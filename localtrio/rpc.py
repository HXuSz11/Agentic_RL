"""主进程 ↔ 引擎子进程的轻量 RPC。

每个引擎（trainer、某个基模的 vLLM 采样副本）都是独立子进程：

- 通过 ``CUDA_VISIBLE_DEVICES`` 绑定到配置里的 GPU；
- 通过 Unix socket + pickle 收发 ``("call", id, method, kwargs)``；
- 调用按提交顺序发送（FIFO），trainer 端串行执行，因此 ``forward_backward`` →
  ``optim_step`` 的顺序与 PyTRIO 一样由提交顺序决定。

用户脚本本身不初始化 CUDA，所以 ``if __name__ == "__main__"`` 之外的代码也不会被
子进程重复执行（子进程通过 ``python -m localtrio.worker`` 启动，不导入用户脚本）。
"""

from __future__ import annotations

import atexit
import itertools
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any

from .config import physical_devices

logger = logging.getLogger("localtrio")

_ALL_WORKERS: list["WorkerHandle"] = []
_ALL_LOCK = threading.Lock()


class WorkerError(RuntimeError):
    """子进程里抛出的异常（附带远端 traceback）。"""


class WorkerHandle:
    """管理一个引擎子进程。"""

    def __init__(
        self,
        name: str,
        role: str,
        spec: dict[str, Any],
        gpus: list[int],
        log_dir: Path,
        inherit_logs: bool = False,
        start_timeout: float = 1800.0,
    ) -> None:
        self.name = name
        self.role = role
        self.gpus = list(gpus)
        self.start_timeout = start_timeout
        self._ids = itertools.count()
        self._pending: dict[int, Future] = {}
        self._pending_lock = threading.Lock()
        self._send_queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._dead_reason: str | None = None
        self.info: dict[str, Any] = {}
        self.inflight = 0

        self._tmpdir = tempfile.mkdtemp(prefix="ltrio-", dir="/tmp" if os.path.isdir("/tmp") else None)
        self._address = os.path.join(self._tmpdir, "w.sock")
        authkey = os.urandom(16)
        self._listener = Listener(self._address, family="AF_UNIX", authkey=authkey)

        log_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        self.log_path = log_dir / f"{safe}-{int(time.time())}.log"
        env = os.environ.copy()
        # 配置里是逻辑编号，映射到外部 CUDA_VISIBLE_DEVICES 下的物理设备
        env["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices(gpus))
        env["LOCALTRIO_WORKER_ADDRESS"] = self._address
        env["LOCALTRIO_WORKER_AUTHKEY"] = authkey.hex()
        env["LOCALTRIO_WORKER_SPEC"] = json.dumps(spec)
        env["LOCALTRIO_PARENT_PID"] = str(os.getpid())
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        # 让子进程能找到 localtrio（即使用户没有 pip install -e .）
        pkg_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        if inherit_logs:
            stdout = None
            self._log_handle = None
        else:
            self._log_handle = open(self.log_path, "ab")  # noqa: SIM115
            stdout = self._log_handle
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "localtrio.worker", role],
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout is not None else None,
            start_new_session=True,
        )
        gpu_text = ",".join(map(str, gpus)) if gpus else "cpu"
        where = "" if inherit_logs else f" (log: {self.log_path})"
        print(f"[localtrio] starting {name} on GPU {gpu_text}{where}", flush=True)

        with _ALL_LOCK:
            _ALL_WORKERS.append(self)
        self._conn: Connection | None = None
        threading.Thread(target=self._connect_and_serve, name=f"{name}-rx", daemon=True).start()
        threading.Thread(target=self._sender, name=f"{name}-tx", daemon=True).start()

    # ------------------------------------------------------------------
    def _log_tail(self, lines: int = 60) -> str:
        try:
            text = self.log_path.read_text(errors="replace").splitlines()
            return "\n".join(text[-lines:])
        except OSError:
            return ""

    def _fail_all(self, reason: str) -> None:
        self._dead_reason = reason
        self._ready.set()
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(WorkerError(reason))

    def _kill_process_group(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if self.proc.poll() is not None:
                break
            try:
                os.killpg(self.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
        try:  # vLLM 派生的子进程也在同一进程组
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    def _abort_start(self, reason: str) -> None:
        """启动失败：让所有等待中的调用报错，并杀掉子进程组，释放它可能占着的显存。"""
        self._fail_all(reason)
        try:
            self._listener.close()
        except Exception:  # noqa: BLE001
            pass
        self._kill_process_group()

    def _connect_and_serve(self) -> None:
        accepted: dict[str, Connection] = {}

        def accept() -> None:
            try:
                accepted["conn"] = self._listener.accept()
            except Exception as error:  # noqa: BLE001
                accepted["error"] = error  # type: ignore[assignment]

        t = threading.Thread(target=accept, daemon=True)
        t.start()
        # 启动超时覆盖“连接 + 加载模型 / 初始化 vLLM”整个阶段，直到收到 ready
        deadline = time.time() + self.start_timeout
        while t.is_alive():
            t.join(timeout=0.5)
            if self.proc.poll() is not None and t.is_alive():
                self._abort_start(f"{self.name} 进程启动失败 (exit={self.proc.returncode})\n{self._log_tail()}")
                return
            if time.time() > deadline:
                self._abort_start(f"{self.name} 启动超时（{self.start_timeout:.0f}s 内没有连上）\n{self._log_tail()}")
                return
        if "conn" not in accepted:
            self._abort_start(f"{self.name} 连接失败: {accepted.get('error')}")
            return
        self._conn = accepted["conn"]
        self._listener.close()
        try:
            while not self._ready.is_set():
                if self._conn.poll(0.5):
                    msg = self._conn.recv()
                    if msg[0] == "ready":
                        self.info = msg[1] or {}
                        self._ready.set()
                        print(f"[localtrio] {self.name} ready", flush=True)
                    elif msg[0] == "fatal":
                        self._abort_start(f"{self.name} 初始化失败:\n{msg[1]}")
                        return
                    continue
                if self.proc.poll() is not None:
                    self._abort_start(f"{self.name} 初始化时进程退出 (exit={self.proc.returncode})\n{self._log_tail()}")
                    return
                if time.time() > deadline:
                    self._abort_start(
                        f"{self.name} 初始化超时（{self.start_timeout:.0f}s 内没有就绪，可调大 worker_start_timeout）\n{self._log_tail()}"
                    )
                    return
            while True:
                msg = self._conn.recv()
                kind = msg[0]
                if kind == "ready":
                    self.info = msg[1] or {}
                    self._ready.set()
                    print(f"[localtrio] {self.name} ready", flush=True)
                elif kind == "fatal":
                    self._fail_all(f"{self.name} 初始化失败:\n{msg[1]}")
                    return
                elif kind in ("result", "error"):
                    with self._pending_lock:
                        fut = self._pending.pop(msg[1], None)
                    if fut is None or fut.done():
                        continue
                    if kind == "result":
                        fut.set_result(msg[2])
                    else:
                        fut.set_exception(WorkerError(f"[{self.name}] {msg[2]}"))
        except (EOFError, OSError):
            code = self.proc.poll()
            self._fail_all(f"{self.name} 进程已退出 (exit={code})\n{self._log_tail()}")

    def _sender(self) -> None:
        self._ready.wait()
        while True:
            item = self._send_queue.get()
            if item is None:
                return
            if item is _SHUTDOWN:
                # shutdown 也经由发送线程发出，避免与正在发送的大请求在同一个连接上交错写入
                try:
                    if self._conn is not None:
                        self._conn.send(("shutdown",))
                except Exception:  # noqa: BLE001
                    pass
                return
            req_id, method, kwargs, fut = item
            if self._dead_reason is not None or self._conn is None:
                if not fut.done():
                    fut.set_exception(WorkerError(self._dead_reason or f"{self.name} 不可用"))
                continue
            try:
                self._conn.send(("call", req_id, method, kwargs))
            except Exception as error:  # noqa: BLE001
                with self._pending_lock:
                    self._pending.pop(req_id, None)
                if not fut.done():
                    fut.set_exception(WorkerError(f"{self.name} 发送失败: {error}"))

    # ------------------------------------------------------------------
    def call(self, method: str, **kwargs: Any) -> Future:
        """异步提交一次调用，返回 ``concurrent.futures.Future``。"""
        fut: Future = Future()
        if self._dead_reason is not None:
            fut.set_exception(WorkerError(self._dead_reason))
            return fut
        req_id = next(self._ids)
        with self._pending_lock:
            self._pending[req_id] = fut
            self.inflight += 1

        def _done(_: Future) -> None:
            with self._pending_lock:
                self.inflight -= 1

        fut.add_done_callback(_done)
        self._send_queue.put((req_id, method, kwargs, fut))
        return fut

    def wait_ready(self, timeout: float | None = None) -> dict[str, Any]:
        self._ready.wait(timeout=timeout)
        if self._dead_reason is not None:
            raise WorkerError(self._dead_reason)
        return self.info

    @property
    def alive(self) -> bool:
        return self._dead_reason is None and self.proc.poll() is None

    def request_shutdown(self) -> None:
        """不再接受新调用，并让发送线程在已排队的请求之后发出 shutdown。"""
        if self._dead_reason is None:
            self._dead_reason = f"{self.name} 已关闭"
        self._send_queue.put(_SHUTDOWN)

    def finish_shutdown(self, timeout: float = 15.0) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        self._kill_process_group()
        self._fail_all(self._dead_reason or f"{self.name} 已关闭")
        self._send_queue.put(None)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def shutdown(self, timeout: float = 15.0) -> None:
        self.request_shutdown()
        self.finish_shutdown(timeout)


_SHUTDOWN = object()


def shutdown_all() -> None:
    """并行关闭所有引擎：先全部发出 shutdown，再统一等待 / 强制结束。"""
    with _ALL_LOCK:
        workers = list(_ALL_WORKERS)
        _ALL_WORKERS.clear()
    for worker in workers:
        try:
            worker.request_shutdown()
        except Exception:  # noqa: BLE001
            pass
    threads = [threading.Thread(target=worker.finish_shutdown, daemon=True) for worker in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)


atexit.register(shutdown_all)


# ----------------------------------------------------------------------
# 子进程侧
# ----------------------------------------------------------------------


def connect_from_worker() -> tuple[Connection, dict[str, Any]]:
    address = os.environ["LOCALTRIO_WORKER_ADDRESS"]
    authkey = bytes.fromhex(os.environ["LOCALTRIO_WORKER_AUTHKEY"])
    spec = json.loads(os.environ.get("LOCALTRIO_WORKER_SPEC", "{}"))
    conn = Client(address, family="AF_UNIX", authkey=authkey)
    return conn, spec


def start_parent_watchdog() -> None:
    """父进程消失后自动退出，避免遗留占用显存的僵尸进程。"""
    parent = int(os.environ.get("LOCALTRIO_PARENT_PID", "0") or 0)

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if parent and os.getppid() != parent:
                try:
                    os.killpg(os.getpgid(0), signal.SIGTERM)
                finally:
                    os._exit(0)

    threading.Thread(target=watch, daemon=True).start()
