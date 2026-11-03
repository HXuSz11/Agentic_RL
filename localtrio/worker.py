"""引擎子进程入口：``python -m localtrio.worker {trainer|sampler}``。

由 :mod:`localtrio.rpc` 启动，不要手动运行。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import traceback
from typing import Any

from .rpc import connect_from_worker, start_parent_watchdog


def _build(role: str, spec: dict[str, Any]) -> Any:
    if role == "trainer":
        from .engines.trainer import TrainerEngine

        return TrainerEngine(spec)
    if role == "sampler":
        backend = spec.get("backend", "vllm")
        if backend == "vllm":
            from .engines.vllm_sampler import VLLMSamplerEngine

            return VLLMSamplerEngine(spec)
        from .engines.hf_sampler import HFSamplerEngine

        return HFSamplerEngine(spec)
    raise ValueError(f"unknown worker role {role}")


def _serve_sync(conn: Any, engine: Any) -> None:
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        except Exception:  # noqa: BLE001 - 消息损坏 / 无法反序列化：记下原因再退出
            traceback.print_exc()
            sys.stderr.flush()
            return
        if msg[0] == "shutdown":
            return
        _, req_id, method, kwargs = msg
        try:
            result = getattr(engine, method)(**kwargs)
            conn.send(("result", req_id, result))
        except Exception:  # noqa: BLE001
            conn.send(("error", req_id, traceback.format_exc()))


async def _serve_async(conn: Any, engine: Any) -> None:
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue = asyncio.Queue()

    def reader() -> None:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                msg = ("shutdown",)
            except Exception:  # noqa: BLE001 - 消息损坏 / 无法反序列化：记下原因再退出
                traceback.print_exc()
                sys.stderr.flush()
                msg = ("shutdown",)
            loop.call_soon_threadsafe(inbox.put_nowait, msg)
            if msg[0] == "shutdown":
                return

    threading.Thread(target=reader, daemon=True).start()
    send_lock = threading.Lock()

    def send(message: tuple) -> None:
        with send_lock:
            conn.send(message)

    async def handle(req_id: int, method: str, kwargs: dict[str, Any]) -> None:
        try:
            fn = getattr(engine, method)
            result = fn(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            send(("result", req_id, result))
        except Exception:  # noqa: BLE001
            send(("error", req_id, traceback.format_exc()))

    tasks: set[asyncio.Task] = set()
    while True:
        msg = await inbox.get()
        if msg[0] == "shutdown":
            break
        _, req_id, method, kwargs = msg
        task = asyncio.create_task(handle(req_id, method, kwargs))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    if hasattr(engine, "shutdown"):
        await engine.shutdown()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    role = sys.argv[1]
    conn, spec = connect_from_worker()
    start_parent_watchdog()
    try:
        engine = _build(role, spec)
        if getattr(engine, "is_async", False):

            async def run() -> None:
                await engine.start()
                conn.send(("ready", engine.info()))
                await _serve_async(conn, engine)

            asyncio.run(run())
            return
        conn.send(("ready", engine.info()))
    except Exception:  # noqa: BLE001
        conn.send(("fatal", traceback.format_exc()))
        sys.exit(1)
    _serve_sync(conn, engine)


if __name__ == "__main__":
    main()
