"""主进程里的资源管理：模型解析、GPU 分配、引擎子进程与采样副本负载均衡。"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from .config import EngineConfig, LocalConfig, load_config
from .models import load_tokenizer, resolve_model_path
from .rpc import WorkerHandle, shutdown_all

logger = logging.getLogger("localtrio")


class LayoutError(RuntimeError):
    """GPU 布局冲突或缺失。"""


class SamplerPool:
    """同一基模的多个采样副本（data parallel）。

    路由策略：带 ``affinity``（prompt 前缀的哈希）的请求优先发往固定副本，让多轮
    Agent 的同一条轨迹命中同一份 vLLM prefix cache；该副本明显比最空闲的副本忙
    （在途请求多出 ``AFFINITY_SLACK`` 个以上）时退回最少负载路由。
    """

    AFFINITY_SLACK = 8

    def __init__(self, base_model: str, replicas: list[WorkerHandle], engine: EngineConfig) -> None:
        self.base_model = base_model
        self.replicas = replicas
        self.engine = engine
        self._rr = 0
        self._lock = threading.Lock()

    def call(self, method: str, affinity: int | None = None, **kwargs: Any) -> Future:
        with self._lock:
            alive = [r for r in self.replicas if r.alive] or self.replicas
            self._rr += 1
            # 先按在途数，再轮询打散
            least = min(alive, key=lambda r: (r.inflight, (alive.index(r) - self._rr) % len(alive)))
            replica = least
            if affinity is not None and len(alive) > 1:
                preferred = alive[affinity % len(alive)]
                if preferred.inflight <= least.inflight + self.AFFINITY_SLACK:
                    replica = preferred
        return replica.call(method, **kwargs)


class Runtime:
    _instance: "Runtime | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "Runtime":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = Runtime()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    def __init__(self) -> None:
        self.cfg: LocalConfig = load_config()
        logging.getLogger("localtrio").setLevel(self.cfg.log_level.upper())
        self.home = self.cfg.home_path
        self.home.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._model_dirs: dict[str, str] = {}
        self._trainers: dict[str, WorkerHandle] = {}
        self._samplers: dict[str, SamplerPool] = {}
        self._gpu_usage: dict[int, float] = {}
        self.backend = self.cfg.sampler_backend
        if self.backend == "vllm" and not self.cfg.gpus:
            logger.warning("没有检测到 GPU，采样后端切换为 hf（仅适合调试）")
            self.backend = "hf"
        if self.backend == "vllm":
            try:
                import importlib.util

                if importlib.util.find_spec("vllm") is None:
                    raise ImportError
            except ImportError:
                logger.warning("没有安装 vLLM，采样后端切换为 hf（速度很慢，建议 pip install vllm）")
                self.backend = "hf"

    # ------------------------------------------------------------------
    def model_dir(self, base_model: str) -> str:
        with self._lock:
            if base_model not in self._model_dirs:
                self._model_dirs[base_model] = resolve_model_path(
                    base_model, self.cfg.model_paths, self.cfg.model_source
                )
            return self._model_dirs[base_model]

    def tokenizer(self, base_model: str) -> Any:
        return load_tokenizer(self.model_dir(base_model))

    @property
    def log_dir(self) -> Path:
        return self.home / "logs"

    def new_run_id(self) -> str:
        return time.strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]

    # ------------------------------------------------------------------ trainer
    def trainer(self, base_model: str) -> WorkerHandle:
        with self._lock:
            handle = self._trainers.get(base_model)
            if handle is not None and handle.alive:
                return handle
            if handle is not None:
                # 旧 trainer 已失败：先确保它的进程组退出、显存释放，再在同一组 GPU 上重建
                handle.shutdown(timeout=5)
            spec = {
                "base_model": base_model,
                "model_dir": self.model_dir(base_model),
                "trainer": dataclasses.asdict(self.cfg.trainer),
            }
            gpus = list(self.cfg.trainer.gpus) if self.cfg.gpus else []
            handle = WorkerHandle(
                name=f"trainer[{base_model}]",
                role="trainer",
                spec=spec,
                gpus=gpus,
                log_dir=self.log_dir,
                inherit_logs=self.cfg.worker_logs == "inherit",
                start_timeout=self.cfg.worker_start_timeout,
            )
            self._trainers[base_model] = handle
            return handle

    # ------------------------------------------------------------------ samplers
    def _place(self, base_model: str, engine: EngineConfig) -> list[list[int]]:
        if not self.cfg.gpus:
            return [[]]
        gpus = list(engine.gpus or [])
        if not gpus:
            raise LayoutError(f"没有为 {base_model} 的采样引擎分配 GPU")
        tp = max(int(engine.tensor_parallel_size), 1)
        if len(gpus) % tp != 0:
            raise LayoutError(f"{base_model}: GPU 数 {len(gpus)} 不能被 tensor_parallel_size={tp} 整除")
        util = float(engine.gpu_memory_utilization) if self.backend == "vllm" else 0.3
        if util > 0.98:
            raise LayoutError(f"{base_model}: gpu_memory_utilization={util} 过高（最大 0.98）")
        for gpu in gpus:
            used = self._gpu_usage.get(gpu, 0.0)
            if used > 0 and used + util > 0.98:
                raise LayoutError(
                    f"GPU {gpu} 已被其他采样引擎占用（已用 {used:.2f}，再加 {util:.2f} 超过显存）。\n"
                    f"{base_model} 需要独立的 GPU：请在 localtrio.yaml 的 models 下为它单独配置 gpus，"
                    f"例如 configs/ 目录里的 OPD 预设（trainer→0，学生→1，教师→2,3）。"
                )
            if gpu in (self.cfg.trainer.gpus or []) and util > 0.6:
                logger.warning(
                    "%s 的采样引擎与训练共用 GPU %d，且 gpu_memory_utilization=%.2f；训练端可能显存不足（参考 configs/1gpu.yaml）",
                    base_model,
                    gpu,
                    util,
                )
        for gpu in gpus:
            self._gpu_usage[gpu] = self._gpu_usage.get(gpu, 0.0) + util
        return [gpus[i : i + tp] for i in range(0, len(gpus), tp)]

    def sampler(self, base_model: str) -> SamplerPool:
        with self._lock:
            pool = self._samplers.get(base_model)
            if pool is not None:
                return pool
            engine = self.cfg.engine_for(base_model)
            if self.cfg.gpus and not self.cfg.explicit_engine(base_model) and any(
                not self.cfg.explicit_engine(m) for m in self._samplers
            ):
                raise LayoutError(
                    f"已经有一个基模在使用 models.default 的 GPU，{base_model} 需要在 localtrio.yaml 里单独配置。\n"
                    "例：\nmodels:\n  default: {gpus: [1]}\n"
                    f"  {base_model}: {{gpus: [2, 3], tensor_parallel_size: 2}}"
                )
            placements = self._place(base_model, engine)
            spec = {
                "base_model": base_model,
                "model_dir": self.model_dir(base_model),
                "backend": self.backend,
                "engine": dataclasses.asdict(engine),
                "attn_implementation": self.cfg.trainer.attn_implementation,
            }
            replicas = []
            for index, gpus in enumerate(placements):
                suffix = f"#{index}" if len(placements) > 1 else ""
                replicas.append(
                    WorkerHandle(
                        name=f"sampler[{base_model}]{suffix}",
                        role="sampler",
                        spec=spec,
                        gpus=gpus,
                        log_dir=self.log_dir,
                        inherit_logs=self.cfg.worker_logs == "inherit",
                        start_timeout=self.cfg.worker_start_timeout,
                    )
                )
            pool = SamplerPool(base_model, replicas, engine)
            self._samplers[base_model] = pool
            return pool

    def shutdown(self) -> None:
        shutdown_all()
        self._trainers.clear()
        self._samplers.clear()
        self._gpu_usage.clear()
