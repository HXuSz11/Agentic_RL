"""localtrio 配置：GPU 布局、模型路径、训练与采样引擎参数。

读取顺序（后者覆盖前者）：

1. 内置默认值（适配 4×A6000 48GB，训练 Qwen3.5-4B 这类小模型）；
2. ``LOCALTRIO_CONFIG`` 指向的 YAML，或当前目录的 ``localtrio.yaml``；
3. ``trio.configure(...)`` 传入的关键字参数；
4. 少量环境变量（``LOCALTRIO_HOME``、``LOCALTRIO_SAMPLER_BACKEND`` 等）。

一份最小 YAML 示例::

    trainer:
      gpus: [0]
    models:
      default:            # 训练基模的采样引擎
        gpus: [1, 2, 3]   # 每张卡一个 vLLM 副本（data parallel）
      Qwen/Qwen3.6-27B:   # OPD 教师
        gpus: [2, 3]
        tensor_parallel_size: 2
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("localtrio")

_LOCK = threading.Lock()
_OVERRIDES: dict[str, Any] = {}

# PyTRIO.configure 的参数（超时、重试、并发等）在本地没有意义，静默接受。
_REMOTE_ONLY_OPTIONS = {
    "api_key",
    "base_url",
    "timeout",
    "sampling_timeout",
    "request_timeout_seconds",
    "retry_max_attempts",
    "retry_initial_delay_seconds",
    "retry_multiplier",
    "retry_max_delay_seconds",
    "retry_jitter_ratio",
    "training_retry_timeout_seconds",
    "control_retry_timeout_seconds",
    "sampling_retry_timeout_seconds",
    "actor_capacity_poll_interval_seconds",
    "actor_capacity_poll_jitter_ratio",
    "actor_capacity_wait_timeout_seconds",
    "sampling_timeout_seconds",
    "sampling_first_chunk_timeout_seconds",
    "sampling_chunk_idle_timeout_seconds",
    "actor_event_submit_timeout_seconds",
    "actor_event_polling_timeout_seconds",
    "training_event_max_in_flight",
    "training_event_queue_capacity",
    "sampling_request_max_in_flight",
    "sampling_request_queue_capacity",
    "actor_heartbeat_enabled",
    "actor_heartbeat_interval_seconds",
    "actor_heartbeat_max_in_flight",
    "env",
}


@dataclass
class TrainerConfig:
    """LoRA 训练引擎（HF transformers + PEFT）。"""

    gpus: list[int] = field(default_factory=lambda: [0])
    dtype: str = "bfloat16"
    # sdpa 在 A6000（Ampere）上开箱即用；装了 flash-attn 可改为 flash_attention_2
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    # 一个 micro-batch 内 padding 后的 token 上限；4B 模型 + 梯度检查点在 48GB 上 16k 较稳妥
    max_tokens_per_microbatch: int = 16384
    # 计算 logprob 时 lm_head 的分块大小，避免一次性物化 [L, vocab] 的 logits
    logits_chunk_size: int = 1024
    # 内置 loss 的归一化分母：active_tokens（默认，对齐 PyTRIO 的 logical-batch 归一化）
    # / samples / sum（Tinker 风格，逐 token 求和）
    loss_normalization: str = "active_tokens"
    # forward_backward_custom 的梯度是否再除以分母；none 表示严格等于用户 loss 的梯度
    custom_loss_normalization: str = "none"
    # LoRA 作用的模块：all-linear = 语言模型里全部投影层（含 Qwen3.5 线性注意力的 in_proj/out_proj，
    # vLLM ≥0.11 支持）；standard = 只用 q/k/v/o + gate/up/down（旧版 vLLM 报 LoRA 模块不支持时改用它）
    lora_target_modules: str | list[str] = "all-linear"
    lora_alpha: float = 32.0
    # 训练端前向保持确定性（custom loss 需要两次前向一致），所有 Dropout 在计算时都关闭，
    # 因此这里填非 0 值也不会生效
    lora_dropout: float = 0.0
    # 是否允许训练 lm_head（PyTRIO 的 train_unembed）；vLLM 的 LoRA 一般不支持，默认忽略
    allow_unembed: bool = False
    seed: int = 0


@dataclass
class EngineConfig:
    """某个基模的采样引擎（vLLM 或 HF）。"""

    gpus: list[int] | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = None
    max_num_seqs: int = 256
    # 单次调度的 token 上限（预填充按此分块）。compute_logprobs / prompt logprobs 会为每个
    # prompt token 计算整词表 fp32 logprob（Qwen3.5 词表约 25 万），显存尖峰约与它成正比：
    # 2048 → 约 4~5GB。只做生成、不算 prompt logprob 时可调大以加快长 prompt 的预填充。
    max_num_batched_tokens: int | None = 2048
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    dtype: str = "bfloat16"
    quantization: str | None = None
    max_lora_rank: int = 64
    max_loras: int = 4
    max_cpu_loras: int = 8
    max_logprobs: int = 64
    enable_lora: bool = True
    # raw_logprobs = 未经 temperature/top-p 处理的模型分布（与训练端 logprob 口径一致）
    logprobs_mode: str = "raw_logprobs"
    limit_mm_per_prompt: dict[str, int] | None = None
    extra_engine_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalConfig:
    home: str = "~/.cache/localtrio"
    gpus: list[int] | None = None
    sampler_backend: str = "vllm"  # vllm | hf
    model_source: str = "auto"  # auto | huggingface | modelscope | local
    model_paths: dict[str, str] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    models: dict[str, EngineConfig] = field(default_factory=dict)
    worker_logs: str = "file"  # file | inherit
    worker_start_timeout: float = 1800.0
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    @property
    def home_path(self) -> Path:
        return Path(os.path.expanduser(self.home)).resolve()

    def engine_for(self, base_model: str) -> EngineConfig:
        """返回基模对应的采样引擎配置；未显式配置时退回 ``default``。"""
        if base_model in self.models:
            return self.models[base_model]
        short = base_model.rsplit("/", 1)[-1]
        for key, value in self.models.items():
            if key.rsplit("/", 1)[-1] == short:
                return value
        return copy.deepcopy(self.models.get("default", EngineConfig()))

    def explicit_engine(self, base_model: str) -> bool:
        short = base_model.rsplit("/", 1)[-1]
        return any(k == base_model or k.rsplit("/", 1)[-1] == short for k in self.models if k != "default")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def visible_devices() -> list[str] | None:
    """外部设置的 CUDA_VISIBLE_DEVICES（物理编号或 UUID）；未设置时返回 None。"""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    return [x.strip() for x in visible.split(",") if x.strip()]


def physical_devices(gpus: list[int]) -> list[str]:
    """把配置里的逻辑 GPU 编号映射为子进程的 CUDA_VISIBLE_DEVICES。

    配置中的编号总是相对“当前可见的 GPU”而言：如果启动脚本前已经设置了
    ``CUDA_VISIBLE_DEVICES=4,5,6,7``，那么配置里的 0 指物理 GPU 4。
    """
    visible = visible_devices()
    if visible is None:
        return [str(g) for g in gpus]
    out = []
    for g in gpus:
        if g >= len(visible):
            raise ValueError(f"配置使用了 GPU {g}，但 CUDA_VISIBLE_DEVICES 只暴露了 {len(visible)} 张卡")
        out.append(visible[g])
    return out


def detect_gpus() -> list[int]:
    """返回可用 GPU 的逻辑编号 0..N-1（遵守外部设置的 CUDA_VISIBLE_DEVICES）。"""
    visible = visible_devices()
    if visible is not None:
        return list(range(len(visible)))
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout
            return [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
        except Exception:  # noqa: BLE001
            return []
    return []


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"未知配置项 {type(obj).__name__}.{key}")
        setattr(obj, key, value)
    return obj


def _from_dict(raw: dict[str, Any]) -> LocalConfig:
    raw = dict(raw)
    cfg = LocalConfig()
    trainer = raw.pop("trainer", None) or {}
    models = raw.pop("models", None) or {}
    _merge_dataclass(cfg.trainer, trainer)
    for name, engine in models.items():
        cfg.models[str(name)] = _merge_dataclass(EngineConfig(), engine or {})
    _merge_dataclass(cfg, raw)
    return cfg


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是映射")
    return data


def config_file() -> Path | None:
    env = os.environ.get("LOCALTRIO_CONFIG")
    if env:
        path = Path(env).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"LOCALTRIO_CONFIG 指向的文件不存在：{path}")
        return path
    local = Path.cwd() / "localtrio.yaml"
    return local if local.exists() else None


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config() -> LocalConfig:
    """组合默认值、YAML、``configure()`` 覆盖和环境变量。"""
    raw: dict[str, Any] = {}
    path = config_file()
    if path is not None:
        raw = _load_yaml(path)
    with _LOCK:
        raw = _deep_update(raw, copy.deepcopy(_OVERRIDES))
    cfg = _from_dict(raw)

    env_map = {
        "LOCALTRIO_HOME": "home",
        "LOCALTRIO_SAMPLER_BACKEND": "sampler_backend",
        "LOCALTRIO_MODEL_SOURCE": "model_source",
        "LOCALTRIO_WORKER_LOGS": "worker_logs",
    }
    for env, attr in env_map.items():
        if os.environ.get(env):
            setattr(cfg, attr, os.environ[env])
    if os.environ.get("LOCALTRIO_TRAINER_GPUS"):
        cfg.trainer.gpus = [int(x) for x in os.environ["LOCALTRIO_TRAINER_GPUS"].split(",") if x.strip()]
    if os.environ.get("LOCALTRIO_SAMPLER_GPUS"):
        cfg.models.setdefault("default", EngineConfig()).gpus = [
            int(x) for x in os.environ["LOCALTRIO_SAMPLER_GPUS"].split(",") if x.strip()
        ]

    if cfg.gpus is None:
        cfg.gpus = detect_gpus()
    default = cfg.models.setdefault("default", EngineConfig())
    if not cfg.gpus:
        # 没有 GPU：只用于 CPU 调试，trainer 与 HF 采样器都跑在 CPU 上
        cfg.trainer.gpus = []
        for engine in cfg.models.values():
            engine.gpus = []
        return cfg
    if default.gpus is None:
        # 默认布局：trainer 之外的卡全部给训练基模的采样引擎；只有一张卡时与 trainer 共用。
        others = [g for g in cfg.gpus if g not in cfg.trainer.gpus]
        default.gpus = others or list(cfg.trainer.gpus)
        if not others:
            default.gpu_memory_utilization = min(default.gpu_memory_utilization, 0.4)
    return cfg


def configure(**options: Any) -> None:
    """兼容 ``pytrio.configure``。

    PyTRIO 的网络/超时/并发参数在本地无意义，会被忽略；localtrio 自己的配置项
    （``home``、``trainer={...}``、``models={...}``、``sampler_backend`` 等）会覆盖 YAML。
    """
    local: dict[str, Any] = {}
    for key, value in options.items():
        if key in _REMOTE_ONLY_OPTIONS or value is None:
            continue
        if key == "log_level":
            logging.getLogger("localtrio").setLevel(str(value).upper())
            local[key] = value
            continue
        if key not in LocalConfig.__dataclass_fields__:
            logger.debug("localtrio.configure: ignore unknown option %s", key)
            continue
        local[key] = value
    with _LOCK:
        _deep_update(_OVERRIDES, local)
