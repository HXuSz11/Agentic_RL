from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineArgs:
    model: str = ""
    tokenizer: str | None = None
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    enable_prefix_caching: bool | None = None
    enforce_eager: bool = False
    dtype: str = "auto"
    quantization: str | None = None
    enable_lora: bool = False
    max_lora_rank: int = 16
    max_loras: int = 1
    max_cpu_loras: int | None = None
    max_logprobs: int = 20
    logprobs_mode: str = "raw_logprobs"
    seed: int = 0
    disable_log_stats: bool = False
    limit_mm_per_prompt: dict[str, Any] = field(default_factory=dict)


@dataclass
class AsyncEngineArgs(EngineArgs):
    pass
