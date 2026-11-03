"""测试用的假 vLLM：只模拟 localtrio 用到的 vLLM 0.33 接口与返回结构（不做真实推理）。

真实结构参考 vllm/outputs.py（RequestOutput / CompletionOutput）、vllm/logprobs.py（Logprob、
prompt_logprobs 首位为 None）、vllm/sampling_params.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sampling_params import RequestOutputKind

__version__ = "0.33.0-fake"


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    max_tokens: int | None = 16
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    output_kind: Any = RequestOutputKind.CUMULATIVE

    def __post_init__(self) -> None:
        if self.top_k < -1:
            raise ValueError(f"top_k must be 0 (disable), or at least 1, got {self.top_k}.")
