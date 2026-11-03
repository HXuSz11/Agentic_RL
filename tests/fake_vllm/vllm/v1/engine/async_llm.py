"""假的 AsyncLLM：生成确定性的 token，并把收到的参数编码进 text 方便测试断言。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Logprob:
    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None
    logprobs: list[dict[int, Logprob]] | None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    prompt_logprobs: list[dict[int, Logprob] | None] | None
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = True


class AsyncLLM:
    def __init__(self, args: Any) -> None:
        self.args = args

    @classmethod
    def from_engine_args(cls, engine_args: Any, **_: Any) -> "AsyncLLM":
        return cls(engine_args)

    def shutdown(self) -> None:
        pass

    async def generate(self, prompt: dict, sampling_params: Any, request_id: str, *, lora_request: Any = None, **_: Any):
        await asyncio.sleep(0.01)
        ids = list(prompt["prompt_token_ids"])
        mm = prompt.get("multi_modal_data")
        info = {
            "lora": getattr(lora_request, "lora_path", None),
            "lora_id": getattr(lora_request, "lora_int_id", None),
            "has_mm": mm is not None,
            "prompt_len": len(ids),
            "top_k": sampling_params.top_k,
            "stop_token_ids": sampling_params.stop_token_ids,
            "n": sampling_params.n,
        }
        outputs = []
        for index in range(sampling_params.n):
            tokens: list[int] = []
            logprobs: list[dict[int, Logprob]] = []
            reason = "length"
            for step in range(int(sampling_params.max_tokens)):
                token = (ids[-1] + step + 1 + index) % 500 + 10
                tokens.append(token)
                logprobs.append({token: Logprob(logprob=-0.1 * (step + 1), rank=1)})
                if sampling_params.stop_token_ids and step == 2:
                    tokens[-1] = sampling_params.stop_token_ids[0]
                    logprobs[-1] = {tokens[-1]: Logprob(logprob=-0.05, rank=1)}
                    reason = "stop"
                    break
            outputs.append(CompletionOutput(index=index, text=json.dumps(info), token_ids=tokens, cumulative_logprob=None, logprobs=logprobs, finish_reason=reason))
        prompt_logprobs = None
        if sampling_params.prompt_logprobs is not None:
            k = sampling_params.prompt_logprobs
            prompt_logprobs = [None]
            for i, token in enumerate(ids[1:], start=1):
                entry = {token: Logprob(logprob=-1.0 - 0.01 * i, rank=k + 1)}
                for r in range(1, k + 1):
                    entry[1000 + r] = Logprob(logprob=-0.1 * r, rank=r)
                prompt_logprobs.append(entry)
        from vllm.sampling_params import RequestOutputKind

        if sampling_params.output_kind == RequestOutputKind.FINAL_ONLY:
            # 与真实 vLLM 的 ParentRequest 一致：所有子请求结束后一次性返回 n 条
            yield RequestOutput(request_id=request_id, prompt_token_ids=ids, prompt_logprobs=prompt_logprobs, outputs=outputs)
            return
        # CUMULATIVE（vLLM 默认）：n>1 时每次 yield 只带“本步有更新的”那个子请求
        for i, completion in enumerate(outputs):
            await asyncio.sleep(0)
            yield RequestOutput(
                request_id=request_id,
                prompt_token_ids=ids,
                prompt_logprobs=prompt_logprobs if i == 0 else None,
                outputs=[completion],
                finished=i == len(outputs) - 1,
            )
