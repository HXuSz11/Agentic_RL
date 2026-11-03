"""``SamplingClient``：对基模或某个 LoRA 权重采样、计算 prompt logprob。"""

from __future__ import annotations

import asyncio
import json
import zlib
from array import array
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .futures import APIFuture
from .runtime import Runtime
from .serialize import model_input_to_wire, sampling_params_to_wire
from .types import ModelInput, SampleResponse, SampleSequence, SamplingParams

_DEFAULT_SAMPLING_PARAMS = SamplingParams()


def _check_weights_path(model_path: str | None) -> str | None:
    if model_path is None:
        return None
    if model_path.startswith("trio://"):
        raise ValidationError(
            f"{model_path} 是 PyTRIO 云端路径；本地版请使用 save_weights_for_sampler 返回的本地目录。",
            code="validation.remote_path",
        )
    path = Path(model_path).expanduser()
    if not (path / "adapter_config.json").exists():
        raise ValidationError(f"{model_path} 不是 LoRA 权重目录（缺少 adapter_config.json）", code="validation.bad_weights")
    return str(path.resolve())


def weights_base_model(model_path: str) -> str | None:
    meta = Path(model_path) / "localtrio_meta.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("base_model")
    return None


class SamplingClient:
    """与 ``pytrio.SamplingClient`` 同名同参。"""

    def __init__(self, base_model: str, model_path: str | None = None, runtime: Runtime | None = None) -> None:
        self._runtime = runtime or Runtime.get()
        self.model_path = _check_weights_path(model_path)
        if self.model_path is not None:
            owner = weights_base_model(self.model_path)
            if owner and owner != base_model and owner.rsplit("/", 1)[-1] != base_model.rsplit("/", 1)[-1]:
                raise ValidationError(f"权重 {model_path} 属于 {owner}，不能用 {base_model} 加载")
        self.model_id = base_model
        self.base_model = base_model
        self.sampling_run_cuid = f"sampler::{base_model}::{self.model_path or 'base'}"
        self._pool = self._runtime.sampler(base_model)

    # ------------------------------------------------------------------
    @staticmethod
    def _validate(prompt: ModelInput, params: SamplingParams, include_prompt_logprobs: bool, topk: int) -> None:
        if prompt.is_empty:
            raise ValidationError("prompt cannot be empty", code="validation.empty_prompt")
        if prompt.has_images and (include_prompt_logprobs or topk > 0):
            raise ValidationError("prompt logprobs do not support image chunks", code="validation.image_logprobs_unsupported")
        if params.max_tokens is None or params.max_tokens <= 0:
            raise ValidationError("sampling_params.max_tokens must be greater than 0", code="validation.invalid_sampling_request")

    def _submit(
        self,
        prompt: ModelInput,
        num_samples: int,
        sampling_params: SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
        return_text: bool,
    ) -> APIFuture[SampleResponse]:
        self._validate(prompt, sampling_params, include_prompt_logprobs, topk_prompt_logprobs)
        if num_samples < 1:
            raise ValidationError("num_samples must be >= 1")
        chunks = model_input_to_wire(prompt)
        fut = self._pool.call(
            "generate",
            affinity=_prefix_affinity(chunks),
            chunks=chunks,
            num_samples=int(num_samples),
            params=sampling_params_to_wire(sampling_params),
            lora_path=self.model_path,
            include_prompt_logprobs=bool(include_prompt_logprobs),
            topk_prompt_logprobs=int(topk_prompt_logprobs),
            return_text=bool(return_text),
        )
        return APIFuture(fut).map(_to_response)

    def sample(
        self,
        prompt: ModelInput,
        num_samples: int = 1,
        sampling_params: SamplingParams = _DEFAULT_SAMPLING_PARAMS,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_text: bool = True,
    ) -> APIFuture[SampleResponse]:
        return self._submit(prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs, return_text)

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int = 1,
        sampling_params: SamplingParams = _DEFAULT_SAMPLING_PARAMS,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_text: bool = True,
    ) -> SampleResponse:
        future = self._submit(prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs, return_text)
        return await future

    def compute_logprobs(self, prompt: ModelInput) -> APIFuture[list[float | None]]:
        if prompt.has_images:
            raise ValidationError("compute_logprobs does not support image chunks", code="validation.image_logprobs_unsupported")
        future = self._submit(prompt, 1, SamplingParams(max_tokens=1), True, 0, False)
        return future.map(lambda response: response.prompt_logprobs)

    async def compute_logprobs_async(self, prompt: ModelInput) -> list[float | None]:
        return await self.compute_logprobs(prompt)

    @property
    def task_id(self) -> str:
        """兼容旧接口：采样快照的唯一标识（基模 + 权重目录）。"""
        return self.sampling_run_cuid

    def get_tokenizer(self) -> Any:
        return self._runtime.tokenizer(self.base_model)

    def get_base_model(self) -> str:
        return self.base_model

    def __repr__(self) -> str:
        return f"SamplingClient(base_model={self.base_model!r}, model_path={self.model_path!r})"


_AFFINITY_PREFIX_TOKENS = 512


def _prefix_affinity(chunks: list[tuple]) -> int | None:
    """取 prompt 前 512 个文本 token 的哈希：同一题 / 同一条多轮轨迹落到同一个 vLLM 副本。"""
    prefix: list[int] = []
    for chunk in chunks:
        if chunk[0] != "text":
            break
        prefix.extend(chunk[1][: _AFFINITY_PREFIX_TOKENS - len(prefix)])
        if len(prefix) >= _AFFINITY_PREFIX_TOKENS:
            break
    if not prefix:
        return None
    return zlib.crc32(array("q", prefix).tobytes())


def _to_response(raw: dict[str, Any]) -> SampleResponse:
    return SampleResponse(
        sequences=[SampleSequence(**seq) for seq in raw["sequences"]],
        prompt_logprobs=raw.get("prompt_logprobs") or [],
        topk_prompt_logprobs=raw.get("topk_prompt_logprobs") or [],
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
    )


async def gather_limited(coros: list, limit: int = 64) -> list:
    """小工具：限制并发地 gather（实验代码可选用）。"""
    sem = asyncio.Semaphore(limit)

    async def run(coro: Any) -> Any:
        async with sem:
            return await coro

    return await asyncio.gather(*(run(c) for c in coros))
