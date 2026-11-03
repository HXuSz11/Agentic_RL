"""把公共类型转换为发送给引擎子进程的轻量结构（tuple / numpy）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import Datum, EncodedTextChunk, ModelInput, SamplingParams


def model_input_to_wire(model_input: ModelInput) -> list[tuple]:
    chunks: list[tuple] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, EncodedTextChunk):
            chunks.append(("text", list(chunk.tokens)))
        else:
            chunks.append(("image", bytes(chunk.data), chunk.format, chunk.expected_tokens))
    return chunks


def datum_to_wire(datum: Datum, keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    """``keys`` 指定要发给 trainer 的 loss 输入；其余字段（例如 custom loss 用的标量）留在本地。"""
    items = datum.loss_fn_inputs.items() if keys is None else ((k, datum.loss_fn_inputs[k]) for k in keys if k in datum.loss_fn_inputs)
    return {
        "chunks": model_input_to_wire(datum.model_input),
        "inputs": {name: np.asarray(t.data, dtype=np.float32 if t.dtype == "float32" else np.int64).reshape(t.shape) for name, t in items},
    }


def sampling_params_to_wire(params: SamplingParams) -> dict[str, Any]:
    stop = params.stop
    if stop is not None and not isinstance(stop, str):
        stop = list(stop)
    return {
        "max_tokens": params.max_tokens,
        "seed": params.seed,
        "stop": stop,
        "temperature": params.temperature,
        "top_k": params.top_k,
        "top_p": params.top_p,
    }
