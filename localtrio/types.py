"""localtrio 的公共数据类型。

这些类型的字段与行为对齐 PyTRIO 0.2.9 的用户接口（``Datum``、``ModelInput``、
``TensorData``、``SamplingParams`` 等），实验代码可以把 ``import pytrio as trio``
换成 ``import localtrio as trio`` 而不改其他逻辑。
"""

from __future__ import annotations

import base64
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel as _PydanticBaseModel
from pydantic import ConfigDict, Field, field_serializer, field_validator, model_validator

if TYPE_CHECKING:  # pragma: no cover
    import torch


class BaseModel(_PydanticBaseModel):
    """统一的校验配置：禁止未知字段，允许修改。"""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=False)


# ---------------------------------------------------------------------------
# 张量
# ---------------------------------------------------------------------------


class TensorData(BaseModel):
    """一维存储 + 逻辑 shape 的数值张量，支持 numpy / torch / list 输入。"""

    data: list[float] | list[int]
    dtype: Literal["float32", "int64"]
    shape: list[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _convert_input(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return cls.from_torch(value).model_dump()
        except ImportError:  # pragma: no cover - torch 总是随 localtrio 安装
            pass
        if isinstance(value, np.ndarray):
            return cls.from_numpy(value).model_dump()
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if arr.size == 0:
                return {"data": [], "dtype": "float32", "shape": [0]}
            if arr.dtype.kind not in ("i", "u", "f", "b"):
                raise ValueError("Only numeric lists are allowed for TensorData")
            if arr.dtype.kind == "f":
                return {"data": arr.astype(np.float32).reshape(-1).tolist(), "dtype": "float32", "shape": list(arr.shape)}
            return {"data": arr.astype(np.int64).reshape(-1).tolist(), "dtype": "int64", "shape": list(arr.shape)}
        return value

    @model_validator(mode="after")
    def _validate(self) -> "TensorData":
        if "shape" not in self.model_fields_set:
            self.shape = [len(self.data)]
        if any(dim < 0 for dim in self.shape):
            raise ValueError("Negative dimensions are not allowed in shape")
        expected = math.prod(self.shape) if self.shape else 1
        if expected != len(self.data):
            raise ValueError(
                f"Unexpected data length {len(self.data)} for shape {self.shape} (expected {expected})"
            )
        if self.dtype == "float32":
            self.data = [float(x) for x in self.data]
        else:
            out: list[int] = []
            for x in self.data:
                if isinstance(x, float) and not x.is_integer():
                    raise ValueError(f"Expected integer values for dtype 'int64', got float {x}")
                out.append(int(x))
            self.data = out
        return self

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "TensorData":
        if np.issubdtype(array.dtype, np.floating):
            return cls(data=array.astype(np.float32).reshape(-1).tolist(), dtype="float32", shape=list(array.shape))
        if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
            return cls(data=array.astype(np.int64).reshape(-1).tolist(), dtype="int64", shape=list(array.shape))
        raise ValueError(f"Unsupported numpy dtype for TensorData: {array.dtype}")

    @classmethod
    def from_torch(cls, tensor: "torch.Tensor") -> "TensorData":
        import torch

        tensor = tensor.detach().cpu()
        if tensor.dtype.is_floating_point:
            return cls(data=tensor.to(torch.float32).reshape(-1).tolist(), dtype="float32", shape=list(tensor.shape))
        return cls(data=tensor.to(torch.int64).reshape(-1).tolist(), dtype="int64", shape=list(tensor.shape))

    def to_numpy(self) -> np.ndarray:
        np_dtype = np.float32 if self.dtype == "float32" else np.int64
        return np.asarray(self.data, dtype=np_dtype).reshape(self.shape)

    def to_torch(self) -> "torch.Tensor":
        import torch

        return torch.from_numpy(self.to_numpy())

    def tolist(self) -> list[Any]:
        return list(self.data)

    def __len__(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# 模型输入
# ---------------------------------------------------------------------------


class EncodedTextChunk(BaseModel):
    """一段已经编码好的 token。"""

    tokens: list[int]
    type: Literal["encoded_text"] = "encoded_text"

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


class ImageChunk(BaseModel):
    """一张 PNG/JPEG 图片。它在序列里替代一个 ``<|image_pad|>`` 占位符。

    ``expected_tokens`` 是图片展开后的视觉 token 数，由与基模匹配的 image
    processor 计算（``patches // merge_size**2``）。本地训练端会再算一次并校验。
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    data: bytes = Field(min_length=1)
    format: Literal["png", "jpeg"]
    expected_tokens: int | None = Field(default=None, gt=0)
    type: Literal["image"] = "image"

    @field_validator("data", mode="before")
    @classmethod
    def _decode_b64(cls, value: object) -> object:
        if isinstance(value, str):
            return base64.b64decode(value, validate=True)
        return value

    @field_serializer("data", when_used="json")
    def _encode_b64(self, value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    @property
    def length(self) -> int:
        if self.expected_tokens is None:
            raise ValueError("image chunks require expected_tokens to calculate length")
        return self.expected_tokens

    def __len__(self) -> int:
        return self.length


ModelInputChunk = Annotated[EncodedTextChunk | ImageChunk, Field(discriminator="type")]


class ModelInput(BaseModel):
    """由文本 token 片段和图片片段组成的模型输入。"""

    chunks: list[ModelInputChunk]

    @classmethod
    def from_ints(cls, tokens: Sequence[int]) -> "ModelInput":
        return cls(chunks=[EncodedTextChunk(tokens=[int(t) for t in tokens])])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    def append(self, chunk: EncodedTextChunk | ImageChunk) -> "ModelInput":
        return ModelInput(chunks=[*self.chunks, chunk])

    def append_int(self, token: int) -> "ModelInput":
        return self.append(EncodedTextChunk(tokens=[int(token)]))

    def to_ints(self) -> list[int]:
        tokens: list[int] = []
        for chunk in self.chunks:
            if isinstance(chunk, ImageChunk):
                raise ValueError("model input contains image chunks and cannot be converted to token IDs")
            tokens.extend(chunk.tokens)
        return tokens

    def tolist(self) -> list[int]:
        return self.to_ints()

    @property
    def length(self) -> int:
        return len(self)

    def __len__(self) -> int:
        total = 0
        for chunk in self.chunks:
            if isinstance(chunk, EncodedTextChunk):
                total += len(chunk.tokens)
            else:
                if chunk.expected_tokens is None:
                    raise ValueError("image chunks require expected_tokens to calculate model input length")
                total += chunk.expected_tokens
        return total

    @property
    def has_images(self) -> bool:
        return any(isinstance(c, ImageChunk) for c in self.chunks)

    @property
    def is_empty(self) -> bool:
        return not any(isinstance(c, ImageChunk) or len(c.tokens) > 0 for c in self.chunks)


class Datum(BaseModel):
    """一条训练样本：模型输入 + 按 loss 约定命名的张量。"""

    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]

    @model_validator(mode="before")
    @classmethod
    def _convert_tensors(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        inputs = data.get("loss_fn_inputs")
        if not isinstance(inputs, dict):
            return data
        return {**data, "loss_fn_inputs": {k: TensorData.model_validate(v) for k, v in inputs.items()}}


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------


class SamplingParams(BaseModel):
    """采样参数。``top_k=-1`` 表示不限制。"""

    max_tokens: int | None = None
    seed: int | None = None
    stop: str | Sequence[str] | Sequence[int] | None = None
    temperature: float = 1
    top_k: int = -1
    top_p: float = 1


class SampleSequence(BaseModel):
    """一条生成序列。``tokens`` 包含终止 token（如 ``<|im_end|>``）。"""

    stop_reason: str
    tokens: list[int]
    text: str | None = None
    logprobs: list[float]


class SampleResponse(BaseModel):
    sequences: list[SampleSequence]
    prompt_logprobs: list[float | None] = Field(default_factory=list)
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    type: Literal["sample"] = "sample"


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

LossFnOutput = dict[str, TensorData]


class ForwardBackwardOutput(BaseModel):
    loss_fn_outputs: list[LossFnOutput]
    metrics: dict[str, float]
    debug: dict[str, object] = Field(default_factory=dict)


class AdamParams(BaseModel):
    """AdamW 参数，默认值与 PyTRIO 相同。"""

    learning_rate: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class OptimStepResponse(BaseModel):
    metrics: dict[str, float]


class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    model: str
    size: int
    type: Literal["save-weights"] = "save-weights"


class SaveWeightsResponse(BaseModel):
    path: str
    model: str
    size: int
    type: Literal["save-checkpoint"] = "save-checkpoint"


class SupportedModel(BaseModel):
    model_name: str
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow", protected_namespaces=())


class GetServerCapabilitiesResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")
    supported_models: list[SupportedModel] = Field(default_factory=list)


from .futures import APIFuture  # noqa: E402  (re-export, 与 pytrio.types 一致)

__all__ = [
    "APIFuture",
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "GetServerCapabilitiesResponse",
    "ImageChunk",
    "LossFnOutput",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepResponse",
    "SampleResponse",
    "SampleSequence",
    "SamplingParams",
    "SaveWeightsForSamplerResponse",
    "SaveWeightsResponse",
    "SupportedModel",
    "TensorData",
]
