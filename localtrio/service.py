"""``ServiceClient``：创建训练与采样客户端的入口（对应 PyTRIO 的根客户端）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .runtime import Runtime
from .sampling import SamplingClient
from .training import TrainingClient
from .types import GetServerCapabilitiesResponse, SupportedModel


def _state_meta(path: str) -> dict[str, Any]:
    if path.startswith("trio://"):
        raise ValidationError(f"{path} 是 PyTRIO 云端 checkpoint；本地版请使用 save_state 返回的本地目录")
    meta_path = Path(path).expanduser() / "localtrio_meta.json"
    if not meta_path.exists():
        raise ValidationError(f"{path} 不是 localtrio checkpoint（缺少 localtrio_meta.json）")
    return json.loads(meta_path.read_text())


class ServiceClient:
    """与 ``pytrio.ServiceClient`` 同名同参；``api_key`` 等远程参数会被忽略。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._runtime = Runtime.get()

    # ------------------------------------------------------------------ info
    def get_supported_models(self) -> list[str]:
        models = set(self._runtime.cfg.model_paths) | {m for m in self._runtime.cfg.models if m != "default"}
        return sorted(models) or ["<任意 HuggingFace / ModelScope 模型名或本地路径>"]

    def get_server_capabilities(self) -> GetServerCapabilitiesResponse:
        return GetServerCapabilitiesResponse(supported_models=[SupportedModel(model_name=m) for m in self.get_supported_models()])

    async def get_server_capabilities_async(self) -> GetServerCapabilitiesResponse:
        return self.get_server_capabilities()

    # ------------------------------------------------------------------ training
    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        **_: Any,
    ) -> TrainingClient:
        return TrainingClient(
            base_model,
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            runtime=self._runtime,
        )

    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, seed: int | None = None, train_mlp: bool = True, train_attn: bool = True, train_unembed: bool = True, **kwargs: Any) -> TrainingClient:
        return self.create_lora_training_client(base_model, rank, seed, train_mlp, train_attn, train_unembed, **kwargs)

    def _from_state(self, path: str, load_optimizer: bool) -> TrainingClient:
        meta = _state_meta(path)
        return TrainingClient(
            meta["base_model"],
            rank=int(meta["rank"]),
            seed=meta.get("seed"),
            train_mlp=bool(meta["train_mlp"]),
            train_attn=bool(meta["train_attn"]),
            train_unembed=bool(meta.get("train_unembed", False)),
            init_state=str(Path(path).expanduser().resolve()),
            load_optimizer=load_optimizer,
            runtime=self._runtime,
        )

    def create_training_client_from_state(self, path: str) -> TrainingClient:
        """从 ``save_state`` 目录恢复 LoRA 权重（不恢复优化器）。"""
        return self._from_state(path, load_optimizer=False)

    async def create_training_client_from_state_async(self, path: str) -> TrainingClient:
        return self._from_state(path, load_optimizer=False)

    def create_training_client_from_state_with_optimizer(self, path: str) -> TrainingClient:
        """从 ``save_state`` 目录恢复 LoRA 权重和 AdamW 状态。"""
        return self._from_state(path, load_optimizer=True)

    async def create_training_client_from_state_with_optimizer_async(self, path: str) -> TrainingClient:
        return self._from_state(path, load_optimizer=True)

    # ------------------------------------------------------------------ sampling
    def create_sampling_client(self, base_model: str | None = None, model_path: str | None = None, **_: Any) -> SamplingClient:
        if base_model is None:
            if model_path is None:
                raise ValidationError("create_sampling_client 需要 base_model 或 model_path")
            base_model = _state_meta(model_path)["base_model"]
        return SamplingClient(base_model, model_path=model_path, runtime=self._runtime)

    async def create_sampling_client_async(self, base_model: str | None = None, model_path: str | None = None, **kwargs: Any) -> SamplingClient:
        return self.create_sampling_client(base_model, model_path, **kwargs)

    def create_rest_client(self) -> Any:
        raise NotImplementedError("本地版没有 REST 控制面；checkpoint 就是 LOCALTRIO_HOME/runs 下的目录")

    def close(self) -> None:
        """释放全部 GPU 引擎（脚本结束时也会自动释放）。"""
        Runtime.reset()
