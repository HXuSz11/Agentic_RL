"""``TrainingClient``：一个 LoRA 训练 run 的前向、反向、优化与保存。"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ValidationError
from .futures import APIFuture
from .runtime import Runtime
from .sampling import SamplingClient
from .serialize import datum_to_wire
from .types import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    OptimStepResponse,
    SaveWeightsForSamplerResponse,
    SaveWeightsResponse,
    TensorData,
)

logger = logging.getLogger("localtrio")

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?$")
_KEEP_TEMP_WEIGHTS = 4


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValidationError("checkpoint name must be a string")
    normalized = name.replace(" ", "-")
    if not _NAME_PATTERN.fullmatch(normalized):
        raise ValidationError(f"checkpoint name {name!r} 只能包含字母、数字、. _ -")
    return normalized[:200]


def _to_output(raw: dict[str, Any], extra_metrics: dict[str, float] | None = None) -> ForwardBackwardOutput:
    outputs = [
        {key: TensorData.from_numpy(np.asarray(value)) for key, value in item.items()} for item in raw["loss_fn_outputs"]
    ]
    metrics = {k: float(v) for k, v in raw["metrics"].items()}
    if extra_metrics:
        metrics.update({k: float(v) for k, v in extra_metrics.items()})
    return ForwardBackwardOutput(loss_fn_outputs=outputs, metrics=metrics)


def _invoke_custom_loss(loss_fn: Callable, data: list[Datum], logprobs: list[Any], config: dict[str, object] | None) -> Any:
    if config is None:
        return loss_fn(data, logprobs)
    try:
        inspect.signature(loss_fn).bind(data, logprobs, config)
    except (TypeError, ValueError):
        return loss_fn(data, logprobs)
    return loss_fn(data, logprobs, config)


class TrainingClient:
    """与 ``pytrio.TrainingClient`` 同名同参。

    所有调用按提交顺序在 trainer 进程中串行执行，同步方法返回 :class:`APIFuture`，
    ``*_async`` 方法 ``await`` 后同样得到 :class:`APIFuture`（与 PyTRIO 一致）。
    """

    def __init__(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        init_state: str | None = None,
        load_optimizer: bool = False,
        runtime: Runtime | None = None,
    ) -> None:
        self._runtime = runtime or Runtime.get()
        self.model_id = base_model
        self.model_alias = base_model
        self.base_model = base_model
        self.rank = rank
        self.training_run_cuid = self._runtime.new_run_id()
        self.run_dir = self._runtime.home / "runs" / self.training_run_cuid
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._worker = self._runtime.trainer(base_model)
        self._temp_weights: list[Path] = []
        self._init = self._worker.call(
            "create_run",
            run_id=self.training_run_cuid,
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            init_state=init_state,
            load_optimizer=load_optimizer,
        )
        self._init.add_done_callback(self._report_init)

    def _report_init(self, fut: Any) -> None:
        if fut.exception() is None:
            info = fut.result()
            print(
                f"[localtrio] training run {self.training_run_cuid}: base={self.base_model} rank={self.rank} "
                f"trainable={info['trainable_params'] / 1e6:.1f}M targets={','.join(info['target_modules'])}",
                flush=True,
            )

    def _call(self, method: str, **kwargs: Any) -> Any:
        return self._worker.call(method, run_id=self.training_run_cuid, **kwargs)

    @staticmethod
    def _validate_batch(data: list[Datum]) -> None:
        if not data:
            raise ValidationError("training data cannot be empty", code="validation.empty_training_data")
        for idx, datum in enumerate(data):
            if "target_tokens" not in datum.loss_fn_inputs:
                raise ValidationError(f"target_tokens is required (datum {idx})")
            target = datum.loss_fn_inputs["target_tokens"]
            if target.dtype != "int64":
                raise ValidationError("target_tokens must have dtype int64")
            if datum.model_input.is_empty:
                raise ValidationError("training samples cannot be empty")

    # ------------------------------------------------------------------ forward / backward
    def _forward(self, data: list[Datum], loss_fn: str, loss_fn_config: dict[str, object] | None, auto_shift: bool, backward: bool) -> APIFuture[ForwardBackwardOutput]:
        self._validate_batch(data)
        fut = self._call(
            "forward",
            batch=[datum_to_wire(d) for d in data],
            loss_fn=loss_fn,
            loss_fn_config=dict(loss_fn_config or {}),
            auto_shift=bool(auto_shift),
            backward=backward,
        )
        return APIFuture(fut).map(_to_output)

    def forward(self, data: list[Datum], loss_fn: str = "cross_entropy", loss_fn_config: dict[str, object] | None = None, auto_shift: bool = False) -> APIFuture[ForwardBackwardOutput]:
        """只做前向（不累积梯度），返回逐样本 ``logprobs`` 与 loss 指标。"""
        return self._forward(data, loss_fn, loss_fn_config, auto_shift, backward=False)

    async def forward_async(self, data: list[Datum], loss_fn: str = "cross_entropy", loss_fn_config: dict[str, object] | None = None, auto_shift: bool = False) -> APIFuture[ForwardBackwardOutput]:
        return self.forward(data, loss_fn, loss_fn_config, auto_shift)

    def forward_backward(self, data: list[Datum], loss_fn: str = "cross_entropy", loss_fn_config: dict[str, object] | None = None, auto_shift: bool = False) -> APIFuture[ForwardBackwardOutput]:
        """前向 + 反向，梯度在 trainer 里累积，直到下一次 ``optim_step``。"""
        return self._forward(data, loss_fn, loss_fn_config, auto_shift, backward=True)

    async def forward_backward_async(self, data: list[Datum], loss_fn: str = "cross_entropy", loss_fn_config: dict[str, object] | None = None, auto_shift: bool = False) -> APIFuture[ForwardBackwardOutput]:
        return self.forward_backward(data, loss_fn, loss_fn_config, auto_shift)

    def _custom_second_phase(self, data: list[Datum], forward_result: ForwardBackwardOutput, loss_fn: Callable, loss_fn_config: dict[str, object] | None) -> APIFuture[ForwardBackwardOutput]:
        import torch

        logprobs = []
        for output in forward_result.loss_fn_outputs:
            raw = output["logprobs"]
            logprobs.append(torch.tensor(raw.data, dtype=torch.float32).reshape(raw.shape).requires_grad_(True))
        loss, custom_metrics = _invoke_custom_loss(loss_fn, data, logprobs, loss_fn_config)
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            raise ValidationError("Custom loss output has no gradients. Ensure loss_fn returns a tensor with gradients.")
        loss.backward()
        grads = []
        for lp in logprobs:
            grads.append((lp.grad if lp.grad is not None else torch.zeros_like(lp)).detach().cpu().numpy())
        fut = self._call("linear_backward", batch=[datum_to_wire(d) for d in data], grads=grads)
        return APIFuture(fut).map(lambda raw: _to_output(raw, dict(custom_metrics or {})))

    def forward_backward_custom(self, data: list[Datum], loss_fn: Callable, loss_fn_config: dict[str, object] | None = None) -> APIFuture[ForwardBackwardOutput]:
        """自定义 loss：先前向拿到可求导 logprob，本地算 loss，再把梯度反传到 LoRA。"""
        forward_result = self.forward(data, "cross_entropy").result()
        return self._custom_second_phase(data, forward_result, loss_fn, loss_fn_config)

    async def forward_backward_custom_async(self, data: list[Datum], loss_fn: Callable, loss_fn_config: dict[str, object] | None = None) -> APIFuture[ForwardBackwardOutput]:
        forward_result = await self.forward(data, "cross_entropy")
        return self._custom_second_phase(data, forward_result, loss_fn, loss_fn_config)

    # ------------------------------------------------------------------ optimizer
    def optim_step(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        fut = self._call("optim_step", adam=adam_params.model_dump())
        return APIFuture(fut).map(lambda metrics: OptimStepResponse(metrics=metrics))

    async def optim_step_async(self, adam_params: AdamParams) -> APIFuture[OptimStepResponse]:
        return self.optim_step(adam_params)

    # ------------------------------------------------------------------ checkpoints
    def save_state(self, name: str, ttl_seconds: int | None = None, overwrite: bool = False) -> APIFuture[SaveWeightsResponse]:
        """保存 LoRA 权重 + AdamW 状态，可用 ``create_training_client_from_state`` 恢复。"""
        path = self.run_dir / "states" / _normalize_name(name)
        if path.exists():
            if not overwrite:
                return APIFuture.failed(ValidationError(f"checkpoint {path} 已存在（overwrite=False）"))  # type: ignore[return-value]
            shutil.rmtree(path)
        fut = self._call("save", path=str(path), kind="state")
        return APIFuture(fut).map(lambda r: SaveWeightsResponse(path=r["path"], model=self.base_model, size=int(r["size"])))

    async def save_state_async(self, name: str, ttl_seconds: int | None = None, overwrite: bool = False) -> APIFuture[SaveWeightsResponse]:
        return self.save_state(name, ttl_seconds, overwrite)

    def save_weights_for_sampler(self, name: str, ttl_seconds: int | None = None) -> APIFuture[SaveWeightsForSamplerResponse]:
        """保存可被 vLLM 直接加载的 PEFT adapter 目录，返回其绝对路径。"""
        path = self.run_dir / "sampler_weights" / _normalize_name(name)
        if path.exists():
            shutil.rmtree(path)
        return self._save_sampler(path)

    def _save_sampler(self, path: Path) -> APIFuture[SaveWeightsForSamplerResponse]:
        fut = self._call("save", path=str(path), kind="sampler")
        return APIFuture(fut).map(lambda r: SaveWeightsForSamplerResponse(path=r["path"], model=self.base_model, size=int(r["size"])))

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None) -> APIFuture[SaveWeightsForSamplerResponse]:
        return self.save_weights_for_sampler(name, ttl_seconds)

    def load_state(self, path: str) -> APIFuture[dict[str, object]]:
        fut = self._call("load_state", path=str(path), load_optimizer=False)
        return APIFuture(fut).map(lambda _: {})

    async def load_state_async(self, path: str) -> APIFuture[dict[str, object]]:
        return self.load_state(path)

    def load_state_with_optimizer(self, path: str) -> APIFuture[dict[str, object]]:
        fut = self._call("load_state", path=str(path), load_optimizer=True)
        return APIFuture(fut).map(lambda _: {})

    async def load_state_with_optimizer_async(self, path: str) -> APIFuture[dict[str, object]]:
        return self.load_state_with_optimizer(path)

    # ------------------------------------------------------------------ sampling
    def create_sampling_client(self, model_path: str) -> SamplingClient:
        return SamplingClient(self.base_model, model_path=model_path, runtime=self._runtime)

    async def create_sampling_client_async(self, model_path: str) -> SamplingClient:
        return self.create_sampling_client(model_path)

    def save_weights_and_get_sampling_client(self) -> SamplingClient:
        """保存当前权重到临时目录并返回绑定它的 SamplingClient（每个 RL step 调用一次）。"""
        path = self.run_dir / "tmp_sampler" / f"temp-sampling-{uuid.uuid4().hex[:12]}"
        result = self._save_sampler(path).result()
        self._rotate_temp(Path(result.path))
        return self.create_sampling_client(result.path)

    async def save_weights_and_get_sampling_client_async(self) -> SamplingClient:
        path = self.run_dir / "tmp_sampler" / f"temp-sampling-{uuid.uuid4().hex[:12]}"
        result = await self._save_sampler(path)
        self._rotate_temp(Path(result.path))
        return self.create_sampling_client(result.path)

    def _rotate_temp(self, path: Path) -> None:
        self._temp_weights.append(path)
        while len(self._temp_weights) > _KEEP_TEMP_WEIGHTS:
            shutil.rmtree(self._temp_weights.pop(0), ignore_errors=True)

    # ------------------------------------------------------------------ misc
    def get_tokenizer(self) -> Any:
        return self._runtime.tokenizer(self.base_model)

    def get_info(self) -> dict[str, Any]:
        return {"run_id": self.training_run_cuid, "base_model": self.base_model, "run_dir": str(self.run_dir), **self._init.result()}

    def close(self) -> None:
        try:
            self._worker.call("drop_run", run_id=self.training_run_cuid).result(timeout=60)
        except Exception:  # noqa: BLE001
            pass

    async def close_async(self) -> None:
        await asyncio.to_thread(self.close)

    def __repr__(self) -> str:
        return f"TrainingClient(run={self.training_run_cuid!r}, base_model={self.base_model!r})"
