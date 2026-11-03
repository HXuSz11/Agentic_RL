"""``TrainingClient``：一个 LoRA 训练 run 的前向、反向、优化与保存。"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import queue
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import weakref

import numpy as np

from .errors import ValidationError
from .futures import APIFuture
from .losses import LOSS_INPUT_SCHEMAS, normalize_loss_fn
from .runtime import Runtime
from .sampling import SamplingClient
from .serialize import datum_to_wire
from .types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
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


_DROP_QUEUE: "queue.SimpleQueue[tuple[Any, str]]" = queue.SimpleQueue()
_DROP_THREAD: threading.Thread | None = None
_DROP_THREAD_LOCK = threading.Lock()


def _drain_drops() -> None:
    while True:
        worker, run_id = _DROP_QUEUE.get()
        try:
            if worker.alive:
                worker.call("drop_run", run_id=run_id)
        except Exception:  # noqa: BLE001
            pass


def _ensure_drop_thread() -> None:
    global _DROP_THREAD
    with _DROP_THREAD_LOCK:
        if _DROP_THREAD is None or not _DROP_THREAD.is_alive():
            _DROP_THREAD = threading.Thread(target=_drain_drops, name="localtrio-drop-run", daemon=True)
            _DROP_THREAD.start()


def _drop_run(worker: Any, run_id: str) -> None:
    """TrainingClient 被回收时调用（不能引用 client 本身）。

    finalizer 可能在任意位置被 GC 触发（甚至当前线程正持有 RPC 的锁时），所以这里
    只往可重入安全的 SimpleQueue 里放一条记录，由后台线程真正发出 drop_run。
    """
    _DROP_QUEUE.put((worker, run_id))


def _report_init(run_id: str, base_model: str, rank: int, fut: Any) -> None:
    error = fut.exception()
    if error is None:
        info = fut.result()
        print(
            f"[localtrio] training run {run_id}: base={base_model} rank={rank} "
            f"trainable={info['trainable_params'] / 1e6:.1f}M targets={','.join(info['target_modules'])}",
            flush=True,
        )
    else:
        logger.error("训练 run %s 创建失败：%s", run_id, error)
        print(f"[localtrio] training run {run_id} 创建失败：\n{error}", flush=True)


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
        engine = self._runtime.cfg.engine_for(base_model)
        if engine.enable_lora and rank > int(engine.max_lora_rank):
            # 提前报错：否则要训练到第一次采样时才会被 vLLM 拒绝
            raise ValidationError(
                f"LoRA rank {rank} 超过采样引擎的 max_lora_rank={engine.max_lora_rank}；"
                "调小 rank，或在配置的 models 下调大 max_lora_rank",
                code="validation.lora_rank",
            )
        _ensure_drop_thread()
        self.model_id = base_model
        self.model_alias = base_model
        self.base_model = base_model
        self.rank = rank
        self.training_run_cuid = self._runtime.new_run_id()
        self.run_dir = self._runtime.home / "runs" / self.training_run_cuid
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._worker = self._runtime.trainer(base_model)
        self._temp_weights: list[tuple[Path, Any]] = []
        self._vocab: int | None = None
        # 没有显式 close() 就被丢弃时，也让 trainer 释放这个 run 占用的显存
        self._finalizer = weakref.finalize(self, _drop_run, self._worker, self.training_run_cuid)
        self._finalizer.atexit = False
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
        # 回调不能引用 self：否则 client → _init → 回调 → client 形成引用环，
        # 丢弃 client 时要等循环 GC 才回收，trainer 里这个 run 的显存也迟迟不释放
        self._init.add_done_callback(functools.partial(_report_init, self.training_run_cuid, self.base_model, self.rank))

    def _call(self, method: str, **kwargs: Any) -> Any:
        if self._init.done() and self._init.exception() is not None:
            from concurrent.futures import Future

            fut: Future = Future()
            fut.set_exception(RuntimeError(f"训练 run {self.training_run_cuid} 创建失败，无法执行 {method}：\n{self._init.exception()}"))
            return fut
        return self._worker.call(method, run_id=self.training_run_cuid, **kwargs)

    def _vocab_limit(self) -> int:
        """合法 token id 的上界：max(tokenizer 长度, 模型词表大小)。"""
        if self._vocab is None:
            from .models import read_config_json

            cfg = read_config_json(self._runtime.model_dir(self.base_model))
            sizes = [len(self.get_tokenizer()), int(cfg.get("vocab_size") or 0), int((cfg.get("text_config") or {}).get("vocab_size") or 0)]
            self._vocab = max(sizes)
        return self._vocab

    @staticmethod
    def _wire_keys(loss_fn: str, datum: Datum) -> tuple[str, ...]:
        """只把当前 loss 需要的字段发给 trainer（与 PyTRIO 一致）；其余字段留在本地给 custom loss 用。"""
        keys = LOSS_INPUT_SCHEMAS[loss_fn]
        if loss_fn == "cross_entropy" and "weights" in datum.loss_fn_inputs:
            keys = (*keys, "weights")
        return keys

    def _validate_batch(self, data: list[Datum], loss_fn: str) -> str:
        """与 PyTRIO SDK 的客户端校验一致：必需字段、形状、有限值、token id 范围。"""
        if not data:
            raise ValidationError("training data cannot be empty", code="validation.empty_training_data")
        try:
            name = normalize_loss_fn(loss_fn)
        except ValueError as error:
            raise ValidationError(str(error)) from None
        vocab = self._vocab_limit()
        for idx, datum in enumerate(data):
            if datum.model_input.is_empty:
                raise ValidationError(f"training samples cannot be empty (datum {idx})", code="validation.empty_training_sample")
            for field_name in LOSS_INPUT_SCHEMAS[name]:
                if field_name not in datum.loss_fn_inputs:
                    raise ValidationError(f"{field_name} is required (datum {idx}, loss_fn={name})")
            if datum.loss_fn_inputs["target_tokens"].dtype != "int64":
                raise ValidationError(f"target_tokens must have dtype int64 (datum {idx})")
            try:
                length: int | None = len(datum.model_input)
            except ValueError:
                length = None  # 图片缺少 expected_tokens，由训练端推断
            for key in self._wire_keys(name, datum):
                tensor = datum.loss_fn_inputs[key]
                if length is not None and tensor.shape != [length]:
                    raise ValidationError(f"{key} must have shape [{length}], got {tensor.shape} (datum {idx})")
                if tensor.dtype == "float32" and not np.isfinite(np.asarray(tensor.data, dtype=np.float64)).all():
                    raise ValidationError(f"{key} contains NaN/inf (datum {idx})", code="validation.non_finite")
            targets = np.asarray(datum.loss_fn_inputs["target_tokens"].data, dtype=np.int64)
            if (((targets < 0) | (targets >= vocab)) & (targets != -100)).any():
                raise ValidationError(f"target_tokens 有超出词表的 id（datum {idx}，词表 {vocab}，-100 表示忽略）", code="validation.token_id_out_of_range")
            for chunk in datum.model_input.chunks:
                if isinstance(chunk, EncodedTextChunk) and chunk.tokens:
                    ids = np.asarray(chunk.tokens, dtype=np.int64)
                    if ((ids < 0) | (ids >= vocab)).any():
                        raise ValidationError(f"model_input 有超出词表的 token id（datum {idx}，词表 {vocab}）", code="validation.token_id_out_of_range")
        return name

    # ------------------------------------------------------------------ forward / backward
    def _forward(self, data: list[Datum], loss_fn: str, loss_fn_config: dict[str, object] | None, auto_shift: bool, backward: bool) -> APIFuture[ForwardBackwardOutput]:
        name = self._validate_batch(data, loss_fn)
        fut = self._call(
            "forward",
            batch=[datum_to_wire(d, self._wire_keys(name, d)) for d in data],
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
        for idx, lp in enumerate(logprobs):
            grad = (lp.grad if lp.grad is not None else torch.zeros_like(lp)).detach().cpu()
            if not torch.isfinite(grad).all():
                raise ValidationError(f"custom loss 对第 {idx} 条样本 logprob 的梯度含 NaN/inf", code="validation.non_finite")
            grads.append(grad.numpy())
        fut = self._call("linear_backward", batch=[datum_to_wire(d, ("target_tokens",)) for d in data], grads=grads)
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
        if path.exists() and not overwrite:
            return APIFuture.failed(ValidationError(f"checkpoint {path} 已存在（overwrite=False）"))  # type: ignore[return-value]
        # 覆盖时由 trainer 在真正执行保存时原子替换（先写临时目录再 rename），
        # 排队期间旧目录保持可用
        fut = self._call("save", path=str(path), kind="state")
        return APIFuture(fut).map(lambda r: SaveWeightsResponse(path=r["path"], model=self.base_model, size=int(r["size"])))

    async def save_state_async(self, name: str, ttl_seconds: int | None = None, overwrite: bool = False) -> APIFuture[SaveWeightsResponse]:
        return self.save_state(name, ttl_seconds, overwrite)

    def save_weights_for_sampler(self, name: str, ttl_seconds: int | None = None) -> APIFuture[SaveWeightsForSamplerResponse]:
        """保存可被 vLLM 直接加载的 PEFT adapter 目录，返回其绝对路径。"""
        path = self.run_dir / "sampler_weights" / _normalize_name(name)
        # 同名目录由 trainer 原子替换；采样端按“路径 + 修改时间”识别新版本
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
        return self._temp_client(Path(result.path))

    async def save_weights_and_get_sampling_client_async(self) -> SamplingClient:
        path = self.run_dir / "tmp_sampler" / f"temp-sampling-{uuid.uuid4().hex[:12]}"
        result = await self._save_sampler(path)
        return self._temp_client(Path(result.path))

    def _temp_client(self, path: Path) -> SamplingClient:
        client = self.create_sampling_client(str(path))
        self._temp_weights.append((path, weakref.ref(client)))
        self._rotate_temp()
        return client

    def _rotate_temp(self) -> None:
        """只保留最近 4 份临时权重；更早的目录在没有 SamplingClient 引用它之后才删除。

        vLLM 会按需（以及 LRU 淘汰后）从磁盘重新加载 adapter，所以仍在使用的目录不能删。
        """
        keep = self._temp_weights[-_KEEP_TEMP_WEIGHTS:]
        older = self._temp_weights[:-_KEEP_TEMP_WEIGHTS]
        alive = []
        for path, ref in older:
            if ref() is None:
                shutil.rmtree(path, ignore_errors=True)
            else:
                alive.append((path, ref))
        self._temp_weights = alive + keep

    # ------------------------------------------------------------------ misc
    def get_tokenizer(self) -> Any:
        return self._runtime.tokenizer(self.base_model)

    def get_info(self) -> dict[str, Any]:
        return {"run_id": self.training_run_cuid, "base_model": self.base_model, "run_dir": str(self.run_dir), **self._init.result()}

    def close(self) -> None:
        """释放 trainer 里这个 run 的 adapter、优化器状态和梯度，并清理临时采样权重。"""
        self._finalizer.detach()
        try:
            self._worker.call("drop_run", run_id=self.training_run_cuid).result(timeout=60)
        except Exception:  # noqa: BLE001
            pass
        # 仍被 SamplingClient 引用的临时权重保留（close 之后还可能用它做最终评测），
        # 在那个 client 被回收时再删除目录（rmtree 不涉及 RPC 锁，放在 finalizer 里安全）
        for path, ref in self._temp_weights:
            client = ref()
            if client is None:
                shutil.rmtree(path, ignore_errors=True)
            else:
                weakref.finalize(client, shutil.rmtree, str(path), True)
        self._temp_weights = []

    async def close_async(self) -> None:
        await asyncio.to_thread(self.close)

    def __repr__(self) -> str:
        return f"TrainingClient(run={self.training_run_cuid!r}, base_model={self.base_model!r})"
