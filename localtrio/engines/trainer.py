"""LoRA 训练引擎：HF transformers + PEFT，运行在 trainer 子进程里。

一个 trainer 进程加载一份基模，可以同时托管多个训练 run（每个 run 是一个独立的
LoRA adapter + 自己的 AdamW 状态）。这样 Medical OPD 里“一个 state 做教师、另一个
继续训练”的写法不需要加载两份基模。

关键实现：

- **logprob 计算**：把 ``lm_head`` 临时替换为恒等映射拿到最后一层 hidden states，
  再按 ``logits_chunk_size`` 分块计算 ``log_softmax`` 并 gather，配合 checkpoint，
  不会一次性物化 ``[seq_len, vocab]`` 的 logits（Qwen3.5 词表约 25 万）。
- **micro-batch**：按长度排序后右 padding 打包，单批 token 上限
  ``max_tokens_per_microbatch``；一次 ``forward_backward`` 内的所有 micro-batch
  共享同一个 logical-batch 分母，梯度等价于整批计算。
- **图片**：``ImageChunk`` 展开为 ``expected_tokens`` 个 ``<|image_pad|>``，像素由
  与基模匹配的 image processor 处理；视觉塔不加 LoRA。
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from .. import losses
from ..models import image_token_id, is_multimodal, load_hf_model, load_image_processor, load_tokenizer
from .common import expand_chunks

logger = logging.getLogger("localtrio.trainer")

STANDARD_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
STANDARD_MLP = ("gate_proj", "up_proj", "down_proj")
# Qwen3.5 / Qwen3-Next 的线性注意力（Gated DeltaNet）投影；all-linear 模式才会加 LoRA
LINEAR_ATTN = ("in_proj_qkv", "in_proj_z", "in_proj_qkvz", "in_proj_ba", "in_proj_a", "in_proj_b", "out_proj")
VISION_EXCLUDE = r".*(visual|vision_tower|vision_model|vision_encoder|mm_projector|multi_modal_projector|merger|audio).*"


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: D401
        return x


@dataclass
class Prepared:
    input_ids: torch.Tensor  # [L]
    inputs: dict[str, torch.Tensor]  # target_tokens / weights / logprobs / advantages
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())

    @property
    def has_images(self) -> bool:
        return self.pixel_values is not None


@dataclass
class Run:
    run_id: str
    adapter: str
    base_model: str
    rank: int
    lora_config: Any
    params: list[nn.Parameter] = field(default_factory=list)
    optimizer: torch.optim.Optimizer | None = None
    step: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _chunk_logprobs(head: nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 直接调用输出层模块：lm_head 挂了 LoRA（train_unembed）时也能正确计算
    logits = head(hidden).float()
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)


class TrainerEngine:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        cfg = spec["trainer"]
        self.cfg = cfg
        self.base_model = spec["base_model"]
        self.model_dir = spec["model_dir"]
        self.device_count = torch.cuda.device_count()
        torch.manual_seed(int(cfg.get("seed", 0)))

        if self.device_count == 0:
            device_map: Any = {"": "cpu"}
            dtype = "float32" if cfg.get("dtype", "bfloat16") == "bfloat16" and os.environ.get("LOCALTRIO_CPU_BF16") != "1" else cfg["dtype"]
        elif self.device_count == 1:
            device_map = {"": 0}
            dtype = cfg.get("dtype", "bfloat16")
        else:
            device_map = "auto"
            dtype = cfg.get("dtype", "bfloat16")
        self.dtype = dtype
        t0 = time.time()
        model = load_hf_model(
            self.model_dir,
            dtype=dtype,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            device_map=device_map,
        )
        model.config.use_cache = False
        for p in model.parameters():
            p.requires_grad_(False)
        if cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        self.base = model
        self.model: Any = model  # 第一个 run 创建后变成 PeftModel
        self.runs: dict[str, Run] = {}
        self.tokenizer = load_tokenizer(self.model_dir)
        self.multimodal = is_multimodal(self.model_dir)
        self.image_processor = load_image_processor(self.model_dir) if self.multimodal else None
        self.image_token_id = image_token_id(self.model_dir, self.tokenizer) if self.multimodal else None
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self._accepts_mm_types = "mm_token_type_ids" in inspect.signature(model.forward).parameters
        # 在挂 LoRA 之前记录语言模型里的 Linear 名称（挂上之后类型会变成 lora.Linear）
        vision = re.compile(VISION_EXCLUDE)
        self._linear_names = {
            name.rsplit(".", 1)[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and not vision.fullmatch(name)
        }
        self._has_vision = any(vision.fullmatch(name) for name, _ in model.named_modules())
        logger.info("trainer loaded %s in %.1fs (dtype=%s)", self.base_model, time.time() - t0, dtype)

    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "model_dir": self.model_dir,
            "devices": self.device_count,
            "dtype": self.dtype,
            "multimodal": self.multimodal,
            "vocab_size": len(self.tokenizer),
        }

    @property
    def input_device(self) -> torch.device:
        return self.base.get_input_embeddings().weight.device

    # ------------------------------------------------------------------ runs
    def _target_modules(self, train_mlp: bool, train_attn: bool) -> list[str]:
        mode = self.cfg.get("lora_target_modules", "all-linear")
        if isinstance(mode, list):
            return list(mode)
        names: list[str] = []
        if train_attn:
            names.extend(STANDARD_ATTN)
            if mode == "all-linear":
                names.extend(LINEAR_ATTN)
        if train_mlp:
            names.extend(STANDARD_MLP)
        selected = [n for n in names if n in self._linear_names]
        if not selected:
            raise ValueError(f"在 {self.base_model} 中找不到可加 LoRA 的模块（候选 {names}）")
        return selected

    def create_run(
        self,
        run_id: str,
        rank: int,
        seed: int | None,
        train_mlp: bool,
        train_attn: bool,
        train_unembed: bool,
        init_state: str | None = None,
        load_optimizer: bool = False,
    ) -> dict[str, Any]:
        from peft import LoraConfig, get_peft_model

        if init_state is not None:
            meta = json.loads((Path(init_state) / "localtrio_meta.json").read_text())
            rank = int(meta["rank"])
            train_mlp, train_attn = bool(meta["train_mlp"]), bool(meta["train_attn"])
            train_unembed = bool(meta.get("train_unembed", False))
            if meta.get("base_model") not in (None, self.base_model):
                raise ValueError(f"checkpoint 基模 {meta.get('base_model')} 与当前 {self.base_model} 不一致")
        if not 1 <= rank <= 128:
            raise ValueError("LoRA rank 需要在 [1, 128] 范围内（PyTRIO 为 [4, 64]）")
        if train_unembed and not self.cfg.get("allow_unembed", False):
            # PyTRIO 默认 train_unembed=True。vLLM 只对部分模型类声明了 lm_head LoRA
            # （Qwen3.5 多模态类没有），所以默认不训练输出层，避免采样端加载 adapter 失败。
            logger.info("train_unembed=True 被忽略（设置 trainer.allow_unembed: true 可开启，需要 vLLM 支持该模型的 lm_head LoRA）")
            train_unembed = False
        targets = self._target_modules(train_mlp, train_attn)
        if train_unembed:
            # 以 LoRA 的形式训练输出层（vLLM 不支持 modules_to_save 形式的 lm_head）
            targets = [*targets, "lm_head"]
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=float(self.cfg.get("lora_alpha", 32.0)),
            lora_dropout=float(self.cfg.get("lora_dropout", 0.0)),
            target_modules=targets,
            exclude_modules=VISION_EXCLUDE if self._has_vision else None,
            bias="none",
            task_type="CAUSAL_LM",
        )
        adapter = "r" + re.sub(r"[^0-9a-zA-Z]", "", run_id)[-24:]
        torch.manual_seed(int(seed) if seed is not None else int(self.cfg.get("seed", 0)) + len(self.runs))
        if not self.runs and not hasattr(self.model, "peft_config"):
            self.model = get_peft_model(self.base, lora_cfg, adapter_name=adapter)
        else:
            self.model.add_adapter(adapter, lora_cfg)
        run = Run(run_id=run_id, adapter=adapter, base_model=self.base_model, rank=rank, lora_config=lora_cfg)
        run.meta = {
            "base_model": self.base_model,
            "rank": rank,
            "seed": seed,
            "train_mlp": train_mlp,
            "train_attn": train_attn,
            "train_unembed": train_unembed,
            "target_modules": targets,
            "lora_alpha": lora_cfg.lora_alpha,
        }
        self.runs[run_id] = run
        self._activate(run)
        run.params = [p for n, p in self.model.named_parameters() if f".{adapter}." in n or n.endswith(f".{adapter}")]
        if not run.params:
            raise RuntimeError("LoRA 参数为空，检查 target_modules 配置")
        if init_state is not None:
            self._load_state(run, Path(init_state), load_optimizer)
        n_params = sum(p.numel() for p in run.params)
        return {"run_id": run_id, "trainable_params": n_params, "target_modules": targets}

    def drop_run(self, run_id: str) -> bool:
        run = self.runs.pop(run_id, None)
        if run is None:
            return False
        try:
            self.model.delete_adapter(run.adapter)
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def _activate(self, run: Run) -> None:
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(run.adapter)
        for name, param in self.model.named_parameters():
            if "lora_" in name or "modules_to_save" in name:
                param.requires_grad_(f".{run.adapter}." in name or name.endswith(f".{run.adapter}"))

    # ------------------------------------------------------------------ inputs
    def _prepare(self, datum: dict[str, Any]) -> Prepared:
        if not self.multimodal and any(c[0] != "text" for c in datum["chunks"]):
            raise ValueError(f"{self.base_model} 不是多模态模型，不能接受 ImageChunk")
        ids, pixel_values, grid, _ = expand_chunks(datum["chunks"], self.image_processor, self.image_token_id)
        length = len(ids)
        tensors: dict[str, torch.Tensor] = {}
        for key, value in datum["inputs"].items():
            arr = np.asarray(value)
            if arr.shape != (length,):
                raise ValueError(f"{key} must have shape [{length}], got {list(arr.shape)}")
            tensors[key] = torch.from_numpy(arr.astype(np.int64 if key == "target_tokens" else np.float32))
        return Prepared(
            input_ids=torch.tensor(ids, dtype=torch.long),
            inputs=tensors,
            pixel_values=pixel_values,
            image_grid_thw=grid,
        )

    def _microbatches(self, prepared: list[Prepared]) -> list[list[int]]:
        budget = int(self.cfg.get("max_tokens_per_microbatch", 16384))
        order = sorted(range(len(prepared)), key=lambda i: -prepared[i].length)
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for idx in order:
            item = prepared[idx]
            # 图文样本与纯文本样本一样按 token 预算打包：pixel_values 按样本顺序拼接，
            # 模型内部按 image_pad 出现顺序把视觉特征填回各自位置。
            new_max = max(current_max, item.length)
            if current and new_max * (len(current) + 1) > budget:
                batches.append(current)
                current, current_max = [], 0
                new_max = item.length
            current.append(idx)
            current_max = new_max
        if current:
            batches.append(current)
        return batches

    @contextmanager
    def _hidden_head(self):
        """暂时把 lm_head 换成恒等映射，让 forward 直接输出最后一层 hidden states。"""
        head = self.base.get_output_embeddings()
        parent, attr = None, None
        for name, module in self.base.named_modules():
            for child_name, child in module.named_children():
                if child is head:
                    parent, attr = module, child_name
                    break
            if parent is not None:
                break
        if parent is None:
            raise RuntimeError("找不到 lm_head")
        setattr(parent, attr, _Identity())
        try:
            yield head
        finally:
            setattr(parent, attr, head)

    def _token_logprobs(self, items: list[Prepared], grad: bool) -> list[torch.Tensor]:
        device = self.input_device
        max_len = max(item.length for item in items)
        batch = len(items)
        input_ids = torch.full((batch, max_len), self.pad_id, dtype=torch.long)
        attention = torch.zeros((batch, max_len), dtype=torch.long)
        targets = torch.zeros((batch, max_len), dtype=torch.long)
        for b, item in enumerate(items):
            n = item.length
            input_ids[b, :n] = item.input_ids
            attention[b, :n] = 1
            t = item.inputs["target_tokens"].clone()
            t[t < 0] = 0
            targets[b, :n] = t
        kwargs: dict[str, Any] = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention.to(device),
            "use_cache": False,
        }
        if any(item.has_images for item in items):
            kwargs["pixel_values"] = torch.cat([i.pixel_values for i in items if i.pixel_values is not None]).to(device)
            kwargs["image_grid_thw"] = torch.cat([i.image_grid_thw for i in items if i.image_grid_thw is not None]).to(device)
            if self._accepts_mm_types:
                kwargs["mm_token_type_ids"] = (input_ids == int(self.image_token_id)).to(torch.int32).to(device)

        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, self._hidden_head() as head:
            out = self.model(**kwargs)
            hidden = out.logits if hasattr(out, "logits") else out[0]
            head_device = next(head.parameters()).device
            # 只保留真实 token（去掉右侧 padding）并展平成 [N, H]，再按总 token 数分块：
            # 每块 logits 为 [chunk, vocab]，与 batch 里有多少条序列无关。
            valid = attention.bool().to(hidden.device)
            flat_hidden = hidden[valid].to(head_device)
            flat_targets = targets.to(hidden.device)[valid].to(head_device)
            chunk = int(self.cfg.get("logits_chunk_size", 1024))
            pieces = []
            for start in range(0, flat_hidden.shape[0], chunk):
                h = flat_hidden[start : start + chunk]
                t = flat_targets[start : start + chunk]
                if grad:
                    lp = torch.utils.checkpoint.checkpoint(_chunk_logprobs, head, h, t, use_reentrant=False)
                else:
                    lp = _chunk_logprobs(head, h, t)
                pieces.append(lp)
            flat_logprobs = torch.cat(pieces)
        return list(flat_logprobs.split([item.length for item in items]))

    # ------------------------------------------------------------------ API
    def forward(
        self,
        run_id: str,
        batch: list[dict[str, Any]],
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None,
        auto_shift: bool,
        backward: bool,
    ) -> dict[str, Any]:
        run = self._run(run_id)
        self._activate(run)
        loss_fn = losses.normalize_loss_fn(loss_fn)
        prepared = [self._prepare(d) for d in batch]
        for idx, item in enumerate(prepared):
            for name in losses.LOSS_INPUT_SCHEMAS[loss_fn]:
                if name not in item.inputs:
                    raise ValueError(f"{name} is required (datum {idx}, loss_fn={loss_fn})")
        if auto_shift:
            prepared = [self._shift(item) for item in prepared]
        counts = losses.count_tokens(loss_fn, [p.inputs for p in prepared], auto_shift=False)
        denom = losses.denominator(loss_fn, counts, self.cfg.get("loss_normalization", "active_tokens"))
        return self._run_batches(prepared, loss_fn, loss_fn_config, backward, denom, counts)

    @staticmethod
    def _shift(item: Prepared) -> Prepared:
        """auto_shift：target 与输入逐位对齐，位置 i 改为预测 target[i+1]，丢弃首个 label。"""
        item.inputs = {
            k: torch.cat([v[1:], torch.full((1,), -100 if k == "target_tokens" else 0, dtype=v.dtype)])
            for k, v in item.inputs.items()
        }
        return item

    def _run_batches(
        self,
        prepared: list[Prepared],
        loss_fn: str,
        loss_fn_config: dict[str, Any] | None,
        backward: bool,
        denom: float,
        counts: losses.TokenCounts,
        linear_grads: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        outputs: list[dict[str, np.ndarray] | None] = [None] * len(prepared)
        loss_sum = 0.0
        for mb in self._microbatches(prepared):
            items = [prepared[i] for i in mb]
            lps = self._token_logprobs(items, grad=backward)
            total: torch.Tensor | None = None
            for idx, item, lp in zip(mb, items, lps, strict=True):
                inputs = {k: v.to(lp.device) for k, v in item.inputs.items()}
                mask = losses.valid_mask(inputs["target_tokens"], auto_shift=False)
                if linear_grads is None:
                    per_token = losses.per_token_loss(loss_fn, lp, inputs, loss_fn_config)
                else:
                    g = linear_grads[idx].to(lp.device, torch.float32)
                    per_token = torch.where(mask, g * lp.float(), torch.zeros_like(lp, dtype=torch.float32))
                loss_sum += float(per_token.detach().sum().item())
                total = per_token.sum() if total is None else total + per_token.sum()
                outputs[idx] = {
                    "logprobs": torch.where(mask, lp.detach().float(), torch.zeros_like(lp, dtype=torch.float32)).cpu().numpy(),
                    "elementwise_loss": per_token.detach().float().cpu().numpy(),
                }
            if backward and total is not None:
                (total / denom).backward()
        token_count = counts.total_active_tokens
        metrics = {
            "loss_sum": loss_sum,
            "loss_mean": loss_sum / token_count if token_count > 0 else 0.0,
            "token_count": float(token_count),
        }
        return {"loss_fn_outputs": outputs, "metrics": metrics}

    def linear_backward(self, run_id: str, batch: list[dict[str, Any]], grads: list[np.ndarray]) -> dict[str, Any]:
        """custom loss 第二阶段：以 ``sum(g * logprob)`` 为代理 loss 反向。

        ``g = dL/dlogprob`` 来自用户 loss 在主进程里的 autograd，代理 loss 对参数的梯度
        与用户 loss 完全相同（与 PyTRIO 把 ``-g`` 作为 cross_entropy 权重是同一个技巧）。
        """
        run = self._run(run_id)
        self._activate(run)
        prepared = [self._prepare(d) for d in batch]
        grad_tensors = [torch.from_numpy(np.asarray(g, dtype=np.float32).reshape(-1)) for g in grads]
        for item, g in zip(prepared, grad_tensors, strict=True):
            if g.numel() != item.length:
                raise ValueError(f"custom loss 梯度长度 {g.numel()} 与样本长度 {item.length} 不一致")
            item.inputs["weights"] = -g
        counts = losses.count_tokens("cross_entropy", [p.inputs for p in prepared], auto_shift=False)
        mode = self.cfg.get("custom_loss_normalization", "none")
        if mode == "same_as_builtin":
            mode = self.cfg.get("loss_normalization", "active_tokens")
        denom = losses.denominator("cross_entropy", counts, mode)
        return self._run_batches(prepared, "cross_entropy", None, True, denom, counts, linear_grads=grad_tensors)

    def optim_step(self, run_id: str, adam: dict[str, float]) -> dict[str, float]:
        run = self._run(run_id)
        lr = float(adam.get("learning_rate", 1e-4))
        betas = (float(adam.get("beta1", 0.9)), float(adam.get("beta2", 0.95)))
        eps = float(adam.get("eps", 1e-12))
        wd = float(adam.get("weight_decay", 0.0))
        if run.optimizer is None:
            run.optimizer = torch.optim.AdamW(run.params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        for group in run.optimizer.param_groups:
            group.update(lr=lr, betas=betas, eps=eps, weight_decay=wd)
        clip = float(adam.get("grad_clip_norm", 0.0))
        with_grad = [p for p in run.params if p.grad is not None]
        if with_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(with_grad, max_norm=clip if clip > 0 else math.inf)
            grad_norm_value = float(grad_norm.item())
        else:
            grad_norm_value = 0.0
        run.optimizer.step()
        run.optimizer.zero_grad(set_to_none=True)
        run.step += 1
        return {"grad_norm": grad_norm_value, "learning_rate": lr, "step": float(run.step)}

    # ------------------------------------------------------------------ checkpoints
    def _adapter_state(self, run: Run, dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(self.model, adapter_name=run.adapter)
        out = {}
        for key, value in state.items():
            value = value.detach().cpu().contiguous()
            if dtype is not None and value.is_floating_point():
                value = value.to(dtype)
            out[key] = value
        return out

    def save(self, run_id: str, path: str, kind: str) -> dict[str, Any]:
        from safetensors.torch import save_file

        run = self._run(run_id)
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        weights_dtype = torch.bfloat16 if kind == "sampler" and self.dtype == "bfloat16" else None
        save_file(self._adapter_state(run, weights_dtype), str(target / "adapter_model.safetensors"))
        config = run.lora_config.to_dict()
        config["base_model_name_or_path"] = self.model_dir
        config["inference_mode"] = True
        for key, value in list(config.items()):
            if isinstance(value, set):
                config[key] = sorted(value)
        (target / "adapter_config.json").write_text(json.dumps(config, indent=2, default=str))
        meta = {**run.meta, "kind": kind, "step": run.step, "run_id": run.run_id, "saved_at": time.time()}
        (target / "localtrio_meta.json").write_text(json.dumps(meta, indent=2))
        if kind == "state" and run.optimizer is not None:
            torch.save(run.optimizer.state_dict(), target / "optimizer.pt")
        size = sum(f.stat().st_size for f in target.iterdir() if f.is_file())
        return {"path": str(target), "size": size}

    def _load_state(self, run: Run, path: Path, load_optimizer: bool) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = load_file(str(path / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(self.model, state, adapter_name=run.adapter)
        unexpected = getattr(result, "unexpected_keys", []) or []
        if unexpected:
            raise RuntimeError(f"checkpoint 中有无法匹配的权重：{unexpected[:5]}")
        meta = json.loads((path / "localtrio_meta.json").read_text())
        run.step = int(meta.get("step", 0))
        if load_optimizer:
            opt_path = path / "optimizer.pt"
            if not opt_path.exists():
                raise FileNotFoundError(f"{path} 没有 optimizer.pt（只有 save_state 保存的 checkpoint 才含优化器状态）")
            run.optimizer = torch.optim.AdamW(run.params, lr=1e-4)
            run.optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))

    def load_state(self, run_id: str, path: str, load_optimizer: bool) -> dict[str, Any]:
        run = self._run(run_id)
        self._load_state(run, Path(path), load_optimizer)
        return {"step": run.step}

    def _run(self, run_id: str) -> Run:
        if run_id not in self.runs:
            raise KeyError(f"训练 run {run_id} 不存在")
        return self.runs[run_id]

    def memory(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            f"cuda:{i}": torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())
        }
