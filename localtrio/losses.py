"""内置 loss 的逐 token 计算，与 PyTRIO / Tinker 的约定一致。

记号：``lp`` 为当前策略在 target token 上的 logprob（可求导），``old`` 为采样时
的 logprob，``A`` 为 advantage，``ratio = exp(lp - old)``。

======================  ================================================
loss_fn                 逐 token loss
======================  ================================================
cross_entropy           ``-w * lp``
importance_sampling     ``-ratio * A``
ppo                     ``-min(ratio * A, clip(ratio, lo, hi) * A)``
cispo                   ``-sg(clip(ratio, lo, hi)) * lp * A``
dro                     ``-(lp * A - 0.5 * beta * (lp - old) ** 2)``
======================  ================================================

``target_tokens == -100`` 的位置不参与计算。``loss_fn_config`` 中的
``clip_low_threshold`` / ``clip_high_threshold`` 是 ratio 的绝对上下界（不是 epsilon）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LOSS_INPUT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "cross_entropy": ("target_tokens",),
    "importance_sampling": ("target_tokens", "logprobs", "advantages"),
    "ppo": ("target_tokens", "logprobs", "advantages"),
    "cispo": ("target_tokens", "logprobs", "advantages"),
    "dro": ("target_tokens", "logprobs", "advantages"),
}

DEFAULT_CLIP = {
    "ppo": (0.8, 1.2),
    "cispo": (0.0, 4.0),
}


def normalize_loss_fn(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in LOSS_INPUT_SCHEMAS:
        raise ValueError(f"Unsupported loss function type: {name}")
    return normalized


@dataclass
class TokenCounts:
    """一个 logical batch 的全局分母（与 PyTRIO 的 GradientAccumulationTotals 同口径）。"""

    total_samples: int
    total_active_tokens: int
    total_advantage_tokens: int


def valid_mask(target_tokens: torch.Tensor, auto_shift: bool) -> torch.Tensor:
    mask = target_tokens != -100
    if auto_shift and mask.numel() > 0:
        mask = mask.clone()
        mask[0] = False
    return mask


def active_mask(loss_fn: str, inputs: dict[str, torch.Tensor], auto_shift: bool) -> torch.Tensor:
    mask = valid_mask(inputs["target_tokens"], auto_shift)
    if loss_fn == "cross_entropy" and "weights" in inputs:
        mask = mask & (inputs["weights"] != 0)
    return mask


def count_tokens(loss_fn: str, batch: list[dict[str, torch.Tensor]], auto_shift: bool) -> TokenCounts:
    active = 0
    advantage = 0
    for inputs in batch:
        mask = active_mask(loss_fn, inputs, auto_shift)
        active += int(mask.sum().item())
        if "advantages" in inputs:
            advantage += int((mask & (inputs["advantages"] != 0)).sum().item())
    return TokenCounts(total_samples=len(batch), total_active_tokens=active, total_advantage_tokens=advantage)


def denominator(loss_fn: str, counts: TokenCounts, mode: str) -> float:
    mode = mode.lower()
    if mode in ("sum", "none"):
        return 1.0
    if mode == "samples":
        return float(max(counts.total_samples, 1))
    if mode in ("active_tokens", "tokens", "token_mean"):
        if loss_fn == "dro":
            return float(max(counts.total_advantage_tokens, 1))
        return float(max(counts.total_active_tokens, 1))
    raise ValueError(f"未知的 loss_normalization：{mode}")


def per_token_loss(
    loss_fn: str,
    target_logprobs: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    config: dict[str, object] | None,
    auto_shift: bool = False,
) -> torch.Tensor:
    """返回 shape 与 ``target_logprobs`` 相同的逐 token loss（无效位置为 0）。"""
    config = config or {}
    lp = target_logprobs.float()
    mask = valid_mask(inputs["target_tokens"].to(lp.device), auto_shift)

    if loss_fn == "cross_entropy":
        weights = inputs.get("weights")
        w = torch.ones_like(lp) if weights is None else weights.to(lp.device, torch.float32)
        loss = -w * lp
    else:
        old = inputs["logprobs"].to(lp.device, torch.float32)
        adv = inputs["advantages"].to(lp.device, torch.float32)
        log_ratio = torch.where(mask, lp - old, torch.zeros_like(lp))
        ratio = torch.exp(log_ratio)
        if loss_fn == "importance_sampling":
            loss = -ratio * adv
        elif loss_fn == "ppo":
            low, high = _clip_bounds(loss_fn, config)
            clipped = torch.clamp(ratio, low, high)
            loss = -torch.minimum(ratio * adv, clipped * adv)
        elif loss_fn == "cispo":
            low, high = _clip_bounds(loss_fn, config)
            clipped = torch.clamp(ratio, low, high).detach()
            loss = -clipped * lp * adv
        elif loss_fn == "dro":
            beta = float(config.get("beta", 0.05))
            loss = -(lp * adv - 0.5 * beta * (lp - old) ** 2)
        else:  # pragma: no cover - normalize_loss_fn 已拦截
            raise ValueError(loss_fn)
    return torch.where(mask, loss, torch.zeros_like(loss))


def _clip_bounds(loss_fn: str, config: dict[str, object]) -> tuple[float, float]:
    low, high = DEFAULT_CLIP[loss_fn]
    low = float(config.get("clip_low_threshold", low))  # type: ignore[arg-type]
    high = float(config.get("clip_high_threshold", high))  # type: ignore[arg-type]
    if low > high:
        raise ValueError("clip_low_threshold must be <= clip_high_threshold")
    return low, high
