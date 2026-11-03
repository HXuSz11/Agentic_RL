"""直接在进程内测试训练引擎的数值正确性（CPU，微型随机模型）。"""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import torch

from localtrio.config import TrainerConfig
from localtrio.engines.trainer import TrainerEngine


def make_engine(model_dir: str, **overrides) -> TrainerEngine:
    cfg = TrainerConfig(gpus=[], gradient_checkpointing=False, logits_chunk_size=7, max_tokens_per_microbatch=4096)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    from dataclasses import asdict

    return TrainerEngine({"base_model": model_dir, "model_dir": model_dir, "trainer": asdict(cfg)})


def text_datum(rng: np.random.Generator, length: int, vocab: int, prompt: int = 3) -> dict:
    ids = rng.integers(10, vocab - 1, size=length).tolist()
    targets = ids[1:] + [int(rng.integers(10, vocab - 1))]
    targets = [0] * prompt + targets[prompt:]
    adv = [0.0] * prompt + [float(rng.normal())] * (length - prompt)
    old = [0.0] * prompt + list(rng.normal(-3, 0.5, size=length - prompt))
    return {
        "chunks": [("text", ids)],
        "inputs": {
            "target_tokens": np.asarray(targets, dtype=np.int64),
            "logprobs": np.asarray(old, dtype=np.float32),
            "advantages": np.asarray(adv, dtype=np.float32),
        },
    }


def reference_logprobs(engine: TrainerEngine, datum: dict) -> torch.Tensor:
    ids = torch.tensor([datum["chunks"][0][1]])
    with torch.no_grad():
        logits = engine.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].float()
    targets = torch.from_numpy(datum["inputs"]["target_tokens"])
    return torch.log_softmax(logits, -1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)


def lora_grads(engine: TrainerEngine, run_id: str) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1).clone() if p.grad is not None else torch.zeros(p.numel()) for p in engine.runs[run_id].params])


def zero_grads(engine: TrainerEngine, run_id: str) -> None:
    for p in engine.runs[run_id].params:
        p.grad = None


@pytest.fixture(scope="module")
def engine(text_model):
    eng = make_engine(text_model)
    eng.create_run("run_a", rank=8, seed=1, train_mlp=True, train_attn=True, train_unembed=True)
    # 打破 LoRA B=0 的初始对称，让 adapter 真的影响输出
    with torch.no_grad():
        for p in eng.runs["run_a"].params:
            p.add_(torch.randn_like(p) * 0.05)
    return eng


def test_chunked_logprobs_match_full_logits(engine):
    rng = np.random.default_rng(0)
    vocab = len(engine.tokenizer)
    data = [text_datum(rng, n, vocab) for n in (5, 17, 30)]
    out = engine.forward("run_a", data, "cross_entropy", None, auto_shift=False, backward=False)
    for datum, item in zip(data, out["loss_fn_outputs"], strict=True):
        ref = reference_logprobs(engine, datum)
        np.testing.assert_allclose(item["logprobs"], ref.numpy(), atol=1e-5)


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo", "cispo", "dro"])
def test_builtin_loss_values(engine, loss_fn):
    rng = np.random.default_rng(1)
    vocab = len(engine.tokenizer)
    data = [text_datum(rng, n, vocab) for n in (9, 14)]
    cfg = {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}
    out = engine.forward("run_a", data, loss_fn, cfg, auto_shift=False, backward=False)
    total = 0.0
    for datum in data:
        lp = reference_logprobs(engine, datum)
        old = torch.from_numpy(datum["inputs"]["logprobs"])
        adv = torch.from_numpy(datum["inputs"]["advantages"])
        ratio = torch.exp(lp - old)
        if loss_fn == "importance_sampling":
            ref = -ratio * adv
        elif loss_fn == "ppo":
            ref = -torch.minimum(ratio * adv, ratio.clamp(0.9, 1.1) * adv)
        elif loss_fn == "cispo":
            ref = -ratio.clamp(0.9, 1.1) * lp * adv
        else:
            ref = -(lp * adv - 0.5 * 0.05 * (lp - old) ** 2)
        total += float(ref.sum())
    assert out["metrics"]["loss_sum"] == pytest.approx(total, rel=1e-4, abs=1e-4)
    active = sum(len(d["inputs"]["target_tokens"]) for d in data)
    assert out["metrics"]["token_count"] == active


def test_custom_loss_linearization_is_exact(engine):
    """linear_backward 的梯度必须等于直接对用户 loss 求导的梯度。"""
    rng = np.random.default_rng(2)
    vocab = len(engine.tokenizer)
    data = [text_datum(rng, n, vocab) for n in (8, 21)]

    def user_loss(lps: list[torch.Tensor]) -> torch.Tensor:
        # 一个非线性的序列级 loss（类似 GSPO）
        total = 0
        for lp, d in zip(lps, data, strict=True):
            old = torch.from_numpy(d["inputs"]["logprobs"])
            adv = torch.from_numpy(d["inputs"]["advantages"])
            mask = adv != 0
            seq_ratio = torch.exp(((lp - old) * mask).sum() / mask.sum())
            total = total - torch.minimum(seq_ratio * adv[mask][0], seq_ratio.clamp(0.95, 1.05) * adv[mask][0])
        return total / len(lps)

    # (a) 直接求导
    zero_grads(engine, "run_a")
    engine._activate(engine.runs["run_a"])
    prepared = [engine._prepare(d) for d in data]
    lps = engine._token_logprobs(prepared, grad=True)
    user_loss(lps).backward()
    direct = lora_grads(engine, "run_a")
    zero_grads(engine, "run_a")

    # (b) PyTRIO 风格：先 forward 拿 logprob，主进程算梯度，再 linear_backward
    fwd = engine.forward("run_a", data, "cross_entropy", None, False, False)
    leaves = [torch.tensor(o["logprobs"], requires_grad=True) for o in fwd["loss_fn_outputs"]]
    user_loss(leaves).backward()
    engine.linear_backward("run_a", data, [leaf.grad.numpy() for leaf in leaves])
    linear = lora_grads(engine, "run_a")
    zero_grads(engine, "run_a")
    assert direct.abs().max() > 0
    torch.testing.assert_close(linear, direct, rtol=1e-4, atol=1e-6)


def test_microbatch_invariance(text_model):
    rng = np.random.default_rng(3)
    grads = []
    for budget in (4096, 20):
        eng = make_engine(text_model, max_tokens_per_microbatch=budget)
        eng.create_run("r", rank=4, seed=7, train_mlp=True, train_attn=True, train_unembed=False)
        with torch.no_grad():
            torch.manual_seed(5)
            for p in eng.runs["r"].params:
                p.copy_(torch.randn_like(p) * 0.05)
        vocab = len(eng.tokenizer)
        rng = np.random.default_rng(3)
        data = [text_datum(rng, n, vocab) for n in (6, 11, 19, 4)]
        eng.forward("r", data, "importance_sampling", None, False, True)
        grads.append(lora_grads(eng, "r"))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-7)


def test_optim_step_and_state_roundtrip(text_model, tmp_path):
    eng = make_engine(text_model)
    eng.create_run("r1", rank=4, seed=3, train_mlp=True, train_attn=True, train_unembed=False)
    rng = np.random.default_rng(4)
    vocab = len(eng.tokenizer)
    data = [text_datum(rng, 12, vocab)]
    before = eng.forward("r1", data, "cross_entropy", None, False, False)["loss_fn_outputs"][0]["logprobs"].copy()
    for _ in range(3):
        eng.forward("r1", data, "importance_sampling", None, False, True)
        metrics = eng.optim_step("r1", {"learning_rate": 1e-2})
    assert metrics["grad_norm"] > 0
    after = eng.forward("r1", data, "cross_entropy", None, False, False)["loss_fn_outputs"][0]["logprobs"]
    assert not np.allclose(before, after)
    info = eng.save("r1", str(tmp_path / "state"), kind="state")
    assert info["size"] > 0
    assert (tmp_path / "state" / "adapter_config.json").exists()
    assert (tmp_path / "state" / "optimizer.pt").exists()
    eng.create_run("r2", rank=4, seed=99, train_mlp=True, train_attn=True, train_unembed=False, init_state=str(tmp_path / "state"), load_optimizer=True)
    restored = eng.forward("r2", data, "cross_entropy", None, False, False)["loss_fn_outputs"][0]["logprobs"]
    np.testing.assert_allclose(restored, after, atol=1e-6)
    # 两个 run 共享一份基模，互不干扰
    again = eng.forward("r1", data, "cross_entropy", None, False, False)["loss_fn_outputs"][0]["logprobs"]
    np.testing.assert_allclose(again, after, atol=1e-6)


def test_adapter_loads_in_peft(text_model, tmp_path):
    """导出的 sampler 权重是标准 PEFT 目录（vLLM / PEFT 都能加载）。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    eng = make_engine(text_model)
    eng.create_run("r", rank=4, seed=3, train_mlp=True, train_attn=True, train_unembed=False)
    with torch.no_grad():
        for p in eng.runs["r"].params:
            p.add_(torch.randn_like(p) * 0.05)
    eng.save("r", str(tmp_path / "w"), kind="sampler")
    rng = np.random.default_rng(5)
    datum = text_datum(rng, 10, len(eng.tokenizer))
    ours = eng.forward("r", [datum], "cross_entropy", None, False, False)["loss_fn_outputs"][0]["logprobs"]
    base = AutoModelForCausalLM.from_pretrained(text_model, dtype=torch.float32)
    peft = PeftModel.from_pretrained(base, str(tmp_path / "w"))
    ids = torch.tensor([datum["chunks"][0][1]])
    with torch.no_grad():
        logits = peft(input_ids=ids).logits[0].float()
    ref = torch.log_softmax(logits, -1).gather(-1, torch.from_numpy(datum["inputs"]["target_tokens"]).unsqueeze(-1)).squeeze(-1)
    np.testing.assert_allclose(ours, ref.numpy(), atol=1e-5)


def test_vision_language_model(vl_model):
    """Qwen3.5 结构：线性注意力层 + 图片输入的前向/反向。"""
    from PIL import Image

    eng = make_engine(vl_model)
    info = eng.create_run("v", rank=4, seed=0, train_mlp=True, train_attn=True, train_unembed=False)
    # all-linear：全注意力 q/k/v/o、线性注意力 in_proj_*/out_proj 和 MLP 都挂 LoRA
    assert {"q_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "down_proj"} <= set(info["target_modules"])
    assert not any("visual" in n for n, p in eng.model.named_parameters() if p.requires_grad)

    tok = eng.tokenizer
    image = Image.fromarray((np.random.default_rng(0).random((48, 64, 3)) * 255).astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    patches = eng.image_processor.get_number_of_image_patches(48, 64, images_kwargs={})
    n_img = patches // eng.image_processor.merge_size ** 2
    before = tok.encode("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)
    after = tok.encode("<|vision_end|>what?<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    completion = tok.encode("answer A<|im_end|>", add_special_tokens=False)
    length = len(before) + n_img + len(after) + len(completion) - 1
    obs = length - len(completion)
    datum = {
        "chunks": [("text", before), ("image", buffer.getvalue(), "png", n_img), ("text", after + completion[:-1])],
        "inputs": {
            "target_tokens": np.asarray([0] * obs + completion, dtype=np.int64),
            "logprobs": np.asarray([0.0] * obs + [-2.0] * len(completion), dtype=np.float32),
            "advantages": np.asarray([0.0] * obs + [1.0] * len(completion), dtype=np.float32),
        },
    }
    out = eng.forward("v", [datum], "importance_sampling", None, False, True)
    assert out["metrics"]["token_count"] == length
    assert len(out["loss_fn_outputs"][0]["logprobs"]) == length
    assert sum(p.grad.abs().sum() for p in eng.runs["v"].params if p.grad is not None) > 0
    bad = copy.deepcopy(datum)
    bad["chunks"][1] = ("image", buffer.getvalue(), "png", n_img + 1)
    with pytest.raises(ValueError, match="expected_tokens"):
        eng.forward("v", [bad], "importance_sampling", None, False, False)


def _image_datum(eng, seed: int, size: tuple[int, int], text: str):
    from PIL import Image

    rng = np.random.default_rng(seed)
    image = Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    n_img = eng.image_processor.get_number_of_image_patches(size[1], size[0], images_kwargs={}) // eng.image_processor.merge_size ** 2
    tok = eng.tokenizer
    before = tok.encode("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)
    after = tok.encode("<|vision_end|>" + text + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    completion = tok.encode("answer " + text + "<|im_end|>", add_special_tokens=False)
    length = len(before) + n_img + len(after) + len(completion) - 1
    obs = length - len(completion)
    return {
        "chunks": [("text", before), ("image", buffer.getvalue(), "png", n_img), ("text", after + completion[:-1])],
        "inputs": {
            "target_tokens": np.asarray([0] * obs + completion, dtype=np.int64),
            "logprobs": np.asarray([0.0] * obs + [-2.0] * len(completion), dtype=np.float32),
            "advantages": np.asarray([0.0] * obs + [float(seed % 3) - 1.0] * len(completion), dtype=np.float32),
        },
    }


def test_vision_microbatch_invariance(vl_model):
    """多条图文样本放进同一个 micro-batch，与逐条计算的 logprob 和梯度一致。"""
    results = []
    for budget in (4096, 1):
        eng = make_engine(vl_model, max_tokens_per_microbatch=budget)
        eng.create_run("v", rank=4, seed=0, train_mlp=True, train_attn=True, train_unembed=False)
        with torch.no_grad():
            torch.manual_seed(11)
            for p in eng.runs["v"].params:
                p.copy_(torch.randn_like(p) * 0.05)
        data = [_image_datum(eng, 1, (48, 64), "A B"), _image_datum(eng, 2, (64, 40), "C"), text_datum(np.random.default_rng(9), 12, len(eng.tokenizer))]
        assert len(eng._microbatches([eng._prepare(d) for d in data])) == (1 if budget == 4096 else 3)
        out = eng.forward("v", data, "importance_sampling", None, False, True)
        results.append(([o["logprobs"] for o in out["loss_fn_outputs"]], lora_grads(eng, "v")))
    for a, b in zip(results[0][0], results[1][0], strict=True):
        np.testing.assert_allclose(a, b, atol=2e-5)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-3, atol=1e-6)
