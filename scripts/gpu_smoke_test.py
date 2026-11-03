"""在真实 GPU 上快速自检 localtrio（建议在自己的机器上第一次运行任何章节前先跑一遍）。

检查项：
1. vLLM 组采样（num_samples>1）是否返回全部序列；
2. vLLM 采样 logprob 与训练端 forward 的 logprob 是否一致（on-policy ratio ≈ 1）；
3. forward_backward + optim_step 后，新的 LoRA 是否被采样端热加载（目标序列概率上升）；
4. compute_logprobs / top-k prompt logprobs；
5. save_state → create_training_client_from_state 往返；
6. （可选，--image）图文 prompt 的采样与训练。

用法（仓库根目录）：
    uv run python scripts/gpu_smoke_test.py                       # 默认 Qwen/Qwen3.5-4B
    uv run python scripts/gpu_smoke_test.py --base-model Qwen/Qwen3.5-0.8B --image
    LOCALTRIO_CONFIG=configs/1gpu.yaml uv run python scripts/gpu_smoke_test.py

结果只用于判断环境是否正常，不代表任何训练效果。
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import localtrio as trio  # noqa: E402


def build_datum(prompt, tokens, logprobs, advantage):
    obs = len(prompt) - 1
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(prompt + tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray([0] * obs + tokens, dtype=np.int64),
            "logprobs": np.asarray([0.0] * obs + logprobs, dtype=np.float32),
            "advantages": np.asarray([0.0] * obs + [advantage] * len(tokens), dtype=np.float32),
        },
    )


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="localtrio GPU 自检")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--image", action="store_true", help="同时测试图文输入（需要多模态基模）")
    parser.add_argument("--logprob-tol", type=float, default=0.05, help="采样端与训练端逐 token logprob 的平均绝对误差上限")
    args = parser.parse_args()

    results = []
    t0 = time.time()
    service = trio.ServiceClient()
    tc = service.create_lora_training_client(base_model=args.base_model, rank=16, seed=0)
    tok = tc.get_tokenizer()
    messages = [{"role": "user", "content": "What is 17 * 23? Answer with a number in \\boxed{}."}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tok.encode(prompt_text, add_special_tokens=False)
    params = trio.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, stop=[t for t in [tok.eos_token, "<|im_end|>"] if t])

    sampler = tc.save_weights_and_get_sampling_client()
    res = sampler.sample(trio.ModelInput.from_ints(prompt), num_samples=args.group_size, sampling_params=params).result()
    print(f"引擎启动 + 首次采样耗时 {time.time() - t0:.0f}s", flush=True)
    results.append(check("组采样返回全部序列", len(res.sequences) == args.group_size, f"{len(res.sequences)}/{args.group_size}"))
    results.append(check("token 与 logprob 等长", all(len(s.tokens) == len(s.logprobs) > 0 for s in res.sequences)))

    datums = [build_datum(prompt, s.tokens, s.logprobs, 1.0) for s in res.sequences]
    fwd = tc.forward(datums, loss_fn="cross_entropy").result()
    diffs = []
    for seq, out in zip(res.sequences, fwd.loss_fn_outputs, strict=True):
        train_lp = np.asarray(out["logprobs"].data)[len(prompt) - 1 :]
        diffs.append(np.abs(train_lp - np.asarray(seq.logprobs)))
    mean_diff = float(np.mean(np.concatenate(diffs)))
    max_diff = float(np.max(np.concatenate(diffs)))
    results.append(
        check(
            "采样端 logprob ≈ 训练端 logprob",
            mean_diff < args.logprob_tol,
            f"平均 |Δ|={mean_diff:.4f}，最大 |Δ|={max_diff:.3f}，on-policy ratio 均值≈{math.exp(mean_diff):.3f}",
        )
    )

    target = res.sequences[0]
    full = trio.ModelInput.from_ints(prompt + target.tokens)
    before = sampler.compute_logprobs(full).result()
    results.append(check("compute_logprobs 首位为 None、长度正确", before[0] is None and len(before) == len(prompt) + len(target.tokens)))
    top = sampler.sample(trio.ModelInput.from_ints(prompt), sampling_params=trio.SamplingParams(max_tokens=1), include_prompt_logprobs=True, topk_prompt_logprobs=5).result()
    results.append(check("top-k prompt logprobs", top.topk_prompt_logprobs[0] is None and len(top.topk_prompt_logprobs[1]) == 5))

    good = build_datum(prompt, target.tokens, target.logprobs, 1.0)
    for _ in range(4):
        fb = tc.forward_backward([good], loss_fn="importance_sampling")
        opt = tc.optim_step(trio.AdamParams(learning_rate=2e-4))
        fb.result()
        metrics = opt.result().metrics
    results.append(check("optim_step 产生非零梯度", metrics.get("grad_norm", 0.0) > 0, f"grad_norm={metrics.get('grad_norm', 0.0):.4g}"))
    trained = tc.save_weights_and_get_sampling_client()
    after = trained.compute_logprobs(full).result()
    gain = float(np.sum(after[len(prompt):]) - np.sum(before[len(prompt):]))
    results.append(check("新 LoRA 被采样端加载（目标序列 logprob 上升）", gain > 0, f"Δ sum logprob = {gain:+.3f}"))

    state = tc.save_state(name="smoke").result()
    restored = service.create_training_client_from_state(state.path)
    a = np.asarray(tc.forward([good]).result().loss_fn_outputs[0]["logprobs"].data)
    b = np.asarray(restored.forward([good]).result().loss_fn_outputs[0]["logprobs"].data)
    results.append(check("save_state 往返一致", float(np.max(np.abs(a - b))) < 1e-3, f"max |Δ|={float(np.max(np.abs(a - b))):.2e}，目录 {state.path}"))

    if args.image:
        from PIL import Image

        processor = trio.get_image_processor(args.base_model)
        image = Image.fromarray((np.random.default_rng(0).random((224, 320, 3)) * 255).astype("uint8"))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        n_img = processor.get_number_of_image_patches(image.height, image.width, images_kwargs={}) // processor.merge_size**2
        text = tok.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": "x"}, {"type": "text", "text": "Describe the image."}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        before_img, after_img = text.split("<|image_pad|>")
        chunks = [
            trio.types.EncodedTextChunk(tokens=tok.encode(before_img, add_special_tokens=False)),
            trio.ImageChunk(data=buf.getvalue(), format="png", expected_tokens=n_img),
            trio.types.EncodedTextChunk(tokens=tok.encode(after_img, add_special_tokens=False)),
        ]
        mm_prompt = trio.ModelInput(chunks=chunks)
        vres = trained.sample(mm_prompt, num_samples=2, sampling_params=trio.SamplingParams(max_tokens=16)).result()
        results.append(check("图文采样", len(vres.sequences) == 2 and vres.input_tokens == len(mm_prompt), f"input_tokens={vres.input_tokens}, 本地={len(mm_prompt)}"))
        seq = vres.sequences[0]
        obs = len(mm_prompt) - 1
        datum = trio.Datum(
            model_input=trio.ModelInput(chunks=[*chunks, trio.types.EncodedTextChunk(tokens=seq.tokens[:-1])]),
            loss_fn_inputs={
                "target_tokens": np.asarray([0] * obs + seq.tokens, dtype=np.int64),
                "logprobs": np.asarray([0.0] * obs + seq.logprobs, dtype=np.float32),
                "advantages": np.asarray([0.0] * obs + [1.0] * len(seq.tokens), dtype=np.float32),
            },
        )
        out = tc.forward([datum], loss_fn="importance_sampling").result()
        vdiff = float(np.mean(np.abs(np.asarray(out.loss_fn_outputs[0]["logprobs"].data)[obs:] - np.asarray(seq.logprobs))))
        results.append(check("图文：采样端 logprob ≈ 训练端 logprob", vdiff < args.logprob_tol * 2, f"平均 |Δ|={vdiff:.4f}"))

    print(f"\n{sum(results)}/{len(results)} 项通过，总耗时 {time.time() - t0:.0f}s；引擎日志在 {trio.load_config().home_path / 'logs'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
