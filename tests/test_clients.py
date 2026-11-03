"""通过公共 API（ServiceClient → 子进程引擎）做端到端测试。

CPU + HF 采样后端；验证的重点是：

- 采样端返回的 logprob 与训练端 forward 的 logprob 一致（on-policy 时 ratio≈1）；
- 同步 / 异步接口的返回类型与 PyTRIO 一致；
- 训练确实改变了采样分布；checkpoint 可以恢复。
"""

from __future__ import annotations

import asyncio
import io
import os

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def trio(tmp_path_factory):
    os.environ["LOCALTRIO_HOME"] = str(tmp_path_factory.mktemp("home"))
    os.environ["LOCALTRIO_SAMPLER_BACKEND"] = "hf"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import localtrio

    localtrio.shutdown()
    yield localtrio
    localtrio.shutdown()


def build_datum(trio, prompt, tokens, logprobs, advantage):
    obs = len(prompt) - 1
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(prompt + tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray([0] * obs + tokens, dtype=np.int64),
            "logprobs": np.asarray([0.0] * obs + logprobs, dtype=np.float32),
            "advantages": np.asarray([0.0] * obs + [advantage] * len(tokens), dtype=np.float32),
        },
    )


def test_end_to_end_sync(trio, text_model):
    service = trio.ServiceClient()
    tc = service.create_lora_training_client(base_model=text_model, rank=8, seed=0)
    tok = tc.get_tokenizer()
    prompt = tok.apply_chat_template([{"role": "user", "content": "How many r's are in strawberry?"}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    params = trio.SamplingParams(max_tokens=12, temperature=1.0, stop=["<|im_end|>"])

    sampler = tc.save_weights_and_get_sampling_client()
    result = sampler.sample(prompt=trio.ModelInput.from_ints(prompt_ids), num_samples=3, sampling_params=params).result()
    assert isinstance(result, trio.types.SampleResponse)
    assert len(result.sequences) == 3
    assert result.input_tokens == len(prompt_ids)
    for seq in result.sequences:
        assert len(seq.tokens) == len(seq.logprobs) > 0
        assert seq.stop_reason in ("stop", "length")
        assert isinstance(seq.text, str)

    # 采样端 logprob == 训练端 forward logprob
    datums = [build_datum(trio, prompt_ids, s.tokens, s.logprobs, 1.0) for s in result.sequences]
    fwd = tc.forward(datums, loss_fn="cross_entropy").result()
    for seq, out in zip(result.sequences, fwd.loss_fn_outputs, strict=True):
        train_lp = np.asarray(out["logprobs"].data)[len(prompt_ids) - 1 :]
        np.testing.assert_allclose(train_lp, seq.logprobs, atol=2e-4)

    # compute_logprobs 与 forward 一致
    full = prompt_ids + result.sequences[0].tokens
    lps = sampler.compute_logprobs(trio.ModelInput.from_ints(full)).result()
    assert lps[0] is None and len(lps) == len(full)
    np.testing.assert_allclose(lps[len(prompt_ids):], result.sequences[0].logprobs, atol=2e-4)

    # 训练：把第一个回答当作好样本反复强化，它的概率应当上升
    target = result.sequences[0]
    good = build_datum(trio, prompt_ids, target.tokens, target.logprobs, 1.0)
    before = float(np.sum(target.logprobs))
    for _ in range(5):
        fb = tc.forward_backward([good], loss_fn="importance_sampling")
        opt = tc.optim_step(trio.AdamParams(learning_rate=5e-2))
        assert isinstance(fb.result(), trio.types.ForwardBackwardOutput)
        assert "grad_norm" in opt.result().metrics
    saved = tc.save_weights_for_sampler(name="final").result()
    assert os.path.isdir(saved.path) and saved.size > 0
    trained = service.create_sampling_client(base_model=text_model, model_path=saved.path)
    after = trained.compute_logprobs(trio.ModelInput.from_ints(prompt_ids + target.tokens)).result()
    assert float(np.sum(after[len(prompt_ids):])) > before + 1.0

    # 基模采样不受 LoRA 影响
    base = service.create_sampling_client(base_model=text_model)
    base_lp = base.compute_logprobs(trio.ModelInput.from_ints(prompt_ids + target.tokens)).result()
    np.testing.assert_allclose(base_lp[len(prompt_ids):], target.logprobs, atol=2e-4)

    # PPO 的 loss_fn_config、custom loss
    fb = tc.forward_backward([good], loss_fn="ppo", loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.28}).result()
    assert fb.metrics["token_count"] > 0

    def custom(data, logprobs_list):
        loss = sum(-(lp * 0.1).sum() for lp in logprobs_list)
        return loss, {"custom/metric": 1.5}

    plain = trio.Datum(model_input=good.model_input, loss_fn_inputs={"target_tokens": good.loss_fn_inputs["target_tokens"]})
    out = tc.forward_backward_custom([plain], custom).result()
    assert out.metrics["custom/metric"] == 1.5
    tc.optim_step(trio.AdamParams(learning_rate=1e-3)).result()

    # 保存 / 恢复
    state = tc.save_state(name="ckpt").result()
    restored = service.create_training_client_from_state_with_optimizer(state.path)
    a = tc.forward([plain]).result().loss_fn_outputs[0]["logprobs"].data
    b = restored.forward([plain]).result().loss_fn_outputs[0]["logprobs"].data
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_async_api_and_topk(trio, text_model):
    async def main():
        service = trio.ServiceClient()
        tc = await service.create_lora_training_client_async(base_model=text_model, rank=4)
        tok = tc.get_tokenizer()
        ids = tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        sampler = await tc.save_weights_and_get_sampling_client_async()
        params = trio.SamplingParams(max_tokens=6, seed=1)
        responses = await asyncio.gather(*[sampler.sample_async(prompt=trio.ModelInput.from_ints(ids), num_samples=2, sampling_params=params) for _ in range(3)])
        assert all(len(r.sequences) == 2 for r in responses)
        seq = responses[0].sequences[0]
        datum = build_datum(trio, ids, seq.tokens, seq.logprobs, 0.5)
        fb_future = await tc.forward_backward_async([datum], loss_fn="cispo")
        opt_future = await tc.optim_step_async(trio.AdamParams(learning_rate=1e-3))
        assert isinstance(fb_future, trio.APIFuture)
        fb = await fb_future
        await opt_future
        assert fb.metrics["loss_mean"] == pytest.approx(fb.metrics["loss_sum"] / fb.metrics["token_count"])
        lps = await sampler.compute_logprobs_async(trio.ModelInput.from_ints(ids))
        assert lps[0] is None
        top = await sampler.sample_async(
            prompt=trio.ModelInput.from_ints(ids),
            sampling_params=trio.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=3,
        )
        assert top.topk_prompt_logprobs[0] is None
        assert len(top.topk_prompt_logprobs[1]) == 3
        best = top.topk_prompt_logprobs[1][0][1]
        assert best >= top.topk_prompt_logprobs[1][-1][1]

    asyncio.run(main())


def test_vision_sampling_and_training(trio, vl_model):
    from PIL import Image
    from transformers import AutoImageProcessor

    service = trio.ServiceClient()
    tc = service.create_lora_training_client(base_model=vl_model, rank=4)
    tok = tc.get_tokenizer()
    processor = AutoImageProcessor.from_pretrained(vl_model)
    image = Image.fromarray((np.random.default_rng(1).random((40, 56, 3)) * 255).astype("uint8"))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    n_img = processor.get_number_of_image_patches(image.height, image.width, images_kwargs={}) // processor.merge_size**2
    text = tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "选项 A B C D"}, {"type": "image", "image": "x"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    before, after = text.split("<|image_pad|>")
    chunks = [
        trio.types.EncodedTextChunk(tokens=tok.encode(before, add_special_tokens=False)),
        trio.ImageChunk(data=buf.getvalue(), format="png", expected_tokens=n_img),
        trio.types.EncodedTextChunk(tokens=tok.encode(after, add_special_tokens=False)),
    ]
    prompt = trio.ModelInput(chunks=chunks)
    sampler = tc.save_weights_and_get_sampling_client()
    res = sampler.sample(prompt=prompt, num_samples=2, sampling_params=trio.SamplingParams(max_tokens=5)).result()
    assert res.input_tokens == len(prompt)
    seq = res.sequences[0]
    obs = len(prompt) - 1
    datum = trio.Datum(
        model_input=trio.ModelInput(chunks=[*chunks, trio.types.EncodedTextChunk(tokens=seq.tokens[:-1])]),
        loss_fn_inputs={
            "target_tokens": np.asarray([0] * obs + seq.tokens, dtype=np.int64),
            "logprobs": np.asarray([0.0] * obs + seq.logprobs, dtype=np.float32),
            "advantages": np.asarray([0.0] * obs + [1.0] * len(seq.tokens), dtype=np.float32),
        },
    )
    out = tc.forward([datum], loss_fn="importance_sampling").result()
    train_lp = np.asarray(out.loss_fn_outputs[0]["logprobs"].data)[obs:]
    np.testing.assert_allclose(train_lp, seq.logprobs, atol=5e-4)
    tc.forward_backward([datum], loss_fn="importance_sampling").result()
    tc.optim_step(trio.AdamParams(learning_rate=1e-3)).result()
    with pytest.raises(trio.ValidationError):
        sampler.compute_logprobs(prompt)
    _ = torch  # noqa: F841
