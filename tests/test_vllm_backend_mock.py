"""用假 vLLM（tests/fake_vllm）测试 vLLM 采样后端的参数构造、结果解析与异步服务循环。

真实 vLLM 需要 GPU；这里验证的是 localtrio 这一侧的代码：
LoRARequest 是否带上 sampler_weights 路径、stop 字符串是否转成 stop_token_ids、
top_k 关闭值、prompt logprobs / top-k 的解析、图文 prompt 的构造。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = textwrap.dedent(
    r'''
    import io, json, sys
    import numpy as np
    import localtrio as trio
    from PIL import Image

    text_model, vl_model = sys.argv[1], sys.argv[2]
    service = trio.ServiceClient()
    tc = service.create_lora_training_client(base_model=text_model, rank=4)
    tok = tc.get_tokenizer()
    ids = tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    sampler = tc.save_weights_and_get_sampling_client()
    res = sampler.sample(trio.ModelInput.from_ints(ids), num_samples=2,
                         sampling_params=trio.SamplingParams(max_tokens=8, stop=["<|im_end|>"])).result()
    seq = res.sequences[0]
    info = json.loads(seq.text)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert info["lora"] == sampler.model_path, info
    assert info["top_k"] == 0 and info["n"] == 2, info
    assert info["stop_token_ids"] == [im_end], info
    assert seq.stop_reason == "stop" and seq.tokens[-1] == im_end, seq
    assert seq.logprobs == [-0.1, -0.2, -0.05], seq.logprobs
    assert res.input_tokens == len(ids) and res.output_tokens == 6

    # 第二个 step 的权重目录会分配新的 lora_int_id
    tc.forward_backward([trio.Datum(model_input=trio.ModelInput.from_ints(ids[:-1]),
                                    loss_fn_inputs={"target_tokens": np.asarray(ids[1:], dtype=np.int64)})]).result()
    tc.optim_step(trio.AdamParams(learning_rate=1e-3)).result()
    sampler2 = tc.save_weights_and_get_sampling_client()
    info2 = json.loads(sampler2.sample(trio.ModelInput.from_ints(ids), sampling_params=trio.SamplingParams(max_tokens=2)).result().sequences[0].text)
    assert info2["lora"] == sampler2.model_path and info2["lora_id"] != info["lora_id"], (info, info2)

    lps = sampler.compute_logprobs(trio.ModelInput.from_ints(ids)).result()
    assert lps[0] is None and len(lps) == len(ids)
    assert abs(lps[1] + 1.01) < 1e-6, lps[:3]
    top = sampler.sample(trio.ModelInput.from_ints(ids), sampling_params=trio.SamplingParams(max_tokens=1),
                         include_prompt_logprobs=True, topk_prompt_logprobs=2).result()
    assert top.topk_prompt_logprobs[0] is None
    assert top.topk_prompt_logprobs[1] == [(1001, -0.1), (1002, -0.2)], top.topk_prompt_logprobs[1]

    base = service.create_sampling_client(base_model=text_model)
    assert json.loads(base.sample(trio.ModelInput.from_ints(ids), sampling_params=trio.SamplingParams(max_tokens=2)).result().sequences[0].text)["lora"] is None

    # 图文 prompt：一个 <|image_pad|> + multi_modal_data
    vl = service.create_sampling_client(base_model=vl_model)
    vtok = vl.get_tokenizer()
    img = Image.new("RGB", (40, 40), "white"); buf = io.BytesIO(); img.save(buf, format="PNG")
    before = vtok.encode("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)
    after = vtok.encode("<|vision_end|>?<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    prompt = trio.ModelInput(chunks=[trio.types.EncodedTextChunk(tokens=before),
                                     trio.ImageChunk(data=buf.getvalue(), format="png", expected_tokens=4),
                                     trio.types.EncodedTextChunk(tokens=after)])
    vres = vl.sample(prompt, sampling_params=trio.SamplingParams(max_tokens=3)).result()
    vinfo = json.loads(vres.sequences[0].text)
    assert vinfo["has_mm"] and vinfo["prompt_len"] == len(before) + 1 + len(after), vinfo
    assert vres.input_tokens == len(prompt), (vres.input_tokens, len(prompt))
    print("MOCK-VLLM-OK")
    '''
)


def test_vllm_backend_with_fake_engine(tmp_path, text_model, vl_model):
    script = tmp_path / "run.py"
    script.write_text(SCRIPT)
    config = tmp_path / "localtrio.yaml"
    config.write_text(
        "sampler_backend: vllm\n"
        "gpus: [0]\n"
        "trainer: {gpus: [0]}\n"
        "models:\n"
        "  default: {gpus: [0], gpu_memory_utilization: 0.3}\n"
        f"  {vl_model}: {{gpus: [0], gpu_memory_utilization: 0.3}}\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(ROOT / "tests" / "fake_vllm"), str(ROOT)]),
            "LOCALTRIO_CONFIG": str(config),
            "LOCALTRIO_HOME": str(tmp_path / "home"),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    env.pop("LOCALTRIO_SAMPLER_BACKEND", None)
    proc = subprocess.run(
        [sys.executable, str(script), text_model, vl_model],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    logs = "\n".join(p.read_text(errors="replace")[-3000:] for p in (tmp_path / "home" / "logs").glob("*.log")) if (tmp_path / "home" / "logs").exists() else ""
    assert "MOCK-VLLM-OK" in proc.stdout, proc.stdout + proc.stderr + logs
