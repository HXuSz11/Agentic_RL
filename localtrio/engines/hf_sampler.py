"""HF transformers 采样引擎（后备方案）。

用途：没有安装 vLLM、vLLM 暂不支持某个新架构，或在 CPU 上做单元测试。速度远慢于
vLLM，但返回结构完全相同：生成 token、每个 token 在**原始模型分布**下的 logprob、
prompt logprob 与 top-k。
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

import torch

from ..models import image_token_id, is_multimodal, load_hf_model, load_image_processor, load_tokenizer
from .common import default_stop_ids, expand_chunks, split_stop

logger = logging.getLogger("localtrio.hf_sampler")


class HFSamplerEngine:
    is_async = False

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.model_dir = spec["model_dir"]
        engine = spec.get("engine", {})
        cuda = torch.cuda.is_available()
        dtype = engine.get("dtype", "bfloat16") if cuda else "float32"
        t0 = time.time()
        self.base = load_hf_model(
            self.model_dir,
            dtype=dtype,
            attn_implementation=spec.get("attn_implementation", "sdpa"),
            device_map={"": 0} if cuda else {"": "cpu"},
        )
        self.base.eval()
        self.model: Any = self.base
        self.tokenizer = load_tokenizer(self.model_dir)
        self.multimodal = is_multimodal(self.model_dir)
        self.image_processor = load_image_processor(self.model_dir) if self.multimodal else None
        self.image_token = image_token_id(self.model_dir, self.tokenizer) if self.multimodal else None
        self.default_stops = default_stop_ids(self.model_dir, self.tokenizer)
        self.adapters: OrderedDict[str, str] = OrderedDict()
        self.max_adapters = int(engine.get("max_loras", 4))
        logger.info("HF sampler loaded %s in %.1fs", self.model_dir, time.time() - t0)

    def info(self) -> dict[str, Any]:
        return {"backend": "hf", "model_dir": self.model_dir}

    @property
    def device(self) -> torch.device:
        return self.base.get_input_embeddings().weight.device

    # ------------------------------------------------------------------
    def _use_adapter(self, lora_path: str | None) -> None:
        if lora_path is None:
            if hasattr(self.model, "disable_adapter_layers"):
                self.model.disable_adapter_layers()
            return
        from peft import PeftModel

        name = self.adapters.get(lora_path)
        if name is None:
            name = f"a{len(self.adapters)}_{int(time.time() * 1000) % 10**8}"
            if isinstance(self.model, PeftModel):
                self.model.load_adapter(lora_path, adapter_name=name)
            else:
                self.model = PeftModel.from_pretrained(self.base, lora_path, adapter_name=name)
            self.adapters[lora_path] = name
            while len(self.adapters) > self.max_adapters:
                old_path, old_name = self.adapters.popitem(last=False)
                if old_name != name:
                    self.model.delete_adapter(old_name)
        self.adapters.move_to_end(lora_path)
        self.model.enable_adapter_layers()
        self.model.set_adapter(name)
        self.model.eval()

    def _model_kwargs(self, ids: list[int], pixel_values: Any, grid: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"input_ids": torch.tensor([ids], device=self.device)}
        kwargs["attention_mask"] = torch.ones_like(kwargs["input_ids"])
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(self.device)
            kwargs["image_grid_thw"] = grid.to(self.device)
            kwargs["mm_token_type_ids"] = (kwargs["input_ids"] == int(self.image_token)).to(torch.int32)
        return kwargs

    @torch.no_grad()
    def generate(
        self,
        chunks: list[tuple],
        num_samples: int,
        params: dict[str, Any],
        lora_path: str | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_text: bool = True,
    ) -> dict[str, Any]:
        self._use_adapter(lora_path)
        ids, pixel_values, grid, _ = expand_chunks(chunks, self.image_processor, self.image_token)
        result: dict[str, Any] = {"input_tokens": len(ids), "prompt_logprobs": [], "topk_prompt_logprobs": []}

        if include_prompt_logprobs or topk_prompt_logprobs > 0:
            out = self.model(**self._model_kwargs(ids, pixel_values, grid), use_cache=False)
            logp = torch.log_softmax(out.logits[0].float(), dim=-1)
            prompt_lp: list[float | None] = [None]
            topk: list[Any] = [None]
            for i in range(1, len(ids)):
                prompt_lp.append(float(logp[i - 1, ids[i]].item()))
                if topk_prompt_logprobs > 0:
                    vals, idxs = torch.topk(logp[i - 1], topk_prompt_logprobs)
                    topk.append([(int(t), float(v)) for t, v in zip(idxs.tolist(), vals.tolist(), strict=True)])
            result["prompt_logprobs"] = prompt_lp if include_prompt_logprobs else []
            result["topk_prompt_logprobs"] = topk if topk_prompt_logprobs > 0 else []

        stop_texts, stop_ids = split_stop(params.get("stop"), self.tokenizer)
        stop_ids = list(dict.fromkeys(stop_ids + self.default_stops))
        max_tokens = int(params["max_tokens"])
        temperature = float(params.get("temperature", 1.0))
        top_k = int(params.get("top_k", -1))
        top_p = float(params.get("top_p", 1.0))
        sequences = self._generate(ids, pixel_values, grid, num_samples, max_tokens, temperature, top_k, top_p, stop_ids, stop_texts, params.get("seed"), return_text)
        result["sequences"] = sequences
        result["output_tokens"] = sum(len(s["tokens"]) for s in sequences)
        return result

    def _generate(self, ids, pixel_values, grid, n, max_tokens, temperature, top_k, top_p, stop_ids, stop_texts, seed, return_text):
        from transformers import GenerationConfig

        kwargs = self._model_kwargs(ids, pixel_values, grid)
        # 显式给出全部采样参数，避免继承 checkpoint 自带 generation_config 里的 top_k / top_p
        gen = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_k=top_k if top_k > 0 else 0,
            top_p=top_p if temperature > 0 else None,
            num_return_sequences=n,
            eos_token_id=stop_ids or None,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (stop_ids[0] if stop_ids else 0),
            stop_strings=stop_texts or None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        if seed is not None:
            torch.manual_seed(int(seed))
        out = self.model.generate(**kwargs, generation_config=gen, tokenizer=self.tokenizer if stop_texts else None)
        prompt_len = kwargs["input_ids"].shape[1]
        sequences = []
        stop_set = set(stop_ids)
        for b in range(out.sequences.shape[0]):
            generated = out.sequences[b, prompt_len:].tolist()
            length, reason = len(generated), "length"
            for i, token in enumerate(generated):
                if token in stop_set:
                    length, reason = i + 1, "stop"
                    break
                if stop_texts:
                    tail = self.tokenizer.decode(generated[max(0, i - 31) : i + 1], skip_special_tokens=True)
                    if any(t in tail for t in stop_texts):
                        length, reason = i + 1, "stop"
                        break
            tokens = generated[:length]
            logprobs = [
                float(torch.log_softmax(out.logits[i][b].float(), dim=-1)[token].item()) for i, token in enumerate(tokens)
            ]
            text = None
            if return_text:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                for t in stop_texts:
                    if t in text:
                        text = text[: text.index(t)]
            sequences.append({"tokens": tokens, "logprobs": logprobs, "stop_reason": reason, "text": text})
        return sequences
