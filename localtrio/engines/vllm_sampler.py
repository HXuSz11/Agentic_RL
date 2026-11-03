"""vLLM 采样引擎（默认后端）。

- 使用 ``AsyncLLM``（vLLM V1），同一副本上的所有请求由 vLLM 连续批处理；
- 训练端每次 ``save_weights_for_sampler`` 都写一个新的 PEFT adapter 目录，这里用
  ``LoRARequest`` 按路径热加载，不需要重启引擎；
- 输入是 token id（``TokensPrompt``），与训练端看到的序列逐 token 一致；图片以
  单个 ``<|image_pad|>`` + ``multi_modal_data`` 传入，由 vLLM 按真实尺寸展开；
- 返回的 logprob 默认是原始模型分布（``logprobs_mode=raw_logprobs``），与训练端
  ``forward`` 的口径一致，重要性比率在 on-policy 时约等于 1。
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from typing import Any

from ..models import image_token_id, is_multimodal, load_tokenizer
from .common import adapter_version, decode_image, split_stop

logger = logging.getLogger("localtrio.vllm_sampler")


def _logprob_value(entry: Any, token: int) -> float:
    if entry is None:
        return float("nan")
    if isinstance(entry, dict):
        item = entry.get(token)
        if item is None:
            return float("nan")
        return float(getattr(item, "logprob", item))
    return float(getattr(entry, "logprob", entry))


class VLLMSamplerEngine:
    is_async = True

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.model_dir = spec["model_dir"]
        self.engine_cfg = spec.get("engine", {})
        self.tokenizer = load_tokenizer(self.model_dir)
        self.multimodal = is_multimodal(self.model_dir)
        self.image_token = image_token_id(self.model_dir, self.tokenizer) if self.multimodal else None
        self.lora_ids: dict[str, int] = {}
        self.engine: Any = None

    async def start(self) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs

        cfg = self.engine_cfg
        wanted: dict[str, Any] = {
            "model": self.model_dir,
            "tokenizer": self.model_dir,
            "trust_remote_code": True,
            "tensor_parallel_size": int(cfg.get("tensor_parallel_size", 1)),
            "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.85)),
            "max_model_len": cfg.get("max_model_len"),
            "max_num_seqs": int(cfg.get("max_num_seqs", 256)),
            "max_num_batched_tokens": cfg.get("max_num_batched_tokens"),
            "enable_prefix_caching": bool(cfg.get("enable_prefix_caching", True)),
            "enforce_eager": bool(cfg.get("enforce_eager", False)),
            "dtype": cfg.get("dtype", "bfloat16"),
            "quantization": cfg.get("quantization"),
            "enable_lora": bool(cfg.get("enable_lora", True)),
            "max_lora_rank": int(cfg.get("max_lora_rank", 64)),
            "max_loras": int(cfg.get("max_loras", 4)),
            "max_cpu_loras": int(cfg.get("max_cpu_loras", 8)),
            "max_logprobs": int(cfg.get("max_logprobs", 64)),
            "logprobs_mode": cfg.get("logprobs_mode", "raw_logprobs"),
            "seed": 0,
            "disable_log_stats": True,
        }
        if self.multimodal:
            # 多模态模型默认每个 prompt 最多 8 张图（Spec-o3 一条轨迹最多 8 张），并关闭视频输入，
            # 避免 vLLM 按更大的默认上限预留编码器显存。可在配置里覆盖。
            wanted["limit_mm_per_prompt"] = cfg.get("limit_mm_per_prompt") or {"image": 8, "video": 0}
        if not wanted["enable_lora"]:
            for key in ("max_lora_rank", "max_loras", "max_cpu_loras"):
                wanted.pop(key)
        wanted.update(cfg.get("extra_engine_args") or {})
        fields = {f.name for f in dataclasses.fields(AsyncEngineArgs)}
        dropped = [k for k in wanted if k not in fields]
        if dropped:
            logger.warning("当前 vLLM 版本不支持这些参数，已忽略：%s", dropped)
        args = AsyncEngineArgs(**{k: v for k, v in wanted.items() if k in fields and v is not None})
        t0 = time.time()
        try:
            from vllm.v1.engine.async_llm import AsyncLLM

            self.engine = AsyncLLM.from_engine_args(args)
        except ImportError:  # 旧版 vLLM
            from vllm import AsyncLLMEngine

            self.engine = AsyncLLMEngine.from_engine_args(args)
        logger.info("vLLM engine ready for %s in %.1fs", self.model_dir, time.time() - t0)

    def info(self) -> dict[str, Any]:
        import vllm

        return {"backend": "vllm", "vllm_version": getattr(vllm, "__version__", "?"), "model_dir": self.model_dir}

    async def shutdown(self) -> None:
        if self.engine is not None and hasattr(self.engine, "shutdown"):
            try:
                self.engine.shutdown()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    def _lora_request(self, path: str | None) -> Any:
        if path is None:
            return None
        from vllm.lora.request import LoRARequest

        # 以“路径 + 权重文件修改时间”为键：同名目录被覆盖保存后会分到新的 lora_int_id，
        # 避免 vLLM 继续使用缓存里的旧 adapter
        key = adapter_version(path)
        if key not in self.lora_ids:
            self.lora_ids[key] = len(self.lora_ids) + 1
        lora_id = self.lora_ids[key]
        return LoRARequest(lora_name=f"lora-{lora_id}", lora_int_id=lora_id, lora_path=path)

    async def generate(
        self,
        chunks: list[tuple],
        num_samples: int,
        params: dict[str, Any],
        lora_path: str | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        return_text: bool = True,
    ) -> dict[str, Any]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompt_ids: list[int] = []
        images = []
        expected_image_tokens = 0
        for chunk in chunks:
            if chunk[0] == "text":
                prompt_ids.extend(int(t) for t in chunk[1])
            else:
                if self.image_token is None:
                    raise ValueError("当前模型不是多模态模型，不能接受 ImageChunk")
                prompt_ids.append(int(self.image_token))
                images.append(decode_image(chunk[1]))
                expected_image_tokens += int(chunk[3] or 0)
        prompt: dict[str, Any] = {"prompt_token_ids": prompt_ids}
        if images:
            prompt["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}

        stop_texts, stop_ids = split_stop(params.get("stop"), self.tokenizer)
        want_prompt_lp = include_prompt_logprobs or topk_prompt_logprobs > 0
        top_k = int(params.get("top_k", -1))
        options = dict(
            n=int(num_samples),
            max_tokens=int(params["max_tokens"]),
            temperature=float(params.get("temperature", 1.0)),
            top_p=float(params.get("top_p", 1.0)),
            seed=params.get("seed"),
            stop=stop_texts or None,
            stop_token_ids=stop_ids or None,
            logprobs=0,
            prompt_logprobs=max(int(topk_prompt_logprobs), 0) if want_prompt_lp else None,
            detokenize=bool(return_text or stop_texts),
            skip_special_tokens=True,
        )
        # n>1 时 vLLM 把请求拆成 n 个子请求；默认的 CUMULATIVE 模式每次只返回“本步有更新的”
        # 子请求，最后一次 yield 只含最后结束的那几条。改用 FINAL_ONLY（结束时一次性给出全部 n
        # 条），并在下面按 index 累积所有 yield，兼容不支持 output_kind 的旧版本。
        try:
            from vllm.sampling_params import RequestOutputKind

            options["output_kind"] = RequestOutputKind.FINAL_ONLY
        except ImportError:  # pragma: no cover - 旧版 vLLM
            pass
        try:  # 新版 vLLM 用 0 表示不限制 top_k，旧版用 -1
            sampling = SamplingParams(top_k=top_k if top_k > 0 else 0, **options)
        except ValueError:
            sampling = SamplingParams(top_k=top_k if top_k > 0 else -1, **options)
        except TypeError:
            options.pop("output_kind", None)
            sampling = SamplingParams(top_k=top_k if top_k > 0 else 0, **options)
        request_id = uuid.uuid4().hex
        final = None
        prompt_logprobs_seen = None
        completions: dict[int, Any] = {}
        async for output in self.engine.generate(
            TokensPrompt(**prompt), sampling, request_id, lora_request=self._lora_request(lora_path)
        ):
            final = output
            for completion in output.outputs or []:
                if completion is not None:
                    completions[int(completion.index)] = completion
            if getattr(output, "prompt_logprobs", None):
                prompt_logprobs_seen = output.prompt_logprobs
        if final is None:
            raise RuntimeError("vLLM 没有返回结果")
        if len(completions) != int(num_samples):
            raise RuntimeError(f"vLLM 只返回了 {len(completions)} / {num_samples} 条序列（request {request_id}）")

        sequences = []
        for _, completion in sorted(completions.items()):
            tokens = [int(t) for t in completion.token_ids]
            entries = completion.logprobs or []
            logprobs = [_logprob_value(entries[i] if i < len(entries) else None, tok) for i, tok in enumerate(tokens)]
            if any(lp != lp for lp in logprobs):  # NaN：vLLM 没给出采样 token 的 logprob
                raise RuntimeError("vLLM 返回的采样 logprob 缺失（检查 max_logprobs / logprobs_mode 配置）")
            reason = completion.finish_reason or "length"
            sequences.append(
                {
                    "tokens": tokens,
                    "logprobs": logprobs,
                    "stop_reason": "stop" if reason == "stop" else str(reason),
                    "text": completion.text if return_text else None,
                }
            )

        result: dict[str, Any] = {"sequences": sequences, "prompt_logprobs": [], "topk_prompt_logprobs": []}
        if images:
            # vLLM 返回的 prompt_token_ids 是展开视觉占位符之后的真实序列；用它核对训练端的
            # expected_tokens。拿不到展开结果时（旧版本）退回客户端给出的估计值。
            expanded = list(getattr(final, "prompt_token_ids", None) or [])
            if len(expanded) > len(prompt_ids):
                result["input_tokens"] = len(expanded)
            else:
                result["input_tokens"] = len(prompt_ids) - len(images) + expected_image_tokens
        else:
            result["input_tokens"] = len(prompt_ids)
        result["output_tokens"] = sum(len(s["tokens"]) for s in sequences)
        if want_prompt_lp:
            raw = final.prompt_logprobs or prompt_logprobs_seen or []
            prompt_lp: list[float | None] = []
            topk: list[Any] = []
            for i, token in enumerate(prompt_ids):
                entry = raw[i] if i < len(raw) else None
                if entry is None:
                    prompt_lp.append(None)
                    topk.append(None)
                    continue
                value = _logprob_value(entry, token)
                prompt_lp.append(None if value != value else value)  # NaN → None
                if topk_prompt_logprobs > 0 and isinstance(entry, dict):
                    ranked = sorted(entry.items(), key=lambda kv: getattr(kv[1], "rank", 0) or 0)
                    picked = [
                        (int(tid), float(lp.logprob))
                        for tid, lp in ranked
                        if (getattr(lp, "rank", 1) or 1) <= topk_prompt_logprobs
                    ][:topk_prompt_logprobs]
                    topk.append(picked)
                else:
                    topk.append(None)
            result["prompt_logprobs"] = prompt_lp if include_prompt_logprobs else []
            result["topk_prompt_logprobs"] = topk if topk_prompt_logprobs > 0 else []
        return result
