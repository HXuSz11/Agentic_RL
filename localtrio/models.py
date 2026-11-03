"""基模路径解析、tokenizer 与模型类加载。"""

from __future__ import annotations

import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("localtrio")
_DOWNLOAD_LOCK = threading.Lock()

IMAGE_PAD_TOKEN = "<|image_pad|>"


def resolve_model_path(name: str, model_paths: dict[str, str] | None = None, source: str = "auto") -> str:
    """把 ``Qwen/Qwen3.5-4B`` 这样的模型名解析为本地目录。

    优先级：配置里的 ``model_paths`` 映射 → 已存在的本地路径 → ModelScope /
    HuggingFace 下载（``source=auto`` 时，设置了 ``HF_ENDPOINT`` 或可访问 HF 就用 HF，
    否则用 ModelScope；国内网络建议在配置里写 ``model_source: modelscope``）。
    """
    model_paths = model_paths or {}
    mapped = model_paths.get(name) or model_paths.get(name.rsplit("/", 1)[-1])
    if mapped:
        path = Path(os.path.expanduser(mapped))
        if not path.exists():
            raise FileNotFoundError(f"model_paths[{name!r}] = {mapped} 不存在")
        return str(path.resolve())
    if Path(os.path.expanduser(name)).exists():
        return str(Path(os.path.expanduser(name)).resolve())

    source = (source or "auto").lower()
    with _DOWNLOAD_LOCK:
        if source in ("modelscope", "ms"):
            return _download_modelscope(name)
        if source in ("huggingface", "hf"):
            return _download_hf(name)
        if source == "local":
            raise FileNotFoundError(f"model_source=local，但找不到模型 {name}；请在 model_paths 中配置路径")
        # auto：先查两边的本地缓存，再尝试下载
        for fn in (_cached_hf, _cached_modelscope):
            cached = fn(name)
            if cached:
                return cached
        errors: list[str] = []
        order = (_download_hf, _download_modelscope)
        if os.environ.get("LOCALTRIO_PREFER_MODELSCOPE", "").lower() in ("1", "true", "yes"):
            order = (_download_modelscope, _download_hf)
        for fn in order:
            try:
                return fn(name)
            except Exception as error:  # noqa: BLE001
                errors.append(f"{fn.__name__}: {error}")
        raise RuntimeError(f"无法获取模型 {name}：\n" + "\n".join(errors))


def _cached_hf(name: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def _cached_modelscope(name: str) -> str | None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        return snapshot_download(name, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def _download_hf(name: str) -> str:
    from huggingface_hub import snapshot_download

    logger.info("Downloading %s from HuggingFace ...", name)
    return snapshot_download(name)


def _download_modelscope(name: str) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    logger.info("Downloading %s from ModelScope ...", name)
    return snapshot_download(name)


def read_config_json(model_dir: str) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_multimodal(model_dir: str) -> bool:
    cfg = read_config_json(model_dir)
    if "vision_config" in cfg:
        return True
    archs = cfg.get("architectures") or []
    return any("ConditionalGeneration" in a or "VL" in a for a in archs)


@lru_cache(maxsize=16)
def load_tokenizer(model_dir: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


@lru_cache(maxsize=8)
def load_image_processor(model_dir: str) -> Any:
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)


def image_token_id(model_dir: str, tokenizer: Any | None = None) -> int:
    cfg = read_config_json(model_dir)
    for key in ("image_token_id", "image_token_index"):
        if isinstance(cfg.get(key), int):
            return int(cfg[key])
    tokenizer = tokenizer or load_tokenizer(model_dir)
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"模型 {model_dir} 没有 {IMAGE_PAD_TOKEN}，不能接受图片输入")
    return int(token_id)


def load_hf_model(model_dir: str, *, dtype: str, attn_implementation: str, device_map: Any) -> Any:
    """按 checkpoint 的架构加载模型：多模态模型用 ImageTextToText，其余用 CausalLM。

    LoRA 的模块名与 HF checkpoint 保持一致，vLLM 才能直接加载导出的 adapter。
    """
    import torch
    import transformers

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    loaders = []
    if is_multimodal(model_dir):
        auto_mm = getattr(transformers, "AutoModelForImageTextToText", None)
        if auto_mm is not None:
            loaders.append(auto_mm)
    loaders.append(transformers.AutoModelForCausalLM)
    last_error: Exception | None = None
    for loader in loaders:
        try:
            return loader.from_pretrained(model_dir, **kwargs)
        except TypeError as error:
            # 旧版 transformers 使用 torch_dtype 参数名
            if "dtype" in str(error):
                kw = dict(kwargs)
                kw["torch_dtype"] = kw.pop("dtype")
                try:
                    return loader.from_pretrained(model_dir, **kw)
                except Exception as inner:  # noqa: BLE001
                    last_error = inner
                    continue
            last_error = error
        except (ValueError, KeyError) as error:
            last_error = error
    raise RuntimeError(f"无法加载模型 {model_dir}: {last_error}")
