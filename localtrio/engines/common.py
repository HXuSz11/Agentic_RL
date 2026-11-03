"""采样引擎共用的小工具：停止条件拆分、图片解码。"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any


def split_stop(stop: Any, tokenizer: Any) -> tuple[list[str], list[int]]:
    """把 PyTRIO 的 ``stop``（字符串 / 字符串列表 / token id 列表）拆成文本与 token 两类。

    ``<|im_end|>`` 这类特殊 token 在 ``skip_special_tokens=True`` 的解码文本里不会出现，
    必须转换成 stop token id，否则字符串匹配永远不会触发。
    """
    if stop is None:
        return [], []
    if isinstance(stop, str):
        items: Sequence[Any] = [stop]
    else:
        items = list(stop)
    texts: list[str] = []
    ids: list[int] = []
    special = set(getattr(tokenizer, "all_special_tokens", []) or [])
    added = getattr(tokenizer, "get_added_vocab", lambda: {})() or {}
    for item in items:
        if isinstance(item, int):
            ids.append(int(item))
            continue
        if not item:
            continue
        if item in special or item in added:
            token_id = tokenizer.convert_tokens_to_ids(item)
            if isinstance(token_id, int) and token_id >= 0:
                ids.append(token_id)
                continue
        texts.append(str(item))
    return list(dict.fromkeys(texts)), list(dict.fromkeys(ids))


def adapter_version(path: str) -> str:
    """LoRA 目录的版本键：路径 + 权重文件的修改时间（同名目录被覆盖后键会变化）。"""
    import os

    weights = os.path.join(path, "adapter_model.safetensors")
    try:
        stamp = os.stat(weights).st_mtime_ns
    except OSError:
        stamp = 0
    return f"{path}@{stamp}"


def decode_image(data: bytes) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def default_stop_ids(model_dir: str, tokenizer: Any) -> list[int]:
    """未指定 stop 时与 vLLM 一致：使用 generation_config 的 eos_token_id。"""
    ids: list[int] = []
    try:
        from transformers import GenerationConfig

        gen = GenerationConfig.from_pretrained(model_dir)
        eos = gen.eos_token_id
        if isinstance(eos, int):
            ids.append(eos)
        elif isinstance(eos, (list, tuple)):
            ids.extend(int(x) for x in eos)
    except Exception:  # noqa: BLE001
        pass
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    return list(dict.fromkeys(ids))


def expand_chunks(chunks: list[tuple], image_processor: Any, image_token: int | None) -> tuple[list[int], Any, Any, list[int]]:
    """把 ``[("text", ids) | ("image", bytes, fmt, expected)]`` 展开为完整 token 序列。

    返回 ``(input_ids, pixel_values, image_grid_thw, image_token_counts)``；
    每张图片占 ``grid_t*grid_h*grid_w // merge_size**2`` 个 ``<|image_pad|>``。
    """
    ids: list[int] = []
    images = []
    expected: list[int | None] = []
    for chunk in chunks:
        if chunk[0] == "text":
            ids.extend(int(t) for t in chunk[1])
        else:
            images.append(decode_image(chunk[1]))
            expected.append(chunk[3])
            ids.append(-1 - (len(images) - 1))
    if not images:
        return ids, None, None, []
    if image_processor is None or image_token is None:
        raise ValueError("当前模型不是多模态模型，不能接受 ImageChunk")
    processed = image_processor(images=images, return_tensors="pt")
    grid = processed["image_grid_thw"]
    merge = int(getattr(image_processor, "merge_size", 2))
    counts = [int(g.prod().item()) // (merge * merge) for g in grid]
    out: list[int] = []
    for token in ids:
        if token < 0:
            idx = -1 - token
            if expected[idx] is not None and expected[idx] != counts[idx]:
                raise ValueError(
                    f"第 {idx} 张图片 expected_tokens={expected[idx]}，image processor 计算为 {counts[idx]}"
                )
            out.extend([int(image_token)] * counts[idx])
        else:
            out.append(token)
    return out, processed["pixel_values"], grid, counts
