"""构造离线测试用的微型模型（随机权重），不需要联网下载。

生成两个目录：

- ``tiny-qwen3``：纯文本 Qwen3ForCausalLM；
- ``tiny-qwen35-vl``：Qwen3_5ForConditionalGeneration（线性注意力 + 全注意力混合层，
  带视觉塔与 image processor），用来测试 LoRA 目标模块和图文输入。

用法：python tests/make_tiny_models.py /path/to/output
"""

from __future__ import annotations

import sys
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for part in message['content'] %}"
    "{% if part['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endif %}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}{% endif %}"
    "{% endif %}"
)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]


def build_tokenizer(out: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [
        "How many r's are in strawberry? Provide a numerical answer without units, written inside \\boxed{}.",
        "Let's spell the word out and number all the letters. There are three r's. \\boxed{3}",
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
        "user assistant system 0 1 2 3 4 5 6 7 8 9 + - * / = ( ) { } [ ] , . ? ! answer think",
        "请根据图片解答下面的几何选择题。选项 A B C D 最终答案",
    ] * 20
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[1:],
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def main(root: Path) -> None:
    import torch
    from transformers import GenerationConfig, Qwen2VLImageProcessor, Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(root)
    vocab = len(tokenizer)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    text_dir = root / "tiny-qwen3"
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        eos_token_id=eos,
        pad_token_id=pad,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(cfg).save_pretrained(text_dir)
    tokenizer.save_pretrained(text_dir)
    GenerationConfig(eos_token_id=[eos], pad_token_id=pad).save_pretrained(text_dir)

    vl_dir = root / "tiny-qwen35-vl"
    vl_cfg = Qwen3_5Config(
        text_config=dict(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            max_position_embeddings=4096,
            eos_token_id=eos,
            pad_token_id=pad,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3], "mrope_interleaved": True, "partial_rotary_factor": 1.0},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=256,
        ),
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=pad,
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        tie_word_embeddings=False,
    )
    Qwen3_5ForConditionalGeneration(vl_cfg).save_pretrained(vl_dir)
    tokenizer.save_pretrained(vl_dir)
    GenerationConfig(eos_token_id=[eos], pad_token_id=pad).save_pretrained(vl_dir)
    Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=32 * 32,
        max_pixels=128 * 128,
    ).save_pretrained(vl_dir)
    print(text_dir)
    print(vl_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/localtrio-tiny"))
