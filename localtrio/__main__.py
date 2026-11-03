"""localtrio 命令行工具。

    python -m localtrio info                      # 查看 GPU 布局与配置
    python -m localtrio download Qwen/Qwen3.5-4B  # 预下载模型（HF 或 ModelScope）
    python -m localtrio serve --base-model Qwen/Qwen3.5-4B --lora step100=/path/to/sampler_weights/step100 --gpus 1
                                                  # 启动 OpenAI 兼容服务（评测脚本用）；
                                                  # 未识别的参数原样转发给 vLLM，例如 --enforce-eager
    python -m localtrio merge --adapter /path/to/sampler_weights/final --out ./merged
                                                  # 把 LoRA 合并进基模，导出完整权重
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def cmd_info(_: argparse.Namespace) -> None:
    from .config import config_file, detect_gpus, load_config

    cfg = load_config()
    print(f"config file : {config_file() or '(内置默认值)'}")
    print(f"home        : {cfg.home_path}")
    print(f"GPUs        : {detect_gpus() or '无（CPU 调试模式）'}")
    print(f"sampler     : {cfg.sampler_backend}")
    print(f"trainer     : GPU {cfg.trainer.gpus}  lora={cfg.trainer.lora_target_modules} alpha={cfg.trainer.lora_alpha}")
    for name, engine in cfg.models.items():
        tp = engine.tensor_parallel_size
        replicas = len(engine.gpus or []) // max(tp, 1)
        print(f"sampler[{name}] : GPU {engine.gpus} tp={tp} replicas={replicas} mem={engine.gpu_memory_utilization}")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv"], check=False)
    try:
        import vllm

        print(f"vllm        : {vllm.__version__}")
    except ImportError:
        print("vllm        : 未安装（将使用很慢的 HF 采样后端）")


def cmd_download(args: argparse.Namespace) -> None:
    from .config import load_config
    from .models import resolve_model_path

    cfg = load_config()
    for name in args.models:
        print(name, "->", resolve_model_path(name, cfg.model_paths, args.source or cfg.model_source))


def cmd_serve(args: argparse.Namespace) -> None:
    from .config import load_config
    from .models import resolve_model_path

    cfg = load_config()
    model_dir = resolve_model_path(args.base_model, cfg.model_paths, cfg.model_source)
    # `vllm serve` 是 vLLM 当前推荐的入口（python -m vllm.entrypoints.openai.api_server 已弃用）
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model_dir,
        "--served-model-name",
        args.base_model,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    if args.max_model_len and args.max_model_len > 0:
        cmd += ["--max-model-len", str(args.max_model_len)]
    loras = []
    for item in args.lora or []:
        if "=" not in item:
            raise SystemExit(f"--lora 需要 name=path 格式：{item}")
        name, path = item.split("=", 1)
        loras.append(f"{name}={Path(path).expanduser().resolve()}")
    cmd += ["--enable-lora", "--max-lora-rank", str(args.max_lora_rank)]
    if loras:
        cmd += ["--lora-modules", *loras]
    cmd += args.extra
    from .config import physical_devices

    env = os.environ.copy()
    if args.gpus:
        # 与配置文件一致：编号相对于当前可见的 GPU
        env["CUDA_VISIBLE_DEVICES"] = ",".join(physical_devices([int(x) for x in args.gpus.split(",") if x.strip()]))
    # 允许评测时通过 /v1/load_lora_adapter 动态加载更多 checkpoint
    env.setdefault("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True")
    print("$", " ".join(cmd), flush=True)
    print(f"OpenAI base_url: http://127.0.0.1:{args.port}/v1  模型名：{args.base_model}"
          + (f"，LoRA：{', '.join(i.split('=')[0] for i in loras)}" if loras else ""), flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


def cmd_merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel

    from .config import load_config
    from .models import load_hf_model, resolve_model_path

    adapter = Path(args.adapter).expanduser().resolve()
    meta = json.loads((adapter / "localtrio_meta.json").read_text()) if (adapter / "localtrio_meta.json").exists() else {}
    base = args.base_model or meta.get("base_model")
    if not base:
        raise SystemExit("需要 --base-model（adapter 目录里没有 localtrio_meta.json）")
    cfg = load_config()
    model_dir = resolve_model_path(base, cfg.model_paths, cfg.model_source)
    model = load_hf_model(model_dir, dtype=args.dtype, attn_implementation="sdpa", device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out = Path(args.out).expanduser()
    merged.save_pretrained(out, safe_serialization=True)
    for fname in os.listdir(model_dir):
        if fname.endswith((".json", ".jinja", ".txt", ".model", ".tiktoken")) and not fname.startswith(("model", "config")):
            if not (out / fname).exists():
                shutil.copy(Path(model_dir) / fname, out / fname)
    print(f"merged weights saved to {out} (dtype={args.dtype})")
    _ = torch


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m localtrio")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="查看配置与 GPU 布局").set_defaults(fn=cmd_info)

    p = sub.add_parser("download", help="预下载模型")
    p.add_argument("models", nargs="+")
    p.add_argument("--source", choices=["huggingface", "modelscope"], default=None)
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("serve", help="启动 OpenAI 兼容的 vLLM 服务（可挂载 LoRA）")
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora", action="append", help="name=path，可重复")
    p.add_argument("--gpus", default=None, help="例如 0 或 2,3")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=32768, help="默认 32768；传 0 使用模型自带上限")
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.set_defaults(fn=cmd_serve, extra=[])

    p = sub.add_parser("merge", help="把 LoRA 合并进基模")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.set_defaults(fn=cmd_merge)

    # serve 不认识的参数（如 --data-parallel-size 3、--enforce-eager）原样转发给 vLLM；
    # 也兼容用 "--" 把它们隔开的写法。
    args, unknown = parser.parse_known_args()
    if unknown:
        if args.cmd != "serve":
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        args.extra = [item for item in unknown if item != "--"]
    args.fn(args)


if __name__ == "__main__":
    main()
