"""C-Eval 中文通用能力评测脚本。

本地版：通过 OpenAI 兼容接口评测模型，默认连接 `python -m localtrio serve`
启动的本地 vLLM 服务（http://127.0.0.1:8000/v1），不需要 API key。
`--model` 可以传基模名称、serve 时用 --lora 注册的名字，也可以直接传本地
sampler_weights 目录（脚本会调用 /v1/load_lora_adapter 动态挂载）。

先在另一个终端启动服务（GPU 1，端口 8000）：
uv run python -m localtrio serve --base-model Qwen/Qwen3.5-4B --gpus 1 --port 8000 --max-model-len 16384

再运行评测：
uv run python 01-eval-ceval.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = SCRIPT_DIR / "datasets" / "ceval-non-med" / "test_sample.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval-results"
# 原版为 PyTRIO 的 OpenAI 兼容端点；本地版指向 `python -m localtrio serve`。
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
CHOICE_SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-Eval 中文通用能力评测")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="C-Eval JSONL 文件")
    parser.add_argument("--output-path", type=Path, default=None, help="预测结果 JSONL 输出路径；不传则按 model 自动生成")
    parser.add_argument("--metrics-path", type=Path, default=None, help="准确率指标 JSON 输出路径；默认根据 output-path 自动生成")
    parser.add_argument("--limit", type=int, default=0, help="最多评测多少条；<=0 表示全量")

    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI 兼容服务的 base_url；默认本地 localtrio serve")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="读取 API key 的环境变量名；未设置时使用 EMPTY（本地 vLLM 不校验）")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="基模名、serve 时注册的 LoRA 名，或本地 sampler_weights 目录")
    parser.add_argument("--max-tokens", type=int, default=4096, help="每题最多生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.01, help="采样 temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="采样 top_p")
    parser.add_argument("--concurrency", type=int, default=4, help="并发请求数量")
    parser.add_argument("--timeout", type=float, default=600.0, help="单次请求超时时间，单位秒（本地单卡长输出较慢，默认比原版的 120 秒更宽松）")
    parser.add_argument("--show-errors", type=int, default=5, help="终端最多展示多少条错误样例")
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    return args


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model.strip()).strip("-").lower()
    return slug or "model"


def default_output_path(model: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"ceval_eval_{model_slug(model)}.jsonl"


def load_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到评测文件：{path}，请先运行 00-download-dataset.py")

    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "answer_idx" not in row:
                raise ValueError(f"评测样本缺少 answer_idx 字段：{row.keys()}")
            examples.append(row)
            if limit > 0 and len(examples) >= limit:
                break
    return examples


def build_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    options = row.get("options") or {}
    if question and all(key in options for key in ("A", "B", "C", "D")):
        return "\n".join(
            [
                "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
                "",
                f"题目：{question}",
                f"A. {options['A']}",
                f"B. {options['B']}",
                f"C. {options['C']}",
                f"D. {options['D']}",
                "",
                "答案：",
            ]
        )
    if "prompt" in row:
        return str(row["prompt"])
    raise ValueError(f"评测样本缺少 question/options 或 prompt 字段：{row.keys()}")


LOCAL_SERVE_HINT = (
    "uv run python -m localtrio serve --base-model Qwen/Qwen3.5-4B "
    "--lora <名字>=<sampler_weights 目录> --gpus 1 --port 8000 --max-model-len 16384"
)


def lora_name_for(path: Path) -> str:
    """sampler_weights 目录 → 服务里的 LoRA 名：<run_id>-<权重名>，避免不同 run 的同名权重冲突。"""
    if path.parent.name == "sampler_weights":
        return f"{path.parent.parent.name}-{path.name}"
    return path.name


def resolve_served_model(args: argparse.Namespace, api_key: str) -> str:
    """把 --model 解析成 OpenAI 请求里的 model 字段。

    - 基模名（如 Qwen/Qwen3.5-4B）或 ``localtrio serve --lora 名字=目录`` 注册的名字：原样使用；
    - 本地 sampler_weights 目录（含 adapter_config.json，对应原版的 trio:// 路径）：
      通过 vLLM 的 /v1/load_lora_adapter 动态挂载（localtrio serve 已开启运行时 LoRA 更新），
      再用 <run_id>-<权重名> 作为模型名。
    """
    base_url = args.base_url.rstrip("/")
    path = Path(args.model).expanduser()
    is_adapter_dir = (path / "adapter_config.json").exists()
    try:
        with httpx.Client(timeout=args.timeout, headers={"Authorization": f"Bearer {api_key}"}) as http:
            served: set[str] = set()
            try:
                response = http.get(f"{base_url}/models")
                if response.status_code == 200:
                    served = {str(item.get("id")) for item in response.json().get("data", [])}
            except ValueError:
                pass
            if not is_adapter_dir:
                if served and args.model not in served:
                    print(f"Warning: 服务中没有模型 {args.model!r}，当前可用：{sorted(served)}")
                return args.model
            lora_path = path.resolve()
            lora_name = lora_name_for(lora_path)
            if lora_name not in served:
                response = http.post(
                    f"{base_url}/load_lora_adapter",
                    json={"lora_name": lora_name, "lora_path": str(lora_path)},
                )
                if response.status_code != 200 and "already" not in response.text:
                    raise RuntimeError(
                        f"动态加载 LoRA 失败（HTTP {response.status_code}）：{response.text}\n"
                        "请确认服务由 `python -m localtrio serve` 启动（会设置 "
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True），或在启动时用 --lora 名字=目录 注册后传名字。"
                    )
                print(f"Loaded LoRA adapter {lora_name} <- {lora_path}")
            return lora_name
    except httpx.ConnectError as error:
        raise RuntimeError(
            f"无法连接 {args.base_url}。请先在另一个终端启动本地 OpenAI 兼容服务，例如：\n  {LOCAL_SERVE_HINT}"
        ) from error


def create_client(args: argparse.Namespace) -> AsyncOpenAI:
    load_dotenv(SCRIPT_DIR / ".env", encoding="utf-8-sig", override=True)
    # 本地 vLLM 服务不校验 API key；没有设置时用占位符 EMPTY。
    api_key = os.getenv(args.api_key_env) or "EMPTY"
    args.served_model = resolve_served_model(args, api_key)

    return AsyncOpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout,
    )


async def infer_one(client: AsyncOpenAI, args: argparse.Namespace, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=args.served_model,
        messages=[
            {"role": "system", "content": CHOICE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    return response.choices[0].message.content or ""


def parse_choice(text: str) -> str | None:
    raw_text = text.strip()
    if "</think>" in raw_text:
        segments = [raw_text.rsplit("</think>", 1)[-1]]
    else:
        segments = [raw_text]

    for segment in segments:
        normalized = segment.strip().upper()
        if not normalized:
            continue
        exact_match = re.fullmatch(r"[^A-Z]*([ABCD])[^A-Z]*", normalized)
        if exact_match:
            return exact_match.group(1)

        patterns = [
            r"(?:FINAL\s*(?:ANSWER|OUTPUT)|ANSWER|OUTPUT|最终答案|正确答案|答案|选项|选择|应选|故选)\s*(?:IS|是|为|:|：|->)?\s*[\(\[【]?\s*([ABCD])",
            r"[\(\[【]\s*([ABCD])\s*[\)\]】]",
        ]
        candidates: list[str] = []
        for pattern in patterns:
            candidates.extend(re.findall(pattern, normalized))
        if candidates:
            return candidates[-1]

        for line in reversed(normalized.splitlines()):
            line_match = re.fullmatch(r"\s*([ABCD])\s*[\.。:：、]?\s*", line)
            if line_match:
                return line_match.group(1)

    fallback = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", segments[0].upper()[-1000:])
    if fallback:
        return fallback[-1]
    return None


async def evaluate_one(
    index: int,
    row: dict[str, Any],
    client: AsyncOpenAI,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt = build_prompt(row)
    async with semaphore:
        output = await infer_one(client, args, prompt)

    prediction = parse_choice(output)
    answer = str(row["answer_idx"]).strip().upper()
    is_correct = prediction == answer
    return {
        "index": index,
        "row_id": row.get("row_id"),
        "subject": row.get("subject"),
        "source_split": row.get("source_split"),
        "question": row.get("question"),
        "answer_idx": answer,
        "prediction": prediction,
        "correct": is_correct,
        "output": output,
        "prompt": prompt,
        "model": args.model,
    }


def subject_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("subject") or "unknown")].append(record)

    metrics = {}
    for subject, items in sorted(grouped.items()):
        total = len(items)
        correct = sum(1 for item in items if item["correct"])
        invalid = sum(1 for item in items if item["prediction"] is None)
        metrics[subject] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "invalid": invalid,
        }
    return metrics


async def evaluate(args: argparse.Namespace) -> None:
    examples = load_examples(args.dataset_path, args.limit)
    if not examples:
        raise ValueError(f"评测文件为空：{args.dataset_path}")

    client = create_client(args)
    args.output_path = args.output_path or default_output_path(args.served_model)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(examples)} C-Eval examples")
    print(f"API base_url: {args.base_url}")
    print(f"Model: {args.model}")
    if args.served_model != args.model:
        print(f"Served model name: {args.served_model}")
    print(f"Concurrency: {args.concurrency}")

    start = time.time()
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(evaluate_one(index, row, client, args, semaphore))
        for index, row in enumerate(examples)
    ]
    records: list[dict[str, Any]] = []
    try:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="C-Eval eval", unit="item"):
            records.append(await future)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()

    records.sort(key=lambda item: item["index"])
    correct = sum(1 for item in records if item["correct"])
    invalid = sum(1 for item in records if item["prediction"] is None)
    errors = [item for item in records if not item["correct"]][: args.show_errors]

    with args.output_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(examples)
    accuracy = correct / total if total else 0.0
    elapsed = time.time() - start
    by_subject = subject_metrics(records)
    macro_accuracy = (
        sum(item["accuracy"] for item in by_subject.values()) / len(by_subject)
        if by_subject
        else 0.0
    )
    metrics = {
        "accuracy": accuracy,
        "macro_subject_accuracy": macro_accuracy,
        "correct": correct,
        "total": total,
        "invalid": invalid,
        "elapsed_seconds": elapsed,
        "model": args.model,
        "served_model": args.served_model,
        "concurrency": args.concurrency,
        "dataset_path": str(args.dataset_path),
        "output_path": str(args.output_path),
        "by_subject": by_subject,
    }
    metrics_path = args.metrics_path or args.output_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"accuracy {accuracy:.4f} | macro_subject_accuracy {macro_accuracy:.4f} | "
        f"correct {correct}/{total} | invalid {invalid} | time {elapsed:.2f}s"
    )
    print(f"Saved predictions: {args.output_path}")
    print(f"Saved metrics: {metrics_path}")

    if errors:
        print("\n错误样例：")
        for item in errors:
            question = str(item["question"])[:80]
            print(
                f"- #{item['index']} subject={item['subject']} pred={item['prediction']} "
                f"answer={item['answer_idx']} question={question}"
            )


def main() -> None:
    asyncio.run(evaluate(parse_args()))


if __name__ == "__main__":
    main()
