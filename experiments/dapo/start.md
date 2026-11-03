# DAPO 快速启动（本地 4×A6000 版）

> 本页是原教程快速启动指南的本地 GPU 版本：训练与采样通过 [localtrio](../../docs/content/guide/localtrio.md) 在自己的 GPU 上执行，不需要 `trio login`、API Key，也没有计费。算法讲解见 [正文](./readme.md)。

以下命令均从仓库根目录运行。

## 0. 安装环境

```bash
uv sync --extra kernels            # torch / transformers / peft / vllm + Qwen3.5 线性注意力训练 kernel
uv run python -m localtrio info    # 查看检测到的 GPU 与布局
uv run swanlab login               # 可选；不用 SwanLab 就在训练命令里写 --swanlab-mode disabled
```

- 要求 Python `>=3.12` 和可用的 NVIDIA 驱动。`--extra kernels` 编译失败时可以先只运行 `uv sync`，训练能跑但会明显变慢。
- 基模 `Qwen/Qwen3.5-4B` 在第一次训练时自动从 HuggingFace 下载，也可以提前下载：`uv run python -m localtrio download Qwen/Qwen3.5-4B`。
- 国内网络：设置 `HF_ENDPOINT=https://hf-mirror.com`（模型和数据集都会走镜像），或者用 `LOCALTRIO_MODEL_SOURCE=modelscope` 让模型从 ModelScope 下载。

## 1. GPU 布局

不提供任何配置文件时，localtrio 使用内置的 4×A6000 默认布局（与 [`configs/4xa6000.yaml`](../../configs/4xa6000.yaml) 相同），本章直接用它即可：

| GPU | 进程 | 作用 |
| --- | --- | --- |
| GPU0 | trainer（transformers + PEFT） | LoRA 前向 / 反向、AdamW 更新、保存 checkpoint |
| GPU1、GPU2、GPU3 | 3 个 vLLM 采样副本（data parallel） | group rollout、AIME25 评测；每个 step 按路径热加载最新 LoRA |

其他情况：

- 只有 1 张 48GB 卡：在命令前加 `LOCALTRIO_CONFIG=configs/1gpu.yaml`，训练与采样共用 GPU0，速度明显变慢。
- 想临时换卡：`LOCALTRIO_TRAINER_GPUS=0 LOCALTRIO_SAMPLER_GPUS=1,2` 这类环境变量会覆盖布局。
- 权重、checkpoint 和子进程日志默认写在 `~/.cache/localtrio/`，可以用 `LOCALTRIO_HOME=/大容量磁盘/localtrio` 改位置。

GPU 显存规划的更多细节见 [4×A6000 显存与 GPU 布局](../../docs/content/guide/hardware.md)。

## 2. 准备数据

```bash
uv run python experiments/dapo/prepare_data.py
```

这条命令会从 HuggingFace 下载固定 revision 的 DAPO-Math-17K 并去重、切分，同时下载 AIME25：

```text
experiments/dapo/datasets/
├── train.jsonl     # 17,126 道训练题
├── dev.jsonl       # 50 道 dev 题
└── aime_2025/      # 30 道评测题
```

已经有本地 parquet 时可以用 `--raw /path/to/dapo-math-17k.parquet` 跳过 DAPO-Math-17K 的下载；AIME25 仍会从 HuggingFace 下载（`experiments/dapo/datasets/aime_2025/` 已存在且校验通过时跳过）。

## 3. 启动训练

DAPO（正文使用的缩小配置）：

```bash
uv run python experiments/dapo/train.py \
    --algorithm dapo \
    --max-steps 10 \
    --groups-per-step 4 \
    --group-size 8 \
    --max-candidate-multiplier 2 \
    --max-prompt-tokens 512 \
    --max-tokens 4096 \
    --overlong-cache 1024 \
    --swanlab-mode online
```

同一入口也支持 `--algorithm grpo`，可用于 matched baseline（GRPO 不使用 `--max-candidate-multiplier`）。

输出位置：

- 启动时会打印 `[localtrio] training run run_YYYYmmdd_HHMMSS_xxxxxx: ...`，这个 run 的所有文件都在 `~/.cache/localtrio/runs/<run_id>/` 下；
- 每 `--save-every` 步（默认 25）和训练结束时保存两份 checkpoint，日志里会打印完整路径：
  - `states/<run-name>-final-state/`：LoRA 权重 + AdamW 状态，用于续训；
  - `sampler_weights/<run-name>-final-sampler-weights/`：PEFT adapter，用于采样和评测；
  - `<run-name>` 默认是 `dapo-qwen35-4b-10steps`（GRPO 为 `grpo-qwen35-4b-10steps`），可用 `--run-name` 修改；
- SwanLab 本地日志在 `experiments/dapo/swanlog/`；trainer 与 vLLM 子进程的日志在 `~/.cache/localtrio/logs/`。

## 4. 运行评测

先评测 Base Model（结果写到 `experiments/dapo/eval-results/aime25-base.jsonl`）：

```bash
uv run python experiments/dapo/eval.py
```

再评测训练得到的 sampler weights。`--model-path` 直接传本地目录：

```bash
RUN_DIR=$(ls -dt "${LOCALTRIO_HOME:-$HOME/.cache/localtrio}"/runs/run_* | head -1)   # 最近一次训练（若在 YAML 里改了 home，换成对应目录）
ls "$RUN_DIR/sampler_weights"

uv run python experiments/dapo/eval.py \
    --model-path "$RUN_DIR/sampler_weights/dapo-qwen35-4b-10steps-final-sampler-weights" \
    --output experiments/dapo/eval-results/aime25-dapo-step10.jsonl
```

评测只启动 vLLM，不加载 trainer。训练已经结束时，可以在命令前加 `LOCALTRIO_SAMPLER_GPUS=0,1,2,3`，让 4 张卡各跑一个副本。每个 JSONL 前 30 行是逐题结果，最后一行是 Average@12、Pass@12、Format 等汇总。

## 5. 资源估算（估算，未实测）

| 项目 | 估算 |
| --- | --- |
| GPU0 显存 | 基模 bf16 约 9～10GB + LoRA/优化器约 1～2GB + 激活，常规约 15～25GB，其中含 lm_head 分块约 1～3GB |
| GPU1/2/3 显存 | 每卡 vLLM 预分配约 40GB（`gpu_memory_utilization=0.85`） |
| GRPO 每步 | 约 2～4 分钟（1 个 rollout wave + 一次 custom loss 更新） |
| DAPO 每步 | 约 2～10 分钟（1～5 个 rollout wave，取决于补采次数） |
| 10 step 试跑 | 约 0.5～1.5 小时，不含首次下载模型和启动 vLLM |
| AIME25 评测 | 每次约 10～20 分钟（360 条，上限 8,192 token） |

这些数字是按 A6000 显存带宽和算力推算的数量级，不是实测值，推算过程见正文的 [本地资源估算](./readme.md)。实际耗时以 SwanLab 的 `time/step_seconds` 为准。

## 常见问题

**GPU0（trainer）显存不足（CUDA OOM）**

先调小 micro-batch 的 token 上限 `trainer.max_tokens_per_microbatch`（默认 16384）。计算 logprob 时 lm_head 按 `logits_chunk_size` 个 token 分块，每块是 `logits_chunk_size × 约 25 万词表` 的 fp32 logits（默认 1024 时约 1～3GB，与序列条数无关）；显存仍紧张时可以再调小 `trainer.logits_chunk_size`（默认 1024，可依次试 512、256）。写一个只包含改动项的配置文件即可，其余项沿用默认布局：

```yaml
# my-localtrio.yaml
trainer:
  max_tokens_per_microbatch: 8192
  logits_chunk_size: 256
```

```bash
LOCALTRIO_CONFIG=my-localtrio.yaml uv run python experiments/dapo/train.py --algorithm dapo ...
```

改回原文最初的长 CoT 配置（`--max-tokens 8192 --overlong-cache 2048`）时最容易遇到这个问题；单条序列超过上限时会单独成为一个 micro-batch。

**vLLM 启动或采样时显存不足、提示 KV cache 不够**

在配置文件里调低 `models.default.gpu_memory_utilization`（例如 0.80）或 `max_num_seqs`，也可以给 `max_model_len` 设一个够用的值（训练需要 `max_prompt_tokens + max_tokens`，AIME25 评测默认需要约 12,300）：

```yaml
models:
  default:
    gpu_memory_utilization: 0.80
    max_model_len: 16384
```

**vLLM 报某个 LoRA 模块不支持**

默认 `trainer.lora_target_modules: all-linear` 会给 Qwen3.5 线性注意力的投影层也加 LoRA，较旧的 vLLM 可能不支持。在配置文件里改成 `standard`（只用 q/k/v/o 与 MLP 投影）后重新训练：

```yaml
trainer:
  lora_target_modules: standard
```

**`--model-path` 报“是 PyTRIO 云端路径”**

本地版没有 `trio://` 路径，请传训练日志里 `Saved sampler weights:` 打印的本地目录（目录里有 `adapter_config.json`）。

**训练还在跑，想同时评测**

评测会在默认的 GPU1/2/3 上再启动一组 vLLM，和正在训练的采样副本抢显存。建议训练结束后再评测；确实要并行时，用 `LOCALTRIO_SAMPLER_GPUS` 指定空闲的卡。两个 checkpoint 也可以分别用 `LOCALTRIO_SAMPLER_GPUS=0,1` 和 `LOCALTRIO_SAMPLER_GPUS=2,3` 并行评测。

**引擎启动失败，想看详细报错**

trainer 和 vLLM 子进程的日志在 `~/.cache/localtrio/logs/`；设置 `LOCALTRIO_WORKER_LOGS=inherit` 可以直接打印到终端。

**没有 SwanLab 账号**

训练命令改成 `--swanlab-mode disabled`，每步的 reward、有效组数和 token 数仍会打印在终端。
