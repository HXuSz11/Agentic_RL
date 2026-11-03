# OPSD 快速启动（本地 4×A6000 版）

> 本页是原教程快速启动指南的本地 GPU 版本：训练与采样通过 [localtrio](../../docs/content/guide/localtrio.md) 在自己的 GPU 上执行，不需要 `trio login`、API Key，也没有计费。算法讲解和原文结果见 [正文](./readme.md)。

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
| GPU0 | trainer（transformers + PEFT） | Student 的 LoRA 前向 / 反向、AdamW 更新、保存 checkpoint |
| GPU1、GPU2、GPU3 | 3 个 vLLM 副本（data parallel） | Student 采样（挂当前 LoRA）+ 固定 Teacher 的 `compute_logprobs`（同一基模，不挂 LoRA） |

固定 Teacher 就是 step-0 的同一个基模，和 Student 共用这组 vLLM 副本，不需要额外配置，也不会多加载一份模型。

其他情况：

- 只有 1 张 48GB 卡：在命令前加 `LOCALTRIO_CONFIG=configs/1gpu.yaml`，训练与采样共用 GPU0，速度明显变慢。
- 想临时换卡：`LOCALTRIO_TRAINER_GPUS=0 LOCALTRIO_SAMPLER_GPUS=1,2` 这类环境变量会覆盖布局。
- 权重、checkpoint 和子进程日志默认写在 `~/.cache/localtrio/`，可以用 `LOCALTRIO_HOME=/大容量磁盘/localtrio` 改位置。

GPU 显存规划的更多细节见 [4×A6000 显存与 GPU 布局](../../docs/content/guide/hardware.md)。

## 2. 准备数据

```bash
uv run python experiments/opsd/00-datasets.py
```

这条命令会下载并校验固定 revision 的 OPSD 训练集和 AIME25 评测集：

```text
experiments/opsd/datasets/
├── openthoughts_math_30k_opsd/   # 29,434 条 problem + solution
└── aime_2025/                     # 30 道评测题
```

只下载其中一份：`--only opsd` 或 `--only aime25`。

## 3. 启动训练

与正文实验相同的 100-step 异步 OPSD：

```bash
uv run python experiments/opsd/01-opsd-async.py \
    --steps 100 \
    --batch-size 32 \
    --group-size 1 \
    --max-tokens 1024 \
    --sample-size 0 \
    --save-every-steps 25 \
    --max-concurrency 32 \
    --swanlab-mode online
```

先试跑确认环境时，可以用 `--steps 10 --batch-size 4 --sample-size 100 --save-every-steps 5 --swanlab-mode disabled`。逐题串行、便于阅读的同步版是 `01-opsd-sync.py`，参数相同（没有 `--max-concurrency`）。

输出位置：

- 启动时会打印 `[localtrio] training run run_YYYYmmdd_HHMMSS_xxxxxx: ...`，这个 run 的所有文件都在 `~/.cache/localtrio/runs/<run_id>/` 下；
- 每 25 步保存两份 checkpoint，日志里的 `Saved state [...]` / `Saved sampler weights [...]` 会打印完整路径（`<run-name>` 默认是 `opsd-async-qwen35-4b-sampled-token-steps100`，可用 `--run-name` 修改）：
  - `states/<run-name>-step000025/`：LoRA 权重 + AdamW 状态，用于续训；
  - `sampler_weights/<run-name>-step000025/`：PEFT adapter，用于采样和评测；
- SwanLab 本地日志在 `experiments/opsd/swanlog/`；trainer 与 vLLM 子进程的日志在 `~/.cache/localtrio/logs/`。

## 4. 运行评测

先评测 Base Model：

```bash
uv run python experiments/opsd/00-eval-aime25.py \
    --val-n 12 \
    --max-tokens 38912 \
    --temperature 1.0 \
    --enable-thinking false \
    --output experiments/opsd/eval-results/aime25-base.jsonl
```

再依次评测 4 个 checkpoint。`--model-path` 直接传本地目录，输出文件名保持 `aime25-sampler-steps<N>.jsonl`，`analysis.py` 按这个名字识别：

```bash
RUN_DIR=$(ls -dt ~/.cache/localtrio/runs/run_* | head -1)   # 最近一次训练
ls "$RUN_DIR/sampler_weights"

for s in 25 50 75 100; do
  uv run python experiments/opsd/00-eval-aime25.py \
      --val-n 12 \
      --max-tokens 38912 \
      --temperature 1.0 \
      --enable-thinking false \
      --model-path "$RUN_DIR/sampler_weights/opsd-async-qwen35-4b-sampled-token-steps100-step$(printf %06d $s)" \
      --output experiments/opsd/eval-results/aime25-sampler-steps$s.jsonl
done
```

评测只启动 vLLM，不加载 trainer。训练已经结束时，可以在命令前加 `LOCALTRIO_SAMPLER_GPUS=0,1,2,3`，让 4 张卡各跑一个副本。每个 JSONL 前 30 行是逐题结果，最后一行是 Average@12、Pass@12、Format 等汇总。

最后汇总成图：

```bash
uv run python experiments/opsd/analysis.py
```

图默认写到 `experiments/opsd/eval-results/aime25-opsd-progress.png`，不会覆盖正文引用的原文结果图 `images/aime25-opsd-progress.png`。

## 5. 资源估算（估算，未实测）

| 项目 | 估算 |
| --- | --- |
| GPU0 显存 | 基模 bf16 约 9～10GB + LoRA rank 64/优化器约 2～3GB + 激活，常规约 15～25GB；短序列多时 logprob 分块会更高 |
| GPU1/2/3 显存 | 每卡 vLLM 预分配约 40GB（`gpu_memory_utilization=0.85`） |
| 每个训练 step | 约 1～1.5 分钟（32 条 ≤1,024 token 的 rollout + Teacher 打分 + 一次更新） |
| 100 steps | 约 1.5～2.5 小时，不含首次下载模型和启动 vLLM |
| AIME25 评测 | 每个模型状态约 20～40 分钟（360 条，上限 38,912 token），Base + 4 个 checkpoint 约 2～3.5 小时 |

这些数字是按 A6000 显存带宽和算力推算的数量级，不是实测值，推算过程见正文的 [本地资源估算](./readme.md)。评测时间主要由最长的那几条回答决定。

## 常见问题

**GPU0（trainer）显存不足（CUDA OOM）**

OPSD 的每条序列只有约 1k token，一个 micro-batch（默认上限 16384 token）会装下十几条序列；计算 logprob 时的分块张量大小约为“序列数 × `logits_chunk_size` × 约 25 万的词表”，是 trainer 显存峰值的主要来源。遇到 OOM 时调小 `trainer.logits_chunk_size`（默认 1024，可依次试 512、256），必要时再调小 `trainer.max_tokens_per_microbatch`。写一个只包含改动项的配置文件即可，其余项沿用默认布局：

```yaml
# my-localtrio.yaml
trainer:
  logits_chunk_size: 256
  max_tokens_per_microbatch: 8192
```

```bash
LOCALTRIO_CONFIG=my-localtrio.yaml uv run python experiments/opsd/01-opsd-async.py --steps 100 ...
```

micro-batch 只影响显存和速度，内置 `importance_sampling` loss 按整批的 token 数归一化，结果与整批计算等价。

**vLLM 在 Teacher 打分或评测时显存不足、提示 KV cache 不够**

Teacher 的 `compute_logprobs` 要为带参考解答的整段 prompt 计算逐 token logprob，评测又允许单条回答长达 38,912 token，这两处对 vLLM 显存的压力最大。可以在配置文件里调低 `gpu_memory_utilization`、限制每次调度的 token 数，并把 `max_model_len` 设成刚好够用的长度（评测需要 prompt + 38,912）：

```yaml
models:
  default:
    gpu_memory_utilization: 0.80
    max_num_batched_tokens: 2048
    max_model_len: 40960
```

也可以调小训练的 `--max-concurrency` 或评测的 `--concurrency`，减少同时在途的请求。

**vLLM 报某个 LoRA 模块不支持**

默认 `trainer.lora_target_modules: all-linear` 会给 Qwen3.5 线性注意力的投影层也加 LoRA，较旧的 vLLM 可能不支持。在配置文件里改成 `standard`（只用 q/k/v/o 与 MLP 投影）后重新训练：

```yaml
trainer:
  lora_target_modules: standard
```

**`--lora-rank` 能不能超过 64**

脚本沿用原版的 `[4, 64]` 校验；localtrio 的 vLLM 引擎默认 `max_lora_rank: 64`，两者一致，不建议改。

**`--model-path` 报“是 PyTRIO 云端路径”**

本地版没有 `trio://` 路径，请传训练日志里 `Saved sampler weights` 打印的本地目录（目录里有 `adapter_config.json`）。

**训练还在跑，想同时评测**

评测会在默认的 GPU1/2/3 上再启动一组 vLLM，和正在训练的采样副本抢显存。建议训练结束后再评测；确实要并行时，用 `LOCALTRIO_SAMPLER_GPUS` 指定空闲的卡。多个 checkpoint 也可以分别用 `LOCALTRIO_SAMPLER_GPUS=0,1` 和 `LOCALTRIO_SAMPLER_GPUS=2,3` 并行评测。

**引擎启动失败，想看详细报错**

trainer 和 vLLM 子进程的日志在 `~/.cache/localtrio/logs/`；设置 `LOCALTRIO_WORKER_LOGS=inherit` 可以直接打印到终端。

**没有 SwanLab 账号**

训练命令改成 `--swanlab-mode disabled`（或 `--no-swanlab`），每步的 reverse KL、token 数和耗时仍会打印在终端。
