# GRPO 快速启动（本地 GPU 版）

> 原理与代码讲解见 [GRPO 正文](./readme.md)。本章是 Datawhale [开源原教程](https://github.com/datawhalechina/agentic-rl-lab)的本地 GPU 版本，代码通过 [localtrio](../../docs/content/guide/localtrio.md) 在自己的 GPU 上训练和采样，不需要 `trio login`、API key，也没有计费。

以下命令均从仓库根目录运行，Python ≥ 3.12。

## 0. GPU 布局

本章只用一个基模（`Qwen/Qwen3.5-4B`），直接使用**默认布局**，不需要配置文件。localtrio 按检测到的卡数分配：

| GPU | 用途 |
| --- | --- |
| GPU 0 | LoRA 训练（transformers + PEFT，bf16 + 梯度检查点） |
| 其余每张卡 | 各一个 vLLM 采样副本，每个 step 热加载最新 LoRA |

只有 1 张卡时，训练和采样共用 GPU 0。启动前可以确认一下：

```bash
uv run python -m localtrio info
```

不需要 4 张卡。本章的最低与推荐配置（估算，详见 [显存与 GPU 布局规划](../../docs/content/guide/hardware.md)）：

| 你的卡 | 怎么跑 | 速度（相对 4 卡） |
| --- | --- | --- |
| 1 张 ≥48GB（最低） | 命令前加 `LOCALTRIO_CONFIG=configs/1gpu.yaml`（训练与采样共用 GPU 0） | 约 1/3～1/4 |
| 2 张 ≥24GB（最低） | 直接运行；24GB 卡加 `LOCALTRIO_CONFIG=configs/24gb.yaml`（训练 micro-batch 8192） | 约 1/2～1/3 |
| 4 张（推荐） | 直接运行，与 [`configs/4gpu.yaml`](../../configs/4gpu.yaml) 相同；3 张、8 张卡也直接运行。24GB 卡同样加 `LOCALTRIO_CONFIG=configs/24gb.yaml` | 基准 |

## 1. 安装依赖

```bash
uv sync --extra kernels   # kernels：Qwen3.5 线性注意力的训练加速，强烈建议安装
uv run swanlab login      # 可选；只用 --swanlab-mode disabled 时不需要
```

可以提前下载模型（约 9～10 GB），否则第一次训练时会自动下载：

```bash
uv run python -m localtrio download Qwen/Qwen3.5-4B
# 国内网络：
uv run python -m localtrio download Qwen/Qwen3.5-4B --source modelscope
```

用 ModelScope 下载后，训练时也让 localtrio 从 ModelScope 解析模型：在命令前加 `LOCALTRIO_MODEL_SOURCE=modelscope`，或在配置文件里写 `model_source: modelscope`。

## 2. 准备数据

无需单独执行下载脚本。训练首次启动时会通过 `datasets` 自动下载 `openai/gsm8k` 的 train split（7,473 条），并使用本地 Hugging Face 缓存。国内网络可以设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 3. 冒烟测试

先用极小的配置确认环境、模型下载和引擎启动都正常：

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 1 \
    --batch-size 2 \
    --group-size 2 \
    --max-tokens 64 \
    --loss-fn importance_sampling \
    --swanlab-mode disabled
```

第一次运行会先下载模型，再启动 1 个 trainer 和每张采样卡一个 vLLM 子进程，需要几分钟。`max-tokens=64` 时回答经常被截断、全组 reward 都是 0，日志里出现 `datums: 0` 属于正常现象，说明这一步没有训练信号。

## 4. 启动训练

同步版（正文主要讲这一份，按题目逐个采样）：

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

异步版（逻辑相同，batch 内所有题目并发采样，有多个采样副本时，不同题目按 prompt 哈希分到各个副本上并行生成；batch 较小时不一定每个副本都用上，正式实验推荐）：

```bash
uv run python experiments/grpo/02-demo-async.py \
    --steps 10 \
    --batch-size 4 \
    --group-size 8 \
    --max-tokens 512 \
    --loss-fn importance_sampling \
    --swanlab-mode online
```

`03-grpo.py` 是只保留 `importance_sampling` / `ppo` 两个内置 loss 的教学版，参数与同步版相同（没有 CISPO 相关参数，默认 SwanLab 项目为 `agentic-rl-local-grpo-tashan`）。

比较不同 loss 时，保持其他参数一致，仅修改：

```bash
--loss-fn importance_sampling
--loss-fn ppo
--loss-fn cispo        # 01 / 02 支持；用 forward_backward_custom 实现
```

常用参数（三个脚本相同，默认值以代码为准）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-model` | `Qwen/Qwen3.5-4B` | HuggingFace / ModelScope 模型名或本地目录 |
| `--lora-rank` | `32` | LoRA rank |
| `--steps` | `10` | 训练步数 |
| `--all-data` | 关 | 全量 GSM8K train 走一遍（batch 4 时约 1,869 步），打开后忽略 `--steps` |
| `--batch-size` | `4` | 每步题目数 |
| `--group-size` | `4` | 每题采样数（正文命令用 8） |
| `--max-tokens` | `1024` | 每条回答最多生成 token 数（正文命令用 512） |
| `--temperature` / `--top-p` | `1.0` / `1.0` | 采样参数 |
| `--learning-rate` | `4e-5` | Adam 学习率（`--beta1 0.9`、`--beta2 0.95`） |
| `--loss-fn` | `importance_sampling` | `importance_sampling` / `ppo` / `cispo` |
| `--cispo-clip-low-threshold` / `--cispo-clip-high-threshold` | `0.0` / `4.0` | 仅 01 / 02：CISPO 的 ratio 上下界 |
| `--swanlab-mode` | `online` | `online` 或 `disabled` |

## 5. 输出与评测

本章没有独立的 `eval.py`。训练过程会在终端和 SwanLab 中记录 `reward`、`frac_degenerate`、`rollout/avg_gen_len`、`datums`、`train_tokens` 与 `loss_mean`（CISPO 还有 `cispo/clip_fraction` 等）。训练结束时打印最终权重的**本地目录**：

```text
Saved weights name: grpo-sync-qwen35-4b-gsm8k-importance-sampling-steps10, path: /home/<you>/.cache/localtrio/runs/run_<时间>_<id>/sampler_weights/grpo-sync-qwen35-4b-gsm8k-importance-sampling-steps10
```

目录结构：

```text
~/.cache/localtrio/                 # 可用 LOCALTRIO_HOME 改到大容量磁盘
├── runs/<run_id>/
│   ├── sampler_weights/<名字>/      # 最终 LoRA（PEFT adapter，vLLM 可直接加载）
│   └── tmp_sampler/                # 训练中每步的临时权重，默认保留最近 4 份
└── logs/                           # trainer / vLLM 子进程日志
```

想手动看看训练后的模型怎么答题，可以在训练结束后（GPU 空出来后）把这个目录挂到本地 OpenAI 兼容服务上（只有 1 张卡时把 `--gpus 1` 改成 `--gpus 0`）：

```bash
uv run python -m localtrio serve \
    --base-model Qwen/Qwen3.5-4B \
    --lora grpo=/home/<you>/.cache/localtrio/runs/<run_id>/sampler_weights/<名字> \
    --gpus 1 --port 8000
```

然后用 `base_url=http://127.0.0.1:8000/v1`、模型名 `grpo`（或基模名 `Qwen/Qwen3.5-4B` 对照）、任意 api key（如 `EMPTY`）调用。需要完整权重时可以用 `uv run python -m localtrio merge --adapter <目录> --out ./merged` 把 LoRA 合并进基模。

## 6. 资源估算

> 以下为按模型大小和默认参数推算的**估算值，未实测**。

| 项目 | 估算 |
| --- | --- |
| GPU 0 显存（训练） | 约 15～25 GB（4B bf16 权重约 9～10 GB + LoRA / AdamW 约 1～2 GB + 梯度检查点激活）；24GB 卡用 micro-batch 8192（`configs/24gb.yaml` 已调好），约 13～18 GB |
| 采样卡显存 | vLLM 按 `gpu_memory_utilization=0.85` 预占每卡 85%，大部分是 KV cache：48GB 卡约 40 GB，24GB 卡约 20 GB |
| 首次启动 | 下载模型 + 加载各引擎，约 2～5 分钟 |
| 每步耗时 | 按 4 张 48GB 卡估算，`batch 4 × group 8 × 512 tokens`：同步版约 0.5～1.5 分钟，异步版更快 |
| 10 步试跑 | 按 4 张卡估算，连同启动约 10～25 分钟 |
| 卡更少时 | 2 张卡约慢 2～3 倍，1 张卡（`1gpu.yaml`）约慢 3～4 倍 |

## 7. 常见问题

**训练端（GPU0）OOM。** trainer 按 padding 后的 token 预算（`trainer.max_tokens_per_microbatch`）把整个 batch 拆成多个 micro-batch 逐个反向，峰值显存由这个预算约束，和 `--batch-size`、`--group-size` 基本无关。按下面的顺序调：先在配置文件里把 `trainer.max_tokens_per_microbatch` 从 16384 调到 8192 或 4096（一个 step 会拆成更多 micro-batch，梯度不变，只是更慢）；再把 `trainer.logits_chunk_size` 从 1024 调到 512；保持 `trainer.gradient_checkpointing: true`。仍然不够时再缩短 `--max-tokens`，它限制单条序列的长度，主要在单条序列就超过 micro-batch 预算时起作用。调小 `--batch-size` / `--group-size` 几乎省不下训练端显存，还会改变实验设置。

**vLLM 启动时报显存不足或 KV cache 不足。** 这两种报错的调法相反，先看报错原文：

- 报 `Free memory on device ... is less than desired GPU memory utilization` 或 CUDA OOM（多半是同一张卡上还有别的进程）：调小 `models.default.gpu_memory_utilization`（如 0.75）。
- 报 `... KV cache is needed, which is larger than the available KV cache memory`（24GB 卡上常见）：模型自带的 `max_model_len` 放不下，设置 `models.default.max_model_len: 8192`，或在显存允许时调大 `gpu_memory_utilization`。这时再调小 utilization 只会让可用的 KV cache 更少。

**vLLM 报 LoRA 模块不支持**（例如提到 `in_proj_qkv`、`out_proj` 等线性注意力投影）。把 `trainer.lora_target_modules` 改成 `standard`，只在 q/k/v/o + gate/up/down 上加 LoRA。

修改以上配置的方式：写一个只包含要改的项的 YAML（没写的项沿用默认值，可选项见 [`configs/4gpu.yaml`](../../configs/4gpu.yaml) 里的注释），再用 `LOCALTRIO_CONFIG` 指定，例如：

```yaml
# my-grpo.yaml（没写 gpus：仍按卡数自动分配，GPU 0 训练、其余采样）
trainer:
  max_tokens_per_microbatch: 8192
  lora_target_modules: standard
models:
  default:
    gpu_memory_utilization: 0.75
    max_model_len: 8192
```

```bash
LOCALTRIO_CONFIG=my-grpo.yaml uv run python experiments/grpo/02-demo-async.py --steps 10 --batch-size 4 --group-size 8 --max-tokens 512
```

**训练很慢。** 确认装了 `uv sync --extra kernels`（否则 Qwen3.5 的线性注意力走纯 PyTorch 实现）；用 `02-demo-async.py` 让各个采样副本并发工作；采样吞吐随采样卡数近似线性增长，卡少时这是最主要的瓶颈。

**`degenerate` 一直是 100%、`datums: 0`。** 同一题的回答全对或全错，没有相对优势信号。调大 `--group-size` 或 `--max-tokens`（回答被截断时拿不到 `\boxed{}`）。

**引擎启动失败或卡住。** 子进程日志在 `~/.cache/localtrio/logs/`；也可以加 `LOCALTRIO_WORKER_LOGS=inherit` 让日志直接打印到终端。用 `nvidia-smi` 检查是否有其他进程占着 GPU。

**想换 GPU 而不写配置文件。** 用环境变量指定训练卡和采样卡，例如 `LOCALTRIO_TRAINER_GPUS=0 LOCALTRIO_SAMPLER_GPUS=1,2`（GPU 0 训练、GPU 1–2 采样，其余卡留给别的任务）。

**原版快速启动里说 `02-demo-async.py` 启动会报 timeout 配置错误。** 本地版已经删掉了那段 PyTRIO 专用的 `trio.configure(...)`，可以直接运行。
