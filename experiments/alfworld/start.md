# ALFWorld Agentic RL 快速启动（本地 4×A6000 版）

> 完整原理、原文训练结果与代码拆解请查看 [ALFWorld Agentic RL Blog](./readme.md)。本章通过 localtrio 在本地 GPU 上运行，原理与 API 对照见 [localtrio 指南](../../docs/content/guide/localtrio.md)，显存与 GPU 布局规划见 [4×A6000 硬件指南](../../docs/content/guide/hardware.md)。

以下命令均从**仓库根目录**运行。所有计算都在本机完成：TextWorld / PDDL 环境跑在 CPU 上，rollout 由 vLLM 采样，LoRA 训练由 transformers + PEFT 完成。不需要 `trio login`，不需要 API Key，也没有计费。

> **算力提示**
>
> 正式配置每个 update 采样 64 条长轨迹，单条完整序列最多 12K tokens，80 updates 共 5,120 条轨迹。按第 8 节的估算，4×A6000 上完整训练需要十几个小时。建议先运行 1-step 验证，再启动完整训练。

## 0. GPU 布局

本章只训练一个基模 `Qwen/Qwen3.5-4B`，**直接使用默认布局即可，不需要任何配置文件**：

| 设备 | 角色 |
| --- | --- |
| GPU 0 | LoRA 训练（transformers + PEFT，bf16 + 梯度检查点） |
| GPU 1、2、3 | vLLM 采样，3 个副本做 data parallel，每个 update 热加载最新 LoRA |
| CPU | TextWorld / PDDL 环境，每条训练轨迹一个子进程 |

默认布局与预设 [`configs/4xa6000.yaml`](../../configs/4xa6000.yaml) 相同。查看本机实际生效的布局：

```bash
uv run python -m localtrio info
```

## 1. 安装依赖

ALFWorld 使用可选依赖安装，不会影响仓库中的其他章节：

```bash
uv sync --extra kernels --extra alfworld
```

- `kernels`：Qwen3.5 线性注意力（Gated DeltaNet）的训练加速 kernel，强烈建议安装；
- `alfworld`：`alfworld==0.4.2` 与 `spacy==3.8.13`，TextWorld 环境在 CPU 上运行。

Python 版本以仓库根目录的 `pyproject.toml` 为准（`>=3.12`）。`swanlab login` 只在使用 `--swanlab-mode online` 时需要。

## 2. 下载模型与 ALFWorld 数据

模型在第一次创建训练 / 采样客户端时自动下载，也可以提前下载：

```bash
uv run python -m localtrio download Qwen/Qwen3.5-4B
# 国内网络可改用 ModelScope，或设置 HF_ENDPOINT 使用镜像
uv run python -m localtrio download Qwen/Qwen3.5-4B --source modelscope
```

下载 ALFWorld 数据：

```bash
uv run --extra alfworld alfworld-download \
    --data-dir "$PWD/experiments/alfworld/datasets/alfworld"
```

训练和评测默认读取该目录，无需额外传入 `--data-root`。[AgentOPSD](../agentopsd/start.md) 与 [TEMPO](../tempo/readme.md) 章节用的是同一份数据，可以用 `--data-root` 指向这里，不必重复下载。

## 3. 运行 1-step 验证

先用一个游戏、8 条轨迹检查数据、环境、采样、reward、PPO 和 checkpoint 链路：

```bash
uv run --extra alfworld python experiments/alfworld/train.py \
    --max-steps 1 \
    --games-per-batch 1 \
    --group-size 8 \
    --max-episode-steps 10 \
    --save-every 0 \
    --run-name alfworld-smoke \
    --swanlab-mode disabled
```

第一次运行要加载模型、启动 trainer 和 3 个 vLLM 副本（含 CUDA graph 捕获），启动阶段本身就要几分钟。`--save-every 0` 只关闭中间 checkpoint；训练结束后仍会保存最终 state 和 sampler weights。

## 4. 启动 80-step 正式训练

```bash
uv run --extra alfworld python experiments/alfworld/train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --games-per-batch 8 \
    --group-size 8 \
    --max-episode-steps 50 \
    --max-trajectory-tokens 12000 \
    --max-assistant-tokens 2048 \
    --temperature 1.0 \
    --top-p 1.0 \
    --learning-rate 1e-6 \
    --save-every 20 \
    --run-name alfworld-agent-rl-qwen35-4b \
    --swanlab-mode online
```

所有超参数与原文一致。启动时会先打印 run ID，每 20 updates 打印一次两类**本地目录**：

```text
[localtrio] training run run_YYYYMMDD_HHMMSS_xxxxxx: base=Qwen/Qwen3.5-4B rank=32 ...
Saved state: /home/<you>/.cache/localtrio/runs/run_YYYYMMDD_HHMMSS_xxxxxx/states/alfworld-agent-rl-qwen35-4b-update-20-state
Saved sampler weights: /home/<you>/.cache/localtrio/runs/run_YYYYMMDD_HHMMSS_xxxxxx/sampler_weights/alfworld-agent-rl-qwen35-4b-update-20-weights
```

评测 checkpoint 时使用 `Saved sampler weights`，不要使用 state 目录。

`train.py` 保留了原文的两个常量：`MAX_SEQUENCE_TOKENS=12000`（单条完整轨迹上限，[`rollout.py`](./rollout.py)）和 `MAX_TRAIN_BATCH_SIZE=128`（单次 `forward_backward` 的 Datum 数上限，[`train.py`](./train.py)）。它们在原文里是 PyTRIO 的平台限制；本地没有这类限制，保留它们是为了让轨迹长度和每次调用的 loss 归一化口径与原文一致，同时控制训练显存。

## 5. 输出在哪里

| 内容 | 位置 |
| --- | --- |
| 训练 state（LoRA + AdamW 状态） | `~/.cache/localtrio/runs/<run_id>/states/<name>` |
| sampler weights（PEFT adapter，供评测） | `~/.cache/localtrio/runs/<run_id>/sampler_weights/<name>` |
| 每个 update 的临时采样快照 | `~/.cache/localtrio/runs/<run_id>/tmp_sampler/`，只保留最近 4 个 |
| trainer / vLLM 子进程日志 | `~/.cache/localtrio/logs/` |
| 评测 JSONL | `--output` 指定的路径，下文示例写到 `experiments/alfworld/eval_results/` |

目录根可以用环境变量 `LOCALTRIO_HOME=/data/localtrio` 或配置文件里的 `home` 改到大容量磁盘。训练轨迹只保留在进程内存中，不写盘；评测时每一局的完整轨迹都会写进 JSONL。

## 6. 评测

### 6.1 评测 Base Model

完整评测覆盖 `valid_seen=140` 与 `valid_unseen=134`，一共 274 个游戏：

```bash
uv run --extra alfworld python experiments/alfworld/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/alfworld/eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

先检查链路时可以加 `--limit-per-split 2 --games-per-batch 2`，每个 split 只评 2 局。

### 6.2 评测 checkpoint

原文的 checkpoint 保存在 PyTRIO 云端（`trio://...` 路径），本地无法加载；请评测自己训练出来的 sampler weights。把 `RUN_DIR` 换成训练日志里打印的 run 目录：

```bash
RUN_DIR=~/.cache/localtrio/runs/run_YYYYMMDD_HHMMSS_xxxxxx

uv run --extra alfworld python experiments/alfworld/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path "$RUN_DIR/sampler_weights/alfworld-agent-rl-qwen35-4b-update-40-weights" \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/alfworld/eval_results/checkpoint-40steps.jsonl \
    --swanlab-mode disabled

uv run --extra alfworld python experiments/alfworld/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path "$RUN_DIR/sampler_weights/alfworld-agent-rl-qwen35-4b-update-80-weights" \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/alfworld/eval_results/checkpoint-80steps.jsonl \
    --swanlab-mode disabled
```

`eval.py` 默认拒绝覆盖已有 JSONL。需要重新评测同一路径时，显式增加 `--overwrite-output`。

评测只用到采样引擎，GPU 0 会空闲。**不和训练同时运行时**，可以让 4 张卡都跑 vLLM 副本：

```bash
LOCALTRIO_SAMPLER_GPUS=0,1,2,3 uv run --extra alfworld python experiments/alfworld/eval.py ...
```

评测时每个游戏只采一条轨迹，TextWorld 在进程内运行、环境步进串行执行，`--games-per-batch` 主要提高的是采样端的并发。

## 7. 生成评测图

`analysis.py` 默认读取以下三个文件：

```text
experiments/alfworld/eval_results/base-qwen35-4b-eval.jsonl
experiments/alfworld/eval_results/checkpoint-40steps.jsonl
experiments/alfworld/eval_results/checkpoint-80steps.jsonl
```

默认输出路径 `experiments/alfworld/images/alfworld_checkpoint_evaluation.png` 正是文章里那张原文结果图，直接运行会把它覆盖掉。建议把本地结果写到别处：

```bash
uv run python experiments/alfworld/analysis.py \
    --output experiments/alfworld/eval_results/alfworld_checkpoint_evaluation_local.png
```

## 8. 本地资源估算

> 下表是根据模型规模、原文 token 用量和 A6000 算力做的**粗略估算，未经实测**，实际数值取决于 vLLM / transformers 版本、是否安装 `kernels`、CPU 性能和轨迹长度分布。

| 项目 | 估算 |
| --- | --- |
| GPU 0（训练） | Qwen3.5-4B bf16 权重约 8～10GB；加上 LoRA、AdamW 和 12K token 单条轨迹的激活（梯度检查点 + 分块 logprob），峰值大约 20～35GB |
| GPU 1～3（vLLM） | 每张卡按 `gpu_memory_utilization=0.85` 预占约 40GB，权重之外都用作 KV cache |
| 单个 update | rollout（最多 50 轮环境交互）约 3～8 分钟，训练约 5～8 分钟 |
| 1-step 验证 | 约 5～15 分钟，大部分是模型加载与 vLLM 启动 |
| 80-step 正式训练 | 约 10～20 小时（原文在 PyTRIO 上约 12 小时） |
| 一次完整评测（274 局） | 约 0.5～1.5 小时 |
| CPU / 内存 | 正式配置同时持有 64 个 TextWorld 子进程；建议 8 核以上、32GB 以上内存。CPU 核数少时环境步进会成为瓶颈 |
| 磁盘 | 模型约 10GB；每个 sampler weights 是一个 rank-32 LoRA adapter，state 额外包含 AdamW 状态，体积约为前者的数倍 |

训练时间的估算依据：原文 PyTRIO 会话的用量统计为 prefilling 843.51M、train 62.24M、sample 14.28M tokens，折合每个 update 约 78 万训练 token、18 万采样 token。

## 9. 常见问题

**训练进程（GPU 0）OOM。** 先确认装了 `--extra kernels`。再写一份配置文件调小显存相关参数（文件可以放在任意位置）：

```yaml
# ~/localtrio-alfworld.yaml：在默认 4×A6000 布局上做保守的显存设置
trainer:
  gpus: [0]
  max_tokens_per_microbatch: 8192   # 默认 16384；单条超过上限的轨迹会单独成批
  logits_chunk_size: 512            # 默认 1024；降低 lm_head 分块大小
models:
  default:
    gpus: [1, 2, 3]
    gpu_memory_utilization: 0.80    # 默认 0.85
    max_model_len: 16384            # 12K 轨迹 + 余量
```

```bash
LOCALTRIO_CONFIG=~/localtrio-alfworld.yaml uv run --extra alfworld python experiments/alfworld/train.py ...
```

`max_tokens_per_microbatch` 只影响多条短轨迹拼成的 micro-batch；一条 12K 的轨迹无论如何都会单独前向。若单条长轨迹仍然 OOM，最后的手段是调小 `--max-trajectory-tokens`，但这会改变实验配置，结果不再与原文可比。

**vLLM OOM。** 在配置的 `models.default` 下调小 `gpu_memory_utilization`（如 0.80）或 `max_num_seqs`（如 128）。

**vLLM 报 LoRA 模块不支持**（例如线性注意力的 `in_proj_*` / `out_proj`）。旧版 vLLM 不支持给 Qwen3.5 线性注意力投影挂 LoRA，在配置里改为只训练标准投影：

```yaml
trainer:
  lora_target_modules: standard     # 只用 q/k/v/o + gate/up/down
```

这会改变 LoRA 的作用范围，结果可能与原文不同。

**vLLM 启动时报 `max_model_len` 超过 KV cache 容量。** 在配置的 `models.default` 下设置 `max_model_len: 16384`。

**CPU 或内存吃紧。** 调试时可以减小 `--games-per-batch`；也可以加 `--no-environment-asynchronous` 改用进程内环境，环境步进会串行，速度更慢，但不影响算法。

**只有一张 GPU。** 使用 `LOCALTRIO_CONFIG=configs/1gpu.yaml`，训练与采样共用 GPU 0，速度会慢很多。

**想看 trainer / vLLM 的详细日志。** 日志默认写到 `~/.cache/localtrio/logs/`；设置 `LOCALTRIO_WORKER_LOGS=inherit` 可以直接打印到终端。

**本地结果和原文不一样。** 原文结果来自 PyTRIO 云端。本地的 LoRA 实现细节（如 `lora_alpha`、作用模块、不训练 `lm_head`）、采样引擎数值和随机种子行为都可能不同，加上在线 rollout 本身方差较大，数值有差异是正常的；请以同一套本地代码跑出的 Base / checkpoint 对比为准。
