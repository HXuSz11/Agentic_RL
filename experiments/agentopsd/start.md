# AgentOPSD 快速启动（本地 4×A6000 版）

> 完整原理、原文训练结果与代码拆解请查看 [AgentOPSD Blog](./readme.md)。本章通过 localtrio 在本地 GPU 上运行，原理与 API 对照见 [localtrio 指南](../../docs/content/guide/localtrio.md)，显存与 GPU 布局规划见 [4×A6000 硬件指南](../../docs/content/guide/hardware.md)。

以下命令均从**仓库根目录**运行。所有计算都在本机完成：ALFWorld / TextWorld 环境跑在 CPU 上，Student rollout 与 Self-Teacher 复评由 vLLM 采样，LoRA 训练由 transformers + PEFT 完成。不需要 `trio login`，不需要 API Key，也没有计费。

> **算力提示**
>
> 正式配置每个 update 采样 `16 × 8 = 128` 条长轨迹（单条最长 14,336 tokens），80 steps 共 10,240 条 rollout，并对每条轨迹再做一次整轨迹 Teacher 复评。按第 9 节的估算，4×A6000 上完整训练需要十几个小时。建议先跑通 1-step smoke，再启动完整训练。

## 0. GPU 布局

本章只有一个基模 `Qwen/Qwen3.5-4B`。Student 与 Self-Teacher 是同一个参数快照，Teacher 只是换成了带 Skill 的 prompt，两者调用的是同一个 `SamplingClient`，因此共用同一组 vLLM 副本。**直接使用默认布局即可，不需要任何配置文件**：

| 设备 | 角色 |
| --- | --- |
| GPU 0 | LoRA 训练（transformers + PEFT，bf16 + 梯度检查点） |
| GPU 1、2、3 | vLLM 采样，3 个副本做 data parallel；Student rollout 与 Teacher `compute_logprobs` 共用，每步热加载最新 LoRA |
| CPU | ALFWorld / TextWorld 环境，每条训练轨迹一个子进程 |

默认布局与预设 [`configs/4xa6000.yaml`](../../configs/4xa6000.yaml) 相同。查看本机实际生效的布局：

```bash
uv run python -m localtrio info
```

## 1. 安装依赖

```bash
uv sync --extra kernels --extra alfworld
```

若 kernels 编译失败（缺 CUDA 工具链），可以先运行 `uv sync --extra alfworld` 继续，训练会退回较慢的纯 PyTorch 实现；详见 [快速开始](../../docs/content/guide/quickstart.md)。

- `kernels`：Qwen3.5 线性注意力（Gated DeltaNet）的训练加速 kernel，强烈建议安装；
- `alfworld`：`alfworld==0.4.2` 与 `spacy==3.8.13`，TextWorld 环境在 CPU 上运行。

Python 版本以仓库根目录的 `pyproject.toml` 为准（`>=3.12`）。`uv run swanlab login` 只在使用 `--swanlab-mode online` 时需要。检查关键依赖版本：

```bash
uv run --extra alfworld python -c 'from importlib.metadata import version as v; [print(p, v(p)) for p in ("torch", "transformers", "peft", "vllm", "alfworld", "swanlab")]'
```

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
    --data-dir "$PWD/experiments/agentopsd/datasets/alfworld"
```

训练和评测默认读取该目录，无需额外传入 `--data-root`。如果已经为 [ALFWorld](../alfworld/start.md) 或 [TEMPO](../tempo/readme.md) 章节下载过同一份数据，也可以用 `--data-root` 直接指向那份目录，不必重复下载。

## 3. 查看可用参数

```bash
uv run --extra alfworld python experiments/agentopsd/train.py --help
uv run --extra alfworld python experiments/agentopsd/eval.py --help
uv run python experiments/agentopsd/analysis.py --help
```

## 4. 运行 1-step smoke

先用一个游戏、4 条轨迹检查 ALFWorld 环境、Student rollout、同快照 Teacher 复评、turn credit、PPO 和 checkpoint 链路：

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --max-steps 1 \
    --tasks-per-update 1 \
    --group-size 4 \
    --max-turns 20 \
    --teacher-concurrency 4 \
    --save-every 1 \
    --seed 3 \
    --run-name agentopsd-smoke \
    --swanlab-mode disabled
```

第一次运行要加载模型、启动 trainer 和 3 个 vLLM 副本（含 CUDA graph 捕获），启动阶段本身就要几分钟。单个游戏的 4 条轨迹可能全部成功或全部失败，此时 group-relative advantage 为 0，脚本会保留 rollout 指标并跳过参数更新。需要稳定验证 backward 时，可以提高 `--tasks-per-update`。

## 5. 启动 80-step 正式训练

```bash
uv run --extra alfworld python experiments/agentopsd/train.py \
    --base-model Qwen/Qwen3.5-4B \
    --max-steps 80 \
    --tasks-per-update 16 \
    --group-size 8 \
    --max-turns 50 \
    --max-trajectory-tokens 14336 \
    --max-action-tokens 512 \
    --temperature 1.0 \
    --top-p 1.0 \
    --teacher-concurrency 16 \
    --gamma 0.95 \
    --reshape-lambda 0.5 \
    --weight-bound 0.2 \
    --ppo-clip-low 0.8 \
    --ppo-clip-high 1.24 \
    --lora-rank 32 \
    --learning-rate 4e-6 \
    --save-every 40 \
    --seed 42 \
    --run-name agentopsd-alfworld-qwen35-4b \
    --swanlab-project agentic-rl-local-agentopsd \
    --swanlab-mode online
```

所有超参数与原文一致。启动时会先打印 run ID，Step 40、Step 80 和训练结束时各打印一次两类**本地目录**：

```text
[localtrio] training run run_YYYYMMDD_HHMMSS_xxxxxx: base=Qwen/Qwen3.5-4B rank=32 ...
Saved state: /home/<you>/.cache/localtrio/runs/run_YYYYMMDD_HHMMSS_xxxxxx/states/agentopsd-alfworld-qwen35-4b-step-40-state
Saved sampler weights: /home/<you>/.cache/localtrio/runs/run_YYYYMMDD_HHMMSS_xxxxxx/sampler_weights/agentopsd-alfworld-qwen35-4b-step-40-weights
```

`eval.py` 使用 `Saved sampler weights` 目录；state 目录用于续训，不要传给 `eval.py`。

## 6. 输出在哪里

| 内容 | 位置 |
| --- | --- |
| 训练 state（LoRA + AdamW 状态） | `~/.cache/localtrio/runs/<run_id>/states/<name>` |
| sampler weights（PEFT adapter，供评测） | `~/.cache/localtrio/runs/<run_id>/sampler_weights/<name>` |
| 每步的临时采样快照 | `~/.cache/localtrio/runs/<run_id>/tmp_sampler/`，默认保留最近 4 个（仍被引用的会多留） |
| trainer / vLLM 子进程日志 | `~/.cache/localtrio/logs/` |
| 评测 JSONL | `--output` 指定的路径，下文示例写到 `experiments/agentopsd/eval_results/` |

目录根可以用环境变量 `LOCALTRIO_HOME=/data/localtrio` 或配置文件里的 `home` 改到大容量磁盘。`train.py` 本身没有续训参数；如需从 state 继续训练，可在自己的脚本里用 `trio.ServiceClient().create_training_client_from_state_with_optimizer(<state 目录>)` 恢复。

## 7. 评测

### 7.1 快速检查评测链路

只评测 seen / unseen 各 2 个游戏，适合先检查输出路径和环境：

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --limit-per-split 2 \
    --games-per-batch 2 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/base-smoke.jsonl \
    --swanlab-mode disabled
```

### 7.2 完整评测 Base Model

完整评测覆盖 `valid_seen=140` 与 `valid_unseen=134`，一共 274 个游戏：

```bash
uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/base-qwen35-4b-eval.jsonl \
    --swanlab-mode disabled
```

### 7.3 评测 Step 40 与 Step 80 checkpoint

原文的 checkpoint 保存在 PyTRIO 云端（`trio://...` 路径），本地无法加载；请评测自己训练出来的 sampler weights。把 `RUN_DIR` 换成训练日志里打印的 run 目录：

```bash
RUN_DIR=~/.cache/localtrio/runs/run_YYYYMMDD_HHMMSS_xxxxxx

uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path "$RUN_DIR/sampler_weights/agentopsd-alfworld-qwen35-4b-step-40-weights" \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step40.jsonl \
    --swanlab-mode disabled

uv run --extra alfworld python experiments/agentopsd/eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path "$RUN_DIR/sampler_weights/agentopsd-alfworld-qwen35-4b-step-80-weights" \
    --split all \
    --games-per-batch 16 \
    --temperature 0.01 \
    --seed 42 \
    --output experiments/agentopsd/eval_results/checkpoint-eval-step80.jsonl \
    --swanlab-mode disabled
```

每个 JSONL 保存 274 条完整轨迹，最后一行是 summary。`eval.py` 默认保护已有 JSONL；确认需要重跑同一输出文件时，显式增加 `--overwrite-output`。

评测只用到采样引擎，GPU 0 会空闲。**不和训练同时运行时**，可以让 4 张卡都跑 vLLM 副本：

```bash
LOCALTRIO_SAMPLER_GPUS=0,1,2,3 uv run --extra alfworld python experiments/agentopsd/eval.py ...
```

## 8. 生成 checkpoint 对比图

`analysis.py` 默认读取：

```text
experiments/agentopsd/eval_results/base-qwen35-4b-eval.jsonl
experiments/agentopsd/eval_results/checkpoint-eval-step40.jsonl
experiments/agentopsd/eval_results/checkpoint-eval-step80.jsonl
```

默认输出路径 `experiments/agentopsd/images/agentopsd_checkpoint_evaluation.png` 正是文章里那张原文结果图，直接运行会把它覆盖掉。建议把本地结果写到别处：

```bash
uv run python experiments/agentopsd/analysis.py \
    --result-dir experiments/agentopsd/eval_results \
    --output experiments/agentopsd/eval_results/agentopsd_checkpoint_evaluation_local.png \
    --dpi 300
```

## 9. 本地资源估算

> 下表是根据模型规模、原文 token 用量和 A6000 算力做的**粗略估算，未经实测**，实际数值取决于 vLLM / transformers 版本、是否安装 `kernels`、CPU 性能和轨迹长度分布。

| 项目 | 估算 |
| --- | --- |
| GPU 0（训练） | Qwen3.5-4B bf16 权重约 8～10GB；加上 LoRA、AdamW 和 14K token 单条轨迹的激活（梯度检查点 + 分块 logprob），峰值大约 20～35GB |
| GPU 1～3（vLLM） | 每张卡按 `gpu_memory_utilization=0.85` 预占约 40GB，权重之外都用作 KV cache；Teacher 对 16K 前缀算 `compute_logprobs` 时会有额外的整词表 logprob 显存尖峰 |
| 单个 update | Student rollout 约 3～7 分钟，Teacher 复评约 1～3 分钟，训练约 3～5 分钟 |
| 1-step smoke | 约 5～15 分钟，大部分是模型加载与 vLLM 启动 |
| 80-step 正式训练 | 约 10～20 小时 |
| 一次完整评测（274 局） | 约 0.5～1.5 小时 |
| CPU / 内存 | 正式配置同时持有 128 个 TextWorld 子进程（16 个游戏组 × 8 条分支），每轮各游戏组的环境步进在线程池里并发执行；建议 16 核以上、64GB 以上内存。CPU 核数少时环境会成为瓶颈 |
| 磁盘 | 模型约 10GB；每个 sampler weights 是一个 rank-32 LoRA adapter，state 额外包含 AdamW 状态，体积约为前者的数倍 |

训练时间的估算依据：原文 PyTRIO 会话的用量统计为 train 30.52M tokens、sample 9.98M tokens、prefilling 762.21M tokens（其中 549.26M 命中缓存），折合每个 update 约 38 万训练 token、12 万采样 token。

## 10. 常见问题

**训练进程（GPU 0）OOM。** 先确认装了 `--extra kernels`。再写一份配置文件调小显存相关参数（文件可以放在任意位置）：

```yaml
# ~/localtrio-agentopsd.yaml：在默认 4×A6000 布局上做保守的显存设置
trainer:
  gpus: [0]
  max_tokens_per_microbatch: 8192   # 默认 16384；单条超过上限的轨迹会单独成批
  logits_chunk_size: 512            # 默认 1024；降低 lm_head 分块大小
models:
  default:
    gpus: [1, 2, 3]
    gpu_memory_utilization: 0.80    # 默认 0.85；也给 Teacher compute_logprobs 的显存尖峰（约 0.5~1GB）留余量
    max_model_len: 16384            # 与 MAX_SEQUENCE_TOKENS 对齐
```

```bash
LOCALTRIO_CONFIG=~/localtrio-agentopsd.yaml uv run --extra alfworld python experiments/agentopsd/train.py ...
```

`max_tokens_per_microbatch` 只影响多条短轨迹拼成的 micro-batch；一条 14K 的轨迹无论如何都会单独前向。若单条长轨迹仍然 OOM，最后的手段是调小 `--max-trajectory-tokens`，但这会改变实验配置，结果不再与原文可比。

**vLLM 在 Teacher 复评阶段 OOM。** 使用上面 YAML 中 `models.default` 的三项设置，必要时再把 `--teacher-concurrency` 从 16 调小（只影响并发，不影响算法）。

**vLLM 报 LoRA 模块不支持**（例如线性注意力的 `in_proj_*` / `out_proj`）。旧版 vLLM 不支持给 Qwen3.5 线性注意力投影挂 LoRA，在配置里改为只训练标准投影：

```yaml
trainer:
  lora_target_modules: standard     # 只用 q/k/v/o + gate/up/down
```

这会改变 LoRA 的作用范围，结果可能与原文不同。

**vLLM 启动时报 `max_model_len` 超过 KV cache 容量。** 在配置的 `models.default` 下设置 `max_model_len: 16384`。

**CPU 或内存吃紧。** 正式配置每步有 128 个 TextWorld 子进程。调试时可以减小 `--tasks-per-update`；也可以加 `--no-environment-asynchronous` 改用进程内环境，环境步进会串行，速度更慢，但不影响算法。

**只有一张 GPU。** 使用 `LOCALTRIO_CONFIG=configs/1gpu.yaml`，训练与采样共用 GPU 0，速度会慢很多。

**想看 trainer / vLLM 的详细日志。** 日志默认写到 `~/.cache/localtrio/logs/`；设置 `LOCALTRIO_WORKER_LOGS=inherit` 可以直接打印到终端。

**本地结果和原文不一样。** 原文结果来自 PyTRIO 云端。本地的 LoRA 实现细节（如 `lora_alpha`、作用模块）、采样引擎数值和随机种子行为都可能不同，加上 ALFWorld 在线 rollout 本身方差较大，数值有差异是正常的；请以同一套本地代码跑出的 Base / checkpoint 对比为准。
