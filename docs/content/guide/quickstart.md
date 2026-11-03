---
title: 安装与第一个实验：用自己的显卡跑通 GRPO
description: 需要几张卡、安装 PyTorch、transformers、PEFT 与 vLLM，检查 GPU 布局，下载 Qwen3.5-4B，并在 GSM8K 上用本地显卡启动第一个 GRPO 训练（1 张 48GB 或 2 张 24GB 卡即可）。
---

# 安装与第一个实验

本地版不需要任何训练平台账号：实验脚本在主进程里组织数据和训练循环，训练与采样由 [localtrio](./localtrio) 在你自己的 NVIDIA 显卡上执行。

## 需要几张卡

不需要 4 张卡。localtrio 按检测到的卡数自动分工：默认 GPU 0 训练 LoRA，其余卡各跑一个 vLLM 采样副本；只有 1 张卡时两者共用。卡越多采样越快，实验设置不用改。本页第一个 GRPO 实验的跑法（速度为估算）：

| 你的卡 | 怎么跑 | 速度（相对 4 卡，估算） |
| --- | --- | --- |
| 1 张 ≥48GB | 直接运行也会自动共用 GPU 0；建议加 `LOCALTRIO_CONFIG=configs/1gpu.yaml`（训练 micro-batch 8192） | 约 1/3～1/4 |
| 2 张 ≥24GB | 直接运行，不需要配置文件（GPU 0 训练、GPU 1 采样）；24GB 卡加 `LOCALTRIO_CONFIG=configs/24gb.yaml` | 约 1/2～1/3 |
| 3 张、4 张或更多 | 直接运行，GPU 0 训练，其余每张卡一个采样副本；24GB 卡同样加 `configs/24gb.yaml` | 3 张约 2/3～3/4；4 张为基准 |

显卡要求是 Ampere 或更新的架构（bf16），每张卡最好 ≥24GB；只有 1 张 24GB / 32GB 的卡时放不下训练加采样两份 4B 权重和 KV cache，只能用 `--base-model Qwen/Qwen3.5-0.8B` 加小参数跑通代码路径（已不是原文的实验设置）。各章的最低与推荐配置、24GB / 48GB / 80GB 显卡分别能做什么，见[显存与 GPU 布局](./hardware)。

## 准备

- Linux x86_64，NVIDIA 驱动 ≥ 580：仓库的 `uv.lock` 锁定了 vLLM 0.33.0 与它要求的 torch 2.14.0，后者在 PyPI 上的 wheel 基于 CUDA 13.0。装完后用下面的命令确认 `torch.cuda.is_available()` 为 `True`。
- Python 3.12 或 3.13，[uv](https://docs.astral.sh/uv/)。
- 磁盘：模型（4B 约 10GB，27B 约 55GB）+ 每个 run 的 checkpoint，建议预留 100GB 以上 SSD。
- 网络：能访问 HuggingFace（或配置镜像 / ModelScope）以及各章节用到的数据集。

## 获取代码与安装依赖

```bash
git clone https://github.com/HXuSz11/Agentic_RL.git
cd Agentic_RL
uv sync                      # torch、transformers、peft、vllm 与实验依赖
uv sync --extra kernels      # 推荐：Qwen3.5 线性注意力的训练 kernel（flash-linear-attention、causal-conv1d）
```

`uv sync` 按仓库里的 `uv.lock` 安装，各包版本与作者核对时一致（例如 swanlab 0.10.x：更早的 0.6.x 不接受各章默认的 `--swanlab-mode online`）。`causal-conv1d` 没有匹配的预编译 wheel 时会从源码编译，需要本机有与 torch 一致的 CUDA 13.x 工具链（`nvcc`）；编译失败可以先跳过这一步，训练会退回较慢的纯 PyTorch 实现。ALFWorld / TEMPO / AgentOPSD 还需要 alfworld 依赖：`uv sync --extra kernels --extra alfworld`（kernels 编译失败时用 `uv sync --extra alfworld`）。

::: warning uv sync 会移除没写上的 extra
`uv sync` 默认精确同步：这次命令没有指定的 extra 会被卸载。比如装过 kernels 之后再单独执行 `uv sync --extra dev` 或 `uv sync --extra alfworld`，flash-linear-attention / causal-conv1d 会被删掉，训练悄悄退回纯 PyTorch 实现。所以每次都把要用的 extras 写全（如 `uv sync --extra kernels --extra dev`），或者加 `--inexact` 只装不删。`uv run` 不会移除已经装好的包。
:::

## 检查环境与 GPU 布局

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
uv run python -m localtrio info
```

`info` 会列出检测到的卡和当前布局。不提供配置文件时按卡数自动分配：

| 卡数 | 训练 | 采样 |
| --- | --- | --- |
| 1 张 | GPU 0 | 同在 GPU 0（vLLM 只占 40% 显存） |
| 2 张 | GPU 0 | GPU 1，1 个副本 |
| 4 张 | GPU 0 | GPU 1–3，各一个副本 |

例如 2 张卡、没有配置文件时，输出里应能看到：

```text
config file : (内置默认值)
GPUs        : [0, 1]
trainer     : GPU [0]  lora=all-linear alpha=32.0
sampler[default] : GPU [1] tp=1 replicas=1 mem=0.85
```

想换分工可以不写配置文件，直接用环境变量，例如 `LOCALTRIO_TRAINER_GPUS=0 LOCALTRIO_SAMPLER_GPUS=1,2`；预设布局见[显存与 GPU 布局](./hardware#预设布局)。

## 下载模型

```bash
uv run python -m localtrio download Qwen/Qwen3.5-4B
```

国内网络可以二选一：

```bash
export HF_ENDPOINT=https://hf-mirror.com      # 走 HuggingFace 镜像
export LOCALTRIO_MODEL_SOURCE=modelscope      # 或改从 ModelScope 下载
```

已经下载好的模型可以在配置文件里用 `model_paths` 指向本地目录，见 [localtrio 配置](./localtrio#配置)。

## 实验记录（可选）

```bash
uv run swanlab login
```

不想记录时，给训练脚本加 `--swanlab-mode disabled`（大多数章节支持这个参数，以各章快速启动为准）。

## GPU 自检（第一次在新机器上运行前）

```bash
uv run python scripts/gpu_smoke_test.py                 # 默认 Qwen/Qwen3.5-4B，几分钟
uv run python scripts/gpu_smoke_test.py --image         # 同时检查图文输入
```

它会依次检查：vLLM 组采样是否返回全部序列、采样端 logprob 是否与训练端一致（on-policy ratio≈1）、训练一步后新 LoRA 是否被采样端热加载、`compute_logprobs` / top-k、`save_state` 往返。全部 `PASS` 再开始正式实验；某项 `FAIL` 时把输出和 `~/.cache/localtrio/logs/` 里的引擎日志一起贴到 Issue。

## 运行第一个 GRPO

不需要配置文件，直接运行（GPU 按卡数自动分配）：

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 2 --batch-size 4 --group-size 4 --max-tokens 512 \
    --loss-fn importance_sampling --swanlab-mode disabled
```

- 只有 1 张 ≥48GB 的卡：建议在命令前加 `LOCALTRIO_CONFIG=configs/1gpu.yaml`。
- 24GB 卡（2 张或更多）：建议在命令前加 `LOCALTRIO_CONFIG=configs/24gb.yaml`（训练 micro-batch 8192，GPU 仍按卡数自动分配）；vLLM 启动时报 KV cache 不足，就把这个文件里 `# max_model_len: 16384` 的注释去掉。

第一次运行会依次看到（2 张卡时；单卡时采样也在 GPU 0，卡更多时还会启动 `#1`、`#2`…）：

```text
[localtrio] starting trainer[Qwen/Qwen3.5-4B] on GPU 0 (log: ~/.cache/localtrio/logs/...)
[localtrio] training run run_xxx: base=Qwen/Qwen3.5-4B rank=32 trainable=...M targets=q_proj,...
[localtrio] starting sampler[Qwen/Qwen3.5-4B]#0 on GPU 1 ...
[localtrio] sampler[Qwen/Qwen3.5-4B]#0 ready
Step  0 | reward: ... | degenerate: ... | loss_mean: ...
```

vLLM 第一次启动要编译 CUDA graph，可能需要几分钟；之后同一个脚本里的每一步都复用已经启动的引擎。训练结束时打印的 `path` 就是最终 LoRA 权重目录（`~/.cache/localtrio/runs/<run>/sampler_weights/<name>`）。

跑通之后，按 [GRPO 快速启动](/experiments/grpo/run)调大 `--steps`、`--batch-size`，并配合 [GRPO 算法拆解](/experiments/grpo/)对照「准备 prompt → 组采样 → 奖励与 advantage → 更新策略」四个步骤。

::: tip 引擎日志
引擎在子进程里运行，日志默认写到 `~/.cache/localtrio/logs/`。启动失败时报错会附上日志末尾；想直接在终端看到 vLLM 和训练端日志，运行前设置 `LOCALTRIO_WORKER_LOGS=inherit`。
:::

## 没有 GPU 也能检查代码吗？

可以检查代码路径，但只适合微型模型：

```bash
uv sync --extra kernels --extra dev   # 机器上编译不了 kernels 时改成 uv sync --extra dev
uv run pytest tests -q
```

测试会在 CPU 上构造随机初始化的微型 Qwen3 / Qwen3.5-VL 模型，走一遍训练、采样、保存和恢复。真实章节请在 GPU 机器上运行。

## 记录一次可复查的实验

保留模型名称、checkpoint 目录、数据划分、随机种子、采样预算、GPU 型号与张数、布局（`python -m localtrio info` 的输出）和评测命令。对比实验时，先对齐这些条件，再讨论方法的效果。

下一步可以从[实验手册](/experiments/)选择方向，或查看[学习路线](./learning-path)。
