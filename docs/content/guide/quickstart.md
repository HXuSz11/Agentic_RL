---
title: 安装与第一个实验：在 4×A6000 上跑通 GRPO
description: 安装 PyTorch、transformers、PEFT 与 vLLM，检查 GPU 布局，下载 Qwen3.5-4B，并在 GSM8K 上用本地 GPU 启动第一个 GRPO 训练。
---

# 安装与第一个实验

本地版不需要任何训练平台账号：实验脚本在主进程里组织数据和训练循环，训练与采样由 [localtrio](./localtrio) 在你的 GPU 上执行。

## 准备

- Linux x86_64，NVIDIA 驱动足够新（PyPI 上的 PyTorch / vLLM 默认 wheel 基于较新的 CUDA；装完后用下面的命令确认 `torch.cuda.is_available()` 为 `True`）。
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

`causal-conv1d` 没有匹配的预编译 wheel 时会从源码编译，需要本机有 CUDA 工具链；编译失败可以先跳过这一步，训练会退回较慢的纯 PyTorch 实现。ALFWorld / TEMPO / AgentOPSD 还需要 alfworld 依赖：`uv sync --extra kernels --extra alfworld`（kernels 编译失败时用 `uv sync --extra alfworld`）。

::: warning uv sync 会移除没写上的 extra
`uv sync` 默认精确同步：这次命令没有指定的 extra 会被卸载。比如装过 kernels 之后再单独执行 `uv sync --extra dev` 或 `uv sync --extra alfworld`，flash-linear-attention / causal-conv1d 会被删掉，训练悄悄退回纯 PyTorch 实现。所以每次都把要用的 extras 写全（如 `uv sync --extra kernels --extra dev`），或者加 `--inexact` 只装不删。`uv run` 不会移除已经装好的包。
:::

## 检查环境与 GPU 布局

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
uv run python -m localtrio info
```

`info` 会列出当前布局。不提供配置文件时，4 卡机器上的默认布局是：GPU0 训练，GPU1/2/3 各一个 vLLM 采样副本。其他布局见[显存与 GPU 布局](./hardware)。

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

```bash
uv run python experiments/grpo/01-demo-sync.py \
    --steps 2 --batch-size 4 --group-size 4 --max-tokens 512 \
    --loss-fn importance_sampling --swanlab-mode disabled
```

第一次运行会依次看到：

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

保留模型名称、checkpoint 目录、数据划分、随机种子、采样预算、GPU 布局（`python -m localtrio info` 的输出）和评测命令。对比实验时，先对齐这些条件，再讨论方法的效果。

下一步可以从[实验手册](/experiments/)选择方向，或查看[学习路线](./learning-path)。
