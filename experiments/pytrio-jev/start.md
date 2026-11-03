# Jev 决策模型：本地快速启动

本页是原教程 PyTrio-Jev 快速启动的本地 GPU 版本：训练与推理通过 [localtrio](../../docs/content/guide/localtrio.md) 在 4×RTX A6000 上运行，不需要 `trio login`、API key，也没有计费。原理与原文结果见 [正文](./readme.md)，显存规划见 [4×A6000 硬件指南](../../docs/content/guide/hardware.md)。

## 0. 安装环境

在仓库根目录执行（Python ≥ 3.12）：

```bash
uv sync --extra kernels            # kernels：Qwen3.5 线性注意力的训练加速，强烈建议安装
uv run python -m localtrio info    # 查看 GPU 布局、vLLM 版本
swanlab login                      # 可选；也可以给训练脚本加 --swanlab-mode offline / disabled
```

模型默认从 HuggingFace 下载。国内网络可以设置 `HF_ENDPOINT`（HF 镜像），或改用 ModelScope：

```bash
export LOCALTRIO_MODEL_SOURCE=modelscope
uv run python -m localtrio download Qwen/Qwen3.5-4B   # 可选：提前下载基模（约 9–10GB）
```

数据准备脚本还会从 HuggingFace 下载 `typed-decisions` / `ToolACE`，并从 `raw.githubusercontent.com` 下载 Intern-Decision 的评测文件，网络受限时请配置代理或镜像。

## 1. GPU 布局

这一章只有一个基模（`Qwen/Qwen3.5-4B`），直接用**内置默认布局**，不需要任何配置文件（等价于 [`configs/4xa6000.yaml`](../../configs/4xa6000.yaml)）：

| 脚本 | GPU0 | GPU1 / GPU2 / GPU3 |
| --- | --- | --- |
| `train.py` | LoRA 训练 | 空闲（训练过程不采样） |
| `inference.py` / `calibrate.py` / `eval.py` | 空闲 | 各一个 vLLM 副本 |

- 评测时想把 GPU0 也用上：在命令前加 `LOCALTRIO_SAMPLER_GPUS=0,1,2,3`。
- 只有一张 48GB 卡：在命令前加 `LOCALTRIO_CONFIG=../../configs/1gpu.yaml`（训练与采样共用 GPU0，速度更慢）。

## 2. 准备数据

以下命令都在 `experiments/pytrio-jev/` 目录内运行：

```bash
cd experiments/pytrio-jev

# typed-decisions 960 案例 + ToolACE 1200 条，以及五项评测集
uv run python 00-prepare-data.py
```

产物在 `datasets/`：`train.jsonl`（2160 条）、`dev.jsonl`、`calibration.jsonl` 和 `eval/*.jsonl`。

## 3. 训练

```bash
# 冒烟：64 条跑 1 个 epoch，确认 loss 下降、marker 对齐正常
uv run python train.py --max-samples 64 --epochs 1 --save-every 0 --swanlab-mode offline

# 正式训练：2160 条 × 2 epoch ÷ 16 = 270 步
uv run python train.py --epochs 2 --batch-size 16 --save-every 60 --swanlab-mode online
```

每次保存都会打印一行 `保存权重 pytrio-jev-qwen35-4b-step-<N>: <本地目录>`。权重是可以直接给 vLLM 加载的 PEFT adapter 目录，位置是：

```text
~/.cache/localtrio/runs/<run_id>/sampler_weights/pytrio-jev-qwen35-4b-step-<N>
```

找不到输出时可以直接列出来（目录根可用 `LOCALTRIO_HOME` 修改）：

```bash
ls -d ~/.cache/localtrio/runs/*/sampler_weights/pytrio-jev-*
WEIGHTS=~/.cache/localtrio/runs/<run_id>/sampler_weights/pytrio-jev-qwen35-4b-step-270
```

trainer / vLLM 子进程的日志在 `~/.cache/localtrio/logs/`。

## 4. 推理、校准与评测

`--weights` 直接传上一步的本地目录（原版这里是 `trio://` 路径）；不传则评 base 模型。

```bash
# 内置样例试跑推理
uv run python inference.py --weights "$WEIGHTS"

# 温度校准：在 calibration 划分上拟合 T，写出 outputs/calibration.json
uv run python calibrate.py --weights "$WEIGHTS"

# 五项基准评测（先校准，后读 test）
uv run python eval.py --weights "$WEIGHTS" --calibration outputs/calibration.json --tag step-270

# base 模型对照
uv run python eval.py --tag base

# 汇总对照参考基线，写出 outputs/summary.md
uv run python analysis.py
```

评测结果写在 `outputs/eval-<tag>.json`。`--concurrency`（默认 15）只控制同时在途的请求数，localtrio 会把请求分给各个 vLLM 副本；3 个副本时可以调到 32–64 来加速，不影响评测口径。

## 5. 资源估算

> 以下为按模型规模和 token 量推算的**估算值，未实测**。

| 项目 | 估算 |
| --- | --- |
| GPU0 显存（训练） | 约 15–25GB：权重 9–10GB + LoRA/AdamW 约 1–2GB + 激活（梯度检查点） |
| 采样卡显存 | 每个 vLLM 副本预占约 40GB（`gpu_memory_utilization=0.85`） |
| 训练 270 步 | 约 0.5–1.5 小时（含模型加载；未安装 kernels 会明显更慢） |
| 校准 + 五项评测 | 合计约十几到几十分钟（3 个 vLLM 副本） |
| 磁盘 | 基模约 9–10GB；每个 LoRA 权重几十到几百 MB |

## 6. 常见问题

**怎么改 localtrio 配置？** 写一份自己的 YAML（例如复制 `configs/4xa6000.yaml` 再修改），用 `LOCALTRIO_CONFIG` 指向它；也可以把它命名为 `localtrio.yaml` 放在**当前工作目录**（本章是 `experiments/pytrio-jev/`），会被自动读取。例如：

```yaml
trainer:
  gpus: [0]
  max_tokens_per_microbatch: 8192   # 训练 OOM 时调小
  # lora_target_modules: standard  # 仅当 vLLM 报 LoRA 模块不支持时启用（需重新训练）
models:
  default:
    gpus: [1, 2, 3]
    gpu_memory_utilization: 0.80    # 采样 OOM 时调小
    max_num_batched_tokens: 1024    # 默认 2048；长 prompt + prompt logprobs 显存紧张时调小
```

```bash
LOCALTRIO_CONFIG=../../my-jev.yaml uv run python train.py ...
```

- **训练 OOM**：调小 `trainer.max_tokens_per_microbatch`（如 8192 或 4096）。它只改变 micro-batch 切分，梯度仍按整个 batch 的 active token 数归一化，不改变实验设置；不建议改 `--batch-size`。
- **评测 / 校准时 vLLM OOM**：受限 softmax 需要整段 prompt 的 top-k prompt logprobs，长 prompt 比较吃显存。依次尝试调小 `--concurrency`、把 `models.default.max_num_batched_tokens` 从默认 2048 调小到 1024、调低 `models.default.gpu_memory_utilization`。
- **vLLM 报 LoRA 不支持某个模块**（如 Qwen3.5 线性注意力的 `in_proj_*` / `out_proj`）：把 `trainer.lora_target_modules` 改成 `standard`（只用 q/k/v/o + gate/up/down），然后**重新训练**，已有 adapter 无法直接转换。
- **报 `prompt logprobs ... max_logprobs`**：某个字段的候选数超过了 vLLM 的上限（默认 64），调大 `models.default.max_logprobs`。
- **报 `trio://` 是 PyTRIO 云端路径**：本地版只接受 `save_weights_for_sampler` 返回的本地目录，请把命令里的 `trio://...` 换成 `~/.cache/localtrio/runs/...` 下的目录。
- **数值与原文不同**：原文结果来自 PyTRIO 云端；本地版默认不训练 lm_head（见正文第 4 节），且 vLLM 批处理存在数值误差，准确率与原文会有差异（幅度未实测；原文中仅采样端数值误差就带来约 0.1–2 个点的波动）。
