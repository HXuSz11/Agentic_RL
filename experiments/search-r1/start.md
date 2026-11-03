# Search-R1 快速启动

本页是本地 4×RTX A6000（48GB）版的运行手册：训练与采样通过 [localtrio](../../docs/content/guide/localtrio.md) 跑在自己的 GPU 上，搜索仍调用 DeepSeek 官方 Search API 或 Wikipedia。算法讲解见 [readme.md](./readme.md)。

以下命令均从**仓库根目录**运行。

## 1. 安装环境

```bash
uv sync --extra kernels          # torch / transformers / peft / vllm + Qwen3.5 线性注意力训练 kernel
uv run python -m localtrio info  # 查看 GPU、vLLM 版本与当前布局
uv run swanlab login             # 可选；不记录到云端时给训练命令加 --swanlab-mode disabled
```

- 不需要 `trio login`、`PYTRIO_API_KEY`，也没有训练计费。
- 第一次运行时会自动下载 `Qwen/Qwen3.5-4B`（默认从 HuggingFace）。国内网络可以先设置 `export HF_ENDPOINT=https://hf-mirror.com`，或 `export LOCALTRIO_MODEL_SOURCE=modelscope`；也可以提前下载：`uv run python -m localtrio download Qwen/Qwen3.5-4B`。
- 不装 `kernels` 扩展也能跑，但 Qwen3.5 的线性注意力会退回纯 PyTorch 实现，训练更慢、更占显存。

## 2. GPU 布局

本章使用内置默认布局，不需要任何配置文件（等价于 [`configs/4xa6000.yaml`](../../configs/4xa6000.yaml)）：

| GPU | 角色 |
| --- | --- |
| GPU0 | LoRA 训练（transformers + PEFT，bf16 + 梯度检查点） |
| GPU1 / GPU2 / GPU3 | 各一个 vLLM 采样副本（data parallel，LoRA 热加载） |

搜索请求在 CPU 上由主进程发出。显存与布局的整体说明见 [4×A6000 硬件规划](../../docs/content/guide/hardware.md)。

## 3. 准备数据

```bash
uv run python experiments/search-r1/prepare_data.py
```

数据从 ModelScope 下载固定版本，生成：

```text
experiments/search-r1/datasets/
├── train.jsonl    # 169,615 道训练题（NQ + HotpotQA）
├── dev.jsonl      # 固定 70 道评测题（7 个 benchmark × 10）
└── test.jsonl     # 完整评测池
```

## 4. 配置搜索后端

```bash
cp experiments/search-r1/.env.example experiments/search-r1/.env
```

默认使用 DeepSeek 官方 Search API，在 `experiments/search-r1/.env` 中填写：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
```

也兼容 `DEEPSEEK_KEY`。每次请求默认返回 Top 3，把 `snippet` 短摘要连同标题、来源、URL 交给模型。DeepSeek Search 按次计费，价格见[官方说明](https://api-docs.deepseek.com/zh-cn/guides/search/)。

Wikipedia 后端（`--search-backend wikipedia`）无需 API Key，按 Wikimedia 公开限额默认 3 个并发。

## 5. 训练

DeepSeek Search 后端，20 step：

```bash
uv run python experiments/search-r1/train.py \
    --max-steps 20 \
    --questions-per-batch 8 \
    --group-size 8 \
    --save-every 5 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend deepseek \
    --run-name search-r1-qwen35-4b-deepseek \
    --swanlab-mode online
```

Wikipedia 后端：把 `--search-backend` 改成 `wikipedia`，`--run-name` 改成 `search-r1-qwen35-4b-wikipedia`。

正式训练可把 `--max-steps` 改为 `100`、`--save-every` 改为 `50`。每个 step 把完整 rollout batch（8 题 × 8 条轨迹）一次提交给 `forward_backward`，由 localtrio 的 trainer 按 `trainer.max_tokens_per_microbatch` 自动拆 micro-batch、按整批 active token 数归一化并累积梯度，再执行一次 `optim_step`。单条训练轨迹上限为 `8,192` tokens。

**输出在哪里：** 每 `--save-every` 步和训练结束时各保存一次，终端打印两行本地路径：

```text
Saved state: ~/.cache/localtrio/runs/<run_id>/states/<run-name>-step-20-state
Saved sampler weights: ~/.cache/localtrio/runs/<run_id>/sampler_weights/<run-name>-step-20-weights
```

`states/` 用于续训，`sampler_weights/` 用于评测；最终 checkpoint 名为 `<run-name>-final-*`。目录根可用 `LOCALTRIO_HOME` 改到大容量磁盘。引擎日志在 `~/.cache/localtrio/logs/`。

**续训：** `--resume-state ~/.cache/localtrio/runs/<run_id>/states/<run-name>-step-20-state`。它只恢复 LoRA 权重（`create_training_client_from_state`），不恢复优化器状态，step 计数和数据位置都从 0 开始。

## 6. 评测

评测只启动 vLLM，不启动 trainer。训练结束后可以在命令前加 `LOCALTRIO_SAMPLER_GPUS=0,1,2,3`，用 4 张卡采样；训练进行中不要同时评测。

先评测 Base Model：

```bash
uv run python experiments/search-r1/eval.py \
    --batch-size 16 \
    --base-model Qwen/Qwen3.5-4B \
    --search-backend deepseek \
    --search-count 3 \
    --search-concurrency 16 \
    --search-timeout 60 \
    --output experiments/search-r1/eval_result/eval_results_base_deepseek_api.jsonl
```

再评测训练得到的 sampler weights（`--model-path` 填训练终端打印的 `Saved sampler weights` 目录）：

```bash
uv run python experiments/search-r1/eval.py \
    --batch-size 16 \
    --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/search-r1-qwen35-4b-deepseek-step-20-weights \
    --search-backend deepseek \
    --search-count 3 \
    --search-concurrency 16 \
    --search-timeout 60 \
    --output experiments/search-r1/eval_result/eval_results_rl_step_20_deepseek_api.jsonl
```

- 评测 Wikipedia 后端时，两条命令都改为 `--search-backend wikipedia --search-concurrency 3 --search-timeout 15`。Base 与 checkpoint 必须使用相同的后端、摘要字段和搜索配置。
- 先加 `--limit 20` 可以快速检查整条链路。
- `--output` 是逐题 JSONL，最后一行是 `type=summary` 的汇总指标；终端也会打印同一份指标。

## 7. 绘图

```bash
uv run python experiments/search-r1/analyse.py --preset deepseek-api \
    --output experiments/search-r1/eval_result/local_base_vs_step30.png
```

`deepseek-api` preset 读取 `eval_result/eval_results_base.jsonl` 和 `eval_result/eval_results_rl_step_30.jsonl`。不加 `--output` 时会写到 `images/deepseek_api_base_vs_step30.png`，覆盖 readme 引用的原文结果图。

## 8. 资源估算

> 以下为按模型规模与原文配置做的**粗略估算，未经实测**；搜索 API 的延迟波动对总耗时影响很大。

| 项目 | 估算（未实测） |
| --- | --- |
| GPU0 训练显存 | 约 15–25 GB（4B bf16 权重约 9 GB + LoRA/AdamW + 激活，单条 ≤ 8,192 token） |
| GPU1–3 采样显存 | 每卡约 40 GB（vLLM 按 `gpu_memory_utilization=0.85` 预占） |
| 单个 step | rollout 约 2–4 分钟 + 训练约 1–3 分钟 |
| 20 step / 100 step | 约 1.5–3 小时 / 约 7–13 小时 |
| 70 题评测 | 约 10–20 分钟 |
| 磁盘 | 每次保存的 LoRA 几十到几百 MB；每步的临时采样权重只保留最近 4 份 |

## 9. 常见问题

改 localtrio 参数时，复制一份预设再用 `LOCALTRIO_CONFIG` 指定，例如：

```bash
cp configs/4xa6000.yaml my-search-r1.yaml   # 按下面的说明修改
LOCALTRIO_CONFIG=my-search-r1.yaml uv run python experiments/search-r1/train.py ...
```

**训练卡 OOM（trainer 日志出现 `CUDA out of memory`）**：在 YAML 里调小 `trainer.max_tokens_per_microbatch`（16384 → 8192 → 4096），必要时再调小 `trainer.logits_chunk_size`（1024 → 512）。这只改变 micro-batch 的切法，不改变梯度。

**vLLM 启动失败或采样卡 OOM**：调小 `models.default.gpu_memory_utilization`（0.85 → 0.75），或设置 `models.default.max_model_len: 10240`（需大于本章 8,192 的轨迹上限）。

**vLLM 加载 LoRA 时报 target modules 不支持**：把 `trainer.lora_target_modules` 改成 `standard`（只训练 q/k/v/o + MLP），重新训练。

**`--model-path` 报“不是 LoRA 权重目录”**：要传 `sampler_weights/` 下的目录，而不是 `states/`；也不能再用原文的 `trio://` 路径。

**搜索大量 429 或超时**：调小 `--search-concurrency`，或调大 `--search-timeout`；先在 SwanLab 里确认 `search/success_rate` 正常再看 reward。

**想看引擎日志**：日志在 `~/.cache/localtrio/logs/`；设置 `LOCALTRIO_WORKER_LOGS=inherit` 可直接打印到终端。
