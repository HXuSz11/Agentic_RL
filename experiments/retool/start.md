# ReTool 快速启动

本页是本地 4×RTX A6000（48GB）版的运行手册：训练与采样通过 [localtrio](../../docs/content/guide/localtrio.md) 跑在自己的 GPU 上，代码解释器沙箱在本机 CPU 上起子进程执行。算法讲解见 [readme.md](./readme.md)。

以下命令均从**仓库根目录**运行。

## 1. 安装环境

```bash
uv sync --extra kernels          # torch / transformers / peft / vllm + Qwen3.5 线性注意力训练 kernel
uv run python -m localtrio info  # 查看 GPU、vLLM 版本与当前布局
uv run swanlab login             # 可选；不记录到云端时给训练命令加 --swanlab-mode disabled
```

- 不需要 `trio login`、`PYTRIO_API_KEY`，也没有训练计费。本章也没有外部 API：沙箱、reward 判定（`math_verify`）都在本地。
- 第一次运行时会自动下载 `Qwen/Qwen3.5-4B`（默认从 HuggingFace）。国内网络可以先设置 `export HF_ENDPOINT=https://hf-mirror.com`，或 `export LOCALTRIO_MODEL_SOURCE=modelscope`；也可以提前下载：`uv run python -m localtrio download Qwen/Qwen3.5-4B`。
- 沙箱用当前虚拟环境的 Python（`sys.executable`）执行模型写的代码，模型能 `import` 的库就是 `uv sync` 装好的那些（numpy、sympy 等）。

## 2. GPU 布局

本章使用内置默认布局，不需要任何配置文件（等价于 [`configs/4gpu.yaml`](../../configs/4gpu.yaml)）：

| 资源 | 角色 |
| --- | --- |
| GPU0 | LoRA 训练（transformers + PEFT，bf16 + 梯度检查点） |
| GPU1 / GPU2 / GPU3 | 各一个 vLLM 采样副本（data parallel，LoRA 热加载） |
| CPU | 代码沙箱，默认 8 个并发子进程（`--sandbox-workers`），单次 30 秒超时（`--sandbox-timeout`） |

显存与布局的整体说明见 [4×A6000 硬件规划](../../docs/content/guide/hardware.md)。

## 3. 准备数据

训练集（DAPO-Math-17k）与 AIME25 评测集：

```bash
uv run python experiments/retool/prepare_data.py
uv run python experiments/opsd/00-datasets.py --only aime25
```

生成：

```text
experiments/retool/datasets/
├── train.jsonl                  # 训练集（question + answer）
└── dev.jsonl                    # 50 条开发集
experiments/opsd/datasets/aime_2025/   # AIME25，30 题（eval.py 默认读取这里）
```

`prepare_data.py` 默认从 HuggingFace Hub 加载 `BytedTsinghua-SIA/DAPO-Math-17k`，原始数据存在 HF 缓存目录（默认 `~/.cache/huggingface/`），不会写进 `datasets/`；国内网络可以设置 `HF_ENDPOINT` 指向镜像。`datasets/raw/dapo-math-17k.parquet` 是可选输入：网络不稳时先手动下载到这里，脚本会优先读取它。

## 4. 训练

先运行 20-step 验证实验：

```bash
uv run python experiments/retool/train.py \
    --max-steps 20 \
    --save-every 5 \
    --run-name retool-qwen35-4b-step20
```

完整实验把 `--max-steps` 改为 `200`、`--save-every` 改为 `50`，`--run-name` 改为 `retool-qwen35-4b-step200`。默认每步 8 题 × 8 条轨迹（`--questions-per-batch 8 --group-size 8`），其余超参与原文一致：lr `4e-5`、Adam β (0.9, 0.95)、`ppo` clip 0.8 / 1.28、单条轨迹 ≤ 8,192 token。

只想检查代码路径时，可以先跑一个小规模测试：

```bash
uv run python experiments/retool/train.py \
    --max-steps 2 \
    --questions-per-batch 2 \
    --group-size 8 \
    --save-every 1 \
    --swanlab-mode disabled
```

**输出在哪里：** 每 `--save-every` 步和训练结束时各保存一次，终端打印两行本地路径：

```text
Saved state: ~/.cache/localtrio/runs/<run_id>/states/<run-name>-step-50-state
Saved sampler weights: ~/.cache/localtrio/runs/<run_id>/sampler_weights/<run-name>-step-50-weights
```

最终 checkpoint 名为 `<run-name>-final-state` / `<run-name>-final-weights`。目录根可用 `LOCALTRIO_HOME` 改到大容量磁盘；引擎日志在 `~/.cache/localtrio/logs/`。本章的 `train.py` 没有续训参数，state 目录主要用作保险。

## 5. 评测

评测只启动 vLLM，不启动 trainer。训练结束后可以在命令前加 `LOCALTRIO_SAMPLER_GPUS=0,1,2,3`，用 4 张卡采样；训练进行中不要同时评测。

先评测 Base Model：

```bash
uv run python experiments/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --output experiments/retool/eval-results/aime25-retool-base.jsonl
```

再评测训练得到的 sampler weights（`--model-path` 填训练终端打印的 `Saved sampler weights` 目录）：

```bash
uv run python experiments/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/retool-qwen35-4b-step200-final-weights \
    --output experiments/retool/eval-results/aime25-retool-step200.jsonl
```

- `--mode text-only` 禁用工具做单轮推理，作为对照组（`--max-tokens` 控制单答案长度，默认 8192）。
- 先加 `--limit 2` 可以快速检查流程。
- 输出 JSONL 每行一道题，最后一行是 `type=summary`（Average@N、Pass@N、Format、代码调用统计）；retool 模式按 `--chunk-size` 道题一批逐批落盘。

画 checkpoint 对比图前，[`analysis.py`](./analysis.py) 要求 `eval-results/` 里 `CHECKPOINTS` 列出的 6 个文件齐全（`aime25-retool-{base,step20,step50,step100,step150,step200}.jsonl`），缺任何一个都会报 `FileNotFoundError`，而且各文件的 `--val-n` 必须一致。除了上面的 Base 和 step200，还要评测 4 个 checkpoint（`<run_id>` 换成对应那次训练的真实目录，两次训练的 run_id 不同）：

```bash
# step20：来自 20-step 验证跑（--run-name retool-qwen35-4b-step20）的最终 checkpoint
uv run python experiments/retool/eval.py \
    --mode retool \
    --val-n 12 \
    --temperature 1.0 \
    --top-p 0.7 \
    --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/retool-qwen35-4b-step20-final-weights \
    --output experiments/retool/eval-results/aime25-retool-step20.jsonl

# step50 / step100 / step150：来自 200-step 完整训练（--save-every 50）的中间 checkpoint
for step in 50 100 150; do
  uv run python experiments/retool/eval.py \
      --mode retool \
      --val-n 12 \
      --temperature 1.0 \
      --top-p 0.7 \
      --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/retool-qwen35-4b-step200-step-${step}-weights \
      --output experiments/retool/eval-results/aime25-retool-step${step}.jsonl
done
```

6 个文件齐全后画图：

```bash
uv run python experiments/retool/analysis.py --output experiments/retool/eval-results/local_checkpoints.png
```

不加 `--output` 会覆盖 readme 引用的原文结果图 `images/checkpoint_avg_pass_format.png`。

## 6. 资源估算

> 以下为按模型规模和原文记录的 token 用量（200 step 约 15.97M train / 15.84M sample token）推算的**粗略估算，未经实测**。

| 项目 | 估算（未实测） |
| --- | --- |
| GPU0 训练显存 | 约 15–25 GB（4B bf16 权重约 9 GB + LoRA/AdamW + 激活，单条 ≤ 8,192 token） |
| GPU1–3 采样显存 | 每卡约 40 GB（vLLM 按 `gpu_memory_utilization=0.85` 预占） |
| 单个 step | rollout（最多 6 轮串行生成 + 代码执行）约 2–4 分钟 + 训练约 1–1.5 分钟 |
| 20 step / 200 step | 约 1–1.8 小时 / 约 10–18 小时 |
| AIME25 评测一次 | 360 条轨迹，约 10–25 分钟 |

## 7. 常见问题

改 localtrio 参数时，复制一份预设再用 `LOCALTRIO_CONFIG` 指定：

```bash
cp configs/4gpu.yaml my-retool.yaml   # 按下面的说明修改
LOCALTRIO_CONFIG=my-retool.yaml uv run python experiments/retool/train.py ...
```

**训练卡 OOM**：调小 `trainer.max_tokens_per_microbatch`（16384 → 8192 → 4096），必要时再调小 `trainer.logits_chunk_size`（1024 → 512）。不要改 `train.py` 里的 `MAX_MICRO_BATCH_*`：那两个常量决定每次 `forward_backward` 装多少条 Datum，会影响梯度加权；trainer 的 micro-batch 拆分不影响梯度。

**vLLM 启动失败或采样卡 OOM**：调小 `models.default.gpu_memory_utilization`（0.85 → 0.75），或设置 `models.default.max_model_len: 10240`（需大于本章 8,192 的轨迹上限）。

**vLLM 加载 LoRA 时报 target modules 不支持**：把 `trainer.lora_target_modules` 改成 `standard`（只训练 q/k/v/o + MLP），重新训练。

**loss 归一化和原文不完全一样？** `train.py` 按原版 PyTRIO“每次请求内对样本取 mean”的约定缩放 advantage；localtrio 默认按每次请求的 active token 数归一化。想严格按逐样本均值，在 YAML 里加 `trainer.loss_normalization: samples`。默认设置与 PyTRIO 0.2.9 文档的口径一致，详见 [localtrio 指南](../../docs/content/guide/localtrio.md)。

**`--model-path` 报“不是 LoRA 权重目录（缺少 adapter_config.json）”**：说明该路径下没有 `adapter_config.json`，通常是 run_id 或 checkpoint 名写错、目录不存在，或者只写到了 `sampler_weights/` 这一层，没写到具体的 checkpoint 目录。请直接复制训练终端 `Saved sampler weights:` 那一行的完整路径。评测建议用 `sampler_weights/`（bf16，体积更小），但 `states/` 目录本身也能通过这个检查。原文的 `trio://` 路径会报另一个错误：“是 PyTRIO 云端路径”。

**沙箱超时或 CPU 被打满**：调小 `--sandbox-workers`；沙箱子进程已把 BLAS 线程数压到 1。不要无人值守运行：文件读写只靠 system prompt 约束。

**想看引擎日志**：日志在 `~/.cache/localtrio/logs/`；设置 `LOCALTRIO_WORKER_LOGS=inherit` 可直接打印到终端。
