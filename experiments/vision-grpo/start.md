# GeoQA Vision GRPO 快速启动（本地 GPU 版）

> 完整原理、原文训练结果与实现边界请查看 [Vision GRPO 正文](./readme.md)。本章是 Datawhale [开源原教程](https://github.com/datawhalechina/agentic-rl-lab)的本地 GPU 版本，代码通过 [localtrio](../../docs/content/guide/localtrio.md) 在自己的 GPU 上训练和采样，不需要 `trio login`、API key，也没有计费。

以下命令均从仓库根目录运行，Python ≥ 3.12。

## 0. GPU 布局

本章只用一个基模（`Qwen/Qwen3.5-4B`，原生支持图文输入），直接使用**默认布局**，不需要配置文件。localtrio 按检测到的卡数分配：

| GPU | 用途 |
| --- | --- |
| GPU 0 | LoRA 训练（transformers + PEFT；视觉塔冻结，只在语言模型上加 LoRA） |
| 其余每张卡 | 各一个 vLLM 采样副本（图文输入），每个 step 热加载最新 LoRA |

只有 1 张卡时，训练和采样共用 GPU 0；用 `uv run python -m localtrio info` 可以确认实际布局。

不需要 4 张卡。本章的最低与推荐配置（估算，详见 [显存与 GPU 布局规划](../../docs/content/guide/hardware.md)）：

| 你的卡 | 怎么跑 | 速度（相对 4 卡） |
| --- | --- | --- |
| 1 张 ≥48GB（最低） | 命令前加 `LOCALTRIO_CONFIG=configs/1gpu.yaml`（训练与采样共用 GPU 0） | 约 1/3～1/4 |
| 2 张 ≥24GB（最低） | 直接运行；24GB 卡加 `LOCALTRIO_CONFIG=configs/24gb.yaml`（训练 micro-batch 8192） | 约 1/2～1/3 |
| 4 张（推荐） | 直接运行，与 [`configs/4gpu.yaml`](../../configs/4gpu.yaml) 相同；3 张、8 张卡也直接运行。24GB 卡同样加 `LOCALTRIO_CONFIG=configs/24gb.yaml` | 基准 |

图片 token 计入序列长度，显存估算和 OOM 处理见第 10、11 节。

## 1. 安装依赖

```bash
uv sync --extra kernels   # kernels：Qwen3.5 线性注意力的训练加速，强烈建议安装
uv run swanlab login      # 可选；调试时用 --swanlab-mode disabled
```

可以提前下载模型（约 9～10 GB），否则第一次训练时会自动下载：

```bash
uv run python -m localtrio download Qwen/Qwen3.5-4B
# 国内网络：
uv run python -m localtrio download Qwen/Qwen3.5-4B --source modelscope
```

用 ModelScope 下载后，训练和评测时在命令前加 `LOCALTRIO_MODEL_SOURCE=modelscope`（或在配置文件里写 `model_source: modelscope`）。`train.py` / `eval.py` 里的 image processor 也会通过 localtrio 解析到同一个模型目录。

## 2. 下载并拆分 GeoQA

```bash
uv run python experiments/vision-grpo/download-dataset.py
```

默认生成：

```text
experiments/vision-grpo/datasets/train.parquet  # 3,503 条
experiments/vision-grpo/datasets/test.parquet   # 固定 100 条（原始 test 用 seed=42 打乱后取最后 100 条）
```

需要锁定数据集 revision 时：

```bash
uv run python experiments/vision-grpo/download-dataset.py \
  --revision DATASET_COMMIT
```

数据从 Hugging Face 下载，国内网络可以设置 `HF_ENDPOINT=https://hf-mirror.com`。

## 3. 检查命令与 GPU 布局

```bash
uv run python -m localtrio info
uv run python experiments/vision-grpo/train.py --help
uv run python experiments/vision-grpo/eval.py --help
```

## 4. 运行 1-step 小规模验证

```bash
uv run python experiments/vision-grpo/train.py \
  --steps 1 \
  --batch-size 1 \
  --group-size 4 \
  --max-tokens 256 \
  --save-every 0 \
  --no-save-weights \
  --show-samples \
  --swanlab-mode disabled
```

第一次运行会下载模型并启动 1 个 trainer 和每张采样卡一个 vLLM 子进程，需要几分钟。这个配置会检查真实图片采样、视觉 token 对齐、boxed reward 和 group advantage。单题四条回答可能全部同分，出现这种情况时该组会被跳过，日志中的 `datums` 为 0；想验证完整的反向链路，可以把 `--batch-size` 调到 4。

## 5. 启动 20-step 小规模训练

```bash
uv run python experiments/vision-grpo/train.py \
  --steps 20 \
  --batch-size 8 \
  --group-size 8 \
  --max-tokens 1024 \
  --learning-rate 4e-5 \
  --save-every 10 \
  --swanlab-mode online
```

## 6. 启动 100-step 正式训练

```bash
uv run python experiments/vision-grpo/train.py \
  --base-model Qwen/Qwen3.5-4B \
  --lora-rank 32 \
  --steps 100 \
  --batch-size 8 \
  --group-size 8 \
  --max-tokens 1024 \
  --temperature 1.0 \
  --top-p 1.0 \
  --learning-rate 4e-5 \
  --save-every 25 \
  --experiment-name vision-grpo-qwen35-4b-geoqa \
  --weights-name vision-grpo-qwen35-4b-geoqa \
  --swanlab-mode online
```

每 25 steps 会保存并打印两个**本地目录**：

```text
Sampler 权重：/home/<you>/.cache/localtrio/runs/<run_id>/sampler_weights/vision-grpo-qwen35-4b-geoqa-step-25-sampler
State 权重：/home/<you>/.cache/localtrio/runs/<run_id>/states/vision-grpo-qwen35-4b-geoqa-step-25-state
```

评测时使用 `sampler_weights` 目录；`states` 目录还带有 AdamW 状态，可以在自己的代码里用 `trio.ServiceClient().create_training_client_from_state_with_optimizer(<states 目录>)` 恢复训练（`train.py` 本身没有续训参数）。权重根目录默认是 `~/.cache/localtrio`，可用 `LOCALTRIO_HOME` 改到大容量磁盘。

`train.py` 常用参数（默认值以代码为准）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-model` | `Qwen/Qwen3.5-4B` | 模型名或本地目录 |
| `--lora-rank` | `32` | LoRA rank |
| `--steps` | `10` | 训练步数 |
| `--batch-size` / `--group-size` | `4` / `4` | 每步题目数 / 每题采样数（正式实验用 8 / 8） |
| `--max-tokens` | `64` | 每条回答最多生成 token 数（正式实验用 1024） |
| `--learning-rate` | `4e-5` | Adam 学习率 |
| `--max-samples` | `0` | 只用按 `--seed` 打乱后的前 N 条训练数据（即随机子集），0 表示全部 |
| `--save-every` | `10` | 每隔多少步保存，0 表示只保存最终 checkpoint |
| `--no-save-weights` | — | 不保存任何权重（调试用） |
| `--show-samples` | — | 打印每条回答，检查格式与 reward |
| `--swanlab-mode` | `online` | `online` / `local` / `offline` / `disabled` |

## 7. 评测 Base

评测只启动 vLLM 采样副本（默认 GPU 0 以外每张卡一个；只有 1 张卡时用 GPU 0），请在训练结束、GPU 空出来之后运行：

```bash
uv run python experiments/vision-grpo/eval.py \
  --base-model Qwen/Qwen3.5-4B \
  --max-tokens 1024 \
  --limit 100 \
  --output experiments/vision-grpo/eval-results-base.json
```

## 8. 评测 checkpoint

把 `--model-path` 替换为训练日志打印的 sampler 权重目录：

```bash
uv run python experiments/vision-grpo/eval.py \
  --base-model Qwen/Qwen3.5-4B \
  --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/vision-grpo-qwen35-4b-geoqa-step-100-sampler \
  --max-tokens 1024 \
  --limit 100 \
  --output experiments/vision-grpo/eval-results-step-100.json
```

每次评测一个模型，并为 Base 与 checkpoint 使用不同输出文件。终端会打印 Accuracy、Format rate 和结果文件路径；JSON 里有逐样本的预测与回答文本。多卡时评测默认空着 GPU 0；想多一个副本，可以在命令前把所有卡都列进 `LOCALTRIO_SAMPLER_GPUS`（可选），例如 4 张卡时 `LOCALTRIO_SAMPLER_GPUS=0,1,2,3`，2 张卡时 `LOCALTRIO_SAMPLER_GPUS=0,1`。

## 9. 生成结果图

```bash
uv run python experiments/vision-grpo/analysis.py \
  --output experiments/vision-grpo/images/eval-comparison-local.png
```

`analysis.py` 绘制的是脚本里写死的 `BASE_RESULTS` 与 `GRPO_RESULTS`（当前为原文在 PyTRIO 上的 Base `71.0% / 75.0%` 与 step 100 `87.0% / 91.0%`），不会读取评测 JSON。本地实验完成后，先把这两个字典改成自己的结果；用 `--output` 写到新文件，避免覆盖原文的 `images/eval-comparison.png`。

## 10. 资源估算

> 以下为按模型大小、默认参数和原文记录的 token 量推算的**估算值，未实测**。

| 项目 | 估算 |
| --- | --- |
| GPU 0 显存（训练） | 约 15～25 GB（4B bf16 权重约 9～10 GB + LoRA / AdamW 约 1～2 GB + 激活）；24GB 卡用 micro-batch 8192（`configs/24gb.yaml` 已调好） |
| 采样卡显存 | vLLM 按 `gpu_memory_utilization=0.85` 预占每卡 85%：48GB 卡约 40 GB，24GB 卡约 20 GB |
| 首次启动 | 下载模型 + 加载各引擎，约 2～5 分钟 |
| 每步耗时 | 按 4 张 48GB 卡估算，`batch 8 × group 8 × 1024 tokens`：约 0.5～2 分钟 |
| 20 steps | 按 4 张卡估算，约 15～40 分钟 |
| 100 steps | 按 4 张卡估算，约 1～3 小时 |
| 卡更少时 | 2 张卡约慢 2～3 倍，1 张卡（`1gpu.yaml`）约慢 3～4 倍 |
| 评测 100 题 | 引擎启动之外约几分钟 |

## 11. 常见问题

**训练端（GPU0）OOM。** 图文样本和文本样本一样按 token 预算打包（视觉 token 计入长度），先在配置里把 `trainer.max_tokens_per_microbatch` 从 16384 调到 8192，再考虑调小 `--max-tokens`，或把 `trainer.logits_chunk_size` 从 1024 调到 512。保持 `trainer.gradient_checkpointing: true`。

**vLLM 启动时报显存不足或 KV cache 不足。** 这两种报错的调法相反，先看报错原文：

- 报 `Free memory on device ... is less than desired GPU memory utilization` 或 CUDA OOM（多半是同一张卡上还有别的进程）：调小 `models.default.gpu_memory_utilization`（如 0.75）。
- 报 `... KV cache is needed, which is larger than the available KV cache memory`（24GB 卡上常见）：模型自带的 `max_model_len` 放不下，设置 `models.default.max_model_len: 8192`，或在显存允许时调大 `gpu_memory_utilization`。这时再调小 utilization 只会让可用的 KV cache 更少。

**vLLM 报 LoRA 模块不支持**（例如提到 `in_proj_qkv`、`out_proj` 等线性注意力投影）。把 `trainer.lora_target_modules` 改成 `standard`，只在 q/k/v/o + gate/up/down 上加 LoRA。

修改以上配置的方式：写一个只包含要改的项的 YAML（没写的项沿用默认值，可选项见 [`configs/4gpu.yaml`](../../configs/4gpu.yaml) 里的注释），再用 `LOCALTRIO_CONFIG` 指定：

```yaml
# my-vision.yaml（没写 gpus：仍按卡数自动分配，GPU 0 训练、其余采样）
trainer:
  max_tokens_per_microbatch: 8192
  lora_target_modules: standard
models:
  default:
    gpu_memory_utilization: 0.75
    max_model_len: 8192
```

```bash
LOCALTRIO_CONFIG=my-vision.yaml uv run python experiments/vision-grpo/train.py --steps 20 --batch-size 8 --group-size 8 --max-tokens 1024
```

**报错“图文 prompt 长度不一致：local=…, sampler=…”或“第 0 张图片 expected_tokens=…，image processor 计算为 …”。** 两条报错说的是同一件事：`encode_image()` 估计的视觉 token 数和引擎实际展开的数量不一致。默认的 vLLM 采样后端下通常先看到第一条，它来自 `train.py` 采样后的对齐检查，比较的是 vLLM 展开视觉占位符后的真实 prompt 长度和按 `encode_image()` 算出的本地长度；第二条来自 localtrio 的 `expand_chunks`，在训练端 `forward_backward` 处理 Datum 时触发，用 `hf` 采样后端（如 `configs/cpu-debug.yaml`）时在采样阶段就会触发。排查方法相同：确认没有改动 image processor 的分辨率参数，并且 `train.py` / `eval.py` 与引擎加载的是同一个模型目录（脚本里的 `resolve_model_dir()` 负责这件事）。

**换成多图数据集时 vLLM 报图片数量超限。** localtrio 对多模态模型默认 `limit_mm_per_prompt: {image: 8, video: 0}`。一条 prompt 超过 8 张图时，在配置里把 `models.default.limit_mm_per_prompt` 调大，例如 `{image: 16, video: 0}`。这一项会整体替换默认值，记得保留 `video: 0`，否则 vLLM 会把没写的视频上限当成 999，并按这个上限预留编码器显存。GeoQA 每题只有一张图，默认即可。

**`degenerate` 很高、`datums` 很少。** 同一题的回答全对或全错。调大 `--group-size`；回答被截断时也拿不到 `\boxed{}`，可以检查 `rollout/completion_tokens_mean` 是否贴近 `--max-tokens`。

**评测时报 GPU 被占用或显存不足。** 训练还没结束时，训练进程的采样副本还占着那些卡；等训练结束，或者用 `LOCALTRIO_SAMPLER_GPUS` 指定空闲的卡。

**引擎启动失败或卡住。** 子进程日志在 `~/.cache/localtrio/logs/`；也可以加 `LOCALTRIO_WORKER_LOGS=inherit` 让日志直接打印到终端。
