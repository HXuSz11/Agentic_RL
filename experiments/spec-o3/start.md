# Spec-o3 快速开始

按顺序完成：安装环境（可选：预下载基模）→ 确认 GPU 布局 → 下载数据 → SFT 冷启动 → GRPO → 评测。

本页是本地 GPU 版：SFT、GRPO 与采样通过 [localtrio](../../docs/content/guide/localtrio.md) 跑在你自己的 NVIDIA 显卡上（图文输入走 `ImageChunk`），光谱绘图工具在本机 CPU 上运行。不要求 4 张卡，最低配置见第 2 节。原理与原文结果见 [readme.md](./readme.md)，原文公开的训练曲线见 [SwanLab](https://swanlab.cn/@kmno4/agentic-rl-lab-spec-o3/v1/rjj3zx/runs)。

## 1. 安装环境

在仓库根目录执行：

```bash
uv sync --extra kernels          # torch / transformers / peft / vllm + Qwen3.5 线性注意力训练 kernel
uv run python -m localtrio info  # 查看 GPU、vLLM 版本与当前布局
uv run python -m localtrio download Qwen/Qwen3.5-4B --source huggingface   # 可选：提前下载基模权重；不执行时首次训练会自动下载；训练 9B 时再加 Qwen/Qwen3.5-9B
cd experiments/spec-o3
uv run swanlab login             # 可选；训练命令改用 --swanlab-mode disabled 时可跳过
```

**后续命令都在 `experiments/spec-o3/` 中执行。**

- 不需要 `trio login`、`PYTRIO_API_KEY`，也没有训练计费。
- 基模权重会在首次训练或评测时由 localtrio 自动下载，上面的 `localtrio download` 只是把下载时间提前（9B 实验执行 `uv run python -m localtrio download Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B --source huggingface`）。`prepare_data.py` 和评测 4B 时的 `eval.py`（默认使用固定的 `--model-revision`）会用 `transformers` 从 Hugging Face 读取 tokenizer 与 image processor，HF 缓存里因此可能留下一个只含这些文件、没有权重的快照；localtrio 会跳过这种快照，首次使用时照常下载完整权重。`train_sft.py`、`train_rl.py` 默认 `--model-revision main`，直接从 localtrio 实际使用的模型目录读取 tokenizer 与 image processor（必要时先下载）。
- 国内网络请设置 `export HF_ENDPOINT=https://hf-mirror.com`，tokenizer 读取和上面的 `download` 都会走镜像。也可以改从 ModelScope 下载：`download` 时用 `--source modelscope` 即可，`auto` 模式会找到这份缓存，后续命令无需额外设置。HF 无法访问、又没有提前下载时，可在训练和评测命令前加 `LOCALTRIO_MODEL_SOURCE=modelscope`（或在 YAML 里写 `model_source: modelscope`），让 localtrio 不再先尝试 HF。
- `prepare_data.py` 会在脚本内把 `HF_ENDPOINT` 固定为 `https://huggingface.co`（原版设置），下载数据时需要能访问 Hugging Face 官方站点。

## 2. GPU 布局

不写配置文件时，localtrio 按检测到的卡数自动分工：GPU 0 训练 LoRA，其余卡各跑一个 vLLM 采样副本，只有 1 张卡时两者共用。按自己的卡选预设，后面的 SFT / RL 命令前照表加上（在本目录下运行；速度相对 4 卡估算）：

| 基模 | 你的卡 | 命令前加 | 分工 |
| --- | --- | --- | --- |
| Qwen3.5-4B | 1 张 ≥48GB（最低） | `LOCALTRIO_CONFIG=../../configs/1gpu.yaml`，或不加 | 训练与采样共用 GPU0（vLLM 占 40%），约慢 3～4 倍 |
| Qwen3.5-4B | 2 张 ≥24GB（最低） | 直接运行；24GB 卡加 `LOCALTRIO_CONFIG=../../configs/24gb.yaml` | GPU0 训练（24GB 卡 micro-batch 8192），其余卡采样，约慢 2～3 倍 |
| Qwen3.5-4B | 4 张（推荐） | 不加（与 [`configs/4gpu.yaml`](../../configs/4gpu.yaml) 相同） | GPU0 训练，GPU1–3 采样 |
| Qwen3.5-9B | 2 张 ≥48GB（最低） | `LOCALTRIO_CONFIG=../../configs/2gpu.yaml` | GPU0 训练（micro-batch 8192），GPU1 采样 |
| Qwen3.5-9B | 4 张 ≥48GB（推荐） | `LOCALTRIO_CONFIG=../../configs/4gpu-big-student.yaml`（[预设](../../configs/4gpu-big-student.yaml)） | GPU0–1 训练（按层切分），GPU2–3 采样 |

24GB 卡放不下 9B；24GB 的采样卡跑 4B 时还要设置 `models.default.max_model_len: 20480`（见第 9 节）。

用 `LOCALTRIO_CONFIG=<预设> uv run python -m localtrio info` 确认生效。评测只需要 vLLM，4B 和 9B 都可以用默认布局。显存与布局的整体说明见 [硬件规划](../../docs/content/guide/hardware.md)。

## 3. 下载并整理数据

```bash
uv run python prepare_data.py sft
uv run python prepare_data.py bench
```

先完成 `sft`，再执行 `bench`：后者需要读取 SFT 的对象列表来划分开发集。数据与图片直接保存在项目内：

| 路径 | 用途 |
| --- | --- |
| `datasets/sft/sft_train.jsonl` | SFT 训练，848 条 |
| `datasets/sft/sft_val.jsonl` | SFT 验证，94 条 |
| `datasets/sft/images/` | SFT 轨迹图片 |
| `datasets/rl/bench_rl_train.jsonl` | RL 训练，3,108 条 |
| `datasets/rl/bench_dev.jsonl` | 开发集评测，256 条 |
| `datasets/rl/bench_test.jsonl` | 完整测试集，6,754 条 |
| `datasets/rl/images/` | 初始光谱图 |
| `datasets/rl/spectra/` | 工具绘图所需的光谱数组 |

重新执行会复用已下载文件。准备结果中的图片与光谱路径为绝对路径，移动目录或换机器后需重新整理。文本清理记录保存在 `datasets/sft/sft_cleaning.jsonl`，数据统计保存在各目录的 `*_stats.json`。

## 4. SFT 冷启动：2 个 epoch

```bash
uv run python train_sft.py \
    --base-model Qwen/Qwen3.5-4B \
    --epochs 2 \
    --batch-size 4 \
    --rank 32 \
    --learning-rate 2e-5 \
    --seed 42 \
    --run-name spec-o3-sft \
    --swanlab-mode online
```

默认读取 `datasets/sft/`。每个 epoch 完成验证后，在本地保存两份权重，并打印路径：

| 终端打印 | 本地目录 | 后续用途 |
| --- | --- | --- |
| `state_path（继续训练）` | `~/.cache/localtrio/runs/<run_id>/states/spec-o3-sft-epoch-<N>-state` | 传给 `train_rl.py --state-path` |
| `sampler_path（推理评测）` | `~/.cache/localtrio/runs/<run_id>/sampler_weights/spec-o3-sft-epoch-<N>-sampler` | 传给 `eval.py --model-path` |

**保存好 epoch 2 的两个路径**（也可以在 `~/.cache/localtrio/runs/` 下按 run 名找到）。目录根可用 `LOCALTRIO_HOME` 改到大容量磁盘；引擎日志在 `~/.cache/localtrio/logs/`。

训练 9B 时使用独立的 `--run-name`，4 张卡用双卡训练布局（下例）；只有 2 张 48GB 卡时把配置换成 `../../configs/2gpu.yaml`：

```bash
LOCALTRIO_CONFIG=../../configs/4gpu-big-student.yaml uv run python train_sft.py \
    --base-model Qwen/Qwen3.5-9B \
    --epochs 2 \
    --batch-size 4 \
    --rank 32 \
    --learning-rate 2e-5 \
    --seed 42 \
    --run-name spec-o3-sft-9b \
    --swanlab-mode online
```

后续 9B 的 RL 从这份 9B SFT 的 `state_path` 开始。

## 5. 从 SFT state 开始 GRPO：1 个 epoch

把 `--state-path` 换成 **SFT epoch 2 打印的 `state_path`**：

```bash
uv run python train_rl.py \
    --state-path ~/.cache/localtrio/runs/<run_id>/states/spec-o3-sft-epoch-2-state \
    --epochs 1 \
    --batch-size 8 \
    --group-size 8 \
    --learning-rate 1e-6 \
    --max-turns 8 \
    --max-tokens 2048 \
    --max-seq-len 16384 \
    --temperature 1.0 \
    --top-p 1.0 \
    --save-every 100 \
    --seed 42 \
    --run-name spec-o3-rl \
    --swanlab-mode online
```

每批 8 道题，每题采样 8 条轨迹；一个 epoch 共 24,864 条轨迹、389 个 rollout batch。每 100 个累计 rollout step 和每个 epoch 结束时，都会保存并打印 state、sampler 两种本地路径（`spec-o3-rl-step-<N>-*`、`spec-o3-rl-epoch-<N>-*`）。

- `--max-tokens` 是单轮生成上限；`--max-seq-len` 包含文字、图片和工具历史。`--max-turns 8` 包含最终回答这一轮。
- 整批 Datum 一次交给 `forward_backward`，localtrio 的 trainer 按 `trainer.max_tokens_per_microbatch` 自动拆 micro-batch 并累积梯度，无需设置 mini batch。
- 从 SFT 接 RL 默认使用新的优化器。继续已有 RL 训练时，可换成 RL 的 `state_path` 并加 `--resume-optimizer`；它会恢复优化器状态，但脚本仍从新一轮数据遍历开始，不恢复之前的样本位置。
- 要训练 3 个 epoch，改为 `--epochs 3`。9B 传入 9B SFT 的 `state_path`，命令前加 `LOCALTRIO_CONFIG=../../configs/4gpu-big-student.yaml`（只有 2 张 48GB 卡时用 `../../configs/2gpu.yaml`）；4B 与 9B 使用各自的 `--run-name`。

## 6. 评测 Base、SFT 和 RL

评测只启动 vLLM，不启动 trainer；训练进行中不要同时评测。训练结束后，可以在下面每条命令前加 `LOCALTRIO_SAMPLER_GPUS`，让所有卡都采样（例如 4 张卡写 `0,1,2,3`，2 张卡写 `0,1`）。

下面的命令统一使用 6,144 token 的单轮上限和 16,384 token 的总上下文上限，默认评测 `datasets/rl/bench_dev.jsonl` 中的 256 条样本。

### Base

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --output outputs/base-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### SFT epoch 2

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/spec-o3-sft-epoch-2-sampler \
    --output outputs/sft-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### RL epoch 1

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-4B \
    --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/spec-o3-rl-epoch-1-sampler \
    --output outputs/rl-dev-6144 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

### 4B / 9B RL epoch 3

把 `--model-path` 换成对应 run 的 `...-epoch-3-sampler` 目录；9B 还要把 `--base-model` 改为 `Qwen/Qwen3.5-9B`（评测不需要 9B 预设）。下面假设 9B RL 的 `--run-name` 是 `spec-o3-rl-9b`：

```bash
uv run python eval.py \
    --base-model Qwen/Qwen3.5-9B \
    --model-path ~/.cache/localtrio/runs/<run_id>/sampler_weights/spec-o3-rl-9b-epoch-3-sampler \
    --output outputs/rl-9b-e3 \
    --max-tokens 6144 \
    --max-seq-len 16384 \
    --concurrency 16 \
    --temperature 0.6 \
    --seed 42
```

`--base-model` 必须与 sampler 权重的基模一致；`--model-revision` 可指定 tokenizer 与图像处理器的 HF 文件版本，默认 4B 固定版本、9B 使用 `main`。不传 `--model-path` 就评测 Base，并自动追加最终答案格式提醒；SFT/RL 使用原始提示词。`--output` 必须填写，各模型使用不同目录。

每个输出目录中包含：

- `metrics.json`：评测配置与分类指标。
- `trajectories.jsonl`：逐样本的完整交互记录。
- `images/`：评测过程中工具生成的局部光谱图。

readme 结果表里原文的历史单轮预算为 4B Base 6,132、4B SFT 2,048、三组 RL 6,144 token；这里的 Base/SFT 命令统一用 6,144，结果需单独记录。完整测试集可通过 `--data datasets/rl/bench_test.jsonl` 指定；正式使用前需处理统计中记录的 2 个 SFT/test 光谱对象重叠。

## 7. 生成结果图

```bash
uv run python analysis.py
```

输出 `images/results_comparison.png` 和同名矢量 PDF。脚本绘制的是原文在 PyTRIO 上的五组固定结果；取得本地评测结果后，需要同步修改 [`analysis.py`](./analysis.py) 中的数值与图注（建议另存一份，保留原文结果图）。

## 8. 资源估算

> 以下数字按模型规模和原文记录的 token 用量（SFT 约 6.11M train token；4B RL epoch 1 约 239M prefilling / 79M train / 26M sample token）推算，是**粗略估算，未经实测**。显存和时间以每卡 48GB、4 张卡（4B 默认布局，9B 用 `4gpu-big-student.yaml`）为例；2 张卡约慢 2～3 倍，1 张卡约慢 3～4 倍。最低与推荐配置见第 2 节。

| 项目 | 估算（未实测） |
| --- | --- |
| 4B 训练卡（GPU0）显存 | 约 20–30 GB（单条 16,384 token、含图片）；24GB 卡用 micro-batch 8192（`configs/24gb.yaml` 已设好） |
| 9B 训练显存 | 双卡（GPU0+1）时每卡约 15–25 GB；2 张卡时单卡训练，需要 48GB 卡 |
| 采样卡显存 | 每卡 vLLM 预占显存的 85%（48GB 卡约 40 GB） |
| 4B SFT 2 epoch（424 step） | 约 1–2 小时 |
| 4B RL 1 epoch（389 个 rollout batch） | 约 1.5–3 天 |
| 9B RL 1 epoch | 约为 4B 的 2 倍或更多 |
| 256 条开发集评测 | 约 0.5–2 小时（并发 16，可调大 `--concurrency`） |

## 9. 常见问题

改 localtrio 参数时，复制一份预设再用 `LOCALTRIO_CONFIG` 指定（路径相对于当前目录）：

```bash
cp ../../configs/4gpu.yaml my-spec-o3.yaml   # 按第 2 节选：1gpu / 2gpu / 4gpu / 4gpu-big-student（9B）
LOCALTRIO_CONFIG=my-spec-o3.yaml uv run python train_rl.py ...
```

**训练卡 OOM**：调小 `trainer.max_tokens_per_microbatch`（4B 默认 16384 → 8192，24GB 卡仍 OOM 时 → 4096；9B 预设 12288 → 8192），必要时再调小 `trainer.logits_chunk_size`（1024 → 512）。单条超长样本会单独成一个 micro-batch，图片 token 也计入长度。

**vLLM 启动失败或采样卡 OOM**：同卡还有别的进程、报可用显存不足时，调小 `models.default.gpu_memory_utilization`（0.85 → 0.75）；报 KV cache 放不下 `max_model_len`（24GB 卡上常见）时，设置 `models.default.max_model_len: 20480`（需大于 `--max-seq-len` 16384）。

**长上下文采样慢（可选提速）**：localtrio 默认 `models.default.max_num_batched_tokens: 2048`，是偏保守的预填充分块；本章只做生成，可以调到 8192 加快多轮图文长 prompt 的预填充，代价是 KV cache 略少。

**vLLM 报单个 prompt 的图片数超限**：localtrio 默认 `limit_mm_per_prompt: {image: 8, video: 0}`，正好覆盖 `--max-turns 8`（1 张初始图 + 7 张工具图），按本页命令不会超限。把 `--max-turns` 调到 N>8 时，在 `models.default` 下设置 `limit_mm_per_prompt: {image: N, video: 0}`；这一项会整体替换默认值，记得保留 `video: 0`，否则 vLLM 会重新为视频预留编码显存。

**训练端报 `expected_tokens` 与 image processor 不一致**：`expected_tokens` 由脚本用 Hugging Face 上的 image processor 计算，训练端用 localtrio 下载的模型目录里的 processor 校验。两边的模型版本需要一致：确认 `--model-revision` 与 localtrio 下载的版本对应，或在 localtrio 配置的 `model_paths` 里指向同一份本地快照。

**vLLM 加载 LoRA 时报 target modules 不支持**：把 `trainer.lora_target_modules` 改成 `standard`（只训练 q/k/v/o + MLP），从 SFT 重新训练。

**loss 归一化和原文注释不一致？** 脚本按原版“服务端求和”的口径先除以整批 assistant token 数；localtrio 默认（`trainer.loss_normalization: active_tokens`，与 PyTRIO 0.2.9 文档一致）还会再除以该批 active token 数：SFT 为该批 assistant token 数，RL 为该批全部非 `-100` 位置（含问题、图片和工具观察）。这个因子随 batch 变化（RL 中还随轨迹长度和被跳过的组数变化），会改变各步更新的相对大小，并不是一个常数缩放。想按原注释口径（逐 token 平均）训练，在 YAML 里加 `trainer.loss_normalization: sum`。

**本地训练的参数范围和原文不完全相同？** 是的。`train_sft.py` 创建训练客户端时没有传 `train_unembed`，PyTRIO 默认 `train_unembed=True`，原文在 PyTRIO 上还训练了输出层 `lm_head`；localtrio 默认忽略这一项，只在语言模型的全部线性层（`trainer.lora_target_modules: all-linear`）上加 LoRA，RL 从 SFT state 继续，也沿用这组模块。本地结果可能因此与原文不同。想尽量对齐，可在 YAML 里设 `trainer.allow_unembed: true`（以 LoRA 形式训练 `lm_head`），前提是你的 vLLM 能为这个多模态模型加载带 `lm_head` 的 adapter，详见 [localtrio 说明](../../docs/content/guide/localtrio.md)。

**`--state-path` / `--model-path` 报错**：`--state-path` 要传 `states/` 下的目录，`--model-path` 要传 `sampler_weights/` 下的目录；原文的 `trio://` 路径和 PyTRIO 权重控制台在本地版都不存在。

**查看完整参数**：

```bash
uv run python prepare_data.py --help
uv run python train_sft.py --help
uv run python train_rl.py --help
uv run python eval.py --help
```
