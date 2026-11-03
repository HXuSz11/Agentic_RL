# Tournament-GRPO 快速启动

本页是本地 4×RTX A6000（48GB）版的运行手册：策略模型的训练与采样通过 [localtrio](../../docs/content/guide/localtrio.md) 跑在自己的 GPU 上；搜索（DeepSeek Search API、Jina Reader）、训练裁判（Jev）和 RACE 评分模型（DeepSeek）仍是外部 API。算法讲解见 [readme.md](./readme.md)。

以下命令都在**仓库根目录**运行。

## 0. 安装环境

```bash
uv sync --extra kernels          # torch / transformers / peft / vllm + Qwen3.5 线性注意力训练 kernel
uv run python -m localtrio info  # 查看 GPU、vLLM 版本与当前布局
uv run swanlab login             # 可选；不记录到云端时给命令加 --swanlab-mode disabled
```

- 不需要 `trio login`、`PYTRIO_API_KEY`，也没有训练计费；外部 API 的 Key 见第 3 步。
- 第一次运行时会自动下载 `Qwen/Qwen3.5-4B`（默认从 HuggingFace）。`train.py` 和 `prepare_data.py` 还会用 `transformers` 直接从 Hugging Face 读取 tokenizer。国内网络可设置 `export HF_ENDPOINT=https://hf-mirror.com`；localtrio 下载模型也可以改用 `export LOCALTRIO_MODEL_SOURCE=modelscope`。
- 强烈建议装 `kernels` 扩展：本章单条轨迹最长约 2.2 万 token，不装时 Qwen3.5 线性注意力退回纯 PyTorch 实现，训练明显更慢、更占显存。

## 1. GPU 布局

默认使用内置布局，不需要任何配置文件（等价于 [`configs/4xa6000.yaml`](../../configs/4xa6000.yaml)）：

| GPU | 角色 |
| --- | --- |
| GPU0 | LoRA 训练（transformers + PEFT，bf16 + 梯度检查点；custom loss 先前向、再反向） |
| GPU1 / GPU2 / GPU3 | 各一个 vLLM 采样副本（data parallel，LoRA 热加载） |

裁判、搜索、网页读取都走外部 API，不占 GPU。想用本机模型当裁判时布局需要调整，见第 6 节。显存与布局的整体说明见 [4×A6000 硬件规划](../../docs/content/guide/hardware.md)。

## 2. 准备数据

```bash
uv run python experiments/tournament-grpo/prepare_data.py
```

脚本下载作者固定 commit 的训练数据和 Deep Research Bench，并用 `Qwen/Qwen3.5-4B` tokenizer 去掉超过 4096 token 的题目：

```text
experiments/tournament-grpo/datasets/
├── raw/                      # 作者原始 Parquet
├── train.jsonl               # 8,378 条
├── val.jsonl                 # 440 条
└── deep_research_bench/      # query / criteria / reference 各 100 条
```

## 3. 配置外部 API

```bash
cp -n experiments/tournament-grpo/.env.example experiments/tournament-grpo/.env
```

在 `experiments/tournament-grpo/.env` 中填写搜索、训练裁判和评测裁判：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
JINA_API_KEY=
JINA_READER_BASE_URL=https://r.jina.ai

TOURNAMENT_JUDGE_BACKEND=jev
TOURNAMENT_JUDGE_API_URL=https://api.typesafe.ai/v1/systemone
TOURNAMENT_JUDGE_MODEL_NAME=jev-1.14.0
TOURNAMENT_JUDGE_API_KEY=your_jev_api_key

RACE_JUDGE_API_URL=https://api.deepseek.com/v1/responses
RACE_JUDGE_MODEL_NAME=deepseek-flash
RACE_JUDGE_API_KEY=your_deepseek_api_key
```

- `JINA_API_KEY` 可以留空，此时限速 20 RPM。国内可把 `JINA_READER_BASE_URL` 改成 `https://r.jinaai.cn`。
- Jev 官方接口在中国大陆无法直接访问。工具和裁判请求不读取 shell 的 `HTTP(S)_PROXY`，国内读者需要开启系统级全局代理，或把 `TOURNAMENT_JUDGE_API_URL` 换成国内可访问的兼容服务。
- 搜索和 Jina 也可以在 `.env` 中分别设置 `DEEPSEEK_SEARCH_PROXY`、`JINA_READER_PROXY`。

检查 Jev 是否可用：

```bash
uv run python experiments/tournament-grpo/judge.py
```

## 4. 训练

### 单步验收

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 1 \
  --questions-per-batch 1 \
  --group-size 8 \
  --run-name tournament-grpo-smoke
```

这一步采样 1 道题、8 条轨迹，进行 28 场裁判比较，并更新一次权重。第一次运行还要加载模型、启动 4 个引擎进程，会多花几分钟。运行后检查 `experiments/tournament-grpo/outputs/tournament-grpo-smoke/train.jsonl`：`judge/comparisons` 为 28，`reward/has_answer` 大于 0。

### 20-step 训练

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 20 \
  --questions-per-batch 8 \
  --group-size 8 \
  --learning-rate 4e-5 \
  --save-every 10 \
  --run-name tournament-grpo-20step
```

每步 64 条轨迹、224 场裁判比较。裁判请求重试 3 次仍失败时，训练会停止。修好接口后，从最近的 checkpoint 续训（恢复 LoRA 权重、AdamW 状态和数据位置，`--max-steps` 是累计步数）：

```bash
uv run python experiments/tournament-grpo/train.py \
  --max-steps 20 \
  --resume experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-10.json \
  --run-name tournament-grpo-20step-resumed
```

续训后 `checkpoint-step-20.json` 位于 `outputs/tournament-grpo-20step-resumed/`，第 5 节评测的 `--checkpoint` 要改成 `experiments/tournament-grpo/outputs/tournament-grpo-20step-resumed/checkpoint-step-20.json`。

**输出在哪里：** 训练日志位于 `experiments/tournament-grpo/outputs/<run-name>/`：

```text
train.jsonl                 # 每步指标
judge.jsonl                 # 每场比赛的候选、胜者和裁判输出
rollout-step-*.jsonl        # 第 1 步和 checkpoint 步的完整轨迹
checkpoint-step-*.json      # 记录 state_path 与 sampler_path
```

`checkpoint-step-N.json` 里的两个路径是 localtrio 保存的本地目录，不再是 `trio://`：

```json
{
  "step": 20,
  "state_path": "/home/<you>/.cache/localtrio/runs/<run_id>/states/step-20-state",
  "sampler_path": "/home/<you>/.cache/localtrio/runs/<run_id>/sampler_weights/step-20-weights",
  "config": { "...": "..." }
}
```

目录根可用 `LOCALTRIO_HOME` 改到大容量磁盘；移动权重目录后要同步修改 JSON 里的路径。引擎日志在 `<LOCALTRIO_HOME>/logs/`（默认 `~/.cache/localtrio/logs/`）。

## 5. Deep Research Bench 评测

`generate` 阶段只启动 vLLM，不启动 trainer；训练结束后可以在命令前加 `LOCALTRIO_SAMPLER_GPUS=0,1,2,3`，用 4 张卡采样。训练进行中不要同时评测。

先为 Base 和 Step 20 生成研究报告：

```bash
uv run python experiments/tournament-grpo/eval.py generate \
  --batch-size 16
uv run python experiments/tournament-grpo/eval.py generate \
  --batch-size 16 \
  --checkpoint experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-20.json
```

再用 RACE 打分（只调用外部评分模型，不占 GPU）：

```bash
uv run python experiments/tournament-grpo/eval.py score \
  --judge-concurrency 8
uv run python experiments/tournament-grpo/eval.py score \
  --judge-concurrency 8 \
  --checkpoint experiments/tournament-grpo/outputs/tournament-grpo-20step/checkpoint-step-20.json
```

评测使用 100 道题，每题贪心生成 1 条轨迹。生成和评分都逐题保存，中断后重跑同一命令即可继续。可以先加 `--limit 4`，只跑前 4 题检查流程。

结果按 policy 模型、权重和裁判模型分开保存。权重目录名取 `sampler_path` 的最后三段（`<run_id>/sampler_weights/<name>`）：

```text
experiments/tournament-grpo/outputs/drb/Qwen--Qwen3.5-4B/
├── base/
│   ├── articles.jsonl
│   ├── generation-summary.json
│   └── race/deepseek-flash/race-summary.json
└── run_<日期>_<时间>_<id>--sampler_weights--step-20-weights/
    └── ...
```

训练曲线在 SwanLab 的 `tournament-grpo` 项目中查看。优先看 `reward/*` 和 `rollout/*`，指标含义见 readme 第 7 节。

## 6. 可选：用本机模型当裁判

默认行为不变：训练裁判是 Jev，RACE 评分是 DeepSeek。如果想用自己的 GPU 跑一个 OpenAI 兼容的裁判，可以用 `python -m localtrio serve` 起 vLLM 服务，再把 `.env` 切到作者原版的 `chat` 协议。

**训练期间**，4 张卡已被训练和采样占满，需要先把采样缩到 GPU1、GPU2，把 GPU3 留给裁判：

```bash
# 终端 1：GPU3 上启动裁判服务
uv run python -m localtrio serve --base-model <裁判模型> --gpus 3 --port 8001 --max-model-len 32768 --no-lora

# 终端 2：训练时采样只用 GPU1、GPU2
LOCALTRIO_SAMPLER_GPUS=1,2 uv run python experiments/tournament-grpo/train.py \
  --max-steps 20 --questions-per-batch 8 --group-size 8 --save-every 10 \
  --run-name tournament-grpo-20step-localjudge
```

也可以写进 YAML：复制 `configs/4xa6000.yaml`，把 `models.default.gpus` 改成 `[1, 2]`，再用 `LOCALTRIO_CONFIG` 指定。`.env` 改为：

```dotenv
TOURNAMENT_JUDGE_BACKEND=chat
TOURNAMENT_JUDGE_API_URL=http://127.0.0.1:8001/v1/chat/completions
TOURNAMENT_JUDGE_MODEL_NAME=<裁判模型，与 --base-model 相同>
TOURNAMENT_JUDGE_API_KEY=
```

裁判模型的要求：非思考（Instruct）模型，输出里不能夹带推理过程，否则胜者解析会出错；上下文至少能放下 rubric 加两份候选报告（建议 `--max-model-len 32768`）；bf16 权重约 14B 以内，才能在单张 48GB 卡上留出 32K 上下文的 KV cache（估算，未实测）。20B 级模型需要量化，或改成双卡张量并行（例如 `--gpus 2,3 --tp 2`，此时训练采样只留 GPU1：`LOCALTRIO_SAMPLER_GPUS=1`）。作者原版用的是 Qwen2.5-72B-Instruct，单卡放不下。命令里的 `--no-lora` 表示只服务基模，不预留 LoRA 显存。

**评测的 RACE 打分**发生在训练结束之后，4 张卡都空闲，可以起更大的评分模型，例如 4 卡张量并行（显存是否够用取决于模型，未实测）。评分模型要选原生上下文 ≥64K 的非思考 Instruct 模型，例如 Qwen3-30B-A3B-Instruct-2507（4 卡张量并行放得下，估算）：

```bash
uv run python -m localtrio serve --base-model <评分模型> --gpus 0,1,2,3 --tp 4 --port 8001 --max-model-len 65536 --no-lora
```

上下文要开到 65536：RACE 每次请求固定 `max_tokens=8192`，最长的参考报告 prompt 约 2.5 万 token（按 Qwen3 tokenizer 估算），用 32768 时少数题会被 vLLM 拒绝（较早的版本如 0.30 及以前直接返回 400）或输出被截断（较新的版本如 0.33 会把输出预算截短），`score` 会在这几题反复中止。不建议写 `--max-model-len 0`：它会用模型自带的上限（Qwen3-2507 系列为 256K），有些模型的 KV cache 放不下，启动会失败。原生只有 32K 的模型（如 Qwen2.5-72B-Instruct）不适合直接当这里的评分模型。

```dotenv
RACE_JUDGE_API_URL=http://127.0.0.1:8001/v1/chat/completions
RACE_JUDGE_MODEL_NAME=<评分模型，与 --base-model 相同>
RACE_JUDGE_API_KEY=
```

RACE 脚本遇到 `reasoning_content` 会直接报错，同样要用非思考模型。换了裁判或评分模型，reward 和 RACE 分数都不能与原文直接比较。

## 7. 资源估算

> 以下为按模型规模和原文记录（20 步训练 4 小时 3 分钟，其中裁判约 2 小时 22 分钟）推算的**粗略估算，未经实测**。裁判、搜索和 Jina 限速都是外部因素。

| 项目 | 估算（未实测） |
| --- | --- |
| GPU0 训练显存 | 单条轨迹最长约 2.2 万 token（4,096 prompt + 18,000 response），峰值约 15–25 GB（与[硬件规划](../../docs/content/guide/hardware.md)口径一致） |
| GPU1–3 采样显存 | 每卡 vLLM 预占约 40 GB |
| 单个 step | rollout 约 3–6 分钟 + 裁判约 7 分钟 + 训练约 4–7 分钟 |
| 20 step | 约 5–7 小时 |
| 生成 100 份报告 | 约 5–20 分钟 |
| 外部 API 费用 | 参考原文：训练期 DeepSeek Search 约 17 元 + Jev 约 1 元，RACE 评测约 10 元 |

## 8. 常见问题

改 localtrio 参数时，复制一份预设再用 `LOCALTRIO_CONFIG` 指定：

```bash
cp configs/4xa6000.yaml my-tournament.yaml   # 按下面的说明修改
LOCALTRIO_CONFIG=my-tournament.yaml uv run python experiments/tournament-grpo/train.py ...
```

**训练卡 OOM**：本章轨迹很长，先确认装了 `kernels` 扩展；再调小 `trainer.max_tokens_per_microbatch`（16384 → 8192）和 `trainer.logits_chunk_size`（1024 → 512）。单条超过上限的轨迹会单独成一个 micro-batch，所以真正决定峰值的是最长那条轨迹。

**vLLM 启动失败或采样卡 OOM**：调小 `models.default.gpu_memory_utilization`（0.85 → 0.75），或设置 `models.default.max_model_len: 32768`（必须大于 rollout 的 `max_model_tokens=25000`）。

**vLLM 加载 LoRA 时报 target modules 不支持**：把 `trainer.lora_target_modules` 改成 `standard`（只训练 q/k/v/o + MLP），重新训练。

**长上下文采样慢（可选提速）**：localtrio 默认 `models.default.max_num_batched_tokens: 2048`，是偏保守的预填充分块；本章只做生成，可以调到 8192 加快多轮长 prompt 的预填充，代价是 KV cache 略少。

**训练中途报 judge 失败并停止**：这是有意的设计（不用随机胜者更新）。修好网络或 Key 后用 `--resume` 从最近的 `checkpoint-step-N.json` 继续。

**Jina 读网页很慢**：无 Key 时限速 20 RPM，跨进程共享；填 `JINA_API_KEY` 并把 `JINA_RATE_LIMIT_RPM` 调到 500。

**`--checkpoint` 报“不是 LoRA 权重目录（缺少 adapter_config.json）”**：JSON 里的 `sampler_path` 是本机绝对路径，换机器或移动了 runs 目录后需要手动修改 `sampler_path`。原文的 `trio://` checkpoint 不能在本地版使用。
