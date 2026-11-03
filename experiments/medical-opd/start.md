# Medical OPD：本地快速启动

本页是原教程 Medical OPD 快速启动的本地 GPU 版本：训练通过 [localtrio](../../docs/content/guide/localtrio.md) 在 4×RTX A6000 上运行，评测连接本地 `python -m localtrio serve` 启动的 OpenAI 兼容服务。不需要 `trio login`、API key，也没有计费。实验设计与原文结果见 [正文](./readme.md)，显存规划见 [4×A6000 硬件指南](../../docs/content/guide/hardware.md)。

以下命令均从**仓库根目录**运行。

## 0. 安装环境

```bash
uv sync --extra kernels            # Python ≥ 3.12；kernels 是 Qwen3.5 线性注意力的训练加速，强烈建议安装
uv run python -m localtrio info    # 查看 GPU 布局与 vLLM 版本
uv run swanlab login               # 可选；也可以给脚本加 --swanlab-mode offline / disabled
```

模型默认从 HuggingFace 下载，数据集（medical-o1、MedQA、C-Eval）也来自 HuggingFace。国内网络可以设置 `HF_ENDPOINT`（HF 镜像）；模型也可以改用 ModelScope：

```bash
export LOCALTRIO_MODEL_SOURCE=modelscope
uv run python -m localtrio download Qwen/Qwen3.5-4B   # 可选：提前下载基模
```

不需要复制 `.env`：评测脚本默认连接 `http://127.0.0.1:8000/v1`，没有 API key 时自动用 `EMPTY`。只有评测其他 OpenAI 兼容服务时，才需要参考 [`.env.example`](./.env.example) 设置 `OPENAI_API_KEY`。

## 1. GPU 布局

Student、Medical Teacher（SFT LoRA）和 Base Teacher 都是 `Qwen/Qwen3.5-4B`，所有训练脚本都用**内置默认布局**，不需要 `LOCALTRIO_CONFIG`（等价于 [`configs/4gpu.yaml`](../../configs/4gpu.yaml)）：

| GPU | 用途 |
| --- | --- |
| GPU0 | LoRA 训练（transformers + PEFT）；一个 trainer 进程可以同时托管多个 LoRA run |
| GPU1、GPU2、GPU3 | 各一个 vLLM 副本：Student 采样、Teacher `compute_logprobs`，按请求热加载不同 LoRA |

想在训练的同时评测之前的 checkpoint，可以让训练只用两张采样卡，把 GPU3 留给评测服务：在训练命令前加 `LOCALTRIO_SAMPLER_GPUS=1,2`，评测服务用 `--gpus 3`。

## 2. 下载数据

```bash
uv run python experiments/medical-opd/00-download-dataset.py
```

数据写入 `experiments/medical-opd/datasets/`：Medical SFT（`medical-o1-reasoning-SFT-zh/train.jsonl`）、MedQA-zh 评测（`medqa-zh-4options/test_sample.jsonl`，600 题）和 C-Eval non-med（`ceval-non-med/opd_train.jsonl` 与 `test_sample.jsonl`，300 题）。只想检查链路时加 `--medical-sft-sample-size 100 --medqa-sample-size 20 --ceval-train-size 100 --ceval-test-size 20`。

## 3. 训练 Medical SFT Teacher

```bash
uv run python experiments/medical-opd/02-medical-sft.py \
    --num-epochs 3 \
    --batch-size 16 \
    --max-length 2048 \
    --swanlab-mode online
```

先试跑可以加 `--sample-size 100 --num-epochs 1 --batch-size 2 --swanlab-mode disabled`。每个 epoch 结束会打印两个本地目录：

```text
Saved state [epoch003]:   ~/.cache/localtrio/runs/<SFT_RUN_ID>/states/sft-medical-qwen35-4b-medical-o1-reasoning-sft-zh-full-epochs3-epoch003
Saved weights [epoch003]: ~/.cache/localtrio/runs/<SFT_RUN_ID>/sampler_weights/sft-medical-qwen35-4b-medical-o1-reasoning-sft-zh-full-epochs3-epoch003
```

后面作为 Medical Teacher 的是 **epoch 3 的 `sampler_weights` 目录**。忘记记录时可以直接查找：

```bash
LT_HOME="${LOCALTRIO_HOME:-$HOME/.cache/localtrio}"   # 兼容 LOCALTRIO_HOME
ls -d "$LT_HOME"/runs/*/sampler_weights/sft-medical-*-epoch003
SFT_WEIGHTS=$(ls -d "$LT_HOME"/runs/*/sampler_weights/sft-medical-*-epoch003 | tail -n 1)
```

如果在 YAML 里改过 `home:`，上面的 `ls` 找不到目录，直接把训练日志里 `Saved weights [epoch003]:` 打印的路径填给 `SFT_WEIGHTS`。

## 4. 方案一 SAR-OPD

**Stage 2：Medical OPD**（fresh Student，Teacher = SFT epoch 3）：

```bash
uv run python experiments/medical-opd/03-medical-opd-async.py \
    --teacher-model-path "$SFT_WEIGHTS" \
    --steps 300 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 0 \
    --max-tokens 2048 \
    --learning-rate 4e-5 \
    --save-every-steps 300 \
    --swanlab-mode online
```

先试跑可以用 `--steps 10 --batch-size 2 --group-size 2 --sample-size 100 --save-every-steps 5 --swanlab-mode disabled`。同步版 `03-medical-opd-sync.py` 参数相同，更适合逐步调试（用同步版时保存的目录名是 `opd-medical-sync-…`，下面的 glob 两种都能匹配）。结束时打印 step 300 的 `state` 和 `weights` 两个目录：

```bash
LT_HOME="${LOCALTRIO_HOME:-$HOME/.cache/localtrio}"
OPD_STATE=$(ls -d "$LT_HOME"/runs/*/states/opd-medical-*-step000300 | tail -n 1)
```

同样，改过 `home:` 时直接用日志里 `Saved checkpoint [step000300] state:` 打印的路径。

**Stage 3：Base-anchor C-Eval OPD**（从 Medical OPD 的 state 继续，Teacher = 原始 Base）：

```bash
uv run python experiments/medical-opd/04-ceval-opd-async.py \
    --student-state-path "$OPD_STATE" \
    --steps 300 \
    --batch-size 4 \
    --group-size 4 \
    --sample-size 0 \
    --max-tokens 2048 \
    --learning-rate 5e-6 \
    --save-every-steps 50 \
    --swanlab-mode online
```

`--student-state-path` 必须是 `states/` 目录（`save_state` 的结果），传 `sampler_weights/` 目录会直接报错。脚本先保存一份 source-copy state，再在正式 run 里保存 step 0 和之后每 50 步的 state + sampler weights，名字形如 `opd-ceval-async-qwen35-4b-base-anchor-ppo-steps300-step000050`。

## 5. 方案二 IDT-OPD

```bash
uv run python experiments/medical-opd/05-interleaved-multi-teacher-opd.py \
    --medical-teacher-model-path "$SFT_WEIGHTS" \
    --steps 600 \
    --medical-steps-per-cycle 1 \
    --ceval-steps-per-cycle 1 \
    --batch-size 4 \
    --group-size 4 \
    --max-tokens 2048 \
    --learning-rate 1e-5 \
    --save-every-steps 50 \
    --swanlab-mode online
```

`2:1` 调度把 `--medical-steps-per-cycle` 改成 2，并把 `--save-every-steps` 改成 3 的倍数（如 60）。

## 6. 评测

先在一个终端启动本地 OpenAI 兼容服务（vLLM，GPU1，端口 8000）：

```bash
uv run python -m localtrio serve --base-model Qwen/Qwen3.5-4B --gpus 1 --port 8000 --max-model-len 16384
```

在另一个终端评测。`--model` 可以是基模名，也可以直接是某个 `sampler_weights` 目录（脚本会调用 `/v1/load_lora_adapter` 动态挂载，模型名记为 `<run_id>-<权重名>`）：

```bash
WEIGHTS=~/.cache/localtrio/runs/<RUN_ID>/sampler_weights/<WEIGHTS_NAME>

# Base 对照
uv run python experiments/medical-opd/01-eval-medical.py --model Qwen/Qwen3.5-4B --max-tokens 1024 --concurrency 16
uv run python experiments/medical-opd/01-eval-ceval.py   --model Qwen/Qwen3.5-4B --max-tokens 8192 --concurrency 16

# 某个 checkpoint
uv run python experiments/medical-opd/01-eval-medical.py --model "$WEIGHTS" --max-tokens 1024 --concurrency 16
uv run python experiments/medical-opd/01-eval-ceval.py   --model "$WEIGHTS" --max-tokens 8192 --concurrency 16
```

也可以在启动服务时注册固定名字，评测时传名字：`--lora sar300=<目录>` → `--model sar300`。想并行评测两个基准，可以在 GPU2 上再起一个 `--port 8001` 的服务，并给第二个评测加 `--base-url http://127.0.0.1:8001/v1`。

逐题结果与汇总指标写入 `experiments/medical-opd/eval-results/`（`medical_eval_<模型名>.jsonl` / `.metrics.json`、`ceval_eval_<模型名>.jsonl` / `.metrics.json`）。`analysis.py` 画的是原文的硬编码结果，复现后可以改成自己的数值，并用 `--output-dir` 输出到别处，避免覆盖文章引用的原图。

## 7. 资源估算

> 以下为按模型规模和 token 量推算的**估算值，未实测**。

| 项目 | 估算 |
| --- | --- |
| GPU0 显存（训练） | 峰值约 20–30GB（4B 权重 9–10GB + LoRA/AdamW + 梯度检查点下的激活） |
| GPU1–3 显存（采样） | 每个 vLLM 副本预占约 40GB（`gpu_memory_utilization=0.85`） |
| 评测服务 | 单卡预占约 43GB（`localtrio serve` 默认 `gpu_memory_utilization=0.9`） |
| Medical SFT | 每步约 10–25 秒；总步数 = 3 × ⌈样本数 / 16⌉（启动时打印） |
| Medical OPD / Base-anchor OPD（各 300 步） | 每步约 1–1.5 分钟，各约 5–8 小时 |
| IDT-OPD（600 步） | 约 10–16 小时 |
| MedQA-zh / C-Eval 评测 | 单卡服务约 10–30 分钟 / 0.5–1.5 小时 |
| 磁盘 | 基模约 9–10GB；每个 sampler weights 几十到几百 MB，state 约为其数倍 |

## 8. 常见问题

**怎么改 localtrio 配置？** 写一份自己的 YAML（例如复制 `configs/4gpu.yaml` 修改后存为 `my-medical.yaml`），在命令前加 `LOCALTRIO_CONFIG=my-medical.yaml`；也可以命名为 `localtrio.yaml` 放在当前工作目录（仓库根目录）自动读取。常用项：

```yaml
trainer:
  gpus: [0]
  max_tokens_per_microbatch: 8192   # 训练 OOM 时调小
  lora_target_modules: standard     # vLLM 报 LoRA 模块不支持时改用
models:
  default:
    gpus: [1, 2, 3]
    gpu_memory_utilization: 0.80    # 采样 OOM 时调小
    max_num_seqs: 128
```

- **训练 OOM**：调小 `trainer.max_tokens_per_microbatch`（如 8192）。它只改变 micro-batch 切分，梯度仍按整个 batch 的 active token 数归一化，不改变实验设置；`--batch-size`、`--group-size`、`--max-tokens` 会改变实验本身，尽量不动。
- **vLLM 采样 OOM**：调低 `models.default.gpu_memory_utilization` 或 `max_num_seqs`；评测服务则调低 `--gpu-memory-utilization`、`--max-model-len` 或评测脚本的 `--concurrency`。
- **vLLM 报 LoRA 不支持某个模块**（如 Qwen3.5 线性注意力的 `in_proj_*` / `out_proj`）：把 `trainer.lora_target_modules` 改成 `standard`（只用 q/k/v/o + gate/up/down），然后从 SFT 开始**重新训练**；已有 adapter 不能直接转换。
- **评测报 `max_lora_rank`**：`localtrio serve` 默认 `--max-lora-rank 64`，训练用的 rank 更大时调高它。
- **评测报无法连接 / 动态加载 LoRA 失败**：确认服务已经启动、端口一致；动态加载依赖 `localtrio serve` 设置的 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`，自己直接起 vLLM 时请加上这个环境变量，或用 `--lora 名字=目录` 预先注册。
- **评测请求超时**：本地单卡生成 8192 token 较慢，默认超时已从原版的 120 秒放宽到 600 秒；仍超时时调大 `--timeout` 或调小 `--concurrency`。
- **报 `trio://` 是 PyTRIO 云端路径**：本地版只接受 `~/.cache/localtrio/runs/...` 下的本地目录。
- **训练中断后续训**：state 目录可以用 `create_training_client_from_state_with_optimizer` 恢复 LoRA 和 AdamW 状态；本章脚本的第三阶段按原文设计只恢复权重、不继承优化器。
- **数值与原文不同**：原文结果来自 PyTRIO 云端（训练、采样与 OpenAI 兼容服务都在服务端）；本地版的 vLLM 数值误差、chat template 处理都可能带来差异，请以本地评测的 Base 为对照。训练端还有两处实现差异：
  - **输出层 `lm_head`**：`02`、`03`、`05` 脚本创建训练客户端时没有传 `train_unembed`，PyTRIO 默认 `train_unembed=True`，原文还训练了 `lm_head`；localtrio 默认不训练。设 `trainer.allow_unembed: true` 可开启（以 LoRA 形式训练 `lm_head`），前提是你的 vLLM 能为这个模型加载带 `lm_head` 的 adapter；各阶段要用同一份配置，第三阶段从训练过 `lm_head` 的 state 恢复时也需要这项设置。
  - **LoRA 缩放**：`lora_alpha` 默认是 32，不随 rank 变化（可用 `trainer.lora_alpha` 修改），所以缩放系数 `alpha / rank` 随 rank 变化：Medical OPD 与 IDT-OPD 的 fresh Student 默认 rank 16，系数是 2；Medical SFT 默认 rank 32，系数是 1。PyTRIO 服务端的 alpha 未公开，照搬原文的学习率时本地的有效步长可能与原文不同，SFT 与 OPD 阶段的学习率也不能直接按数值比较。

  详见正文“评测 sampler weights”一节末尾的说明和 [localtrio 说明](../../docs/content/guide/localtrio.md#与-pytrio-的已知差异)。
