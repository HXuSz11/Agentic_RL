---
title: 4×A6000 显存与 GPU 布局规划
description: 在 4 张 RTX A6000（48GB）上分配训练卡、采样卡与教师卡：4B、9B、27B 模型的显存估算、各章节推荐布局与 OOM 排查顺序。
---

# 显存与 GPU 布局规划

RTX A6000 是 Ampere 架构的 48GB 显卡：支持 bf16 和 FlashAttention-2，不支持 FP8。本页的数字都是**按模型尺寸推算的估算值**，不是实测；第一次运行某个章节时请先用小参数，并用 `nvidia-smi` 观察实际占用。

## 一张卡负责什么

localtrio 把“训练”和“采样”放在不同的进程、不同的卡上：

| 角色 | 进程 | 显存主要花在哪里 |
| --- | --- | --- |
| 训练（trainer） | HF transformers + PEFT | bf16 基模权重 + LoRA 的 fp32 权重/梯度/AdamW 状态 + 激活（开启梯度检查点） |
| 采样（sampler） | vLLM，每卡一个副本 | bf16 基模权重 + KV cache / 线性注意力状态（由 `gpu_memory_utilization` 控制总占比） |
| 教师（teacher，可选） | vLLM | 另一个基模时才需要单独的卡；同一基模的教师复用采样副本 |

RL 训练里大部分时间花在采样，所以默认把 3 张卡都给 vLLM。

## 预设布局

| 预设 | GPU0 | GPU1 | GPU2 | GPU3 | 适用章节 |
| --- | --- | --- | --- | --- | --- |
| 默认（不写配置，或 `configs/4xa6000.yaml`） | 训练 | 采样 #1 | 采样 #2 | 采样 #3 | 除 General OPD 外的全部 4B 章节 |
| `configs/4xa6000-opd-27b.yaml` | 学生训练 | 学生采样 | 27B 教师（TP=2） | 27B 教师（TP=2） | General OPD |
| `configs/4xa6000-big-student.yaml` | 训练（切分） | 训练（切分） | 采样 #1 | 采样 #2 | 9B 学生（如 Spec-o3 的 9B 实验） |
| `configs/1gpu.yaml` | 训练 + 采样（vLLM 40%） | — | — | — | 只有一张卡时 |
| `configs/cpu-debug.yaml` | CPU + HF 采样 | | | | 只用于检查代码路径 |

切换方法：`LOCALTRIO_CONFIG=configs/<预设>.yaml uv run python ...`；用 `uv run python -m localtrio info` 确认生效。

## 显存估算

### 模型权重（bf16）

| 模型 | 参数量 | 权重显存 | 说明 |
| --- | --- | --- | --- |
| Qwen3.5-4B（本仓库默认学生） | ~4–5B（含视觉塔） | ~9–10 GB | 单卡即可训练和采样 |
| Qwen3.5-9B | ~9–10B | ~19–20 GB | 单卡可训练（调小 micro-batch），双卡更宽裕 |
| Qwen3.6-27B（General OPD 教师） | ~27B | ~54 GB | 超过单卡，必须 `tensor_parallel_size: 2` |

### 训练卡（4B，LoRA rank 32，梯度检查点）

- LoRA 参数：all-linear、rank 32 约为基模参数的 1–2%，fp32 权重 + 梯度 + AdamW 两个动量 ≈ 每参数 16 字节 → **约 1–2 GB**。
- 激活：梯度检查点只保留每层输入，约 `hidden × 2 字节 × 层数` 每 token；4B 模型约 0.15 MB/token，16k token 的 micro-batch 约 **2–3 GB**，再加上重算一层时的峰值和 lm_head 分块（`logits_chunk_size=1024` 时约 1–3 GB）。
- 合计：默认 `max_tokens_per_microbatch: 16384` 时大约 **15–25 GB**，48GB 卡有余量；长轨迹章节（ALFWorld 12K、Tournament-GRPO 18K）也在这个量级，单条超长样本会单独成一个 micro-batch。

### 采样卡（vLLM）

`gpu_memory_utilization: 0.85` 时每卡约 40GB 归 vLLM：4B 权重约 10GB，其余 ~30GB 给 KV cache 和线性注意力状态。Qwen3.5 只有约 1/4 的层是全注意力层，KV cache 相当省，组采样（`n=group_size`）和长回答都能放下。

## 各章推荐

| 章节 | 布局 | 备注 |
| --- | --- | --- |
| Loss Functions / GRPO / DAPO / GSPO | 默认 | DAPO 的长 CoT（4096–8192 token）若训练 OOM，调小 `max_tokens_per_microbatch` |
| Vision GRPO / Spec-o3（4B） | 默认 | 图片 token 计入序列长度 |
| Spec-o3（9B） | `4xa6000-big-student` | 或在默认布局下把 micro-batch 调到 8192 |
| Jev 决策模型 | 默认 | 主要是 SFT + 推理，资源占用小 |
| General OPD | `4xa6000-opd-27b` | 教师只做 `compute_logprobs`；想省卡可换更小教师（效果会变） |
| Medical OPD / OPSD / AgentOPSD | 默认 | 教师是同一基模（基模本身或 SFT LoRA），复用采样副本 |
| Search-R1 / ReTool / Tournament-GRPO | 默认 | 搜索、沙箱、裁判在 CPU / 外部 API；多轮上下文较长 |
| ALFWorld / TEMPO | 默认 | TextWorld 环境在 CPU 上运行，CPU 核数影响并发 |

想在训练的同时用本地模型当裁判或评测服务：把 `models.default.gpus` 缩成 `[1, 2]`，留 GPU3 给 `uv run python -m localtrio serve --base-model <裁判/评测模型> --gpus 3`。

## OOM 排查顺序

1. **训练卡 OOM**（trainer 日志里的 `CUDA out of memory`）：
   - 调小 `trainer.max_tokens_per_microbatch`（16384 → 8192 → 4096）；
   - 调小 `trainer.logits_chunk_size`（1024 → 512）；
   - 确认 `gradient_checkpointing: true`；
   - 9B 以上用双卡训练布局。
2. **采样卡 OOM / vLLM 启动失败**：
   - 启动时报 `Free memory on device ... is less than desired GPU memory utilization` 或 CUDA OOM（常见于同卡还有别的进程）：调小 `models.default.gpu_memory_utilization`（0.85 → 0.75）；
   - 启动时报 `... KV cache is needed, which is larger than the available KV cache memory`（模型自带的 `max_model_len` 放不下）：设置 `models.default.max_model_len`（如 32768），或在显存允许时调大 `gpu_memory_utilization`，这时调小它只会让 KV cache 更少；
   - 运行中做 `compute_logprobs` / prompt logprobs 时 OOM：vLLM 0.33 默认的 Model Runner V2 按固定 1024 行分块计算整词表 logprob，额外的显存尖峰约 0.5–1GB，不随 `max_num_batched_tokens` 变化，启动时的显存估算也只覆盖其中一部分。先调低这张卡上各引擎的 `gpu_memory_utilization` 给它留余量，再考虑调小 `max_model_len` 或并发数。调小 `max_num_batched_tokens` 只在设置了 `VLLM_USE_V2_MODEL_RUNNER=0`（旧 runner 会一次算完整个预填充块）时才压得住这个尖峰；
   - `max_num_batched_tokens` 默认 2048。只做生成的引擎可以调大到 8192，加快长 prompt 预填充，代价是 KV cache 略少；
   - 与训练共卡（`1gpu.yaml`）时把 vLLM 压到 0.35–0.40。
3. **两个基模抢同一组卡**：localtrio 会直接报错并提示在 YAML 里单独分配 GPU。

## 速度相关

- 安装 Qwen3.5 线性注意力的训练 kernel：`uv sync --extra kernels`（flash-linear-attention + causal-conv1d）；之后要装 dev / alfworld 时把 kernels 一起写上（如 `uv sync --extra kernels --extra alfworld`），否则 `uv sync` 会把 kernels 卸掉。不装也能跑，但训练端会退回纯 PyTorch 实现，长序列明显更慢、更占显存。
- 想用 FlashAttention-2：安装 `flash-attn` 后把 `trainer.attn_implementation` 改成 `flash_attention_2`。
- 采样吞吐随采样卡数近似线性增长；训练端是单进程，batch 很大时可以调大 micro-batch 提高利用率。
- 每个 RL step 都会写一次 LoRA 权重（bf16，几十到几百 MB），把 `home` 放在 SSD 上。
