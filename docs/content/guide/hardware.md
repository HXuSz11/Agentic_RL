---
title: 显存与 GPU 布局规划
description: 1 张、2 张还是 4 张卡？各章节的最低与推荐配置，24GB、48GB、80GB 显卡的显存估算，预设布局怎么选，以及 OOM 时先调哪个参数。
---

# 显存与 GPU 布局规划

本仓库不绑定某一款显卡，也**不要求 4 张卡**：localtrio 启动时会检测本机有几张卡，按“GPU 0 训练、其余采样”自动分配；只有 1 张卡时训练和采样共用它。卡越多，采样副本越多，跑得越快。

本页的数字都是**按模型尺寸推算的估算值**，还没有在每种显卡上实测。第一次运行某个章节时先用小参数，并用 `nvidia-smi` 观察实际占用。

## 需要什么样的显卡

- NVIDIA 显卡，**Ampere 或更新的架构**（需要 bf16）：RTX 3090 / 4090 / 5090、RTX A6000 / RTX 6000 Ada、L40S、A100、H100 等都可以。
- 驱动 ≥ 580：仓库 `uv.lock` 锁定的 torch 2.14 / vLLM 0.33 基于 CUDA 13.0。
- 每张卡最好 ≥ 24GB。张量并行（例如 27B 教师 `tensor_parallel_size: 2`）用的几张卡最好型号相同；训练卡和采样卡型号可以不同。
- 多卡之间不需要 NVLink：训练和各个采样副本是独立进程，每步只通过本地磁盘交换几十到几百 MB 的 LoRA 权重。

## 一定要 4 张卡吗？

不需要。不写配置文件时，按检测到的卡数自动分配：

| 卡数 | 自动布局 | 能跑什么（4B 学生，Qwen3.5-4B） | 相对 4 卡的速度（估算） |
| --- | --- | --- | --- |
| 1 张 | 训练与采样共用 GPU 0，vLLM 只占 40% 显存（`configs/1gpu.yaml`） | 单卡需 **≥48GB**；全部单基模章节 | 约 1/3～1/4 |
| 2 张 | GPU 0 训练，GPU 1 采样（`configs/2gpu.yaml`） | 每卡 ≥24GB；全部单基模章节。每卡 ≥48GB 时还能跑 9B 学生，以及 General OPD 换成 9B 教师 | 约 1/2～1/3 |
| 3 张 | GPU 0 训练，GPU 1–2 采样 | 同 2 张，采样更快 | 约 2/3～3/4 |
| 4 张 | GPU 0 训练，GPU 1–3 采样（`configs/4gpu.yaml`） | 全部章节；每卡 ≥48GB 时 General OPD 可以用 27B 教师预设 | 基准 |
| 8 张 | GPU 0 训练，GPU 1–7 采样 | 采样更快；也可以用 `CUDA_VISIBLE_DEVICES` 把机器分成两组，同时跑两个实验 | 采样部分接近线性加速 |

RL 训练的大部分时间花在采样上，所以多出来的卡默认都给 vLLM。卡少时只是慢，实验设置不用改：组大小、batch、学习率都保持原样，结果在统计意义上可比。

只有 1 张 24GB 或 32GB 的卡时，放不下“训练 + vLLM”两份 4B 权重和 KV cache。可以用 `--base-model Qwen/Qwen3.5-0.8B` 和小参数跑通代码路径，但这已经不是原文的实验设置。

## 按显存看每张卡能做什么

| 单卡显存 | 训练卡（LoRA） | 采样卡（vLLM） | 说明 |
| --- | --- | --- | --- |
| 24GB（RTX 3090 / 4090） | 4B：把 `trainer.max_tokens_per_microbatch` 调到 8192 | 4B：约 10GB 留给 KV cache，长上下文章节设置 `max_model_len` | 9B 学生和 27B 教师都不建议 |
| 32GB（RTX 5090） | 4B 宽裕 | 4B 宽裕 | 9B 训练仍偏紧 |
| 48GB（RTX A6000 / 6000 Ada、L40S） | 4B 宽裕；9B 单卡可训（micro-batch 8192） | 4B / 9B | 单卡可以同时做训练和采样（4B） |
| 80GB（A100 / H100） | 9B 宽裕 | 27B 教师单卡即可（约 54GB 权重） | General OPD 的 27B 教师不再需要两张卡 |

## 一张卡负责什么

localtrio 把“训练”和“采样”放在不同的进程上，卡够时也放在不同的卡上：

| 角色 | 进程 | 显存主要花在哪里 |
| --- | --- | --- |
| 训练（trainer） | HF transformers + PEFT | bf16 基模权重 + LoRA 的 fp32 权重/梯度/AdamW 状态 + 激活（开启梯度检查点） |
| 采样（sampler） | vLLM，每卡一个副本 | bf16 基模权重 + KV cache / 线性注意力状态（由 `gpu_memory_utilization` 控制总占比） |
| 教师（teacher，可选） | vLLM | 另一个基模时才需要单独的卡；同一基模的教师复用采样副本 |

## 预设布局

| 预设 | 卡数 | 分工 | 适用 |
| --- | --- | --- | --- |
| 不写配置 | 任意 | GPU 0 训练，其余采样；单卡共用 | 绝大多数章节 |
| `configs/1gpu.yaml` | 1（≥48GB） | 训练 + 采样共卡（vLLM 40%），micro-batch 8192 | 只有一张大显存卡 |
| `configs/2gpu.yaml` | 2（≥24GB） | GPU 0 训练（micro-batch 8192），GPU 1 采样 | 两张卡，或两张 24GB 卡 |
| `configs/4gpu.yaml` | 4 | GPU 0 训练，GPU 1–3 采样 | 与自动布局相同，只是把取值写明方便修改 |
| `configs/4gpu-big-student.yaml` | 4 | GPU 0–1 训练（按层切分），GPU 2–3 采样 | 9B 学生（如 Spec-o3 的 9B 实验），想要更宽裕的布局时 |
| `configs/4gpu-opd-27b.yaml` | 4（≥48GB） | GPU 0 学生训练，GPU 1 学生采样，GPU 2–3 27B 教师（TP=2） | General OPD 的 27B 教师；80GB 卡可改成教师单卡 |
| `configs/cpu-debug.yaml` | 0 | CPU + HF 采样 | 只用于检查代码路径 |

切换方法：`LOCALTRIO_CONFIG=configs/<预设>.yaml uv run python ...`；用 `uv run python -m localtrio info` 确认生效。也可以只用环境变量改分工，例如 `LOCALTRIO_TRAINER_GPUS=0 LOCALTRIO_SAMPLER_GPUS=1,2`。

## 各章最低与推荐配置

| 章节 | 最低配置（估算） | 推荐 | 备注 |
| --- | --- | --- | --- |
| Loss Functions | 不需要 GPU | — | 只有讲解，没有训练代码 |
| GRPO / DAPO / GSPO | 1×48GB 或 2×24GB | 4 张卡 | DAPO 的长 CoT（4096–8192 token）训练 OOM 时调小 `max_tokens_per_microbatch` |
| Vision GRPO / Spec-o3（4B） | 1×48GB 或 2×24GB | 4 张卡 | 图片 token 计入序列长度 |
| Spec-o3（9B） | 2×48GB（`2gpu.yaml`） | 4×48GB（`4gpu-big-student.yaml`） | 24GB 卡上 9B 的训练和采样都放不下 |
| Jev 决策模型 | 1×48GB 或 2×24GB | 2 张卡 | 以 SFT 和推理为主，资源占用小 |
| General OPD | 2×48GB（换 9B 教师，见章节内示例） | 4×48GB 或 2–3×80GB（27B 教师） | 教师只做 `compute_logprobs` |
| Medical OPD / OPSD / AgentOPSD | 1×48GB 或 2×24GB | 4 张卡 | 教师是同一基模（基模本身或 SFT LoRA），复用采样副本，不需要额外的卡 |
| Search-R1 / ReTool / Tournament-GRPO | 1×48GB 或 2×24GB | 4 张卡 | 搜索、沙箱、裁判在 CPU / 外部 API；多轮上下文较长，24GB 采样卡要设置 `max_model_len` |
| ALFWorld / TEMPO | 1×48GB 或 2×24GB | 4 张卡 | TextWorld 环境在 CPU 上运行，CPU 核数影响并发 |

想在训练的同时用本地模型当裁判或评测服务：给它单独留一张卡，例如 4 张卡时把 `models.default.gpus` 缩成 `[1, 2]`，再用 `uv run python -m localtrio serve --base-model <裁判/评测模型> --gpus 3` 启动。

## 显存估算

### 模型权重（bf16）

| 模型 | 参数量 | 权重显存 | 说明 |
| --- | --- | --- | --- |
| Qwen3.5-4B（本仓库默认学生） | ~4–5B（含视觉塔） | ~9–10 GB | 24GB 卡可以单独训练或单独采样；一张卡同时做两件事需要 ≥48GB |
| Qwen3.5-9B | ~9–10B | ~19–20 GB | 48GB 卡可训练（调小 micro-batch），双卡切分更宽裕 |
| Qwen3.6-27B（General OPD 教师） | ~27B | ~54 GB | 48GB 卡要 `tensor_parallel_size: 2`；80GB 卡单卡即可 |

### 训练卡（4B，LoRA rank 32，梯度检查点）

- LoRA 参数：all-linear、rank 32 约为基模参数的 1–2%，fp32 权重 + 梯度 + AdamW 两个动量 ≈ 每参数 16 字节 → **约 1–2 GB**。
- 激活：梯度检查点只保留每层输入，约 `hidden × 2 字节 × 层数` 每 token；4B 模型约 0.15 MB/token，16k token 的 micro-batch 约 **2–3 GB**，再加上重算一层时的峰值和 lm_head 分块（`logits_chunk_size=1024` 时约 1–3 GB）。
- 合计：默认 `max_tokens_per_microbatch: 16384` 时大约 **15–25 GB**，48GB 卡有余量；24GB 卡把它调到 8192（约 13–18 GB）。长轨迹章节（ALFWorld 12K、Tournament-GRPO 18K）里单条超长样本会单独成一个 micro-batch，24GB 卡上仍 OOM 时再降到 4096、`logits_chunk_size` 调到 512。

### 采样卡（vLLM）

`gpu_memory_utilization: 0.85` 时 vLLM 用掉每卡显存的 85%：48GB 卡约 40GB（4B 权重约 10GB，其余 ~30GB 给 KV cache 和线性注意力状态），24GB 卡约 20GB（KV cache 只剩 ~10GB）。Qwen3.5 只有约 1/4 的层是全注意力层，KV cache 相当省，组采样（`n=group_size`）和长回答在 48GB 卡上都能放下；24GB 卡上若启动时报 KV cache 不足，设置 `models.default.max_model_len`（如 16384 或 32768，需大于各章 prompt + 回答长度）。

## OOM 排查顺序

1. **训练卡 OOM**（trainer 日志里的 `CUDA out of memory`）：
   - 调小 `trainer.max_tokens_per_microbatch`（16384 → 8192 → 4096）；它只改变 micro-batch 切分，梯度仍按整个 batch 归一化，不改变实验设置；
   - 调小 `trainer.logits_chunk_size`（1024 → 512）；
   - 确认 `gradient_checkpointing: true`；
   - 9B 以上的学生用双卡训练布局（`4gpu-big-student.yaml`）。
2. **采样卡 OOM / vLLM 启动失败**：
   - 启动时报 `Free memory on device ... is less than desired GPU memory utilization` 或 CUDA OOM（常见于同卡还有别的进程）：调小 `models.default.gpu_memory_utilization`（0.85 → 0.75）；
   - 启动时报 `... KV cache is needed, which is larger than the available KV cache memory`（模型自带的 `max_model_len` 放不下，24GB 卡上常见）：设置 `models.default.max_model_len`（如 32768），或在显存允许时调大 `gpu_memory_utilization`，这时调小它只会让 KV cache 更少；
   - 运行中做 `compute_logprobs` / prompt logprobs 时 OOM：vLLM 0.33 默认的 Model Runner V2 按固定 1024 行分块计算整词表 logprob，额外的显存尖峰约 0.5–1GB，不随 `max_num_batched_tokens` 变化，启动时的显存估算也只覆盖其中一部分。先调低这张卡上各引擎的 `gpu_memory_utilization` 给它留余量，再考虑调小 `max_model_len` 或并发数。调小 `max_num_batched_tokens` 只在设置了 `VLLM_USE_V2_MODEL_RUNNER=0`（旧 runner 会一次算完整个预填充块）时才压得住这个尖峰；
   - `max_num_batched_tokens` 默认 2048。只做生成的引擎可以调大到 8192，加快长 prompt 预填充，代价是 KV cache 略少；
   - 与训练共卡（单卡、`1gpu.yaml`）时把 vLLM 压到 0.35–0.40。
3. **两个基模抢同一组卡**：localtrio 会直接报错并提示在 YAML 里单独分配 GPU。

## 速度相关

- 安装 Qwen3.5 线性注意力的训练 kernel：`uv sync --extra kernels`（flash-linear-attention + causal-conv1d）；之后要装 dev / alfworld 时把 kernels 一起写上（如 `uv sync --extra kernels --extra alfworld`），否则 `uv sync` 会把 kernels 卸掉。不装也能跑，但训练端会退回纯 PyTorch 实现，长序列明显更慢、更占显存。
- 想用 FlashAttention-2：安装 `flash-attn` 后把 `trainer.attn_implementation` 改成 `flash_attention_2`。
- 采样吞吐随采样卡数近似线性增长；训练端是单进程，batch 很大时可以调大 micro-batch 提高利用率。
- 每个 RL step 都会写一次 LoRA 权重（bf16，几十到几百 MB），把 `home` 放在 SSD 上。
