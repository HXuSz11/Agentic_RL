---
title: localtrio：在自己的 GPU 上运行 PyTRIO 风格的 Train API
description: localtrio 用 HF transformers + PEFT 训练 LoRA、用 vLLM 采样，接口与 PyTRIO 0.2.9 同名同参。本文说明架构、API 对照、loss 语义、权重同步与迁移方法。
---

# localtrio：本地版 Train API

原教程的每个实验都只和一组很小的接口打交道：创建 LoRA 训练会话、用当前权重采样、把带 advantage 的样本交给 `forward_backward`、`optim_step`、保存权重。PyTRIO（以及 Tinker）把这组接口放在云端；**localtrio 在本地把同一组接口实现了一遍**，所以实验代码只需要改一行：

```python
import localtrio as trio   # 原来是 import pytrio as trio
```

## 架构

```text
你的实验脚本（主进程，不占用 GPU）
│  数据、reward、advantage、训练循环、自定义 loss（CPU 上的 autograd）
│
├── trainer 子进程   CUDA_VISIBLE_DEVICES=0
│     HF transformers + PEFT LoRA（bf16 基模 + fp32 LoRA/AdamW）
│     forward / forward_backward / optim_step / save_state
│     一个基模进程可托管多个训练 run（每个 run 一个 adapter + 一份优化器状态）
│
└── sampler 子进程 ×N   CUDA_VISIBLE_DEVICES=1 / 2 / 3（每卡一个副本）
      vLLM AsyncLLM，连续批处理；token-in / token-out
      LoRA 热加载：每个 sampler_weights 目录 = 一个 LoRARequest
```

- 主进程与子进程通过 Unix socket 通信；调用按提交顺序发送，trainer 串行执行，所以 `forward_backward` → `optim_step` 的先后关系与 PyTRIO 一样由提交顺序决定。
- 引擎在第一次用到时启动并常驻，脚本退出时自动清理（包括 vLLM 派生的进程）。启动日志写在 `~/.cache/localtrio/logs/`。
- 子进程通过 `python -m localtrio.worker` 启动，不会重复执行你的脚本。

## 权重同步：每一步怎样让采样用上新策略

```text
optim_step ──► save_weights_and_get_sampling_client()
                 │ trainer 把当前 LoRA 写成标准 PEFT 目录（bf16，~几十到几百 MB）
                 ▼
               runs/<run>/tmp_sampler/temp-sampling-xxxx/
                 │ 返回绑定该目录的 SamplingClient
                 ▼
               sample() ──► vLLM 用 LoRARequest(path) 热加载，不重启引擎
```

临时目录只保留最近 4 个。同一基模的所有 LoRA（不同 step、教师 SFT 权重）以及不带 LoRA 的基模共用同一组 vLLM 副本，所以 OPSD 的 step-0 教师、Medical OPD 的 SFT 教师都不需要额外显卡。

## API 对照

| PyTRIO 0.2.9 | localtrio | 本地实现 |
| --- | --- | --- |
| `trio.ServiceClient()` | 同名 | 读取配置，管理引擎子进程；`api_key` 等参数被忽略 |
| `create_lora_training_client(base_model, rank, seed, train_mlp, train_attn, train_unembed)` | 同名（含 `_async`） | PEFT `LoraConfig`，`lora_alpha=32`；`train_unembed` 默认忽略（见下方“已知差异”） |
| `create_training_client_from_state(path)` / `_with_optimizer` | 同名（含 `_async`） | 从 `save_state` 目录恢复 adapter（和 AdamW 状态） |
| `create_sampling_client(base_model, model_path=None)` | 同名（含 `_async`） | `model_path` 是本地 sampler_weights 目录 |
| `training_client.forward(data, loss_fn)` | 同名 | 无梯度前向，返回逐 token `logprobs` |
| `forward_backward(data, loss_fn, loss_fn_config)` | 同名 | 前向 + 反向，梯度累积到下一次 `optim_step` |
| `forward_backward_custom(data, loss_fn)` | 同名 | 先前向取 logprob，主进程算 loss 与 `dL/dlogprob`，再在 trainer 上反传（梯度严格等于用户 loss 的梯度） |
| `optim_step(AdamParams)` | 同名 | `torch.optim.AdamW`，每步可改 lr；`grad_clip_norm>0` 时裁剪；返回 `grad_norm` |
| `save_state(name)` / `save_weights_for_sampler(name)` | 同名 | 返回**本地绝对路径**；`ttl_seconds` 被忽略 |
| `save_weights_and_get_sampling_client()` | 同名（含 `_async`） | 见上节 |
| `sampling_client.sample(prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs, return_text)` | 同名（含 `_async`） | vLLM `TokensPrompt` + `SamplingParams(n=num_samples, logprobs=0)` |
| `compute_logprobs(prompt)` | 同名（含 `_async`） | vLLM `prompt_logprobs`，首位为 `None` |
| `trio.Datum / ModelInput / ImageChunk / TensorData / SamplingParams / AdamParams` | 同名同字段 | pydantic 类型，行为一致 |
| `APIFuture`：`.result()` / `await` | 同名 | 同步方法返回 future；`*_async` 在 `await` 后同样得到 future（`sample_async`、`compute_logprobs_async` 直接返回结果），与 PyTRIO 一致 |
| `trio.configure(timeout=..., ...)` | 同名 | 网络与超时参数被忽略；可传 localtrio 自己的配置项 |

## Loss 语义

记 `lp` 为当前策略在 target token 上的 logprob，`old` 为采样时 logprob，`A` 为 advantage，`ratio = exp(lp - old)`：

| `loss_fn` | 逐 token loss | `loss_fn_config` |
| --- | --- | --- |
| `cross_entropy` | `-w · lp` | `weights` 作为 `loss_fn_inputs` 传入，缺省为 1 |
| `importance_sampling` | `-ratio · A` | — |
| `ppo` | `-min(ratio · A, clip(ratio, lo, hi) · A)` | `clip_low_threshold`（默认 0.8）、`clip_high_threshold`（默认 1.2） |
| `cispo` | `-sg(clip(ratio, lo, hi)) · lp · A` | 默认 `lo=0`、`hi=4` |
| `dro` | `-(lp · A - 0.5 · β · (lp - old)²)` | `beta`（默认 0.05） |

- `target_tokens == -100` 的位置不参与计算；prompt 位置一般用 target 0、advantage 0 占位（贡献为 0）。
- **归一化**：一次 `forward_backward` 调用就是一个 logical batch，内置 loss 的梯度按整批的 active token 数归一化（active token = `target != -100` 的位置，与 PyTRIO SDK 统计 `total_active_tokens` 的口径一致，prompt / observation 位置若用 target 0 占位也会计入）；micro-batch 拆分不改变梯度。全局可在配置里改成 `samples`（除以样本数）或 `sum`（Tinker 风格，不归一化），也可以按调用覆盖：`forward_backward(data, "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28, "normalization": "sum"})`。原教程各章对 PyTRIO 服务端口径的假设不完全相同，相关章节的快速启动里写了建议值；Adam 对整体常数缩放不敏感，口径差异主要影响不同 step 之间的相对权重。
- 返回的 `metrics` 与 PyTRIO 同名：`loss_sum`、`token_count`、`loss_mean = loss_sum / token_count`；custom loss 返回的指标会合并进去。
- 采样端返回的 logprob 是**原始模型分布**（vLLM `logprobs_mode=raw_logprobs`），与训练端 `forward` 口径一致，on-policy 时 ratio ≈ 1。

## 图文输入

`ImageChunk` 在序列里替代一个 `<|image_pad|>`：

- 训练端用与基模匹配的 image processor 处理像素，把占位符展开为 `expected_tokens` 个视觉 token，并**校验** `expected_tokens` 与 processor 的计算一致；Qwen3.5 需要的 `mm_token_type_ids` 自动生成；视觉塔不加 LoRA。
- 采样端给 vLLM 一个 `<|image_pad|>` + `multi_modal_data`，由 vLLM 按真实尺寸展开。
- 与 PyTRIO 相同：带图片的 prompt 不支持 `compute_logprobs` / prompt logprobs。
- 图文样本和文本样本一样按 `max_tokens_per_microbatch` 打包（视觉 token 计入长度）。
- vLLM 端默认 `limit_mm_per_prompt: {image: 8, video: 0}`，一条 prompt 需要更多图片时在 `models.default.limit_mm_per_prompt` 里调大。
- 与引擎一致的模型目录 / image processor：`trio.get_model_dir(base_model)`、`trio.get_image_processor(base_model)`。

## LoRA 作用范围

默认 `trainer.lora_target_modules: all-linear`：语言模型里的全部投影层——全注意力层的 `q/k/v/o_proj`、Qwen3.5 线性注意力（Gated DeltaNet）层的 `in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj`，以及 MLP 的 `gate/up/down_proj`。vLLM ≥0.11 对 Qwen3.5 的这些模块都支持 LoRA。如果你的 vLLM 版本加载 adapter 时报 “expected target modules ...”，把它改成 `standard`（只用 q/k/v/o + MLP）。

## 配置

不提供配置时使用内置默认布局（GPU0 训练，其余卡做采样）。需要改布局时写一个 YAML，并通过环境变量指定：

```bash
LOCALTRIO_CONFIG=configs/4xa6000-opd-27b.yaml uv run python experiments/general-opd/01-demo-sync.py
```

常用配置项：

```yaml
home: ~/.cache/localtrio          # checkpoint 与日志目录
model_source: auto                # auto | huggingface | modelscope
model_paths:                      # 可选：模型名 → 本地目录
  Qwen/Qwen3.5-4B: /data/models/Qwen3.5-4B
trainer:
  gpus: [0]
  attn_implementation: sdpa       # flash_attention_2 需要安装 flash-attn
  max_tokens_per_microbatch: 16384
  lora_target_modules: all-linear
  loss_normalization: active_tokens
models:
  default:                        # 训练基模的采样引擎
    gpus: [1, 2, 3]
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.85
    max_model_len: null           # null = 模型自带上限
    max_num_batched_tokens: 2048  # 预填充分块；compute_logprobs 的显存尖峰与它成正比
  Qwen/Qwen3.6-27B:               # 其他基模（例如 OPD 教师）必须单独分配 GPU
    gpus: [2, 3]
    tensor_parallel_size: 2
```

环境变量：`LOCALTRIO_CONFIG`、`LOCALTRIO_HOME`、`LOCALTRIO_TRAINER_GPUS=0`、`LOCALTRIO_SAMPLER_GPUS=1,2,3`、`LOCALTRIO_SAMPLER_BACKEND=vllm|hf`、`LOCALTRIO_MODEL_SOURCE`、`LOCALTRIO_WORKER_LOGS=inherit`（把引擎日志直接打印到终端）。

## 命令行工具

```bash
uv run python -m localtrio info                                  # 当前配置、GPU、vLLM 版本
uv run python -m localtrio download Qwen/Qwen3.5-4B              # 预下载模型
uv run python -m localtrio serve --base-model Qwen/Qwen3.5-4B \
    --lora step100=$HOME/.cache/localtrio/runs/<run>/sampler_weights/step100 --gpus 1
                                                                 # OpenAI 兼容服务（base_url http://127.0.0.1:8000/v1）
uv run python -m localtrio merge --adapter <sampler_weights 目录> --out ./merged
                                                                 # 把 LoRA 合并进基模，导出完整权重
```

`serve` 适合用 OpenAI SDK 写的评测脚本（例如 Medical OPD 的 MedQA / C-Eval 评测）：模型名填基模名或 `--lora` 里的名字，API key 任意。`--max-model-len` 默认 32768（传 0 用模型自带上限）；只服务基模（例如大模型裁判）时加 `--no-lora` 省下 LoRA 预留显存；不认识的参数会原样转发给 vLLM，例如 `--enforce-eager`、`--data-parallel-size 3`。服务开启了 `VLLM_ALLOW_RUNTIME_LORA_UPDATING`，评测脚本可以通过 `/v1/load_lora_adapter` 动态挂载新的 checkpoint。

## 从 PyTRIO 代码迁移

1. `import pytrio as trio` → `import localtrio as trio`；删掉 `trio login`、`PYTRIO_API_KEY`。
2. 把 `trio://...` 路径换成 `save_weights_for_sampler` / `save_state` 返回的本地目录。
3. 调用了 PyTRIO OpenAI 兼容端点（`https://pytrio.cn/api/openai/v1`）的脚本，改用 `python -m localtrio serve` 起的本地服务。
4. 如果用到了第二个基模（例如更大的教师），在 YAML 里给它单独分配 GPU。

和 Tinker 的对应关系也一样：Tinker 的 `ServiceClient → TrainingClient / SamplingClient → forward_backward / optim_step / sample` 与这里同构，差别主要在 checkpoint 路径和 loss 归一化口径，迁移时按上面的 loss 表核对。

## 与 PyTRIO 的已知差异

- 没有远程控制面：`create_rest_client()` 不可用，checkpoint 就是磁盘上的目录。
- LoRA rank 允许 1–128（PyTRIO 为 4–64）；vLLM 侧 `max_lora_rank` 默认 64，用更大的 rank 时同步修改 `models.default.max_lora_rank`。
- **`train_unembed`**：PyTRIO 默认 `True`（输出层也参与训练）。vLLM 0.33 只为纯文本 Qwen3.5 类声明了 `lm_head` LoRA，本 Lab 加载的 Qwen3.5 多模态类没有声明，采样端可能无法加载带 `lm_head` 的 adapter，所以本地默认忽略它。设置 `trainer.allow_unembed: true` 后会以 LoRA 形式训练 `lm_head`（训练端的 logprob 计算已支持），请先确认你的 vLLM 能加载这样的 adapter。这是本地结果与原文可能不同的原因之一。
- **LoRA 缩放**：`lora_alpha` 固定为 32（`trainer.lora_alpha` 可改），缩放系数 `alpha / rank` 随 rank 变化：rank 32 为 1，rank 64（OPSD）为 0.5。PyTRIO 服务端的取值未公开，跨 rank 比较等效学习率时注意这一点。
- **采样 logprob 口径**：默认返回原始模型分布的 logprob（与训练端一致，ratio 在 on-policy 时≈1）。如果某个实验需要“经过 temperature / top-k 处理后”的 logprob，在配置里设置 `models.default.logprobs_mode: processed_logprobs`；这会让 OPSD 这类在 temperature≠1 下计算 reverse KL 的数值与默认口径不同。
- 保存时的 `ttl_seconds` 被忽略（本地 checkpoint 不会过期，需要自己清理 `runs/` 目录）。
- 兼容接口：`SamplingClient.task_id`（等于 `sampling_run_cuid`）、`Datum.validate_loss_inputs(loss_fn)`。
- PyTRIO 服务端的具体数值实现（kernel、精度、归一化细节）不公开，本地实现按文档语义复现，数值不会逐位一致。

## 测试

```bash
uv sync --extra dev
uv run pytest tests -q        # CPU 即可运行：构造随机初始化的微型 Qwen3 / Qwen3.5-VL 模型
```

测试覆盖：分块 logprob 与完整 logits 一致、四种策略 loss 的数值、custom loss 梯度与直接求导一致、micro-batch 不变性、AdamW 更新与 checkpoint 往返、导出 adapter 可被 PEFT 加载、Qwen3.5 线性注意力 + 图片输入的前向/反向，以及通过 `ServiceClient` 走子进程的端到端流程（采样与训练 logprob 一致、训练后目标序列概率上升、异步接口、top-k prompt logprobs）。vLLM 后端需要 GPU，不在 CPU 测试范围内。
