---
title: 关于 Agentic RL 本地实战
description: Datawhale 开源 Agentic RL 教程的本地 GPU 改写版：同样的 16 篇算法笔记，代码通过 localtrio 在自己的 4×A6000 上训练与采样。
---

# 关于本地版

这是 Datawhale [开源 Agentic RL 教程](https://github.com/datawhalechina/agentic-rl-lab)的**本地 GPU 版本**。原教程用 [PyTRIO](https://pytrio.com) 把模型训练和采样交给云端服务，本地只写数据、奖励和实验循环；这里保留全部算法讲解和实验代码，把“云端那一半”换成了运行在自己显卡上的 [localtrio](./localtrio)。

目标硬件是 **4 张 NVIDIA RTX A6000（48GB）**，也提供单卡和 CPU 调试配置。

## 和原教程有什么不同

| | 原教程 | 本地版 |
| --- | --- | --- |
| 训练 / 采样 | PyTRIO 云端（`import pytrio as trio`） | localtrio（`import localtrio as trio`），HF transformers + PEFT 训练，vLLM 采样 |
| 账号与费用 | 需要 PyTRIO 账号与额度 | 不需要账号，消耗的是自己的 GPU 时间 |
| checkpoint | `trio://...` 远程路径 | 本地目录 `~/.cache/localtrio/runs/<run>/...`，标准 PEFT 格式 |
| GPU 规划 | 不需要关心 | 需要：默认 GPU0 训练、GPU1-3 采样，见[显存与 GPU 布局](./hardware) |
| 实验代码 | — | 只替换 import 和少量云端相关代码，算法与超参保持原样 |
| 实验结果 | 原作者在 PyTRIO 上的记录 | 原文结果图保留并注明来源；本地复现的数值需要以自己的运行为准 |

localtrio 的接口与 PyTRIO 0.2.9 同名同参：`ServiceClient`、`create_lora_training_client`、`forward_backward`、`forward_backward_custom`、`optim_step`、`save_weights_and_get_sampling_client`、`sample`、`compute_logprobs`……原教程里关于“训练循环怎么写”的讲解都可以直接对照本地代码阅读。

## 每篇实验里有什么

一篇论文从提出想法到真正跑起来，中间藏着不少细节：采样怎样组织，奖励怎样计算，哪些 token 参与更新，评测又和训练有什么区别。笔记沿着这些问题展开：

1. **论文与直觉**：方法在解决什么问题，核心变量是什么。
2. **代码与训练**：数据、reward、loss、训练循环怎样连接起来。
3. **本地运行**：用哪种 GPU 布局、显存大概多少、命令是什么。
4. **结果与边界**：原文的观察，以及当前实现和原方法之间的差异。

不同章节的分数不能直接放在同一张榜单里比较；算法级实现也有各自的简化，请结合每篇文末的限制阅读。

## 实现的可信度

- localtrio 的训练数学（五种内置 loss、custom loss 梯度、micro-batch 梯度累积、checkpoint 保存/恢复、采样端与训练端 logprob 一致性、Qwen3.5 图文输入）有一组 CPU 上用微型模型跑的测试，见仓库 `tests/`。
- vLLM 采样后端按 vLLM 0.33 的接口编写，需要在有 GPU 的机器上运行；各章“本地资源估算”是按模型尺寸推算的数字，**不是实测值**。第一次跑某章时建议先用快速启动里的小规模参数。

## 致谢与许可

- 原教程作者：[KMnO4-zx（宋志学）](https://github.com/KMnO4-zx) 与 Datawhale；内容贡献者 [MRLE](https://github.com/mrle0429)（Tournament-GRPO）。
- 本仓库沿用原项目的 Apache-2.0 许可，改动见各文件与提交记录。
- 实验记录工具：[SwanLab](https://swanlab.cn/)（可用 `--swanlab-mode disabled` 关闭）。

## 从哪里开始

- 第一次接触：先看[学习路线](./learning-path)，再按[安装与第一个实验](./quickstart)跑通 GRPO。
- 想先搞清楚本地是怎么跑的：读 [localtrio：本地版 Train API](./localtrio) 和[显存与 GPU 布局](./hardware)。
- 已经有训练经验：在[实验手册](/experiments/)中按研究方向选一篇。

问题与复现记录欢迎提交到 [GitHub Issues](https://github.com/HXuSz11/Agentic_RL/issues)，附上模型名称、运行命令、`python -m localtrio info` 的输出和必要日志（引擎日志在 `~/.cache/localtrio/logs/`）。
