# 第 0 篇： 强化学习基础-损失函数篇

![三种 RL loss 的卡通解释](./images/rl_losses_cartoon_cover.png)

<div align="center">
  <a href="https://www.zhihu.com/people/feng-qi-xia-pian" target="_blank"><img alt="Zhihu" src="https://img.shields.io/badge/Zhihu-知乎-4362f6"></a>
  <a href="https://www.xiaohongshu.com/user/profile/63c2055e000000002502c58c" target="_blank"><img alt="Rednote" src="https://img.shields.io/badge/Rednote-小红书-e93c49"></a>
  <a href="https://github.com/datawhalechina/agentic-rl-lab"><img alt="visitors" src="https://komarev.com/ghpvc/?username=KMnO4-zx-agentic-rl-lab-loss-function&amp;label=visitors&amp;color=1283c3&amp;style=flat"></a>
</div>

> **本地 GPU 版说明**
>
> 本章是 Datawhale [Agentic RL Lab](https://github.com/datawhalechina/agentic-rl-lab) 原教程的本地 GPU 版本。原系列代码基于 PyTRIO 云端训练服务；本仓库改用 [localtrio](../../docs/content/guide/localtrio.md)（接口与 PyTRIO 同名同参，训练用 transformers + PEFT LoRA，采样用 vLLM），在 4×RTX A6000（48GB）上运行。本章没有训练代码，第 5 节补充了这三个 loss 在 localtrio 训练端的逐 token 实现。感谢原作者的开源与分享。

- 左边：Importance Sampling。模型走过一串 token，有的选择被奖励，有的选择被惩罚；loss 的工作就是把好选择的概率推高，把坏选择的概率压低。
- 中间：PPO。模型不能因为一次奖励就猛冲，clip 像限速门，防止一步改太大。
- 右边：CISPO。clip 后的 ratio 像一个固定砝码，只调节梯度大小；真正被优化的是当前 token 的 `logprob`。

接下来，我将开启我的强化学习算法分享之路，因为最近在研究强化学习算法，感觉很有趣，似乎找到了最原始的快乐哈哈哈。之前想研究 agent-rl，发现如果用 verl 的话，需要训推一体，然后至少一个 4 卡或者 8 卡的节点，大部分精力都花在照顾 infra 上。

原作者当时用 PyTRIO 云端服务绕开了这些；本地版把这一层换成了 localtrio：实验代码还是同样的 `TrainingClient` / `SamplingClient` 调用，背后是自己机器上的 4 张 A6000——GPU0 跑 LoRA 训练，GPU1～3 各跑一个 vLLM 采样副本，权重同步、checkpoint 都在本地目录里完成。这样就可以潜心研究或者复现当前论文的强化学习算法。

所以这个系列会复现 10 篇论文的强化学习算法，代码全部开源，并且都能在本地 GPU 上运行。复现的算法包括但不限于 GRPO、GSPO、DAPO、OPD 等各种变体的 OPD！

今天是分享的第零篇，强化学习基础-损失函数篇。

很多人第一次看强化学习 loss，会有一个很自然的困惑：监督学习的 loss 我能理解，答案对不上就惩罚；可强化学习里，模型自己生成了一段话，后面才拿到一个 reward，那 loss 到底在优化什么？

我的理解是，先不要把它想成复杂公式。强化学习里的很多 loss，本质是在做一件事：

> 如果某些 token 后来被证明对结果有帮助，就让模型下次更容易生成它们；如果某些 token 拖了后腿，就让模型下次少生成它们。

这件事拆开之后，就会出现三个变量：

- `advantage`：这一步是好是坏。
- `sampling_logprobs`：生成答案时，采样模型当时有多喜欢这个 token。
- `target_logprobs`：训练时，当前模型现在有多喜欢这个 token。

用这三个东西，就能理解 Importance Sampling、PPO 和 CISPO 的区别。

### 1. Importance Sampling

Importance Sampling 最直接。它问的是：当前模型和采样模型对同一个 token 的概率比例是多少？

```math
\mathcal{L}_{\text{IS}}(\theta) = \mathbb{E}_{x\sim q}\Bigl[\frac{p_\theta(x)}{q(x)}A(x)\Bigr]
```

实现时再把概率比例写成 `exp(target_logprobs - sampling_logprobs)`，并因为优化器最小化 loss，所以代码里加负号：

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
loss = -(prob_ratio * advantages).sum()
```

如果 advantage 是正的，说明这个 token 有用，训练会推高它的概率。

如果 advantage 是负的，说明这个 token 没用，训练会压低它的概率。

这里的负号只是因为优化器默认最小化 loss。我们真正想最大化的是 `ratio * advantage` 这个目标。

### 2. PPO

PPO 解决的是另一个问题：如果 ratio 太大，模型可能一次更新就走太远。

所以 PPO 给 ratio 加了一个 clip：

```math
\mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_{x \sim q}\left[\min\left(\frac{p_\theta(x)}{q(x)} \cdot A(x), \text{clip}\left(\frac{p_\theta(x)}{q(x)}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right) \cdot A(x)\right)\right]
```

比如一个 token 的 advantage 很高，ratio 已经到了 `1.8`，说明当前模型已经比采样时更喜欢它很多。PPO 不会继续按 `1.8` 的力度推，而是把它限制到比如 `1.2`。这就是 PPO 的核心直觉：可以学习，但别一步冲太远。

### 3. CISPO

CISPO 和 PPO 的区别更细一点。

PPO 是 clip objective。

CISPO 是 clip ratio，然后把这个 ratio 当成固定系数，去乘 `logprob * advantage`。

```math
\mathcal{L}_{\text{CISPO}}(\theta) = -\mathbb{E}_{x \sim q}\left[\textbf{sg}\left( \text{clip}\left(\frac{p_\theta(x)}{q(x)}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\right) \right) \cdot \log p_\theta(x) \cdot A(x)\right]
```

这里最关键的是 `detach`。它表示 clipped ratio 不再参与梯度回传，只作为一个固定权重。真正被优化的是 `target_logprob`。

所以 CISPO 可以理解成：给每个 token 的学习力度加一个上限，但不直接把某些 token 的梯度剪没。

### 4. 一句话比较

| Loss                | 人话比喻             | 核心动作                                      | 最容易误解的点                                           |
| ------------------- | -------------------- | --------------------------------------------- | -------------------------------------------------------- |
| Importance Sampling | 按功劳发奖金         | `ratio * advantage`                           | loss 可能为负，这是因为代码在最小化负目标                |
| PPO                 | 奖金有预算上限       | `min(unclipped, clipped)`                     | `min` 不是随便取小，而是在做保守更新                     |
| CISPO               | 给学习力度挂固定砝码 | `detach(clipped_ratio) * logprob * advantage` | `.detach()` 后 ratio 不走梯度，主要是 `logprob` 在被优化 |

更短的总结：

```text
IS:    有用就提高概率，没用就降低概率；ratio 负责修正采样偏差。
PPO:   还是这么做，但 ratio 不能太离谱，防止策略更新过猛。
CISPO: ratio 也会被 clip，但 clip 后只当固定权重，真正优化 logprob。
```

### 5. 在 localtrio 里，这三个 loss 怎么逐 token 计算

后面的大多数章节不会自己写 `exp(...)` 和 `clip(...)`，而是把 rollout 打包成 `Datum`，交给训练端的内置 loss；少数章节（GRPO 的 CISPO 版本、DAPO、GSPO、Tournament-GRPO）则通过 `forward_backward_custom` 在自定义 loss 里自己写 ratio 和 clip（见本节最后一条）。内置 loss 的用法如下：

```python
import localtrio as trio

fwd_bwd = training_client.forward_backward(
    datums,
    loss_fn="ppo",  # 或 "importance_sampling" / "cispo"
    loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
)
training_client.optim_step(trio.AdamParams(learning_rate=4e-5)).result()
print(fwd_bwd.result().metrics)  # loss_sum / loss_mean / token_count
```

每个 `Datum` 的 `loss_fn_inputs` 里放三样东西，和前面的三个变量一一对应：

| 字段 | 对应变量 | 说明 |
| --- | --- | --- |
| `target_tokens` | 被打分的 token | 训练端用当前 LoRA 重新前向，得到 `target_logprobs` |
| `logprobs` | `sampling_logprobs` | 采样时 vLLM 返回的 logprob（原始模型分布，不受 temperature / top-p 影响） |
| `advantages` | `advantage` | 每个 token 的好坏 |

训练端（[`localtrio/losses.py`](../../localtrio/losses.py)）的逐 token 公式，记 `lp = target_logprobs`、`old = sampling_logprobs`、`ratio = exp(lp - old)`：

| `loss_fn` | 逐 token loss | 默认 clip 区间 |
| --- | --- | --- |
| `importance_sampling` | `-ratio * A` | 无 |
| `ppo` | `-min(ratio * A, clip(ratio, lo, hi) * A)` | `[0.8, 1.2]` |
| `cispo` | `-detach(clip(ratio, lo, hi)) * lp * A` | `[0.0, 4.0]` |

有三个细节和上面的示意代码不完全一样，读后面的章节时值得记住：

- **clip 用的是 ratio 的绝对上下界**。`clip_low_threshold` / `clip_high_threshold` 对应公式里的 `1 - ε_low` / `1 + ε_high`，所以 `ε = 0.2` 要写成 `0.8` 和 `1.2`。
- **不是 `.sum()`，而是按整个 logical batch 的 active token 数归一化**。一次 `forward_backward` 里的所有样本共享同一个分母，即使训练端为了省显存把它们拆成多个 micro-batch，梯度也和整批一起算相同。`target_tokens == -100` 的位置不参与计算，也不计入分母；advantage 为 0 的位置（比如 prompt）loss 为 0，但只要 target 不是 -100 就仍计入分母。分母口径可以用配置项 `trainer.loss_normalization` 改成 `samples` 或 `sum`。
- **自定义 loss 不再额外归一化**。localtrio 默认（`trainer.custom_loss_normalization: none`）下，`forward_backward_custom` 回传的梯度严格等于你自己写的 loss 的梯度，所以 [GRPO 一章](../grpo/readme.md) 里用 custom loss 写的 CISPO 保持了 `.sum()` 的口径。原版 PyTRIO 0.2.9 的 SDK 则把 custom loss 换成 `weights = -dL/dlogprob` 的 `cross_entropy` 再提交，并把梯度非 0 的 token 数作为分母信息发给服务端（SDK 注释只保证梯度方向相同），实际尺度可能多除以这个 token 数；想按这个口径训练，设 `trainer.custom_loss_normalization: same_as_builtin`，分母就沿用上一条的 `trainer.loss_normalization`。整体尺度对 Adam 影响不大，但开了 `grad_clip_norm` 时就会有区别。

更完整的 API 对照见 [localtrio 指南](../../docs/content/guide/localtrio.md)。下一篇 [GRPO](../grpo/readme.md) 就会用同一套 rollout 数据，把这三个 loss 放在一起跑。
