---
title: 参与贡献：本地 GPU 版的复现记录与改进
description: 如何为 Agentic RL Lab 本地 GPU 版提交复现记录、修复章节、改进 localtrio，以及章节结构与质量要求。
---

# 参与贡献

本仓库是 [Datawhale Agentic RL Lab](https://github.com/datawhalechina/agentic-rl-lab) 的本地 GPU 改写版。最有价值的贡献是**在真实显卡上跑出来的复现记录**：原文结果来自 PyTRIO 云端，本地版的每一章都需要有人在自己的机器上验证。

新增算法章节、修改原文讲解，建议同时向[上游仓库](https://github.com/datawhalechina/agentic-rl-lab/blob/main/CONTRIBUTING.md)提交，让两边的内容保持一致。

## 贡献方式

| 类型 | 流程 |
| --- | --- |
| 本地复现记录（GPU 型号、布局、耗时、显存、指标） | 开 Issue 或直接提 PR，补充到对应章节的「本地资源估算 / 本地结果」小节 |
| 修复章节在本地跑不通的问题 | 提 PR，描述里附 `python -m localtrio info` 输出、运行命令和引擎日志 |
| 改进 localtrio（新后端、性能、接口缺口） | 提 PR，并补充 `tests/` 里的测试 |
| 错别字、文档改进 | 直接提 PR |

## 章节结构约定

```text
experiments/algorithm-name/
├── readme.md          # 章节正文
├── start.md           # 快速启动：环境 → GPU 布局 → 数据 → 训练 → 评测
├── data.py / prepare_data.py
├── reward.py / rollout.py
├── train.py / eval.py / analysis.py
├── images/
└── .env.example       # 只放外部 API（搜索、裁判等）的 key 模板
```

- 所有脚本从**仓库根目录**运行：`uv run python experiments/algorithm-name/train.py ...`。
- 代码使用 `import localtrio as trio`，接口与 PyTRIO 保持同名同参，便于和上游对照。
- `start.md` 必须让新读者按顺序复制命令就能跑通，并写清楚用哪个 GPU 布局（默认或 `configs/` 里的预设）。

## 质量要求

1. **真实跑通**：PR 里注明 GPU 型号与数量、布局、单次训练耗时，并附训练记录（SwanLab 链接或日志）。
2. **估算与实测分开写**：没有实测的显存和耗时必须标注“估算”。
3. **有效果证据**：在固定评测集上给出 Base vs checkpoint 的对比，并写清评测设置。
4. **诚实边界**：说明单次小规模训练能支持和不能支持的结论；本地数值与原文不一致时如实记录。

## 工程规范

- 依赖用 [uv](https://docs.astral.sh/uv/) 管理（`pyproject.toml`），Python `>=3.12`。
- **不要提交**：`.env` 与密钥、数据集、`swanlog/`、评测输出、`~/.cache/localtrio` 下的 checkpoint。
- 改动 `localtrio/` 后运行 `uv run pytest tests -q`；改动网站后运行 `npm run docs:build && npm run docs:check`。
- 图片放在章节目录的 `images/` 下，正文用相对路径引用。

## 交流

- 本仓库：[GitHub Issues](https://github.com/HXuSz11/Agentic_RL/issues)
- 原教程社区：Datawhale [Agentic RL Lab](https://github.com/datawhalechina/agentic-rl-lab)（微信群见上游 [Issue #13](https://github.com/datawhalechina/agentic-rl-lab/issues/13)）
