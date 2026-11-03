// 页面元数据（搜索标题、快速启动简介），与正文分开维护。
export const metadata = {
  'loss-functions': {
    seoTitle: '强化学习损失函数：Importance Sampling、PPO 与 CISPO',
  },
  grpo: {
    seoTitle: 'GRPO 本地实战：在 4×A6000 上用 GSM8K 训练 Qwen3.5',
    runDescription: '安装依赖与 vLLM，使用默认 4 卡布局运行 GSM8K 上的 GRPO 训练，了解采样参数、三种 loss 与日志指标。',
  },
  'vision-grpo': {
    seoTitle: 'Vision GRPO 本地多模态强化学习：GeoQA 图文推理',
    runDescription: '准备 GeoQA 图片数据，配置图文输入，在本地 GPU 上运行 Vision GRPO 训练与几何问答评测。',
  },
  'pytrio-jev': {
    seoTitle: 'Jev 决策模型本地实战：LoRA 训练、分类推理与温度校准',
    runDescription: '准备 Typed Decisions、ToolACE 与 Jevbench 数据，在本地运行 Qwen3.5-4B LoRA 训练、决策推理、温度校准和五项基准评测。',
  },
  'general-opd': {
    seoTitle: 'On-Policy Distillation 本地实战：4B 学生、27B 教师与 Reverse KL',
  },
  'medical-opd': {
    seoTitle: '医疗 On-Policy Distillation：SAR-OPD、IDT-OPD 与多教师（本地 GPU 版）',
    runDescription: '准备医疗蒸馏数据，在本地运行 Medical SFT、SAR-OPD 和 IDT-OPD，并用本地 vLLM 服务评测。',
  },
  dapo: {
    seoTitle: 'DAPO 原理与本地训练：动态采样、Clip-Higher 与四项改进',
    runDescription: '在本地 GPU 上启动 DAPO 训练，配置 Dynamic Sampling 与采样参数，查看四项改进相关的训练日志。',
  },
  gspo: {
    seoTitle: 'GSPO 原理与核心 Loss 实现：序列级重要性采样（本地 GPU 版）',
    runDescription: '运行 GSPO 核心 loss 实验，设置策略更新与采样参数，并用本地 checkpoint 做评测。',
  },
  opsd: {
    seoTitle: 'OPSD 自蒸馏本地实战：固定教师与学生自采样轨迹',
    runDescription: '运行 OPSD 数学推理自蒸馏，配置固定 step-0 Teacher、采样预算与训练指标记录。',
  },
  agentopsd: {
    seoTitle: 'AgentOPSD 本地实战：Self-Teacher 与多轮 Agent 信用分配',
    runDescription: '准备 ALFWorld 与 Skill，在本地运行 AgentOPSD 训练、checkpoint 评测和配对分析。',
  },
  'search-r1': {
    seoTitle: 'Search-R1 本地复现：训练会多轮搜索的 LLM Agent',
    runDescription: '配置 DeepSeek Search API 或免密钥 Wikipedia，在本地 GPU 上运行 Search-R1 多轮搜索训练与评测。',
  },
  retool: {
    seoTitle: 'ReTool 本地复现：Python 工具调用与 Agent 强化学习',
    runDescription: '连接 ReTool 本地 Python 沙箱，启动思考与代码执行交织的训练，检查工具输出与 rollout 指标。',
  },
  alfworld: {
    seoTitle: 'ALFWorld Agent 强化学习本地实战：TextWorld 环境与长轨迹训练',
    runDescription: '安装 ALFWorld 和 TextWorld 依赖、准备环境数据，在本地运行家务 Agent 训练与评测。',
  },
  tempo: {
    seoTitle: 'TEMPO 算法级复现：Macro-step 优化与生成式 Critic（本地 GPU 版）',
  },
  'spec-o3': {
    seoTitle: 'Spec-o3 多模态 Agent 本地实战：SFT、GRPO 与天体光谱审核',
    runDescription: '下载 Spec-o3 数据，在本地运行 SFT、GRPO、开发集评测与绘图，4B 与 9B 学生各有 GPU 布局。',
  },
  'tournament-grpo': {
    author: { name: 'MRLE（原作） · 本地 GPU 版', url: 'https://github.com/mrle0429' },
    seoTitle: 'Tournament-GRPO 原理与本地复现：研究报告淘汰赛奖励',
    runDescription: '准备研究数据，配置 DeepSeek 搜索、Jina Reader 与裁判，在本地 GPU 上运行 Tournament-GRPO 训练和评测。',
  },
}
