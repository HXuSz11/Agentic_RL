import { metadata } from './metadata.mjs'
// 唯一的章节注册表。正文仍保存在 experiments/<id>/ 原目录里。
export const groups = [
  { id: 'foundations', label: '基础与优化', note: '理解一次策略更新', number: '01' },
  { id: 'distillation', label: '蒸馏与自学习', note: '把反馈变成学习信号', number: '02' },
  { id: 'agents', label: '工具与环境', note: '让模型走进交互环境', number: '03' },
  { id: 'frontiers', label: '多模态与前沿', note: '探索更长、更复杂的任务', number: '04' },
]

export const chapters = [
  { id: 'loss-functions', number: '01', title: 'Loss Functions', subtitle: '从直觉理解强化学习损失', group: 'foundations', source: 'experiments/loss-functions/readme.md', code: false, description: '从 importance sampling 到 PPO、CISPO，看清每一次策略更新到底在优化什么，以及它们在 localtrio 训练端怎样逐 token 计算。', tags: ['入门', '目标函数'] },
  { id: 'grpo', number: '02', title: 'GRPO', subtitle: '从组内比较开始', group: 'foundations', source: 'experiments/grpo/readme.md', start: 'experiments/grpo/start.md', description: '在 GSM8K 上走通采样、奖励与更新：一张卡训练 LoRA，其余卡用 vLLM 组采样，亲手比较三种 loss 的训练行为。', tags: ['GSM8K', '组相对优势'] },
  { id: 'vision-grpo', number: '03', title: 'Vision GRPO', subtitle: '让强化学习看见图片', group: 'frontiers', source: 'experiments/vision-grpo/readme.md', start: 'experiments/vision-grpo/start.md', description: '在 GeoQA 上接入图像输入，把 Qwen3.5 的图文推理、vLLM 多模态采样与可验证奖励串起来。', tags: ['视觉语言模型', 'GeoQA'] },
  { id: 'pytrio-jev', number: '04', title: 'Jev 决策模型', subtitle: '训练自己的分类决策模型', group: 'frontiers', source: 'experiments/pytrio-jev/readme.md', start: 'experiments/pytrio-jev/start.md', description: '用 2160 条数据在本地 LoRA 微调 Qwen3.5-4B，走通 marker 监督、受限 softmax 推理、温度校准与五项基准评测。', tags: ['分类决策', '概率校准'] },
  { id: 'general-opd', number: '05', title: 'General OPD', subtitle: '让学生在自己的轨迹上学习', group: 'distillation', source: 'experiments/general-opd/readme.md', description: '用 DeepMath-103K 串起学生采样、27B 教师打分与 reverse KL；卡不够时可以换成 9B 教师，双卡就能跑。', tags: ['蒸馏', 'DeepMath'] },
  { id: 'medical-opd', number: '06', title: 'Medical OPD', subtitle: '专业能力与通用能力的平衡', group: 'distillation', source: 'experiments/medical-opd/readme.md', start: 'experiments/medical-opd/start.md', description: '从本地 Medical SFT 出发，实验 SAR-OPD 与 IDT-OPD，用本地 vLLM 服务评测医疗与通用能力的变化。', tags: ['医疗任务', '多教师'] },
  { id: 'dapo', number: '07', title: 'DAPO', subtitle: '拆开训练稳定性的四个改进', group: 'foundations', source: 'experiments/dapo/readme.md', start: 'experiments/dapo/start.md', description: '拆解四项核心设计，分析 Dynamic Sampling 在本地 vLLM 采样下的收益与时间成本。', tags: ['动态采样', '训练稳定性'] },
  { id: 'gspo', number: '08', title: 'GSPO', subtitle: '把优化单位提升到序列', group: 'foundations', source: 'experiments/gspo/readme.md', start: 'experiments/gspo/start.md', description: '用 forward_backward_custom 实现 sequence-level 重要性比率与裁剪，厘清核心 loss 和完整训练的边界。', tags: ['序列级优化', '自定义 Loss'] },
  { id: 'opsd', number: '09', title: 'OPSD', subtitle: '从自己生成的轨迹中蒸馏', group: 'distillation', source: 'experiments/opsd/readme.md', start: 'experiments/opsd/start.md', description: '固定 step-0 Teacher（同一基模、不带 LoRA），在学生自采样轨迹上构造蒸馏信号，教师与学生共用一组 vLLM。', tags: ['自蒸馏', '数学推理'] },
  { id: 'agentopsd', number: '10', title: 'AgentOPSD', subtitle: '把信用分配到每一轮交互', group: 'frontiers', source: 'experiments/agentopsd/readme.md', start: 'experiments/agentopsd/start.md', description: '让带 Skill 的 Self-Teacher 给出 turn-level 信号，在本地 ALFWorld 环境里分析多轮 Agent 的学习过程。', tags: ['信用分配', 'Self-Teacher'] },
  { id: 'search-r1', number: '11', title: 'Search-R1', subtitle: '学会在思考中搜索', group: 'agents', source: 'experiments/search-r1/readme.md', start: 'experiments/search-r1/start.md', description: '用 DeepSeek Search API 或 Wikipedia 获取证据，在本地 GPU 上复现多轮搜索训练，并比较 Base 与 checkpoint。', tags: ['在线搜索', '多轮交互'] },
  { id: 'retool', number: '12', title: 'ReTool', subtitle: '把 Python 变成推理工具', group: 'agents', source: 'experiments/retool/readme.md', start: 'experiments/retool/start.md', description: '连接本地代码沙箱，复现思考与代码执行交织进行的 Agentic RL，工具 token 只做上下文不参与 loss。', tags: ['代码沙箱', '工具调用'] },
  { id: 'alfworld', number: '13', title: 'ALFWorld', subtitle: '训练一个会做家务的 Agent', group: 'agents', source: 'experiments/alfworld/readme.md', start: 'experiments/alfworld/start.md', description: '在本地 TextWorld 环境里处理 12K 长轨迹、环境反馈与家务任务，环境跑 CPU、模型跑 GPU。', tags: ['具身环境', '长轨迹'] },
  { id: 'tempo', number: '14', title: 'TEMPO', subtitle: '用小段轨迹学习长程任务', group: 'frontiers', source: 'experiments/tempo/readme.md', description: '在 ALFWorld 中探索 macro-step 优化与生成式 critic，记录算法级复现的边界。', tags: ['Macro-step', '生成式 Critic'] },
  { id: 'spec-o3', number: '15', title: 'Spec-o3', subtitle: '用强化学习寻找天上的星星', group: 'frontiers', source: 'experiments/spec-o3/readme.md', start: 'experiments/spec-o3/start.md', description: '从 SFT 到 GRPO，让模型边看光谱、边思考、边调用工具完成天体候选审核；9B 学生用双卡训练布局。', tags: ['光谱分析', '多模态工具'] },
  { id: 'tournament-grpo', number: '16', title: 'Tournament-GRPO', subtitle: '让研究报告打一场淘汰赛', group: 'agents', source: 'experiments/tournament-grpo/readme.md', start: 'experiments/tournament-grpo/start.md', description: '让同题研究报告两两比赛，用淘汰赛胜场构造 GRPO 奖励；策略在本地训练，裁判仍走外部 API。', tags: ['淘汰赛奖励', 'Deep Research'] },
]

export const repoUrl = 'https://github.com/HXuSz11/Agentic_RL'
// GitHub 的 HEAD 会解析到仓库默认分支，链接不依赖具体分支名
export const repoRef = 'HEAD'
export const upstreamUrl = 'https://github.com/datawhalechina/agentic-rl-lab'
export const chapterLink = (chapter) => `/experiments/${chapter.id}/`
export const localizedChapters = () => chapters.map(c => ({ ...c, ...metadata[c.id] }))
export const localizedGroups = () => groups

// 网站自己渲染标题区，正文里的封面图在页面中隐藏（算法图与结果图全部保留）。
export const articleCovers = {
  grpo: './images/GRPO.png',
  'general-opd': './images/OPD-deepmath.png',
  'medical-opd': './images/OPD-head.png',
  'search-r1': './images/封面.png',
  opsd: './images/head.png',
  retool: './images/封面.png',
  dapo: './images/head.png',
  gspo: './images/封面.png',
  alfworld: './images/封面.png',
  'vision-grpo': './images/vision-grpo.png',
  tempo: './images/封面.png',
  agentopsd: './images/封面.png',
  'spec-o3': './images/封面.png',
  'pytrio-jev': './images/head.png',
  'tournament-grpo': './images/封面.png',
}
