// 站点地址：GitHub Pages 项目页 https://hxusz11.github.io/Agentic_RL/
export const siteOrigin = 'https://hxusz11.github.io'
export const siteBase = '/Agentic_RL/'
export const siteUrl = siteOrigin + siteBase
// 站点名称：导航栏、页面标题、分享卡片共用，改名只需改这里（以及 public/social-card.svg）
export const siteName = 'Agentic RL 本地实战'
export const siteDescription = '在自己的 4 张 RTX A6000 上跑通 16 个 LLM 与 Agent 强化学习实验：localtrio 用 transformers + PEFT 训练 LoRA、vLLM 采样，接口与 PyTRIO 同名同参。'
// 原教程作者；本地版改写见各页说明
export const author = { name: 'KMnO4-zx（原作） · 本地 GPU 版', url: 'https://github.com/datawhalechina/agentic-rl-lab' }
export const routePath = (relativePath) => relativePath.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '.html')
export const pageUrl = (relativePath) => siteUrl + routePath(relativePath)
export const localizedPath = (path) => path
