// 站点地址：GitHub Pages 项目页 https://hxusz11.github.io/Agentic_RL/
export const siteOrigin = 'https://hxusz11.github.io'
export const siteBase = '/Agentic_RL/'
export const siteUrl = siteOrigin + siteBase
// 原教程作者；本地版改写见各页说明
export const author = { name: 'KMnO4-zx（原作） · 本地 GPU 版', url: 'https://github.com/datawhalechina/agentic-rl-lab' }
export const routePath = (relativePath) => relativePath.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '.html')
export const pageUrl = (relativePath) => siteUrl + routePath(relativePath)
export const localizedPath = (path) => path
