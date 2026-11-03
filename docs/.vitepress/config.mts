import { defineConfig } from 'vitepress'
import { slug } from 'github-slugger'
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { localizedChapters, repoUrl, chapterLink } from '../chapters.mjs'
import { siteUrl, siteBase } from '../site.mjs'
import { addSeo, pageDates } from './seo.mjs'
import { imageSize, imageHints } from './images.mjs'
import { rewriteRawMarkdown } from './raw-markdown.mjs'

// Serve every page's raw markdown next to its HTML so readers can copy it
// or hand the URL to an LLM (mirrors the page path, e.g. /experiments/grpo/index.md).
// A UTF-8 BOM keeps browsers from misdetecting the encoding when the
// static host serves .md without a charset header.
function markdownFiles(srcDir: string) {
  const files: string[] = []
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry)
      if (statSync(full).isDirectory()) { if (entry !== 'public') walk(full); continue }
      if (entry.endsWith('.md')) files.push(relative(srcDir, full).replaceAll('\\', '/'))
    }
  }
  walk(srcDir)
  return files
}

function copyRawMarkdown(srcDir: string, outDir: string) {
  const pages = markdownFiles(srcDir)
  for (const page of pages) {
    const dest = join(outDir, page)
    mkdirSync(dirname(dest), { recursive: true })
    writeFileSync(dest, '\uFEFF' + rewriteRawMarkdown(readFileSync(join(srcDir, page), 'utf8'), page, pages))
  }
}

function navigation() {
  const chapters = localizedChapters()
  return {
    nav: [
      { text: '首页', link: '/' },
      { text: '实验手册', link: '/experiments/', activeMatch: '/experiments/' },
      { text: '本地训练', link: '/guide/localtrio.html', activeMatch: '/guide/(localtrio|hardware|quickstart)' },
      { text: '学习路线', link: '/guide/learning-path.html' },
      { text: '关于', link: '/guide/introduction.html' },
    ],
    sidebar: [
      { text: '从这里开始', items: [
        { text: '关于本地版', link: '/guide/introduction.html' },
        { text: '学习路线', link: '/guide/learning-path.html' },
        { text: '安装与第一个实验', link: '/guide/quickstart.html' },
        { text: 'GRPO、DAPO 与 GSPO', link: '/guide/grpo-dapo-gspo.html' },
        { text: '参与贡献', link: '/guide/contributing.html' },
        { text: '全部实验', link: '/experiments/' },
      ] },
      { text: '本地训练（4×A6000）', items: [
        { text: 'localtrio：本地版 Train API', link: '/guide/localtrio.html' },
        { text: '显存与 GPU 布局规划', link: '/guide/hardware.html' },
      ] },
      {
        text: '实验阅读顺序', collapsed: false,
        items: chapters.map(c => ({
          text: `${c.number}  ${c.title}`, link: chapterLink(c),
          ...(c.start ? { collapsed: true, items: [{ text: '快速启动', link: `${chapterLink(c)}run.html` }] } : {}),
        })),
      },
    ],
    outline: { label: '本页目录', level: [2, 3] as [number, number] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换至浅色',
    darkModeSwitchTitle: '切换至深色',
    sidebarMenuLabel: '章节目录', returnToTopLabel: '回到顶部',
    skipToContentLabel: '跳到正文',
    editLink: {
      pattern: ({ frontmatter, relativePath }) => {
        // VitePress 会把这个函数序列化到浏览器里，必须自包含。
        const repository = 'https://github.com/HXuSz11/Agentic_RL'
        return frontmatter.source ? `${repository}/edit/HEAD/${frontmatter.source}` : `${repository}/edit/HEAD/docs/content/${relativePath}`
      },
      text: '在 GitHub 上编辑此页',
    },
    notFound: { title: '这页笔记还没找到', quote: '换一个入口，继续探索强化学习。', linkLabel: '返回首页', linkText: '返回首页' },
    footer: { message: '原教程：Datawhale Agentic RL Lab（Apache-2.0）· 本地 GPU 版：localtrio + transformers + PEFT + vLLM', copyright: '开放代码，记录实验，也记录边界。' },
  }
}

export default defineConfig({
  title: 'Agentic RL Lab · Local',
  description: 'Agentic RL Lab 的本地 GPU 版：用 localtrio（transformers + PEFT + vLLM）在 4×A6000 上复现 GRPO、DAPO、GSPO、OPD、Search-R1、ReTool、ALFWorld 与多模态 Agent。',
  lang: 'zh-CN', base: siteBase, srcDir: './content', cleanUrls: false, lastUpdated: false,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${siteBase}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#fdfbf9' }],
  ],
  transformPageData: addSeo,
  buildEnd: (siteConfig) => copyRawMarkdown(siteConfig.srcDir, siteConfig.outDir),
  vite: {
    plugins: [{
      name: 'raw-markdown-charset',
      // The dev server bypasses buildEnd; give source .md responses an
      // explicit charset so Chinese text does not render as mojibake.
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.includes('.md')) res.setHeader('Content-Type', 'text/markdown; charset=utf-8')
          next()
        })
      },
    }],
  },
  // VitePress's generated 404 bypasses transformPageData.
  transformHead: ({ pageData }) => pageData.relativePath === '404.md'
    ? [['meta', { name: 'robots', content: 'noindex' }]] : [],
  sitemap: {
    hostname: siteUrl,
    transformItems: items => items.map(item => ({ ...item, lastmod: pageDates.get(item.url) })),
  },
  markdown: {
    math: true, image: { lazyLoading: true }, anchor: { slugify: slug },
    theme: { light: 'github-light-high-contrast', dark: 'github-dark' },
    config(md) {
      const renderImage = md.renderer.rules.image!
      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const size = imageSize(tokens[idx].attrGet('src'))
        if (size) {
          tokens[idx].attrSet('width', String(size.width))
          tokens[idx].attrSet('height', String(size.height))
        }
        tokens[idx].attrSet('decoding', 'async')
        return renderImage(tokens, idx, options, env, self)
      }
      const block = md.renderer.rules.html_block!
      const inline = md.renderer.rules.html_inline!
      for (const [type, renderer] of [['html_block', block], ['html_inline', inline]] as const) {
        md.renderer.rules[type] = (tokens, idx, options, env, self) => imageHints(renderer(tokens, idx, options, env, self))
          .replace(/\bsrc="(\/media\/[^\"]+)"/g, (_, url) => `:src="'${siteBase.slice(0, -1) + url}'"`)
      }
    },
  },
  themeConfig: {
    ...navigation(),
    logo: { src: '/favicon.svg', width: 30, height: 30 }, siteTitle: 'Agentic RL Lab · Local',
    socialLinks: [{ icon: 'github', link: repoUrl, ariaLabel: 'GitHub' }],
    search: {
      provider: 'local',
      options: {
        _render(src, env, md) {
          const html = md.render(src, env)
          if (env.frontmatter?.search === false) return ''
          return env.frontmatter?.chapter
            ? `<h1 id="article-title">${md.utils.escapeHtml(env.frontmatter.title)}<a href="#article-title" class="header-anchor">#</a></h1><p>${md.utils.escapeHtml(env.frontmatter.description)}</p>${html}`
            : html
        },
        miniSearch: { options: {
          tokenize: text => Array.from(new Intl.Segmenter('zh', { granularity: 'word' }).segment(text), part => part.segment).filter(word => /[\p{L}\p{N}]/u.test(word)),
        } },
        locales: { root: { translations: {
          button: { buttonText: '搜索实验与笔记', buttonAriaLabel: '搜索实验与笔记' },
          modal: {
            displayDetails: '显示详情', resetButtonTitle: '清空搜索', backButtonTitle: '关闭搜索', noResultsText: '没有找到相关内容',
            footer: { selectText: '选择', selectKeyAriaLabel: '回车', navigateText: '切换', navigateUpKeyAriaLabel: '向上', navigateDownKeyAriaLabel: '向下', closeText: '关闭', closeKeyAriaLabel: 'Esc' },
          },
        } } },
      },
    },
  },
})
