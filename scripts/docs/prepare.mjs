import { readFile, writeFile, mkdir, copyFile, rm, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import MarkdownIt from 'markdown-it'
import { chapters, localizedChapters, localizedGroups, repoUrl, repoRef, upstreamUrl, chapterLink, articleCovers } from '../../docs/chapters.mjs'
import { sourceDates } from '../../docs/.vitepress/seo.mjs'

const root = fileURLToPath(new URL('../../', import.meta.url))
const content = path.join(root, 'docs/content')
const media = path.join(content, 'public/media')
const parser = new MarkdownIt({ html: true })
const routes = new Map(chapters.flatMap(c => [
  [c.source, chapterLink(c)], ...(c.start ? [[c.start, `${chapterLink(c)}run`]] : []),
]))
routes.set('README.md', '/guide/introduction')
// 章节里可以用相对路径链接手写指南页（docs/content/guide/*.md），这里映射成站内路由
for (const page of ['introduction', 'learning-path', 'quickstart', 'localtrio', 'hardware', 'grpo-dapo-gspo', 'contributing']) {
  routes.set(`docs/content/guide/${page}.md`, `/guide/${page}`)
}
const copied = new Set()

// 只清理自动生成的章节目录和复制的配图，不动手写页面。
await rm(media, { recursive: true, force: true })
for (const c of chapters) await rm(path.join(content, 'experiments', c.id), { recursive: true, force: true })

async function resolveUrl(raw, source, image = false) {
  let url = raw.replace(/&amp;/g, '&')
  for (const prefix of [`${repoUrl}/blob/main/`, `${repoUrl}/blob/HEAD/`, `${upstreamUrl}/blob/main/`]) {
    // 指向本仓库同名文件的绝对链接也转为站内处理；上游链接只在本仓库有同路径文件时才转换
    if (url.startsWith(prefix)) {
      const candidate = url.slice(prefix.length).split('#')[0]
      if (prefix !== `${upstreamUrl}/blob/main/` || (await stat(path.join(root, decodeURIComponent(candidate))).catch(() => null))) url = '/' + url.slice(prefix.length)
    }
  }
  if (/^(?:[a-z]+:|\/\/|#)/i.test(url)) return raw
  const hashAt = url.indexOf('#')
  const hash = hashAt < 0 ? '' : url.slice(hashAt)
  const local = decodeURIComponent((hashAt < 0 ? url : url.slice(0, hashAt)).split('?')[0])
  const relative = path.posix.normalize(local.startsWith('/') ? local.slice(1) : path.posix.join(path.posix.dirname(source), local))
  if (relative.startsWith('../')) throw new Error(`Path escapes repository: ${source}: ${raw}`)
  if (routes.has(relative)) return routes.get(relative) + hash
  const absolute = path.join(root, relative)
  const info = await stat(absolute).catch(() => null)
  if (!info) throw new Error(`Missing local target: ${source}: ${raw} -> ${relative}`)
  if (image || /\.(png|jpe?g|gif|svg|webp|avif)$/i.test(relative)) {
    if (!copied.has(relative)) {
      await mkdir(path.dirname(path.join(media, relative)), { recursive: true })
      await copyFile(absolute, path.join(media, relative))
      copied.add(relative)
    }
    return '/media/' + relative.split('/').map(encodeURIComponent).join('/') + hash
  }
  return `${repoUrl}/${info.isDirectory() ? 'tree' : 'blob'}/${repoRef}/${relative.split('/').map(encodeURIComponent).join('/')}${hash}`
}

async function transform(body, source) {
  body = body.replace(/```math\r?\n([\s\S]*?)```/g, (_, equation) => `$$\n${equation.trim()}\n$$`)
  body = body.replace(/<div\b[^>]*>[\s\S]*?<\/div>/gi, block => /(?:shields\.io|visitor-badge|komarev\.com)/.test(block) ? '' : block)
  const destinations = new Map()
  function visit(tokens) {
    for (const token of tokens) {
      if (token.type === 'image') destinations.set(token.attrGet('src'), true)
      if (token.type === 'link_open') destinations.set(token.attrGet('href'), false)
      if (token.children) visit(token.children)
    }
  }
  visit(parser.parse(body, {}))
  for (const match of body.matchAll(/\b(src|href)=["']([^"']+)["']/g)) destinations.set(match[2], match[1] === 'src')
  for (const [url, isImage] of destinations) {
    const resolved = await resolveUrl(url, source, isImage)
    if (resolved === url) continue
    for (const original of new Set([url, decodeURI(url)])) {
      body = body.split(`](${original})`).join(`](${resolved})`)
      body = body.split(`](<${original}>)`).join(`](${resolved})`)
      body = body.split(`src="${original}"`).join(`src="${resolved}"`)
      body = body.split(`href="${original}"`).join(`href="${resolved}"`)
      body = body.split(`src='${original}'`).join(`src='${resolved}'`)
      body = body.split(`href='${original}'`).join(`href='${resolved}'`)
    }
  }
  // 空 alt 的图片用文件名生成说明文字
  body = body.replace(/!\[\]\(([^)]+)\)/g, (_, url) => {
    const label = decodeURIComponent(url.split('/').at(-1).split('#')[0]).replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ')
    return `![${label}](${url})`
  })
  return body
}

let pages = 0
for (const chapter of localizedChapters()) {
  for (const isRun of [false, true]) {
    const source = isRun ? chapter.start : chapter.source
    if (!source) continue
    let body = await readFile(path.join(root, source), 'utf8')
    body = body.replace(/^# .+\r?\n/, '')
    const cover = !isRun && articleCovers[chapter.id]
    if (cover) {
      body = body.replace(/!\[[^\]]*\]\(([^)]+)\)/g, (match, url) => decodeURI(url.replace(/^<|>$/g, '')) === cover ? '' : match)
      body = body.replace(/<div\b[^>]*>\s*<img\b[^>]*>\s*<\/div>/gi, block => block.includes(`src="${cover}"`) ? '' : block)
    }
    body = await transform(body, source)
    const frontmatter = {
      title: isRun ? `${chapter.title} 快速启动：环境、GPU 布局与运行命令` : chapter.seoTitle,
      description: isRun ? chapter.runDescription : chapter.description,
      chapter: chapter.id, source, isRun,
      ...(chapter.author ? { author: chapter.author } : {}),
      sourceDates: sourceDates(source),
      readingTime: Math.max(1, Math.ceil(body.replace(/```[\s\S]*?```/g, '').length / 450)),
      group: localizedGroups().find(g => g.id === chapter.group).label,
    }
    if (!frontmatter.title || !frontmatter.description) throw new Error(`Missing title/description metadata for ${chapter.id}${isRun ? ' (run)' : ''}`)
    const output = path.join(content, 'experiments', chapter.id, isRun ? 'run.md' : 'index.md')
    await mkdir(path.dirname(output), { recursive: true })
    await writeFile(output, `---\n${Object.entries(frontmatter).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n')}\n---\n\n${body.trim()}\n`)
    pages++
  }
}
console.log(`Prepared ${pages} pages from ${chapters.length} chapters; copied ${copied.size} referenced images.`)
