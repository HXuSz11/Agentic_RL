<script setup>
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { chapterLink, repoUrl, repoRef } from '../../../chapters.mjs'
import { useLocale } from '../composables.js'
import Icon from './Icon.vue'
import MarkdownActions from './MarkdownActions.vue'
const { frontmatter } = useData()
const { locale, chapters } = useLocale()
const chapter = computed(() => chapters.value.find(c => c.id === frontmatter.value.chapter))
</script>

<template>
  <header v-if="chapter" :class="['article-intro', 'g-' + chapter.group]">
    <MarkdownActions />
    <p class="article-kicker"><span class="g-chip"><i class="g-dot" aria-hidden="true" />{{ frontmatter.group }}</span><span>第 {{ chapter.number }} 篇</span></p>
    <h1 id="article-title">{{ chapter.title }}<span>{{ frontmatter.isRun ? '快速启动：环境、GPU 布局与运行命令' : chapter.subtitle }}</span></h1>
    <p class="article-description">{{ frontmatter.description }}</p>
    <div class="article-meta">
      <span><Icon name="book" :size="15" /> 约 {{ frontmatter.readingTime }} 分钟</span>
      <a :href="frontmatter.authorUrl" rel="author">{{ frontmatter.author }}</a>
      <span>更新于 <time :datetime="frontmatter.modified">{{ frontmatter.modified }}</time></span>
      <span v-for="tag in chapter.tags" :key="tag" class="article-tag">{{ tag }}</span>
    </div>
    <div class="article-actions">
      <a v-if="chapter.start && !frontmatter.isRun" class="x-button primary small" :href="withBase(`${chapterLink(chapter, locale)}run.html`)"><Icon name="flask" :size="16" /> 快速启动</a>
      <a v-if="frontmatter.isRun" class="x-button ghost small" :href="withBase(chapterLink(chapter, locale))"><Icon name="book" :size="16" /> 阅读算法拆解</a>
      <a class="x-button ghost small" :href="`${repoUrl}/tree/${repoRef}/${chapter.source.slice(0, chapter.source.lastIndexOf('/'))}`" target="_blank" rel="noreferrer"><Icon name="code" :size="16" /> 查看代码 <Icon name="diagonal" :size="14" /></a>
    </div>
  </header>
  <div v-else-if="frontmatter.modified" class="guide-intro"><MarkdownActions /><p class="guide-meta"><a :href="frontmatter.authorUrl" rel="author">{{ frontmatter.author }}</a> · 更新于 <time :datetime="frontmatter.modified">{{ frontmatter.modified }}</time></p></div>
</template>
