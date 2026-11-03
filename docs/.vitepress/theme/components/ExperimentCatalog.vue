<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { withBase } from 'vitepress'
import { chapterLink } from '../../../chapters.mjs'
import Icon from './Icon.vue'
import { useLocale } from '../composables.js'
const { locale, chapters, groups, path } = useLocale()
const selected = ref('all')
const query = ref('')
const filtered = computed(() => chapters.value.filter(c => (selected.value === 'all' || c.group === selected.value) && `${c.title} ${c.subtitle} ${c.description} ${c.tags.join(' ')}`.toLowerCase().includes(query.value.toLowerCase().trim())))
const groupLabel = (id) => groups.value.find(g => g.id === id)?.label
const groupNote = (id) => groups.value.find(g => g.id === id)?.note
function readTopic() { const topic = new URLSearchParams(location.search).get('topic'); selected.value = groups.value.some(g => g.id === topic) ? topic : 'all' }
function selectTopic(id) { selected.value = id; const url = new URL(location.href); id === 'all' ? url.searchParams.delete('topic') : url.searchParams.set('topic', id); history.replaceState(null, '', url) }
onMounted(() => { readTopic(); window.addEventListener('popstate', readTopic) })
onUnmounted(() => window.removeEventListener('popstate', readTopic))
</script>

<template>
  <main class="catalog x-width">
    <header class="catalog-header">
      <h1>实验手册</h1>
      <p>{{ chapters.length }} 篇算法拆解与本地复现记录：从一个训练循环出发，走到会搜索、会调用工具、会看图的 Agent。每篇都写明了 GPU 布局和运行命令。</p>
    </header>
    <div class="catalog-controls">
      <div class="group-tabs" role="group" aria-label="按研究方向筛选">
        <button type="button" :aria-pressed="selected === 'all'" :class="{ active: selected === 'all' }" @click="selectTopic('all')">全部<span>{{ chapters.length }}</span></button>
        <button v-for="group in groups" :key="group.id" type="button" :class="['g-' + group.id, { active: selected === group.id }]" :aria-pressed="selected === group.id" @click="selectTopic(group.id)"><i class="g-dot" aria-hidden="true" />{{ group.label }}<span>{{ chapters.filter(c => c.group === group.id).length }}</span></button>
      </div>
      <label class="catalog-search"><Icon name="search" :size="17" /><input v-model="query" aria-label="筛选实验" placeholder="搜索算法、任务或关键词…"></label>
    </div>
    <p v-if="selected !== 'all'" class="catalog-note">{{ groupLabel(selected) }}：{{ groupNote(selected) }}</p>
    <div class="catalog-count" aria-live="polite">共 {{ filtered.length }} 个实验</div>
    <div class="catalog-list">
      <a v-for="chapter in filtered" :key="chapter.id" :href="withBase(chapterLink(chapter, locale))" :class="['catalog-row', 'g-' + chapter.group]">
        <span class="catalog-number">{{ chapter.number }}</span>
        <div class="catalog-name"><h2>{{ chapter.title }}</h2><p>{{ chapter.subtitle }}</p></div>
        <p class="catalog-description">{{ chapter.description }}</p>
        <span class="g-chip catalog-group"><i class="g-dot" aria-hidden="true" />{{ groupLabel(chapter.group) }}</span>
        <Icon name="arrow" :size="18" />
      </a>
      <div v-if="!filtered.length" class="catalog-empty"><Icon name="search" :size="30" /><h2>没有找到相关实验</h2><p>试试 GRPO、搜索、蒸馏，或者重置筛选。</p><button class="x-button ghost" @click="query = ''; selectTopic('all')">查看全部实验</button></div>
    </div>
    <p class="catalog-bottom">第一次来？按 <a :href="withBase(path('/guide/learning-path.html'))">学习路线</a> 找到适合自己的起点。</p>
  </main>
</template>
