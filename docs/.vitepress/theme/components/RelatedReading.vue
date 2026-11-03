<script setup>
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { chapterLink } from '../../../chapters.mjs'
import { useLocale } from '../composables.js'
import Icon from './Icon.vue'
const { frontmatter } = useData()
const { locale, chapters, path } = useLocale()
const current = computed(() => chapters.value.find(c => c.id === frontmatter.value.chapter))
const related = computed(() => current.value ? chapters.value.filter(c => c.group === current.value.group && c.id !== current.value.id) : [])
</script>
<template>
  <aside v-if="related.length" class="related-reading" aria-label="继续阅读">
    <h2>继续阅读</h2>
    <ul class="related-list">
      <li v-for="chapter in related" :key="chapter.id"><a :class="'g-' + chapter.group" :href="withBase(chapterLink(chapter, locale))"><span class="related-num">{{ chapter.number }}</span><span class="related-title">{{ chapter.title }}<small>{{ chapter.subtitle }}</small></span><Icon name="arrow" :size="15" /></a></li>
    </ul>
    <p class="related-links">
      <a :href="withBase(path('/guide/learning-path.html'))">完整学习路线</a>
      <a :href="withBase(path('/guide/localtrio.html'))">localtrio：本地版 Train API</a>
      <a :href="withBase(path('/guide/hardware.html'))">4×A6000 显存与 GPU 布局</a>
    </p>
  </aside>
</template>
