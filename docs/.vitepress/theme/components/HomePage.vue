<script setup>
import { withBase } from 'vitepress'
import { repoUrl, upstreamUrl, chapterLink } from '../../../chapters.mjs'
import { computed, ref } from 'vue'
import { useLocale } from '../composables.js'
import Icon from './Icon.vue'
import GpuLayout from './GpuLayout.vue'

const { locale, chapters, groups, path } = useLocale()
const activeGroup = ref('all')
const visibleChapters = computed(() => activeGroup.value === 'all' ? chapters.value : chapters.value.filter(c => c.group === activeGroup.value))
const groupLabel = (id) => groups.value.find(g => g.id === id)?.label
const groupCount = (id) => chapters.value.filter(c => c.group === id).length

const commands = [
  'git clone https://github.com/HXuSz11/Agentic_RL.git',
  'cd Agentic_RL',
  'uv sync',
  'uv run python scripts/gpu_smoke_test.py',
]
const copied = ref(false)
const failed = ref(false)
async function copyCommands() {
  try {
    await navigator.clipboard.writeText(commands.join('\n'))
    copied.value = true
    setTimeout(() => (copied.value = false), 1800)
  } catch {
    failed.value = true
    setTimeout(() => (failed.value = false), 1800)
  }
}

const steps = [
  { icon: 'layers', where: '采样卡 · vLLM', title: '采样', text: '每个 prompt 生成一组回答，同时返回逐 token 的 logprob。' },
  { icon: 'spark', where: '训练脚本主进程', title: '打分', text: '规则奖励、裁判模型或环境反馈给出 reward，再算组内优势。' },
  { icon: 'chip', where: '训练卡 · transformers + PEFT', title: '更新', text: 'forward_backward 与 optim_step 更新 LoRA，接口与 PyTRIO 同名同参。' },
  { icon: 'route', where: 'save_weights_and_get_sampling_client', title: '同步', text: '新 LoRA 写到本地，采样副本热加载，下一步就用上新策略。' },
]
const guides = [
  { href: '/guide/localtrio.html', icon: 'code', title: 'localtrio：本地版 Train API', text: 'forward_backward、optim_step、sample 怎样在本地实现，loss 语义和 PyTRIO 有哪些差异。' },
  { href: '/guide/hardware.html', icon: 'chip', title: '显存与 GPU 布局', text: '1 张、2 张还是 4 张卡？各章最低配置，24GB、48GB、80GB 显卡怎么分工，OOM 时先调哪个参数。' },
  { href: '/guide/learning-path.html', icon: 'route', title: '学习路线', text: '从 Loss Functions 和 GRPO 出发，按目标挑选蒸馏、工具调用或多模态方向。' },
]
</script>

<template>
  <main class="home">
    <section class="hero">
      <div class="hero-bg" aria-hidden="true" />
      <div class="hero-inner x-width">
        <div class="hero-copy">
          <p class="hero-pill"><span class="pulse-dot" aria-hidden="true" />本地 GPU 版 · 1 张卡起步，卡越多越快</p>
          <h1>把 {{ chapters.length }} 个 Agentic RL 实验<br>跑在<span class="hero-gradient">自己的显卡</span>上</h1>
          <p class="hero-lead"><a class="hero-credit" :href="upstreamUrl" target="_blank" rel="noreferrer">Datawhale 原教程</a>依赖 PyTRIO 云端训练服务。这里用 <strong>localtrio</strong> 在本地复刻同名同参的 Train API：transformers + PEFT 训练 LoRA，vLLM 负责采样，每一步自动把新权重交给采样端。不用写配置，按本机有几张卡自动分配训练卡和采样卡。</p>
          <div class="hero-actions">
            <a class="x-button primary" :href="withBase(path('/guide/quickstart.html'))">开始第一个实验 <Icon name="arrow" :size="17" /></a>
            <a class="x-button ghost" :href="withBase(path('/experiments/'))">浏览全部实验 <Icon name="book" :size="17" /></a>
          </div>
          <div class="terminal">
            <div class="terminal-bar">
              <span class="terminal-lights" aria-hidden="true"><i /><i /><i /></span>
              <span>快速开始</span>
              <button type="button" class="terminal-copy" aria-label="复制快速开始命令" @click="copyCommands"><Icon :name="copied ? 'check' : 'copy'" :size="14" />{{ copied ? '已复制' : failed ? '复制失败' : '复制' }}</button>
              <span class="sr-only" role="status">{{ copied ? '命令已复制到剪贴板' : failed ? '复制失败，请手动选择命令' : '' }}</span>
            </div>
            <pre><code><span v-for="cmd in commands" :key="cmd" class="terminal-line"><span class="prompt" aria-hidden="true">$</span>{{ cmd }}</span></code></pre>
          </div>
        </div>
        <GpuLayout />
      </div>
    </section>

    <section class="stats x-width" aria-label="概览">
      <div><strong>{{ chapters.length }}</strong><span>篇实验笔记</span></div>
      <div><strong>{{ groups.length }}</strong><span>个研究方向</span></div>
      <div><strong>1<small>张卡起步</small></strong><span>单卡 ≥48GB，或多张 ≥24GB</span></div>
      <div><strong>1</strong><span>套与 PyTRIO 同名同参的 API</span></div>
    </section>

    <section class="section x-width">
      <header class="section-head">
        <div><h2>一步训练里发生了什么</h2><p>所有章节共用同一个循环，区别只在奖励怎么来、loss 怎么写。</p></div>
        <a class="text-link" :href="withBase(path('/guide/localtrio.html'))">localtrio 原理 <Icon name="arrow" :size="16" /></a>
      </header>
      <ol class="steps">
        <li v-for="(step, i) in steps" :key="step.title" class="step">
          <div class="step-top"><span class="step-icon"><Icon :name="step.icon" :size="19" /></span><span class="step-index">0{{ i + 1 }}</span></div>
          <h3>{{ step.title }}</h3>
          <p class="step-where">{{ step.where }}</p>
          <p>{{ step.text }}</p>
        </li>
      </ol>
    </section>

    <section class="section x-width">
      <header class="section-head">
        <div><h2>全部实验</h2><p>新读者建议先读 <a :href="withBase(path('/experiments/loss-functions/'))">Loss Functions</a> 和 <a :href="withBase(path('/experiments/grpo/'))">GRPO</a>。</p></div>
        <a class="text-link" :href="withBase(path('/experiments/'))">实验手册 <Icon name="arrow" :size="16" /></a>
      </header>
      <div class="group-tabs" role="group" aria-label="按方向筛选">
        <button type="button" :aria-pressed="activeGroup === 'all'" :class="{ active: activeGroup === 'all' }" @click="activeGroup = 'all'">全部<span>{{ chapters.length }}</span></button>
        <button v-for="group in groups" :key="group.id" type="button" :class="['g-' + group.id, { active: activeGroup === group.id }]" :aria-pressed="activeGroup === group.id" @click="activeGroup = group.id"><i class="g-dot" aria-hidden="true" />{{ group.label }}<span>{{ groupCount(group.id) }}</span></button>
      </div>
      <div class="card-grid">
        <a v-for="chapter in visibleChapters" :key="chapter.id" :href="withBase(chapterLink(chapter, locale))" :class="['x-card', 'g-' + chapter.group]">
          <div class="card-top"><span class="card-num">{{ chapter.number }}</span><span class="g-chip"><i class="g-dot" aria-hidden="true" />{{ groupLabel(chapter.group) }}</span></div>
          <h3>{{ chapter.title }}</h3>
          <p class="card-sub">{{ chapter.subtitle }}</p>
          <p class="card-desc">{{ chapter.description }}</p>
          <div class="card-foot"><span v-for="tag in chapter.tags" :key="tag" class="card-tag">{{ tag }}</span><span class="card-go" aria-hidden="true"><Icon name="arrow" :size="15" /></span></div>
        </a>
      </div>
    </section>

    <section class="section x-width">
      <header class="section-head"><div><h2>本地训练指南</h2><p>跑实验之前，先弄清楚训练、采样和显存是怎么安排的。</p></div></header>
      <div class="guide-grid">
        <a v-for="guide in guides" :key="guide.href" class="guide-card" :href="withBase(path(guide.href))">
          <span class="guide-icon"><Icon :name="guide.icon" :size="20" /></span>
          <h3>{{ guide.title }}</h3>
          <p>{{ guide.text }}</p>
          <span class="guide-go">阅读 <Icon name="arrow" :size="15" /></span>
        </a>
      </div>
    </section>

    <section class="cta x-width">
      <div>
        <h2>跑出结果了？</h2>
        <p>原文的指标来自 PyTRIO 云端，本地数值以你自己的实验记录为准。欢迎把你在自己显卡上的复现结果、显卡型号与显存数据、踩过的坑提交到仓库。</p>
      </div>
      <a class="x-button primary" :href="repoUrl" target="_blank" rel="noreferrer">在 GitHub 参与 <Icon name="diagonal" :size="16" /></a>
    </section>
  </main>
</template>
