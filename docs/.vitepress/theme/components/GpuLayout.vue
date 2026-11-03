<script setup>
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'

// 与 localtrio/config.py 的自动布局一致：GPU 0 训练，其余卡全部做 vLLM 采样；只有 1 张卡时共用。
const options = [
  { count: 1, label: '1 张卡', note: '单卡需 ≥48GB：训练与采样共用，vLLM 只占 40% 显存', speed: '约 4 卡速度的 1/3～1/4' },
  { count: 2, label: '2 张卡', note: '每卡 ≥24GB 即可跑 4B 的单基模章节', speed: '约 4 卡速度的 1/2～1/3' },
  { count: 4, label: '4 张卡', note: '全部章节；每卡 ≥48GB 时可用 27B 教师预设', speed: '采样 3 个副本并行' },
  { count: 8, label: '8 张卡', note: '多出来的卡都做采样，也可以分两组同时跑两个实验', speed: '采样 7 个副本并行' },
]
const selected = ref(2)
const current = computed(() => options.find(o => o.count === selected.value))
const samplers = computed(() => Array.from({ length: Math.max(selected.value - 1, 0) }, (_, i) => i + 1))
const grouped = computed(() => samplers.value.length > 3)
</script>

<template>
  <figure class="rig" aria-labelledby="rig-title" aria-describedby="rig-desc">
    <figcaption class="rig-head">
      <span class="rig-lights" aria-hidden="true"><i /><i /><i /></span>
      <span id="rig-title">自动 GPU 布局</span>
      <code>python -m localtrio info</code>
    </figcaption>
    <div class="rig-tabs" role="group" aria-label="选择显卡数量">
      <button v-for="o in options" :key="o.count" type="button" :aria-pressed="selected === o.count" :class="{ active: selected === o.count }" @click="selected = o.count">{{ o.label }}</button>
    </div>
    <p id="rig-desc" class="sr-only">不需要写配置文件：GPU 0 用 transformers 与 PEFT 训练 LoRA，其余每张卡运行一个 vLLM 采样副本；只有一张卡时训练和采样共用。当前选择 {{ current.label }}：{{ current.note }}。</p>
    <ol class="rig-cards" aria-live="polite">
      <li v-if="selected === 1" class="rig-card is-train is-shared">
        <span class="rig-id">GPU&nbsp;0</span>
        <div class="rig-body">
          <p class="rig-role">训练 + 采样<span>共用一张卡</span></p>
          <p class="rig-stack">transformers + PEFT · vLLM 40%</p>
          <span class="rig-meter is-split" aria-hidden="true"><i /><b /></span>
        </div>
      </li>
      <template v-else>
        <li class="rig-card is-train">
          <span class="rig-id">GPU&nbsp;0</span>
          <div class="rig-body">
            <p class="rig-role">训练<span>LoRA</span></p>
            <p class="rig-stack">transformers + PEFT · 梯度检查点</p>
            <span class="rig-meter" aria-hidden="true"><i /></span>
          </div>
        </li>
        <template v-if="!grouped">
          <li v-for="n in samplers" :key="n" class="rig-card is-sample" :style="{ '--delay': `${n * 0.35}s` }">
            <span class="rig-id">GPU&nbsp;{{ n }}</span>
            <div class="rig-body">
              <p class="rig-role">采样<span>副本 #{{ n }}</span></p>
              <p class="rig-stack">vLLM · LoRA 热加载</p>
              <span class="rig-meter" aria-hidden="true"><i /></span>
            </div>
          </li>
        </template>
        <li v-else class="rig-card is-sample">
          <span class="rig-id">GPU&nbsp;1–{{ samplers.length }}</span>
          <div class="rig-body">
            <p class="rig-role">采样<span>{{ samplers.length }} 个副本</span></p>
            <p class="rig-stack">vLLM · 按 prompt 前缀分发请求</p>
            <span class="rig-chips" aria-hidden="true"><i v-for="n in samplers" :key="n" :style="{ '--delay': `${n * 0.2}s` }" /></span>
          </div>
        </li>
      </template>
    </ol>
    <div class="rig-foot">
      <p><strong>{{ current.note }}</strong><span>{{ current.speed }}</span></p>
      <a :href="withBase('/guide/hardware.html')">各章最低配置 →</a>
    </div>
  </figure>
</template>
