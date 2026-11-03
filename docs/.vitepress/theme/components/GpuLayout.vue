<script setup>
// 默认 4×A6000 布局示意（与 configs/4xa6000.yaml 一致）：GPU0 训练，GPU1-3 各一个 vLLM 采样副本。
const samplers = [1, 2, 3]
</script>

<template>
  <figure class="rig" aria-labelledby="rig-title" aria-describedby="rig-desc">
    <figcaption class="rig-head">
      <span class="rig-lights" aria-hidden="true"><i /><i /><i /></span>
      <span id="rig-title">默认 GPU 布局</span>
      <code>configs/4xa6000.yaml</code>
    </figcaption>
    <p id="rig-desc" class="sr-only">GPU0 用 transformers 与 PEFT 训练 LoRA；GPU1 到 GPU3 各运行一个 vLLM 采样副本。每一步训练后，新的 LoRA 权重同步给三个采样副本热加载。示意图，不代表实测显存。</p>
    <ol class="rig-cards">
      <li class="rig-card is-train">
        <span class="rig-id">GPU&nbsp;0</span>
        <div class="rig-body">
          <p class="rig-role">训练<span>LoRA</span></p>
          <p class="rig-stack">transformers + PEFT · 梯度检查点</p>
          <span class="rig-meter" aria-hidden="true"><i /></span>
        </div>
        <span class="rig-mem">48&nbsp;GB</span>
      </li>
      <li v-for="n in samplers" :key="n" class="rig-card is-sample" :style="{ '--delay': `${n * 0.35}s` }">
        <span class="rig-id">GPU&nbsp;{{ n }}</span>
        <div class="rig-body">
          <p class="rig-role">采样<span>副本 #{{ n }}</span></p>
          <p class="rig-stack">vLLM · 显存预留 85%</p>
          <span class="rig-meter" aria-hidden="true"><i /></span>
        </div>
        <span class="rig-mem">48&nbsp;GB</span>
      </li>
    </ol>
    <div class="rig-sync">
      <span class="rig-wire" aria-hidden="true"><i /></span>
      <p><code>save_weights_and_<wbr>get_sampling_client()</code> 每步把新 LoRA 交给三个副本热加载</p>
    </div>
  </figure>
</template>
