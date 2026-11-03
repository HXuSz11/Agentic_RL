import DefaultTheme from 'vitepress/theme'
import { inBrowser, useData } from 'vitepress'
import { h, watch } from 'vue'
import PageIntro from './components/PageIntro.vue'
import RelatedReading from './components/RelatedReading.vue'
import HomePage from './components/HomePage.vue'
import ExperimentCatalog from './components/ExperimentCatalog.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  // “继续阅读”放在上一篇/下一篇之前，翻页导航仍是正文后的最后一块
  Layout: () => h(DefaultTheme.Layout, null, { 'doc-before': () => h(PageIntro), 'doc-footer-before': () => h(RelatedReading) }),
  enhanceApp({ app }) {
    app.component('SiteHome', HomePage)
    app.component('ExperimentCatalog', ExperimentCatalog)
  },
  setup() {
    // 手机浏览器地址栏颜色跟随站内的深浅色切换
    const { isDark } = useData()
    if (inBrowser) {
      watch(isDark, dark => {
        for (const meta of document.querySelectorAll('meta[name="theme-color"]')) meta.setAttribute('content', dark ? '#0d1116' : '#ffffff')
      }, { immediate: true })
    }
  },
}
