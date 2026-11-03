import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import PageIntro from './components/PageIntro.vue'
import RelatedReading from './components/RelatedReading.vue'
import HomePage from './components/HomePage.vue'
import ExperimentCatalog from './components/ExperimentCatalog.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => h(DefaultTheme.Layout, null, { 'doc-before': () => h(PageIntro), 'doc-after': () => h(RelatedReading) }),
  enhanceApp({ app }) {
    app.component('SiteHome', HomePage)
    app.component('ExperimentCatalog', ExperimentCatalog)
  },
}
