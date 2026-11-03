import { computed } from 'vue'
import { localizedChapters, localizedGroups } from '../../chapters.mjs'
// 站点只有中文；保留 t() 接口，方便组件沿用原写法。
export function useLocale() {
  const en = computed(() => false)
  const locale = computed(() => 'zh')
  const chapters = computed(() => localizedChapters())
  const groups = computed(() => localizedGroups())
  const path = (value) => value
  const t = (zh) => zh
  return { en, locale, chapters, groups, path, t }
}
