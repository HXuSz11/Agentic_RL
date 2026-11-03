<script setup>
defineProps({ name: { type: String, default: 'arrow' }, size: { type: Number, default: 20 } })
const paths = {
  arrow: 'M4 12h16m-6-6 6 6-6 6',
  diagonal: 'M7 17 17 7M8 7h9v9',
  book: 'M12 5v15M3 4c4-1 7 0 9 2 2-2 5-3 9-2v15c-4-1-7 0-9 2-2-2-5-3-9-2Z',
  code: 'm8 6-6 6 6 6m8-12 6 6-6 6m-3-16-2 20',
  flask: 'M9 3h6m-5 0v7L4 20c-.3.5 0 1 1 1h14c1 0 1.3-.5 1-1l-6-10V3M7 15h10',
  search: 'M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
  layers: 'm12 3 10 6-10 6L2 9Zm-10 12 10 6 10-6M2 12l10 6 10-6',
  check: 'm5 12 4 4L19 6',
  copy: 'M9 9h11v11H9zM5 15H4V4h11v1',
  chip: 'M7 7h10v10H7zM9 3v4m6-4v4M9 17v4m6-4v4M3 9h4m-4 6h4m10-6h4m-4 6h4',
  route: 'M6 19a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm12-10a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM8 17h7a3 3 0 0 0 0-6H9a3 3 0 0 1 0-6h7',
  spark: 'M12 3v4m0 10v4M3 12h4m10 0h4M6 6l2.5 2.5m7 7L18 18M6 18l2.5-2.5m7-7L18 6',
}
</script>
<template><svg :width="size" :height="size" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path :d="paths[name] || paths.arrow" /></svg></template>
