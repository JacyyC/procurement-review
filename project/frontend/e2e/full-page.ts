import type { Page } from '@playwright/test'

/** Full-page capture must start at zero: offscreen fixed skip links otherwise
 * appear inside the stitched image at the previous viewport's scroll offset. */
export async function fullPage(page: Page, options: Parameters<Page['screenshot']>[0]) {
  await page.evaluate(async () => {
    await document.fonts.ready
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
    await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))
  })
  return page.screenshot(options)
}
