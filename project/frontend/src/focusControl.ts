/** Focus without opening native controls or starting business operations. */
export function focusControl(id: string) {
  const control = document.getElementById(id)
  if (!(control instanceof HTMLElement) || control.matches(':disabled')) return
  control.focus({ preventScroll: true })
  control.scrollIntoView({ block: 'nearest', behavior: 'instant' })
}
