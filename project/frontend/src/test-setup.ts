import '@testing-library/jest-dom/vitest'

// jsdom has no layout. Real focus/scroll visibility is checked by Playwright.
Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: () => {} })
