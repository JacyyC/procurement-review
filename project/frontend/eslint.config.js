import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist', 'playwright-report', 'test-results', 'e2e/probe/dist'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    // The e2e bootstrap helpers run in Node, not in the browser.
    files: ['e2e/**/*.mjs'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: { ...globals.node },
    },
  },
  {
    // The production probe's console recorder is a plain browser script injected before
    // any page script runs, so it is neither a module nor Node code.
    files: ['e2e/probe/console-scan.js'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'script',
      globals: { ...globals.browser },
    },
  },
  {
    // The recorder wraps functions the page may not export; named unused catches are noise.
    files: ['e2e/probe/console-scan.js'],
    rules: { '@typescript-eslint/no-unused-vars': 'off' },
  },
  {
    files: ['**/*.{ts,tsx}'],
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs['recommended-latest'].rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
  {
    // The probe entry is a test asset that mounts deliberately broken components; it is
    // not a hot-reloadable product module.
    files: ['e2e/probe/main.tsx'],
    rules: { 'react-refresh/only-export-components': 'off' },
  },
)
