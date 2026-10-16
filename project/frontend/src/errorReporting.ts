/**
 * The workbench's single safe diagnostic outlet (FIX-04B / C-AUD-07).
 *
 * React reports a render error three ways, and the workbench must own all of them:
 * `onCaughtError` for an error a boundary handled, `onUncaughtError` for one that
 * unmounted the root, and `onRecoverableError` for one React recovered from on its own.
 * With none configured, React 19's production default calls `console.error(error)`
 * directly, which prints the whole thrown value — including a rendered payload and its
 * stack — into the console.
 *
 * The rule this module enforces is that a diagnostic describes *where* something failed,
 * never *what* it carried:
 *
 *  - the scope is a value from a fixed local vocabulary, so a future caller cannot smuggle
 *    a string into the one field the outlet prints;
 *  - the category is one of this module's own constants;
 *  - the only description of the thrown value is its JavaScript *type*, drawn from another
 *    fixed set.  `name`, `message`, `stack`, `cause`, `componentStack` and any other
 *    property are never read, because all of them are controlled by whoever threw.
 *
 * No global `console` replacement happens here: the outlet is a normal function that the
 * boundary and the root both call, so the console keeps working for everything else.
 */

import type { RootOptions } from 'react-dom/client'

/** The scopes the workbench declares.  Anything else falls back to the first entry. */
const SCOPES = ['工作台', '审查视图', 'React 根'] as const

export type DiagnosticScope = (typeof SCOPES)[number]

/** Map any caller-supplied label onto the declared vocabulary. */
export function scopeOf(scope: unknown): DiagnosticScope {
  return SCOPES.includes(scope as DiagnosticScope) ? (scope as DiagnosticScope) : '工作台'
}

/** Fixed categories; a thrown value never contributes to one. */
export type DiagnosticCategory =
  | 'render_failed'
  | 'root_caught'
  | 'root_uncaught'
  | 'root_recoverable'

const CATEGORY_OF_ROOT_CALLBACK = {
  onCaughtError: 'root_caught',
  onUncaughtError: 'root_uncaught',
  onRecoverableError: 'root_recoverable',
} as const satisfies Record<string, DiagnosticCategory>

/** The only thing we are willing to say about a thrown value: its JS type. */
function thrownKind(value: unknown): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (value instanceof Error) return 'error'
  const kind = typeof value
  if (kind === 'string' || kind === 'number' || kind === 'boolean') return kind
  if (kind === 'symbol' || kind === 'function' || kind === 'object') return kind
  if (kind === 'bigint') return 'bigint'
  return 'unknown'
}

function emit(scope: DiagnosticScope, category: DiagnosticCategory, thrown: unknown): void {
  console.error(`[review-workbench] ${scope} ${category} (thrown: ${thrownKind(thrown)})`)
}

/**
 * Report one render failure.  `info` is accepted so the React callback signature fits, but
 * it is never read: `componentStack` names components and comes from the same stack.
 */
export function reportRenderFailure(
  scope: unknown,
  thrown: unknown,
  category: DiagnosticCategory = 'render_failed',
): void {
  emit(scopeOf(scope), category, thrown)
}

/**
 * Root options for `createRoot`/`hydrateRoot`.  The product entry and the production
 * probe both use this object, so the verified behaviour is the shipped behaviour.
 */
export const ROOT_ERROR_OPTIONS: RootOptions = {
  onCaughtError: (error) => reportRenderFailure('React 根', error, CATEGORY_OF_ROOT_CALLBACK.onCaughtError),
  onUncaughtError: (error) =>
    reportRenderFailure('React 根', error, CATEGORY_OF_ROOT_CALLBACK.onUncaughtError),
  onRecoverableError: (error) =>
    reportRenderFailure('React 根', error, CATEGORY_OF_ROOT_CALLBACK.onRecoverableError),
}
