import { Component, Fragment, type ReactNode } from 'react'

import { reportRenderFailure } from './errorReporting'

/**
 * Recoverable render boundary for the workbench.
 *
 * The response contract (`apiContract.ts`) rejects malformed payloads before they reach
 * React, so reaching this boundary means a genuine rendering exception — a bug, not a bad
 * response.  Three rules follow from that:
 *
 *  - the fallback keeps an escape hatch open: the human can re-read the view, or
 *    clear the credential and sign in again, instead of facing a blank page;
 *  - **resetting never replays a write.**  `onReset` is owned by `App` and only
 *    clears derived view state and re-issues *reads*.  A confirm/reject/cancel/
 *    resume that already left the browser is never re-sent by a boundary reset,
 *    because the workbench has no retry path for writes at all;
 *  - diagnostics go through `errorReporting.ts`, the same outlet the React root uses.
 *    Nothing about the thrown value is printed — not its name, message, stack, cause or
 *    component stack — because a component that renders a payload is exactly the case
 *    where the thrown value carries that payload (FIX-04B / C-AUD-07).
 */
interface Props {
  children: ReactNode
  /** Called once per reset, after the boundary clears its own failed state. */
  onReset: () => void
  /** Optional escape hatch back to the credential screen. */
  onSignOut?: () => void
  /** Human-readable scope, e.g. 工作台 or 审查视图. */
  scope: string
}

interface State {
  failed: boolean
  /** Remounts the children so a recoverable failure really gets a fresh render. */
  resetKey: number
}

export class ReviewErrorBoundary extends Component<Props, State> {
  state: State = { failed: false, resetKey: 0 }

  static getDerivedStateFromError(): Partial<State> {
    return { failed: true }
  }

  componentDidCatch(thrown: unknown): void {
    // A component may throw anything at all, including `null`; the outlet takes the
    // value as `unknown` and reports only its type.
    reportRenderFailure(this.props.scope, thrown)
  }

  private reset = (): void => {
    this.setState((state) => ({ failed: false, resetKey: state.resetKey + 1 }))
    this.props.onReset()
  }

  private signOut = (): void => {
    this.setState((state) => ({ failed: false, resetKey: state.resetKey + 1 }))
    this.props.onSignOut?.()
  }

  render(): ReactNode {
    if (!this.state.failed) {
      // `key` remounts the subtree on reset, so a view that threw once does not
      // inherit half-initialised state on the next attempt.  A fragment keeps the
      // boundary invisible to layout: no extra box is inserted into the grid.
      return <Fragment key={this.state.resetKey}>{this.props.children}</Fragment>
    }
    return (
      <section className="boundary-state" role="alert" data-testid="review-error-boundary">
        <h2>界面无法显示当前视图</h2>
        <p>
          这是本地的界面错误，服务端记录没有被修改。你可以重新读取当前视图；已提交的
          人工动作不会被自动重放。
        </p>
        <div className="boundary-actions">
          <button type="button" className="primary-button" onClick={this.reset}>
            重新读取视图
          </button>
          {this.props.onSignOut && (
            <button
              type="button"
              className="icon-text-button"
              onClick={this.signOut}
            >
              清除凭据并返回登录
            </button>
          )}
        </div>
      </section>
    )
  }
}
