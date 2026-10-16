"""Endpoint wiring for the Review API routers (split out of ``api.py``, V1-06).

``ApiWiring`` carries the resolved runtime plus the guard chain every business
endpoint shares: fail closed when the store is not configured, resolve the
identity, enforce the scope, then the Case grant — and keep enough on
``request.state`` for the error handlers to classify denials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

from .api_support import bearer_token, request_id
from .review_service import ReviewServiceError
from .review_store import CaseRegistration
from .security_contracts import AuthContext, Scope, SecurityError, SecurityOperation

if TYPE_CHECKING:
    from .api import SecurityRuntime


class ApiWiring:
    """The per-app guard chain bound to one resolved ``SecurityRuntime``."""

    def __init__(self, runtime: SecurityRuntime) -> None:
        self._runtime = runtime

    @property
    def failure_reason(self) -> str | None:
        """The unavailability reason, when the runtime came up unconfigured."""

        return self._runtime.failure

    def require_ready(self) -> SecurityRuntime:
        if self._runtime.failure is not None or self._runtime.service is None:
            raise ReviewServiceError(
                status_code=503,
                code="security_not_configured",
                message="The local security store is not configured.",
            )
        return self._runtime

    def context(
        self,
        request: Request,
        *,
        operation: SecurityOperation,
        scope: Scope | None = None,
    ) -> AuthContext:
        active = self.require_ready()
        assert active.authorization is not None  # guaranteed when ready
        context = active.authorization.resolve(
            bearer_token(request),
            request_id=request_id(request),
            operation=operation,
        )
        request.state.auth_context = context
        request.state.operation = operation
        if scope is not None:
            active.authorization.require_scope(
                context,
                scope,
                request_id=request_id(request),
                operation=operation,
            )
        return context

    def case(
        self,
        request: Request,
        context: AuthContext,
        case_id: str,
        *,
        operation: SecurityOperation,
    ) -> CaseRegistration:
        active = self.require_ready()
        assert active.authorization is not None
        return active.authorization.require_case(
            context,
            case_id,
            request_id=request_id(request),
            operation=operation,
        )

    def granted_case_ids(self, context: AuthContext) -> tuple[str, ...]:
        active = self.require_ready()
        assert active.store is not None
        return tuple(
            item.case_id
            for item in active.store.list_granted_cases(
                tenant_id=context.tenant_id,
                subject_id=context.subject_id,
            )
        )

    def deny_not_found(
        self,
        request: Request,
        context: AuthContext,
        *,
        operation: SecurityOperation,
        case_id: str,
    ) -> SecurityError:
        active = self.require_ready()
        assert active.authorization is not None
        return active.authorization.deny_not_found(
            request_id=request_id(request),
            operation=operation,
            context=context,
            case_id=case_id,
        )

    def ensure_case_still_granted(
        self,
        request: Request,
        context: AuthContext,
        case_id: str,
        *,
        operation: SecurityOperation,
    ) -> None:
        active = self.require_ready()
        assert active.store is not None
        if not active.store.case_granted(
            tenant_id=context.tenant_id,
            case_id=case_id,
            subject_id=context.subject_id,
        ):
            raise self.deny_not_found(
                request,
                context,
                operation=operation,
                case_id=case_id,
            )


__all__ = ["ApiWiring"]
