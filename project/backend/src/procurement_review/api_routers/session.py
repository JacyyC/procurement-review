"""Session surface: the caller's own verified identity (V1-06)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..api_contracts import SessionResponse
from ..api_wiring import ApiWiring
from ..security_contracts import SecurityOperation


def build_session_router(wiring: ApiWiring) -> APIRouter:
    router = APIRouter()

    @router.get("/session", response_model=SessionResponse)
    async def session(request: Request) -> SessionResponse:
        context = wiring.context(request, operation=SecurityOperation.SESSION)
        return SessionResponse(
            subject_id=context.subject_id,
            tenant_id=context.tenant_id,
            credential_id=context.credential_id,
            authorization_revision=context.authorization_revision,
            roles=tuple(item.value for item in context.roles),
            scopes=tuple(item.value for item in context.scopes),
            issued_at=context.issued_at,
            expires_at=context.expires_at,
        )

    return router


__all__ = ["build_session_router"]
