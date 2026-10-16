"""Human-action surface: confirm/reject/cancel/resume on a review run (V1-06)."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response

from ..api_contracts import (
    HumanActionRequestV2,
    ResumeRequestV2,
    ReviewRunResponseV2,
)
from ..api_support import IDEMPOTENCY_HEADER_PARAMS, resolve_idempotency_key
from ..api_wiring import ApiWiring
from ..contracts import ApprovalActionType
from ..security_contracts import Scope, SecurityOperation


def _perform_action(
    wiring: ApiWiring,
    request: Request,
    run_id: str,
    action: ApprovalActionType,
    body: HumanActionRequestV2,
    idempotency_header: str | None,
) -> ReviewRunResponseV2:
    operation = {
        ApprovalActionType.CONFIRM: SecurityOperation.REVIEW_CONFIRM,
        ApprovalActionType.REJECT: SecurityOperation.REVIEW_REJECT,
        ApprovalActionType.CANCEL: SecurityOperation.REVIEW_CANCEL,
    }[action]
    required_scope = {
        ApprovalActionType.CONFIRM: Scope.REVIEW_DECIDE,
        ApprovalActionType.REJECT: Scope.REVIEW_DECIDE,
        ApprovalActionType.CANCEL: Scope.REVIEW_CANCEL,
    }[action]
    context = wiring.context(request, operation=operation, scope=required_scope)
    key = resolve_idempotency_key(idempotency_header, body.idempotency_key)
    resolved_body = body.model_copy(update={"idempotency_key": key})
    active = wiring.require_ready()
    assert active.service is not None
    # The grant is decided from the minimal identity read before any business record is
    # loaded: an action by a caller who may not touch this Case keeps its own 404 however
    # the stored record looks.
    wiring.ensure_case_still_granted(
        request,
        context,
        active.service.run_case_id(run_id, context=context),
        operation=operation,
    )
    return active.service.apply_action(
        run_id=run_id,
        action=action,
        request=resolved_body,
        context=context,
    )


def build_actions_router(wiring: ApiWiring) -> APIRouter:
    router = APIRouter()

    @router.post("/reviews/{run_id}/confirm", response_model=ReviewRunResponseV2)
    async def confirm_review(
        request: Request,
        run_id: str,
        body: HumanActionRequestV2,
        idempotency_header: str | None = Header(**IDEMPOTENCY_HEADER_PARAMS),
    ) -> ReviewRunResponseV2:
        return _perform_action(
            wiring,
            request,
            run_id,
            ApprovalActionType.CONFIRM,
            body,
            idempotency_header,
        )

    @router.post("/reviews/{run_id}/reject", response_model=ReviewRunResponseV2)
    async def reject_review(
        request: Request,
        run_id: str,
        body: HumanActionRequestV2,
        idempotency_header: str | None = Header(**IDEMPOTENCY_HEADER_PARAMS),
    ) -> ReviewRunResponseV2:
        return _perform_action(
            wiring, request, run_id, ApprovalActionType.REJECT, body, idempotency_header
        )

    @router.post("/reviews/{run_id}/cancel", response_model=ReviewRunResponseV2)
    async def cancel_review(
        request: Request,
        run_id: str,
        body: HumanActionRequestV2,
        idempotency_header: str | None = Header(**IDEMPOTENCY_HEADER_PARAMS),
    ) -> ReviewRunResponseV2:
        return _perform_action(
            wiring, request, run_id, ApprovalActionType.CANCEL, body, idempotency_header
        )

    @router.post("/reviews/{run_id}/resume", response_model=ReviewRunResponseV2)
    async def resume_review(
        request: Request,
        run_id: str,
        body: ResumeRequestV2,
        response: Response,
        idempotency_header: str | None = Header(**IDEMPOTENCY_HEADER_PARAMS),
    ) -> ReviewRunResponseV2:
        context = wiring.context(
            request,
            operation=SecurityOperation.REVIEW_RESUME,
            scope=Scope.REVIEW_RESUME,
        )
        key = resolve_idempotency_key(idempotency_header, body.idempotency_key)
        resolved_body = body.model_copy(update={"idempotency_key": key})
        active = wiring.require_ready()
        assert active.service is not None
        # The resume's own contract decides what a damaged record means, so the grant
        # pre-check asks for the Run's Case without reading the record at all.
        case_id = active.service.run_case_id(run_id, context=context)
        wiring.ensure_case_still_granted(
            request,
            context,
            case_id,
            operation=SecurityOperation.REVIEW_RESUME,
        )
        result, created = await active.service.resume_review(
            run_id,
            resolved_body,
            context=context,
            idempotency_key=key,
        )
        response.status_code = 202 if created else 200
        return result

    return router


__all__ = ["build_actions_router"]
