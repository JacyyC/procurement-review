"""Review-run surface: list/create/read plus run-scoped trace and audit (V1-06)."""

from __future__ import annotations

from fastapi import APIRouter, Header, Request, Response

from ..api_contracts import (
    AuditResponseV2,
    ReviewCreateRequestV2,
    ReviewListResponseV2,
    ReviewRunResponseV2,
    TraceResponse,
)
from ..api_support import IDEMPOTENCY_HEADER_PARAMS, resolve_idempotency_key
from ..api_wiring import ApiWiring
from ..security_contracts import Scope, SecurityOperation


def build_reviews_router(wiring: ApiWiring) -> APIRouter:
    router = APIRouter()

    @router.get("/reviews", response_model=ReviewListResponseV2)
    async def list_reviews(request: Request) -> ReviewListResponseV2:
        context = wiring.context(
            request,
            operation=SecurityOperation.REVIEW_LIST,
            scope=Scope.REVIEW_READ,
        )
        active = wiring.require_ready()
        assert active.service is not None
        return active.service.list_reviews(
            context=context,
            granted_case_ids=wiring.granted_case_ids(context),
        )

    @router.post("/reviews", response_model=ReviewRunResponseV2, status_code=202)
    async def create_review(
        request: Request,
        body: ReviewCreateRequestV2,
        response: Response,
        idempotency_header: str | None = Header(**IDEMPOTENCY_HEADER_PARAMS),
    ) -> ReviewRunResponseV2:
        context = wiring.context(
            request,
            operation=SecurityOperation.REVIEW_CREATE,
            scope=Scope.REVIEW_CREATE,
        )
        key = resolve_idempotency_key(idempotency_header, body.idempotency_key)
        registration = wiring.case(
            request,
            context,
            body.case_id,
            operation=SecurityOperation.REVIEW_CREATE,
        )
        active = wiring.require_ready()
        assert active.authorization is not None and active.service is not None
        scope = active.authorization.build_runtime_scope(
            context,
            registration,
            generation=1,
            deadline_at=active.authorization.runtime_scope_deadline(30.0),
        )
        result, created = await active.service.create_review(
            body,
            registration=registration,
            scope=scope,
            context=context,
            idempotency_key=key,
        )
        response.status_code = 202 if created else 200
        response.headers["Location"] = f"/api/v2/reviews/{result.run_id}"
        return result

    @router.get("/reviews/{run_id}", response_model=ReviewRunResponseV2)
    async def get_review(request: Request, run_id: str) -> ReviewRunResponseV2:
        context = wiring.context(
            request,
            operation=SecurityOperation.REVIEW_DETAIL,
            scope=Scope.REVIEW_READ,
        )
        active = wiring.require_ready()
        assert active.service is not None
        # Identity, then tenant and Case grant, then the record: the object decision is
        # made from the minimal identity read, so a damaged record can neither become this
        # caller's answer nor tell them that an object they may not read is damaged.
        wiring.ensure_case_still_granted(
            request,
            context,
            active.service.run_case_id(run_id, context=context),
            operation=SecurityOperation.REVIEW_DETAIL,
        )
        return active.service.get_review(run_id, context=context)

    return router


def build_review_records_router(wiring: ApiWiring) -> APIRouter:
    router = APIRouter()

    @router.get("/reviews/{run_id}/trace", response_model=TraceResponse)
    async def get_trace(request: Request, run_id: str) -> TraceResponse:
        context = wiring.context(
            request,
            operation=SecurityOperation.TRACE_READ,
            scope=Scope.TRACE_READ,
        )
        active = wiring.require_ready()
        assert active.service is not None
        wiring.ensure_case_still_granted(
            request,
            context,
            active.service.run_case_id(run_id, context=context),
            operation=SecurityOperation.TRACE_READ,
        )
        return TraceResponse(trace=active.service.get_trace(run_id, context=context))

    @router.get("/reviews/{run_id}/audit", response_model=AuditResponseV2)
    async def get_audit(request: Request, run_id: str) -> AuditResponseV2:
        context = wiring.context(
            request,
            operation=SecurityOperation.AUDIT_READ,
            scope=Scope.AUDIT_READ,
        )
        active = wiring.require_ready()
        assert active.service is not None
        wiring.ensure_case_still_granted(
            request,
            context,
            active.service.run_case_id(run_id, context=context),
            operation=SecurityOperation.AUDIT_READ,
        )
        events, anomalies = active.service.get_audit_projection(run_id, context=context)
        return AuditResponseV2(
            run_id=run_id,
            events=events,
            legacy_sequence_anomalies=anomalies,
        )

    return router


__all__ = ["build_review_records_router", "build_reviews_router"]
