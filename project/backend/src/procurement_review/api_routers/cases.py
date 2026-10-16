"""Case surface: only the Cases the credential is granted (V1-06)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..api_contracts import CaseListResponseV2, CaseOptionV2
from ..api_wiring import ApiWiring
from ..security_contracts import Scope, SecurityOperation


def build_cases_router(wiring: ApiWiring) -> APIRouter:
    router = APIRouter()

    @router.get("/cases", response_model=CaseListResponseV2)
    async def list_cases(request: Request) -> CaseListResponseV2:
        context = wiring.context(
            request,
            operation=SecurityOperation.CASE_LIST,
            scope=Scope.CASE_READ,
        )
        active = wiring.require_ready()
        assert active.store is not None
        registrations = active.store.list_granted_cases(
            tenant_id=context.tenant_id,
            subject_id=context.subject_id,
        )
        return CaseListResponseV2(
            cases=tuple(
                CaseOptionV2(
                    case_id=item.case_id,
                    case_version=item.case_version,
                    label=item.label,
                    case_type=item.case_type,
                    scenario=item.scenario,
                    document_version=item.document_version,
                    dataset_version=item.dataset_version,
                    index_version=item.index_version,
                    rules_version=item.rules_version,
                )
                for item in registrations
            )
        )

    return router


__all__ = ["build_cases_router"]
