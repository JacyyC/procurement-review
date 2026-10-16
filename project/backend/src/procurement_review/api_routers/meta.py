"""Meta surface: the v2 health probe and the retired v1 catch-all (V1-06)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..api_contracts import API_VERSION_V2, HealthResponseV2
from ..api_wiring import ApiWiring


def build_retired_router() -> APIRouter:
    """The fixed, data-free 410 for every retired v1 path."""

    router = APIRouter(prefix="/api/v1")

    def _retired_response() -> JSONResponse:
        return JSONResponse(
            status_code=410,
            content={
                "error": {
                    "code": "api_version_retired",
                    "message": (
                        "review-api.v1 is retired; authenticate against review-api.v2."
                    ),
                    "request_id": "retired",
                    "details": {"current_api_version": API_VERSION_V2},
                }
            },
        )

    for retired_method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        suffix = retired_method.lower()
        router.add_api_route(
            "",
            _retired_response,
            methods=[retired_method],
            name=f"retired_api_version_root_{suffix}",
        )
        router.add_api_route(
            "/{retired_path:path}",
            _retired_response,
            methods=[retired_method],
            name=f"retired_api_version_path_{suffix}",
        )
    return router


def build_meta_router(wiring: ApiWiring) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponseV2)
    async def health() -> HealthResponseV2:
        return HealthResponseV2(
            security_store="not_configured" if wiring.failure_reason else "ready"
        )

    return router


__all__ = ["build_meta_router", "build_retired_router"]
