"""Authenticated local Review API (``review-api.v2``).

The v1 business surface is retired: it only returns a fixed, data-free 410 so an old
client cannot reach an unauthenticated path.  Every v2 business endpoint resolves a
server-side identity first, then a scope, then a Case grant, and only then reads or
mutates persisted review state.

``create_app`` is only the assembly point (V1-06, AC-07): the routers live per
resource in ``api_routers/``, the shared guard chain in ``api_wiring.py``, the
error envelopes in ``api_error_handlers.py`` and the pure helpers in
``api_support.py``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from .api_contracts import API_VERSION_V2
from .api_error_handlers import register_error_handlers
from .api_routers import (
    build_actions_router,
    build_cases_router,
    build_meta_router,
    build_retired_router,
    build_review_records_router,
    build_reviews_router,
    build_session_router,
)
from .api_support import allowed_origins, error_response, host_is_local, timing_override
from .api_wiring import ApiWiring
from .authorization import AuthorizationService
from .case_catalog import CASE_CATALOG
from .execution_profile import RealProfileProblem, execution_profile
from .fault_hooks import FaultController
from .logging_setup import configure_logging
from .review_repository import ReviewRepository
from .review_service import (
    HEARTBEAT_SECONDS,
    LEASE_SECONDS,
    SCAN_INTERVAL_SECONDS,
    ReviewService,
)
from .review_store import ReviewStore

# The frozen V1-01 case catalog remains importable from this module for the offline
# contracts that replay the accepted nine scenarios.
CASE_OPTIONS = CASE_CATALOG

REVIEW_DATABASE_URL_ENV = "REVIEW_DATABASE_URL"

logger = logging.getLogger("procurement_review.api")


class SecurityRuntime:
    """The composition root: store, repository, authorization and review service.

    Every member is optional because the app must still start without a configured
    store (``/health`` and the retired v1 surface stay observable); ``failure`` then
    carries the reason and every business endpoint fails closed with 503.
    """

    def __init__(
        self,
        *,
        store: ReviewStore | None = None,
        authorization: AuthorizationService | None = None,
        repository: ReviewRepository | None = None,
        service: ReviewService | None = None,
        failure: str | None = None,
    ) -> None:
        self.store = store
        self.authorization = authorization
        self.repository = repository
        self.service = service
        self.failure = failure

    @classmethod
    def build(cls, store: ReviewStore) -> SecurityRuntime:
        authorization = AuthorizationService(store=store)
        fault = FaultController.from_environment()
        repository = ReviewRepository(store, fault=fault)
        profile = execution_profile()
        if isinstance(profile, RealProfileProblem):
            # A real profile that is missing its endpoint, identities, credential or an
            # explicit deadline is a controlled failure: the workbench does not fall back
            # to the synthetic line, and it says which profile failed to configure.
            return cls.unavailable(profile.reason)
        service = ReviewService(
            repository=repository,
            runner=profile.runner(store),
            fault=fault,
            lease_seconds=timing_override("REVIEW_LEASE_SECONDS", LEASE_SECONDS),
            heartbeat_seconds=timing_override(
                "REVIEW_HEARTBEAT_SECONDS", HEARTBEAT_SECONDS
            ),
            scan_interval_seconds=timing_override(
                "REVIEW_SCAN_INTERVAL_SECONDS", SCAN_INTERVAL_SECONDS
            ),
            generation_deadline_seconds=profile.generation_deadline_seconds,
        )
        return cls(
            store=store,
            authorization=authorization,
            repository=repository,
            service=service,
        )

    @classmethod
    def from_environment(cls) -> SecurityRuntime | None:
        database_url = os.environ.get(REVIEW_DATABASE_URL_ENV)
        if not database_url:
            return None
        try:
            store = ReviewStore.from_url(database_url)
            store.ensure_migrated()
            return cls.build(store)
        except Exception:  # noqa: BLE001 - fail closed instead of serving unauthenticated
            return cls.unavailable("review_store_not_ready")

    @classmethod
    def unavailable(
        cls, reason: str = "review_database_url_not_configured"
    ) -> SecurityRuntime:
        """The typed stand-in for an unconfigured store; no constructor bypass."""

        return cls(failure=reason)

    async def shutdown(self) -> None:
        if self.service is not None:
            await self.service.shutdown()
        if self.store is not None:
            self.store.close()


def create_app(
    *,
    service: ReviewService | None = None,
    runtime: SecurityRuntime | None = None,
    store: ReviewStore | None = None,
    authorization: AuthorizationService | None = None,
    repository: ReviewRepository | None = None,
) -> FastAPI:
    """Build the authenticated app.

    Without an explicit runtime the app reads ``REVIEW_DATABASE_URL``.  When no
    security store is available the app still starts (so ``/health`` and the retired v1
    surface stay observable) but every business endpoint fails closed with 503 — there
    is no in-memory fallback path.
    """

    configure_logging()

    resolved: SecurityRuntime | None = runtime
    if resolved is None and store is not None:
        built = SecurityRuntime.build(store)
        if authorization is not None:
            built.authorization = authorization
        if repository is not None:
            built.repository = repository
        if service is not None:
            built.service = service
        resolved = built
    if resolved is None:
        resolved = SecurityRuntime.from_environment()
    if resolved is None:
        resolved = SecurityRuntime.unavailable()

    origins = allowed_origins()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.runtime = resolved
        logger.info(
            "api startup: security_store=%s",
            "not_configured" if resolved.failure else "ready",
        )
        if resolved.service is not None:
            resolved.service.start_scanner()
        try:
            yield
        finally:
            await resolved.shutdown()
            logger.info("api shutdown complete")

    application = FastAPI(
        title="Local Procurement Review API",
        summary="Test-only review suggestions and explicit local human decisions.",
        description=(
            "This API runs deterministic local fixtures behind local credentials. "
            "It does not perform a real procurement approval or write to an external "
            "business system."
        ),
        version=API_VERSION_V2,
        docs_url="/api/v2/docs",
        redoc_url=None,
        openapi_url="/api/v2/openapi.json",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
        expose_headers=["X-Request-ID", "X-Execution-Mode"],
        max_age=600,
    )

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = f"request-{uuid4().hex}"
        if not host_is_local(request.headers.get("host")):
            return error_response(
                request=request,
                status_code=403,
                code="local_origin_required",
                message="This local service only accepts loopback requests.",
            )
        origin = request.headers.get("origin")
        if origin is not None and origin not in origins:
            return error_response(
                request=request,
                status_code=403,
                code="local_origin_required",
                message="This local service only accepts the configured workbench origin.",
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Execution-Mode"] = "test_only"
        return response

    register_error_handlers(application, resolved)
    application.include_router(build_retired_router())

    wiring = ApiWiring(resolved)
    for router in (
        build_meta_router(wiring),
        build_session_router(wiring),
        build_cases_router(wiring),
        build_reviews_router(wiring),
        build_actions_router(wiring),
        build_review_records_router(wiring),
    ):
        application.include_router(router, prefix="/api/v2")
    return application


app = create_app()


def main() -> None:
    """Start the local authenticated API development server."""

    import uvicorn

    uvicorn.run(
        "procurement_review.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()


__all__: Iterable[str] = (
    "REVIEW_DATABASE_URL_ENV",
    "SecurityRuntime",
    "create_app",
)
