"""Uniform error-envelope handlers for the Review API (split out of ``api.py``).

Every expected failure leaves the surface as the same structured envelope; the
two security-relevant rejections (unknown/foreign objects and reserved authority
fields) are also recorded on the denial ledger, without anchoring another
tenant's inventory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from .api_contracts import API_VERSION_V2
from .api_support import error_response, request_id, safe_issue
from .record_reads import (
    RECORD_UNAVAILABLE_CODE,
    RECORD_UNAVAILABLE_MESSAGE,
    record_unavailable,
)
from .review_service import ReviewServiceError
from .run_errors import RecordIncompleteError
from .review_store import ReviewStoreError
from .security_contracts import DenialReason, SecurityError, SecurityOperation

if TYPE_CHECKING:
    from .api import SecurityRuntime


def register_error_handlers(application: FastAPI, resolved: SecurityRuntime) -> None:
    """Install the shared exception handlers on one application."""

    @application.exception_handler(SecurityError)
    async def handle_security_error(
        request: Request,
        error: SecurityError,
    ) -> JSONResponse:
        return error_response(
            request=request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @application.exception_handler(ReviewServiceError)
    async def handle_service_error(
        request: Request,
        error: ReviewServiceError,
    ) -> JSONResponse:
        context = getattr(request.state, "auth_context", None)
        operation = getattr(request.state, "operation", None)
        if error.status_code == 404 and context is not None and operation is not None:
            # A missing, foreign or ungranted object is a rejection: record it without a
            # Case anchor so nothing about another tenant's inventory is written down.
            assert resolved.authorization is not None
            resolved.authorization.record_denial(
                request_id=request_id(request),
                operation=operation,
                reason=DenialReason.UNKNOWN_RESOURCE,
                actor_subject_id=context.subject_id,
                actor_tenant_id=context.tenant_id,
            )
        return error_response(
            request=request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @application.exception_handler(RecordIncompleteError)
    async def handle_incomplete_record(
        request: Request,
        error: RecordIncompleteError,
    ) -> JSONResponse:
        # A committed record that is missing an identity the read projection has to report
        # is as unreadable as a damaged one, and it gets the same fixed answer: the caller
        # still receives a JSON envelope, never a server traceback (AC-04.2).
        del error
        return error_response(
            request=request,
            status_code=record_unavailable().status_code,
            code=RECORD_UNAVAILABLE_CODE,
            message=RECORD_UNAVAILABLE_MESSAGE,
            details={},
        )

    @application.exception_handler(SQLAlchemyError)
    async def handle_storage_error(
        request: Request,
        error: SQLAlchemyError,
    ) -> JSONResponse:
        # A store outage is a dependency failure, not a caller error: the answer is a
        # structured 503 with no internal detail.
        return error_response(
            request=request,
            status_code=503,
            code="security_dependency_unavailable",
            message="The local review store is unavailable.",
            details={},
        )

    @application.exception_handler(ReviewStoreError)
    async def handle_store_error(
        request: Request,
        error: ReviewStoreError,
    ) -> JSONResponse:
        return error_response(
            request=request,
            status_code=503,
            code="security_dependency_unavailable",
            message="The local review store is unavailable.",
            details={},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        issues = [safe_issue(item) for item in error.errors()]
        reserved = any(item["type"] == "reserved_field_rejected" for item in issues)
        version_mismatch = any(
            item["location"][-1:] == ["api_version"] and item["type"] == "literal_error"
            for item in issues
        )
        if reserved and resolved is not None and resolved.authorization is not None:
            # A rejected authority field is a security event; the ledger covers it even
            # though no identity was resolved for this request.
            resolved.authorization.record_denial(
                request_id=request_id(request),
                operation=SecurityOperation.REVIEW_CREATE,
                reason=DenialReason.RESERVED_FIELD_REJECTED,
                actor_subject_id=None,
                actor_tenant_id=None,
            )
        if reserved:
            status_code, code, message = (
                422,
                "reserved_field_rejected",
                "The request carries a reserved authority field.",
            )
        elif version_mismatch:
            status_code, code, message = (
                409,
                "api_version_mismatch",
                f"The request API version does not match {API_VERSION_V2}.",
            )
        else:
            status_code, code, message = (
                422,
                "invalid_request",
                "The request does not match the Review API contract.",
            )
        return error_response(
            request=request,
            status_code=status_code,
            code=code,
            message=message,
            details={"issues": issues},
        )


__all__ = ["register_error_handlers"]
