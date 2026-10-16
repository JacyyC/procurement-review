"""Shared helpers for the Review API surface (split out of ``api.py``, V1-06).

Everything here is free of runtime wiring: request IDs, the uniform error
envelope, local-only guards, idempotency-key resolution and the validation-issue
projection that never echoes rejected field names or input values.
"""

from __future__ import annotations

import os
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse

from .api_contracts import ErrorBody, ErrorEnvelope
from .review_service import ReviewServiceError

REVIEW_ALLOWED_ORIGINS_ENV = "REVIEW_ALLOWED_ORIGINS"
DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")
RESERVED_AUTHORITY_FIELDS = frozenset(
    {
        "actor",
        "human_actor",
        "initiator",
        "tenant",
        "tenant_id",
        "subject",
        "subject_id",
        "scopes",
        "scope",
        "authorization_revision",
        "credential",
        "credential_id",
        "token",
        "access_token",
    }
)

# The single Idempotency-Key header contract shared by every mutating endpoint.
IDEMPOTENCY_HEADER_PARAMS: dict[str, object] = {
    "default": None,
    "alias": "Idempotency-Key",
    "min_length": 1,
    "max_length": 160,
    "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
}


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", f"request-{uuid4().hex}")


def error_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=request_id(request),
            details=details or {},
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json"),
    )


def timing_override(name: str, default: float) -> float:
    """Read one local timing override; the frozen defaults stay in the contract."""

    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def allowed_origins() -> tuple[str, ...]:
    raw = os.environ.get(REVIEW_ALLOWED_ORIGINS_ENV)
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or DEFAULT_ALLOWED_ORIGINS


def host_is_local(value: str | None) -> bool:
    if not value:
        return True
    host, _, _port = value.rpartition(":")
    candidate = host or value
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    return candidate in DEFAULT_ALLOWED_HOSTS


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header is None:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def resolve_idempotency_key(
    header: str | None,
    body: str | None,
) -> str:
    if header is not None and body is not None and header != body:
        raise ReviewServiceError(
            status_code=409,
            code="idempotency_conflict",
            message="Header and body idempotency keys must match.",
        )
    key = header or body
    if key is None:
        raise ReviewServiceError(
            status_code=422,
            code="idempotency_key_required",
            message="An Idempotency-Key header or body idempotency_key is required.",
        )
    return key


def safe_issue(issue: object) -> dict[str, object]:
    """Project one validation issue without echoing rejected field names or input."""

    data = dict(issue) if isinstance(issue, dict) else {}
    location = [str(item) for item in data.get("loc", ())]
    issue_type = str(data.get("type", "validation_error"))
    if issue_type == "extra_forbidden":
        return {
            "location": ["body"],
            "type": "reserved_field_rejected",
            "message": "Reserved or unknown field rejected.",
        }
    if location and location[-1] in RESERVED_AUTHORITY_FIELDS:
        return {
            "location": [location[0]] if location else ["body"],
            "type": "reserved_field_rejected",
            "message": "Reserved authority field rejected.",
        }
    return {
        "location": location,
        "type": issue_type,
        "message": "Invalid request value.",
    }


__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "DEFAULT_ALLOWED_ORIGINS",
    "IDEMPOTENCY_HEADER_PARAMS",
    "RESERVED_AUTHORITY_FIELDS",
    "REVIEW_ALLOWED_ORIGINS_ENV",
    "allowed_origins",
    "bearer_token",
    "error_response",
    "host_is_local",
    "request_id",
    "resolve_idempotency_key",
    "safe_issue",
    "timing_override",
]
