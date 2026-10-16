"""Product security contracts for the V1-03 local review runtime.

These contracts describe *authorization*, not business evidence.  They are kept in
their own versioned module so the frozen V1-01/V1-02 domain, rule and evaluation
contracts keep their exact semantics.

Nothing in this module stores or transports a credential value: a credential is
only ever represented here by its opaque ID and by server-side metadata.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .contracts import Checksum, Identifier, StrictContract

SECURITY_CONTRACT_VERSION = "review-security.v1"
AUTH_CONTEXT_VERSION = "auth-context.v1"
AUTHORIZATION_POLICY_VERSION = "review-authz.v1"
RUNTIME_SCOPE_VERSION = "runtime-scope.v1"
CREDENTIAL_VERSION = "local-credential.v1"
RESOURCE_REGISTRY_VERSION = "review-resource-registry.v1"

CREDENTIAL_DIGEST_ALGORITHM = "sha256"
CREDENTIAL_MIN_ENTROPY_BITS = 256
CREDENTIAL_DEFAULT_TTL_SECONDS = 86_400

Revision = Annotated[int, Field(strict=True, ge=1)]
Generation = Annotated[int, Field(strict=True, ge=1)]


class Scope(StrEnum):
    """Fixed operation scopes; roles only ever seed these, they do not replace them."""

    CASE_READ = "case:read"
    REVIEW_READ = "review:read"
    REVIEW_CREATE = "review:create"
    REVIEW_CANCEL = "review:cancel"
    REVIEW_RESUME = "review:resume"
    REVIEW_DECIDE = "review:decide"
    TRACE_READ = "trace:read"
    AUDIT_READ = "audit:read"


class RoleBinding(StrEnum):
    """Seed role combinations; every decision is still made on scopes and grants."""

    READER = "reader"
    OPERATOR = "operator"
    APPROVER = "approver"
    AUDITOR = "auditor"


_READER_SCOPES = frozenset({Scope.CASE_READ, Scope.REVIEW_READ})
_OPERATOR_SCOPES = _READER_SCOPES | frozenset(
    {Scope.REVIEW_CREATE, Scope.REVIEW_CANCEL, Scope.REVIEW_RESUME, Scope.TRACE_READ}
)
_APPROVER_SCOPES = _READER_SCOPES | frozenset({Scope.REVIEW_DECIDE, Scope.AUDIT_READ})
_AUDITOR_SCOPES = _READER_SCOPES | frozenset({Scope.TRACE_READ, Scope.AUDIT_READ})

ROLE_SCOPES: dict[RoleBinding, frozenset[Scope]] = {
    RoleBinding.READER: _READER_SCOPES,
    RoleBinding.OPERATOR: _OPERATOR_SCOPES,
    RoleBinding.APPROVER: _APPROVER_SCOPES,
    RoleBinding.AUDITOR: _AUDITOR_SCOPES,
}


def scopes_for_roles(roles: tuple[RoleBinding, ...]) -> frozenset[Scope]:
    """Return the union of the seeded role scopes; unknown roles contribute nothing."""

    merged: set[Scope] = set()
    for role in roles:
        merged |= ROLE_SCOPES[role]
    return frozenset(merged)


class CredentialState(StrEnum):
    """Server-side classification of one presented credential."""

    ACTIVE = "active"
    UNKNOWN = "unknown_credential"
    EXPIRED = "expired_credential"
    REVOKED = "revoked_credential"
    DISABLED_SUBJECT = "disabled_subject"
    MALFORMED = "malformed_credential"


class DenialReason(StrEnum):
    """Stable, non-reflective categories for access-denial records."""

    MISSING_CREDENTIAL = "missing_credential"
    UNKNOWN_CREDENTIAL = "unknown_credential"
    EXPIRED_CREDENTIAL = "expired_credential"
    REVOKED_CREDENTIAL = "revoked_credential"
    DISABLED_SUBJECT = "disabled_subject"
    MALFORMED_CREDENTIAL = "malformed_credential"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    CASE_NOT_AUTHORIZED = "case_not_authorized"
    UNKNOWN_RESOURCE = "unknown_resource"
    RESERVED_FIELD_REJECTED = "reserved_field_rejected"


class SecurityOperation(StrEnum):
    """Operation family recorded on a denial; it never contains a target payload."""

    HEALTH = "health"
    SESSION = "session"
    CASE_LIST = "case_list"
    CASE_DETAIL = "case_detail"
    REVIEW_LIST = "review_list"
    REVIEW_CREATE = "review_create"
    REVIEW_DETAIL = "review_detail"
    REVIEW_CONFIRM = "review_confirm"
    REVIEW_REJECT = "review_reject"
    REVIEW_CANCEL = "review_cancel"
    REVIEW_RESUME = "review_resume"
    TRACE_READ = "trace_read"
    AUDIT_READ = "audit_read"
    RETIRED_API = "retired_api"
    MCP_TOOL = "mcp_tool"
    RETRIEVAL = "retrieval"


class AuthContext(StrictContract):
    """Server-derived identity; it is never accepted from a request payload."""

    auth_context_version: Literal["auth-context.v1"] = AUTH_CONTEXT_VERSION
    subject_id: Identifier
    tenant_id: Identifier
    credential_id: Identifier
    authorization_revision: Revision
    roles: tuple[RoleBinding, ...]
    scopes: tuple[Scope, ...]
    issued_at: datetime
    expires_at: datetime

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("credential timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def roles_and_scopes_are_consistent(self) -> AuthContext:
        if not self.roles:
            raise ValueError("an authenticated context requires at least one role")
        if self.scopes != tuple(sorted(set(self.scopes), key=lambda item: item.value)):
            raise ValueError("auth context scopes must be unique and sorted")
        if set(self.scopes) - scopes_for_roles(self.roles):
            raise ValueError("auth context scopes must come from the seeded roles")
        if self.expires_at <= self.issued_at:
            raise ValueError("credential expiry must follow its issue time")
        return self

    def has_scope(self, scope: Scope) -> bool:
        return scope in self.scopes


class RuntimeScope(StrictContract):
    """Immutable per-execution scope built by the trusted composition root.

    The scope is derived from the resource registry for exactly one tenant/Case and
    one execution generation.  Model output, tool arguments and browser payloads can
    never widen it.
    """

    runtime_scope_version: Literal["runtime-scope.v1"] = RUNTIME_SCOPE_VERSION
    tenant_id: Identifier
    case_id: Identifier
    case_version: Identifier
    case_type: Identifier
    scenario: Identifier
    subject_id: Identifier
    credential_id: Identifier
    authorization_revision: Revision
    generation: Generation
    document_version: Identifier
    dataset_version: Identifier
    index_version: Identifier
    rules_version: Identifier
    case_document_ids: tuple[Identifier, ...]
    shared_document_ids: tuple[Identifier, ...]
    allowed_order_ids: tuple[Identifier, ...]
    allowed_supplier_ids: tuple[Identifier, ...]
    allowed_transaction_ids: tuple[Identifier, ...]
    issued_at: datetime
    deadline_at: datetime

    @field_validator("issued_at", "deadline_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("runtime scope timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def collections_are_unique_and_sorted(self) -> RuntimeScope:
        for name in (
            "case_document_ids",
            "shared_document_ids",
            "allowed_order_ids",
            "allowed_supplier_ids",
            "allowed_transaction_ids",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"runtime scope {name} must be unique and sorted")
        if set(self.case_document_ids) & set(self.shared_document_ids):
            raise ValueError("a document cannot be both case-owned and shared")
        if not self.case_document_ids and not self.shared_document_ids:
            raise ValueError("a runtime scope must allow at least one document")
        if not self.allowed_order_ids:
            raise ValueError("a runtime scope must allow at least one order")
        if self.deadline_at <= self.issued_at:
            raise ValueError("runtime scope deadline must follow its issue time")
        return self


class CredentialMetadata(StrictContract):
    """Non-secret server-side credential record returned by the maintenance CLI."""

    credential_version: Literal["local-credential.v1"] = CREDENTIAL_VERSION
    credential_id: Identifier
    subject_id: Identifier
    tenant_id: Identifier
    digest_algorithm: Literal["sha256"] = CREDENTIAL_DIGEST_ALGORITHM
    issued_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    revoked_reason: Identifier | None = None

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("credential timestamps must be timezone-aware")
        return value

    @field_validator("revoked_at")
    @classmethod
    def require_aware_revocation(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("revoked_at must be timezone-aware")
        return value


class AccessDenialEvent(StrictContract):
    """A rejection record that cannot leak another tenant's object details.

    Case-bound denials are readable with the Case audit permission; denials without a
    resolvable Case are only readable from the trusted local maintenance CLI.
    """

    denial_schema_version: Literal["access-denial.v1"] = "access-denial.v1"
    denial_id: Identifier
    sequence: Annotated[int, Field(strict=True, ge=1)]
    occurred_at: datetime
    request_id: Identifier
    operation: SecurityOperation
    reason: DenialReason
    actor_subject_id: Identifier | None = None
    actor_tenant_id: Identifier | None = None
    case_id: Identifier | None = None
    target_digest: Checksum | None = None

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value


class SecurityError(RuntimeError):
    """Expected authorization failure carrying a stable public contract."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        reason: DenialReason,
        operation: SecurityOperation,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.reason = reason
        self.operation = operation
        self.details = details or {}


def unauthenticated(
    reason: DenialReason, operation: SecurityOperation
) -> SecurityError:
    """Return the single 401 contract used for every unusable credential."""

    return SecurityError(
        status_code=401,
        code="unauthenticated",
        message="A valid local credential is required for this operation.",
        reason=reason,
        operation=operation,
    )


def forbidden(operation: SecurityOperation) -> SecurityError:
    """Return the 403 contract for an authenticated subject without the scope."""

    return SecurityError(
        status_code=403,
        code="insufficient_scope",
        message="The authenticated subject lacks the required scope.",
        reason=DenialReason.INSUFFICIENT_SCOPE,
        operation=operation,
    )


def not_found(
    operation: SecurityOperation,
    *,
    reason: DenialReason = DenialReason.CASE_NOT_AUTHORIZED,
    target_digest: str | None = None,
) -> SecurityError:
    """Return the single 404 contract for missing, foreign, and ungranted objects.

    The same shape is used for every case so a caller cannot probe another tenant's
    object inventory through status codes, counts or error details.
    """

    details = {"target_digest": target_digest} if target_digest else None
    return SecurityError(
        status_code=404,
        code="resource_not_found",
        message="No accessible resource exists for the requested identifier.",
        reason=reason,
        operation=operation,
        details=details,
    )
