"""Server-side identity resolution and Case authorization for the V1-03 runtime."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from .review_store import (
    CaseRegistration,
    CredentialRow,
    ReviewStore,
    ReviewStoreError,
    digest_token,
)
from .security_contracts import (
    CREDENTIAL_DEFAULT_TTL_SECONDS,
    CREDENTIAL_MIN_ENTROPY_BITS,
    AccessDenialEvent,
    AuthContext,
    CredentialState,
    DenialReason,
    RoleBinding,
    RuntimeScope,
    Scope,
    SecurityError,
    SecurityOperation,
    forbidden,
    not_found,
    scopes_for_roles,
    unauthenticated,
)

TOKEN_PREFIX = "rvk1"
_TOKEN_SECRET_BYTES = 32
_ALLOWED_ROLES = {item.value for item in RoleBinding}
_VALID_SCOPES = frozenset(Scope)

logger = logging.getLogger("procurement_review.authorization")


class CredentialFormatError(RuntimeError):
    """Raised when a presented token is not a well-formed local credential."""


AUTHORITY_CREDENTIAL_UNKNOWN = "credential_unknown"
AUTHORITY_CREDENTIAL_REVOKED = "credential_revoked"
AUTHORITY_CREDENTIAL_EXPIRED = "credential_expired"
AUTHORITY_SUBJECT_DISABLED = "subject_disabled"
AUTHORITY_SUBJECT_UNROLED = "subject_unroled"
AUTHORITY_GRANT_REVOKED = "case_grant_revoked"
AUTHORITY_REVISION_CHANGED = "authorization_revision_changed"

#: The denial category each lapsed authority corresponds to, so a request whose authority
#: was withdrawn inside its own transaction answers like the gate would have.
AUTHORITY_DENIAL_REASON: dict[str, DenialReason] = {
    AUTHORITY_CREDENTIAL_UNKNOWN: DenialReason.UNKNOWN_CREDENTIAL,
    AUTHORITY_CREDENTIAL_REVOKED: DenialReason.REVOKED_CREDENTIAL,
    AUTHORITY_CREDENTIAL_EXPIRED: DenialReason.EXPIRED_CREDENTIAL,
    AUTHORITY_SUBJECT_DISABLED: DenialReason.DISABLED_SUBJECT,
    AUTHORITY_SUBJECT_UNROLED: DenialReason.INSUFFICIENT_SCOPE,
    AUTHORITY_GRANT_REVOKED: DenialReason.CASE_NOT_AUTHORIZED,
    AUTHORITY_REVISION_CHANGED: DenialReason.CASE_NOT_AUTHORIZED,
}


def generate_credential(credential_id: str | None = None) -> tuple[str, str, str]:
    """Return one opaque token, its credential ID, and its server-side digest."""

    resolved_id = credential_id or f"cred-{uuid.uuid4().hex}"
    secret = secrets.token_urlsafe(_TOKEN_SECRET_BYTES)
    if len(secret.encode("utf-8")) * 8 < CREDENTIAL_MIN_ENTROPY_BITS:
        raise CredentialFormatError("generated credential is below the entropy floor")
    token = f"{TOKEN_PREFIX}.{resolved_id}.{secret}"
    return token, resolved_id, digest_token(token)


def token_digest(token: str) -> str:
    """Return the digest of one presented token without retaining the value."""

    return digest_token(token)


def parse_token(token: str) -> str | None:
    """Return the credential ID embedded in a token, or ``None`` when malformed."""

    parts = token.split(".")
    if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
        return None
    credential_id, secret = parts[1], parts[2]
    if not credential_id or not secret:
        return None
    if not all(item.isalnum() or item in "-_" for item in credential_id):
        return None
    try:
        raw = base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4))
    except Exception:  # noqa: BLE001 - a malformed token is an expected input
        return None
    if not raw:
        return None
    return credential_id


@dataclass(frozen=True, slots=True)
class DenialContext:
    request_id: str
    operation: SecurityOperation
    occurred_at: datetime


class AuthorizationService:
    """Resolves the trusted AuthContext and enforces scope + Case grant."""

    def __init__(
        self,
        *,
        store: ReviewStore,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
        credential_ttl_seconds: int = CREDENTIAL_DEFAULT_TTL_SECONDS,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: f"denial-{uuid.uuid4().hex}")
        self._credential_ttl_seconds = credential_ttl_seconds
        self._denial_events: list[AccessDenialEvent] = []

    @property
    def store(self) -> ReviewStore:
        return self._store

    def now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("authorization clock must be timezone-aware")
        return value

    def credential_ttl(self) -> timedelta:
        return timedelta(seconds=self._credential_ttl_seconds)

    # -------------------------------------------------------------- resolution

    def resolve(
        self,
        token: str | None,
        *,
        request_id: str,
        operation: SecurityOperation,
    ) -> AuthContext:
        """Return the server-derived identity or raise the unified 401 contract."""

        if token is None or not token.strip():
            raise self._deny(
                request_id=request_id,
                operation=operation,
                reason=DenialReason.MISSING_CREDENTIAL,
            )
        if parse_token(token) is None:
            raise self._deny(
                request_id=request_id,
                operation=operation,
                reason=DenialReason.MALFORMED_CREDENTIAL,
            )
        try:
            state, row = self._store.classify_credential(
                token_digest(token), now=self.now()
            )
        except ReviewStoreError as error:
            raise self._dependency_unavailable(operation) from error
        except Exception as error:  # noqa: BLE001 - database outage is an expected state
            raise self._dependency_unavailable(operation) from error
        if state is not CredentialState.ACTIVE or row is None:
            raise self._deny(
                request_id=request_id,
                operation=operation,
                reason=DenialReason(state.value),
                row=None if state is CredentialState.UNKNOWN else row,
            )
        return self._context_from_row(row)

    def _context_from_row(self, row: CredentialRow) -> AuthContext:
        roles = tuple(RoleBinding(item) for item in row.roles if item in _ALLOWED_ROLES)
        if not roles:
            raise forbidden(SecurityOperation.SESSION)
        scopes = tuple(sorted(scopes_for_roles(roles), key=lambda item: item.value))
        return AuthContext(
            subject_id=row.subject_id,
            tenant_id=row.tenant_id,
            credential_id=row.credential_id,
            authorization_revision=row.authorization_revision,
            roles=roles,
            scopes=scopes,
            issued_at=row.issued_at,
            expires_at=row.expires_at,
        )

    def revalidate(
        self,
        context: AuthContext,
        *,
        request_id: str,
        operation: SecurityOperation,
    ) -> AuthContext:
        """Re-read the current identity and revision before a privileged commit."""

        try:
            state, row = self._store.classify_credential_by_id(
                context.credential_id, now=self.now()
            )
        except Exception as error:  # noqa: BLE001 - a database outage is an expected state
            raise self._dependency_unavailable(operation) from error
        if state is not CredentialState.ACTIVE or row is None:
            raise self._deny(
                request_id=request_id,
                operation=operation,
                reason=DenialReason(state.value),
                row=row,
            )
        return self._context_from_row(row)

    # ----------------------------------------------------------------- scopes

    def require_scope(
        self,
        context: AuthContext,
        scope: Scope,
        *,
        request_id: str,
        operation: SecurityOperation,
    ) -> None:
        if scope not in context.scopes:
            raise self._deny(
                request_id=request_id,
                operation=operation,
                reason=DenialReason.INSUFFICIENT_SCOPE,
                context=context,
                error=forbidden(operation),
            )

    def require_case(
        self,
        context: AuthContext,
        case_id: str,
        *,
        request_id: str,
        operation: SecurityOperation,
    ) -> CaseRegistration:
        """Return the Case registration only when the same tenant holds a live grant."""

        try:
            registration = self._store.load_case(
                tenant_id=context.tenant_id,
                case_id=case_id,
                subject_id=context.subject_id,
            )
        except Exception as error:  # noqa: BLE001 - a database outage is an expected state
            raise self._dependency_unavailable(operation) from error
        if registration is None:
            try:
                same_tenant = self._store.case_exists(
                    tenant_id=context.tenant_id, case_id=case_id
                )
            except Exception:  # noqa: BLE001 - only used to classify the denial record
                same_tenant = False
            raise self._deny(
                request_id=request_id,
                operation=operation,
                reason=(
                    DenialReason.CASE_NOT_AUTHORIZED
                    if same_tenant
                    else DenialReason.UNKNOWN_RESOURCE
                ),
                context=context,
                case_id=case_id if same_tenant else None,
                target_digest=_target_digest(case_id),
                error=not_found(
                    operation,
                    reason=(
                        DenialReason.CASE_NOT_AUTHORIZED
                        if same_tenant
                        else DenialReason.UNKNOWN_RESOURCE
                    ),
                ),
            )
        return registration

    def build_runtime_scope(
        self,
        context: AuthContext,
        registration: CaseRegistration,
        *,
        generation: int,
        deadline_at: datetime,
    ) -> RuntimeScope:
        issued_at = self.now()
        return RuntimeScope(
            tenant_id=registration.tenant_id,
            case_id=registration.case_id,
            case_version=registration.case_version,
            case_type=registration.case_type,
            scenario=registration.scenario,
            subject_id=context.subject_id,
            credential_id=context.credential_id,
            authorization_revision=context.authorization_revision,
            generation=generation,
            document_version=registration.document_version,
            dataset_version=registration.dataset_version,
            index_version=registration.index_version,
            rules_version=registration.rules_version,
            case_document_ids=registration.case_document_ids,
            shared_document_ids=registration.shared_document_ids,
            allowed_order_ids=registration.allowed_order_ids,
            allowed_supplier_ids=registration.allowed_supplier_ids,
            allowed_transaction_ids=registration.allowed_transaction_ids,
            issued_at=issued_at,
            deadline_at=deadline_at,
        )

    def runtime_scope_deadline(self, seconds: float) -> datetime:
        return self.now() + timedelta(seconds=seconds)

    # ---------------------------------------------------------------- denials

    def deny_not_found(
        self,
        *,
        request_id: str,
        operation: SecurityOperation,
        context: AuthContext,
        case_id: str,
        target_digest: str | None = None,
    ) -> SecurityError:
        """Record and return the single 404 contract for an ungranted object."""

        return self._deny(
            request_id=request_id,
            operation=operation,
            reason=DenialReason.CASE_NOT_AUTHORIZED,
            context=context,
            case_id=case_id,
            target_digest=target_digest,
            error=not_found(operation, reason=DenialReason.CASE_NOT_AUTHORIZED),
        )

    def _dependency_unavailable(self, operation: SecurityOperation) -> SecurityError:
        return SecurityError(
            status_code=503,
            code="security_dependency_unavailable",
            message="The local authorization store is unavailable.",
            reason=DenialReason.UNKNOWN_RESOURCE,
            operation=operation,
        )

    def _deny(
        self,
        *,
        request_id: str,
        operation: SecurityOperation,
        reason: DenialReason,
        context: AuthContext | None = None,
        row: CredentialRow | None = None,
        case_id: str | None = None,
        target_digest: str | None = None,
        error: SecurityError | None = None,
    ) -> SecurityError:
        raised = error or unauthenticated(reason, operation)
        subject_id = (
            context.subject_id if context else (row.subject_id if row else None)
        )
        tenant_id = context.tenant_id if context else (row.tenant_id if row else None)
        self.record_denial(
            request_id=request_id,
            operation=operation,
            reason=reason,
            actor_subject_id=subject_id,
            actor_tenant_id=tenant_id,
            case_id=case_id,
            target_digest=target_digest,
        )
        return raised

    def record_denial(
        self,
        *,
        request_id: str,
        operation: SecurityOperation,
        reason: DenialReason,
        actor_subject_id: str | None,
        actor_tenant_id: str | None,
        case_id: str | None = None,
        target_digest: str | None = None,
    ) -> AccessDenialEvent:
        """Best-effort denial ledger; a storage outage is reported, never masked."""

        event = AccessDenialEvent(
            denial_id=self._id_factory(),
            sequence=1,
            occurred_at=self.now(),
            request_id=request_id,
            operation=operation,
            reason=reason,
            actor_subject_id=actor_subject_id,
            actor_tenant_id=actor_tenant_id,
            case_id=case_id,
            target_digest=target_digest,
        )
        try:
            sequence = self._store.append_denial(
                denial_id=event.denial_id,
                occurred_at=event.occurred_at,
                request_id=event.request_id,
                operation=event.operation.value,
                reason=event.reason.value,
                actor_subject_id=event.actor_subject_id,
                actor_tenant_id=event.actor_tenant_id,
                case_id=event.case_id,
                target_digest=event.target_digest,
            )
        except Exception:  # noqa: BLE001 - the rejection itself must still happen
            sequence = None
        if sequence is not None:
            event = event.model_copy(update={"sequence": sequence})
        self._denial_events.append(event)
        # Diagnostic surface only: stable IDs and enum codes, never the presented
        # credential or any business payload.
        logger.info(
            "access denied: operation=%s reason=%s request_id=%s",
            event.operation.value,
            event.reason.value,
            event.request_id,
        )
        return event

    def local_denial_events(self) -> tuple[AccessDenialEvent, ...]:
        return tuple(self._denial_events)


def _target_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def live_execution_authority(
    connection: Any,
    *,
    tenant_id: str,
    subject_id: str,
    credential_id: str,
    case_id: str,
    authorization_revision: int,
    now: datetime | Callable[[], datetime],
    lock: bool = True,
) -> str | None:
    """Return the first reason why live execution authority no longer holds.

    One shared implementation for every protected step, so a read admission, an attempt
    booking, a heartbeat and a state commit cannot drift apart.  The caller supplies the
    connection, which keeps the check inside the caller's own transaction.

    ``review_authorization_revision`` is the single linearization point.  Every revocation
    entry point locks that row ``FOR UPDATE`` before it touches the credential, subject or
    grant rows, so holding ``FOR SHARE`` on it across a check and the write that follows it
    means a revocation either finishes first and is seen here, or waits for this
    transaction.  The credential and subject rows therefore need no lock of their own,
    which keeps the runtime role's privileges unchanged; the Case grant row keeps the share
    lock it has always taken.  A caller that must not lock at all — the read-only stdio MCP
    role — passes ``lock=False`` and still sees every committed revocation.
    """

    revision = connection.execute(
        text(
            "SELECT revision FROM review_authorization_revision "
            f"WHERE tenant_id = :t{' FOR SHARE' if lock else ''}"
        ),
        {"t": tenant_id},
    ).first()
    if revision is None or int(revision.revision) != authorization_revision:
        return AUTHORITY_REVISION_CHANGED
    identity = connection.execute(
        text(
            """
            SELECT c.revoked_at, c.expires_at, s.disabled,
                   (SELECT count(*) FROM review_subject_role r
                     WHERE r.tenant_id = c.tenant_id AND r.subject_id = c.subject_id)
                   AS roles
            FROM review_credential c
            JOIN review_subject s
              ON s.tenant_id = c.tenant_id AND s.subject_id = c.subject_id
            WHERE c.credential_id = :c AND c.tenant_id = :t AND s.subject_id = :s
            """
        ),
        {"c": credential_id, "t": tenant_id, "s": subject_id},
    ).first()
    if identity is None:
        return AUTHORITY_CREDENTIAL_UNKNOWN
    if identity.revoked_at is not None:
        return AUTHORITY_CREDENTIAL_REVOKED
    if identity.expires_at <= (now() if callable(now) else now):
        return AUTHORITY_CREDENTIAL_EXPIRED
    if bool(identity.disabled):
        return AUTHORITY_SUBJECT_DISABLED
    if int(identity.roles) == 0:
        return AUTHORITY_SUBJECT_UNROLED
    grant = connection.execute(
        text(
            "SELECT revoked_at FROM review_case_grant WHERE tenant_id = :t "
            f"AND case_id = :c AND subject_id = :s{' FOR SHARE' if lock else ''}"
        ),
        {"t": tenant_id, "c": case_id, "s": subject_id},
    ).first()
    if grant is None or grant.revoked_at is not None:
        return AUTHORITY_GRANT_REVOKED
    if identity.expires_at <= (now() if callable(now) else now):
        return AUTHORITY_CREDENTIAL_EXPIRED
    return None


def scope_admission(store: ReviewStore, scope: RuntimeScope) -> Callable[..., None]:
    """Return the live-authority gate for a process holding only a ``RuntimeScope``.

    The protected stdio entry point checks its scope once at startup, which is not the
    moment any record is read.  This gate is called by the read implementation at each
    tool entry, so a revocation that lands after startup still stops new reads.

    A preload that reads the database hands the gate its own connection, and the check then
    runs on that transaction; a tool read of an already-built index calls it with nothing
    and the gate owns a short transaction of its own.  The read-only stdio role takes no row
    lock either way, so it cannot block a revocation — it only has to see it.
    """

    def admit(connection: Any | None = None, *, site: str | None = None) -> None:
        del site
        if connection is not None:
            reason = live_execution_authority(
                connection,
                tenant_id=scope.tenant_id,
                subject_id=scope.subject_id,
                credential_id=scope.credential_id,
                case_id=scope.case_id,
                authorization_revision=scope.authorization_revision,
                now=datetime.now(UTC),
                lock=False,
            )
        else:
            with store.transaction() as owned:
                reason = live_execution_authority(
                    owned,
                    tenant_id=scope.tenant_id,
                    subject_id=scope.subject_id,
                    credential_id=scope.credential_id,
                    case_id=scope.case_id,
                    authorization_revision=scope.authorization_revision,
                    now=datetime.now(UTC),
                    lock=False,
                )
        if reason is not None:
            raise SecurityError(
                status_code=403,
                code="runtime_scope_not_authorized",
                message="The runtime scope is no longer authorized.",
                reason=DenialReason.CASE_NOT_AUTHORIZED,
                operation=SecurityOperation.MCP_TOOL,
            )

    return admit


def verify_scope_authorization(store: ReviewStore, scope: RuntimeScope) -> None:
    """Re-check one runtime scope against the live authorization store.

    Used by processes that hold a scope but no browser credential (the stdio MCP
    entry point).  It reuses the same registry and grant checks as the API instead of
    trusting the file that carried the scope.
    """

    registration = store.load_case(
        tenant_id=scope.tenant_id,
        case_id=scope.case_id,
        subject_id=scope.subject_id,
    )
    if registration is None:
        raise SecurityError(
            status_code=403,
            code="runtime_scope_not_authorized",
            message="The runtime scope is no longer authorized.",
            reason=DenialReason.CASE_NOT_AUTHORIZED,
            operation=SecurityOperation.MCP_TOOL,
        )
    if (
        registration.tenant_id != scope.tenant_id
        or registration.case_id != scope.case_id
    ):
        raise SecurityError(
            status_code=403,
            code="runtime_scope_not_authorized",
            message="The runtime scope is no longer authorized.",
            reason=DenialReason.CASE_NOT_AUTHORIZED,
            operation=SecurityOperation.MCP_TOOL,
        )
    if (
        registration.case_version != scope.case_version
        or registration.document_version != scope.document_version
        or registration.dataset_version != scope.dataset_version
        or registration.index_version != scope.index_version
    ):
        raise SecurityError(
            status_code=409,
            code="runtime_scope_version_mismatch",
            message="The registered Case versions no longer match the runtime scope.",
            reason=DenialReason.CASE_NOT_AUTHORIZED,
            operation=SecurityOperation.MCP_TOOL,
        )
    if (
        registration.allowed_order_ids != scope.allowed_order_ids
        or registration.allowed_supplier_ids != scope.allowed_supplier_ids
        or registration.allowed_transaction_ids != scope.allowed_transaction_ids
    ):
        raise SecurityError(
            status_code=403,
            code="runtime_scope_not_authorized",
            message="The runtime scope no longer matches the Case registration.",
            reason=DenialReason.CASE_NOT_AUTHORIZED,
            operation=SecurityOperation.MCP_TOOL,
        )


__all__ = [
    "AUTHORITY_CREDENTIAL_EXPIRED",
    "AUTHORITY_CREDENTIAL_REVOKED",
    "AUTHORITY_CREDENTIAL_UNKNOWN",
    "AUTHORITY_GRANT_REVOKED",
    "AUTHORITY_REVISION_CHANGED",
    "AUTHORITY_SUBJECT_DISABLED",
    "AUTHORITY_SUBJECT_UNROLED",
    "AuthorizationService",
    "CredentialFormatError",
    "DenialContext",
    "TOKEN_PREFIX",
    "generate_credential",
    "live_execution_authority",
    "parse_token",
    "scope_admission",
    "token_digest",
    "verify_scope_authorization",
]
