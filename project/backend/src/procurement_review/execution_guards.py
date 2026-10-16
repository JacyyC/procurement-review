"""The live-authority and lease guards shared by every protected persistence step."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import text

from . import admitted_reads
from .authorization import (
    AUTHORITY_CREDENTIAL_EXPIRED,
    AUTHORITY_CREDENTIAL_REVOKED,
    AUTHORITY_CREDENTIAL_UNKNOWN,
    AUTHORITY_GRANT_REVOKED,
    AUTHORITY_REVISION_CHANGED,
    AUTHORITY_SUBJECT_DISABLED,
    AUTHORITY_SUBJECT_UNROLED,
    live_execution_authority,
)
from .contracts import ReviewLifecycleStatus
from .run_errors import (
    GrantRevokedError,
    RunNotFoundError,
    StaleAuthorizationError,
)
from .run_records import GenerationIdentity
from .review_store import ReviewStore


INVALIDATION_GENERATION_MISSING = "generation_missing"
INVALIDATION_NO_ACTIVE_GENERATION = "no_active_generation"
INVALIDATION_SUPERSEDED = "superseded_generation"
INVALIDATION_LEASE_EXPIRED = "lease_expired"
INVALIDATION_DEADLINE_EXPIRED = "generation_deadline_exceeded"
INVALIDATION_RUN_NOT_PROCESSING = "run_not_processing"
INVALIDATION_CHECKPOINTS_CHANGED = "checkpoints_changed"

AUTHORITY_REASONS = frozenset(
    {
        AUTHORITY_CREDENTIAL_EXPIRED,
        AUTHORITY_CREDENTIAL_REVOKED,
        AUTHORITY_CREDENTIAL_UNKNOWN,
        AUTHORITY_GRANT_REVOKED,
        AUTHORITY_REVISION_CHANGED,
        AUTHORITY_SUBJECT_DISABLED,
        AUTHORITY_SUBJECT_UNROLED,
    }
)


class ExecutionInvalidated(BaseException):
    """This execution generation may no longer perform a protected step."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    @property
    def authority_lost(self) -> bool:
        """Whether the whole generation lost live authorization."""

        return self.reason in AUTHORITY_REASONS


def current_time(now: datetime | Callable[[], datetime]) -> datetime:
    return now() if callable(now) else now


class ExecutionGuards:
    """One guard boundary, bound to a store and the injected fault plan."""

    def __init__(self, store: ReviewStore, fault: Any | None = None) -> None:
        self._store = store
        self._fault = fault

    @staticmethod
    def generation_identity(
        connection: Any,
        *,
        run_id: str,
        generation: int,
    ) -> GenerationIdentity | None:
        """Read the write-once identity of one generation without taking row locks."""

        row = connection.execute(
            text(
                """
                SELECT r.tenant_id, r.case_id, r.status AS run_status,
                       g.subject_id, g.credential_id, COALESCE(g.authorization_revision, 0) AS authorization_revision,
                       g.status AS generation_status, g.lease_expires_at
                FROM review_run r
                LEFT JOIN review_run_generation g
                  ON g.run_id = r.run_id AND g.generation = :generation
                WHERE r.run_id = :run_id
                """
            ),
            {"run_id": run_id, "generation": generation},
        ).first()
        if row is None:
            return None
        return GenerationIdentity(*row)

    @staticmethod
    def require_live_authority(
        connection: Any,
        *,
        tenant_id: str,
        case_id: str,
        subject_id: str,
        credential_id: str,
        authorization_revision: int,
        now: datetime | Callable[[], datetime],
    ) -> None:
        """Run the one live-authority check on the caller's transaction."""

        reason = live_execution_authority(
            connection,
            tenant_id=tenant_id,
            subject_id=subject_id,
            credential_id=credential_id,
            case_id=case_id,
            authorization_revision=authorization_revision,
            now=now,
        )
        if reason is None:
            return
        if reason == AUTHORITY_GRANT_REVOKED:
            raise GrantRevokedError("the acting subject no longer holds a Case grant")
        if reason == AUTHORITY_REVISION_CHANGED:
            raise StaleAuthorizationError("the tenant authorization revision moved")
        raise ExecutionInvalidated(reason)

    @staticmethod
    def require_generation_authority(
        connection: Any,
        *,
        tenant_id: str,
        case_id: str,
        subject_id: str,
        credential_id: str,
        authorization_revision: int,
        now: datetime | Callable[[], datetime],
    ) -> None:
        """The same live check, reported as the loss of the whole generation."""

        reason = live_execution_authority(
            connection,
            tenant_id=tenant_id,
            subject_id=subject_id,
            credential_id=credential_id,
            case_id=case_id,
            authorization_revision=authorization_revision,
            now=now,
        )
        if reason is not None:
            raise ExecutionInvalidated(reason)

    def lock_active_generation(
        self,
        connection: Any,
        *,
        run_id: str,
        generation: int,
        now: datetime | Callable[[], datetime],
        owner_id: str | None = None,
    ) -> str | None:
        """Lock revision → grant → Run → generation, then refresh temporal facts."""
        identity = self.generation_identity(
            connection, run_id=run_id, generation=generation
        )
        if identity is None:
            raise RunNotFoundError(run_id)
        if identity.subject_id is None:
            raise ExecutionInvalidated(INVALIDATION_GENERATION_MISSING)
        self.require_generation_authority(
            connection,
            tenant_id=identity.tenant_id,
            case_id=identity.case_id,
            subject_id=identity.subject_id,
            credential_id=str(identity.credential_id),
            authorization_revision=identity.authorization_revision,
            now=now,
        )
        row = connection.execute(
            text("SELECT status FROM review_run WHERE run_id = :run_id FOR UPDATE"),
            {"run_id": run_id},
        ).first()
        if row is None:
            raise RunNotFoundError(run_id)
        if ReviewLifecycleStatus(row.status) is not ReviewLifecycleStatus.PROCESSING:
            return INVALIDATION_RUN_NOT_PROCESSING
        active = connection.execute(
            text(
                "SELECT generation, owner_id FROM review_run_generation "
                "WHERE run_id = :run_id AND status = 'active' FOR UPDATE"
            ),
            {"run_id": run_id},
        ).first()
        if active is None:
            return INVALIDATION_NO_ACTIVE_GENERATION
        if int(active.generation) != generation or (
            owner_id is not None and active.owner_id != owner_id
        ):
            return INVALIDATION_SUPERSEDED
        self.require_execution_time(
            connection, run_id=run_id, generation=generation, now=now
        )
        return None

    def require_execution_time(
        self,
        connection: Any,
        *,
        run_id: str,
        generation: int,
        now: datetime | Callable[[], datetime],
    ) -> None:
        """Recheck already-locked facts without acquiring any upstream lock.

        Safe after downstream Trace/audit writes; expiry raises to roll back the whole
        transaction. This also refreshes credential expiry after *all* lock waits.
        """
        identity = self.generation_identity(
            connection, run_id=run_id, generation=generation
        )
        limits = connection.execute(
            text(
                "SELECT lease_expires_at, deadline_at FROM review_run_generation "
                "WHERE run_id = :r AND generation = :g"
            ),
            {"r": run_id, "g": generation},
        ).first()
        if identity is None or limits is None:
            raise ExecutionInvalidated(INVALIDATION_GENERATION_MISSING)
        reason = live_execution_authority(
            connection,
            tenant_id=identity.tenant_id,
            case_id=identity.case_id,
            subject_id=identity.subject_id,
            credential_id=identity.credential_id,
            authorization_revision=identity.authorization_revision,
            now=now,
            lock=False,
        )
        if reason is not None:
            raise ExecutionInvalidated(reason)
        instant = current_time(now)
        if limits.deadline_at <= instant:
            raise ExecutionInvalidated(INVALIDATION_DEADLINE_EXPIRED)
        if limits.lease_expires_at <= instant:
            raise ExecutionInvalidated(INVALIDATION_LEASE_EXPIRED)

    def assert_generation_live(
        self,
        *,
        run_id: str,
        generation: int,
        now: datetime | Callable[[], datetime],
        owner_id: str | None = None,
        site: str,
    ) -> None:
        """Admit one actual read on behalf of the generation that is doing it."""

        with self._store.transaction() as connection:
            self.require_generation_live(
                connection,
                run_id=run_id,
                generation=generation,
                now=now,
                owner_id=owner_id,
                on_authorized=(
                    None
                    if self._fault is None
                    else lambda: self._fault.barrier(f"during_read_admission:{site}")
                ),
            )

    #: Admit a read on the caller's own transaction: the implementation lives in
    #: ``admitted_reads`` next to the reads it covers, and binding it here keeps the
    #: guard API the repository and the checkpoint store already call unchanged.
    require_generation_live = admitted_reads.require_generation_live


__all__ = [
    "AUTHORITY_REASONS",
    "ExecutionGuards",
    "ExecutionInvalidated",
    "INVALIDATION_CHECKPOINTS_CHANGED",
    "INVALIDATION_GENERATION_MISSING",
    "INVALIDATION_LEASE_EXPIRED",
    "INVALIDATION_DEADLINE_EXPIRED",
    "INVALIDATION_NO_ACTIVE_GENERATION",
    "INVALIDATION_RUN_NOT_PROCESSING",
    "INVALIDATION_SUPERSEDED",
]
