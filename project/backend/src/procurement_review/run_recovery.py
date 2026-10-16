"""The resume/recovery transaction of the persisted review run (FIX-03A split).

``resume_run`` decides one recovery in a single transaction: live authority, the existing
operation, the new-operation checks and every write they imply share it, so no window can
leave a Run that consumed budget without a generation, or an audit that claims a state
nobody committed.  It lives beside the rest of the repository because it owns exactly one
transaction of the same boundary; every value it writes is projected from a closed record
or a closed update type.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from sqlalchemy import text

from .contracts import ReviewLifecycleStatus
from .execution_guards import INVALIDATION_CHECKPOINTS_CHANGED
from .record_contracts import (
    GenerationStatusPatch,
    IdempotencyRecord,
    RunResumePatch,
    RunStatusPatch,
)
from .record_guard import guard_record
from .record_reads import verify_checkpoint_rows
from .run_audit import append_audit as append_audit_event
from .run_errors import (
    IllegalTransitionError,
    RecordUnavailableError,
    RepositoryError,
    RunNotFoundError,
    VersionConflictError,
)
from .run_idempotency import (
    checkpoints_digest,
    claim_operation,
    idempotency_key_digest,
    load_operation,
    resume_replay,
)
from .run_records import (
    IDEMPOTENT_APPLIED,
    IDEMPOTENT_FAILED,
    AuditDraft,
    ResumeOutcome,
    TraceDraft,
    load_run,
)
from .run_trace import append_trace as append_trace_events, trace_identity


class RecoveryTransactions:
    """The resume half of the repository boundary, shared by its one implementation.

    The class expects the hosting repository's ``_store``, ``_guards`` and optional
    ``_fault`` attributes and its ``_open_generation_locked`` step; it is mixed into
    :class:`~procurement_review.review_repository.ReviewRepository` and never used alone.
    """

    def resume_run(
        self,
        *,
        run_id: str,
        actor_subject_id: str,
        actor_credential_id: str,
        actor_authorization_revision: int,
        idempotency_key: str,
        request_digest: str,
        expected_version: int,
        owner_id: str,
        now: datetime,
        lease_seconds: float,
        deadline_seconds: float,
        resume_limit: int,
        incompatibility: str | None,
        expected_checkpoints_digest: str,
        audit: AuditDraft,
        failure_audit: AuditDraft,
        events: Sequence[TraceDraft] = (),
        rejection_events: Callable[[str], Sequence[TraceDraft]] = lambda code: (),
    ) -> ResumeOutcome:
        """Decide one resume in a single transaction.

        Live authority, the existing operation, the new-operation checks and every write
        they imply share this transaction, so no window can leave a Run that consumed
        budget without a generation, or an audit that claims a state nobody committed.
        """

        key_digest = idempotency_key_digest(idempotency_key)
        with self._store.transaction() as connection:
            unguarded = connection.execute(
                text(
                    "SELECT tenant_id, case_id FROM review_run WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            ).first()
            if unguarded is None:
                raise RunNotFoundError(run_id)
            self._guards.require_live_authority(
                connection,
                tenant_id=unguarded.tenant_id,
                case_id=unguarded.case_id,
                subject_id=actor_subject_id,
                credential_id=actor_credential_id,
                authorization_revision=actor_authorization_revision,
                now=now,
            )
            existing = load_operation(
                connection,
                tenant_id=unguarded.tenant_id,
                subject_id=actor_subject_id,
                family="resume",
                key_digest=key_digest,
            )
            if existing is not None:
                return resume_replay(connection, run_id, existing, request_digest)

            self._resume_barrier("before_run_lock")
            row = connection.execute(
                text("SELECT * FROM review_run WHERE run_id = :run_id FOR UPDATE"),
                {"run_id": run_id},
            ).first()
            if row is None:
                raise RunNotFoundError(run_id)
            # The Run lock serializes this Run; it says nothing about another Run that used
            # the same key, so the operation is re-read unconditionally here instead of only
            # when the revision moved.  A duplicate that committed while this transaction
            # waited must replay, and a duplicate that spent this key on another Run must
            # answer the contract conflict — never a version error and never a 503.
            committed = load_operation(
                connection,
                tenant_id=unguarded.tenant_id,
                subject_id=actor_subject_id,
                family="resume",
                key_digest=key_digest,
            )
            if committed is not None:
                return resume_replay(connection, run_id, committed, request_digest)
            if int(row.revision) != expected_version:
                raise VersionConflictError(
                    expected=expected_version, current=int(row.revision)
                )
            status = ReviewLifecycleStatus(row.status)
            if status is not ReviewLifecycleStatus.INTERRUPTED:
                raise IllegalTransitionError(action="resume", status=status.value)
            run = load_run(connection, run_id)
            assert run is not None
            try:
                # The adopted set is verified as complete stored rows before anything is
                # reused from it: a checkpoint that no longer satisfies its contract takes
                # the same permanent rejection as a set that moved under the decision.
                verify_checkpoint_rows(run.raw.checkpoints)
            except RecordUnavailableError:
                incompatibility = INVALIDATION_CHECKPOINTS_CHANGED
            if checkpoints_digest(run.checkpoints) != expected_checkpoints_digest:
                # The checkpoints moved under a decision computed before the lock; the
                # safe answer is the same permanent rejection, decided from these rows.
                incompatibility = INVALIDATION_CHECKPOINTS_CHANGED
            failure_code = incompatibility or (
                "resume_budget_exhausted"
                if int(row.resume_count) >= resume_limit
                else None
            )
            if failure_code is not None:
                return self._resume_rejected(
                    connection,
                    run_id=run_id,
                    row=row,
                    actor_subject_id=actor_subject_id,
                    key_digest=key_digest,
                    request_digest=request_digest,
                    failure_code=failure_code,
                    now=now,
                    audit=failure_audit,
                    events=rejection_events(failure_code),
                )

            self._resume_barrier("before_claim")
            mapping = guard_record(
                "idempotency",
                {
                    "tenant_id": unguarded.tenant_id,
                    "subject_id": actor_subject_id,
                    "operation_family": "resume",
                    "key_digest": key_digest,
                    "request_digest": request_digest,
                    "run_id": run_id,
                    "action": "resume",
                    "receipt_id": None,
                    "first_revision": int(row.revision),
                    "outcome": IDEMPOTENT_APPLIED,
                    "failure_code": None,
                    "created_at": now,
                },
                location="idempotency",
            )
            assert isinstance(mapping, IdempotencyRecord)
            # The claim — not the read above — is what makes this the one operation of the
            # key: another Run may have committed it between that read and this statement,
            # and only the unique namespace can arbitrate that window.  Nothing has been
            # written yet, so a loser leaves no budget, generation, audit or trace behind.
            if not claim_operation(connection, mapping):
                committed = load_operation(
                    connection,
                    tenant_id=unguarded.tenant_id,
                    subject_id=actor_subject_id,
                    family="resume",
                    key_digest=key_digest,
                )
                if committed is None:
                    raise RepositoryError(
                        "the resume key could not be claimed or read back"
                    )
                return resume_replay(connection, run_id, committed, request_digest)
            self._resume_barrier("idempotency")
            resume_patch = guard_record(
                "run_resume",
                {"current_subject_id": actor_subject_id, "updated_at": now},
                location="run_resume",
            )
            assert isinstance(resume_patch, RunResumePatch)
            connection.execute(
                text(
                    "UPDATE review_run SET resume_count = resume_count + 1, "
                    "current_subject_id = :subject, updated_at = :now "
                    "WHERE run_id = :run_id"
                ),
                {
                    "run_id": run_id,
                    "subject": resume_patch.current_subject_id,
                    "now": resume_patch.updated_at,
                },
            )
            self._resume_barrier("resume_count")
            generation = self._open_generation_locked(
                connection,
                run_id=run_id,
                subject_id=actor_subject_id,
                credential_id=actor_credential_id,
                authorization_revision=actor_authorization_revision,
                owner_id=owner_id,
                now=now,
                lease_seconds=lease_seconds,
                deadline_seconds=deadline_seconds,
            )
            self._resume_barrier("generation")
            self._resume_barrier("before_audit")
            append_audit_event(
                connection, run_id=run_id, draft=audit, fault=self._fault
            )
            append_trace_events(
                connection,
                run_id=run_id,
                trace_id=trace_identity(connection, run_id),
                drafts=events,
                now=now,
            )
            self._resume_barrier("after_audit")
            committed = load_run(connection, run_id)
            assert committed is not None
            self._resume_barrier("before_commit")
            return ResumeOutcome(
                run=committed, kind=IDEMPOTENT_APPLIED, generation=generation.generation
            )

    def _resume_rejected(
        self,
        connection: Any,
        *,
        run_id: str,
        row: Any,
        actor_subject_id: str,
        key_digest: str,
        request_digest: str,
        failure_code: str,
        now: datetime,
        audit: AuditDraft,
        events: Sequence[TraceDraft],
    ) -> ResumeOutcome:
        """Decide a permanently unrecoverable resume without a second transaction."""

        audit = replace(audit, detail_code=failure_code)
        revision = int(row.revision) + 1
        message = "The review cannot be resumed; create a new review."
        rejected = guard_record(
            "run_status",
            {
                "status": ReviewLifecycleStatus.FAILED,
                "error_code": failure_code,
                "error_message": message,
                "current_subject_id": actor_subject_id,
            },
            location="run_status",
        )
        assert isinstance(rejected, RunStatusPatch)
        cancelled = guard_record(
            "generation_status", {"status": "cancelled"}, location="generation_status"
        )
        assert isinstance(cancelled, GenerationStatusPatch)
        mapping = guard_record(
            "idempotency",
            {
                "tenant_id": row.tenant_id,
                "subject_id": actor_subject_id,
                "operation_family": "resume",
                "key_digest": key_digest,
                "request_digest": request_digest,
                "run_id": run_id,
                "action": "resume",
                "receipt_id": None,
                "first_revision": int(row.revision),
                "outcome": IDEMPOTENT_FAILED,
                "failure_code": failure_code,
                "created_at": now,
            },
            location="idempotency",
        )
        assert isinstance(mapping, IdempotencyRecord)
        # A permanent rejection is one operation of the subject's namespace exactly like a
        # successful resume is, so it claims the key first and only then rejects this Run.
        # Claiming after the state change would let one Run be failed by another Run's key.
        if not claim_operation(connection, mapping):
            committed = load_operation(
                connection,
                tenant_id=row.tenant_id,
                subject_id=actor_subject_id,
                family="resume",
                key_digest=key_digest,
            )
            if committed is None:
                raise RepositoryError(
                    "the resume key could not be claimed or read back"
                )
            return resume_replay(connection, run_id, committed, request_digest)
        connection.execute(
            text(
                "UPDATE review_run SET status = 'failed', revision = :revision, "
                "error_code = :code, error_message = :message, "
                "current_subject_id = :subject, updated_at = :now "
                "WHERE run_id = :run_id"
            ),
            {
                "run_id": run_id,
                "revision": revision,
                "code": rejected.error_code,
                "message": rejected.error_message,
                "subject": rejected.current_subject_id,
                "now": now,
            },
        )
        connection.execute(
            text(
                "UPDATE review_run_generation SET status = :status "
                "WHERE run_id = :run_id AND status = 'active'"
            ),
            {"run_id": run_id, "status": cancelled.status},
        )
        self._resume_barrier("idempotency")
        append_audit_event(connection, run_id=run_id, draft=audit, fault=self._fault)
        append_trace_events(
            connection,
            run_id=run_id,
            trace_id=trace_identity(connection, run_id),
            drafts=events,
            now=now,
        )
        committed = load_run(connection, run_id)
        assert committed is not None
        self._resume_barrier("before_commit")
        return ResumeOutcome(
            run=committed, kind=IDEMPOTENT_FAILED, failure_code=failure_code
        )

    def _resume_barrier(self, stage: str) -> None:
        """Reach one stage of the resume transaction when a plan asks for it."""

        if self._fault is not None:
            self._fault.barrier(f"during_resume_commit:{stage}")
