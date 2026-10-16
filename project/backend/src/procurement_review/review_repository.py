"""PostgreSQL repository for the persisted V1-03 review run."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text

from .execution_guards import (
    current_time,
    AUTHORITY_REASONS,
    ExecutionGuards,
    ExecutionInvalidated,
    INVALIDATION_CHECKPOINTS_CHANGED,
    INVALIDATION_GENERATION_MISSING,
    INVALIDATION_LEASE_EXPIRED,
    INVALIDATION_NO_ACTIVE_GENERATION,
    INVALIDATION_RUN_NOT_PROCESSING,
    INVALIDATION_SUPERSEDED,
)
from .field_guard import scan_forbidden_keys, scan_payload
from .contracts import ApprovalActionType, ReviewLifecycleStatus, WorkerKind
from .api_contracts import ReceiptViewV2
from .audit_contracts import CheckpointRecord
from .record_contracts import (
    AttemptAbandonPatch,
    AttemptCompletionPatch,
    AttemptDiscardPatch,
    CheckpointPayload,
    GenerationHeartbeatPatch,
    GenerationRecord,
    GenerationStatusPatch,
    IdempotencyRecord,
    ReceiptRecord,
    RunRecordWithTrace,
    RunStatusPatch,
)
from .record_guard import guard_record, validate_stored_record
from .record_reads import (
    verify_audit_records,
    verify_checkpoint_rows,
    verify_run_record,
    verify_trace_records,
)
from .record_rows import json_value, row_mapping
from .run_audit import append_audit as append_audit_event
from .run_idempotency import (
    checkpoints_digest,
    claim_operation,
    create_replay,
    idempotency_key_digest,
    load_operation,
)
from .run_recovery import RecoveryTransactions
from .run_trace import (
    append_trace as append_trace_events,
    envelope as trace_envelope_view,
    lock_run,
    trace_identity,
)
from .run_errors import (
    FieldGuardError,
    GrantRevokedError,
    IdempotencyConflictError,
    IllegalTransitionError,
    RepositoryError,
    RunNotFoundError,
    StaleAuthorizationError,
    SummaryChecksumError,
    VersionConflictError,
)
from .run_records import (
    ACTIVE_GENERATION_STATUS,
    AuditDraft,
    TraceDraft,
    IDEMPOTENT_APPLIED,
    IDEMPOTENT_FAILED,
    IDEMPOTENT_REPLAY,
    PENDING_STATUSES,
    TERMINAL_STATUSES,
    ActionOutcome,
    AttemptRecord,
    BranchCommit,
    CreateOutcome,
    GenerationIdentity,
    PersistedRun,
    ResumeOutcome,
    RunAuthorizationIdentity,
    RunGeneration,
    RunHeader,
    _json,
    load_run,
)
from .review_store import ReviewStore
from .trace import canonical_checksum

# Diagnostic surface only (V1-06, AC-08): stable IDs, never business payloads.
logger = logging.getLogger("procurement_review.review_repository")


def guard_payload(payload: object, *, location: str) -> None:
    """Screen one ad-hoc payload with the raw value and key patterns."""

    report = scan_payload(payload, location=location)
    report = (
        scan_forbidden_keys(payload, location=location)
        if not report.findings
        else report
    )
    if report.findings:
        categories = sorted({item.category for item in report.findings})
        # Categories and the caller's guard location are fixed names; the JSON path
        # and matched text never reach the log (AC-08: IDs and codes only).
        logger.info(
            "field guard rejected: location=%s categories=%s",
            location,
            ",".join(categories),
        )
        raise FieldGuardError(
            f"the {location} payload violates the field guard: {', '.join(categories)}"
        )


class ReviewRepository(RecoveryTransactions):
    """Narrow persistence boundary; every method owns exactly one transaction."""

    def __init__(self, store: ReviewStore, *, fault: Any | None = None) -> None:
        self._store = store
        self._fault = (
            fault if fault is not None and getattr(fault, "enabled", False) else None
        )
        self._guards = ExecutionGuards(store, self._fault)

    @property
    def store(self) -> ReviewStore:
        return self._store

    def assert_generation_live(
        self,
        *,
        run_id: str,
        generation: int,
        now: datetime | Callable[[], datetime],
        owner_id: str | None = None,
        site: str,
    ) -> None:
        """Admit one actual read on behalf of the generation performing it."""

        self._guards.assert_generation_live(
            run_id=run_id, generation=generation, now=now, owner_id=owner_id, site=site
        )

    def require_generation_live(
        self,
        connection: Any,
        *,
        run_id: str,
        generation: int,
        now: datetime | Callable[[], datetime],
        owner_id: str | None = None,
    ) -> None:
        """Admit a protected local read on the caller's transaction."""

        self._guards.require_generation_live(
            connection, run_id=run_id, generation=generation, now=now, owner_id=owner_id
        )

    def load_run(self, run_id: str) -> PersistedRun | None:
        with self._store.transaction() as connection:
            return load_run(connection, run_id)

    def list_runs(
        self,
        *,
        tenant_id: str,
        case_ids: Sequence[str] | None = None,
        limit: int = 200,
    ) -> tuple[PersistedRun, ...]:
        with self._store.transaction() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT run_id FROM review_run
                    WHERE tenant_id = :t
                      AND (:all_cases OR case_id = ANY(:cases))
                    ORDER BY created_at DESC, run_id DESC
                    LIMIT :limit
                    """
                ),
                {
                    "t": tenant_id,
                    "all_cases": case_ids is None,
                    "cases": list(case_ids or ()),
                    "limit": limit,
                },
            ).scalars()
            return tuple(
                run
                for run in (
                    load_run(connection, run_id, with_events=False)
                    for run_id in tuple(rows)
                )
                if run is not None
            )

    def run_authorization_identity(
        self, run_id: str
    ) -> RunAuthorizationIdentity | None:
        """The metadata one access decision needs, and nothing else."""

        with self._store.transaction() as connection:
            row = connection.execute(
                text(
                    "SELECT tenant_id, case_id FROM review_run WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            ).first()
        if row is None:
            return None
        return RunAuthorizationIdentity(
            run_id=run_id, tenant_id=row.tenant_id, case_id=row.case_id
        )

    def attempt_totals(self, run_id: str) -> dict[str, int]:
        with self._store.transaction() as connection:
            rows = connection.execute(
                text(
                    "SELECT worker, count(*) AS total FROM review_worker_attempt "
                    "WHERE run_id = :run_id GROUP BY worker"
                ),
                {"run_id": run_id},
            ).all()
        return {row.worker: int(row.total) for row in rows}

    def create_run(
        self,
        *,
        run_id: str,
        trace_id: str,
        tenant_id: str,
        case_id: str,
        case_version: str,
        subject_id: str,
        credential_id: str,
        authorization_revision: int,
        request_checksum: str,
        document_version: str,
        dataset_version: str,
        index_version: str,
        rules_version: str,
        isolation: Mapping[str, Any],
        idempotency_key: str,
        request_digest: str,
        owner_id: str,
        now: datetime,
        lease_seconds: float,
        deadline_seconds: float,
        events: Sequence[TraceDraft],
    ) -> CreateOutcome:
        key_digest = idempotency_key_digest(idempotency_key)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        deadline_at = now + timedelta(seconds=deadline_seconds)
        # Validate all records, including idempotency and Trace identity, before opening the transaction.
        isolated = guard_record(
            "isolation",
            {"case_type": "purchase_order", **dict(isolation)},
            location="isolation",
            strict=False,
        )
        created = guard_record(
            "run",
            {
                "run_id": run_id,
                "tenant_id": tenant_id,
                "case_id": case_id,
                "case_version": case_version,
                "status": ReviewLifecycleStatus.PROCESSING,
                "revision": 1,
                "resume_count": 0,
                "created_by_subject": subject_id,
                "current_subject_id": subject_id,
                "request_checksum": request_checksum,
                "document_version": document_version,
                "dataset_version": dataset_version,
                "index_version": index_version,
                "rules_version": rules_version,
                "created_at": now,
                "updated_at": now,
                "error_code": None,
                "error_message": None,
                "trace_id": trace_id,
            },
            location="run",
        )
        assert isinstance(created, RunRecordWithTrace)
        generation_record = guard_record(
            "generation",
            {
                "run_id": run_id,
                "generation": 1,
                "subject_id": subject_id,
                "credential_id": credential_id,
                "authorization_revision": authorization_revision,
                "owner_id": owner_id,
                "status": ACTIVE_GENERATION_STATUS,
                "heartbeat_at": now,
                "lease_expires_at": lease_expires_at,
                "deadline_at": deadline_at,
                "created_at": now,
            },
            location="generation",
        )
        assert isinstance(generation_record, GenerationRecord)
        create_mapping = guard_record(
            "idempotency",
            {
                "tenant_id": tenant_id,
                "subject_id": subject_id,
                "operation_family": "create",
                "key_digest": key_digest,
                "request_digest": request_digest,
                "run_id": run_id,
                "action": None,
                "receipt_id": None,
                "first_revision": created.revision,
                "outcome": IDEMPOTENT_APPLIED,
                "failure_code": None,
                "created_at": now,
            },
            location="idempotency",
        )
        assert isinstance(create_mapping, IdempotencyRecord)
        with self._store.transaction() as connection:
            # Authorize before creating or locating a replayed Run.
            self._guards.require_live_authority(
                connection,
                tenant_id=tenant_id,
                case_id=case_id,
                subject_id=subject_id,
                credential_id=credential_id,
                authorization_revision=authorization_revision,
                now=now,
            )
            existing = load_operation(
                connection,
                tenant_id=tenant_id,
                subject_id=subject_id,
                family="create",
                key_digest=key_digest,
            )
            if existing is not None:
                return create_replay(connection, existing, request_digest)
            if self._fault is not None:
                # Test-only: park every contender that has already read "no mapping",
                # so the claim below runs against a real race instead of a lucky one.
                self._fault.barrier("during_create_claim")
            # Claim before inserting the Run; uniqueness arbitration sends the loser to replay.
            if not claim_operation(connection, create_mapping):
                # Read the committed winner; the authorized request digest decides replay versus conflict.
                committed = load_operation(
                    connection,
                    tenant_id=tenant_id,
                    subject_id=subject_id,
                    family="create",
                    key_digest=key_digest,
                )
                if committed is None:
                    raise RepositoryError(
                        "the create key could not be claimed or read back"
                    )
                return create_replay(connection, committed, request_digest)

            connection.execute(
                text(
                    """
                    INSERT INTO review_run (
                        run_id, tenant_id, case_id, case_version, status, revision,
                        resume_count, created_by_subject, current_subject_id,
                        request_checksum, document_version, dataset_version,
                        index_version, rules_version, isolation, created_at, updated_at,
                        trace_id
                    ) VALUES (
                        :run_id, :tenant, :case_id, :case_version, :status, :revision,
                        :resume_count, :subject, :subject, :request_checksum, :document,
                        :dataset, :index, :rules, CAST(:isolation AS jsonb), :now, :now,
                        :trace_id
                    )
                    """
                ),
                {
                    "run_id": created.run_id,
                    "tenant": created.tenant_id,
                    "case_id": created.case_id,
                    "case_version": created.case_version,
                    "status": created.status.value,
                    "revision": created.revision,
                    "resume_count": created.resume_count,
                    "subject": created.created_by_subject,
                    "request_checksum": created.request_checksum,
                    "document": created.document_version,
                    "dataset": created.dataset_version,
                    "index": created.index_version,
                    "rules": created.rules_version,
                    "isolation": _json(isolated.model_dump(mode="json")),
                    "now": created.created_at,
                    "trace_id": created.trace_id,
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO review_run_generation (
                        run_id, generation, subject_id, credential_id,
                        authorization_revision, owner_id, status, heartbeat_at,
                        lease_expires_at, deadline_at, created_at
                    ) VALUES (
                        :run_id, :generation, :subject, :credential, :revision, :owner,
                        :status, :now, :lease, :deadline, :now
                    )
                    """
                ),
                {
                    "run_id": generation_record.run_id,
                    "generation": generation_record.generation,
                    "subject": generation_record.subject_id,
                    "credential": generation_record.credential_id,
                    "revision": generation_record.authorization_revision,
                    "owner": generation_record.owner_id,
                    "status": generation_record.status,
                    "now": generation_record.created_at,
                    "lease": generation_record.lease_expires_at,
                    "deadline": generation_record.deadline_at,
                },
            )
            append_trace_events(
                connection,
                run_id=run_id,
                trace_id=trace_id,
                drafts=events,
                now=now,
            )
            run = load_run(connection, run_id)
            assert run is not None
            return CreateOutcome(run=run, created=True)

    def _open_generation_locked(
        self,
        connection: Any,
        *,
        run_id: str,
        subject_id: str,
        credential_id: str,
        authorization_revision: int,
        owner_id: str,
        now: datetime,
        lease_seconds: float,
        deadline_seconds: float,
    ) -> RunGeneration:
        """Open the next execution generation inside the caller's locked transaction."""

        current = connection.execute(
            text(
                "SELECT coalesce(max(generation), 0) AS g FROM review_run_generation "
                "WHERE run_id = :run_id"
            ),
            {"run_id": run_id},
        ).scalar_one()
        generation = int(current) + 1
        expected_lease = now + timedelta(seconds=lease_seconds)
        expected_deadline = now + timedelta(seconds=deadline_seconds)
        opened = guard_record(
            "generation",
            {
                "run_id": run_id,
                "generation": generation,
                "subject_id": subject_id,
                "credential_id": credential_id,
                "authorization_revision": authorization_revision,
                "owner_id": owner_id,
                "status": ACTIVE_GENERATION_STATUS,
                "heartbeat_at": now,
                "lease_expires_at": expected_lease,
                "deadline_at": expected_deadline,
                "created_at": now,
            },
            location="generation",
        )
        assert isinstance(opened, GenerationRecord)
        superseded = guard_record(
            "generation_status", {"status": "superseded"}, location="generation_status"
        )
        assert isinstance(superseded, GenerationStatusPatch)
        connection.execute(
            text(
                "UPDATE review_run_generation SET status = :status "
                "WHERE run_id = :run_id AND status = 'active'"
            ),
            {"run_id": run_id, "status": superseded.status},
        )
        reopened = guard_record(
            "run_status",
            {
                "status": ReviewLifecycleStatus.PROCESSING,
                "error_code": None,
                "error_message": None,
                "current_subject_id": subject_id,
            },
            location="run_status",
        )
        assert isinstance(reopened, RunStatusPatch)
        connection.execute(
            text(
                """
                UPDATE review_run
                SET status = :status, revision = revision + 1,
                    error_code = :error_code, error_message = :error_message,
                    current_subject_id = :subject, updated_at = :now
                WHERE run_id = :run_id
                """
            ),
            {
                "run_id": run_id,
                "status": reopened.status.value,
                "error_code": reopened.error_code,
                "error_message": reopened.error_message,
                "subject": reopened.current_subject_id,
                "now": now,
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO review_run_generation (
                    run_id, generation, subject_id, credential_id,
                    authorization_revision, owner_id, status, heartbeat_at,
                    lease_expires_at, deadline_at, created_at
                ) VALUES (
                    :run_id, :generation, :subject, :credential, :revision, :owner,
                    :status, :now, :lease, :deadline, :now
                )
                """
            ),
            {
                "run_id": opened.run_id,
                "generation": opened.generation,
                "subject": opened.subject_id,
                "credential": opened.credential_id,
                "revision": opened.authorization_revision,
                "owner": opened.owner_id,
                "status": opened.status,
                "now": opened.created_at,
                "lease": opened.lease_expires_at,
                "deadline": opened.deadline_at,
            },
        )
        return RunGeneration(
            run_id=run_id,
            generation=generation,
            subject_id=subject_id,
            credential_id=credential_id,
            authorization_revision=authorization_revision,
            owner_id=owner_id,
            status=ACTIVE_GENERATION_STATUS,
            heartbeat_at=now,
            lease_expires_at=expected_lease,
            deadline_at=expected_deadline,
            created_at=now,
        )

    def heartbeat(
        self,
        *,
        run_id: str,
        generation: int,
        owner_id: str,
        now: datetime | Callable[[], datetime],
        lease_seconds: float,
    ) -> bool:
        clock = now
        try:
            with self._store.transaction() as connection:
                reason = self._guards.lock_active_generation(
                    connection,
                    run_id=run_id,
                    generation=generation,
                    owner_id=owner_id,
                    now=clock,
                )
                if reason is not None:
                    return False
                now = current_time(clock)
                renewal = guard_record(
                    "generation_heartbeat",
                    {
                        "heartbeat_at": now,
                        "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    },
                    location="generation_heartbeat",
                )
                assert isinstance(renewal, GenerationHeartbeatPatch)
                # The final check precedes renewal: renewal cannot make an expired lease live.
                self._guards.require_execution_time(
                    connection, run_id=run_id, generation=generation, now=clock
                )
                connection.execute(
                    text(
                        "UPDATE review_run_generation SET heartbeat_at=:now, lease_expires_at=:lease "
                        "WHERE run_id=:r AND generation=:g AND owner_id=:o AND status='active'"
                    ),
                    {
                        "r": run_id,
                        "g": generation,
                        "o": owner_id,
                        "now": renewal.heartbeat_at,
                        "lease": renewal.lease_expires_at,
                    },
                )
                return True
        except ExecutionInvalidated:
            return False

    def begin_attempt(
        self,
        *,
        run_id: str,
        generation: int,
        worker: str,
        attempt_no: int,
        now: datetime | Callable[[], datetime],
        owner_id: str | None = None,
    ) -> None:
        """Record the attempt start before any read-only port is called."""

        clock = now
        with self._store.transaction() as connection:
            discarded = self._guards.lock_active_generation(
                connection,
                run_id=run_id,
                generation=generation,
                now=now,
                owner_id=owner_id,
            )
            if discarded is not None:
                raise ExecutionInvalidated(discarded)
            now = current_time(clock)
            guard_record(
                "attempt",
                {
                    "run_id": run_id,
                    "generation": generation,
                    "worker": worker,
                    "attempt_no": attempt_no,
                    "started_at": now,
                },
                location="attempt",
            )
            connection.execute(
                text(
                    """
                    INSERT INTO review_worker_attempt (
                        run_id, generation, worker, attempt_no, started_at
                    ) VALUES (:run_id, :generation, :worker, :attempt, :now)
                    ON CONFLICT (run_id, generation, worker, attempt_no) DO NOTHING
                    """
                ),
                {
                    "run_id": run_id,
                    "generation": generation,
                    "worker": worker,
                    "attempt": attempt_no,
                    "now": now,
                },
            )
            self._guards.require_execution_time(
                connection, run_id=run_id, generation=generation, now=clock
            )

    def finish_attempt(
        self,
        *,
        run_id: str,
        generation: int,
        worker: str,
        attempt_no: int,
        outcome: str,
        error_code: str | None,
        now: datetime,
        abandoned: bool = False,
        discarded_late: bool = False,
    ) -> None:
        guard_record(
            "attempt_outcome",
            {
                "outcome": outcome,
                "error_code": error_code,
                "abandoned": abandoned,
                "discarded_late": discarded_late,
            },
            location="attempt_outcome",
        )
        with self._store.transaction() as connection:
            connection.execute(
                text(
                    """
                    UPDATE review_worker_attempt
                    SET finished_at = :now,
                        outcome = :outcome,
                        last_error = :error,
                        first_error = coalesce(first_error, :error),
                        abandoned = abandoned OR :abandoned,
                        discarded_late = discarded_late OR :discarded
                    WHERE run_id = :run_id AND generation = :generation
                      AND worker = :worker AND attempt_no = :attempt
                    """
                ),
                {
                    "run_id": run_id,
                    "generation": generation,
                    "worker": worker,
                    "attempt": attempt_no,
                    "outcome": outcome,
                    "error": error_code,
                    "now": now,
                    "abandoned": abandoned,
                    "discarded": discarded_late,
                },
            )

    def close_generation(
        self,
        *,
        run_id: str,
        generation: int,
        status: str,
        now: datetime,
        abandon_open_attempts: bool = True,
    ) -> None:
        closed = guard_record(
            "generation_status", {"status": status}, location="generation_status"
        )
        assert isinstance(closed, GenerationStatusPatch)
        abandoned = guard_record(
            "attempt_abandon",
            {"finished_at": now, "outcome": "abandoned"},
            location="attempt_abandon",
        )
        assert isinstance(abandoned, AttemptAbandonPatch)
        with self._store.transaction() as connection:
            lock_run(connection, run_id)
            connection.execute(
                text(
                    "UPDATE review_run_generation SET status = :status "
                    "WHERE run_id = :run_id AND generation = :generation AND status = 'active'"
                ),
                {
                    "run_id": run_id,
                    "generation": generation,
                    "status": closed.status,
                },
            )
            if abandon_open_attempts:
                connection.execute(
                    text(
                        """
                        UPDATE review_worker_attempt
                        SET abandoned = :abandoned,
                            finished_at = coalesce(finished_at, :now),
                            outcome = coalesce(outcome, :outcome)
                        WHERE run_id = :run_id AND generation = :generation
                          AND finished_at IS NULL
                        """
                    ),
                    {
                        "run_id": run_id,
                        "generation": generation,
                        "now": abandoned.finished_at,
                        "abandoned": abandoned.abandoned,
                        "outcome": abandoned.outcome,
                    },
                )

    def adopted_checkpoint(self, run_id: str, worker: str) -> Mapping[str, Any] | None:
        with self._store.transaction() as connection:
            row = connection.execute(
                text(
                    "SELECT * FROM review_branch_checkpoint "
                    "WHERE run_id = :run_id AND worker = :worker"
                ),
                {"run_id": run_id, "worker": worker},
            ).first()
        if row is None:
            return None
        stored = row_mapping(row)
        # Validate the complete checkpoint row before reuse; damaged records stop execution.
        verify_checkpoint_rows((stored,))
        payload = json_value(stored.get("result_payload"))
        return {
            "input_checksum": stored.get("input_checksum"),
            "contract_checksum": stored.get("contract_checksum"),
            "result": payload.get("result") if isinstance(payload, Mapping) else None,
        }

    def commit_branch(
        self,
        *,
        run_id: str,
        generation: int,
        worker: str,
        attempt_no: int,
        input_checksum: str,
        contract_checksum: str,
        result_payload: Mapping[str, Any],
        now: datetime | Callable[[], datetime],
        owner_id: str | None = None,
        events: Sequence[TraceDraft] = (),
    ) -> BranchCommit:
        """Adopt one validated branch result, atomically with its attempt and trace."""

        clock = now
        with self._store.transaction() as connection:
            guard = self._guards.lock_active_generation(
                connection,
                run_id=run_id,
                generation=generation,
                now=now,
                owner_id=owner_id,
            )
            if guard is not None:
                return BranchCommit(adopted=False, reason=guard)
            if self._fault is not None:
                # Test-only: let a scenario break the database inside this transaction.
                self._fault.barrier(f"during_checkpoint_commit:{worker}")
            now = current_time(clock)
            latest = connection.execute(
                text(
                    "SELECT coalesce(max(attempt_no), 0) FROM review_worker_attempt "
                    "WHERE run_id = :run_id AND worker = :worker"
                ),
                {"run_id": run_id, "worker": worker},
            ).scalar_one()
            if int(latest) > attempt_no:
                late = guard_record(
                    "attempt_discard",
                    {"discarded_late": True},
                    location="attempt_discard",
                )
                assert isinstance(late, AttemptDiscardPatch)
                connection.execute(
                    text(
                        "UPDATE review_worker_attempt SET discarded_late = :late "
                        "WHERE run_id = :run_id AND generation = :generation "
                        "AND worker = :worker AND attempt_no = :attempt"
                    ),
                    {
                        "run_id": run_id,
                        "generation": generation,
                        "worker": worker,
                        "attempt": attempt_no,
                        "late": late.discarded_late,
                    },
                )
                return BranchCommit(adopted=False, reason="superseded_attempt")
            # Both relation columns and payload come from the validated records below.
            adopted = guard_record(
                "checkpoint",
                {
                    "run_id": run_id,
                    "worker": worker,
                    "generation": generation,
                    "attempt_no": attempt_no,
                    "input_checksum": input_checksum,
                    "contract_checksum": contract_checksum,
                    "result_checksum": canonical_checksum(dict(result_payload)),
                },
                location="checkpoint",
            )
            assert isinstance(adopted, CheckpointRecord)
            stored = guard_record(
                "checkpoint_payload",
                result_payload,
                location="checkpoint_payload",
                strict=False,
            )
            assert isinstance(stored, CheckpointPayload)
            inserted = connection.execute(
                text(
                    """
                    INSERT INTO review_branch_checkpoint (
                        run_id, worker, generation, attempt_no, input_checksum,
                        contract_checksum, result_checksum, result_payload, adopted_at
                    ) VALUES (
                        :run_id, :worker, :generation, :attempt, :input, :contract,
                        :checksum, CAST(:payload AS jsonb), :now
                    )
                    ON CONFLICT (run_id, worker) DO NOTHING
                    """
                ),
                {
                    "run_id": adopted.run_id,
                    "worker": adopted.worker,
                    "generation": adopted.generation,
                    "attempt": adopted.attempt_no,
                    "input": adopted.input_checksum,
                    "contract": adopted.contract_checksum,
                    "checksum": adopted.result_checksum,
                    "payload": _json(stored.model_dump(mode="json")),
                    "now": now,
                },
            )
            if not inserted.rowcount:
                duplicate = guard_record(
                    "attempt_discard",
                    {"discarded_late": True},
                    location="attempt_discard",
                )
                assert isinstance(duplicate, AttemptDiscardPatch)
                connection.execute(
                    text(
                        """
                        UPDATE review_worker_attempt SET discarded_late = :late
                        WHERE run_id = :run_id AND generation = :generation
                          AND worker = :worker AND attempt_no = :attempt
                        """
                    ),
                    {
                        "run_id": run_id,
                        "generation": generation,
                        "worker": worker,
                        "attempt": attempt_no,
                        "late": duplicate.discarded_late,
                    },
                )
                return BranchCommit(adopted=False, reason="already_adopted")
            completed = guard_record(
                "attempt_completion",
                {"finished_at": now, "outcome": "completed"},
                location="attempt_completion",
            )
            assert isinstance(completed, AttemptCompletionPatch)
            connection.execute(
                text(
                    """
                    UPDATE review_worker_attempt
                    SET finished_at = :now, outcome = :outcome,
                        last_error = :last_error
                    WHERE run_id = :run_id AND generation = :generation
                      AND worker = :worker AND attempt_no = :attempt
                    """
                ),
                {
                    "run_id": run_id,
                    "generation": generation,
                    "worker": worker,
                    "attempt": attempt_no,
                    "now": completed.finished_at,
                    "outcome": completed.outcome,
                    "last_error": completed.last_error,
                },
            )
            append_trace_events(
                connection,
                run_id=run_id,
                trace_id=trace_identity(connection, run_id),
                drafts=events,
                now=now,
            )
            self._guards.require_execution_time(
                connection, run_id=run_id, generation=generation, now=clock
            )
        logger.info("run branch committed: run_id=%s worker=%s", run_id, worker)
        return BranchCommit(adopted=True, reason="committed")

    def commit_summary(
        self,
        *,
        run_id: str,
        generation: int,
        summary_payload: Mapping[str, Any],
        execution_payload: Mapping[str, Any],
        summary_checksum: str,
        gate_status: ReviewLifecycleStatus,
        now: datetime | Callable[[], datetime],
        owner_id: str | None = None,
        events: Sequence[TraceDraft] = (),
    ) -> bool:
        summary_record = guard_record(
            "summary", summary_payload, location="summary", strict=False
        )
        execution_record = guard_record(
            "execution", execution_payload, location="execution", strict=False
        )
        clock = now
        with self._store.transaction() as connection:
            guard = self._guards.lock_active_generation(
                connection,
                run_id=run_id,
                generation=generation,
                now=now,
                owner_id=owner_id,
            )
            if guard is not None:
                return False
            now = current_time(clock)
            row = connection.execute(
                text(
                    "SELECT revision FROM review_run WHERE run_id = :run_id FOR UPDATE"
                ),
                {"run_id": run_id},
            ).first()
            if row is None:
                raise RunNotFoundError(run_id)
            revision = int(row.revision) + 1
            connection.execute(
                text(
                    """
                    INSERT INTO review_run_summary (
                        run_id, summary_checksum, summary_payload, execution_payload,
                        human_gate_status, committed_revision, created_at
                    ) VALUES (
                        :run_id, :checksum, CAST(:payload AS jsonb),
                        CAST(:execution AS jsonb), :gate, :revision, :now
                    )
                    ON CONFLICT (run_id) DO NOTHING
                    """
                ),
                {
                    "run_id": run_id,
                    "checksum": summary_checksum,
                    "payload": _json(summary_record.model_dump(mode="json")),
                    "execution": _json(execution_record.model_dump(mode="json")),
                    "gate": gate_status.value,
                    "revision": revision,
                    "now": now,
                },
            )
            gated = guard_record(
                "run_status",
                {"status": gate_status, "error_code": None},
                location="run_status",
            )
            assert isinstance(gated, RunStatusPatch)
            connection.execute(
                text(
                    """
                    UPDATE review_run
                    SET status = :status, revision = :revision, updated_at = :now
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "status": gated.status.value,
                    "revision": revision,
                    "now": now,
                },
            )
            append_trace_events(
                connection,
                run_id=run_id,
                trace_id=trace_identity(connection, run_id),
                drafts=events,
                now=now,
            )
            if self._fault is not None:
                self._fault.barrier("during_summary_commit")
            self._guards.require_execution_time(
                connection, run_id=run_id, generation=generation, now=clock
            )
            return True

    def mark_interrupted(
        self,
        *,
        run_id: str,
        generation: int | None = None,
        owner_id: str | None = None,
        expired_only: bool = False,
        reason_code: str,
        now: datetime | Callable[[], datetime],
        events: Sequence[TraceDraft] = (),
        audit: AuditDraft | None = None,
    ) -> bool:
        clock = now
        with self._store.transaction() as connection:
            row = connection.execute(
                text("SELECT * FROM review_run WHERE run_id = :run_id FOR UPDATE"),
                {"run_id": run_id},
            ).first()
            if row is None:
                raise RunNotFoundError(run_id)
            if (
                ReviewLifecycleStatus(row.status)
                is not ReviewLifecycleStatus.PROCESSING
            ):
                return False
            active = connection.execute(
                text(
                    "SELECT generation, owner_id, lease_expires_at, deadline_at FROM review_run_generation "
                    "WHERE run_id=:r ORDER BY generation DESC LIMIT 1 FOR UPDATE"
                ),
                {"r": run_id},
            ).first()
            if (
                generation is not None
                and active is not None
                and (
                    active.generation != generation
                    or (owner_id is not None and active.owner_id != owner_id)
                )
            ):
                return False
            now = current_time(clock)
            if expired_only and (
                active is None or active.lease_expires_at > current_time(now)
            ):
                return False
            revision = int(row.revision) + 1
            interrupted = guard_record(
                "run_status",
                {
                    "status": ReviewLifecycleStatus.INTERRUPTED,
                    "error_code": reason_code,
                },
                location="run_status",
            )
            assert isinstance(interrupted, RunStatusPatch)
            abandoned = guard_record(
                "attempt_abandon",
                {"finished_at": now, "outcome": "abandoned"},
                location="attempt_abandon",
            )
            assert isinstance(abandoned, AttemptAbandonPatch)
            expired = guard_record(
                "generation_status", {"status": "expired"}, location="generation_status"
            )
            assert isinstance(expired, GenerationStatusPatch)
            connection.execute(
                text(
                    "UPDATE review_run SET status = :status, revision = :revision, "
                    "error_code = :code, updated_at = :now WHERE run_id = :run_id"
                ),
                {
                    "run_id": run_id,
                    "status": interrupted.status.value,
                    "revision": revision,
                    "code": interrupted.error_code,
                    "now": now,
                },
            )
            connection.execute(
                text(
                    "UPDATE review_run_generation SET status = :status "
                    "WHERE run_id = :run_id AND status = 'active'"
                ),
                {"run_id": run_id, "status": expired.status},
            )
            connection.execute(
                text(
                    """
                    UPDATE review_worker_attempt
                    SET abandoned = :abandoned,
                        finished_at = coalesce(finished_at, :now),
                        outcome = coalesce(outcome, :outcome)
                    WHERE run_id = :run_id AND finished_at IS NULL
                    """
                ),
                {
                    "run_id": run_id,
                    "now": abandoned.finished_at,
                    "abandoned": abandoned.abandoned,
                    "outcome": abandoned.outcome,
                },
            )
            append_trace_events(
                connection,
                run_id=run_id,
                trace_id=trace_identity(connection, run_id),
                drafts=events,
                now=now,
            )
            if audit is not None:
                append_audit_event(
                    connection, run_id=run_id, draft=audit, fault=self._fault
                )
            return True

    def mark_failed(
        self,
        *,
        run_id: str,
        generation: int | None = None,
        owner_id: str | None = None,
        error_code: str,
        error_message: str,
        now: datetime,
        events: Sequence[TraceDraft] = (),
    ) -> bool:
        with self._store.transaction() as connection:
            row = connection.execute(
                text("SELECT * FROM review_run WHERE run_id = :run_id FOR UPDATE"),
                {"run_id": run_id},
            ).first()
            if row is None:
                raise RunNotFoundError(run_id)
            if ReviewLifecycleStatus(row.status) not in {
                ReviewLifecycleStatus.PROCESSING,
                ReviewLifecycleStatus.INTERRUPTED,
            }:
                return False
            active = connection.execute(
                text(
                    "SELECT generation, owner_id, lease_expires_at, deadline_at FROM review_run_generation "
                    "WHERE run_id=:r ORDER BY generation DESC LIMIT 1 FOR UPDATE"
                ),
                {"r": run_id},
            ).first()
            if (
                generation is not None
                and active is not None
                and (
                    active.generation != generation
                    or (owner_id is not None and active.owner_id != owner_id)
                )
            ):
                return False
            revision = int(row.revision) + 1
            failed = guard_record(
                "run_status",
                {
                    "status": ReviewLifecycleStatus.FAILED,
                    "error_code": error_code,
                    "error_message": error_message,
                },
                location="run_status",
            )
            assert isinstance(failed, RunStatusPatch)
            cancelled = guard_record(
                "generation_status",
                {"status": "cancelled"},
                location="generation_status",
            )
            assert isinstance(cancelled, GenerationStatusPatch)
            connection.execute(
                text(
                    "UPDATE review_run SET status = :status, revision = :revision, "
                    "error_code = :code, error_message = :message, updated_at = :now "
                    "WHERE run_id = :run_id"
                ),
                {
                    "run_id": run_id,
                    "status": failed.status.value,
                    "revision": revision,
                    "code": failed.error_code,
                    "message": failed.error_message,
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
            append_trace_events(
                connection,
                run_id=run_id,
                trace_id=trace_identity(connection, run_id),
                drafts=events,
                now=now,
            )
            return True

    def apply_human_action(
        self,
        *,
        run_id: str,
        action: ApprovalActionType,
        expected_version: int,
        expected_summary_checksum: str | None,
        actor_subject_id: str,
        actor_credential_id: str,
        actor_authorization_revision: int,
        idempotency_key: str,
        request_digest: str,
        now: datetime,
        receipt_id: str | None,
        receipt_payload: Mapping[str, Any] | None,
        audit: AuditDraft,
        events: Sequence[TraceDraft] = (),
    ) -> ActionOutcome:
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
            row = connection.execute(
                text("SELECT * FROM review_run WHERE run_id = :run_id FOR UPDATE"),
                {"run_id": run_id},
            ).first()
            if row is None:
                raise RunNotFoundError(run_id)
            existing = connection.execute(
                text(
                    "SELECT * FROM review_idempotency WHERE tenant_id = :t "
                    "AND subject_id = :s AND operation_family = 'human_action' "
                    "AND key_digest = :k"
                ),
                {
                    "t": row.tenant_id,
                    "s": actor_subject_id,
                    "k": key_digest,
                },
            ).first()
            if existing is not None:
                if existing.request_digest != request_digest:
                    raise IdempotencyConflictError(
                        "the action key was already used with a different request"
                    )
                current = load_run(connection, run_id)
                assert current is not None
                return ActionOutcome(run=current, replay=True)

            status = ReviewLifecycleStatus(row.status)
            if int(row.revision) != expected_version:
                raise VersionConflictError(
                    expected=expected_version, current=int(row.revision)
                )
            target = {
                ApprovalActionType.CONFIRM: ReviewLifecycleStatus.CONFIRMED,
                ApprovalActionType.REJECT: ReviewLifecycleStatus.REJECTED,
                ApprovalActionType.CANCEL: ReviewLifecycleStatus.CANCELLED,
            }[action]
            if target not in LEGAL_TRANSITIONS[status]:
                raise IllegalTransitionError(action=action.value, status=status.value)

            revision = int(row.revision) + 1

            stored_receipt: ReceiptViewV2 | None = None
            receipt: ReceiptRecord | None = None
            if action in {ApprovalActionType.CONFIRM, ApprovalActionType.REJECT}:
                summary_row = connection.execute(
                    text(
                        "SELECT summary_checksum FROM review_run_summary "
                        "WHERE run_id = :run_id FOR SHARE"
                    ),
                    {"run_id": run_id},
                ).first()
                if summary_row is None:
                    raise IllegalTransitionError(
                        action=action.value, status=status.value
                    )
                if summary_row.summary_checksum != expected_summary_checksum:
                    raise SummaryChecksumError(current=summary_row.summary_checksum)
                if receipt_id is None or receipt_payload is None:
                    raise RepositoryError("a decision requires its receipt payload")
                stored_receipt = guard_record(
                    "receipt_payload",
                    receipt_payload,
                    location="receipt_payload",
                    strict=False,
                )
                assert isinstance(stored_receipt, ReceiptViewV2)
                receipt = guard_record(
                    "receipt",
                    {
                        "receipt_id": receipt_id,
                        "run_id": run_id,
                        "case_id": row.case_id,
                        "human_actor": actor_subject_id,
                        "decision": stored_receipt.decision.value,
                        "reviewed_recommendation": (
                            stored_receipt.reviewed_recommendation.value
                        ),
                        "recorded_at": now,
                        "idempotency_digest": key_digest,
                    },
                    location="receipt",
                )
                assert isinstance(receipt, ReceiptRecord)

            revision = int(row.revision) + 1
            action_mapping = guard_record(
                "idempotency",
                {
                    "tenant_id": row.tenant_id,
                    "subject_id": actor_subject_id,
                    "operation_family": "human_action",
                    "key_digest": key_digest,
                    "request_digest": request_digest,
                    "run_id": run_id,
                    "action": action.value,
                    "receipt_id": receipt_id,
                    "first_revision": revision,
                    "outcome": IDEMPOTENT_APPLIED,
                    "failure_code": None,
                    "created_at": now,
                },
                location="idempotency",
            )
            assert isinstance(action_mapping, IdempotencyRecord)
            # Claim the cross-Run namespace before writes; the loser answers the committed operation.
            if not claim_operation(connection, action_mapping):
                committed = connection.execute(
                    text(
                        "SELECT * FROM review_idempotency WHERE tenant_id = :t "
                        "AND subject_id = :s AND operation_family = 'human_action' "
                        "AND key_digest = :k"
                    ),
                    {"t": row.tenant_id, "s": actor_subject_id, "k": key_digest},
                ).first()
                if committed is None:
                    raise RepositoryError(
                        "the action key could not be claimed or read back"
                    )
                if committed.request_digest != request_digest:
                    raise IdempotencyConflictError(
                        "the action key was already used with a different request"
                    )
                current = load_run(connection, run_id)
                assert current is not None
                return ActionOutcome(run=current, replay=True)

            if receipt is not None and stored_receipt is not None:
                connection.execute(
                    text(
                        """
                        INSERT INTO review_receipt (
                            receipt_id, run_id, case_id, human_actor, decision,
                            reviewed_recommendation, recorded_at, idempotency_digest,
                            receipt_payload
                        ) VALUES (
                            :receipt_id, :run_id, :case_id, :actor, :decision,
                            :recommendation, :now, :digest, CAST(:payload AS jsonb)
                        )
                        """
                    ),
                    {
                        "receipt_id": receipt.receipt_id,
                        "run_id": receipt.run_id,
                        "case_id": receipt.case_id,
                        "actor": receipt.human_actor,
                        "decision": receipt.decision,
                        "recommendation": receipt.reviewed_recommendation,
                        "now": receipt.recorded_at,
                        "digest": receipt.idempotency_digest,
                        "payload": _json(stored_receipt.model_dump(mode="json")),
                    },
                )

            decided = guard_record(
                "run_status",
                {"status": target, "error_code": None},
                location="run_status",
            )
            assert isinstance(decided, RunStatusPatch)
            cancelled = guard_record(
                "generation_status",
                {"status": "cancelled"},
                location="generation_status",
            )
            assert isinstance(cancelled, GenerationStatusPatch)
            abandoned = guard_record(
                "attempt_abandon",
                {"finished_at": now, "outcome": "abandoned"},
                location="attempt_abandon",
            )
            assert isinstance(abandoned, AttemptAbandonPatch)
            connection.execute(
                text(
                    "UPDATE review_run SET status = :status, revision = :revision, "
                    "updated_at = :now WHERE run_id = :run_id"
                ),
                {
                    "run_id": run_id,
                    "status": decided.status.value,
                    "revision": revision,
                    "now": now,
                },
            )
            if action is ApprovalActionType.CANCEL:
                connection.execute(
                    text(
                        "UPDATE review_run_generation SET status = :status "
                        "WHERE run_id = :run_id AND status = 'active'"
                    ),
                    {"run_id": run_id, "status": cancelled.status},
                )
                connection.execute(
                    text(
                        """
                        UPDATE review_worker_attempt
                        SET abandoned = :abandoned,
                            finished_at = coalesce(finished_at, :now),
                            outcome = coalesce(outcome, :outcome)
                        WHERE run_id = :run_id AND finished_at IS NULL
                        """
                    ),
                    {
                        "run_id": run_id,
                        "now": abandoned.finished_at,
                        "abandoned": abandoned.abandoned,
                        "outcome": abandoned.outcome,
                    },
                )
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
            current = load_run(connection, run_id)
            assert current is not None
            return ActionOutcome(run=current, replay=False)

    def scan_expired_leases(
        self,
        *,
        now: datetime | Callable[[], datetime],
        events_factory,
        audit_factory,
    ) -> tuple[str, ...]:
        """Mark every processing Run whose active lease expired as interrupted."""

        with self._store.transaction() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT r.run_id, g.generation, g.owner_id
                    FROM review_run r
                    JOIN review_run_generation g
                      ON g.run_id = r.run_id AND g.status = 'active'
                    WHERE r.status = 'processing' AND g.lease_expires_at <= :now
                    ORDER BY r.run_id
                    """
                ),
                {"now": current_time(now)},
            ).all()
            stale = tuple(rows)
        interrupted = []
        for candidate in stale:
            run_id = str(candidate.run_id)
            if self.mark_interrupted(
                run_id=run_id,
                generation=candidate.generation,
                owner_id=candidate.owner_id,
                expired_only=True,
                reason_code="lease_expired",
                now=now,
                events=events_factory(run_id),
                audit=audit_factory(run_id),
            ):
                interrupted.append(run_id)
        return tuple(interrupted)

    def append_trace(
        self,
        *,
        run_id: str,
        drafts: Sequence[TraceDraft],
        now: datetime,
    ) -> tuple[Any, ...]:
        """Append events to one Run's own Trace sequence."""

        with self._store.transaction() as connection:
            lock_run(connection, run_id)
            return append_trace_events(
                connection,
                run_id=run_id,
                trace_id=trace_identity(connection, run_id),
                drafts=drafts,
                now=now,
            )

    def trace_envelope(self, run_id: str, *, complete: bool) -> Any:
        run = self.load_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        verify_run_record(run)
        verify_trace_records(run.trace_events)
        return trace_envelope_view(run, complete=complete)

    def audit_events(self, run_id: str) -> tuple[Any, ...]:
        return self.audit_projection(run_id)[0]

    def audit_projection(self, run_id: str) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        """The audit events of one Run, plus the legacy sequence anomalies."""

        run = self.load_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        verify_run_record(run)
        verify_audit_records(run.audit_events)
        events = tuple(
            validate_stored_record("audit", item) for item in run.audit_events
        )
        return events, run.audit_sequence_anomalies

    def worker_trace_drafts(
        self,
        run_id: str,
        *,
        generation: int,
        worker: WorkerKind,
        attempt_no: int,
    ) -> Iterable[TraceDraft]:
        del run_id, generation, worker, attempt_no
        return ()


LEGAL_TRANSITIONS: dict[ReviewLifecycleStatus, frozenset[ReviewLifecycleStatus]] = {
    ReviewLifecycleStatus.PROCESSING: frozenset(
        {
            ReviewLifecycleStatus.PENDING_CONFIRMATION,
            ReviewLifecycleStatus.PENDING_HUMAN_REVIEW,
            ReviewLifecycleStatus.CANCELLED,
            ReviewLifecycleStatus.FAILED,
            ReviewLifecycleStatus.INTERRUPTED,
        }
    ),
    ReviewLifecycleStatus.INTERRUPTED: frozenset(
        {
            ReviewLifecycleStatus.PROCESSING,
            ReviewLifecycleStatus.CANCELLED,
            ReviewLifecycleStatus.FAILED,
        }
    ),
    ReviewLifecycleStatus.PENDING_CONFIRMATION: frozenset(
        {
            ReviewLifecycleStatus.CONFIRMED,
            ReviewLifecycleStatus.REJECTED,
            ReviewLifecycleStatus.CANCELLED,
        }
    ),
    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW: frozenset(
        {
            ReviewLifecycleStatus.CONFIRMED,
            ReviewLifecycleStatus.REJECTED,
            ReviewLifecycleStatus.CANCELLED,
        }
    ),
    ReviewLifecycleStatus.CONFIRMED: frozenset(),
    ReviewLifecycleStatus.REJECTED: frozenset(),
    ReviewLifecycleStatus.CANCELLED: frozenset(),
    ReviewLifecycleStatus.FAILED: frozenset(),
}


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


__all__ = [
    "ACTIVE_GENERATION_STATUS",
    "ActionOutcome",
    "AttemptRecord",
    "AuditDraft",
    "TraceDraft",
    "AUTHORITY_REASONS",
    "BranchCommit",
    "ExecutionInvalidated",
    "CreateOutcome",
    "GenerationIdentity",
    "RunAuthorizationIdentity",
    "GrantRevokedError",
    "IdempotencyConflictError",
    "IDEMPOTENT_APPLIED",
    "IDEMPOTENT_FAILED",
    "IDEMPOTENT_REPLAY",
    "IllegalTransitionError",
    "INVALIDATION_CHECKPOINTS_CHANGED",
    "INVALIDATION_GENERATION_MISSING",
    "INVALIDATION_LEASE_EXPIRED",
    "INVALIDATION_NO_ACTIVE_GENERATION",
    "INVALIDATION_RUN_NOT_PROCESSING",
    "INVALIDATION_SUPERSEDED",
    "LEGAL_TRANSITIONS",
    "PENDING_STATUSES",
    "PersistedRun",
    "RepositoryError",
    "ResumeOutcome",
    "ReviewRepository",
    "RunGeneration",
    "RunHeader",
    "RunNotFoundError",
    "StaleAuthorizationError",
    "SummaryChecksumError",
    "TERMINAL_STATUSES",
    "FieldGuardError",
    "TraceDraft",
    "VersionConflictError",
    "checkpoints_digest",
    "guard_payload",
    "idempotency_key_digest",
    "new_id",
]
