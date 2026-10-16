"""Persisted, credential-scoped review lifecycle for the V1-03 runtime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.exc import SQLAlchemyError

from .api_contracts import (
    HumanActionRequestV2,
    ReceiptViewV2,
    ResumeRequestV2,
    ReviewCreateRequestV2,
    ReviewListResponseV2,
    ReviewRunResponseV2,
)
from .audit_contracts import AuditEventClass
from .authorization import AUTHORITY_DENIAL_REASON
from .checkpointing import (
    IncompatibleCheckpointError,
    StorageUnavailableSignal,
)
from .contracts import (
    ApprovalActionType,
    HumanDecision,
    HumanReviewStatus,
    ReviewLifecycleStatus,
    ReviewSummary,
    RunEventType,
)
from .fault_hooks import FaultController
from .precise_run_contracts import ExecutionPayload
from .precise_run_state import (
    checkpoint_reusable,
    run_isolation,
    snapshot_from_isolation,
)
from .review_repository import (
    AuditDraft,
    ExecutionInvalidated,
    GrantRevokedError,
    INVALIDATION_GENERATION_MISSING,
    IDEMPOTENT_FAILED,
    IdempotencyConflictError,
    IllegalTransitionError,
    PersistedRun,
    ReviewRepository,
    RunNotFoundError,
    StaleAuthorizationError,
    SummaryChecksumError,
    TraceDraft,
    VersionConflictError,
    checkpoints_digest,
    new_id,
)
from .record_guard import validate_stored_model
from .record_reads import (
    record_refusal,
    verify_checkpoint_rows,
    verify_run_record,
)
from .review_store import CaseRegistration, ReviewStoreError
from .security_contracts import (
    AuthContext,
    DenialReason,
    RuntimeScope,
    SecurityError,
    SecurityOperation,
    unauthenticated,
)
from .run_projection import (
    COMPLETE_STATUSES,
    build_run_response,
    registration_from_run,
)
from .scoped_runtime import ScopeIntegrityError
from .run_errors import RecordUnavailableError
from .run_stop import BatchStoppedSignal
from .service_errors import ReviewServiceError
from .trace import canonical_checksum

API_TRANSPORT_VERSION = "review-api.v2"


LEASE_SECONDS = 15.0
HEARTBEAT_SECONDS = 5.0
SCAN_INTERVAL_SECONDS = 1.0
GENERATION_DEADLINE_SECONDS = 30.0
RUN_RESUME_LIMIT = 2

# Whole-generation loss is distinct from a branch-local rejection.
AUTHORIZATION_REVOKED = "authorization_revoked"

# Diagnostics contain IDs/codes only; business records stay in Trace/audit.
logger = logging.getLogger("procurement_review.review_service")


class ReviewRunner(Protocol):
    async def run(
        self,
        *,
        run_id: str,
        scope: RuntimeScope,
        registration: CaseRegistration,
        generation: int,
        repository: ReviewRepository,
        deadline_at: datetime,
        fault: FaultController | None,
        heartbeat: Callable[[], bool],
        owner_id: str,
        now: Callable[[], datetime],
    ) -> ExecutionPayload: ...


class IncompatibleCheckpointFailure(RuntimeError):
    """Raised when recovery cannot reuse the committed checkpoints."""


class ReviewService:
    """Own the persisted Run state machine without changing review risk semantics."""

    def __init__(
        self,
        *,
        repository: ReviewRepository,
        runner: ReviewRunner,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        lease_seconds: float = LEASE_SECONDS,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        scan_interval_seconds: float = SCAN_INTERVAL_SECONDS,
        generation_deadline_seconds: float = GENERATION_DEADLINE_SECONDS,
        resume_limit: int = RUN_RESUME_LIMIT,
        shutdown_timeout_seconds: float = 5.0,
        fault: FaultController | None = None,
    ) -> None:
        self._repository = repository
        self._runner = runner
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or _default_id_factory
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._scan_interval_seconds = scan_interval_seconds
        self._generation_deadline_seconds = generation_deadline_seconds
        self._resume_limit = resume_limit
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._fault = fault if fault is not None and fault.enabled else None
        self._tasks: dict[tuple[str, int], asyncio.Task[None]] = {}
        self._scanner: asyncio.Task[None] | None = None
        self._abandoned_tasks = 0

    @property
    def repository(self) -> ReviewRepository:
        return self._repository

    @property
    def abandoned_task_count(self) -> int:
        """Execution tasks that did not unwind inside the bounded shutdown window."""

        return self._abandoned_tasks

    def now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("review service clock must be timezone-aware")
        return value

    async def create_review(
        self,
        request: ReviewCreateRequestV2,
        *,
        registration: CaseRegistration,
        scope: RuntimeScope,
        context: AuthContext,
        idempotency_key: str,
    ) -> tuple[ReviewRunResponseV2, bool]:
        if registration.tenant_id != context.tenant_id:
            raise _not_found(request.case_id)
        if registration.case_version != request.case_version:
            raise ReviewServiceError(
                status_code=409,
                code="case_version_mismatch",
                message="The requested Case version is not the registered version.",
                details={"registered_case_version": registration.case_version},
            )
        del scope
        request_digest = canonical_checksum(
            {
                "case_id": request.case_id,
                "case_version": request.case_version,
                "tenant_id": registration.tenant_id,
            }
        )
        run_id = self._id_factory("run")
        trace_id = self._id_factory("trace")
        now = self.now()
        events = (
            TraceDraft(
                event_type=RunEventType.RUN_STARTED.value,
                step="review_run",
                input_summary={
                    "case_id": registration.case_id,
                    "case_type": registration.case_type,
                    "scenario": registration.scenario,
                    "execution_mode": "test_only",
                },
                source_versions={
                    "api": API_TRANSPORT_VERSION,
                    "document": registration.document_version,
                    "index": registration.index_version,
                    "dataset": registration.dataset_version,
                    "rules": registration.rules_version,
                },
            ),
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="validate_input",
                output_summary={"status": "valid"},
                source_versions={"api": API_TRANSPORT_VERSION},
            ),
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="create_run",
                output_summary={"status": "processing"},
                source_versions={"api": API_TRANSPORT_VERSION},
            ),
        )
        try:
            outcome = self._repository.create_run(
                run_id=run_id,
                trace_id=trace_id,
                tenant_id=registration.tenant_id,
                case_id=registration.case_id,
                case_version=registration.case_version,
                subject_id=context.subject_id,
                credential_id=context.credential_id,
                authorization_revision=context.authorization_revision,
                request_checksum=canonical_checksum(
                    {
                        "case_id": request.case_id,
                        "case_version": request.case_version,
                        "request_digest": request_digest,
                    }
                ),
                document_version=registration.document_version,
                dataset_version=registration.dataset_version,
                index_version=registration.index_version,
                rules_version=registration.rules_version,
                isolation=run_isolation(registration, self._runner),
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                owner_id=new_id("owner"),
                now=now,
                lease_seconds=self._lease_seconds,
                deadline_seconds=self._generation_deadline_seconds,
                events=events,
            )
        except IdempotencyConflictError as error:
            raise ReviewServiceError(
                status_code=409,
                code="idempotency_conflict",
                message=(
                    "The request idempotency key was already used with a different "
                    "payload."
                ),
            ) from error
        except ExecutionInvalidated as error:
            raise _authority_lost(error, SecurityOperation.REVIEW_CREATE) from error
        if outcome.created:
            if self._fault is not None:
                self._fault.barrier("after_create_commit")
            generation = (
                outcome.run.generation.generation if outcome.run.generation else 1
            )
            logger.info(
                "run created: run_id=%s case_id=%s generation=%s",
                run_id,
                registration.case_id,
                generation,
            )
            self._schedule(outcome.run.header.run_id, generation)
        # Replays pass the same complete-record verification as detail reads.
        return self._verified_response(outcome.run), outcome.created

    def _schedule(self, run_id: str, generation: int) -> None:
        """Schedule exactly one task per (run, generation)."""

        key = (run_id, generation)
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            return
        self._tasks[key] = asyncio.create_task(
            self._execute(run_id, generation),
            name=f"review-execution:{run_id}:{generation}",
        )

    async def _execute(self, run_id: str, generation: int) -> None:
        owner_id: str | None = None
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            run = self._repository.load_run(run_id)
            if run is None or run.header.status is not ReviewLifecycleStatus.PROCESSING:
                return
            if run.generation is None:
                self._fail_run(
                    run_id,
                    generation,
                    "generation_missing",
                    "execution generation missing",
                    owner_id=owner_id,
                )
                return
            if run.generation.generation != generation:
                # Stale tasks cannot change their replacement.
                return
            owner_id = run.generation.owner_id
            try:
                self._repository.assert_generation_live(
                    run_id=run_id,
                    generation=generation,
                    now=self.now,
                    owner_id=owner_id,
                    site="dispatch",
                )
            except ExecutionInvalidated as error:
                self._stop_invalidated(run_id, generation, error, owner_id=owner_id)
                return
            scope, registration = self._runtime_from_run(run, generation)
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(run_id, generation, owner_id),
                name=f"review-heartbeat:{run_id}",
            )
            execution = await self._runner.run(
                run_id=run_id,
                scope=scope,
                registration=registration,
                generation=generation,
                repository=self._repository,
                deadline_at=run.generation.deadline_at,
                fault=self._fault,
                heartbeat=lambda: self._heartbeat(run_id, generation, owner_id),
                owner_id=owner_id,
                now=self.now,
            )
            self._commit_summary(run_id, generation, execution, owner_id=owner_id)
        except asyncio.CancelledError:
            raise
        except BaseExceptionGroup as group:
            # Unwrap the MCP TaskGroup before classifying the stop signal.
            incompatible = _find_signal(group, IncompatibleCheckpointError)
            unavailable = _find_signal(group, StorageUnavailableSignal)
            invalidated = _find_signal(group, ExecutionInvalidated)
            batch_stopped = _find_signal(group, BatchStoppedSignal)
            if incompatible is not None:
                self._fail_run(
                    run_id,
                    generation,
                    "checkpoint_incompatible",
                    incompatible.message,
                    owner_id=owner_id,
                )
            elif unavailable is not None:
                self._interrupt_or_fail(
                    run_id, generation, "storage_unavailable", owner_id=owner_id
                )
            elif invalidated is not None:
                self._stop_invalidated(
                    run_id, generation, invalidated, owner_id=owner_id
                )
            elif batch_stopped is not None:
                # The evaluation batch had already stopped sending; the Run records that
                # fact instead of pretending its own execution failed.
                self._interrupt_or_fail(
                    run_id, generation, "batch_stopped", owner_id=owner_id
                )
            else:
                self._fail_run(
                    run_id,
                    generation,
                    "review_execution_failed",
                    "The local review execution failed closed.",
                    owner_id=owner_id,
                )
        except (StorageUnavailableSignal, IncompatibleCheckpointError) as error:
            if isinstance(error, IncompatibleCheckpointError):
                self._fail_run(
                    run_id,
                    generation,
                    "checkpoint_incompatible",
                    error.message,
                    owner_id=owner_id,
                )
            else:
                self._interrupt_or_fail(
                    run_id, generation, "storage_unavailable", owner_id=owner_id
                )
        except BatchStoppedSignal:
            self._interrupt_or_fail(
                run_id, generation, "batch_stopped", owner_id=owner_id
            )
        except ExecutionInvalidated as error:
            self._stop_invalidated(run_id, generation, error, owner_id=owner_id)
        except IncompatibleCheckpointFailure as error:
            self._fail_run(
                run_id,
                generation,
                "checkpoint_incompatible",
                str(error),
                owner_id=owner_id,
            )
        except (ReviewStoreError, SQLAlchemyError):
            # Preserve checkpoints for explicit recovery after storage returns.
            self._interrupt_or_fail(
                run_id, generation, "storage_unavailable", owner_id=owner_id
            )
        except Exception as error:  # noqa: BLE001 - runtime failure is a closed API state
            # Material that cannot serve a branch scope keeps its own named code.
            named = isinstance(error, ScopeIntegrityError)
            self._fail_run(
                run_id,
                generation,
                "material_scope_incomplete" if named else "review_execution_failed",
                str(error)[:200]
                if named
                else "The local review execution failed closed.",
                owner_id=owner_id,
            )
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await heartbeat_task
            self._tasks.pop((run_id, generation), None)

    def _commit_summary(
        self,
        run_id: str,
        generation: int,
        execution: ExecutionPayload,
        *,
        owner_id: str | None = None,
    ) -> None:
        summary = execution.summary
        gate_status = (
            ReviewLifecycleStatus.PENDING_CONFIRMATION
            if summary.human_review_status is HumanReviewStatus.PENDING_CONFIRMATION
            else ReviewLifecycleStatus.PENDING_HUMAN_REVIEW
        )
        events = (
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="deterministic_aggregator",
                output_summary={
                    "recommendation": summary.recommendation.value,
                    "highest_risk": summary.highest_risk.value,
                    "finding_count": len(summary.findings),
                    "missing_count": len(summary.missing_information),
                    "conflict_count": len(summary.conflicts),
                    "failure_count": len(summary.failure_reasons),
                },
                source_versions={"rules": summary.rules_version},
                generation=generation,
            ),
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="review_workflow",
                output_summary={"status": gate_status.value},
                source_versions={"execution": execution.execution_version},
                generation=generation,
            ),
        )
        committed = self._repository.commit_summary(
            run_id=run_id,
            generation=generation,
            summary_payload=summary.model_dump(mode="json"),
            execution_payload=execution.model_dump(mode="json"),
            summary_checksum=canonical_checksum(summary.model_dump(mode="json")),
            owner_id=owner_id,
            gate_status=gate_status,
            now=self.now,
            events=events,
        )
        if committed:
            logger.info("run summary committed: run_id=%s", run_id)
            self._repository.close_generation(
                run_id=run_id,
                generation=generation,
                status="completed",
                now=self.now(),
                abandon_open_attempts=False,
            )
            if self._fault is not None:
                self._fault.barrier("after_summary_commit")

    def _interrupt_or_fail(
        self,
        run_id: str,
        generation: int,
        reason_code: str = "storage_unavailable",
        *,
        owner_id: str | None = None,
    ) -> None:
        try:
            run = self._repository.load_run(run_id)
        except SQLAlchemyError:
            # The store is still unavailable: the lease scanner owns this transition.
            return
        if run is None or run.header.status is not ReviewLifecycleStatus.PROCESSING:
            return
        if run.generation is not None and run.generation.generation != generation:
            # A newer generation already owns the Run.
            return
        try:
            changed = self._repository.mark_interrupted(
                run_id=run_id,
                generation=generation,
                owner_id=owner_id,
                reason_code=reason_code,
                now=self.now(),
                events=(
                    TraceDraft(
                        event_type=RunEventType.STEP_FAILED.value,
                        step="review_workflow",
                        output_summary={
                            "status": ReviewLifecycleStatus.INTERRUPTED.value
                        },
                        source_versions={"execution": "review-execution.v1"},
                        error=(
                            "The local review execution stopped without a committed result."
                        ),
                        generation=generation,
                    ),
                ),
                audit=AuditDraft(
                    audit_id=self._id_factory("audit"),
                    event_class=AuditEventClass.LIFECYCLE,
                    occurred_at=self.now(),
                    new_status=ReviewLifecycleStatus.INTERRUPTED,
                    detail_code=reason_code,
                ),
            )
        except SQLAlchemyError:
            return
        if changed:
            logger.info(
                "run interrupted: run_id=%s generation=%s reason=%s",
                run_id,
                generation,
                reason_code,
            )

    def _fail_run(
        self,
        run_id: str,
        generation: int,
        code: str,
        message: str,
        *,
        owner_id: str | None = None,
    ) -> None:
        run = self._repository.load_run(run_id)
        if run is None:
            return
        if run.generation is not None and run.generation.generation != generation:
            return
        self._repository.mark_failed(
            run_id=run_id,
            generation=generation,
            owner_id=owner_id,
            error_code=code,
            error_message=_bounded(message),
            now=self.now(),
            events=(
                TraceDraft(
                    event_type=RunEventType.RUN_FAILED.value,
                    step="review_run",
                    output_summary={"status": ReviewLifecycleStatus.FAILED.value},
                    source_versions={"execution": "review-execution.v1"},
                    error=_bounded(message),
                ),
            ),
        )

    async def _heartbeat_loop(
        self, run_id: str, generation: int, owner_id: str
    ) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            if not self._heartbeat(run_id, generation, owner_id):
                return

    def _heartbeat(self, run_id: str, generation: int, owner_id: str) -> bool:
        renewed = self._repository.heartbeat(
            run_id=run_id,
            generation=generation,
            owner_id=owner_id,
            now=self.now,
            lease_seconds=self._lease_seconds,
        )
        if not renewed:
            try:
                self._repository.assert_generation_live(
                    run_id=run_id,
                    generation=generation,
                    owner_id=owner_id,
                    now=self.now,
                    site="heartbeat",
                )
            except ExecutionInvalidated as error:
                self._stop_invalidated(run_id, generation, error, owner_id=owner_id)
        return renewed

    def _stop_invalidated(
        self,
        run_id: str,
        generation: int,
        error: ExecutionInvalidated,
        *,
        owner_id: str | None = None,
    ) -> None:
        """Turn one lost generation into a controlled interruption."""

        reason = AUTHORIZATION_REVOKED if error.authority_lost else error.reason
        self._interrupt_or_fail(run_id, generation, reason, owner_id=owner_id)

    async def resume_review(
        self,
        run_id: str,
        request: ResumeRequestV2,
        *,
        context: AuthContext,
        idempotency_key: str,
    ) -> tuple[ReviewRunResponseV2, bool]:
        run = self._repository.load_run(run_id)
        if run is None or run.header.tenant_id != context.tenant_id:
            raise _not_found(run_id)
        if request.expected_version is None:
            raise ReviewServiceError(
                status_code=422,
                code="expected_version_required",
                message="expected_version is required to resume a review.",
            )
        request_digest = canonical_checksum(
            {
                "operation": "resume",
                "run_id": run_id,
                "expected_version": request.expected_version,
            }
        )
        # The transaction revalidates this projected compatibility decision.
        incompatibility = self._checkpoint_incompatibility(run)
        try:
            outcome = self._repository.resume_run(
                run_id=run_id,
                actor_subject_id=context.subject_id,
                actor_credential_id=context.credential_id,
                actor_authorization_revision=context.authorization_revision,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                expected_version=request.expected_version,
                owner_id=new_id("owner"),
                now=self.now(),
                lease_seconds=self._lease_seconds,
                deadline_seconds=self._generation_deadline_seconds,
                resume_limit=self._resume_limit,
                incompatibility=incompatibility,
                expected_checkpoints_digest=checkpoints_digest(run.checkpoints),
                audit=self._resume_audit(run, context),
                failure_audit=self._resume_rejection_audit(run, context),
                events=self._resume_events(run),
                rejection_events=self._rejection_events,
            )
        except ExecutionInvalidated as error:
            raise _authority_lost(error, SecurityOperation.REVIEW_RESUME) from error
        except GrantRevokedError as error:
            raise _not_found(run_id) from error
        except StaleAuthorizationError as error:
            raise ReviewServiceError(
                status_code=409,
                code="authorization_revision_changed",
                message="The authorization changed; re-read the review before resuming.",
            ) from error
        except IdempotencyConflictError as error:
            raise ReviewServiceError(
                status_code=409,
                code="idempotency_conflict",
                message="The resume key was already used with a different request.",
            ) from error
        except VersionConflictError as error:
            raise ReviewServiceError(
                status_code=409,
                code="version_conflict",
                message="The review version is stale; refresh before acting.",
                details={
                    "expected_version": error.expected,
                    "current_version": error.current,
                },
            ) from error
        except IllegalTransitionError as error:
            raise ReviewServiceError(
                status_code=409,
                code="invalid_state_transition",
                message=(
                    "Only an interrupted review can be resumed; a pending or terminal "
                    "review must be read or decided instead."
                ),
                details={"action": error.action, "status": error.status},
            ) from error

        if outcome.kind == IDEMPOTENT_FAILED:
            raise _not_recoverable(outcome.failure_code)
        if not outcome.schedules:
            logger.info("run resume replayed: run_id=%s", run_id)
            return self._verified_response(outcome.run), False
        if self._fault is not None:
            # A crash here leaves a complete generation discoverable by the scanner.
            self._fault.barrier("after_resume_commit")
        assert outcome.generation is not None
        logger.info(
            "run resumed: run_id=%s generation=%s",
            run_id,
            outcome.generation,
        )
        self._schedule(run_id, outcome.generation)
        if self._fault is not None:
            self._fault.barrier("after_resume_dispatch")
        return self._verified_response(outcome.run), True

    def _resume_audit(self, run: PersistedRun, context: AuthContext) -> AuditDraft:
        return AuditDraft(
            audit_id=self._id_factory("audit"),
            event_class=AuditEventClass.RECOVERY,
            occurred_at=self.now(),
            actor=context.subject_id,
            authorization_revision=context.authorization_revision,
            previous_status=run.header.status,
            new_status=ReviewLifecycleStatus.PROCESSING,
            previous_revision=run.header.revision,
            detail_code="resume_requested",
        )

    def _resume_rejection_audit(
        self, run: PersistedRun, context: AuthContext
    ) -> AuditDraft:
        return AuditDraft(
            audit_id=self._id_factory("audit"),
            event_class=AuditEventClass.RECOVERY,
            occurred_at=self.now(),
            actor=context.subject_id,
            authorization_revision=context.authorization_revision,
            previous_status=run.header.status,
            new_status=ReviewLifecycleStatus.FAILED,
            previous_revision=run.header.revision,
        )

    def _resume_events(self, run: PersistedRun) -> tuple[TraceDraft, ...]:
        generation = run.generation.generation if run.generation else 0
        return (
            TraceDraft(
                event_type=RunEventType.STEP_COMPLETED.value,
                step="review_recovery",
                output_summary={"status": ReviewLifecycleStatus.PROCESSING.value},
                source_versions={"recovery": "review-recovery.v1"},
                generation=generation + 1,
            ),
        )

    def _rejection_events(self, reason: str) -> tuple[TraceDraft, ...]:
        return (
            TraceDraft(
                event_type=RunEventType.RUN_FAILED.value,
                step="review_run",
                output_summary={"status": ReviewLifecycleStatus.FAILED.value},
                source_versions={"recovery": "review-recovery.v1"},
                error=_bounded(reason),
            ),
        )

    def _checkpoint_incompatibility(self, run: PersistedRun) -> str | None:
        """Return a stable reason when an adopted checkpoint cannot be reused."""

        if not run.checkpoints or run.generation is None:
            return None
        try:
            # Damaged checkpoint records take the permanent incompatibility path.
            verify_checkpoint_rows(run.raw.checkpoints)
        except RecordUnavailableError:
            return "checkpoint_incompatible"
        try:
            with record_refusal():
                scope, _registration = self._runtime_from_run(
                    run, run.generation.generation
                )
        except IncompatibleCheckpointFailure:
            return INVALIDATION_GENERATION_MISSING
        run_frozen = snapshot_from_isolation(getattr(run.header, "isolation", None))
        for worker, payload in run.checkpoints.items():
            if not checkpoint_reusable(payload, scope, worker, snapshot=run_frozen):
                return "checkpoint_incompatible"
        return None

    async def scan_interruptions(self) -> tuple[str, ...]:
        """Mark every Run whose execution lease expired without a heartbeat."""

        interrupted = self._repository.scan_expired_leases(
            now=self.now,
            events_factory=lambda run_id: (
                TraceDraft(
                    event_type=RunEventType.STEP_FAILED.value,
                    step="review_workflow",
                    output_summary={
                        "status": ReviewLifecycleStatus.INTERRUPTED.value,
                        "reason": "lease_expired",
                    },
                    source_versions={"recovery": "review-recovery.v1"},
                    error="The execution lease expired.",
                ),
            ),
            audit_factory=lambda run_id: AuditDraft(
                audit_id=self._id_factory("audit"),
                event_class=AuditEventClass.LIFECYCLE,
                occurred_at=self.now(),
                new_status=ReviewLifecycleStatus.INTERRUPTED,
                detail_code="lease_expired",
            ),
        )
        for run_id in interrupted:
            logger.info("run interrupted: run_id=%s reason=lease_expired", run_id)
        return interrupted

    def start_scanner(self) -> None:
        if self._scanner is not None:
            return

        async def loop() -> None:
            while True:
                await asyncio.sleep(self._scan_interval_seconds)
                with contextlib.suppress(Exception):
                    await self.scan_interruptions()

        self._scanner = asyncio.create_task(loop(), name="review-interruption-scan")

    def apply_action(
        self,
        *,
        run_id: str,
        action: ApprovalActionType,
        request: HumanActionRequestV2,
        context: AuthContext,
    ) -> ReviewRunResponseV2:
        if request.idempotency_key is None:
            raise ReviewServiceError(
                status_code=422,
                code="idempotency_key_required",
                message="An idempotency key is required for every human action.",
            )
        if request.expected_version is None:
            raise ReviewServiceError(
                status_code=422,
                code="expected_version_required",
                message="expected_version is required for every human action.",
            )
        if (
            action in {ApprovalActionType.CONFIRM, ApprovalActionType.REJECT}
            and request.expected_summary_checksum is None
        ):
            raise ReviewServiceError(
                status_code=422,
                code="expected_summary_checksum_required",
                message=(
                    "expected_summary_checksum is required to decide on a review result."
                ),
            )
        run = self._repository.load_run(run_id)
        if run is None or run.header.tenant_id != context.tenant_id:
            raise _not_found(run_id)
        # Authorized actions must still validate the whole record (AC-04.2).
        with record_refusal():
            verify_run_record(run)
        request_digest = canonical_checksum(
            {
                "run_id": run_id,
                "action": action.value,
                "expected_version": request.expected_version,
                "expected_summary_checksum": request.expected_summary_checksum,
            }
        )
        previous_status = run.header.status
        target_status = {
            ApprovalActionType.CONFIRM: ReviewLifecycleStatus.CONFIRMED,
            ApprovalActionType.REJECT: ReviewLifecycleStatus.REJECTED,
            ApprovalActionType.CANCEL: ReviewLifecycleStatus.CANCELLED,
        }[action]
        decision = {
            ApprovalActionType.CONFIRM: HumanDecision.CONFIRM_RECOMMENDATION,
            ApprovalActionType.REJECT: HumanDecision.REJECT_RECOMMENDATION,
        }.get(action)
        receipt_id: str | None = None
        receipt_payload: dict[str, object] | None = None
        if decision is not None:
            if run.summary_payload is None:
                raise ReviewServiceError(
                    status_code=409,
                    code="invalid_state_transition",
                    message="A review result must exist before a decision is recorded.",
                    details={"status": previous_status.value},
                )
            summary = validate_stored_model(ReviewSummary, run.summary_payload)
            receipt_id = self._id_factory("receipt")
            receipt_payload = ReceiptViewV2(
                receipt_id=receipt_id,
                run_id=run_id,
                case_id=run.header.case_id,
                human_actor=context.subject_id,
                decision=decision,
                reviewed_recommendation=summary.recommendation,
                recorded_at=self.now(),
                idempotency_key_digest=_digest(request.idempotency_key),
            ).model_dump(mode="json")
        try:
            outcome = self._repository.apply_human_action(
                run_id=run_id,
                action=action,
                expected_version=request.expected_version,
                expected_summary_checksum=request.expected_summary_checksum,
                actor_subject_id=context.subject_id,
                actor_credential_id=context.credential_id,
                actor_authorization_revision=context.authorization_revision,
                idempotency_key=request.idempotency_key,
                request_digest=request_digest,
                now=self.now(),
                receipt_id=receipt_id,
                receipt_payload=receipt_payload,
                audit=AuditDraft(
                    audit_id=self._id_factory("audit"),
                    event_class=AuditEventClass.APPROVAL,
                    occurred_at=self.now(),
                    action_type=action,
                    actor=context.subject_id,
                    authorization_revision=context.authorization_revision,
                    previous_status=previous_status,
                    new_status=target_status,
                    previous_revision=run.header.revision,
                    new_revision=run.header.revision + 1,
                    result_checksum=run.summary_checksum,
                    idempotency_digest=_digest(request.idempotency_key),
                    receipt_id=receipt_id,
                ),
                events=self._action_events(
                    action=action,
                    previous_status=previous_status,
                    target_status=target_status,
                    decision=decision,
                ),
            )
        except RunNotFoundError as error:
            raise _not_found(run_id) from error
        except ExecutionInvalidated as error:
            raise _authority_lost(error, _ACTION_OPERATIONS[action]) from error
        except GrantRevokedError as error:
            raise _not_found(run_id) from error
        except StaleAuthorizationError as error:
            raise ReviewServiceError(
                status_code=409,
                code="authorization_revision_changed",
                message="The authorization changed; re-read the review before acting.",
            ) from error
        except VersionConflictError as error:
            raise ReviewServiceError(
                status_code=409,
                code="version_conflict",
                message="The review version is stale; refresh before acting.",
                details={
                    "expected_version": error.expected,
                    "current_version": error.current,
                },
            ) from error
        except IllegalTransitionError as error:
            raise ReviewServiceError(
                status_code=409,
                code="invalid_state_transition",
                message=(
                    f"Action {error.action} is not legal while the review is "
                    f"{error.status}."
                ),
                details={"action": error.action, "status": error.status},
            ) from error
        except SummaryChecksumError as error:
            raise ReviewServiceError(
                status_code=409,
                code="summary_checksum_mismatch",
                message="The presented review summary checksum is stale.",
                details={"summary_checksum": error.current},
            ) from error
        except IdempotencyConflictError as error:
            raise ReviewServiceError(
                status_code=409,
                code="idempotency_conflict",
                message="The action key was already used with a different payload.",
            ) from error
        if action is ApprovalActionType.CANCEL:
            for key in [item for item in self._tasks if item[0] == run_id]:
                task = self._tasks.pop(key)
                if not task.done():
                    task.cancel()
        logger.info(
            "human action applied: run_id=%s action=%s status=%s",
            run_id,
            action.value,
            target_status.value,
        )
        return self._verified_response(outcome.run)

    def _action_events(
        self,
        *,
        action: ApprovalActionType,
        previous_status: ReviewLifecycleStatus,
        target_status: ReviewLifecycleStatus,
        decision: HumanDecision | None,
    ) -> tuple[TraceDraft, ...]:
        events: list[TraceDraft] = []
        if previous_status in {
            ReviewLifecycleStatus.PROCESSING,
            ReviewLifecycleStatus.INTERRUPTED,
        }:
            events.append(
                TraceDraft(
                    event_type=RunEventType.STEP_CANCELLED.value,
                    step="review_workflow",
                    output_summary={"status": ReviewLifecycleStatus.CANCELLED.value},
                    source_versions={"execution": "review-execution.v1"},
                )
            )
        events.append(
            TraceDraft(
                event_type=(
                    RunEventType.CONFIRMATION_RECORDED.value
                    if decision is HumanDecision.CONFIRM_RECOMMENDATION
                    else RunEventType.HUMAN_ACTION_RECORDED.value
                ),
                step="human_gate",
                output_summary={"action": action.value, "status": target_status.value},
                source_versions={"approval_audit": "approval-audit.v2"},
            )
        )
        events.append(
            TraceDraft(
                event_type=(
                    RunEventType.RUN_CANCELLED.value
                    if action is ApprovalActionType.CANCEL
                    else RunEventType.RUN_COMPLETED.value
                ),
                step="review_run",
                output_summary={"status": target_status.value},
                source_versions={"api": API_TRANSPORT_VERSION},
            )
        )
        return tuple(events)

    def run_case_id(self, run_id: str, *, context: AuthContext) -> str:
        """The Case one Run belongs to, read before any business record is."""

        identity = self._repository.run_authorization_identity(run_id)
        if identity is None or identity.tenant_id != context.tenant_id:
            raise _not_found(run_id)
        return identity.case_id

    def get_review(self, run_id: str, *, context: AuthContext) -> ReviewRunResponseV2:
        return self._verified_response(self._load_authorized(run_id, context=context))

    def _load_authorized(self, run_id: str, *, context: AuthContext) -> PersistedRun:
        """Load one Run for a caller whose own access decision has already been made."""

        run = self._repository.load_run(run_id)
        if run is None or run.header.tenant_id != context.tenant_id:
            raise _not_found(run_id)
        return run

    def list_reviews(
        self,
        *,
        context: AuthContext,
        granted_case_ids: Sequence[str],
    ) -> ReviewListResponseV2:
        runs = self._repository.list_runs(
            tenant_id=context.tenant_id,
            case_ids=tuple(sorted(granted_case_ids)),
        )
        # A list verifies every whole Run and never silently drops polluted records.
        return ReviewListResponseV2(
            runs=tuple(self._verified_response(item) for item in runs)
        )

    def get_trace(self, run_id: str, *, context: AuthContext) -> object:
        run = self._load_authorized(run_id, context=context)
        with record_refusal():
            return self._repository.trace_envelope(
                run_id, complete=run.header.status in COMPLETE_STATUSES
            )

    def get_audit(self, run_id: str, *, context: AuthContext) -> tuple[object, ...]:
        self._load_authorized(run_id, context=context)
        with record_refusal():
            return self._repository.audit_events(run_id)

    def get_audit_projection(
        self, run_id: str, *, context: AuthContext
    ) -> tuple[tuple[object, ...], tuple[str, ...]]:
        """The audit events plus the legacy sequence anomalies of one Run."""

        self._load_authorized(run_id, context=context)
        with record_refusal():
            return self._repository.audit_projection(run_id)

    def get_scope(self, run_id: str, *, context: AuthContext) -> RuntimeScope:
        run = self._load_authorized(run_id, context=context)
        generation = run.generation.generation if run.generation else 1
        scope, _registration = self._runtime_from_run(run, generation)
        return scope

    async def shutdown(self) -> None:
        if self._scanner is not None:
            self._scanner.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._scanner
            self._scanner = None
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        _done, pending = await asyncio.wait(
            tasks, timeout=self._shutdown_timeout_seconds
        )
        if pending:
            self._abandoned_tasks += len(pending)

    def _runtime_from_run(
        self, run: PersistedRun, generation: int
    ) -> tuple[RuntimeScope, CaseRegistration]:
        registration = registration_from_run(run)
        active = run.generation
        if active is None or active.generation != generation:
            raise IncompatibleCheckpointFailure(
                "the committed execution generation is missing"
            )
        scope = RuntimeScope(
            tenant_id=run.header.tenant_id,
            case_id=run.header.case_id,
            case_version=run.header.case_version,
            case_type=registration.case_type,
            scenario=registration.scenario,
            subject_id=active.subject_id,
            credential_id=active.credential_id,
            authorization_revision=max(1, active.authorization_revision),
            generation=active.generation,
            document_version=run.header.document_version,
            dataset_version=run.header.dataset_version,
            index_version=run.header.index_version,
            rules_version=run.header.rules_version,
            case_document_ids=registration.case_document_ids,
            shared_document_ids=registration.shared_document_ids,
            allowed_order_ids=registration.allowed_order_ids,
            allowed_supplier_ids=registration.allowed_supplier_ids,
            allowed_transaction_ids=registration.allowed_transaction_ids,
            issued_at=active.created_at,
            deadline_at=active.deadline_at,
        )
        return scope, registration

    def _verified_response(self, run: PersistedRun) -> ReviewRunResponseV2:
        """The one outlet for an answer that carries a Run."""

        with record_refusal():
            verify_run_record(run)
            return build_run_response(run, resume_limit=self._resume_limit)


_ACTION_OPERATIONS = {
    ApprovalActionType.CONFIRM: SecurityOperation.REVIEW_CONFIRM,
    ApprovalActionType.REJECT: SecurityOperation.REVIEW_REJECT,
    ApprovalActionType.CANCEL: SecurityOperation.REVIEW_CANCEL,
}


def _find_signal(group: BaseException, kind: type[BaseException]):
    """Return the first nested exception of ``kind`` inside one exception group."""

    if isinstance(group, kind):
        return group
    if isinstance(group, BaseExceptionGroup):
        for item in group.exceptions:
            found = _find_signal(item, kind)
            if found is not None:
                return found
    return None


def _default_id_factory(prefix: str) -> str:
    return new_id(prefix)


def _digest(value: str) -> str:
    return canonical_checksum({"idempotency_key": value})


def _bounded(value: str) -> str:
    return value[:500] if value else "The local review execution failed closed."


def _authority_lost(
    error: ExecutionInvalidated, operation: SecurityOperation
) -> SecurityError:
    """The unified 401 for an authority withdrawn inside the caller's transaction."""

    return unauthenticated(
        AUTHORITY_DENIAL_REASON.get(error.reason, DenialReason.UNKNOWN_CREDENTIAL),
        operation,
    )


def _not_recoverable(reason: str | None) -> ReviewServiceError:
    """The safe 409 for a resume that can never succeed."""

    return ReviewServiceError(
        status_code=409,
        code="review_not_recoverable",
        message=(
            "This review cannot be resumed; create a new review for the same Case."
        ),
        details={"reason": reason or "checkpoint_incompatible"},
    )


def _not_found(run_id: str) -> ReviewServiceError:
    del run_id
    return ReviewServiceError(
        status_code=404,
        code="resource_not_found",
        message="No accessible resource exists for the requested identifier.",
    )


__all__ = [
    "AUTHORIZATION_REVOKED",
    "GENERATION_DEADLINE_SECONDS",
    "HEARTBEAT_SECONDS",
    "IncompatibleCheckpointFailure",
    "LEASE_SECONDS",
    "RUN_RESUME_LIMIT",
    "ReviewRunner",
    "ReviewService",
    "ReviewServiceError",
]
