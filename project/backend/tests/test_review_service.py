"""Persisted ``ReviewService`` lifecycle, idempotency, recovery and isolation tests.

``ReviewService`` is now backed by PostgreSQL: the database is the only authority for
Run state, checkpoints, attempts, idempotency, receipts and audit, and the in-process
service holds nothing but the handles of the executions it currently runs.  Every
scenario here therefore drives the real service over the real V1-03 security harness
(one disposable database per test, a real ``ReviewStore`` registration, a server-derived
:class:`AuthContext` and a real :class:`RuntimeScope` built by
:class:`AuthorizationService`), and reads the committed facts back through
``harness.repository`` and through a *second* service bound to the same store.

The only double is a mutable clock, used by the interruption scenario; no runner is
faked, so the persisted attempt, checkpoint and summary facts are the ones the product
runtime actually writes.  Every scenario shuts its service down inside a ``finally``: an
execution task still pending when the event loop closes can hang the process.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from procurement_review.api_contracts import (
    HumanActionRequestV2,
    ResumeRequestV2,
    ReviewCreateRequestV2,
    ReviewListResponseV2,
    ReviewRunResponseV2,
)
from procurement_review.audit_contracts import (
    ApprovalAuditEventV2,
    AuditEventClass,
)
from procurement_review.contracts import (
    ApprovalActionType,
    HumanDecision,
    RecommendationStatus,
    ReviewLifecycleStatus,
    RunEventType,
    WorkerKind,
)
from procurement_review.review_repository import (
    LEGAL_TRANSITIONS,
    PersistedRun,
    idempotency_key_digest,
)
from procurement_review.review_service import ReviewService, ReviewServiceError
from procurement_review.review_store import CaseRegistration
from procurement_review.security_contracts import (
    AuthContext,
    RuntimeScope,
    SecurityOperation,
)
from procurement_review.trace import TraceEnvelope, canonical_checksum

from v1_03_support import (
    ALPHA_APPROVER,
    ALPHA_OPERATOR,
    ALPHA_TENANT,
    BETA_OPERATOR,
    BETA_TENANT,
    DEMO_SUBJECT,
    DEMO_TENANT,
    SecurityHarness,
    build_service,
)

LOW_CASE = "CASE-LOW-001"
HIGH_CASE = "CASE-HIGH-001"
CONFLICT_CASE = "CASE-CONFLICT-001"
CANCEL_CASE = "CASE-CANCEL-001"
PARTIAL_FAILURE_CASE = "CASE-PARTIAL-FAILURE-001"
ALL_FAILURE_CASE = "CASE-ALL-FAILURE-001"
VERSION_MISMATCH_CASE = "CASE-VERSION-001"
RUNTIME_FAILURE_CASE = "CASE-RUNTIME-FAILURE-001"
ALPHA_CASE_ONE = "CASE-ALPHA-001"
BETA_CASE_TWO = "CASE-BETA-002"

WORKERS = ("contract", "policy", "procurement_data")
SETTLED_STATUSES = (
    ReviewLifecycleStatus.PENDING_CONFIRMATION,
    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW,
    ReviewLifecycleStatus.CONFIRMED,
    ReviewLifecycleStatus.REJECTED,
    ReviewLifecycleStatus.CANCELLED,
    ReviewLifecycleStatus.FAILED,
)

# A syntactically valid checksum used where the state transition, not the summary
# comparison, is the behaviour under test.
ZERO_CHECKSUM = "sha256:" + "0" * 64
CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class MutableClock:
    """A test clock the scenario moves; the service never reads wall time itself."""

    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, delta: timedelta) -> None:
        self.value = self.value + delta


@dataclass(frozen=True, slots=True)
class ScopedCase:
    """One authenticated, scope-bound view of exactly one registered Case."""

    harness: SecurityHarness
    service: ReviewService
    context: AuthContext
    registration: CaseRegistration
    scope: RuntimeScope

    @property
    def case_id(self) -> str:
        return self.registration.case_id

    def create_request(
        self,
        *,
        case_id: str | None = None,
        case_version: str | None = None,
    ) -> ReviewCreateRequestV2:
        return ReviewCreateRequestV2(
            case_id=self.case_id if case_id is None else case_id,
            case_version=(
                self.registration.case_version if case_version is None else case_version
            ),
        )

    async def create(
        self,
        idempotency_key: str,
        *,
        request: ReviewCreateRequestV2 | None = None,
        registration: CaseRegistration | None = None,
        scope: RuntimeScope | None = None,
    ) -> tuple[ReviewRunResponseV2, bool]:
        return await self.service.create_review(
            self.create_request() if request is None else request,
            registration=self.registration if registration is None else registration,
            scope=self.scope if scope is None else scope,
            context=self.context,
            idempotency_key=idempotency_key,
        )

    async def create_and_wait(
        self,
        idempotency_key: str,
        *,
        request: ReviewCreateRequestV2 | None = None,
    ) -> ReviewRunResponseV2:
        created, _ = await self.create(idempotency_key, request=request)
        return await self.wait(created.run_id)

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> ReviewRunResponseV2:
        return await self.wait_for(
            run_id, SETTLED_STATUSES, timeout_seconds=timeout_seconds
        )

    async def wait_for(
        self,
        run_id: str,
        statuses: tuple[ReviewLifecycleStatus, ...],
        *,
        timeout_seconds: float = 30.0,
    ) -> ReviewRunResponseV2:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        result = self.get(run_id)
        while result.status not in statuses:
            assert loop.time() < deadline, (
                f"run {run_id} never reached {statuses}: {result.status}"
            )
            await asyncio.sleep(0.01)
            result = self.get(run_id)
        return result

    def get(self, run_id: str) -> ReviewRunResponseV2:
        return self.service.get_review(run_id, context=self.context)

    def action(
        self,
        run_id: str,
        action: ApprovalActionType,
        request: HumanActionRequestV2,
    ) -> ReviewRunResponseV2:
        return self.service.apply_action(
            run_id=run_id,
            action=action,
            request=request,
            context=self.context,
        )

    def granted_case_ids(self) -> tuple[str, ...]:
        return tuple(
            item.case_id
            for item in self.harness.store.list_granted_cases(
                tenant_id=self.context.tenant_id,
                subject_id=self.context.subject_id,
            )
        )

    def list_runs(
        self,
        granted_case_ids: tuple[str, ...] | None = None,
    ) -> ReviewListResponseV2:
        return self.service.list_reviews(
            context=self.context,
            granted_case_ids=(
                self.granted_case_ids()
                if granted_case_ids is None
                else granted_case_ids
            ),
        )

    def trace(self, run_id: str) -> TraceEnvelope:
        return self.service.get_trace(run_id, context=self.context)

    def audit(self, run_id: str) -> tuple[ApprovalAuditEventV2, ...]:
        return self.service.get_audit(run_id, context=self.context)

    def scope_of(self, run_id: str) -> RuntimeScope:
        return self.service.get_scope(run_id, context=self.context)

    def persisted(self, run_id: str) -> PersistedRun | None:
        return self.harness.repository.load_run(run_id)


def open_case(
    harness: SecurityHarness,
    subject_id: str,
    case_id: str,
    *,
    service: ReviewService,
    generation: int = 1,
) -> ScopedCase:
    """Derive identity, Case registration and runtime scope from one credential."""

    token = harness.issue(subject_id)
    context = harness.authorization.resolve(
        token,
        request_id="test-review-service",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    registration = harness.authorization.require_case(
        context,
        case_id,
        request_id="test-review-service",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    scope = harness.authorization.build_runtime_scope(
        context,
        registration,
        generation=generation,
        deadline_at=harness.authorization.runtime_scope_deadline(30.0),
    )
    return ScopedCase(
        harness=harness,
        service=service,
        context=context,
        registration=registration,
        scope=scope,
    )


def run_scenario(
    harness: SecurityHarness,
    exercise: Callable[[ReviewService], Awaitable[None]],
    **service_kwargs: object,
) -> None:
    """Run one async scenario and always drain the service's executions."""

    service = build_service(harness.store, **service_kwargs)

    async def main() -> None:
        try:
            await exercise(service)
        finally:
            await service.shutdown()

    asyncio.run(main())


def action_request(
    idempotency_key: str | None,
    *,
    expected_version: int | None = None,
    expected_summary_checksum: str | None = None,
) -> HumanActionRequestV2:
    return HumanActionRequestV2(
        idempotency_key=idempotency_key,
        expected_version=expected_version,
        expected_summary_checksum=expected_summary_checksum,
    )


def stored_scalar(harness: SecurityHarness, sql: str, **parameters: object) -> int:
    """Read one persisted count straight from PostgreSQL, bypassing the service."""

    with harness.store.transaction() as connection:
        return int(connection.execute(text(sql), parameters).scalar_one())


def receipt_count(harness: SecurityHarness, run_id: str) -> int:
    return stored_scalar(
        harness,
        "SELECT count(*) FROM review_receipt WHERE run_id = :run_id",
        run_id=run_id,
    )


# ------------------------------------------------------------------- happy path


def test_low_case_produces_suggestion_evidence_facts_trace_and_checksum(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        created, is_new = await case.create("create-low-001")
        assert is_new is True
        assert created.status is ReviewLifecycleStatus.PROCESSING
        assert created.version == 1 and created.revision == 1
        assert created.summary is None
        assert created.summary_checksum is None
        assert created.receipt is None
        assert created.audit_event_count == 0
        assert created.recovery is not None
        assert created.recovery.adopted_workers == ()
        assert created.recovery.attempt_totals == {}

        # The committed generation rebuilds the same authorised scope, not a wider one.
        restored = case.scope_of(created.run_id)
        assert restored.tenant_id == case.scope.tenant_id
        assert restored.case_id == case.scope.case_id
        assert restored.case_version == case.scope.case_version
        assert restored.subject_id == case.scope.subject_id
        assert restored.credential_id == case.scope.credential_id
        assert restored.generation == case.scope.generation
        assert restored.index_version == case.scope.index_version
        assert restored.dataset_version == case.scope.dataset_version
        assert restored.allowed_order_ids == case.scope.allowed_order_ids

        pending = await case.wait(created.run_id)
        assert pending.status is ReviewLifecycleStatus.PENDING_CONFIRMATION
        assert pending.version == 2 and pending.revision == 2
        summary = pending.summary
        assert summary is not None
        assert summary.recommendation is RecommendationStatus.RECOMMEND_APPROVE
        assert summary.is_final_decision is False
        assert summary.run_id == pending.run_id
        assert pending.branch_failures == ()
        assert len(pending.evidence_refs) == 2
        assert len(pending.business_facts) == 5
        assert all(fact.read_only is True for fact in pending.business_facts)
        assert all(
            fact.dataset_version == case.scope.dataset_version
            for fact in pending.business_facts
        )
        assert pending.receipt is None
        assert pending.audit_event_count == 0
        assert case.audit(pending.run_id) == ()

        # The committed checksum is the canonical JSON SHA-256 of the stored summary.
        persisted = case.persisted(pending.run_id)
        assert persisted is not None
        assert persisted.summary_payload is not None
        assert persisted.summary_checksum == pending.summary_checksum
        canonical = json.dumps(
            persisted.summary_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        independent = "sha256:" + hashlib.sha256(canonical).hexdigest()
        assert pending.summary_checksum is not None
        assert CHECKSUM_PATTERN.fullmatch(pending.summary_checksum)
        assert pending.summary_checksum == independent
        assert pending.summary_checksum == canonical_checksum(
            summary.model_dump(mode="json")
        )
        assert case.get(pending.run_id).summary_checksum == independent

        # The trace is a separate, ordered record of the same execution.
        trace = case.trace(pending.run_id)
        assert trace.run_id == pending.run_id
        assert trace.complete is False
        assert len(trace.events) == pending.trace_event_count
        assert trace.events[0].event_type is RunEventType.RUN_STARTED
        assert tuple(event.event_id for event in trace.events) == tuple(
            sorted(event.event_id for event in trace.events)
        )

    run_scenario(fresh_harness, exercise)


# ----------------------------------------------------------------- idempotency


def test_create_replays_its_key_and_rejects_a_conflict_or_a_stale_case_version(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        first, is_new = await case.create("create-replay")
        assert is_new is True

        replay, replay_is_new = await case.create("create-replay")
        assert replay_is_new is False
        assert replay.run_id == first.run_id
        assert replay == first
        assert replay.status is ReviewLifecycleStatus.PROCESSING
        assert (
            stored_scalar(
                fresh_harness,
                "SELECT count(*) FROM review_idempotency WHERE run_id = :run_id",
                run_id=first.run_id,
            )
            == 1
        )

        # The same key with a different Case is a conflict, not a replay.
        other = open_case(fresh_harness, DEMO_SUBJECT, HIGH_CASE, service=service)
        with pytest.raises(ReviewServiceError) as conflict:
            await case.create(
                "create-replay",
                request=other.create_request(),
                registration=other.registration,
                scope=other.scope,
            )
        assert conflict.value.status_code == 409
        assert conflict.value.code == "idempotency_conflict"

        # A registered Case whose version moved is rejected before any run exists.
        with pytest.raises(ReviewServiceError) as stale_version:
            await case.create(
                "create-stale-version",
                request=case.create_request(case_version=f"{LOW_CASE.lower()}.v999"),
            )
        assert stale_version.value.status_code == 409
        assert stale_version.value.code == "case_version_mismatch"
        assert stale_version.value.details == {
            "registered_case_version": case.registration.case_version
        }

        # A new key still reaches the other Case and commits exactly one more run.
        fresh, fresh_is_new = await case.create(
            "create-other",
            request=other.create_request(),
            registration=other.registration,
            scope=other.scope,
        )
        assert fresh_is_new is True
        assert fresh.run_id != first.run_id
        assert (await other.wait(fresh.run_id)).status is (
            ReviewLifecycleStatus.PENDING_CONFIRMATION
        )
        assert {run.run_id for run in case.list_runs().runs} == {
            first.run_id,
            fresh.run_id,
        }

    run_scenario(fresh_harness, exercise)


def test_concurrent_duplicate_create_commits_exactly_one_run(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        results = await asyncio.gather(
            *(case.create("concurrent-create") for _ in range(12))
        )

        run_ids = {item[0].run_id for item in results}
        assert len(run_ids) == 1
        assert sum(1 for item in results if item[1]) == 1
        assert all(item[1] is False for item in results[1:])
        assert all(item[0] == results[0][0] for item in results)

        run_id = results[0][0].run_id
        pending = await case.wait(run_id)
        assert pending.status is ReviewLifecycleStatus.PENDING_CONFIRMATION
        assert len(case.list_runs().runs) == 1
        assert (
            stored_scalar(
                fresh_harness,
                "SELECT count(*) FROM review_run WHERE tenant_id = :tenant "
                "AND case_id = :case_id",
                tenant=DEMO_TENANT,
                case_id=LOW_CASE,
            )
            == 1
        )
        assert (
            stored_scalar(
                fresh_harness,
                "SELECT count(*) FROM review_idempotency "
                "WHERE operation_family = 'create' AND run_id = :run_id",
                run_id=run_id,
            )
            == 1
        )

    run_scenario(fresh_harness, exercise)


def test_idempotency_keys_are_namespaced_by_tenant_and_subject(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        alpha = open_case(
            fresh_harness, ALPHA_OPERATOR, ALPHA_CASE_ONE, service=service
        )
        approver = open_case(
            fresh_harness, ALPHA_APPROVER, ALPHA_CASE_ONE, service=service
        )
        beta = open_case(fresh_harness, BETA_OPERATOR, ALPHA_CASE_ONE, service=service)
        # Two tenants register the same Case ID with different content.
        assert alpha.context.tenant_id == ALPHA_TENANT
        assert beta.context.tenant_id == BETA_TENANT
        assert alpha.registration.case_id == beta.registration.case_id
        assert alpha.registration.checksum != beta.registration.checksum

        first, is_new = await alpha.create("shared-key")
        same_subject, replay_is_new = await alpha.create("shared-key")
        other_subject, other_subject_is_new = await approver.create("shared-key")
        other_tenant, other_tenant_is_new = await beta.create("shared-key")

        assert is_new is True
        assert replay_is_new is False
        assert same_subject.run_id == first.run_id
        assert other_subject_is_new is True
        assert other_subject.run_id != first.run_id
        assert other_tenant_is_new is True
        assert other_tenant.run_id != first.run_id
        assert len({first.run_id, other_subject.run_id, other_tenant.run_id}) == 3

        # One raw key leaves one digest but three independent namespaced records.
        digest = idempotency_key_digest("shared-key")
        assert (
            stored_scalar(
                fresh_harness,
                "SELECT count(*) FROM review_idempotency "
                "WHERE key_digest = :digest AND operation_family = 'create'",
                digest=digest,
            )
            == 3
        )
        assert (
            stored_scalar(
                fresh_harness,
                "SELECT count(DISTINCT tenant_id || '/' || subject_id) "
                "FROM review_idempotency WHERE key_digest = :digest "
                "AND operation_family = 'create'",
                digest=digest,
            )
            == 3
        )

        for scoped, run in (
            (alpha, first),
            (approver, other_subject),
            (beta, other_tenant),
        ):
            finished = await scoped.wait(run.run_id)
            assert finished.status in SETTLED_STATUSES

    run_scenario(fresh_harness, exercise)


# ---------------------------------------------------------------- transitions


def test_legal_transitions_are_frozen_and_include_interrupted() -> None:
    assert LEGAL_TRANSITIONS == {
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


def test_cancel_is_legal_from_every_non_terminal_state(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        # Processing has no result yet, so a decision is illegal and only cancel fits.
        processing = open_case(
            fresh_harness, DEMO_SUBJECT, CANCEL_CASE, service=service
        )
        created, _ = await processing.create("transitions-processing")
        assert created.status is ReviewLifecycleStatus.PROCESSING
        for action in (ApprovalActionType.CONFIRM, ApprovalActionType.REJECT):
            with pytest.raises(ReviewServiceError) as premature:
                processing.action(
                    created.run_id,
                    action,
                    action_request(
                        f"premature-{action.value}",
                        expected_version=created.version,
                        expected_summary_checksum=ZERO_CHECKSUM,
                    ),
                )
            assert premature.value.status_code == 409
            assert premature.value.code == "invalid_state_transition"
            # A decision before a committed result is refused before any receipt exists.
            assert premature.value.details == {
                "status": ReviewLifecycleStatus.PROCESSING.value
            }
        cancelled_processing = processing.action(
            created.run_id,
            ApprovalActionType.CANCEL,
            action_request("cancel-processing", expected_version=created.version),
        )
        assert cancelled_processing.status is ReviewLifecycleStatus.CANCELLED
        assert cancelled_processing.receipt is None

        # A pending confirmation may be cancelled without deciding anything.
        confirmable = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        pending = await confirmable.create_and_wait("transitions-pending-confirmation")
        assert pending.status is ReviewLifecycleStatus.PENDING_CONFIRMATION
        cancelled_pending = confirmable.action(
            pending.run_id,
            ApprovalActionType.CANCEL,
            action_request(
                "cancel-pending-confirmation", expected_version=pending.version
            ),
        )
        assert cancelled_pending.status is ReviewLifecycleStatus.CANCELLED
        assert cancelled_pending.receipt is None
        # Cancelling keeps the committed suggestion but records no decision.
        assert cancelled_pending.summary == pending.summary
        assert cancelled_pending.summary_checksum == pending.summary_checksum
        assert cancelled_pending.audit_event_count == 1

        # A pending human review is cancellable as well.
        reviewing = open_case(
            fresh_harness, DEMO_SUBJECT, CONFLICT_CASE, service=service
        )
        human = await reviewing.create_and_wait("transitions-pending-human-review")
        assert human.status is ReviewLifecycleStatus.PENDING_HUMAN_REVIEW
        assert human.summary is not None
        cancelled_human = reviewing.action(
            human.run_id,
            ApprovalActionType.CANCEL,
            action_request(
                "cancel-pending-human-review", expected_version=human.version
            ),
        )
        assert cancelled_human.status is ReviewLifecycleStatus.CANCELLED
        assert cancelled_human.receipt is None

    run_scenario(fresh_harness, exercise)


def test_cancel_is_legal_from_an_interrupted_run(
    fresh_harness: SecurityHarness,
) -> None:
    clock = MutableClock(datetime.now(UTC))

    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, CANCEL_CASE, service=service)
        created, _ = await case.create("interrupt-create")
        assert created.status is ReviewLifecycleStatus.PROCESSING

        # The execution lease expires without a heartbeat: the scanner interrupts it.
        clock.advance(timedelta(minutes=5))
        assert await service.scan_interruptions() == (created.run_id,)
        interrupted = case.get(created.run_id)
        assert interrupted.status is ReviewLifecycleStatus.INTERRUPTED
        assert interrupted.revision == created.revision + 1
        assert interrupted.recovery is not None
        assert interrupted.recovery.resumable is True
        assert interrupted.recovery.interrupted_reason == "lease_expired"

        cancelled = case.action(
            created.run_id,
            ApprovalActionType.CANCEL,
            action_request("cancel-interrupted", expected_version=interrupted.version),
        )
        assert cancelled.status is ReviewLifecycleStatus.CANCELLED
        assert cancelled.receipt is None
        assert cancelled.summary is None
        assert cancelled.recovery is not None
        assert cancelled.recovery.resumable is False

        # The interrupted generation never commits a late result over the decision.
        await asyncio.sleep(0.6)
        settled = case.get(created.run_id)
        assert settled.status is ReviewLifecycleStatus.CANCELLED
        assert settled.summary is None
        assert settled.summary_checksum is None
        assert receipt_count(fresh_harness, created.run_id) == 0

    run_scenario(fresh_harness, exercise, clock=clock, lease_seconds=0.05)


def test_every_action_on_a_terminal_run_is_illegal(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        pending = await case.create_and_wait("terminal-create")
        confirmed = case.action(
            pending.run_id,
            ApprovalActionType.CONFIRM,
            action_request(
                "terminal-confirm",
                expected_version=pending.version,
                expected_summary_checksum=pending.summary_checksum,
            ),
        )
        assert confirmed.status is ReviewLifecycleStatus.CONFIRMED

        for action in (
            ApprovalActionType.CONFIRM,
            ApprovalActionType.REJECT,
            ApprovalActionType.CANCEL,
        ):
            with pytest.raises(ReviewServiceError) as terminal:
                case.action(
                    confirmed.run_id,
                    action,
                    action_request(
                        f"after-{action.value}",
                        expected_version=confirmed.version,
                        expected_summary_checksum=pending.summary_checksum,
                    ),
                )
            assert terminal.value.status_code == 409
            assert terminal.value.code == "invalid_state_transition"
            assert terminal.value.details == {
                "action": action.value,
                "status": ReviewLifecycleStatus.CONFIRMED.value,
            }

        settled = case.get(confirmed.run_id)
        assert settled.status is ReviewLifecycleStatus.CONFIRMED
        assert settled.revision == confirmed.revision
        assert settled.receipt == confirmed.receipt
        assert len(case.audit(confirmed.run_id)) == 1
        assert receipt_count(fresh_harness, confirmed.run_id) == 1

    run_scenario(fresh_harness, exercise)


# -------------------------------------------------------------------- actions


def test_confirm_replay_is_identical_and_a_new_key_or_checksum_is_rejected(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        pending = await case.create_and_wait("replay-create")
        checksum = pending.summary_checksum
        assert checksum is not None

        first = case.action(
            pending.run_id,
            ApprovalActionType.CONFIRM,
            action_request(
                "replay-confirm",
                expected_version=pending.version,
                expected_summary_checksum=checksum,
            ),
        )
        assert first.status is ReviewLifecycleStatus.CONFIRMED
        assert first.revision == pending.revision + 1
        assert first.receipt is not None
        assert first.receipt.run_id == pending.run_id
        assert first.receipt.decision is HumanDecision.CONFIRM_RECOMMENDATION
        assert first.receipt.human_actor == DEMO_SUBJECT
        assert first.receipt.reviewed_recommendation is (
            RecommendationStatus.RECOMMEND_APPROVE
        )
        assert first.receipt.idempotency_key_digest == canonical_checksum(
            {"idempotency_key": "replay-confirm"}
        )
        assert first.receipt.external_write_performed is False
        assert first.audit_event_count == 1

        # The same key replays even though its expected_version is now stale.
        replay = case.action(
            pending.run_id,
            ApprovalActionType.CONFIRM,
            action_request(
                "replay-confirm",
                expected_version=pending.version,
                expected_summary_checksum=checksum,
            ),
        )
        assert replay == first
        assert replay.receipt == first.receipt
        assert replay.revision == first.revision
        persisted = case.persisted(pending.run_id)
        assert persisted is not None
        assert persisted.header.revision == first.revision
        assert persisted.receipt_payload == first.receipt.model_dump(mode="json")
        assert len(case.audit(pending.run_id)) == 1
        assert receipt_count(fresh_harness, pending.run_id) == 1

        # A new key with the stale version is a conflict, not a second decision.
        with pytest.raises(ReviewServiceError) as stale:
            case.action(
                pending.run_id,
                ApprovalActionType.REJECT,
                action_request(
                    "replay-new-key",
                    expected_version=pending.version,
                    expected_summary_checksum=checksum,
                ),
            )
        assert stale.value.status_code == 409
        assert stale.value.code == "version_conflict"
        assert stale.value.details == {
            "expected_version": pending.version,
            "current_version": first.revision,
        }

        # A bogus checksum on a live decision is a summary mismatch.
        other = open_case(fresh_harness, DEMO_SUBJECT, HIGH_CASE, service=service)
        other_pending = await other.create_and_wait("replay-checksum-create")
        other_checksum = other_pending.summary_checksum
        assert other_checksum is not None
        with pytest.raises(ReviewServiceError) as mismatch:
            other.action(
                other_pending.run_id,
                ApprovalActionType.CONFIRM,
                action_request(
                    "replay-bogus-checksum",
                    expected_version=other_pending.version,
                    expected_summary_checksum=ZERO_CHECKSUM,
                ),
            )
        assert mismatch.value.status_code == 409
        assert mismatch.value.code == "summary_checksum_mismatch"
        assert mismatch.value.details == {"summary_checksum": other_checksum}
        assert other.get(other_pending.run_id).status is (
            ReviewLifecycleStatus.PENDING_CONFIRMATION
        )
        assert other.audit(other_pending.run_id) == ()

    run_scenario(fresh_harness, exercise)


def test_actions_require_a_key_a_version_and_a_summary_checksum(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        pending = await case.create_and_wait("mandatory-create")
        checksum = pending.summary_checksum
        assert checksum is not None

        attempts = (
            (
                "idempotency_key_required",
                ApprovalActionType.CONFIRM,
                action_request(
                    None,
                    expected_version=pending.version,
                    expected_summary_checksum=checksum,
                ),
            ),
            (
                "expected_version_required",
                ApprovalActionType.CONFIRM,
                action_request(
                    "mandatory-no-version", expected_summary_checksum=checksum
                ),
            ),
            (
                "expected_summary_checksum_required",
                ApprovalActionType.CONFIRM,
                action_request(
                    "mandatory-no-checksum", expected_version=pending.version
                ),
            ),
            (
                "expected_summary_checksum_required",
                ApprovalActionType.REJECT,
                action_request(
                    "mandatory-no-checksum", expected_version=pending.version
                ),
            ),
        )
        for code, action, request in attempts:
            with pytest.raises(ReviewServiceError) as rejected:
                case.action(pending.run_id, action, request)
            assert rejected.value.status_code == 422
            assert rejected.value.code == code
        assert case.audit(pending.run_id) == ()
        assert receipt_count(fresh_harness, pending.run_id) == 0
        assert case.get(pending.run_id).status is (
            ReviewLifecycleStatus.PENDING_CONFIRMATION
        )

        # Cancel decides nothing about the summary, so it needs no checksum.
        cancelled = case.action(
            pending.run_id,
            ApprovalActionType.CANCEL,
            action_request("mandatory-cancel", expected_version=pending.version),
        )
        assert cancelled.status is ReviewLifecycleStatus.CANCELLED
        assert cancelled.receipt is None
        assert cancelled.audit_event_count == 1

    run_scenario(fresh_harness, exercise)


def test_two_sequential_confirms_leave_one_receipt_and_one_terminal_state(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        pending = await case.create_and_wait("serialised-confirm-create")
        checksum = pending.summary_checksum
        assert checksum is not None

        first = case.action(
            pending.run_id,
            ApprovalActionType.CONFIRM,
            action_request(
                "serialised-confirm-one",
                expected_version=pending.version,
                expected_summary_checksum=checksum,
            ),
        )
        assert first.status is ReviewLifecycleStatus.CONFIRMED
        assert first.receipt is not None

        with pytest.raises(ReviewServiceError) as second:
            case.action(
                pending.run_id,
                ApprovalActionType.CONFIRM,
                action_request(
                    "serialised-confirm-two",
                    expected_version=first.version,
                    expected_summary_checksum=checksum,
                ),
            )
        assert second.value.status_code == 409
        assert second.value.code == "invalid_state_transition"
        assert second.value.details == {
            "action": ApprovalActionType.CONFIRM.value,
            "status": ReviewLifecycleStatus.CONFIRMED.value,
        }

        settled = case.get(pending.run_id)
        assert settled.status is ReviewLifecycleStatus.CONFIRMED
        assert settled.revision == first.revision
        assert settled.receipt == first.receipt
        assert settled.audit_event_count == 1
        assert len(case.audit(pending.run_id)) == 1
        assert receipt_count(fresh_harness, pending.run_id) == 1

    run_scenario(fresh_harness, exercise)


# ------------------------------------------------------------------ isolation


def test_a_foreign_tenant_run_is_a_same_shape_404_and_absent_from_the_list(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        alpha = open_case(
            fresh_harness, ALPHA_OPERATOR, ALPHA_CASE_ONE, service=service
        )
        beta = open_case(fresh_harness, BETA_OPERATOR, ALPHA_CASE_ONE, service=service)
        beta_own = open_case(
            fresh_harness, BETA_OPERATOR, BETA_CASE_TWO, service=service
        )

        alpha_run = await alpha.create_and_wait("alpha-create")
        beta_run = await beta_own.create_and_wait("beta-create")
        assert alpha_run.run_id != beta_run.run_id

        assert alpha.get(alpha_run.run_id).run_id == alpha_run.run_id
        assert beta.get(beta_run.run_id).run_id == beta_run.run_id

        with pytest.raises(ReviewServiceError) as foreign:
            beta.get(alpha_run.run_id)
        with pytest.raises(ReviewServiceError) as unknown:
            beta.get("run-never-created")
        assert foreign.value.status_code == 404
        assert foreign.value.code == "resource_not_found"
        assert (
            foreign.value.status_code,
            foreign.value.code,
            foreign.value.message,
            foreign.value.details,
        ) == (
            unknown.value.status_code,
            unknown.value.code,
            unknown.value.message,
            unknown.value.details,
        )

        for read in (
            lambda: beta.trace(alpha_run.run_id),
            lambda: beta.audit(alpha_run.run_id),
            lambda: beta.scope_of(alpha_run.run_id),
        ):
            with pytest.raises(ReviewServiceError) as denied:
                read()
            assert denied.value.status_code == 404
            assert denied.value.code == "resource_not_found"

        with pytest.raises(ReviewServiceError) as denied_action:
            beta.action(
                alpha_run.run_id,
                ApprovalActionType.CANCEL,
                action_request("beta-cancel", expected_version=alpha_run.version),
            )
        assert denied_action.value.code == "resource_not_found"

        with pytest.raises(ReviewServiceError) as denied_resume:
            await beta.service.resume_review(
                alpha_run.run_id,
                ResumeRequestV2(expected_version=alpha_run.version),
                context=beta.context,
                idempotency_key="beta-resume",
            )
        assert denied_resume.value.status_code == 404
        assert denied_resume.value.code == "resource_not_found"

        with pytest.raises(ReviewServiceError) as cross_tenant_create:
            await beta.create(
                "beta-create",
                request=alpha.create_request(),
                registration=alpha.registration,
                scope=alpha.scope,
            )
        assert cross_tenant_create.value.status_code == 404
        assert cross_tenant_create.value.code == "resource_not_found"

        # Listing is narrowed by tenant *and* by the caller's live Case grants.
        assert {run.run_id for run in beta.list_runs().runs} == {beta_run.run_id}
        assert {run.run_id for run in alpha.list_runs().runs} == {alpha_run.run_id}
        assert beta.list_runs(granted_case_ids=(ALPHA_CASE_ONE,)).runs == ()
        assert alpha.list_runs(granted_case_ids=(BETA_CASE_TWO,)).runs == ()
        assert beta.list_runs(granted_case_ids=()).runs == ()

    run_scenario(fresh_harness, exercise)


# ------------------------------------------------------------------- recovery


def test_persisted_attempts_checkpoints_and_history_survive_a_new_service(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        pending = await case.create_and_wait("history-create")
        assert pending.status is ReviewLifecycleStatus.PENDING_CONFIRMATION

        run = fresh_harness.repository.load_run(pending.run_id)
        assert run is not None
        assert run.header.status is ReviewLifecycleStatus.PENDING_CONFIRMATION
        assert run.header.revision == pending.revision == 2
        assert run.summary_checksum == pending.summary_checksum
        assert run.summary_payload is not None
        assert run.receipt_payload is None

        # One adopted checkpoint and exactly one completed attempt per worker.
        assert sorted(run.checkpoints) == list(WORKERS)
        for worker, payload in run.checkpoints.items():
            assert set(payload) == {"input_checksum", "contract_checksum", "result"}
            assert CHECKSUM_PATTERN.fullmatch(str(payload["input_checksum"]))
            assert CHECKSUM_PATTERN.fullmatch(str(payload["contract_checksum"]))
            assert payload["result"]["worker"] == worker
            assert payload["result"]["outcome"] == "completed"

        assert len(run.attempts) == len(WORKERS)
        attempts = {item.worker: item for item in run.attempts}
        assert sorted(attempts) == list(WORKERS)
        for attempt in attempts.values():
            assert attempt.attempt_no == 1
            assert attempt.generation == 1
            assert attempt.outcome == "completed"
            assert attempt.finished_at is not None
            assert attempt.abandoned is False
            assert attempt.discarded_late is False
            assert attempt.last_error is None
        assert pending.recovery is not None
        assert pending.recovery.adopted_workers == WORKERS
        assert pending.recovery.attempt_totals == {worker: 1 for worker in WORKERS}
        assert pending.recovery.generation == 1

        # A second service bound to the same store reads the identical history back.
        reopened = build_service(fresh_harness.store)
        try:
            again = reopened.get_review(pending.run_id, context=case.context)
            assert again == pending
            assert reopened.get_trace(pending.run_id, context=case.context) == (
                case.trace(pending.run_id)
            )
            assert reopened.get_audit(pending.run_id, context=case.context) == ()
            restored = reopened.get_scope(pending.run_id, context=case.context)
            assert restored.generation == 1
            assert restored.case_id == case.case_id
            assert again.recovery is not None
            assert again.recovery.adopted_workers == WORKERS
            assert again.recovery.attempt_totals == {worker: 1 for worker in WORKERS}
        finally:
            await reopened.shutdown()

    run_scenario(fresh_harness, exercise)


# ------------------------------------------------------------- closed failures


def test_partial_and_total_worker_failures_reach_the_human_gate(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        partial = open_case(
            fresh_harness, DEMO_SUBJECT, PARTIAL_FAILURE_CASE, service=service
        )
        partial_run = await partial.create_and_wait("create-partial-failure")
        assert partial_run.status is ReviewLifecycleStatus.PENDING_HUMAN_REVIEW
        assert partial_run.summary is not None
        assert partial_run.summary.recommendation is (
            RecommendationStatus.INSUFFICIENT_INFORMATION
        )
        assert partial_run.summary.is_final_decision is False
        assert partial_run.receipt is None
        assert tuple(
            (item.worker, item.code) for item in partial_run.branch_failures
        ) == ((WorkerKind.CONTRACT, "model_failure"),)

        total = open_case(
            fresh_harness, DEMO_SUBJECT, ALL_FAILURE_CASE, service=service
        )
        total_run = await total.create_and_wait("create-all-failure")
        assert total_run.status is ReviewLifecycleStatus.PENDING_HUMAN_REVIEW
        assert total_run.summary is not None
        assert total_run.summary.recommendation is (
            RecommendationStatus.INSUFFICIENT_INFORMATION
        )
        assert total_run.summary.is_final_decision is False
        assert total_run.receipt is None
        assert tuple(
            (item.worker, item.code) for item in total_run.branch_failures
        ) == (
            (WorkerKind.CONTRACT, "model_failure"),
            (WorkerKind.POLICY, "model_failure"),
            (WorkerKind.PROCUREMENT_DATA, "model_failure"),
        )

    run_scenario(fresh_harness, exercise)


def test_version_mismatch_case_fails_closed_with_visible_version_errors(
    fresh_harness: SecurityHarness,
) -> None:
    """The registered Case pins a stale index and dataset version for its workers."""

    async def exercise(service: ReviewService) -> None:
        case = open_case(
            fresh_harness, DEMO_SUBJECT, VERSION_MISMATCH_CASE, service=service
        )
        assert case.registration.scenario == "version_mismatch"
        assert case.scope.index_version == case.registration.index_version

        result = await case.create_and_wait("create-version-mismatch")
        assert result.status is ReviewLifecycleStatus.PENDING_HUMAN_REVIEW
        assert result.summary is not None
        assert result.summary.recommendation is (
            RecommendationStatus.INSUFFICIENT_INFORMATION
        )
        assert result.summary.is_final_decision is False
        assert result.receipt is None
        assert result.error_code is None
        assert tuple((item.worker, item.code) for item in result.branch_failures) == (
            (WorkerKind.CONTRACT, "version_mismatch"),
            (WorkerKind.POLICY, "version_mismatch"),
            (WorkerKind.PROCUREMENT_DATA, "version_mismatch"),
        )

    run_scenario(fresh_harness, exercise)


def test_declared_runtime_failure_fails_closed_and_replays_without_rerunning(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(
            fresh_harness, DEMO_SUBJECT, RUNTIME_FAILURE_CASE, service=service
        )
        created, is_new = await case.create("create-runtime-failure")
        assert is_new is True
        failed = await case.wait(created.run_id)

        assert failed.status is ReviewLifecycleStatus.FAILED
        assert failed.summary is None
        assert failed.summary_checksum is None
        assert failed.receipt is None
        assert failed.audit_event_count == 0
        assert failed.error_code == "review_execution_failed"
        # The injected failure detail never reaches the persisted contract.
        assert failed.error_message == "The local review execution failed closed."
        assert "injected" not in failed.error_message
        trace = case.trace(failed.run_id)
        assert trace.complete is True
        assert trace.events[-1].event_type is RunEventType.RUN_FAILED
        assert case.audit(failed.run_id) == ()

        # The failed key replays the failed run; a new key may retry.
        replay, replay_is_new = await case.create("create-runtime-failure")
        assert replay_is_new is False
        assert replay.run_id == failed.run_id
        assert replay == failed

        retry, retry_is_new = await case.create("retry-runtime-failure")
        assert retry_is_new is True
        assert retry.run_id != failed.run_id
        retried = await case.wait(retry.run_id)
        assert retried.status is ReviewLifecycleStatus.FAILED
        assert len(case.list_runs().runs) == 2
        for run_id in (failed.run_id, retried.run_id):
            persisted = case.persisted(run_id)
            assert persisted is not None
            assert persisted.checkpoints == {}
            assert persisted.summary_payload is None

    run_scenario(fresh_harness, exercise)


# ---------------------------------------------------- separated local records


def test_trace_audit_and_receipt_stay_separate_and_never_store_the_raw_key(
    fresh_harness: SecurityHarness,
) -> None:
    async def exercise(service: ReviewService) -> None:
        case = open_case(fresh_harness, DEMO_SUBJECT, LOW_CASE, service=service)
        pending = await case.create_and_wait("separation-create")
        completed = case.action(
            pending.run_id,
            ApprovalActionType.CONFIRM,
            action_request(
                "separation-confirm",
                expected_version=pending.version,
                expected_summary_checksum=pending.summary_checksum,
            ),
        )
        trace = case.trace(completed.run_id)
        audit = case.audit(completed.run_id)
        assert completed.receipt is not None

        assert trace.complete is True
        assert completed.trace_event_count == len(trace.events)
        assert trace.events[-1].event_type is RunEventType.RUN_COMPLETED
        assert any(
            event.event_type is RunEventType.CONFIRMATION_RECORDED
            for event in trace.events
        )
        assert tuple(event.event_id for event in trace.events) == tuple(
            sorted(event.event_id for event in trace.events)
        )

        assert len(audit) == 1
        event = audit[0]
        assert event.event_class is AuditEventClass.APPROVAL
        assert event.action_type is ApprovalActionType.CONFIRM
        assert event.actor == DEMO_SUBJECT
        assert event.previous_status is ReviewLifecycleStatus.PENDING_CONFIRMATION
        assert event.new_status is ReviewLifecycleStatus.CONFIRMED
        assert event.receipt_id == completed.receipt.receipt_id
        assert event.idempotency_digest == (completed.receipt.idempotency_key_digest)
        assert event.external_write_performed is False

        record = json.dumps(
            {
                "trace": trace.model_dump(mode="json"),
                "audit": [item.model_dump(mode="json") for item in audit],
                "receipt": completed.receipt.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        assert "separation-confirm" not in record
        assert "evidence_refs" not in record
        assert "business_fact_refs" not in record

        # PostgreSQL stores only the digest of the raw key.
        assert (
            stored_scalar(
                fresh_harness,
                "SELECT count(*) FROM review_idempotency WHERE key_digest = :raw",
                raw="separation-confirm",
            )
            == 0
        )
        assert (
            stored_scalar(
                fresh_harness,
                "SELECT count(*) FROM review_idempotency "
                "WHERE key_digest = :digest AND operation_family = 'human_action'",
                digest=idempotency_key_digest("separation-confirm"),
            )
            == 1
        )

    run_scenario(fresh_harness, exercise)
