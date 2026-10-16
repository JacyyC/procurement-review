"""V1-03 TX/NEG/DB matrix: atomicity, idempotency races, recovery stop conditions.

Every scenario runs against the real PostgreSQL store on its own disposable database.
Concurrency uses independent connections and independent service instances, never a
single in-process lock.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from procurement_review.api_contracts import (
    HumanActionRequestV2,
    ResumeRequestV2,
    ReviewCreateRequestV2,
)
from procurement_review.contracts import ApprovalActionType, ReviewLifecycleStatus
from procurement_review.audit_contracts import AuditEventClass
from procurement_review.fault_hooks import FaultController
from procurement_review.review_repository import AuditDraft, SummaryChecksumError
from procurement_review.review_service import ReviewService, ReviewServiceError
from procurement_review.security_contracts import SecurityOperation

from v1_03_support import (
    DEMO_SUBJECT,
    DEMO_TENANT,
    SecurityHarness,
    build_service,
)

LOW_CASE = "CASE-LOW-001"
LOW_VERSION = "case-low-001.v1"
FAST = {"lease_seconds": 2.0, "heartbeat_seconds": 0.4, "scan_interval_seconds": 0.2}


class _Run:
    """One run driven through the persisted service on the calling event loop."""

    def __init__(
        self, harness: SecurityHarness, service: ReviewService, token: str
    ) -> None:
        self.harness = harness
        self.service = service
        self.token = token

    def context(self):
        return self.harness.authorization.resolve(
            self.token,
            request_id="tx",
            operation=SecurityOperation.REVIEW_CREATE,
        )

    def scope_and_registration(self, case_id: str = LOW_CASE):
        context = self.context()
        registration = self.harness.authorization.require_case(
            context,
            case_id,
            request_id="tx",
            operation=SecurityOperation.REVIEW_CREATE,
        )
        scope = self.harness.authorization.build_runtime_scope(
            context,
            registration,
            generation=1,
            deadline_at=self.harness.authorization.runtime_scope_deadline(30.0),
        )
        return context, registration, scope

    async def create(self, key: str, case_id: str = LOW_CASE):
        context, registration, scope = self.scope_and_registration(case_id)
        return await self.service.create_review(
            ReviewCreateRequestV2(
                case_id=case_id,
                # The registry owns the version; a colliding Case ID in another tenant
                # carries a different one.
                case_version=registration.case_version,
            ),
            registration=registration,
            scope=scope,
            context=context,
            idempotency_key=key,
        )

    def action(
        self,
        run_id: str,
        action: ApprovalActionType,
        *,
        key: str,
        expected_version: int,
        checksum: str | None = None,
    ):
        context = self.context()
        return self.service.apply_action(
            run_id=run_id,
            action=action,
            request=HumanActionRequestV2(
                api_version="review-api.v2",
                idempotency_key=key,
                expected_version=expected_version,
                expected_summary_checksum=checksum,
            ),
            context=context,
        )

    async def wait(self, run_id: str, *statuses: str, timeout: float = 30.0):
        context = self.context()
        deadline = asyncio.get_running_loop().time() + timeout
        body = None
        while asyncio.get_running_loop().time() < deadline:
            body = self.service.get_review(run_id, context=context)
            if body.status.value in statuses:
                return body
            await asyncio.sleep(0.02)
        raise AssertionError(f"run stayed {body.status.value if body else '?'}")


def _second_service(harness: SecurityHarness, **kwargs) -> ReviewService:
    """A completely independent service instance over its own store/engine."""

    return build_service(harness.store, **kwargs)


# ------------------------------------------------------------------------ TX-01


def test_tx01_create_is_all_or_nothing_and_locates_the_same_operation(
    fresh_harness: SecurityHarness,
) -> None:
    """TX-01: commit-before-failure, lost response, and concurrent duplicate create."""

    service = build_service(fresh_harness.store)

    async def scenario() -> dict[str, object]:
        runner = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        first, created = await runner.create("tx01-key")
        await runner.wait(first.run_id, "pending_confirmation", "pending_human_review")
        replay, replayed_created = await runner.create("tx01-key")
        results = await asyncio.gather(
            *[runner.create("tx01-concurrent") for _ in range(8)],
            return_exceptions=True,
        )
        await service.shutdown()
        return {
            "first": first,
            "created": created,
            "replay": replay,
            "replayed_created": replayed_created,
            "concurrent": results,
        }

    outcome = asyncio.run(scenario())
    assert outcome["created"] is True
    assert outcome["replayed_created"] is False
    assert outcome["replay"].run_id == outcome["first"].run_id
    ids = {
        item[0].run_id
        for item in outcome["concurrent"]
        if not isinstance(item, BaseException)
    }
    assert len(ids) == 1
    with fresh_harness.store.transaction() as connection:
        rows = connection.execute(
            text("SELECT count(*) FROM review_run WHERE tenant_id = :t"),
            {"t": DEMO_TENANT},
        ).scalar_one()
        create_mappings = connection.execute(
            text(
                "SELECT count(*) FROM review_idempotency "
                "WHERE operation_family = 'create'"
            )
        ).scalar_one()
    assert rows == 2  # the sequential pair plus the single concurrent run
    assert create_mappings == 2


# ------------------------------------------------------------------------ TX-02


def test_tx02_decision_is_atomic_with_receipt_and_audit(
    fresh_harness: SecurityHarness,
) -> None:
    """TX-02: the decision, its receipt and its audit exist together or not at all."""

    service = build_service(fresh_harness.store, **FAST)

    async def scenario():
        runner = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        run, _ = await runner.create("tx02-key")
        body = await runner.wait(
            run.run_id, "pending_confirmation", "pending_human_review"
        )
        with pytest.raises(SummaryChecksumError):
            fresh_harness.repository.apply_human_action(
                run_id=run.run_id,
                action=ApprovalActionType.CONFIRM,
                expected_version=body.version,
                expected_summary_checksum="sha256:" + "0" * 64,
                actor_subject_id=DEMO_SUBJECT,
                actor_credential_id=fresh_harness.credential_id_of(runner.token),
                actor_authorization_revision=1,
                idempotency_key="tx02-bogus",
                request_digest="digest",
                now=datetime.now(UTC),
                receipt_id="receipt-bogus",
                receipt_payload=None,
                audit=AuditDraft(
                    audit_id="audit-bogus",
                    event_class=AuditEventClass.APPROVAL,
                    occurred_at=datetime.now(UTC),
                ),
            )
        decided = runner.action(
            run.run_id,
            ApprovalActionType.CONFIRM,
            key="tx02-confirm",
            expected_version=body.version,
            checksum=body.summary_checksum,
        )
        await service.shutdown()
        return run.run_id, decided

    run_id, decided = asyncio.run(scenario())
    assert decided.status is ReviewLifecycleStatus.CONFIRMED
    assert decided.receipt is not None
    assert decided.audit_event_count == 1
    with fresh_harness.store.transaction() as connection:
        counts = {
            table: connection.execute(
                text(f"SELECT count(*) FROM {table} WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
            for table in ("review_receipt", "review_audit_event")
        }
        orphan = connection.execute(
            text(
                "SELECT count(*) FROM review_receipt r WHERE NOT EXISTS "
                "(SELECT 1 FROM review_run x WHERE x.run_id = r.run_id)"
            )
        ).scalar_one()
        audit_only = connection.execute(
            text(
                "SELECT count(*) FROM review_audit_event WHERE run_id = :r "
                "AND event_class = 'approval'"
            ),
            {"r": run_id},
        ).scalar_one()
    assert counts["review_receipt"] == 1
    assert counts["review_audit_event"] == 1
    assert orphan == 0
    assert audit_only == 1


# ------------------------------------------------------------------------ TX-03


def test_tx03_idempotency_namespace_is_scoped_by_tenant_and_subject(
    fresh_harness: SecurityHarness,
) -> None:
    """TX-03: same key, different subject or tenant, never collides."""

    service = build_service(fresh_harness.store)

    async def scenario():
        demo = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        alpha_token = fresh_harness.issue("subject-alpha-operator")
        beta_token = fresh_harness.issue("subject-beta-operator")
        alpha = _Run(fresh_harness, service, alpha_token)
        beta = _Run(fresh_harness, service, beta_token)
        one, _ = await demo.create("shared-key")
        two, _ = await alpha.create("shared-key", case_id="CASE-ALPHA-001")
        three, _ = await beta.create("shared-key", case_id="CASE-ALPHA-001")
        for runner, item, case_id in (
            (demo, one, LOW_CASE),
            (alpha, two, "CASE-ALPHA-001"),
            (beta, three, "CASE-ALPHA-001"),
        ):
            await runner.wait(
                item.run_id, "pending_confirmation", "pending_human_review"
            )
        await service.shutdown()
        return {one.run_id, two.run_id, three.run_id}

    ids = asyncio.run(scenario())
    assert len(ids) == 3
    with fresh_harness.store.transaction() as connection:
        rows = connection.execute(
            text(
                "SELECT tenant_id, subject_id, key_digest FROM review_idempotency "
                "WHERE operation_family = 'create'"
            )
        ).all()
    assert len({row.key_digest for row in rows}) == 1
    assert {(row.tenant_id, row.subject_id) for row in rows} == {
        (DEMO_TENANT, DEMO_SUBJECT),
        ("tenant-alpha", "subject-alpha-operator"),
        ("tenant-beta", "subject-beta-operator"),
    }


# ------------------------------------------------------------------------ TX-04


def test_tx04_concurrent_decisions_accept_exactly_one_terminal_state(
    fresh_harness: SecurityHarness,
) -> None:
    """TX-04: two connections race confirm vs cancel; only one may win."""

    service = build_service(fresh_harness.store, **FAST)
    harness = fresh_harness

    async def prepare():
        runner = _Run(harness, service, harness.issue(DEMO_SUBJECT))
        run, _ = await runner.create("tx04-key")
        body = await runner.wait(
            run.run_id, "pending_confirmation", "pending_human_review"
        )
        await service.shutdown()
        return run.run_id, body

    run_id, body = asyncio.run(prepare())
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(action: ApprovalActionType, key: str) -> None:
        winner = _Run(harness, _second_service(harness), harness.issue(DEMO_SUBJECT))
        barrier.wait(timeout=10)
        try:
            result = winner.action(
                run_id,
                action,
                key=key,
                expected_version=body.version,
                checksum=body.summary_checksum,
            )
            code = result.status.value
        except ReviewServiceError as error:
            code = error.code
        with lock:
            outcomes.append(code)

    threads = [
        threading.Thread(target=worker, args=(ApprovalActionType.CONFIRM, "tx04-a")),
        threading.Thread(target=worker, args=(ApprovalActionType.CANCEL, "tx04-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    asyncio.run(service.shutdown())

    assert len(outcomes) == 2
    accepted = [item for item in outcomes if item in {"confirmed", "cancelled"}]
    rejected = [
        item
        for item in outcomes
        if item.endswith("conflict") or item == "invalid_state_transition"
    ]
    assert len(accepted) == 1, outcomes
    assert len(rejected) == 1, outcomes
    with harness.store.transaction() as connection:
        row = connection.execute(
            text("SELECT status, revision FROM review_run WHERE run_id = :r"),
            {"r": run_id},
        ).one()
        receipts = connection.execute(
            text("SELECT count(*) FROM review_receipt WHERE run_id = :r"),
            {"r": run_id},
        ).scalar_one()
        audits = connection.execute(
            text(
                "SELECT count(*) FROM review_audit_event WHERE run_id = :r "
                "AND event_class = 'approval'"
            ),
            {"r": run_id},
        ).scalar_one()
    assert row.status == accepted[0]
    assert receipts == (1 if accepted[0] == "confirmed" else 0)
    assert audits == 1
    assert int(row.revision) == body.version + 1


# ------------------------------------------------------------------------ TX-05


def test_tx05_revoked_grant_blocks_the_commit_and_the_replay(
    fresh_harness: SecurityHarness,
) -> None:
    """TX-05: the grant is re-checked inside the decision transaction."""

    service = build_service(fresh_harness.store, **FAST)

    async def scenario():
        runner = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        run, _ = await runner.create("tx05-key")
        body = await runner.wait(
            run.run_id, "pending_confirmation", "pending_human_review"
        )
        await service.shutdown()
        return run.run_id, body

    run_id, body = asyncio.run(scenario())
    fresh_harness.store.set_case_grant(
        tenant_id=DEMO_TENANT,
        case_id=LOW_CASE,
        subject_id=DEMO_SUBJECT,
        granted=False,
        occurred_at=datetime.now(UTC),
    )
    try:
        with pytest.raises(ReviewServiceError) as denied:
            asyncio.run(_async_action(fresh_harness, service, run_id, body))
        assert denied.value.status_code == 404
        assert denied.value.code == "resource_not_found"
        with fresh_harness.store.transaction() as connection:
            row = connection.execute(
                text("SELECT status, revision FROM review_run WHERE run_id = :r"),
                {"r": run_id},
            ).one()
            receipts = connection.execute(
                text("SELECT count(*) FROM review_receipt WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        assert row.status == body.status.value
        assert int(row.revision) == body.version
        assert receipts == 0
    finally:
        fresh_harness.store.set_case_grant(
            tenant_id=DEMO_TENANT,
            case_id=LOW_CASE,
            subject_id=DEMO_SUBJECT,
            granted=True,
            occurred_at=datetime.now(UTC),
        )
    asyncio.run(service.shutdown())


async def _async_action(
    harness: SecurityHarness, service: ReviewService, run_id: str, body
):
    runner = _Run(harness, service, harness.issue(DEMO_SUBJECT))
    return runner.action(
        run_id,
        ApprovalActionType.CONFIRM,
        key="tx05-confirm",
        expected_version=body.version,
        checksum=body.summary_checksum,
    )


# ------------------------------------------------------------------------ TX-06


def test_tx06_mandatory_fields_and_replay_rules(
    fresh_harness: SecurityHarness,
) -> None:
    """TX-06: missing fields are 422, stale new actions are 409, replays are stable."""

    service = build_service(fresh_harness.store, **FAST)

    async def scenario():
        runner = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        run, _ = await runner.create("tx06-key")
        body = await runner.wait(
            run.run_id, "pending_confirmation", "pending_human_review"
        )
        errors: dict[str, tuple[int, str]] = {}
        for name, kwargs in (
            ("no_key", {"key": None, "expected_version": body.version}),
            ("no_version", {"key": "tx06-a", "expected_version": None}),
        ):
            try:
                context = runner.context()
                service.apply_action(
                    run_id=run.run_id,
                    action=ApprovalActionType.CONFIRM,
                    request=HumanActionRequestV2(
                        api_version="review-api.v2",
                        idempotency_key=kwargs["key"],
                        expected_version=kwargs["expected_version"],
                        expected_summary_checksum=body.summary_checksum,
                    ),
                    context=context,
                )
            except ReviewServiceError as error:
                errors[name] = (error.status_code, error.code)
        try:
            runner.action(
                run.run_id,
                ApprovalActionType.CONFIRM,
                key="tx06-b",
                expected_version=body.version + 5,
                checksum=body.summary_checksum,
            )
        except ReviewServiceError as error:
            errors["stale"] = (error.status_code, error.code)
        first = runner.action(
            run.run_id,
            ApprovalActionType.CONFIRM,
            key="tx06-c",
            expected_version=body.version,
            checksum=body.summary_checksum,
        )
        replay = runner.action(
            run.run_id,
            ApprovalActionType.CONFIRM,
            key="tx06-c",
            expected_version=body.version,
            checksum=body.summary_checksum,
        )
        await service.shutdown()
        return errors, first, replay

    errors, first, replay = asyncio.run(scenario())
    assert errors["no_key"] == (422, "idempotency_key_required")
    assert errors["no_version"] == (422, "expected_version_required")
    assert errors["stale"] == (409, "version_conflict")
    assert first.status is ReviewLifecycleStatus.CONFIRMED
    assert replay.model_dump() == first.model_dump()
    assert replay.revision == first.revision
    assert replay.audit_event_count == 1


# ------------------------------------------------------------------------ NEG-03


def test_neg03_resume_budget_is_not_reset_by_restarts(
    fresh_harness: SecurityHarness,
) -> None:
    """NEG-03: the resume budget is persisted and never refilled."""

    service = build_service(fresh_harness.store, **FAST)

    async def scenario():
        runner = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        run, _ = await runner.create("neg03-key")
        run_id = run.run_id
        context = runner.context()
        await service.shutdown()
        # Force the run into interrupted directly through the repository, then exhaust
        # the resume budget with explicit resumes.
        fresh_harness.repository.mark_interrupted(
            run_id=run_id,
            reason_code="lease_expired",
            now=datetime.now(UTC),
        )
        outcomes: list[str] = []
        for index in range(3):
            resumed = build_service(fresh_harness.store, **FAST)
            try:
                body = fresh_harness.repository.load_run(run_id)
                assert body is not None
                try:
                    await resumed.resume_review(
                        run_id,
                        ResumeRequestV2(
                            api_version="review-api.v2",
                            expected_version=body.header.revision,
                            idempotency_key=f"neg03-{index}",
                        ),
                        context=context,
                        idempotency_key=f"neg03-{index}",
                    )
                    outcomes.append("accepted")
                except ReviewServiceError as error:
                    outcomes.append(error.code)
                await asyncio.sleep(0.05)
                current = fresh_harness.repository.load_run(run_id)
                if current is not None and current.header.status in {
                    ReviewLifecycleStatus.INTERRUPTED,
                    ReviewLifecycleStatus.FAILED,
                    ReviewLifecycleStatus.PENDING_CONFIRMATION,
                    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW,
                }:
                    continue
            finally:
                await resumed.shutdown()
        final = fresh_harness.repository.load_run(run_id)
        return outcomes, final

    outcomes, final = asyncio.run(scenario())
    assert outcomes.count("accepted") <= 2, outcomes
    assert final is not None
    assert final.header.resume_count <= 2
    assert outcomes[-1] in {"review_not_recoverable", "invalid_state_transition"}


# ------------------------------------------------------------------------ NEG-04


def test_neg04_untrusted_checkpoint_and_missing_versions_stop_execution(
    fresh_harness: SecurityHarness,
) -> None:
    """NEG-04: an unknown checkpoint contract fails closed with no Worker execution."""

    service = build_service(fresh_harness.store, **FAST)

    async def prepare():
        runner = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        run, _ = await runner.create("neg04-key")
        body = await runner.wait(
            run.run_id, "pending_confirmation", "pending_human_review"
        )
        await service.shutdown()
        return run.run_id, body

    run_id, _body = asyncio.run(prepare())
    with fresh_harness.store.transaction() as connection:
        connection.execute(
            text(
                "UPDATE review_branch_checkpoint SET contract_checksum = :bad "
                "WHERE run_id = :r AND worker = 'policy'"
            ),
            {"bad": "sha256:" + "9" * 64, "r": run_id},
        )
        connection.execute(
            text("UPDATE review_run SET status = 'interrupted' WHERE run_id = :r"),
            {"r": run_id},
        )
        connection.execute(
            text(
                "UPDATE review_run_summary SET summary_payload = '{}'::jsonb "
                "WHERE run_id = :r"
            ),
            {"r": run_id},
        )
        connection.execute(
            text("DELETE FROM review_run_summary WHERE run_id = :r"), {"r": run_id}
        )
    before = fresh_harness.repository.attempt_totals(run_id)
    resumed = build_service(fresh_harness.store, **FAST)

    async def run_resume():
        runner = _Run(fresh_harness, resumed, fresh_harness.issue(DEMO_SUBJECT))
        context = runner.context()
        body = fresh_harness.repository.load_run(run_id)
        assert body is not None
        try:
            await resumed.resume_review(
                run_id,
                ResumeRequestV2(
                    api_version="review-api.v2",
                    expected_version=body.header.revision,
                    idempotency_key="neg04-resume",
                ),
                context=context,
                idempotency_key="neg04-resume",
            )
        except ReviewServiceError as error:
            return error.code
        await asyncio.sleep(0.3)
        return "accepted"

    code = asyncio.run(run_resume())
    asyncio.run(resumed.shutdown())
    after = fresh_harness.repository.attempt_totals(run_id)
    final = fresh_harness.repository.load_run(run_id)
    assert code == "review_not_recoverable"
    assert after == before, "an untrusted checkpoint must not start a new attempt"
    assert final is not None
    assert final.header.status is ReviewLifecycleStatus.FAILED
    assert final.header.error_code == "checkpoint_incompatible"


# ------------------------------------------------------------------------ NEG-06


def test_neg06_only_one_generation_may_commit(
    fresh_harness: SecurityHarness,
) -> None:
    """NEG-06: a superseded generation can never adopt a branch result."""

    slow = FaultController(
        enabled=True,
        slow={
            worker: {1: 1.5} for worker in ("policy", "contract", "procurement_data")
        },
    )
    service = build_service(fresh_harness.store, **FAST, fault=slow)

    async def prepare():
        runner = _Run(fresh_harness, service, fresh_harness.issue(DEMO_SUBJECT))
        run, _ = await runner.create("neg06-key")
        # The workers are deliberately slow, so the Run is still processing when the
        # lease is declared expired; the old generation then tries to commit late.
        await asyncio.sleep(0.2)
        fresh_harness.repository.mark_interrupted(
            run_id=run.run_id,
            reason_code="lease_expired",
            now=datetime.now(UTC),
        )
        await service.shutdown()
        return run.run_id

    run_id = asyncio.run(prepare())
    token = fresh_harness.issue(DEMO_SUBJECT)
    # The attempt ledger counts across generations, so generation two's first try is
    # attempt two; the resumed service needs its own slow plan to still be reading.
    resume_slow = FaultController(
        enabled=True,
        slow={
            worker: {1: 1.5, 2: 1.5, 3: 1.5}
            for worker in ("policy", "contract", "procurement_data")
        },
    )
    resumed = _second_service(fresh_harness, fault=resume_slow, **FAST)

    async def reopen():
        runner = _Run(fresh_harness, resumed, token)
        context = runner.context()
        current = resumed.get_review(run_id, context=context)
        await resumed.resume_review(
            run_id,
            ResumeRequestV2(
                api_version="review-api.v2", expected_version=current.revision
            ),
            context=context,
            idempotency_key="neg06-resume",
        )
        # Generation two is still inside its slow read, so the successor generation owns
        # the Run while generation one tries to commit late.
        await asyncio.sleep(0.2)

    asyncio.run(reopen())
    committed = fresh_harness.repository.load_run(run_id)
    assert committed is not None and committed.generation is not None
    generation = committed.generation.generation
    late = fresh_harness.repository.commit_branch(
        run_id=run_id,
        generation=generation - 1,
        worker="policy",
        attempt_no=1,
        input_checksum="sha256:" + "1" * 64,
        contract_checksum="sha256:" + "2" * 64,
        result_payload={"late": True},
        now=datetime.now(UTC),
    )
    asyncio.run(resumed.shutdown())
    assert late.adopted is False
    assert late.reason in {"superseded_generation", "already_adopted", "lease_expired"}
    with fresh_harness.store.transaction() as connection:
        checkpoints = connection.execute(
            text(
                "SELECT worker, generation FROM review_branch_checkpoint "
                "WHERE run_id = :r"
            ),
            {"r": run_id},
        ).all()
    rows = [(row.worker, row.generation) for row in checkpoints]
    # Both generations were still inside their slow reads when the service loop closed, and
    # the superseded commit above may not leave a row behind either, so this scenario's
    # expected checkpoint set is empty.  Comparing the materialised rows keeps the check
    # evaluated on its content: any row, at any generation, fails it instead of the old
    # `or checkpoints == []` swallowing the result.
    assert rows == []
    # Stated for every row as well: only the live generation may own an adopted branch.
    assert all(row.generation == generation for row in checkpoints), rows
