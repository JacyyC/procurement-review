"""FIX-01B V-B-01..V-B-06: record order, identity, closure and compatibility.

Every scenario runs against its own disposable PostgreSQL database on a real isolated
server.  Concurrency is created with the product's own inert fault barriers (parked at a
named commit boundary and released from the test) plus independent engines, so "the other
transaction had already committed when this one looked" is a fact of the run rather than
a probability; where a step really is concurrent, the assertions are about durable facts.

The module also covers what the B batch changed about *reading*: a Run and its Trace
events share one persisted identity, an audit number is the number its committed row
holds, and a legacy row that disagrees is corrected by a versioned read projection
without rewriting history.
"""

from __future__ import annotations

import logging
import random
import secrets
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from procurement_review.api import create_app
from procurement_review.audit_contracts import AuditEventClass
from procurement_review.authorization import AuthorizationService
from procurement_review.contracts import (
    RedactionStatus,
    ReviewLifecycleStatus,
    RunEventType,
)
from procurement_review.fault_hooks import BarrierSpec, FaultController
from procurement_review.field_guard import (
    SENSITIVE_VALUE_PATTERNS,
    canary_values,
)
from procurement_review.local_runtime import ScopedWorkflowRunner
from procurement_review.record_guard import guard_record
from procurement_review.review_repository import (
    AuditDraft,
    ReviewRepository,
    TraceDraft,
    checkpoints_digest,
)
from procurement_review.review_schema import REVIEW_MIGRATIONS, migration_checksum
from procurement_review.review_service import ReviewService
from procurement_review.review_store import ReviewStore
from procurement_review.run_errors import (
    FieldGuardError,
    RecordContractError,
    RecordIncompleteError,
)
from procurement_review.trace import canonical_checksum

from fix01a_support import (
    Fix01aFixture,
    barrier_dir,
    release_barrier,
    settle,
    wait_barrier,
)

API = "/api/v2"
API_VERSION_V2 = "review-api.v2"
BARRIER_WAIT_SECONDS = 90.0
PHONE_SHAPED_RUN = "run-19436117970e4786bf273aa093fa7332"
NATIONAL_ID_SHAPED_TRACE = "trace-3ce9a90712644c308669546852246626"
BOUNDARY_SEED = 20260913


# --------------------------------------------------------------------------- helpers


@contextmanager
def database(name: str) -> Iterator[Fix01aFixture]:
    fixture = Fix01aFixture(name)
    try:
        yield fixture
    finally:
        fixture.close()


def once_fault(directory: Path, *names: str) -> FaultController:
    """A plan that parks the first arrival at each named boundary and lets the rest pass.

    A race scenario needs one transaction to stand inside its critical section while a
    second one really enters the same code path; ``once`` is what makes the second writer
    a writer instead of a second parked observer.
    """

    return FaultController(
        enabled=True,
        directory=directory,
        barriers={
            name: BarrierSpec(
                action="wait",
                timeout_seconds=BARRIER_WAIT_SECONDS,
                once=True,
            )
            for name in names
        },
    )


@dataclass(slots=True)
class ApiNode:
    """One independent service instance (own engine, own app) over the same database.

    The client is entered as a context manager on construction, so the application's own
    event loop (and the execution task one create schedules on it) lives as long as the
    node does — a per-request portal would cancel the scheduling and leave the Run in
    ``processing`` forever.
    """

    store: ReviewStore
    service: ReviewService
    repository: ReviewRepository
    client: TestClient

    def close(self) -> None:
        self.client.__exit__(None, None, None)
        self.store.close()


def api_node(
    fixture: Fix01aFixture,
    tag: str,
    *,
    fault: FaultController | None = None,
    id_factory: Any | None = None,
    clock: Any | None = None,
    resume_limit: int = 2,
) -> ApiNode:
    store = ReviewStore.from_url(fixture.url_for(tag))
    repository = ReviewRepository(store, fault=fault)
    service = ReviewService(
        repository=repository,
        runner=ScopedWorkflowRunner(store=store),
        fault=fault,
        id_factory=id_factory,
        clock=clock,
        resume_limit=resume_limit,
        lease_seconds=1000.0,
        heartbeat_seconds=1000.0,
        scan_interval_seconds=0.2,
    )
    application = create_app(
        store=store,
        authorization=AuthorizationService(store=store),
        repository=repository,
        service=service,
    )
    client = TestClient(application, base_url="http://127.0.0.1")
    client.__enter__()
    return ApiNode(
        store=store,
        service=service,
        repository=repository,
        client=client,
    )


def headers(token: str, key: str | None = None) -> dict[str, str]:
    values = {"Authorization": f"Bearer {token}"}
    if key is not None:
        values["Idempotency-Key"] = key
    return values


def create_body(case_id: str, case_version: str) -> dict[str, str]:
    return {
        "api_version": API_VERSION_V2,
        "case_id": case_id,
        "case_version": case_version,
    }


def create_run(
    node: ApiNode, ident: Any, *, key: str, case_id: str | None = None
) -> str:
    response = node.client.post(
        f"{API}/reviews",
        headers=headers(ident.token, key),
        json=create_body(
            case_id or ident.registration.case_id, ident.registration.case_version
        ),
    )
    assert response.status_code == 202, response.text
    return str(response.json()["run_id"])


def audit_rows(fixture: Fix01aFixture) -> list[tuple[int, int, str]]:
    """Every committed audit row as ``(column sequence, JSON sequence, audit_id)``."""

    with fixture.driver.transaction() as connection:
        rows = connection.execute(
            text(
                "SELECT sequence, event_payload->>'sequence' AS json_sequence, audit_id "
                "FROM review_audit_event ORDER BY sequence"
            )
        ).all()
    return [(int(row.sequence), int(row.json_sequence), row.audit_id) for row in rows]


def allocator_next(fixture: Fix01aFixture, stream: str = "audit") -> int:
    with fixture.driver.transaction() as connection:
        return int(
            connection.execute(
                text(
                    "SELECT next_value FROM review_sequence_allocator WHERE stream = :s"
                ),
                {"s": stream},
            ).scalar_one()
        )


def blocked_on_allocator(fixture: Fix01aFixture) -> list[str]:
    """Backends currently waiting for the allocator row held by another transaction.

    A transaction that waits for a row lock is reported by PostgreSQL as waiting on the
    blocking *transaction* (``wait_event_type='Lock'``, ``wait_event='transactionid'``),
    not as an ungranted relation lock, so the predicate is the waiting statement plus the
    blocking PID the server itself reports.
    """

    with fixture.driver.transaction() as connection:
        rows = connection.execute(
            text(
                """
                SELECT a.pid, a.wait_event, a.blockers
                FROM (
                    SELECT pid, wait_event, pg_blocking_pids(pid) AS blockers, query
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                      AND pid <> pg_backend_pid()
                      AND wait_event_type = 'Lock'
                      AND cardinality(pg_blocking_pids(pid)) > 0
                ) a
                WHERE a.query LIKE '%review_sequence_allocator%'
                ORDER BY a.pid
                """
            )
        ).all()
    return [f"{row.pid}:{row.wait_event}:{sorted(row.blockers)}" for row in rows]


def audit_draft(subject: str, credential_revision: int, **overrides: Any) -> AuditDraft:
    values: dict[str, Any] = {
        "audit_id": f"audit-{secrets.token_hex(16)}",
        "event_class": AuditEventClass.LIFECYCLE,
        "occurred_at": datetime.now(UTC),
        "new_status": ReviewLifecycleStatus.INTERRUPTED,
        "detail_code": "storage_unavailable",
    }
    values.update(overrides)
    return AuditDraft(**values)


def interrupt(repository: ReviewRepository, run_id: str) -> bool:
    return repository.mark_interrupted(
        run_id=run_id,
        reason_code="storage_unavailable",
        now=datetime.now(UTC),
        events=(
            TraceDraft(
                event_type=RunEventType.STEP_FAILED.value,
                step="review_workflow",
                output_summary={"status": "interrupted"},
                source_versions={"execution": "review-execution.v1"},
                error="The local review execution stopped without a committed result.",
            ),
        ),
        audit=audit_draft("subject-local-demo", 1),
    )


def photo_shaped_id_factory(prefix: str) -> str:
    """The two identities fixed by AC-04.5, plus deterministic siblings."""

    if prefix == "run":
        return PHONE_SHAPED_RUN
    if prefix == "trace":
        return NATIONAL_ID_SHAPED_TRACE
    return f"{prefix}-19436117970e4786bf273aa093fa7332"


# ------------------------------------------------------------------ V-B-01 order


def test_audit_sequence_follows_commit_order_across_connections() -> None:
    """V-B-01: the second writer waits for the first commit and takes the next number."""

    with database("v_b_01_order") as fixture:
        directory = barrier_dir("v-b-01-order")
        fault = once_fault(directory, "during_audit_sequence")
        first_node = api_node(fixture, "v1b_order_a", fault=fault)
        second_node = api_node(fixture, "v1b_order_b", fault=fault)
        ident = fixture.identity()
        token = ident.token
        try:
            first_run = create_run(first_node, ident, key="v-b-01-a")
            second_run = create_run(second_node, ident, key="v-b-01-b")
            settle(fixture, first_run, "pending_confirmation")
            settle(fixture, second_run, "pending_confirmation")
            version = int(
                first_node.client.get(
                    f"{API}/reviews/{first_run}", headers=headers(token)
                ).json()["version"]
            )
            answers: dict[str, Any] = {}

            def cancelled(label: str, node: ApiNode, run_id: str) -> None:
                response = node.client.post(
                    f"{API}/reviews/{run_id}/cancel",
                    headers=headers(token, f"v-b-01-{label}"),
                    json={"api_version": API_VERSION_V2, "expected_version": version},
                )
                answers[label] = (response.status_code, response.json())

            parked = threading.Thread(
                target=cancelled, args=("parked", first_node, first_run), daemon=True
            )
            parked.start()
            wait_barrier(directory, "during_audit_sequence", timeout=60.0)
            waiting = threading.Thread(
                target=cancelled, args=("waiting", second_node, second_run), daemon=True
            )
            waiting.start()
            # The waiter is visible in the database while the first transaction is parked:
            # its wait is an observed fact, not an assumption about timing.
            deadline = time.monotonic() + 30.0
            observed: list[str] = []
            while time.monotonic() < deadline and not observed:
                observed = blocked_on_allocator(fixture)
                if not observed:
                    time.sleep(0.05)
            assert observed, (
                f"the second audit writer never waited for the allocator: {answers}"
            )
            release_barrier(directory, "during_audit_sequence")
            for thread in (parked, waiting):
                thread.join(timeout=120)
            assert not parked.is_alive() and not waiting.is_alive()
            assert {label: value[0] for label, value in answers.items()} == {
                "parked": 200,
                "waiting": 200,
            }, answers

            rows = audit_rows(fixture)
            assert len(rows) == 2, rows
            columns = [column for column, _json, _id in rows]
            assert columns == sorted(set(columns)), rows
            assert all(column == json_value for column, json_value, _id in rows), rows
            assert len({audit_id for _column, _json, audit_id in rows}) == 2
            assert allocator_next(fixture) == rows[-1][0] + 1
            # The same numbers come back through the read projection of each Run, and the
            # Run that committed first holds the lower number.
            projected: dict[str, int] = {}
            for label, (node, run_id) in {
                "parked": (first_node, first_run),
                "waiting": (second_node, second_run),
            }.items():
                response = node.client.get(
                    f"{API}/reviews/{run_id}/audit", headers=headers(token)
                )
                assert response.status_code == 200, response.text
                payload = response.json()
                assert payload["sequence_projection_version"] == (
                    "approval-audit.sequence-projection.v1"
                )
                assert payload["legacy_sequence_anomalies"] == []
                numbers = [item["sequence"] for item in payload["events"]]
                assert len(numbers) == 1, payload
                projected[label] = numbers[0]
            assert sorted(projected.values()) == columns
            assert projected["parked"] < projected["waiting"]
        finally:
            first_node.close()
            second_node.close()


def test_audit_numbers_ignore_the_clock_and_are_stable_for_a_new_reader() -> None:
    """V-B-01: the number never comes from the wall clock, and reading it is stable."""

    with database("v_b_01_clock") as fixture:

        class SteppedClock:
            """One clock the scenario moves backwards between the two writes."""

            def __init__(self) -> None:
                self.value = datetime.now(UTC)

            def __call__(self) -> datetime:
                return self.value

        clock = SteppedClock()
        node = api_node(fixture, "v1b_clock", clock=clock)
        ident = fixture.identity()
        token = ident.token
        try:
            run_ids = [
                create_run(node, ident, key=f"v-b-01-clock-{i}") for i in range(2)
            ]
            for index, run_id in enumerate(run_ids):
                settle(fixture, run_id, "pending_confirmation")
                version = int(
                    node.client.get(
                        f"{API}/reviews/{run_id}", headers=headers(token)
                    ).json()["version"]
                )
                if index == 1:
                    # The wall clock moves back five hours between the two events; the
                    # allocated number must still express the committed order.
                    clock.value = clock.value - timedelta(hours=5)
                response = node.client.post(
                    f"{API}/reviews/{run_id}/cancel",
                    headers=headers(token, f"v-b-01-clock-cancel-{run_id}"),
                    json={"api_version": API_VERSION_V2, "expected_version": version},
                )
                assert response.status_code == 200, response.text
            rows = audit_rows(fixture)
            assert [column for column, _json, _id in rows] == [1, 2], rows
            with fixture.driver.transaction() as connection:
                occurred = list(
                    connection.execute(
                        text(
                            "SELECT occurred_at FROM review_audit_event "
                            "ORDER BY sequence"
                        )
                    ).scalars()
                )
            assert occurred[1] < occurred[0], occurred
            reader = ReviewStore.from_url(fixture.url_for("v1b_clock_reader"))
            try:
                repository = ReviewRepository(reader)
                first, anomalies = repository.audit_projection(run_ids[0])
                assert [item.sequence for item in first] == [1]
                assert anomalies == ()
                second, _anomalies = repository.audit_projection(run_ids[1])
                assert [item.sequence for item in second] == [2]
            finally:
                reader.close()
        finally:
            node.close()


def test_every_audit_entry_point_shares_one_allocated_stream() -> None:
    """V-B-01: approval, lifecycle, recovery and failed recovery use one stream."""

    with database("v_b_01_writers") as fixture:
        node = api_node(fixture, "v1b_writers")
        ident = fixture.identity()
        token = ident.token
        repository = node.repository

        approved = create_run(node, ident, key="v-b-01-w-approval")
        settle(fixture, approved, "pending_confirmation")
        version = int(
            node.client.get(f"{API}/reviews/{approved}", headers=headers(token)).json()[
                "version"
            ]
        )
        cancelled = node.client.post(
            f"{API}/reviews/{approved}/cancel",
            headers=headers(token, "v-b-01-w-cancel"),
            json={"api_version": API_VERSION_V2, "expected_version": version},
        )
        assert cancelled.status_code == 200, cancelled.text

        resumed = create_run(node, ident, key="v-b-01-w-interrupt")
        assert interrupt(repository, resumed) is True
        current = node.client.get(
            f"{API}/reviews/{resumed}", headers=headers(token)
        ).json()
        response = node.client.post(
            f"{API}/reviews/{resumed}/resume",
            headers=headers(token, "v-b-01-w-resume"),
            json={
                "api_version": API_VERSION_V2,
                "expected_version": int(current["version"]),
            },
        )
        assert response.status_code == 202, response.text

        # A resume whose budget was already spent registers a failed recovery.
        spent = api_node(fixture, "v1b_writers_spent", resume_limit=0)
        try:
            spent_run = create_run(spent, ident, key="v-b-01-w-spent")
            assert interrupt(spent.repository, spent_run) is True
            current = spent.client.get(
                f"{API}/reviews/{spent_run}", headers=headers(token)
            ).json()
            refused = spent.client.post(
                f"{API}/reviews/{spent_run}/resume",
                headers=headers(token, "v-b-01-w-spent-resume"),
                json={
                    "api_version": API_VERSION_V2,
                    "expected_version": int(current["version"]),
                },
            )
            assert refused.status_code == 409, refused.text
            assert refused.json()["error"]["code"] == "review_not_recoverable"
        finally:
            spent.close()

        failed = create_run(node, ident, key="v-b-01-w-failed")
        assert (
            repository.mark_failed(
                run_id=failed,
                error_code="review_execution_failed",
                error_message="The local review execution failed closed.",
                now=datetime.now(UTC),
                events=(
                    TraceDraft(
                        event_type=RunEventType.RUN_FAILED.value,
                        step="review_run",
                        output_summary={"status": "failed"},
                        source_versions={"execution": "review-execution.v1"},
                        error="The local review execution failed closed.",
                    ),
                ),
            )
            is True
        )

        rows = audit_rows(fixture)
        columns = [column for column, _json, _id in rows]
        assert columns == sorted(set(columns)), rows
        assert all(column == json_value for column, json_value, _id in rows), rows
        assert len(columns) == 5, rows
        audit = node.client.get(
            f"{API}/reviews/{approved}/audit", headers=headers(token)
        )
        assert audit.status_code == 200, audit.text
        assert [item["event_class"] for item in audit.json()["events"]] == ["approval"]
        assert allocator_next(fixture) == columns[-1] + 1
        node.close()


def test_a_refused_state_transaction_leaves_no_audit_and_no_number() -> None:
    """V-B-01: a guard failure after the audit is allocated rolls the whole step back."""

    with database("v_b_01_rollback") as fixture:
        node = api_node(fixture, "v1b_rollback")
        ident = fixture.identity()
        repository = node.repository
        run_id = create_run(node, ident, key="v-b-01-rollback")
        assert interrupt(repository, run_id) is True
        before_rows = audit_rows(fixture)
        before_next = allocator_next(fixture)
        before_ledger = canonical_checksum(fixture.ledger(run_id))
        poisoned = canary_values()[0][1]
        with pytest.raises(FieldGuardError):
            repository.resume_run(
                run_id=run_id,
                actor_subject_id=ident.context.subject_id,
                actor_credential_id=ident.context.credential_id,
                actor_authorization_revision=ident.context.authorization_revision,
                idempotency_key="v-b-01-rollback-resume",
                request_digest=canonical_checksum({"resume": "poisoned"}),
                expected_version=int(fixture.ledger(run_id)["revision"]),
                owner_id="owner-" + secrets.token_hex(16),
                now=datetime.now(UTC),
                lease_seconds=1000.0,
                deadline_seconds=1000.0,
                resume_limit=2,
                incompatibility=None,
                expected_checkpoints_digest=checkpoints_digest(
                    node.repository.load_run(run_id).checkpoints
                ),
                audit=audit_draft(
                    ident.context.subject_id,
                    ident.context.authorization_revision,
                    event_class=AuditEventClass.RECOVERY,
                    new_status=ReviewLifecycleStatus.PROCESSING,
                    detail_code="resume_requested",
                ),
                failure_audit=audit_draft(
                    ident.context.subject_id,
                    ident.context.authorization_revision,
                    event_class=AuditEventClass.RECOVERY,
                    new_status=ReviewLifecycleStatus.FAILED,
                ),
                events=(
                    TraceDraft(
                        event_type=RunEventType.RUN_STARTED.value,
                        step="review_recovery",
                        output_summary={"status": "processing", "reason": poisoned},
                        source_versions={"recovery": "review-recovery.v1"},
                    ),
                ),
            )
        # The refused transaction wrote nothing at all, and the number it had taken is
        # not visible as a hole in the committed order.
        assert audit_rows(fixture) == before_rows
        assert canonical_checksum(fixture.ledger(run_id)) == before_ledger
        assert allocator_next(fixture) == before_next
        # The next committed event takes the number the refused transaction had used.
        response = node.client.post(
            f"{API}/reviews/{run_id}/resume",
            headers=headers(ident.token, "v-b-01-rollback-resume-ok"),
            json={
                "api_version": API_VERSION_V2,
                "expected_version": int(fixture.ledger(run_id)["revision"]),
            },
        )
        assert response.status_code == 202, response.text
        after_rows = audit_rows(fixture)
        assert [column for column, _json, _id in after_rows] == [
            column for column, _json, _id in before_rows
        ] + [before_next]
        node.close()


def test_trace_ordinals_stay_unique_under_concurrent_appenders() -> None:
    """V-B-01: one Run's Trace sequence is unique and ordered under real contention."""

    with database("v_b_01_trace") as fixture:
        node = api_node(fixture, "v1b_trace")
        ident = fixture.identity()
        run_id = create_run(node, ident, key="v-b-01-trace")
        settle(fixture, run_id, "pending_confirmation")
        before = _trace_count(fixture, run_id)
        repository = node.repository
        start = threading.Barrier(9)
        answers: list[str] = []
        lock = threading.Lock()

        def appender(index: int) -> None:
            start.wait(timeout=30)
            try:
                repository.append_trace(
                    run_id=run_id,
                    drafts=tuple(
                        TraceDraft(
                            event_type=RunEventType.STEP_COMPLETED.value,
                            step=f"contention.{index}.{item}",
                            output_summary={"status": "ok"},
                            source_versions={"api": API_VERSION_V2},
                        )
                        for item in range(3)
                    ),
                    now=datetime.now(UTC),
                )
                value = "ok"
            except Exception as error:  # noqa: BLE001 - the answer is asserted
                value = f"{type(error).__name__}: {error}"
            with lock:
                answers.append(value)

        threads = [
            threading.Thread(target=appender, args=(index,), daemon=True)
            for index in range(8)
        ]
        for thread in threads:
            thread.start()
        start.wait(timeout=30)
        for thread in threads:
            thread.join(timeout=120)
        assert answers == ["ok"] * 8, answers
        with fixture.driver.transaction() as connection:
            ordinals = [
                int(item)
                for item in connection.execute(
                    text(
                        "SELECT event_ordinal FROM review_trace_event "
                        "WHERE run_id = :r ORDER BY event_ordinal"
                    ),
                    {"r": run_id},
                ).scalars()
            ]
            identities = [
                str(item)
                for item in connection.execute(
                    text("SELECT event_id FROM review_trace_event WHERE run_id = :r"),
                    {"r": run_id},
                ).scalars()
            ]
        assert ordinals == list(range(1, len(ordinals) + 1)), ordinals
        assert len(ordinals) == before + 8 * 3
        assert len(set(identities)) == len(identities)
        node.close()


def _trace_count(fixture: Fix01aFixture, run_id: str) -> int:
    with fixture.driver.transaction() as connection:
        return int(
            connection.execute(
                text("SELECT count(*) FROM review_trace_event WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        )


# ------------------------------------------------------------------ V-B-02 races


def test_concurrent_same_key_create_returns_created_and_replay() -> None:
    """V-B-02: two instances that both read "no mapping" stay inside the contract."""

    with database("v_b_02_create") as fixture:
        directory = barrier_dir("v-b-02-create")
        fault = once_fault(directory, "during_create_claim")
        first_node = api_node(fixture, "v1b_create_a", fault=fault)
        second_node = api_node(fixture, "v1b_create_b", fault=fault)
        ident = fixture.identity()
        body = create_body(ident.registration.case_id, ident.registration.case_version)
        answers: dict[str, Any] = {}
        start = threading.Barrier(3)

        def contender(label: str) -> None:
            node = first_node if label == "a" else second_node
            start.wait(timeout=30)
            response = node.client.post(
                f"{API}/reviews",
                headers=headers(ident.token, "v-b-02-shared"),
                json=body,
            )
            answers[label] = (response.status_code, response.json())

        try:
            threads = [
                threading.Thread(target=contender, args=(label,), daemon=True)
                for label in ("a", "b")
            ]
            for thread in threads:
                thread.start()
            start.wait(timeout=30)
            # One contender stands inside the critical section (past its own "no mapping"
            # read); the other really enters the same claim path.  Releasing lets the race
            # resolve through the database instead of through luck.
            wait_barrier(directory, "during_create_claim", timeout=60.0)
            release_barrier(directory, "during_create_claim")
            for thread in threads:
                thread.join(timeout=120)
            assert not any(thread.is_alive() for thread in threads)
            assert sorted(value[0] for value in answers.values()) == [200, 202], answers
            run_ids = {value[1]["run_id"] for value in answers.values()}
            assert len(run_ids) == 1, answers
            with fixture.driver.transaction() as connection:
                runs = list(
                    connection.execute(text("SELECT run_id FROM review_run")).scalars()
                )
                mappings = connection.execute(
                    text(
                        "SELECT operation_family, run_id, outcome "
                        "FROM review_idempotency ORDER BY operation_family"
                    )
                ).all()
            assert runs == list(run_ids)
            assert [
                (row.operation_family, row.run_id, row.outcome) for row in mappings
            ] == [("create", runs[0], "applied")]
        finally:
            first_node.close()
            second_node.close()


def test_concurrent_same_key_create_with_another_body_conflicts() -> None:
    """V-B-02: the same subject reusing the key for another Case is a contract conflict."""

    with database("v_b_02_conflict") as fixture:
        directory = barrier_dir("v-b-02-conflict")
        fault = once_fault(directory, "during_create_claim")
        first_node = api_node(fixture, "v1b_conflict_a", fault=fault)
        second_node = api_node(fixture, "v1b_conflict_b", fault=fault)
        low = fixture.identity()
        high = fixture.identity(case_id="CASE-HIGH-001")
        assert low.context.subject_id == high.context.subject_id
        answers: dict[str, Any] = {}
        start = threading.Barrier(3)

        def contender(label: str) -> None:
            node = first_node if label == "a" else second_node
            identity = low if label == "a" else high
            start.wait(timeout=30)
            response = node.client.post(
                f"{API}/reviews",
                headers=headers(identity.token, "v-b-02-conflict"),
                json=create_body(
                    identity.registration.case_id, identity.registration.case_version
                ),
            )
            answers[label] = (response.status_code, response.json())

        try:
            threads = [
                threading.Thread(target=contender, args=(label,), daemon=True)
                for label in ("a", "b")
            ]
            for thread in threads:
                thread.start()
            start.wait(timeout=30)
            wait_barrier(directory, "during_create_claim", timeout=60.0)
            release_barrier(directory, "during_create_claim")
            for thread in threads:
                thread.join(timeout=120)
            assert not any(thread.is_alive() for thread in threads)
            codes = sorted(value[0] for value in answers.values())
            assert codes == [202, 409], answers
            conflict = next(value for value in answers.values() if value[0] == 409)
            assert conflict[1]["error"]["code"] == "idempotency_conflict"
            with fixture.driver.transaction() as connection:
                assert (
                    int(
                        connection.execute(
                            text("SELECT count(*) FROM review_run")
                        ).scalar_one()
                    )
                    == 1
                )
        finally:
            first_node.close()
            second_node.close()


def test_the_same_key_in_another_subject_never_reads_a_foreign_operation() -> None:
    """V-B-02: the idempotency namespace is the subject's own, not the tenant's."""

    with database("v_b_02_subject") as fixture:
        directory = barrier_dir("v-b-02-subject")
        fault = once_fault(directory, "during_create_claim")
        first_node = api_node(fixture, "v1b_subject_a", fault=fault)
        second_node = api_node(fixture, "v1b_subject_b", fault=fault)
        operator = fixture.identity()
        other = fixture.identity(
            subject_id="subject-alpha-operator",
            case_id="CASE-ALPHA-001",
            tenant_id="tenant-alpha",
        )
        assert other.context.subject_id != operator.context.subject_id
        assert other.context.tenant_id != operator.context.tenant_id
        answers: dict[str, Any] = {}
        start = threading.Barrier(3)

        def contender(label: str) -> None:
            node = first_node if label == "a" else second_node
            identity = operator if label == "a" else other
            start.wait(timeout=30)
            response = node.client.post(
                f"{API}/reviews",
                headers=headers(identity.token, "v-b-02-subject-shared"),
                json=create_body(
                    identity.registration.case_id, identity.registration.case_version
                ),
            )
            answers[label] = (response.status_code, response.json())

        try:
            threads = [
                threading.Thread(target=contender, args=(label,), daemon=True)
                for label in ("a", "b")
            ]
            for thread in threads:
                thread.start()
            start.wait(timeout=30)
            wait_barrier(directory, "during_create_claim", timeout=60.0)
            release_barrier(directory, "during_create_claim")
            for thread in threads:
                thread.join(timeout=120)
            assert not any(thread.is_alive() for thread in threads)
            assert sorted(value[0] for value in answers.values()) == [202, 202], answers
            assert len({value[1]["run_id"] for value in answers.values()}) == 2
            with fixture.driver.transaction() as connection:
                rows = connection.execute(
                    text("SELECT subject_id, key_digest FROM review_idempotency")
                ).all()
            assert len(rows) == 2
            assert len({(row.subject_id, row.key_digest) for row in rows}) == 2
        finally:
            first_node.close()
            second_node.close()


def test_concurrent_same_key_resume_replays_instead_of_conflicting() -> None:
    """V-B-02/AF-11: the duplicate resume is answered by the committed operation."""

    with database("v_b_02_resume") as fixture:
        directory = barrier_dir("v-b-02-resume")
        fault = once_fault(directory, "during_resume_commit:before_run_lock")
        original_node = api_node(fixture, "v1b_resume_a")
        duplicate_node = api_node(fixture, "v1b_resume_b", fault=fault)
        ident = fixture.identity()
        token = ident.token
        try:
            run_id = create_run(original_node, ident, key="v-b-02-resume-create")
            settle(fixture, run_id, "pending_confirmation")
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_run SET status = 'interrupted', "
                        "revision = revision + 1, error_code = 'storage_unavailable' "
                        "WHERE run_id = :r"
                    ),
                    {"r": run_id},
                )
                connection.execute(
                    text(
                        "UPDATE review_run_generation SET status = 'expired' "
                        "WHERE run_id = :r AND status = 'active'"
                    ),
                    {"r": run_id},
                )
                version = int(
                    connection.execute(
                        text("SELECT revision FROM review_run WHERE run_id = :r"),
                        {"r": run_id},
                    ).scalar_one()
                )
            body = {"api_version": API_VERSION_V2, "expected_version": version}
            answers: dict[str, Any] = {}

            def resume(label: str, node: ApiNode) -> None:
                response = node.client.post(
                    f"{API}/reviews/{run_id}/resume",
                    headers=headers(token, "v-b-02-resume-shared"),
                    json=body,
                )
                answers[label] = (response.status_code, response.json())

            # The duplicate reaches its idempotency read before the Run lock exists and
            # parks there; the original then runs the whole operation to completion.
            duplicate = threading.Thread(
                target=resume, args=("duplicate", duplicate_node), daemon=True
            )
            duplicate.start()
            wait_barrier(
                directory, "during_resume_commit:before_run_lock", timeout=60.0
            )
            original = threading.Thread(
                target=resume, args=("original", original_node), daemon=True
            )
            original.start()
            original.join(timeout=120)
            release_barrier(directory, "during_resume_commit:before_run_lock")
            duplicate.join(timeout=120)
            assert not duplicate.is_alive() and not original.is_alive()
            assert {label: value[0] for label, value in answers.items()} == {
                "original": 202,
                "duplicate": 200,
            }, answers
            # Both answers are the same Run; the replay reports the committed projection
            # as it stands when it is built, so its revision only moves forward.
            assert answers["duplicate"][1]["run_id"] == answers["original"][1]["run_id"]
            assert (
                answers["duplicate"][1]["version"] >= answers["original"][1]["version"]
            )
            with fixture.driver.transaction() as connection:
                resume_count = int(
                    connection.execute(
                        text("SELECT resume_count FROM review_run WHERE run_id = :r"),
                        {"r": run_id},
                    ).scalar_one()
                )
                generations = int(
                    connection.execute(
                        text(
                            "SELECT count(*) FROM review_run_generation "
                            "WHERE run_id = :r"
                        ),
                        {"r": run_id},
                    ).scalar_one()
                )
                recovery = int(
                    connection.execute(
                        text(
                            "SELECT count(*) FROM review_audit_event WHERE run_id = :r "
                            "AND event_class = 'recovery'"
                        ),
                        {"r": run_id},
                    ).scalar_one()
                )
                mappings = list(
                    connection.execute(
                        text(
                            "SELECT outcome FROM review_idempotency WHERE run_id = :r "
                            "AND operation_family = 'resume'"
                        ),
                        {"r": run_id},
                    ).scalars()
                )
            assert (resume_count, generations, recovery) == (1, 2, 1)
            assert mappings == ["applied"]

            # Same key, different body: a contract conflict and still no new write.
            other = original_node.client.post(
                f"{API}/reviews/{run_id}/resume",
                headers=headers(token, "v-b-02-resume-shared"),
                json={"api_version": API_VERSION_V2, "expected_version": version - 1},
            )
            assert other.status_code == 409, other.text
            assert other.json()["error"]["code"] == "idempotency_conflict"
            assert int(fixture.ledger(run_id)["resume_count"]) == 1
        finally:
            original_node.close()
            duplicate_node.close()


# ------------------------------------------------------------------ V-B-03 identity


def test_run_trace_identity_is_the_persisted_one_everywhere() -> None:
    """V-B-03: Run detail/list, envelope and every event share one persisted identity."""

    with database("v_b_03_identity") as fixture:
        node = api_node(fixture, "v1b_identity")
        ident = fixture.identity()
        token = ident.token
        run_id = create_run(node, ident, key="v-b-03")
        settle(fixture, run_id, "pending_confirmation")
        detail = node.client.get(
            f"{API}/reviews/{run_id}", headers=headers(token)
        ).json()
        listing = node.client.get(f"{API}/reviews", headers=headers(token)).json()
        trace = node.client.get(
            f"{API}/reviews/{run_id}/trace", headers=headers(token)
        ).json()["trace"]
        identity = detail["trace_id"]
        assert identity == trace["trace_id"]
        assert listing["runs"][0]["trace_id"] == identity
        assert {event["trace_id"] for event in trace["events"]} == {identity}
        assert identity != f"trace-{run_id}"
        with fixture.driver.transaction() as connection:
            stored = connection.execute(
                text("SELECT trace_id FROM review_run WHERE run_id = :r"),
                {"r": run_id},
            ).scalar_one()
        assert stored == identity
        # Reading again — also from a brand new engine — never produces a new identity.
        reader = ReviewStore.from_url(fixture.url_for("v1b_identity_reader"))
        try:
            again = ReviewRepository(reader).trace_envelope(run_id, complete=False)
            assert again.trace_id == identity
            assert {event.trace_id for event in again.events} == {identity}
        finally:
            reader.close()
        node.close()


def test_legacy_run_without_a_recorded_identity_reads_its_event_identity() -> None:
    """V-B-03: a row committed before the column existed keeps the identity it wrote."""

    with database("v_b_03_legacy") as fixture:
        node = api_node(fixture, "v1b_legacy")
        ident = fixture.identity()
        token = ident.token
        run_id = create_run(node, ident, key="v-b-03-legacy")
        settle(fixture, run_id, "pending_confirmation")
        with fixture.driver.transaction() as connection:
            recorded = connection.execute(
                text(
                    "SELECT trace_id FROM review_trace_event WHERE run_id = :r "
                    "ORDER BY sequence LIMIT 1"
                ),
                {"r": run_id},
            ).scalar_one()
            connection.execute(
                text("UPDATE review_run SET trace_id = NULL WHERE run_id = :r"),
                {"r": run_id},
            )
        detail = node.client.get(
            f"{API}/reviews/{run_id}", headers=headers(token)
        ).json()
        trace = node.client.get(
            f"{API}/reviews/{run_id}/trace", headers=headers(token)
        ).json()["trace"]
        listing = node.client.get(f"{API}/reviews", headers=headers(token)).json()
        assert detail["trace_id"] == recorded == trace["trace_id"]
        assert listing["runs"][0]["trace_id"] == recorded
        node.close()


def test_run_without_any_trace_identity_fails_closed() -> None:
    """V-B-03: an incomplete record is reported, never given a fabricated identity."""

    with database("v_b_03_incomplete") as fixture:
        node = api_node(fixture, "v1b_incomplete")
        ident = fixture.identity()
        run_id = f"run-{secrets.token_hex(16)}"
        with fixture.driver.transaction() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO review_run (
                        run_id, tenant_id, case_id, case_version, status, revision,
                        resume_count, created_by_subject, current_subject_id,
                        request_checksum, document_version, dataset_version,
                        index_version, rules_version, isolation, created_at,
                        updated_at
                    ) VALUES (
                        :run_id, :tenant, :case_id, 'case-v1', 'processing', 1, 0,
                        :subject, :subject, :checksum, 'doc-v1', 'data-v1',
                        'index-v1', '2.0.0', CAST(:isolation AS jsonb), :now, :now
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "tenant": ident.registration.tenant_id,
                    "case_id": ident.registration.case_id,
                    "subject": ident.context.subject_id,
                    "checksum": ident.registration.checksum,
                    "isolation": (
                        '{"case_type": "purchase_order", "purchase_order_id": "PO-001"}'
                    ),
                    "now": datetime.now(UTC),
                },
            )
        with pytest.raises(RecordIncompleteError):
            node.service.get_review(run_id, context=ident.context)
        with pytest.raises(RecordIncompleteError):
            node.repository.trace_envelope(run_id, complete=False)
        with fixture.driver.transaction() as connection:
            assert (
                connection.execute(
                    text("SELECT trace_id FROM review_run WHERE run_id = :r"),
                    {"r": run_id},
                ).scalar_one()
                is None
            )
        node.close()


def test_no_event_run_keeps_the_identity_recorded_with_it() -> None:
    """V-B-03: the no-event boundary reads the Run record, not the Run ID."""

    with database("v_b_03_no_events") as fixture:
        node = api_node(fixture, "v1b_no_events")
        ident = fixture.identity()
        run_id = f"run-{secrets.token_hex(16)}"
        trace_id = f"trace-{secrets.token_hex(16)}"
        outcome = node.repository.create_run(
            run_id=run_id,
            trace_id=trace_id,
            tenant_id=ident.registration.tenant_id,
            case_id=ident.registration.case_id,
            case_version=ident.registration.case_version,
            subject_id=ident.context.subject_id,
            credential_id=ident.context.credential_id,
            authorization_revision=ident.context.authorization_revision,
            request_checksum=ident.registration.checksum,
            document_version=ident.registration.document_version,
            dataset_version=ident.registration.dataset_version,
            index_version=ident.registration.index_version,
            rules_version=ident.registration.rules_version,
            isolation={
                "case_type": ident.registration.case_type,
                "scenario": ident.registration.scenario,
                "purchase_order_id": ident.registration.purchase_order_id,
                "case_document_ids": list(ident.registration.case_document_ids),
                "shared_document_ids": list(ident.registration.shared_document_ids),
                "allowed_order_ids": list(ident.registration.allowed_order_ids),
                "allowed_supplier_ids": list(ident.registration.allowed_supplier_ids),
                "allowed_transaction_ids": list(
                    ident.registration.allowed_transaction_ids
                ),
            },
            idempotency_key="v-b-03-no-events",
            request_digest=canonical_checksum({"no_events": True}),
            owner_id="owner-" + secrets.token_hex(16),
            now=datetime.now(UTC),
            lease_seconds=1000.0,
            deadline_seconds=1000.0,
            events=(),
        )
        assert outcome.created is True
        assert outcome.run.persisted_trace_id == trace_id
        assert outcome.run.trace_event_count == 0
        # The Run keeps the identity it was created with, while the envelope — which
        # covers at least one committed event — is refused instead of being fabricated.
        with pytest.raises(RecordIncompleteError):
            node.repository.trace_envelope(run_id, complete=False)
        reloaded = node.repository.load_run(run_id)
        assert reloaded is not None and reloaded.persisted_trace_id == trace_id
        node.close()


# ------------------------------------------------------------------ V-B-04 closure


def _valid_samples(run_id: str, trace_id: str) -> dict[str, dict[str, Any]]:
    """One valid, closed record per kind, as the write paths build them."""

    now = datetime.now(UTC)
    checksum = "sha256:" + "a" * 64
    return {
        "run": {
            "run_id": run_id,
            "trace_id": trace_id,
            "tenant_id": "tenant-local-demo",
            "case_id": "CASE-LOW-001",
            "case_version": "case-v1",
            "status": ReviewLifecycleStatus.PROCESSING,
            "revision": 1,
            "resume_count": 0,
            "created_by_subject": "subject-local-demo",
            "current_subject_id": "subject-local-demo",
            "request_checksum": checksum,
            "document_version": "doc-v1",
            "dataset_version": "data-v1",
            "index_version": "index-v1",
            "rules_version": "2.0.0",
            "created_at": now,
            "updated_at": now,
        },
        "generation": {
            "run_id": run_id,
            "generation": 1,
            "subject_id": "subject-local-demo",
            "credential_id": "cred-" + "b" * 32,
            "authorization_revision": 1,
            "owner_id": "owner-" + "c" * 32,
            "status": "active",
            "heartbeat_at": now,
            "lease_expires_at": now,
            "deadline_at": now,
            "created_at": now,
        },
        "attempt": {
            "run_id": run_id,
            "generation": 1,
            "worker": "policy",
            "attempt_no": 1,
            "started_at": now,
        },
        "idempotency": {
            "tenant_id": "tenant-local-demo",
            "subject_id": "subject-local-demo",
            "operation_family": "create",
            "key_digest": checksum,
            "request_digest": checksum,
            "run_id": run_id,
            "created_at": now,
        },
        "receipt": {
            "receipt_id": "receipt-" + "d" * 32,
            "run_id": run_id,
            "case_id": "CASE-LOW-001",
            "human_actor": "subject-local-demo",
            "decision": "confirm_recommendation",
            "reviewed_recommendation": "recommend_approve",
            "recorded_at": now,
            "idempotency_digest": checksum,
        },
        "trace": {
            "event_id": f"event-{run_id}-0001",
            "run_id": run_id,
            "trace_id": trace_id,
            "event_type": RunEventType.STEP_COMPLETED,
            "step": "validate_input",
            "occurred_at": now,
            "input_summary": {"case_id": "CASE-LOW-001"},
            "output_summary": {"status": "valid"},
            "input_checksum": checksum,
            "output_checksum": checksum,
            "source_versions": {"api": API_VERSION_V2},
            "duration_ms": 0,
            "redaction_status": RedactionStatus.NOT_REQUIRED,
            "error": None,
        },
        "audit": {
            "audit_id": "audit-" + "e" * 32,
            "sequence": 1,
            "tenant_id": "tenant-local-demo",
            "case_id": "CASE-LOW-001",
            "run_id": run_id,
            "event_class": AuditEventClass.LIFECYCLE,
            "new_status": ReviewLifecycleStatus.INTERRUPTED,
            "occurred_at": now,
            "detail_code": "storage_unavailable",
        },
    }


@pytest.mark.parametrize(
    "kind",
    ["run", "generation", "attempt", "idempotency", "receipt", "trace", "audit"],
)
def test_every_record_type_rejects_unregistered_keys(kind: str) -> None:
    """V-B-04: an unknown key at the top level or nested is refused, not dropped."""

    run_id = "run-" + "1" * 32
    trace_id = "trace-" + "2" * 32
    sample = _valid_samples(run_id, trace_id)[kind]
    guard_record(kind, dict(sample))
    with pytest.raises(RecordContractError):
        guard_record(kind, {**sample, "customer_notes": {"freeform": "ordinary text"}})
    nested = dict(sample)
    if kind == "trace":
        nested["output_summary"] = {"status": "valid", "notes": {"freeform": "x"}}
    elif kind == "run":
        nested["status"] = {"value": "processing"}
    else:
        nested["extra"] = {"freeform": "x"}
    with pytest.raises(RecordContractError):
        guard_record(kind, nested)


def test_registered_names_are_not_legal_in_the_wrong_place() -> None:
    """V-B-04: a known key with the wrong shape is refused, not silently accepted."""

    run_id = "run-" + "1" * 32
    trace_id = "trace-" + "2" * 32
    sample = _valid_samples(run_id, trace_id)["trace"]
    nested_mapping = dict(sample)
    nested_mapping["output_summary"] = {"status": {"value": "valid"}}
    with pytest.raises(RecordContractError):
        guard_record("trace", nested_mapping)
    unregistered = dict(sample)
    unregistered["input_summary"] = {"case_id": "CASE-LOW-001", "case_notes": "text"}
    with pytest.raises(RecordContractError):
        guard_record("trace", unregistered)
    unregistered_stream = dict(sample)
    unregistered_stream["source_versions"] = {"api": API_VERSION_V2, "notes": "text"}
    with pytest.raises(RecordContractError):
        guard_record("trace", unregistered_stream)


@pytest.mark.parametrize("canary_kind,canary", canary_values())
def test_canaries_are_refused_in_every_free_text_slot(
    canary_kind: str, canary: str, caplog: pytest.LogCaptureFixture
) -> None:
    """V-B-04: every synthetic canary is refused, and no log line echoes it."""

    del canary_kind
    run_id = "run-" + "1" * 32
    trace_id = "trace-" + "2" * 32
    with caplog.at_level(logging.INFO):
        for kind, slot in (
            ("trace", "error"),
            ("trace", "output_summary"),
            ("run", "error_message"),
            ("audit", "detail_code"),
            ("receipt", "human_actor"),
        ):
            record = _valid_samples(run_id, trace_id)[kind]
            if slot == "output_summary":
                record["output_summary"] = {"status": canary}
            elif slot == "error":
                record["error"] = canary
                record["event_type"] = RunEventType.STEP_FAILED
            else:
                record[slot] = canary
            with pytest.raises(RecordContractError):
                guard_record(kind, record)
    assert canary not in caplog.text


def test_typed_slots_accept_registered_identities_and_digests() -> None:
    """V-B-04/X-8: a real digest or minted identity is trusted by type and format."""

    run_id = PHONE_SHAPED_RUN
    trace_id = NATIONAL_ID_SHAPED_TRACE
    sample = _valid_samples(run_id, trace_id)["trace"]
    assert guard_record("trace", sample) is not None
    digest = "sha256:" + "9" * 64
    sample["input_checksum"] = digest
    sample["output_checksum"] = digest
    assert guard_record("trace", sample) is not None
    # A phone number pasted into an identifier slot is refused, and so is a value that
    # only pretends to be a minted identity.
    for value in (
        "13800001234",
        "run-13800001234",
        "run-110101199003070000",
        "run-13800001234abcdef",
    ):
        poisoned = _valid_samples(run_id, trace_id)["run"]
        poisoned["run_id"] = value
        with pytest.raises(RecordContractError):
            guard_record("run", poisoned)


def test_persisted_summary_records_reject_unknown_nested_keys() -> None:
    """V-B-04: the stored summary, execution and checkpoint are closed at every depth."""

    with database("v_b_04_nested") as fixture:
        node = api_node(fixture, "v1b_nested")
        ident = fixture.identity()
        run_id = create_run(node, ident, key="v-b-04-nested")
        settle(fixture, run_id, "pending_confirmation")
        with fixture.driver.transaction() as connection:
            summary = connection.execute(
                text(
                    "SELECT summary_payload FROM review_run_summary WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
            execution = connection.execute(
                text(
                    "SELECT execution_payload FROM review_run_summary WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
            checkpoints = [
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT result_payload FROM review_branch_checkpoint "
                        "WHERE run_id = :r"
                    ),
                    {"r": run_id},
                ).all()
            ]
        # The stored records pass their own closed contract as they are.
        guard_record("summary", summary, strict=False)
        guard_record("execution", execution, strict=False)
        for payload in checkpoints:
            guard_record("checkpoint_payload", payload, strict=False)
        # An unknown key anywhere in the tree is refused.
        for poisoned in (
            {**summary, "customer_notes": "x"},
            {**summary, "findings": [{**summary["findings"][0], "notes": "x"}]},
        ):
            with pytest.raises(RecordContractError):
                guard_record("summary", poisoned, strict=False)
        with pytest.raises(RecordContractError):
            guard_record("execution", {**execution, "extra": 1}, strict=False)
        checkpoint = checkpoints[0]
        poisoned_checkpoint = {
            **checkpoint,
            "result": {**checkpoint["result"], "notes": ["x"]},
        }
        with pytest.raises(RecordContractError):
            guard_record("checkpoint_payload", poisoned_checkpoint, strict=False)
        business_fact = summary["findings"][0]["business_fact_refs"]
        if business_fact:
            poisoned_fact = {
                **summary,
                "findings": [
                    {
                        **summary["findings"][0],
                        "business_fact_refs": [
                            {**business_fact[0], "result": {"new_field": "x"}}
                        ],
                    }
                ],
            }
            with pytest.raises(RecordContractError):
                guard_record("summary", poisoned_fact, strict=False)
        node.close()


def test_a_clean_run_persists_and_reads_back() -> None:
    """V-B-04: the closed guards do not block a normal bounded business record."""

    with database("v_b_04_clean") as fixture:
        node = api_node(fixture, "v1b_clean")
        ident = fixture.identity()
        token = ident.token
        run_id = create_run(node, ident, key="v-b-04-clean")
        settle(fixture, run_id, "pending_confirmation")
        detail = node.client.get(
            f"{API}/reviews/{run_id}", headers=headers(token)
        ).json()
        assert detail["summary"]["findings"], detail
        assert detail["summary_checksum"]
        assert detail["trace_event_count"] >= 5
        node.close()


# ------------------------------------------------------------------ V-B-05 X-8


def test_fixed_identities_run_the_whole_path() -> None:
    """V-B-05: both AC-04.5 identities survive create, commit, nesting and export."""

    with database("v_b_05_fixed") as fixture:
        node = api_node(fixture, "v1b_fixed", id_factory=photo_shaped_id_factory)
        ident = fixture.identity()
        token = ident.token
        response = node.client.post(
            f"{API}/reviews",
            headers=headers(token, "v-b-05-fixed"),
            json=create_body(
                ident.registration.case_id, ident.registration.case_version
            ),
        )
        assert response.status_code == 202, response.text
        payload = response.json()
        assert payload["run_id"] == PHONE_SHAPED_RUN
        assert payload["trace_id"] == NATIONAL_ID_SHAPED_TRACE
        run_id = payload["run_id"]
        settle(fixture, run_id, "pending_confirmation")
        detail = node.client.get(
            f"{API}/reviews/{run_id}", headers=headers(token)
        ).json()
        assert detail["status"] == "pending_confirmation"
        assert detail["summary"]["run_id"] == run_id
        assert detail["summary_checksum"] is not None
        assert detail["trace_event_count"] >= 5
        audit = node.client.get(
            f"{API}/reviews/{run_id}/audit", headers=headers(token)
        ).json()
        assert audit["legacy_sequence_anomalies"] == []
        trace = node.client.get(
            f"{API}/reviews/{run_id}/trace", headers=headers(token)
        ).json()["trace"]
        assert trace["trace_id"] == detail["trace_id"]
        with fixture.driver.transaction() as connection:
            stored = connection.execute(
                text(
                    "SELECT summary_payload->>'run_id' FROM review_run_summary "
                    "WHERE run_id = :r"
                ),
                {"r": run_id},
            ).scalar_one()
        assert stored == run_id
        node.close()


def test_fixed_seed_boundary_samples_are_judged_by_type() -> None:
    """V-B-05: a fixed seed (20260913) produces the boundary shapes the type decides."""

    def has_sensitive_shape(value: str) -> bool:
        return any(
            pattern.search(value) for _category, pattern in SENSITIVE_VALUE_PATTERNS
        )

    def trace_with_error(value: str) -> dict[str, Any]:
        record = _valid_samples(value, NATIONAL_ID_SHAPED_TRACE)["trace"]
        record["error"] = f"branch failed: {value}"
        record["event_type"] = RunEventType.STEP_FAILED
        return record

    rng = random.Random(BOUNDARY_SEED)
    clean = 0
    shaped = 0
    for _ in range(400):
        hexed = "".join(rng.choice("0123456789abcdef") for _ in range(32))
        minted = f"run-{hexed}"
        # A minted identity is trusted by type and format wherever it appears…
        guard_record("run", _valid_samples(minted, NATIONAL_ID_SHAPED_TRACE)["run"])
        if has_sensitive_shape(minted):
            shaped += 1
            # …while the same string in a free-text slot is not an identity: the digit
            # run inside it is a real sensitive shape there.
            with pytest.raises(RecordContractError):
                guard_record("trace", trace_with_error(minted))
        else:
            clean += 1
            guard_record("trace", trace_with_error(minted))
    assert clean + shaped == 400, (clean, shaped)
    assert clean > shaped, (clean, shaped)

    # The same boundary is produced on purpose, not only by sampling: an 11-digit mobile
    # shape and an 18-digit national-ID shape inside a minted 32-hex identity.
    injected = (
        f"run-{'a'}{'b'}{'13800001234'}{'c' * 19}",
        f"run-{'a'}{'b'}{'110101199003070000'}{'c' * 12}",
    )
    for minted in injected:
        value = minted.split("-", 1)[1]
        assert len(value) == 32 and has_sensitive_shape(minted)
        guard_record("run", _valid_samples(minted, NATIONAL_ID_SHAPED_TRACE)["run"])
        with pytest.raises(RecordContractError):
            guard_record("trace", trace_with_error(minted))

    # A value that only pretends to be one of these identities is still refused.
    for disguised in (
        "13800001234",
        "run-13800001234",
        "run-110101199003070000",
        "run-13800001234abcdef",
    ):
        poisoned = _valid_samples(PHONE_SHAPED_RUN, NATIONAL_ID_SHAPED_TRACE)["run"]
        poisoned["run_id"] = disguised
        with pytest.raises(RecordContractError):
            guard_record("run", poisoned)
    # The documented boundary: a 32-hex value is an identity by registered format, even
    # when its digits happen to contain a phone-shaped run.
    boundary = f"run-{'13800001234'}{'0' * 21}"
    assert len(boundary.split("-", 1)[1]) == 32
    poisoned = _valid_samples(PHONE_SHAPED_RUN, NATIONAL_ID_SHAPED_TRACE)["run"]
    poisoned["run_id"] = boundary
    assert guard_record("run", poisoned) is not None


# ------------------------------------------------------------------ V-B-06 compat


def _apply_migrations_upto(store: ReviewStore, version: int) -> None:
    """Bring a fresh database to one historical schema version.

    The disposable scenario database is created migrated, so the historical shape is
    rebuilt from scratch here: the schema is dropped and only the migrations up to the
    requested version are applied, exactly as an older build would have left it.
    """

    with store.transaction() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    with store.transaction() as connection:
        for migration in REVIEW_MIGRATIONS:
            if migration.version > version:
                break
            for statement in migration.statements:
                connection.execute(text(statement))
            connection.execute(
                text(
                    "INSERT INTO review_schema_migration (version, name, checksum) "
                    "VALUES (:version, :name, :checksum)"
                ),
                {
                    "version": migration.version,
                    "name": migration.name,
                    "checksum": migration_checksum(migration),
                },
            )


def _legacy_database(fixture: Fix01aFixture) -> str:
    """Build the historical shapes the B batch has to stay compatible with."""

    now = datetime.now(UTC)
    run_id = "run-" + "3" * 32
    trace_id = "trace-" + "4" * 32
    checksum = "sha256:" + "6" * 64
    with fixture.driver.transaction() as connection:
        connection.execute(
            text(
                """
                INSERT INTO review_run (
                    run_id, tenant_id, case_id, case_version, status, revision,
                    resume_count, created_by_subject, current_subject_id,
                    request_checksum, document_version, dataset_version,
                    index_version, rules_version, isolation, created_at, updated_at
                ) VALUES (
                    :run_id, 'tenant-local-demo', 'CASE-LOW-001', 'case-v1',
                    'pending_confirmation', 2, 0, 'subject-local-demo',
                    'subject-local-demo', :checksum, 'doc-v1', 'data-v1', 'index-v1',
                    '2.0.0', '{}'::jsonb, :now, :now
                )
                """
            ),
            {"run_id": run_id, "checksum": "sha256:" + "5" * 64, "now": now},
        )
        connection.execute(
            text(
                """
                INSERT INTO review_trace_event (
                    event_ordinal, event_id, trace_id, run_id, event_type, step,
                    occurred_at, input_summary, output_summary, input_checksum,
                    output_checksum, source_versions, duration_ms, error, event_payload
                ) VALUES (
                    1, :event_id, :trace_id, :run_id, 'run_started', 'review_run',
                    :now, '{}'::jsonb, '{}'::jsonb, :checksum, :checksum,
                    '{"api":"review-api.v2"}'::jsonb, 0, NULL, '{}'::jsonb
                )
                """
            ),
            {
                "event_id": f"event-{run_id}-0001",
                "trace_id": trace_id,
                "run_id": run_id,
                "now": now,
                "checksum": checksum,
            },
        )
        # Two legacy audit rows whose JSON sequence disagrees with the relation column:
        # exactly the shape the old max()+1 writer left behind across two Runs.
        for index, (column, json_value) in enumerate(((1, 1), (3, 2))):
            audit_id = f"audit-{'7' * 31}{index}"
            connection.execute(
                text(
                    """
                    INSERT INTO review_audit_event (
                        sequence, audit_id, run_id, case_id, event_class, action_type,
                        actor, authorization_revision, previous_status, new_status,
                        previous_revision, new_revision, result_checksum,
                        idempotency_digest, receipt_id, api_version, occurred_at,
                        event_payload
                    ) VALUES (
                        :sequence, :audit_id, :run_id, 'CASE-LOW-001', 'lifecycle',
                        NULL, NULL, NULL, NULL, 'interrupted', NULL, NULL, NULL,
                        NULL, NULL, 'review-api.v2', :now, CAST(:payload AS jsonb)
                    )
                    """
                ),
                {
                    "sequence": column,
                    "audit_id": audit_id,
                    "run_id": run_id,
                    "now": now,
                    "payload": (
                        '{"audit_schema_version":"approval-audit.v2",'
                        f'"audit_id":"{audit_id}","sequence":{json_value},'
                        '"tenant_id":"tenant-local-demo","case_id":"CASE-LOW-001",'
                        f'"run_id":"{run_id}","event_class":"lifecycle",'
                        '"new_status":"interrupted","api_version":"review-api.v2",'
                        f'"occurred_at":"{now.isoformat()}","simulation_only":true,'
                        '"external_write_performed":false}'
                    ),
                },
            )
        connection.execute(
            text(
                "INSERT INTO review_tenant (tenant_id, display_name) "
                "VALUES ('tenant-local-demo', 'Local demo') ON CONFLICT DO NOTHING"
            )
        )
    return run_id


def test_upgrade_backfills_identity_and_sequence_without_rewriting_history() -> None:
    """V-B-06: a v3 database upgrades in place, and the stored payload stays history."""

    with database("v_b_06_upgrade") as fixture:
        store = ReviewStore.from_url(fixture.url_for("v1b_upgrade"))
        try:
            _apply_migrations_upto(store, 3)
            run_id = _legacy_database(fixture)
            with fixture.driver.transaction() as connection:
                before = [
                    (int(row.sequence), row.event_payload)
                    for row in connection.execute(
                        text(
                            "SELECT sequence, event_payload::text FROM "
                            "review_audit_event ORDER BY sequence"
                        )
                    ).all()
                ]
            report = store.migrate()
            assert report.applied == (4,), report
            with fixture.driver.transaction() as connection:
                after = [
                    (int(row.sequence), row.event_payload)
                    for row in connection.execute(
                        text(
                            "SELECT sequence, event_payload::text FROM "
                            "review_audit_event ORDER BY sequence"
                        )
                    ).all()
                ]
                identity = connection.execute(
                    text("SELECT trace_id FROM review_run")
                ).scalar_one()
                next_value = connection.execute(
                    text(
                        "SELECT next_value FROM review_sequence_allocator "
                        "WHERE stream = 'audit'"
                    )
                ).scalar_one()
            assert before == after
            assert identity == "trace-" + "4" * 32
            assert int(next_value) == 4
            repository = ReviewRepository(store)
            events, anomalies = repository.audit_projection(run_id)
            assert [item.sequence for item in events] == [1, 3]
            assert anomalies == (f"audit-{'7' * 31}1",)
            again = store.migrate()
            assert again.applied == ()
            assert again.already_present == (1, 2, 3, 4)
            assert repository.load_run(run_id) is not None
        finally:
            store.close()


def test_a_failed_upgrade_rolls_back_and_can_be_retried() -> None:
    """V-B-06: an upgrade that fails leaves no half-applied schema behind."""

    with database("v_b_06_retry") as fixture:
        store = ReviewStore.from_url(fixture.url_for("v1b_retry"))
        try:
            _apply_migrations_upto(store, 3)
            run_id = _legacy_database(fixture)
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "CREATE TABLE review_sequence_allocator ("
                        "stream text PRIMARY KEY, next_value bigint NOT NULL, "
                        "unexpected text NOT NULL)"
                    )
                )
            with pytest.raises(Exception):
                store.migrate()
            with fixture.driver.transaction() as connection:
                versions = [
                    int(item)
                    for item in connection.execute(
                        text(
                            "SELECT version FROM review_schema_migration "
                            "ORDER BY version"
                        )
                    ).scalars()
                ]
                columns = [
                    str(item)
                    for item in connection.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'review_run'"
                        )
                    ).scalars()
                ]
            assert versions == [1, 2, 3], versions
            assert "trace_id" not in columns, columns
            with fixture.driver.transaction() as connection:
                connection.execute(text("DROP TABLE review_sequence_allocator"))
            report = store.migrate()
            assert report.applied == (4,)
            with fixture.driver.transaction() as connection:
                assert (
                    connection.execute(
                        text("SELECT trace_id FROM review_run")
                    ).scalar_one()
                    == "trace-" + "4" * 32
                )
            assert ReviewRepository(store).load_run(run_id) is not None
        finally:
            store.close()


def test_unknown_migration_version_fails_closed() -> None:
    """V-B-06: a database carrying migrations this build does not know is refused."""

    from procurement_review.review_store import UnknownMigrationVersionError

    with database("v_b_06_unknown") as fixture:
        store = ReviewStore.from_url(fixture.url_for("v1b_unknown"))
        try:
            store.migrate()
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "INSERT INTO review_schema_migration (version, name, checksum) "
                        "VALUES (99, 'from-the-future', 'sha256:x')"
                    )
                )
            with pytest.raises(UnknownMigrationVersionError):
                store.migrate()
        finally:
            store.close()


def _reset_roles(admin: Any, roles: tuple[str, ...]) -> None:
    """Drop the delivery roles and every privilege they hold in this database."""

    existing = {
        str(row[0])
        for row in admin.execute(
            "SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)", (list(roles),)
        ).fetchall()
    }
    if existing:
        admin.execute("DROP OWNED BY " + ", ".join(sorted(existing)))
    for role in roles:
        admin.execute(f"DROP ROLE IF EXISTS {role}")


def test_runtime_role_reaches_the_allocator_and_nothing_more() -> None:
    """V-B-06: the delivered runtime role advances the allocator and not the registry."""

    scripts = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import init_bootstrap  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    assert "review_sequence_allocator" in init_bootstrap.RUNTIME_WRITABLE_TABLES

    import psycopg

    with database("v_b_06_role") as fixture:
        head, _, tail = fixture.url_for("v1b_role").partition("?")
        scheme_and_host = head.replace(
            "postgresql+psycopg://", "postgresql://"
        ).rpartition("/")[0]
        database_dsn = f"{scheme_and_host}/{fixture.database}?{tail}"
        runtime_dsn = f"{scheme_and_host}/{fixture.database}"
        try:
            with psycopg.connect(database_dsn, autocommit=True) as admin:
                _reset_roles(admin, init_bootstrap.ALL_ROLES)
                for role in init_bootstrap.ALL_ROLES:
                    admin.execute(f"CREATE ROLE {role} LOGIN")
                init_bootstrap.grant_table_privileges(admin)
            with psycopg.connect(runtime_dsn, user="review_runtime") as runtime:
                value = runtime.execute(
                    "SELECT next_value FROM review_sequence_allocator "
                    "WHERE stream = 'audit' FOR UPDATE"
                ).fetchone()[0]
                assert int(value) >= 1
                runtime.execute(
                    "UPDATE review_sequence_allocator SET next_value = next_value + 1 "
                    "WHERE stream = 'audit'"
                )
                runtime.rollback()
                for statement in (
                    "UPDATE review_authorization_revision SET revision = revision + 1",
                    "UPDATE review_case_grant SET granted_at = now()",
                    "UPDATE review_case_registry SET label = 'rewritten'",
                    "INSERT INTO review_procurement_record (tenant_id, case_id, "
                    "record_kind, record_id, dataset_version, as_of, payload) VALUES "
                    "('tenant-local-demo', 'CASE-LOW-001', 'order', 'PO-ROLE', "
                    "'security-procurement.v1', DATE '2026-06-30', '{}'::jsonb)",
                ):
                    with pytest.raises(psycopg.errors.Error):
                        runtime.execute(statement)
                    runtime.rollback()
        finally:
            with psycopg.connect(database_dsn, autocommit=True) as admin:
                _reset_roles(admin, init_bootstrap.ALL_ROLES)
