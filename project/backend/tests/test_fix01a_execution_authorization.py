"""FIX-01A V-A-01 and V-A-02: live execution authority under real interleavings.

Each scenario parks the service at one named commit boundary with the product's own file
barrier, commits a revocation from a second database backend while it stands there, then
releases it and reads the durable facts back.  A parked barrier on the service's own event
loop stops that loop, so "the revocation was already committed when this step ran" is a fact
of the run rather than a timing guess.  Where a step really does run concurrently — a
retrieval port call is dispatched to a worker thread — the assertions are about durable
database facts and the read-site counters only.

Two scenarios stand apart from that matrix on purpose.  One clean execution publishes the
absolute number of reads each protected site enters, so a site that stopped asking for
admission cannot hide inside a matrix that only measures increments.  The other loads a real
stdio startup context and shows that its scope — checked once when the process started — is
re-checked by every tool entry against a revocation committed on another connection.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from sqlalchemy import text

from procurement_review.api_contracts import ResumeRequestV2
from procurement_review.authorization import (
    AUTHORITY_CREDENTIAL_EXPIRED,
    AUTHORITY_CREDENTIAL_REVOKED,
    AUTHORITY_GRANT_REVOKED,
    AUTHORITY_REVISION_CHANGED,
    AUTHORITY_SUBJECT_DISABLED,
)
from procurement_review.contracts import ReviewLifecycleStatus
from procurement_review.fault_hooks import (
    FAULT_APPLICATION_NAME_ENV,
    BarrierSpec,
    FaultController,
)
from procurement_review.mcp_server import (
    GET_PURCHASE_ORDER_TOOL,
    GET_SUPPLIER_TOOL,
    LIST_HISTORICAL_TRANSACTIONS_TOOL,
    ProcurementToolErrorCode,
    ProcurementToolOutcome,
    create_procurement_mcp_server,
)
from procurement_review.mcp_startup import MCP_STARTUP_VERSION, load_startup_context
from procurement_review.procurement_store import ProcurementScopeError
from procurement_review.review_repository import ExecutionInvalidated
from procurement_review.scoped_runtime import ScopedReviewRuntime
from procurement_review.security_contracts import RuntimeScope, SecurityError

from fix01a_support import (
    Fix01aFixture,
    Identity,
    Loop,
    barrier_dir,
    parked_fault,
    release_barrier,
    runtime,
    settle,
    wait_barrier,
)

LOW_CASE = "CASE-LOW-001"

#: Every revocation kind named by V-A-01, committed on the independent connection.
REVOCATIONS: dict[str, tuple[str, Callable[[Fix01aFixture, Identity], None]]] = {
    "credential_revoked": (
        AUTHORITY_CREDENTIAL_REVOKED,
        lambda fixture, ident: fixture.revoke_credential(ident.credential_id),
    ),
    "credential_expired": (
        AUTHORITY_CREDENTIAL_EXPIRED,
        lambda fixture, ident: fixture.expire_credential(ident.credential_id),
    ),
    "subject_disabled": (
        AUTHORITY_SUBJECT_DISABLED,
        lambda fixture, ident: fixture.disable_subject(ident.context.subject_id),
    ),
    "case_grant_revoked": (
        AUTHORITY_GRANT_REVOKED,
        lambda fixture, ident: fixture.revoke_case_grant(
            ident.context.subject_id, LOW_CASE
        ),
    ),
    "authorization_revision_changed": (
        AUTHORITY_REVISION_CHANGED,
        lambda fixture, ident: fixture.bump_authorization_revision(),
    ),
}

#: The service-loop boundaries V-A-01 names: first dispatch, the attempt booking, a branch
#: adoption and the summary commit.
PARK_POINTS = (
    "after_create_commit",
    "attempt_started:policy",
    "before_branch_commit:policy",
    "after_all_branches_committed",
)

SETTLED = (
    "interrupted",
    "failed",
    "cancelled",
    "pending_confirmation",
    "pending_human_review",
)


@contextmanager
def database(name: str) -> Iterator[Fix01aFixture]:
    fixture = Fix01aFixture(name)
    try:
        yield fixture
    finally:
        fixture.close()


def additions(before: list[str], after: list[str]) -> list[str]:
    """What appeared since, asserting that nothing already committed was rewritten."""

    missing = [item for item in before if item not in after]
    assert not missing, f"committed history was rewritten: {missing}"
    return [item for item in after if item not in before]


def adopted_branch_results(
    fixture: Fix01aFixture, run_id: str
) -> dict[str, dict[str, object]]:
    """The branch results a Run actually adopted, read back from the checkpoint store."""

    with fixture.probe.transaction() as connection:
        rows = connection.execute(
            text(
                "SELECT worker, generation, attempt_no, result_payload "
                "FROM review_branch_checkpoint WHERE run_id = :r ORDER BY worker"
            ),
            {"r": run_id},
        ).all()
    return {
        str(row.worker): {
            "generation": int(row.generation),
            "attempt_no": int(row.attempt_no),
            "result": dict(row.result_payload)["result"],
        }
        for row in rows
    }


def durable_footprint(fixture: Fix01aFixture) -> dict[str, int]:
    """Row counts of every ledger table, over the whole scenario database.

    A read-only entry point must leave this untouched; comparing it before and after one
    call is what turns "the refusal wrote nothing" into a database fact.
    """

    tables = (
        "review_run",
        "review_run_generation",
        "review_run_summary",
        "review_worker_attempt",
        "review_branch_checkpoint",
        "review_audit_event",
        "review_trace_event",
        "review_idempotency",
        "review_procurement_record",
    )
    with fixture.probe.transaction() as connection:
        return {
            table: int(
                connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            )
            for table in tables
        }


def stored_record_payload(
    fixture: Fix01aFixture, *, scope: RuntimeScope, record_id: str
) -> dict[str, object]:
    """The stored body of one registered procurement record, as the database holds it."""

    with fixture.probe.transaction() as connection:
        return dict(
            connection.execute(
                text(
                    "SELECT payload FROM review_procurement_record "
                    "WHERE tenant_id = :t AND case_id = :c AND record_id = :r"
                ),
                {"t": scope.tenant_id, "c": scope.case_id, "r": record_id},
            ).scalar_one()
        )


def drive_parked(
    fixture: Fix01aFixture,
    barrier: str,
    revocation: Callable[[Fix01aFixture, Identity], None],
    *,
    key: str,
) -> dict[str, object]:
    """Park one execution at a boundary, revoke behind it, then let it run."""

    directory = barrier_dir(f"{barrier}-{key}")
    fault = parked_fault(directory, barrier)
    active = runtime(fixture, fault)
    ident = fixture.identity()
    loop = Loop()
    try:
        handle = loop.start(
            active.service.create_review(
                fixture.create_request(ident),
                registration=ident.registration,
                scope=ident.scope,
                context=ident.context,
                idempotency_key=key,
            )
        )
        wait_barrier(directory, barrier)
        run_id = fixture.latest_run()
        before = fixture.ledger(run_id)
        reads_before = active.reads_entered
        revocation(fixture, ident)
        release_barrier(directory, barrier)
        handle.result(timeout=120)
        after = settle(fixture, run_id, *SETTLED)
        return {
            "run_id": run_id,
            "before": before,
            "after": after,
            "reads_before": reads_before,
            "reads_after": active.reads_entered,
            "denials": active.read_denials,
        }
    finally:
        release_barrier(directory, barrier)
        loop.call(active.service.shutdown(), timeout=30.0)
        loop.close()


@pytest.mark.parametrize("park_point", PARK_POINTS)
@pytest.mark.parametrize("revocation", sorted(REVOCATIONS))
def test_revocation_stops_every_later_step(park_point: str, revocation: str) -> None:
    """V-A-01: no new read, attempt, adoption or summary after a committed revocation."""

    _, apply = REVOCATIONS[revocation]
    with database("matrix") as fixture:
        outcome = drive_parked(
            fixture, park_point, apply, key=f"{park_point}-{revocation}"
        )
        before = outcome["before"]
        after = outcome["after"]
        assert isinstance(before, dict) and isinstance(after, dict)

        assert additions(before["attempts"], after["attempts"]) == []
        assert additions(before["checkpoints"], after["checkpoints"]) == []
        assert after["summaries"] == before["summaries"] == 0
        assert after["status"] == ReviewLifecycleStatus.INTERRUPTED.value
        assert after["error_code"] == "authorization_revoked"

        # The only new durable facts are the interruption itself.
        new_trace = additions(before["trace"], after["trace"])
        assert all("review_workflow:step_failed" in item for item in new_trace), (
            new_trace
        )
        new_audit = additions(before["audit"], after["audit"])
        assert new_audit, after
        assert all(
            "lifecycle:interrupted:authorization_revoked" in item for item in new_audit
        ), new_audit

        # Real read-entry counts: once the loop unfroze, no further read body started.
        assert outcome["reads_after"] == outcome["reads_before"]
        denials = outcome["denials"]
        assert isinstance(denials, dict)
        if park_point == "attempt_started:policy":
            # The parked branch's very next step is its own read admission, so the refusal
            # is recorded at the read site rather than only at a later commit.
            assert denials.get("policy", 0) >= 1, denials
            assert outcome["reads_after"].get("policy", 0) == 0


def test_in_flight_result_is_never_adopted_or_summarized() -> None:
    """V-A-01: a read issued before the revocation may finish, and stops there."""

    barrier = "during_port_read:policy"
    directory = barrier_dir("inflight")
    fault = parked_fault(directory, barrier)
    fixture = Fix01aFixture("inflight")
    loop = Loop()
    active = runtime(fixture, fault)
    ident = fixture.identity()
    try:
        handle = loop.start(
            active.service.create_review(
                fixture.create_request(ident),
                registration=ident.registration,
                scope=ident.scope,
                context=ident.context,
                idempotency_key="inflight",
            )
        )
        wait_barrier(directory, barrier)
        run_id = fixture.latest_run()
        assert fixture.revoke_credential(ident.credential_id)
        release_barrier(directory, barrier)
        handle.result(timeout=120)
        body = settle(fixture, run_id, *SETTLED)
        assert body["summaries"] == 0
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value
        assert body["error_code"] == "authorization_revoked"
        # The already-issued read completed, yet its result reached neither the record nor
        # the aggregation: no adoption, no aggregator event, no checkpoint.
        assert not [item for item in body["checkpoints"] if item.startswith("policy@")]
        assert not [item for item in body["trace"] if "worker.policy" in item]
        assert not [
            item for item in body["trace"] if "deterministic_aggregator" in item
        ]
    finally:
        release_barrier(directory, barrier)
        loop.call(active.service.shutdown(), timeout=30.0)
        loop.close()
        fixture.close()


def test_authority_lock_covers_the_check_and_not_the_read() -> None:
    """V-A-02: a revocation linearizes with the check and never waits for the read.

    Half one parks inside the admission transaction, which holds the tenant revision row
    share lock, so a concurrent revision bump has to time out.  Half two parks inside the
    read body, after that transaction closed, and the same bump commits at once.
    """

    def bump_while_parked(barrier: str) -> str | None:
        directory = barrier_dir(f"lockscope-{barrier}")
        fault = parked_fault(directory, barrier)
        fixture = Fix01aFixture("lockscope")
        loop = Loop()
        active = runtime(fixture, fault)
        ident = fixture.identity()
        engine = fixture.admin_engine(autocommit=False)
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key=barrier,
                )
            )
            wait_barrier(directory, barrier)
            with engine.connect() as connection:
                connection.execute(text("SET LOCAL lock_timeout = '2s'"))
                try:
                    connection.execute(
                        text(
                            "UPDATE review_authorization_revision "
                            "SET revision = revision + 1 WHERE tenant_id = :t"
                        ),
                        {"t": ident.context.tenant_id},
                    )
                    connection.commit()
                    answer = None
                except Exception as error:  # noqa: BLE001 - the answer is asserted below
                    connection.rollback()
                    answer = str(error)
            release_barrier(directory, barrier)
            handle.result(timeout=120)
            return answer
        finally:
            release_barrier(directory, barrier)
            engine.dispose()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()
            fixture.close()

    blocked = bump_while_parked("during_read_admission:dispatch")
    assert blocked is not None, "a revocation slipped past a held admission"
    assert "lock timeout" in blocked.lower(), blocked
    assert bump_while_parked("during_port_read:policy") is None, (
        "an authorization row lock was still held while the port was read"
    )


def test_heartbeat_cannot_renew_a_lost_generation() -> None:
    """V-A-02: a revoked generation stops renewing and is then interrupted."""

    fixture = Fix01aFixture("heartbeat")
    slow = FaultController(
        enabled=True,
        slow={
            worker: {1: 4.0} for worker in ("policy", "contract", "procurement_data")
        },
    )
    loop = Loop()
    active = runtime(fixture, slow, lease_seconds=30.0, heartbeat_seconds=0.2)
    try:
        ident = fixture.identity()
        created, _ = loop.call(
            active.service.create_review(
                fixture.create_request(ident),
                registration=ident.registration,
                scope=ident.scope,
                context=ident.context,
                idempotency_key="heartbeat",
            )
        )
        run_id = created.run_id
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not fixture.ledger(run_id)["attempts"]:
            time.sleep(0.05)
        assert fixture.ledger(run_id)["attempts"], "the branches never started reading"
        assert fixture.revoke_credential(ident.credential_id)
        time.sleep(0.6)  # several heartbeat intervals, so a live lease would have moved
        first = fixture.ledger(run_id)["leases"]
        time.sleep(1.2)
        assert fixture.ledger(run_id)["leases"] == first, "the lease kept being renewed"
        body = settle(fixture, run_id, "interrupted")
        assert body["error_code"] == "authorization_revoked"
        assert body["summaries"] == 0
    finally:
        loop.call(active.service.shutdown(), timeout=30.0)
        loop.close()
        fixture.close()


def test_store_outage_stops_progress_without_faking_a_persisted_interrupt() -> None:
    """V-A-02: with the store unreachable the read stops and nothing is invented."""

    barrier = "during_read_admission:dispatch"
    directory = barrier_dir("outage")
    from v1_03_support import require_admin_dsn

    # The terminate action breaks every backend carrying this connection tag.  Only the
    # application's engine is tagged with it: the scenario still reads and scans over the
    # fixture's own driver/probe tags, and nothing else running can be caught in the outage.
    tag = f"fix01a_outage_{secrets.token_hex(4)}"
    previous_tag = os.environ.get(FAULT_APPLICATION_NAME_ENV)
    os.environ[FAULT_APPLICATION_NAME_ENV] = tag
    fault = FaultController(
        enabled=True,
        directory=directory,
        admin_dsn=require_admin_dsn(),
        barriers={barrier: BarrierSpec(action="terminate", timeout_seconds=2.0)},
    )
    fixture = Fix01aFixture("outage", application_name=tag)
    loop = Loop()
    active = runtime(
        fixture,
        fault,
        lease_seconds=2.0,
        heartbeat_seconds=0.5,
        scan_interval_seconds=0.2,
    )
    settled = False

    def restore_environment() -> None:
        if previous_tag is None:
            os.environ.pop(FAULT_APPLICATION_NAME_ENV, None)
        else:
            os.environ[FAULT_APPLICATION_NAME_ENV] = previous_tag

    try:
        ident = fixture.identity()
        handle = loop.start(
            active.service.create_review(
                fixture.create_request(ident),
                registration=ident.registration,
                scope=ident.scope,
                context=ident.context,
                idempotency_key="outage",
            )
        )
        wait_barrier(directory, barrier)
        run_id = fixture.latest_run()
        release_barrier(directory, barrier)
        handle.result(timeout=120)
        # No scanner runs in this harness, so the lease expiry and the product scanner
        # decide what the interruption is allowed to claim.  The scan goes through the
        # observation connection on purpose: the injected outage takes every backend
        # tagged as the application, which is exactly the connection it must not need.
        deadline = time.monotonic() + 30.0
        body = fixture.ledger(run_id)
        while time.monotonic() < deadline:
            fixture.observer.scan_expired_leases(
                now=datetime.now(UTC),
                events_factory=lambda _run_id: (),
                audit_factory=lambda _run_id: None,
            )
            body = fixture.ledger(run_id)
            if body["status"] != ReviewLifecycleStatus.PROCESSING.value:
                break
            time.sleep(0.2)
        # Nothing was read, adopted or summarized on the strength of an unreachable store,
        # and the Run never claims a completion it did not commit.
        assert sum(active.reads_entered.values()) == 0
        assert body["attempts"] == []
        assert body["summaries"] == 0
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value
        assert body["error_code"] in {"storage_unavailable", "lease_expired"}, body
        settled = True
    finally:
        # Every step runs on its own: this scenario has just poisoned the connection it
        # came in on, and a loop left polling would starve the later timing scenarios.
        cleanup = (
            lambda: loop.call(active.service.shutdown(), timeout=30.0),
            loop.close,
            fixture.close,
            restore_environment,
        )
        failure: BaseException | None = None
        for step in cleanup:
            try:
                step()
            except BaseException as error:  # noqa: BLE001 - keep tearing down the rest
                failure = failure or error
        if failure is not None and settled:
            raise failure


def test_superseded_generation_cannot_alter_the_taken_over_run() -> None:
    """V-A-02: a new authorized operator takes over; the old generation changes nothing."""

    barrier = "before_branch_commit:policy"
    directory = barrier_dir("superseded")
    fault = parked_fault(directory, barrier)
    fixture = Fix01aFixture("superseded")
    loop = Loop()
    taker_loop = Loop()
    old = runtime(fixture, fault, lease_seconds=2.0, heartbeat_seconds=50.0)
    taker = runtime(fixture, None, lease_seconds=60.0, heartbeat_seconds=50.0)
    try:
        ident = fixture.identity()
        handle = loop.start(
            old.service.create_review(
                fixture.create_request(ident),
                registration=ident.registration,
                scope=ident.scope,
                context=ident.context,
                idempotency_key="superseded",
            )
        )
        wait_barrier(directory, barrier)
        run_id = fixture.latest_run()
        before = fixture.ledger(run_id)
        # The service loop is parked, so the interruption and the take-up run on the
        # harness' own connections: the lease really expires and the scanner owns it.
        time.sleep(2.5)
        scanned = fixture.repository.scan_expired_leases(
            now=datetime.now(UTC),
            events_factory=lambda _run_id: (),
            audit_factory=lambda _run_id: None,
        )
        assert scanned == (run_id,)
        assert fixture.ledger(run_id)["status"] == "interrupted"
        fresh = fixture.identity()  # the same still-granted subject, a live credential
        current = taker.service.get_review(run_id, context=fresh.context)
        taker_loop.call(
            taker.service.resume_review(
                run_id,
                ResumeRequestV2(expected_version=current.revision),
                context=fresh.context,
                idempotency_key="take-over",
            )
        )
        release_barrier(directory, barrier)
        handle.result(timeout=120)
        body = settle(fixture, run_id, "pending_confirmation", "pending_human_review")
        grown = additions(before["checkpoints"], body["checkpoints"])
        late = [item for item in grown if item.endswith("@1")]
        assert late == [], f"the superseded generation committed: {late}"
        assert grown, "the new generation adopted nothing either"
        assert all(item.endswith("@2") for item in grown), grown
        assert body["resume_count"] == 1
        assert body["summaries"] == 1
    finally:
        release_barrier(directory, barrier)
        loop.call(old.service.shutdown(), timeout=30.0)
        taker_loop.call(taker.service.shutdown(), timeout=30.0)
        loop.close()
        taker_loop.close()
        fixture.close()


def test_out_of_scope_tool_argument_stays_a_branch_rejection() -> None:
    """AC-01 item 5: one bad tool argument never becomes a whole-generation event."""

    with database("scope") as fixture:
        ident = fixture.identity()
        admissions: list[tuple[str | None, bool]] = []

        def admission(connection: Any = None, *, site: str | None = None) -> None:
            admissions.append((site, connection is not None))

        scoped = ScopedReviewRuntime.build(
            store=fixture.app, scope=ident.scope, admission=admission
        )
        # The assembly's own protected preloads run through this same gate, each in the
        # transaction that reads its data (FIX-06A).  The assertions below are about the
        # *tool* boundary — an out-of-scope argument must never reach the authority check —
        # so the preload admissions are recorded and then set aside.
        assert admissions == [
            ("retrieval_index", True),
            ("procurement_records", True),
            ("evidence_index", True),
        ], admissions
        admissions.clear()
        allowed = ident.scope.allowed_order_ids
        assert allowed
        outside = next(
            item
            for item in ("PO-001", "PO-002", "PO-003", "PO-004", "PO-005")
            if item not in allowed
        )
        server = create_procurement_mcp_server(scoped.procurement, scope=ident.scope)

        def read_order(record_id: str) -> dict[str, object]:
            """One call over the production MCP boundary, answered by its own response."""

            async def scenario() -> dict[str, object]:
                async with Client(server, raise_exceptions=True) as client:
                    answer = await client.call_tool(
                        GET_PURCHASE_ORDER_TOOL,
                        {
                            "purchase_order_id": record_id,
                            "dataset_version": ident.scope.dataset_version,
                        },
                    )
                assert answer.structured_content is not None
                return dict(answer.structured_content)

            return asyncio.run(scenario())

        denied = read_order(outside)
        assert denied["outcome"] == ProcurementToolOutcome.SCOPE_DENIED.value
        assert denied["error_code"] == ProcurementToolErrorCode.OUT_OF_SCOPE.value
        assert denied["facts"] == []
        assert admissions == [], "an out-of-scope argument reached the authority check"

        # One real execution first, so "nothing new was written" is a claim over ledger
        # rows that exist rather than over an empty database.
        active = runtime(fixture, None)
        loop = Loop()
        run_id = ""
        try:
            created, _ = loop.call(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="scope-rejection",
                )
            )
            run_id = created.run_id
            settle(fixture, run_id, "pending_confirmation", "pending_human_review")
        finally:
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        before = fixture.ledger(run_id)
        adopted = adopted_branch_results(fixture, run_id)
        assert before["status"] == "pending_confirmation", before
        assert sorted(adopted) == ["contract", "policy", "procurement_data"], adopted
        assert [row["generation"] for row in adopted.values()] == [1, 1, 1]
        assert all(
            row["result"]["outcome"] == "completed" for row in adopted.values()
        ), adopted
        footprint = durable_footprint(fixture)
        assert sum(footprint.values()) > 0, footprint

        # The same rejection against a Run that already holds adopted branch results stays
        # the tool's answer: no new checkpoint, summary, attempt or trace row appears, and
        # the branch results already committed are not rewritten.
        assert (
            read_order(outside)["outcome"] == ProcurementToolOutcome.SCOPE_DENIED.value
        )
        assert admissions == [], "a rejected tool argument reached the authority check"
        assert (
            read_order(allowed[0])["outcome"] == ProcurementToolOutcome.KNOWN.value
        ), "an in-scope read never reached the authority gate"
        assert admissions == [(None, False)], admissions
        assert fixture.ledger(run_id) == before
        assert adopted_branch_results(fixture, run_id) == adopted
        assert durable_footprint(fixture) == footprint
        with pytest.raises(ProcurementScopeError):
            scoped.procurement.get_purchase_order(outside, ident.scope.dataset_version)
        assert admissions == [(None, False)]


def test_concurrent_revocation_and_commits_do_not_deadlock() -> None:
    """V-A-02: the shared lock order stays live under real concurrent contention."""

    fixture = Fix01aFixture("deadlock")
    loop = Loop()
    stop = threading.Event()
    errors: list[str] = []
    admission_errors: list[str] = []
    active = runtime(fixture, None, lease_seconds=60.0, heartbeat_seconds=1.0)
    try:
        ident = fixture.identity()
        created, _ = loop.call(
            active.service.create_review(
                fixture.create_request(ident),
                registration=ident.registration,
                scope=ident.scope,
                context=ident.context,
                idempotency_key="deadlock-create",
            )
        )
        run_id = created.run_id
        barrier = threading.Barrier(10)

        def contend(index: int) -> None:
            engine = fixture.admin_engine()
            try:
                barrier.wait(timeout=30)
                for _ in range(8):
                    with engine.connect() as connection:
                        try:
                            if index % 3 == 0:
                                connection.execute(
                                    text(
                                        "SELECT revision FROM "
                                        "review_authorization_revision "
                                        "WHERE tenant_id = :t FOR SHARE"
                                    ),
                                    {"t": ident.context.tenant_id},
                                )
                            elif index % 3 == 1:
                                connection.execute(
                                    text(
                                        "UPDATE review_authorization_revision "
                                        "SET revision = revision + 1 "
                                        "WHERE tenant_id = :t"
                                    ),
                                    {"t": ident.context.tenant_id},
                                )
                            else:
                                connection.execute(
                                    text(
                                        "SELECT status FROM review_run "
                                        "WHERE run_id = :r FOR UPDATE"
                                    ),
                                    {"r": run_id},
                                )
                        except Exception as error:  # noqa: BLE001 - answer is asserted
                            errors.append(f"{type(error).__name__}:{error}")
                    if stop.is_set():
                        return
            finally:
                engine.dispose()

        workers = [
            threading.Thread(target=contend, args=(index,), daemon=True)
            for index in range(9)
        ]
        for worker in workers:
            worker.start()
        barrier.wait(timeout=30)
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline and any(item.is_alive() for item in workers):
            try:
                active.repository.assert_generation_live(
                    run_id=run_id,
                    generation=1,
                    now=datetime.now(UTC),
                    site="contention",
                )
            except ExecutionInvalidated:
                pass
            except Exception as error:  # noqa: BLE001 - judged with the thread answers
                admission_errors.append(f"{type(error).__name__}:{error}")
        stop.set()
        for worker in workers:
            worker.join(timeout=30)
            assert not worker.is_alive(), "a contending connection never finished"
        reporting = errors + admission_errors
        assert [item for item in reporting if "40P01" in item] == [], reporting
        assert admission_errors == [], admission_errors
        assert errors == [], errors
    finally:
        stop.set()
        loop.call(active.service.shutdown(), timeout=30.0)
        loop.close()
        fixture.close()


def test_one_clean_run_enters_every_protected_read_site() -> None:
    """V-A-01 positive control: the absolute read counts one clean execution must make.

    The revocation matrix above can only show that a read *stops* after a committed
    revocation; on its own it cannot notice a read entry point that stopped asking for
    admission at all.  This test fixes what one successful Fake run actually enters, per
    site, independently of any increment measured against a revocation.
    """

    with database("reads") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        loop = Loop()
        run_id = ""
        try:
            created, _ = loop.call(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="reads-positive-control",
                )
            )
            run_id = created.run_id
            settle(fixture, run_id, "pending_confirmation", "pending_human_review")
        finally:
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()
        # The control is only worth anything on a run that really finished every branch.
        body = fixture.ledger(run_id)
        assert body["status"] == "pending_confirmation", body
        assert body["summaries"] == 1
        assert sorted(body["checkpoints"]) == [
            "contract@1",
            "policy@1",
            "procurement_data@1",
        ]
        # The two document Workers read the index once each; the procurement branch answers
        # the fixed three-tool sequence, so each protected site carries its own count.
        assert active.reads_entered == {
            "policy": 1,
            "contract": 1,
            "procurement_data": 3,
        }, active.reads_entered
        assert active.read_denials == {}, active.read_denials


def stdio_context(fixture: Fix01aFixture, ident: Identity, directory: Path) -> Path:
    """Write one protected startup context for this scenario's database and return it."""

    directory.mkdir()
    os.chmod(directory, 0o700)
    context_file = directory / "context.json"
    context_file.write_text(
        json.dumps(
            {
                "startup_context_version": MCP_STARTUP_VERSION,
                "database_url": fixture.url_for("fix01a_stdio"),
                "scope": ident.scope.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    os.chmod(context_file, 0o600)
    return context_file


def stdio_calls(scope: RuntimeScope) -> tuple[tuple[str, dict[str, str]], ...]:
    """The three protected read entries of the stdio MCP server, each with one argument."""

    order = scope.allowed_order_ids[0]
    return (
        (GET_SUPPLIER_TOOL, {"supplier_id": scope.allowed_supplier_ids[0]}),
        (GET_PURCHASE_ORDER_TOOL, {"purchase_order_id": order}),
        (LIST_HISTORICAL_TRANSACTIONS_TOOL, {"purchase_order_id": order}),
    )


def backend_pids(entries: list[str], tag: str | None = None) -> set[int]:
    """The PIDs behind ``fix01a_support``'s ``pid:…`` readings, optionally one tag only."""

    if tag is None:
        return {int(item.split(":", 1)[0]) for item in entries}
    return {int(item.split(":", 1)[0]) for item in entries if item.endswith(f":{tag}")}


def test_stdio_read_gate_rechecks_authority_at_every_tool_entry(
    tmp_path: Path,
) -> None:
    """V-A-01: a stdio scope that passed startup still stops every later read.

    ``verify_scope_authorization`` runs once when the startup context is loaded, which is
    not the moment any record is read.  AF-02 is closed here: the gate the three tools
    actually stand behind is :func:`authorization.scope_admission`, and a revocation
    committed on a second connection after startup has to stop each of them.
    """

    with database("stdio") as fixture:
        ident = fixture.identity()
        context = load_startup_context(stdio_context(fixture, ident, tmp_path / "gate"))
        try:
            scope = context.scope
            server = create_procurement_mcp_server(context.reader, scope=scope)
            calls = stdio_calls(scope)

            def answer(name: str, arguments: dict[str, str]) -> Any:
                async def scenario() -> Any:
                    async with Client(server, raise_exceptions=True) as client:
                        return await client.call_tool(
                            name,
                            {**arguments, "dataset_version": scope.dataset_version},
                        )

                return asyncio.run(scenario())

            order = scope.allowed_order_ids[0]
            before = {name: answer(name, arguments) for name, arguments in calls}
            assert [
                dict(item.structured_content)["outcome"] for item in before.values()
            ] == ["KNOWN"] * 3
            # The startup context opened its own engine, so the revocation below arrives
            # over a connection this reader never shares — the same boundary a real stdio
            # process sits behind.
            backends = fixture.app_backends()
            stdio_pids = backend_pids(backends, "fix01a_stdio")
            app_pids = backend_pids(backends, "fix01a_app")
            assert stdio_pids and app_pids, backends
            assert not stdio_pids & app_pids, backends
            footprint = durable_footprint(fixture)
            stored = stored_record_payload(fixture, scope=scope, record_id=order)
            read_order = dict(before[GET_PURCHASE_ORDER_TOOL].structured_content)
            assert [dict(item)["result"] for item in read_order["facts"]][0] == stored

            assert fixture.revoke_credential(ident.credential_id)

            for name, arguments in calls:
                # The refusal travels as an error answer with no record behind it; the MCP
                # wire format carries no stable code, so the code is asserted at the gate
                # the tool body calls, which is what produced that error.
                refused = answer(name, arguments)
                assert refused.is_error is True, name
                assert refused.structured_content is None, name
                read = getattr(context.reader, name)
                with pytest.raises(SecurityError) as refusal:
                    read(**arguments, dataset_version=scope.dataset_version)
                assert refusal.value.code == "runtime_scope_not_authorized", name
                assert refusal.value.status_code == 403, name
            # Nothing was written on the strength of the revoked scope, and the records the
            # process had already read were left exactly as stored.
            assert durable_footprint(fixture) == footprint
            assert (
                stored_record_payload(fixture, scope=scope, record_id=order) == stored
            )
        finally:
            context.close()


def test_stdio_read_gate_never_waits_for_a_held_authorization_lock(
    tmp_path: Path,
) -> None:
    """V-A-01/V-A-02: the read-only stdio gate takes no row lock and still sees commits.

    The stdio role runs :func:`authorization.live_execution_authority` with ``lock=False``.
    With another backend holding the tenant revision row ``FOR UPDATE``, its reads must
    still be admitted, and the moment that holder commits a revision change the same reads
    must stop — a stale startup conclusion may not carry them.
    """

    with database("stdiolock") as fixture:
        ident = fixture.identity()
        context = load_startup_context(stdio_context(fixture, ident, tmp_path / "lock"))
        server = create_procurement_mcp_server(context.reader, scope=context.scope)
        calls = stdio_calls(context.scope)
        engine = fixture.admin_engine(autocommit=False)
        holding = threading.Event()
        release = threading.Event()
        answers: list[str] = []

        def hold_the_revision_row() -> None:
            with engine.connect() as connection:
                connection.execute(
                    text(
                        "SELECT revision FROM review_authorization_revision "
                        "WHERE tenant_id = :t FOR UPDATE"
                    ),
                    {"t": ident.context.tenant_id},
                )
                holding.set()
                release.wait(timeout=60.0)
                connection.execute(
                    text(
                        "UPDATE review_authorization_revision "
                        "SET revision = revision + 1 WHERE tenant_id = :t"
                    ),
                    {"t": ident.context.tenant_id},
                )
                connection.commit()
                answers.append("committed")

        def admitted(name: str, arguments: dict[str, str]) -> bool:
            async def scenario() -> bool:
                async with Client(server, raise_exceptions=True) as client:
                    answer = await client.call_tool(
                        name,
                        {
                            **arguments,
                            "dataset_version": context.scope.dataset_version,
                        },
                    )
                return not answer.is_error

            return asyncio.run(scenario())

        try:
            holder = threading.Thread(target=hold_the_revision_row, daemon=True)
            holder.start()
            assert holding.wait(timeout=30.0), "the row lock was never taken"
            try:
                for name, arguments in calls:
                    assert admitted(name, arguments), (
                        f"{name} waited for a lock its role must not take"
                    )
                # Sampled while the holder is still inside its transaction: the revision row
                # really is locked by another backend, and none of the locks this database
                # shows belongs to the stdio reader that was just admitted.
                held = fixture.held_review_locks()
                backends = fixture.app_backends()
                contenders = backend_pids(backends, "fix01a_contender")
                stdio_pids = backend_pids(backends, "fix01a_stdio")
                assert any("review_authorization_revision" in item for item in held), (
                    held
                )
                assert contenders and stdio_pids, backends
                assert backend_pids(held) <= contenders, held
                assert not backend_pids(held) & stdio_pids, held
            finally:
                release.set()
                holder.join(timeout=30.0)
            assert answers == ["committed"], answers
            for name, arguments in calls:
                assert not admitted(name, arguments), (
                    f"{name} kept reading after a committed revision change"
                )
                with pytest.raises(SecurityError) as refusal:
                    getattr(context.reader, name)(
                        **arguments, dataset_version=context.scope.dataset_version
                    )
                assert refusal.value.code == "runtime_scope_not_authorized", name
        finally:
            release.set()
            engine.dispose()
            context.close()
