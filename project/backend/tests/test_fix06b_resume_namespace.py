"""FIX-06B V-B-01 and V-B-02: the resume idempotency namespace under real contention.

R-02 showed that one resume key was only unique inside one Run as far as the code was
concerned: the pre-lock lookup, the conditional re-read after the Run lock and the plain
``INSERT`` were three separate decisions, so two requests that used the same key for two
different Runs could both read "no operation".  The second one then met the primary key as
a raw ``IntegrityError`` and the API answered ``503 security_dependency_unavailable`` — a
dependency failure that never happened — instead of the contract's ``409``.  The same
window also misreported a plain same-Run duplicate, because a resume does not move the Run
revision the conditional re-read was keyed on.

Every scenario here runs independent service instances (own engine, own application) over
one database, parks one request inside a named boundary of its resume transaction, lets the
other commit, and then reads the durable facts back.  The namespace is the subject's —
``(tenant, subject, operation family, key digest)`` — and the request digest decides replay
versus conflict.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from procurement_review.contracts import ReviewLifecycleStatus
from procurement_review.fault_hooks import BarrierSpec, FaultController
from procurement_review.run_idempotency import idempotency_key_digest

from fix01a_support import barrier_dir, parked_fault, release_barrier, wait_barrier
from test_fix01b_record_integrity import (  # noqa: F401 - re-exported harness
    API,
    API_VERSION_V2,
    ApiNode,
    api_node,
    audit_draft,
    create_run,
    database,
    headers,
    interrupt,
)

#: The boundaries a resume transaction can be parked at for this work item.
BEFORE_RUN_LOCK = "during_resume_commit:before_run_lock"
BEFORE_CLAIM = "during_resume_commit:before_claim"

#: The barrier that holds a seeded Run in ``processing`` until it is interrupted on purpose.
ATTEMPT_STARTED = "attempt_started:policy"

PENDING = ReviewLifecycleStatus.PENDING_CONFIRMATION.value
INTERRUPTED = ReviewLifecycleStatus.INTERRUPTED.value


def contest_fault(directory: Path, *once_names: str) -> FaultController:
    """Hold seeded executions, and park the first arrival at each resume boundary."""

    barriers = {
        ATTEMPT_STARTED: BarrierSpec(action="wait", timeout_seconds=90.0),
        **{
            name: BarrierSpec(action="wait", timeout_seconds=90.0, once=True)
            for name in once_names
        },
    }
    return FaultController(enabled=True, directory=directory, barriers=barriers)


def resume_once(
    node: ApiNode, ident: Any, run_id: str, *, key: str, expected_version: int
) -> tuple[int, dict[str, Any]]:
    response = node.client.post(
        f"{API}/reviews/{run_id}/resume",
        headers=headers(ident.token, key),
        json={"api_version": API_VERSION_V2, "expected_version": expected_version},
    )
    try:
        body: Any = response.json()
    except ValueError:
        body = {}
    return response.status_code, body if isinstance(body, dict) else {}


def error_code(body: dict[str, Any]) -> str | None:
    error = body.get("error")
    return error.get("code") if isinstance(error, dict) else None


def namespace_rows(fixture: Any) -> list[tuple[str, str, str, str, str, str]]:
    """Every committed operation as ``(family, run_id, outcome, digest, tenant, subject)``."""

    with fixture.driver.transaction() as connection:
        rows = connection.execute(
            text(
                "SELECT operation_family, run_id, outcome, key_digest, tenant_id, "
                "subject_id FROM review_idempotency ORDER BY operation_family, run_id"
            )
        ).all()
    return [
        (
            str(row.operation_family),
            str(row.run_id),
            str(row.outcome),
            row.key_digest,
            str(row.tenant_id),
            str(row.subject_id),
        )
        for row in rows
    ]


def receipt_rows(fixture: Any) -> list[tuple[str, str]]:
    """Every committed local receipt as ``(run_id, decision)``."""

    with fixture.driver.transaction() as connection:
        rows = connection.execute(
            text("SELECT run_id, decision FROM review_receipt ORDER BY run_id")
        ).all()
    return [(str(row.run_id), str(row.decision)) for row in rows]


def namespace_footprint(fixture: Any) -> dict[str, int]:
    """How many namespace, generation, audit and trace rows exist in total."""

    tables = (
        "review_idempotency",
        "review_run_generation",
        "review_audit_event",
        "review_trace_event",
    )
    with fixture.driver.transaction() as connection:
        return {
            table: int(
                connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            )
            for table in tables
        }


@contextmanager
def contending_nodes(
    fixture: Any, directory: Path, *once_names: str
) -> Iterator[tuple[ApiNode, ApiNode]]:
    """A parked node and an independent node over the same database, always disposed."""

    first = api_node(
        fixture, "fix06b_first", fault=contest_fault(directory, *once_names)
    )
    second = api_node(fixture, "fix06b_second")
    try:
        yield first, second
    finally:
        second.close()
        first.close()


def interrupted_run(fixture: Any, ident: Any, *, key: str, name: str) -> str:
    """Create one real Run and stop it deterministically, on a node of its own.

    The Run is held inside its first branch booking, so ``mark_interrupted`` sees a Run that
    is genuinely ``processing`` instead of racing the Fake execution to the finish line.
    """

    directory = barrier_dir(f"fix06b-seed-{name}")
    node = api_node(
        fixture, f"fix06b_seed_{name}", fault=parked_fault(directory, ATTEMPT_STARTED)
    )
    try:
        run_id = create_run(node, ident, key=key)
        wait_barrier(directory, ATTEMPT_STARTED)
        assert interrupt(node.repository, run_id) is True
        release_barrier(directory, ATTEMPT_STARTED)
        body = wait_status(fixture, run_id, INTERRUPTED)
        assert body["status"] == INTERRUPTED, body
        return run_id
    finally:
        release_barrier(directory, ATTEMPT_STARTED)
        node.close()


def wait_status(
    fixture: Any, run_id: str, status: str, *, timeout: float = 90.0
) -> dict:
    deadline = time.monotonic() + timeout
    body = fixture.ledger(run_id)
    while time.monotonic() < deadline:
        if body["status"] == status:
            return body
        time.sleep(0.05)
        body = fixture.ledger(run_id)
    return body


def start_and_park(
    node: ApiNode,
    ident: Any,
    run_id: str,
    *,
    key: str,
    expected_version: int,
    answers: dict[str, Any],
) -> threading.Thread:
    """Start one resume on its own thread; the caller waits for the product barrier."""

    def contest() -> None:
        answers["parked"] = resume_once(
            node, ident, run_id, key=key, expected_version=expected_version
        )

    worker = threading.Thread(target=contest, daemon=True)
    worker.start()
    return worker


def test_cross_run_same_key_race_answers_a_contract_result() -> None:
    """R-02/V-B-01: one key, two Runs, one loser — and the loser is a 409, not a 503."""

    directory = barrier_dir("fix06b-before-lock")
    with database("fix06b_cross_run") as fixture:
        ident = fixture.identity()
        run_a = interrupted_run(fixture, ident, key="seed-a", name="a")
        run_b = interrupted_run(fixture, ident, key="seed-b", name="b")
        version_a = int(fixture.ledger(run_a)["revision"])
        version_b = int(fixture.ledger(run_b)["revision"])
        with contending_nodes(fixture, directory, BEFORE_RUN_LOCK) as (first, second):
            before = namespace_footprint(fixture)
            ledger_before = fixture.ledger(run_a)
            answers: dict[str, Any] = {}
            worker = start_and_park(
                first,
                ident,
                run_a,
                key="shared-key",
                expected_version=version_a,
                answers=answers,
            )
            # The parked request has already read "no operation" and stands before its own
            # Run lock; the second request uses the same key for a different Run.
            wait_barrier(directory, BEFORE_RUN_LOCK, timeout=60.0)
            answers["other"] = resume_once(
                second, ident, run_b, key="shared-key", expected_version=version_b
            )
            release_barrier(directory, BEFORE_RUN_LOCK)
            worker.join(timeout=90)
            assert not worker.is_alive(), "the parked resume never answered"
            after = namespace_footprint(fixture)
            ledger_a = fixture.ledger(run_a)
            rows = [
                row
                for row in namespace_rows(fixture)
                if row[3] == idempotency_key_digest("shared-key")
            ]

        assert answers["other"][0] in {202, 200}, answers["other"]
        assert answers["parked"][0] == 409, (
            "the losing request did not answer a contract conflict: "
            f"{answers['parked']}"
        )
        assert error_code(answers["parked"][1]) == "idempotency_conflict", answers[
            "parked"
        ]
        # One namespace operation, and the loser left no budget, generation or dispatch.
        assert len(rows) == 1 and rows[0][1] == run_b, rows
        # The losing transaction wrote nothing at all: its whole ledger is unchanged.
        assert ledger_a == ledger_before, (ledger_a, ledger_before)
        assert after["review_run_generation"] == before["review_run_generation"] + 1, (
            before,
            after,
        )


def test_same_run_duplicate_is_a_replay_not_a_dependency_failure() -> None:
    """V-B-01: one Run, one key, two contenders — the duplicate replays or conflicts."""

    directory = barrier_dir("fix06b-same-run")
    with database("fix06b_same_run") as fixture:
        ident = fixture.identity()
        run_id = interrupted_run(fixture, ident, key="seed", name="same")
        version = int(fixture.ledger(run_id)["revision"])
        with contending_nodes(fixture, directory, BEFORE_RUN_LOCK) as (first, second):
            answers: dict[str, Any] = {}
            worker = start_and_park(
                first,
                ident,
                run_id,
                key="same-run",
                expected_version=version,
                answers=answers,
            )
            wait_barrier(directory, BEFORE_RUN_LOCK, timeout=60.0)
            answers["other"] = resume_once(
                second, ident, run_id, key="same-run", expected_version=version
            )
            release_barrier(directory, BEFORE_RUN_LOCK)
            worker.join(timeout=90)
            assert not worker.is_alive(), "the parked resume never answered"
            body = wait_status(fixture, run_id, PENDING)
            rows = namespace_rows(fixture)

        codes = sorted({answers["parked"][0], answers["other"][0]})
        assert codes in ([202, 200], [200, 202], [202, 409]), answers
        assert 503 not in codes, answers
        assert len([row for row in rows if row[0] == "resume"]) == 1, rows
        assert body["status"] == PENDING, body
        assert body["resume_count"] == 1, body
        assert body["summaries"] == 1, body


def test_race_inside_the_recheck_window_is_still_a_contract_result() -> None:
    """V-B-01: a commit between the re-read and the claim must not become a 503.

    The parked transaction has already re-read "no operation" under its Run lock and stands
    immediately before it claims the key.  The other one commits that key for a different
    Run in exactly that window, so only the unique constraint can arbitrate — a re-read that
    merely moved earlier would still lose here.
    """

    directory = barrier_dir("fix06b-before-claim")
    with database("fix06b_claim_window") as fixture:
        ident = fixture.identity()
        run_a = interrupted_run(fixture, ident, key="seed-a", name="wa")
        run_b = interrupted_run(fixture, ident, key="seed-b", name="wb")
        version_a = int(fixture.ledger(run_a)["revision"])
        version_b = int(fixture.ledger(run_b)["revision"])
        with contending_nodes(fixture, directory, BEFORE_CLAIM) as (first, second):
            answers: dict[str, Any] = {}
            worker = start_and_park(
                first,
                ident,
                run_a,
                key="window-key",
                expected_version=version_a,
                answers=answers,
            )
            wait_barrier(directory, BEFORE_CLAIM, timeout=60.0)
            answers["other"] = resume_once(
                second, ident, run_b, key="window-key", expected_version=version_b
            )
            assert answers["other"][0] in {202, 200}, answers["other"]
            release_barrier(directory, BEFORE_CLAIM)
            worker.join(timeout=90)
            assert not worker.is_alive(), "the parked resume never answered"
            ledger_a = fixture.ledger(run_a)
            rows = [
                row
                for row in namespace_rows(fixture)
                if row[3] == idempotency_key_digest("window-key")
            ]

        assert answers["parked"][0] == 409, answers["parked"]
        assert error_code(answers["parked"][1]) == "idempotency_conflict", answers[
            "parked"
        ]
        assert len(rows) == 1 and rows[0][1] == run_b, rows
        assert ledger_a["resume_count"] == 0, ledger_a
        assert ledger_a["status"] == INTERRUPTED, ledger_a


@pytest.mark.parametrize("same_body", [True, False])
def test_replay_and_conflict_on_one_run_follow_the_request_digest(
    same_body: bool,
) -> None:
    """V-B-01/AC-02: same key with the same body replays; a different body conflicts."""

    with database(f"fix06b_digest_{int(same_body)}") as fixture:
        ident = fixture.identity()
        run_id = interrupted_run(fixture, ident, key="seed", name=f"d{int(same_body)}")
        version = int(fixture.ledger(run_id)["revision"])
        node = api_node(fixture, "fix06b_digest")
        try:
            first = resume_once(
                node, ident, run_id, key="digest-key", expected_version=version
            )
            assert first[0] == 202, first
            body = wait_status(fixture, run_id, PENDING)
            assert body["status"] == PENDING, body
            before = namespace_footprint(fixture)
            replay_version = version if same_body else version + 1
            second = resume_once(
                node,
                ident,
                run_id,
                key="digest-key",
                expected_version=replay_version,
            )
            after = namespace_footprint(fixture)
            ledger = fixture.ledger(run_id)
        finally:
            node.close()

        if same_body:
            assert second[0] == 200, second
            assert second[1]["recovery"]["resume_count"] == 1, second
            assert ledger["resume_count"] == 1, ledger
        else:
            assert second[0] == 409, second
            assert error_code(second[1]) == "idempotency_conflict", second
        # A replay and a conflict never write: no namespace, generation, audit or trace row.
        assert after == before, (after, before)


def test_a_permanent_rejection_claims_the_same_namespace() -> None:
    """V-B-02: a failed outcome claims the key, and a cross-Run loser still writes nothing.

    The parked request reaches the Run boundary having already read "no operation" for the
    shared key.  The other Run is then permanently rejected under that same key, so the
    parked one has to answer the contract conflict instead of a dependency failure — and the
    losing Run must keep its own state instead of inheriting somebody else's rejection.
    """

    directory = barrier_dir("fix06b-rejected")
    with database("fix06b_rejected") as fixture:
        ident = fixture.identity()
        run_a = interrupted_run(fixture, ident, key="seed-a", name="ra")
        run_b = interrupted_run(fixture, ident, key="seed-b", name="rb")
        version_a = int(fixture.ledger(run_a)["revision"])
        version_b = int(fixture.ledger(run_b)["revision"])
        first = api_node(
            fixture, "fix06b_first", fault=contest_fault(directory, BEFORE_RUN_LOCK)
        )
        second = api_node(fixture, "fix06b_second", resume_limit=0)
        answers: dict[str, Any] = {}
        try:
            worker = start_and_park(
                first,
                ident,
                run_a,
                key="reject-key",
                expected_version=version_a,
                answers=answers,
            )
            wait_barrier(directory, BEFORE_RUN_LOCK, timeout=60.0)
            answers["other"] = resume_once(
                second, ident, run_b, key="reject-key", expected_version=version_b
            )
            assert answers["other"][0] == 409, answers["other"]
            assert error_code(answers["other"][1]) == "review_not_recoverable", answers[
                "other"
            ]
            release_barrier(directory, BEFORE_RUN_LOCK)
            worker.join(timeout=90)
            assert not worker.is_alive(), "the parked resume never answered"
        finally:
            release_barrier(directory, BEFORE_RUN_LOCK)
            second.close()
            first.close()
        ledger_a = fixture.ledger(run_a)
        ledger_b = fixture.ledger(run_b)
        digest = idempotency_key_digest("reject-key")
        failed = [
            row
            for row in namespace_rows(fixture)
            if row[3] == digest and row[2] == "failed"
        ]

        assert answers["parked"][0] == 409, answers["parked"]
        assert error_code(answers["parked"][1]) == "idempotency_conflict", answers[
            "parked"
        ]
        assert ledger_b["status"] == ReviewLifecycleStatus.FAILED.value, ledger_b
        assert len(failed) == 1 and failed[0][1] == run_b, failed
        # The losing Run keeps its own state: it was not rejected by somebody else's key.
        assert ledger_a["status"] == INTERRUPTED, ledger_a
        assert ledger_a["resume_count"] == 0, ledger_a
        # Its own interruption reason, not the other Run's permanent rejection.
        assert ledger_a["error_code"] == "storage_unavailable", ledger_a


def test_barrier_synchronised_contenders_never_answer_503() -> None:
    """V-B-01: three Runs, one key, one barrier — exactly one 202 and never a 503."""

    with database("fix06b_crowd") as fixture:
        ident = fixture.identity()
        run_ids = [
            interrupted_run(fixture, ident, key=f"seed-{index}", name=f"c{index}")
            for index in range(3)
        ]
        versions = [int(fixture.ledger(run_id)["revision"]) for run_id in run_ids]
        nodes = (
            api_node(fixture, "fix06b_crowd_a"),
            api_node(fixture, "fix06b_crowd_b"),
        )
        try:
            answers: list[tuple[int, dict[str, Any]]] = []
            lock = threading.Lock()
            start = threading.Barrier(len(run_ids))

            def contest(index: int) -> None:
                node = nodes[index % len(nodes)]
                start.wait(timeout=30)
                answer = resume_once(
                    node,
                    ident,
                    run_ids[index],
                    key="crowd-key",
                    expected_version=versions[index],
                )
                with lock:
                    answers.append(answer)

            workers = [
                threading.Thread(target=contest, args=(index,), daemon=True)
                for index in range(len(run_ids))
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=120)
                assert not worker.is_alive(), "a contending resume never answered"
            digest = idempotency_key_digest("crowd-key")
            rows = [row for row in namespace_rows(fixture) if row[3] == digest]
        finally:
            for node in nodes:
                node.close()

        codes = sorted(answer[0] for answer in answers)
        assert codes.count(202) == 1, answers
        assert set(codes) <= {202, 409}, answers
        assert 503 not in codes, answers
        assert len(rows) == 1, rows


def test_the_namespace_separates_tenants_subjects_and_keys() -> None:
    """V-B-01: different subjects, tenants and keys never share one operation."""

    with database("fix06b_namespace") as fixture:
        demo = fixture.identity()
        alpha = fixture.identity(
            subject_id="subject-alpha-operator",
            case_id="CASE-ALPHA-001",
            tenant_id="tenant-alpha",
        )
        demo_run = interrupted_run(fixture, ident=demo, key="seed-demo", name="demo")
        alpha_run = interrupted_run(
            fixture, ident=alpha, key="seed-alpha", name="alpha"
        )
        demo_version = int(fixture.ledger(demo_run)["revision"])
        alpha_version = int(fixture.ledger(alpha_run)["revision"])
        node = api_node(fixture, "fix06b_namespaces")
        try:
            # The same raw key under two subjects and two tenants.
            assert (
                resume_once(
                    node, demo, demo_run, key="shared", expected_version=demo_version
                )[0]
                == 202
            )
            assert (
                resume_once(
                    node, alpha, alpha_run, key="shared", expected_version=alpha_version
                )[0]
                == 202
            )
            assert wait_status(fixture, demo_run, PENDING)["status"] == PENDING
            assert wait_status(fixture, alpha_run, PENDING)["status"] in {
                PENDING,
                ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
            }
            other = interrupted_run(fixture, ident=demo, key="seed-other", name="other")
            other_version = int(fixture.ledger(other)["revision"])
            assert (
                resume_once(
                    node, demo, other, key="another-key", expected_version=other_version
                )[0]
                == 202
            )
            rows = [row for row in namespace_rows(fixture) if row[0] == "resume"]
        finally:
            node.close()

        assert len(rows) == 3, rows
        # The same raw key under two subjects and tenants is two namespaces, and the third
        # operation adds a second key: three operations, three namespaces, two key digests.
        assert len({(row[4], row[5], row[0], row[3]) for row in rows}) == 3, rows
        assert len({row[3] for row in rows}) == 2, rows


def test_one_raw_key_is_two_operations_across_families() -> None:
    """V-B-01: the create family and the resume family never collide on one raw key."""

    with database("fix06b_families") as fixture:
        ident = fixture.identity()
        node = api_node(fixture, "fix06b_families")
        try:
            run_id = create_run(node, ident, key="family-key")
            assert interrupt(node.repository, run_id) is True
            assert wait_status(fixture, run_id, INTERRUPTED)["status"] == INTERRUPTED
            version = int(fixture.ledger(run_id)["revision"])
            answer = resume_once(
                node, ident, run_id, key="family-key", expected_version=version
            )
            rows = namespace_rows(fixture)
        finally:
            node.close()

        assert answer[0] == 202, answer
        digest = idempotency_key_digest("family-key")
        families = sorted(row[0] for row in rows if row[3] == digest)
        assert families == ["create", "resume"], rows


def test_commit_then_lost_answer_is_located_by_the_same_key() -> None:
    """V-B-02: a committed resume whose answer was lost is found by the same key alone."""

    with database("fix06b_lost") as fixture:
        ident = fixture.identity()
        run_id = interrupted_run(fixture, ident, key="seed", name="lost")
        version = int(fixture.ledger(run_id)["revision"])
        node = api_node(fixture, "fix06b_lost")
        try:
            assert (
                resume_once(
                    node, ident, run_id, key="lost-key", expected_version=version
                )[0]
                == 202
            )
            assert wait_status(fixture, run_id, PENDING)["status"] == PENDING
            before = namespace_footprint(fixture)
            recovered, body = resume_once(
                node, ident, run_id, key="lost-key", expected_version=version
            )
            after = namespace_footprint(fixture)
        finally:
            node.close()

        assert recovered == 200, body
        assert body["recovery"]["resume_count"] == 1, body
        assert after == before, (after, before)


def test_revocation_is_still_decided_before_the_namespace_is_consulted() -> None:
    """V-B-02: a revoked subject learns nothing from the namespace and writes none of it."""

    with database("fix06b_authority") as fixture:
        ident = fixture.identity()
        run_id = interrupted_run(fixture, ident, key="seed", name="auth")
        version = int(fixture.ledger(run_id)["revision"])
        node = api_node(fixture, "fix06b_authority")
        try:
            assert (
                resume_once(
                    node, ident, run_id, key="auth-key", expected_version=version
                )[0]
                == 202
            )
            assert wait_status(fixture, run_id, PENDING)["status"] == PENDING
            before = namespace_footprint(fixture)
            assert fixture.revoke_credential(ident.credential_id)
            forbidden = resume_once(
                node, ident, run_id, key="auth-key", expected_version=version
            )
            after = namespace_footprint(fixture)
            ledger = fixture.ledger(run_id)
        finally:
            node.close()

        assert forbidden[0] in {401, 403}, forbidden
        assert after == before, (after, before)
        assert ledger["resume_count"] == 1, ledger


def test_a_conflict_is_not_retried_and_not_re_dispatched() -> None:
    """V-B-01: the loser answers inside one transaction, with no retry loop or dispatch."""

    directory = barrier_dir("fix06b-no-retry")
    with database("fix06b_no_retry") as fixture:
        ident = fixture.identity()
        run_a = interrupted_run(fixture, ident, key="seed-a", name="na")
        run_b = interrupted_run(fixture, ident, key="seed-b", name="nb")
        version_a = int(fixture.ledger(run_a)["revision"])
        fault = contest_fault(directory, BEFORE_RUN_LOCK)
        first = api_node(fixture, "fix06b_retry_first", fault=fault)
        second = api_node(fixture, "fix06b_retry_second")
        try:
            assert (
                resume_once(
                    second,
                    ident,
                    run_b,
                    key="retry-key",
                    expected_version=int(fixture.ledger(run_b)["revision"]),
                )[0]
                == 202
            )
            wait_status(fixture, run_b, PENDING)
            before = namespace_footprint(fixture)
            answer = resume_once(
                first, ident, run_a, key="retry-key", expected_version=version_a
            )
            after = namespace_footprint(fixture)
            reached = list(fault.reached)
        finally:
            second.close()
            first.close()

        assert answer[0] == 409, answer
        assert after == before, (after, before)
        assert reached.count(BEFORE_RUN_LOCK) == 0, reached


def test_a_human_action_key_is_one_namespace_operation_across_runs() -> None:
    """V-B-01: the action family claims its key the same way; a cross-Run loser is a 409.

    Two Runs are decided concurrently with one key.  The actions are independent (each Run
    has its own lock), so only the namespace can arbitrate; the loser must answer the
    contract conflict and write nothing at all — neither a receipt nor a status change.
    """

    with database("fix06b_action") as fixture:
        ident = fixture.identity()
        node_a = api_node(fixture, "fix06b_action_a")
        node_b = api_node(fixture, "fix06b_action_b")
        try:
            run_a = create_run(node_a, ident, key="action-seed-a")
            run_b = create_run(node_b, ident, key="action-seed-b")
            assert wait_status(fixture, run_a, PENDING)["status"] == PENDING
            assert wait_status(fixture, run_b, PENDING)["status"] == PENDING
            views = {
                run_id: node.client.get(
                    f"{API}/reviews/{run_id}", headers=headers(ident.token)
                ).json()
                for node, run_id in ((node_a, run_a), (node_b, run_b))
            }
            answers: list[tuple[int, dict[str, Any]]] = []
            lock = threading.Lock()
            start = threading.Barrier(2)

            def confirm(node: ApiNode, run_id: str) -> None:
                start.wait(timeout=30)
                response = node.client.post(
                    f"{API}/reviews/{run_id}/confirm",
                    headers=headers(ident.token, "action-key"),
                    json={
                        "api_version": API_VERSION_V2,
                        "expected_version": int(views[run_id]["version"]),
                        "expected_summary_checksum": views[run_id]["summary_checksum"],
                    },
                )
                try:
                    body: Any = response.json()
                except ValueError:
                    body = {}
                with lock:
                    answers.append((response.status_code, body))

            workers = [
                threading.Thread(target=confirm, args=args, daemon=True)
                for args in ((node_a, run_a), (node_b, run_b))
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=120)
                assert not worker.is_alive(), "a contending action never answered"
            digest = idempotency_key_digest("action-key")
            rows = [row for row in namespace_rows(fixture) if row[3] == digest]
            ledgers = {run_id: fixture.ledger(run_id) for run_id in (run_a, run_b)}
            receipts = receipt_rows(fixture)
        finally:
            node_b.close()
            node_a.close()

        codes = sorted(answer[0] for answer in answers)
        assert 503 not in codes, answers
        assert codes.count(200) == 1, answers
        assert set(codes) <= {200, 409}, answers
        loser = next(answer for answer in answers if answer[0] == 409)
        assert error_code(loser[1]) == "idempotency_conflict", loser
        assert len(rows) == 1, rows
        decided = [
            run_id
            for run_id, ledger in ledgers.items()
            if ledger["status"] == ReviewLifecycleStatus.CONFIRMED.value
        ]
        assert len(decided) == 1, ledgers
        assert rows[0][1] == decided[0], (rows, decided)
        assert len(receipts) == 1 and receipts[0][0] == decided[0], receipts


def test_every_write_entry_point_claims_through_the_one_implementation() -> None:
    """AC-07, structural check: no write path may register a namespace operation itself."""

    import inspect

    from procurement_review import (
        review_repository,
        run_idempotency,
        run_idempotency_namespace,
        run_recovery,
    )

    claim = getattr(run_idempotency, "claim_operation", None)
    assert claim is not None, "the shared conflict-safe claim helper is missing"
    # FIX-06D moved the key itself (lookup + claim) into its own guarded-size module;
    # `run_idempotency` re-exports the one implementation, so this stays a check of the
    # single claim statement rather than of one import site.
    assert claim is run_idempotency_namespace.claim_operation, (
        "run_idempotency must re-export the one conflict-safe claim implementation"
    )
    assert "ON CONFLICT" in inspect.getsource(run_idempotency_namespace), (
        "the shared claim must let the unique constraint arbitrate"
    )
    for module in (review_repository, run_recovery):
        source = inspect.getsource(module)
        assert "INSERT INTO review_idempotency" not in source, (
            f"{module.__name__} still writes the namespace with its own INSERT"
        )
        assert "claim_operation" in source, (
            f"{module.__name__} must claim through the shared helper"
        )
    assert "IntegrityError" not in inspect.getsource(run_recovery), (
        "the resume transaction must not swallow constraint errors"
    )
