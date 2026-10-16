"""FIX-01A V-A-03 and V-A-04: one atomic resume and its idempotency protocol.

Crash windows are produced the only way they are real: a separate operating-system process
runs the API, a file barrier reports the exact commit boundary it reached, and the plan tells
that process to die there.  Each window therefore leaves a genuine PostgreSQL rollback or a
genuine committed transaction, and the surviving process reads the facts back over HTTP and
over its own database connection.

Concurrency is produced the same way: one API process is parked inside its resume
transaction, so a second independent process waits on the real row lock instead of on a
thread that shares the first process' loop.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import text

from procurement_review.authorization import parse_token
from procurement_review.review_store import ReviewStore
from procurement_review.security_recovery_check import (
    FAULT_EXIT_CODE,
    ApiProcess,
    DatabaseFixture,
    _free_port,
    _release,
    _wait_for_barrier,
)

from v1_03_support import require_admin_dsn

LOW_CASE = "CASE-LOW-001"
FAST_TIMING = {
    "REVIEW_LEASE_SECONDS": "3",
    "REVIEW_HEARTBEAT_SECONDS": "0.5",
    "REVIEW_SCAN_INTERVAL_SECONDS": "0.2",
}

#: The stages inside the resume transaction a crash window must cover (V-A-03).
RESUME_STAGES = (
    "idempotency",
    "resume_count",
    "generation",
    "before_audit",
    "after_audit",
    "before_commit",
)

#: The windows after the resume transaction decided the outcome.
AFTER_COMMIT_STAGES = ("after_resume_commit", "after_resume_dispatch")

PENDING = ("pending_confirmation", "pending_human_review")


@dataclass(slots=True)
class Peer:
    """One disposable API process plus the HTTP and database readings around it."""

    fixture: DatabaseFixture
    root: Path
    process: ApiProcess | None = None
    session: httpx.Client | None = None
    token: str = ""

    # ------------------------------------------------------------- processes

    def start(self, plan: dict[str, Any] | None = None, name: str = "api") -> None:
        self.stop()
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        for item in directory.glob("*"):
            item.unlink()
        self.process = ApiProcess(
            fixture=self.fixture,
            fault_dir=directory,
            port=_free_port(),
            plan=plan,
        )
        self.process.start()
        self.process.wait_ready()
        self.token = self.fixture.issue_token()
        self.session = httpx.Client(
            base_url=self.process.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30.0,
        )

    def stop(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.process is not None:
            self.process.stop()
            self.process = None

    def died_at_barrier(self, timeout: float = 90.0) -> None:
        assert self.process is not None
        code = self.process.wait_for_exit(timeout=timeout)
        assert code == FAULT_EXIT_CODE, f"expected the injected exit, got {code}"

    # --------------------------------------------------------------- requests

    def call(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        assert self.session is not None
        headers = {"Idempotency-Key": key} if key else {}
        try:
            response = self.session.request(method, path, json=json, headers=headers)
        except httpx.HTTPError as error:  # a peer that died is an expected answer
            return -1, {"transport": type(error).__name__}
        body: Any = None
        try:
            body = response.json()
        except ValueError:
            body = {}
        return response.status_code, body if isinstance(body, dict) else {}

    def resume(
        self, run_id: str, *, key: str, expected_version: int
    ) -> tuple[int, dict[str, Any]]:
        return self.call(
            "POST",
            f"/api/v2/reviews/{run_id}/resume",
            json={"api_version": "review-api.v2", "expected_version": expected_version},
            key=key,
        )

    def cancel(
        self, run_id: str, *, key: str, expected_version: int
    ) -> tuple[int, Any]:
        return self.call(
            "POST",
            f"/api/v2/reviews/{run_id}/cancel",
            json={"api_version": "review-api.v2", "expected_version": expected_version},
            key=key,
        )

    def create(self, key: str) -> tuple[int, dict[str, Any]]:
        return self.call(
            "POST",
            "/api/v2/reviews",
            json={
                "api_version": "review-api.v2",
                "case_id": LOW_CASE,
                "case_version": f"{LOW_CASE.lower()}.v1",
            },
            key=key,
        )

    def read(self, run_id: str) -> tuple[int, dict[str, Any]]:
        return self.call("GET", f"/api/v2/reviews/{run_id}")

    def version(self, run_id: str) -> int:
        code, body = self.read(run_id)
        assert code == 200, (code, body)
        return int(body["version"])

    def wait_status(self, run_id: str, *statuses: str, timeout: float = 90.0) -> dict:
        deadline = time.monotonic() + timeout
        body: dict[str, Any] = {}
        while time.monotonic() < deadline:
            code, body = self.read(run_id)
            if code == 200 and body.get("status") in statuses:
                return body
            time.sleep(0.1)
        raise AssertionError(f"run {run_id} stayed {body.get('status')}: {body}")

    # ----------------------------------------------------------------- ledger

    def ledger(self, run_id: str) -> dict[str, Any]:
        """Read every table the resume protocol is allowed to touch."""

        store = ReviewStore.from_url(self.fixture.url)
        try:
            with store.transaction() as connection:
                run = connection.execute(
                    text(
                        "SELECT status, revision, resume_count, error_code "
                        "FROM review_run WHERE run_id = :r"
                    ),
                    {"r": run_id},
                ).one()
                tables = {
                    "generations": "SELECT coalesce(array_agg(generation || '=' || "
                    "status ORDER BY generation),'{}') FROM review_run_generation "
                    "WHERE run_id = :r",
                    "idempotency": "SELECT coalesce(array_agg(operation_family || '=' || "
                    "coalesce(outcome,'-') || '=' || coalesce(failure_code,'-') "
                    "ORDER BY operation_family),'{}') FROM review_idempotency "
                    "WHERE run_id = :r",
                    "audit": "SELECT coalesce(array_agg(event_class || ':' || "
                    "coalesce(new_status,'-') || ':' || "
                    "coalesce(event_payload #>> '{detail_code}','-') ORDER BY sequence),"
                    "'{}') FROM review_audit_event WHERE run_id = :r",
                    "trace": "SELECT coalesce(array_agg(event_ordinal || '/' || step || "
                    "':' || event_type ORDER BY event_ordinal),'{}') "
                    "FROM review_trace_event WHERE run_id = :r",
                    "attempts": "SELECT coalesce(array_agg(generation || '/' || worker "
                    "|| '/' || attempt_no ORDER BY generation, worker, attempt_no),'{}') "
                    "FROM review_worker_attempt WHERE run_id = :r",
                    "checkpoints": "SELECT coalesce(array_agg(worker || '@' || generation "
                    "ORDER BY worker),'{}') FROM review_branch_checkpoint WHERE run_id = :r",
                    "summaries": "SELECT count(*) FROM review_run_summary "
                    "WHERE run_id = :r",
                    "receipts": "SELECT count(*) FROM review_receipt WHERE run_id = :r",
                }
                counts = {
                    name: connection.execute(text(sql), {"r": run_id}).scalar_one()
                    for name, sql in tables.items()
                }
        finally:
            store.close()
        return {
            "status": run.status,
            "revision": int(run.revision),
            "resume_count": int(run.resume_count),
            "error_code": run.error_code,
            **{name: _as_list(value) for name, value in counts.items()},
        }


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (int, float)):
        return [str(int(value))]
    return [str(item) for item in value]


#: Every backend of one scenario database except the observing connection itself.
BACKEND_INVENTORY = text(
    """
    SELECT a.pid, a.state, a.wait_event_type, a.wait_event, a.query,
           pg_blocking_pids(a.pid) AS blocked_by
    FROM pg_stat_activity a
    WHERE a.datname = current_database() AND a.pid <> pg_backend_pid()
    ORDER BY a.pid
    """
)


def run_lock_waiters(backends: Sequence[Any]) -> list[Any]:
    """Backends the server itself reports as waiting for the lock on a Run row.

    Either mode counts: the resume path takes the Run row ``FOR UPDATE`` while the request
    that follows it waits on the same row behind that holder.
    """

    return [
        row
        for row in backends
        if list(row.blocked_by)
        and "review_run" in row.query
        and ("FOR UPDATE" in row.query or "FOR SHARE" in row.query)
    ]


def wait_for_run_lock_waiter(database_url: str, timeout: float = 30.0) -> list[Any]:
    """Return the backend inventory once a Run row lock is genuinely being waited for.

    Only the server's own view can tell "the second process reached the lock" apart from
    "the second process is somewhere else", so this reports the scenario as failed instead
    of guessing with a fixed sleep.
    """

    store = ReviewStore.from_url(database_url)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with store.transaction() as connection:
                rows = connection.execute(BACKEND_INVENTORY).all()
            if run_lock_waiters(rows):
                return list(rows)
            time.sleep(0.05)
    finally:
        store.close()
    raise AssertionError("no backend ever waited on the Run row lock")


@contextmanager
def peer_root(root: Path) -> Iterator[Peer]:
    previous = {key: os.environ.get(key) for key in FAST_TIMING}
    os.environ.update(FAST_TIMING)
    os.environ.setdefault("SECURITY_RECOVERY_ADMIN_DSN", require_admin_dsn())
    fixture = DatabaseFixture()
    peer = Peer(fixture=fixture, root=root)
    try:
        yield peer
    finally:
        peer.stop()
        fixture.close()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def to_interrupted(peer: Peer, key: str) -> str:
    """Create a Run, die before its dispatch and let the lease interrupt it."""

    peer.start(plan={"barriers": {"after_create_commit": {"action": "exit"}}}, name=key)
    status, body = peer.create(key)
    peer.died_at_barrier()
    # The create committed and the process died before the answer left, so the only safe
    # move is to send the same key and body again and read the operation back.
    assert status == -1, (status, body)
    peer.stop()
    peer.start(name=f"{key}-after")
    status, located = peer.create(key)
    assert status == 200, (status, located)
    run_id = str(located["run_id"])
    peer.wait_status(run_id, "interrupted")
    return run_id


def with_committed_checkpoints(peer: Peer, key: str) -> str:
    """Reach an interrupted Run whose three branch checkpoints are all committed."""

    peer.start(
        plan={"barriers": {"after_all_branches_committed": {"action": "exit"}}},
        name=key,
    )
    status, body = peer.create(key)
    peer.died_at_barrier()
    assert status == 202, (status, body)
    run_id = str(body["run_id"])
    peer.stop()
    peer.start(name=f"{key}-after")
    peer.wait_status(run_id, "interrupted")
    ledger = peer.ledger(run_id)
    assert len(ledger["checkpoints"]) == 3, ledger
    return run_id


@pytest.mark.parametrize("stage", RESUME_STAGES)
def test_crash_inside_the_resume_transaction_rolls_back_everything(
    tmp_path, stage: str
) -> None:
    """V-A-03: every pre-commit window leaves zero residue, so a retry starts clean."""

    barrier = f"during_resume_commit:{stage}"
    with peer_root(tmp_path / "inside") as peer:
        run_id = to_interrupted(peer, f"inside-{stage}")
        before = peer.ledger(run_id)
        assert before["resume_count"] == 0, before
        assert before["idempotency"] == ["create=applied=-"], before

        version = peer.version(run_id)
        peer.start(plan={"barriers": {barrier: {"action": "exit"}}}, name="die")
        peer.resume(run_id, key=f"resume-{stage}", expected_version=version)
        peer.died_at_barrier()
        peer.stop()

        peer.start(name="after-crash")
        assert peer.ledger(run_id) == before, peer.ledger(run_id)
        assert peer.read(run_id)[1]["status"] == "interrupted"

        # Nothing was committed, so the same key legitimately performs the operation once.
        status, body = peer.resume(
            run_id, key=f"resume-{stage}", expected_version=version
        )
        assert status == 202, (status, body)
        after = peer.ledger(run_id)
        assert after["resume_count"] == 1, after
        assert any(item.startswith("2=") for item in after["generations"]), after
        assert after["idempotency"] == ["create=applied=-", "resume=applied=-"], after
        assert len([item for item in after["audit"] if "recovery:" in item]) == 1, after
        peer.wait_status(run_id, *PENDING)
        assert peer.ledger(run_id)["summaries"] == ["1"]


@pytest.mark.parametrize("stage", AFTER_COMMIT_STAGES)
def test_crash_after_the_resume_commit_is_located_by_replay(
    tmp_path, stage: str
) -> None:
    """V-A-03: a committed resume survives, and a lost response is replayed not guessed."""

    with peer_root(tmp_path / stage) as peer:
        run_id = to_interrupted(peer, f"after-{stage}")
        peer.start(plan={"barriers": {stage: {"action": "exit"}}}, name="die")
        version = peer.version(run_id)
        peer.resume(run_id, key=f"resume-{stage}", expected_version=version)
        peer.died_at_barrier()
        peer.stop()

        peer.start(name="after-crash")
        committed = peer.ledger(run_id)
        # Budget, generation, audit and the idempotency record were written together.
        assert committed["resume_count"] == 1, committed
        assert committed["idempotency"] == ["create=applied=-", "resume=applied=-"], (
            committed
        )
        assert len(committed["generations"]) == 2, committed
        assert committed["summaries"] == ["0"], committed

        if stage == "after_resume_commit":
            # Committed and never dispatched: the lease has to identify it rather than
            # leave a Run hanging in processing.
            peer.wait_status(run_id, "interrupted")
        else:
            peer.wait_status(run_id, "interrupted", *PENDING)

        code, body = peer.resume(
            run_id, key=f"resume-{stage}", expected_version=version
        )
        assert code == 200, (code, body)
        after = peer.ledger(run_id)
        assert after["resume_count"] == 1, after
        assert len(after["generations"]) == 2, after
        recovery = [item for item in after["audit"] if "recovery:" in item]
        assert recovery == [
            item for item in committed["audit"] if "recovery:" in item
        ], after
        assert after["idempotency"] == committed["idempotency"], after
        assert after["receipts"] == committed["receipts"], after


def test_replay_with_the_original_version_is_a_clean_read(tmp_path) -> None:
    """V-A-04/AC-02: the original request may be replayed, and it adds nothing."""

    with peer_root(tmp_path / "replay") as peer:
        run_id = to_interrupted(peer, "replay")
        version = peer.version(run_id)
        peer.start(name="resumer")
        status, body = peer.resume(run_id, key="replay-once", expected_version=version)
        assert status == 202, (status, body)
        peer.wait_status(run_id, *PENDING)
        settled = peer.ledger(run_id)
        assert settled["status"] in PENDING
        assert settled["receipts"] == ["0"], settled

        status, replay = peer.resume(
            run_id, key="replay-once", expected_version=version
        )
        assert status == 200, (status, replay)
        assert replay["run_id"] == run_id
        assert replay["version"] == settled["revision"], replay
        assert peer.ledger(run_id) == settled


def test_same_key_other_bodies_and_wrong_states_answer_409(tmp_path) -> None:
    """V-A-04: same key/different body, stale version and illegal state each answer 409."""

    with peer_root(tmp_path / "conflicts") as peer:
        run_id = to_interrupted(peer, "conflicts")
        version = peer.version(run_id)
        peer.start(name="conflicter")
        status, _body = peer.resume(
            run_id, key="conflict-key", expected_version=version
        )
        assert status == 202
        peer.wait_status(run_id, *PENDING)
        settled = peer.ledger(run_id)

        status, body = peer.resume(
            run_id, key="conflict-key", expected_version=version + 1
        )
        assert status == 409, (status, body)
        assert body["error"]["code"] == "idempotency_conflict", body

        status, body = peer.resume(run_id, key="stale-key", expected_version=version)
        assert status == 409, (status, body)
        assert body["error"]["code"] == "version_conflict", body

        status, body = peer.resume(
            run_id, key="state-key", expected_version=settled["revision"]
        )
        assert status == 409, (status, body)
        assert body["error"]["code"] == "invalid_state_transition", body

        after = peer.ledger(run_id)
        assert after == settled, after


def test_one_resume_key_cannot_be_replayed_onto_another_run(tmp_path) -> None:
    """V-A-04/AF-12: a key is bound to the Run it first served, not re-decided per Run.

    The other conflict cases reuse one Run. This one puts the same key on a second Run that
    is itself resumable, so the only thing that can refuse it is the digest of the operation
    the key already owns — which is what makes it a contract conflict rather than a new
    operation, a version miss or an illegal state.
    """

    with peer_root(tmp_path / "cross-run") as peer:
        first = to_interrupted(peer, "cross-first")
        second = to_interrupted(peer, "cross-second")
        first_version = peer.version(first)
        second_version = peer.version(second)
        peer.start(name="cross-runner")
        status, body = peer.resume(
            first, key="shared-key", expected_version=first_version
        )
        assert status == 202, (status, body)
        peer.wait_status(first, *PENDING)
        settled_first = peer.ledger(first)
        settled_second = peer.ledger(second)
        assert settled_first["status"] in PENDING, settled_first
        assert settled_second["status"] == "interrupted", settled_second
        assert len(settled_first["generations"]) == 2, settled_first
        assert len(settled_second["generations"]) == 1, settled_second

        status, conflict = peer.resume(
            second, key="shared-key", expected_version=second_version
        )
        assert status == 409, (status, conflict)
        assert conflict["error"]["code"] == "idempotency_conflict", conflict
        assert peer.ledger(first) == settled_first, "the key's own Run changed"
        after_second = peer.ledger(second)
        assert after_second == settled_second, after_second
        # No second generation and no resume operation appeared on the other Run.
        assert len(after_second["generations"]) == 1, after_second
        assert after_second["idempotency"] == ["create=applied=-"], after_second


def test_permanent_rejection_is_decided_once_and_replays_identically(tmp_path) -> None:
    """V-A-04/V-A-03: an unrecoverable resume fails atomically and keeps its identity."""

    with peer_root(tmp_path / "rejected") as peer:
        run_id = with_committed_checkpoints(peer, "rejected")
        before = peer.ledger(run_id)
        assert before["status"] == "interrupted", before
        version = peer.version(run_id)
        store = ReviewStore.from_url(peer.fixture.url)
        try:
            with store.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_branch_checkpoint "
                        "SET contract_checksum = 'sha256:' || repeat('0', 64) "
                        "WHERE run_id = :r"
                    ),
                    {"r": run_id},
                )
        finally:
            store.close()

        peer.start(name="rejecter")
        status, body = peer.resume(run_id, key="reject-key", expected_version=version)
        assert status == 409, (status, body)
        assert body["error"]["code"] == "review_not_recoverable", body
        assert body["error"]["details"]["reason"] == "checkpoint_incompatible", body
        rejected = peer.ledger(run_id)
        assert rejected["status"] == "failed", rejected
        assert rejected["error_code"] == "checkpoint_incompatible", rejected
        assert rejected["resume_count"] == 0, rejected
        assert rejected["idempotency"] == [
            "create=applied=-",
            "resume=failed=checkpoint_incompatible",
        ], rejected
        assert len([item for item in rejected["audit"] if "recovery:" in item]) == 1
        assert len(rejected["generations"]) == 1, rejected
        assert rejected["summaries"] == ["0"], rejected

        status, again = peer.resume(run_id, key="reject-key", expected_version=version)
        assert status == 409, (status, again)
        assert again["error"]["code"] == "review_not_recoverable", again
        assert again["error"]["details"]["reason"] == "checkpoint_incompatible", again
        assert peer.ledger(run_id) == rejected, "a replayed failure wrote something new"

        status, other = peer.resume(
            run_id, key="other-key", expected_version=peer.version(run_id)
        )
        assert status == 409, (status, other)
        assert other["error"]["code"] == "invalid_state_transition", other
        assert peer.ledger(run_id) == rejected


def test_resume_budget_is_spent_only_by_committed_generations(tmp_path) -> None:
    """V-A-04: exhaustion is decided in one transaction and replays without a new fact."""

    with peer_root(tmp_path / "budget") as peer:
        run_id = to_interrupted(peer, "budget")
        for index in (1, 2):
            version = peer.version(run_id)
            peer.start(
                plan={"barriers": {"after_resume_commit": {"action": "exit"}}},
                name=f"spend-{index}",
            )
            status, body = peer.resume(
                run_id, key=f"budget-{index}", expected_version=version
            )
            assert status == -1 or status == 202, (status, body)
            peer.died_at_barrier()
            peer.stop()
            peer.start(name=f"budget-after-{index}")
            peer.wait_status(run_id, "interrupted")
            assert peer.ledger(run_id)["resume_count"] == index

        version = peer.version(run_id)
        status, body = peer.resume(run_id, key="budget-3", expected_version=version)
        assert status == 409, (status, body)
        assert body["error"]["code"] == "review_not_recoverable", body
        assert body["error"]["details"]["reason"] == "resume_budget_exhausted", body
        spent = peer.ledger(run_id)
        assert spent["status"] == "failed", spent
        assert spent["error_code"] == "resume_budget_exhausted", spent
        assert spent["idempotency"] == [
            "create=applied=-",
            "resume=applied=-",
            "resume=applied=-",
            "resume=failed=resume_budget_exhausted",
        ], spent

        status, again = peer.resume(run_id, key="budget-3", expected_version=version)
        assert status == 409, (status, again)
        assert again["error"]["details"]["reason"] == "resume_budget_exhausted", again
        assert peer.ledger(run_id) == spent, "the exhausted-budget replay added a fact"


def test_authority_lost_before_a_replay_is_refused(tmp_path) -> None:
    """V-A-04: replay is not a way around a withdrawn credential."""

    with peer_root(tmp_path / "authority") as peer:
        run_id = to_interrupted(peer, "authority")
        version = peer.version(run_id)
        peer.start(name="authority-check")
        # The credential this session actually presents, not the one generation one was
        # opened on: `start` issues a fresh credential for every process.
        status, _body = peer.resume(
            run_id, key="authority-resume", expected_version=version
        )
        assert status == 202
        credential_id = str(parse_token(peer.token) or "")
        assert credential_id
        peer.wait_status(run_id, *PENDING)
        settled = peer.ledger(run_id)
        assert settled["idempotency"] == ["create=applied=-", "resume=applied=-"], (
            settled
        )

        assert peer.fixture.store.revoke_credential(
            credential_id=credential_id,
            revoked_at=datetime.now(UTC),
            reason="fix01a",
        )
        status, body = peer.resume(
            run_id, key="authority-resume", expected_version=version
        )
        assert status == 401, (status, body)
        assert body["error"]["code"] == "unauthenticated", body
        assert peer.ledger(run_id) == settled, "a refused replay changed the ledger"


def test_two_processes_racing_one_resume_produce_one_generation(tmp_path) -> None:
    """V-A-04: two independent processes serialize on the row lock, never on a guess."""

    barrier = "during_resume_commit:before_commit"
    with peer_root(tmp_path / "race") as peer:
        run_id = to_interrupted(peer, "race")
        version = peer.version(run_id)
        peer.start(
            plan={
                "barriers": {
                    barrier: {"action": "wait", "timeout_seconds": 40.0},
                }
            },
            name="first",
        )
        parked = peer.process
        assert parked is not None
        answers: dict[str, tuple[int, dict[str, Any]]] = {}

        def first() -> None:
            answers["a"] = peer.resume(run_id, key="race-a", expected_version=version)

        thread_a = threading.Thread(target=first, daemon=True)
        thread_a.start()
        _wait_for_barrier(parked.fault_dir, barrier)

        second = Peer(fixture=peer.fixture, root=peer.root)
        second.start(name="second")
        thread_b = threading.Thread(
            target=lambda: answers.__setitem__(
                "b", second.resume(run_id, key="race-b", expected_version=version)
            ),
            daemon=True,
        )
        thread_b.start()
        # Not a sleep that guesses where the second process got to: this returns only once
        # the database itself reports a backend waiting on the Run row lock for another PID.
        backends = wait_for_run_lock_waiter(peer.fixture.url)
        waiting = run_lock_waiters(backends)
        assert len(waiting) == 1, backends
        waiter = waiting[0]
        assert waiter.state == "active", waiter
        assert waiter.wait_event_type == "Lock", waiter
        assert waiter.wait_event == "transactionid", waiter
        blockers = [row for row in backends if row.pid in list(waiter.blocked_by)]
        assert len(blockers) == 1, waiter.blocked_by
        # The PID holding it up is a different backend, parked inside its own transaction:
        # the first process, stopped at its commit barrier while it holds the Run row lock.
        assert blockers[0].pid != waiter.pid, blockers
        assert blockers[0].state == "idle in transaction", blockers[0]
        _release(parked.fault_dir, barrier)
        thread_a.join(timeout=90)
        thread_b.join(timeout=90)
        assert not thread_a.is_alive() and not thread_b.is_alive()
        assert answers["a"][0] == 202, answers
        assert answers["b"][0] == 409, answers
        assert answers["b"][1]["error"]["code"] == "version_conflict", answers
        second.stop()
        peer.stop()

        peer.start(name="observer")
        ledger = peer.ledger(run_id)
        assert ledger["resume_count"] == 1, ledger
        assert len(ledger["generations"]) == 2, ledger
        assert ledger["idempotency"] == ["create=applied=-", "resume=applied=-"], ledger
        assert len([item for item in ledger["audit"] if "recovery:" in item]) == 1, (
            ledger
        )


def test_cancel_and_resume_orders_are_decided_by_commit(tmp_path) -> None:
    """V-A-04: cancel and resume in both orders leave one legal outcome each."""

    with peer_root(tmp_path / "orders") as peer:
        # Cancel first: a terminal Run has no automatic out-edge.
        first_run = to_interrupted(peer, "cancel-first")
        version = peer.version(first_run)
        peer.start(name="canceller")
        code, body = peer.cancel(
            first_run, key="cancel-first", expected_version=version
        )
        assert code == 200, (code, body)
        code, body = peer.resume(
            first_run, key="after-cancel", expected_version=peer.version(first_run)
        )
        assert code == 409, (code, body)
        assert body["error"]["code"] == "invalid_state_transition", body
        assert peer.ledger(first_run)["resume_count"] == 0

        # Resume first, then cancel: the resume stays, the Run ends cancelled.
        second_run = to_interrupted(peer, "resume-first")
        second_version = peer.version(second_run)
        peer.start(name="resumer")
        code, body = peer.resume(
            second_run, key="resume-then-cancel", expected_version=second_version
        )
        assert code == 202, (code, body)
        current = peer.version(second_run)
        code, body = peer.cancel(
            second_run, key="cancel-after-resume", expected_version=current
        )
        assert code == 200, (code, body)
        ledger = peer.ledger(second_run)
        assert ledger["status"] == "cancelled", ledger
        assert ledger["resume_count"] == 1, ledger
        assert ledger["receipts"] == ["0"], ledger
        code, body = peer.resume(
            second_run, key="after-cancel-two", expected_version=current
        )
        assert code == 409, (code, body)
