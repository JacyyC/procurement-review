"""FIX-06A V-A-01 and V-A-02: the preload of a Run's runtime under live authority.

R-01 showed that the dispatch admission and the actual preload reads were two different
transactions.  A credential revoked between them still let the assembly read documents,
retrieval scores and procurement records, because only *port calls* asked for admission
and every preload read had already happened before the first port call.

These scenarios therefore observe the real statements the application engine issues — the
Worker read counters stayed empty while the preload was reading, so they cannot be the
proof — and they cover both orders of the one linearization point that matters:

* the revocation is committed while the execution stands before ``ScopedWorkflowRunner.run``
  entered (the window the original probe measured), and
* the preload read obtains the authority share lock first, so a revocation behind it has to
  wait for that short transaction to end.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import event, text

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
from procurement_review.local_runtime import ScopedWorkflowRunner
from procurement_review.review_store import ReviewStore
from procurement_review.service_errors import ReviewServiceError

from fix01a_support import (
    APP_TAG,
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
from v1_03_support import require_admin_dsn

#: The three tables R-01 measured reads against; they carry the preload's own data.
PROTECTED_TABLES = (
    "review_document_registry",
    "review_retrieval_score",
    "review_procurement_record",
)

SETTLED = (
    "interrupted",
    "failed",
    "cancelled",
    "pending_confirmation",
    "pending_human_review",
)

#: The three admission sites a preload has to go through, one per protected read.
PRELOAD_SITES = ("evidence_index", "procurement_records", "retrieval_index")

#: Every revocation kind V-A-01 names, committed on the fixture's independent connection.
REVOCATIONS: dict[str, Callable[[Fix01aFixture, Identity], None]] = {
    AUTHORITY_CREDENTIAL_REVOKED: lambda fixture, ident: fixture.revoke_credential(
        ident.credential_id
    ),
    AUTHORITY_CREDENTIAL_EXPIRED: lambda fixture, ident: fixture.expire_credential(
        ident.credential_id
    ),
    AUTHORITY_SUBJECT_DISABLED: lambda fixture, ident: fixture.disable_subject(
        ident.context.subject_id
    ),
    AUTHORITY_GRANT_REVOKED: lambda fixture, ident: fixture.revoke_case_grant(
        ident.context.subject_id, ident.registration.case_id
    ),
    AUTHORITY_REVISION_CHANGED: lambda fixture, ident: (
        fixture.bump_authorization_revision()
    ),
}


@contextmanager
def database(name: str, *, application_name: str = APP_TAG) -> Iterator[Fix01aFixture]:
    fixture = Fix01aFixture(name, application_name=application_name)
    try:
        yield fixture
    finally:
        fixture.close()


@dataclass(slots=True)
class ProtectedReads:
    """Observe the protected SQL the application engine really issues.

    ``before_cursor_execute`` sees the statement the execution is about to run, which makes
    "this read started after the revocation was committed" a fact of the run: the scenario
    flips the recording switch only once the revocation's commit has returned.
    ``after_cursor_execute`` carries the row count that statement actually produced, so a
    read that ran and returned data cannot be mistaken for one that never started.
    """

    store: ReviewStore
    park_on: str | None = None
    statements: list[str] = field(default_factory=list)
    rows: list[tuple[str, int]] = field(default_factory=list)
    parked: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    _recording: bool = False
    _parked_once: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def install(self) -> None:
        event.listen(self.store.engine, "before_cursor_execute", self._before)
        event.listen(self.store.engine, "after_cursor_execute", self._after)

    def remove(self) -> None:
        event.remove(self.store.engine, "before_cursor_execute", self._before)
        event.remove(self.store.engine, "after_cursor_execute", self._after)

    def arm(self) -> None:
        """Start recording; the caller does this after the revocation's commit returned."""

        self._recording = True

    @staticmethod
    def _tables(statement: str) -> tuple[str, ...]:
        return tuple(name for name in PROTECTED_TABLES if name in statement)

    def _before(
        self,
        connection: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        del connection, cursor, parameters, context, executemany
        hit = self._tables(statement)
        if not hit:
            return
        if self.park_on is not None and self.park_on in hit and not self._parked_once:
            self._parked_once = True
            self.parked.set()
            if not self.release.wait(timeout=90.0):
                raise AssertionError("the parked preload read was never released")
        if not self._recording:
            return
        with self._lock:
            self.statements.append(" ".join(statement.split()))

    def _after(
        self,
        connection: Any,
        cursor: Any,
        statement: str,
        parameters: Any,
        context: Any,
        executemany: bool,
    ) -> None:
        del connection, parameters, context, executemany
        hit = self._tables(statement)
        if not hit or not self._recording:
            return
        with self._lock:
            self.rows.append((hit[0], int(cursor.rowcount)))

    def summary(self) -> dict[str, Any]:
        """The observed reads, as the assertion messages and Evidence need them."""

        with self._lock:
            return {"statements": list(self.statements), "rows": list(self.rows)}


@dataclass(slots=True)
class ParkedRunner:
    """Park one execution at the entry of ``ScopedWorkflowRunner.run``.

    This is exactly the window R-01 measured: the service has already admitted the dispatch
    on live authority, and the scope-bound runtime — including every preload read — has not
    been built yet.  The wrapper waits for a release the scenario sets once the revocation's
    commit has returned; there is no sleep and no product short-circuit in the path.
    """

    runner: ScopedWorkflowRunner
    reached: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    _original: Any = None

    def install(self) -> None:
        self._original = self.runner.run
        original = self._original
        reached, release = self.reached, self.release

        async def parked(*args: Any, **kwargs: Any) -> Any:
            reached.set()
            if not release.wait(timeout=90.0):
                raise AssertionError("the parked execution was never released")
            return await original(*args, **kwargs)

        self.runner.run = parked  # type: ignore[method-assign]

    def reopen(self) -> None:
        """Arm the same park for a second execution over the same runtime."""

        self.reached.clear()
        self.release.clear()

    def restore(self) -> None:
        if self._original is not None:
            self.runner.run = self._original  # type: ignore[method-assign]


def create_run(
    active: Any, fixture: Fix01aFixture, ident: Identity, *, key: str
) -> Any:
    return active.service.create_review(
        fixture.create_request(ident),
        registration=ident.registration,
        scope=ident.scope,
        context=ident.context,
        idempotency_key=key,
    )


def test_preload_reads_nothing_after_a_committed_revocation() -> None:
    """R-01/V-A-01: with the revocation committed first, the preload reads no data.

    The dispatch admission has already run and passed; the revocation is committed while
    the execution stands before the assembly.  What is asserted is the product's own SQL:
    no protected SELECT may start after that commit, whatever the Worker counters say.
    """

    with database("preload_revoked") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        observation = ProtectedReads(fixture.app)
        park = ParkedRunner(active.runner)
        loop = Loop()
        observation.install()
        park.install()
        try:
            handle = loop.start(
                create_run(active, fixture, ident, key="preload-revoked")
            )
            assert park.reached.wait(timeout=60.0), "the assembly was never reached"
            run_id = fixture.latest_run()
            # The commit has returned before the execution may continue, so every protected
            # statement recorded from here on is a read the revocation should have stopped.
            assert fixture.revoke_credential(ident.credential_id)
            observation.arm()
            park.release.set()
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED)
            observed = observation.summary()
        finally:
            park.release.set()
            park.restore()
            observation.remove()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert observed["statements"] == [], (
            "the preload read protected data after the revocation was committed: "
            f"{observed}"
        )
        assert observed["rows"] == [], observed
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
        assert body["error_code"] == "authorization_revoked", body
        assert body["checkpoints"] == []
        assert body["summaries"] == 0


@pytest.mark.parametrize("reason", sorted(REVOCATIONS))
def test_every_revocation_kind_stops_the_preload(reason: str) -> None:
    """V-A-01/V-A-02: revocation, expiry, disable, grant loss and revision all linearize."""

    with database(f"preload_kind_{reason}") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        observation = ProtectedReads(fixture.app)
        park = ParkedRunner(active.runner)
        loop = Loop()
        observation.install()
        park.install()
        try:
            handle = loop.start(
                create_run(active, fixture, ident, key=f"kind-{reason}")
            )
            assert park.reached.wait(timeout=60.0)
            run_id = fixture.latest_run()
            REVOCATIONS[reason](fixture, ident)
            observation.arm()
            park.release.set()
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED)
            observed = observation.summary()
        finally:
            park.release.set()
            park.restore()
            observation.remove()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert observed["statements"] == [], f"{reason}: {observed}"
        assert observed["rows"] == [], f"{reason}: {observed}"
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, (reason, body)
        assert body["error_code"] == "authorization_revoked", (reason, body)
        assert body["checkpoints"] == [], (reason, body)
        assert body["summaries"] == 0, (reason, body)


def test_resume_preload_reads_nothing_after_a_committed_revocation() -> None:
    """V-A-01: the resume entry builds the same runtime, so it needs the same admission."""

    with database("preload_resume") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        observation = ProtectedReads(fixture.app)
        park = ParkedRunner(active.runner)
        loop = Loop()
        observation.install()
        park.install()
        try:
            seed = loop.start(create_run(active, fixture, ident, key="preload-seed"))
            assert park.reached.wait(timeout=60.0)
            run_id = fixture.latest_run()
            assert fixture.revoke_credential(ident.credential_id)
            park.release.set()
            seed.result(timeout=120)
            seeded = settle(fixture, run_id, *SETTLED)
            assert seeded["status"] == ReviewLifecycleStatus.INTERRUPTED.value, seeded

            # The resume runs on a live credential for the same still-granted subject, so the
            # only reason it can stop is the revocation committed before its own preload.
            fresh = fixture.identity()
            current = active.service.get_review(run_id, context=fresh.context)
            park.reopen()
            resumed_handle = loop.start(
                active.service.resume_review(
                    run_id,
                    ResumeRequestV2(expected_version=current.revision),
                    context=fresh.context,
                    idempotency_key="preload-resume",
                )
            )
            assert park.reached.wait(timeout=60.0), (
                "the resume assembly was never reached"
            )
            assert fixture.revoke_credential(fresh.credential_id)
            observation.arm()
            park.release.set()
            resumed_handle.result(timeout=120)
            resumed = settle(fixture, run_id, *SETTLED)
            observed = observation.summary()
        finally:
            park.release.set()
            park.restore()
            observation.remove()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert observed["statements"] == [], (
            "the resumed assembly read protected data after the revocation was committed: "
            f"{observed}"
        )
        assert observed["rows"] == [], observed
        assert resumed["status"] == ReviewLifecycleStatus.INTERRUPTED.value, resumed
        assert resumed["error_code"] == "authorization_revoked", resumed
        assert resumed["summaries"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            "UPDATE review_run_generation SET status = 'superseded' "
            "WHERE run_id = :r AND generation = 1",
            "no_active_generation",
        ),
        (
            "UPDATE review_run_generation "
            "SET lease_expires_at = now() - interval '1 minute' "
            "WHERE run_id = :r AND generation = 1",
            "lease_expired",
        ),
    ],
)
def test_preload_admission_covers_generation_and_lease_not_only_credentials(
    mutation: str, expected_reason: str
) -> None:
    """V-A-02: the preload admission checks the generation it serves, not just the token."""

    with database(f"preload_state_{expected_reason}") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        observation = ProtectedReads(fixture.app)
        park = ParkedRunner(active.runner)
        loop = Loop()
        observation.install()
        park.install()
        try:
            handle = loop.start(
                create_run(active, fixture, ident, key=f"state-{expected_reason}")
            )
            assert park.reached.wait(timeout=60.0)
            run_id = fixture.latest_run()
            with fixture.driver.transaction() as connection:
                connection.execute(text(mutation), {"r": run_id})
            observation.arm()
            park.release.set()
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED)
            observed = observation.summary()
        finally:
            park.release.set()
            park.restore()
            observation.remove()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert observed["statements"] == [], observed
        assert observed["rows"] == [], observed
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
        assert body["error_code"] == expected_reason, body
        assert body["summaries"] == 0


def test_preload_read_holds_the_authority_lock_until_its_transaction_ends() -> None:
    """V-A-01: a revocation that arrives second waits; the preload read owns the lock.

    The execution is parked inside the protected SELECT itself, which is the moment the
    read's own transaction is open.  A concurrent revision bump must therefore time out
    while that transaction is alive — otherwise the check and the read are two different
    transactions and a revocation can slip between them.
    """

    with database("preload_locks") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        observation = ProtectedReads(fixture.app, park_on="review_document_registry")
        loop = Loop()
        engine = fixture.admin_engine(autocommit=False)
        observation.install()
        try:
            handle = loop.start(create_run(active, fixture, ident, key="preload-locks"))
            assert observation.parked.wait(timeout=60.0), (
                "the preload read never reached the protected SELECT"
            )
            run_id = fixture.latest_run()
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
            observation.release.set()
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED)
        finally:
            observation.release.set()
            observation.remove()
            engine.dispose()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert answer is not None, (
            "a revocation committed while the preload read held the authority lock"
        )
        assert "lock timeout" in answer.lower(), answer
        # The read that held the lock was a legal one, so the Run still finishes normally.
        assert body["status"] != ReviewLifecycleStatus.PROCESSING.value, body


def test_preload_admission_stops_when_the_authorization_store_is_unreachable() -> None:
    """V-A-02: an unreachable store stops the preload instead of reading past it.

    The outage is injected at the preload's own admission boundary with the product's
    ``terminate`` action, which breaks every backend carrying the application's connection
    tag — the same boundary the dispatch-window outage scenario uses.  The run may then only
    be settled by the lease scanner over a connection the outage does not touch.
    """

    tag = f"fix06a_preload_outage_{secrets.token_hex(4)}"
    barrier = "during_guarded_read:retrieval_index"
    previous = os.environ.get(FAULT_APPLICATION_NAME_ENV)
    os.environ[FAULT_APPLICATION_NAME_ENV] = tag
    directory = barrier_dir("preload-outage")
    fault = FaultController(
        enabled=True,
        directory=directory,
        admin_dsn=require_admin_dsn(),
        barriers={barrier: BarrierSpec(action="terminate", timeout_seconds=2.0)},
    )
    try:
        with database("preload_outage", application_name=tag) as fixture:
            ident = fixture.identity()
            active = runtime(
                fixture,
                fault,
                lease_seconds=2.0,
                heartbeat_seconds=0.5,
                scan_interval_seconds=0.2,
            )
            observation = ProtectedReads(fixture.app)
            loop = Loop()
            observation.install()
            try:
                created, _ = loop.call(
                    create_run(active, fixture, ident, key="preload-outage")
                )
                run_id = created.run_id
                wait_barrier(directory, barrier)
                observation.arm()
                deadline = time.monotonic() + 30.0
                body = fixture.ledger(run_id)
                while time.monotonic() < deadline:
                    # The scan runs over the observation connection on purpose: the injected
                    # outage takes every backend tagged as the application.
                    fixture.observer.scan_expired_leases(
                        now=datetime.now(UTC),
                        events_factory=lambda _run_id: (),
                        audit_factory=lambda _run_id: None,
                    )
                    body = fixture.ledger(run_id)
                    if body["status"] != ReviewLifecycleStatus.PROCESSING.value:
                        break
                    time.sleep(0.2)
                observed = observation.summary()
            finally:
                observation.remove()
                loop.call(active.service.shutdown(), timeout=30.0)
                loop.close()

            # Nothing was read on the strength of an unreachable store, and the Run never
            # claims a completion it did not commit.
            assert observed["rows"] == [], observed
            assert body["attempts"] == [], body
            assert body["summaries"] == 0, body
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] in {"storage_unavailable", "lease_expired"}, body
    finally:
        if previous is None:
            os.environ.pop(FAULT_APPLICATION_NAME_ENV, None)
        else:
            os.environ[FAULT_APPLICATION_NAME_ENV] = previous


def test_legal_resume_rebuilds_the_runtime_and_completes() -> None:
    """V-A-01/V-A-02 positive control: a legal resume passes the same preload admission."""

    directory = barrier_dir("preload-legal-resume")
    fault = parked_fault(directory, "attempt_started:policy")
    with database("preload_resume_legal") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, fault)
        loop = Loop()
        try:
            handle = loop.start(create_run(active, fixture, ident, key="resume-legal"))
            wait_barrier(directory, "attempt_started:policy")
            run_id = fixture.latest_run()
            assert fixture.revoke_credential(ident.credential_id)
            release_barrier(directory, "attempt_started:policy")
            handle.result(timeout=120)
            interrupted = settle(fixture, run_id, *SETTLED)
            assert interrupted["status"] == ReviewLifecycleStatus.INTERRUPTED.value, (
                interrupted
            )

            fresh = fixture.identity()
            current = active.service.get_review(run_id, context=fresh.context)
            _, scheduled = loop.call(
                active.service.resume_review(
                    run_id,
                    ResumeRequestV2(expected_version=current.revision),
                    context=fresh.context,
                    idempotency_key="resume-legal",
                )
            )
            assert scheduled is True, "the legal resume was answered as a replay"
            body = settle(
                fixture, run_id, "pending_confirmation", "pending_human_review"
            )
        finally:
            release_barrier(directory, "attempt_started:policy")
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert body["status"] == "pending_confirmation", body
        assert body["resume_count"] == 1, body
        assert sorted(body["checkpoints"]) == [
            "contract@2",
            "policy@2",
            "procurement_data@2",
        ], body
        assert body["summaries"] == 1, body
        # The resumed generation built its own runtime, and each of its protected preloads
        # was admitted in the transaction it read in — exactly once per generation.
        assert active.preload_reads == dict.fromkeys(PRELOAD_SITES, 2), (
            active.preload_reads
        )


def test_preload_check_and_read_share_one_transaction() -> None:
    """V-A-01, independent assertion: the live check and the preload read are one transaction.

    This test deliberately does not use the new admission telemetry or the guarded-read
    barrier.  It reconstructs the statement stream of the application engine together with
    its commit boundaries and requires that every protected preload SELECT is preceded, on
    the very same connection and with no commit in between, by the authority share lock on
    the tenant revision row.  A "check once more before building" patch would fail here:
    the check would sit in its own transaction and the read in the next one.
    """

    events: list[tuple[str, int, str]] = []
    held: list[Any] = []
    engine: Any = None
    with database("preload_tx") as fixture:
        engine = fixture.app.engine

        def before_cursor(
            connection: Any,
            cursor: Any,
            statement: str,
            parameters: Any,
            context: Any,
            executemany: bool,
        ) -> None:
            del cursor, parameters, context, executemany
            held.append(connection)
            events.append(("sql", id(connection), " ".join(statement.split())))

        def record_commit(connection: Any) -> None:
            events.append(("commit", id(connection), ""))

        def record_rollback(connection: Any) -> None:
            events.append(("rollback", id(connection), ""))

        event.listen(engine, "before_cursor_execute", before_cursor)
        event.listen(engine, "commit", record_commit)
        event.listen(engine, "rollback", record_rollback)
        ident = fixture.identity()
        active = runtime(fixture, None)
        loop = Loop()
        try:
            created, _ = loop.call(create_run(active, fixture, ident, key="preload-tx"))
            run_id = created.run_id
            body = settle(
                fixture, run_id, "pending_confirmation", "pending_human_review"
            )
        finally:
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()
            event.remove(engine, "before_cursor_execute", before_cursor)
            event.remove(engine, "commit", record_commit)
            event.remove(engine, "rollback", record_rollback)

        assert body["status"] == "pending_confirmation", body
        # Only the application engine is observed, so every statement below is product SQL.
        assert held, "the engine never issued the observed statements"
        protected = [
            index
            for index, entry in enumerate(events)
            if entry[0] == "sql"
            and any(table in entry[2] for table in PROTECTED_TABLES)
        ]
        assert protected, events
        for index in protected:
            connection_id = events[index][1]
            admitted = False
            for earlier in range(index - 1, -1, -1):
                kind, other, statement = events[earlier]
                if other != connection_id:
                    continue
                if kind in {"commit", "rollback"}:
                    break
                if (
                    "FOR SHARE" in statement
                    and "review_authorization_revision" in statement
                ):
                    admitted = True
                    break
            assert admitted, (
                "a protected preload read ran outside the transaction that admitted it: "
                f"{events[index][2][:80]}"
            )
        # One admitted transaction per protected preload site, and no more.
        assert sum(1 for _, _, text_ in events if "FOR SHARE" in text_) >= len(
            PRELOAD_SITES
        ), events


def test_clean_run_admits_every_preload_site_and_still_completes() -> None:
    """V-A-01 positive control: no preload site hides from admission, and legal runs pass."""

    with database("preload_clean") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        loop = Loop()
        try:
            created, _ = loop.call(
                create_run(active, fixture, ident, key="preload-clean")
            )
            run_id = created.run_id
            body = settle(
                fixture, run_id, "pending_confirmation", "pending_human_review"
            )
        finally:
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert body["status"] == "pending_confirmation", body
        assert sorted(body["checkpoints"]) == [
            "contract@1",
            "policy@1",
            "procurement_data@1",
        ], body
        assert body["summaries"] == 1, body
        # Every protected preload read asked for admission exactly once, and the Worker read
        # accounting the earlier stages fixed is unchanged by the preload's own admissions.
        assert active.preload_reads == dict.fromkeys(PRELOAD_SITES, 1), (
            active.preload_reads
        )
        assert active.reads_entered == {
            "policy": 1,
            "contract": 1,
            "procurement_data": 3,
        }, active.reads_entered
        assert active.read_denials == {}, active.read_denials


def test_preload_denial_is_reported_per_site() -> None:
    """V-A-02: a refused preload names the site it refused instead of one blanket code."""

    with database("preload_sites") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        observation = ProtectedReads(fixture.app)
        park = ParkedRunner(active.runner)
        loop = Loop()
        observation.install()
        park.install()
        try:
            handle = loop.start(create_run(active, fixture, ident, key="preload-sites"))
            assert park.reached.wait(timeout=60.0)
            run_id = fixture.latest_run()
            fixture.disable_subject(ident.context.subject_id)
            observation.arm()
            park.release.set()
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED)
            observed = observation.summary()
        finally:
            park.release.set()
            park.restore()
            observation.remove()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert observed["statements"] == [], observed
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
        # The first protected read of the assembly is the retrieval index; it is the one that
        # must carry the refusal, and no later site may have been admitted.
        assert active.read_denials.get("retrieval_index", 0) == 1, active.read_denials
        assert active.preload_reads == {}, active.preload_reads


def test_preload_contention_keeps_the_one_lock_order() -> None:
    """V-A-02: concurrent revocations against a live preload never produce a deadlock."""

    with database("preload_contention") as fixture:
        ident = fixture.identity()
        active = runtime(fixture, None)
        loop = Loop()
        errors: list[str] = []
        stop = threading.Event()
        engine = fixture.admin_engine(autocommit=False)

        def contend() -> None:
            while not stop.is_set():
                with engine.connect() as connection:
                    connection.execute(text("SET LOCAL lock_timeout = '1s'"))
                    try:
                        connection.execute(
                            text(
                                "UPDATE review_authorization_revision "
                                "SET revision = revision + 1 WHERE tenant_id = :t"
                            ),
                            {"t": ident.context.tenant_id},
                        )
                        connection.commit()
                    except Exception as error:  # noqa: BLE001 - only deadlocks matter
                        connection.rollback()
                        errors.append(f"{type(error).__name__}:{error}")

        worker = threading.Thread(target=contend, daemon=True)
        worker.start()
        created = None
        body: dict[str, Any] | None = None
        refused: str | None = None
        try:
            try:
                created, _ = loop.call(
                    create_run(active, fixture, ident, key="preload-contention")
                )
            except ReviewServiceError as error:
                # A revision that moved before the dispatch is a legal contract answer, not
                # a lock-order failure; the deadlock check below still applies to it.
                refused = error.code
            if created is not None:
                body = settle(fixture, created.run_id, *SETTLED)
        finally:
            stop.set()
            worker.join(timeout=10)
            engine.dispose()
            loop.call(active.service.shutdown(), timeout=30.0)
            loop.close()

        assert [item for item in errors if "40P01" in item] == [], errors
        assert refused in {None, "authorization_revision_changed"}, refused
        assert created is not None or refused is not None
        if created is not None:
            assert body is not None
            assert body["summaries"] <= 1, body
