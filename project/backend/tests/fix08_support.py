"""Shared harness for the FIX-08 result-consumption boundary verifications.

One real read is performed by the production port, and the response it has *already*
returned is held before the Worker can adopt it.  While it is held, a second engine
commits a real credential revocation and confirms it; only then is the response released.
A recorder above the model adapter reports what actually entered the model, so "an
attempt was booked", "a check had passed earlier" or "a counter moved" is never mistaken
for "the request was issued".

The same recorder can hold a model response on the way back, which is the other half of
the boundary: a call issued while the generation was still authorized must not have its
late result adopted.

Nothing here guesses at timing with a bare sleep, and nothing trusts an in-process flag
to represent the database's authorization state: every revocation is committed and
confirmed on the fixture's own independent connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from sqlalchemy import create_engine, text

from fix01a_support import Fix01aFixture, require_admin_dsn
from procurement_review.checkpointing import current_execution_worker
from procurement_review.model_adapters import DeterministicFakeModelAdapter
from procurement_review.procurement_mcp_client import OfficialProcurementMCPClient
from procurement_review.scoped_runtime import ScopedRetrievalPort

HOLD_TIMEOUT_SECONDS = 30.0

#: The fixed topological site names the Workers report.  They name the boundary, never a
#: Case, a tenant, a fixture count or a port.
RETRIEVAL_BOUNDARY = "retrieval_result"
RETRIEVAL_FAILURE_BOUNDARY = "retrieval_read_failed"
PROCUREMENT_BOUNDARY = "procurement_result"
PROCUREMENT_FAILURE_BOUNDARY = "procurement_read_failed"
MODEL_RESULT_BOUNDARY = "model_result"
MODEL_FAILURE_BOUNDARY = "model_failed"

WORKERS = ("policy", "contract", "procurement_data")

#: The read boundary a Worker's own code reports first, by Worker name.
READ_BOUNDARY = {
    "policy": RETRIEVAL_BOUNDARY,
    "contract": RETRIEVAL_BOUNDARY,
    "procurement_data": PROCUREMENT_BOUNDARY,
}
FAILURE_BOUNDARY = {
    "policy": RETRIEVAL_FAILURE_BOUNDARY,
    "contract": RETRIEVAL_FAILURE_BOUNDARY,
    "procurement_data": PROCUREMENT_FAILURE_BOUNDARY,
}


#: The gate value that holds every Worker's read at once.
ALL_WORKERS = "*"


@dataclass
class ReadGate:
    """Hold one already-returned read, or one already-returned model response.

    ``worker`` may be one Worker's name or :data:`ALL_WORKERS`.  In the latter case the gate
    opens only after every Worker's read has returned, which turns "no branch reached its
    commit before the others were refused" into a fact of the release order rather than a
    race between three concurrent branches.
    """

    worker: str
    #: Hold only the first read to arrive and let later ones pass.  A scenario that starts a
    #: second Run while the first is parked needs this: holding every Run's read would make
    #: the healthy Run's branch time out instead of proving it is unaffected.
    hold_once: bool = False
    fetched: threading.Event = field(default_factory=threading.Event)
    revoked: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    arrived: set[str] = field(default_factory=set)

    @property
    def expected(self) -> int:
        return len(WORKERS) if self.worker == ALL_WORKERS else 1

    def matches(self, worker: str) -> bool:
        if self.worker != ALL_WORKERS and self.worker != worker:
            return False
        return not (self.hold_once and self.fetched.is_set())

    def _arrive(self, worker: str) -> None:
        self.arrived.add(worker)
        if len(self.arrived) >= self.expected:
            self.fetched.set()

    def _hold_sync(self, worker: str) -> None:
        self._arrive(worker)
        if not self.release.wait(HOLD_TIMEOUT_SECONDS):
            raise AssertionError("the held response was never released")

    async def hold(self, worker: str) -> None:
        self._arrive(worker)
        released = await asyncio.to_thread(self.release.wait, HOLD_TIMEOUT_SECONDS)
        if not released:
            raise AssertionError("the held response was never released")

    def let_go(self) -> None:
        self.release.set()


@dataclass
class ModelRecorder:
    """The exact record of what crossed the model adapter boundary."""

    gate: ReadGate | None = None
    hold_response: bool = False
    raises: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    requests: list[Any] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": [dict(call) for call in self.calls],
            "by_worker": {
                worker: sum(1 for call in self.calls if call["worker"] == worker)
                for worker in WORKERS
            },
            "entered_after_revocation": [
                call
                for call in self.calls
                if call["revocation_committed"]
                and (
                    call["nonempty_evidence_context_count"]
                    or call["business_fact_count"]
                )
            ],
        }


class _RecordedCall:
    """The handle of one already-received recorded call; only its result is asynchronous."""

    __slots__ = ("_model", "_request", "_findings")

    def __init__(self, model: RecordingModel, request: Any, findings: Any) -> None:
        self._model = model
        self._request = request
        self._findings = findings

    async def result(self) -> Any:
        return await self._model.deliver(self._request, self._findings)


class RecordingModel(DeterministicFakeModelAdapter):
    """The frozen ``test_only`` fixture plus an exact record of its model boundary.

    The receipt is recorded in ``start``: the adapter has taken the immutable request and
    resolved it to the exact answer it will return, while the caller's admitting
    transaction is still open.  ``generate`` keeps working for callers that use the
    coroutine entry and records exactly the same one receipt.
    """

    def __init__(self, recorder: ModelRecorder) -> None:
        super().__init__()
        self._recorder = recorder

    def start(self, request: Any) -> _RecordedCall:
        recorder = self._recorder
        evidence = tuple(getattr(request, "evidence_contexts", ()) or ())
        facts = tuple(getattr(request, "business_facts", ()) or ())
        recorder.requests.append(request)
        recorder.calls.append(
            {
                "worker": request.worker.value,
                "case_id": request.case_id,
                "revocation_committed": bool(
                    recorder.gate is not None and recorder.gate.revoked.is_set()
                ),
                "evidence_context_count": len(evidence),
                "nonempty_evidence_context_count": sum(
                    bool(item.content) for item in evidence
                ),
                "business_fact_count": len(facts),
            }
        )
        return _RecordedCall(self, request, self.resolve(request))

    async def generate(self, request: Any) -> Any:
        return await self.start(request).result()

    async def deliver(self, request: Any, findings: Any) -> Any:
        """Hand back the answer that was already resolved; the receipt has been recorded."""

        recorder = self._recorder
        if recorder.raises:
            raise RuntimeError("injected model adapter failure")
        if (
            recorder.hold_response
            and recorder.gate is not None
            and recorder.gate.matches(request.worker.value)
        ):
            # The response exists; it is held before the Worker can validate or adopt it.
            await recorder.gate.hold(request.worker.value)
        return findings


@contextlib.contextmanager
def held_reads(gate: ReadGate, *, fail_read: bool = False) -> Iterator[None]:
    """Hold the two production read entry points, optionally failing the target read."""

    original_retrieve = ScopedRetrievalPort.retrieve
    original_collect = OfficialProcurementMCPClient.collect_review_facts

    def held_worker() -> str | None:
        """The gated Worker whose read is running on this thread, if there is one."""

        running = current_execution_worker()
        if running in WORKERS and gate.matches(running):
            return running
        return None

    def held_retrieve(port: Any, request: Any) -> Any:
        worker = held_worker()
        if worker is not None and fail_read:
            gate._hold_sync(worker)
            raise RuntimeError("injected retrieval port failure")
        response = original_retrieve(port, request)
        if worker is not None:
            gate._hold_sync(worker)
        return response

    async def held_collect(client: Any, request: Any) -> Any:
        target = gate.matches("procurement_data")
        if target and fail_read:
            await gate.hold("procurement_data")
            raise RuntimeError("injected MCP client failure")
        response = await original_collect(client, request)
        if target:
            await gate.hold("procurement_data")
        return response

    with patch.object(ScopedRetrievalPort, "retrieve", held_retrieve):
        with patch.object(
            OfficialProcurementMCPClient, "collect_review_facts", held_collect
        ):
            yield


class StoreOutage:
    """Make one scenario's database genuinely unreachable, then restore it.

    Two real PostgreSQL facts are combined so the outage is a fact of the database rather
    than a race:

    * ``ALTER DATABASE ... ALLOW_CONNECTIONS false`` refuses every *new* connection to this
      database, so a pool that reconnects cannot quietly succeed (the store is built with
      ``pool_pre_ping``, which would otherwise open a fresh connection and never notice).
    * every backend already carrying the application's own connection tag is terminated, so
      the connections the pool still holds are dead too.

    The scenario's own driver and observation engines keep the connections they already
    opened, so the run can still be read and settled while the application cannot reach the
    store at all.
    """

    def __init__(self, fixture: Fix01aFixture) -> None:
        self._fixture = fixture
        # A connection to the maintenance database, never to the scenario's own database:
        # PostgreSQL refuses to disallow connections for the database it is connected to.
        self._engine = create_engine(require_admin_dsn(), isolation_level="AUTOCOMMIT")
        self._admin = self._engine.connect()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._refused = False

    def refuse(self) -> None:
        self._admin.execute(
            text(f'ALTER DATABASE "{self._fixture.database}" ALLOW_CONNECTIONS false')
        )
        self._refused = True
        self._thread = threading.Thread(target=self._terminate_loop, daemon=True)
        self._thread.start()

    def _terminate_loop(self) -> None:
        engine = create_engine(require_admin_dsn(), isolation_level="AUTOCOMMIT")
        try:
            while not self._stop.is_set():
                with engine.connect() as connection:
                    connection.execute(
                        text(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                            "WHERE application_name = :tag AND pid <> pg_backend_pid()"
                        ),
                        {"tag": self._fixture.application_name},
                    )
                time.sleep(0.05)
        finally:
            engine.dispose()

    def wait_until_unreachable(self, timeout: float = 30.0) -> None:
        """Block until the application holds no live backend and cannot open one.

        Refusing new connections and terminating the existing ones are two separate
        PostgreSQL facts; releasing the held response in between would let the pool answer
        from a connection that has not been taken away yet.  This waits for the second fact.
        """

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            alive = self._admin.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = :db AND application_name = :tag"
                ),
                {"db": self._fixture.database, "tag": self._fixture.application_name},
            ).scalar_one()
            if int(alive) == 0:
                return
            time.sleep(0.05)
        raise AssertionError("the application kept a live backend through the outage")

    def __enter__(self) -> StoreOutage:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def restore(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._refused:
            self._admin.execute(
                text(
                    f'ALTER DATABASE "{self._fixture.database}" ALLOW_CONNECTIONS true'
                )
            )
            self._refused = False

    def close(self) -> None:
        self.restore()
        self._admin.close()
        self._engine.dispose()


async def wait_for_execution_task(
    service: Any, run_id: str, generation: int, timeout: float = 60.0
) -> None:
    """Block until every branch of one dispatch has settled.

    ``create_review`` returns as soon as the Run is committed and the execution is
    dispatched, so waiting for the response says nothing about whether the three branches
    are finished.  The service's own execution task is the exact boundary: it completes
    only after the graph, its fan-in and the branch accounting are done, which is what an
    assertion about "no model call happened" has to be made after.
    """

    # The dispatch registry is private, and deliberately so: this is an observation of an
    # existing boundary, not a product capability added for a test.
    tracked = service._tasks.get((run_id, generation))
    if tracked is None:
        # A dispatch that has already finished removes itself from the registry, so a
        # missing entry means there is nothing left to wait for.
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tracked.done():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} generation {generation} never settled")


@contextlib.contextmanager
def new_database(name: str, **kwargs: Any) -> Iterator[Fix01aFixture]:
    """One disposable database, named so a failure report identifies the scenario."""

    fixture = Fix01aFixture(f"{name}_{secrets.token_hex(3)}", **kwargs)
    try:
        yield fixture
    finally:
        fixture.close()
