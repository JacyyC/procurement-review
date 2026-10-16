"""Shared harness for the FIX-09 model-invocation boundary verifications.

Three observations are recorded separately and are never collapsed into one counter:

* ``request_factory_executed`` — how many times the Worker's own request factory actually
  ran.  It is observed by wrapping the runner's call boundary.  Before FIX-09B the runner
  has no boundary at all, so the harness reports ``boundary_attached = False`` and the
  count stays 0 while a protected payload still reaches the adapter; that mismatch is the
  defect the counterexample is written for, and it is stated as an assertion rather than
  hidden behind a compatibility shim.
* ``model_call_entered`` — how many times a model adapter *synchronously received* a
  request, recorded at the adapter together with which receive API took it.  Creating a
  coroutine, booking an attempt, writing an intent or moving a telemetry counter is not an
  entry, and nothing here treats one as such.
* ``model_result_adopted`` — how many returned model results passed the Worker's
  result-consumption boundary, read from the execution's own telemetry and cross-checked
  against the committed checkpoints and the evidence verifier.

No scenario decides an interleaving with a bare sleep: every one of them parks the real
execution on the product's own file barrier or on a real committed database fact.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from fix08_support import ReadGate
from procurement_review.model_adapters import DeterministicFakeModelAdapter
from procurement_review.worker_contracts import WorkerModelRequest

#: The fixed three Worker names, in the order the topology declares them.
WORKERS = ("policy", "contract", "procurement_data")

#: The first consumption boundary each Worker reports for its own read.
READ_BOUNDARY = {
    "policy": "retrieval_result",
    "contract": "retrieval_result",
    "procurement_data": "procurement_result",
}

#: The two receive APIs a model adapter can offer.  ``start`` is the synchronous receive
#: FIX-09B introduces; ``generate`` is the pre-existing coroutine entry, recorded so a
#: scenario can state which one actually took the payload.
SYNC_START = "start"
COROUTINE_GENERATE = "generate"

#: The telemetry site suffix one already-returned model result is admitted at.
MODEL_RESULT_SUFFIX = ".model_result"

#: The one site name a Worker hands to the model-call boundary.  It names the boundary and
#: the Worker, never a Case, a tenant, a fixture count or a port.
MODEL_ENTRY_SUFFIX = "model_entry"


def entry_site(worker: str) -> str:
    """The call site one Worker reports to the model-call boundary."""

    return f"{worker}.{MODEL_ENTRY_SUFFIX}"


@dataclass
class BoundaryRecorder:
    """The exact record of what a model adapter synchronously received."""

    gate: ReadGate | None = None
    hold_response: bool = False
    #: Raises inside the synchronous ``start`` receive, before any handle exists.
    raises_at_start: bool = False
    #: Raises while the received call's result is being awaited.
    raises: bool = False
    #: Answers this instead of the frozen fixture, to exercise an invalid model output.
    output: Any | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    #: Strong references, so an entry can never be confused with a later request.
    requests: list[Any] = field(default_factory=list)

    def note_entry(self, request: Any, *, api: str) -> None:
        """Record one synchronous receipt; a request is never counted twice."""

        if any(item["request"] is request for item in self.entries):
            return
        evidence = tuple(getattr(request, "evidence_contexts", ()) or ())
        facts = tuple(getattr(request, "business_facts", ()) or ())
        worker = getattr(getattr(request, "worker", None), "value", "unknown")
        self.requests.append(request)
        self.entries.append(
            {
                "request": request,
                "worker": worker,
                "case_id": getattr(request, "case_id", None),
                "receive_api": api,
                "revocation_committed": bool(
                    self.gate is not None and self.gate.revoked.is_set()
                ),
                "evidence_context_count": len(evidence),
                "nonempty_evidence_context_count": sum(
                    bool(item.content) for item in evidence
                ),
                "business_fact_count": len(facts),
            }
        )

    def by_worker(self) -> dict[str, int]:
        return {
            worker: sum(1 for item in self.entries if item["worker"] == worker)
            for worker in WORKERS
        }

    def with_protected_payload(self) -> dict[str, int]:
        """Entries per Worker that carried a non-empty protected context or fact set."""

        return {
            worker: sum(
                1
                for item in self.entries
                if item["worker"] == worker
                and (
                    item["nonempty_evidence_context_count"]
                    or item["business_fact_count"]
                )
            )
            for worker in WORKERS
        }

    def receive_apis(self) -> list[str]:
        return [str(item["receive_api"]) for item in self.entries]

    def public_entries(self) -> list[dict[str, Any]]:
        """The record without the request objects, for assertion messages."""

        return [
            {key: value for key, value in item.items() if key != "request"}
            for item in self.entries
        ]


class BoundaryRecordingModel(DeterministicFakeModelAdapter):
    """The frozen ``test_only`` fixture plus an exact record of its synchronous receipt.

    ``start`` is the synchronous receive API: it takes the request and resolves it to the
    exact answer the adapter will hand back, while the caller's admitting transaction is
    still open.  ``generate`` is the pre-existing coroutine entry: it is recorded too,
    because a counterexample that only watched ``start`` would report "zero entries" on an
    implementation that still hands the whole context to the adapter through ``generate``.
    """

    def __init__(self, recorder: BoundaryRecorder) -> None:
        super().__init__()
        self._recorder = recorder

    def start(self, request: WorkerModelRequest) -> ModelCallHandle:
        self._recorder.note_entry(request, api=SYNC_START)
        if self._recorder.raises_at_start:
            raise RuntimeError("injected synchronous model receive failure")
        return ModelCallHandle(self, request, self._answer(request))

    async def generate(self, request: WorkerModelRequest) -> object:
        self._recorder.note_entry(request, api=COROUTINE_GENERATE)
        if self._recorder.raises_at_start:
            raise RuntimeError("injected synchronous model receive failure")
        return await self.deliver(request, self._answer(request))

    def _answer(self, request: WorkerModelRequest) -> object:
        """Resolve the request at receipt, or hand back the injected answer instead."""

        if self._recorder.output is not None:
            return self._recorder.output
        return self.resolve(request)

    async def deliver(self, request: WorkerModelRequest, findings: object) -> object:
        """Hand back the answer that was already resolved; the receipt is recorded."""

        recorder = self._recorder
        if recorder.raises:
            raise RuntimeError("injected model adapter failure")
        if (
            recorder.hold_response
            and recorder.gate is not None
            and recorder.gate.matches(request.worker.value)
        ):
            await recorder.gate.hold(request.worker.value)
        return findings


class ModelCallHandle:
    """One received call; only its result is asynchronous."""

    __slots__ = ("adapter", "request", "findings")

    def __init__(
        self,
        adapter: BoundaryRecordingModel,
        request: WorkerModelRequest,
        findings: object,
    ):
        self.adapter = adapter
        self.request = request
        self.findings = findings

    async def result(self) -> object:
        return await self.adapter.deliver(self.request, self.findings)


@dataclass
class InvocationProbe:
    """The recording wrapper over one execution's model-call boundary."""

    attached: bool = False
    factory_executions: int = 0
    sites: list[str] = field(default_factory=list)
    #: Every boundary the runner built, so a scenario can ask an old one directly whether
    #: it may still enter — the generation binding, not just the current one.
    invocations: list[Any] = field(default_factory=list)

    def note_factory(self, site: str) -> None:
        self.factory_executions += 1
        self.sites.append(site)


def attach_invocation(runner: Any, probe: InvocationProbe) -> InvocationProbe:
    """Wrap the runner's own call boundary so the factory count is exact.

    The boundary is built per execution by :meth:`ScopedWorkflowRunner.build_model_invocation`.
    Before FIX-09B that method does not exist, so nothing is attached and the probe stays at
    zero; the counterexample then fails on the behaviour — a protected payload reached the
    adapter without any boundary having run — which is exactly the defect it is written for.
    """

    builder = getattr(runner, "build_model_invocation", None)
    if builder is None:
        return probe

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        inner = builder(*args, **kwargs)
        probe.invocations.append(inner)
        return _RecordingInvocation(inner, probe)

    runner.build_model_invocation = wrapped
    probe.attached = True
    return probe


class _RecordingInvocation:
    """Run-scoped recording wrapper: it only counts what the boundary really built."""

    def __init__(self, inner: Any, probe: InvocationProbe) -> None:
        self._inner = inner
        self._probe = probe

    def enter(self, *, site: str, request_factory: Any) -> Any:
        def counted() -> Any:
            self._probe.note_factory(site)
            return request_factory()

        return self._inner.enter(site=site, request_factory=counted)


@dataclass(frozen=True, slots=True)
class Observations:
    """The three independent observations of one scenario, plus its raw entry record."""

    boundary_attached: bool
    request_factory_executed: int
    factory_sites: list[str]
    model_call_entered: dict[str, int]
    entered_with_protected_payload: dict[str, int]
    model_result_adopted: dict[str, int]
    receive_apis: list[str]
    entries: list[dict[str, Any]]


def admit_results(runner: Any, *, suffix: str) -> dict[str, int]:
    """Read the execution's own admission counters for one boundary suffix."""

    counts = runner.telemetry.result_admissions
    return {worker: int(counts.get(f"{worker}{suffix}", 0)) for worker in WORKERS}


def entry_denials(runner: Any) -> dict[str, int]:
    """Read the model-entry refusals the call boundary itself recorded."""

    return dict(runner.telemetry.model_entry_denials)


def wait_for_ungranted_lock(
    fixture: Any,
    application_name: str,
    *,
    timeout: float = 30.0,
) -> int:
    """Block until one tagged backend waits for a lock, and report how many it waits for.

    "The competing transaction is blocked" is a fact of the database, not a timing guess:
    PostgreSQL publishes the lock it is waiting for as an ungranted row in ``pg_locks``.  A
    fixed sleep could never distinguish a blocked revocation from a slow one.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with fixture.probe.transaction() as connection:
            waiting = int(
                connection.execute(
                    text(
                        """
                        SELECT count(*) FROM pg_locks l
                        JOIN pg_stat_activity a ON a.pid = l.pid
                        WHERE NOT l.granted AND a.application_name = :tag
                        """
                    ),
                    {"tag": application_name},
                ).scalar_one()
            )
        if waiting:
            return waiting
        time.sleep(0.02)
    raise AssertionError(
        f"{application_name} never waited for a lock; the admitting transaction had "
        "already been released"
    )


class BackgroundRevocation:
    """Commit one real revocation on the fixture's independent engine, off the loop.

    A revocation that has to *wait* cannot be issued from the driving thread without
    deadlocking the scenario, so it runs in its own thread and reports when it committed.
    """

    def __init__(self, fixture: Any, credential_id: str) -> None:
        self._fixture = fixture
        self._credential_id = credential_id
        self.committed = threading.Event()
        self.revoked = False
        self.seconds = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        started = time.monotonic()
        self.revoked = bool(self._fixture.revoke_credential(self._credential_id))
        self.seconds = time.monotonic() - started
        self.committed.set()

    def join(self, timeout: float = 30.0) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)
        assert self.committed.is_set(), "the revocation never committed"


def observe(
    runner: Any, recorder: BoundaryRecorder, probe: InvocationProbe
) -> Observations:
    """Collect the three observations without collapsing them into one number."""

    return Observations(
        boundary_attached=probe.attached,
        request_factory_executed=probe.factory_executions,
        factory_sites=list(probe.sites),
        model_call_entered=recorder.by_worker(),
        entered_with_protected_payload=recorder.with_protected_payload(),
        model_result_adopted=admit_results(runner, suffix=MODEL_RESULT_SUFFIX),
        receive_apis=recorder.receive_apis(),
        entries=recorder.public_entries(),
    )


__all__ = [
    "COROUTINE_GENERATE",
    "BackgroundRevocation",
    "MODEL_ENTRY_SUFFIX",
    "MODEL_RESULT_SUFFIX",
    "READ_BOUNDARY",
    "SYNC_START",
    "WORKERS",
    "BoundaryRecorder",
    "BoundaryRecordingModel",
    "InvocationProbe",
    "ModelCallHandle",
    "Observations",
    "admit_results",
    "attach_invocation",
    "entry_denials",
    "entry_site",
    "observe",
    "wait_for_ungranted_lock",
]
