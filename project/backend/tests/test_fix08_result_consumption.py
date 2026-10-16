"""FIX-08: the result-consumption boundary of the three read-only Workers.

The registered defect (AUDIT-02 AF-01) is that a read response which *returned after* the
acting credential had already been revoked was still used to build ``EvidenceContext`` /
``WorkerModelRequest`` and to call the model.  A final ``interrupted`` and a zero summary
only prove the write was refused; they say nothing about what the execution adopted on the
way there.

Every scenario below drives the real ``ScopedWorkflowRunner`` over a real database, holds
a real read response at its return, commits a real revocation on the fixture's own
independent connection, and only then releases it.  The recorder above the model adapter
is what makes "an affected model call happened" a measurement of the adapter boundary
rather than an inference from a booked attempt or a passed check.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from fix01a_support import Fix01aFixture, Loop, runtime, settle
from fix08_support import (
    FAILURE_BOUNDARY,
    MODEL_FAILURE_BOUNDARY,
    MODEL_RESULT_BOUNDARY,
    READ_BOUNDARY,
    WORKERS,
    ModelRecorder,
    StoreOutage,
    wait_for_execution_task,
    ReadGate,
    RecordingModel,
    held_reads,
    new_database,
)
from procurement_review.contracts import ReviewLifecycleStatus
from procurement_review.model_invocation import OfflineModelInvocation
from procurement_review.result_admission import OFFLINE_RESULT_CONSUMPTION

SETTLED = (
    ReviewLifecycleStatus.PENDING_CONFIRMATION.value,
    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
    ReviewLifecycleStatus.INTERRUPTED.value,
    ReviewLifecycleStatus.FAILED.value,
)


def _run_review(
    fixture: Fix01aFixture,
    *,
    worker: str,
    revoke: bool,
    key: str,
    recorder: ModelRecorder,
    gate: ReadGate,
    fail_read: bool = False,
    lease_seconds: float = 1000.0,
) -> dict[str, Any]:
    """Hold one Worker's read, optionally revoke, then release, settle and report."""

    active = runtime(
        fixture,
        lease_seconds=lease_seconds,
        heartbeat_seconds=max(lease_seconds / 2.0, 0.5),
        scan_interval_seconds=0.2,
    )
    active.runner._model = RecordingModel(recorder)
    seen: list[Any] = []
    active.runner._evidence_verifier = seen.append
    loop = Loop()
    ident = fixture.identity()
    try:
        with held_reads(gate, fail_read=fail_read):
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key=key,
                )
            )
            assert gate.fetched.wait(60.0), f"the real {worker} read never returned"
            run_id = fixture.latest_run()
            revoked = False
            if revoke:
                revoked = bool(fixture.revoke_credential(ident.credential_id))
                assert revoked, "the revocation was not committed"
                gate.revoked.set()
            gate.let_go()
            handle.result(timeout=120)
        body = settle(fixture, run_id, *SETTLED, timeout=60.0)
        return {
            "run_id": run_id,
            "revocation_committed": revoked,
            "model": recorder.snapshot(),
            "verifier_executions": seen,
            "ledger": body,
            "telemetry": {
                "result_admissions": dict(
                    getattr(active.runner.telemetry, "result_admissions", {})
                ),
                "result_denials": dict(
                    getattr(active.runner.telemetry, "result_denials", {})
                ),
                "reads_entered": active.reads_entered,
                "read_denials": active.read_denials,
            },
        }
    finally:
        gate.let_go()
        loop.call(active.service.shutdown(), timeout=60.0)
        loop.close()


def _enter_model_call(
    fixture: Fix01aFixture,
    *,
    worker: str,
    revoke: bool,
    key: str,
    recorder: ModelRecorder,
    gate: ReadGate,
) -> dict[str, Any]:
    """Hold one Worker's *model response*, optionally revoke, then release and settle."""

    active = runtime(fixture)
    active.runner._model = RecordingModel(recorder)
    seen: list[Any] = []
    active.runner._evidence_verifier = seen.append
    loop = Loop()
    ident = fixture.identity()
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
        assert _wait_for_call(recorder, worker, 60.0), (
            f"the {worker} model call was never entered"
        )
        run_id = fixture.latest_run()
        revoked = False
        if revoke:
            revoked = bool(fixture.revoke_credential(ident.credential_id))
            assert revoked, "the revocation was not committed"
            gate.revoked.set()
        gate.let_go()
        handle.result(timeout=120)
        body = settle(fixture, run_id, *SETTLED, timeout=60.0)
        return {
            "run_id": run_id,
            "revocation_committed": revoked,
            "model": recorder.snapshot(),
            "verifier_executions": seen,
            "ledger": body,
        }
    finally:
        gate.let_go()
        loop.call(active.service.shutdown(), timeout=60.0)
        loop.close()


def _wait_for_call(recorder: ModelRecorder, worker: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(call["worker"] == worker for call in recorder.calls):
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------------------- the read-return boundary


@pytest.mark.parametrize("worker", WORKERS)
def test_a_late_read_result_never_reaches_the_model_after_a_committed_revocation(
    worker: str,
) -> None:
    """AC-01: a response returned after the revocation is not consumed at all.

    The revocation is committed and confirmed on the fixture's independent connection
    while the Worker still holds the response it already received, so "the affected Worker
    entered the model with a non-empty context" is decided by the release order and not by
    a race.
    """

    gate = ReadGate(worker=worker)
    recorder = ModelRecorder(gate=gate)
    with new_database(f"late_{worker}") as fixture:
        outcome = _run_review(
            fixture,
            worker=worker,
            revoke=True,
            key=f"late-{worker}",
            recorder=recorder,
            gate=gate,
        )
        model = outcome["model"]
        body = outcome["ledger"]

        assert outcome["revocation_committed"] is True
        # The affected Worker never entered the model at all.
        assert model["by_worker"][worker] == 0, model["calls"]
        # No call anywhere carried a late read payload after the revocation.
        assert model["entered_after_revocation"] == [], model["calls"]
        # The late response was not assembled into an execution the runner verified.
        assert outcome["verifier_executions"] == []
        # Nothing durable came from it either: the generation stopped as it should.
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
        assert body["error_code"] == "authorization_revoked", body
        assert body["summaries"] == 0, body
        assert not [
            item for item in body["checkpoints"] if item.startswith(f"{worker}@")
        ], body["checkpoints"]
        # The refusal is recorded at the affected Worker's own consumption boundary.
        denials = outcome["telemetry"]["result_denials"]
        assert any(site.startswith(f"{worker}.") for site in denials), denials


@pytest.mark.parametrize("worker", WORKERS)
def test_the_same_held_read_completes_normally_without_a_revocation(
    worker: str,
) -> None:
    """The control walks the identical pause/release chain and still succeeds."""

    gate = ReadGate(worker=worker)
    recorder = ModelRecorder(gate=gate)
    with new_database(f"control_{worker}") as fixture:
        outcome = _run_review(
            fixture,
            worker=worker,
            revoke=False,
            key=f"control-{worker}",
            recorder=recorder,
            gate=gate,
        )
        model = outcome["model"]
        body = outcome["ledger"]

        assert outcome["revocation_committed"] is False
        assert model["by_worker"] == {
            "policy": 1,
            "contract": 1,
            "procurement_data": 1,
        }, model["calls"]
        assert model["entered_after_revocation"] == []
        assert len(outcome["verifier_executions"]) == 1
        assert body["status"] == ReviewLifecycleStatus.PENDING_CONFIRMATION.value, body
        assert body["summaries"] == 1, body
        assert sorted(body["checkpoints"]) == [
            "contract@1",
            "policy@1",
            "procurement_data@1",
        ], body["checkpoints"]
        # Every Worker's consumption boundary really ran in the production assembly.
        admissions = outcome["telemetry"]["result_admissions"]
        for name in WORKERS:
            assert any(site.startswith(f"{name}.") for site in admissions), admissions


@pytest.mark.parametrize("worker", WORKERS)
def test_a_read_failure_after_a_revocation_stops_the_generation(worker: str) -> None:
    """AC-02: a failed read is not degraded into one summarizable branch failure."""

    gate = ReadGate(worker=worker)
    recorder = ModelRecorder(gate=gate)
    with new_database(f"readfail_{worker}") as fixture:
        outcome = _run_review(
            fixture,
            worker=worker,
            revoke=True,
            key=f"readfail-{worker}",
            recorder=recorder,
            gate=gate,
            fail_read=True,
        )
        body = outcome["ledger"]

        assert outcome["revocation_committed"] is True
        # The refused Worker never reached the model, and no call carried late payload.
        assert outcome["model"]["by_worker"][worker] == 0, outcome["model"]["calls"]
        assert outcome["model"]["entered_after_revocation"] == [], outcome["model"]
        # No branch result of the refused Worker was handed on to the aggregator.
        assert outcome["verifier_executions"] == []
        assert body["summaries"] == 0, body
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
        assert body["error_code"] == "authorization_revoked", body
        denials = outcome["telemetry"]["result_denials"]
        failure_site = f"{worker}.{FAILURE_BOUNDARY[worker]}"
        assert failure_site in denials, denials


# ------------------------------------------------------------ the model-return boundary


@pytest.mark.parametrize("worker", WORKERS)
def test_a_model_response_that_arrives_after_a_revocation_is_not_adopted(
    worker: str,
) -> None:
    """AC-03: a call issued under a live authority is not retracted; its result is dropped.

    The call is shown to have really been entered — the adapter recorded it — so "the
    attempt was booked" is never used as evidence that a request went out.  What must not
    happen is that the late response is validated, wrapped and handed on.
    """

    gate = ReadGate(worker=worker)
    recorder = ModelRecorder(gate=gate, hold_response=True)
    with new_database(f"model_{worker}") as fixture:
        outcome = _enter_model_call(
            fixture,
            worker=worker,
            revoke=True,
            key=f"model-{worker}",
            recorder=recorder,
            gate=gate,
        )
        model = outcome["model"]
        body = outcome["ledger"]

        assert outcome["revocation_committed"] is True
        entered = [call for call in model["calls"] if call["worker"] == worker]
        assert len(entered) == 1, model["calls"]
        assert entered[0]["revocation_committed"] is False, entered
        # The late response arrived after the revocation and was not adopted downstream.
        assert outcome["verifier_executions"] == []
        assert body["summaries"] == 0, body
        assert not [
            item for item in body["checkpoints"] if item.startswith(f"{worker}@")
        ], body["checkpoints"]
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
        assert body["error_code"] == "authorization_revoked", body


@pytest.mark.parametrize("worker", WORKERS)
def test_a_model_response_that_arrives_without_a_revocation_is_adopted(
    worker: str,
) -> None:
    """The model-return control: the identical hold still completes the Run."""

    gate = ReadGate(worker=worker)
    recorder = ModelRecorder(gate=gate, hold_response=True)
    with new_database(f"modelok_{worker}") as fixture:
        outcome = _enter_model_call(
            fixture,
            worker=worker,
            revoke=False,
            key=f"model-ok-{worker}",
            recorder=recorder,
            gate=gate,
        )
        body = outcome["ledger"]

        assert (
            len(
                [call for call in outcome["model"]["calls"] if call["worker"] == worker]
            )
            == 1
        )
        assert len(outcome["verifier_executions"]) == 1
        assert body["summaries"] == 1, body
        assert body["status"] == ReviewLifecycleStatus.PENDING_CONFIRMATION.value, body


@pytest.mark.parametrize("worker", WORKERS)
def test_a_failing_model_branch_never_receives_the_late_payload(worker: str) -> None:
    """AC-01/AC-02: the fact-carrying failure exit is behind the same boundary.

    ``_model_failure_result`` can carry the procurement branch's business facts, so a
    model that raises must not even be entered once the generation has lost authority.
    """

    gate = ReadGate(worker=worker)
    recorder = ModelRecorder(gate=gate, raises=True)
    with new_database(f"modelfail_{worker}") as fixture:
        outcome = _run_review(
            fixture,
            worker=worker,
            revoke=True,
            key=f"modelfail-{worker}",
            recorder=recorder,
            gate=gate,
        )
        body = outcome["ledger"]

        assert outcome["revocation_committed"] is True
        assert outcome["model"]["by_worker"][worker] == 0, outcome["model"]["calls"]
        assert outcome["model"]["entered_after_revocation"] == [], outcome["model"]
        assert outcome["verifier_executions"] == []
        assert body["summaries"] == 0, body
        assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
        assert body["error_code"] == "authorization_revoked", body
        denials = outcome["telemetry"]["result_denials"]
        assert any(site.startswith(f"{worker}.") for site in denials), denials


def test_the_failing_model_branch_still_fails_closed_while_authorized() -> None:
    """Adjacency: with a live authority the same failing model keeps its contract."""

    gate = ReadGate(worker="procurement_data")
    recorder = ModelRecorder(gate=gate, raises=True)
    with new_database("modelfail_ok") as fixture:
        outcome = _run_review(
            fixture,
            worker="procurement_data",
            revoke=False,
            key="modelfail-ok",
            recorder=recorder,
            gate=gate,
        )
        body = outcome["ledger"]
        assert outcome["model"]["by_worker"]["procurement_data"] == 1
        assert body["status"] == ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value, body
        assert body["summaries"] == 1, body


# ------------------------------------------------------------------- assembly proofs


class _UnusedRetrieval:
    """A capability port that is never reached: only the assembly is under test."""

    test_only = True

    def retrieve(self, request: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError("the retrieval port must not be reached")


class _UnusedMCP:
    """The procurement capability port, likewise only present so assembly can proceed."""

    async def collect_review_facts(self, request: Any) -> Any:  # pragma: no cover
        raise AssertionError("the MCP port must not be reached")


def test_a_worker_refuses_to_be_assembled_without_a_consumption_policy() -> None:
    """AC-05: the boundary is injected, never defaulted to a silent pass.

    Every other argument is valid here on purpose: the only reason these calls may raise is
    that the consumption policy is missing or is not a policy at all.
    """

    from procurement_review.workers import (
        ContractWorker,
        PolicyWorker,
        ProcurementAnomalyWorker,
    )

    model = RecordingModel(ModelRecorder())
    invocation = OfflineModelInvocation(receiver=model)
    with pytest.raises(TypeError, match="consumption"):
        PolicyWorker(  # type: ignore[call-arg]
            retrieval=_UnusedRetrieval(), model=model, invocation=invocation
        )
    with pytest.raises(TypeError, match="consumption"):
        ContractWorker(  # type: ignore[call-arg]
            retrieval=_UnusedRetrieval(), model=model, invocation=invocation
        )
    with pytest.raises(TypeError, match="consumption"):
        ProcurementAnomalyWorker(  # type: ignore[call-arg]
            mcp=_UnusedMCP(), model=model, invocation=invocation
        )
    # A value that merely happens to be present is refused rather than silently accepted.
    with pytest.raises(TypeError, match="consumption"):
        ContractWorker(  # type: ignore[arg-type]
            retrieval=_UnusedRetrieval(),
            model=model,
            consumption=object(),
            invocation=invocation,
        )
    # The model-call boundary is a second mandatory capability: a Worker is never
    # assembled with a consumption policy but no way to authorize and receive a call.
    with pytest.raises(TypeError, match="invocation"):
        PolicyWorker(  # type: ignore[call-arg]
            retrieval=_UnusedRetrieval(),
            model=model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
        )
    with pytest.raises(TypeError, match="invocation"):
        ProcurementAnomalyWorker(  # type: ignore[call-arg]
            mcp=_UnusedMCP(),
            model=model,
            consumption=OFFLINE_RESULT_CONSUMPTION,
        )


def test_the_offline_runner_has_no_database_identity_to_fake() -> None:
    """AC-04: the offline path runs without a Run, a generation or a review database."""

    from procurement_review.api_contracts import ReviewCreateRequest, ReviewScenario
    from procurement_review.local_runtime import LocalWorkflowRunner

    loop = Loop()
    try:
        execution = loop.call(
            LocalWorkflowRunner().run(
                run_id="run-fix08-offline",
                request=ReviewCreateRequest(
                    case_id="CASE-LOW-001",
                    scenario=ReviewScenario.LOW,
                ),
            ),
            timeout=120.0,
        )
    finally:
        loop.close()
    assert len(execution.branch_results) == 3
    assert execution.summary is not None


# ------------------------------------------------------------------ storage failures


def test_a_store_outage_at_the_consumption_boundary_stops_the_run() -> None:
    """AC-02: an unreachable authority store stops progress instead of trusting a cache.

    The outage is a fact of the database, not a race: new connections to the scenario's
    database are refused and every backend carrying the application's own tag is
    terminated, so a pool with ``pool_pre_ping`` cannot quietly reconnect.  The scenario's
    own driver and observation engines keep the connections they already opened.
    """

    import os
    from datetime import UTC, datetime

    from procurement_review.fault_hooks import FAULT_APPLICATION_NAME_ENV

    tag = f"fix08_outage_{os.urandom(3).hex()}"
    previous = os.environ.get(FAULT_APPLICATION_NAME_ENV)
    os.environ[FAULT_APPLICATION_NAME_ENV] = tag
    gate = ReadGate(worker="policy")
    recorder = ModelRecorder(gate=gate)
    loop = Loop()
    settled = False

    def restore_environment() -> None:
        if previous is None:
            os.environ.pop(FAULT_APPLICATION_NAME_ENV, None)
        else:
            os.environ[FAULT_APPLICATION_NAME_ENV] = previous

    try:
        with new_database("outage", application_name=tag) as fixture:
            active = runtime(
                fixture,
                lease_seconds=2.0,
                heartbeat_seconds=0.5,
                scan_interval_seconds=0.2,
            )
            active.runner._model = RecordingModel(recorder)
            ident = fixture.identity()
            with StoreOutage(fixture) as outage:
                with held_reads(gate):
                    handle = loop.start(
                        active.service.create_review(
                            fixture.create_request(ident),
                            registration=ident.registration,
                            scope=ident.scope,
                            context=ident.context,
                            idempotency_key="fix08-outage",
                        )
                    )
                    assert gate.fetched.wait(60.0), (
                        "the real policy read never returned"
                    )
                    run_id = fixture.latest_run()
                    # The observers must already hold their own connections before the
                    # application's database stops accepting any new one.
                    warmed = fixture.ledger(run_id)
                    outage.refuse()
                    outage.wait_until_unreachable()
                    gate.let_go()
                    handle.result(timeout=120)
                    # ``create_review`` answers as soon as the Run is dispatched, so the
                    # branches have to be waited for before "no model call happened" means
                    # anything.  This is the service's own execution boundary.
                    loop.call(
                        wait_for_execution_task(active.service, run_id, 1), timeout=90.0
                    )
                body = warmed
                deadline = time.monotonic() + 60.0
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
                outage.restore()
                # The refused response was never adopted, and no completion was invented.
                assert recorder.snapshot()["by_worker"]["policy"] == 0, recorder.calls
                assert body["summaries"] == 0, body
                assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
                assert body["error_code"] in {
                    "storage_unavailable",
                    "lease_expired",
                }, body
                loop.call(active.service.shutdown(), timeout=60.0)
                settled = True
    finally:
        try:
            loop.close()
        finally:
            restore_environment()
        if not settled:
            pytest.fail("the outage scenario did not reach a settled Run")


# ------------------------------------------------------------------------- adjacency


def test_the_reported_boundaries_are_fixed_topological_names() -> None:
    """The sites name the boundary and the Worker, never a Case, tenant or fixture."""

    assert READ_BOUNDARY == {
        "policy": "retrieval_result",
        "contract": "retrieval_result",
        "procurement_data": "procurement_result",
    }
    assert FAILURE_BOUNDARY == {
        "policy": "retrieval_read_failed",
        "contract": "retrieval_read_failed",
        "procurement_data": "procurement_read_failed",
    }
    assert MODEL_RESULT_BOUNDARY == "model_result"
    assert MODEL_FAILURE_BOUNDARY == "model_failed"
