"""FIX-09A: the model-call entry boundary, stated as a formal counterexample.

FIX-08 closed the *result* half of the model boundary: a response that comes back after the
generation lost its authority is dropped before it is validated, wrapped, checkpointed or
summarized.  It left the *entry* half open.  ``RunResultAdmission.admit_result`` returns
when its own short transaction closes, and the Worker only afterwards builds the protected
context and hands it to the model adapter, so a revocation committed in that window still
gets one call carrying the whole context.

These scenarios drive the real ``ScopedWorkflowRunner`` over a real database and record the
three observations separately (:mod:`fix09_support`):

* ``request_factory_executed`` — the Worker's request factory actually ran;
* ``model_call_entered`` — a model adapter synchronously received a request, and through
  which receive API;
* ``model_result_adopted`` — a returned result passed the result-consumption boundary.

The interleaving is not decided by a sleep.  Each scenario parks the real execution on the
product's own fault barrier, commits the revocation on the fixture's *independent* engine,
and only then releases it, so "the revocation was committed while the execution stood
before the model entry" is a fact of the run rather than a probability.

On the pre-FIX-09B implementation the first two scenarios fail: a payload does reach the
adapter (and therefore also with no boundary factory having run), which is precisely R-01.
The third scenario is the control that already holds and must keep holding.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from sqlalchemy import text

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
from fix08_support import ReadGate, new_database
from fix09_support import (
    READ_BOUNDARY,
    WORKERS,
    BackgroundRevocation,
    BoundaryRecorder,
    BoundaryRecordingModel,
    InvocationProbe,
    Observations,
    attach_invocation,
    entry_site,
    observe,
    wait_for_ungranted_lock,
)
from procurement_review.contracts import ReviewLifecycleStatus

SETTLED = (
    ReviewLifecycleStatus.PENDING_CONFIRMATION.value,
    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
    ReviewLifecycleStatus.INTERRUPTED.value,
    ReviewLifecycleStatus.FAILED.value,
    ReviewLifecycleStatus.CANCELLED.value,
)
REVOKED = "authorization_revoked"


def _boundary(worker: str) -> str:
    """The parked site that stands after one Worker's read admission has passed."""

    return f"after_result_admission:{worker}.{READ_BOUNDARY[worker]}"


def _harness(
    fixture: Fix01aFixture,
    *,
    fault: Any | None = None,
    gate: Any | None = None,
) -> tuple[Any, BoundaryRecorder, InvocationProbe, list[Any], Loop, Identity]:
    """One recording execution: real service, recording adapter, recording boundary."""

    active = runtime(fixture, fault)
    recorder = BoundaryRecorder(gate=gate, hold_response=gate is not None)
    active.runner._model = BoundaryRecordingModel(recorder)
    probe = attach_invocation(active.runner, InvocationProbe())
    seen: list[Any] = []
    active.runner._evidence_verifier = seen.append
    return active, recorder, probe, seen, Loop(), fixture.identity()


# ------------------------------------- entry: a committed revocation stops the call


@pytest.mark.parametrize("worker", WORKERS)
def test_a_revocation_committed_before_the_model_entry_never_reaches_the_adapter(
    worker: str,
) -> None:
    """AC-01: the entry is the linearization point, not the check that preceded it.

    The execution is parked *after* its read-result admission passed and before the model
    entry, exactly the window FIX-08 left open.  A real credential revocation is committed
    by the fixture's independent connection in that window.  Under the contract, the
    Worker's request must never be built and the adapter must never receive anything.
    """

    boundary = _boundary(worker)
    directory = barrier_dir(f"fix09_entry_{worker}")
    fault = parked_fault(directory, boundary)
    with new_database(f"entry_{worker}") as fixture:
        active, recorder, probe, seen, loop, ident = _harness(fixture, fault=fault)
        run_id = ""
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key=f"fix09-entry-{worker}",
                )
            )
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            assert fixture.revoke_credential(ident.credential_id), (
                "the revocation was not committed"
            )
            release_barrier(directory, boundary)
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            observations = observe(active.runner, recorder, probe)

            assert observations.model_call_entered[worker] == 0, _report(
                "a model adapter received a request after the revocation committed",
                observations,
            )
            assert observations.entered_with_protected_payload[worker] == 0, _report(
                "protected context or facts reached a model adapter after the revocation",
                observations,
            )
            assert observations.request_factory_executed == len(
                observations.factory_sites
            )
            assert entry_site(worker) not in observations.factory_sites, _report(
                "the revoked Worker's request factory ran", observations
            )
            assert observations.model_result_adopted[worker] == 0, _report(
                "a returned model result was adopted after the revocation", observations
            )
            assert seen == [], observations
            # A sibling that reached its own model while the generation was still live may
            # have committed; the parked Worker may not have anything derived from its read.
            assert not [
                item for item in body["checkpoints"] if item.startswith(f"{worker}@")
            ], body["checkpoints"]
            assert body["summaries"] == 0, body
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] == REVOKED, body
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def _report(headline: str, observations: Observations) -> str:
    """One failure message carrying all three observations, never just one counter."""

    return (
        f"{headline}\n"
        f"  boundary_attached={observations.boundary_attached}\n"
        f"  request_factory_executed={observations.request_factory_executed} "
        f"(sites={observations.factory_sites})\n"
        f"  model_call_entered={observations.model_call_entered} "
        f"(receive_apis={observations.receive_apis})\n"
        f"  entered_with_protected_payload="
        f"{observations.entered_with_protected_payload}\n"
        f"  model_result_adopted={observations.model_result_adopted}\n"
        f"  entries={observations.entries}"
    )


# ------------------------------------- the authorized control: the factory and the entry agree


def test_every_payload_the_adapter_received_was_built_inside_the_call_boundary() -> (
    None
):
    """AC-01/AC-05: no protected payload may reach the adapter without the boundary.

    The control Run is never revoked, so all three Workers complete.  The count of adapter
    receipts and the count of request-factory executions come from two different places and
    must agree: an entry that no boundary produced is a context handed to the model outside
    the authorization step, whatever the count of attempts or telemetry events says.
    """

    with new_database("fix09_entry_control") as fixture:
        active, recorder, probe, _seen, loop, ident = _harness(fixture)
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="fix09-entry-control",
                )
            )
            handle.result(timeout=120)
            run_id = fixture.latest_run()
            body = settle(
                fixture,
                run_id,
                ReviewLifecycleStatus.PENDING_CONFIRMATION.value,
                ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
                timeout=60.0,
            )
            observations = observe(active.runner, recorder, probe)
            entered_total = sum(observations.model_call_entered.values())

            assert observations.model_call_entered == {
                "policy": 1,
                "contract": 1,
                "procurement_data": 1,
            }, _report(
                "the control Run did not enter each Worker's model once", observations
            )
            assert observations.entered_with_protected_payload == {
                "policy": 1,
                "contract": 1,
                "procurement_data": 1,
            }, _report("the control Run lost a Worker's context", observations)
            assert probe.factory_executions == entered_total, _report(
                "a payload reached the adapter without the call boundary building it",
                observations,
            )
            assert observations.model_result_adopted == {
                "policy": 1,
                "contract": 1,
                "procurement_data": 1,
            }, _report(
                "the control Run did not adopt each returned result", observations
            )
            assert body["summaries"] == 1, body
            assert len(body["checkpoints"]) == 3, body["checkpoints"]
        finally:
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


# ------------------------------------- receipt: the call may be late, its result may not land


def test_a_call_received_before_the_revocation_is_late_and_never_adopted() -> None:
    """AC-01/AC-02: the adapter receipt is the point of no return, and only for one call.

    The model response is held after the adapter has received the request, so the
    revocation commits while the call is genuinely in flight.  That call is allowed to
    finish — it was received under a live authorization — but its result must be dropped,
    and the revocation must not wait for the model, nor for any database lock the receipt
    left behind.
    """

    gate = ReadGate(worker="contract")
    with new_database("fix09_receipt") as fixture:
        active, recorder, probe, _seen, loop, ident = _harness(fixture, gate=gate)
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="fix09-receipt",
                )
            )
            assert _wait_for_worker(recorder, "contract", 60.0), (
                f"the contract model call was never received: {recorder.public_entries()}"
            )
            run_id = fixture.latest_run()
            # While the call is in flight, a competing transaction must be able to take the
            # row every live-authority check linearizes on: that can only take long if some
            # branch holds its share lock across the model.
            blocked_for = _time_to_take_authority_row(fixture, ident)
            assert blocked_for < 3.0, (
                f"the authority row stayed locked for {blocked_for:.2f}s during the model"
            )
            started = time.monotonic()
            assert fixture.revoke_credential(ident.credential_id)
            committed = time.monotonic() - started
            assert committed < 3.0, (
                f"the revocation waited {committed:.2f}s for the model"
            )
            assert not gate.release.is_set()
            gate.revoked.set()
            gate.let_go()
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            observations = observe(active.runner, recorder, probe)

            assert observations.model_call_entered["contract"] == 1, _report(
                "the received call was not recorded", observations
            )
            assert observations.model_result_adopted["contract"] == 0, _report(
                "the late response was adopted after the revocation", observations
            )
            assert body["summaries"] == 0, body
            assert not [
                item for item in body["checkpoints"] if item.startswith("contract@")
            ], body["checkpoints"]
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] == REVOKED, body
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def _wait_for_worker(recorder: BoundaryRecorder, worker: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if recorder.by_worker()[worker]:
            return True
        time.sleep(0.02)
    return False


def _time_to_take_authority_row(fixture: Fix01aFixture, ident: Identity) -> float:
    """Seconds a competing transaction needs for the authority linearization row."""

    started = time.monotonic()
    with fixture.driver.transaction() as connection:
        connection.execute(
            text(
                "SELECT revision FROM review_authorization_revision "
                "WHERE tenant_id = :t FOR UPDATE"
            ),
            {"t": ident.context.tenant_id},
        )
    return time.monotonic() - started


# ------------------------- FIX-09B: the entry is one database linearization step


@pytest.mark.parametrize("worker", WORKERS)
def test_the_adapter_receipt_happens_inside_the_admitting_transaction(
    worker: str,
) -> None:
    """AC-01: the receipt and the authorization re-read share one short transaction.

    The execution is parked after the adapter has synchronously received the request and
    before that transaction has been released.  A real revocation is then issued on the
    fixture's independent engine: it must be *waiting for a database lock* — the fact that
    the receipt really is inside the transaction — and it must complete right after the
    transaction closes, without waiting for the model at all.
    """

    barrier = f"after_model_call_received:{entry_site(worker)}"
    directory = barrier_dir(f"fix09_receipt_{worker}")
    fault = parked_fault(directory, barrier)
    with new_database(f"receipt_{worker}") as fixture:
        active, recorder, probe, _seen, loop, ident = _harness(fixture, fault=fault)
        revocation = BackgroundRevocation(fixture, ident.credential_id)
        run_id = ""
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key=f"fix09-receipt-{worker}",
                )
            )
            wait_barrier(directory, barrier)
            run_id = fixture.latest_run()
            observations = observe(active.runner, recorder, probe)

            # The adapter has the request; the transaction that admitted it is still open.
            assert observations.model_call_entered[worker] == 1, _report(
                "the adapter did not receive the request inside the boundary",
                observations,
            )
            assert observations.entered_with_protected_payload[worker] == 1, _report(
                "the received request carried no protected context", observations
            )
            assert observations.factory_sites.count(entry_site(worker)) == 1, _report(
                "the parked Worker's request factory did not run exactly once inside "
                "the boundary",
                observations,
            )

            revocation.start()
            waiting = wait_for_ungranted_lock(fixture, "fix01a_driver")
            assert waiting >= 1, waiting
            assert not revocation.committed.is_set(), (
                "the revocation committed while the admitting transaction was still open"
            )

            release_barrier(directory, barrier)
            revocation.join(timeout=30.0)
            assert revocation.revoked, "the revocation did not commit"
            assert revocation.seconds < 3.0, (
                f"the revocation waited {revocation.seconds:.2f}s for the short transaction"
            )

            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            settled = observe(active.runner, recorder, probe)

            # The call was received while the generation was still live, so it may finish;
            # its answer may not be adopted, and the Worker may not have checkpointed.
            assert settled.model_call_entered[worker] == 1, _report(
                "the received call was not exactly one", settled
            )
            assert settled.model_result_adopted[worker] == 0, _report(
                "the late answer was adopted after the revocation", settled
            )
            assert not [
                item for item in body["checkpoints"] if item.startswith(f"{worker}@")
            ], body["checkpoints"]
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] == REVOKED, body
        finally:
            release_barrier(directory, barrier)
            revocation.join(timeout=30.0)
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


@pytest.mark.parametrize("worker", WORKERS)
def test_the_live_authority_is_re_read_before_the_request_exists(worker: str) -> None:
    """AC-01/AC-05: nothing protected exists before the authorization step has passed.

    The execution is parked after the live authority has been re-read and its share locks
    are held, and before the request factory has run.  The revocation issued at that moment
    must be waiting for a lock, and no request may exist yet — which is what makes "the
    factory only runs inside an authorized boundary" an observable fact rather than a
    reading of the source.
    """

    barrier = f"during_model_entry_authorization:{entry_site(worker)}"
    directory = barrier_dir(f"fix09_authorize_{worker}")
    fault = parked_fault(directory, barrier)
    with new_database(f"authorize_{worker}") as fixture:
        active, recorder, probe, _seen, loop, ident = _harness(fixture, fault=fault)
        revocation = BackgroundRevocation(fixture, ident.credential_id)
        run_id = ""
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key=f"fix09-authorize-{worker}",
                )
            )
            wait_barrier(directory, barrier)
            run_id = fixture.latest_run()
            observations = observe(active.runner, recorder, probe)

            assert entry_site(worker) not in observations.factory_sites, _report(
                "the parked Worker's request was built before its authorization step "
                "had passed",
                observations,
            )
            assert observations.model_call_entered[worker] == 0, _report(
                "the adapter received a request before the authorization step passed",
                observations,
            )

            revocation.start()
            assert wait_for_ungranted_lock(fixture, "fix01a_driver") >= 1
            assert not revocation.committed.is_set(), (
                "the revocation committed while the authorization locks were held"
            )

            # The authorization passed first, so this call is received and allowed to
            # finish; the revocation that was waiting then commits and the answer is dropped.
            release_barrier(directory, barrier)
            revocation.join(timeout=30.0)
            assert revocation.revoked, "the revocation did not commit"
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            settled = observe(active.runner, recorder, probe)

            assert settled.factory_sites.count(entry_site(worker)) == 1, _report(
                "the factory did not run exactly once after the authorization passed",
                settled,
            )
            assert settled.model_call_entered[worker] == 1, _report(
                "the authorized call was not received exactly once", settled
            )
            assert settled.model_result_adopted[worker] == 0, _report(
                "the answer was adopted after the revocation", settled
            )
            assert body["summaries"] == 0, body
        finally:
            release_barrier(directory, barrier)
            revocation.join(timeout=30.0)
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()
