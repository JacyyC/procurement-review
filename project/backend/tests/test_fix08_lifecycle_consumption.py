"""FIX-08C: the result-consumption boundary across the execution's whole lifecycle.

FIX-08B proved the boundary itself.  This module proves it holds for every way an execution
can lose the right to keep going, for the interleavings that sit *around* a passed
admission, and for the lifecycle contracts the fix must not have changed.

Every scenario drives the real ``ScopedWorkflowRunner`` over a real database and changes the
authorization state on the fixture's own independent connection, so no assertion depends on
an in-process flag, a static ``RuntimeScope`` or a test-only boolean.
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
from fix08_support import (
    ALL_WORKERS,
    RETRIEVAL_BOUNDARY,
    ModelRecorder,
    ReadGate,
    RecordingModel,
    held_reads,
    new_database,
)
from procurement_review.api_contracts import ApprovalActionType, HumanActionRequestV2
from procurement_review.contracts import ReviewLifecycleStatus

SETTLED = (
    ReviewLifecycleStatus.PENDING_CONFIRMATION.value,
    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
    ReviewLifecycleStatus.INTERRUPTED.value,
    ReviewLifecycleStatus.FAILED.value,
    ReviewLifecycleStatus.CANCELLED.value,
)

#: Every real way the acting authority can lapse, applied on the independent connection.
LAPSED_AUTHORITY = {
    "credential_revoked": lambda fixture, ident: fixture.revoke_credential(
        ident.credential_id
    ),
    "credential_expired": lambda fixture, ident: fixture.expire_credential(
        ident.credential_id
    ),
    "subject_disabled": lambda fixture, ident: fixture.disable_subject(
        ident.context.subject_id
    ),
    "subject_unroled": lambda fixture, ident: _unrole_subject(
        fixture, ident.context.subject_id
    ),
    "case_grant_revoked": lambda fixture, ident: fixture.revoke_case_grant(
        ident.context.subject_id, ident.registration.case_id
    ),
    "authorization_revision_changed": lambda fixture, ident: (
        fixture.bump_authorization_revision()
    ),
}


def _unrole_subject(fixture: Fix01aFixture, subject_id: str) -> int:
    """Remove every role of one subject on the independent connection."""

    with fixture.driver.transaction() as connection:
        result = connection.execute(
            text("DELETE FROM review_subject_role WHERE subject_id = :s"),
            {"s": subject_id},
        )
    return int(result.rowcount)


def _base(
    fixture: Fix01aFixture,
    *,
    worker: str,
    recorder: ModelRecorder,
) -> tuple[Any, Loop, Identity, list[Any]]:
    active = runtime(fixture)
    active.runner._model = RecordingModel(recorder)
    seen: list[Any] = []
    active.runner._evidence_verifier = seen.append
    return active, Loop(), fixture.identity(), seen


# ------------------------------------------------------- V-C-01: every lapse is caught


@pytest.mark.parametrize("reason", sorted(LAPSED_AUTHORITY))
def test_every_lapsed_authority_is_caught_before_the_result_is_consumed(
    reason: str,
) -> None:
    """AC-02: the boundary re-reads the live authority, not a cached decision.

    The same held response is released after each kind of real database change, so all six
    reasons have to be seen at the consumption point itself.  Nothing here mutates the
    ``RuntimeScope``: the scope is unchanged in every one of these scenarios.
    """

    # Every Worker's real read is held, so no branch can reach a commit before the others
    # are refused: the whole execution is decided at the consumption boundary.
    gate = ReadGate(worker=ALL_WORKERS)
    recorder = ModelRecorder(gate=gate)
    with new_database(f"lapse_{reason}") as fixture:
        active, loop, ident, seen = _base(fixture, worker="policy", recorder=recorder)
        try:
            with held_reads(gate):
                handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(ident),
                        registration=ident.registration,
                        scope=ident.scope,
                        context=ident.context,
                        idempotency_key=f"lapse-{reason}",
                    )
                )
                assert gate.fetched.wait(60.0), (
                    f"the real reads never returned: {sorted(gate.arrived)}"
                )
                run_id = fixture.latest_run()
                LAPSED_AUTHORITY[reason](fixture, ident)
                gate.let_go()
                handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)

            # No Worker consumed its already-returned result, and nothing was adopted.
            assert recorder.snapshot()["by_worker"] == {
                "policy": 0,
                "contract": 0,
                "procurement_data": 0,
            }, recorder.calls
            assert seen == []
            assert body["checkpoints"] == [], body["checkpoints"]
            assert body["summaries"] == 0, body
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] == "authorization_revoked", body
            # At least one refusal is recorded at a Worker's own consumption boundary.  Which
            # branch observes the lapse first is not fixed — a sibling's commit guard may stop
            # the generation first — but every recorded refusal is a consumption refusal.
            denials = active.runner.telemetry.result_denials
            assert denials, (reason, active.runner.telemetry.as_summary())
            assert all(
                site.split(".", 1)[0] in {"policy", "contract", "procurement_data"}
                for site in denials
            ), denials
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


# -------------------------------------- V-C-02: the window around a passed admission


def test_an_admission_that_passed_before_the_revocation_still_drops_its_result() -> (
    None
):
    """AC-03, read through FIX-09: a passed admission never authorizes the next step.

    The execution is parked *after* its read-return admission passed, so the read really
    was issued while the generation was still authorized.  FIX-08 stated the second half of
    the contract here — the response arriving after the revocation is not adopted.  FIX-09
    moved the linearization point for the *model entry* into the adapter's synchronous
    receipt inside the live-authority transaction, so this window may no longer produce a
    call at all: a passed check, a booked attempt or a held context is not evidence that the
    next protected step is still allowed.  The three observations are recorded separately in
    ``test_fix09_model_invocation_boundary.py``.
    """

    directory = barrier_dir("after_admission")
    barrier = f"after_result_admission:policy.{RETRIEVAL_BOUNDARY}"
    fault = parked_fault(directory, barrier)
    recorder = ModelRecorder()
    with new_database("after_admission") as fixture:
        active = runtime(fixture, fault)
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
                    idempotency_key="after-admission",
                )
            )
            wait_barrier(directory, barrier)
            run_id = fixture.latest_run()
            assert gate_committed(fixture, ident), "the revocation was not committed"
            release_barrier(directory, barrier)
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)

            entered = [call for call in recorder.calls if call["worker"] == "policy"]
            # The revocation had already committed when the admission returned: no call may
            # follow from it, and nothing derived from the held read may be carried on.
            assert entered == [], recorder.calls
            assert seen == []
            assert body["summaries"] == 0, body
            assert not [
                item for item in body["checkpoints"] if item.startswith("policy@")
            ]
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] == "authorization_revoked", body
            # The refused generation booked nothing beyond the read it was already in.
            assert [
                item for item in body["attempts"] if item.startswith("1/policy/")
            ] == ["1/policy/1"], body["attempts"]
        finally:
            release_barrier(directory, barrier)
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def gate_committed(fixture: Fix01aFixture, ident: Identity) -> bool:
    return bool(fixture.revoke_credential(ident.credential_id))


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


def test_a_revocation_is_not_blocked_by_an_in_flight_model() -> None:
    """AC-03: no database row lock is held across the model call.

    The model response is held while the driver commits the revocation, so a lock held
    across the model would show up as a slow (or deadlocked) revocation and as granted
    locks on this project's tables held by the application.
    """

    gate = ReadGate(worker="contract")
    recorder = ModelRecorder(gate=gate, hold_response=True)
    with new_database("slow_model") as fixture:
        active = runtime(fixture)
        active.runner._model = RecordingModel(recorder)
        loop = Loop()
        ident = fixture.identity()
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="slow-model",
                )
            )
            assert _wait_for_call(recorder, "contract", 60.0), (
                "the contract model call was never entered"
            )
            run_id = fixture.latest_run()
            # While the model is in flight, a competing transaction can take the tenant
            # revision row the authority check linearizes on.  That can only take long if
            # some branch held its share lock across the model call.
            blocked_for = _time_to_take_authority_row(fixture, ident)
            assert blocked_for < 3.0, (
                f"the authority row stayed locked for {blocked_for:.2f}s during the model"
            )
            # The revocation commits while the model is still parked: the response is
            # released only afterwards, so the commit did not wait for it.
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

            assert recorder.snapshot()["by_worker"]["contract"] == 1, recorder.calls
            assert body["summaries"] == 0, body
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] == "authorization_revoked", body
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


# ------------------------------------------------ V-C-02: cancellation and takeover


def test_cancelling_a_run_does_not_adopt_a_held_read_result() -> None:
    """AC-02/AC-04: a cancelled Run is terminal, so its in-flight read is not consumed."""

    gate = ReadGate(worker="policy")
    recorder = ModelRecorder(gate=gate)
    with new_database("cancel_held") as fixture:
        active, loop, ident, seen = _base(fixture, worker="policy", recorder=recorder)
        try:
            with held_reads(gate):
                handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(ident),
                        registration=ident.registration,
                        scope=ident.scope,
                        context=ident.context,
                        idempotency_key="cancel-held",
                    )
                )
                assert gate.fetched.wait(60.0), "the real policy read never returned"
                run_id = fixture.latest_run()
                current = active.service.get_review(run_id, context=ident.context)
                cancelled = active.service.apply_action(
                    run_id=run_id,
                    action=ApprovalActionType.CANCEL,
                    request=HumanActionRequestV2(
                        expected_version=current.revision,
                        idempotency_key="cancel-held-action",
                    ),
                    context=ident.context,
                )
                assert cancelled.status is ReviewLifecycleStatus.CANCELLED
                gate.let_go()
                handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)

            assert recorder.snapshot()["by_worker"]["policy"] == 0, recorder.calls
            assert seen == []
            assert body["summaries"] == 0, body
            assert body["status"] == ReviewLifecycleStatus.CANCELLED.value, body
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def test_a_superseded_generation_cannot_adopt_a_held_read_result() -> None:
    """AC-02/AC-03: an expired lease ends the generation; the late result is not consumed."""

    gate = ReadGate(worker="policy")
    recorder = ModelRecorder(gate=gate)
    with new_database("lease_held") as fixture:
        active = runtime(
            fixture,
            lease_seconds=2.0,
            heartbeat_seconds=1000.0,
            scan_interval_seconds=0.2,
        )
        active.runner._model = RecordingModel(recorder)
        seen: list[Any] = []
        active.runner._evidence_verifier = seen.append
        loop = Loop()
        ident = fixture.identity()
        try:
            with held_reads(gate):
                handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(ident),
                        registration=ident.registration,
                        scope=ident.scope,
                        context=ident.context,
                        idempotency_key="lease-held",
                    )
                )
                assert gate.fetched.wait(60.0), "the real policy read never returned"
                run_id = fixture.latest_run()
                body = fixture.ledger(run_id)
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    fixture.observer.scan_expired_leases(
                        now=_now(),
                        events_factory=lambda _r: (),
                        audit_factory=lambda _r: None,
                    )
                    body = fixture.ledger(run_id)
                    if body["status"] != ReviewLifecycleStatus.PROCESSING.value:
                        break
                    time.sleep(0.2)
                assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
                gate.let_go()
                handle.result(timeout=120)
            settled = settle(fixture, run_id, *SETTLED, timeout=60.0)

            assert recorder.snapshot()["by_worker"]["policy"] == 0, recorder.calls
            assert seen == []
            assert settled["summaries"] == 0, settled
            # The other two branches committed while the lease was still live; the refused
            # result of the expired generation is what must not appear.
            assert not [
                item for item in settled["checkpoints"] if item.startswith("policy@")
            ], settled["checkpoints"]
            assert settled["status"] == ReviewLifecycleStatus.INTERRUPTED.value, settled
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def _now() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)


# ------------------------------------------- V-C-03: budgets and retry discipline


def test_a_refused_consumption_is_a_run_level_stop_not_a_retried_branch() -> None:
    """AC-04: losing the authority is not a transient read failure.

    The affected Worker books exactly one attempt and is never retried, because the
    generation stops rather than summarizing a fail-closed branch.
    """

    gate = ReadGate(worker="policy")
    recorder = ModelRecorder(gate=gate)
    with new_database("no_retry") as fixture:
        active, loop, ident, _seen = _base(fixture, worker="policy", recorder=recorder)
        try:
            with held_reads(gate):
                handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(ident),
                        registration=ident.registration,
                        scope=ident.scope,
                        context=ident.context,
                        idempotency_key="no-retry",
                    )
                )
                assert gate.fetched.wait(60.0), "the real policy read never returned"
                run_id = fixture.latest_run()
                assert fixture.revoke_credential(ident.credential_id)
                gate.let_go()
                handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)

            assert active.runner.telemetry.retries == 0, (
                active.runner.telemetry.as_summary()
            )
            attempts = [
                item for item in body["attempts"] if item.startswith("1/policy/")
            ]
            assert attempts == ["1/policy/1"], body["attempts"]
            assert body["error_code"] == "authorization_revoked", body
            assert body["summaries"] == 0, body
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def test_the_consumption_boundary_adds_no_read_of_its_own() -> None:
    """AC-03/AC-05: a consumption check is not a read and does not move the read counters.

    The control Run performs exactly the same reads it did before the fix, and the
    consumption counters are disjoint from the read counters.
    """

    gate = ReadGate(worker="policy")
    recorder = ModelRecorder(gate=gate)
    with new_database("no_extra_read") as fixture:
        active, loop, ident, _seen = _base(fixture, worker="policy", recorder=recorder)
        try:
            with held_reads(gate):
                handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(ident),
                        registration=ident.registration,
                        scope=ident.scope,
                        context=ident.context,
                        idempotency_key="no-extra-read",
                    )
                )
                assert gate.fetched.wait(60.0), "the real policy read never returned"
                run_id = fixture.latest_run()
                gate.let_go()
                handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)

            telemetry = active.runner.telemetry
            assert body["status"] == ReviewLifecycleStatus.PENDING_CONFIRMATION.value, (
                body
            )
            assert telemetry.reads_entered == {
                "contract": 1,
                "policy": 1,
                "procurement_data": 3,
            }, telemetry.reads_entered
            assert telemetry.preload_reads == {
                "evidence_index": 1,
                "procurement_records": 1,
                "retrieval_index": 1,
            }, telemetry.preload_reads
            # Consumption admissions are counted apart from reads and never add one.
            assert set(telemetry.result_admissions) == {
                f"policy.{RETRIEVAL_BOUNDARY}",
                "policy.model_result",
                f"contract.{RETRIEVAL_BOUNDARY}",
                "contract.model_result",
                "procurement_data.procurement_result",
                "procurement_data.model_result",
            }, telemetry.result_admissions
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


# ------------------------------------------ V-C-02: two Runs do not share an identity


def test_two_concurrent_runs_do_not_share_a_consumption_identity() -> None:
    """AC-03/AC-04: revoking one Run's credential does not touch the other's execution.

    Both Runs execute in the same service over the same store, so a boundary that carried a
    process-wide identity (or one Run's identity into the other) would show up here as the
    healthy Run losing its model context or the revoked Run keeping it.
    """

    gate = ReadGate(worker="policy", hold_once=True)
    recorder = ModelRecorder(gate=gate)
    with new_database("two_runs") as fixture:
        active = runtime(fixture)
        active.runner._model = RecordingModel(recorder)
        loop = Loop()
        held_ident = fixture.identity(case_id="CASE-LOW-001")
        healthy_ident = fixture.identity(case_id="CASE-HIGH-001")
        try:
            with held_reads(gate):
                held_handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(held_ident),
                        registration=held_ident.registration,
                        scope=held_ident.scope,
                        context=held_ident.context,
                        idempotency_key="two-runs-held",
                    )
                )
                assert gate.fetched.wait(60.0), "the real policy read never returned"
                held_run = fixture.latest_run()
                healthy_handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(healthy_ident),
                        registration=healthy_ident.registration,
                        scope=healthy_ident.scope,
                        context=healthy_ident.context,
                        idempotency_key="two-runs-healthy",
                    )
                )
                healthy_handle.result(timeout=120)
                healthy_run = _latest_run_for(fixture, "CASE-HIGH-001")
                healthy_body = settle(
                    fixture,
                    healthy_run,
                    ReviewLifecycleStatus.PENDING_CONFIRMATION.value,
                    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
                    timeout=60.0,
                )
                assert fixture.revoke_credential(held_ident.credential_id)
                gate.let_go()
                held_handle.result(timeout=120)
            held_body = settle(fixture, held_run, *SETTLED, timeout=60.0)

            # The healthy Run kept its whole model context.  The held Run's other two
            # branches were never held and may legitimately have run before the revocation;
            # what it may not have is a policy call, because that is the read it was parked
            # on, and no call anywhere may have been entered after the revocation.
            calls = recorder.snapshot()["calls"]
            healthy = [call for call in calls if call["case_id"] == "CASE-HIGH-001"]
            parked = [call for call in calls if call["case_id"] == "CASE-LOW-001"]
            assert sorted(call["worker"] for call in healthy) == [
                "contract",
                "policy",
                "procurement_data",
            ], healthy
            assert [
                call["worker"] for call in parked if call["worker"] == "policy"
            ] == [], parked
            assert [call for call in calls if call["revocation_committed"]] == [], calls
            assert healthy_body["summaries"] == 1, healthy_body
            assert len(healthy_body["checkpoints"]) == 3, healthy_body
            assert held_body["summaries"] == 0, held_body
            assert held_body["status"] == ReviewLifecycleStatus.INTERRUPTED.value
            assert held_body["error_code"] == "authorization_revoked", held_body
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def _latest_run_for(fixture: Fix01aFixture, case_id: str) -> str:
    with fixture.probe.transaction() as connection:
        return str(
            connection.execute(
                text(
                    "SELECT run_id FROM review_run WHERE case_id = :c "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"c": case_id},
            ).scalar_one()
        )


# ------------------------------------- V-C-03: the attempt timeout keeps its contract


def test_a_timed_out_attempt_is_retried_within_budget_and_adopts_nothing_late() -> None:
    """AC-04: the fixed attempt budget and retry discipline are unchanged by the boundary.

    The first policy attempt is injected as slower than the per-attempt timeout, so it is
    cancelled inside the attempt while the read thread may still be running.  The late
    result must not be adopted, the retry must stay inside the fixed budget, and the Run
    must still complete normally.
    """

    from procurement_review.fault_hooks import FaultController

    slow = FaultController(enabled=True, slow={"policy": {1: 6.0}})
    recorder = ModelRecorder()
    with new_database("timeout_retry") as fixture:
        active = runtime(fixture, slow)
        active.runner._model = RecordingModel(recorder)
        loop = Loop()
        ident = fixture.identity()
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="timeout-retry",
                )
            )
            handle.result(timeout=120)
            run_id = fixture.latest_run()
            body = settle(
                fixture,
                run_id,
                ReviewLifecycleStatus.PENDING_CONFIRMATION.value,
                ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
                timeout=120.0,
            )

            assert active.runner.telemetry.timeouts == 1, (
                active.runner.telemetry.as_summary()
            )
            assert active.runner.telemetry.retries == 1, (
                active.runner.telemetry.as_summary()
            )
            assert [
                item for item in body["attempts"] if item.startswith("1/policy/")
            ] == ["1/policy/1", "1/policy/2"], body["attempts"]
            # One policy model call, after the successful attempt — never the late one.
            assert recorder.snapshot()["by_worker"] == {
                "policy": 1,
                "contract": 1,
                "procurement_data": 1,
            }, recorder.calls
            assert body["summaries"] == 1, body
            assert len(body["checkpoints"]) == 3, body
        finally:
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()
