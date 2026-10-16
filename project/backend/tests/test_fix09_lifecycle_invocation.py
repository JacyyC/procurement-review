"""FIX-09C: the model-call boundary across every way an execution can change state.

FIX-09B proved the two linearization orders.  This module proves the boundary holds for
every real way an execution loses the right to enter the model, for the model outcomes that
must stay fail-closed, for cancellation, lease, generation and concurrent Runs, and for an
authority store that is genuinely unreachable.

Every scenario drives the real ``ScopedWorkflowRunner`` over a real database and changes the
authorization state on the fixture's own independent connection, so no assertion depends on
an in-process flag, a static ``RuntimeScope`` or a test-only boolean.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import text

from fix01a_support import (
    Fix01aFixture,
    Loop,
    barrier_dir,
    parked_fault,
    release_barrier,
    runtime,
    settle,
    wait_barrier,
)
from fix08_support import (
    RETRIEVAL_BOUNDARY,
    ReadGate,
    StoreOutage,
    held_reads,
    new_database,
)
from fix09_support import (
    WORKERS,
    BackgroundRevocation,
    BoundaryRecorder,
    BoundaryRecordingModel,
    InvocationProbe,
    attach_invocation,
    entry_denials,
    entry_site,
    observe,
)
from procurement_review.api_contracts import ApprovalActionType, HumanActionRequestV2
from procurement_review.contracts import ReviewLifecycleStatus
from procurement_review.execution_guards import ExecutionInvalidated

SETTLED = (
    ReviewLifecycleStatus.PENDING_CONFIRMATION.value,
    ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value,
    ReviewLifecycleStatus.INTERRUPTED.value,
    ReviewLifecycleStatus.FAILED.value,
    ReviewLifecycleStatus.CANCELLED.value,
)
REVOKED = "authorization_revoked"

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


def _now() -> Any:
    return datetime.now(UTC)


def _harness(
    fixture: Fix01aFixture,
    *,
    fault: Any | None = None,
    recorder: BoundaryRecorder | None = None,
    gate: Any | None = None,
    **runtime_kwargs: Any,
) -> tuple[Any, BoundaryRecorder, InvocationProbe, list[Any], Loop, Any]:
    """One recording execution: real service, recording adapter, recording boundary."""

    active = runtime(fixture, fault, **runtime_kwargs)
    observed = recorder or BoundaryRecorder(gate=gate, hold_response=gate is not None)
    active.runner._model = BoundaryRecordingModel(observed)
    probe = attach_invocation(active.runner, InvocationProbe())
    seen: list[Any] = []
    active.runner._evidence_verifier = seen.append
    return active, observed, probe, seen, Loop(), fixture.identity()


# --------------------------------------------- V-C-01: every lapse is refused at the entry


@pytest.mark.parametrize("reason", sorted(LAPSED_AUTHORITY))
def test_every_lapsed_authority_is_refused_at_the_model_entry(reason: str) -> None:
    """AC-03: the entry re-reads the live authority; a cached decision is not enough.

    One Worker's read admission is parked after it passed, so the lapse commits in the
    window FIX-08 left open.  The Worker's request must then never be built, the adapter
    must never receive anything, and the refusal must be recorded at the entry boundary
    itself — not merely as a result that failed to be stored.
    """

    boundary = f"after_result_admission:policy.{RETRIEVAL_BOUNDARY}"
    directory = barrier_dir(f"fix09c_lapse_{reason}")
    fault = parked_fault(directory, boundary)
    with new_database(f"c_lapse_{reason}") as fixture:
        active, recorder, probe, seen, loop, ident = _harness(fixture, fault=fault)
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key=f"fix09c-lapse-{reason}",
                )
            )
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            LAPSED_AUTHORITY[reason](fixture, ident)
            release_barrier(directory, boundary)
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            observations = observe(active.runner, recorder, probe)

            assert observations.model_call_entered["policy"] == 0, observations
            assert entry_site("policy") not in observations.factory_sites, observations
            assert observations.model_result_adopted["policy"] == 0, observations
            # Result consumption can now reject earlier. Independently exercise the
            # captured production entry too, preserving its refusal obligation.
            forbidden_factory = []
            with pytest.raises(ExecutionInvalidated):
                probe.invocations[0].enter(
                    site=entry_site("policy"),
                    request_factory=lambda: forbidden_factory.append("built"),
                )
            assert forbidden_factory == []
            assert entry_denials(active.runner).get(entry_site("policy"), 0) >= 1, (
                active.runner.telemetry.model_entry_denials
            )
            assert seen == []
            assert body["summaries"] == 0, body
            assert not [
                item for item in body["checkpoints"] if item.startswith("policy@")
            ], body["checkpoints"]
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            # Every authority lapse ends the generation under the same contract reason.
            assert body["error_code"] == REVOKED, body
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


# ------------------------------------------------ V-C-02: cancel, lease and generation


def test_a_cancelled_run_never_enters_the_model() -> None:
    """AC-02/AC-03: a terminal Run is terminal; its in-flight read may not reach a model."""

    boundary = f"after_result_admission:policy.{RETRIEVAL_BOUNDARY}"
    directory = barrier_dir("fix09c_cancel")
    fault = parked_fault(directory, boundary)
    with new_database("c_cancel") as fixture:
        active, recorder, probe, _seen, loop, ident = _harness(fixture, fault=fault)
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="fix09c-cancel",
                )
            )
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            current = active.service.get_review(run_id, context=ident.context)
            cancelled = active.service.apply_action(
                run_id=run_id,
                action=ApprovalActionType.CANCEL,
                request=HumanActionRequestV2(
                    expected_version=current.revision,
                    idempotency_key="fix09c-cancel-action",
                ),
                context=ident.context,
            )
            assert cancelled.status is ReviewLifecycleStatus.CANCELLED
            release_barrier(directory, boundary)
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            observations = observe(active.runner, recorder, probe)

            assert observations.model_call_entered["policy"] == 0, observations
            assert body["summaries"] == 0, body
            assert body["status"] == ReviewLifecycleStatus.CANCELLED.value, body
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def test_an_expired_lease_never_enters_the_model_and_the_old_generation_cannot_return() -> (
    None
):
    """AC-03: a lapsed lease ends the generation, and its boundary stays closed for good.

    The first half is the ordinary scenario: the lease expires while one Worker's read is
    held, so the entry must refuse.  The second half asks the boundary that execution built
    whether it may still enter once the generation is gone — a takeover, a replay or a late
    callback must not be able to borrow the old identity.
    """

    boundary = f"after_result_admission:policy.{RETRIEVAL_BOUNDARY}"
    directory = barrier_dir("fix09c_lease")
    fault = parked_fault(directory, boundary)
    with new_database("c_lease") as fixture:
        active, recorder, probe, _seen, loop, ident = _harness(
            fixture,
            fault=fault,
            heartbeat_seconds=1000.0,
            scan_interval_seconds=0.2,
        )
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="fix09c-lease",
                )
            )
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            # The lease is expired as a database fact rather than by a short clock: the
            # scanner is asked to judge against a time past any lease, so a loaded host
            # cannot make the scenario pass or fail by being slow.  The parked Worker's
            # generation is therefore already gone when it is released.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                fixture.observer.scan_expired_leases(
                    now=_now() + timedelta(hours=1),
                    events_factory=lambda _r: (),
                    audit_factory=lambda _r: None,
                )
                if (
                    fixture.ledger(run_id)["status"]
                    != ReviewLifecycleStatus.PROCESSING.value
                ):
                    break
                time.sleep(0.2)
            release_barrier(directory, boundary)
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            observations = observe(active.runner, recorder, probe)

            assert observations.model_call_entered["policy"] == 0, observations
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["summaries"] == 0, body

            # The generation is gone: its own boundary must refuse a later entry outright.
            assert probe.invocations, "the execution built no model-call boundary"
            old = probe.invocations[0]
            built: list[Any] = []
            with pytest.raises(ExecutionInvalidated):
                old.enter(
                    site=entry_site("policy"),
                    request_factory=lambda: built.append("built"),  # type: ignore[arg-type]
                )
            assert built == [], "an expired generation still built a request"
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


# ------------------------------------------------------- V-C-03: two Runs, two identities


def test_two_concurrent_runs_do_not_share_an_entry_identity() -> None:
    """AC-03: revoking one Run's credential does not open or close the other Run's entry."""

    gate = ReadGate(worker="policy", hold_once=True)
    with new_database("c_two_runs") as fixture:
        active, recorder, probe, _seen, loop, _ident = _harness(fixture, gate=gate)
        held_ident = fixture.identity(case_id="CASE-LOW-001")
        healthy_ident = fixture.identity(case_id="CASE-HIGH-001")
        try:
            with held_reads(gate):
                held = loop.start(
                    active.service.create_review(
                        fixture.create_request(held_ident),
                        registration=held_ident.registration,
                        scope=held_ident.scope,
                        context=held_ident.context,
                        idempotency_key="fix09c-two-held",
                    )
                )
                assert gate.fetched.wait(60.0), "the held read never returned"
                held_run = fixture.latest_run()
                healthy = loop.start(
                    active.service.create_review(
                        fixture.create_request(healthy_ident),
                        registration=healthy_ident.registration,
                        scope=healthy_ident.scope,
                        context=healthy_ident.context,
                        idempotency_key="fix09c-two-healthy",
                    )
                )
                healthy.result(timeout=120)
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
                held.result(timeout=120)
            held_body = settle(fixture, held_run, *SETTLED, timeout=60.0)
            observations = observe(active.runner, recorder, probe)

            calls = recorder.entries
            healthy_calls = [
                item for item in calls if item["case_id"] == "CASE-HIGH-001"
            ]
            held_calls = [item for item in calls if item["case_id"] == "CASE-LOW-001"]
            assert sorted(item["worker"] for item in healthy_calls) == [
                "contract",
                "policy",
                "procurement_data",
            ], healthy_calls
            assert [item for item in held_calls if item["worker"] == "policy"] == [], (
                held_calls
            )
            assert healthy_body["summaries"] == 1, healthy_body
            assert len(healthy_body["checkpoints"]) == 3, healthy_body
            assert held_body["summaries"] == 0, held_body
            assert held_body["status"] == ReviewLifecycleStatus.INTERRUPTED.value
            assert held_body["error_code"] == REVOKED, held_body
            assert observations.model_result_adopted["policy"] >= 1, observations
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


# ------------------------------------------- V-C-04: model outcomes stay fail-closed


def test_a_synchronous_receive_failure_is_a_branch_result_and_not_an_entry() -> None:
    """AC-02/AC-04: a failed receipt is fail-closed, and it is not counted as an entry.

    The adapter raises inside ``start`` itself.  The transaction that admitted the call
    rolls back, the Worker turns the failure into its schema-valid unresolved branch, and
    the execution still summarizes — but the entry counter stays at zero, because an
    attempted receipt is not a received call.
    """

    recorder = BoundaryRecorder(raises_at_start=True)
    with new_database("c_start_failure") as fixture:
        active, _recorder, probe, _seen, loop, ident = _harness(
            fixture, recorder=recorder
        )
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="fix09c-start-failure",
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

            # The adapter was handed every request and failed each one; the boundary
            # admitted none of them, so no entry is recorded even though three receipts
            # were attempted, no branch committed a result, and the Run still summarized
            # fail-closed instead of inventing findings.
            assert recorder.by_worker() == {
                "policy": 1,
                "contract": 1,
                "procurement_data": 1,
            }, observations
            assert active.runner.telemetry.model_entries == {}, (
                active.runner.telemetry.model_entries
            )
            assert body["summaries"] == 1, body
            assert body["checkpoints"] == [], body["checkpoints"]
            assert body["status"] == ReviewLifecycleStatus.PENDING_HUMAN_REVIEW.value, (
                body
            )
        finally:
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()


def test_an_asynchronous_model_failure_and_a_bad_output_never_adopt_anything() -> None:
    """AC-02/AC-04: a received call that fails or answers badly still fails closed.

    Both runs below reach the adapter for real — the entry is counted — and both must end
    as the Worker's own unresolved branch: the received request may not be turned into a
    supported finding, a summary of facts it never received, or an adopted result.
    """

    for label, recorder in (
        ("model_exception", BoundaryRecorder(raises=True)),
        ("bad_output", BoundaryRecorder(output={"not": "a Finding tuple"})),
    ):
        with new_database(f"c_{label}") as fixture:
            active = runtime(fixture)
            active.runner._model = BoundaryRecordingModel(recorder)
            probe = attach_invocation(active.runner, InvocationProbe())
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
                        idempotency_key=f"fix09c-{label}",
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

                assert observations.model_call_entered["policy"] == 1, observations
                assert body["summaries"] == 1, body
                assert seen, "the execution produced no result at all"
                policy_result = next(
                    item
                    for item in seen[0].branch_results
                    if item.worker.value == "policy"
                )
                assert policy_result.outcome.value == "failed", policy_result
            finally:
                loop.call(active.service.shutdown(), timeout=60.0)
                loop.close()


def test_an_unreachable_authority_store_stops_the_run_at_the_entry() -> None:
    """AC-02: an unreachable store is fail-closed and never degrades into a Worker branch.

    The outage is a fact of the database, not a race: new connections to the scenario's
    database are refused and every backend carrying the application's own tag is
    terminated.  The boundary's authority re-read then cannot succeed, so the call must not
    be entered and the Run must stop as a storage failure rather than summarize.
    """

    from procurement_review.fault_hooks import FAULT_APPLICATION_NAME_ENV

    tag = f"fix09c_outage_{os.urandom(3).hex()}"
    previous = os.environ.get(FAULT_APPLICATION_NAME_ENV)
    os.environ[FAULT_APPLICATION_NAME_ENV] = tag
    boundary = f"after_result_admission:policy.{RETRIEVAL_BOUNDARY}"
    directory = barrier_dir("fix09c_outage")
    fault = parked_fault(directory, boundary)
    settled = False
    try:
        with new_database("c_outage", application_name=tag) as fixture:
            active, recorder, probe, _seen, loop, ident = _harness(
                fixture,
                fault=fault,
                lease_seconds=2.0,
                heartbeat_seconds=0.5,
                scan_interval_seconds=0.2,
            )
            with StoreOutage(fixture) as outage:
                handle = loop.start(
                    active.service.create_review(
                        fixture.create_request(ident),
                        registration=ident.registration,
                        scope=ident.scope,
                        context=ident.context,
                        idempotency_key="fix09c-outage",
                    )
                )
                wait_barrier(directory, boundary)
                run_id = fixture.latest_run()
                warmed = fixture.ledger(run_id)
                outage.refuse()
                outage.wait_until_unreachable()
                release_barrier(directory, boundary)
                handle.result(timeout=120)
                deadline = time.monotonic() + 60.0
                body = warmed
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
                outage.restore()
                observations = observe(active.runner, recorder, probe)

                assert observations.model_call_entered["policy"] == 0, observations
                assert entry_site("policy") not in observations.factory_sites, (
                    observations
                )
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
            release_barrier(directory, boundary)
        finally:
            if previous is None:
                os.environ.pop(FAULT_APPLICATION_NAME_ENV, None)
            else:
                os.environ[FAULT_APPLICATION_NAME_ENV] = previous
        if not settled:
            pytest.fail("the outage scenario did not reach a settled Run")


# ------------------------------------------- V-C-05: a slow model holds no database lock


@pytest.mark.parametrize("worker", WORKERS)
def test_a_slow_model_never_holds_a_database_lock(worker: str) -> None:
    """AC-02: waiting for the model happens outside every transaction.

    The model response is held while a real revocation is issued on the independent engine.
    If the admitting transaction were still open, that revocation would wait for the model;
    instead it must commit promptly, and a competing transaction must be able to take the
    authority row at the same time.
    """

    gate = ReadGate(worker=worker)
    recorder = BoundaryRecorder(gate=gate, hold_response=True)
    with new_database(f"c_slow_{worker}") as fixture:
        active, _recorder, probe, _seen, loop, ident = _harness(
            fixture, recorder=recorder
        )
        try:
            handle = loop.start(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key=f"fix09c-slow-{worker}",
                )
            )
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline and not recorder.by_worker()[worker]:
                time.sleep(0.02)
            assert recorder.by_worker()[worker] == 1, recorder.public_entries()
            run_id = fixture.latest_run()

            started = time.monotonic()
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "SELECT revision FROM review_authorization_revision "
                        "WHERE tenant_id = :t FOR UPDATE"
                    ),
                    {"t": ident.context.tenant_id},
                )
            blocked_for = time.monotonic() - started
            assert blocked_for < 3.0, (
                f"the authority row stayed locked for {blocked_for:.2f}s during the model"
            )

            revocation = BackgroundRevocation(fixture, ident.credential_id)
            revocation.start()
            revocation.join(timeout=30.0)
            assert revocation.seconds < 3.0, (
                f"the revocation waited {revocation.seconds:.2f}s for the model"
            )
            assert not gate.release.is_set()

            gate.revoked.set()
            gate.let_go()
            handle.result(timeout=120)
            body = settle(fixture, run_id, *SETTLED, timeout=60.0)
            observations = observe(active.runner, recorder, probe)

            assert observations.model_call_entered[worker] == 1, observations
            assert observations.model_result_adopted[worker] == 0, observations
            assert body["summaries"] == 0, body
            assert body["status"] == ReviewLifecycleStatus.INTERRUPTED.value, body
            assert body["error_code"] == REVOKED, body
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60.0)
            loop.close()
