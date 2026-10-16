"""Lock-wait clocks, independent lease/deadline, and generation-bound cleanup."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from fix01a_support import barrier_dir, parked_fault, release_barrier, wait_barrier
from fix08_support import new_database
from fix09_support import wait_for_ungranted_lock
from procurement_review.execution_guards import ExecutionInvalidated
from test_fix09_lifecycle_invocation import _harness
from test_fix_final_cancellation import cancel, drain
from test_fix_final_deadlines import Clock, create, limits


@contextmanager
def parked(name, **settings):
    boundary = "after_result_admission:policy.retrieval_result"
    directory = barrier_dir(name)
    clock = Clock(datetime.now(UTC))
    with new_database(name) as fixture:
        active, recorder, probe, _, loop, ident = _harness(
            fixture, fault=parked_fault(directory, boundary), clock=clock, **settings
        )
        try:
            create(active, loop, fixture, ident)
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            yield (
                fixture,
                active,
                recorder,
                probe,
                loop,
                ident,
                run_id,
                clock,
                limits(fixture, run_id),
            )
        finally:
            release_barrier(directory, boundary)
            loop.call(drain(active.service), timeout=60)
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()


@pytest.mark.parametrize("resource", ["revision", "grant", "run", "generation"])
@pytest.mark.parametrize("expiry", ["deadline", "lease", "credential"])
def test_waiting_for_each_lock_refreshes_time(resource, expiry):
    with parked(f"final_wait_{resource}_{expiry}") as args:
        fixture, active, _, probe, _, ident, run_id, clock, row = args
        if expiry == "lease":
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_run_generation SET lease_expires_at=:at WHERE run_id=:r"
                    ),
                    {"r": run_id, "at": clock() + timedelta(seconds=1)},
                )
        if expiry == "credential":
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_credential SET expires_at=:at WHERE credential_id=:c"
                    ),
                    {"c": ident.credential_id, "at": clock() + timedelta(seconds=1)},
                )
        queries = {
            "revision": "SELECT revision FROM review_authorization_revision WHERE tenant_id=:t FOR UPDATE",
            "grant": "SELECT case_id FROM review_case_grant WHERE tenant_id=:t AND case_id=:c AND subject_id=:s FOR UPDATE",
            "run": "SELECT run_id FROM review_run WHERE run_id=:r FOR UPDATE",
            "generation": "SELECT generation FROM review_run_generation WHERE run_id=:r AND generation=1 FOR UPDATE",
        }
        factory = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(queries[resource]),
                    {
                        "t": ident.context.tenant_id,
                        "c": ident.registration.case_id,
                        "s": ident.context.subject_id,
                        "r": run_id,
                    },
                )
                pending = pool.submit(
                    probe.invocations[0].enter,
                    site="policy.model_entry",
                    request_factory=lambda: factory.append(True),
                )
                assert wait_for_ungranted_lock(fixture, "fix01a_app") >= 1
                assert not pending.done()
                clock.value = (
                    row.deadline_at
                    if expiry == "deadline"
                    else clock() + timedelta(seconds=1)
                )
            failure = None
            try:
                pending.result(timeout=3)
            except BaseException as error:
                failure = error
        assert factory == []
        assert isinstance(failure, ExecutionInvalidated), failure
        assert (
            failure.reason
            == {
                "deadline": "generation_deadline_exceeded",
                "lease": "lease_expired",
                "credential": "credential_expired",
            }[expiry]
        )


@pytest.mark.parametrize("state", ["healthy", "deadline", "lease", "owner"])
def test_heartbeat_respects_both_times_and_owner(state):
    with parked(f"final_heartbeat_{state}") as args:
        fixture, active, _, _, _, _, run_id, clock, row = args
        owner = row.owner_id
        if state == "deadline":
            clock.value = row.deadline_at
        elif state == "lease":
            with fixture.driver.transaction() as connection:
                connection.execute(
                    text(
                        "UPDATE review_run_generation SET lease_expires_at=:at WHERE run_id=:r"
                    ),
                    {"r": run_id, "at": clock()},
                )
        elif state == "owner":
            owner = "owner-stale"
        before = fixture.ledger(run_id)
        renewed = active.service._heartbeat(run_id, 1, owner)
        assert renewed is (state == "healthy")
        body = fixture.ledger(run_id)
        if state in {"deadline", "lease"}:
            assert body["status"] == "interrupted", body
            assert body["error_code"] == (
                "generation_deadline_exceeded"
                if state == "deadline"
                else "lease_expired"
            )
        if state == "owner":
            assert body == before
        with fixture.probe.transaction() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT deadline_at FROM review_run_generation WHERE run_id=:r AND generation=1"
                    ),
                    {"r": run_id},
                ).scalar_one()
                == row.deadline_at
            )


def test_old_cleanup_cannot_interrupt_a_new_owner():
    with parked("final_old_cleanup") as args:
        fixture, active, _, _, _, _, run_id, clock, row = args
        with fixture.driver.transaction() as connection:
            connection.execute(
                text(
                    "UPDATE review_run_generation SET owner_id='owner-replacement' WHERE run_id=:r"
                ),
                {"r": run_id},
            )
        before = fixture.ledger(run_id)
        assert not active.repository.mark_interrupted(
            run_id=run_id,
            generation=1,
            owner_id=row.owner_id,
            reason_code="generation_deadline_exceeded",
            now=clock(),
        )
        assert not active.repository.mark_failed(
            run_id=run_id,
            generation=1,
            owner_id=row.owner_id,
            error_code="review_execution_failed",
            error_message="synthetic failure",
            now=clock(),
        )
        assert fixture.ledger(run_id) == before
        # Restore the deliberately replaced identity so normal test cleanup can finish.
        with fixture.driver.transaction() as connection:
            connection.execute(
                text("UPDATE review_run_generation SET owner_id=:o WHERE run_id=:r"),
                {"r": run_id, "o": row.owner_id},
            )


def test_deadline_does_not_prevent_authorized_cancel():
    with parked("final_expired_cancel") as args:
        fixture, active, _, _, _, ident, run_id, clock, row = args
        clock.value = row.deadline_at
        assert cancel(fixture, ident, run_id).status.value == "cancelled"
        assert (
            active.service.get_review(run_id, context=ident.context).status.value
            == "cancelled"
        )


@pytest.mark.parametrize("action_name", ["confirm", "reject", "cancel"])
def test_deadline_is_not_an_expiry_for_committed_summary_and_human_actions(action_name):
    from procurement_review.api_contracts import HumanActionRequestV2
    from procurement_review.contracts import ApprovalActionType

    clock = Clock(datetime.now(UTC))
    with new_database(f"final_history_{action_name}") as fixture:
        active, _, _, _, loop, ident = _harness(fixture, clock=clock)
        try:
            create(active, loop, fixture, ident)
            loop.call(drain(active.service), timeout=60)
            run_id = fixture.latest_run()
            pending = active.service.get_review(run_id, context=ident.context)
            assert pending.status.value == "pending_confirmation"
            before = fixture.ledger(run_id)
            clock.value += timedelta(seconds=31)
            current = active.service.get_review(run_id, context=ident.context)
            assert current.summary_checksum == pending.summary_checksum
            request = HumanActionRequestV2(
                expected_version=current.revision,
                expected_summary_checksum=current.summary_checksum,
                idempotency_key="final-history",
            )
            result = active.service.apply_action(
                run_id=run_id,
                action=ApprovalActionType(action_name),
                request=request,
                context=ident.context,
            )
            replay = active.service.apply_action(
                run_id=run_id,
                action=ApprovalActionType(action_name),
                request=request,
                context=ident.context,
            )
            assert result.status == replay.status
            assert (
                result.status.value
                == {
                    "confirm": "confirmed",
                    "reject": "rejected",
                    "cancel": "cancelled",
                }[action_name]
            )
            after = fixture.ledger(run_id)
            assert after["checkpoints"] == before["checkpoints"]
            assert after["summaries"] == before["summaries"] == 1
        finally:
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()


def test_deadline_interruption_resumes_with_new_deadline_and_old_cleanup_is_ignored():
    from procurement_review.api_contracts import ResumeRequestV2
    from fix01a_support import Loop
    from procurement_review.review_repository import ReviewRepository
    from procurement_review.review_service import ReviewService
    from procurement_review.local_runtime import ScopedWorkflowRunner

    with parked("final_deadline_resume") as args:
        fixture, active, _, _, loop, ident, run_id, clock, old = args
        clock.value = old.deadline_at
        assert not active.service._heartbeat(run_id, 1, old.owner_id)
        interrupted = active.service.get_review(run_id, context=ident.context)
        assert interrupted.status.value == "interrupted"
        # Independent service owns recovery and never has a reference to the old task.
        recovery_loop = Loop()
        recovered = ReviewService(
            repository=ReviewRepository(fixture.driver),
            runner=ScopedWorkflowRunner(store=fixture.driver),
            clock=clock,
            lease_seconds=1000,
            heartbeat_seconds=1000,
        )
        response = recovery_loop.call(
            recovered.resume_review(
                run_id,
                ResumeRequestV2(expected_version=interrupted.revision),
                context=ident.context,
                idempotency_key="final-resume",
            )
        )
        assert response[0].status.value in {"processing", "pending_confirmation"}
        recovery_loop.call(drain(recovered), timeout=60)
        with fixture.probe.transaction() as connection:
            latest = connection.execute(
                text(
                    "SELECT generation, deadline_at FROM review_run_generation WHERE run_id=:r ORDER BY generation DESC LIMIT 1"
                ),
                {"r": run_id},
            ).one()
        assert latest.generation == 2 and latest.deadline_at > old.deadline_at
        assert fixture.ledger(run_id)["status"] == "pending_confirmation"
        before = fixture.ledger(run_id)
        assert not active.repository.mark_interrupted(
            run_id=run_id,
            generation=1,
            owner_id=old.owner_id,
            reason_code="lease_expired",
            now=clock(),
        )
        assert not active.repository.heartbeat(
            run_id=run_id,
            generation=1,
            owner_id=old.owner_id,
            now=clock,
            lease_seconds=1000,
        )
        assert fixture.ledger(run_id) == before
        recovery_loop.call(recovered.shutdown(), timeout=60)
        recovery_loop.close()


def test_retry_after_deadline_cannot_book_a_new_attempt(monkeypatch):
    from procurement_review.fault_hooks import FaultController
    from test_fix_final_deadlines import assert_expired

    clock = Clock(datetime.now(UTC))
    fault = FaultController(enabled=True, transient={"policy": {1: 1}})
    with new_database("final_retry_deadline") as fixture:
        active, recorder, _, _, loop, ident = _harness(
            fixture, clock=clock, fault=fault
        )
        finish = active.repository.finish_attempt

        def finish_and_expire(**kwargs):
            finish(**kwargs)
            if kwargs["worker"] == "policy":
                clock.value = limits(fixture, kwargs["run_id"]).deadline_at

        monkeypatch.setattr(active.repository, "finish_attempt", finish_and_expire)
        try:
            create(active, loop, fixture, ident)
            loop.call(drain(active.service), timeout=60)
            run_id = fixture.latest_run()
            body = assert_expired(active, fixture, ident, run_id)
            assert [a for a in body["attempts"] if "/policy/" in a] == ["1/policy/1"]
            assert recorder.by_worker()["policy"] == 0
        finally:
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()
