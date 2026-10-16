"""Cancellation orders observed with independent services and PostgreSQL locks."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text

from fix01a_support import barrier_dir, parked_fault, release_barrier, wait_barrier
from fix08_support import ReadGate, new_database
from fix09_support import (
    READ_BOUNDARY,
    WORKERS,
    entry_site,
    observe,
    wait_for_ungranted_lock,
)
from procurement_review.api_contracts import HumanActionRequestV2
from procurement_review.contracts import ApprovalActionType, ReviewLifecycleStatus
from procurement_review.local_runtime import ScopedWorkflowRunner
from procurement_review.review_repository import ReviewRepository
from procurement_review.review_service import ReviewService
from test_fix09_model_invocation_boundary import _harness


async def drain(service):
    """Wait for execution, not merely the already committed terminal database row."""
    await asyncio.gather(*tuple(service._tasks.values()), return_exceptions=True)


def cancel(fixture, ident, run_id):
    service = ReviewService(
        repository=ReviewRepository(fixture.driver),
        runner=ScopedWorkflowRunner(store=fixture.driver),
    )
    current = service.get_review(run_id, context=ident.context)
    return service.apply_action(
        run_id=run_id,
        action=ApprovalActionType.CANCEL,
        request=HumanActionRequestV2(
            expected_version=current.revision, idempotency_key="final-cancel"
        ),
        context=ident.context,
    )


@pytest.mark.parametrize("worker", WORKERS)
@pytest.mark.parametrize("entry_first", [False, True])
def test_cancel_and_model_receipt_have_a_database_order(worker, entry_first):
    boundary = (
        f"during_model_entry_authorization:{entry_site(worker)}"
        if entry_first
        else f"after_result_admission:{worker}.{READ_BOUNDARY[worker]}"
    )
    directory = barrier_dir(f"final_cancel_{worker}_{entry_first}")
    gate = ReadGate(worker=worker)
    with new_database(f"final_cancel_{worker}") as fixture:
        active, recorder, probe, _, loop, ident = _harness(
            fixture, fault=parked_fault(directory, boundary), gate=gate
        )
        pool = ThreadPoolExecutor(max_workers=1)
        contender = None
        try:
            loop.call(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="final-create",
                )
            )
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            assert recorder.by_worker()[worker] == 0
            assert entry_site(worker) not in probe.sites
            contender = pool.submit(cancel, fixture, ident, run_id)
            if entry_first:
                assert wait_for_ungranted_lock(fixture, "fix01a_driver") > 0
                assert not contender.done()
                release_barrier(directory, boundary)
            response = contender.result(timeout=30)
            assert response.status is ReviewLifecycleStatus.CANCELLED
            assert fixture.ledger(run_id)["status"] == "cancelled"
            if not entry_first:
                assert recorder.by_worker()[worker] == 0
                release_barrier(directory, boundary)
            # The response remains withheld until cancellation is actually committed.
            gate.let_go()
            loop.call(drain(active.service), timeout=60)
            result = observe(active.runner, recorder, probe)
            assert result.model_call_entered[worker] == int(entry_first), result
            assert result.entered_with_protected_payload[worker] == int(entry_first)
            assert result.factory_sites.count(entry_site(worker)) == int(entry_first)
            assert result.model_result_adopted[worker] == 0
            body = fixture.ledger(run_id)
            assert body["status"] == "cancelled" and body["summaries"] == 0
            assert f"{worker}@1" not in body["checkpoints"]
            with fixture.probe.transaction() as connection:
                assert (
                    connection.execute(
                        text("SELECT count(*) FROM review_receipt WHERE run_id = :r"),
                        {"r": run_id},
                    ).scalar_one()
                    == 0
                )
        finally:
            release_barrier(directory, boundary)
            gate.let_go()
            pool.shutdown(wait=True)
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()


@pytest.mark.parametrize("worker", WORKERS)
@pytest.mark.parametrize("late_error", [False, True])
def test_slow_model_does_not_delay_cancellation(worker, late_error):
    gate = ReadGate(worker=worker)
    with new_database(f"final_slow_{worker}") as fixture:
        active, recorder, _, _, loop, ident = _harness(fixture, gate=gate)
        deliver = active.runner._model.deliver

        async def late_delivery(request, findings):
            result = await deliver(request, findings)
            if late_error and request.worker.value == worker:
                raise ValueError("synthetic late response failure")
            return result

        active.runner._model.deliver = late_delivery
        try:
            loop.call(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="final-slow",
                )
            )
            assert gate.fetched.wait(30)
            assert recorder.by_worker()[worker] == 1
            with ThreadPoolExecutor(max_workers=1) as pool:
                response = pool.submit(
                    cancel, fixture, ident, fixture.latest_run()
                ).result(3)
            assert response.status is ReviewLifecycleStatus.CANCELLED
            assert not gate.release.is_set()
            gate.let_go()
            loop.call(drain(active.service), timeout=60)
            assert (
                active.runner.telemetry.result_admissions.get(
                    f"{worker}.model_result", 0
                )
                == 0
            )
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()


@pytest.mark.parametrize("failure", ["factory", "start", "transaction"])
def test_failed_entry_releases_lifecycle_locks(failure, monkeypatch):
    from contextlib import contextmanager
    from sqlalchemy.exc import OperationalError
    from procurement_review.checkpointing import StorageUnavailableSignal

    boundary = "after_result_admission:policy.retrieval_result"
    directory = barrier_dir(f"final_failure_{failure}")
    with new_database(f"final_failure_{failure}") as fixture:
        active, _, probe, _, loop, ident = _harness(
            fixture, fault=parked_fault(directory, boundary)
        )
        try:
            loop.call(
                active.service.create_review(
                    fixture.create_request(ident),
                    registration=ident.registration,
                    scope=ident.scope,
                    context=ident.context,
                    idempotency_key="final-failure",
                )
            )
            wait_barrier(directory, boundary)
            invocation = probe.invocations[0]
            built = []

            def factory():
                built.append(True)
                if failure == "factory":
                    raise ValueError("synthetic factory failure")
                return object()

            class Receiver:
                def start(self, request):
                    if failure == "start":
                        raise ValueError("synthetic start failure")
                    return object()

            original_transaction = fixture.app.transaction

            @contextmanager
            def failed_commit():
                with original_transaction() as connection:
                    yield connection
                    raise OperationalError("synthetic commit failure", None, None)

            with monkeypatch.context() as scoped:
                scoped.setattr(invocation, "_receiver", Receiver())
                if failure == "transaction":
                    scoped.setattr(fixture.app, "transaction", failed_commit)
                with pytest.raises(
                    StorageUnavailableSignal if failure == "transaction" else ValueError
                ):
                    invocation.enter(site="policy.model_entry", request_factory=factory)
            assert built == [True]
            run_id = fixture.latest_run()
            with fixture.probe.transaction() as connection:
                connection.execute(
                    text(
                        "SELECT run_id FROM review_run WHERE run_id=:r FOR UPDATE NOWAIT"
                    ),
                    {"r": run_id},
                )
                connection.execute(
                    text(
                        "SELECT generation FROM review_run_generation WHERE run_id=:r FOR UPDATE NOWAIT"
                    ),
                    {"r": run_id},
                )
            assert (
                cancel(fixture, ident, run_id).status is ReviewLifecycleStatus.CANCELLED
            )
            release_barrier(directory, boundary)
            loop.call(drain(active.service), timeout=60)
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()
