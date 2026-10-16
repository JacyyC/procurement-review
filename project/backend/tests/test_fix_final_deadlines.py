"""Persistent generation deadlines at real execution and publication boundaries."""

import time
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from fix01a_support import barrier_dir, parked_fault, release_barrier, wait_barrier
from fix08_support import READ_BOUNDARY, ReadGate, new_database
from fix09_support import WORKERS, entry_site, observe
from test_fix09_lifecycle_invocation import _harness
from test_fix_final_cancellation import drain


@dataclass
class Clock:
    value: datetime

    def __call__(self):
        return self.value


def limits(fixture, run_id):
    with fixture.probe.transaction() as connection:
        return connection.execute(
            text(
                "SELECT generation, owner_id, deadline_at, lease_expires_at "
                "FROM review_run_generation WHERE run_id=:r AND status='active'"
            ),
            {"r": run_id},
        ).one()


def create(active, loop, fixture, ident):
    loop.call(
        active.service.create_review(
            fixture.create_request(ident),
            registration=ident.registration,
            scope=ident.scope,
            context=ident.context,
            idempotency_key="final-deadline",
        )
    )


def assert_expired(active, fixture, ident, run_id):
    body = fixture.ledger(run_id)
    assert body["status"] == "interrupted", body
    assert body["error_code"] == "generation_deadline_exceeded", body
    assert body["summaries"] == 0, body
    # Closed records remain readable through the actual guarded user projection.
    assert (
        active.service.get_review(run_id, context=ident.context).status.value
        == "interrupted"
    )
    with fixture.probe.transaction() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM review_worker_attempt WHERE run_id=:r AND finished_at IS NULL"
                ),
                {"r": run_id},
            ).scalar_one()
            == 0
        )
    return body


@pytest.mark.parametrize("worker", WORKERS)
def test_natural_deadline_stops_execution_while_lease_remains_live(worker):
    boundary = f"after_result_admission:{worker}.{READ_BOUNDARY[worker]}"
    directory = barrier_dir(f"final_natural_{worker}")
    with new_database(f"final_natural_{worker}") as fixture:
        active, recorder, probe, _, loop, ident = _harness(
            fixture,
            fault=parked_fault(directory, boundary),
            generation_deadline_seconds=2.0,
        )
        try:
            create(active, loop, fixture, ident)
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            row = limits(fixture, run_id)
            assert datetime.now(UTC) < row.deadline_at
            assert recorder.by_worker()[worker] == 0
            time.sleep(
                max(0, (row.deadline_at - datetime.now(UTC)).total_seconds()) + 0.01
            )
            assert row.deadline_at <= datetime.now(UTC) < row.lease_expires_at
            release_barrier(directory, boundary)
            loop.call(drain(active.service), timeout=60)
            observed = observe(active.runner, recorder, probe)
            assert observed.model_call_entered[worker] == 0, observed
            assert observed.entered_with_protected_payload[worker] == 0
            assert entry_site(worker) not in observed.factory_sites
            assert observed.model_result_adopted[worker] == 0
            assert_expired(active, fixture, ident, run_id)
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()


@pytest.mark.parametrize("worker", WORKERS)
@pytest.mark.parametrize("point", ["attempt", "factory", "model_result", "checkpoint"])
def test_expiry_at_execution_barriers_rolls_back_new_suggestions(worker, point, caplog):
    caplog.set_level(logging.INFO, logger="procurement_review")
    logger = logging.getLogger("procurement_review")
    logger.addHandler(caplog.handler)
    boundary = {
        "attempt": f"attempt_started:{worker}",
        "factory": f"during_model_entry_authorization:{entry_site(worker)}",
        "model_result": f"after_result_admission:{worker}.model_result",
        "checkpoint": f"during_checkpoint_commit:{worker}",
    }[point]
    directory = barrier_dir(f"final_time_{worker}_{point}")
    clock = Clock(datetime.now(UTC))
    with new_database(f"final_time_{worker}_{point}") as fixture:
        active, recorder, probe, _, loop, ident = _harness(
            fixture,
            fault=parked_fault(directory, boundary),
            clock=clock,
        )
        try:
            create(active, loop, fixture, ident)
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            row = limits(fixture, run_id)
            clock.value = (
                row.deadline_at
            )  # Equality is expired; the lease remains valid.
            assert clock() < row.lease_expires_at
            release_barrier(directory, boundary)
            loop.call(drain(active.service), timeout=60)
            result = observe(active.runner, recorder, probe)
            assert result.model_call_entered[worker] == int(
                point in {"model_result", "checkpoint"}
            )
            if point != "checkpoint":
                assert result.model_result_adopted[worker] == 0
            body = assert_expired(active, fixture, ident, run_id)
            assert f"{worker}@1" not in body["checkpoints"]
            assert not any(
                f"run branch committed: run_id={run_id} worker={worker}"
                in item.getMessage()
                for item in caplog.records
            )
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()
            logger.removeHandler(caplog.handler)


def test_summary_commit_expiry_rolls_back_summary_but_preserves_legal_checkpoints(
    caplog,
):
    caplog.set_level(logging.INFO, logger="procurement_review")
    logger = logging.getLogger("procurement_review")
    logger.addHandler(caplog.handler)
    boundary = "during_summary_commit"
    directory = barrier_dir("final_summary_time")
    clock = Clock(datetime.now(UTC))
    with new_database("final_summary_time") as fixture:
        active, _, _, _, loop, ident = _harness(
            fixture,
            fault=parked_fault(directory, boundary),
            clock=clock,
        )
        try:
            create(active, loop, fixture, ident)
            wait_barrier(directory, boundary)
            run_id = fixture.latest_run()
            row = limits(fixture, run_id)
            before = fixture.ledger(run_id)
            assert len(before["checkpoints"]) == 3
            clock.value = row.deadline_at
            release_barrier(directory, boundary)
            loop.call(drain(active.service), timeout=60)
            body = assert_expired(active, fixture, ident, run_id)
            assert body["checkpoints"] == before["checkpoints"]
            assert (
                sum(
                    "run branch committed:" in item.getMessage()
                    for item in caplog.records
                )
                == 3
            )
            assert not any(
                "run summary committed:" in item.getMessage() for item in caplog.records
            )
        finally:
            release_barrier(directory, boundary)
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()
            logger.removeHandler(caplog.handler)


@pytest.mark.parametrize("worker", WORKERS)
@pytest.mark.parametrize("late_error", [False, True])
def test_model_response_or_exception_after_deadline_is_not_adopted(worker, late_error):
    gate = ReadGate(worker=worker)
    clock = Clock(datetime.now(UTC))
    with new_database(f"final_response_{worker}") as fixture:
        active, recorder, probe, _, loop, ident = _harness(
            fixture, clock=clock, gate=gate
        )
        deliver = active.runner._model.deliver

        async def delivered(request, findings):
            answer = await deliver(request, findings)
            if late_error and request.worker.value == worker:
                raise ValueError("synthetic late failure")
            return answer

        active.runner._model.deliver = delivered
        try:
            create(active, loop, fixture, ident)
            assert gate.fetched.wait(30)
            run_id = fixture.latest_run()
            clock.value = limits(fixture, run_id).deadline_at
            gate.let_go()
            loop.call(drain(active.service), timeout=60)
            result = observe(active.runner, recorder, probe)
            assert result.model_call_entered[worker] == 1
            assert result.model_result_adopted[worker] == 0
            assert_expired(active, fixture, ident, run_id)
        finally:
            gate.let_go()
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()


@pytest.mark.parametrize("worker", WORKERS)
def test_request_factory_crossing_deadline_cannot_reach_adapter(worker):
    clock = Clock(datetime.now(UTC))
    with new_database(f"final_factory_{worker}") as fixture:
        active, recorder, probe, _, loop, ident = _harness(fixture, clock=clock)
        builder = active.runner.build_model_invocation

        def build(**kwargs):
            invocation = builder(**kwargs)

            class CrossingFactory:
                def enter(self, *, site, request_factory):
                    def crossing():
                        request = request_factory()
                        if site == entry_site(worker):
                            clock.value += timedelta(seconds=31)
                        return request

                    return invocation.enter(site=site, request_factory=crossing)

            return CrossingFactory()

        active.runner.build_model_invocation = build
        try:
            create(active, loop, fixture, ident)
            loop.call(drain(active.service), timeout=60)
            run_id = fixture.latest_run()
            assert probe.sites.count(entry_site(worker)) == 1
            assert recorder.by_worker()[worker] == 0
            assert_expired(active, fixture, ident, run_id)
        finally:
            loop.call(active.service.shutdown(), timeout=60)
            loop.close()
