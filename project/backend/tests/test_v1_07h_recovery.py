"""H preparation checks for actual corrections, settlement and runner admission."""

import asyncio
import copy
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import httpx
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from procurement_review.eval_h_fixture import make_disposable
from procurement_review.eval_h_seal import HSeal
from procurement_review.eval_h_runner import run, setup
from procurement_review.eval_h_mock import mock_transport
from procurement_review.eval_h_journal import Journal
from procurement_review.eval_h_consumption import completion_proof
from procurement_review.eval_h_score import rebuild as score
from procurement_review.eval_h_report import rebuild as report
from procurement_review.eval_h_audit import audit, freeze, replay
from procurement_review.eval_h_storage import write


@contextmanager
def batch(tmp_path, suffix):
    value = make_disposable(tmp_path / ".local", 2, batch_id="disposable-h-" + suffix)
    seal = HSeal(Path(value["root"]))
    seal.generate(
        seal.directory / "private/recipe.json",
        seal.directory / "private/seed.bin",
        seal.directory / "private/registry.json",
    )
    seal.seal()
    base = os.environ["TEST_POSTGRES_DSN"]
    name = "v107h_" + seal.plan().batch_id.replace("-", "_")
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(base).set(database=name).render_as_string(hide_password=False)
    try:
        setup(seal.directory, url)
        yield seal, url
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def finish(seal):
    proof = completion_proof(seal.directory)
    assert proof["complete"], proof["failures"]
    seal.complete()
    seal.release()
    value = score(seal.directory)
    write(seal.directory / "score.json", value)
    seal.scored(seal.directory / "score.json")
    rebuilt = report(seal.directory)
    write(seal.directory / "result.json", rebuilt)
    return value, rebuilt


def metering_counterexamples(seal):
    from procurement_review.eval_h_product_checks import h_metering_failures

    observation = Journal(seal.directory).committed(
        seal.plan().case_ids[0], 101, "multi_agent"
    )
    binding = observation["binding"]
    registration = dict(
        binding=dict(requested_model=seal.plan().configuration.requested_model)
    )
    run_id = observation["row"]["run_id"]
    assert not h_metering_failures(registration, run_id, binding)
    changed = copy.deepcopy(binding)
    changed["execution"]["metering"]["input_tokens_known"] = 1
    assert any(
        "ledger_mismatch" in error
        for error in h_metering_failures(registration, run_id, changed)
    )
    for invalid in (True, 1, [], "other-model"):
        changed = copy.deepcopy(binding)
        event = next(
            event for event in changed["events"] if event["step"] == "model_attempt"
        )
        event["output_summary"]["receipt_model"] = invalid
        assert h_metering_failures(registration, run_id, changed)


def test_legal_correction_preserves_first_failure_and_full_final_success(tmp_path):
    with batch(tmp_path, "correction-test") as (seal, _):
        requests = []
        asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("correction", requests),
            )
        )
        value, rebuilt = finish(seal)
        metering_counterexamples(seal)
        assert len(requests) == 18
        for arm in ("multi_agent", "single_agent"):
            first = value["repeats"]["101"][arm]["q01"]
            assert first["first"]["numerator"] == 0
            assert first["final"]["value"] == 1
            assert value["repeats"]["202"][arm]["q01"]["first"]["value"] == 1
        assert rebuilt["Q-12"]["complete"] is True
        assert replay(seal.directory)["verified"]
        assert audit(seal.directory)["audit_passed"]


def test_minted_run_identity_with_pii_shaped_digits_persists(tmp_path, monkeypatch):
    from procurement_review import review_service
    from procurement_review.record_guard import guard_record, scan_persisted
    from procurement_review.run_errors import RecordContractError

    failing_id = "run-7ae399608865446198057d36970f588d"
    original = review_service.new_id
    allocated = []

    def new_id(prefix):
        value = failing_id if prefix == "run" and not allocated else original(prefix)
        if prefix == "run":
            allocated.append(value)
        return value

    monkeypatch.setattr(review_service, "new_id", new_id)
    with batch(tmp_path, "minted-id-test") as (seal, _):
        requests = []
        asyncio.run(
            run(seal.directory, transport_override=mock_transport("material", requests))
        )
        assert len(requests) == 12
        proof = completion_proof(seal.directory)
        assert proof["complete"], proof["failures"]
        observation = Journal(seal.directory).committed(
            seal.plan().case_ids[0], 101, "multi_agent"
        )
        assert observation["row"]["run_id"] == failing_id
        execution = observation["binding"]["execution"]
        for value in (failing_id, "run-19436117970e4786bf273aa093fa7332"):
            body = copy.deepcopy(execution)
            body["run_id"] = body["summary"]["run_id"] = value
            assert scan_persisted(body, kind="execution", location="minted-id").clean
            guard_record("execution", body, strict=False)
        for value in ("19436117970", failing_id + "-text"):
            body = copy.deepcopy(execution)
            body["run_id"] = body["summary"]["run_id"] = value
            assert not scan_persisted(body, kind="execution", location="raw-pii").clean
            with pytest.raises(RecordContractError):
                guard_record("execution", body, strict=False)
        body = copy.deepcopy(execution)
        body["stop_reason"] = failing_id
        assert not scan_persisted(body, kind="execution", location="free-text").clean
        assert finish(seal)[1]["Q-12"]["complete"] is True


def test_http_failure_keeps_plan_denominators_unknown_usage_and_failure_report(
    tmp_path,
):
    with batch(tmp_path, "http-failure-test") as (seal, _):
        requests = []
        asyncio.run(
            run(seal.directory, transport_override=mock_transport("http500", requests))
        )
        value, rebuilt = finish(seal)
        metering_counterexamples(seal)
        for seed in ("101", "202"):
            for arm, count in (("multi_agent", 4), ("single_agent", 2)):
                item = value["repeats"][seed][arm]
                assert item["q01"]["first"]["denominator"] == count
                assert item["q01"]["first"]["numerator"] == 0
                assert item["q02"]["all"]["denominator"] == 24
                assert item["q07"]["semantic"]["denominator"] > 0
                assert item["q07"]["semantic"]["value"] == 1
                assert all(
                    not verdict["spans"]
                    for verdict in item["semantic_verdicts"].values()
                )
                assert item["q09"]["all_sample_p95_ms"] is not None
                assert item["q09"]["success_p95_ms"] is None
                assert item["q09"]["failed"] == 2
                assert item["q09"]["censored"] == 0
                assert item["metrics"]["q09_pipeline"] == {
                    "non_expected_pipeline_failures": 2,
                    "planned_runs": 2,
                }
                assert item["gates"]["Q-09"] is False
        assert rebuilt["Q-12"]["complete"] is False
        assert (
            "usage_unknown_comparison_unverified"
            in rebuilt["Q-12"]["incomplete_reasons"]
        )
        assert all(
            row["cost_value"] is None and row["unknown_usage"] > 0
            for row in rebuilt["Q-12"]["resources"].values()
        )
        assert replay(seal.directory)["verified"] is True
        assert audit(seal.directory)["comparison_complete"] is False


def test_mock_timeout_preserves_all_sample_upper_bound_and_censoring(tmp_path):
    def timeout(request):
        raise httpx.ReadTimeout("controlled-h-mock-timeout", request=request)

    with batch(tmp_path, "timeout-test") as (seal, _):
        asyncio.run(
            run(seal.directory, transport_override=httpx.MockTransport(timeout))
        )
        value, rebuilt = finish(seal)
        for repeat in value["repeats"].values():
            for item in repeat.values():
                assert item["metrics"]["q09_pipeline"] == {
                    "non_expected_pipeline_failures": 2,
                    "planned_runs": 2,
                }
                assert item["q09"]["failed"] == item["q09"]["censored"] == 2
                assert item["q09"]["all_sample_p95_ms"] is not None
                assert item["q09"]["success_p95_ms"] is None
                assert item["gates"]["Q-09"] is False
        assert rebuilt["stage_acceptance"] is False
        assert replay(seal.directory)["verified"]


def test_incomplete_report_never_opens_labels_and_cannot_freeze(tmp_path, monkeypatch):
    value = make_disposable(
        tmp_path / ".local", 2, batch_id="disposable-h-incomplete-test"
    )
    seal = HSeal(Path(value["root"]))
    seal.generate(
        seal.directory / "private/recipe.json",
        seal.directory / "private/seed.bin",
        seal.directory / "private/registry.json",
    )
    seal.seal()
    original = Path.open

    def no_labels(path, *args, **kwargs):
        if path.name == "labels.json":
            pytest.fail("failure report opened sealed labels")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", no_labels)
        rebuilt = report(seal.directory)
    assert rebuilt["Q-12"]["complete"] is False
    assert rebuilt["Q-12"]["coverage"]["planned"] == 8
    assert rebuilt["Q-12"]["coverage"]["observed"] == 0
    write(seal.directory / "result.json", rebuilt)
    assert replay(seal.directory)["verified"]
    with pytest.raises(ValueError, match="freeze_audit_failed"):
        freeze(seal.directory, tmp_path / "bad-freeze.json")
    assert seal.record()["state"] == "sealed"


def test_concurrent_run_is_refused_without_blocking_the_active_event_loop(tmp_path):
    with batch(tmp_path, "concurrency-test") as (seal, _):
        requests = []

        async def concurrent():
            first = asyncio.create_task(
                run(
                    seal.directory,
                    transport_override=mock_transport("material", requests),
                )
            )
            await asyncio.sleep(0)
            with pytest.raises(ValueError, match="concurrent_operation_in_progress"):
                await run(seal.directory, transport_override=mock_transport("material"))
            return await asyncio.wait_for(first, timeout=60)

        assert asyncio.run(concurrent())["committed"] == 8
        assert len(requests) == 12
        assert completion_proof(seal.directory)["complete"]


def test_explicit_product_resume_reuses_checkpoint_and_preserves_unknown(tmp_path):
    from procurement_review.checkpointing import StorageUnavailableSignal
    from procurement_review.eval_h_product import multi
    from procurement_review.eval_h_storage import read
    from procurement_review.precise_run_transport import PreciseRunChatTransport
    from procurement_review.review_store import ReviewStore

    class Once:
        enabled = True
        fired = False

        def barrier(self, site):
            if (
                site == "during_model_entry_authorization:precise.contract"
                and not self.fired
            ):
                self.fired = True
                raise StorageUnavailableSignal("h-disposable-injected-storage-outage")

    with batch(tmp_path, "resume-test") as (seal, url):
        state = read(seal.directory / "private/runtime.json")
        store = ReviewStore.from_url(url)
        requests = []
        transport = PreciseRunChatTransport(
            seal.plan().configuration.model_copy(update={"seed": 101}),
            SecretStr("h-offline-only"),
            catalog=None,
            transport=mock_transport("material", requests),
        )
        fault = Once()
        case = next(
            case for case in seal.runner_cases() if case.case_id == "V107-D-9001"
        )
        try:
            row, binding = asyncio.run(
                multi(
                    store,
                    state["token"],
                    seal.plan(),
                    case,
                    101,
                    transport,
                    resume=True,
                    fault=fault,
                )
            )
        finally:
            store.close()
        write(tmp_path / "resume-binding.json", binding)
        assert fault.fired
        assert len(requests) == 2, f"mock_sends={len(requests)}"
        assert binding["resume_count"] == 1
        assert binding["execution"]["metering"]["reused_branches"] == 1
        assert binding["execution"]["metering"]["unknown_usage_attempts"] == 1
        assert row["status"] == "pending_confirmation"
        assert len(binding["attempts"]) == 3
        assert binding["checkpoints"][1]["worker"] == "policy"
        assert binding["checkpoints"][1]["generation"] == 1
        asyncio.run(
            run(
                seal.directory,
                transport_override=mock_transport("material", requests),
                resume=True,
            )
        )
        proof = completion_proof(seal.directory)
        assert proof["complete"], proof["failures"]
        assert len(requests) == 12


def test_settled_unknown_joint_task_is_kept_as_failure_and_never_resent(tmp_path):
    with batch(tmp_path, "joint-unknown-test") as (seal, _):
        journal = Journal(seal.directory)
        case = seal.plan().case_ids[0]
        journal.reserve(case, 101, "single_agent", "joint", "a" * 64)
        journal.recover_unknown(case, 101, "single_agent")
        requests = []
        asyncio.run(
            run(seal.directory, transport_override=mock_transport("material", requests))
        )
        assert len(requests) == 11
        own = journal.committed(case, 101, "single_agent")
        assert own["row"]["status"] == "failed"
        assert own["attempts"][0]["receipt"]["usage_unknown"] is True
        assert completion_proof(seal.directory)["complete"]


def test_abandoned_attempt_without_receipt_is_not_projected_as_known_success():
    from procurement_review.eval_h_product_checks import product_attempts

    binding = dict(
        events=[],
        execution={},
        attempts=[
            dict(
                generation=1,
                worker="policy",
                attempt_no=1,
                finished_at=None,
                abandoned=True,
                discarded_late=True,
                last_error=None,
            )
        ],
    )
    projected = product_attempts(binding)
    assert projected[0]["status"] == "settled"
    assert projected[0]["effective_error"] == "interrupted_unknown"
    assert projected[0]["receipt"]["input_usage"] is None
    assert projected[0]["receipt"]["request_sha256"] is None
