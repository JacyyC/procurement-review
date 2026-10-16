"""Legal concurrent, recovered, human and unadopted records for G v5."""

import copy
from datetime import datetime, timedelta

import pytest

from procurement_review.precise_dev_paired_metering import completeness_report_v3
from procurement_review.trace import canonical_checksum
from test_v1_07g_consumption_v5 import rejected
from v1_07g_consumption_support import (
    fixture_v5,
    recovery_fixture,
    event,
    seal,
    sort_events,
)


@pytest.fixture(scope="module")
def baseline():
    return fixture_v5()


def first_run(data):
    return next(iter(data[2]["runs"].values()))


def positive(data):
    report = completeness_report_v3(*data)
    assert report["complete"], report["failures"]
    assert (report["planned_pairs"], report["planned_model_tasks"]) == (24, 48)


def test_concurrent_workers_are_not_globally_serial(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    run["attempts"][1]["started_at"] = (
        datetime.fromisoformat(run["created_at"]) + timedelta(milliseconds=1)
    ).isoformat()
    positive(data)


def test_trace_one_microsecond_clock_adjustment(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    finish = datetime.fromisoformat(run["summary"]["created_at"])
    for index, e in enumerate(
        [e for e in run["events"] if e["step"] == "model_attempt"]
    ):
        e["occurred_at"] = (finish + timedelta(microseconds=index)).isoformat()
    for e in run["events"]:
        if e["step"] == "deterministic_aggregator":
            e["occurred_at"] = (finish + timedelta(microseconds=2)).isoformat()
        elif e["step"] == "review_workflow":
            e["occurred_at"] = (finish + timedelta(microseconds=3)).isoformat()
    row = next(
        row for row in data[1]["observations"] if data[2]["runs"][row["run_id"]] is run
    )
    row["observed_at"] = (finish + timedelta(microseconds=5)).isoformat()
    sort_events(run)
    seal(data)
    positive(data)


def test_untagged_production_interruption_can_recover(baseline):
    data = recovery_fixture(baseline)
    run = first_run(data)
    next(
        e for e in run["events"] if e["output_summary"].get("status") == "interrupted"
    )["generation"] = None
    positive(data)


def test_lease_scanner_interruption_can_recover(baseline):
    data = recovery_fixture(baseline)
    run = first_run(data)
    stop = next(
        e
        for e in run["events"]
        if e["step"] == "review_workflow" and e["event_type"] == "step_failed"
    )
    stop.update(generation=None)
    stop["output_summary"]["reason"] = "lease_expired"
    stop["output_checksum"] = canonical_checksum(stop["output_summary"])
    positive(data)


def test_recovered_generation_has_one_interruption(baseline):
    data = recovery_fixture(baseline)
    run = first_run(data)
    stop = copy.deepcopy(
        next(e for e in run["events"] if e["event_type"] == "step_failed")
    )
    stop["occurred_at"] = (
        datetime.fromisoformat(stop["occurred_at"]) + timedelta(microseconds=1)
    ).isoformat()
    run["events"].append(stop)
    sort_events(run)
    rejected(data, "trace_interruption_count")


def test_untagged_interruption_does_not_invalidate_new_generation(baseline):
    data = recovery_fixture(baseline)
    run = first_run(data)
    stop = next(
        e for e in run["events"] if e["output_summary"].get("status") == "interrupted"
    )
    stop["generation"] = None
    start = datetime.fromisoformat(run["generations"][-1]["created_at"])
    finish = start + timedelta(seconds=2)
    run["attempts"][0].update(
        generation=2,
        started_at=(start + timedelta(milliseconds=1)).isoformat(),
        finished_at=finish.isoformat(),
    )
    run["checkpoints"][0].update(generation=2, adopted_at=finish.isoformat())
    receipt = next(
        e
        for e in run["events"]
        if e["step"] == "model_attempt" and e["worker"] == "policy"
    )
    receipt.update(generation=2, occurred_at=finish.isoformat())
    run["execution"]["metering"]["reused_branches"] = 1
    sort_events(run)
    seal(data)
    positive(data)


def human_fixture(baseline, action):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    finish = datetime.fromisoformat(run["summary"]["created_at"]) + timedelta(seconds=2)
    status = {"confirm": "confirmed", "reject": "rejected", "cancel": "cancelled"}[
        action
    ]
    run["events"].extend(
        [
            event(
                run,
                "human_gate",
                {"action": action, "status": status},
                finish,
                kind="confirmation_recorded"
                if action == "confirm"
                else "human_action_recorded",
            ),
            event(
                run,
                "review_run",
                {"status": status},
                finish + timedelta(microseconds=1),
                kind="run_cancelled" if action == "cancel" else "run_completed",
            ),
        ]
    )
    run["status"] = status
    sort_events(run)
    return data


@pytest.mark.parametrize("action", ["confirm", "reject", "cancel"])
def test_observed_then_human_action(baseline, action):
    positive(human_fixture(baseline, action))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_gate",
        "missing_final",
        "wrong_content",
        "wrong_type",
        "wrong_generation",
        "early",
    ],
)
def test_human_transaction_counterexamples(baseline, mutation):
    data = human_fixture(baseline, "confirm")
    run = first_run(data)
    gate = next(e for e in run["events"] if e["step"] == "human_gate")
    final = run["events"][-1]
    if mutation.startswith("missing"):
        run["events"].remove(gate if mutation == "missing_gate" else final)
    elif mutation == "wrong_content":
        final["output_summary"] = {"status": "rejected"}
        final["output_checksum"] = canonical_checksum(final["output_summary"])
    elif mutation == "wrong_type":
        gate["event_type"] = "human_action_recorded"
    elif mutation == "wrong_generation":
        final["generation"] = 1
    else:
        time = datetime.fromisoformat(run["created_at"]) + timedelta(seconds=1)
        gate["occurred_at"] = time.isoformat()
        final["occurred_at"] = (time + timedelta(microseconds=1)).isoformat()
    sort_events(run)
    rejected(data, "human_before_summary" if mutation == "early" else "trace_human_")


def test_unknown_usage_is_still_complete(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    e = next(e for e in run["events"] if e["step"] == "model_attempt")
    e["output_summary"].update(usage_unknown=True, input_usage=None, output_usage=None)
    e["output_checksum"] = canonical_checksum(e["output_summary"])
    run["execution"]["policy"]["attempts"][0].update(
        usage_unknown=True, input_tokens=None, output_tokens=None
    )
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        run["execution"]["policy"]["attempts"]
    )
    run["execution"]["metering"].update(
        unknown_usage_attempts=1,
        usage_complete=False,
        input_tokens_known=100,
        output_tokens_known=40,
    )
    seal(data)
    positive(data)


def test_citation_correction_preserves_distinct_errors():
    data = fixture_v5(True)
    run = first_run(data)
    run["attempts"][0].update(
        first_error="citation_missing", last_error="citation_missing"
    )
    run["execution"]["policy"]["attempts"][0]["error_code"] = "citation_missing"
    receipt = next(e for e in run["events"] if e["step"] == "model_attempt")
    receipt["event_type"] = "step_completed"
    receipt["output_summary"].update(
        outcome="completed", attempt_error=None, next_correction_kind="citation"
    )
    receipt["output_checksum"] = canonical_checksum(receipt["output_summary"])
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        run["execution"]["policy"]["attempts"]
    )
    run["execution"]["metering"].update(contract_corrections=0, citation_corrections=1)
    seal(data)
    positive(data)


def partial_grounding_fixture(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    # A final answer with surviving duties can be adopted without another retry.
    run["attempts"][0].update(first_error="citation_missing", last_error=None)
    branch = run["execution"]["policy"]
    branch.update(
        support_failures=["citation_missing"], failure_code="citation_missing"
    )
    branch["attempts"][0]["error_code"] = "citation_missing"
    commit = run["checkpoints"][0]["result_payload"]["result"]
    commit["support_failures"] = ["citation_missing"]
    commit["attempts"] = copy.deepcopy(branch["attempts"])
    return seal(data)


def test_final_grounding_refusal_can_adopt_surviving_duties(baseline):
    positive(partial_grounding_fixture(baseline))


def test_final_grounding_error_needs_grounding_evidence(baseline):
    data = partial_grounding_fixture(baseline)
    first_run(data)["execution"]["policy"]["support_failures"] = []
    rejected(data, "ledger_grounding_error")


@pytest.mark.parametrize("workers", [("policy",), ("policy", "contract")])
def test_late_receipts_may_settle_without_adoption(baseline, workers):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    run["checkpoints"] = [
        cp for cp in run["checkpoints"] if cp["worker"] not in workers
    ]
    deadline = datetime.fromisoformat(run["generations"][0]["deadline_at"])
    for index, worker in enumerate(workers, 1):
        run["execution"][worker].update(
            outcome="failed", failure_code="deadline_exhausted", reported=[], spans=[]
        )
        a = next(a for a in run["attempts"] if a["worker"] == worker)
        a.update(
            discarded_late=True,
            finished_at=(deadline + timedelta(seconds=index)).isoformat(),
        )
        e = next(e for e in run["events"] if e.get("worker") == worker)
        e["occurred_at"] = a["finished_at"]
    sort_events(run)
    seal(data)
    positive(data)


def test_late_receipt_errors_still_reconcile(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    run["checkpoints"] = [cp for cp in run["checkpoints"] if cp["worker"] != "policy"]
    run["execution"]["policy"].update(
        outcome="failed", failure_code="deadline_exhausted", reported=[], spans=[]
    )
    row = run["attempts"][0]
    row.update(discarded_late=True, last_error="invalid_schema")
    seal(data)
    rejected(data, "ledger_last_error")


def unreturned_recovery_fixture(baseline):
    data = recovery_fixture(baseline)
    run = first_run(data)
    start = datetime.fromisoformat(run["generations"][-1]["created_at"])
    finish = start + timedelta(seconds=2)
    old = run["attempts"][0]
    latest = copy.deepcopy(old)
    latest.update(
        generation=2,
        attempt_no=2,
        started_at=(start + timedelta(milliseconds=1)).isoformat(),
        finished_at=finish.isoformat(),
    )
    old.update(
        outcome="abandoned",
        abandoned=True,
        finished_at=(start - timedelta(seconds=5)).isoformat(),
    )
    run["attempts"].append(latest)
    run["checkpoints"][0].update(
        generation=2, attempt_no=2, adopted_at=finish.isoformat()
    )
    branch = run["execution"]["policy"]
    branch["attempts"][0]["attempt_no"] = 2
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        branch["attempts"]
    )
    receipt = next(
        e
        for e in run["events"]
        if e["step"] == "model_attempt" and e["worker"] == "policy"
    )
    receipt.update(generation=2, attempt=2, occurred_at=finish.isoformat())
    receipt["output_summary"]["attempt_no"] = 2
    receipt["output_checksum"] = canonical_checksum(receipt["output_summary"])
    sort_events(run)
    from procurement_review.precise_run_state import metering_from_ledger

    receipts = [
        {**e["output_summary"], "attempt": e["attempt"]}
        for e in run["events"]
        if e["step"] == "model_attempt"
    ]
    run["execution"]["metering"] = metering_from_ledger(
        run["attempts"],
        receipts=receipts,
        requested_model=data[0]["binding"]["requested_model"],
        resolved_models=[e["receipt_model"] for e in receipts],
        explicit_recoveries=1,
        reused_branches=1,
    ).model_dump(mode="json")
    return seal(data)


def test_abandoned_unreturned_recovery_preserves_unknown(baseline):
    data = unreturned_recovery_fixture(baseline)
    positive(data)
    meter = first_run(data)["execution"]["metering"]
    assert meter["unknown_usage_attempts"] == 1 and meter["usage_complete"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("first_error", "invalid_schema"),
        ("last_error", "invalid_schema"),
        ("outcome", "completed"),
    ],
)
def test_unreturned_attempt_has_only_its_abandonment_state(baseline, field, value):
    data = unreturned_recovery_fixture(baseline)
    first_run(data)["attempts"][0][field] = value
    rejected(data, "ledger_unreturned_state")
