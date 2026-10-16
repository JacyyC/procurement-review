"""Independent planned-denominator, source, timing and paired-statistic expectations."""

import copy
from types import SimpleNamespace

import pytest

from procurement_review.eval_h_contracts import ARMS
from procurement_review.eval_h_fixture import fixture_case, fixture_label
from procurement_review.eval_h_journal import Journal
from procurement_review.eval_h_material import source_identities
from procurement_review.eval_h_projection import project
from procurement_review.eval_h_report import paired_statistics
from procurement_review.eval_h_score import latency, repeat_score
from procurement_review.gold_v38_dev_quality import dev_quality_cases
from test_v1_07h_preparation import pg_batch as _pg_batch

pg_batch = _pg_batch


def inputs(seal):
    cases = seal.runner_cases()
    labels = tuple(fixture_label(authored) for authored in dev_quality_cases())
    observations = [
        Journal(seal.directory).committed(*key)
        for key in seal.plan().observation_keys()
    ]
    return seal.plan(), cases, labels, observations


def test_first_answer_needs_independent_semantics_even_with_success_receipt(pg_batch):
    seal, _ = pg_batch
    plan, cases, labels, observations = inputs(seal)
    baseline = repeat_score(plan, cases, labels, observations, 101)
    assert baseline["multi_agent"]["q01"]["first"]["numerator"] == 24
    changed = copy.deepcopy(observations)
    own = next(
        row
        for row in changed
        if row["repeat_seed"] == 101 and row["arm"] == "multi_agent"
    )
    own["row"]["spans"][0]["value"] = "unsupported independent value"
    result = repeat_score(plan, cases, labels, changed, 101)
    assert result["multi_agent"]["q01"]["first"]["numerator"] == 23
    assert result["multi_agent"]["q01"]["first"]["denominator"] == 24
    assert result["multi_agent"]["gates"]["Q-01"] is False
    assert result["single_agent"]["q01"]["first"]["numerator"] == 12


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "wrong.v1"),
        ("content_sha256", "0" * 64),
        ("locator", "wrong-locator"),
        ("source_id", "cross-scope-source"),
    ],
)
def test_q08_requires_whole_identity_without_shrinking_the_denominator(
    pg_batch, field, value
):
    seal, _ = pg_batch
    plan, cases, labels, observations = inputs(seal)
    changed = copy.deepcopy(observations)
    own = next(
        row
        for row in changed
        if row["repeat_seed"] == 101 and row["arm"] == "multi_agent"
    )
    own["retrieval"]["policy"]["original"][0][field] = value
    scored = repeat_score(plan, cases, labels, changed, 101)["multi_agent"]
    metric = scored["q08"]["policy:original"]
    assert metric["numerator"] == 11 and metric["denominator"] == 12
    assert metric["invalid_identity"] + metric["out_of_scope"] > 0
    assert scored["gates"]["Q-08"] is False


def test_q08_counts_required_evidence_and_keeps_negative_recall_separate(pg_batch):
    seal, _ = pg_batch
    plan, cases, labels, observations = inputs(seal)
    case = cases[0]
    extra = case.documents[0].model_copy(
        update={
            "source_id": "extra-policy-source",
            "document_id": "extra-policy-document",
        }
    )
    assert extra.role == "policy"
    case = case.model_copy(update={"documents": (*case.documents, extra)})
    label = next(label for label in labels if label.case_id == case.case_id)
    label = label.model_copy(
        update={
            "required_sources": {
                **label.required_sources,
                "policy": (*label.required_sources["policy"], extra.source_id),
            }
        }
    )
    one_plan = plan.model_copy(update={"case_ids": (case.case_id,)})
    result = repeat_score(one_plan, [case], [label], observations, 101)
    assert result["multi_agent"]["q08"]["policy:original"]["denominator"] == 2
    assert result["multi_agent"]["q08"]["policy:original"]["numerator"] == 1
    changed = copy.deepcopy(observations)
    for own in changed:
        if own["case_id"] == case.case_id:
            for variant in ("original", "without_case_id"):
                own["retrieval"]["policy"][variant].append(
                    source_identities(one_plan, case)[extra.source_id]
                )
    positive = repeat_score(one_plan, [case], [label], changed, 101)
    assert positive["multi_agent"]["q08"]["policy:original"]["value"] == 1
    negative = label.model_copy(
        update={"required_sources": {**label.required_sources, "policy": ()}}
    )
    result = repeat_score(one_plan, [case], [negative], observations, 101)[
        "multi_agent"
    ]["q08"]["policy:original"]
    assert result["denominator"] == result["numerator"] == 0 and result["value"] is None
    assert result["negative_false_recall"] == 1


def test_all_sample_nearest_rank_and_deadline_include_failures():
    rows = [
        dict(
            elapsed_ms=index * 1000,
            row=dict(status="failed" if index >= 190 else "pending_confirmation"),
        )
        for index in range(1, 201)
    ]
    result = latency(rows, 200)
    assert result["all_sample_p95_ms"] == 190000
    assert result["success_p95_ms"] == 180000
    assert result["failed"] == 11 and result["censored"] == 0
    assert result["deadline_satisfied"] is False
    assert latency(rows[:-1], 200)["all_sample_p95_ms"] is None
    rows[0]["elapsed_ms"] = True
    assert latency(rows, 200)["all_sample_p95_ms"] is None


def test_h_projection_preserves_native_pipeline_failure_for_pending_human_review():
    binding = {"execution": {"decision": {"pipeline_failures": ["transport"]}}}
    result = project({"status": "pending_human_review", "findings": []}, binding)
    assert result["pipeline_failures"] == ["transport"]
    assert result["status"] == "pending_human_review"


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("failure", [True, False])
def test_q09_counts_pipeline_failure_per_run_and_keeps_business_unknown(arm, failure):
    authored = dev_quality_cases()[:2]
    cases = [fixture_case(item) for item in authored]
    labels = [fixture_label(item) for item in authored]
    plan = SimpleNamespace(
        case_ids=tuple(case.case_id for case in cases),
        tenant_id="h-metric-tenant",
        dataset_version="h-metric-facts.v1",
        document_version=cases[0].documents[0].version,
        index_version="h-metric-index.v1",
    )
    observations = [
        dict(
            case_id=case.case_id,
            arm=arm,
            repeat_seed=101,
            elapsed_ms=1000 + index,
            row=dict(
                status="pending_human_review",
                terminal_status=200,
                recommendation="insufficient_information",
                pipeline_failures=["document_branch_failed", "transport"]
                if failure
                else [],
            ),
            attempts=[],
        )
        for index, case in enumerate(cases)
    ]
    result = repeat_score(plan, cases, labels, observations, 101)[arm]
    assert result["metrics"]["q09_pipeline"] == {
        "non_expected_pipeline_failures": 2 if failure else 0,
        "planned_runs": 2,
    }
    assert result["gates"]["Q-09"] is (not failure)
    assert result["q09"]["all_sample_p95_ms"] == 1001
    assert result["q09"]["success_p95_ms"] == (None if failure else 1001)
    assert result["q09"]["failed"] == (2 if failure else 0)
    assert result["q09"]["censored"] == 0


def test_latency_separates_returned_failure_timeout_cancel_and_legal_recovery():
    rows = [
        dict(
            elapsed_ms=1000,
            row=dict(
                status="pending_human_review",
                terminal_status=200,
                pipeline_failures=["transport"],
            ),
        ),
        dict(
            elapsed_ms=90000,
            row=dict(
                status="pending_human_review",
                terminal_status=200,
                pipeline_failures=["timeout"],
            ),
        ),
        dict(elapsed_ms=2000, row=dict(status="cancelled", terminal_status=200)),
        dict(
            elapsed_ms=5000,
            row=dict(
                status="pending_confirmation",
                terminal_status=200,
                pipeline_failures=[],
                branches={
                    "policy": dict(
                        outcome="completed",
                        attempts=[dict(error_code="timeout"), dict(error_code=None)],
                    )
                },
            ),
        ),
    ]
    result = latency(rows, 4)
    assert result["all_sample_p95_ms"] == 90000
    assert result["success_p95_ms"] == 5000
    assert result["failed"] == 3 and result["censored"] == 2
    assert result["cancelled"] == 1


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("last_error", ["timeout", "server_error"])
def test_censoring_uses_last_durable_receipt_not_a_recovered_earlier_timeout(
    arm, last_error
):
    rows = [
        dict(
            arm=arm,
            elapsed_ms=60000,
            row=dict(
                status="failed",
                terminal_status=200,
                pipeline_failures=["document_model_failed"],
            ),
            attempts=[
                dict(
                    branch="joint" if arm == "single_agent" else "policy",
                    attempt_no=number,
                    receipt={"error_code": error, "attempt_error": error},
                )
                for number, error in ((1, "timeout"), (2, last_error))
            ],
        )
    ]
    result = latency(rows, 1)
    assert result["failed"] == 1 and result["success_p95_ms"] is None
    assert result["all_sample_p95_ms"] == 60000
    assert result["censored"] == int(last_error == "timeout")


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_paired_win_tie_loss_and_intervals_never_add_a_direction_gate(direction):
    authored = dev_quality_cases()[:2]
    cases = [fixture_case(item) for item in authored]
    truths = {item.case_id: fixture_label(item) for item in authored}
    plan = SimpleNamespace(repeat_seeds=(101, 202))
    rows = {}
    for case in cases:
        for seed in plan.repeat_seeds:
            for arm in ARMS:
                correct = direction == 0 or (arm == "multi_agent") == (direction == 1)
                rows[case.case_id, seed, arm] = dict(
                    recommendation=truths[case.case_id].recommendation.value
                    if correct
                    else "needs_human_review"
                )
    mean, interval, per_case, by_class = paired_statistics(plan, cases, truths, rows)
    assert mean == direction and interval == (direction, direction)
    assert set(per_case.values()) == {direction}
    assert (
        sum(
            item[{1: "win", 0: "tie", -1: "loss"}[direction]]
            for item in by_class.values()
        )
        == 4
    )
