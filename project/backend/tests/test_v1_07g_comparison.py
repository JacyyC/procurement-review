"""G's neutral Q-12: plan-driven denominators, win/tie/loss, and honest gaps.

Every expectation here is hand-built or read back from a frozen plan.  Nothing in
this module is model quality evidence, and no test sends a request.
"""

from __future__ import annotations

import json

import pytest

from procurement_review.eval_g_comparison import score_comparison
from procurement_review.eval_g_contracts import (
    DOCUMENT_OBLIGATIONS,
    ArmPlan,
    ArmShape,
    ClaimCollectionStatus,
    ComparisonArm,
    ComparisonObservationV1,
    HistoryMetering,
    ModelBranch,
    PriorLedger,
    TerminalState,
)
from procurement_review.eval_g_plan import (
    arm_shapes,
    comparison_implementation_fingerprints,
    history_metering,
    load_plan,
)
from procurement_review.eval_g_report import build_report_v5
from procurement_review.eval_v2_contracts import EvalTopology
from procurement_review.eval_v2_metrics import score_predictions
from procurement_review.gold_v3_contracts import V3Recommendation, V3Split
from v1_07b_support import prediction_for, v3_bundle
from v1_07g_support import SEEDS, frozen_plan, observation, resources

MULTI = ComparisonArm.MULTI_AGENT
SINGLE = ComparisonArm.SINGLE_AGENT


def _plan(case_ids=("V107-D-0001", "V107-D-0002")):
    return frozen_plan(list(case_ids))


def _pair(plan, *, multi="recommend_approve", single="recommend_approve", **kwargs):
    """One settled observation per arm for every planned Case and repeat."""

    rows = []
    for arm, recommendation in ((MULTI, multi), (SINGLE, single)):
        for case_id, seed in plan.arm[arm].observation_keys():
            rows.append(
                observation(
                    arm=arm,
                    case_id=case_id,
                    repeat_seed=seed,
                    plan_sha256=plan.plan_sha256,
                    recommendation=recommendation,
                    first_send_count=len(plan.arm[arm].shape.model_branches),
                    first_pass_valid_count=len(plan.arm[arm].shape.model_branches),
                    physical_request_count=len(plan.arm[arm].shape.model_branches),
                    **kwargs,
                )
            )
    return rows


def _resources(plan, *, multi_requests=8, single_requests=4, namespace=None):
    scope = namespace or plan.history.namespace
    return {
        MULTI: resources(
            arm=MULTI,
            namespace=scope,
            physical_requests=multi_requests,
            known_input_tokens=800,
            known_output_tokens=1_600,
        ),
        SINGLE: resources(
            arm=SINGLE,
            namespace=scope,
            physical_requests=single_requests,
            known_input_tokens=800,
            known_output_tokens=800,
        ),
    }


def _expected(plan, value="recommend_approve"):
    return {case_id: value for case_id in plan.arm[MULTI].case_ids}


def _ledger_matching(plan, observations, resources_map):
    """Reconcile the arm ledgers with the observations the batch settled."""

    for arm, item in resources_map.items():
        rows = [row for row in observations if row.arm is arm]
        item.__dict__  # frozen model; rebuild instead
        resources_map[arm] = item.model_copy(
            update={
                "physical_requests": sum(row.physical_request_count for row in rows),
                "known_input_tokens": sum(row.known_input_tokens or 0 for row in rows),
                "known_output_tokens": sum(
                    row.known_output_tokens or 0 for row in rows
                ),
                "usage_unknown_attempts": sum(
                    row.usage_unknown_attempts for row in rows
                ),
                "receipts": sum(row.physical_request_count for row in rows),
                "unreturned_attempts": 0,
            }
        )
    return resources_map


def test_plan_denominators_are_preregistered_and_exclude_zero_call_branches():
    plan = _plan()
    multi, single = plan.arm[MULTI], plan.arm[SINGLE]
    assert plan.planned_observations() == 8
    # Two isolated document first sends versus one joint first send per observation.
    assert (multi.planned_first_sends, single.planned_first_sends) == (8, 4)
    assert (multi.max_physical_requests, single.max_physical_requests) == (16, 8)
    assert plan.planned_physical_capacity() == 24
    # The deterministic procurement and control branches issue no model request,
    # so they never enter a first-send denominator.
    assert multi.shape.deterministic_branches == ("procurement", "control")
    assert "procurement" not in {b.branch_id for b in multi.shape.model_branches}
    assert multi.planned_first_sends == len(multi.task_keys())
    assert single.planned_first_sends == len(single.task_keys())
    assert plan.arm[MULTI].shape.obligation_coverage() == tuple(
        sorted(
            set(DOCUMENT_OBLIGATIONS) | {"O-03", "O-04", "O-07", "O-10", "O-11", "O-12"}
        )
    )


def test_unfair_plans_are_rejected_before_any_request():
    multi, _single = arm_shapes()
    with pytest.raises(ValueError, match="identical registered case set"):
        _rebuild_with_unequal_cases()
    with pytest.raises(ValueError, match="exactly one joint first send"):
        ArmShape(
            arm_id=SINGLE,
            topology=EvalTopology.SINGLE_AGENT,
            model_branches=(
                ModelBranch(branch_id="joint", obligation_ids=DOCUMENT_OBLIGATIONS[:3]),
                ModelBranch(branch_id="extra", obligation_ids=DOCUMENT_OBLIGATIONS[3:]),
            ),
        )
    with pytest.raises(ValueError, match="keeps exactly its own role"):
        ArmShape(
            arm_id=MULTI,
            topology=EvalTopology.MULTI_AGENT,
            model_branches=(
                # The full duty catalog is still covered, but one isolated branch
                # now reaches across into the other document's role.
                ModelBranch(
                    branch_id="policy", obligation_ids=("O-01", "O-02", "O-05")
                ),
                ModelBranch(
                    branch_id="contract", obligation_ids=("O-06", "O-08", "O-09")
                ),
            ),
        )
    with pytest.raises(ValueError, match="two shared attempts per first send"):
        ArmPlan(
            shape=multi,
            case_ids=("V107-D-0001",),
            repeat_seeds=SEEDS,
            planned_observations=2,
            planned_first_sends=4,
            max_physical_requests=12,
        )


def _rebuild_with_unequal_cases():
    from procurement_review.eval_g_contracts import ComparisonPlanV1
    from procurement_review.eval_g_plan import fairness
    from v1_07g_support import FAIRNESS_FINGERPRINTS

    multi, single = arm_shapes()
    left = ArmPlan(
        shape=multi,
        case_ids=("V107-D-0001",),
        repeat_seeds=SEEDS,
        planned_observations=2,
        planned_first_sends=4,
        max_physical_requests=8,
    )
    right = ArmPlan(
        shape=single,
        case_ids=("V107-D-0002",),
        repeat_seeds=SEEDS,
        planned_observations=2,
        planned_first_sends=2,
        max_physical_requests=4,
    )
    return ComparisonPlanV1(
        plan_id="v1-07g-unequal",
        stage_id="V1-07G",
        split="dev",
        gold_set_version="gold-set.v3.1",
        fairness=fairness(truth_contract="v1-07-truth.v1.1", **FAIRNESS_FINGERPRINTS),
        history=history_metering(),
        arms=(left, right),
        authorization="test",
        stop_conditions=("cancelled",),
    )


@pytest.mark.parametrize(
    ("single_answer", "expected_outcome", "expected_value"),
    [
        ("recommend_reject", "multi_agent_ahead", 1.0),
        ("recommend_approve", "tie", 0.0),
        ("recommend_approve", "single_agent_ahead", -1.0),
    ],
)
def test_win_tie_and_loss_all_settle_and_none_of_them_gates(
    single_answer, expected_outcome, expected_value
):
    plan = _plan()
    truth = {
        "V107-D-0001": "recommend_approve",
        "V107-D-0002": "recommend_approve",
    }
    if expected_outcome == "single_agent_ahead":
        truth = {key: "recommend_reject" for key in truth}
        rows = _pair(plan, multi="recommend_approve", single="recommend_reject")
    else:
        rows = _pair(plan, multi="recommend_approve", single=single_answer)
    result = score_comparison(
        plan,
        rows,
        expected=truth,
        resources=_ledger_matching(plan, rows, _resources(plan)),
    )
    gate = result["Q-12"]
    delta = gate["paired_delta"]["multi_minus_single_accuracy"]
    assert delta["value"] == expected_value
    assert gate["paired_delta"]["outcome"] == expected_outcome
    # A loss is a settled comparison, not a failed one.
    assert gate["complete"] is True
    assert gate["meets_threshold"] is True
    assert gate["direction_gate"] is None
    assert gate["incomplete_reasons"] == []
    assert delta["paired_cases"] == 2 and delta["paired_observations"] == 4


def test_a_whole_missing_arm_keeps_its_planned_denominator():
    plan = _plan()
    rows = [row for row in _pair(plan) if row.arm is MULTI]
    result = score_comparison(
        plan,
        rows,
        expected=_expected(plan),
        resources=_ledger_matching(plan, rows, _resources(plan)),
    )
    coverage = result["Q-12"]["coverage"]
    assert coverage["planned_observations"] == 8
    assert coverage["terminal_observations"] == 4
    assert coverage["coverage"]["denominator"] == 8
    assert coverage["coverage"]["value"] == 0.5
    assert coverage["by_arm"]["single_agent"]["planned_observations"] == 4
    assert coverage["by_arm"]["single_agent"]["terminal_observations"] == 0
    reasons = result["Q-12"]["incomplete_reasons"]
    assert "arm_missing:single_agent" in reasons
    assert "unsettled_pair" in reasons
    assert result["Q-12"]["complete"] is False
    assert result["Q-12"]["paired_delta"]["outcome"] is None


def test_the_old_v1_coverage_denominator_shrinks_with_a_missing_arm():
    """The frozen A～F consumer is unchanged; this is why G needed a new version."""

    _cases, truths_file, _manifest = v3_bundle()
    truths = list(truths_file.truths)
    dev = [item for item in truths if item.case_id.startswith("V107-D-")]
    predictions = [
        prediction_for(truth, topology=EvalTopology.MULTI_AGENT, repeat_seed=seed)
        for seed in SEEDS
        for truth in dev
    ]
    legacy = score_predictions(truths, predictions, split=V3Split.DEV)
    # Only the multi-agent arm ever ran, yet the old coverage still reads complete
    # because its denominator is derived from the slices that happened to appear.
    assert legacy["observation_coverage"]["denominator"] == len(dev) * len(SEEDS)
    assert legacy["observation_coverage"]["meets_threshold"] is True
    assert legacy["Q-12"]["topology"] == {}
    plan = _plan([item.case_id for item in dev])
    rows = [
        observation(
            arm=MULTI,
            case_id=item.case_id,
            repeat_seed=seed,
            plan_sha256=plan.plan_sha256,
            first_send_count=2,
            physical_request_count=2,
        )
        for seed in SEEDS
        for item in dev
    ]
    result = score_comparison(
        plan,
        rows,
        expected={item.case_id: item.expected_recommendation for item in dev},
        resources={
            MULTI: resources(
                arm=MULTI,
                namespace=plan.history.namespace,
                physical_requests=len(rows) * 2,
                known_input_tokens=1,
                known_output_tokens=1,
            )
        },
    )
    coverage = result["Q-12"]["coverage"]
    assert coverage["planned_observations"] == 2 * len(dev) * len(SEEDS)
    assert coverage["coverage"]["value"] == 0.5
    assert "arm_missing:single_agent" in result["Q-12"]["incomplete_reasons"]


def test_a_missing_repeat_or_projection_never_settles():
    plan = _plan()
    complete = _pair(plan)
    without_repeat = [
        row
        for row in complete
        if not (row.arm is SINGLE and row.repeat_seed == SEEDS[1])
    ]
    result = score_comparison(
        plan,
        without_repeat,
        expected=_expected(plan),
        resources=_ledger_matching(plan, without_repeat, _resources(plan)),
    )
    assert "repeat_missing" in result["Q-12"]["incomplete_reasons"]
    assert result["Q-12"]["records"]["unsettled_pairs"]["count"] == 2

    blocked = [
        row.model_copy(
            update={
                "claim_collection_status": ClaimCollectionStatus.BLOCKED_BY_PROTECTION,
                "claim_count": 0,
                "claim_digest_sha256": None,
            }
        )
        if row.arm is MULTI
        else row
        for row in complete
    ]
    result = score_comparison(
        plan,
        blocked,
        expected=_expected(plan),
        resources=_ledger_matching(plan, blocked, _resources(plan)),
    )
    # A protection block keeps the audit unfinished; zero claims is never a pass.
    assert "claim_collection_incomplete" in result["Q-12"]["incomplete_reasons"]
    assert result["Q-12"]["complete"] is False


def test_unknown_cost_is_declared_and_known_cost_is_compared():
    plan = _plan()
    rows = _pair(plan)
    unknown = _ledger_matching(plan, rows, _resources(plan))
    result = score_comparison(plan, rows, expected=_expected(plan), resources=unknown)
    comparison = result["Q-12"]["resource_comparison"]
    assert comparison["cost_status"] == "unknown"
    assert comparison["cost_ratio"] is None
    assert result["Q-12"]["complete"] is True
    known = {
        arm: item.model_copy(
            update={"cost_status": "known", "cost_value": 2.0 if arm is MULTI else 1.0}
        )
        for arm, item in unknown.items()
    }
    result = score_comparison(plan, rows, expected=_expected(plan), resources=known)
    assert result["Q-12"]["resource_comparison"]["cost_status"] == "known"
    assert result["Q-12"]["resource_comparison"]["cost_ratio"] == 2.0
    assert result["Q-12"]["complete"] is True


def test_missing_known_tokens_or_timing_is_incomplete_not_estimated():
    """The contract rejects them outright; the scorer still fails closed on a tamper."""

    plan = _plan()
    settled = _pair(plan)[0]
    for field, message in (
        ("known_input_tokens", "settled usage"),
        ("total_latency_ms", "measured latency"),
    ):
        body = settled.model_dump(mode="json")
        body[field] = None
        if field == "known_input_tokens":
            body["known_output_tokens"] = None
        with pytest.raises(ValueError, match=message):
            ComparisonObservationV1.model_validate_json(json.dumps(body))
    # A tampered durable record that skipped validation is still caught downstream,
    # and it is never estimated into a pass.
    for field, reason in (
        ("known_input_tokens", "known_token_usage_missing"),
        ("total_latency_ms", "known_timing_missing"),
    ):
        rows = [
            ComparisonObservationV1.model_construct(
                **{
                    **row.model_dump(),
                    field: None,
                    **(
                        {"known_output_tokens": None}
                        if field != "total_latency_ms"
                        else {}
                    ),
                }
            )
            for row in _pair(plan)
        ]
        result = score_comparison(
            plan,
            rows,
            expected=_expected(plan),
            resources=_ledger_matching(plan, rows, _resources(plan)),
        )
        assert reason in result["Q-12"]["incomplete_reasons"]
        assert result["Q-12"]["complete"] is False


def test_declared_unknown_usage_stays_visible_and_keeps_the_totals_a_lower_bound():
    plan = _plan()
    rows = [
        row.model_copy(
            update={
                "known_input_tokens": 0,
                "known_output_tokens": 0,
                "usage_unknown_attempts": row.physical_request_count,
            }
        )
        for row in _pair(plan)
    ]
    result = score_comparison(
        plan,
        rows,
        expected=_expected(plan),
        resources=_ledger_matching(plan, rows, _resources(plan)),
    )
    comparison = result["Q-12"]["resource_comparison"]
    assert comparison["token_totals_are_lower_bound"] is True
    assert comparison["declared_unknown_usage_attempts"] == 12
    assert result["Q-12"]["records"]["declared_unknown_usage"]["count"] == 8
    # Honest, locatable and still unverified: a declared unknown does not settle
    # the token comparison, and it is never estimated into a pass.
    assert "resource_usage_unknown" in result["Q-12"]["incomplete_reasons"]
    assert result["Q-12"]["complete"] is False


def test_unfair_inputs_and_ledger_drift_are_incomplete():
    plan = _plan()
    rows = _pair(plan)
    ledger = _ledger_matching(plan, rows, _resources(plan))
    unfair = score_comparison(
        plan,
        rows,
        expected=_expected(plan),
        resources=ledger,
        observed_fairness={"expected_resolved_model": "some-other-model"},
    )
    assert "unfair_input" in unfair["Q-12"]["incomplete_reasons"]
    assert unfair["Q-12"]["fairness"]["mismatches"]["expected_resolved_model"] == {
        "registered": "deepseek-v4-1-flash",
        "observed": "some-other-model",
    }
    wrong_namespace = _ledger_matching(
        plan, rows, _resources(plan, namespace="v1-07-retired-g-h")
    )
    drifted = score_comparison(
        plan, rows, expected=_expected(plan), resources=wrong_namespace
    )
    assert "metering_namespace_mismatch" in drifted["Q-12"]["incomplete_reasons"]
    inflated = {
        MULTI: ledger[MULTI].model_copy(update={"physical_requests": 999}),
        SINGLE: ledger[SINGLE],
    }
    result = score_comparison(plan, rows, expected=_expected(plan), resources=inflated)
    reasons = result["Q-12"]["incomplete_reasons"]
    assert "ledger_mismatch:multi_agent" in reasons
    assert "physical_capacity_exceeded" in reasons


def test_a_failed_first_send_stays_in_the_q01_denominator():
    plan, rows, expected, ledger = _pair_plan_with_failure()
    result = score_comparison(plan, rows, expected=expected, resources=ledger)
    per_arm = result["Q-12"]["first_send_validity_by_arm"]["multi_agent"]
    assert per_arm["denominator"] == 8
    # Two planned first sends came back invalid; the denominator stays at eight.
    assert per_arm["numerator"] == 4
    assert per_arm["meets_threshold"] is False
    assert (
        result["Q-12"]["first_send_validity_by_arm"]["single_agent"]["denominator"] == 4
    )
    assert per_arm["deterministic_zero_call_branches"] == ["procurement", "control"]


def _pair_plan_with_failure():
    plan = _plan()
    rows = _pair(plan)
    # Two of the eight planned multi-agent first sends came back invalid.
    rows = [
        row.model_copy(update={"first_pass_valid_count": 0})
        if row.arm is MULTI and row.repeat_seed == SEEDS[0]
        else row
        for row in rows
    ]
    return plan, rows, _expected(plan), _ledger_matching(plan, rows, _resources(plan))


def test_a_zero_call_branch_is_neither_a_first_send_nor_a_gap():
    plan = _plan()
    rows = [
        row.model_copy(
            update={
                "first_send_count": 0,
                "first_pass_valid_count": 0,
                "physical_request_count": 0,
                "known_input_tokens": 0,
                "known_output_tokens": 0,
            }
        )
        for row in _pair(plan)
    ]
    result = score_comparison(
        plan,
        rows,
        expected=_expected(plan),
        resources=_ledger_matching(plan, rows, _resources(plan)),
    )
    # Nothing was issued, so the shortfall is a real accounting gap, not a pass.
    assert "first_send_shortfall" in result["Q-12"]["incomplete_reasons"]
    assert result["Q-12"]["records"]["first_send_shortfall"] == {
        "multi_agent": 8,
        "single_agent": 4,
    }
    assert (
        result["Q-12"]["coverage"]["by_arm"]["multi_agent"]["coverage"][
            "meets_threshold"
        ]
        is True
    )


def test_history_namespaces_stay_separate_and_are_read_from_real_artifacts():
    metering = history_metering()
    assert metering.namespace == "v1-07g-comparison"
    scopes = {item.namespace: item for item in metering.prior_ledgers}
    assert "v1-07-retired-g-h" in scopes
    retired = scopes["v1-07-retired-g-h"]
    # The retired inventory is read back at its registered cardinality, not zeroed.
    assert (retired.physical_sent, retired.receipts) == (4993, 4988)
    assert retired.unreturned == 5 and retired.usage_unknown == 11
    assert retired.cost_status == "unknown" and retired.cost_value is None
    assert "v1-02-v1-05-real-line" in scopes
    with pytest.raises(ValueError, match="must differ from every prior scope"):
        HistoryMetering(
            namespace="v1-07g-duplicate",
            prior_ledgers=(
                PriorLedger(
                    namespace="v1-07g-duplicate",
                    source_entry="test",
                    source_status="registered_only",
                    count_basis="exact",
                    physical_sent=1,
                    receipts=1,
                    unreturned=0,
                    usage_unknown=0,
                    known_input_tokens=0,
                    known_output_tokens=0,
                ),
            ),
        )
    with pytest.raises(ValueError, match="current G/H line"):
        HistoryMetering(namespace="v1-07f-oracle-runner", prior_ledgers=())


def test_a_frozen_plan_detects_tampering_on_read_back(tmp_path):
    plan = _plan()
    path = tmp_path / "plan.json"
    import json

    path.write_text(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False))
    assert load_plan(path).plan_sha256 == plan.plan_sha256
    tampered = json.loads(path.read_text())
    tampered["authorization"] = "a-different-authorization"
    path.write_text(json.dumps(tampered, ensure_ascii=False))
    with pytest.raises(ValueError, match="does not match its content"):
        load_plan(path)


def test_report_v5_separates_product_quality_from_comparison_completeness():
    plan = _plan()
    rows = _pair(plan)
    ledger = _ledger_matching(plan, rows, _resources(plan))
    quality = {
        MULTI: {
            "scoring_version": "v1-07g-score.v1",
            "plan_denominator_matches": True,
            "combined": {
                "counts": {"cases": 4},
                "per_class": {},
                "runner_p95_ms": 1_500,
                "execution_p95_ms": 1_490,
                "gates": {"Q-03_recommendation_accuracy": True, "Q-02": None},
            },
            "consistency": {"value": 1.0, "meets_threshold": True},
        },
        SINGLE: {
            "scoring_version": "v1-07g-score.v1",
            "plan_denominator_matches": True,
            "combined": {
                "counts": {"cases": 4},
                "per_class": {},
                "runner_p95_ms": 900,
                "execution_p95_ms": 890,
                "gates": {"Q-03_recommendation_accuracy": False},
            },
            "consistency": {"value": 1.0, "meets_threshold": True},
        },
    }
    report = build_report_v5(
        plan,
        rows,
        expected=_expected(plan),
        resources=ledger,
        arm_quality=quality,
        implementation=comparison_implementation_fingerprints(),
    )
    assert report["schema_version"] == "eval-report.v5"
    assert report["stage_acceptance"] is False
    assert report["network_calls"] == 0
    product = report["product_quality"]
    # An undetermined product gate fails closed instead of being reported as a pass.
    assert product["multi_agent"]["meets_product_gates"] is False
    assert product["multi_agent"]["undetermined_gates"] == ["Q-02"]
    assert product["single_agent"]["failed_gates"] == ["Q-03_recommendation_accuracy"]
    # The comparison is complete and settled even though one arm's quality failed.
    assert report["Q-12"]["complete"] is True
    assert report["Q-12"]["paired_delta"]["outcome"] == "tie"
    assert report["history_metering"]["namespace"] == plan.history.namespace
    assert report["implementation"]


def test_unregistered_and_duplicate_observations_are_rejected():
    plan = _plan()
    rows = _pair(plan)
    stranger = observation(
        arm=MULTI,
        case_id="V107-D-9999",
        repeat_seed=SEEDS[0],
        plan_sha256=plan.plan_sha256,
        first_send_count=2,
        physical_request_count=2,
    )
    result = score_comparison(
        plan,
        [*rows, stranger],
        expected=_expected(plan),
        resources=_ledger_matching(plan, rows, _resources(plan)),
    )
    assert "unregistered_observation" in result["Q-12"]["incomplete_reasons"]
    duplicated = score_comparison(
        plan,
        [*rows, rows[0]],
        expected=_expected(plan),
        resources=_ledger_matching(plan, rows, _resources(plan)),
    )
    assert "duplicate_observation" in duplicated["Q-12"]["incomplete_reasons"]
    foreign = [row.model_copy(update={"plan_sha256": "f" * 64}) for row in rows]
    result = score_comparison(
        plan,
        foreign,
        expected=_expected(plan),
        resources=_ledger_matching(plan, [], _resources(plan)),
    )
    assert "plan_digest_mismatch" in result["Q-12"]["incomplete_reasons"]


def test_an_unsettled_observation_is_terminal_but_not_comparable():
    plan = _plan()
    rows = _pair(plan)
    rows[0] = rows[0].model_copy(
        update={
            "terminal_state": TerminalState.CANCELLED,
            "recommendation": None,
            "decision_sha256": None,
            "claim_collection_status": ClaimCollectionStatus.ABSENT,
            "claim_count": 0,
            "claim_digest_sha256": None,
            "total_latency_ms": None,
            "first_pass_valid_count": 0,
            "stop_reason": "authorization_revoked",
        }
    )
    result = score_comparison(
        plan,
        rows,
        expected=_expected(plan),
        resources=_ledger_matching(plan, rows, _resources(plan)),
    )
    reasons = result["Q-12"]["incomplete_reasons"]
    assert "unsettled_observations" in reasons and "unsettled_pair" in reasons
    assert result["Q-12"]["complete"] is False
    assert (
        result["Q-12"]["paired_delta"]["multi_minus_single_accuracy"]["unsettled_pairs"]
        == 1
    )


def test_a_rejected_observation_contract_cannot_claim_a_response_it_lacks():
    plan = _plan()
    with pytest.raises(ValueError, match="settled usage"):
        observation(
            arm=MULTI,
            case_id="V107-D-0001",
            repeat_seed=SEEDS[0],
            plan_sha256=plan.plan_sha256,
            known_input_tokens=None,
            known_output_tokens=None,
        )
    with pytest.raises(ValueError, match="adopts no decision"):
        ComparisonObservationV1(
            arm=MULTI,
            case_id="V107-D-0001",
            repeat_seed=SEEDS[0],
            plan_sha256=plan.plan_sha256,
            terminal_state=TerminalState.FAILED,
            recommendation=V3Recommendation.APPROVE,
            stop_reason="document_model_failed",
        )
    with pytest.raises(ValueError, match="cannot exceed the first sends"):
        observation(
            arm=MULTI,
            case_id="V107-D-0001",
            repeat_seed=SEEDS[0],
            plan_sha256=plan.plan_sha256,
            first_send_count=1,
            first_pass_valid_count=1,
            corrected_valid_count=1,
        )
