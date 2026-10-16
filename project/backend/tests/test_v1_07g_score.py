"""The arm-agnostic scorer and the independent audit of a frozen batch.

The multi-agent projections come from the real frozen F pipeline behind a Mock
transport.  The joint-arm projection is a merged shape fixture: it proves the
scorer no longer hardcodes two branches.  Neither is model quality evidence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from procurement_review.eval_f_context import load_cases
from procurement_review.eval_f_score import score_batch
from procurement_review.eval_g_contracts import ComparisonArm, TerminalState
from procurement_review.eval_g_plan import arm_shapes
from procurement_review.eval_g_report import observation_from, resources_from_ledger
from procurement_review.eval_g_score import G_SCORE_V1, load_truth, score_arm
from procurement_review.eval_v2_contracts import EvalTopology
from procurement_review.eval_v2_report import build_report_v4
from v1_07b_support import prediction_for, v3_bundle
from v1_07g_support import (
    SEEDS,
    frozen_plan,
    joint_projection,
    multi_agent_projection,
    persist_batch,
    real_multi_agent_case,
    write_batch_inputs,
)

MULTI = ComparisonArm.MULTI_AGENT
SINGLE = ComparisonArm.SINGLE_AGENT
SCRIPT = Path(__file__).parents[2] / "scripts/comparison_report.py"
_spec = importlib.util.spec_from_file_location("comparison_report", SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.fixture(scope="module")
def real_batch(tmp_path_factory):
    """Two DEV cases x two repeats, scored for both registered arm shapes."""

    tmp_path = tmp_path_factory.mktemp("g-score")
    cases = load_cases("dev", "gold-set.v3.1")[:2]
    plan = frozen_plan([case.case_id for case in cases])
    multi, single = arm_shapes()
    multi_projections, single_projections, attempts = [], [], {MULTI: [], SINGLE: []}
    for case in cases:
        for seed in SEEDS:
            decision, branches, ledger_rows = real_multi_agent_case(
                tmp_path, case, seed=seed
            )
            projection = multi_agent_projection(
                case, decision, branches, plan_sha256=plan.plan_sha256, seed=seed
            )
            multi_projections.append(projection)
            attempts[MULTI].extend(ledger_rows)
            joint = joint_projection(projection, plan_sha256=plan.plan_sha256)
            single_projections.append(joint)
            attempts[SINGLE].append(
                {
                    "task_key": f"{case.case_id}.{seed}.joint",
                    "attempt_no": 1,
                    "receipt_json": json.dumps(
                        {
                            "usage_unknown": False,
                            "input_tokens": 100,
                            "output_tokens": 200,
                        }
                    ),
                }
            )
    return dict(
        root=tmp_path,
        plan=plan,
        cases={case.case_id: case for case in cases},
        shapes={MULTI: multi, SINGLE: single},
        projections={MULTI: multi_projections, SINGLE: single_projections},
        attempts=attempts,
    )


def test_the_branch_shape_comes_from_the_plan_not_from_the_scorer(real_batch):
    truths = load_truth("gold-set.v3.1")
    plan = real_batch["plan"]
    scored = {
        arm: score_arm(
            real_batch["cases"],
            truths,
            real_batch["projections"][arm],
            plan=plan.arm[arm],
        )
        for arm in (MULTI, SINGLE)
    }
    observations = len(plan.arm[MULTI].observation_keys())
    assert scored[MULTI]["planned_first_sends"] == 2 * observations
    assert scored[SINGLE]["planned_first_sends"] == observations
    for arm in (MULTI, SINGLE):
        result = scored[arm]
        assert result["contract"] == G_SCORE_V1
        assert result["plan_denominator_matches"] is True
        assert (
            result["combined"]["counts"]["planned_branches"]
            == (result["planned_first_sends"])
        )
        assert result["combined"]["counts"]["cases"] == observations
        # Both arms answer the identical twelve-duty catalog per Case.
        assert result["combined"]["counts"]["obligations"] == 12 * observations
        assert result["stage_acceptance"] is False
        gates = result["combined"]["gates"]
        assert "Q-01_first_pass" in gates and "Q-10_repeat_consistency" in gates
        assert gates["Q-11_no_forbidden_export"] is True
    # The joint arm is scored as one first send, not as two renamed branches.
    assert scored[SINGLE]["combined"]["counts"]["planned_branches"] == observations
    assert scored[MULTI]["combined"]["counts"]["planned_branches"] == 2 * observations


def test_a_missing_projection_keeps_its_planned_duties(real_batch):
    truths = load_truth("gold-set.v3.1")
    plan = real_batch["plan"]
    partial = real_batch["projections"][MULTI][:1]
    result = score_arm(real_batch["cases"], truths, partial, plan=plan.arm[MULTI])
    counts = result["combined"]["counts"]
    assert counts["missing_cases"] == len(plan.arm[MULTI].observation_keys()) - 1
    # The denominator is the registered plan, so a gap can never shrink it.
    assert counts["planned_branches"] == plan.arm[MULTI].planned_first_sends
    assert result["plan_denominator_matches"] is True
    assert counts["pipeline_failed"] >= counts["missing_cases"]
    assert result["combined"]["gates"]["Q-01_first_pass"] is False


def test_the_frozen_f_and_v4_consumers_still_read_their_own_history(real_batch):
    """G adds a version; it does not reinterpret the A～F consumers."""

    _cases, truths_file, _manifest = v3_bundle()
    truths = list(truths_file.truths)
    dev = [item for item in truths if item.case_id.startswith("V107-D-")][:4]
    predictions = [
        prediction_for(truth, topology=topology, repeat_seed=601)
        for topology in (EvalTopology.MULTI_AGENT, EvalTopology.SINGLE_AGENT)
        for truth in dev
    ]
    report = build_report_v4(truths, predictions)
    assert report["schema_version"] == "eval-report.v4"
    assert report["metrics_version"] == "v1-07-eval-metrics.v1"
    # The frozen Q-12 still requires a positive gain: that is the historical
    # contract, preserved on purpose, and the reason G needed a new version.
    assert (
        report["scores"]["dev"]["Q-12"]["topology"]["multi_minus_single_accuracy"][
            "threshold"
        ]
        == 0.03
    )
    from procurement_review.eval_v2_metrics import score_predictions
    from procurement_review.gold_v3_contracts import V3Split

    legacy = score_predictions(truths, predictions, split=V3Split.DEV)
    assert legacy["metrics_version"] == "v1-07-eval-metrics.v1"
    # F's scorer keeps its own frozen two-branch shape and its own contract name.
    assert (
        score_batch([], [], [], "gold-set.v3.1")["contract"] == "eval-precise-score.v1"
    )


def test_the_independent_audit_reads_a_frozen_batch_back(real_batch, tmp_path):
    plan = real_batch["plan"]
    folder = write_batch_inputs(
        tmp_path / "batch",
        plan,
        projections=real_batch["projections"],
        attempts=real_batch["attempts"],
    )
    _plan_body, report, resources, observations = audit.rebuild(folder)
    assert all(item.terminal_state is TerminalState.COMMITTED for item in observations)
    persist_batch(folder, plan, observations, resources, report)
    result = audit.build_audit(folder)
    assert result["audit_passed"] is True, result["audit_checks"]
    assert result["network_calls"] == 0
    assert result["stage_acceptance"] is False
    assert result["comparison_complete"] is True
    assert result["product_gates"]
    assert result["outcome"] in {"multi_agent_ahead", "tie", "single_agent_ahead"}
    assert result["resource_comparison"]["request_ratio"] == 2.0
    # Reading the batch twice gives the identical audit; nothing was rewritten.
    again = audit.build_audit(folder)
    assert again["source_sha256"] == result["source_sha256"]
    assert again["audit_checks"] == result["audit_checks"]


def test_the_audit_rejects_a_tampered_stored_result(real_batch, tmp_path):
    plan = real_batch["plan"]
    folder = write_batch_inputs(
        tmp_path / "batch",
        plan,
        projections=real_batch["projections"],
        attempts=real_batch["attempts"],
    )
    _plan_body, report, resources, observations = audit.rebuild(folder)
    persist_batch(folder, plan, observations, resources, report)
    stored = json.loads((folder / "result.json").read_text())
    stored["Q-12"]["paired_delta"]["outcome"] = "multi_agent_ahead"
    stored["Q-12"]["complete"] = True
    (folder / "result.json").write_text(json.dumps(stored, ensure_ascii=False))
    result = audit.build_audit(folder)
    assert result["audit_passed"] is False
    assert result["audit_checks"]["recomputation_matches_stored_result"] is False


def test_the_audit_reports_an_incomplete_batch_as_incomplete(real_batch, tmp_path):
    plan = real_batch["plan"]
    partial = {MULTI: real_batch["projections"][MULTI], SINGLE: []}
    folder = write_batch_inputs(
        tmp_path / "batch",
        plan,
        projections=partial,
        attempts={MULTI: real_batch["attempts"][MULTI], SINGLE: []},
    )
    _plan_body, report, resources, observations = audit.rebuild(folder)
    persist_batch(folder, plan, observations, resources, report)
    result = audit.build_audit(folder)
    assert result["comparison_complete"] is False
    assert "arm_missing:single_agent" in result["incomplete_reasons"]
    # An unfinished comparison is reported as unfinished, never as a pass.
    assert result["audit_checks"]["stored_observations_match_recomputation"] is True


def test_observations_carry_a_bounded_claim_digest_and_no_payload(real_batch):
    plan = real_batch["plan"]
    truths = load_truth("gold-set.v3.1")
    scored = score_arm(
        real_batch["cases"],
        truths,
        real_batch["projections"][MULTI],
        plan=plan.arm[MULTI],
    )
    rows = {(row["case_id"], row["repeat_seed"]): row for row in scored["rows"]}
    key = next(iter(rows))
    projection = next(
        item
        for item in real_batch["projections"][MULTI]
        if (item.case_id, item.repeat_seed) == key
    )
    own_attempts = [
        attempt
        for attempt in real_batch["attempts"][MULTI]
        if attempt["task_key"].startswith(f"{key[0]}.{key[1]}.")
    ]
    observation = observation_from(
        projection,
        arm=MULTI,
        shape=plan.arm[MULTI].shape,
        plan_sha256=plan.plan_sha256,
        attempts=own_attempts,
        scored_row=rows[key],
    )
    body = json.dumps(observation.model_dump(mode="json"))
    assert observation.claim_count == 12
    assert observation.claim_digest_sha256
    assert observation.first_send_count == 2
    assert observation.physical_request_count == 2
    assert observation.known_input_tokens == 200
    # The projection stays bounded and closed: no prompt, response or source text.
    assert "sources" not in body and "content" not in body
    assert len(body) < 2_000
    metering = resources_from_ledger(
        MULTI, plan.history.namespace, real_batch["attempts"][MULTI]
    )
    assert metering.physical_requests == len(real_batch["attempts"][MULTI])
    assert metering.cost_status == "unknown" and metering.cost_value is None
