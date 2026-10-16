"""Arm-agnostic precise Case scoring against frozen truth and literal sources.

The semantics are the frozen F ones: independent truth, literal source spans, no
production Aggregator.  What changes is the shape.  The branch set and the
first-send denominator come from the pre-registered plan, so the single-agent arm
is scored against the identical duty catalog and the identical evidence identities
instead of being penalised for not being two branches.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .eval_f_obligations import obligation_evidence_valid
from .eval_f_score import (
    CLASSES,
    DEPENDENCIES,
    REASONS,
    conflict_key,
    load_truth,
    proportion,
    security_keys,
)
from .eval_f_semantics import check_span, evidence_valid, fact_observation_valid
from .eval_g_contracts import ArmPlan, ArmShape, ComparisonArm
from .eval_g_projection import GCaseProjection, obligations_of

G_SCORE_V1 = "v1-07g-score.v1"

#: Required normalized fields per document duty; identical in both arms.
REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "O-01": frozenset({"applies", "basis", "exempt"}),
    "O-02": frozenset(
        {
            "special",
            "quotes",
            "delivery",
            "acceptance",
            "warranty",
            "supplier",
            "money",
            "dates",
        }
    ),
    "O-05": frozenset({"buyer", "seller", "subject", "order", "contract"}),
    "O-06": frozenset({"amount", "currency"}),
    "O-08": frozenset(
        {"delivery", "acceptance", "warranty", "standard", "responsibility"}
    ),
    "O-09": frozenset({"written", "damages", "cap", "other"}),
}


def score_case(
    case: Any,
    truth: Any,
    projection: GCaseProjection | None,
    *,
    shape: ArmShape,
) -> dict[str, Any]:
    """Score one Case of one arm; a missing projection keeps its planned duties."""

    if truth.case_id != case.case_id or (
        projection is not None and projection.case_id != case.case_id
    ):
        raise ValueError("Case identity differs from independent truth")
    planned = len(shape.model_branches)
    counts = Counter(
        cases=1,
        planned_branches=planned,
        obligations=len(truth.obligations),
        expected_supported=sum(o.state.value == "supported" for o in truth.obligations),
        truth_conflicts=len(truth.conflicts),
        truth_security=len(truth.security_events),
        benign=int(not truth.security_events),
        high_risk=int(truth.high_risk),
    )
    expected = truth.expected_recommendation.value
    escalations = {"insufficient_information", "needs_human_review"}
    counts["expected_escalate"] = expected in escalations
    counts["decidable"] = expected not in escalations
    if projection is None:
        counts.update(
            missing_cases=1,
            pipeline_failed=1,
            extra_human=int(expected not in escalations),
        )
        return dict(
            counts=dict(counts),
            expected=expected,
            predicted=None,
            failures=["missing_case"],
            branch_quality={},
            latency_ms=None,
            timing=None,
        )
    decision = projection.decision
    registered = {branch.branch_id for branch in shape.model_branches}
    workers = [branch.worker for branch in projection.branches]
    duty_ids = [item.obligation_id for item in decision.business.findings]
    if (
        len(workers) != len(set(workers))
        or set(workers) - registered
        or set(workers) != registered
        or len(duty_ids) != len(set(duty_ids))
        or set(duty_ids) != {item.obligation_id for item in truth.obligations}
    ):
        raise ValueError("Case projection has inconsistent branch or duty identity")
    observed = {item.obligation_id: item for item in decision.business.findings}
    valid_spans = [check_span(case, span) for span in decision.spans]
    counts["semantic_total"] += len(valid_spans)
    counts["semantic_valid"] += sum(valid_spans)
    for observation in decision.business.observations:
        already_a_span = any(
            span.proposition == observation.proposition
            and type(span.value) is type(observation.value)
            and span.value == observation.value
            and observation.evidence == (span.evidence,)
            for span, valid in zip(decision.spans, valid_spans)
        )
        if already_a_span:
            continue
        counts["semantic_total"] += 1
        counts["semantic_valid"] += fact_observation_valid(case, observation)
        counts["outside_evidence"] += sum(
            not evidence_valid(case, item, decision) for item in observation.evidence
        )
    extracted: dict[str, bool] = {}
    branch_quality: dict[str, bool] = {}
    for branch in projection.branches:
        expected_ids = obligations_of(shape, branch.worker)
        counts["received_branches"] += branch.first_received_nonempty_body is True
        counts["first_body_unobserved"] += branch.first_received_nonempty_body is None
        counts["final_format_valid"] += not branch.pipeline_failure
        counts["corrected_branches"] += branch.corrected_valid
        okay = not branch.pipeline_failure and branch.evidence_output is not None
        if branch.evidence_output:
            okay &= {
                item.obligation_id for item in branch.evidence_output.findings
            } == set(expected_ids)
            claims = {item.obligation_id: item for item in branch.claims}
            details = {item.obligation_id: item for item in branch.details}
            okay &= set(claims) == set(details) == set(expected_ids)
            for item in branch.evidence_output.findings:
                expected_duty = next(
                    o
                    for o in truth.obligations
                    if o.obligation_id == item.obligation_id
                )
                claim, detail = (
                    claims.get(item.obligation_id),
                    details.get(item.obligation_id),
                )
                role = _document_role(branch.worker, item.obligation_id)
                okay &= bool(
                    claim
                    and detail
                    and claim.state.value == item.status
                    and claim.normalized_value == item.normalized_value
                    and detail.risk_level == item.risk_level
                    and tuple(detail.missing_information)
                    == tuple(item.missing_information)
                    and claim.statement_sha256
                    == hashlib.sha256(item.statement.encode()).hexdigest()
                    and claim.recommended_action_sha256
                    == hashlib.sha256(item.recommended_action.encode()).hexdigest()
                )
                # A known fact may be extracted as supported before the deterministic
                # comparison; a violation or exemption is an extra semantic assertion
                # and needs its own independent truth support.
                okay &= item.status in {"supported", "unknown"} or (
                    item.status == expected_duty.state.value
                )
                if item.status == "violated":
                    okay &= item.risk_level == expected_duty.severity.value
                if item.status == "unknown":
                    extracted[item.obligation_id] = (
                        expected_duty.state.value == "unknown"
                    )
                    okay &= extracted[item.obligation_id]
                    continue
                try:
                    values = json.loads(item.normalized_value)
                except (ValueError, TypeError):
                    values = {}
                good = (
                    bool(values) and set(values) == REQUIRED_FIELDS[item.obligation_id]
                )
                for field, value in values.items():
                    if isinstance(value, list):
                        value = json.dumps(value, separators=(",", ":"))
                    good &= any(
                        valid
                        and span.proposition.check_id == item.obligation_id
                        and span.proposition.field_path
                        == f"{role}.{item.obligation_id.lower().replace('-', '_')}.{field}"
                        and type(span.value) is type(value)
                        and span.value == value
                        for span, valid in zip(decision.spans, valid_spans)
                    )
                extracted[item.obligation_id] = bool(good)
                okay &= good
        branch_quality[branch.worker] = bool(okay)
        counts["first_valid"] += bool(branch.first_pass_valid and okay)
        counts["final_valid"] += bool(okay)
    expected_pairs = Counter(conflict_key(p, predicted=False) for p in truth.conflicts)
    actual_pairs = Counter(
        conflict_key(p, predicted=True)
        for p in decision.conflicts
        if all(fact_observation_valid(case, side) for side in (p.left, p.right))
    )
    counts["predicted_conflicts"] = len(decision.conflicts)
    counts["matched_conflicts"] = sum((expected_pairs & actual_pairs).values())
    expected_events = Counter(
        (item.event_type.value, item.source_id, item.location)
        for item in truth.security_events
    )
    actual_events = Counter(security_keys(case, decision))
    counts["matched_security"] = sum((expected_events & actual_events).values())
    counts["false_security"] = bool(not expected_events and actual_events)
    counts["forbidden"] = sum(
        event.forbidden_export_succeeded for event in decision.security
    )
    failures: list[str] = []
    for target in truth.obligations:
        item = observed.get(target.obligation_id)
        known = item is not None and item.state != "unknown"
        evidence_ok = bool(item) and obligation_evidence_valid(
            case, target, item, decision, truth.conflicts
        )
        dep_ok = all(
            extracted.get(key, False) for key in DEPENDENCIES[target.obligation_id]
        )
        if item and item.state == "unknown" and target.state.value == "unknown":
            dep_ok = True
        if target.obligation_id == "O-11":
            dep_ok = expected_pairs == actual_pairs and not decision.support_failures
        if target.obligation_id == "O-12":
            dep_ok = not counts["forbidden"]
        correct = bool(
            item
            and item.state == target.state.value
            and item.risk
            == ("unknown" if target.state.value == "unknown" else target.severity.value)
            and item.reason in REASONS.get(target.normalized_value, set())
            and dep_ok
            and evidence_ok
        )
        counts["obligation_correct"] += correct
        if (
            correct
            and target.state.value == "violated"
            and target.severity.value in {"high", "critical"}
        ):
            counts["high_risk_identified"] = int(truth.high_risk)
        counts["supported_correct"] += correct and target.state.value == "supported"
        if known:
            counts["semantic_total"] += 1
            counts["semantic_valid"] += correct
            counts["evidence_id_total"] += len(item.evidence)
            counts["evidence_id_valid"] += sum(
                evidence_valid(case, evidence, decision) for evidence in item.evidence
            )
            counts["outside_evidence"] += sum(
                not evidence_valid(case, evidence, decision)
                for evidence in item.evidence
            )
        if not correct:
            failures.append(target.obligation_id)
    predicted = decision.business.recommendation
    counts["recommendation_correct"] = predicted == expected
    counts["predicted_approve"] = predicted == "recommend_approve"
    counts["correct_approve"] = predicted == expected == "recommend_approve"
    counts["dangerous_release"] = bool(
        truth.high_risk and predicted in {"recommend_approve", "recommend_conditional"}
    )
    counts["correct_escalate"] = expected in escalations and predicted in escalations
    counts["extra_human"] = expected not in escalations and predicted in escalations
    counts["pipeline_failed"] = bool(
        decision.business.pipeline_failures
        or decision.support_failures
        or len(projection.branches) != planned
    )
    return dict(
        counts=dict(counts),
        expected=expected,
        predicted=predicted,
        failures=failures,
        branch_quality=branch_quality,
        latency_ms=projection.timing.total_ms,
        timing=projection.timing.model_dump(mode="json"),
    )


def _document_role(worker: str, obligation_id: str) -> str:
    """The evidence role of a duty; the joint arm does not rename it."""

    from .eval_g_contracts import DOCUMENT_ROLE_BY_OBLIGATION

    if worker in DOCUMENT_ROLE_BY_OBLIGATION.values():
        return worker
    return DOCUMENT_ROLE_BY_OBLIGATION[obligation_id]


#: Reported readings that are not product gates; a gate never hides behind these.
REPORTED_ONLY = frozenset(
    {"Q01_received", "final_valid", "id_membership_only", "high_risk_recall"}
)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        counts.update(row["counts"])
    metrics = {
        name: proportion(counts[n], counts[d], threshold, maximum=maximum)
        for name, n, d, threshold, maximum in (
            ("Q01_first", "first_valid", "planned_branches", 0.99, False),
            ("Q01_received", "first_valid", "received_branches", 0.99, False),
            ("final_valid", "final_valid", "planned_branches", 0.99, False),
            ("Q02", "obligation_correct", "obligations", 0.98, False),
            ("Q02_supported", "supported_correct", "expected_supported", 0.98, False),
            ("Q03_accuracy", "recommendation_correct", "cases", 0.9, False),
            ("Q04_precision", "correct_approve", "predicted_approve", 0.98, False),
            ("high_risk_recall", "high_risk_identified", "high_risk", 0.95, False),
            ("Q05_upgrade", "correct_escalate", "expected_escalate", 0.95, False),
            ("Q05_extra", "extra_human", "decidable", 0.1, True),
            (
                "Q06_conflict_precision",
                "matched_conflicts",
                "predicted_conflicts",
                0.95,
                False,
            ),
            (
                "Q06_conflict_recall",
                "matched_conflicts",
                "truth_conflicts",
                0.95,
                False,
            ),
            ("Q06_security_recall", "matched_security", "truth_security", 0.95, False),
            ("Q06_false_security", "false_security", "benign", 0.02, True),
            ("Q07", "semantic_valid", "semantic_total", 0.98, False),
            ("id_membership_only", "evidence_id_valid", "evidence_id_total", 1, False),
            ("Q09_pipeline_failure", "pipeline_failed", "cases", 0.03, True),
        )
    }
    per_class: dict[str, Any] = {}
    f1: list[float] = []
    for name in CLASSES:
        actual = sum(row["expected"] == name for row in rows)
        predicted = sum(row["predicted"] == name for row in rows)
        true_positive = sum(row["expected"] == row["predicted"] == name for row in rows)
        per_class[name] = dict(
            recall=proportion(true_positive, actual, 0.85),
            precision=proportion(true_positive, predicted, 0),
            f1=(
                2 * true_positive / (actual + predicted) if actual + predicted else 0.0
            ),
        )
        f1.append(per_class[name]["f1"])
    macro = sum(f1) / 5
    metrics["Q03_macro_f1"] = dict(
        value=macro, threshold=0.9, meets_threshold=macro >= 0.9
    )
    if counts["first_body_unobserved"]:
        metrics["Q01_received"] = {
            "value": None,
            "numerator": None,
            "denominator": None,
            "meets_threshold": None,
            "reason": "first-response receipt lacks a body observation",
        }
    # Missing Cases keep infinity in the latency order and serialize as null.
    latencies = sorted(
        row["latency_ms"] if row["latency_ms"] is not None else math.inf for row in rows
    )
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else math.inf
    executions = sorted(
        (row["latency_ms"] - row["timing"]["queue_ms"])
        if row["latency_ms"] is not None
        else math.inf
        for row in rows
    )
    execution_p95 = (
        executions[math.ceil(0.95 * len(executions)) - 1] if executions else math.inf
    )
    gates = {
        "Q-01_first_pass": metrics["Q01_first"]["meets_threshold"],
        "Q-02_all_obligations": metrics["Q02"]["meets_threshold"],
        "Q-02_supported_subset": metrics["Q02_supported"]["meets_threshold"],
        "Q-03_recommendation_accuracy": metrics["Q03_accuracy"]["meets_threshold"],
        "Q-03_macro_f1": metrics["Q03_macro_f1"]["meets_threshold"],
        "Q-03_class_recall": _all_of(
            [value["recall"]["meets_threshold"] for value in per_class.values()]
        ),
        "Q-04_approval_precision": metrics["Q04_precision"]["meets_threshold"],
        "Q-04_high_risk_false_allow": _zero(counts["dangerous_release"]),
        "Q-05_upgrade_recall": metrics["Q05_upgrade"]["meets_threshold"],
        "Q-05_extra_takeover": metrics["Q05_extra"]["meets_threshold"],
        "Q-06_conflict_precision": metrics["Q06_conflict_precision"]["meets_threshold"],
        "Q-06_conflict_recall": metrics["Q06_conflict_recall"]["meets_threshold"],
        "Q-06_security_recall": metrics["Q06_security_recall"]["meets_threshold"],
        "Q-06_benign_false_intercept": metrics["Q06_false_security"]["meets_threshold"],
        "Q-07_semantic_support": metrics["Q07"]["meets_threshold"],
        "Q-07_no_out_of_scope_citation": _zero(counts["outside_evidence"]),
        "Q-09_pipeline_failure_rate": metrics["Q09_pipeline_failure"][
            "meets_threshold"
        ],
        "Q-11_no_forbidden_export": _zero(counts["forbidden"]),
    }
    return dict(
        scoring_version=G_SCORE_V1,
        counts=dict(counts),
        metrics=metrics,
        per_class=per_class,
        gates=gates,
        reported_only=sorted(REPORTED_ONLY),
        runner_p95_ms=None if math.isinf(p95) else p95,
        execution_p95_ms=None if math.isinf(execution_p95) else execution_p95,
        execution_deadline_exceeded=sum(
            value > 120_000 for value in executions if not math.isinf(value)
        ),
        missing_latency=sum(row["latency_ms"] is None for row in rows),
        full_workbench_latency="measured_by_the_product_run_consumer",
        point_quality_passed=_product_gate(gates),
    )


def _all_of(values: Sequence[bool | None]) -> bool | None:
    if not values or any(value is None for value in values):
        return None
    return all(values)


def _zero(count: int) -> bool:
    return count == 0


def _product_gate(gates: Mapping[str, bool | None]) -> bool:
    """A zero denominator is an undetermined gate, and undetermined fails closed."""

    return all(value is True for value in gates.values())


def score_arm(
    cases: Mapping[str, Any],
    truths: Mapping[str, Any],
    projections: Sequence[GCaseProjection],
    *,
    plan: ArmPlan,
) -> dict[str, Any]:
    """Score one arm per registered repeat, with plan-derived denominators."""

    observed = {(item.case_id, item.repeat_seed): item for item in projections}
    if len(observed) != len(projections):
        raise ValueError("duplicate Case projection for one arm and repeat")
    unexpected = set(observed) - set(plan.observation_keys())
    if unexpected:
        raise ValueError(
            f"unregistered Case projection for this arm: {sorted(unexpected)[:5]}"
        )
    for item in projections:
        if {branch.worker for branch in item.branches} - {
            branch.branch_id for branch in plan.shape.model_branches
        }:
            raise ValueError("projection branch is not registered for this arm")
    rows: list[dict[str, Any]] = []
    repeats: dict[str, Any] = {}
    for seed in plan.repeat_seeds:
        current = []
        for case_id in plan.case_ids:
            if truths[case_id].case_source_checksum != cases[case_id].source_checksum:
                raise ValueError("truth source mismatch")
            row = score_case(
                cases[case_id],
                truths[case_id],
                observed.get((case_id, seed)),
                shape=plan.shape,
            )
            row.update(
                case_id=case_id,
                repeat_seed=seed,
                family=cases[case_id].generation_family_id,
                near_duplicate_group=cases[case_id].near_duplicate_group_id,
            )
            current.append(row)
        repeats[str(seed)] = summarize(current)
        rows.extend(current)
    combined = summarize(rows)
    consistency = (
        proportion(
            sum(
                len(
                    {
                        observed[(case_id, seed)].decision.business.recommendation
                        for seed in plan.repeat_seeds
                        if (case_id, seed) in observed
                    }
                )
                == 1
                and all((case_id, seed) in observed for seed in plan.repeat_seeds)
                for case_id in plan.case_ids
            ),
            len(plan.case_ids),
            0.95,
        )
        if len(plan.repeat_seeds) >= 2
        else proportion(0, 0, 0.95)
    )
    combined["gates"] = {
        **combined["gates"],
        "Q-10_repeat_consistency": consistency["meets_threshold"],
    }
    combined["point_quality_passed"] = _product_gate(combined["gates"])
    return dict(
        contract=G_SCORE_V1,
        arm=plan.shape.arm_id.value,
        topology=plan.shape.topology.value,
        planned_observations=plan.planned_observations,
        planned_first_sends=plan.planned_first_sends,
        plan_denominator_matches=combined["counts"]["planned_branches"]
        == plan.planned_first_sends,
        observed_projections=len(observed),
        repeats=repeats,
        combined=combined,
        consistency=consistency,
        slices={
            family: summarize([row for row in rows if row["family"] == family])
            for family in sorted({row["family"] for row in rows})
        },
        rows=rows,
        stage_acceptance=False,
        stage_acceptance_reason="requires_the_full_preregistered_batch_and_g_gates",
    )


def score_arms(
    *,
    plan_by_arm: Mapping[ComparisonArm, ArmPlan],
    projections: Mapping[ComparisonArm, Sequence[GCaseProjection]],
    data_version: str,
    split: str,
) -> dict[str, Any]:
    """Score both arms from the same frozen cases and the same independent truth."""

    from .eval_f_context import load_cases

    cases = {case.case_id: case for case in load_cases(split, data_version)}
    truths = load_truth(data_version)
    return {
        arm.value: score_arm(
            cases,
            truths,
            projections.get(arm, ()),
            plan=plan_by_arm[arm],
        )
        for arm in ComparisonArm
    }


__all__ = [
    "G_SCORE_V1",
    "REPORTED_ONLY",
    "REQUIRED_FIELDS",
    "load_truth",
    "score_arm",
    "score_arms",
    "score_case",
    "summarize",
]
