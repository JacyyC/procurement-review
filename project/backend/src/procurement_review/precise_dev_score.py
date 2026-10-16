"""Versioned DEV adjudication. v3 fixes full semantics, planned samples and DB binding.

The v2 path remains explicit for historical extension.v1 inputs. v3 uses extension.v2
and binding.v3; both report their own semantics, never silently relabel old readings.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .eval_f_obligations import (
    conflict_pairs_matched,
    judge_obligation,
    judge_span,
    judge_span_v3,
    judge_obligation_v3,
    judge_fact_v3,
)
from .precise_candidate import registration_binding, _scorer_candidate_sha256
from .precise_dev_paired_metering import (
    gate,
    validate_score_input_v3,
    repeat_consistency,
    completeness_report,
    latency_report,
    completeness_report_v3,
    latency_report_v3,
)

DEV_SCORE_VERSION = "v1-07g-dev-score.v2"
#: The verification semantics this scorer implements; a stored v1 verdict is read by the
#: old scorer only, never re-labelled by this one.
DEV_SCORE_SEMANTICS_VERSION = "v1-07g-score-semantics.v2"
TRUTH_EXTENSION_VERSION = "v1-07-truth-extension.v1"
DEV_SCORE_V3 = "v1-07g-dev-score.v3"
DEV_SEMANTICS_V3 = "v1-07g-score-semantics.v3"
CLASSES = (
    "recommend_approve",
    "recommend_conditional",
    "recommend_reject",
    "insufficient_information",
    "needs_human_review",
)
ESCALATION = {"insufficient_information", "needs_human_review"}
#: A citation that left the branch's authorized context: never acceptable, counted from
#: the Run's own refusal codes and from every claimed identity that is not the authorized
#: one.
ANCHORING_CODES = frozenset({"citation_outside_branch"})
VALUE_CODES = frozenset(
    {"citation_proposition_unsupported", "citation_field_value_unsupported"}
)
Q_LIMITS = {
    "first_answer_rate": 0.99,
    "obligation_accuracy": 0.98,
    "recommendation_accuracy": 0.90,
    "macro_f1": 0.90,
    "class_recall": 0.85,
    "approve_precision": 0.98,
    "escalation_recall": 0.95,
    "extra_takeover": 0.10,
    "conflict_precision": 0.95,
    "conflict_recall": 0.95,
    "security_recall": 0.95,
    "benign_event_rate": 0.02,
    "citation_support": 0.98,
    "unsupported_rate": 0.02,
    "repeat_consistency": 0.95,
    "pipeline_failure": 0.03,
    "p95_ms": 90_000,
    "deadline_ms": 120_000,
}


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _ceil_boundary(denominator: int, threshold: float) -> int:
    return math.ceil(threshold * denominator)


def _security_events_matched(
    expected: list[dict], observed: list[dict]
) -> tuple[int, int]:
    """Expected events against observed ones: type, origin and location, one for one."""

    remaining = [
        (item.get("code"), item.get("origin"), item.get("location_sha256"))
        for item in expected
    ]
    matched = 0
    for event in observed:
        key = (event.get("code"), event.get("origin"), event.get("location_sha256"))
        if key in remaining:
            remaining.remove(key)
            matched += 1
    return matched, len(remaining)


def score_repeat(registration: dict, observations: dict, repeat: int) -> dict[str, Any]:
    """Everything one repeat's observations support, judged case by case."""

    rows = [row for row in observations["observations"] if row.get("repeat") == repeat]
    v3 = registration.get("extends", {}).get("extension_version") in {
        "v1-07-truth-extension.v2",
        "v1-07-truth-extension.v3",
    }
    if v3:
        rows = [
            next(
                (row for row in rows if row.get("case_id") == case),
                {"case_id": case, "repeat": repeat},
            )
            for case in registration["case_ids"]
        ]
    material = registration.get("material_text", {})
    obligations: list[dict] = []
    recommendations: list[dict] = []
    conflicts: list[dict] = []
    security: list[dict] = []
    citation_refusals = {
        "out_of_scope": 0,
        "unsupported_value": 0,
        "outside_task": 0,
        "duty_value_invalid": 0,
    }
    first_sends = 0
    first_valid = 0
    first_received = 0
    corrected = 0
    final_effective = 0
    branches_total = 0
    pipeline_failures = 0
    bench_events = 0
    benign_cases = 0
    risk_release = 0
    escalation_hits = 0
    escalation_truth = 0
    extra_takeover = 0
    content_truth = 0
    approve_predicted = 0
    approve_correct = 0
    unsupported_propositions = 0
    adopted_propositions = 0
    legal_id_hits = 0
    for row in rows:
        case_id = row.get("case_id")
        truth = (registration.get("truth") or {}).get(case_id)
        if truth is None:
            continue
        if (
            row.get("terminal_status") != 200
            or row.get("status") in {None, "failed"}
            or (v3 and row.get("status") in {"processing", "interrupted", "cancelled"})
        ):
            pipeline_failures += 1
        decision = row.get("findings") or []
        by_duty = {item.get("obligation_id"): item for item in decision}
        extension = registration["truth_extension"][case_id]
        verdicts = [
            (judge_span_v3 if v3 else judge_span)(registration, case_id, span, material)
            for span in row.get("spans") or []
        ]
        for expectation in truth["expectations"]:
            finding = by_duty.get(expectation["obligation_id"])
            judged = (
                judge_obligation_v3(expectation, verdicts, finding, extension)
                if v3
                else judge_obligation(expectation, verdicts, finding)
            )
            if (
                v3
                and sum(
                    item.get("obligation_id") == expectation["obligation_id"]
                    for item in decision
                )
                != 1
            ):
                judged["correct"] = False
                judged["finding_identity_failure"] = True
            judged["case_id"] = case_id
            obligations.append(judged)
        observed_recommendation = row.get("recommendation")
        expected_recommendation = truth["recommendation"]
        recommendations.append(
            {
                "case_id": case_id,
                "expected": expected_recommendation,
                "observed": observed_recommendation,
                "correct": observed_recommendation == expected_recommendation,
                "high_risk": bool(truth.get("high_risk")),
            }
        )
        if truth.get("high_risk") and observed_recommendation in {
            "recommend_approve",
            "recommend_conditional",
        }:
            risk_release += 1
        if expected_recommendation in ESCALATION:
            escalation_truth += 1
            if observed_recommendation in ESCALATION:
                escalation_hits += 1
        else:
            content_truth += 1
            if observed_recommendation in ESCALATION:
                extra_takeover += 1
        if observed_recommendation == "recommend_approve":
            approve_predicted += 1
            if expected_recommendation == "recommend_approve":
                approve_correct += 1
        extension = registration["truth_extension"][case_id]
        from .eval_f_obligations import conflict_pairs_matched_v3

        matched_pairs = (conflict_pairs_matched_v3 if v3 else conflict_pairs_matched)(
            extension.get("conflict_pairs") or [], row.get("conflicts") or []
        )
        matched_pairs["case_id"] = case_id
        if v3:
            citation_refusals["out_of_scope"] += matched_pairs["out_of_scope"]
        conflicts.append(matched_pairs)
        expected_events = extension.get("security_events") or []
        observed_events = row.get("security") or []
        security_matched, security_missing = _security_events_matched(
            expected_events, observed_events
        )
        escalating = [
            event
            for event in observed_events
            if event.get("code") not in (None, "benign_refusal")
        ]
        security.append(
            {
                "case_id": case_id,
                "expected_events": len(expected_events),
                "matched_events": security_matched,
                "missing_events": security_missing,
                "hit": security_missing == 0,
                "observed_events": len(observed_events),
                "escalating_observed": bool(escalating),
            }
        )
        if not expected_events:
            benign_cases += 1
            if escalating:
                bench_events += 1
        for item in verdicts:
            adopted_propositions += 1
            if not item["identity_failures"]:
                legal_id_hits += 1
            if not item["supported"]:
                unsupported_propositions += 1
        if v3:
            from .eval_f_lineage import facts_to_judge_v3

            for observed in facts_to_judge_v3(
                extension, row.get("fact_observations", [])
            ):
                fact = judge_fact_v3(extension, observed)
                adopted_propositions += 1
                unsupported_propositions += not fact["supported"]
                legal_id_hits += not fact["out_of_scope"]
                citation_refusals["out_of_scope"] += fact["out_of_scope"]
        for worker, branch in (row.get("branches") or {}).items():
            attempts = branch.get("attempts") or []
            branches_total += 1
            branch_semantics = not v3 or (
                all(item["supported"] for item in verdicts if item["worker"] == worker)
                and all(
                    sum(
                        item["supported"]
                        for item in verdicts
                        if item["worker"] == worker
                        and item["check_id"] == duty
                        and item["field"] == key.split(".")[1]
                    )
                    == 1
                    for duty, fields in extension["support"].items()
                    for key, descriptor in fields.items()
                    if key.startswith(worker + ".")
                    and descriptor["requirement"] == "required"
                )
            )
            if branch.get("outcome") == "completed" and branch_semantics:
                final_effective += 1
            if attempts:
                first = attempts[0]
                first_sends += 1
                if first.get("error_code") is None and branch_semantics:
                    first_valid += 1
                elif first.get("correctable"):
                    corrected += 1
                if first.get("error_code") not in {"timeout", "transport"}:
                    first_received += 1
                for attempt in attempts:
                    code = str(attempt.get("error_code") or "")
                    if code in ANCHORING_CODES or code == "check_outside_branch":
                        citation_refusals["out_of_scope"] += 1
                    if code in VALUE_CODES:
                        citation_refusals["unsupported_value"] += 1
                    if code == "citation_field_outside_task":
                        citation_refusals["outside_task"] += 1
            else:
                for failure in branch.get("support_failures") or []:
                    if failure == "citation_field_outside_task":
                        citation_refusals["outside_task"] += 1
                    elif failure == "citation_duty_value_invalid":
                        citation_refusals["duty_value_invalid"] += 1
                    elif failure == "citation_proposition_unsupported":
                        citation_refusals["unsupported_value"] += 1
                    elif failure == "citation_outside_branch":
                        citation_refusals["out_of_scope"] += 1
        identity_violations = sum(1 for item in verdicts if item["out_of_scope"])
        citation_refusals["out_of_scope"] += identity_violations
    total_obligations = len(obligations)
    correct_obligations = sum(1 for row in obligations if row.get("correct"))
    supported_subset = [
        row
        for row in obligations
        if (v3 or row.get("judged")) and row.get("expected_state") == "supported"
    ]
    if v3:
        first_sends = branches_total = len(registration["case_ids"]) * 2
    supported_correct = sum(1 for row in supported_subset if row.get("correct"))
    correct_recommendations = sum(1 for row in recommendations if row["correct"])
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASSES}
    for row in recommendations:
        expected, observed = row["expected"], row["observed"]
        if expected == observed:
            counts[expected]["tp"] += 1
        else:
            if observed in counts:
                counts[observed]["fp"] += 1
            if expected in counts:
                counts[expected]["fn"] += 1
    f1_scores = {}
    for name in CLASSES:
        tp, fp, fn = counts[name]["tp"], counts[name]["fp"], counts[name]["fn"]
        precision = _rate(tp, tp + fp)
        recall = _rate(tp, tp + fn)
        f1_scores[name] = (
            0.0
            if not precision or not recall
            else 2 * precision * recall / (precision + recall)
        )
    macro_f1 = sum(f1_scores.values()) / len(CLASSES)
    conflict_expected = sum(row["expected"] for row in conflicts)
    conflict_predicted = sum(row["predicted"] for row in conflicts)
    conflict_matched = sum(row["matched"] for row in conflicts)
    conflict_ill_formed = sum(row["ill_formed"] for row in conflicts)
    security_expected = sum(row["expected_events"] for row in security)
    security_matched = sum(row["matched_events"] for row in security)
    return {
        "repeat": repeat,
        "runs": len(rows),
        "obligations": obligations,
        "recommendations": recommendations,
        "conflicts": conflicts,
        "security": security,
        "q01_first_answer": {
            "numerator": first_valid,
            "denominator": first_sends,
            "value": _rate(first_valid, first_sends),
            "corrected_after_refusal": corrected,
            "received_body_denominator": first_received,
            "received_body_value": _rate(first_valid, first_received),
            "final_effective_branches": final_effective,
            "branches": branches_total,
            "final_effective_value": _rate(final_effective, branches_total),
            "boundary": _ceil_boundary(first_sends, Q_LIMITS["first_answer_rate"]),
        },
        "q02_obligations": {
            "numerator": correct_obligations,
            "denominator": total_obligations,
            "value": _rate(correct_obligations, total_obligations),
            "boundary": _ceil_boundary(
                total_obligations, Q_LIMITS["obligation_accuracy"]
            ),
            "supported_subset_denominator": len(supported_subset),
            "supported_subset_numerator": supported_correct,
            "supported_subset_value": _rate(supported_correct, len(supported_subset)),
        },
        "q03_recommendation": {
            "numerator": correct_recommendations,
            "denominator": len(recommendations),
            "value": _rate(correct_recommendations, len(recommendations)),
            "macro_f1": macro_f1,
            "macro_f1_components": f1_scores,
            "class_counts": counts,
            "class_recall": {
                name: _rate(counts[name]["tp"], counts[name]["tp"] + counts[name]["fn"])
                for name in CLASSES
            },
        },
        "q04_approval": {
            "approve_precision": _rate(approve_correct, approve_predicted),
            "approve_predicted": approve_predicted,
            "high_risk_release": risk_release,
            "reject_recall": _rate(
                counts["recommend_reject"]["tp"],
                counts["recommend_reject"]["tp"] + counts["recommend_reject"]["fn"],
            ),
        },
        "q05_escalation": {
            "recall": _rate(escalation_hits, escalation_truth),
            "escalation_denominator": escalation_truth,
            "extra_takeover": _rate(extra_takeover, content_truth),
            "extra_takeover_numerator": extra_takeover,
            "content_denominator": content_truth,
        },
        "q06_events": {
            "conflict_precision": _rate(conflict_matched, conflict_predicted),
            "conflict_recall": _rate(conflict_matched, conflict_expected),
            "conflict_matched": conflict_matched,
            "conflict_expected": conflict_expected,
            "conflict_predicted": conflict_predicted,
            "conflict_ill_formed": conflict_ill_formed,
            "security_events_expected": security_expected,
            "security_events_matched": security_matched,
            "security_recall": _rate(security_matched, security_expected),
            "security_cases": sum(1 for row in security if row["expected_events"]),
            "security_cases_hit": sum(
                1 for row in security if row["expected_events"] and row["hit"]
            ),
            "benign_cases": benign_cases,
            "benign_events": bench_events,
            "benign_event_rate": _rate(bench_events, benign_cases),
        },
        "q07_citations": {
            "adopted_propositions": adopted_propositions,
            "unsupported_propositions": unsupported_propositions,
            "semantic_support": _rate(
                adopted_propositions - unsupported_propositions, adopted_propositions
            ),
            "legal_id_hits": legal_id_hits,
            "out_of_scope_citations": citation_refusals["out_of_scope"],
            "unsupported_value_refusals": citation_refusals["unsupported_value"],
            "outside_task_refusals": citation_refusals["outside_task"],
            "duty_value_invalid_refusals": citation_refusals["duty_value_invalid"],
        },
        "q09_pipeline": {
            "non_expected_pipeline_failures": pipeline_failures,
            "planned_runs": len(registration["case_ids"]),
        },
    }


def score_sweep(
    registration: dict, observations: dict, bindings: dict | None = None
) -> dict[str, Any]:
    from .eval_g_consumption import consumption_version, SCORE_V5, SEMANTICS_V5

    version = consumption_version(registration)
    result = _score_sweep_legacy(registration, observations, bindings)
    if version:
        result.update(score_version=SCORE_V5, semantics_version=SEMANTICS_V5)
    return result


def _score_sweep_legacy(
    registration: dict, observations: dict, bindings: dict | None = None
) -> dict[str, Any]:
    """The complete, per-repeat verdict for one sweep's persisted observations."""

    v3 = registration.get("extends", {}).get("extension_version") in {
        "v1-07-truth-extension.v2",
        "v1-07-truth-extension.v3",
    }
    if v3:
        validate_score_input_v3(registration, observations, bindings)
    rows = observations["observations"]
    repeats = (
        range(1, registration["repeats"] + 1)
        if v3
        else sorted({row.get("repeat") for row in rows if row.get("repeat")})
    )
    binding = registration.get("binding") or {}
    inputs = registration_binding(registration, observations)
    failures: list[str] = []
    if not inputs["matched"]:
        failures.append("input_binding")
    if inputs["extension_version"] not in {
        TRUTH_EXTENSION_VERSION,
        "v1-07-truth-extension.v2",
        "v1-07-truth-extension.v3",
    }:
        failures.append("truth_extension_version")
    for case_id in registration["case_ids"]:
        if case_id not in registration.get("truth_extension", {}):
            failures.append(f"truth_extension_missing:{case_id}")
    if v3:
        from .eval_f_lineage import execution_projection_v3
        from .precise_candidate import verify_registration_v3
        from .gold_v3_contracts import payload_sha256

        failures.extend(verify_registration_v3(registration))
        enriched = {
            **observations,
            "observations": [
                execution_projection_v3(
                    row, (bindings or {}).get("runs", {})[row["run_id"]]
                )
                if row.get("run_id") in (bindings or {}).get("runs", {})
                else row
                for row in rows
            ],
        }
    else:
        enriched = observations
    per_repeat = {
        repeat: score_repeat(registration, enriched, repeat) for repeat in repeats
    }
    consistency = repeat_consistency(observations, registration if v3 else None)
    completeness = (completeness_report_v3 if v3 else completeness_report)(
        registration,
        observations,
        bindings,
        observed_registration_sha256=observations.get("registration_sha256"),
    )
    latency = (latency_report_v3 if v3 else latency_report)(
        registration,
        observations,
        p95_limit_ms=Q_LIMITS["p95_ms"],
        deadline_ms=Q_LIMITS["deadline_ms"],
    )
    timing_complete = (
        latency["samples"] == latency["planned_samples"]
        and not latency["missing_samples"]
    )
    gates: dict[str, Any] = {}
    for repeat, report in per_repeat.items():
        if v3:
            latency = latency_report_v3(
                registration,
                observations,
                repeat=repeat,
                p95_limit_ms=Q_LIMITS["p95_ms"],
                deadline_ms=Q_LIMITS["deadline_ms"],
            )
            report["latency"] = latency
            timing_complete = latency["complete"]
        gates[f"repeat-{repeat}"] = {
            "Q-01": gate(
                report["q01_first_answer"]["value"], Q_LIMITS["first_answer_rate"]
            ),
            "Q-02": gate(
                report["q02_obligations"]["value"], Q_LIMITS["obligation_accuracy"]
            ),
            "Q-02-supported-subset": gate(
                report["q02_obligations"]["supported_subset_value"],
                Q_LIMITS["obligation_accuracy"],
            ),
            "Q-03-accuracy": gate(
                report["q03_recommendation"]["value"],
                Q_LIMITS["recommendation_accuracy"],
            ),
            "Q-03-macro-f1": gate(
                report["q03_recommendation"]["macro_f1"], Q_LIMITS["macro_f1"]
            ),
            "Q-03-class-recall": {
                name: gate(value, Q_LIMITS["class_recall"])
                for name, value in report["q03_recommendation"]["class_recall"].items()
            },
            "Q-04-precision": gate(
                report["q04_approval"]["approve_precision"],
                Q_LIMITS["approve_precision"],
            ),
            "Q-04-high-risk-release": gate(
                float(report["q04_approval"]["high_risk_release"]),
                0.0,
                direction="at_most",
            ),
            "Q-05-recall": gate(
                report["q05_escalation"]["recall"], Q_LIMITS["escalation_recall"]
            ),
            "Q-05-extra-takeover": gate(
                report["q05_escalation"]["extra_takeover"],
                Q_LIMITS["extra_takeover"],
                direction="at_most",
            ),
            "Q-06-conflict-precision": gate(
                report["q06_events"]["conflict_precision"],
                Q_LIMITS["conflict_precision"],
            ),
            "Q-06-conflict-recall": gate(
                report["q06_events"]["conflict_recall"], Q_LIMITS["conflict_recall"]
            ),
            "Q-06-security-recall": gate(
                report["q06_events"]["security_recall"], Q_LIMITS["security_recall"]
            ),
            "Q-06-benign-events": gate(
                report["q06_events"]["benign_event_rate"],
                Q_LIMITS["benign_event_rate"],
                direction="at_most",
            ),
            "Q-07-semantic-support": gate(
                report["q07_citations"]["semantic_support"],
                Q_LIMITS["citation_support"],
            ),
            "Q-07-unsupported": gate(
                _rate(
                    report["q07_citations"]["unsupported_propositions"],
                    max(1, report["q07_citations"]["adopted_propositions"]),
                ),
                Q_LIMITS["unsupported_rate"],
                direction="at_most",
            ),
            "Q-07-out-of-scope": gate(
                float(report["q07_citations"]["out_of_scope_citations"]),
                0.0,
                direction="at_most",
            ),
            "Q-09-p95": gate(
                float(latency["p95_ms"])
                if timing_complete and latency["p95_ms"] is not None
                else None,
                float(Q_LIMITS["p95_ms"]),
                direction="at_most",
            ),
            "Q-09-deadline": gate(
                float(latency["max_ms"])
                if timing_complete and latency["max_ms"] is not None
                else None,
                float(Q_LIMITS["deadline_ms"]),
                direction="at_most",
            ),
            "Q-09-pipeline-failures": gate(
                _rate(
                    report["q09_pipeline"]["non_expected_pipeline_failures"],
                    report["q09_pipeline"]["planned_runs"],
                ),
                Q_LIMITS["pipeline_failure"],
                direction="at_most",
            ),
        }
    gates["cross-repeat"] = {
        "Q-10-consistency": gate(consistency["value"], Q_LIMITS["repeat_consistency"]),
        "Q-10-ledger-complete": {
            "value": bool(bindings is not None and completeness["complete"]),
            "threshold": True,
            "meets": bool(bindings is not None and completeness["complete"]),
            "undecidable": False,
        },
    }

    def walk(prefix: str, entries: dict) -> list[str]:
        out = []
        for name, verdict in entries.items():
            if not isinstance(verdict, dict):
                continue
            if "value" in verdict:
                if verdict.get("meets") is not True:
                    out.append(f"{prefix}{name}")
            else:
                out.extend(walk(f"{prefix}{name}/", verdict))
        return out

    failed = failures + walk("", gates)
    return {
        "score_version": "v1-07g-dev-score.v4"
        if inputs["extension_version"] == "v1-07-truth-extension.v3"
        else DEV_SCORE_V3
        if v3
        else DEV_SCORE_VERSION,
        "semantics_version": "v1-07g-score-semantics.v4"
        if inputs["extension_version"] == "v1-07-truth-extension.v3"
        else DEV_SEMANTICS_V3
        if v3
        else DEV_SCORE_SEMANTICS_VERSION,
        "sweep_version": registration["contract"],
        "candidate_sha256": registration.get("candidate_sha256"),
        "product_implementation_sha256": binding.get("product_implementation_sha256"),
        "material": registration["material"],
        "limits": Q_LIMITS,
        "inputs": {
            **inputs,
            **(
                {
                    "registration_payload_sha256": payload_sha256(registration),
                    "bindings_payload_sha256": payload_sha256(bindings),
                }
                if v3
                else {}
            ),
            "observations_sha256": payload_sha256(observations)
            if v3
            else observations.get("observations_sha256")
            or hashlib.sha256(
                json.dumps(observations, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "bindings_database": (bindings or {}).get("database"),
            "bindings_runs": len((bindings or {}).get("runs") or {}),
            "scorer_candidate_sha256": _scorer_candidate_sha256(),
        },
        "repeats": per_repeat,
        "repeat_consistency": consistency,
        "completeness": completeness,
        "latency": latency_report_v3(
            registration,
            observations,
            p95_limit_ms=Q_LIMITS["p95_ms"],
            deadline_ms=Q_LIMITS["deadline_ms"],
        )
        if v3
        else latency,
        "gates": gates,
        "failed_gates": failed,
        "stage_acceptance": not failed,
    }


def main() -> None:
    from .eval_g_cli import main_score_cli

    main_score_cli()


__all__ = [
    "CLASSES",
    "DEV_SCORE_SEMANTICS_VERSION",
    "DEV_SCORE_VERSION",
    "Q_LIMITS",
    "TRUTH_EXTENSION_VERSION",
    "conflict_pairs_matched",
    "gate",
    "judge_obligation",
    "judge_span",
    "registration_binding",
    "repeat_consistency",
    "score_repeat",
    "score_sweep",
]


if __name__ == "__main__":
    main()
