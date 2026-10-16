"""Neutral Q-12: fair-comparison completeness, scored apart from product quality.

Every denominator comes from the frozen plan, so a missing arm keeps its full
planned denominator instead of shrinking it.  Winning, tying and losing are all
settled outcomes; none of them is a gate.  Unknown cost stays unknown, and a
missing known token or timing figure makes the comparison incomplete rather than
being estimated away.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .eval_g_contracts import (
    COMPARISON_METRICS_V2,
    ArmResources,
    ClaimCollectionStatus,
    ComparisonArm,
    ComparisonObservationV1,
    ComparisonPlanV1,
    TerminalState,
)
from .eval_v2_metrics import _cluster_bootstrap, _metric
from .gold_v3_contracts import V3Recommendation

#: Reasons carry a bounded sample so a large batch stays readable.
_MAX_REPORTED_KEYS = 12
ObservationKey = tuple[str, int]
Registered = Mapping[ComparisonArm, Mapping[ObservationKey, ComparisonObservationV1]]
RECOMMENDATION_CLASSES = tuple(item.value for item in V3Recommendation)


def _bounded(keys: Sequence[Any]) -> dict[str, Any]:
    sample = [list(key) if isinstance(key, tuple) else key for key in keys]
    return {
        "count": len(keys),
        "sample": sample[:_MAX_REPORTED_KEYS],
        "truncated": len(keys) > _MAX_REPORTED_KEYS,
    }


def _outcome(value: float | None) -> str | None:
    """Classify the paired difference; every branch is a settled, valid result."""

    if value is None:
        return None
    if value > 0:
        return "multi_agent_ahead"
    if value < 0:
        return "single_agent_ahead"
    return "tie"


def _ratio(left: int | float | None, right: int | float | None) -> float | None:
    if left is None or right is None or not right:
        return None
    return left / right


def coverage_report(
    plan: ComparisonPlanV1,
    observations: Sequence[ComparisonObservationV1],
) -> tuple[dict[str, Any], list[str], dict[ComparisonArm, dict[ObservationKey, Any]]]:
    """Compare terminal observations against the plan, never against themselves."""

    reasons: set[str] = set()
    registered: dict[ComparisonArm, dict[ObservationKey, ComparisonObservationV1]] = {
        arm: {} for arm in ComparisonArm
    }
    duplicates: list[tuple[str, str, int]] = []
    unregistered: list[tuple[str, str, int]] = []
    digest_mismatches: list[tuple[str, str, int]] = []
    for item in observations:
        label = (item.arm.value, item.case_id, item.repeat_seed)
        if item.plan_sha256 != plan.plan_sha256:
            digest_mismatches.append(label)
            continue
        planned = plan.arm.get(item.arm)
        if planned is None or item.key not in set(planned.observation_keys()):
            unregistered.append(label)
            continue
        if item.key in registered[item.arm]:
            duplicates.append(label)
            continue
        registered[item.arm][item.key] = item
    if digest_mismatches:
        reasons.add("plan_digest_mismatch")
    if unregistered:
        reasons.add("unregistered_observation")
    if duplicates:
        reasons.add("duplicate_observation")

    per_arm: dict[str, Any] = {}
    missing: list[tuple[str, str, int]] = []
    short_repeats: list[tuple[str, str]] = []
    for arm, planned in sorted(plan.arm.items(), key=lambda entry: entry[0].value):
        observed = registered[arm]
        absent = sorted(set(planned.observation_keys()) - set(observed))
        missing.extend((arm.value, case_id, seed) for case_id, seed in absent)
        if not observed:
            reasons.add(f"arm_missing:{arm.value}")
        elif absent:
            reasons.add(f"observation_missing:{arm.value}")
        counts: dict[str, int] = defaultdict(int)
        for case_id, _ in observed:
            counts[case_id] += 1
        for case_id in sorted(set(planned.case_ids)):
            if counts[case_id] < len(planned.repeat_seeds):
                short_repeats.append((arm.value, case_id))
        per_arm[arm.value] = {
            "planned_observations": planned.planned_observations,
            "terminal_observations": len(observed),
            "missing_observations": _bounded(absent),
            "planned_first_sends": planned.planned_first_sends,
            "issued_first_sends": sum(
                item.first_send_count for item in observed.values()
            ),
            "first_pass_valid": sum(
                item.first_pass_valid_count for item in observed.values()
            ),
            "valid_after_one_additional_attempt": sum(
                item.first_pass_valid_count + item.corrected_valid_count
                for item in observed.values()
            ),
            "physical_requests": sum(
                item.physical_request_count for item in observed.values()
            ),
            "physical_capacity": planned.max_physical_requests,
            "coverage": _metric(
                len(observed), planned.planned_observations, threshold=1.0
            ),
        }
    if short_repeats:
        reasons.add("repeat_missing")
    total_observed = sum(len(items) for items in registered.values())
    return (
        {
            "planned_observations": plan.planned_observations(),
            "terminal_observations": total_observed,
            "coverage": _metric(
                total_observed, plan.planned_observations(), threshold=1.0
            ),
            "by_arm": per_arm,
            "missing_observations": _bounded(missing),
            "cases_with_missing_repeats": _bounded(short_repeats),
            "unregistered_observations": _bounded(unregistered),
            "duplicate_observations": _bounded(duplicates),
            "plan_digest_mismatches": _bounded(digest_mismatches),
        },
        sorted(reasons),
        registered,
    )


def record_completeness(
    plan: ComparisonPlanV1, registered: Registered
) -> tuple[dict[str, Any], list[str]]:
    """Require settled projections, claims, known tokens and known timing."""

    reasons: set[str] = set()
    buckets: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    unknown_usage: list[tuple[str, str, int]] = []
    for arm, observed in sorted(registered.items(), key=lambda entry: entry[0].value):
        for key, item in sorted(observed.items()):
            label = (arm.value, key[0], key[1])
            if item.terminal_state is not TerminalState.COMMITTED:
                buckets["unsettled_observations"].append(label)
                continue
            if item.decision_sha256 is None:
                buckets["projection_missing"].append(label)
            if item.claim_collection_status is not ClaimCollectionStatus.COMPLETE:
                buckets["claim_collection_incomplete"].append(label)
            if item.known_input_tokens is None:
                buckets["known_token_usage_missing"].append(label)
            if item.usage_unknown_attempts:
                unknown_usage.append(label)
            if item.total_latency_ms is None:
                buckets["known_timing_missing"].append(label)
    pairs_without_decision: list[ObservationKey] = []
    for key in sorted(plan.arm[ComparisonArm.MULTI_AGENT].observation_keys()):
        settled = all(
            (item := registered.get(arm, {}).get(key)) is not None
            and item.terminal_state is TerminalState.COMMITTED
            and item.recommendation is not None
            for arm in ComparisonArm
        )
        if not settled:
            pairs_without_decision.append(key)
    for name, labels in buckets.items():
        if labels:
            reasons.add(name)
    if unknown_usage:
        # A declared unknown is honest, but it leaves the token comparison unverified.
        reasons.add("resource_usage_unknown")
    if pairs_without_decision:
        reasons.add("unsettled_pair")
    shortfall = {
        arm.value: plan.arm[arm].planned_first_sends
        - sum(item.first_send_count for item in registered.get(arm, {}).values())
        for arm in plan.arm
    }
    if any(shortfall.values()) and not pairs_without_decision:
        reasons.add("first_send_shortfall")
    return (
        {
            **{name: _bounded(labels) for name, labels in buckets.items()},
            "unsettled_pairs": _bounded(pairs_without_decision),
            "declared_unknown_usage": _bounded(unknown_usage),
            "first_send_shortfall": shortfall,
        },
        sorted(reasons),
    )


def fairness_report(
    plan: ComparisonPlanV1,
    observed_fairness: Mapping[str, Any] | None,
    resources: Mapping[ComparisonArm, ArmResources],
) -> tuple[dict[str, Any], list[str]]:
    """Compare what actually ran against the frozen fairness contract."""

    reasons: set[str] = set()
    declared = plan.fairness.model_dump(mode="json")
    mismatches: dict[str, dict[str, Any]] = {}
    for field, value in (observed_fairness or {}).items():
        if field not in declared or declared[field] != value:
            mismatches[field] = {"registered": declared.get(field), "observed": value}
    if mismatches:
        reasons.add("unfair_input")
    namespace_problems = sorted(
        arm.value
        for arm, item in resources.items()
        if item.namespace != plan.history.namespace
    )
    if namespace_problems:
        reasons.add("metering_namespace_mismatch")
    capacity_problems = sorted(
        arm.value
        for arm, item in resources.items()
        if arm in plan.arm
        and item.physical_requests > plan.arm[arm].max_physical_requests
    )
    if capacity_problems:
        reasons.add("physical_capacity_exceeded")
    absent = sorted(arm.value for arm in plan.arm if arm not in resources)
    if absent:
        reasons.add("arm_resources_missing")
    return (
        {
            "registered": declared,
            "observed": dict(observed_fairness or {}),
            "mismatches": mismatches,
            "metering_namespace": plan.history.namespace,
            "arm_namespaces": {
                arm.value: item.namespace for arm, item in resources.items()
            },
            "namespace_mismatch_arms": namespace_problems,
            "capacity_exceeded_arms": capacity_problems,
            "missing_resource_arms": absent,
            "prior_scopes_kept_separate": [
                item.namespace for item in plan.history.prior_ledgers
            ],
        },
        sorted(reasons),
    )


def resource_comparison(
    resources: Mapping[ComparisonArm, ArmResources],
) -> dict[str, Any]:
    """Report the real price of the topology difference without gating on it."""

    multi = resources.get(ComparisonArm.MULTI_AGENT)
    single = resources.get(ComparisonArm.SINGLE_AGENT)
    unknown_usage = sum(item.usage_unknown_attempts for item in resources.values())
    unreturned = sum(item.unreturned_attempts for item in resources.values())
    cost_known = all(
        item is not None and item.cost_status == "known" for item in (multi, single)
    )
    return {
        "status": "measured" if multi is not None and single is not None else "missing",
        "multi_agent": multi.model_dump(mode="json") if multi else None,
        "single_agent": single.model_dump(mode="json") if single else None,
        "request_ratio": _ratio(
            multi.physical_requests if multi else None,
            single.physical_requests if single else None,
        ),
        "input_token_ratio": _ratio(
            multi.known_input_tokens if multi else None,
            single.known_input_tokens if single else None,
        ),
        "output_token_ratio": _ratio(
            multi.known_output_tokens if multi else None,
            single.known_output_tokens if single else None,
        ),
        "p95_latency_ratio": _ratio(
            multi.p95_total_latency_ms if multi else None,
            single.p95_total_latency_ms if single else None,
        ),
        "cost_status": "known" if cost_known else "unknown",
        "cost_ratio": _ratio(
            multi.cost_value if multi and multi.cost_status == "known" else None,
            single.cost_value if single and single.cost_status == "known" else None,
        ),
        "declared_unknown_usage_attempts": unknown_usage,
        "unreturned_attempts": unreturned,
        "token_totals_are_lower_bound": bool(unknown_usage or unreturned),
        "no_direction_or_cost_gate": True,
    }


def paired_delta(
    plan: ComparisonPlanV1,
    registered: Registered,
    expected: Mapping[str, V3Recommendation | str],
    near_duplicate_groups: Mapping[str, str] | None,
    *,
    bootstrap_seed: int = 20260919,
) -> dict[str, Any]:
    """Multi minus single accuracy, averaged within Case then clustered by group."""

    multi = registered.get(ComparisonArm.MULTI_AGENT, {})
    single = registered.get(ComparisonArm.SINGLE_AGENT, {})
    per_case: dict[str, list[float]] = defaultdict(list)
    per_class: dict[str, dict[str, int]] = {
        name: {"multi_agent_ahead": 0, "tie": 0, "single_agent_ahead": 0}
        for name in RECOMMENDATION_CLASSES
    }
    unsettled = 0
    for key in sorted(plan.arm[ComparisonArm.MULTI_AGENT].observation_keys()):
        left, right = multi.get(key), single.get(key)
        committed = (
            left is not None
            and right is not None
            and left.terminal_state is TerminalState.COMMITTED
            and right.terminal_state is TerminalState.COMMITTED
            and left.recommendation is not None
            and right.recommendation is not None
        )
        if not committed:
            unsettled += 1
            continue
        truth = expected.get(key[0])
        if truth is None:
            raise ValueError("comparison requires the expected recommendation per case")
        wanted = truth.value if isinstance(truth, V3Recommendation) else str(truth)
        left_hit = float(left.recommendation.value == wanted)
        right_hit = float(right.recommendation.value == wanted)
        per_case[key[0]].append(left_hit - right_hit)
        counts = per_class[wanted]
        if left_hit > right_hit:
            counts["multi_agent_ahead"] += 1
        elif left_hit < right_hit:
            counts["single_agent_ahead"] += 1
        else:
            counts["tie"] += 1
    if not per_case:
        return {
            "multi_minus_single_accuracy": {
                "value": None,
                "cluster_bootstrap_ci": None,
                "paired_cases": 0,
                "paired_observations": 0,
                "clusters": 0,
                "unsettled_pairs": unsettled,
            },
            "outcome": None,
            "by_expected_class": per_class,
            "direction_gate": None,
            "note": "no settled pair; the difference is undetermined, not zero",
        }
    rows = [(case_id, sum(v) / len(v)) for case_id, v in sorted(per_case.items())]
    clusters: dict[str, list[float]] = defaultdict(list)
    for case_id, delta in rows:
        clusters[(near_duplicate_groups or {}).get(case_id, case_id)].append(delta)
    value = sum(delta for _, delta in rows) / len(rows)
    interval = _cluster_bootstrap(clusters, seed=bootstrap_seed)
    return {
        "multi_minus_single_accuracy": {
            "value": value,
            "cluster_bootstrap_ci": list(interval) if interval else None,
            "paired_cases": len(rows),
            "paired_observations": sum(len(v) for v in per_case.values()),
            "clusters": len(clusters),
            "unsettled_pairs": unsettled,
            "bootstrap_seed": bootstrap_seed,
            "cluster_field": "near_duplicate_group_id",
        },
        "outcome": _outcome(value),
        "by_expected_class": per_class,
        "direction_gate": None,
        "note": "win, tie and loss are all settled outcomes; none of them gates Q-12",
    }


def ledger_reconciliation(
    registered: Registered, resources: Mapping[ComparisonArm, ArmResources]
) -> tuple[dict[str, Any], list[str]]:
    """The projections and the durable ledger must tell the same story."""

    reasons: set[str] = set()
    per_arm: dict[str, Any] = {}
    for arm, item in sorted(resources.items(), key=lambda entry: entry[0].value):
        observed = registered.get(arm, {})
        actual = {
            "physical_requests": sum(
                entry.physical_request_count for entry in observed.values()
            ),
            "known_input_tokens": sum(
                entry.known_input_tokens or 0 for entry in observed.values()
            ),
            "known_output_tokens": sum(
                entry.known_output_tokens or 0 for entry in observed.values()
            ),
            "usage_unknown_attempts": sum(
                entry.usage_unknown_attempts for entry in observed.values()
            ),
        }
        recorded = {
            "physical_requests": item.physical_requests,
            "known_input_tokens": item.known_input_tokens,
            "known_output_tokens": item.known_output_tokens,
            "usage_unknown_attempts": item.usage_unknown_attempts,
        }
        differences = {
            field: {"observed": actual[field], "ledger": recorded[field]}
            for field in actual
            if actual[field] != recorded[field]
        }
        per_arm[arm.value] = {
            "observed": actual,
            "ledger": recorded,
            "differences": differences,
        }
        if differences:
            reasons.add(f"ledger_mismatch:{arm.value}")
    return {"by_arm": per_arm}, sorted(reasons)


def product_quality(
    arm_quality: Mapping[ComparisonArm, Mapping[str, Any]],
) -> dict[str, Any]:
    """Judge the absolute product gates apart from the comparison's fairness.

    A zero denominator leaves a gate undetermined, and an undetermined product
    gate fails closed rather than being reported as a pass.
    """

    result: dict[str, Any] = {}
    for arm, summary in sorted(arm_quality.items(), key=lambda entry: entry[0].value):
        combined = summary.get("combined") or summary
        gates = dict(combined.get("gates") or summary.get("gates") or {})
        undecided = sorted(name for name, value in gates.items() if value is None)
        failed = sorted(name for name, value in gates.items() if value is False)
        result[arm.value] = {
            "scoring_version": summary.get("scoring_version")
            or combined.get("scoring_version"),
            "counts": dict(combined.get("counts") or {}),
            "per_class": dict(combined.get("per_class") or {}),
            "runner_p95_ms": combined.get("runner_p95_ms"),
            "execution_p95_ms": combined.get("execution_p95_ms"),
            "consistency": summary.get("consistency"),
            "plan_denominator_matches": summary.get("plan_denominator_matches"),
            "gates": gates,
            "failed_gates": failed,
            "undetermined_gates": undecided,
            "meets_product_gates": (
                None if not gates else not failed and not undecided
            ),
        }
    return result


def score_comparison(
    plan: ComparisonPlanV1,
    observations: Sequence[ComparisonObservationV1],
    *,
    expected: Mapping[str, V3Recommendation | str],
    resources: Mapping[ComparisonArm, ArmResources],
    arm_quality: Mapping[ComparisonArm, Mapping[str, Any]] | None = None,
    observed_fairness: Mapping[str, Any] | None = None,
    near_duplicate_groups: Mapping[str, str] | None = None,
    bootstrap_seed: int = 20260919,
) -> dict[str, Any]:
    """Score one pre-registered two-arm comparison under the neutral Q-12 contract."""

    coverage, coverage_reasons, registered = coverage_report(plan, observations)
    records, record_reasons = record_completeness(plan, registered)
    fairness, fairness_reasons = fairness_report(plan, observed_fairness, resources)
    ledger, ledger_reasons = ledger_reconciliation(registered, resources)
    delta = paired_delta(
        plan, registered, expected, near_duplicate_groups, bootstrap_seed=bootstrap_seed
    )
    reasons = sorted(
        set([*coverage_reasons, *record_reasons, *fairness_reasons, *ledger_reasons])
    )
    first_sends: dict[str, Any] = {}
    for arm in plan.arm:
        reported = coverage["by_arm"][arm.value]
        planned = plan.arm[arm].planned_first_sends
        first_sends[arm.value] = {
            **_metric(reported["first_pass_valid"], planned, threshold=0.99),
            "valid_after_one_additional_attempt": _metric(
                reported["valid_after_one_additional_attempt"], planned, threshold=0.99
            ),
            "planned_first_sends": planned,
            "deterministic_zero_call_branches": list(
                plan.arm[arm].shape.deterministic_branches
            ),
        }
    return {
        "metrics_version": COMPARISON_METRICS_V2,
        "comparison_contract": "v1-07g-comparison.v1",
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "stage_id": plan.stage_id,
        "split": plan.split,
        "product_quality": product_quality(arm_quality or {}),
        "Q-12": {
            "coverage": coverage,
            "records": records,
            "fairness": fairness,
            "ledger_reconciliation": ledger,
            "resource_comparison": resource_comparison(resources),
            "first_send_validity_by_arm": first_sends,
            "paired_delta": delta,
            "complete": not reasons,
            "incomplete_reasons": reasons,
            "meets_threshold": not reasons,
            "direction_gate": None,
            "history_namespaces": {
                "current": plan.history.namespace,
                "prior": [item.namespace for item in plan.history.prior_ledgers],
            },
        },
    }


__all__ = [
    "coverage_report",
    "fairness_report",
    "ledger_reconciliation",
    "paired_delta",
    "product_quality",
    "record_completeness",
    "resource_comparison",
    "score_comparison",
]
