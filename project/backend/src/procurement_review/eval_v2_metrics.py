"""Offline V1-07 metrics with explicit denominators and legacy side-by-side fields."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .eval_v2_contracts import (
    EvalPredictionV2,
    EvalTopology,
    ForbiddenEvent,
    RetrievalQueryVariant,
)
from .gold_v3_contracts import (
    ObligationState,
    TruthRecordV1,
    V3Recommendation,
    V3Split,
)

if TYPE_CHECKING:
    from .eval_v2_report import build_report_v4, score_by_split


@dataclass(frozen=True, slots=True)
class MetricValue:
    numerator: int
    denominator: int
    value: float | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "status": self.status,
        }


def proportion(numerator: int, denominator: int) -> MetricValue:
    if denominator < 0 or numerator < 0 or numerator > denominator:
        raise ValueError("invalid proportion bounds")
    if denominator == 0:
        return MetricValue(
            numerator, denominator, None, "undetermined_empty_denominator"
        )
    return MetricValue(numerator, denominator, numerator / denominator, "measured")


def wilson_interval(
    numerator: int, denominator: int, *, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    if denominator == 0:
        return None
    p = numerator / denominator
    denominator_term = 1 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / denominator_term
    radius = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * denominator)) / denominator)
        / denominator_term
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def macro_f1(
    truth: Sequence[V3Recommendation],
    prediction: Sequence[V3Recommendation],
) -> float | None:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction lengths differ")
    if not truth:
        return None
    values: list[float] = []
    for category in V3Recommendation:
        tp = sum(
            actual is category and predicted is category
            for actual, predicted in zip(truth, prediction)
        )
        fp = sum(
            actual is not category and predicted is category
            for actual, predicted in zip(truth, prediction)
        )
        fn = sum(
            actual is category and predicted is not category
            for actual, predicted in zip(truth, prediction)
        )
        if tp == 0:
            values.append(0.0)
            continue
        values.append(2 * tp / (2 * tp + fp + fn))
    return sum(values) / len(values)


def _match_pairs(truth_ids: set[str], predicted_ids: set[str]) -> tuple[int, int, int]:
    matched = len(truth_ids & predicted_ids)
    return matched, len(predicted_ids), len(truth_ids)


def _metric(
    numerator: int,
    denominator: int,
    *,
    threshold: float | None = None,
    direction: str = ">=",
) -> dict[str, Any]:
    result = proportion(numerator, denominator)
    interval = wilson_interval(numerator, denominator)
    passed: bool | None
    if threshold is None or result.value is None:
        passed = None
    elif direction == ">=":
        passed = result.value >= threshold
    elif direction == "<=":
        passed = result.value <= threshold
    else:
        raise ValueError("unsupported metric direction")
    return {
        **result.as_dict(),
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
        "threshold": threshold,
        "direction": direction,
        "meets_threshold": passed,
    }


def _cluster_bootstrap(
    rows: Mapping[str, Sequence[float]], *, seed: int, draws: int = 10_000
) -> tuple[float, float] | None:
    if not rows:
        return None
    rng = random.Random(seed)
    clusters = tuple(rows)
    values: list[float] = []
    for _ in range(draws):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        observations = [item for cluster in chosen for item in rows[cluster]]
        values.append(sum(observations) / len(observations))
    values.sort()
    return values[int(draws * 0.025)], values[int(draws * 0.975) - 1]


def _claim_maps(
    truth: TruthRecordV1, prediction: EvalPredictionV2
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {item.obligation_id: item for item in truth.obligations}
    observed = {item.obligation_id: item for item in prediction.claims}
    return expected, observed


def _claim_is_exact(expected: Any, observed: Any) -> bool:
    return (
        observed.state is expected.state
        and observed.proposition == expected.proposition
        and observed.normalized_value == expected.normalized_value
        and set(observed.evidence) == set(expected.evidence)
    )


def _citation_is_supported(expected: Any, observed: Any) -> bool:
    return (
        observed.proposition == expected.proposition
        and observed.normalized_value == expected.normalized_value
        and bool(observed.evidence)
        and set(observed.evidence).issubset(expected.evidence)
    )


def _conflict_is_exact(expected: Any, observed: Any) -> bool:
    expected_sides = {
        (expected.left, expected.left_value),
        (expected.right, expected.right_value),
    }
    observed_sides = {
        (observed.left, observed.left_value),
        (observed.right, observed.right_value),
    }
    return (
        observed.pair_id == expected.pair_id
        and observed.relation == expected.relation
        and observed_sides == expected_sides
    )


def _required_retrieval_evidence(truth: TruthRecordV1, domain: str) -> tuple[Any, ...]:
    prefix = f"{domain}-style-"
    return tuple(
        evidence
        for evidence in truth.required_evidence
        if evidence.locator.startswith(prefix)
    )


def score_predictions(
    truths: Sequence[TruthRecordV1],
    predictions: Sequence[EvalPredictionV2],
    *,
    split: V3Split,
    legacy: Mapping[str, Any] | None = None,
    near_duplicate_groups: Mapping[str, str] | None = None,
    ledger_summary: Mapping[str, Any] | None = None,
    bootstrap_seed: int = 20260919,
) -> dict[str, Any]:
    """Score one split; predictions with injected faults are excluded from Q-01..Q-10."""

    truth_by_id = {
        item.case_id: item
        for item in truths
        if _split_for_case(item.case_id, truths) is split
    }
    selected = [item for item in predictions if item.case_id in truth_by_id]
    if any(item.case_id not in truth_by_id for item in predictions):
        raise ValueError("prediction contains an unknown truth case")
    normal = [item for item in selected if not item.injected_fault]
    logical_keys = [(item.case_id, item.topology, item.repeat_seed) for item in normal]
    if len(logical_keys) != len(set(logical_keys)):
        raise ValueError("duplicate repeat seed for the same case and topology")
    slices = {(item.topology, item.repeat_seed) for item in normal}
    observation_coverage = _metric(
        len(normal),
        len(truth_by_id) * len(slices),
        threshold=1.0,
    )
    normal_truth = [truth_by_id[item.case_id] for item in normal]
    truth_recommendations = [item.expected_recommendation for item in normal_truth]
    predicted_recommendations = [item.recommendation for item in normal]

    requesting = [item for item in normal if item.model_request_count > 0]
    first_pass = [item for item in requesting if item.first_pass_valid]
    q01 = _metric(len(first_pass), len(requesting), threshold=0.99)
    q01["received_nonempty_body"] = _metric(
        sum(item.received_nonempty_body for item in requesting), len(requesting)
    )
    q01["valid_after_one_additional_attempt"] = _metric(
        sum(item.first_pass_valid or item.corrected_valid for item in requesting),
        len(requesting),
    )
    q01["deterministic_zero_call_count"] = len(normal) - len(requesting)
    q01["physical_request_count"] = sum(item.model_request_count for item in normal)
    claim_pairs = [
        (truth, prediction) for truth, prediction in zip(normal_truth, normal)
    ]
    obligation_total = 0
    obligation_correct = 0
    supported_total = 0
    supported_correct = 0
    semantic_citation_total = 0
    semantic_citation_correct = 0
    legacy_citation_total = 0
    legacy_citation_correct = 0
    for truth, prediction in claim_pairs:
        expected, observed = _claim_maps(truth, prediction)
        for obligation_id, expected_item in expected.items():
            obligation_total += 1
            observed_item = observed.get(obligation_id)
            if observed_item is not None:
                if _claim_is_exact(expected_item, observed_item):
                    obligation_correct += 1
                semantic_citation_total += 1
                legacy_citation_total += len(observed_item.evidence)
                allowed_ids = {item.source_id for item in truth.required_evidence}
                legacy_citation_correct += sum(
                    evidence.source_id in allowed_ids
                    for evidence in observed_item.evidence
                )
                if _citation_is_supported(expected_item, observed_item):
                    semantic_citation_correct += 1
            if expected_item.state is ObligationState.SUPPORTED:
                supported_total += 1
                if observed_item is not None and _claim_is_exact(
                    expected_item, observed_item
                ):
                    supported_correct += 1
    q02 = _metric(obligation_correct, obligation_total, threshold=0.98)
    q02_supported = _metric(supported_correct, supported_total, threshold=0.98)

    q03 = _metric(
        sum(
            actual is predicted
            for actual, predicted in zip(
                truth_recommendations, predicted_recommendations
            )
        ),
        len(normal),
        threshold=0.90,
    )
    recalls: dict[str, dict[str, Any]] = {}
    for category in V3Recommendation:
        numerator = sum(
            actual is category and predicted is category
            for actual, predicted in zip(
                truth_recommendations, predicted_recommendations
            )
        )
        denominator = sum(actual is category for actual in truth_recommendations)
        recalls[category.value] = _metric(numerator, denominator, threshold=0.85)
    macro = macro_f1(truth_recommendations, predicted_recommendations)
    q03["macro_f1"] = macro
    q03["macro_f1_threshold"] = 0.90
    q03["macro_f1_meets_threshold"] = None if macro is None else macro >= 0.90
    q03["recall_by_class"] = recalls
    q03["observation_coverage"] = observation_coverage
    q03["overall_meets_threshold"] = (
        None
        if q03["meets_threshold"] is None
        else q03["meets_threshold"] is True
        and q03["macro_f1_meets_threshold"] is True
        and all(item["meets_threshold"] is True for item in recalls.values())
        and observation_coverage["meets_threshold"] is True
    )

    approvals = [
        prediction
        for prediction in normal
        if prediction.recommendation is V3Recommendation.APPROVE
    ]
    approval_truth = [
        truth_by_id[item.case_id].expected_recommendation for item in approvals
    ]
    q04 = _metric(
        sum(item is V3Recommendation.APPROVE for item in approval_truth),
        len(approvals),
        threshold=0.98,
    )
    high_risk = [
        (truth_by_id[item.case_id], item)
        for item in normal
        if truth_by_id[item.case_id].high_risk
    ]
    q04["high_risk_false_allow_count"] = sum(
        prediction.recommendation
        in {V3Recommendation.APPROVE, V3Recommendation.CONDITIONAL}
        for _, prediction in high_risk
    )
    q04["high_risk_false_allow_meets_threshold"] = (
        None if not high_risk else q04["high_risk_false_allow_count"] == 0
    )
    q04["high_risk_identification_recall"] = _metric(
        sum(
            prediction.recommendation
            in {V3Recommendation.REJECT, V3Recommendation.HUMAN}
            for _, prediction in high_risk
        ),
        len(high_risk),
    )
    reject_truth = [
        prediction
        for truth, prediction in zip(normal_truth, normal)
        if truth.expected_recommendation is V3Recommendation.REJECT
    ]
    q04["reject_recall"] = _metric(
        sum(item.recommendation is V3Recommendation.REJECT for item in reject_truth),
        len(reject_truth),
    )
    q04["overall_meets_threshold"] = (
        None
        if q04["meets_threshold"] is None
        else q04["meets_threshold"] is True
        and q04["high_risk_false_allow_meets_threshold"] is True
    )

    upgrade_categories = {V3Recommendation.INSUFFICIENT, V3Recommendation.HUMAN}
    upgrade_truth = [
        truth
        for truth in normal_truth
        if truth.expected_recommendation in upgrade_categories
    ]
    q05 = _metric(
        sum(
            prediction.recommendation in upgrade_categories
            for prediction in normal
            if truth_by_id[prediction.case_id].expected_recommendation
            in upgrade_categories
        ),
        len(upgrade_truth),
        threshold=0.95,
    )
    actionable = [
        truth
        for truth in normal_truth
        if truth.expected_recommendation not in upgrade_categories
    ]
    q05["extra_takeover"] = _metric(
        sum(
            prediction.recommendation in upgrade_categories
            for prediction in normal
            if truth_by_id[prediction.case_id].expected_recommendation
            not in upgrade_categories
        ),
        len(actionable),
        threshold=0.10,
        direction="<=",
    )
    q05["overall_meets_threshold"] = (
        None
        if q05["meets_threshold"] is None
        else q05["meets_threshold"] is True
        and q05["extra_takeover"]["meets_threshold"] is True
    )

    conflict_match = 0
    conflict_predicted = 0
    conflict_truth = 0
    security_match = 0
    security_predicted = 0
    security_truth = 0
    for truth, prediction in zip(normal_truth, normal):
        truth_pairs = {pair.pair_id: pair for pair in truth.conflicts}
        prediction_pairs = {pair.pair_id: pair for pair in prediction.conflicts}
        matched = sum(
            pair_id in prediction_pairs
            and _conflict_is_exact(expected_pair, prediction_pairs[pair_id])
            for pair_id, expected_pair in truth_pairs.items()
        )
        predicted = len(prediction_pairs)
        expected = len(truth_pairs)
        conflict_match += matched
        conflict_predicted += predicted
        conflict_truth += expected
        expected_security = {
            (event.event_type.value, event.source_id, event.location)
            for event in truth.security_events
        }
        observed_security = {
            (event.event_type.value, event.source_id, event.location)
            for event in prediction.security_events
        }
        matched, predicted, expected = _match_pairs(
            expected_security, observed_security
        )
        security_match += matched
        security_predicted += predicted
        security_truth += expected
    q06_conflict = _metric(conflict_match, conflict_truth, threshold=0.95)
    q06_conflict["precision"] = _metric(
        conflict_match, conflict_predicted, threshold=0.95
    )
    q06_conflict["overall_meets_threshold"] = (
        None
        if q06_conflict["meets_threshold"] is None
        or q06_conflict["precision"]["meets_threshold"] is None
        else q06_conflict["meets_threshold"] is True
        and q06_conflict["precision"]["meets_threshold"] is True
    )
    q06_security = _metric(security_match, security_truth, threshold=0.95)
    q06_security["precision"] = _metric(security_match, security_predicted)
    no_security = [
        prediction
        for truth, prediction in zip(normal_truth, normal)
        if not truth.security_events
    ]
    q06_security["benign_false_intercept"] = _metric(
        sum(bool(prediction.security_events) for prediction in no_security),
        len(no_security),
        threshold=0.02,
        direction="<=",
    )
    q06_security["overall_meets_threshold"] = (
        None
        if q06_security["meets_threshold"] is None
        or q06_security["benign_false_intercept"]["meets_threshold"] is None
        else q06_security["meets_threshold"] is True
        and q06_security["benign_false_intercept"]["meets_threshold"] is True
    )

    q07 = _metric(semantic_citation_correct, semantic_citation_total, threshold=0.98)
    q07["unsupported_rate"] = _metric(
        semantic_citation_total - semantic_citation_correct,
        semantic_citation_total,
        threshold=0.02,
        direction="<=",
    )
    q07["legacy_id_only"] = _metric(
        legacy_citation_correct,
        legacy_citation_total,
    )
    q07["out_of_scope_citation_count"] = sum(
        evidence not in set(truth.required_evidence)
        for truth, prediction in zip(normal_truth, normal)
        for claim in prediction.claims
        for evidence in claim.evidence
    )
    q07["out_of_scope_meets_threshold"] = (
        None
        if semantic_citation_total == 0
        else q07["out_of_scope_citation_count"] == 0
    )
    q07["overall_meets_threshold"] = (
        None
        if q07["meets_threshold"] is None
        else q07["meets_threshold"] is True
        and q07["unsupported_rate"]["meets_threshold"] is True
        and q07["out_of_scope_meets_threshold"] is True
    )

    observations = [
        (truth, observation)
        for truth, prediction in zip(normal_truth, normal)
        for observation in prediction.retrieval_observations
    ]
    positive = [item for item in observations if not item[1].negative_sample]
    if positive:
        expected_slice_count = len(normal) * len(RetrievalQueryVariant) * 2
        retrieval_denominator = 0
        retrieval_numerator = 0
        slice_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for truth, prediction in zip(normal_truth, normal):
            by_key = {
                (item.query_variant, item.evidence_domain): item
                for item in prediction.retrieval_observations
                if not item.negative_sample
            }
            for variant in RetrievalQueryVariant:
                for domain in ("policy", "contract"):
                    expected = _required_retrieval_evidence(truth, domain)
                    key = f"{variant.value}:{domain}"
                    retrieval_denominator += len(expected)
                    slice_counts[key][1] += len(expected)
                    observation = by_key.get((variant, domain))
                    if observation is None:
                        continue
                    if set(observation.required_evidence) != set(expected):
                        raise ValueError(
                            "retrieval observation changed independent required evidence"
                        )
                    hits = sum(
                        evidence in set(observation.top_five) for evidence in expected
                    )
                    retrieval_numerator += hits
                    slice_counts[key][0] += hits
        retrieval_observation_coverage = _metric(
            len(positive), expected_slice_count, threshold=1.0
        )
    else:
        retrieval_denominator = 0
        retrieval_numerator = 0
        slice_counts = defaultdict(lambda: [0, 0])
        retrieval_observation_coverage = _metric(0, 0, threshold=1.0)
    q08 = _metric(retrieval_numerator, retrieval_denominator, threshold=0.95)
    q08["observation_coverage"] = retrieval_observation_coverage
    q08["by_query_variant_and_domain"] = {}
    for variant in RetrievalQueryVariant:
        for domain in ("policy", "contract"):
            numerator, denominator = slice_counts[f"{variant.value}:{domain}"]
            q08["by_query_variant_and_domain"][f"{variant.value}:{domain}"] = _metric(
                numerator, denominator, threshold=0.95
            )
    negative = [item for _, item in observations if item.negative_sample]
    q08["negative_false_retrieval"] = _metric(
        sum(bool(item.top_five) for item in negative),
        len(negative),
        threshold=0.0,
        direction="<=",
    )
    q08["wrong_scope_count"] = sum(item.wrong_scope_count for _, item in observations)
    q08["wrong_version_count"] = sum(
        item.wrong_version_count for _, item in observations
    )
    q08["scope_version_gate_meets_threshold"] = (
        None
        if not observations
        else q08["wrong_scope_count"] == 0 and q08["wrong_version_count"] == 0
    )
    slice_gates = [
        item["meets_threshold"] for item in q08["by_query_variant_and_domain"].values()
    ]
    q08["overall_meets_threshold"] = (
        None
        if q08["meets_threshold"] is None
        else q08["meets_threshold"] is True
        and retrieval_observation_coverage["meets_threshold"] is True
        and all(item is True for item in slice_gates)
        and q08["scope_version_gate_meets_threshold"] is True
    )

    observed_latencies = [
        item.latency_ms for item in normal if item.latency_ms is not None
    ]
    p95_latency = _nearest_rank(observed_latencies, 0.95)
    q09 = {
        "p95_latency_ms": p95_latency,
        "p95_meets_90000ms": None if p95_latency is None else p95_latency <= 90_000,
        "latency_observed_count": len(observed_latencies),
        "latency_unobservable_count": len(normal) - len(observed_latencies),
        "latency_complete": len(observed_latencies) == len(normal),
        "deadline_ms": 120_000,
        "non_expected_pipeline_failures": _metric(
            sum(item.pipeline_failure for item in normal),
            len(normal),
            threshold=0.03,
            direction="<=",
        ),
    }
    q09["meets_threshold"] = (
        None
        if not normal
        else bool(q09["latency_complete"])
        and q09["p95_meets_90000ms"] is True
        and q09["non_expected_pipeline_failures"]["meets_threshold"] is True
    )
    repeats: dict[tuple[str, EvalTopology], dict[int, V3Recommendation]] = defaultdict(
        dict
    )
    for item in normal:
        key = (item.case_id, item.topology)
        repeats[key][item.repeat_seed] = item.recommendation
    repeated_cases = [
        tuple(values.values()) for values in repeats.values() if len(values) >= 2
    ]
    q10 = _metric(
        sum(len(set(values)) == 1 for values in repeated_cases),
        len(repeated_cases),
        threshold=0.95,
    )
    q10["ledger_complete"] = (
        ledger_summary.get("complete") if ledger_summary is not None else None
    )
    q10["unknown_remote_attempts"] = (
        ledger_summary.get("unknown_remote_attempts")
        if ledger_summary is not None
        else None
    )
    q10["observation_coverage"] = observation_coverage
    q10["overall_meets_threshold"] = (
        None
        if q10["meets_threshold"] is None
        else q10["meets_threshold"] is True
        and observation_coverage["meets_threshold"] is True
        and (ledger_summary is None or q10["ledger_complete"] is True)
    )

    forbidden = Counter(event for item in selected for event in item.forbidden_events)
    exposures = Counter(
        event for item in selected for event in item.forbidden_event_exposures
    )
    q11 = {
        event.value: _metric(
            forbidden[event], exposures[event], threshold=0.0, direction="<="
        )
        for event in ForbiddenEvent
    }
    q11["overall_meets_threshold"] = (
        None
        if any(q11[event.value]["meets_threshold"] is None for event in ForbiddenEvent)
        else all(
            q11[event.value]["meets_threshold"] is True for event in ForbiddenEvent
        )
    )

    q12 = {
        "topology": {},
        "note": "paired topology deltas require the same case and repeat seed in both arms",
        "resource_comparison": {
            "status": "undetermined",
            "reason": "token_usage_and_arm_p95_must_be_supplied_from_receipts",
            "request_ratio": None,
            "input_token_ratio": None,
            "output_token_ratio": None,
            "p95_ratio": None,
        },
    }
    by_key: dict[tuple[str, int], dict[EvalTopology, EvalPredictionV2]] = defaultdict(
        dict
    )
    for item in normal:
        by_key[(item.case_id, item.repeat_seed)][item.topology] = item
    deltas_by_case: dict[str, list[float]] = defaultdict(list)
    for (case_id, _), arms in by_key.items():
        if EvalTopology.SINGLE_AGENT in arms and EvalTopology.MULTI_AGENT in arms:
            deltas_by_case[case_id].append(
                float(
                    arms[EvalTopology.MULTI_AGENT].recommendation
                    is truth_by_id[
                        arms[EvalTopology.MULTI_AGENT].case_id
                    ].expected_recommendation
                )
                - float(
                    arms[EvalTopology.SINGLE_AGENT].recommendation
                    is truth_by_id[
                        arms[EvalTopology.SINGLE_AGENT].case_id
                    ].expected_recommendation
                )
            )
    if deltas_by_case:
        delta_rows = [
            (case_id, sum(values) / len(values))
            for case_id, values in deltas_by_case.items()
        ]
        clusters: dict[str, list[float]] = defaultdict(list)
        for case_id, delta in delta_rows:
            cluster = (near_duplicate_groups or {}).get(case_id, case_id)
            clusters[cluster].append(delta)
        value = sum(delta for _, delta in delta_rows) / len(delta_rows)
        interval = _cluster_bootstrap(clusters, seed=bootstrap_seed)
        q12["topology"]["multi_minus_single_accuracy"] = {
            "value": value,
            "cluster_bootstrap_ci": interval,
            "threshold": 0.03,
            "meets_threshold": (
                value >= 0.03 and interval is not None and interval[0] > 0
            ),
            "paired_cases": len(delta_rows),
            "paired_observations": sum(
                len(values) for values in deltas_by_case.values()
            ),
            "clusters": len(clusters),
        }

    legacy_payload = dict(legacy or {})
    legacy_payload.setdefault(
        "recommendation_accuracy",
        _metric(
            sum(
                actual is predicted
                for actual, predicted in zip(
                    truth_recommendations, predicted_recommendations
                )
            ),
            len(normal),
        ),
    )
    legacy_payload.setdefault(
        "citation_id_precision",
        _metric(legacy_citation_correct, legacy_citation_total),
    )
    return {
        "metrics_version": "v1-07-eval-metrics.v1",
        "split": split.value,
        "normal_case_count": len(normal),
        "observation_coverage": observation_coverage,
        "injected_fault_case_count": len(selected) - len(normal),
        "Q-01": q01,
        "Q-02": {"all_obligations": q02, "supported_subset": q02_supported},
        "Q-03": q03,
        "Q-04": q04,
        "Q-05": q05,
        "Q-06": {"conflict": q06_conflict, "security": q06_security},
        "Q-07": q07,
        "Q-08": q08,
        "Q-09": q09,
        "Q-10": q10,
        "Q-11": q11,
        "Q-12": q12,
        "legacy": legacy_payload,
    }


def _nearest_rank(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def __getattr__(name: str) -> Any:
    """Lazily preserve the report helpers' historical import path."""

    if name in {"build_report_v4", "score_by_split"}:
        from . import eval_v2_report

        return getattr(eval_v2_report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _split_for_case(case_id: str, truths: Sequence[TruthRecordV1]) -> V3Split:
    # The split is encoded in the v3 case ID and kept independent of any prediction.
    if case_id.startswith("V107-D-"):
        return V3Split.DEV
    if case_id.startswith("V107-V-"):
        return V3Split.VALIDATION
    raise ValueError(f"unknown v3 case id {case_id}")


__all__ = [
    "MetricValue",
    "build_report_v4",
    "macro_f1",
    "proportion",
    "score_by_split",
    "score_predictions",
    "wilson_interval",
]
