from __future__ import annotations

import pytest

from procurement_review.eval_v2_contracts import EvalTopology, RetrievalObservation
from procurement_review.eval_v2_metrics import (
    build_report_v4,
    macro_f1,
    proportion,
    score_predictions,
    wilson_interval,
)
from procurement_review.gold_v3_contracts import (
    ObligationState,
    V3Recommendation,
    V3Split,
)
from v1_07b_support import prediction_for, v3_bundle


def _dev_truths():
    return tuple(
        item for item in v3_bundle()[1].truths if item.case_id.startswith("V107-D-")
    )


def _different(value: V3Recommendation) -> V3Recommendation:
    return (
        V3Recommendation.APPROVE
        if value is not V3Recommendation.APPROVE
        else V3Recommendation.REJECT
    )


def test_perfect_dev_predictions_exercise_every_q_metric() -> None:
    truths = _dev_truths()
    predictions = tuple(prediction_for(item) for item in truths)
    report = score_predictions(truths, predictions, split=V3Split.DEV)
    assert set(report) >= {f"Q-{index:02d}" for index in range(1, 13)}
    assert report["Q-01"]["value"] == 1.0
    assert report["Q-02"]["all_obligations"]["value"] == 1.0
    assert report["Q-02"]["supported_subset"]["value"] == 1.0
    assert report["Q-03"]["value"] == 1.0
    assert report["Q-03"]["macro_f1"] == 1.0
    assert report["Q-04"]["high_risk_false_allow_count"] == 0
    assert report["Q-05"]["value"] == 1.0
    assert report["Q-06"]["conflict"]["value"] == 1.0
    assert report["Q-06"]["security"]["value"] == 1.0
    assert report["Q-07"]["value"] == 1.0
    assert report["Q-07"]["out_of_scope_citation_count"] == 0
    assert report["Q-08"]["value"] == 1.0
    assert report["Q-09"]["p95_latency_ms"] == 1
    assert report["Q-11"]["secret_leak"]["value"] == 0.0
    assert report["Q-03"]["overall_meets_threshold"] is True
    assert report["Q-04"]["overall_meets_threshold"] is True
    assert report["Q-05"]["overall_meets_threshold"] is True
    assert report["Q-06"]["conflict"]["overall_meets_threshold"] is True
    assert report["Q-07"]["overall_meets_threshold"] is True
    assert report["Q-08"]["overall_meets_threshold"] is True
    assert report["Q-11"]["overall_meets_threshold"] is True


def test_empty_denominators_are_undetermined_instead_of_perfect() -> None:
    value = proportion(0, 0)
    assert value.value is None
    assert value.status == "undetermined_empty_denominator"
    assert wilson_interval(0, 0) is None
    assert macro_f1((), ()) is None
    report = score_predictions(_dev_truths(), (), split=V3Split.DEV)
    assert report["Q-01"]["meets_threshold"] is None
    assert (
        report["Q-03"]["recall_by_class"]["recommend_approve"]["meets_threshold"]
        is None
    )
    assert report["Q-04"]["meets_threshold"] is None
    assert report["Q-06"]["conflict"]["precision"]["value"] is None
    assert report["Q-07"]["value"] is None
    assert report["Q-11"]["external_write"]["meets_threshold"] is None


def test_all_unknown_and_all_reject_cannot_game_the_quality_gates() -> None:
    truths = _dev_truths()
    unknown_predictions = []
    reject_predictions = []
    for truth in truths:
        base = prediction_for(truth)
        unknown_claims = tuple(
            claim.model_copy(update={"state": ObligationState.UNKNOWN})
            for claim in base.claims
        )
        unknown_predictions.append(
            base.model_copy(
                update={
                    "recommendation": V3Recommendation.INSUFFICIENT,
                    "claims": unknown_claims,
                }
            )
        )
        reject_predictions.append(
            base.model_copy(update={"recommendation": V3Recommendation.REJECT})
        )
    unknown = score_predictions(truths, unknown_predictions, split=V3Split.DEV)
    assert unknown["Q-02"]["all_obligations"]["value"] < 0.20
    assert unknown["Q-03"]["meets_threshold"] is False
    assert unknown["Q-04"]["meets_threshold"] is None
    reject = score_predictions(truths, reject_predictions, split=V3Split.DEV)
    assert reject["Q-03"]["meets_threshold"] is False
    assert reject["Q-03"]["macro_f1_meets_threshold"] is False
    assert reject["Q-04"]["meets_threshold"] is None


def test_correct_evidence_id_with_wrong_proposition_or_value_gets_no_semantic_credit() -> (
    None
):
    truth = _dev_truths()[0]
    base = prediction_for(truth)
    first = base.claims[0]
    wrong_proposition = first.proposition.model_copy(
        update={"field_path": "policy.wrong_claim"}
    )
    wrong_claim = first.model_copy(update={"proposition": wrong_proposition})
    prediction = base.model_copy(update={"claims": (wrong_claim, *base.claims[1:])})
    report = score_predictions((truth,), (prediction,), split=V3Split.DEV)
    assert report["Q-07"]["legacy_id_only"]["value"] == 1.0
    assert report["Q-07"]["value"] == pytest.approx(11 / 12)
    wrong_value = first.model_copy(update={"normalized_value": "wrong_value"})
    prediction = base.model_copy(update={"claims": (wrong_value, *base.claims[1:])})
    report = score_predictions((truth,), (prediction,), split=V3Split.DEV)
    assert report["Q-07"]["value"] == pytest.approx(11 / 12)


def test_correct_conflict_id_with_wrong_side_value_gets_no_conflict_credit() -> None:
    truth = next(item for item in _dev_truths() if item.conflicts)
    base = prediction_for(truth)
    conflict = base.conflicts[0]
    forged = conflict.model_copy(update={"right_value": "forged-value"})
    prediction = base.model_copy(update={"conflicts": (forged,)})
    report = score_predictions((truth,), (prediction,), split=V3Split.DEV)
    assert report["Q-06"]["conflict"]["numerator"] == 0
    assert report["Q-06"]["conflict"]["denominator"] == 1
    assert report["Q-06"]["conflict"]["precision"]["value"] == 0.0


def test_deterministic_zero_call_branch_is_excluded_from_q01_denominator() -> None:
    first, second = _dev_truths()[:2]
    zero_call = prediction_for(first, model_request_count=0)
    invalid = prediction_for(second, first_pass_valid=False)
    invalid = invalid.model_copy(
        update={"received_nonempty_body": False, "latency_ms": None}
    )
    report = score_predictions((first, second), (zero_call, invalid), split=V3Split.DEV)
    assert report["Q-01"]["denominator"] == 1
    assert report["Q-01"]["numerator"] == 0
    assert report["Q-01"]["deterministic_zero_call_count"] == 1
    assert report["Q-09"]["latency_unobservable_count"] == 2
    assert report["Q-09"]["meets_threshold"] is False


def test_duplicate_repeat_seed_for_same_case_and_topology_fails_closed() -> None:
    truth = _dev_truths()[0]
    prediction = prediction_for(truth)
    with pytest.raises(ValueError, match="duplicate repeat seed"):
        score_predictions((truth,), (prediction, prediction), split=V3Split.DEV)


def test_cluster_bootstrap_is_deterministic_and_clusters_near_duplicates() -> None:
    truths = _dev_truths()[:4]
    predictions = []
    groups = {}
    for index, truth in enumerate(truths):
        predictions.append(prediction_for(truth, topology=EvalTopology.MULTI_AGENT))
        predictions.append(
            prediction_for(
                truth,
                topology=EvalTopology.SINGLE_AGENT,
                recommendation=_different(truth.expected_recommendation),
            )
        )
        groups[truth.case_id] = f"cluster-{index // 2}"
    first = score_predictions(
        truths,
        predictions,
        split=V3Split.DEV,
        near_duplicate_groups=groups,
    )["Q-12"]["topology"]["multi_minus_single_accuracy"]
    second = score_predictions(
        truths,
        predictions,
        split=V3Split.DEV,
        near_duplicate_groups=groups,
    )["Q-12"]["topology"]["multi_minus_single_accuracy"]
    assert first == second
    assert first["value"] == 1.0
    assert first["cluster_bootstrap_ci"] == (1.0, 1.0)
    assert first["paired_cases"] == 4
    assert first["clusters"] == 2


def test_q08_reports_registered_slices_and_wrong_scope_without_counting_negatives() -> (
    None
):
    truth = _dev_truths()[0]
    base = prediction_for(truth)
    first = base.retrieval_observations[0].model_copy(
        update={"top_five": (), "wrong_scope_count": 1}
    )
    prediction = base.model_copy(
        update={"retrieval_observations": (first, *base.retrieval_observations[1:])}
    )
    report = score_predictions((truth,), (prediction,), split=V3Split.DEV)
    assert report["Q-08"]["value"] < 1.0
    assert report["Q-08"]["wrong_scope_count"] == 1
    assert report["Q-08"]["scope_version_gate_meets_threshold"] is False
    assert report["Q-08"]["overall_meets_threshold"] is False
    assert set(report["Q-08"]["by_query_variant_and_domain"]) == {
        "original:policy",
        "original:contract",
        "without_case_id:policy",
        "without_case_id:contract",
    }
    assert report["Q-08"]["negative_false_retrieval"]["value"] is None


def test_q08_does_not_infer_retrieval_recall_from_generic_evidence() -> None:
    truth = _dev_truths()[0]
    prediction = prediction_for(truth).model_copy(update={"retrieval_observations": ()})
    assert prediction.retrieved_evidence
    report = score_predictions((truth,), (prediction,), split=V3Split.DEV)
    assert report["Q-08"]["value"] is None
    assert report["Q-08"]["meets_threshold"] is None
    assert report["Q-08"]["scope_version_gate_meets_threshold"] is None
    assert report["Q-08"]["overall_meets_threshold"] is None
    assert all(
        item["value"] is None
        for item in report["Q-08"]["by_query_variant_and_domain"].values()
    )


def test_q08_rejects_self_reported_required_evidence() -> None:
    truth = _dev_truths()[0]
    base = prediction_for(truth)
    first = base.retrieval_observations[0]
    forged = RetrievalObservation(
        query_variant=first.query_variant,
        evidence_domain=first.evidence_domain,
        required_evidence=(truth.required_evidence[-1],),
        top_five=(truth.required_evidence[-1],),
    )
    prediction = base.model_copy(
        update={"retrieval_observations": (forged, *base.retrieval_observations[1:])}
    )
    with pytest.raises(ValueError, match="independent required evidence"):
        score_predictions((truth,), (prediction,), split=V3Split.DEV)


def test_q08_incomplete_registered_slice_matrix_cannot_pass_overall() -> None:
    truth = _dev_truths()[0]
    base = prediction_for(truth)
    prediction = base.model_copy(
        update={"retrieval_observations": base.retrieval_observations[:1]}
    )
    report = score_predictions((truth,), (prediction,), split=V3Split.DEV)
    assert report["Q-08"]["value"] < 1.0
    assert report["Q-08"]["observation_coverage"]["value"] == 0.25
    assert report["Q-08"]["overall_meets_threshold"] is False


def test_partial_case_matrix_is_visible_and_cannot_pass_q03_overall() -> None:
    truths = _dev_truths()[:2]
    report = score_predictions(
        truths,
        (prediction_for(truths[0]),),
        split=V3Split.DEV,
    )
    assert report["Q-03"]["value"] == 1.0
    assert report["observation_coverage"]["value"] == 0.5
    assert report["Q-03"]["overall_meets_threshold"] is False


def test_q12_averages_repeats_within_case_before_cluster_bootstrap() -> None:
    first, second = _dev_truths()[:2]
    predictions = (
        prediction_for(first, topology=EvalTopology.MULTI_AGENT, repeat_seed=1),
        prediction_for(
            first,
            topology=EvalTopology.SINGLE_AGENT,
            repeat_seed=1,
            recommendation=_different(first.expected_recommendation),
        ),
        prediction_for(first, topology=EvalTopology.MULTI_AGENT, repeat_seed=2),
        prediction_for(
            first,
            topology=EvalTopology.SINGLE_AGENT,
            repeat_seed=2,
            recommendation=_different(first.expected_recommendation),
        ),
        prediction_for(
            second,
            topology=EvalTopology.MULTI_AGENT,
            repeat_seed=1,
            recommendation=_different(second.expected_recommendation),
        ),
        prediction_for(second, topology=EvalTopology.SINGLE_AGENT, repeat_seed=1),
    )
    result = score_predictions(
        (first, second),
        predictions,
        split=V3Split.DEV,
    )["Q-12"]["topology"]["multi_minus_single_accuracy"]
    assert result["value"] == 0.0
    assert result["paired_cases"] == 2
    assert result["paired_observations"] == 3


def test_new_report_uses_v4_and_keeps_legacy_metrics_side_by_side() -> None:
    truths = v3_bundle()[1].truths
    predictions = tuple(prediction_for(item) for item in truths)
    report = build_report_v4(
        truths,
        predictions,
        legacy={"frozen_formula": {"value": 0.25}},
    )
    assert report["schema_version"] == "eval-report.v4"
    assert set(report["scores"]) == {"dev", "validation"}
    assert report["scores"]["dev"]["legacy"]["frozen_formula"] == {"value": 0.25}
    assert set(report["scores"]["dev"]["by_topology_and_seed"]) == {
        "multi_agent:20260919"
    }
