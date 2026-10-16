"""Counterfeit evidence must not earn semantic or obligation credit."""

import asyncio

import httpx
import pytest

from procurement_review.eval_f_context import DATA, load_cases
from procurement_review.eval_f_contracts import FCaseProjection
from procurement_review.eval_f_registration import prepare, registration
from procurement_review.eval_f_run import execute
from procurement_review.eval_f_score import score_case
from procurement_review.gold_v31_contracts import TruthFileV11
from v1_07f_support import response_for


@pytest.fixture(scope="module")
def observations(tmp_path_factory):
    root = tmp_path_factory.mktemp("duty-binding") / ".local"
    path = root / "registration.json"
    prepare(path)
    asyncio.run(
        execute(
            path,
            "minimum",
            transport_override=httpx.MockTransport(response_for),
            offline_root=root / "live",
        )
    )
    directory = next((root / "live").glob("*/minimum"))
    seed = registration()["candidates"]["minimum"]["repeat_seeds"][0]
    cases = {c.case_id: c for c in load_cases()}
    truths = {
        t.case_id: t
        for t in TruthFileV11.model_validate_json(
            (DATA / "truth.json").read_bytes()
        ).truths
    }
    return {
        c: (
            cases[c],
            truths[c],
            FCaseProjection.model_validate_json(
                (directory / f"{c}.{seed}.json").read_bytes()
            ),
        )
        for c in registration()["candidates"]["minimum"]["case_ids"]
    }


def replace_finding(projection, key, **changes):
    rows = tuple(
        f.model_copy(update=changes) if f.obligation_id == key else f
        for f in projection.decision.business.findings
    )
    return projection.model_copy(
        update={
            "decision": projection.decision.model_copy(
                update={
                    "business": projection.decision.business.model_copy(
                        update={"findings": rows}
                    )
                }
            )
        }
    )


def test_missing_case_preserves_escalation_and_decidable_denominators(observations):
    from procurement_review.eval_f_score import summarize

    rows = [score_case(case, truth, None) for case, truth, _ in observations.values()]
    result = summarize(rows)
    assert result["metrics"]["Q05_upgrade"]["denominator"] == 3
    assert result["metrics"]["Q05_upgrade"]["numerator"] == 0
    assert result["metrics"]["Q05_extra"]["denominator"] == 3
    assert result["metrics"]["Q05_extra"]["numerator"] == 3
    assert result["counts"]["missing_cases"] == 6
    assert not result["point_quality_passed"]


@pytest.mark.parametrize(
    "text",
    [
        "not WARRANTY at least 90 days",
        "并不要求质保至少90天",
        "尚未提供质保至少90天的约定",
        "WARRANTY at least 90 days is only an illustration",
    ],
)
def test_independent_semantic_score_rejects_negated_or_illustrative_rules(text):
    from procurement_review.eval_f_semantics import matches_value

    assert not matches_value("warranty", 90, text, "policy")


@pytest.mark.parametrize("key", [f"O-{i:02}" for i in range(1, 13)])
def test_each_duty_rejects_other_dutys_legal_evidence(observations, key):
    case, truth, projection = observations["V107-D-0001"]
    source = "O-03" if key in {"O-11", "O-12"} else "O-12"
    unrelated = next(
        f.evidence
        for f in projection.decision.business.findings
        if f.obligation_id == source
    )
    broken = replace_finding(projection, key, evidence=unrelated)
    score = score_case(case, truth, broken)
    assert key in score["failures"]
    assert score["counts"]["obligation_correct"] <= 11
    assert score["counts"]["semantic_valid"] < score["counts"]["semantic_total"]


@pytest.mark.parametrize(
    "key", ["O-03", "O-04", "O-05", "O-06", "O-07", "O-08", "O-10"]
)
def test_omitting_one_required_evidence_field_is_not_complete(observations, key):
    case, truth, projection = observations["V107-D-0001"]
    finding = next(
        f for f in projection.decision.business.findings if f.obligation_id == key
    )
    for omitted in range(len(finding.evidence)):
        partial = finding.evidence[:omitted] + finding.evidence[omitted + 1 :]
        score = score_case(
            case, truth, replace_finding(projection, key, evidence=partial)
        )
        assert key in score["failures"]


def test_unknown_must_identify_actual_missing_fact(observations):
    case, truth, projection = observations["V107-D-0004"]
    assert "O-04" not in score_case(case, truth, projection)["failures"]
    broken = replace_finding(projection, "O-04", missing=("unrelated_source",))
    assert "O-04" in score_case(case, truth, broken)["failures"]


def test_wrong_violation_severity_cannot_earn_obligation_credit(observations):
    case, truth, projection = observations["V107-D-0003"]
    broken = replace_finding(projection, "O-03", risk="low", blocking=False)
    assert "O-03" in score_case(case, truth, broken)["failures"]


def test_foreign_case_projection_cannot_be_scored_under_a_matching_source(observations):
    case, truth, projection = observations["V107-D-0001"]
    with pytest.raises(ValueError, match="Case identity"):
        score_case(case, truth, projection.model_copy(update={"case_id": "other"}))


def test_duplicate_branch_cannot_inflate_first_send_numerator(observations):
    case, truth, projection = observations["V107-D-0001"]
    with pytest.raises(ValueError, match="branch or duty identity"):
        score_case(
            case,
            truth,
            projection.model_copy(
                update={
                    "branches": (projection.branches[0], projection.branches[0]),
                }
            ),
        )


def test_otherwise_correct_observation_cannot_cite_another_field(observations):
    case, truth, projection = observations["V107-D-0001"]
    original = projection.decision.business.observations[0]
    unrelated = next(
        f.evidence[0]
        for f in projection.decision.business.findings
        if f.obligation_id == "O-12"
    )
    rows = (
        original.model_copy(update={"evidence": (unrelated,)}),
        *projection.decision.business.observations[1:],
    )
    broken = projection.model_copy(
        update={
            "decision": projection.decision.model_copy(
                update={
                    "business": projection.decision.business.model_copy(
                        update={"observations": rows}
                    )
                }
            )
        }
    )
    score = score_case(case, truth, broken)
    assert score["counts"]["semantic_valid"] < score["counts"]["semantic_total"]


def test_fact_observation_is_bound_to_its_duty(observations):
    from procurement_review.eval_f_semantics import fact_observation_valid

    case, _, projection = observations["V107-D-0001"]
    original = projection.decision.business.observations[0]
    assert fact_observation_valid(case, original)
    broken = original.model_copy(
        update={
            "proposition": original.proposition.model_copy(update={"check_id": "O-12"})
        }
    )
    assert not fact_observation_valid(case, broken)


def test_model_semantic_violation_is_not_valid_just_because_values_are_supported(
    observations,
):
    from procurement_review.gold_v3_contracts import ObligationState

    case, truth, projection = observations["V107-D-0001"]
    branch = projection.branches[0]
    findings = tuple(
        f.model_copy(
            update={
                "status": "violated",
                "assertion": "non_compliant",
                "risk_level": "high",
            }
        )
        if f.obligation_id == "O-01"
        else f
        for f in branch.evidence_output.findings
    )
    claims = tuple(
        c.model_copy(update={"state": ObligationState.VIOLATED})
        if c.obligation_id == "O-01"
        else c
        for c in branch.claims
    )
    details = tuple(
        c.model_copy(update={"risk_level": "high"}) if c.obligation_id == "O-01" else c
        for c in branch.details
    )
    branch = branch.model_copy(
        update={
            "evidence_output": branch.evidence_output.model_copy(
                update={"findings": findings}
            ),
            "claims": claims,
            "details": details,
        }
    )
    broken = projection.model_copy(
        update={"branches": (branch, projection.branches[1])}
    )
    result = score_case(case, truth, broken)
    assert result["counts"]["first_valid"] == 1
    assert result["counts"]["final_valid"] == 1


def test_missing_or_inconsistent_wire_sidecars_cannot_earn_validity(observations):
    case, truth, projection = observations["V107-D-0001"]
    branch = projection.branches[0]
    for changes in (
        {"claims": ()},
        {"details": ()},
        {
            "evidence_output": branch.evidence_output.model_copy(
                update={"findings": branch.evidence_output.findings[:1]}
            )
        },
    ):
        broken = projection.model_copy(
            update={
                "branches": (branch.model_copy(update=changes), projection.branches[1])
            }
        )
        assert score_case(case, truth, broken)["counts"]["first_valid"] == 1
