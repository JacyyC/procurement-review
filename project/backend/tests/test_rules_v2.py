"""Rules 2.0: the V1-05 contract revision keeps safety and gains decidability.

Every test here is offline and model-free. The frozen rules 1.0 behaviour is covered by
``test_rules.py``; this file asserts what ``resources/rules/v2.json`` changed and what it
deliberately did not change.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from procurement_review.contracts import (
    BusinessFact,
    CauseClass,
    EscalationClass,
    EvidenceRef,
    FactStatus,
    Finding,
    FindingAssertion,
    FindingStatus,
    HumanReviewStatus,
    RecommendationStatus,
    ReviewSummary,
    RiskCategory,
    RiskLevel,
    SourceKind,
    WorkerKind,
)
from procurement_review.rules import (
    REQUIRED_FINDING_KEYS,
    RULE_SET,
    RULE_SET_V2,
    RULE_SETS,
    RULES_VERSION,
    RULES_VERSION_V2,
    aggregate_findings,
    classify_cause,
    escalation_class,
    is_generated_reason_code,
    rule_set_for_version,
)

CHECKSUM = "sha256:" + "b" * 64
NOW = datetime(2026, 9, 11, 8, 0, tzinfo=UTC)


def evidence(source_id: str) -> EvidenceRef:
    return EvidenceRef(
        source_kind=SourceKind.PUBLIC_DOCUMENT,
        source_id=source_id,
        version="v1",
        chunk_id=f"chunk-{source_id}",
        content_checksum=CHECKSUM,
    )


def business_fact(record_id: str = "po-001") -> BusinessFact:
    return BusinessFact(
        tool_name="get_purchase_order",
        record_id=record_id,
        dataset_version="fixture-v1",
        as_of=NOW,
        status=FactStatus.KNOWN,
        result={"amount": 1200},
    )


def supported(
    key: str,
    category: RiskCategory,
    worker: WorkerKind,
    *,
    risk: RiskLevel = RiskLevel.LOW,
    assertion: FindingAssertion = FindingAssertion.RISK_ABSENT,
    refs: tuple[EvidenceRef, ...] | None = None,
    facts: tuple[BusinessFact, ...] | None = None,
    notes: tuple[str, ...] = (),
) -> Finding:
    if refs is None and facts is None:
        refs = (evidence(key),)
    if facts is None and worker is WorkerKind.PROCUREMENT_DATA:
        refs, facts = (), (business_fact(key),)
    return Finding(
        worker=worker,
        finding_key=key,
        category=category,
        risk_level=risk,
        assertion=assertion,
        statement=f"Supported result for {key}.",
        status=FindingStatus.SUPPORTED,
        evidence_refs=refs or (),
        business_fact_refs=facts or (),
        missing_information=notes,
        recommended_action="Continue deterministic review.",
    )


def unresolved(
    key: str,
    category: RiskCategory,
    worker: WorkerKind,
    *,
    status: FindingStatus = FindingStatus.INSUFFICIENT_EVIDENCE,
) -> Finding:
    return Finding(
        worker=worker,
        finding_key=key,
        category=category,
        risk_level=RiskLevel.UNKNOWN,
        assertion=FindingAssertion.UNKNOWN,
        statement=f"{key} could not be decided.",
        status=status,
        missing_information=(f"{key} evidence",),
        recommended_action="Require human review.",
    )


def complete_low_risk() -> tuple[Finding, ...]:
    return (
        supported(
            REQUIRED_FINDING_KEYS[0],
            RiskCategory.POLICY_COMPLIANCE,
            WorkerKind.POLICY,
        ),
        supported(
            REQUIRED_FINDING_KEYS[1],
            RiskCategory.CONTRACT_TERMS,
            WorkerKind.CONTRACT,
        ),
        supported(
            REQUIRED_FINDING_KEYS[2],
            RiskCategory.PROCUREMENT_ANOMALY,
            WorkerKind.PROCUREMENT_DATA,
        ),
    )


def v2(findings: tuple[Finding, ...], **kwargs: object) -> ReviewSummary:
    return aggregate_findings(
        run_id="run-v2",
        case_id="case-v2",
        findings=findings,
        rule_set=RULE_SET_V2,
        **kwargs,  # type: ignore[arg-type]
    )


def v1(findings: tuple[Finding, ...], **kwargs: object) -> ReviewSummary:
    return aggregate_findings(
        run_id="run-v1",
        case_id="case-v1",
        findings=findings,
        **kwargs,  # type: ignore[arg-type]
    )


def test_v2_is_a_separate_versioned_resource_that_keeps_the_frozen_vocabulary() -> None:
    assert RULES_VERSION == "1.0.0"
    assert RULES_VERSION_V2 == "2.0.0"
    assert set(RULE_SETS) == {"1.0.0", "2.0.0"}
    assert RULE_SET.gate_mode == "all_gaps"
    assert RULE_SET.conflict_anchors == ("finding_key",)
    assert RULE_SET.escalation_precedence == ()
    assert RULE_SET.coverage_required_recommendations == ()
    assert RULE_SET_V2.gate_mode == "directional"
    # The M2 promise: what a model is asked to name must not change.
    assert RULE_SET_V2.required_finding_keys == RULE_SET.required_finding_keys
    assert RULE_SET_V2.required_finding_contracts == RULE_SET.required_finding_contracts
    assert RULE_SET_V2.risk_order == RULE_SET.risk_order
    assert (
        RULE_SET_V2.recommendation_by_highest_risk
        == RULE_SET.recommendation_by_highest_risk
    )
    assert RULE_SET_V2.conflict_anchors == (
        "finding_key",
        "evidence_ref",
        "business_fact",
    )
    assert RULE_SET_V2.escalation_precedence == ("conflict", "pipeline", "evidence")
    assert RULE_SET_V2.coverage_required_recommendations == (
        RecommendationStatus.RECOMMEND_APPROVE,
        RecommendationStatus.RECOMMEND_CONDITIONAL,
    )
    with pytest.raises(ValueError, match="unknown rules version"):
        rule_set_for_version("3.0.0")


def test_a_supported_high_risk_is_actionable_while_other_branches_stay_open() -> None:
    findings = (
        supported(
            REQUIRED_FINDING_KEYS[2],
            RiskCategory.PROCUREMENT_ANOMALY,
            WorkerKind.PROCUREMENT_DATA,
            risk=RiskLevel.HIGH,
            assertion=FindingAssertion.RISK_PRESENT,
        ),
        unresolved(
            REQUIRED_FINDING_KEYS[0], RiskCategory.POLICY_COMPLIANCE, WorkerKind.POLICY
        ),
        unresolved(
            REQUIRED_FINDING_KEYS[1], RiskCategory.CONTRACT_TERMS, WorkerKind.CONTRACT
        ),
    )

    frozen = v1(findings)
    revised = v2(findings)

    assert frozen.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert revised.recommendation is RecommendationStatus.RECOMMEND_REJECT
    assert revised.highest_risk is RiskLevel.HIGH
    assert revised.human_review_status is HumanReviewStatus.PENDING_CONFIRMATION
    assert escalation_class(revised) is EscalationClass.CONTENT
    assert any(
        code.startswith("evidence:unresolved_finding:")
        for code in revised.missing_information
    )


def test_an_approval_still_requires_every_required_dimension_to_be_supported() -> None:
    complete = v2(complete_low_risk())
    assert complete.recommendation is RecommendationStatus.RECOMMEND_APPROVE
    assert complete.missing_information == ()
    assert complete.failure_reasons == ()
    assert escalation_class(complete) is EscalationClass.CONTENT

    policy_only = v2(complete_low_risk()[:1] + complete_low_risk()[2:])
    assert policy_only.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert policy_only.human_review_status is HumanReviewStatus.REQUIRED
    assert escalation_class(policy_only) is EscalationClass.EVIDENCE_INSUFFICIENT
    assert "evidence:coverage_incomplete:contract.required_terms" in (
        policy_only.missing_information
    )
    assert "coverage:missing_required_finding:contract.required_terms" in (
        policy_only.missing_information
    )


def test_a_branch_failure_closes_the_end_to_end_result_regardless_of_risk() -> None:
    findings = complete_low_risk() + (
        supported(
            "policy.exception",
            RiskCategory.POLICY_COMPLIANCE,
            WorkerKind.POLICY,
            risk=RiskLevel.CRITICAL,
            assertion=FindingAssertion.RISK_PRESENT,
        ),
    )
    summary = v2(findings, upstream_failures=("worker_failure:contract:timeout",))

    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert summary.human_review_status is HumanReviewStatus.REQUIRED
    assert escalation_class(summary) is EscalationClass.PIPELINE_INCOMPLETE
    assert "pipeline:worker_failure:contract:timeout" in summary.failure_reasons
    assert is_generated_reason_code("pipeline:worker_failure:contract:timeout")


def test_conflict_on_a_shared_reference_is_detected_within_one_workers_own_findings() -> (
    None
):
    shared = evidence("case-doc-0001")
    findings = (
        *complete_low_risk(),
        supported(
            "contract.auto_renewal_present",
            RiskCategory.CONTRACT_TERMS,
            WorkerKind.CONTRACT,
            risk=RiskLevel.HIGH,
            assertion=FindingAssertion.RISK_PRESENT,
            refs=(shared,),
        ),
        supported(
            "contract.auto_renewal_absent",
            RiskCategory.CONTRACT_TERMS,
            WorkerKind.CONTRACT,
            risk=RiskLevel.LOW,
            assertion=FindingAssertion.RISK_ABSENT,
            refs=(shared,),
        ),
    )

    frozen = v1(findings)
    revised = v2(findings)

    assert frozen.conflicts == ()
    assert len(revised.conflicts) == 1
    conflict = revised.conflicts[0]
    assert conflict.reason.value == "opposing_assertions"
    assert conflict.finding_key == "conflict:evidence_ref:case-doc-0001/v1"
    assert {item.value for item in conflict.assertions} == {
        "risk_present",
        "risk_absent",
    }
    # R-4: an absence never cancels a presence; the case escalates instead.
    assert revised.recommendation is RecommendationStatus.NEEDS_HUMAN_REVIEW
    assert revised.highest_risk is RiskLevel.HIGH
    assert escalation_class(revised) is EscalationClass.CONFLICT


def test_a_conflict_still_adjudicates_when_a_branch_also_failed() -> None:
    shared = evidence("case-doc-0002")
    findings = (
        supported(
            "a_present",
            RiskCategory.DATA_QUALITY,
            WorkerKind.POLICY,
            risk=RiskLevel.HIGH,
            assertion=FindingAssertion.RISK_PRESENT,
            refs=(shared,),
        ),
        supported(
            "a_absent",
            RiskCategory.DATA_QUALITY,
            WorkerKind.POLICY,
            risk=RiskLevel.LOW,
            assertion=FindingAssertion.RISK_ABSENT,
            refs=(shared,),
        ),
    )
    summary = v2(findings, input_errors=("missing_worker_branch:contract",))

    assert summary.conflicts
    assert summary.recommendation is RecommendationStatus.NEEDS_HUMAN_REVIEW
    assert escalation_class(summary) is EscalationClass.CONFLICT
    # The pipeline cause is never absorbed by the conflict class: it stays recorded.
    assert summary.failure_reasons == ("pipeline:missing_worker_branch:contract",)


def test_a_branch_failure_without_a_conflict_closes_the_result() -> None:
    summary = v2(
        complete_low_risk(),
        upstream_failures=("worker_failure:contract:timeout",),
    )
    assert summary.conflicts == ()
    assert summary.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert escalation_class(summary) is EscalationClass.PIPELINE_INCOMPLETE


def test_a_shared_key_conflict_keeps_the_rules_1_subject_name() -> None:
    findings = complete_low_risk() + (
        supported(
            "shared.fixture_conflict",
            RiskCategory.POLICY_COMPLIANCE,
            WorkerKind.POLICY,
            risk=RiskLevel.MEDIUM,
            assertion=FindingAssertion.RISK_PRESENT,
        ),
        supported(
            "shared.fixture_conflict",
            RiskCategory.CONTRACT_TERMS,
            WorkerKind.CONTRACT,
            assertion=FindingAssertion.RISK_ABSENT,
        ),
    )
    assert v1(findings).conflicts == v2(findings).conflicts


def test_a_wrong_source_required_key_is_recorded_but_no_longer_gates() -> None:
    mismatched = list(complete_low_risk())
    mismatched[0] = supported(
        REQUIRED_FINDING_KEYS[0],
        RiskCategory.POLICY_COMPLIANCE,
        WorkerKind.CONTRACT,
    )
    frozen = v1(tuple(mismatched))
    assert frozen.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert (
        f"required_finding_source_mismatch:{REQUIRED_FINDING_KEYS[0]}"
        in frozen.missing_information
    )

    demoted = v2(tuple(mismatched))
    assert (
        f"coverage:required_finding_source_mismatch:{REQUIRED_FINDING_KEYS[0]}"
        in demoted.missing_information
    )
    assert demoted.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION

    risky = (*mismatched, unsupported_high_risk())
    rejecting = v2(risky)
    assert rejecting.recommendation is RecommendationStatus.RECOMMEND_REJECT
    # R-4: the mismatch never hides a supported high risk found elsewhere.
    assert rejecting.highest_risk is RiskLevel.HIGH


def unsupported_high_risk() -> Finding:
    return supported(
        "procurement.blocked_supplier",
        RiskCategory.SUPPLIER_INTEGRITY,
        WorkerKind.PROCUREMENT_DATA,
        risk=RiskLevel.HIGH,
        assertion=FindingAssertion.RISK_PRESENT,
    )


def test_model_self_reported_gaps_are_content_notes_and_never_a_gate() -> None:
    findings = tuple(
        finding.model_copy(
            update={"missing_information": ("supplier.due_diligence_status",)}
        )
        for finding in complete_low_risk()
    )
    frozen = v1(findings)
    revised = v2(findings)

    assert frozen.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert revised.recommendation is RecommendationStatus.RECOMMEND_APPROVE
    assert "supplier.due_diligence_status" in revised.missing_information
    assert classify_cause("supplier.due_diligence_status") is CauseClass.CONTENT_NOTE


def test_an_unshapeable_cannot_be_silently_dropped_from_the_pipeline_class() -> None:
    summary = v2(
        complete_low_risk(),
        upstream_failures=("a caller wrote free-form prose here",),
    )
    assert len(summary.failure_reasons) == 1
    assert summary.failure_reasons[0].startswith("pipeline:unlisted_cause:")
    assert "a caller wrote free-form prose here" in summary.missing_information
    assert escalation_class(summary) is EscalationClass.PIPELINE_INCOMPLETE


def test_v2_is_deterministic_for_input_order() -> None:
    findings = complete_low_risk() + (
        supported(
            "procurement.amount_mismatch",
            RiskCategory.FINANCIAL_EXPOSURE,
            WorkerKind.PROCUREMENT_DATA,
            risk=RiskLevel.MEDIUM,
            assertion=FindingAssertion.RISK_PRESENT,
        ),
    )
    forward = v2(findings)
    reverse = v2(tuple(reversed(findings)))
    assert forward.model_dump_json() == reverse.model_dump_json()
    assert forward.recommendation is RecommendationStatus.RECOMMEND_CONDITIONAL


def test_cause_classification_only_accepts_registered_shapes() -> None:
    assert (
        classify_cause("pipeline:worker_failure:policy:timeout") is CauseClass.PIPELINE
    )
    assert (
        classify_cause("evidence:unresolved_finding:contract.required_terms:unknown")
        is CauseClass.EVIDENCE
    )
    assert (
        classify_cause("coverage:critical_evidence_missing:policy.x:evidence_ref")
        is CauseClass.COVERAGE
    )
    assert classify_cause("worker_failure:policy:timeout") is CauseClass.PIPELINE
    assert classify_cause("evidence: the invoice is missing") is CauseClass.CONTENT_NOTE
    assert classify_cause("缺失 evidence:invoice") is CauseClass.CONTENT_NOTE
    assert not is_generated_reason_code("evidence: the invoice is missing")
    assert is_generated_reason_code(
        "unresolved_finding:contract.required_terms:unknown"
    )


def test_the_contract_refuses_a_v2_summary_that_gates_inconsistently() -> None:
    approved = json.loads(v2(complete_low_risk()).model_dump_json())

    def rebuild(**changes: object) -> ReviewSummary:
        return ReviewSummary.model_validate_json(
            json.dumps({**approved, **changes}, ensure_ascii=False)
        )

    with pytest.raises(ValidationError, match="approval requires complete"):
        rebuild(missing_information=["evidence:coverage_incomplete:policy.x"])

    with pytest.raises(ValidationError, match="no pipeline failure"):
        rebuild(
            recommendation="recommend_reject",
            highest_risk="high",
            failure_reasons=["pipeline:worker_failure:policy:timeout"],
        )

    with pytest.raises(ValidationError, match="not a known aggregation contract"):
        # Rules 1, 2 and 3 are the registered aggregation contracts (3.x is the rules
        # 3.1.0 line's own summary contract); a version outside them is refused rather
        # than being validated under the wrong gate semantics.
        rebuild(rules_version="4.0.0")
