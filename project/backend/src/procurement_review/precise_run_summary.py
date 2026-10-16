"""The product summary projection of one precise decision.

The execution record carries the exact propositions and evidence identities; this
projection states which of the decision's duties and causes the product gate must
act on, and which stored source each of them names.  It is bounded on purpose and
never invents a source: a supported duty whose evidence cannot be mapped back to
the material the Worker was handed refuses to commit.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from .business_v3_procurement import canonical_json_bytes
from .business_v3_types import BusinessContextV3
from .contracts import (
    BusinessFact,
    EvidenceRef,
    Finding,
    FindingAssertion,
    FindingStatus,
    HumanReviewStatus,
    RecommendationStatus,
    ReviewSummary,
    RiskCategory,
    RiskLevel,
    WorkerKind,
)

#: The product's risk vocabulary has no "no risk" member, so a rules 3.1 duty that
#: reports no risk is recorded at the lowest known level instead of inventing one.
PRODUCT_RISK = {
    "none": RiskLevel.LOW,
    "low": RiskLevel.LOW,
    "medium": RiskLevel.MEDIUM,
    "high": RiskLevel.HIGH,
    "critical": RiskLevel.CRITICAL,
    "unknown": RiskLevel.UNKNOWN,
}

#: The duties the product summary carries, with the Worker each belongs to.  O-11
#: and O-12 are the control rows and stay in the execution record itself.
DUTY_OWNERS: dict[str, WorkerKind] = {
    "O-01": WorkerKind.POLICY,
    "O-02": WorkerKind.POLICY,
    "O-04": WorkerKind.POLICY,
    "O-05": WorkerKind.CONTRACT,
    "O-06": WorkerKind.CONTRACT,
    "O-08": WorkerKind.CONTRACT,
    "O-09": WorkerKind.CONTRACT,
    "O-03": WorkerKind.PROCUREMENT_DATA,
    "O-07": WorkerKind.PROCUREMENT_DATA,
    "O-10": WorkerKind.PROCUREMENT_DATA,
}
DUTY_CATEGORIES: dict[str, RiskCategory] = {
    "O-01": RiskCategory.POLICY_COMPLIANCE,
    "O-02": RiskCategory.POLICY_COMPLIANCE,
    "O-04": RiskCategory.POLICY_COMPLIANCE,
    "O-05": RiskCategory.CONTRACT_TERMS,
    "O-06": RiskCategory.CONTRACT_TERMS,
    "O-08": RiskCategory.CONTRACT_TERMS,
    "O-09": RiskCategory.CONTRACT_TERMS,
    "O-03": RiskCategory.SUPPLIER_INTEGRITY,
    "O-07": RiskCategory.DATA_QUALITY,
    "O-10": RiskCategory.DATA_QUALITY,
}


def sha256_of_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first(values: tuple, fallback: str) -> str:
    return values[0] if values else fallback


def business_context(scope: Any, procurement_read: Any) -> BusinessContextV3:
    """Build the business context from the scope and the read facts, never a label."""

    order = next(
        (
            fact
            for fact in procurement_read.facts
            if fact.tool_name == "get_purchase_order"
        ),
        None,
    )
    body = dict(order.result) if order is not None else {}
    as_of = order.as_of.date() if order is not None else datetime.now(UTC).date()
    return BusinessContextV3(
        scope_id=f"{scope.tenant_id}:{scope.case_id}",
        purchase_order_id=str(
            body.get(
                "purchase_order_id", _first(scope.allowed_order_ids, scope.case_id)
            )
        ),
        supplier_id=str(
            body.get("supplier_id", _first(scope.allowed_supplier_ids, scope.case_id))
        ),
        contract_id=str(
            body.get(
                "contract_reference", _first(scope.allowed_order_ids, scope.case_id)
            )
        ),
        as_of=as_of,
        dataset_version=scope.dataset_version,
    )


def evidence_map(hits_by_worker: dict[str, tuple], procurement_facts: tuple) -> dict:
    """Map every evidence identity a branch may cite back to its product source.

    Document identities resolve to the product reference of the hit the Worker was
    actually handed; fact identities resolve to the read-only fact beside it.  A
    duty whose source cannot be named this way is refused rather than committed.
    """

    result: dict[str, dict[str, Any]] = {}
    for hits in hits_by_worker.values():
        for hit in hits:
            result.setdefault(hit.source_id, {})[
                hit.content_checksum.removeprefix("sha256:")
            ] = hit.evidence_ref
    for fact in procurement_facts:
        digest = hashlib.sha256(canonical_json_bytes(fact.result)).hexdigest()
        result.setdefault(fact.record_id, {})[digest] = fact
    return result


def summary_from_precise(
    *,
    run_id: str,
    case_id: str,
    decision: Any,
    security: tuple,
    conflicts: tuple,
    evidence_map: dict,
    procurement_facts: tuple,
) -> ReviewSummary:
    """Project the precise decision into the product's gate summary."""

    findings: list[Finding] = []
    for row in decision.findings:
        if row.obligation_id not in DUTY_OWNERS:
            continue
        supported = row.state in {"supported", "violated", "not_applicable"}
        references: list[EvidenceRef] = []
        facts: list[BusinessFact] = []
        for identity in row.evidence:
            mapped = evidence_map.get(identity.source_id, {}).get(
                identity.content_sha256
            )
            if isinstance(mapped, EvidenceRef):
                references.append(mapped)
            elif isinstance(mapped, BusinessFact):
                facts.append(mapped)
        if supported and not references and not facts:
            raise ValueError(
                f"a supported duty must name its source: {row.obligation_id}"
            )
        findings.append(
            Finding(
                worker=DUTY_OWNERS[row.obligation_id],
                finding_key=row.obligation_id,
                category=DUTY_CATEGORIES[row.obligation_id],
                risk_level=PRODUCT_RISK[row.risk],
                assertion=(
                    FindingAssertion.RISK_PRESENT
                    if row.state == "violated"
                    else FindingAssertion.RISK_ABSENT
                    if supported
                    else FindingAssertion.UNKNOWN
                ),
                statement=f"rules 3.1.0 {row.obligation_id}: {row.reason}",
                status=FindingStatus.SUPPORTED if supported else FindingStatus.UNKNOWN,
                evidence_refs=tuple(references),
                business_fact_refs=tuple(facts),
                missing_information=tuple(row.missing)
                if row.state == "unknown"
                else (),
                recommended_action=("人工核对来源" if supported else "补充所缺证据"),
            )
        )
    causes: list[str] = []
    for row in decision.findings:
        if row.state == "unknown":
            causes.extend(row.missing)
    causes.extend(f"conflict:{pair.pair_id}" for pair in conflicts)
    causes.extend(f"security:{event.code}" for event in security if event.escalates)
    if (
        decision.recommendation == RecommendationStatus.NEEDS_HUMAN_REVIEW.value
        and not causes
    ):
        causes.append("human_review_required:non_remediable_violation")
    recommendation = RecommendationStatus(decision.recommendation)
    gate_status = (
        HumanReviewStatus.PENDING_CONFIRMATION
        if recommendation
        in {
            RecommendationStatus.RECOMMEND_APPROVE,
            RecommendationStatus.RECOMMEND_CONDITIONAL,
            RecommendationStatus.RECOMMEND_REJECT,
        }
        else HumanReviewStatus.REQUIRED
    )
    highest = RiskLevel.UNKNOWN
    for level in ("critical", "high", "medium", "low", "none"):
        if any(row.risk == level for row in decision.findings):
            highest = PRODUCT_RISK[level]
            break
    del procurement_facts
    return ReviewSummary(
        run_id=run_id,
        case_id=case_id,
        rules_version=decision.rules_version,
        findings=tuple(findings),
        highest_risk=highest,
        recommendation=recommendation,
        human_review_status=gate_status,
        conflicts=(),
        missing_information=tuple(dict.fromkeys(causes)),
        failure_reasons=tuple(decision.pipeline_failures),
        is_final_decision=False,
    )


__all__ = [
    "DUTY_CATEGORIES",
    "DUTY_OWNERS",
    "PRODUCT_RISK",
    "business_context",
    "evidence_map",
    "sha256_of_text",
    "summary_from_precise",
]
