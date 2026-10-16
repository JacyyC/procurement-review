"""Independent synthetic-source adjudication, never used as production rules.

Unlike v3's boolean-only O-09 and unconditional O-05, these text obligations
require the actual branch-visible terms. Missing text cannot be supplied by MCP.
"""

import re
from .gold_v3_contracts import (
    ObligationState,
    TruthSeverity,
    V3Recommendation,
    payload_sha256,
)
from .gold_v3_oracle import derive_truth as legacy_truth
from .gold_v31_contracts import TruthRecordV11


def source_sufficiency(case) -> dict[str, bool]:
    text = case.model_input.contract_document.content
    facts = case.model_input.business_facts
    # Authored from the business specification, independent of the generator.
    identity = all(
        value and value in text
        for value in (
            facts.supplier_id,
            facts.purchase_order_id,
            facts.contract_reference,
            facts.counterparty_reference,
        )
    )
    parties = ("买方" in text and "卖方" in text) or (
        "BUYER=" in text and "SELLER=" in text
    )
    subject = bool(re.search(r"(?:标的：|SUBJECT=)[^\n;|。]+", text))
    acceptance = (
        "到货数量逐项相符且完好率100%" in text
        or "all quantities match and 100% undamaged" in text
    )
    warranty = (
        "卖方免费修复或更换缺陷交付物" in text
        or "seller repairs or replaces defective deliverables at no charge" in text
    )
    explicit_cap = bool(
        re.search(
            r"(?:累计赔偿责任上限为|LIABILITY_LIMIT=)\s*\d+\.\d{2}\s+(?:CNY|USD)", text
        )
    )
    change = (
        "双方书面签署后变更生效" in text
        or "CHANGE=effective after both parties sign in writing" in text
    )
    return {
        "O-05": bool(identity and parties and subject),
        "O-08": acceptance and warranty,
        "O-09": explicit_cap and change,
    }


def _recommendation(obligations, conflicts, security_events):
    if conflicts or security_events:
        return V3Recommendation.HUMAN
    unknown = {
        o.obligation_id for o in obligations if o.state is ObligationState.UNKNOWN
    }
    high = any(
        o.state is ObligationState.VIOLATED
        and o.severity in {TruthSeverity.HIGH, TruthSeverity.CRITICAL}
        for o in obligations
    )
    if high and not unknown.intersection({"O-01", "O-10"}):
        return V3Recommendation.REJECT
    if unknown:
        return V3Recommendation.INSUFFICIENT
    if any(o.state is ObligationState.VIOLATED for o in obligations):
        return V3Recommendation.CONDITIONAL
    return V3Recommendation.APPROVE


def derive_truth(case) -> TruthRecordV11:
    prior = legacy_truth(case)
    sufficiency = source_sufficiency(case)
    obligations = []
    for item in prior.obligations:
        if item.obligation_id in sufficiency and not sufficiency[item.obligation_id]:
            item = item.model_copy(
                update={
                    "state": ObligationState.UNKNOWN,
                    "normalized_value": "document_terms_insufficient",
                    "severity": TruthSeverity.NONE,
                    "reason_code": "visible_contract_terms_missing",
                    # Procurement booleans do not prove private contract prose.
                    "evidence": tuple(
                        e
                        for e in item.evidence
                        if e.source_id == case.model_input.contract_document.source_id
                    ),
                }
            )
        obligations.append(item)
    values = {
        name: getattr(prior, name)
        for name in type(prior).model_fields
        if name != "truth_checksum"
    }
    values.update(
        truth_contract="v1-07-truth.v1.1",
        obligations=tuple(obligations),
        expected_recommendation=_recommendation(
            tuple(obligations), prior.conflicts, prior.security_events
        ),
    )
    view = TruthRecordV11.model_construct(**values, truth_checksum="0" * 64)
    values["truth_checksum"] = payload_sha256(
        view.model_dump(mode="json", exclude={"truth_checksum"})
    )
    return TruthRecordV11(**values)
