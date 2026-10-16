"""Independent source sufficiency for new documents; no production imports.

The business specification and arithmetic oracle remain the frozen v3 rules.
Only the new source expressions are adjudicated here. Original truths are never
rewritten, and no model answer is an input to this oracle.
"""

import re
from .gold_v3_contracts import ObligationState, TruthSeverity, payload_sha256
from .gold_v3_oracle import derive_truth as business_truth
from .gold_v31_oracle import source_sufficiency, _recommendation
from .gold_v32_contracts import TruthRecordV12


def sufficient_contract(case):
    if case.language == "zh-CN":
        return source_sufficiency(case)
    text = case.model_input.contract_document.content
    f = case.model_input.business_facts
    identity = all(
        value and value in text
        for value in (
            f.purchase_order_id,
            f.supplier_id,
            f.contract_reference,
            f.counterparty_reference,
        )
    )
    parties = bool(re.search(r"Buyer:\s*V107-.+?seller:\s*V107-", text, re.I))
    subject = bool(re.search(r"subject:\s*[^.;\n]+", text, re.I))
    standard = "every ordered unit delivered without damage" in text
    warranty = "seller repairs or replaces defects free of charge" in text
    written = "only after both parties sign in writing" in text
    cap = bool(re.search(r"Liability cap:\s*\d+\.\d{2}\s+(CNY|USD)", text))
    return {
        "O-05": bool(identity and parties and subject),
        "O-08": standard and warranty,
        "O-09": written and cap,
    }


def derive_truth(case):
    prior = business_truth(case)
    sufficiency = sufficient_contract(case)
    duties = []
    for item in prior.obligations:
        if item.obligation_id in sufficiency and not sufficiency[item.obligation_id]:
            item = item.model_copy(
                update={
                    "state": ObligationState.UNKNOWN,
                    "normalized_value": "document_terms_insufficient",
                    "severity": TruthSeverity.NONE,
                    "reason_code": "visible_contract_terms_missing",
                    "evidence": tuple(
                        e
                        for e in item.evidence
                        if e.source_id == case.model_input.contract_document.source_id
                    ),
                }
            )
        duties.append(item)
    values = prior.model_dump(mode="python", exclude={"truth_checksum"})
    values.update(
        truth_contract="v1-07-truth.v1.2",
        obligations=tuple(duties),
        expected_recommendation=_recommendation(
            duties, prior.conflicts, prior.security_events
        ),
    )
    values.update(
        required_evidence=prior.required_evidence,
        conflicts=prior.conflicts,
        security_events=prior.security_events,
    )
    view = TruthRecordV12.model_construct(**values, truth_checksum="0" * 64)
    return TruthRecordV12(
        **values,
        truth_checksum=payload_sha256(
            view.model_dump(mode="json", exclude={"truth_checksum"})
        ),
    )
