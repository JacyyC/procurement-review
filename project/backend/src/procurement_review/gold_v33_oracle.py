"""Independent source sufficiency oracle for v3.3 documents."""

import re
from .gold_v3_contracts import ObligationState, TruthSeverity, payload_sha256
from .gold_v3_oracle import derive_truth as business_truth
from .gold_v31_oracle import source_sufficiency, _recommendation
from .gold_v33_contracts import TruthRecordV13


def sufficient_contract(case):
    if case.language == "zh-CN":
        return source_sufficiency(case)
    text = case.model_input.contract_document.content
    f = case.model_input.business_facts
    identity = all(
        v and v in text
        for v in (
            f.purchase_order_id,
            f.supplier_id,
            f.contract_reference,
            f.counterparty_reference,
        )
    )
    parties = bool(re.search(r"Buyer:.*?seller:", text, re.I | re.S))
    subject = bool(re.search(r"subject:\s*[^.;\n]+", text, re.I))
    # This is a source sufficiency oracle for the frozen synthetic documents,
    # not an open-ended semantic classifier. Require the full operative clause.
    standard = any(
        "Acceptance standard: " + clause + "." in text
        for clause in (
            "all units conform to the signed specification",
            "each unit passes the agreed functional test",
            "every delivered unit is intact and complete",
        )
    )
    warranty = any(
        "Warranty responsibility: " + clause + "." in text
        for clause in (
            "seller replaces defective units without charge",
            "seller repairs faults at its own expense",
            "seller supplies replacement parts free of charge",
        )
    )
    written = "only after both parties sign in writing" in text
    cap = bool(re.search(r"Liability cap:\s*\d+\.\d{2}\s+(?:CNY|USD)", text))
    damages = any(
        "Damages: " + clause + "." in text
        for clause in (
            "late dispatch costs 0.3% of the order amount per day",
            "late supply costs 0.2% of the invoice amount per week",
            "late shipment costs 0.1% of the contract price per day",
        )
    )
    return {
        "O-05": identity and parties and subject,
        "O-08": standard and warranty,
        "O-09": written and damages and cap,
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
        truth_contract="v1-07-truth.v1.3",
        obligations=tuple(duties),
        expected_recommendation=_recommendation(
            duties, prior.conflicts, prior.security_events
        ),
        required_evidence=prior.required_evidence,
        conflicts=prior.conflicts,
        security_events=prior.security_events,
    )
    view = TruthRecordV13.model_construct(**values, truth_checksum="0" * 64)
    return TruthRecordV13(
        **values,
        truth_checksum=payload_sha256(
            view.model_dump(mode="json", exclude={"truth_checksum"})
        ),
    )
