"""H adjudication extends the independent field judge to multiple actual sources."""

import copy

from .eval_f_obligations import (
    judge_span_v3,
    judge_obligation,
    judge_fact_v3,
    identity_key,
)
from .eval_h_material import source_identities
from .gold_v3_contracts import payload_sha256
from .precise_candidate import canonical_json_sha256

DOCUMENT_ROLES = {
    "O-01": {"policy"},
    "O-02": {"policy"},
    "O-03": {"policy"},
    "O-04": {"policy", "contract"},
    "O-05": {"contract"},
    "O-06": {"policy", "contract"},
    "O-07": {"policy"},
    "O-08": {"policy", "contract"},
    "O-09": {"contract"},
}
FACT_FIELDS = {
    "O-03": {
        "supplier." + field
        for field in (
            "registry_status",
            "procurement_status",
            "tax_identifier",
            "due_diligence_status",
            "sanctions_screening",
        )
    },
    "O-04": {
        "purchase_order.competitive_quote_count",
        "purchase_order.contract_reference",
        "historical_transaction.counterparty_reference",
        "purchase_order.transaction_category",
    },
    "O-05": {
        "purchase_order.contract_reference",
        "historical_transaction.counterparty_reference",
    },
    "O-06": {
        "purchase_order." + field
        for field in (
            "amount_minor",
            "currency",
            "invoice_amount_minor",
            "invoice_currency",
        )
    }
    | {"historical_transaction.amount_minor", "historical_transaction.currency"},
    "O-07": {"purchase_order.ordered_on", "historical_transaction.booked_on"},
    "O-10": {
        kind + ".identity"
        for kind in ("purchase_order", "supplier", "historical_transaction")
    },
    "O-11": {"purchase_order.currency", "purchase_order.invoice_currency"},
}


def judge(plan, case, label, row, registration):
    sources = source_identities(plan, case)
    verdicts = []
    extension = registration["truth_extension"][case.case_id]
    for span in row.get("spans") or []:
        local = copy.deepcopy(registration)
        role = span.get("worker")
        document = next(
            (
                document
                for document in case.documents
                if document.role == role and document.source_id == span.get("source_id")
            ),
            None,
        )
        if document:
            local["truth_extension"][case.case_id]["branch_sources"][role] = sources[
                document.source_id
            ]
            local["material_text"][case.case_id][role] = document.content
        verdicts.append(
            judge_span_v3(local, case.case_id, span, local["material_text"])
        )
    findings = row.get("findings") or []
    obligations = []
    for expectation in label.expectations:
        duty = expectation["obligation_id"]
        own = [finding for finding in findings if finding.get("obligation_id") == duty]
        finding = own[0] if len(own) == 1 else None
        required = {
            key: descriptor
            for key, descriptor in extension["support"].get(duty, {}).items()
            if descriptor["requirement"] == "required"
        }
        judged = judge_obligation(
            {**expectation, "values": {}, "contains": {}, "support": required},
            verdicts,
            finding,
        )
        errors = []
        if finding is not None:
            for field in ("risk", "blocking"):
                if field == "risk" and expectation["state"] == "unknown":
                    continue
                if (
                    type(finding.get(field)) is not type(expectation[field])
                    or finding.get(field) != expectation[field]
                ):
                    errors.append("wrong_" + field)
            evidence = finding.get("evidence") or []
            allowed = {
                identity_key(sources[document.source_id])
                for document in case.documents
                if document.role in DOCUMENT_ROLES.get(duty, set())
            }
            allowed.update(
                (fact["source_id"], fact["version"], locator, fact["content_sha256"])
                for locator, fact in extension["facts"].items()
                if locator in FACT_FIELDS.get(duty, set())
            )
            refs = [
                "ref-" + canonical_json_sha256(source)
                for role in ("policy", "contract")
                for source in row.get("model_sources", {}).get(role, [])
            ]
            control = (
                "trusted-review-control",
                "business-review.v3.1",
                "source-and-output-inspection",
                payload_sha256(
                    {"sources": refs, "events": row.get("native_security", [])}
                ),
            )
            seen = []
            for identity in evidence:
                if identity.get("source_id") == "trusted-review-control" and duty in {
                    "O-11",
                    "O-12",
                }:
                    if identity_key(identity) != control:
                        errors.append("control_identity")
                elif identity_key(identity) not in allowed:
                    errors.append("evidence_identity")
                seen.append(identity_key(identity))
            if len(set(seen)) != len(seen):
                errors.append("duplicate_evidence")
            seen_fields = [
                (verdict["worker"], verdict["field"])
                for verdict in verdicts
                if verdict["check_id"] == duty
            ]
            if len(set(seen_fields)) != len(seen_fields):
                errors.append("duplicate_claim")
            if set(expectation.get("fact_locators", [])) - {
                identity["locator"] for identity in evidence
            }:
                errors.append("missing_fact_evidence")
            if finding.get("state") != "unknown":
                for role in DOCUMENT_ROLES.get(duty, set()):
                    if not any(
                        identity["source_id"] in label.required_sources[role]
                        for identity in evidence
                    ):
                        errors.append("missing_document_evidence:" + role)
                if duty == "O-10" and FACT_FIELDS[duty] - {
                    identity["locator"] for identity in evidence
                }:
                    errors.append("missing_identity_evidence")
                if duty in {"O-11", "O-12"} and not evidence:
                    errors.append("missing_control_evidence")
        else:
            errors.append("missing_or_duplicate_finding")
        judged.update(
            correct=judged["correct"] and not errors, evidence_failures=errors
        )
        obligations.append(judged)
    facts = [
        judge_fact_v3(extension, fact) for fact in row.get("fact_observations", [])
    ]
    return dict(
        obligations=obligations,
        spans=verdicts,
        facts=facts,
        propositions=len(verdicts) + len(facts),
        supported=sum(verdict["supported"] for verdict in (*verdicts, *facts)),
        outside=sum(verdict["out_of_scope"] for verdict in (*verdicts, *facts)),
    )
