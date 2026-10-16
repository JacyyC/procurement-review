"""Deterministic rules 3 business aggregation of validated, isolated results."""

import json

from .business_v3_types import (
    BusinessDecisionV3,
    finding,
    BusinessObservationV3,
    BusinessEvidenceIdentity,
    BusinessPropositionIdentity as PropositionIdentity,
)
from .business_v3_rules import (
    supplier_check,
    quote_check,
    amount_check,
    date_check,
    contract_checks,
)


def decide_business_v3(
    findings,
    *,
    pipeline_failures=(),
    security_events=(),
    conflict_pairs=(),
    observations=(),
):
    rows = list(findings)
    ids = [f.obligation_id for f in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate obligation")
    for index in range(1, 11):
        obligation = f"O-{index:02}"
        if obligation not in ids:
            rows.append(
                finding(
                    obligation,
                    "unknown",
                    "required_obligation_absent",
                    missing=("validated_obligation",),
                )
            )
    rows.sort(key=lambda f: f.obligation_id)
    unknown = tuple(f.obligation_id for f in rows if f.state == "unknown" or f.missing)
    violations = tuple(f.obligation_id for f in rows if f.state == "violated")
    blocking = any(f.blocking and f.risk in {"high", "critical"} for f in rows)
    uncertain_authority = any(
        f.obligation_id in {"O-01", "O-10"} and f.state == "unknown" for f in rows
    )
    critical_unknown = any(f.critical and f.state == "unknown" for f in rows)
    if security_events or conflict_pairs:
        recommendation = "needs_human_review"
    elif pipeline_failures or uncertain_authority:
        recommendation = "insufficient_information"
    elif blocking:
        recommendation = "recommend_reject"
    elif critical_unknown:
        recommendation = "insufficient_information"
    elif violations:
        recommendation = (
            "recommend_conditional"
            if all(
                f.remediable and not f.blocking and f.risk in {"low", "medium"}
                for f in rows
                if f.state == "violated"
            )
            else "needs_human_review"
        )
    else:
        recommendation = "recommend_approve"
    return BusinessDecisionV3(
        recommendation=recommendation,
        findings=tuple(rows),
        violations=violations,
        unknown_obligations=unknown,
        pipeline_failures=tuple(sorted(set(pipeline_failures))),
        security_events=tuple(sorted(set(security_events))),
        conflict_pairs=tuple(sorted(set(conflict_pairs))),
        observations=tuple(observations),
    )


def business_observations(context, procurement, documents):
    rows = []
    for kind, facts in (
        ("purchase_order", (procurement.order,)),
        ("supplier", (procurement.supplier,)),
        ("transaction", procurement.transactions),
    ):
        for fact in facts:
            if fact is None:
                continue
            for field in (
                "currency",
                "invoice_currency",
                "amount_minor",
                "invoice_amount_minor",
                "ordered_on",
                "booked_on",
                "registry_status",
                "procurement_status",
            ):
                if field not in fact.result:
                    continue
                from .business_v3_procurement import field_evidence

                value = fact.result[field]
                if type(value) not in {str, int, bool, type(None)}:
                    value = None
                check = (
                    "O-06"
                    if "amount" in field or "currency" in field
                    else "O-07"
                    if field.endswith("_on")
                    else "O-03"
                )
                rows.append(
                    BusinessObservationV3(
                        proposition=PropositionIdentity(
                            scope_id=context.scope_id,
                            subject_type=kind,
                            subject_id=fact.record_id,
                            check_id=check,
                            field_path=f"{kind}.{field}",
                            effective_at=context.as_of,
                            source_version=fact.dataset_version,
                        ),
                        value=value,
                        evidence=(field_evidence(fact, field),),
                    )
                )
    for document in documents:
        for observed in document.findings:
            value = document.values.get(observed.finding.obligation_id)
            if value is None:
                continue
            for field, raw in value.model_dump().items():
                if isinstance(raw, tuple):
                    raw = json.dumps(raw, separators=(",", ":"))
                for evidence in observed.evidence:
                    rows.append(
                        BusinessObservationV3(
                            proposition=PropositionIdentity(
                                **(
                                    observed.proposition.model_dump()
                                    | {
                                        "field_path": f"{document.worker}.{observed.finding.obligation_id.lower().replace('-', '_')}.{field}",
                                        "source_version": evidence.version,
                                    }
                                )
                            ),
                            value=raw,
                            evidence=(
                                BusinessEvidenceIdentity(**evidence.model_dump()),
                            ),
                        )
                    )
    return tuple(rows)


def aggregate_business_v3(
    context,
    policy,
    contract,
    procurement,
    *,
    pipeline_failures=(),
    security_events=(),
    conflict_pairs=(),
):
    if policy.worker != "policy" or contract.worker != "contract":
        raise ValueError("document role mismatch")
    scope = policy.values.get("O-01")
    limits = policy.values.get("O-02")
    rows = [procurement.source_finding]
    if scope is None:
        rows.append(
            finding(
                "O-01",
                "unknown",
                "policy_applicability_missing",
                policy.evidence("O-01"),
                missing=("applicability_basis",),
            )
        )
    else:
        rows.append(
            finding(
                "O-01",
                "supported" if scope.applies else "not_applicable",
                "policy_applicable" if scope.applies else "explicit_policy_exemption",
                policy.evidence("O-01"),
            )
        )
    rows.append(
        finding(
            "O-02",
            "supported" if limits else "unknown",
            "policy_constraints_explicit" if limits else "policy_constraints_missing",
            policy.evidence("O-02"),
            missing=() if limits else ("policy_constraints",),
        )
    )
    rows.extend(
        (
            supplier_check(procurement, policy),
            quote_check(procurement, policy),
            amount_check(procurement, contract, policy),
            date_check(context, procurement, policy),
        )
    )
    rows.extend(contract_checks(context, procurement, contract, policy))
    if scope is not None and not scope.applies:
        rows = [
            finding(
                row.obligation_id,
                "not_applicable",
                "explicit_policy_exemption",
                policy.evidence("O-01"),
            )
            if row.obligation_id in scope.exempt
            else row
            for row in rows
        ]
    # Text-only violations are model semantic judgments, not manufactured from
    # status names. Preserve their evidence and risk alongside pure comparisons.
    for document in (policy, contract):
        for observed in document.findings:
            item = observed.finding
            if (
                item.obligation_id in document.values
                and item.status == "violated"
                and item.obligation_id in {"O-02", "O-05", "O-08", "O-09"}
            ):
                index = next(
                    i
                    for i, row in enumerate(rows)
                    if row.obligation_id == item.obligation_id
                )
                if rows[index].state == "supported":
                    rows[index] = finding(
                        item.obligation_id,
                        "violated",
                        "document_semantic_violation",
                        observed.evidence,
                        risk=item.risk_level,
                        blocking=item.risk_level in {"high", "critical"},
                        remediable=item.risk_level in {"low", "medium"},
                    )
    return decide_business_v3(
        rows,
        pipeline_failures=(
            *pipeline_failures,
            *policy.failures,
            *contract.failures,
            *procurement.failures,
        ),
        security_events=security_events,
        conflict_pairs=conflict_pairs,
        observations=business_observations(context, procurement, (policy, contract)),
    )
