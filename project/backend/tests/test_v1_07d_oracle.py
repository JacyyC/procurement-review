"""Independent, hand-adjudicated business examples; no production-generated truth."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json

import pytest
from pydantic import ValidationError

from procurement_review.business_v3_aggregate import aggregate_business_v3
from procurement_review.business_v3_documents import (
    decode_document_values,
    document_request_v3,
)
from procurement_review.business_v3_procurement import review_procurement_v3
from procurement_review.business_v3_types import BusinessDecisionV3
from procurement_review.model_wire_v2 import parse_findings
from v1_07d_support import (
    context,
    document,
    evaluate,
    mcp_result,
    response_body,
    sources,
    states,
)


# The right column is adjudicated from Spec 5.3, independently of rule functions.
# Each line records why a decision changes. Neither the corpus nor a Case ID is
# consulted. These are boundary examples, not a claim of real-model accuracy.
ORACLE = [
    ({}, "recommend_approve", "All ten duties have evidence."),
    (
        {"order": {"competitive_quote_count": 2}},
        "recommend_conditional",
        "One quotation short; remediable.",
    ),
    (
        {"order": {"competitive_quote_count": None}},
        "insufficient_information",
        "Unknown quotation count is not zero.",
    ),
    (
        {"order": {"invoice_amount_minor": 10011}},
        "recommend_reject",
        "One cent exceeds the agreed amount.",
    ),
    (
        {"order": {"invoice_currency": "USD"}},
        "needs_human_review",
        "Explicit order/invoice currency equality is violated.",
    ),
    (
        {"transactions": [{"currency": "USD"}]},
        "insufficient_information",
        "Transaction cap cannot be calculated without a rate.",
    ),
    (
        {"transactions": [{"amount_minor": 10011}]},
        "recommend_reject",
        "Same-currency transaction total exceeds the order.",
    ),
    (
        {"transactions": [{"amount_minor": 0}]},
        "recommend_approve",
        "Zero is a known amount below the cap.",
    ),
    (
        {"transactions": [{"amount_minor": None}]},
        "insufficient_information",
        "Missing amount cannot enter the sum.",
    ),
    (
        {"transactions": [{"booked_on": "2026-02-27"}]},
        "recommend_reject",
        "Booking precedes the order.",
    ),
    (
        {"transactions": [{"booked_on": "2026-02-28T00:30:00+08:00"}]},
        "recommend_approve",
        "Same local calendar day is allowed.",
    ),
    (
        {"supplier": {"procurement_status": "blocked"}},
        "recommend_reject",
        "Active registration does not cancel a procurement ban.",
    ),
    (
        {"supplier": {"sanctions_screening": "potential_match"}},
        "recommend_reject",
        "The applicable screening condition is explicitly unmet.",
    ),
    (
        {"supplier": {"procurement_status": "unknown"}},
        "insufficient_information",
        "Unknown status is not a proven ban.",
    ),
    (
        {"order": {"contract_reference": {"id": "CONTRACT-A"}}},
        "insufficient_information",
        "An object is not an admitted reference.",
    ),
    (
        {"transactions": [{"counterparty_reference": ["BUYER-A"]}]},
        "insufficient_information",
        "A list is not an admitted reference.",
    ),
    (
        {"changes": {"O-08": {"acceptance": 8}}},
        "recommend_conditional",
        "The contract exceeds the seven-day acceptance limit.",
    ),
    (
        {"unknown": ("O-08",)},
        "insufficient_information",
        "No acceptance standard cannot be treated as compliance.",
    ),
    (
        {"changes": {"O-09": {"written": False}}},
        "recommend_conditional",
        "Written change protection is remediable.",
    ),
    (
        {"unknown": ("O-01",), "supplier": {"procurement_status": "blocked"}},
        "insufficient_information",
        "Without applicability, a rule cannot justify rejection.",
    ),
]


@pytest.mark.parametrize("inputs,expected,rationale", ORACLE)
def test_independent_business_oracle(inputs, expected, rationale):
    result = evaluate(**inputs)
    assert result.recommendation == expected, rationale


@pytest.mark.parametrize("factor", [1, 7, 10000, 900719925])
def test_unit_preserving_scaling_and_transaction_permutation(factor):
    total = 10010 * factor
    price = format(Decimal(total) / 100, ".2f")
    txs = [
        {"transaction_id": "TX-1", "amount_minor": 5010 * factor},
        {"transaction_id": "TX-2", "amount_minor": 5000 * factor},
    ]
    args = dict(
        order={"amount_minor": total, "invoice_amount_minor": total},
        changes={"O-06": {"amount": price}},
    )
    assert evaluate(**args, transactions=txs).recommendation == "recommend_approve"
    assert (
        evaluate(**args, transactions=list(reversed(txs))).recommendation
        == "recommend_approve"
    )
    txs[1]["amount_minor"] += 1
    assert evaluate(**args, transactions=txs).recommendation == "recommend_reject"


def test_partial_exemption_does_not_hide_other_missing_contract_terms():
    result = evaluate(
        changes={"O-01": {"applies": False, "basis": "仅豁免比价", "exempt": ["O-04"]}},
        unknown=("O-09",),
    )
    assert states(result)["O-04"] == "not_applicable"
    assert states(result)["O-09"] == "unknown"
    assert result.recommendation == "insufficient_information"


def test_asof_is_checked_in_case_timezone_and_unknown_version_fails_closed():
    ctx = context()
    result = mcp_result()
    equivalent = datetime(2026, 6, 29, 16, tzinfo=UTC)
    result = result.model_copy(
        update={
            "facts": tuple(
                f.model_copy(update={"as_of": equivalent}) for f in result.facts
            )
        }
    )
    assert not review_procurement_v3(
        ctx, result, authorized_scope_id=ctx.scope_id
    ).failures
    result = result.model_copy(
        update={
            "facts": tuple(
                f.model_copy(update={"as_of": equivalent - timedelta(seconds=1)})
                for f in result.facts
            )
        }
    )
    assert review_procurement_v3(ctx, result, authorized_scope_id=ctx.scope_id).failures


def test_new_decision_roundtrips_generic_business_ids_and_values_are_immutable():
    result = evaluate()
    assert BusinessDecisionV3.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(TypeError):
        document("policy").values["O-01"] = None
    with pytest.raises(ValidationError):
        context().__class__(**(context().model_dump() | {"timezone": "not-a-timezone"}))


@pytest.mark.parametrize(
    "key,value",
    [
        ("special", None),
        ("special", ""),
        ("quotes", 2.5),
        ("supplier", "approved_by_model"),
    ],
)
def test_typed_specialty_and_policy_cannot_be_guessed(key, value):
    request = document_request_v3(
        context(),
        case_id="V107-D-0099",
        worker="policy",
        question="核对专项",
        sources=sources("policy"),
    )
    body = response_body(request, changes={"O-02": {key: value}})
    decoded = decode_document_values(request, parse_findings(json.dumps(body), request))
    assert decoded.failures and "O-02" not in decoded.values


def test_applicability_state_cannot_disagree_with_typed_value():
    request = document_request_v3(
        context(),
        case_id="V107-D-0099",
        worker="policy",
        question="核对",
        sources=sources("policy"),
    )
    body = response_body(request)
    body["findings"][0].update(status="not_applicable", assertion="not_applicable")
    decoded = decode_document_values(request, parse_findings(json.dumps(body), request))
    assert decoded.failures and "O-01" not in decoded.values


def test_semantic_contract_violation_is_not_silently_changed_to_supported():
    request = document_request_v3(
        context(),
        case_id="V107-D-0099",
        worker="contract",
        question="核对争议条款",
        sources=sources("contract"),
    )
    body = response_body(request)
    body["findings"][-1].update(
        status="violated", assertion="non_compliant", risk_level="high"
    )
    contract = decode_document_values(
        request, parse_findings(json.dumps(body), request)
    )
    procurement = review_procurement_v3(
        context(), mcp_result(), authorized_scope_id=context().scope_id
    )
    result = aggregate_business_v3(context(), document("policy"), contract, procurement)
    assert result.recommendation == "recommend_reject"
    assert states(result)["O-09"] == "violated"


def test_renaming_every_entity_and_case_does_not_change_business_outcome():
    ctx = context().model_copy(
        update={
            "purchase_order_id": "NEW-ORDER",
            "supplier_id": "NEW-SELLER",
            "contract_id": "NEW-CONTRACT",
        }
    )
    changes = {
        "O-05": {
            "order": "NEW-ORDER",
            "seller": "NEW-SELLER",
            "contract": "NEW-CONTRACT",
            "buyer": "NEW-BUYER",
        }
    }
    result = mcp_result(
        order={
            "purchase_order_id": "NEW-ORDER",
            "supplier_id": "NEW-SELLER",
            "contract_reference": "NEW-CONTRACT",
        },
        supplier={"supplier_id": "NEW-SELLER"},
        transactions=[
            {
                "purchase_order_id": "NEW-ORDER",
                "supplier_id": "NEW-SELLER",
                "counterparty_reference": "NEW-BUYER",
            }
        ],
    )
    result = result.model_copy(
        update={
            "facts": tuple(
                f.model_copy(
                    update={
                        "record_id": "NEW-ORDER"
                        if f.record_id == "ORDER-A"
                        else "NEW-SELLER"
                        if f.record_id == "SELLER-A"
                        else f.record_id
                    }
                )
                for f in result.facts
            )
        }
    )
    procurement = review_procurement_v3(ctx, result, authorized_scope_id=ctx.scope_id)
    observed = aggregate_business_v3(
        ctx,
        document("policy", ctx=ctx),
        document("contract", ctx=ctx, changes=changes),
        procurement,
    )
    assert observed.recommendation == "recommend_approve"


def test_c_base_source_identity_is_accepted_without_subclass_assumptions():
    from procurement_review.gold_v3_contracts import EvidenceIdentity
    from procurement_review.model_wire_v2 import SourceSlice

    docs = []
    for role in ("policy", "contract"):
        source = sources(role)[0]
        identity = EvidenceIdentity(
            **(source.identity.model_dump() | {"source_id": f"V107-{role.upper()}"})
        )
        source = SourceSlice(
            scope_id=source.scope_id,
            worker=role,
            identity=identity,
            content=source.content,
        )
        request = document_request_v3(
            context(),
            case_id="V107-D-0001",
            worker=role,
            question="核对",
            sources=(source,),
        )
        docs.append(
            decode_document_values(
                request, parse_findings(json.dumps(response_body(request)), request)
            )
        )
    procurement = review_procurement_v3(
        context(), mcp_result(), authorized_scope_id=context().scope_id
    )
    result = aggregate_business_v3(context(), *docs, procurement)
    assert result.recommendation == "recommend_approve"
    assert BusinessDecisionV3.model_validate_json(result.model_dump_json()) == result
