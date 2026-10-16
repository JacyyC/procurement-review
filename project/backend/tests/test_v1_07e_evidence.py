from datetime import timedelta
import json

import pytest

from procurement_review.business_v3_conflicts import conflict_pairs_v3
from procurement_review.business_v3_evidence import (
    parse_evidence_v3,
    ground_document_v3,
)
from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.business_v3_support import (
    SourceCatalogV3,
    SupportIssue,
    supports_field,
)
from procurement_review.business_v3_types import BusinessObservationV3
from procurement_review.model_wire_v2 import WireIssue
from v1_07d_support import context
from v1_07e_support import CONTRACT, located, request, evidence_body, grounded


def observations(spans):
    return tuple(
        BusinessObservationV3(
            proposition=s.proposition, value=s.value, evidence=(s.evidence,)
        )
        for s in spans
    )


def test_all_fields_require_independent_positioned_support_and_roundtrip():
    policy, contract = grounded("policy"), grounded("contract")
    assert len(policy.spans) == 11 and len(contract.spans) == 16
    assert not policy.support_failures and not contract.support_failures
    assert all(
        span.end > span.start and span.span_sha256
        for span in (*policy.spans, *contract.spans)
    )
    assert not conflict_pairs_v3(observations((*policy.spans, *contract.spans)))


@pytest.mark.parametrize(
    "mutation", ["field", "value", "span", "uncited", "object", "time"]
)
def test_legal_source_id_does_not_prove_the_wrong_proposition(mutation):
    entry = located("contract")
    entries = (entry, located("contract", source_id="DOC-other"))
    req = request("contract", entries)
    body = evidence_body(req)
    witness = next(w for w in body["witnesses"] if w["field"] == "warranty")
    if mutation == "field":
        witness["start"] = entry.source.content.index("交付10日")
        witness["end"] = witness["start"] + len("交付10日")
        witness["value"] = 10
    elif mutation == "value":
        witness["value"] = 999
    elif mutation == "span":
        witness["end"] = 99999
    elif mutation == "uncited":
        witness["source_ref"] = entries[1].source.ref
    elif mutation == "object":
        entries = (located("contract", subject_id="OTHER-ORDER"), entries[1])
    else:
        entries = (
            located("contract", effective_at=context().as_of - timedelta(days=1)),
            entries[1],
        )
    doc, witnesses = parse_evidence_v3(
        json.dumps(body), req, protection=ProtectedOutputV3()
    )
    result = ground_document_v3(req, doc, witnesses, SourceCatalogV3(entries))
    assert result.support_failures and "O-08" not in result.document.values


def test_same_source_chunks_and_versions_never_overwrite_each_other():
    entries = (
        located("contract", locator="clause-1"),
        located(
            "contract", CONTRACT.replace("质保120日", "质保30日"), locator="clause-2"
        ),
        located(
            "contract",
            CONTRACT.replace("质保120日", "质保180日"),
            version="document.v2",
            locator="clause-1",
        ),
    )
    catalog = SourceCatalogV3(entries)
    assert len({e.source.ref for e in entries}) == 3
    assert [catalog.resolve(e.source.ref).source.content for e in entries] == [
        e.source.content for e in entries
    ]
    with pytest.raises(SupportIssue, match="source_identity_content_drift"):
        SourceCatalogV3(
            (
                entries[0],
                located(
                    "contract",
                    CONTRACT.replace("质保120日", "质保30日"),
                    locator="clause-1",
                ),
            )
        )


def contradictory_single_finding(*, version="document.v1"):
    entries = (
        located("contract"),
        located(
            "contract",
            CONTRACT.replace("质保120日", "质保30日"),
            source_id="DOC-second",
            version=version,
        ),
    )
    req = request("contract", entries)
    body = evidence_body(req)
    row = next(row for row in body["findings"] if row["obligation_id"] == "O-08")
    row["evidence_refs"].append(entries[1].source.ref)
    body["witnesses"].append(
        dict(
            obligation_id="O-08",
            field="warranty",
            value=30,
            source_ref=entries[1].source.ref,
            start=0,
            end=len(entries[1].source.content),
        )
    )
    doc, witnesses = parse_evidence_v3(
        json.dumps(body), req, protection=ProtectedOutputV3()
    )
    return entries, ground_document_v3(req, doc, witnesses, SourceCatalogV3(entries))


def test_single_finding_can_express_real_contradiction_with_two_sides():
    _, result = contradictory_single_finding()
    assert not result.support_failures
    pairs = conflict_pairs_v3(observations(result.spans))
    assert len(pairs) == 1 and pairs[0].kind == "business_conflict"
    assert {pairs[0].left.value, pairs[0].right.value} == {30, 120}
    assert {
        pairs[0].left.evidence[0].source_id,
        pairs[0].right.evidence[0].source_id,
    } == {"DOC-contract", "DOC-second"}
    reverse = conflict_pairs_v3(tuple(reversed(observations(result.spans))))
    assert reverse == pairs
    assert conflict_pairs_v3(observations(result.spans) * 2) == pairs


def test_different_legal_versions_are_not_forced_into_a_conflict():
    _, result = contradictory_single_finding(version="document.v2")
    assert not result.support_failures
    assert not conflict_pairs_v3(observations(result.spans))


@pytest.mark.parametrize(
    "update",
    [
        {"subject_id": "OTHER"},
        {"scope_id": "OTHER"},
        {"field_path": "contract.o_08.delivery"},
        {"check_id": "O-09"},
        {"effective_at": context().as_of + timedelta(days=1)},
        {"source_version": "different"},
    ],
)
def test_same_key_different_object_matter_time_version_does_not_conflict(update):
    span = next(
        s
        for s in grounded("contract").spans
        if s.proposition.field_path.endswith(".warranty")
    )
    first = observations((span,))[0]
    other = first.model_copy(
        update={"proposition": first.proposition.model_copy(update=update), "value": 30}
    )
    assert not conflict_pairs_v3((first, other))


@pytest.mark.parametrize(
    "unknown", [None, "UNKNOWN", "unknown", "missing", "null", "unresolved"]
)
def test_unknown_is_not_the_opposite_of_a_known_value(unknown):
    row = observations(grounded("contract").spans)[0]
    assert not conflict_pairs_v3((row, row.model_copy(update={"value": unknown})))


@pytest.mark.parametrize(
    "span", ["质保条款已列明", "组织验收", "赔偿上限按条款执行", "责任上限存在"]
)
def test_titles_and_generic_presence_are_not_substantive_terms(span):
    assert not supports_field("contract", "O-08", "standard", span, span)
    assert not supports_field("contract", "O-09", "cap", span, span)


def test_negated_and_affirmative_written_change_claims_need_the_right_clause():
    assert supports_field("contract", "O-09", "written", True, "变更须双方书面签署")
    assert supports_field("contract", "O-09", "written", False, "变更无需书面")
    assert not supports_field("contract", "O-09", "written", True, "变更无需书面")
    assert not supports_field(
        "contract", "O-09", "written", True, "变更须双方书面签署；变更无需书面"
    )


def test_sidecar_cannot_forge_pipeline_authority_or_add_other_worker_fields():
    req = request("policy")
    body = evidence_body(req)
    body["pipeline_failures"] = ["authorization_granted"]
    with pytest.raises(WireIssue, match="evidence_envelope_invalid"):
        parse_evidence_v3(json.dumps(body), req, protection=ProtectedOutputV3())
    del body["pipeline_failures"]
    body["witnesses"][0]["obligation_id"] = "O-08"
    doc, witnesses = parse_evidence_v3(
        json.dumps(body), req, protection=ProtectedOutputV3()
    )
    result = ground_document_v3(
        req, doc, witnesses, SourceCatalogV3((located("policy"),))
    )
    assert result.document.values == {} and result.support_failures == (
        "citation_field_outside_task",
    )


@pytest.mark.parametrize(
    "obligation,field,value,span",
    [
        (
            "O-08",
            "standard",
            "卖方免费修复更换",
            "标准：数量匹配且100%完好，质保责任：卖方免费修复更换。",
        ),
        (
            "O-08",
            "responsibility",
            "数量匹配且100%完好",
            "标准：数量匹配且100%完好，质保责任：卖方免费修复更换。",
        ),
        ("O-09", "cap", "100000.00 CNY", "价款100000.00 CNY。责任上限条款已列明。"),
        ("O-08", "warranty", 120, "错误示例：质保120日"),
        ("O-08", "warranty", 120, "not WARRANTY=120 days"),
    ],
)
def test_right_value_in_wrong_field_or_counterfactual_clause_is_unsupported(
    obligation, field, value, span
):
    assert not supports_field("contract", obligation, field, value, span)


@pytest.mark.parametrize(
    "worker,obligation,field,value,span",
    [
        ("contract", "O-08", "delivery", 12, "DELIVERY=12 days"),
        ("policy", "O-02", "warranty", 180, "WARRANTY>=180 days"),
        (
            "contract",
            "O-05",
            "buyer",
            "CLIENT-NEW",
            "BUYER=CLIENT-NEW; SELLER=SELLER-NEW",
        ),
        ("contract", "O-06", "amount", "875.63", "PRICE=875.63 USD"),
        ("contract", "O-09", "cap", "1234.00 USD", "LIABILITY_LIMIT=1234.00 USD"),
    ],
)
def test_new_language_words_values_and_entities_are_source_derived(
    worker, obligation, field, value, span
):
    assert supports_field(worker, obligation, field, value, span)


@pytest.mark.parametrize(
    "span",
    ["本制度不适用于该采购", "不适用范围包括该采购", "does not apply to this order"],
)
def test_negated_applicability_cannot_support_affirmative_claim(span):
    assert not supports_field("policy", "O-01", "applies", True, span)


@pytest.mark.parametrize(
    "span", ["本制度不适用于该采购", "not applicable to this order"]
)
def test_explicit_negative_applicability_supports_only_negative_claim(span):
    assert supports_field("policy", "O-01", "applies", False, span)
    assert not supports_field("policy", "O-01", "applies", True, span)


def test_mixed_applicability_needs_separate_positioned_clauses():
    span = "适用于采购A；不适用于采购B"
    assert not supports_field("policy", "O-01", "applies", True, span)
    assert not supports_field("policy", "O-01", "applies", False, span)
