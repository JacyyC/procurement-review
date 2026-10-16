"""Identity references: what the source boundary validates is what the rules compare.

The 2026-10-07 audit found a real false rejection: a contract branch copied the sentence
terminator of ``合同CONTRACT: CONTRACT-907。`` into its value, the grounding accepted it as
present in the cited clause, and the frozen identity comparison then read a different
contract, a different order and a different party at once — a high-risk violation and a
``recommend_reject`` for a Case whose every reference matches.  These cases fix that
boundary: the copied formatting is canonicalized before the source boundary validates it,
and every case that is *not* a formatting difference keeps its verdict.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from procurement_review.business_v31_documents import document_request_v31
from procurement_review.business_v31_identity import canonical_reference
from procurement_review.business_v31_review import ground_document_v31_located
from procurement_review.business_v3_procurement import review_procurement_v3
from procurement_review.business_v3_rules import contract_checks
from procurement_review.business_v3_security import ProtectedOutputV3
from procurement_review.business_v3_support import SourceCatalogV3
from procurement_review.eval_f_context import context_for, located_sources
from procurement_review.gold_v37_conditional import (
    as_gold_case,
    conditional_cases,
    conditional_procurement,
)
from procurement_review.precise_answer_v3 import parse_precise_answer
from test_v1_07g_precise_run import _contract_body

CASE = conditional_cases()[6]  # V107-D-9007: every reference matches the Case identity
IDENTITY_FIELDS = ("buyer", "seller", "order", "contract")


def _branch(context=None):
    """The authored Case's own contract branch, its context and its read-only facts."""

    gold = as_gold_case(CASE)
    located = located_sources(gold)
    context = context or context_for(gold)
    sources = tuple(item.source for item in located if item.source.worker == "contract")
    request = document_request_v31(
        context,
        case_id=CASE.case_id,
        worker="contract",
        question=CASE.question,
        sources=sources,
    )
    return request, SourceCatalogV3(located), context


def _answer(request, *, copy_terminator: bool = False, override: dict | None = None):
    """The mock answer of this branch, optionally copying the sentence's punctuation."""

    source = request.sources[0]
    body = _contract_body(source.content)
    for finding in body["findings"]:
        finding["evidence_refs"] = [source.ref]
    for witness in body["witnesses"]:
        witness["source_ref"] = source.ref
        if witness["obligation_id"] != "O-05":
            continue
        if override and witness["field"] in override:
            witness["value"] = override[witness["field"]]
        elif copy_terminator and witness["field"] in IDENTITY_FIELDS:
            witness["value"] = f"{witness['value']}。"
        finding = next(
            item for item in body["findings"] if item["obligation_id"] == "O-05"
        )
        finding["normalized_value"][witness["field"]] = witness["value"]
    return body


def _parse(request, body):
    return parse_precise_answer(
        json.dumps(
            {"schema_version": "finding-evidence.v3", **body}, ensure_ascii=False
        ),
        request,
        protection=ProtectedOutputV3(),
    )


def _ground(request, catalog, body):
    document, witnesses = _parse(request, body)
    grounded, _locations = ground_document_v31_located(
        request, document, witnesses, catalog
    )
    return grounded


def _o05(context, grounded):
    """The frozen O-05 rule applied to the grounded document and the admitted facts."""

    procurement = review_procurement_v3(
        context, conditional_procurement(CASE), authorized_scope_id=context.scope_id
    )
    policy = SimpleNamespace(values={}, evidence=lambda *keys: ())
    rows = contract_checks(context, procurement, grounded.document, policy)
    return next(row for row in rows if row.obligation_id == "O-05")


def test_the_canonicalizer_removes_formatting_only():
    assert canonical_reference("V107-CONTRACT-D-0001-C。") == "V107-CONTRACT-D-0001-C"
    assert canonical_reference("“V107-PO-D-0001”") == "V107-PO-D-0001"
    assert canonical_reference(" V107-SUP-D-0001 ") == "V107-SUP-D-0001"
    # A suffix, a different number or a different case is content, not formatting.
    assert canonical_reference("V107-CONTRACT-D-0001") != canonical_reference(
        "V107-CONTRACT-D-0001-C"
    )
    assert canonical_reference("V107-CONTRACT-D-0002-C") != canonical_reference(
        "V107-CONTRACT-D-0001-C"
    )
    assert canonical_reference("V107-PO-D-0001。；") == "V107-PO-D-0001"
    # Never emptied into something that matches nothing.
    assert canonical_reference("。。") == "。。"


def test_a_copied_sentence_terminator_is_not_a_different_reference():
    request, catalog, context = _branch()
    body = _answer(request, copy_terminator=True)
    grounded = _ground(request, catalog, body)
    assert "O-05" in grounded.document.values
    values = grounded.document.values["O-05"]
    # The value the source boundary validated is the value the rule compares, and the
    # formatting the model copied is gone from it.
    assert values.contract == context.contract_id
    assert values.order == context.purchase_order_id
    assert values.seller == context.supplier_id
    finding = _o05(context, grounded)
    assert finding.state == "supported"
    assert finding.reason == "contract_identity_complete"


def test_the_grounded_witness_carries_the_canonical_value_it_was_validated_as():
    request, catalog, _context = _branch()
    body = _answer(request, copy_terminator=True)
    grounded = _ground(request, catalog, body)
    spans = {
        span.proposition.field_path.rsplit(".", 1)[-1]: span.value
        for span in grounded.spans
        if span.proposition.check_id == "O-05"
    }
    assert spans["contract"] == CASE.facts["contract_reference"]
    assert spans["order"] == CASE.facts["purchase_order_id"]
    assert spans["seller"] == CASE.facts["supplier_id"]
    assert spans["buyer"] == CASE.facts["counterparty_reference"]


def test_a_genuinely_different_reference_is_still_a_violation():
    request, catalog, context = _branch()
    # The clause states the Case's own contract and the model copied it correctly; the
    # *record* this Run is reviewed against names a different contract.  The comparison
    # must still fail, which is what the canonicalization is not allowed to change.
    other = context.model_copy(update={"contract_id": "V107-CONTRACT-D-0001-X"})
    body = _answer(request, copy_terminator=True)
    grounded = _ground(request, catalog, body)
    finding = _o05(other, grounded)
    assert finding.state == "violated"
    assert finding.reason == "contract_identity_mismatch"
    assert finding.risk == "high"
    assert finding.blocking is True


def test_a_different_suffix_is_content_and_not_formatting():
    request, catalog, context = _branch()
    base = CASE.facts["contract_reference"].removesuffix("-C")
    shorter = context.model_copy(update={"contract_id": base})
    body = _answer(request, copy_terminator=True)
    grounded = _ground(request, catalog, body)
    finding = _o05(shorter, grounded)
    assert finding.state == "violated"
    assert finding.reason == "contract_identity_mismatch"


def test_a_reference_that_is_not_in_the_clause_is_still_refused():
    request, catalog, _context = _branch()
    body = _answer(request, override={"order": "V107-PO-D-9999"})
    grounded = _ground(request, catalog, body)
    # The clause never states that order, so the source boundary refuses the claim: the
    # duty is unresolved rather than matched, exactly as before the fix.
    assert "O-05" not in grounded.document.values
    assert "citation_field_value_unsupported" in grounded.support_failures


def test_a_prose_field_keeps_the_punctuation_it_declared():
    request, catalog, _context = _branch()
    subject = f"{_contract_body(request.sources[0].content)['witnesses'][2]['value']}。"
    body = _answer(request, override={"subject": subject})
    wrapped, witnesses = _parse(request, body)
    declared = [
        item
        for item in witnesses
        if item.obligation_id == "O-05" and item.field == "subject"
    ]
    assert declared and declared[0].value == subject
    grounded = _ground(request, catalog, body)
    spans = {
        span.proposition.field_path.rsplit(".", 1)[-1]: span.value
        for span in grounded.spans
        if span.proposition.check_id == "O-05"
    }
    # Only the identity fields are canonicalized; the prose value stays as extracted.
    assert spans["subject"] == subject
    assert "O-05" in wrapped.values


def test_a_clean_answer_is_left_exactly_as_it_was():
    """The same Case without copied punctuation reads the same as before the fix."""

    request, catalog, context = _branch()
    clean = _answer(request, copy_terminator=False)
    for witness in clean["witnesses"]:
        if witness["obligation_id"] == "O-05" and witness["field"] in IDENTITY_FIELDS:
            assert witness["value"] == canonical_reference(witness["value"])
    grounded = _ground(request, catalog, clean)
    assert grounded.document.values["O-05"].contract == context.contract_id
    finding = _o05(context, grounded)
    assert finding.state == "supported"
    assert finding.reason == "contract_identity_complete"
