"""Hand-authored DEV counterexamples, independent of consumed validation."""

import hashlib
import pytest
from procurement_review.business_v3_support import (
    supports_field,
    SourceCatalogV3,
    SupportIssue,
    WitnessV3,
    VerifiedSpanV3,
)
from procurement_review.eval_f_semantics import matches_value, check_span
from procurement_review.eval_f_context import load_cases, context_for, located_sources
from procurement_review.business_v3_documents import document_request_v3
from procurement_review.business_v3_types import (
    BusinessEvidenceIdentity,
    BusinessPropositionIdentity,
)

UNCOMMITTED = [
    ("warranty", 90, "质保至少90天仅为可选方案，不构成强制要求"),
    ("warranty", 90, "质保至少90天的规定已经取消"),
    ("delivery", 15, "DELIVERY at most 15 days is optional"),
    ("warranty", 90, "WARRANTY at least 90 days is a nonbinding target"),
    ("warranty", 90, "WARRANTY at least 90 days takes effect next year"),
]


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize("field,value,text", UNCOMMITTED)
def test_nonbinding_revoked_future_is_not_current_support(
    adjudicator, field, value, text
):
    if adjudicator == "production":
        assert not supports_field("policy", "O-02", field, value, text)
    else:
        assert not matches_value(field, value, text, "policy")


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize(
    "suffix",
    [
        "，但此要求仅为可选方案",
        "；本条款已经取消",
        "。此要求仅供参考",
        "；本条尚未生效",
        "，该要求已废止",
        "；本要求不适用于本订单",
        "; this requirement is not applicable to this order",
        "; unless the supplier agrees otherwise",
        "，除非双方另行约定",
        "; alternatively 30 days",
    ],
)
def test_citation_cannot_cut_away_qualifier(adjudicator, suffix):
    text = "质保至少90天" + suffix
    case = load_cases()[0]
    old = case.model_input.policy_document
    doc = old.model_copy(
        update={
            "content": text,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )
    case = case.model_copy(
        update={
            "model_input": case.model_input.model_copy(update={"policy_document": doc})
        }
    )
    sources = tuple(s for s in located_sources(case) if s.source.worker == "policy")
    req = document_request_v3(
        context_for(case),
        case_id=case.case_id,
        question="核对本采购义务",
        worker="policy",
        sources=tuple(s.source for s in sources),
    )
    witness = WitnessV3(
        obligation_id="O-02",
        field="warranty",
        value=90,
        source_ref=sources[0].source.ref,
        start=0,
        end=len("质保至少90天"),
    )
    if adjudicator == "production":
        with pytest.raises(SupportIssue):
            SourceCatalogV3(sources).verify(req, witness)
        return
    prop = next(c.proposition for c in req.checks if c.proposition.check_id == "O-02")
    span = VerifiedSpanV3(
        evidence=BusinessEvidenceIdentity(**sources[0].source.identity.model_dump()),
        proposition=BusinessPropositionIdentity(
            **(
                prop.model_dump()
                | {"field_path": "policy.o_02.warranty", "source_version": doc.version}
            )
        ),
        value=90,
        start=0,
        end=witness.end,
        span_sha256=hashlib.sha256("质保至少90天".encode()).hexdigest(),
    )
    assert not check_span(case, span)


@pytest.mark.parametrize(
    "text", ["质保至少90天是本订单的强制要求", "WARRANTY at least 90 days is mandatory"]
)
def test_current_binding_control_is_still_supported(text):
    assert supports_field("policy", "O-02", "warranty", 90, text)
    assert matches_value("warranty", 90, text, "policy")
