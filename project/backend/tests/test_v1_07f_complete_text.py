"""Complete semantic text, not merely a literal prefix, is required as a value."""

import pytest
from procurement_review.business_v3_support import supports_field
from procurement_review.eval_f_semantics import matches_value

CASES = (
    (
        "O-08",
        "standard",
        "every device meets the signed accuracy specification",
        "every device",
        "Acceptance standard: every device meets the signed accuracy specification.",
    ),
    (
        "O-08",
        "responsibility",
        "seller repairs defects at its own expense",
        "seller repairs defects",
        "Warranty responsibility: seller repairs defects at its own expense.",
    ),
    (
        "O-09",
        "damages",
        "late shipment costs 0.2% of the invoice amount per week",
        "late shipment costs 0.2%",
        "Damages: late shipment costs 0.2% of the invoice amount per week.",
    ),
    ("O-09", "cap", "95000.00 CNY", "95000.00", "Liability cap: 95000.00 CNY."),
    (
        "O-08",
        "responsibility",
        "seller repairs defects; seller pays all costs",
        "seller repairs defects",
        "Warranty responsibility: seller repairs defects; seller pays all costs.",
    ),
    (
        "O-08",
        "standard",
        "全部设备符合约定精度",
        "全部设备",
        "验收标准：全部设备符合约定精度。",
    ),
    (
        "O-08",
        "responsibility",
        "卖方免费更换故障部件",
        "卖方免费",
        "质保责任：卖方免费更换故障部件。",
    ),
    (
        "O-09",
        "damages",
        "逾期按订单金额每日赔付0.2%",
        "逾期按订单金额",
        "赔偿：逾期按订单金额每日赔付0.2%。",
    ),
)


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("duty,field,complete,truncated,source", CASES)
def test_value_must_retain_the_entire_substantive_clause(
    independent, duty, field, complete, truncated, source
):
    def check(value):
        return (
            matches_value(field, value, source, "contract")
            if independent
            else supports_field("contract", duty, field, value, source)
        )

    assert check(complete)
    assert not check(truncated)


@pytest.mark.parametrize("independent", [False, True])
def test_complete_clause_before_another_label_is_not_forced_to_absorb_it(independent):
    source = "Acceptance standard: every unit passes calibration; Warranty responsibility: seller replaces faulty parts."
    value = "every unit passes calibration"
    assert (
        matches_value("standard", value, source, "contract")
        if independent
        else supports_field("contract", "O-08", "standard", value, source)
    )


@pytest.mark.parametrize("independent", [False, True])
def test_unlabelled_duplicate_cannot_complete_a_truncated_labelled_value(independent):
    source = (
        "Acceptance standard: every device meets the required test.\nNote: every device"
    )
    value = "every device"
    actual = (
        matches_value("standard", value, source, "contract")
        if independent
        else supports_field("contract", "O-08", "standard", value, source)
    )
    assert not actual


def test_clipped_witness_cannot_hide_a_cost_clause_after_semicolon():
    import hashlib
    from procurement_review.eval_f_context import (
        load_cases,
        context_for,
        located_sources,
    )
    from procurement_review.business_v3_documents import document_request_v3
    from procurement_review.business_v3_support import (
        SourceCatalogV3,
        SupportIssue,
        WitnessV3,
        VerifiedSpanV3,
    )
    from procurement_review.eval_f_semantics import check_span

    c = load_cases()[0]
    content = "Warranty responsibility: seller repairs defects; seller bears all costs."
    document = c.model_input.contract_document.model_copy(
        update={
            "content": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    )
    c = c.model_copy(
        update={
            "model_input": c.model_input.model_copy(
                update={"contract_document": document}
            )
        }
    )
    sources = located_sources(c)
    req = document_request_v3(
        context_for(c),
        case_id=c.case_id,
        worker="contract",
        question=c.model_input.question,
        sources=(sources[1].source,),
    )
    w = WitnessV3(
        obligation_id="O-08",
        field="responsibility",
        value="seller repairs defects",
        source_ref=sources[1].source.ref,
        start=0,
        end=content.index(";"),
    )
    with pytest.raises(SupportIssue):
        SourceCatalogV3(sources).verify(req, w)
    slot = next(x.proposition for x in req.checks if x.proposition.check_id == "O-08")
    span = VerifiedSpanV3(
        evidence=sources[1].source.identity,
        proposition=slot.model_copy(
            update={
                "field_path": "contract.o_08.responsibility",
                "source_version": document.version,
            }
        ),
        value=w.value,
        start=w.start,
        end=w.end,
        span_sha256=hashlib.sha256(content[: w.end].encode()).hexdigest(),
    )
    assert not check_span(c, span)
