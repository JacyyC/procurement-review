"""Independent DEV controls for operative scope and exclusion wording."""

import pytest
from procurement_review.business_v3_support import supports_field
from procurement_review.eval_f_semantics import matches_value


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("These controls apply to the purchase in 2026.", True),
        ("The purchasing requirements apply to this purchase during 2026.", True),
        ("These controls may apply to the purchase in 2026.", False),
        ("These controls do not apply to the purchase in 2026.", False),
        ("These controls apply to the purchase in 2026 unless waived.", False),
        ("These controls apply to the purchase in 2026 but are optional.", False),
        ("Do these controls apply to the purchase in 2026?", False),
    ],
)
def test_plural_scope_is_binding_only_when_unqualified(adjudicator, text, expected):
    actual = (
        supports_field("policy", "O-01", "applies", True, text)
        if adjudicator == "production"
        else matches_value("applies", True, text, "policy")
    )
    assert actual is expected


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize(
    "text,value,expected",
    [
        (
            "Other applicable terms: payment records, audit rights; no other obligations apply.",
            "payment records, audit rights; no other obligations apply",
            True,
        ),
        (
            "Applicable rights: inspection and return; no other terms apply.",
            "inspection and return; no other terms apply",
            True,
        ),
        (
            "Other applicable terms: payment records, audit rights; no other obligations apply.",
            "payment records, audit rights",
            False,
        ),
        (
            "Other applicable terms: no other obligations apply.",
            "no other obligations apply",
            False,
        ),
        (
            "Other applicable terms: no audit rights; no other obligations apply.",
            "no audit rights; no other obligations apply",
            False,
        ),
        (
            "Other applicable terms: payment records, audit rights; no other obligations apply unless waived.",
            "payment records, audit rights; no other obligations apply",
            False,
        ),
        (
            "Other applicable terms: payment records, audit rights; no other obligations apply. This is optional.",
            "payment records, audit rights; no other obligations apply",
            False,
        ),
        (
            "Notes: payment records, audit rights; no other obligations apply.",
            "payment records, audit rights; no other obligations apply",
            False,
        ),
        (
            "Other applicable terms: payment records, audit rights; no other obligations apply.\nWarranty responsibility: no other duties apply.",
            "payment records, audit rights; no other obligations apply",
            False,
        ),
        (
            "Other applicable terms: unknown; no other obligations apply.",
            "unknown; no other obligations apply",
            False,
        ),
    ],
)
def test_explicit_exclusive_scope_keeps_positive_terms_and_qualifier(
    adjudicator, text, value, expected
):
    actual = (
        supports_field("contract", "O-09", "other", value, text)
        if adjudicator == "production"
        else matches_value("other", value, text, "contract")
    )
    assert actual is expected


def test_exclusion_does_not_license_another_field():
    text = "Warranty responsibility: no repairs; no other obligations apply."
    assert not supports_field("contract", "O-08", "responsibility", "no repairs", text)
    assert not matches_value("responsibility", "no repairs", text, "contract")
