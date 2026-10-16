"""Independent, hand-written DEV language controls; no validation fixture imports."""

import pytest

from procurement_review.business_v3_support import supports_field
from procurement_review.eval_f_semantics import matches_value


POSITIVE = [
    (
        "policy",
        "O-02",
        "special",
        "delivery<=21;acceptance<=6;warranty>=180",
        "交付不超过 21 日、验收不超过 6 日、质保不少于 180 日",
    ),
    (
        "contract",
        "O-09",
        "written",
        True,
        "Changes become valid only after both parties sign in writing.",
    ),
    (
        "policy",
        "O-02",
        "supplier",
        "all_qualified",
        "Registry status must be active; procurement approved; due diligence verified; screening clear; registered tax.",
    ),
    ("policy", "O-02", "warranty", 180, "Warranty must last at least 180 days."),
    ("policy", "O-02", "delivery", 21, "Delivery shall take no more than 21 days."),
    ("policy", "O-02", "quotes", 4, "A minimum of 4 competitive quotes is required."),
    ("policy", "O-02", "acceptance", 6, "Acceptance is required within 6 days."),
    ("contract", "O-08", "warranty", 180, "Warranty: 180 days."),
    (
        "policy",
        "O-02",
        "special",
        "delivery<=21;acceptance<=6;warranty>=180",
        "Delivery shall take no more than 21 days; acceptance within 6 days; warranty at least 180 days.",
    ),
    (
        "contract",
        "O-08",
        "standard",
        "all units pass the agreed inspection",
        "Acceptance standard: all units pass the agreed inspection.",
    ),
    ("contract", "O-09", "cap", "50000.00 USD", "Liability cap: 50000.00 USD."),
]

NEGATIVE = [
    (
        "policy",
        "O-02",
        "special",
        "delivery<=21;acceptance<=6;warranty>=180",
        "交付不少于 21 日、验收不超过 6 日、质保不少于 180 日",
    ),
    (
        "contract",
        "O-09",
        "written",
        True,
        "Changes become valid before both parties sign in writing.",
    ),
    (
        "policy",
        "O-02",
        "supplier",
        "all_qualified",
        "Registry denies active status; procurement approved; due diligence verified; screening clear; registered tax.",
    ),
    ("policy", "O-02", "warranty", 180, "Warranty at most 180 days."),
    ("policy", "O-02", "delivery", 21, "Delivery at least 21 days."),
    ("policy", "O-02", "delivery", 21, "Delivery at most 21.5 days."),
    ("policy", "O-02", "delivery", 21, "Delivery at most 21 to 30 days."),
    ("policy", "O-02", "delivery", 21, "Delivery at most 21 days is not applicable."),
    (
        "policy",
        "O-02",
        "delivery",
        21,
        "Delivery at most 21 days is no longer binding.",
    ),
    ("policy", "O-02", "warranty", 180, "No warranty at least 180 days is required."),
    ("policy", "O-02", "warranty", 180, "Warranty at least 180 days is optional."),
    (
        "policy",
        "O-02",
        "special",
        "delivery<=21;acceptance<=6;warranty>=180",
        "Delivery at most 21 days; acceptance at most 6 days; warranty at most 180 days.",
    ),
    (
        "contract",
        "O-08",
        "standard",
        "all units pass",
        "No acceptance standard: all units pass.",
    ),
    (
        "contract",
        "O-09",
        "cap",
        "50000.00 USD",
        "Liability cap: 50000.00 USD is a hypothetical example.",
    ),
]


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize("worker,duty,field,value,text", POSITIVE)
def test_explicit_unfamiliar_binding_clause(
    adjudicator, worker, duty, field, value, text
):
    result = (
        supports_field(worker, duty, field, value, text)
        if adjudicator == "production"
        else matches_value(field, value, text, worker)
    )
    assert result


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize("worker,duty,field,value,text", NEGATIVE)
def test_ambiguous_opposite_and_nonbinding_clause(
    adjudicator, worker, duty, field, value, text
):
    result = (
        supports_field(worker, duty, field, value, text)
        if adjudicator == "production"
        else matches_value(field, value, text, worker)
    )
    assert not result


@pytest.mark.parametrize(
    "worker,duty,field,value,text",
    [
        ("policy", "O-01", "delivery", 21, "Delivery at most 21 days."),
        ("contract", "O-09", "warranty", 180, "Warranty: 180 days."),
        ("contract", "O-08", "cap", "50000.00 USD", "Liability cap: 50000.00 USD."),
    ],
)
def test_supported_words_cannot_change_field_owner(worker, duty, field, value, text):
    assert not supports_field(worker, duty, field, value, text)
