"""Hand-authored DEV counterexamples: a duration integer is measured in days."""

import pytest

from procurement_review.business_v3_support import supports_field
from procurement_review.eval_f_semantics import matches_value


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize("worker", ["policy", "contract"])
@pytest.mark.parametrize("field", ["delivery", "acceptance", "warranty"])
@pytest.mark.parametrize(
    "quantity",
    [
        "21 months",
        "21 weeks",
        "21 hours",
        "21 years",
        "21",
        "-21 days",
        "21/2 days",
        "21e2 days",
        "21 daydreams",
    ],
)
def test_duration_needs_explicit_day_units(adjudicator, worker, field, quantity):
    relation = (
        ("at least" if field == "warranty" else "at most")
        if worker == "policy"
        else ":"
    )
    source = (
        f"{field}: {quantity}."
        if worker == "contract"
        else f"{field} {relation} {quantity}."
    )
    result = (
        supports_field(
            worker, "O-02" if worker == "policy" else "O-08", field, 21, source
        )
        if adjudicator == "production"
        else matches_value(field, 21, source, worker)
    )
    assert not result


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize("worker", ["policy", "contract"])
@pytest.mark.parametrize("field", ["delivery", "acceptance", "warranty"])
def test_explicit_days_with_other_units_in_another_clause(adjudicator, worker, field):
    relation = (
        ("at least" if field == "warranty" else "at most")
        if worker == "policy"
        else ":"
    )
    source = f"{field} {relation} 21 days; reporting interval: 2 months."
    result = (
        supports_field(
            worker, "O-02" if worker == "policy" else "O-08", field, 21, source
        )
        if adjudicator == "production"
        else matches_value(field, 21, source, worker)
    )
    assert result


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize(
    "source", ["交付不超过21个月", "交付不超过21周", "交付不超过21小时", "交付不超过21"]
)
def test_chinese_duration_cannot_lose_its_unit(adjudicator, source):
    result = (
        supports_field("policy", "O-02", "delivery", 21, source)
        if adjudicator == "production"
        else matches_value("delivery", 21, source, "policy")
    )
    assert not result


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize(
    "source",
    [
        "Delivery: 21 days or 30 days.",
        "Delivery: 21 days unless extended.",
        "Delivery: 21 days except this order.",
        "Delivery: 21 days, alternatively 30 days.",
        "Delivery: 21 days subject to supplier confirmation.",
        "交付21天或30天。",
        "交付21天，除非供应商另行确认。",
    ],
)
def test_alternative_and_conditional_durations_are_not_exact(adjudicator, source):
    result = (
        supports_field("contract", "O-08", "delivery", 21, source)
        if adjudicator == "production"
        else matches_value("delivery", 21, source, "contract")
    )
    assert not result
