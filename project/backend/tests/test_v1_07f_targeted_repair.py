"""Independent DEV controls for F-RE-13; never imports validation documents."""

import json

import pytest

from procurement_review.business_v3_support import supports_field
from procurement_review.eval_f_semantics import matches_value
from v1_07f_support import body_for_payload, response_for, setup


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("The minimum competitive quote count is 4.", True),
        ("Quote count >= 4.", True),
        ("At least 4 competitive quotes are required.", True),
        ("Quote count >= 4.5.", False),
        ("Quote count >= 4/5.", False),
        ("Quote count >= 4e2.", False),
        ("Quote count >= 4 to 5.", False),
        ("Quote count >= 4-5.", False),
        ("Quote count <= 4.", False),
        ("Quote count >= 4 is optional.", False),
        ("Quotes >= 4e2.", False),
        ("Quotes >= 4/5.", False),
        ("Quotes >= 4-5.", False),
    ],
)
def test_quote_sentence_punctuation_is_not_a_fraction(adjudicator, text, expected):
    result = (
        supports_field("policy", "O-02", "quotes", 4, text)
        if adjudicator == "production"
        else matches_value("quotes", 4, text, "policy")
    )
    assert result is expected


@pytest.mark.parametrize("adjudicator", ["production", "independent"])
@pytest.mark.parametrize(
    "text,value,expected",
    [
        (
            "Damages: late shipment costs 0.2% of the invoice amount per week.",
            "late shipment costs 0.2% of the invoice amount per week",
            True,
        ),
        (
            "Damages: late shipment costs 0.2% of the invoice amount per week.",
            "0.2% of the invoice amount per week",
            False,
        ),
        (
            "Damages: late shipment costs 0.2% of the invoice amount per week unless waived.",
            "late shipment costs 0.2% of the invoice amount per week",
            False,
        ),
        (
            "Damages: late shipment costs 0.2% of the invoice amount per week is optional.",
            "late shipment costs 0.2% of the invoice amount per week",
            False,
        ),
    ],
)
def test_damages_preserves_trigger_rate_basis_and_period(
    adjudicator, text, value, expected
):
    result = (
        supports_field("contract", "O-09", "damages", value, text)
        if adjudicator == "production"
        else matches_value("damages", value, text, "contract")
    )
    assert result is expected


@pytest.mark.parametrize(
    "value,code",
    [
        (
            {"applies": "secret-invalid-value", "basis": "2026", "exempt": []},
            "typed_o_01_applies_bool_type",
        ),
        (
            {"applies": True, "basis": "X" * 65, "exempt": []},
            "typed_o_01_basis_string_too_long",
        ),
        (
            {"applies": True, "basis": "2026", "secret-field": []},
            "typed_o_01_normalized_value_extra_forbidden",
        ),
    ],
)
def test_typed_diagnostic_exposes_only_fixed_field_and_error(tmp_path, value, code):
    def handler(req):
        body = body_for_payload(json.loads(req.content))
        body["findings"][0]["normalized_value"] = json.dumps(value)
        return response_for(req, body=body)

    driver, request, _ = setup(tmp_path, handler)
    from procurement_review.eval_f_transport import evidence_payload
    import httpx

    req = httpx.Request(
        "POST",
        "https://example.invalid",
        json=evidence_payload(driver.transport.config, request),
    )
    result = driver.transport.decode(handler(req), request, "offline-test-key")
    assert result.error_code == code
    assert result.correctable and result.format_fields == ("normalized_value",)
    assert "secret" not in repr(result)
