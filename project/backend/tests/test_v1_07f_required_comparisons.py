"""DEV-only requirement modals preserve a comparison and its numeric domain."""

import pytest
from procurement_review.business_v3_support import supports_field
from procurement_review.eval_f_semantics import matches_value


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Quote count must be at least 4.", True),
        ("Competitive quotes shall be no fewer than 4.", True),
        ("Delivery period must be at most 11 days.", True),
        ("Warranty period shall be at least 73 days.", True),
        ("Quote count must be at most 4.", False),
        ("Competitive quotes shall be no more than 4.", False),
        ("Quote count must be at least 4.5.", False),
        ("Quote count must be at least 4 or 5.", False),
        ("Quote count must be at least 4 unless waived.", False),
        ("Quote count must be at least 4. This requirement is optional.", False),
        ("Quote count need not be at least 4.", False),
        ("Delivery period must be at least 11 days.", False),
        ("Warranty period shall be at most 73 days.", False),
    ],
)
def test_required_comparisons_keep_direction_and_modality(independent, text, expected):
    field, value = (
        ("delivery", 11)
        if text.startswith("Delivery")
        else ("warranty", 73)
        if text.startswith("Warranty")
        else ("quotes", 4)
    )
    duty = "O-02"
    actual = (
        matches_value(field, value, text, "policy")
        if independent
        else supports_field("policy", duty, field, value, text)
    )
    assert actual is expected
