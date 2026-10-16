from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from procurement_review.contracts import HumanReviewStatus, RecommendationStatus
from procurement_review.synthetic import (
    DATASET_VERSION,
    DEFAULT_FIXTURE_PATH,
    DEFAULT_SEED,
    ENVELOPE_SCHEMA_VERSION,
    FIXTURE_SCHEMA_VERSION,
    GENERATOR_VERSION,
    ScenarioKind,
    SyntheticFixtureEnvelope,
    SyntheticFixturePayload,
    check_fixture,
    generate_fixture,
    load_fixture,
    main,
    payload_sha256,
    render_fixture,
)

EXPECTED_V1_DEFAULT_PAYLOAD_SHA256 = (
    "8f224d67342dc6ed45de404a7cd0c2f4c8de4b56f825752491d689cd6c12c6b6"
)


def test_committed_fixture_schema_and_referential_integrity() -> None:
    fixture = load_fixture()
    payload = fixture.payload

    assert fixture.envelope_schema_version == ENVELOPE_SCHEMA_VERSION
    assert payload.fixture_schema_version == FIXTURE_SCHEMA_VERSION
    assert payload.dataset_version == DATASET_VERSION
    assert payload.generator_version == GENERATOR_VERSION
    assert payload.seed == DEFAULT_SEED
    assert fixture.payload_sha256 == payload_sha256(payload)
    assert len(payload.suppliers) == 12
    assert len(payload.purchase_orders) == 24
    assert len(payload.historical_transactions) == 72
    assert len(payload.ground_truth_cases) == 12

    supplier_ids = {item.supplier_id for item in payload.suppliers}
    order_by_id = {item.purchase_order_id: item for item in payload.purchase_orders}
    transaction_by_id = {
        item.transaction_id: item for item in payload.historical_transactions
    }

    assert len(supplier_ids) == len(payload.suppliers)
    assert len(order_by_id) == len(payload.purchase_orders)
    assert len(transaction_by_id) == len(payload.historical_transactions)
    assert all(item.supplier_id in supplier_ids for item in payload.purchase_orders)
    assert all(
        item.purchase_order_id in order_by_id
        and item.supplier_id == order_by_id[item.purchase_order_id].supplier_id
        for item in payload.historical_transactions
    )
    assert all(
        transaction_by_id[transaction_id].purchase_order_id == case.purchase_order_id
        for case in payload.ground_truth_cases
        for transaction_id in case.transaction_ids
    )


def test_ground_truth_covers_safety_scenarios_and_fail_closed_outcomes() -> None:
    payload = load_fixture().payload
    counts = Counter(case.scenario for case in payload.ground_truth_cases)

    assert counts == {
        ScenarioKind.NORMAL: 3,
        ScenarioKind.HIGH_RISK: 3,
        ScenarioKind.MISSING_INFORMATION: 3,
        ScenarioKind.FACT_CONFLICT: 3,
    }

    for case in payload.ground_truth_cases:
        if case.scenario is ScenarioKind.NORMAL:
            assert (
                case.expected_recommendation is RecommendationStatus.RECOMMEND_APPROVE
            )
            assert not case.must_fail_closed
            assert case.expected_review_state is HumanReviewStatus.PENDING_CONFIRMATION
        else:
            assert (
                case.expected_recommendation
                is not RecommendationStatus.RECOMMEND_APPROVE
            )
            assert case.must_fail_closed

        if case.scenario is ScenarioKind.MISSING_INFORMATION:
            assert case.expected_missing_fields
            assert case.expected_review_state is HumanReviewStatus.REQUIRED
        if case.scenario is ScenarioKind.FACT_CONFLICT:
            assert case.expected_conflicts
            assert case.expected_review_state is HumanReviewStatus.REQUIRED


def test_committed_fixture_is_exact_byte_regeneration() -> None:
    actual = DEFAULT_FIXTURE_PATH.read_bytes()
    expected = render_fixture(generate_fixture())

    assert actual == expected
    assert actual.endswith(b"\n")
    assert not actual.endswith(b"\n\n")
    assert b"\r\n" not in actual
    assert check_fixture().payload_sha256 == load_fixture().payload_sha256


def test_version_and_seed_are_deterministic_and_seed_affects_digest() -> None:
    first = generate_fixture(seed=DEFAULT_SEED, dataset_version=DATASET_VERSION)
    second = generate_fixture(seed=DEFAULT_SEED, dataset_version=DATASET_VERSION)
    changed = generate_fixture(seed=DEFAULT_SEED + 1, dataset_version=DATASET_VERSION)

    assert first.payload_sha256 == second.payload_sha256
    assert first.payload_sha256 == EXPECTED_V1_DEFAULT_PAYLOAD_SHA256
    assert render_fixture(first) == render_fixture(second)
    assert changed.payload_sha256 != first.payload_sha256
    assert render_fixture(changed) != render_fixture(first)


@pytest.mark.parametrize(
    ("collection", "index", "field"),
    (
        ("suppliers", 0, "tax_identifier"),
        ("purchase_orders", 0, "contract_reference"),
        ("purchase_orders", 0, "competitive_quote_count"),
        ("historical_transactions", 0, "counterparty_reference"),
    ),
)
def test_normal_case_rejects_missing_critical_evidence(
    collection: str,
    index: int,
    field: str,
) -> None:
    data = generate_fixture().payload.model_dump(mode="json")
    data[collection][index][field] = None

    with pytest.raises(ValidationError, match="source facts do not match.*normal"):
        SyntheticFixturePayload.model_validate(data)


def test_high_risk_case_rejects_relabelled_facts_and_reasons() -> None:
    data = generate_fixture().payload.model_dump(mode="json")
    data["suppliers"][1]["procurement_status"] = "approved"
    data["suppliers"][1]["due_diligence_status"] = "verified"
    data["suppliers"][1]["risk_signals"] = []
    data["ground_truth_cases"][1]["expected_reasons"] = ["baseline_controls_satisfied"]

    with pytest.raises(ValidationError, match="source facts do not match.*high_risk"):
        SyntheticFixturePayload.model_validate(data)


def test_missing_case_requires_exact_missing_fields() -> None:
    data = generate_fixture().payload.model_dump(mode="json")
    data["ground_truth_cases"][2]["expected_missing_fields"] = [
        "supplier.tax_identifier"
    ]

    with pytest.raises(ValidationError, match="ground truth missing fields"):
        SyntheticFixturePayload.model_validate(data)


def test_fact_conflict_rejects_semantically_compatible_supplier_statuses() -> None:
    data = generate_fixture().payload.model_dump(mode="json")
    data["suppliers"][3]["procurement_status"] = "approved"
    data["suppliers"][3]["risk_signals"] = []

    with pytest.raises(
        ValidationError, match="source facts do not match.*fact_conflict"
    ):
        SyntheticFixturePayload.model_validate(data)


def test_fact_conflict_requires_exact_references_values_and_reasons() -> None:
    data = generate_fixture().payload.model_dump(mode="json")
    data["ground_truth_cases"][3]["expected_reasons"] = ["baseline_controls_satisfied"]
    data["ground_truth_cases"][3]["expected_conflicts"][0].update(
        left_reference="unrelated:left",
        left_value="invented",
        right_reference="unrelated:right",
        right_value="invented-too",
    )

    with pytest.raises(ValidationError, match="ground truth reasons"):
        SyntheticFixturePayload.model_validate(data)


def test_fact_conflict_requires_exact_conflict_details() -> None:
    data = generate_fixture().payload.model_dump(mode="json")
    data["ground_truth_cases"][3]["expected_conflicts"][0].update(
        left_reference="unrelated:left",
        left_value="invented",
        right_reference="unrelated:right",
        right_value="invented-too",
    )

    with pytest.raises(ValidationError, match="ground truth conflicts"):
        SyntheticFixturePayload.model_validate(data)


def test_payload_digest_rejects_tampering() -> None:
    data = generate_fixture().model_dump(mode="json")
    data["payload"]["suppliers"][0]["display_name"] = "Synthetic Supplier 99"

    with pytest.raises(ValidationError, match="payload SHA-256 mismatch"):
        SyntheticFixtureEnvelope.model_validate(data)


def test_cli_write_and_check_exact_bytes(tmp_path) -> None:
    fixture_path = tmp_path / "fixture.json"

    assert main(["--write", "--path", str(fixture_path)]) == 0
    assert fixture_path.read_bytes() == render_fixture(generate_fixture())
    assert main(["--check", "--path", str(fixture_path)]) == 0

    fixture_path.write_bytes(fixture_path.read_bytes() + b" ")
    assert main(["--check", "--path", str(fixture_path)]) == 1
