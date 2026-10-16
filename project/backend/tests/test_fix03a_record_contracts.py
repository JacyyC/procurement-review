"""FIX-03A V-03A-01 and V-03A-02: recursive value contracts and identity rules.

V-03A-01 (AC-01) drives the closed value rules of every open mapping: an unregistered key,
an object or an array where a scalar is declared, a string count, a boolean count, a
negative count and an unregistered word are all refused at any depth, while the legal
mapping, list and text still pass.  The real Trace and business-fact write entry points are
included, so the rule is exercised where the product actually persists.

V-03A-02 (AC-02) drives the identity half: only a specific identifier/digest/enum slot
earns the registered-format rule, a minted identity that happens to contain a phone-shaped
digit run stays legal in its own slot and is still screened in a text slot, and no
annotation grants a blanket exemption.

Both groups end with a negative control that re-applies the pre-fix guard in-process and
shows the same assertion would fail without the fix.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from procurement_review.contracts import RunEventType
from procurement_review.field_guard import SENSITIVE_VALUE_PATTERNS, canary_values
from procurement_review.record_guard import guard_record
from procurement_review.record_contracts import CheckpointPayload
from procurement_review.run_errors import RecordContractError

from fix03a_support import open_mapping_reversion
from test_fix01b_record_integrity import (
    BOUNDARY_SEED,
    NATIONAL_ID_SHAPED_TRACE,
    PHONE_SHAPED_RUN,
    _valid_samples,
)

SAMPLE_RUN = "run-" + "1" * 32
SAMPLE_TRACE = "trace-" + "2" * 32
POISONED_TEXT = canary_values()[2][1]
PHONE_CANARY = "13800001234"
#: A minted 32-hex Run identity whose digits happen to contain a phone-shaped run (X-8).
MINTED_WITH_PHONE_SHAPE = f"run-{'a'}{'b'}{PHONE_CANARY}{'c' * 19}"
#: The documented boundary: a minted identity that is *not* itself a sensitive shape.
MINTED_BOUNDARY = f"run-{PHONE_CANARY}{'0' * 21}"


def trace_sample(**change: Any) -> dict[str, Any]:
    """One valid Trace event with the given overrides."""

    record = dict(_valid_samples(SAMPLE_RUN, SAMPLE_TRACE)["trace"])
    record["input_summary"] = dict(record["input_summary"])
    record["output_summary"] = dict(record["output_summary"])
    record.update(change)
    return record


def has_sensitive_shape(value: str) -> bool:
    return any(pattern.search(value) for _category, pattern in SENSITIVE_VALUE_PATTERNS)


# ------------------------------------------------------------------ V-03A-01 types


@pytest.mark.parametrize(
    "field,summary",
    [
        # An unregistered key one level down, inside an array element.
        (
            "output_summary",
            {
                "status": [
                    {"customer_notes": {"freeform": "unregistered ordinary text"}}
                ]
            },
        ),
        # An array where the key declares a scalar word.
        ("output_summary", {"status": ["valid"]}),
        # An object where the key declares a scalar word.
        ("output_summary", {"status": {"value": "valid"}}),
        # A string where the key declares a bounded count.
        ("output_summary", {"finding_count": "anything"}),
        # A boolean is not a count, however truthy it is.
        ("output_summary", {"finding_count": True}),
        # A negative and an out-of-range count.
        ("output_summary", {"finding_count": -1}),
        ("output_summary", {"finding_count": 10**9}),
        # A number where the key declares a registered word.
        ("output_summary", {"status": 12345}),
        # A word outside the registered vocabulary.
        ("output_summary", {"status": "validated"}),
        # An identity-shaped value where the key declares a registered word.
        ("output_summary", {"status": PHONE_SHAPED_RUN}),
        # A registered identity array is a list of bounded text, not of lists.
        ("output_summary", {"business_fact_ids": [["get_supplier:SUP-001"]]}),
        ("output_summary", {"business_fact_ids": "get_supplier:SUP-001"}),
        # A registered identity array may not smuggle an object element.
        ("output_summary", {"evidence_ref_ids": [{"id": "chunk-" + "a" * 64}]}),
        # A version stream outside the registered namespace.
        ("source_versions", {"api": "review-api.v2", "notes": "free text"}),
        # An escape sequence does not make an unregistered word legal.
        ("output_summary", {"status": "valid\u2028extra"}),
        ("output_summary", {"reason": "lease expired"}),
    ],
)
def test_open_mapping_values_are_typed_at_every_depth(
    field: str, summary: dict[str, Any]
) -> None:
    """V-03A-01: the value rules of a registered key are enforced, not just its name."""

    with pytest.raises(RecordContractError):
        guard_record("trace", trace_sample(**{field: summary}))


@pytest.mark.parametrize(
    "field,summary",
    [
        ("output_summary", {"status": "valid"}),
        ("output_summary", {"status": "pending_confirmation"}),
        ("output_summary", {"finding_count": 0}),
        (
            "output_summary",
            {"recommendation": "recommend_approve", "highest_risk": "low"},
        ),
        (
            "output_summary",
            {
                "worker": "procurement_data",
                "outcome": "completed",
                "finding_count": 3,
                "evidence_ref_count": 2,
                "business_fact_count": 5,
                "business_fact_ids": [
                    "get_purchase_order:PO-001",
                    "get_supplier:SUP-001",
                ],
                "evidence_ref_ids": [
                    "chunk-" + "a" * 64,
                    "fixture-contract:source-version-1",
                ],
            },
        ),
        (
            "input_summary",
            {
                "case_id": "CASE-LOW-001",
                "case_type": "purchase_order",
                "scenario": "low",
            },
        ),
        ("output_summary", {"failure_code": "timeout", "reason": "lease_expired"}),
        ("input_summary", {"execution_mode": "test_only"}),
        ("output_summary", {"action": "cancel"}),
        (
            "output_summary",
            {"missing_count": 4, "conflict_count": 1, "failure_count": 0},
        ),
        (
            "source_versions",
            {
                "api": "review-api.v2",
                "worker_result": "worker-result.v1",
                "checkpoint": "review-checkpoint.v1",
            },
        ),
    ],
)
def test_a_legal_registered_mapping_still_passes(
    field: str, summary: dict[str, Any]
) -> None:
    """V-03A-01: the closed value rules do not block the shapes the product writes."""

    assert guard_record("trace", trace_sample(**{field: summary})) is not None


def test_a_registered_key_value_is_refused_in_the_wrong_place() -> None:
    """V-03A-01: a legal key inside another key's value is still not legal."""

    for summary in (
        {"case_id": {"status": "valid"}},
        {"status": {"case_id": "CASE-LOW-001"}},
        {"finding_count": {"finding_count": 1}},
    ):
        with pytest.raises(RecordContractError):
            guard_record("trace", trace_sample(output_summary=summary))


def test_the_business_fact_result_is_closed_by_its_tool() -> None:
    """V-03A-01: a real checkpoint payload is judged by the tool's own result shape."""

    from procurement_review.contracts import (
        BusinessFact,
        FactStatus,
        RunEventType as _RunEventType,
    )
    from datetime import UTC, datetime

    fact = BusinessFact(
        tool_name="get_purchase_order",
        record_id="PO-001",
        dataset_version="synthetic-procurement.v1",
        as_of=datetime.now(UTC),
        status=FactStatus.KNOWN,
        result={
            "purchase_order_id": "PO-001",
            "supplier_id": "SUP-001",
            "status": "pending_review",
            "currency": "CNY",
            "amount_minor": 958900,
            "invoice_amount_minor": 958900,
            "invoice_currency": "CNY",
            "contract_reference": "SYN-CONTRACT-001",
            "ordered_on": "2026-03-27",
            "requesting_unit": "Synthetic Operations",
            "competitive_quote_count": 3,
        },
    )
    del _RunEventType
    for poisoned in (
        {**fact.result, "customer_notes": {"freeform": "ordinary text"}},
        {**fact.result, "amount_minor": "958900"},
        {**fact.result, "amount_minor": True},
        {**fact.result, "status": "completed"},
        {**fact.result, "ordered_on": "27/03/2026"},
        {**fact.result, "requesting_unit": POISONED_TEXT},
        {**fact.result, "purchase_order_id": {"id": "PO-001"}},
    ):
        with pytest.raises(RecordContractError):
            guard_record(
                "checkpoint_payload",
                {
                    "input_checksum": "sha256:" + "a" * 64,
                    "contract_checksum": "sha256:" + "b" * 64,
                    "result": {
                        "schema_version": "1.0",
                        "result_version": "worker-result.v1",
                        "worker": "procurement_data",
                        "outcome": "completed",
                        "findings": [
                            {
                                "schema_version": "1.0",
                                "worker": "procurement_data",
                                "finding_key": "procurement.order_status",
                                "category": "procurement_anomaly",
                                "risk_level": "low",
                                "assertion": "risk_absent",
                                "statement": "The order is pending review.",
                                "status": "supported",
                                "evidence_refs": [],
                                "business_fact_refs": [
                                    {
                                        **fact.model_dump(mode="json"),
                                        "result": poisoned,
                                    }
                                ],
                                "missing_information": [],
                                "recommended_action": "Confirm the order.",
                            }
                        ],
                        "failure_code": None,
                        "failure_detail": None,
                    },
                },
                location="checkpoint_payload",
                strict=False,
            )


def test_an_unregistered_business_fact_tool_is_refused() -> None:
    """V-03A-01: a tool with no registered result shape is not a new fact."""

    payload = {
        "input_checksum": "sha256:" + "a" * 64,
        "contract_checksum": "sha256:" + "b" * 64,
        "result": {
            "schema_version": "1.0",
            "result_version": "worker-result.v1",
            "worker": "policy",
            "outcome": "completed",
            "findings": [
                {
                    "schema_version": "1.0",
                    "worker": "policy",
                    "finding_key": "policy.unknown_tool",
                    "category": "policy_compliance",
                    "risk_level": "low",
                    "assertion": "risk_absent",
                    "statement": "A fact from a tool that has no registered shape.",
                    "status": "supported",
                    "evidence_refs": [],
                    "business_fact_refs": [
                        {
                            "tool_name": "get_unregistered_record",
                            "record_id": "X-1",
                            "dataset_version": "synthetic-procurement.v1",
                            "as_of": "2026-06-30T00:00:00+00:00",
                            "status": "known",
                            "result": {"anything": "goes"},
                        }
                    ],
                    "missing_information": [],
                    "recommended_action": "Confirm the order.",
                }
            ],
            "failure_code": None,
            "failure_detail": None,
        },
    }
    # The stored record still passes its own closed shape (a tool name is an identifier)…
    assert CheckpointPayload.model_validate(payload, strict=False) is not None
    # …and the mapping rules refuse it, because no result shape is registered for it.
    with pytest.raises(RecordContractError):
        guard_record(
            "checkpoint_payload", payload, location="checkpoint_payload", strict=False
        )


# ------------------------------------------------------------------ V-03A-02 identity


def test_the_same_value_is_judged_by_the_slot_that_declares_it() -> None:
    """V-03A-02: identity, text and unregistered formats are separated by type."""

    minted = MINTED_WITH_PHONE_SHAPE
    assert len(minted.split("-", 1)[1]) == 32
    assert has_sensitive_shape(minted)
    # In a registered identifier slot the value is trusted by type *and* format.
    run_sample = dict(_valid_samples(minted, NATIONAL_ID_SHAPED_TRACE)["run"])
    assert guard_record("run", run_sample) is not None
    # The very same string in a free-text slot is text, and its digit run is a real shape.
    poisoned = trace_sample(
        error=f"branch failed: {minted}", event_type=RunEventType.STEP_FAILED
    )
    with pytest.raises(RecordContractError):
        guard_record("trace", poisoned)
    # …and in a bounded-text summary slot as well.
    with pytest.raises(RecordContractError):
        guard_record("trace", trace_sample(input_summary={"case_id": minted}))
    # A value that only pretends to be a minted identity is refused even in an identity
    # slot, and so is a registered identity kind in the wrong format.
    for disguised in (
        PHONE_CANARY,
        "run-13800001234",
        "run-110101199003070000",
        "run-13800001234abcdef",
        "trace-13800001234",
    ):
        candidate = dict(_valid_samples(minted, NATIONAL_ID_SHAPED_TRACE)["run"])
        candidate["run_id"] = disguised
        with pytest.raises(RecordContractError):
            guard_record("run", candidate)
    # The documented boundary: a 32-hex minted identity without a sensitive shape is a
    # legal identity in its own slot and legal text elsewhere.
    boundary = MINTED_BOUNDARY
    assert len(boundary.split("-", 1)[1]) == 32
    assert not has_sensitive_shape(boundary)
    assert (
        guard_record("run", _valid_samples(boundary, NATIONAL_ID_SHAPED_TRACE)["run"])
        is not None
    )
    assert guard_record("trace", trace_sample(input_summary={"case_id": boundary}))


def test_the_no_event_trace_identity_must_be_a_registered_trace_format() -> None:
    """V-03A-02: the Run's own Trace identity is an identity slot with a registered form."""

    sample = dict(_valid_samples(PHONE_SHAPED_RUN, NATIONAL_ID_SHAPED_TRACE)["run"])
    assert guard_record("run", sample) is not None
    for candidate in ("trace-" + "f" * 32, NATIONAL_ID_SHAPED_TRACE):
        accepted = {**sample, "trace_id": candidate}
        assert guard_record("run", accepted) is not None
    for candidate in (PHONE_CANARY, "trace-13800001234", "run-" + "3" * 32, "trace-X"):
        with pytest.raises(RecordContractError):
            guard_record("run", {**sample, "trace_id": candidate})


def test_no_annotation_grants_a_blanket_identity_exemption() -> None:
    """V-03A-02: an unknown or ``Any`` slot is screened as text, never exempted."""

    from pydantic import JsonValue

    from procurement_review import record_guard

    assert record_guard._annotation_kind(Any) == "text"
    assert record_guard._annotation_kind(JsonValue) == "text"
    # Every synthetic canary is refused in a free-text slot of a registered record.
    for _category, canary in canary_values():
        sample = trace_sample(
            error=f"branch failed: {canary}", event_type=RunEventType.STEP_FAILED
        )
        with pytest.raises(RecordContractError):
            guard_record("trace", sample)
        summary = trace_sample(output_summary={"status": canary})
        with pytest.raises(RecordContractError):
            guard_record("trace", summary)


def test_the_fixed_seed_boundary_is_decided_by_the_declared_type() -> None:
    """V-03A-02: seed 20260913 reproduces the boundary the type rules decide."""

    rng = random.Random(BOUNDARY_SEED)
    clean = 0
    shaped = 0
    for _ in range(400):
        minted = "run-" + "".join(rng.choice("0123456789abcdef") for _ in range(32))
        assert (
            guard_record("run", _valid_samples(minted, NATIONAL_ID_SHAPED_TRACE)["run"])
            is not None
        )
        record = trace_sample(
            error=f"branch failed: {minted}", event_type=RunEventType.STEP_FAILED
        )
        if has_sensitive_shape(minted):
            shaped += 1
            with pytest.raises(RecordContractError):
                guard_record("trace", record)
        else:
            clean += 1
            guard_record("trace", record)
    assert clean + shaped == 400, (clean, shaped)
    assert clean > shaped, (clean, shaped)


# ------------------------------------------------------------------ negative control


def test_the_pre_fix_open_mapping_would_accept_the_same_values() -> None:
    """V-03A-01 control: with the pre-fix walk restored the assertions above fail."""

    accepted_before: list[str] = []
    cases = {
        "array_object": {"status": [{"customer_notes": {"freeform": "text"}}]},
        "string_count": {"finding_count": "anything"},
        "numeric_status": {"status": 12345},
        "identity_in_status": {"status": PHONE_SHAPED_RUN},
    }
    with open_mapping_reversion():
        for name, summary in cases.items():
            try:
                guard_record("trace", trace_sample(output_summary=summary))
            except RecordContractError:
                continue
            accepted_before.append(name)
    assert sorted(accepted_before) == sorted(cases), accepted_before
    # …and with the fix in place every one of them is refused, which is what the
    # assertions in this module depend on.
    for summary in cases.values():
        with pytest.raises(RecordContractError):
            guard_record("trace", trace_sample(output_summary=summary))


def test_the_pre_fix_guard_exempted_a_minted_identity_in_a_text_slot() -> None:
    """V-03A-02 control: the old ``Any`` exemption is what the new rule removes."""

    minted = MINTED_WITH_PHONE_SHAPE
    assert has_sensitive_shape(minted)
    record = trace_sample(
        error=f"branch failed: {minted}", event_type=RunEventType.STEP_FAILED
    )
    with pytest.raises(RecordContractError):
        guard_record("trace", record)
    with open_mapping_reversion():
        # The reversion also opens the summary mapping, so the same identity reaches the
        # old ``Any`` slot rule through the summary instead of the error text.
        guard_record("trace", trace_sample(output_summary={"status": minted}))
    with pytest.raises(RecordContractError):
        guard_record("trace", trace_sample(output_summary={"status": minted}))
