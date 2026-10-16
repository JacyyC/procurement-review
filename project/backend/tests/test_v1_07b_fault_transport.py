from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from procurement_review.eval_fault_transport import (
    EXPECTED_LAYER,
    FaultKind,
    FaultLayer,
    FaultOutcome,
    FaultPlanV1,
    ZeroNetworkFaultTransport,
    assert_triggered,
)


def _plan(kind: FaultKind, variant: str | None = None) -> FaultPlanV1:
    values = {
        "task_key": f"fault:{kind.value}:{variant or 'default'}",
        "fault_kind": kind,
        "injection_layer": EXPECTED_LAYER[kind],
        "invalid_variant": variant,
    }
    if kind is FaultKind.VERSION_DRIFT:
        values.update(expected_version="expected-v1", actual_version="drifted-v2")
    return FaultPlanV1(**values)


@pytest.mark.parametrize(
    ("kind", "variant", "error_code"),
    [
        (FaultKind.INVALID_OUTPUT, "bad_json", "invalid_json"),
        (FaultKind.INVALID_OUTPUT, "bad_schema", "invalid_schema"),
        (FaultKind.TOOL_FAILURE, None, "tool_failure"),
        (FaultKind.NO_EVIDENCE, None, "no_evidence"),
        (FaultKind.PERMISSION_DENIED, None, "permission_denied"),
        (FaultKind.CANCELLED, None, "cancelled"),
        (FaultKind.TIMEOUT, None, "timeout"),
        (FaultKind.VERSION_DRIFT, None, "version_drift"),
    ],
)
def test_each_fault_is_actually_triggered_at_its_registered_layer(
    kind: FaultKind, variant: str | None, error_code: str
) -> None:
    transport = ZeroNetworkFaultTransport()
    plan = _plan(kind, variant)
    outcome = asyncio.run(
        transport.execute(task_key=plan.task_key, payload={"case": "T-1"}, plan=plan)
    )
    proof = assert_triggered(outcome, plan)
    assert outcome.error_code == error_code
    assert proof.injection_layer == EXPECTED_LAYER[kind].value
    assert transport.local_attempts == 1
    assert transport.network_calls == 0


@pytest.mark.parametrize(
    "case_failure_mode",
    [
        "model_schema",
        "mcp_failure",
        "no_evidence",
        "version_mismatch",
        "timeout",
    ],
)
def test_case_metadata_never_activates_a_fault(case_failure_mode: str) -> None:
    transport = ZeroNetworkFaultTransport()
    outcome = asyncio.run(
        transport.execute(
            task_key="normal:case",
            payload={"visible": "valid"},
            case_failure_mode=case_failure_mode,
        )
    )
    assert outcome.succeeded is True
    assert outcome.error_code is None
    assert outcome.proof is None
    assert outcome.response is not None
    assert outcome.response["schema_version"] == "test-response.v1"
    assert transport.network_calls == 0


def test_fault_plan_for_another_task_is_a_normal_adjacent_control() -> None:
    transport = ZeroNetworkFaultTransport()
    plan = _plan(FaultKind.TOOL_FAILURE)
    outcome = asyncio.run(
        transport.execute(task_key="normal:other", payload={"ok": True}, plan=plan)
    )
    assert outcome.succeeded is True
    assert outcome.proof is None


def test_wrong_fault_layer_is_rejected_in_plan_and_trigger_proof() -> None:
    with pytest.raises(ValidationError, match="injection layer"):
        FaultPlanV1(
            task_key="fault:wrong-layer",
            fault_kind=FaultKind.TOOL_FAILURE,
            injection_layer=FaultLayer.PARSER,
        )

    plan = _plan(FaultKind.TOOL_FAILURE)
    outcome = asyncio.run(
        ZeroNetworkFaultTransport().execute(
            task_key=plan.task_key, payload={}, plan=plan
        )
    )
    assert outcome.proof is not None
    wrong_proof = outcome.proof.model_copy(update={"injection_layer": "parser"})
    with pytest.raises(AssertionError, match="wrong layer"):
        assert_triggered(
            FaultOutcome(False, outcome.error_code, None, wrong_proof), plan
        )


def test_invalid_output_variant_and_version_drift_require_real_differences() -> None:
    with pytest.raises(ValidationError, match="bad_json/bad_schema"):
        FaultPlanV1(
            task_key="fault:missing-variant",
            fault_kind=FaultKind.INVALID_OUTPUT,
            injection_layer=FaultLayer.PARSER,
        )
    with pytest.raises(ValidationError, match="distinct versions"):
        FaultPlanV1(
            task_key="fault:no-drift",
            fault_kind=FaultKind.VERSION_DRIFT,
            injection_layer=FaultLayer.VERSION,
        )
