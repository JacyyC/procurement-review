"""Zero-network transport that truly triggers the seven V1-07B fault classes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .eval_v2_contracts import FaultTriggerProof
from .gold_v3_contracts import payload_sha256

FAULT_PLAN_VERSION = "eval-fault-plan.v1"


class FaultKind(StrEnum):
    INVALID_OUTPUT = "invalid_output"
    TOOL_FAILURE = "tool_failure"
    NO_EVIDENCE = "no_evidence"
    PERMISSION_DENIED = "permission_denied"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    VERSION_DRIFT = "version_drift"


class FaultLayer(StrEnum):
    PARSER = "parser"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    AUTHORIZATION = "authorization"
    LIFECYCLE = "lifecycle"
    TRANSPORT = "transport"
    VERSION = "version"


EXPECTED_LAYER = {
    FaultKind.INVALID_OUTPUT: FaultLayer.PARSER,
    FaultKind.TOOL_FAILURE: FaultLayer.TOOL,
    FaultKind.NO_EVIDENCE: FaultLayer.RETRIEVAL,
    FaultKind.PERMISSION_DENIED: FaultLayer.AUTHORIZATION,
    FaultKind.CANCELLED: FaultLayer.LIFECYCLE,
    FaultKind.TIMEOUT: FaultLayer.TRANSPORT,
    FaultKind.VERSION_DRIFT: FaultLayer.VERSION,
}


class FaultPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["eval-fault-plan.v1"] = FAULT_PLAN_VERSION
    task_key: Annotated[str, Field(min_length=3, max_length=192)]
    fault_kind: FaultKind
    injection_layer: FaultLayer
    invalid_variant: Literal["bad_json", "bad_schema"] | None = None
    timeout_seconds: Annotated[float, Field(gt=0, le=1.0)] = 0.01
    expected_version: Annotated[str, Field(min_length=1, max_length=96)] = "v1"
    actual_version: Annotated[str, Field(min_length=1, max_length=96)] = "v1"

    @model_validator(mode="after")
    def layer_and_variant_match(self) -> Self:
        if self.injection_layer is not EXPECTED_LAYER[self.fault_kind]:
            raise ValueError("fault plan injection layer does not match fault kind")
        if self.fault_kind is FaultKind.INVALID_OUTPUT:
            if self.invalid_variant is None:
                raise ValueError(
                    "invalid output plan needs a bad_json/bad_schema variant"
                )
        elif self.invalid_variant is not None:
            raise ValueError("only invalid output uses invalid_variant")
        if self.fault_kind is FaultKind.VERSION_DRIFT:
            if self.expected_version == self.actual_version:
                raise ValueError("version drift plan needs distinct versions")
        elif self.expected_version != self.actual_version:
            raise ValueError("only version drift may change the actual version")
        return self


@dataclass(frozen=True, slots=True)
class FaultOutcome:
    succeeded: bool
    error_code: str | None
    response: dict[str, Any] | None
    proof: FaultTriggerProof | None


class _InjectedFault(RuntimeError):
    def __init__(self, code: str, proof: FaultTriggerProof) -> None:
        super().__init__(code)
        self.code = code
        self.proof = proof


class ZeroNetworkFaultTransport:
    """An in-process transport; ``network_calls`` remains zero by construction."""

    def __init__(self) -> None:
        self.network_calls = 0
        self.local_attempts = 0

    @staticmethod
    def _proof(plan: FaultPlanV1, detail: str) -> FaultTriggerProof:
        identity = {
            "task_key": plan.task_key,
            "fault_kind": plan.fault_kind.value,
            "injection_layer": plan.injection_layer.value,
            "detail_code": detail,
        }
        return FaultTriggerProof(
            fault_kind=plan.fault_kind.value,
            injection_layer=plan.injection_layer.value,
            trigger_id=payload_sha256(identity),
            detail_code=detail,
        )

    @staticmethod
    def _parse_response(raw: bytes, plan: FaultPlanV1 | None) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if plan is None:
                raise RuntimeError(
                    "unexpected local transport parse failure"
                ) from error
            raise _InjectedFault(
                "invalid_json",
                ZeroNetworkFaultTransport._proof(plan, "invalid_json_observed"),
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "test-response.v1"
        ):
            if plan is None:
                raise RuntimeError("unexpected local transport schema failure")
            raise _InjectedFault(
                "invalid_schema",
                ZeroNetworkFaultTransport._proof(plan, "invalid_schema_observed"),
            )
        return payload

    async def _timeout(self, plan: FaultPlanV1) -> None:
        proof = self._proof(plan, "transport_timeout_started")

        async def delayed() -> None:
            await asyncio.sleep(plan.timeout_seconds * 10)

        try:
            await asyncio.wait_for(delayed(), timeout=plan.timeout_seconds)
        except TimeoutError as error:
            raise _InjectedFault("timeout", proof) from error

    async def _cancel(self, plan: FaultPlanV1) -> None:
        reached = asyncio.Event()
        proof = self._proof(plan, "task_cancelled_after_start")

        async def waiting() -> None:
            reached.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(waiting())
        await reached.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as error:
            raise _InjectedFault("cancelled", proof) from error

    async def execute(
        self,
        *,
        task_key: str,
        payload: dict[str, Any],
        plan: FaultPlanV1 | None = None,
        case_failure_mode: str | None = None,
    ) -> FaultOutcome:
        """Execute one local attempt; case metadata never activates a fault."""

        del case_failure_mode
        self.local_attempts += 1
        active = plan if plan is not None and plan.task_key == task_key else None
        try:
            if active is not None and active.fault_kind is FaultKind.PERMISSION_DENIED:
                raise _InjectedFault(
                    "permission_denied",
                    self._proof(active, "authorization_denied_before_transport"),
                )
            if active is not None and active.fault_kind is FaultKind.TOOL_FAILURE:
                raise _InjectedFault(
                    "tool_failure",
                    self._proof(active, "tool_exception_observed"),
                )
            if active is not None and active.fault_kind is FaultKind.NO_EVIDENCE:
                evidence: list[dict[str, Any]] = []
                if not evidence:
                    raise _InjectedFault(
                        "no_evidence",
                        self._proof(active, "retrieval_returned_no_evidence"),
                    )
            if active is not None and active.fault_kind is FaultKind.VERSION_DRIFT:
                if active.actual_version != active.expected_version:
                    raise _InjectedFault(
                        "version_drift",
                        self._proof(active, "version_mismatch_observed"),
                    )
            if active is not None and active.fault_kind is FaultKind.CANCELLED:
                await self._cancel(active)
            if active is not None and active.fault_kind is FaultKind.TIMEOUT:
                await self._timeout(active)

            if active is not None and active.fault_kind is FaultKind.INVALID_OUTPUT:
                raw = (
                    b'{"schema_version":'
                    if active.invalid_variant == "bad_json"
                    else b'{"schema_version":"wrong.v1","result":{}}'
                )
            else:
                raw = json.dumps(
                    {
                        "schema_version": "test-response.v1",
                        "result": {
                            "status": "ok",
                            "payload_sha256": payload_sha256(payload),
                        },
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            response = self._parse_response(raw, active)
            return FaultOutcome(True, None, response, None)
        except _InjectedFault as error:
            return FaultOutcome(False, error.code, None, error.proof)


def assert_triggered(outcome: FaultOutcome, plan: FaultPlanV1) -> FaultTriggerProof:
    if outcome.succeeded or outcome.proof is None:
        raise AssertionError("fault plan did not produce a trigger proof")
    proof = outcome.proof
    if proof.fault_kind != plan.fault_kind.value:
        raise AssertionError("fault proof kind differs from plan")
    if proof.injection_layer != plan.injection_layer.value:
        raise AssertionError("fault occurred at the wrong layer")
    if not proof.observed:
        raise AssertionError("fault proof was not observed")
    return proof


__all__ = [
    "EXPECTED_LAYER",
    "FAULT_PLAN_VERSION",
    "FaultKind",
    "FaultLayer",
    "FaultOutcome",
    "FaultPlanV1",
    "ZeroNetworkFaultTransport",
    "assert_triggered",
]
