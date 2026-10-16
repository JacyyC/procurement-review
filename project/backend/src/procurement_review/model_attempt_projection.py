"""Bounded branch observations, never a fabricated Case recommendation."""

from typing import Literal

from pydantic import Field

from .eval_v2_contracts import EvalV2Model, SemanticClaimProjection
from .gold_v3_contracts import CaseId, ObligationId
from .model_wire_v2 import Worker


class ClaimDetail(EvalV2Model):
    obligation_id: ObligationId
    risk_level: Literal["none", "low", "medium", "high", "critical", "unknown"]
    missing_information: tuple[str, ...] = Field(max_length=4)
    # Optional only to read earlier receipts; new live projections retain bounded
    # safe conclusions so semantic review does not rely on hashes alone.
    statement: str | None = Field(default=None, min_length=1, max_length=256)
    recommended_action: str | None = Field(default=None, min_length=1, max_length=256)


class BranchProjectionV2(EvalV2Model):
    schema_version: Literal["eval-projection.v2"] = "eval-projection.v2"
    projection_kind: Literal["model_branch"] = "model_branch"
    case_id: CaseId
    worker: Worker
    claims: tuple[SemanticClaimProjection, ...] = Field(max_length=12)
    details: tuple[ClaimDetail, ...] = Field(max_length=12)
    first_pass_valid: bool
    received_nonempty_body: bool
    corrected_valid: bool
    pipeline_failure: bool
    model_request_count: int = Field(ge=1, le=2)
    latency_ms: int = Field(ge=0)


def load_projection(body: str):
    import json
    from .eval_v2_contracts import EvalPredictionV2

    kind = json.loads(body).get("projection_kind")
    if kind == "precise_branch":
        from .eval_f_contracts import PreciseBranchProjection

        return PreciseBranchProjection.model_validate_json(body)
    if kind == "paired_branch":
        from .precise_dev_paired_wire import PairedProjection

        return PairedProjection.model_validate_json(body)
    if kind == "paired_branch_v2":
        from .precise_dev_paired_wire import PairedProjectionV2

        return PairedProjectionV2.model_validate_json(body)
    model = BranchProjectionV2 if kind == "model_branch" else EvalPredictionV2
    return model.model_validate_json(body)
