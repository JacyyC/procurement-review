"""Durable per-Case projections for either comparison arm.

F's ``eval-precise-case.v1`` is frozen with its two-branch Oracle shape and its
Oracle-only timing scope.  These contracts add the arm identity, the plan digest
and the workbench timing scope without touching that historical version, so an
old batch still reads back under its own contract.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from .business_v3_evidence import EvidenceEnvelopeV3
from .business_v3_review import PreciseDecisionV3
from .business_v3_security import SecurityEventV3
from .eval_f_contracts import RejectedWitness
from .eval_g_contracts import ArmShape, ComparisonArm
from .eval_v2_contracts import EvalV2Model, SemanticClaimProjection
from .gold_v3_contracts import CaseId, HexSha256
from .model_attempt_projection import ClaimDetail
from .precise_answer_v3 import PreciseClaimProjection

G_BRANCH_PROJECTION = "eval-projection.v3"
G_CASE_PROJECTION = "eval-precise-case.v2"

#: The joint arm answers both document roles in one call; the roles themselves stay
#: attached to the evidence, never to the call structure.
GBranchWorker = Literal["policy", "contract", "joint"]


class GBranchProjection(EvalV2Model):
    schema_version: Literal["eval-projection.v3"] = G_BRANCH_PROJECTION
    projection_kind: Literal["g_branch"] = "g_branch"
    arm: ComparisonArm
    case_id: CaseId
    worker: GBranchWorker
    claims: tuple[SemanticClaimProjection | PreciseClaimProjection, ...] = Field(
        max_length=12
    )
    details: tuple[ClaimDetail, ...] = Field(max_length=12)
    first_pass_valid: bool
    received_nonempty_body: bool
    corrected_valid: bool
    pipeline_failure: bool
    model_request_count: int = Field(ge=1, le=2)
    latency_ms: int = Field(ge=0)
    evidence_output: EvidenceEnvelopeV3 | None
    security_events: tuple[SecurityEventV3, ...] = ()
    rejected_witnesses: tuple[RejectedWitness, ...] = Field(default=(), max_length=16)
    first_received_nonempty_body: bool | None = None

    @model_validator(mode="after")
    def adopted_output_only(self) -> Self:
        if self.pipeline_failure and self.evidence_output is not None:
            raise ValueError("failed branch cannot retain adopted evidence")
        if not self.pipeline_failure and self.evidence_output is None:
            raise ValueError("successful branch requires complete evidence projection")
        return self


class GCaseTiming(EvalV2Model):
    queue_ms: int = Field(ge=0)
    context_ms: int = Field(ge=0)
    retrieval_ms: int | None = None
    mcp_ms: int | None = None
    model_ms: int = Field(ge=0)
    extra_attempt_ms: int = Field(ge=0)
    aggregate_ms: int = Field(ge=0)
    total_ms: int = Field(ge=0)
    connect_ms: int | None = None
    first_response_ms: int | None = None
    human_wait_ms: int | None = None
    scope: Literal[
        "product_workbench", "single_agent_runner", "independent_oracle_runner"
    ]


class GCaseProjection(EvalV2Model):
    schema_version: Literal["eval-precise-case.v2"] = G_CASE_PROJECTION
    arm: ComparisonArm
    case_id: CaseId
    repeat_seed: int = Field(gt=0)
    plan_sha256: HexSha256
    configuration_sha256: HexSha256
    decision: PreciseDecisionV3
    branches: tuple[GBranchProjection, ...]
    timing: GCaseTiming

    @model_validator(mode="after")
    def branches_match_the_registered_arm_shape(self) -> Self:
        workers = tuple(branch.worker for branch in self.branches)
        if len(workers) != len(set(workers)):
            raise ValueError("one projection per registered model branch")
        for branch in self.branches:
            if branch.case_id != self.case_id or branch.arm is not self.arm:
                raise ValueError("branch identity differs from its case projection")
        return self

    def branch_ids(self) -> tuple[str, ...]:
        return tuple(branch.worker for branch in self.branches)


def expected_branch_ids(shape: ArmShape) -> tuple[str, ...]:
    return tuple(branch.branch_id for branch in shape.model_branches)


def obligations_of(shape: ArmShape, branch_id: str) -> frozenset[str]:
    for branch in shape.model_branches:
        if branch.branch_id == branch_id:
            return frozenset(branch.obligation_ids)
    raise ValueError(f"branch {branch_id} is not registered for arm {shape.arm_id}")


__all__ = [
    "G_BRANCH_PROJECTION",
    "G_CASE_PROJECTION",
    "GCaseProjection",
    "GCaseTiming",
    "GBranchProjection",
    "GBranchWorker",
    "expected_branch_ids",
    "obligations_of",
]
