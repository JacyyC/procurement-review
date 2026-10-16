"""F-only precise projections, immutable candidate registration, and timing."""

from typing import Annotated, Literal
from pydantic import Field, model_validator

from .business_v3_evidence import EvidenceEnvelopeV3
from .business_v3_review import PreciseDecisionV3
from .business_v3_security import SecurityEventV3
from .chat_transport_v3 import EndpointConfigV3
from .eval_v2_contracts import EvalV2Model
from .eval_v2_contracts import SemanticClaimProjection
from .model_attempt_projection import BranchProjectionV2
from .precise_answer_v3 import PreciseClaimProjection
from .model_attempt_v2 import BudgetProof


class FBudgetProof(BudgetProof):
    # The finite registered experiment size, not a cumulative balance limit.
    allocated_requests: int = Field(gt=0)


class RejectedWitness(EvalV2Model):
    """Bounded, egress-checked audit diagnostic; never an adopted business claim."""

    obligation_id: str = Field(pattern=r"^O-(0[1-9]|1[0-2])$")
    field: str = Field(pattern=r"^[a-z_]+$", max_length=48)
    value: (
        Annotated[str, Field(min_length=1, max_length=256)]
        | Annotated[int, Field(ge=0, le=10**18)]
        | bool
    )
    code: str = Field(pattern=r"^[a-z_]+$", max_length=96)


class PreciseBranchProjection(BranchProjectionV2):
    projection_kind: Literal["precise_branch"] = "precise_branch"
    #: The frozen claim shape stays readable (historical fixtures and the F line build it)
    #: and the precise shape carries the value bound the 3.1.0 wire actually needs.
    claims: tuple[SemanticClaimProjection | PreciseClaimProjection, ...] = Field(
        max_length=12
    )
    evidence_output: EvidenceEnvelopeV3 | None
    security_events: tuple[SecurityEventV3, ...] = ()
    rejected_witnesses: tuple[RejectedWitness, ...] = Field(default=(), max_length=16)
    first_received_nonempty_body: bool | None = None

    @model_validator(mode="after")
    def adopted_output_only(self):
        if self.pipeline_failure and self.evidence_output is not None:
            raise ValueError("failed branch cannot retain adopted evidence")
        if not self.pipeline_failure and self.evidence_output is None:
            raise ValueError("successful branch requires complete evidence projection")
        return self


class FEndpointConfig(EndpointConfigV3):
    experiment_contract: Literal["v1-07f-batch.v1"] = "v1-07f-batch.v1"
    prompt_style: Literal["baseline", "compact"] = "baseline"

    @property
    def fingerprint(self):
        from .eval_f_registration import implementation_fingerprints
        from .gold_v3_contracts import payload_sha256

        return payload_sha256(
            {
                "parent": super().fingerprint,
                "implementation": implementation_fingerprints(),
            }
        )


class CaseTiming(EvalV2Model):
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
    scope: Literal["independent_oracle_runner"] = "independent_oracle_runner"


class FCaseProjection(EvalV2Model):
    schema_version: Literal["eval-precise-case.v1"] = "eval-precise-case.v1"
    case_id: str
    repeat_seed: int
    configuration_sha256: str
    decision: PreciseDecisionV3
    branches: tuple[PreciseBranchProjection, ...]
    timing: CaseTiming
