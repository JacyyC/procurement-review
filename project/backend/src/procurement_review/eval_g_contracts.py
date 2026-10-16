"""Versioned G/H two-arm comparison contracts.

The pre-registered plan is the only source of coverage denominators: an arm that
never ran keeps its full planned denominator instead of shrinking it.  Nothing
here reads truth labels or production aggregation, and the neutral Q-12 gate
never depends on the direction of the measured difference.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .eval_v2_contracts import EvalTopology, EvalV2Model, ForbiddenEvent
from .gold_v3_contracts import CaseId, HexSha256, ObligationId, V3Recommendation

COMPARISON_PLAN_V1 = "v1-07g-comparison-plan.v1"
COMPARISON_OBSERVATION_V1 = "v1-07g-comparison-observation.v1"
COMPARISON_METRICS_V2 = "v1-07-eval-metrics.v2"
COMPARISON_REPORT_V5 = "eval-report.v5"
COMPARISON_AUDIT_V1 = "v1-07g-comparison-audit.v1"

#: One first send plus one shared additional attempt per logical model branch.
MAX_PHYSICAL_PER_BRANCH: Literal[2] = 2

BranchId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,31}$")]
MeteringNamespace = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{2,63}$")]
TaskKey = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{3,192}$")]

#: Duties answered without any model call; identical in both arms by construction.
DETERMINISTIC_OBLIGATIONS: tuple[ObligationId, ...] = (
    "O-03",
    "O-04",
    "O-07",
    "O-10",
    "O-11",
    "O-12",
)
DOCUMENT_OBLIGATIONS: tuple[ObligationId, ...] = (
    "O-01",
    "O-02",
    "O-05",
    "O-06",
    "O-08",
    "O-09",
)
#: The document role of a duty is a property of the evidence, not of the call
#: structure.  Both arms therefore produce identical proposition identities, and
#: the single-agent arm cannot be scored against a role it was never given.
DOCUMENT_ROLE_BY_OBLIGATION: dict[str, Literal["policy", "contract"]] = {
    "O-01": "policy",
    "O-02": "policy",
    "O-05": "contract",
    "O-06": "contract",
    "O-08": "contract",
    "O-09": "contract",
}
ALL_OBLIGATIONS: tuple[ObligationId, ...] = tuple(
    f"O-{index:02d}" for index in range(1, 13)
)


class ComparisonArm(StrEnum):
    """The two pre-registered arms; the topology difference is the object under test."""

    MULTI_AGENT = "multi_agent"
    SINGLE_AGENT = "single_agent"


class TerminalState(StrEnum):
    """Attempt-level terminal outcomes; every planned observation must reach one."""

    COMMITTED = "committed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REMOTE_UNKNOWN = "remote_unknown"
    BLOCKED_UNKNOWN = "blocked_unknown"


class ClaimCollectionStatus(StrEnum):
    """Whether the bounded semantic claims an auditor needs were retained."""

    COMPLETE = "complete"
    BLOCKED_BY_PROTECTION = "blocked_by_protection"
    ABSENT = "absent"


class ModelBranch(EvalV2Model):
    """One logical model branch: a single first send plus one shared retry."""

    branch_id: BranchId
    obligation_ids: tuple[ObligationId, ...] = Field(min_length=1, max_length=12)
    max_physical_requests: Literal[2] = MAX_PHYSICAL_PER_BRANCH

    @model_validator(mode="after")
    def duties_are_unique(self) -> Self:
        if len(set(self.obligation_ids)) != len(self.obligation_ids):
            raise ValueError("a model branch cannot own one obligation twice")
        return self

    @property
    def document_roles(self) -> tuple[str, ...]:
        return tuple(
            sorted({DOCUMENT_ROLE_BY_OBLIGATION[item] for item in self.obligation_ids})
        )


class ArmShape(EvalV2Model):
    """Call structure of one arm.  Evidence identities are not part of the shape."""

    arm_id: ComparisonArm
    topology: EvalTopology
    model_branches: tuple[ModelBranch, ...] = Field(min_length=1, max_length=3)
    deterministic_branches: tuple[BranchId, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def shape_matches_topology_and_covers_duties(self) -> Self:
        expected = (
            EvalTopology.MULTI_AGENT
            if self.arm_id is ComparisonArm.MULTI_AGENT
            else EvalTopology.SINGLE_AGENT
        )
        if self.topology is not expected:
            raise ValueError("arm topology does not match its registered arm id")
        if self.arm_id is ComparisonArm.SINGLE_AGENT and len(self.model_branches) != 1:
            raise ValueError("the single-agent arm issues exactly one joint first send")
        if self.arm_id is ComparisonArm.MULTI_AGENT and len(self.model_branches) < 2:
            raise ValueError("the production arm keeps its isolated document branches")
        branch_ids = [item.branch_id for item in self.model_branches]
        branch_ids.extend(self.deterministic_branches)
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("branch ids must be unique within one arm")
        owned: list[str] = []
        for branch in self.model_branches:
            owned.extend(branch.obligation_ids)
        if len(set(owned)) != len(owned):
            raise ValueError("two model branches cannot own the same obligation")
        if set(owned) != set(DOCUMENT_OBLIGATIONS):
            raise ValueError("model branches must own exactly the document duties")
        for branch in self.model_branches:
            roles = branch.document_roles
            if self.arm_id is ComparisonArm.MULTI_AGENT:
                if len(roles) != 1 or branch.branch_id != roles[0]:
                    raise ValueError(
                        "an isolated document branch keeps exactly its own role"
                    )
            elif len(roles) != 2:
                raise ValueError(
                    "the joint branch must answer both document roles itself"
                )
        return self

    def obligation_coverage(self) -> tuple[str, ...]:
        return tuple(sorted(set(DOCUMENT_OBLIGATIONS) | set(DETERMINISTIC_OBLIGATIONS)))


class FairnessContract(EvalV2Model):
    """The inputs both arms must share.  Anything listed here is not a variable."""

    requested_model: Annotated[str, Field(min_length=1, max_length=160)]
    expected_resolved_model: Annotated[str, Field(min_length=1, max_length=160)]
    temperature: Literal[0.0] = 0.0
    thinking: Literal["disabled"] = "disabled"
    output_mode: Literal["json_schema"] = "json_schema"
    case_concurrency: Annotated[int, Field(ge=1, le=8)]
    single_attempt_seconds: Annotated[float, Field(gt=0.0, le=90.0)]
    run_deadline_seconds: Annotated[float, Field(gt=0.0, le=120.0)]
    materials_sha256: HexSha256
    fact_tools_sha256: HexSha256
    retrieval_index_sha256: HexSha256
    truth_contract: Annotated[str, Field(pattern=r"^v1-07-truth\.v1(\.[0-9]+)?$")]
    scoring_version: Annotated[str, Field(pattern=r"^v1-07g-score\.v[0-9]+$")]
    correction_policy: Literal["one_shared_additional_attempt"] = (
        "one_shared_additional_attempt"
    )

    @model_validator(mode="after")
    def deadlines_stay_within_registered_limits(self) -> Self:
        if self.run_deadline_seconds < self.single_attempt_seconds:
            raise ValueError("a run deadline shorter than one attempt cannot be met")
        return self


class PriorLedger(EvalV2Model):
    """One historical metering scope.  Scopes are never summed together here."""

    namespace: MeteringNamespace
    source_entry: Annotated[str, Field(min_length=1, max_length=240)]
    source_status: Literal["read", "registered_only"]
    count_basis: Literal["exact", "upper_bound"]
    physical_sent: Annotated[int, Field(ge=0)]
    receipts: Annotated[int, Field(ge=0)]
    unreturned: Annotated[int, Field(ge=0)]
    usage_unknown: Annotated[int, Field(ge=0)]
    known_input_tokens: Annotated[int, Field(ge=0)]
    known_output_tokens: Annotated[int, Field(ge=0)]
    cost_status: Literal["unknown", "known"] = "unknown"
    cost_value: float | None = None
    source_sha256: HexSha256 | None = None

    @model_validator(mode="after")
    def unknown_is_never_zero_filled(self) -> Self:
        if self.receipts + self.unreturned > self.physical_sent:
            raise ValueError("receipts and unreturned exceed the physical requests")
        if self.count_basis == "exact" and (
            self.receipts + self.unreturned != self.physical_sent
        ):
            raise ValueError("an exact scope closes receipts against physical requests")
        if self.cost_status == "known":
            if self.cost_value is None:
                raise ValueError("a known cost requires its measured value")
        elif self.cost_value is not None:
            raise ValueError("an unknown cost must not carry a value")
        if self.source_status == "read" and self.source_sha256 is None:
            raise ValueError("a scope read from artifacts keeps their fingerprint")
        return self


class HistoryMetering(EvalV2Model):
    """The new namespace plus every prior scope kept at its own declared cardinality."""

    namespace: MeteringNamespace
    prior_ledgers: tuple[PriorLedger, ...] = Field(max_length=16)
    double_counting_excluded: Literal[True] = True

    @model_validator(mode="after")
    def namespaces_are_distinct(self) -> Self:
        names = [item.namespace for item in self.prior_ledgers]
        if len(set(names)) != len(names):
            raise ValueError("one historical namespace cannot be registered twice")
        if self.namespace in names:
            raise ValueError("the new namespace must differ from every prior scope")
        if not self.namespace.startswith(("v1-07g", "v1-07h")):
            raise ValueError(
                "a new comparison namespace must belong to the current G/H line"
            )
        return self


class ArmPlan(EvalV2Model):
    """One arm's pre-registered observations, first sends and physical capacity."""

    shape: ArmShape
    case_ids: tuple[CaseId, ...] = Field(min_length=1, max_length=400)
    repeat_seeds: tuple[Annotated[int, Field(gt=0)], ...] = Field(
        min_length=1, max_length=4
    )
    planned_observations: Annotated[int, Field(ge=1)]
    planned_first_sends: Annotated[int, Field(ge=1)]
    max_physical_requests: Annotated[int, Field(ge=2)]

    @model_validator(mode="after")
    def denominators_are_the_declared_plan(self) -> Self:
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("an arm cannot register one case twice")
        if len(set(self.repeat_seeds)) != len(self.repeat_seeds):
            raise ValueError("an arm cannot register one repeat seed twice")
        observations = len(self.case_ids) * len(self.repeat_seeds)
        if self.planned_observations != observations:
            raise ValueError("planned observations must equal cases times repeats")
        first_sends = observations * len(self.shape.model_branches)
        if self.planned_first_sends != first_sends:
            raise ValueError(
                "planned first sends must equal observations times branches"
            )
        if self.max_physical_requests != first_sends * 2:
            raise ValueError("physical capacity is two shared attempts per first send")
        return self

    def task_keys(self) -> tuple[TaskKey, ...]:
        """Enumerate every logical first-send task this arm pre-registers."""

        return tuple(
            f"{case_id}.{seed}.{branch.branch_id}"
            for seed in self.repeat_seeds
            for case_id in self.case_ids
            for branch in self.shape.model_branches
        )

    def observation_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (case_id, seed) for seed in self.repeat_seeds for case_id in self.case_ids
        )


class ComparisonPlanV1(EvalV2Model):
    """The frozen two-arm plan every coverage denominator is read from."""

    schema_version: Literal["v1-07g-comparison-plan.v1"] = COMPARISON_PLAN_V1
    plan_id: Annotated[str, Field(pattern=r"^v1-07g-[a-z0-9][a-z0-9-]{2,63}$")]
    stage_id: Literal["V1-07G", "V1-07H"]
    split: Literal["dev", "validation", "blind"]
    gold_set_version: Annotated[str, Field(pattern=r"^gold-set\.v3(\.[0-9]+)?$")]
    fairness: FairnessContract
    history: HistoryMetering
    arms: tuple[ArmPlan, ArmPlan]
    authorization: Annotated[str, Field(min_length=1, max_length=120)]
    stop_conditions: tuple[Annotated[str, Field(pattern=r"^[a-z0-9_]+$")], ...] = Field(
        min_length=1, max_length=16
    )
    plan_sha256: HexSha256 | None = None

    @model_validator(mode="after")
    def arms_are_paired_and_fair(self) -> Self:
        registered = {arm.shape.arm_id: arm for arm in self.arms}
        if len(registered) != 2:
            raise ValueError("a comparison registers exactly one plan per arm")
        multi, single = (
            registered[ComparisonArm.MULTI_AGENT],
            registered[ComparisonArm.SINGLE_AGENT],
        )
        if set(multi.case_ids) != set(single.case_ids):
            raise ValueError("both arms must run the identical registered case set")
        if multi.repeat_seeds != single.repeat_seeds:
            raise ValueError("both arms must run the identical registered repeats")
        if multi.shape.obligation_coverage() != single.shape.obligation_coverage():
            raise ValueError("both arms must answer the identical obligation catalog")
        if self.split == "blind" and self.stage_id == "V1-07G":
            raise ValueError("G never registers the formal blind split")
        return self

    @property
    def arm(self) -> dict[ComparisonArm, ArmPlan]:
        return {item.shape.arm_id: item for item in self.arms}

    def planned_observations(self) -> int:
        return sum(item.planned_observations for item in self.arms)

    def planned_first_sends(self) -> int:
        return sum(item.planned_first_sends for item in self.arms)

    def planned_physical_capacity(self) -> int:
        return sum(item.max_physical_requests for item in self.arms)


class ComparisonObservationV1(EvalV2Model):
    """One bounded terminal observation; never a prompt, response or chain of thought."""

    schema_version: Literal["v1-07g-comparison-observation.v1"] = (
        COMPARISON_OBSERVATION_V1
    )
    arm: ComparisonArm
    case_id: CaseId
    repeat_seed: Annotated[int, Field(gt=0)]
    plan_sha256: HexSha256
    terminal_state: TerminalState
    recommendation: V3Recommendation | None = None
    decision_sha256: HexSha256 | None = None
    claim_collection_status: ClaimCollectionStatus = ClaimCollectionStatus.ABSENT
    claim_count: Annotated[int, Field(ge=0, le=12)] = 0
    claim_digest_sha256: HexSha256 | None = None
    first_send_count: Annotated[int, Field(ge=0, le=6)] = 0
    first_pass_valid_count: Annotated[int, Field(ge=0, le=6)] = 0
    corrected_valid_count: Annotated[int, Field(ge=0, le=6)] = 0
    physical_request_count: Annotated[int, Field(ge=0, le=12)] = 0
    known_input_tokens: Annotated[int, Field(ge=0)] | None = None
    known_output_tokens: Annotated[int, Field(ge=0)] | None = None
    usage_unknown_attempts: Annotated[int, Field(ge=0, le=12)] = 0
    total_latency_ms: Annotated[int, Field(ge=0, le=600_000)] | None = None
    pipeline_failure: bool = False
    stop_reason: (
        Annotated[str, Field(pattern=r"^[a-z0-9_]+$", max_length=64)] | None
    ) = None
    forbidden_events: tuple[ForbiddenEvent, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def terminal_record_is_settled_and_honest(self) -> Self:
        if self.first_pass_valid_count > self.first_send_count:
            raise ValueError("valid first sends cannot exceed the first sends issued")
        if self.first_pass_valid_count + self.corrected_valid_count > (
            self.first_send_count
        ):
            raise ValueError(
                "first-pass and corrected validity cannot exceed the first sends"
            )
        if self.physical_request_count < self.first_send_count:
            raise ValueError("physical requests cannot be fewer than the first sends")
        tokens = (self.known_input_tokens, self.known_output_tokens)
        if any(value is not None for value in tokens) != all(
            value is not None for value in tokens
        ):
            raise ValueError("known usage requires both token counts")
        if self.terminal_state is TerminalState.COMMITTED:
            if self.recommendation is None or self.decision_sha256 is None:
                raise ValueError("a committed observation keeps its adopted decision")
            if self.total_latency_ms is None:
                raise ValueError("a committed observation keeps its measured latency")
            if (
                all(value is None for value in tokens)
                and not self.usage_unknown_attempts
            ):
                raise ValueError(
                    "settled usage is either known or explicitly declared unknown"
                )
            if self.claim_collection_status is ClaimCollectionStatus.COMPLETE:
                if self.claim_count == 0 or self.claim_digest_sha256 is None:
                    raise ValueError(
                        "complete claim collection keeps its bounded digest"
                    )
            elif self.claim_count or self.claim_digest_sha256 is not None:
                raise ValueError(
                    "claims retained without a complete collection status are inconsistent"
                )
        else:
            if self.recommendation is not None or self.decision_sha256 is not None:
                raise ValueError("an unsettled observation adopts no decision")
            if self.stop_reason is None:
                raise ValueError("an unsettled terminal state records its stop reason")
        return self

    @property
    def key(self) -> tuple[str, int]:
        return (self.case_id, self.repeat_seed)


class ArmResources(EvalV2Model):
    """Ledger-derived usage for one arm, kept separate from every prior scope."""

    arm_id: ComparisonArm
    namespace: MeteringNamespace
    physical_requests: Annotated[int, Field(ge=0)]
    receipts: Annotated[int, Field(ge=0)]
    additional_attempts: Annotated[int, Field(ge=0)]
    known_input_tokens: Annotated[int, Field(ge=0)] = 0
    known_output_tokens: Annotated[int, Field(ge=0)] = 0
    usage_unknown_attempts: Annotated[int, Field(ge=0)] = 0
    unreturned_attempts: Annotated[int, Field(ge=0)] = 0
    cost_status: Literal["unknown", "known"] = "unknown"
    cost_value: float | None = None
    p95_total_latency_ms: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def ledger_counts_close(self) -> Self:
        if self.receipts > self.physical_requests:
            raise ValueError("receipts exceed physical requests")
        if self.unreturned_attempts != self.physical_requests - self.receipts:
            raise ValueError("unreturned attempts must close against the receipts")
        if self.additional_attempts > self.physical_requests:
            raise ValueError("additional attempts exceed the physical requests")
        if self.cost_status == "known":
            if self.cost_value is None:
                raise ValueError("a known cost requires its measured value")
        elif self.cost_value is not None:
            raise ValueError("an unknown cost must not carry a value")
        return self


__all__ = [
    "ALL_OBLIGATIONS",
    "ArmPlan",
    "ArmResources",
    "ArmShape",
    "COMPARISON_AUDIT_V1",
    "COMPARISON_METRICS_V2",
    "COMPARISON_OBSERVATION_V1",
    "COMPARISON_PLAN_V1",
    "COMPARISON_REPORT_V5",
    "ClaimCollectionStatus",
    "ComparisonArm",
    "ComparisonObservationV1",
    "ComparisonPlanV1",
    "DETERMINISTIC_OBLIGATIONS",
    "DOCUMENT_OBLIGATIONS",
    "DOCUMENT_ROLE_BY_OBLIGATION",
    "FairnessContract",
    "HistoryMetering",
    "MAX_PHYSICAL_PER_BRANCH",
    "ModelBranch",
    "PriorLedger",
    "TerminalState",
]
