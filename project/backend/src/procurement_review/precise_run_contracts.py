"""The versioned precise execution record the product Run persists.

``review-execution.v1`` is the frozen fan-in of three ``WorkerBranchResult``
branches.  The precise line cannot be projected into that shape without losing the
exact proposition and the source identity the rule set acts on, so it records its
own version instead: the two model branches keep their grounded propositions,
verified spans and support failures, the procurement branch records that it made
no model call at all, and the decision, the orthogonal reasons and the real
metering travel beside them.  Both versions are read back by the version they
declare, and an unknown version fails closed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, RootModel, model_validator

from .business_v3_conditional import BusinessDecisionV31
from .business_v3_conflicts import ConflictPairV3
from .business_v3_security import SecurityEventV3
from .business_v3_support import VerifiedSpanV3
from .chat_transport_v2 import ThinkingMode
from .contracts import EvidenceRef, Identifier, ReviewSummary
from .eval_v2_contracts import EvalV2Model
from .gold_v3_contracts import payload_sha256
from .precise_answer_v3 import PRECISE_VALUE_LIMIT
from .review_graph import ReviewExecution

PRECISE_RUN_CONTRACT = "precise-run.v1"
PRECISE_EXECUTION_VERSION = "review-execution.v2"
PRECISE_RUN_SNAPSHOT_VERSION = "precise-run-snapshot.v1"
PRECISE_BRANCH_COMMIT_VERSION = "precise-branch-commit.v1"
PROCUREMENT_WORKER = "procurement_data"

#: The key the run snapshot travels under inside the Run's frozen Case input.
SNAPSHOT_KEY = "execution_snapshot"


class PreciseBranchOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ATTEMPTS_LEFT = "no_attempts_left"
    #: Not attempted because an explicit control path stopped further sends.  It is a
    #: different fact from "the persistent quota was already spent".
    STOPPED = "stopped"


class PreciseModelAttemptV1(EvalV2Model):
    """One physical model attempt: what was sent, what came back and what it cost."""

    attempt_no: Annotated[int, Field(ge=1, le=8)]
    error_code: str | None = Field(default=None, max_length=96)
    correctable: bool = False
    latency_ms: Annotated[int, Field(ge=0, le=3_600_000)] | None = None
    input_tokens: Annotated[int, Field(ge=0, le=10**9)] | None = None
    output_tokens: Annotated[int, Field(ge=0, le=10**9)] | None = None
    resolved_model: str | None = Field(default=None, max_length=128)
    usage_unknown: bool = False


class PreciseDutyValueV31(EvalV2Model):
    """One duty as this branch reported it, with the value the model normalized.

    The bound is the v3 answer's own canonical value bound: the richest registered duty
    does not fit in the frozen 256 characters, which is exactly why the wire was versioned.
    """

    obligation_id: str = Field(pattern=r"^O-(0[1-9]|1[0-2])$")
    status: Literal["supported", "violated", "unknown", "not_applicable"]
    normalized: str | None = Field(default=None, max_length=PRECISE_VALUE_LIMIT)


class PreciseDocumentBranchV1(EvalV2Model):
    """One model branch as the precise rule set consumes it.

    It records what the branch reported and what the source boundary accepted —
    the bounded normalized values, the verified spans with their evidence identity
    and proposition, and the page of support failures — never the raw answer text.
    """

    worker: Literal["policy", "contract"]
    outcome: PreciseBranchOutcome
    reported: tuple[PreciseDutyValueV31, ...] = Field(max_length=12)
    spans: tuple[VerifiedSpanV3, ...] = Field(max_length=192)
    support_failures: tuple[str, ...] = Field(max_length=32)
    request_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    answer_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attempts: tuple[PreciseModelAttemptV1, ...] = Field(max_length=4)
    evidence_refs: tuple[EvidenceRef, ...] = Field(max_length=64)
    failure_code: str | None = Field(default=None, max_length=96)


class DeterministicProcurementBranchV1(EvalV2Model):
    """The third Worker: a deterministic read-only branch with no model call."""

    worker: Literal["procurement_data"] = PROCUREMENT_WORKER
    status: str = Field(max_length=48)
    dataset_version: str = Field(max_length=96)
    tool_calls: Annotated[int, Field(ge=0, le=64)]
    record_count: Annotated[int, Field(ge=0, le=4096)]
    model_calls: Literal[0] = 0
    detail: str | None = Field(default=None, max_length=500)


class RunMeteringV1(EvalV2Model):
    """What the Run really sent, in the units the ledger really has.

    One opened attempt row is not one send: the row is written before the transport is
    entered, so a process that died in between never sent anything, and a row closed by an
    interruption may or may not have reached the endpoint.  ``attempts_opened`` counts the
    rows, ``proven_sends`` counts the attempts whose own receipt proves a request was sent
    and answered, and ``unresolved_attempts`` plus ``open_attempts`` count the rows whose
    send state is not decidable.  ``input_tokens_known`` is a subtotal over the attempts
    whose usage is known; ``usage_complete`` says whether that subtotal is the whole Run.

    The three correction counts split what the extra attempts were for, so a contract or
    citation correction is never reported as a transport retry and the reverse.  A row
    written before this distinction existed carries none of them and is reported as
    ``other_extra_attempts``, which keeps a historical Run's numbers unchanged.
    """

    physical_attempts: Annotated[int, Field(ge=0, le=64)]
    first_sends: Annotated[int, Field(ge=0, le=64)]
    corrections: Annotated[int, Field(ge=0, le=64)]
    input_tokens_known: Annotated[int, Field(ge=0, le=10**12)] | None = None
    output_tokens_known: Annotated[int, Field(ge=0, le=10**12)] | None = None
    unknown_usage_attempts: Annotated[int, Field(ge=0, le=64)] = 0
    unreturned_attempts: Annotated[int, Field(ge=0, le=64)] = 0
    branch_latency_ms: tuple[
        tuple[str, Annotated[int, Field(ge=0, le=3_600_000)]], ...
    ] = Field(max_length=4)
    requested_model: str | None = Field(default=None, max_length=128)
    resolved_models: tuple[str, ...] = Field(default=(), max_length=8)
    #: The explicit send/return split.  Legacy rows predate these fields and read back with
    #: their own defaults, so a historical Run is never re-interpreted by a new consumer.
    attempts_opened: Annotated[int, Field(ge=0, le=64)] | None = None
    proven_sends: Annotated[int, Field(ge=0, le=64)] | None = None
    unresolved_attempts: Annotated[int, Field(ge=0, le=64)] | None = None
    open_attempts: Annotated[int, Field(ge=0, le=64)] | None = None
    contract_corrections: Annotated[int, Field(ge=0, le=64)] | None = None
    citation_corrections: Annotated[int, Field(ge=0, le=64)] | None = None
    transport_retries: Annotated[int, Field(ge=0, le=64)] | None = None
    other_extra_attempts: Annotated[int, Field(ge=0, le=64)] | None = None
    usage_complete: bool | None = None
    reused_branches: Annotated[int, Field(ge=0, le=4)] | None = None
    explicit_recoveries: Annotated[int, Field(ge=0, le=4)] | None = None


class PreciseRunExecutionV1(EvalV2Model):
    """The versioned execution payload of one precise Run."""

    execution_version: Literal["review-execution.v2"] = PRECISE_EXECUTION_VERSION
    run_id: Identifier = Field(max_length=128)
    case_id: Identifier = Field(max_length=128)
    rules_version: Literal["3.1.0"] = "3.1.0"
    decision: BusinessDecisionV31
    security: tuple[SecurityEventV3, ...] = Field(max_length=64)
    conflicts: tuple[ConflictPairV3, ...] = Field(max_length=32)
    policy: PreciseDocumentBranchV1
    contract: PreciseDocumentBranchV1
    procurement: DeterministicProcurementBranchV1
    metering: RunMeteringV1
    summary: ReviewSummary
    #: The original code of the failure that ended further sending, when one did.  The
    #: branch records keep their own codes; this names why the remaining branch was not
    #: attempted, so a stopped execution is never read as one that simply ran out of quota.
    stop_reason: str | None = Field(default=None, max_length=96)

    @model_validator(mode="after")
    def branches_match_their_roles(self) -> PreciseRunExecutionV1:
        if self.policy.worker != "policy" or self.contract.worker != "contract":
            raise ValueError("precise branches must keep their declared roles")
        if self.summary.rules_version != self.rules_version:
            raise ValueError("the committed summary must declare the decision's rules")
        if self.procurement.model_calls != 0:
            raise ValueError(
                "the procurement branch is deterministic and calls no model"
            )
        return self


class PreciseRunSnapshotV1(EvalV2Model):
    """The execution configuration one precise Run was created with.

    It is written into the Run's frozen Case input at creation and read back by every
    later generation, so a resume reuses the original mode, model identity, sampling
    parameters, wire/prompt/schema digests, rule and material versions, retrieval
    adapters, branch budget and generation deadline instead of whatever the runtime is
    configured with now.  A runtime that no longer matches it stops with a recorded
    reason; it never silently re-runs the Case under different parameters.
    """

    snapshot_version: Literal["precise-run-snapshot.v1"] = PRECISE_RUN_SNAPSHOT_VERSION
    mode: Literal["test_only", "real"]
    requested_model: str = Field(min_length=1, max_length=160)
    resolved_model: str = Field(min_length=1, max_length=160)
    response_mode: Literal["json_schema", "json_object", "text"]
    prompt_style: Literal["baseline", "compact"]
    #: The accepted ``thinking.type`` values of the registered endpoint; an old snapshot
    #: that carries ``disabled`` or nothing keeps its own reading, and a value outside the
    #: accepted set fails validation instead of being silently re-interpreted.
    thinking: ThinkingMode | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    single_attempt_seconds: float = Field(gt=0, le=600)
    branch_timeout_seconds: float = Field(gt=0, le=600)
    max_attempts_per_branch: int = Field(ge=1, le=8)
    generation_deadline_seconds: float = Field(gt=0, le=900)
    rules_version: str = Field(min_length=1, max_length=64)
    document_version: str = Field(min_length=1, max_length=160)
    dataset_version: str = Field(min_length=1, max_length=160)
    index_version: str = Field(min_length=1, max_length=160)
    retrieval_vector_adapter: str = Field(min_length=1, max_length=96)
    retrieval_lexical_adapter: str = Field(min_length=1, max_length=96)
    wire_version: str = Field(min_length=1, max_length=96)
    prompt_version: str = Field(min_length=1, max_length=96)
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def fingerprint(self) -> str:
        return payload_sha256(self.model_dump(mode="json"))


def snapshot_mismatch(
    frozen: PreciseRunSnapshotV1 | None, current: PreciseRunSnapshotV1
) -> str | None:
    """Which declared dimension no longer matches, as one fixed reason word."""

    if frozen is None:
        return "snapshot_missing"
    record = frozen.model_dump(mode="json")
    live = current.model_dump(mode="json")
    for reason, names in (
        ("mode", ("mode",)),
        ("model_identity", ("requested_model", "resolved_model")),
        (
            "sampling_parameters",
            ("thinking", "temperature", "top_p", "seed", "max_tokens"),
        ),
        ("response_mode", ("response_mode", "prompt_style")),
        ("wire_or_prompt", ("wire_version", "prompt_version")),
        ("schema_digest", ("schema_sha256",)),
        ("implementation_digest", ("implementation_sha256",)),
        ("rules_version", ("rules_version",)),
        ("document_version", ("document_version",)),
        ("dataset_version", ("dataset_version",)),
        ("index_version", ("index_version",)),
        (
            "retrieval_adapters",
            ("retrieval_vector_adapter", "retrieval_lexical_adapter"),
        ),
        ("branch_budget", ("max_attempts_per_branch",)),
        (
            "execution_deadline",
            (
                "generation_deadline_seconds",
                "branch_timeout_seconds",
                "single_attempt_seconds",
            ),
        ),
    ):
        if any(record[name] != live[name] for name in names):
            return reason
    return None


def precise_contract_identity(snapshot: PreciseRunSnapshotV1, worker: str) -> str:
    """The contract one stored branch commit must match to stay reusable."""

    return "sha256:" + payload_sha256(
        {"snapshot": snapshot.fingerprint, "worker": worker}
    )


class PreciseAdoptedDutyV1(EvalV2Model):
    """One duty as the committed branch reported it: bounded value and its citations."""

    obligation_id: str = Field(pattern=r"^O-(0[1-9]|1[0-2])$")
    status: Literal["supported", "violated", "unknown", "not_applicable"]
    risk_level: Literal["none", "low", "medium", "high", "critical", "unknown"]
    value: str = Field(min_length=1, max_length=PRECISE_VALUE_LIMIT)
    evidence_refs: tuple[Annotated[str, Field(pattern=r"^ref-[0-9a-f]{64}$")], ...] = (
        Field(default=(), max_length=4)
    )


class PreciseBranchCommitV1(EvalV2Model):
    """The complete, bounded projection of one adopted model branch.

    It keeps the duties with their bounded normalized values and citations, the verified
    spans with their evidence identity, the branch's support and document failures and the
    bounded security events its own output produced — never an answer text, a prompt or a
    reasoning trace.  Only this projection is reusable: a receipt without it is not a
    success.
    """

    commit_version: Literal["precise-branch-commit.v1"] = PRECISE_BRANCH_COMMIT_VERSION
    worker: Literal["policy", "contract"]
    duties: tuple[PreciseAdoptedDutyV1, ...] = Field(min_length=1, max_length=12)
    spans: tuple[VerifiedSpanV3, ...] = Field(default=(), max_length=192)
    document_failures: tuple[str, ...] = Field(default=(), max_length=32)
    support_failures: tuple[str, ...] = Field(default=(), max_length=32)
    security: tuple[SecurityEventV3, ...] = Field(default=(), max_length=32)
    blocked: bool = False
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: tuple[PreciseModelAttemptV1, ...] = Field(default=(), max_length=4)
    evidence_refs: tuple[EvidenceRef, ...] = Field(default=(), max_length=64)


class PreciseCheckpointPayloadV1(EvalV2Model):
    """The checkpoint payload of the precise line: one versioned branch commit."""

    checkpoint_version: Literal["precise-branch-commit.v1"]
    input_checksum: str = Field(min_length=1, max_length=160)
    contract_checksum: str = Field(min_length=1, max_length=160)
    result: PreciseBranchCommitV1


#: The two declared execution shapes.  The discriminator is the recorded version,
#: so an old Run is judged by the shape it was written with and an unknown version
#: has no branch to fall into.
ExecutionPayload = Annotated[
    ReviewExecution | PreciseRunExecutionV1,
    Field(discriminator="execution_version"),
]


class ExecutionRecord(RootModel[ExecutionPayload]):
    """One stored ``execution_payload``, judged by the version it declares."""


__all__ = [
    "PRECISE_BRANCH_COMMIT_VERSION",
    "PRECISE_EXECUTION_VERSION",
    "PRECISE_RUN_CONTRACT",
    "PRECISE_RUN_SNAPSHOT_VERSION",
    "PROCUREMENT_WORKER",
    "SNAPSHOT_KEY",
    "DeterministicProcurementBranchV1",
    "ExecutionPayload",
    "ExecutionRecord",
    "PreciseAdoptedDutyV1",
    "PreciseBranchCommitV1",
    "PreciseBranchOutcome",
    "PreciseCheckpointPayloadV1",
    "PreciseDocumentBranchV1",
    "PreciseDutyValueV31",
    "PreciseModelAttemptV1",
    "PreciseRunExecutionV1",
    "PreciseRunSnapshotV1",
    "RunMeteringV1",
    "precise_contract_identity",
    "snapshot_mismatch",
]
