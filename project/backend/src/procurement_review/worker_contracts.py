"""Framework-independent contracts for isolated read-only review workers."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field, StringConstraints, model_validator

from .contracts import (
    BusinessFact,
    EvidenceRef,
    FactStatus,
    Finding,
    Identifier,
    StrictContract,
    WorkerKind,
)

WORKER_INPUT_VERSION = "worker-input.v1"
WORKER_MODEL_REQUEST_VERSION = "worker-model-request.v1"
SINGLE_AGENT_MODEL_REQUEST_VERSION = "single-agent-model-request.v1"
WORKER_RESULT_VERSION = "worker-result.v1"
PROCUREMENT_READ_RESULT_VERSION = "procurement-read-result.v1"

WorkerQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]
FailureDetail = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]


class PolicyWorkerInput(StrictContract):
    """Minimum input visible to the policy worker."""

    input_version: Literal["worker-input.v1"] = WORKER_INPUT_VERSION
    case_id: Identifier
    query: WorkerQuery
    index_version: Identifier
    model_fixture_id: Identifier


class ContractWorkerInput(StrictContract):
    """Minimum input visible to the contract worker."""

    input_version: Literal["worker-input.v1"] = WORKER_INPUT_VERSION
    case_id: Identifier
    query: WorkerQuery
    index_version: Identifier
    model_fixture_id: Identifier


class ProcurementWorkerInput(StrictContract):
    """Minimum input visible to the procurement anomaly worker."""

    input_version: Literal["worker-input.v1"] = WORKER_INPUT_VERSION
    case_id: Identifier
    purchase_order_id: Identifier
    dataset_version: Identifier
    model_fixture_id: Identifier


class EvidenceContext(StrictContract):
    """Retrieved text paired with the exact reference a model may cite."""

    evidence_ref: EvidenceRef
    content: WorkerQuery


class WorkerModelRequest(StrictContract):
    """Narrow context passed across the replaceable model boundary."""

    request_version: Literal["worker-model-request.v1"] = WORKER_MODEL_REQUEST_VERSION
    worker: WorkerKind
    case_id: Identifier
    model_fixture_id: Identifier
    question: WorkerQuery | None = None
    evidence_contexts: tuple[EvidenceContext, ...] = ()
    business_facts: tuple[BusinessFact, ...] = ()

    @model_validator(mode="after")
    def context_matches_worker(self) -> WorkerModelRequest:
        if self.worker in {WorkerKind.POLICY, WorkerKind.CONTRACT}:
            if not self.evidence_contexts or self.business_facts:
                raise ValueError("document workers require only retrieved evidence")
        elif not self.business_facts or self.evidence_contexts:
            raise ValueError("procurement workers require only MCP business facts")
        return self


class SingleAgentModelRequest(StrictContract):
    """One review pass over the union of the three workers' allowed context.

    Used only to compare agent topologies: the context items are exactly the ones
    the three ``WorkerModelRequest`` calls carry, so no evidence is added and the
    deterministic aggregator stays identical.
    """

    request_version: Literal["single-agent-model-request.v1"] = (
        SINGLE_AGENT_MODEL_REQUEST_VERSION
    )
    case_id: Identifier
    model_fixture_id: Identifier
    question: WorkerQuery
    workers: tuple[WorkerKind, ...] = Field(min_length=3, max_length=3)
    evidence_contexts: tuple[EvidenceContext, ...] = ()
    business_facts: tuple[BusinessFact, ...] = ()

    @model_validator(mode="after")
    def context_matches_workers(self) -> SingleAgentModelRequest:
        if set(self.workers) != set(WorkerKind):
            raise ValueError("single agent request must cover every worker")
        if not self.evidence_contexts and not self.business_facts:
            raise ValueError("single agent request requires model-visible context")
        return self


class WorkerOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerFailureCode(StrEnum):
    NO_EVIDENCE = "no_evidence"
    NO_RECORDS = "no_records"
    VERSION_MISMATCH = "version_mismatch"
    INVALID_INPUT = "invalid_input"
    PORT_FAILURE = "port_failure"
    MODEL_FAILURE = "model_failure"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    TIMEOUT = "timeout"
    INVALID_WORKER_OUTPUT = "invalid_worker_output"


class WorkerBranchResult(StrictContract):
    """One branch result; failed branches still carry a valid unresolved Finding."""

    result_version: Literal["worker-result.v1"] = WORKER_RESULT_VERSION
    worker: WorkerKind
    outcome: WorkerOutcome
    findings: tuple[Finding, ...] = Field(min_length=1)
    failure_code: WorkerFailureCode | None = None
    failure_detail: FailureDetail | None = None

    @model_validator(mode="after")
    def result_is_consistent(self) -> WorkerBranchResult:
        if any(finding.worker is not self.worker for finding in self.findings):
            raise ValueError("every finding must belong to the result worker")
        if self.outcome is WorkerOutcome.COMPLETED:
            if self.failure_code is not None or self.failure_detail is not None:
                raise ValueError("completed worker results cannot carry a failure")
        elif self.failure_code is None or self.failure_detail is None:
            raise ValueError("failed worker results require a code and detail")
        return self


class ProcurementReadStatus(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"
    VERSION_MISMATCH = "version_mismatch"
    UPSTREAM_FAILURE = "upstream_failure"


class ProcurementReadResult(StrictContract):
    """Framework-neutral result of the fixed three-tool MCP read sequence."""

    result_version: Literal["procurement-read-result.v1"] = (
        PROCUREMENT_READ_RESULT_VERSION
    )
    status: ProcurementReadStatus
    requested_dataset_version: Identifier
    facts: tuple[BusinessFact, ...] = ()
    detail: FailureDetail | None = None

    @model_validator(mode="after")
    def status_matches_facts(self) -> ProcurementReadResult:
        if any(
            fact.dataset_version != self.requested_dataset_version
            for fact in self.facts
        ):
            raise ValueError("MCP facts must match the requested dataset version")
        if self.status is ProcurementReadStatus.KNOWN:
            if not self.facts or any(
                fact.status is not FactStatus.KNOWN for fact in self.facts
            ):
                raise ValueError("known MCP context requires known facts")
            if self.detail is not None:
                raise ValueError("known MCP context cannot carry failure detail")
        elif self.status is ProcurementReadStatus.UNKNOWN:
            if not self.facts or not any(
                fact.status is FactStatus.UNKNOWN for fact in self.facts
            ):
                raise ValueError("unknown MCP context must preserve an unknown fact")
            if any(
                fact.status not in {FactStatus.KNOWN, FactStatus.UNKNOWN}
                for fact in self.facts
            ):
                raise ValueError("unknown MCP context may preserve only read facts")
            if self.detail is None:
                raise ValueError("unknown MCP context requires detail")
        elif self.detail is None or any(
            fact.status is not FactStatus.KNOWN for fact in self.facts
        ):
            raise ValueError("failed MCP context may preserve only prior known facts")
        return self


@runtime_checkable
class WorkerModelAdapter(Protocol):
    """Replaceable model boundary for versioned test or local real adapters."""

    @property
    def adapter_version(self) -> str: ...

    @property
    def fixture_version(self) -> str: ...

    @property
    def test_only(self) -> bool: ...

    async def generate(self, request: WorkerModelRequest) -> object: ...


@runtime_checkable
class ModelCallHandle(Protocol):
    """One call the adapter has already received; only its result is asynchronous."""

    async def result(self) -> object: ...


@runtime_checkable
class SynchronousModelReceiver(Protocol):
    """The synchronous receive half of a model adapter.

    ``start`` accepts the immutable request and returns a handle *before* any network,
    vendor or fixture wait begins.  Creating a coroutine, a task, an attempt record or an
    intent is not a receipt: only the adapter taking the request does that.
    """

    def start(self, request: WorkerModelRequest) -> ModelCallHandle: ...


class ModelCall:
    """The receipt of one model call: the request the adapter received, and its handle."""

    __slots__ = ("request", "handle")

    def __init__(self, *, request: WorkerModelRequest, handle: ModelCallHandle) -> None:
        self.request = request
        self.handle = handle

    async def result(self) -> object:
        return await self.handle.result()


@runtime_checkable
class ModelInvocationPort(Protocol):
    """Authorize one model call and hand its request to the adapter.

    The Worker hands over a *factory* and a call site, never a built request: the factory
    runs at the one point where the live authorization has just been re-read, so no
    protected context exists before that point and none is carried past a refusal.  The
    returned :class:`ModelCall` names what was actually received; waiting for the result
    never holds the transaction that admitted it.
    """

    def enter(
        self,
        *,
        site: str,
        request_factory: Callable[[], WorkerModelRequest],
    ) -> ModelCall: ...


@runtime_checkable
class ProcurementMCPPort(Protocol):
    """Only procurement capability exposed to the procurement worker."""

    async def collect_review_facts(
        self,
        request: ProcurementWorkerInput,
    ) -> ProcurementReadResult: ...


@runtime_checkable
class ResultConsumptionPort(Protocol):
    """Admit one already-returned result before a Worker adopts or sends it.

    A booked attempt, a passed read admission or an earlier check is not evidence that the
    *next* step is still allowed: a read that has already returned, and a model answer that
    has already come back, are both re-admitted here before anything derived from them is
    built, validated or handed on.  The production runner binds this to the Run and the
    generation it is executing; an offline caller states its own explicit policy instead of
    being handed a silent pass.
    """

    def admit_result(self, *, site: str) -> None: ...


@runtime_checkable
class PolicyWorkerPort(Protocol):
    async def review(self, request: PolicyWorkerInput) -> WorkerBranchResult: ...


@runtime_checkable
class ContractWorkerPort(Protocol):
    async def review(self, request: ContractWorkerInput) -> WorkerBranchResult: ...


@runtime_checkable
class ProcurementWorkerPort(Protocol):
    async def review(self, request: ProcurementWorkerInput) -> WorkerBranchResult: ...
