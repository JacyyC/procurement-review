"""Three isolated read-only review workers with fail-closed model boundaries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable

from pydantic import TypeAdapter, ValidationError

from .contracts import (
    BusinessFact,
    EvidenceRef,
    Finding,
    FindingAssertion,
    FindingStatus,
    RiskLevel,
    WorkerKind,
)
from .retrieval import (
    RetrievalIssue,
    RetrievalPort,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStatus,
)
from .rules import RULE_SET
from .worker_contracts import (
    ContractWorkerInput,
    EvidenceContext,
    ModelInvocationPort,
    PolicyWorkerInput,
    ProcurementMCPPort,
    ProcurementReadResult,
    ProcurementReadStatus,
    ProcurementWorkerInput,
    ResultConsumptionPort,
    WorkerBranchResult,
    WorkerFailureCode,
    WorkerModelAdapter,
    WorkerModelRequest,
    WorkerOutcome,
)

#: The fixed consumption sites each Worker reports.  They name the boundary and the
#: Worker, never a Case, a tenant, a fixture count or a port.
RETRIEVAL_RESULT = "retrieval_result"
RETRIEVAL_READ_FAILED = "retrieval_read_failed"
PROCUREMENT_RESULT = "procurement_result"
PROCUREMENT_READ_FAILED = "procurement_read_failed"
MODEL_RESULT = "model_result"
MODEL_FAILED = "model_failed"
MODEL_ENTRY = "model_entry"

_FINDING_TUPLE_ADAPTER = TypeAdapter(tuple[Finding, ...])
_REQUIRED_BY_WORKER = {
    contract.worker: (finding_key, contract.category)
    for finding_key, contract in RULE_SET.required_finding_contracts.items()
}
if set(_REQUIRED_BY_WORKER) != set(WorkerKind):
    raise RuntimeError("rules must define exactly one required finding per worker")
if len(_REQUIRED_BY_WORKER) != len(RULE_SET.required_finding_contracts):
    raise RuntimeError("rules cannot define multiple required findings per worker")


def _canonical_model_json(value: EvidenceRef | BusinessFact | Finding) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _serialized(values: Iterable[EvidenceRef | BusinessFact]) -> set[str]:
    return {_canonical_model_json(value) for value in values}


def failed_worker_result(
    worker: WorkerKind,
    code: WorkerFailureCode,
    detail: str,
    *,
    business_facts: tuple[BusinessFact, ...] = (),
) -> WorkerBranchResult:
    """Build the required schema-valid unresolved Finding for a branch failure."""

    finding_key, category = _REQUIRED_BY_WORKER[worker]
    if code in {WorkerFailureCode.NO_EVIDENCE, WorkerFailureCode.NO_RECORDS}:
        status = FindingStatus.INSUFFICIENT_EVIDENCE
    elif code in {
        WorkerFailureCode.VERSION_MISMATCH,
        WorkerFailureCode.INVALID_INPUT,
        WorkerFailureCode.INVALID_MODEL_OUTPUT,
        WorkerFailureCode.INVALID_WORKER_OUTPUT,
    }:
        status = FindingStatus.INVALID_INPUT
    else:
        status = FindingStatus.UPSTREAM_FAILURE
    finding = Finding(
        worker=worker,
        finding_key=finding_key,
        category=category,
        risk_level=RiskLevel.UNKNOWN,
        assertion=FindingAssertion.UNKNOWN,
        statement=f"{worker.value} worker could not produce a supported finding.",
        status=status,
        business_fact_refs=business_facts,
        missing_information=(f"worker_failure:{worker.value}:{code.value}",),
        recommended_action="Require human review after the missing evidence is resolved.",
    )
    return WorkerBranchResult(
        worker=worker,
        outcome=WorkerOutcome.FAILED,
        findings=(finding,),
        failure_code=code,
        failure_detail=detail,
    )


def _require_consumption(consumption: ResultConsumptionPort) -> None:
    """Refuse to assemble a Worker that has no result-consumption policy.

    There is no default and no ``None`` pass: the production runner binds the policy to the
    Run and generation it is executing, and an offline caller has to state its own policy
    explicitly.  A missing argument is already a ``TypeError``; this also rejects a value
    that merely happens to be present.
    """

    if not isinstance(consumption, ResultConsumptionPort):
        raise TypeError("worker requires a result-consumption policy")


def _require_invocation(invocation: ModelInvocationPort) -> None:
    """Refuse to assemble a Worker that has no model-call boundary.

    A consumption policy re-admits a result that has *already* returned; it says nothing
    about whether the next protected step may run, and the two must not be confused.  The
    boundary that authorizes and receives the model call is therefore a separate, mandatory
    capability: the production runner builds one per execution, an offline caller states its
    own named policy, and a Worker without either is not constructible.
    """

    if not isinstance(invocation, ModelInvocationPort):
        raise TypeError("worker requires a model-invocation boundary")


def _require_test_only_model(model: WorkerModelAdapter) -> None:
    if not isinstance(model, WorkerModelAdapter):
        raise TypeError("worker model must implement WorkerModelAdapter")
    if model.test_only is True:
        if not model.adapter_version or not model.fixture_version:
            raise ValueError("test-only model adapter must expose fixed versions")
        return
    if (
        getattr(model, "real_model", False) is True
        and getattr(model, "model_contract_version", None) == "chat-finding-contract.v1"
        and model.adapter_version == "openai-compatible-chat-adapter.v1"
        and model.fixture_version == "chat-finding-contract.v1"
    ):
        return
    raise ValueError(
        "worker model must be explicitly test_only or the versioned real Chat adapter"
    )


def _model_failure_result(
    worker: WorkerKind,
    exc: Exception,
    *,
    business_facts: tuple[BusinessFact, ...] = (),
) -> WorkerBranchResult:
    """Map adapter failures to stable worker outcomes without exposing details."""

    adapter_code = getattr(exc, "code", None)
    if adapter_code == "timeout":
        code = WorkerFailureCode.TIMEOUT
        detail = "model_timeout"
    elif adapter_code in {
        "invalid_json",
        "output_schema",
        "missing_request_id",
        "missing_model",
        "missing_usage",
    }:
        code = WorkerFailureCode.INVALID_MODEL_OUTPUT
        detail = "model_response_contract_invalid"
    else:
        code = WorkerFailureCode.MODEL_FAILURE
        detail = "model_adapter_failure"
    return failed_worker_result(
        worker,
        code,
        detail,
        business_facts=business_facts,
    )


def _validated_model_findings(
    raw_output: object,
    *,
    worker: WorkerKind,
    evidence_refs: tuple[EvidenceRef, ...] = (),
    business_facts: tuple[BusinessFact, ...] = (),
) -> tuple[Finding, ...]:
    findings = _FINDING_TUPLE_ADAPTER.validate_python(raw_output, strict=True)
    if not findings:
        raise ValueError("model output must contain at least one Finding")

    allowed_evidence = _serialized(evidence_refs)
    allowed_facts = _serialized(business_facts)
    required_key, required_category = _REQUIRED_BY_WORKER[worker]
    required_found = False
    for finding in findings:
        if finding.worker is not worker:
            raise ValueError("model output crossed a worker boundary")
        if any(
            _canonical_model_json(item) not in allowed_evidence
            for item in finding.evidence_refs
        ):
            raise ValueError("model output cited evidence outside its worker context")
        if any(
            _canonical_model_json(item) not in allowed_facts
            for item in finding.business_fact_refs
        ):
            raise ValueError("model output cited facts outside its worker context")
        if worker in {WorkerKind.POLICY, WorkerKind.CONTRACT}:
            if finding.business_fact_refs:
                raise ValueError("document workers cannot cite procurement facts")
        elif finding.evidence_refs:
            raise ValueError("procurement worker cannot cite document evidence")
        if finding.finding_key == required_key:
            if finding.category is not required_category:
                raise ValueError("required finding used the wrong risk category")
            required_found = True
    if not required_found:
        raise ValueError("model output omitted the worker's required finding")
    return tuple(sorted(findings, key=_canonical_model_json))


class _DocumentWorker:
    def __init__(
        self,
        *,
        worker: WorkerKind,
        retrieval: RetrievalPort,
        model: WorkerModelAdapter,
        consumption: ResultConsumptionPort,
        invocation: ModelInvocationPort,
    ) -> None:
        if worker not in {WorkerKind.POLICY, WorkerKind.CONTRACT}:
            raise ValueError("document worker kind is invalid")
        if not isinstance(retrieval, RetrievalPort):
            raise TypeError("document worker requires RetrievalPort")
        _require_consumption(consumption)
        _require_invocation(invocation)
        _require_test_only_model(model)
        self._worker = worker
        self._retrieval = retrieval
        self._model = model
        self._consumption = consumption
        self._invocation = invocation

    def _admit_result(self, boundary: str) -> None:
        """Re-admit this generation before anything a returned result carries is used.

        An execution that has lost its authority or lease raises out of here and stops the
        whole generation; this is never turned into one Worker's fail-closed branch result.
        """

        self._consumption.admit_result(site=f"{self._worker.value}.{boundary}")

    async def _review(
        self,
        request: PolicyWorkerInput | ContractWorkerInput,
    ) -> WorkerBranchResult:
        retrieval_request = RetrievalRequest(
            query=request.query,
            index_version=request.index_version,
            limit=10,
            minimum_results=1,
        )
        try:
            raw_response = await asyncio.to_thread(
                self._retrieval.retrieve,
                retrieval_request,
            )
        except Exception:  # noqa: BLE001 - capability ports are untrusted
            # A read that failed is still reported through this generation's authority: a
            # generation that lost it stops instead of summarizing a fail-closed branch.
            self._admit_result(RETRIEVAL_READ_FAILED)
            return failed_worker_result(
                self._worker,
                WorkerFailureCode.PORT_FAILURE,
                "retrieval_port_failure",
            )
        # The read has returned and none of it may be used before the generation is
        # re-checked.  One admission here covers every branch below: the schema check, the
        # version check, the unresolved-issue exits and the known-evidence success path.
        self._admit_result(RETRIEVAL_RESULT)
        if not isinstance(raw_response, RetrievalResponse):
            return failed_worker_result(
                self._worker,
                WorkerFailureCode.PORT_FAILURE,
                "retrieval_response_schema_invalid",
            )
        response = raw_response
        if (
            response.query != request.query
            or response.requested_index_version != request.index_version
            or response.index_version != request.index_version
            or any(
                hit.index_version != request.index_version for hit in response.results
            )
        ):
            return failed_worker_result(
                self._worker,
                WorkerFailureCode.VERSION_MISMATCH,
                "retrieval_response_request_mismatch",
            )
        if response.status is not RetrievalStatus.KNOWN:
            if response.issue in {
                RetrievalIssue.NO_MATCH,
                RetrievalIssue.INSUFFICIENT_EVIDENCE,
            }:
                code = WorkerFailureCode.NO_EVIDENCE
            elif response.issue is RetrievalIssue.INDEX_VERSION_MISMATCH:
                code = WorkerFailureCode.VERSION_MISMATCH
            elif response.status is RetrievalStatus.INVALID_INPUT:
                code = WorkerFailureCode.INVALID_INPUT
            else:
                code = WorkerFailureCode.PORT_FAILURE
            return failed_worker_result(
                self._worker,
                code,
                f"retrieval_{response.issue.value if response.issue else 'failed'}",
            )

        contexts = tuple(
            EvidenceContext(evidence_ref=hit.evidence_ref, content=hit.content)
            for hit in response.results
        )

        # Everything protected — the evidence contexts and the request that carries them —
        # is built *inside* the call boundary, after the live authority has been re-read in
        # the same transaction that lets the adapter receive it.  Building it here, before
        # the boundary, would put the whole context back into the window a revocation can
        # commit in.
        def build_request() -> WorkerModelRequest:
            return WorkerModelRequest(
                worker=self._worker,
                case_id=request.case_id,
                model_fixture_id=request.model_fixture_id,
                question=request.query,
                evidence_contexts=contexts,
            )

        try:
            call = self._invocation.enter(
                site=f"{self._worker.value}.{MODEL_ENTRY}",
                request_factory=build_request,
            )
            # The transaction that admitted the call is already closed: the wait for the
            # answer never holds a database lock.  A lost authority or an unreachable store
            # is a ``BaseException`` signal and travels past this handler as it must.
            raw_findings = await call.result()
        except Exception as exc:  # noqa: BLE001 - the model boundary is untrusted
            self._admit_result(MODEL_FAILED)
            return _model_failure_result(self._worker, exc)
        # A response that arrived after this generation lost its authority is not adopted:
        # it is dropped before validation, wrapping and any later use.
        self._admit_result(MODEL_RESULT)
        try:
            findings = _validated_model_findings(
                raw_findings,
                worker=self._worker,
                evidence_refs=tuple(item.evidence_ref for item in contexts),
            )
        except (TypeError, ValueError, ValidationError):
            return failed_worker_result(
                self._worker,
                WorkerFailureCode.INVALID_MODEL_OUTPUT,
                "test_model_output_invalid",
            )
        return WorkerBranchResult(
            worker=self._worker,
            outcome=WorkerOutcome.COMPLETED,
            findings=findings,
        )


def validate_worker_findings(
    raw_output: object,
    *,
    worker: WorkerKind,
    evidence_refs: tuple[EvidenceRef, ...] = (),
    business_facts: tuple[BusinessFact, ...] = (),
) -> tuple[Finding, ...]:
    """Public entry to the same Finding validation the production workers apply."""

    return _validated_model_findings(
        raw_output,
        worker=worker,
        evidence_refs=evidence_refs,
        business_facts=business_facts,
    )


class PolicyWorker(_DocumentWorker):
    """Policy worker with only Retrieval and test-only model capabilities."""

    def __init__(
        self,
        *,
        retrieval: RetrievalPort,
        model: WorkerModelAdapter,
        consumption: ResultConsumptionPort,
        invocation: ModelInvocationPort,
    ) -> None:
        super().__init__(
            worker=WorkerKind.POLICY,
            retrieval=retrieval,
            model=model,
            consumption=consumption,
            invocation=invocation,
        )

    async def review(self, request: PolicyWorkerInput) -> WorkerBranchResult:
        if not isinstance(request, PolicyWorkerInput):
            return failed_worker_result(
                WorkerKind.POLICY,
                WorkerFailureCode.INVALID_INPUT,
                "policy_worker_input_invalid",
            )
        return await self._review(request)


class ContractWorker(_DocumentWorker):
    """Contract worker with only Retrieval and test-only model capabilities."""

    def __init__(
        self,
        *,
        retrieval: RetrievalPort,
        model: WorkerModelAdapter,
        consumption: ResultConsumptionPort,
        invocation: ModelInvocationPort,
    ) -> None:
        super().__init__(
            worker=WorkerKind.CONTRACT,
            retrieval=retrieval,
            model=model,
            consumption=consumption,
            invocation=invocation,
        )

    async def review(self, request: ContractWorkerInput) -> WorkerBranchResult:
        if not isinstance(request, ContractWorkerInput):
            return failed_worker_result(
                WorkerKind.CONTRACT,
                WorkerFailureCode.INVALID_INPUT,
                "contract_worker_input_invalid",
            )
        return await self._review(request)


class ProcurementAnomalyWorker:
    """Procurement worker with only the official MCP Client port and test model."""

    def __init__(
        self,
        *,
        mcp: ProcurementMCPPort,
        model: WorkerModelAdapter,
        consumption: ResultConsumptionPort,
        invocation: ModelInvocationPort,
    ) -> None:
        if not isinstance(mcp, ProcurementMCPPort):
            raise TypeError("procurement worker requires ProcurementMCPPort")
        _require_consumption(consumption)
        _require_invocation(invocation)
        _require_test_only_model(model)
        self._mcp = mcp
        self._model = model
        self._consumption = consumption
        self._invocation = invocation

    def _admit_result(self, boundary: str) -> None:
        """Re-admit this generation before anything a returned result carries is used."""

        self._consumption.admit_result(
            site=f"{WorkerKind.PROCUREMENT_DATA.value}.{boundary}"
        )

    async def review(self, request: ProcurementWorkerInput) -> WorkerBranchResult:
        if not isinstance(request, ProcurementWorkerInput):
            return failed_worker_result(
                WorkerKind.PROCUREMENT_DATA,
                WorkerFailureCode.INVALID_INPUT,
                "procurement_worker_input_invalid",
            )
        try:
            raw_context = await self._mcp.collect_review_facts(request)
        except Exception:  # noqa: BLE001 - capability ports are untrusted
            self._admit_result(PROCUREMENT_READ_FAILED)
            return failed_worker_result(
                WorkerKind.PROCUREMENT_DATA,
                WorkerFailureCode.PORT_FAILURE,
                "mcp_client_port_failure",
            )
        # The MCP read has returned, and every branch below can carry its ``business_facts``
        # — the version mismatch, the unresolved statuses, the model failure and the invalid
        # output.  One admission here covers all of them, before any of the facts is used.
        self._admit_result(PROCUREMENT_RESULT)
        if not isinstance(raw_context, ProcurementReadResult):
            return failed_worker_result(
                WorkerKind.PROCUREMENT_DATA,
                WorkerFailureCode.PORT_FAILURE,
                "mcp_context_schema_invalid",
            )
        context = raw_context
        if context.requested_dataset_version != request.dataset_version:
            return failed_worker_result(
                WorkerKind.PROCUREMENT_DATA,
                WorkerFailureCode.VERSION_MISMATCH,
                "mcp_context_request_mismatch",
                business_facts=context.facts,
            )
        if context.status is not ProcurementReadStatus.KNOWN:
            if context.status is ProcurementReadStatus.UNKNOWN:
                code = WorkerFailureCode.NO_RECORDS
            elif context.status is ProcurementReadStatus.VERSION_MISMATCH:
                code = WorkerFailureCode.VERSION_MISMATCH
            elif context.status is ProcurementReadStatus.INVALID_INPUT:
                code = WorkerFailureCode.INVALID_INPUT
            else:
                code = WorkerFailureCode.PORT_FAILURE
            return failed_worker_result(
                WorkerKind.PROCUREMENT_DATA,
                code,
                context.detail or "mcp_read_failed",
                business_facts=context.facts,
            )

        # The request that carries the read facts is built inside the call boundary, after
        # the live authority has been re-read in the same transaction that lets the adapter
        # receive it; nothing protected exists before that point.  Only the already-returned
        # read object is retained here, and none of it has been used yet.
        def build_request() -> WorkerModelRequest:
            return WorkerModelRequest(
                worker=WorkerKind.PROCUREMENT_DATA,
                case_id=request.case_id,
                model_fixture_id=request.model_fixture_id,
                business_facts=context.facts,
            )

        try:
            call = self._invocation.enter(
                site=f"{WorkerKind.PROCUREMENT_DATA.value}.{MODEL_ENTRY}",
                request_factory=build_request,
            )
            # Outside the admitting transaction: waiting for the answer holds no lock.  A
            # lost authority or an unreachable store is a ``BaseException`` signal and
            # travels past this handler as it must.
            raw_findings = await call.result()
        except Exception as exc:  # noqa: BLE001 - the model boundary is untrusted
            self._admit_result(MODEL_FAILED)
            return _model_failure_result(
                WorkerKind.PROCUREMENT_DATA,
                exc,
                business_facts=context.facts,
            )
        # A response that arrived after this generation lost its authority is dropped
        # before it is validated or wrapped — including the invalid-output exit below,
        # which would otherwise still carry the branch's facts.
        self._admit_result(MODEL_RESULT)
        try:
            findings = _validated_model_findings(
                raw_findings,
                worker=WorkerKind.PROCUREMENT_DATA,
                business_facts=context.facts,
            )
        except (TypeError, ValueError, ValidationError):
            return failed_worker_result(
                WorkerKind.PROCUREMENT_DATA,
                WorkerFailureCode.INVALID_MODEL_OUTPUT,
                "test_model_output_invalid",
                business_facts=context.facts,
            )
        return WorkerBranchResult(
            worker=WorkerKind.PROCUREMENT_DATA,
            outcome=WorkerOutcome.COMPLETED,
            findings=findings,
        )
