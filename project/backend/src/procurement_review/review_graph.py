"""LangGraph fan-out/fan-in orchestration with deterministic aggregation."""

from __future__ import annotations

import asyncio
import json
import math
import operator
from collections import Counter
from collections.abc import Iterable
from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field, model_validator

from .contracts import Identifier, ReviewSummary, StrictContract, WorkerKind
from .rules import RULE_SET, RuleSet, aggregate_findings
from .worker_contracts import (
    ContractWorkerInput,
    ContractWorkerPort,
    PolicyWorkerInput,
    PolicyWorkerPort,
    ProcurementWorkerInput,
    ProcurementWorkerPort,
    WorkerBranchResult,
    WorkerFailureCode,
    WorkerOutcome,
)
from .workers import failed_worker_result

REVIEW_STATE_VERSION = "review-state.v1"
REVIEW_EXECUTION_VERSION = "review-execution.v1"
VALIDATE_NODE = "validate_input"
POLICY_NODE = "policy_worker"
CONTRACT_NODE = "contract_worker"
PROCUREMENT_NODE = "procurement_worker"
AGGREGATOR_NODE = "deterministic_aggregator"
WORKER_NODES = (POLICY_NODE, CONTRACT_NODE, PROCUREMENT_NODE)


class ReviewInvocation(StrictContract):
    """Versioned input used to initialize the framework-owned graph state."""

    state_version: Literal["review-state.v1"] = REVIEW_STATE_VERSION
    run_id: Identifier
    case_id: Identifier
    policy_input: PolicyWorkerInput
    contract_input: ContractWorkerInput
    procurement_input: ProcurementWorkerInput

    @model_validator(mode="after")
    def worker_inputs_share_case(self) -> ReviewInvocation:
        if {
            self.policy_input.case_id,
            self.contract_input.case_id,
            self.procurement_input.case_id,
        } != {self.case_id}:
            raise ValueError("all worker inputs must use the review case ID")
        return self


class ReviewState(TypedDict):
    """Versioned LangGraph state; branch updates merge through one reducer."""

    state_version: Literal["review-state.v1"]
    run_id: str
    case_id: str
    policy_input: PolicyWorkerInput
    contract_input: ContractWorkerInput
    procurement_input: ProcurementWorkerInput
    branch_results: Annotated[tuple[WorkerBranchResult, ...], operator.add]
    summary: NotRequired[ReviewSummary]


class ReviewExecution(StrictContract):
    """Stable output independent of branch completion order."""

    execution_version: Literal["review-execution.v1"] = REVIEW_EXECUTION_VERSION
    state_version: Literal["review-state.v1"] = REVIEW_STATE_VERSION
    branch_results: tuple[WorkerBranchResult, ...] = Field(min_length=3, max_length=3)
    summary: ReviewSummary

    @model_validator(mode="after")
    def contains_each_worker_once(self) -> ReviewExecution:
        workers = tuple(result.worker for result in self.branch_results)
        if workers != tuple(sorted(WorkerKind, key=lambda item: item.value)):
            raise ValueError("review execution must contain every worker exactly once")
        return self


def _branch_sort_key(result: WorkerBranchResult) -> tuple[str, str]:
    return (
        result.worker.value,
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def aggregate_worker_results(
    *,
    run_id: str,
    case_id: str,
    branch_results: Iterable[object],
    rule_set: RuleSet = RULE_SET,
) -> ReviewSummary:
    """Validate the fan-in and delegate all risk semantics to the selected rule set.

    The default is the frozen rules 1.0 table; a caller that wants the revised contract has
    to say so, so no replay of a historical batch can pick up new semantics by accident.
    """

    validated: list[WorkerBranchResult] = []
    input_errors: list[str] = []
    for item in branch_results:
        if isinstance(item, WorkerBranchResult):
            validated.append(item)
        else:
            input_errors.append("invalid_worker_branch_result")

    ordered = tuple(sorted(validated, key=_branch_sort_key))
    counts = Counter(result.worker for result in ordered)
    for worker in sorted(WorkerKind, key=lambda item: item.value):
        if counts[worker] == 0:
            input_errors.append(f"missing_worker_branch:{worker.value}")
        elif counts[worker] > 1:
            input_errors.append(f"duplicate_worker_branch:{worker.value}")

    upstream_failures = tuple(
        f"worker_failure:{result.worker.value}:{result.failure_code.value}"
        for result in ordered
        if result.outcome is WorkerOutcome.FAILED and result.failure_code is not None
    )
    findings = tuple(finding for result in ordered for finding in result.findings)
    return aggregate_findings(
        run_id=run_id,
        case_id=case_id,
        findings=findings,
        upstream_failures=upstream_failures,
        input_errors=input_errors,
        rule_set=rule_set,
    )


class ReviewWorkflow:
    """Compiled five-node graph with three concurrent read-only branches."""

    def __init__(
        self,
        *,
        policy_worker: PolicyWorkerPort,
        contract_worker: ContractWorkerPort,
        procurement_worker: ProcurementWorkerPort,
        branch_timeout_seconds: float = 5.0,
        rule_set: RuleSet = RULE_SET,
    ) -> None:
        if not isinstance(policy_worker, PolicyWorkerPort):
            raise TypeError("policy_worker must implement PolicyWorkerPort")
        if not isinstance(contract_worker, ContractWorkerPort):
            raise TypeError("contract_worker must implement ContractWorkerPort")
        if not isinstance(procurement_worker, ProcurementWorkerPort):
            raise TypeError("procurement_worker must implement ProcurementWorkerPort")
        if not isinstance(rule_set, RuleSet):
            raise TypeError("rule_set must be a loaded RuleSet")
        if (
            not isinstance(branch_timeout_seconds, float)
            or not math.isfinite(branch_timeout_seconds)
            or branch_timeout_seconds <= 0.0
        ):
            raise ValueError("branch_timeout_seconds must be a positive finite float")

        self._policy_worker = policy_worker
        self._contract_worker = contract_worker
        self._procurement_worker = procurement_worker
        self._branch_timeout_seconds = branch_timeout_seconds
        self._rule_set = rule_set

        builder = StateGraph(ReviewState)
        builder.add_node(VALIDATE_NODE, self._validate_input)
        builder.add_node(POLICY_NODE, self._run_policy)
        builder.add_node(CONTRACT_NODE, self._run_contract)
        builder.add_node(PROCUREMENT_NODE, self._run_procurement)
        builder.add_node(AGGREGATOR_NODE, self._aggregate)
        builder.add_edge(START, VALIDATE_NODE)
        for worker_node in WORKER_NODES:
            builder.add_edge(VALIDATE_NODE, worker_node)
        builder.add_edge(list(WORKER_NODES), AGGREGATOR_NODE)
        builder.add_edge(AGGREGATOR_NODE, END)
        self.graph = builder.compile(name="procurement_review_v1")

    @staticmethod
    def _validate_input(state: ReviewState) -> dict:
        ReviewInvocation(
            state_version=state["state_version"],
            run_id=state["run_id"],
            case_id=state["case_id"],
            policy_input=state["policy_input"],
            contract_input=state["contract_input"],
            procurement_input=state["procurement_input"],
        )
        return {}

    async def _run_branch(
        self,
        *,
        worker: WorkerKind,
        awaitable,
    ) -> dict[str, tuple[WorkerBranchResult, ...]]:
        try:
            raw_result = await asyncio.wait_for(
                awaitable,
                timeout=self._branch_timeout_seconds,
            )
        except TimeoutError:
            result = failed_worker_result(
                worker,
                WorkerFailureCode.TIMEOUT,
                "worker_branch_timeout",
            )
        except Exception:  # noqa: BLE001 - branch failures must reach fan-in
            result = failed_worker_result(
                worker,
                WorkerFailureCode.PORT_FAILURE,
                "worker_branch_exception",
            )
        else:
            if not isinstance(raw_result, WorkerBranchResult):
                result = failed_worker_result(
                    worker,
                    WorkerFailureCode.INVALID_WORKER_OUTPUT,
                    "worker_branch_result_schema_invalid",
                )
            elif raw_result.worker is not worker:
                result = failed_worker_result(
                    worker,
                    WorkerFailureCode.INVALID_WORKER_OUTPUT,
                    "worker_branch_identity_mismatch",
                )
            else:
                result = raw_result
        return {"branch_results": (result,)}

    async def _run_policy(self, state: ReviewState) -> dict:
        return await self._run_branch(
            worker=WorkerKind.POLICY,
            awaitable=self._policy_worker.review(state["policy_input"]),
        )

    async def _run_contract(self, state: ReviewState) -> dict:
        return await self._run_branch(
            worker=WorkerKind.CONTRACT,
            awaitable=self._contract_worker.review(state["contract_input"]),
        )

    async def _run_procurement(self, state: ReviewState) -> dict:
        return await self._run_branch(
            worker=WorkerKind.PROCUREMENT_DATA,
            awaitable=self._procurement_worker.review(state["procurement_input"]),
        )

    def _aggregate(self, state: ReviewState) -> dict[str, ReviewSummary]:
        return {
            "summary": aggregate_worker_results(
                run_id=state["run_id"],
                case_id=state["case_id"],
                branch_results=state.get("branch_results", ()),
                rule_set=self._rule_set,
            )
        }

    async def run(self, invocation: ReviewInvocation) -> ReviewExecution:
        if not isinstance(invocation, ReviewInvocation):
            raise TypeError("ReviewWorkflow.run requires ReviewInvocation")
        state: ReviewState = {
            "state_version": invocation.state_version,
            "run_id": invocation.run_id,
            "case_id": invocation.case_id,
            "policy_input": invocation.policy_input,
            "contract_input": invocation.contract_input,
            "procurement_input": invocation.procurement_input,
            "branch_results": (),
        }
        completed = await self.graph.ainvoke(state)
        ordered_results = tuple(
            sorted(completed["branch_results"], key=_branch_sort_key)
        )
        return ReviewExecution(
            branch_results=ordered_results,
            summary=completed["summary"],
        )
