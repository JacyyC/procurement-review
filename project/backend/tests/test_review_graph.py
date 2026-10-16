from __future__ import annotations

import asyncio
import importlib.metadata

import pytest
from pydantic import ValidationError

import test_retrieval as retrieval_support
from mcp import Client
from test_mcp_server import FakeProcurementReader
from procurement_review.api_contracts import ReviewCreateRequest
from procurement_review.contracts import (
    FindingAssertion,
    HumanReviewStatus,
    RecommendationStatus,
    RiskLevel,
    WorkerKind,
)
from procurement_review.local_runtime import unscoped_test_scope
from procurement_review.mcp_server import create_procurement_mcp_server
from procurement_review.model_invocation import OfflineModelInvocation
from procurement_review.model_adapters import (
    DeterministicFakeModelAdapter,
    load_worker_model_fixture,
)
from procurement_review.procurement_mcp_client import OfficialProcurementMCPClient
from procurement_review.result_admission import OFFLINE_RESULT_CONSUMPTION
from procurement_review.review_graph import (
    AGGREGATOR_NODE,
    CONTRACT_NODE,
    POLICY_NODE,
    PROCUREMENT_NODE,
    REVIEW_STATE_VERSION,
    VALIDATE_NODE,
    ReviewInvocation,
    ReviewWorkflow,
    aggregate_worker_results,
)
from procurement_review.worker_contracts import (
    ContractWorkerInput,
    PolicyWorkerInput,
    ProcurementWorkerInput,
    WorkerBranchResult,
    WorkerFailureCode,
    WorkerOutcome,
)
from procurement_review.workers import (
    ContractWorker,
    PolicyWorker,
    ProcurementAnomalyWorker,
    failed_worker_result,
)

_FIXTURES = {item.fixture_id: item for item in load_worker_model_fixture().fixtures}


def completed_result(fixture_id: str) -> WorkerBranchResult:
    fixture = _FIXTURES[fixture_id]
    return WorkerBranchResult(
        worker=fixture.worker,
        outcome=WorkerOutcome.COMPLETED,
        findings=fixture.findings,
    )


def invocation(**fixture_updates: str) -> ReviewInvocation:
    fixtures = {
        "policy": "policy-low",
        "contract": "contract-low",
        "procurement": "procurement-low",
    }
    fixtures.update(fixture_updates)
    return ReviewInvocation(
        run_id="RUN-001",
        case_id="CASE-001",
        policy_input=PolicyWorkerInput(
            case_id="CASE-001",
            query="competitive bidding threshold",
            index_version="retrieval-index-v1",
            model_fixture_id=fixtures["policy"],
        ),
        contract_input=ContractWorkerInput(
            case_id="CASE-001",
            query="payment terms termination rights",
            index_version="retrieval-index-v1",
            model_fixture_id=fixtures["contract"],
        ),
        procurement_input=ProcurementWorkerInput(
            case_id="CASE-001",
            purchase_order_id="PO-001",
            dataset_version="synthetic-procurement.v1",
            model_fixture_id=fixtures["procurement"],
        ),
    )


class StubWorker:
    def __init__(
        self,
        result: object,
        *,
        delay: float = 0.0,
        raises: bool = False,
        probe=None,
    ) -> None:
        self.result = result
        self.delay = delay
        self.raises = raises
        self.probe = probe
        self.calls = 0

    async def review(self, request) -> object:
        self.calls += 1
        if self.probe is not None:
            await self.probe.enter()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise RuntimeError("injected branch failure")
        return self.result


class OverlapProbe:
    def __init__(self) -> None:
        self.arrivals = 0
        self.active = 0
        self.max_active = 0
        self.release = asyncio.Event()

    async def enter(self) -> None:
        self.arrivals += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.arrivals == 3:
            self.release.set()
        await asyncio.wait_for(self.release.wait(), timeout=0.5)
        await asyncio.sleep(0)
        self.active -= 1


def workflow(
    *,
    policy: StubWorker | None = None,
    contract: StubWorker | None = None,
    procurement: StubWorker | None = None,
    timeout: float = 1.0,
) -> tuple[ReviewWorkflow, tuple[StubWorker, StubWorker, StubWorker]]:
    workers = (
        policy or StubWorker(completed_result("policy-low")),
        contract or StubWorker(completed_result("contract-low")),
        procurement or StubWorker(completed_result("procurement-low")),
    )
    return (
        ReviewWorkflow(
            policy_worker=workers[0],
            contract_worker=workers[1],
            procurement_worker=workers[2],
            branch_timeout_seconds=timeout,
        ),
        workers,
    )


def test_langgraph_version_nodes_edges_and_explicit_join_are_fixed() -> None:
    graph_workflow, _ = workflow()
    drawable = graph_workflow.graph.get_graph()
    nodes = set(drawable.nodes)
    edges = {(edge.source, edge.target) for edge in drawable.edges}

    assert importlib.metadata.version("langgraph") == "1.2.11"
    assert {
        VALIDATE_NODE,
        POLICY_NODE,
        CONTRACT_NODE,
        PROCUREMENT_NODE,
        AGGREGATOR_NODE,
    }.issubset(nodes)
    assert ("__start__", VALIDATE_NODE) in edges
    assert {
        (VALIDATE_NODE, node) for node in (POLICY_NODE, CONTRACT_NODE, PROCUREMENT_NODE)
    }.issubset(edges)
    assert {
        (node, AGGREGATOR_NODE)
        for node in (POLICY_NODE, CONTRACT_NODE, PROCUREMENT_NODE)
    }.issubset(edges)
    assert (AGGREGATOR_NODE, "__end__") in edges
    assert graph_workflow.graph.checkpointer is None


def test_barrier_proves_three_branches_overlap_and_execute_exactly_once() -> None:
    async def exercise():
        probe = OverlapProbe()
        graph_workflow, workers = workflow(
            policy=StubWorker(completed_result("policy-low"), probe=probe),
            contract=StubWorker(completed_result("contract-low"), probe=probe),
            procurement=StubWorker(completed_result("procurement-low"), probe=probe),
        )
        result = await graph_workflow.run(invocation())
        return probe, workers, result

    probe, workers, result = asyncio.run(exercise())

    assert probe.arrivals == 3
    assert probe.max_active == 3
    assert tuple(worker.calls for worker in workers) == (1, 1, 1)
    assert result.summary.recommendation is RecommendationStatus.RECOMMEND_APPROVE
    assert result.summary.human_review_status is (
        HumanReviewStatus.PENDING_CONFIRMATION
    )
    assert result.summary.is_final_decision is False


def test_real_workers_official_mcp_client_graph_and_aggregator_integrate() -> None:
    async def exercise():
        chunks = tuple(
            retrieval_support._chunk(document["source_id"], document["content"])
            for document in retrieval_support.RETRIEVAL_FIXTURE["documents"]
        )
        retriever = retrieval_support._retriever(chunks)
        model = DeterministicFakeModelAdapter()
        reader = FakeProcurementReader()
        # The server enforces one immutable runtime scope; this offline graph test
        # states the fixture Case it reads PO-001 for.
        server = create_procurement_mcp_server(
            reader,
            scope=unscoped_test_scope(ReviewCreateRequest(case_id="CASE-001")),
        )
        async with Client(server, raise_exceptions=True) as client:
            graph_workflow = ReviewWorkflow(
                policy_worker=PolicyWorker(
                    retrieval=retriever,
                    model=model,
                    consumption=OFFLINE_RESULT_CONSUMPTION,
                    invocation=OfflineModelInvocation(receiver=model),
                ),
                contract_worker=ContractWorker(
                    retrieval=retriever,
                    model=model,
                    consumption=OFFLINE_RESULT_CONSUMPTION,
                    invocation=OfflineModelInvocation(receiver=model),
                ),
                procurement_worker=ProcurementAnomalyWorker(
                    mcp=OfficialProcurementMCPClient(client),
                    model=model,
                    consumption=OFFLINE_RESULT_CONSUMPTION,
                    invocation=OfflineModelInvocation(receiver=model),
                ),
                branch_timeout_seconds=1.0,
            )
            return await graph_workflow.run(invocation())

    result = asyncio.run(exercise())

    assert all(
        item.outcome is WorkerOutcome.COMPLETED for item in result.branch_results
    )
    assert result.summary.rules_version == "1.0.0"
    assert result.summary.recommendation is RecommendationStatus.RECOMMEND_APPROVE
    assert result.summary.human_review_status is (
        HumanReviewStatus.PENDING_CONFIRMATION
    )
    assert result.summary.highest_risk is RiskLevel.LOW
    assert len(result.summary.findings) == 3
    assert sum(len(item.evidence_refs) for item in result.summary.findings) == 2
    assert sum(len(item.business_fact_refs) for item in result.summary.findings) == 5


def test_branch_completion_order_cannot_change_output_or_finding_order() -> None:
    async def run_with_delays(delays: tuple[float, float, float]):
        graph_workflow, _ = workflow(
            policy=StubWorker(completed_result("policy-low"), delay=delays[0]),
            contract=StubWorker(completed_result("contract-low"), delay=delays[1]),
            procurement=StubWorker(
                completed_result("procurement-low"), delay=delays[2]
            ),
        )
        return await graph_workflow.run(invocation())

    first = asyncio.run(run_with_delays((0.03, 0.02, 0.01)))
    second = asyncio.run(run_with_delays((0.01, 0.03, 0.02)))

    assert first.model_dump_json() == second.model_dump_json()
    assert tuple(item.worker.value for item in first.branch_results) == (
        "contract",
        "policy",
        "procurement_data",
    )
    assert tuple(item.finding_key for item in first.summary.findings) == tuple(
        item.finding_key for item in second.summary.findings
    )


def test_timeout_exception_invalid_output_and_partial_failure_reach_fan_in() -> None:
    async def exercise():
        graph_workflow, workers = workflow(
            policy=StubWorker(completed_result("policy-low"), delay=0.05),
            contract=StubWorker(completed_result("contract-low"), raises=True),
            procurement=StubWorker({"invalid": "worker output"}),
            timeout=0.01,
        )
        result = await graph_workflow.run(invocation())
        return workers, result

    workers, result = asyncio.run(exercise())
    by_worker = {item.worker: item for item in result.branch_results}

    assert tuple(worker.calls for worker in workers) == (1, 1, 1)
    assert by_worker[WorkerKind.POLICY].failure_code is WorkerFailureCode.TIMEOUT
    assert by_worker[WorkerKind.CONTRACT].failure_code is (
        WorkerFailureCode.PORT_FAILURE
    )
    assert by_worker[WorkerKind.PROCUREMENT_DATA].failure_code is (
        WorkerFailureCode.INVALID_WORKER_OUTPUT
    )
    assert result.summary.recommendation is (
        RecommendationStatus.INSUFFICIENT_INFORMATION
    )
    assert result.summary.human_review_status is HumanReviewStatus.REQUIRED
    assert len(result.summary.failure_reasons) == 6
    assert (
        sum(
            item.startswith("worker_failure:")
            for item in result.summary.failure_reasons
        )
        == 3
    )
    assert (
        sum(
            item.startswith("finding_failure:")
            for item in result.summary.failure_reasons
        )
        == 3
    )


def test_all_worker_failures_never_recommend_approval() -> None:
    async def exercise():
        graph_workflow, workers = workflow(
            policy=StubWorker(None, raises=True),
            contract=StubWorker(None, raises=True),
            procurement=StubWorker(None, raises=True),
        )
        return workers, await graph_workflow.run(invocation())

    workers, result = asyncio.run(exercise())

    assert tuple(worker.calls for worker in workers) == (1, 1, 1)
    assert all(item.outcome is WorkerOutcome.FAILED for item in result.branch_results)
    assert result.summary.recommendation is (
        RecommendationStatus.INSUFFICIENT_INFORMATION
    )
    assert result.summary.highest_risk is RiskLevel.UNKNOWN
    assert len(result.summary.failure_reasons) == 6


def test_deterministic_aggregator_preserves_high_risk_conflict_and_deduplication() -> (
    None
):
    high = aggregate_worker_results(
        run_id="RUN-HIGH",
        case_id="CASE-001",
        branch_results=(
            completed_result("policy-high"),
            completed_result("contract-low"),
            completed_result("procurement-low"),
        ),
    )
    conflict = aggregate_worker_results(
        run_id="RUN-CONFLICT",
        case_id="CASE-001",
        branch_results=(
            completed_result("policy-conflict-present"),
            completed_result("contract-conflict-absent"),
            completed_result("procurement-low"),
        ),
    )

    policy_base = completed_result("policy-low").findings[0]
    duplicate_low = policy_base.model_copy(
        update={
            "finding_key": "policy.fixture_duplicate",
            "assertion": FindingAssertion.RISK_PRESENT,
            "risk_level": RiskLevel.LOW,
            "statement": "Low duplicate fixture signal.",
        }
    )
    duplicate_high = policy_base.model_copy(
        update={
            "finding_key": "policy.fixture_duplicate",
            "assertion": FindingAssertion.RISK_PRESENT,
            "risk_level": RiskLevel.HIGH,
            "statement": "High duplicate fixture signal.",
        }
    )
    policy_with_duplicates = WorkerBranchResult(
        worker=WorkerKind.POLICY,
        outcome=WorkerOutcome.COMPLETED,
        findings=(policy_base, duplicate_low, duplicate_high),
    )
    deduplicated = aggregate_worker_results(
        run_id="RUN-DEDUP",
        case_id="CASE-001",
        branch_results=(
            completed_result("procurement-low"),
            policy_with_duplicates,
            completed_result("contract-low"),
        ),
    )

    assert high.rules_version == "1.0.0"
    assert high.highest_risk is RiskLevel.HIGH
    assert high.recommendation is RecommendationStatus.RECOMMEND_REJECT
    assert high.human_review_status is HumanReviewStatus.PENDING_CONFIRMATION
    assert conflict.recommendation is RecommendationStatus.NEEDS_HUMAN_REVIEW
    assert conflict.human_review_status is HumanReviewStatus.REQUIRED
    assert conflict.conflicts
    duplicate_findings = [
        item
        for item in deduplicated.findings
        if item.finding_key == "policy.fixture_duplicate"
    ]
    assert len(duplicate_findings) == 1
    assert duplicate_findings[0].risk_level is RiskLevel.HIGH
    assert deduplicated.highest_risk is RiskLevel.HIGH


def test_missing_evidence_and_malformed_fan_in_fail_closed() -> None:
    missing = aggregate_worker_results(
        run_id="RUN-MISSING",
        case_id="CASE-001",
        branch_results=(
            failed_worker_result(
                WorkerKind.POLICY,
                WorkerFailureCode.NO_EVIDENCE,
                "retrieval_no_match",
            ),
            completed_result("contract-low"),
            completed_result("procurement-low"),
        ),
    )
    malformed = aggregate_worker_results(
        run_id="RUN-MALFORMED",
        case_id="CASE-001",
        branch_results=(completed_result("contract-low"), {"invalid": "branch"}),
    )

    assert missing.recommendation is RecommendationStatus.INSUFFICIENT_INFORMATION
    assert any("no_evidence" in item for item in missing.failure_reasons)
    assert malformed.recommendation is (RecommendationStatus.INSUFFICIENT_INFORMATION)
    assert "invalid_worker_branch_result" in malformed.failure_reasons
    assert "missing_worker_branch:policy" in malformed.failure_reasons
    assert "missing_worker_branch:procurement_data" in malformed.failure_reasons


def test_review_state_version_and_case_identity_are_rejected_before_execution() -> None:
    payload = invocation().model_dump(mode="json")
    payload["state_version"] = "review-state.v999"
    with pytest.raises(ValidationError):
        ReviewInvocation.model_validate(payload)

    mismatched = invocation().model_dump(mode="json")
    mismatched["policy_input"]["case_id"] = "CASE-OTHER"
    with pytest.raises(ValidationError, match="review case ID"):
        ReviewInvocation.model_validate(mismatched)

    assert invocation().state_version == REVIEW_STATE_VERSION
