"""Official MCP client mapping tests, including the runtime-scope denial path."""

from __future__ import annotations

import asyncio

from mcp import Client

from test_mcp_server import FakeProcurementReader, fixture_scope
from procurement_review.contracts import FactStatus, FindingStatus, WorkerKind
from procurement_review.mcp_server import create_procurement_mcp_server
from procurement_review.model_adapters import DeterministicFakeModelAdapter
from procurement_review.procurement_mcp_client import OfficialProcurementMCPClient
from procurement_review.model_invocation import OfflineModelInvocation
from procurement_review.result_admission import OFFLINE_RESULT_CONSUMPTION
from procurement_review.worker_contracts import (
    ProcurementMCPPort,
    ProcurementReadStatus,
    ProcurementWorkerInput,
    WorkerFailureCode,
    WorkerModelRequest,
    WorkerOutcome,
)
from procurement_review.workers import ProcurementAnomalyWorker


class RecordingReader(FakeProcurementReader):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str]] = []

    def get_purchase_order(self, purchase_order_id: str, dataset_version: str):
        self.calls.append(("get_purchase_order", purchase_order_id))
        return super().get_purchase_order(purchase_order_id, dataset_version)

    def get_supplier(self, supplier_id: str, dataset_version: str):
        self.calls.append(("get_supplier", supplier_id))
        return super().get_supplier(supplier_id, dataset_version)

    def list_historical_transactions(
        self,
        *,
        dataset_version: str,
        purchase_order_id: str | None = None,
        supplier_id: str | None = None,
    ):
        self.calls.append(
            (
                "list_historical_transactions",
                purchase_order_id or supplier_id or "missing-scope",
            )
        )
        return super().list_historical_transactions(
            dataset_version=dataset_version,
            purchase_order_id=purchase_order_id,
            supplier_id=supplier_id,
        )


class RecordingFakeModel:
    adapter_version = "recording-fake-model-v1"
    fixture_version = "worker-model-fixture-v1"
    test_only = True

    def __init__(self) -> None:
        self.delegate = DeterministicFakeModelAdapter()
        self.requests: list[WorkerModelRequest] = []

    async def generate(self, request: WorkerModelRequest) -> object:
        self.requests.append(request)
        return await self.delegate.generate(request)


def procurement_input(**updates) -> ProcurementWorkerInput:
    values = {
        "case_id": "CASE-001",
        "purchase_order_id": "PO-001",
        "dataset_version": "synthetic-procurement.v1",
        "model_fixture_id": "procurement-low",
    }
    values.update(updates)
    return ProcurementWorkerInput(**values)


def test_procurement_worker_uses_official_client_and_all_three_read_tools() -> None:
    async def exercise():
        reader = RecordingReader()
        server = create_procurement_mcp_server(reader, scope=fixture_scope())
        model = RecordingFakeModel()
        async with Client(server, raise_exceptions=True) as client:
            mcp = OfficialProcurementMCPClient(client)
            assert isinstance(mcp, ProcurementMCPPort)
            worker = ProcurementAnomalyWorker(
                mcp=mcp,
                model=model,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=model),
            )
            result = await worker.review(procurement_input())
        return reader, model, result

    reader, model, result = asyncio.run(exercise())

    assert reader.calls == [
        ("get_purchase_order", "PO-001"),
        ("get_supplier", "SUP-001"),
        ("list_historical_transactions", "PO-001"),
    ]
    assert result.outcome is WorkerOutcome.COMPLETED
    assert result.worker is WorkerKind.PROCUREMENT_DATA
    assert result.findings[0].status is FindingStatus.SUPPORTED
    facts = result.findings[0].business_fact_refs
    assert tuple(fact.record_id for fact in facts) == (
        "PO-001",
        "SUP-001",
        "TXN-001",
        "TXN-002",
        "TXN-003",
    )
    assert all(fact.status is FactStatus.KNOWN for fact in facts)
    assert all(fact.read_only is True for fact in facts)
    assert len(model.requests) == 1
    assert model.requests[0].evidence_contexts == ()
    assert model.requests[0].business_facts == facts


def test_procurement_worker_fails_closed_for_all_mcp_failure_outcomes() -> None:
    async def run_case(request: ProcurementWorkerInput, *, fail_reads: bool = False):
        reader = RecordingReader()
        reader.fail_reads = fail_reads
        # PO-999 is registered for the Case but absent from the pinned dataset, so the
        # tool answers UNKNOWN instead of denying the caller's own scope.
        server = create_procurement_mcp_server(
            reader,
            scope=fixture_scope(orders=("PO-001", "PO-999")),
        )
        model = RecordingFakeModel()
        async with Client(server, raise_exceptions=True) as client:
            worker = ProcurementAnomalyWorker(
                mcp=OfficialProcurementMCPClient(client),
                model=model,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=model),
            )
            result = await worker.review(request)
        return result, model

    async def exercise():
        return (
            await run_case(procurement_input(purchase_order_id="PO-999")),
            await run_case(
                procurement_input(dataset_version="synthetic-procurement.v999")
            ),
            await run_case(procurement_input(purchase_order_id="bad-id")),
            await run_case(procurement_input(), fail_reads=True),
        )

    outcomes = asyncio.run(exercise())
    results = tuple(item[0] for item in outcomes)
    models = tuple(item[1] for item in outcomes)

    assert tuple(result.failure_code for result in results) == (
        WorkerFailureCode.NO_RECORDS,
        WorkerFailureCode.VERSION_MISMATCH,
        WorkerFailureCode.INVALID_INPUT,
        WorkerFailureCode.PORT_FAILURE,
    )
    assert all(result.outcome is WorkerOutcome.FAILED for result in results)
    assert all(
        result.findings[0].status is not FindingStatus.SUPPORTED for result in results
    )
    assert all(model.requests == [] for model in models)
    unknown_fact = results[0].findings[0].business_fact_refs[0]
    assert unknown_fact.record_id == "PO-999"
    assert unknown_fact.status is FactStatus.UNKNOWN
    assert unknown_fact.result == {}


def test_official_client_maps_scope_denial_to_a_terminal_failure() -> None:
    """SCOPE_DENIED is never a KNOWN result and never reaches the data layer."""

    async def exercise():
        reader = RecordingReader()
        server = create_procurement_mcp_server(
            reader,
            scope=fixture_scope(
                orders=("PO-900",),
                suppliers=("SUP-900",),
                transactions=("TXN-900",),
            ),
        )
        model = RecordingFakeModel()
        async with Client(server, raise_exceptions=True) as client:
            mcp = OfficialProcurementMCPClient(client)
            denied = await mcp.collect_review_facts(procurement_input())
            client_calls = list(reader.calls)
            reader.calls.clear()
            worker_result = await ProcurementAnomalyWorker(
                mcp=mcp,
                model=model,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=model),
            ).review(procurement_input())
            worker_calls = list(reader.calls)
        return reader, model, denied, worker_result, client_calls, worker_calls

    reader, model, denied, worker_result, client_calls, worker_calls = asyncio.run(
        exercise()
    )

    assert denied.status is ProcurementReadStatus.UPSTREAM_FAILURE
    assert denied.status is not ProcurementReadStatus.KNOWN
    assert denied.detail == "mcp_upstream_failure"
    assert denied.facts == ()
    assert denied.requested_dataset_version == "synthetic-procurement.v1"
    # The denial happens at the tool boundary: no read is ever issued.
    assert client_calls == []
    assert worker_calls == []
    assert model.requests == []
    assert worker_result.outcome is WorkerOutcome.FAILED
    assert worker_result.failure_code is WorkerFailureCode.PORT_FAILURE
    assert worker_result.findings[0].status is FindingStatus.UPSTREAM_FAILURE
    assert worker_result.findings[0].business_fact_refs == ()


def test_official_client_keeps_only_in_scope_facts_when_a_later_read_is_denied() -> (
    None
):
    async def exercise():
        reader = RecordingReader()
        # The order is registered for this Case; its supplier is not.
        server = create_procurement_mcp_server(
            reader,
            scope=fixture_scope(suppliers=("SUP-900",), transactions=("TXN-900",)),
        )
        model = RecordingFakeModel()
        async with Client(server, raise_exceptions=True) as client:
            mcp = OfficialProcurementMCPClient(client)
            denied = await mcp.collect_review_facts(procurement_input())
            client_calls = list(reader.calls)
            reader.calls.clear()
            worker_result = await ProcurementAnomalyWorker(
                mcp=mcp,
                model=model,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=model),
            ).review(procurement_input())
            worker_calls = list(reader.calls)
        return reader, model, denied, worker_result, client_calls, worker_calls

    reader, model, denied, worker_result, client_calls, worker_calls = asyncio.run(
        exercise()
    )

    assert denied.status is ProcurementReadStatus.UPSTREAM_FAILURE
    assert denied.status is not ProcurementReadStatus.KNOWN
    assert denied.detail == "mcp_upstream_failure"
    # Only the fact that was read inside the scope survives the denial.
    assert tuple(fact.record_id for fact in denied.facts) == ("PO-001",)
    assert all(fact.status is FactStatus.KNOWN for fact in denied.facts)
    assert client_calls == [("get_purchase_order", "PO-001")]
    assert worker_calls == [("get_purchase_order", "PO-001")]
    assert model.requests == []
    assert worker_result.failure_code is WorkerFailureCode.PORT_FAILURE
    assert tuple(
        fact.record_id for fact in worker_result.findings[0].business_fact_refs
    ) == ("PO-001",)


def test_procurement_worker_rejects_invalid_model_output_after_real_mcp_reads() -> None:
    class InvalidFakeModel(RecordingFakeModel):
        async def generate(self, request: WorkerModelRequest) -> object:
            self.requests.append(request)
            return {"not": "a declared Finding tuple"}

    async def exercise():
        reader = RecordingReader()
        server = create_procurement_mcp_server(reader, scope=fixture_scope())
        model = InvalidFakeModel()
        async with Client(server, raise_exceptions=True) as client:
            worker = ProcurementAnomalyWorker(
                mcp=OfficialProcurementMCPClient(client),
                model=model,
                consumption=OFFLINE_RESULT_CONSUMPTION,
                invocation=OfflineModelInvocation(receiver=model),
            )
            return await worker.review(procurement_input())

    result = asyncio.run(exercise())

    assert result.failure_code is WorkerFailureCode.INVALID_MODEL_OUTPUT
    assert result.findings[0].status is FindingStatus.INVALID_INPUT
    assert tuple(fact.record_id for fact in result.findings[0].business_fact_refs) == (
        "PO-001",
        "SUP-001",
        "TXN-001",
        "TXN-002",
        "TXN-003",
    )
