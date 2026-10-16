"""Actual PostgreSQL -> official read-only MCP -> rules 3 boundary."""

import asyncio
from datetime import date

from mcp import Client
import pytest
import test_procurement_store

from procurement_review.business_v3_procurement import review_procurement_v3
from procurement_review.business_v3_types import BusinessContextV3
from procurement_review.mcp_server import create_procurement_mcp_server
from procurement_review.procurement_mcp_client import OfficialProcurementMCPClient
from procurement_review.procurement_store import PostgresProcurementStore
from procurement_review.worker_contracts import ProcurementWorkerInput
from test_procurement_store import _fixture_scope


@pytest.fixture
def postgres_engine():
    yield from test_procurement_store.postgres_engine.__wrapped__()


def test_real_read_boundary_and_zero_model_dependency(postgres_engine):
    async def exercise():
        store = PostgresProcurementStore(postgres_engine)
        loaded = store.load_fixture()
        version = loaded.identity.dataset_version
        scope = _fixture_scope(
            dataset_version=version,
            order_ids=("PO-001",),
            supplier_ids=("SUP-001",),
            transaction_ids=("TXN-001", "TXN-002", "TXN-003"),
        )
        server = create_procurement_mcp_server(store, scope=scope)
        async with Client(server, raise_exceptions=True) as client:
            port = OfficialProcurementMCPClient(client)
            result = await port.collect_review_facts(
                ProcurementWorkerInput(
                    case_id=scope.case_id,
                    purchase_order_id="PO-001",
                    dataset_version=version,
                    model_fixture_id="unused-no-model",
                )
            )
        ctx = BusinessContextV3(
            scope_id=scope.case_id,
            purchase_order_id="PO-001",
            supplier_id="SUP-001",
            contract_id="CONTRACT-LOCAL",
            as_of=date(2026, 6, 30),
            dataset_version=version,
        )
        reviewed = review_procurement_v3(ctx, result, authorized_scope_id=scope.case_id)
        assert not reviewed.failures
        assert reviewed.source_finding.state == "supported"
        assert len(reviewed.transactions) == 3
        assert reviewed.model_request_count == 0
        assert all(e.version == version for e in reviewed.source_finding.evidence)
        denied = review_procurement_v3(ctx, result, authorized_scope_id="OUTSIDE")
        assert denied.order is None and denied.failures == (
            "procurement_scope_mismatch",
        )

    asyncio.run(exercise())
