"""Contract tests for the official-SDK procurement MCP server.

Every product server is built for exactly one :class:`RuntimeScope`; these tests use
the synthetic in-memory fixture reader and build a *real* scope over the same IDs, so
the scope boundary is exercised by the production code path rather than by a stub.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
from datetime import UTC, datetime, timedelta

from mcp import Client
import pytest

from procurement_review.contracts import BusinessFact
from procurement_review import mcp_server
from procurement_review import mcp_startup
from procurement_review.mcp_server import (
    GET_PURCHASE_ORDER_TOOL,
    GET_SUPPLIER_TOOL,
    LIST_HISTORICAL_TRANSACTIONS_TOOL,
    MCP_RESPONSE_SCHEMA_VERSION,
    MCP_SERVER_VERSION,
    READ_ONLY_TOOL_NAMES,
    ProcurementToolErrorCode,
    ProcurementToolOutcome,
    ProcurementToolResponse,
    create_procurement_mcp_server,
)
from procurement_review.mcp_startup import STARTUP_FILE_ENV, McpStartupError
from procurement_review.procurement_store import (
    DatasetVersionMismatchError,
    ProcurementDatasetIdentity,
    ProcurementRecord,
    ProcurementStorageUnavailableError,
    read_fixture,
)
from procurement_review.security_contracts import RuntimeScope

SCOPE_TENANT = "tenant-scope-fixture"
SCOPE_SUBJECT = "subject-scope-fixture"
SCOPE_CASE = "CASE-SCOPE-001"
SCOPE_CASE_VERSION = "case-scope-001.v1"
SCOPE_DOCUMENT = "DOC-SCOPE-001"


class FakeProcurementReader:
    """Deterministic test adapter; it is not a production persistence backend."""

    def __init__(self) -> None:
        fixture = read_fixture()
        payload = fixture.payload
        self._identity = ProcurementDatasetIdentity(
            dataset_version=payload.dataset_version,
            fixture_schema_version=payload.fixture_schema_version,
            envelope_schema_version=fixture.envelope_schema_version,
            generator_version=payload.generator_version,
            seed=payload.seed,
            as_of=payload.as_of,
            digest_algorithm=fixture.digest_algorithm,
            payload_canonicalization=fixture.payload_canonicalization,
            payload_sha256=fixture.payload_sha256,
        )
        self._suppliers = {
            item.supplier_id: item.model_dump(mode="json") for item in payload.suppliers
        }
        self._orders = {
            item.purchase_order_id: item.model_dump(mode="json")
            for item in payload.purchase_orders
        }
        self._transactions = tuple(
            item.model_dump(mode="json") for item in payload.historical_transactions
        )
        self.fail_reads = False
        self.reads: list[tuple[str, str]] = []
        self.scope_denials = 0

    @property
    def identity(self) -> ProcurementDatasetIdentity:
        return self._identity

    def note_scope_denial(self) -> None:
        self.scope_denials += 1

    def _check(self, dataset_version: str) -> None:
        if self.fail_reads:
            raise ProcurementStorageUnavailableError("test-only injected failure")
        if dataset_version != self.identity.dataset_version:
            raise DatasetVersionMismatchError("test-only version mismatch")

    def get_supplier(
        self,
        supplier_id: str,
        dataset_version: str,
    ) -> ProcurementRecord | None:
        self.reads.append(("get_supplier", supplier_id))
        self._check(dataset_version)
        payload = self._suppliers.get(supplier_id)
        if payload is None:
            return None
        return ProcurementRecord(
            record_id=supplier_id,
            dataset_version=dataset_version,
            as_of=self.identity.as_of,
            payload=payload,
        )

    def get_purchase_order(
        self,
        purchase_order_id: str,
        dataset_version: str,
    ) -> ProcurementRecord | None:
        self.reads.append(("get_purchase_order", purchase_order_id))
        self._check(dataset_version)
        payload = self._orders.get(purchase_order_id)
        if payload is None:
            return None
        return ProcurementRecord(
            record_id=purchase_order_id,
            dataset_version=dataset_version,
            as_of=self.identity.as_of,
            payload=payload,
        )

    def list_historical_transactions(
        self,
        *,
        dataset_version: str,
        purchase_order_id: str | None = None,
        supplier_id: str | None = None,
    ) -> tuple[ProcurementRecord, ...]:
        self.reads.append(
            (
                "list_historical_transactions",
                purchase_order_id or supplier_id or "missing-scope",
            )
        )
        self._check(dataset_version)
        records = (
            item
            for item in self._transactions
            if (
                purchase_order_id is not None
                and item["purchase_order_id"] == purchase_order_id
            )
            or (supplier_id is not None and item["supplier_id"] == supplier_id)
        )
        return tuple(
            ProcurementRecord(
                record_id=item["transaction_id"],
                dataset_version=dataset_version,
                as_of=self.identity.as_of,
                payload=item,
            )
            for item in records
        )


def fixture_scope(
    *,
    orders: tuple[str, ...] = ("PO-001",),
    suppliers: tuple[str, ...] = ("SUP-001",),
    transactions: tuple[str, ...] = (
        "TXN-001",
        "TXN-002",
        "TXN-003",
        "TXN-004",
        "TXN-005",
        "TXN-006",
    ),
) -> RuntimeScope:
    """Build one real scope over the synthetic fixture's registered record IDs.

    A caller may register an ID that the pinned dataset does not carry (``PO-999``);
    the tool must then answer ``UNKNOWN`` rather than deny the caller's own scope.
    """

    issued_at = datetime.now(UTC)
    return RuntimeScope(
        tenant_id=SCOPE_TENANT,
        case_id=SCOPE_CASE,
        case_version=SCOPE_CASE_VERSION,
        case_type="purchase_order",
        scenario="low",
        subject_id=SCOPE_SUBJECT,
        credential_id="credential-scope-fixture",
        authorization_revision=1,
        generation=1,
        document_version="retrieval-fixture-v1",
        dataset_version="synthetic-procurement.v1",
        index_version="retrieval-index-v1",
        rules_version="1.0.0",
        case_document_ids=(),
        shared_document_ids=(SCOPE_DOCUMENT,),
        allowed_order_ids=tuple(sorted(set(orders))),
        allowed_supplier_ids=tuple(sorted(set(suppliers))),
        allowed_transaction_ids=tuple(sorted(set(transactions))),
        issued_at=issued_at,
        deadline_at=issued_at + timedelta(seconds=30),
    )


def _parse_response(structured_content: dict[str, object]) -> ProcurementToolResponse:
    return ProcurementToolResponse.model_validate_json(
        json.dumps(structured_content, ensure_ascii=False)
    )


def test_official_sdk_version_is_fixed() -> None:
    assert importlib.metadata.version("mcp") == "2.2.0"


def test_mcp_tool_collection_is_exactly_three_reads_with_stable_schemas() -> None:
    async def exercise() -> None:
        reader = FakeProcurementReader()
        server = create_procurement_mcp_server(reader, scope=fixture_scope())
        async with Client(server, raise_exceptions=True) as client:
            assert client.server_info is not None
            assert client.server_info.version == MCP_SERVER_VERSION
            result = await client.list_tools()

        assert tuple(tool.name for tool in result.tools) == READ_ONLY_TOOL_NAMES
        write_fragments = ("create", "insert", "update", "delete", "approve", "sql")
        for tool in result.tools:
            assert not any(
                fragment in tool.name.lower() for fragment in write_fragments
            )
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False
            assert tool.annotations.idempotent_hint is True
            assert tool.annotations.open_world_hint is False
            assert tool.output_schema is not None
            properties = tool.output_schema["properties"]
            assert properties["response_schema_version"]["const"] == (
                MCP_RESPONSE_SCHEMA_VERSION
            )
            assert properties["read_only"]["const"] is True
            assert tool.input_schema["type"] == "object"

        schemas = {tool.name: tool.input_schema for tool in result.tools}
        assert tuple(schemas[GET_SUPPLIER_TOOL]["properties"]) == (
            "supplier_id",
            "dataset_version",
        )
        assert tuple(schemas[GET_PURCHASE_ORDER_TOOL]["properties"]) == (
            "purchase_order_id",
            "dataset_version",
        )
        assert tuple(schemas[LIST_HISTORICAL_TRANSACTIONS_TOOL]["properties"]) == (
            "dataset_version",
            "purchase_order_id",
            "supplier_id",
        )

    asyncio.run(exercise())


def test_mcp_known_results_map_to_business_facts() -> None:
    async def exercise() -> None:
        reader = FakeProcurementReader()
        server = create_procurement_mcp_server(reader, scope=fixture_scope())
        version = reader.identity.dataset_version
        async with Client(server, raise_exceptions=True) as client:
            supplier_result = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {"supplier_id": "SUP-001", "dataset_version": version},
            )
            order_result = await client.call_tool(
                GET_PURCHASE_ORDER_TOOL,
                {"purchase_order_id": "PO-001", "dataset_version": version},
            )
            order_transactions_result = await client.call_tool(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {"purchase_order_id": "PO-001", "dataset_version": version},
            )
            supplier_transactions_result = await client.call_tool(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {"supplier_id": "SUP-001", "dataset_version": version},
            )

        results = (
            supplier_result,
            order_result,
            order_transactions_result,
            supplier_transactions_result,
        )
        assert all(not item.is_error for item in results)
        responses = tuple(
            _parse_response(item.structured_content)  # type: ignore[arg-type]
            for item in results
        )
        assert tuple(response.outcome.value for response in responses) == (
            "KNOWN",
            "KNOWN",
            "KNOWN",
            "KNOWN",
        )
        assert responses[0].facts[0].record_id == "SUP-001"
        assert responses[1].facts[0].record_id == "PO-001"
        assert tuple(fact.record_id for fact in responses[2].facts) == (
            "TXN-001",
            "TXN-002",
            "TXN-003",
        )
        assert len(responses[3].facts) == 6
        for response in responses:
            assert response.dataset_version == version
            assert response.read_only is True
            for fact in response.facts:
                assert isinstance(fact, BusinessFact)
                assert fact.dataset_version == version
                assert fact.as_of.isoformat() == "2026-06-30T00:00:00+00:00"
                assert fact.read_only is True

    asyncio.run(exercise())


def test_mcp_missing_invalid_version_and_storage_failures_are_explicit() -> None:
    async def exercise() -> None:
        reader = FakeProcurementReader()
        server = create_procurement_mcp_server(
            reader,
            scope=fixture_scope(
                orders=("PO-001", "PO-999"),
                suppliers=("SUP-001", "SUP-999"),
            ),
        )
        version = reader.identity.dataset_version
        async with Client(server, raise_exceptions=True) as client:
            missing = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {"supplier_id": "SUP-999", "dataset_version": version},
            )
            invalid = await client.call_tool(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {
                    "purchase_order_id": "PO-001",
                    "supplier_id": "SUP-001",
                    "dataset_version": version,
                },
            )
            invalid_id = await client.call_tool(
                GET_PURCHASE_ORDER_TOOL,
                {"purchase_order_id": "not-an-id", "dataset_version": version},
            )
            mismatch = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {
                    "supplier_id": "SUP-001",
                    "dataset_version": "synthetic-procurement.v999",
                },
            )
            reader.fail_reads = True
            failed = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {"supplier_id": "SUP-001", "dataset_version": version},
            )

        results = (missing, invalid, invalid_id, mismatch, failed)
        assert all(not item.is_error for item in results)
        responses = tuple(
            _parse_response(item.structured_content)  # type: ignore[arg-type]
            for item in results
        )
        assert tuple(response.outcome.value for response in responses) == (
            "UNKNOWN",
            "INVALID_ARGUMENT",
            "INVALID_ARGUMENT",
            "VERSION_MISMATCH",
            "UPSTREAM_FAILURE",
        )
        assert tuple(
            response.error_code.value if response.error_code else None
            for response in responses
        ) == (
            "NOT_FOUND",
            "INVALID_ARGUMENT",
            "INVALID_ARGUMENT",
            "VERSION_MISMATCH",
            "UPSTREAM_FAILURE",
        )
        assert len(responses[0].facts) == 1
        assert responses[0].facts[0].status.value == "unknown"
        assert responses[0].facts[0].result == {}
        assert all(not response.facts for response in responses[1:])

    asyncio.run(exercise())


def test_mcp_out_of_scope_records_are_denied_before_any_read() -> None:
    async def exercise() -> None:
        reader = FakeProcurementReader()
        # SUP-002 and PO-002 exist in the pinned dataset but are not registered for
        # this Case; SUP-900 does not exist at all.  Both must look identical.
        server = create_procurement_mcp_server(reader, scope=fixture_scope())
        version = reader.identity.dataset_version
        async with Client(server, raise_exceptions=True) as client:
            denied_supplier = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {"supplier_id": "SUP-002", "dataset_version": version},
            )
            denied_absent_supplier = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {"supplier_id": "SUP-900", "dataset_version": version},
            )
            denied_order = await client.call_tool(
                GET_PURCHASE_ORDER_TOOL,
                {"purchase_order_id": "PO-002", "dataset_version": version},
            )
            denied_history = await client.call_tool(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {"supplier_id": "SUP-002", "dataset_version": version},
            )
            allowed_supplier = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {"supplier_id": "SUP-001", "dataset_version": version},
            )

        results = (
            denied_supplier,
            denied_absent_supplier,
            denied_order,
            denied_history,
            allowed_supplier,
        )
        assert all(not item.is_error for item in results)
        responses = tuple(
            _parse_response(item.structured_content)  # type: ignore[arg-type]
            for item in results
        )
        denied = responses[:4]
        for response in denied:
            assert response.outcome is ProcurementToolOutcome.SCOPE_DENIED
            assert response.error_code is ProcurementToolErrorCode.OUT_OF_SCOPE
            assert response.facts == ()
            assert response.error_message is not None
        # The identical shape for an existing and a nonexistent foreign ID means the
        # caller cannot probe another Case's record inventory through the tool.
        assert responses[1] == responses[0]
        assert tuple(item.tool_name for item in denied) == (
            GET_SUPPLIER_TOOL,
            GET_SUPPLIER_TOOL,
            GET_PURCHASE_ORDER_TOOL,
            LIST_HISTORICAL_TRANSACTIONS_TOOL,
        )
        # Scope is checked before the reader is asked for a record.
        assert reader.reads == [("get_supplier", "SUP-001")]
        assert reader.scope_denials == 4
        assert responses[4].outcome is ProcurementToolOutcome.KNOWN
        assert responses[4].facts[0].record_id == "SUP-001"

    asyncio.run(exercise())


def test_mcp_allowed_supplier_history_is_narrowed_to_registered_transactions() -> None:
    async def exercise() -> None:
        reader = FakeProcurementReader()
        # SUP-001 has six transactions in the pinned dataset; this Case only
        # registered two of them.
        server = create_procurement_mcp_server(
            reader,
            scope=fixture_scope(transactions=("TXN-001", "TXN-002")),
        )
        version = reader.identity.dataset_version
        async with Client(server, raise_exceptions=True) as client:
            registered = await client.call_tool(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {"supplier_id": "SUP-001", "dataset_version": version},
            )
            unregistered_order = await client.call_tool(
                LIST_HISTORICAL_TRANSACTIONS_TOOL,
                {"purchase_order_id": "PO-002", "dataset_version": version},
            )

        assert not registered.is_error
        response = _parse_response(registered.structured_content)  # type: ignore[arg-type]
        assert response.outcome is ProcurementToolOutcome.KNOWN
        assert tuple(fact.record_id for fact in response.facts) == (
            "TXN-001",
            "TXN-002",
        )
        # Every returned record was actually read for an allowed supplier, but only
        # the registered subset is exposed.
        assert reader.reads == [("list_historical_transactions", "SUP-001")]
        assert not unregistered_order.is_error
        denied = _parse_response(
            unregistered_order.structured_content  # type: ignore[arg-type]
        )
        assert denied.outcome is ProcurementToolOutcome.SCOPE_DENIED
        assert denied.error_code is ProcurementToolErrorCode.OUT_OF_SCOPE
        assert denied.facts == ()

    asyncio.run(exercise())


def test_mcp_scope_without_overlapping_records_cannot_read_anything() -> None:
    async def exercise() -> None:
        reader = FakeProcurementReader()
        server = create_procurement_mcp_server(
            reader,
            scope=fixture_scope(
                orders=("PO-900",),
                suppliers=("SUP-900",),
                transactions=("TXN-900",),
            ),
        )
        version = reader.identity.dataset_version
        async with Client(server, raise_exceptions=True) as client:
            results = (
                await client.call_tool(
                    GET_SUPPLIER_TOOL,
                    {"supplier_id": "SUP-001", "dataset_version": version},
                ),
                await client.call_tool(
                    GET_PURCHASE_ORDER_TOOL,
                    {"purchase_order_id": "PO-001", "dataset_version": version},
                ),
                await client.call_tool(
                    LIST_HISTORICAL_TRANSACTIONS_TOOL,
                    {"purchase_order_id": "PO-001", "dataset_version": version},
                ),
                await client.call_tool(
                    LIST_HISTORICAL_TRANSACTIONS_TOOL,
                    {"supplier_id": "SUP-001", "dataset_version": version},
                ),
            )

        assert all(not item.is_error for item in results)
        responses = tuple(
            _parse_response(item.structured_content)  # type: ignore[arg-type]
            for item in results
        )
        for response in responses:
            assert response.outcome is ProcurementToolOutcome.SCOPE_DENIED
            assert response.error_code is ProcurementToolErrorCode.OUT_OF_SCOPE
            assert response.facts == ()
        assert reader.reads == []

    asyncio.run(exercise())


def test_mcp_factory_refuses_to_build_without_an_explicit_scope() -> None:
    reader = FakeProcurementReader()
    with pytest.raises(TypeError):
        create_procurement_mcp_server(reader)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        create_procurement_mcp_server(reader, scope=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        create_procurement_mcp_server(reader, scope="unrestricted")  # type: ignore[arg-type]


def test_official_sdk_rejects_missing_or_wrong_typed_arguments_explicitly() -> None:
    async def exercise() -> None:
        server = create_procurement_mcp_server(
            FakeProcurementReader(),
            scope=fixture_scope(),
        )
        async with Client(server, raise_exceptions=False) as client:
            missing = await client.call_tool(GET_SUPPLIER_TOOL, {})
            wrong_type = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {
                    "supplier_id": 1,
                    "dataset_version": "synthetic-procurement.v1",
                },
            )

        for result in (missing, wrong_type):
            assert result.is_error is True
            assert result.structured_content is None
            assert result.content

    asyncio.run(exercise())


def test_stdio_main_loads_the_protected_scope_before_serving_and_always_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    scope = fixture_scope()

    class FakeContext:
        reader = "fixture-reader"

        def __init__(self) -> None:
            self.scope = scope

        def close(self) -> None:
            events.append("close")

    class FakeServer:
        def run(self, *, transport: str) -> None:
            events.append(f"run:{transport}")

    def fake_factory(reader: object, *, scope: RuntimeScope) -> FakeServer:
        events.append(f"create:{reader}")
        assert scope is not None and scope.case_id == SCOPE_CASE
        return FakeServer()

    monkeypatch.setattr(mcp_startup, "load_stdio_startup_context", FakeContext)
    monkeypatch.setattr(mcp_server, "create_procurement_mcp_server", fake_factory)

    assert mcp_server.main() == 0
    assert events == ["create:fixture-reader", "run:stdio", "close"]


def test_stdio_main_has_no_default_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STARTUP_FILE_ENV, raising=False)
    built: list[str] = []

    def fake_factory(reader: object, *, scope: RuntimeScope) -> object:
        built.append(f"{reader}:{scope.case_id}")
        return object()

    monkeypatch.setattr(mcp_server, "create_procurement_mcp_server", fake_factory)

    with pytest.raises(McpStartupError):
        mcp_server.main()
    assert built == []
