from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from mcp import Client
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from datetime import UTC, datetime, timedelta

from procurement_review.mcp_server import (
    GET_SUPPLIER_TOOL,
    create_procurement_mcp_server,
)
from procurement_review.security_contracts import RuntimeScope
from procurement_review.procurement_store import (
    DatasetVersionMismatchError,
    FixtureReferenceError,
    FixtureValidationError,
    PayloadChecksumMismatchError,
    PostgresProcurementStore,
    StoredFixtureMismatchError,
    UnsupportedStorageError,
    read_fixture,
)
from procurement_review.synthetic import DEFAULT_FIXTURE_PATH


def _fixture_json() -> dict[str, object]:
    return json.loads(DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))


def _payload_checksum(payload: object) -> str:
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_default_fixture_has_exact_store_identity_and_scale() -> None:
    fixture = read_fixture()

    assert fixture.payload.dataset_version == "synthetic-procurement.v1"
    assert fixture.payload.generator_version == "procurement-synthetic-generator.v1"
    assert fixture.payload.seed == 20260830
    assert fixture.payload.as_of.isoformat() == "2026-06-30"
    assert fixture.payload_sha256 == (
        "8f224d67342dc6ed45de404a7cd0c2f4c8de4b56f825752491d689cd6c12c6b6"
    )
    assert (
        len(fixture.payload.suppliers),
        len(fixture.payload.purchase_orders),
        len(fixture.payload.historical_transactions),
    ) == (12, 24, 72)


def test_fixture_checksum_tampering_fails_closed(tmp_path: Path) -> None:
    data = _fixture_json()
    data["payload"]["suppliers"][0]["display_name"] = "Synthetic Supplier 99"  # type: ignore[index]
    fixture_path = tmp_path / "tampered.json"
    _write_json(fixture_path, data)

    with pytest.raises(PayloadChecksumMismatchError):
        read_fixture(fixture_path)


def test_fixture_reference_tampering_fails_closed(tmp_path: Path) -> None:
    data = _fixture_json()
    payload = data["payload"]
    payload["purchase_orders"][0]["supplier_id"] = "SUP-999"  # type: ignore[index]
    data["payload_sha256"] = _payload_checksum(payload)
    fixture_path = tmp_path / "bad-reference.json"
    _write_json(fixture_path, data)

    with pytest.raises(FixtureReferenceError):
        read_fixture(fixture_path)


def test_fixture_scale_tampering_fails_closed(tmp_path: Path) -> None:
    data = _fixture_json()
    payload = data["payload"]
    payload["historical_transactions"].append(  # type: ignore[index, union-attr]
        dict(payload["historical_transactions"][0])  # type: ignore[index]
    )
    payload["historical_transactions"][-1]["transaction_id"] = "TXN-999"  # type: ignore[index]
    data["payload_sha256"] = _payload_checksum(payload)
    fixture_path = tmp_path / "wrong-scale.json"
    _write_json(fixture_path, data)

    with pytest.raises(FixtureValidationError, match="exactly 12 suppliers"):
        read_fixture(fixture_path)


def test_non_postgresql_storage_is_explicitly_rejected() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    try:
        with pytest.raises(UnsupportedStorageError, match="requires PostgreSQL"):
            PostgresProcurementStore(engine)
    finally:
        engine.dispose()


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    database_url = os.environ.get("TEST_POSTGRES_DSN")
    if database_url is None:
        pytest.skip("TEST_POSTGRES_DSN is required for PostgreSQL integration")
    engine = create_engine(database_url, pool_pre_ping=True)
    table_names = (
        "procurement_historical_transactions_v1",
        "procurement_purchase_orders_v1",
        "procurement_suppliers_v1",
        "procurement_fixture_datasets_v1",
    )
    with engine.begin() as connection:
        for table_name in table_names:
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
    try:
        yield engine
    finally:
        with engine.begin() as connection:
            for table_name in table_names:
                connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
        engine.dispose()


def test_postgresql_load_is_exact_idempotent_and_queryable(
    postgres_engine: Engine,
) -> None:
    store = PostgresProcurementStore(postgres_engine)

    first = store.load_fixture()
    second = store.load_fixture()

    assert first.inserted is True
    assert second.inserted is False
    assert (
        first.supplier_count,
        first.purchase_order_count,
        first.historical_transaction_count,
    ) == (12, 24, 72)
    assert first.identity == second.identity
    assert first.identity.dataset_version == "synthetic-procurement.v1"
    assert first.identity.generator_version == "procurement-synthetic-generator.v1"
    assert first.identity.seed == 20260830
    assert first.identity.as_of.isoformat() == "2026-06-30"

    supplier = store.get_supplier("SUP-001", first.identity.dataset_version)
    order = store.get_purchase_order("PO-001", first.identity.dataset_version)
    by_order = store.list_historical_transactions(
        dataset_version=first.identity.dataset_version,
        purchase_order_id="PO-001",
    )
    by_supplier = store.list_historical_transactions(
        dataset_version=first.identity.dataset_version,
        supplier_id="SUP-001",
    )

    assert supplier is not None and supplier.payload["supplier_id"] == "SUP-001"
    assert order is not None and order.payload["purchase_order_id"] == "PO-001"
    assert tuple(record.record_id for record in by_order) == (
        "TXN-001",
        "TXN-002",
        "TXN-003",
    )
    assert len(by_supplier) == 6
    assert store.get_supplier("SUP-999", first.identity.dataset_version) is None
    with pytest.raises(DatasetVersionMismatchError):
        store.get_supplier("SUP-001", "synthetic-procurement.v999")

    with postgres_engine.connect() as connection:
        counts = tuple(
            connection.execute(text(f"SELECT count(*) FROM {table_name}")).scalar_one()
            for table_name in (
                "procurement_suppliers_v1",
                "procurement_purchase_orders_v1",
                "procurement_historical_transactions_v1",
            )
        )
    assert counts == (12, 24, 72)


def test_postgresql_concurrent_load_is_idempotent(
    postgres_engine: Engine,
) -> None:
    barrier = Barrier(2)

    def load() -> bool:
        store = PostgresProcurementStore(postgres_engine)
        barrier.wait()
        return store.load_fixture().inserted

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(load), executor.submit(load))
        results = tuple(future.result() for future in futures)

    assert sorted(results) == [False, True]


def test_postgresql_rejects_stored_checksum_mismatch(
    postgres_engine: Engine,
) -> None:
    store = PostgresProcurementStore(postgres_engine)
    store.load_fixture()
    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE procurement_fixture_datasets_v1 SET payload_sha256 = :checksum"
            ),
            {"checksum": "0" * 64},
        )

    with pytest.raises(PayloadChecksumMismatchError):
        store.load_fixture()


def test_postgresql_rejects_changed_stored_record(
    postgres_engine: Engine,
) -> None:
    store = PostgresProcurementStore(postgres_engine)
    store.load_fixture()
    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE procurement_suppliers_v1 "
                "SET payload = jsonb_set(payload, '{display_name}', '"
                '"Tampered"'
                "') WHERE supplier_id = 'SUP-001'"
            )
        )

    with pytest.raises(StoredFixtureMismatchError):
        store.load_fixture()


def _fixture_scope(
    *,
    dataset_version: str,
    order_ids: tuple[str, ...],
    supplier_ids: tuple[str, ...],
    transaction_ids: tuple[str, ...] = (),
) -> RuntimeScope:
    """Build an explicit scope for the frozen synthetic fixture records."""

    now = datetime.now(UTC)
    return RuntimeScope(
        tenant_id="tenant-offline-fixture",
        case_id="CASE-OFFLINE-FIXTURE",
        case_version="case-offline-fixture.v1",
        case_type="purchase_order",
        scenario="low",
        subject_id="subject-offline-fixture",
        credential_id="credential-offline-fixture",
        authorization_revision=1,
        generation=1,
        document_version="retrieval-fixture-v1",
        dataset_version=dataset_version,
        index_version="retrieval-index-v1",
        rules_version="2.0.0",
        case_document_ids=(),
        shared_document_ids=("DOC-OFFLINE-FIXTURE",),
        allowed_order_ids=order_ids,
        allowed_supplier_ids=supplier_ids,
        allowed_transaction_ids=transaction_ids,
        issued_at=now,
        deadline_at=now + timedelta(seconds=60),
    )


def test_official_mcp_client_reads_loaded_postgresql_fixture(
    postgres_engine: Engine,
) -> None:
    async def exercise() -> None:
        store = PostgresProcurementStore(postgres_engine)
        load_report = store.load_fixture()
        # The MCP factory has no unrestricted default: the caller must supply the
        # runtime scope this session is bound to.
        server = create_procurement_mcp_server(
            store,
            scope=_fixture_scope(
                dataset_version=load_report.identity.dataset_version,
                order_ids=("PO-001",),
                supplier_ids=("SUP-001",),
            ),
        )
        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                GET_SUPPLIER_TOOL,
                {
                    "supplier_id": "SUP-001",
                    "dataset_version": load_report.identity.dataset_version,
                },
            )

        assert result.is_error is False
        assert result.structured_content is not None
        assert result.structured_content["outcome"] == "KNOWN"
        assert result.structured_content["read_only"] is True
        assert result.structured_content["facts"][0]["record_id"] == "SUP-001"
        assert result.structured_content["facts"][0]["result"]["supplier_id"] == (
            "SUP-001"
        )

    asyncio.run(exercise())
