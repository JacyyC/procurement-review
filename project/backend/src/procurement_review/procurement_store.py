"""PostgreSQL-backed, read-only access to the synthetic procurement fixture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    ForeignKeyConstraint,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from .synthetic import (
    DATASET_VERSION,
    DEFAULT_FIXTURE_PATH,
    SyntheticFixtureEnvelope,
)

PROCUREMENT_STORE_SCHEMA_VERSION = "procurement-postgresql-store.v1"
_LOAD_ADVISORY_LOCK_KEY = 0x5631303143

_metadata = MetaData()

_datasets = Table(
    "procurement_fixture_datasets_v1",
    _metadata,
    Column("dataset_version", String(160), primary_key=True),
    Column("fixture_schema_version", String(160), nullable=False),
    Column("envelope_schema_version", String(160), nullable=False),
    Column("generator_version", String(160), nullable=False),
    Column("seed", BigInteger, nullable=False),
    Column("as_of", Date, nullable=False),
    Column("digest_algorithm", String(32), nullable=False),
    Column("payload_canonicalization", String(160), nullable=False),
    Column("payload_sha256", String(64), nullable=False),
)

_suppliers = Table(
    "procurement_suppliers_v1",
    _metadata,
    Column("dataset_version", String(160), nullable=False),
    Column("supplier_id", String(160), nullable=False),
    Column("payload", JSONB, nullable=False),
    PrimaryKeyConstraint("dataset_version", "supplier_id"),
    ForeignKeyConstraint(
        ["dataset_version"],
        ["procurement_fixture_datasets_v1.dataset_version"],
        ondelete="RESTRICT",
    ),
)

_purchase_orders = Table(
    "procurement_purchase_orders_v1",
    _metadata,
    Column("dataset_version", String(160), nullable=False),
    Column("purchase_order_id", String(160), nullable=False),
    Column("supplier_id", String(160), nullable=False),
    Column("payload", JSONB, nullable=False),
    PrimaryKeyConstraint("dataset_version", "purchase_order_id"),
    ForeignKeyConstraint(
        ["dataset_version", "supplier_id"],
        [
            "procurement_suppliers_v1.dataset_version",
            "procurement_suppliers_v1.supplier_id",
        ],
        ondelete="RESTRICT",
    ),
)

_historical_transactions = Table(
    "procurement_historical_transactions_v1",
    _metadata,
    Column("dataset_version", String(160), nullable=False),
    Column("transaction_id", String(160), nullable=False),
    Column("purchase_order_id", String(160), nullable=False),
    Column("supplier_id", String(160), nullable=False),
    Column("payload", JSONB, nullable=False),
    PrimaryKeyConstraint("dataset_version", "transaction_id"),
    ForeignKeyConstraint(
        ["dataset_version", "purchase_order_id"],
        [
            "procurement_purchase_orders_v1.dataset_version",
            "procurement_purchase_orders_v1.purchase_order_id",
        ],
        ondelete="RESTRICT",
    ),
    ForeignKeyConstraint(
        ["dataset_version", "supplier_id"],
        [
            "procurement_suppliers_v1.dataset_version",
            "procurement_suppliers_v1.supplier_id",
        ],
        ondelete="RESTRICT",
    ),
)

Index(
    "ix_procurement_orders_v1_dataset_supplier",
    _purchase_orders.c.dataset_version,
    _purchase_orders.c.supplier_id,
)
Index(
    "ix_procurement_transactions_v1_dataset_order",
    _historical_transactions.c.dataset_version,
    _historical_transactions.c.purchase_order_id,
)
Index(
    "ix_procurement_transactions_v1_dataset_supplier",
    _historical_transactions.c.dataset_version,
    _historical_transactions.c.supplier_id,
)


class ProcurementStoreError(RuntimeError):
    """Base class for explicit fixture and PostgreSQL failures."""


class UnsupportedStorageError(ProcurementStoreError):
    """Raised when a non-PostgreSQL engine is supplied."""


class FixtureValidationError(ProcurementStoreError):
    """Raised when the fixture cannot be parsed or violates its schema."""


class FixtureReferenceError(FixtureValidationError):
    """Raised when fixture relationships are inconsistent."""


class DatasetVersionMismatchError(ProcurementStoreError):
    """Raised when callers or stored data select a different dataset version."""


class PayloadChecksumMismatchError(ProcurementStoreError):
    """Raised when a declared or stored payload checksum does not match."""


class FixtureMetadataMismatchError(ProcurementStoreError):
    """Raised when immutable fixture metadata differs for the same version."""


class StoredFixtureMismatchError(ProcurementStoreError):
    """Raised when persisted records are partial or differ from the fixture."""


class InvalidProcurementQueryError(ProcurementStoreError):
    """Raised when a query does not select exactly one supported scope."""


class ProcurementStorageUnavailableError(ProcurementStoreError):
    """Raised when PostgreSQL cannot complete an operation."""


class ProcurementScopeError(ProcurementStoreError):
    """Raised when a read would leave the trusted per-execution runtime scope."""

    def __init__(
        self, message: str = "the requested record is outside the runtime scope"
    ) -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ProcurementDatasetIdentity:
    dataset_version: str
    fixture_schema_version: str
    envelope_schema_version: str
    generator_version: str
    seed: int
    as_of: date
    digest_algorithm: str
    payload_canonicalization: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class ProcurementRecord:
    record_id: str
    dataset_version: str
    as_of: date
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FixtureLoadReport:
    identity: ProcurementDatasetIdentity
    inserted: bool
    supplier_count: int
    purchase_order_count: int
    historical_transaction_count: int


class ProcurementReader(Protocol):
    """Narrow read port consumed by the MCP server and no future Worker directly."""

    @property
    def identity(self) -> ProcurementDatasetIdentity: ...

    def get_supplier(
        self, supplier_id: str, dataset_version: str
    ) -> ProcurementRecord | None: ...

    def get_purchase_order(
        self, purchase_order_id: str, dataset_version: str
    ) -> ProcurementRecord | None: ...

    def list_historical_transactions(
        self,
        *,
        dataset_version: str,
        purchase_order_id: str | None = None,
        supplier_id: str | None = None,
    ) -> tuple[ProcurementRecord, ...]: ...


def _identity_from_fixture(
    fixture: SyntheticFixtureEnvelope,
) -> ProcurementDatasetIdentity:
    payload = fixture.payload
    return ProcurementDatasetIdentity(
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


def read_fixture(
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
) -> SyntheticFixtureEnvelope:
    """Read and strictly validate the authoritative fixture envelope."""

    try:
        raw = fixture_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise FixtureValidationError("fixture could not be read") from exc

    try:
        fixture = SyntheticFixtureEnvelope.model_validate_json(raw)
    except ValidationError as exc:
        message = str(exc)
        if "payload SHA-256 mismatch" in message:
            raise PayloadChecksumMismatchError(
                "fixture payload checksum does not match"
            ) from exc
        reference_markers = (
            "references an unknown",
            "supplier does not match order",
            "subject references do not match",
            "transaction does not belong",
        )
        if any(marker in message for marker in reference_markers):
            raise FixtureReferenceError("fixture references are inconsistent") from exc
        raise FixtureValidationError("fixture schema validation failed") from exc

    counts = (
        len(fixture.payload.suppliers),
        len(fixture.payload.purchase_orders),
        len(fixture.payload.historical_transactions),
    )
    if counts != (12, 24, 72):
        raise FixtureValidationError(
            "fixture must contain exactly 12 suppliers, 24 purchase orders, "
            "and 72 historical transactions"
        )
    return fixture


class PostgresProcurementStore:
    """Loads one immutable synthetic dataset and exposes only read queries."""

    def __init__(
        self,
        engine: Engine,
        *,
        expected_fixture: SyntheticFixtureEnvelope | None = None,
    ) -> None:
        if engine.dialect.name != "postgresql":
            raise UnsupportedStorageError(
                "the procurement store requires PostgreSQL; SQLite is unsupported"
            )
        fixture = expected_fixture or read_fixture()
        self._engine = engine
        self._identity = _identity_from_fixture(fixture)

    @classmethod
    def from_url(
        cls,
        database_url: str,
        *,
        fixture_path: Path = DEFAULT_FIXTURE_PATH,
    ) -> PostgresProcurementStore:
        fixture = read_fixture(fixture_path)
        engine = create_engine(database_url, pool_pre_ping=True)
        return cls(engine, expected_fixture=fixture)

    @property
    def identity(self) -> ProcurementDatasetIdentity:
        return self._identity

    def close(self) -> None:
        self._engine.dispose()

    def create_schema(self) -> None:
        try:
            _metadata.create_all(self._engine)
        except SQLAlchemyError as exc:
            raise ProcurementStorageUnavailableError(
                "PostgreSQL schema initialization failed"
            ) from exc

    def load_fixture(
        self,
        fixture_path: Path = DEFAULT_FIXTURE_PATH,
    ) -> FixtureLoadReport:
        fixture = read_fixture(fixture_path)
        fixture_identity = _identity_from_fixture(fixture)
        self._require_expected_identity(fixture_identity)

        try:
            with self._engine.begin() as connection:
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_key)"),
                    {"lock_key": _LOAD_ADVISORY_LOCK_KEY},
                )
                _metadata.create_all(connection)
                inserted = self._load_or_validate(connection, fixture)
        except ProcurementStoreError:
            raise
        except SQLAlchemyError as exc:
            raise ProcurementStorageUnavailableError(
                "PostgreSQL fixture load failed"
            ) from exc

        return FixtureLoadReport(
            identity=fixture_identity,
            inserted=inserted,
            supplier_count=len(fixture.payload.suppliers),
            purchase_order_count=len(fixture.payload.purchase_orders),
            historical_transaction_count=len(fixture.payload.historical_transactions),
        )

    def get_supplier(
        self, supplier_id: str, dataset_version: str
    ) -> ProcurementRecord | None:
        statement = select(_suppliers.c.payload).where(
            _suppliers.c.dataset_version == dataset_version,
            _suppliers.c.supplier_id == supplier_id,
        )
        return self._get_one(
            statement,
            record_id=supplier_id,
            dataset_version=dataset_version,
        )

    def get_purchase_order(
        self, purchase_order_id: str, dataset_version: str
    ) -> ProcurementRecord | None:
        statement = select(_purchase_orders.c.payload).where(
            _purchase_orders.c.dataset_version == dataset_version,
            _purchase_orders.c.purchase_order_id == purchase_order_id,
        )
        return self._get_one(
            statement,
            record_id=purchase_order_id,
            dataset_version=dataset_version,
        )

    def list_historical_transactions(
        self,
        *,
        dataset_version: str,
        purchase_order_id: str | None = None,
        supplier_id: str | None = None,
    ) -> tuple[ProcurementRecord, ...]:
        if (purchase_order_id is None) == (supplier_id is None):
            raise InvalidProcurementQueryError(
                "select exactly one of purchase_order_id or supplier_id"
            )
        statement = select(
            _historical_transactions.c.transaction_id,
            _historical_transactions.c.payload,
        ).where(_historical_transactions.c.dataset_version == dataset_version)
        if purchase_order_id is not None:
            statement = statement.where(
                _historical_transactions.c.purchase_order_id == purchase_order_id
            )
        else:
            statement = statement.where(
                _historical_transactions.c.supplier_id == supplier_id
            )
        statement = statement.order_by(_historical_transactions.c.transaction_id)

        try:
            with self._engine.connect() as connection:
                identity = self._read_and_validate_identity(connection, dataset_version)
                rows = connection.execute(statement).mappings().all()
        except ProcurementStoreError:
            raise
        except SQLAlchemyError as exc:
            raise ProcurementStorageUnavailableError(
                "PostgreSQL transaction query failed"
            ) from exc

        return tuple(
            ProcurementRecord(
                record_id=row["transaction_id"],
                dataset_version=identity.dataset_version,
                as_of=identity.as_of,
                payload=dict(row["payload"]),
            )
            for row in rows
        )

    def _get_one(
        self,
        statement: Any,
        *,
        record_id: str,
        dataset_version: str,
    ) -> ProcurementRecord | None:
        try:
            with self._engine.connect() as connection:
                identity = self._read_and_validate_identity(connection, dataset_version)
                payload = connection.execute(statement).scalar_one_or_none()
        except ProcurementStoreError:
            raise
        except SQLAlchemyError as exc:
            raise ProcurementStorageUnavailableError(
                "PostgreSQL record query failed"
            ) from exc

        if payload is None:
            return None
        return ProcurementRecord(
            record_id=record_id,
            dataset_version=identity.dataset_version,
            as_of=identity.as_of,
            payload=dict(payload),
        )

    def _load_or_validate(
        self,
        connection: Connection,
        fixture: SyntheticFixtureEnvelope,
    ) -> bool:
        stored_rows = connection.execute(select(_datasets)).mappings().all()
        if stored_rows:
            if len(stored_rows) != 1:
                raise DatasetVersionMismatchError(
                    "the store contains more than one dataset version"
                )
            stored_identity = self._identity_from_row(stored_rows[0])
            self._require_expected_identity(stored_identity)
            self._validate_stored_records(connection, fixture)
            return False

        existing_record_count = sum(
            connection.execute(select(func.count()).select_from(table)).scalar_one()
            for table in (_suppliers, _purchase_orders, _historical_transactions)
        )
        if existing_record_count:
            raise StoredFixtureMismatchError(
                "records exist without immutable dataset metadata"
            )

        identity = _identity_from_fixture(fixture)
        connection.execute(insert(_datasets), asdict(identity))
        connection.execute(
            insert(_suppliers),
            [
                {
                    "dataset_version": identity.dataset_version,
                    "supplier_id": item.supplier_id,
                    "payload": item.model_dump(mode="json"),
                }
                for item in fixture.payload.suppliers
            ],
        )
        connection.execute(
            insert(_purchase_orders),
            [
                {
                    "dataset_version": identity.dataset_version,
                    "purchase_order_id": item.purchase_order_id,
                    "supplier_id": item.supplier_id,
                    "payload": item.model_dump(mode="json"),
                }
                for item in fixture.payload.purchase_orders
            ],
        )
        connection.execute(
            insert(_historical_transactions),
            [
                {
                    "dataset_version": identity.dataset_version,
                    "transaction_id": item.transaction_id,
                    "purchase_order_id": item.purchase_order_id,
                    "supplier_id": item.supplier_id,
                    "payload": item.model_dump(mode="json"),
                }
                for item in fixture.payload.historical_transactions
            ],
        )
        return True

    def _validate_stored_records(
        self,
        connection: Connection,
        fixture: SyntheticFixtureEnvelope,
    ) -> None:
        dataset_version = fixture.payload.dataset_version
        actual_suppliers = {
            row["supplier_id"]: dict(row["payload"])
            for row in connection.execute(
                select(_suppliers.c.supplier_id, _suppliers.c.payload).where(
                    _suppliers.c.dataset_version == dataset_version
                )
            ).mappings()
        }
        expected_suppliers = {
            item.supplier_id: item.model_dump(mode="json")
            for item in fixture.payload.suppliers
        }

        actual_orders = {
            (row["purchase_order_id"], row["supplier_id"]): dict(row["payload"])
            for row in connection.execute(
                select(
                    _purchase_orders.c.purchase_order_id,
                    _purchase_orders.c.supplier_id,
                    _purchase_orders.c.payload,
                ).where(_purchase_orders.c.dataset_version == dataset_version)
            ).mappings()
        }
        expected_orders = {
            (item.purchase_order_id, item.supplier_id): item.model_dump(mode="json")
            for item in fixture.payload.purchase_orders
        }

        actual_transactions = {
            (
                row["transaction_id"],
                row["purchase_order_id"],
                row["supplier_id"],
            ): dict(row["payload"])
            for row in connection.execute(
                select(
                    _historical_transactions.c.transaction_id,
                    _historical_transactions.c.purchase_order_id,
                    _historical_transactions.c.supplier_id,
                    _historical_transactions.c.payload,
                ).where(_historical_transactions.c.dataset_version == dataset_version)
            ).mappings()
        }
        expected_transactions = {
            (item.transaction_id, item.purchase_order_id, item.supplier_id): (
                item.model_dump(mode="json")
            )
            for item in fixture.payload.historical_transactions
        }

        if (
            actual_suppliers != expected_suppliers
            or actual_orders != expected_orders
            or actual_transactions != expected_transactions
        ):
            raise StoredFixtureMismatchError(
                "persisted records do not exactly match the fixture payload"
            )

    def _read_and_validate_identity(
        self,
        connection: Connection,
        dataset_version: str,
    ) -> ProcurementDatasetIdentity:
        if dataset_version != self._identity.dataset_version:
            raise DatasetVersionMismatchError(
                f"expected dataset version {self._identity.dataset_version}"
            )
        row = (
            connection.execute(
                select(_datasets).where(_datasets.c.dataset_version == dataset_version)
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            stored_versions = tuple(
                connection.execute(select(_datasets.c.dataset_version)).scalars()
            )
            if stored_versions:
                raise DatasetVersionMismatchError(
                    "the requested dataset version is not loaded"
                )
            raise StoredFixtureMismatchError("the procurement fixture is not loaded")
        identity = self._identity_from_row(row)
        self._require_expected_identity(identity)
        return identity

    def _require_expected_identity(
        self,
        identity: ProcurementDatasetIdentity,
    ) -> None:
        if identity.dataset_version != self._identity.dataset_version:
            raise DatasetVersionMismatchError(
                f"expected dataset version {self._identity.dataset_version}"
            )
        if identity.payload_sha256 != self._identity.payload_sha256:
            raise PayloadChecksumMismatchError(
                "stored payload checksum does not match the expected fixture"
            )
        if identity != self._identity:
            raise FixtureMetadataMismatchError(
                "immutable fixture metadata differs for the same dataset version"
            )

    @staticmethod
    def _identity_from_row(row: Any) -> ProcurementDatasetIdentity:
        return ProcurementDatasetIdentity(
            dataset_version=row["dataset_version"],
            fixture_schema_version=row["fixture_schema_version"],
            envelope_schema_version=row["envelope_schema_version"],
            generator_version=row["generator_version"],
            seed=row["seed"],
            as_of=row["as_of"],
            digest_algorithm=row["digest_algorithm"],
            payload_canonicalization=row["payload_canonicalization"],
            payload_sha256=row["payload_sha256"],
        )


def ensure_supported_dataset_version(dataset_version: str) -> None:
    if dataset_version != DATASET_VERSION:
        raise DatasetVersionMismatchError(f"expected dataset version {DATASET_VERSION}")
