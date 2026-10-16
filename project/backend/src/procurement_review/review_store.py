"""Versioned PostgreSQL persistence for the V1-03 security and review runtime.

Stage A owns the identity, authorization and resource-registry schema.  Stage B adds
the Run/checkpoint/audit schema through the same migration list, so an empty database
is always brought up by one explicit, re-runnable command.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from .review_schema import REVIEW_MIGRATIONS, migration_checksum
from .security_contracts import (
    CREDENTIAL_DIGEST_ALGORITHM,
    CredentialState,
)
from . import admitted_reads

STORE_SCHEMA_VERSION = "review-store.v1"


class ReviewStoreError(RuntimeError):
    """Base class for explicit storage failures surfaced to the API layer."""


class UnsupportedReviewStorageError(ReviewStoreError):
    """Raised when the configured engine is not PostgreSQL."""


class MigrationError(ReviewStoreError):
    """Raised when the applied schema does not match the expected migration list."""


class SeedConflictError(ReviewStoreError):
    """Raised when a seed would overwrite different immutable registry content."""


class UnknownMigrationVersionError(MigrationError):
    """Raised when the database carries a migration this build does not know."""


@dataclass(frozen=True, slots=True)
class MigrationReport:
    applied: tuple[int, ...]
    already_present: tuple[int, ...]
    total: int


@dataclass(frozen=True, slots=True)
class SeedReport:
    fixture_version: str
    tenants: int
    subjects: int
    cases: int
    documents: int
    retrieval_scores: int
    procurement_records: int
    already_present: bool


@dataclass(frozen=True, slots=True)
class CredentialRow:
    credential_id: str
    tenant_id: str
    subject_id: str
    token_digest: str
    digest_algorithm: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    subject_disabled: bool
    authorization_revision: int
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseRegistration:
    tenant_id: str
    case_id: str
    label: str
    case_type: str
    scenario: str
    purchase_order_id: str
    case_version: str
    document_version: str
    dataset_version: str
    index_version: str
    rules_version: str
    checksum: str
    case_document_ids: tuple[str, ...]
    shared_document_ids: tuple[str, ...]
    allowed_order_ids: tuple[str, ...]
    allowed_supplier_ids: tuple[str, ...]
    allowed_transaction_ids: tuple[str, ...]
    granted_subject_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegisteredDocument:
    document_id: str
    ownership: str
    tenant_id: str | None
    case_id: str | None
    category: str
    source_id: str
    source_version_id: str
    document_version: str
    index_version: str
    content_checksum: str
    chunk_payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RegisteredProcurementRecord:
    tenant_id: str
    case_id: str
    record_kind: str
    record_id: str
    dataset_version: str
    as_of: str
    payload: Mapping[str, Any]

    def payload_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class DenialRow:
    denial_id: str
    sequence: int
    occurred_at: datetime
    request_id: str
    operation: str
    reason: str
    actor_subject_id: str | None
    actor_tenant_id: str | None
    case_id: str | None
    target_digest: str | None


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


class ReviewStore:
    """Narrow PostgreSQL persistence boundary for identity and review state."""

    def __init__(self, engine: Engine) -> None:
        if engine.dialect.name != "postgresql":
            raise UnsupportedReviewStorageError(
                "the review store requires PostgreSQL; no in-memory fallback exists"
            )
        self._engine = engine

    @classmethod
    def from_url(cls, database_url: str) -> ReviewStore:
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
        )
        return cls(engine)

    @property
    def engine(self) -> Engine:
        return self._engine

    def close(self) -> None:
        self._engine.dispose()

    # ------------------------------------------------------------------ schema

    def applied_migrations(self) -> tuple[int, ...]:
        with self._engine.begin() as connection:
            if not _table_exists(connection, "review_schema_migration"):
                return ()
            rows = connection.execute(
                text("SELECT version FROM review_schema_migration ORDER BY version")
            ).scalars()
            return tuple(int(item) for item in rows)

    def migrate(self) -> MigrationReport:
        known = {migration.version: migration for migration in REVIEW_MIGRATIONS}
        applied: list[int] = []
        present: list[int] = []
        with self._engine.begin() as connection:
            existing = ()
            if _table_exists(connection, "review_schema_migration"):
                existing = tuple(
                    int(item)
                    for item in connection.execute(
                        text(
                            "SELECT version FROM review_schema_migration ORDER BY version"
                        )
                    ).scalars()
                )
            unknown = sorted(set(existing) - set(known))
            if unknown:
                raise UnknownMigrationVersionError(
                    "the database carries review migrations this build does not know: "
                    + ", ".join(str(item) for item in unknown)
                )
            for version in sorted(known):
                migration = known[version]
                if version in existing:
                    present.append(version)
                    continue
                for statement in migration.statements:
                    connection.execute(text(statement))
                connection.execute(
                    text(
                        "INSERT INTO review_schema_migration (version, name, checksum) "
                        "VALUES (:version, :name, :checksum)"
                    ),
                    {
                        "version": migration.version,
                        "name": migration.name,
                        "checksum": migration_checksum(migration),
                    },
                )
                applied.append(version)
        return MigrationReport(
            applied=tuple(applied),
            already_present=tuple(present),
            total=len(known),
        )

    def ensure_migrated(self) -> None:
        known = {migration.version for migration in REVIEW_MIGRATIONS}
        if set(self.applied_migrations()) != known:
            raise MigrationError(
                "the review schema is not migrated; run `review_admin migrate` first"
            )

    # ------------------------------------------------------------------- seed

    def seed(self, plan) -> SeedReport:
        """Idempotently load one non-secret synthetic registry fixture."""

        from .security_fixture import SecuritySeedPlan

        if not isinstance(plan, SecuritySeedPlan):
            raise SeedConflictError("seed requires a SecuritySeedPlan")
        checksum = plan.checksum()
        present = False
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    "SELECT checksum FROM review_seed_marker WHERE fixture_version = :v"
                ),
                {"v": plan.fixture_version},
            ).first()
            if row is not None:
                if row[0] != checksum:
                    raise SeedConflictError(
                        "the registry already holds a different definition for "
                        f"{plan.fixture_version}"
                    )
                present = True
            if not present:
                _insert_seed(connection, plan)
                connection.execute(
                    text(
                        "INSERT INTO review_seed_marker (fixture_version, checksum) "
                        "VALUES (:v, :c)"
                    ),
                    {"v": plan.fixture_version, "c": checksum},
                )
        return SeedReport(
            fixture_version=plan.fixture_version,
            tenants=len(plan.tenants),
            subjects=len(plan.subjects),
            cases=len(plan.cases),
            documents=len(plan.documents),
            retrieval_scores=plan.retrieval_score_count,
            procurement_records=len(plan.records),
            already_present=present,
        )

    # --------------------------------------------------------------- identity

    def credential_by_digest(self, token_digest: str) -> CredentialRow | None:
        return self._credential_row("c.token_digest = :key", {"key": token_digest})

    def credential_by_id(self, credential_id: str) -> CredentialRow | None:
        return self._credential_row("c.credential_id = :key", {"key": credential_id})

    def _credential_row(
        self, predicate: str, parameters: dict[str, Any]
    ) -> CredentialRow | None:
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    f"""
                    SELECT c.credential_id, c.tenant_id, c.subject_id, c.token_digest,
                           c.digest_algorithm, c.issued_at, c.expires_at, c.revoked_at,
                           s.disabled, r.revision
                    FROM review_credential c
                    JOIN review_subject s
                      ON s.tenant_id = c.tenant_id AND s.subject_id = c.subject_id
                    JOIN review_authorization_revision r ON r.tenant_id = c.tenant_id
                    WHERE {predicate}
                    """
                ),
                parameters,
            ).first()
            if row is None:
                return None
            roles = tuple(
                connection.execute(
                    text(
                        "SELECT role FROM review_subject_role "
                        "WHERE tenant_id = :t AND subject_id = :s ORDER BY role"
                    ),
                    {"t": row.tenant_id, "s": row.subject_id},
                ).scalars()
            )
        return CredentialRow(
            credential_id=row.credential_id,
            tenant_id=row.tenant_id,
            subject_id=row.subject_id,
            token_digest=row.token_digest,
            digest_algorithm=row.digest_algorithm,
            issued_at=row.issued_at,
            expires_at=row.expires_at,
            revoked_at=row.revoked_at,
            subject_disabled=bool(row.disabled),
            authorization_revision=int(row.revision),
            roles=roles,
        )

    def classify_credential(
        self,
        token_digest: str,
        *,
        now: datetime,
    ) -> tuple[CredentialState, CredentialRow | None]:
        return self._classify(self.credential_by_digest(token_digest), now=now)

    def classify_credential_by_id(
        self,
        credential_id: str,
        *,
        now: datetime,
    ) -> tuple[CredentialState, CredentialRow | None]:
        return self._classify(self.credential_by_id(credential_id), now=now)

    @staticmethod
    def _classify(
        row: CredentialRow | None, *, now: datetime
    ) -> tuple[CredentialState, CredentialRow | None]:
        if row is None:
            return CredentialState.UNKNOWN, None
        if row.digest_algorithm != CREDENTIAL_DIGEST_ALGORITHM:
            return CredentialState.UNKNOWN, row
        if row.revoked_at is not None:
            return CredentialState.REVOKED, row
        if row.subject_disabled:
            return CredentialState.DISABLED_SUBJECT, row
        if row.expires_at <= now:
            return CredentialState.EXPIRED, row
        return CredentialState.ACTIVE, row

    def authorization_revision(self, tenant_id: str) -> int:
        with self._engine.begin() as connection:
            value = connection.execute(
                text(
                    "SELECT revision FROM review_authorization_revision "
                    "WHERE tenant_id = :t"
                ),
                {"t": tenant_id},
            ).scalar()
        if value is None:
            raise ReviewStoreError(f"unknown tenant {tenant_id}")
        return int(value)

    def issue_credential(
        self,
        *,
        credential_id: str,
        tenant_id: str,
        subject_id: str,
        token_digest: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO review_credential (
                        credential_id, tenant_id, subject_id, token_digest,
                        digest_algorithm, issued_at, expires_at
                    ) VALUES (
                        :credential_id, :tenant_id, :subject_id, :digest,
                        :algorithm, :issued_at, :expires_at
                    )
                    """
                ),
                {
                    "credential_id": credential_id,
                    "tenant_id": tenant_id,
                    "subject_id": subject_id,
                    "digest": token_digest,
                    "algorithm": CREDENTIAL_DIGEST_ALGORITHM,
                    "issued_at": issued_at,
                    "expires_at": expires_at,
                },
            )

    def revoke_credential(
        self,
        *,
        credential_id: str,
        revoked_at: datetime,
        reason: str,
    ) -> bool:
        with self._engine.begin() as connection:
            # A credential never changes tenant, so this unlocked read is safe and lets
            # the revocation take the shared lock order: revision row first.
            tenant_id = connection.execute(
                text(
                    "SELECT tenant_id FROM review_credential WHERE credential_id = :c"
                ),
                {"c": credential_id},
            ).scalar()
            if tenant_id is None:
                return False
            _lock_revision_for_update(connection, str(tenant_id))
            result = connection.execute(
                text(
                    "UPDATE review_credential SET revoked_at = :at, revoked_reason = :r "
                    "WHERE credential_id = :c AND revoked_at IS NULL"
                ),
                {"at": revoked_at, "r": reason, "c": credential_id},
            )
            if result.rowcount:
                _bump_revision(connection, str(tenant_id))
            return bool(result.rowcount)

    def set_subject_disabled(self, *, subject_id: str, disabled: bool) -> int:
        with self._engine.begin() as connection:
            tenants = sorted(
                str(row[0])
                for row in connection.execute(
                    text("SELECT tenant_id FROM review_subject WHERE subject_id = :s"),
                    {"s": subject_id},
                ).all()
            )
            for tenant_id in tenants:
                _lock_revision_for_update(connection, tenant_id)
            result = connection.execute(
                text("UPDATE review_subject SET disabled = :d WHERE subject_id = :s"),
                {"d": disabled, "s": subject_id},
            )
            affected = int(result.rowcount)
            for tenant_id in tenants:
                _bump_revision(connection, tenant_id)
            return affected

    def set_case_grant(
        self,
        *,
        tenant_id: str,
        case_id: str,
        subject_id: str,
        granted: bool,
        occurred_at: datetime,
    ) -> bool:
        with self._engine.begin() as connection:
            _lock_revision_for_update(connection, tenant_id)
            if granted:
                connection.execute(
                    text(
                        """
                        INSERT INTO review_case_grant (
                            tenant_id, case_id, subject_id, granted_at, revoked_at
                        ) VALUES (:t, :c, :s, :at, NULL)
                        ON CONFLICT (tenant_id, case_id, subject_id)
                        DO UPDATE SET revoked_at = NULL, granted_at = :at
                        """
                    ),
                    {"t": tenant_id, "c": case_id, "s": subject_id, "at": occurred_at},
                )
            else:
                result = connection.execute(
                    text(
                        "UPDATE review_case_grant SET revoked_at = :at "
                        "WHERE tenant_id = :t AND case_id = :c AND subject_id = :s "
                        "AND revoked_at IS NULL"
                    ),
                    {"t": tenant_id, "c": case_id, "s": subject_id, "at": occurred_at},
                )
                if not result.rowcount:
                    return False
            _bump_revision(connection, tenant_id)
            return True

    def credential_metadata(self, credential_id: str) -> dict[str, Any] | None:
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT credential_id, tenant_id, subject_id, digest_algorithm,
                           issued_at, expires_at, revoked_at, revoked_reason
                    FROM review_credential WHERE credential_id = :c
                    """
                ),
                {"c": credential_id},
            ).first()
        return dict(row._mapping) if row is not None else None

    def tenant_ids(self) -> tuple[str, ...]:
        with self._engine.begin() as connection:
            return tuple(
                connection.execute(
                    text("SELECT tenant_id FROM review_tenant ORDER BY tenant_id")
                ).scalars()
            )

    def subject_ids(self, tenant_id: str) -> tuple[str, ...]:
        with self._engine.begin() as connection:
            return tuple(
                connection.execute(
                    text(
                        "SELECT subject_id FROM review_subject WHERE tenant_id = :t "
                        "ORDER BY subject_id"
                    ),
                    {"t": tenant_id},
                ).scalars()
            )

    # --------------------------------------------------------------- registry

    def list_granted_cases(
        self, *, tenant_id: str, subject_id: str
    ) -> tuple[CaseRegistration, ...]:
        with self._engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT c.case_id
                    FROM review_case_registry c
                    JOIN review_case_grant g
                      ON g.tenant_id = c.tenant_id AND g.case_id = c.case_id
                    WHERE c.tenant_id = :t AND g.subject_id = :s
                      AND g.revoked_at IS NULL
                    ORDER BY c.case_id
                    """
                ),
                {"t": tenant_id, "s": subject_id},
            ).scalars()
            case_ids = tuple(rows)
        return tuple(
            item
            for item in (
                self.load_case(
                    tenant_id=tenant_id, case_id=case_id, subject_id=subject_id
                )
                for case_id in case_ids
            )
            if item is not None
        )

    def load_case(
        self,
        *,
        tenant_id: str,
        case_id: str,
        subject_id: str,
    ) -> CaseRegistration | None:
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT c.*, g.revoked_at AS grant_revoked_at
                    FROM review_case_registry c
                    LEFT JOIN review_case_grant g
                      ON g.tenant_id = c.tenant_id AND g.case_id = c.case_id
                     AND g.subject_id = :s
                    WHERE c.tenant_id = :t AND c.case_id = :c
                    """
                ),
                {"t": tenant_id, "c": case_id, "s": subject_id},
            ).first()
            if row is None or row.grant_revoked_at is not None:
                return None
            case_documents = tuple(
                connection.execute(
                    text(
                        "SELECT document_id FROM review_case_document "
                        "WHERE tenant_id = :t AND case_id = :c AND ownership = 'case_document' "
                        "ORDER BY document_id"
                    ),
                    {"t": tenant_id, "c": case_id},
                ).scalars()
            )
            shared_documents = tuple(
                connection.execute(
                    text(
                        "SELECT document_id FROM review_case_document "
                        "WHERE tenant_id = :t AND case_id = :c "
                        "AND ownership = 'shared_fixture' ORDER BY document_id"
                    ),
                    {"t": tenant_id, "c": case_id},
                ).scalars()
            )
            procurement: dict[str, list[str]] = {
                "order": [],
                "supplier": [],
                "transaction": [],
            }
            for kind, record_id in connection.execute(
                text(
                    "SELECT record_kind, record_id FROM review_case_procurement "
                    "WHERE tenant_id = :t AND case_id = :c ORDER BY record_kind, record_id"
                ),
                {"t": tenant_id, "c": case_id},
            ):
                procurement.setdefault(kind, []).append(record_id)
            granted = tuple(
                connection.execute(
                    text(
                        "SELECT subject_id FROM review_case_grant "
                        "WHERE tenant_id = :t AND case_id = :c AND revoked_at IS NULL "
                        "ORDER BY subject_id"
                    ),
                    {"t": tenant_id, "c": case_id},
                ).scalars()
            )
        return CaseRegistration(
            tenant_id=row.tenant_id,
            case_id=row.case_id,
            label=row.label,
            case_type=row.case_type,
            scenario=row.scenario,
            purchase_order_id=row.purchase_order_id,
            case_version=row.case_version,
            document_version=row.document_version,
            dataset_version=row.dataset_version,
            index_version=row.index_version,
            rules_version=row.rules_version,
            checksum=row.checksum,
            case_document_ids=case_documents,
            shared_document_ids=shared_documents,
            allowed_order_ids=tuple(procurement.get("order", ())),
            allowed_supplier_ids=tuple(procurement.get("supplier", ())),
            allowed_transaction_ids=tuple(procurement.get("transaction", ())),
            granted_subject_ids=granted,
        )

    def case_granted(self, *, tenant_id: str, case_id: str, subject_id: str) -> bool:
        with self._engine.begin() as connection:
            return bool(
                connection.execute(
                    text(
                        "SELECT 1 FROM review_case_grant WHERE tenant_id = :t "
                        "AND case_id = :c AND subject_id = :s AND revoked_at IS NULL"
                    ),
                    {"t": tenant_id, "c": case_id, "s": subject_id},
                ).first()
            )

    def case_exists(self, *, tenant_id: str, case_id: str) -> bool:
        with self._engine.begin() as connection:
            return bool(
                connection.execute(
                    text(
                        "SELECT 1 FROM review_case_registry "
                        "WHERE tenant_id = :t AND case_id = :c"
                    ),
                    {"t": tenant_id, "c": case_id},
                ).first()
            )

    def load_documents(
        self,
        *,
        case_document_ids: Sequence[str],
        shared_document_ids: Sequence[str],
        tenant_id: str,
        case_id: str,
        document_version: str,
        index_version: str,
    ) -> tuple[RegisteredDocument, ...]:
        with self._engine.begin() as connection:
            return admitted_reads.load_documents(
                connection,
                case_document_ids=case_document_ids,
                shared_document_ids=shared_document_ids,
                tenant_id=tenant_id,
                case_id=case_id,
                document_version=document_version,
                index_version=index_version,
            )

    def retrieval_scores(
        self,
        *,
        document_version: str,
        index_version: str,
        chunk_ids: Sequence[str],
    ) -> dict[str, dict[str, float]]:
        if not chunk_ids:
            return {}
        with self._engine.begin() as connection:
            return admitted_reads.retrieval_scores(
                connection,
                document_version=document_version,
                index_version=index_version,
                chunk_ids=chunk_ids,
            )

    def procurement_records(
        self,
        *,
        tenant_id: str,
        case_id: str,
        dataset_version: str,
    ) -> tuple[RegisteredProcurementRecord, ...]:
        with self._engine.begin() as connection:
            return admitted_reads.procurement_records(
                connection,
                tenant_id=tenant_id,
                case_id=case_id,
                dataset_version=dataset_version,
            )

    # ------------------------------------------------------------- denials

    def append_denial(
        self,
        *,
        denial_id: str,
        occurred_at: datetime,
        request_id: str,
        operation: str,
        reason: str,
        actor_subject_id: str | None,
        actor_tenant_id: str | None,
        case_id: str | None,
        target_digest: str | None,
    ) -> int | None:
        """Append one denial record; returns the database-assigned sequence."""

        try:
            with self._engine.begin() as connection:
                sequence = connection.execute(
                    text(
                        """
                        INSERT INTO review_access_denial (
                            denial_id, occurred_at, request_id, operation, reason,
                            actor_subject_id, actor_tenant_id, case_id, target_digest
                        ) VALUES (
                            :denial_id, :occurred_at, :request_id, :operation, :reason,
                            :subject, :tenant, :case_id, :target
                        ) RETURNING sequence
                        """
                    ),
                    {
                        "denial_id": denial_id,
                        "occurred_at": occurred_at,
                        "request_id": request_id,
                        "operation": operation,
                        "reason": reason,
                        "subject": actor_subject_id,
                        "tenant": actor_tenant_id,
                        "case_id": case_id,
                        "target": target_digest,
                    },
                ).scalar_one()
        except SQLAlchemyError:
            return None
        return int(sequence)

    def list_case_denials(
        self, *, tenant_id: str, case_id: str, limit: int = 200
    ) -> tuple[DenialRow, ...]:
        with self._engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT * FROM review_access_denial
                    WHERE actor_tenant_id = :t AND case_id = :c
                    ORDER BY sequence LIMIT :limit
                    """
                ),
                {"t": tenant_id, "c": case_id, "limit": limit},
            ).all()
        return tuple(_denial_row(row) for row in rows)

    def list_unanchored_denials(self, limit: int = 200) -> tuple[DenialRow, ...]:
        with self._engine.begin() as connection:
            rows = connection.execute(
                text(
                    "SELECT * FROM review_access_denial WHERE case_id IS NULL "
                    "ORDER BY sequence LIMIT :limit"
                ),
                {"limit": limit},
            ).all()
        return tuple(_denial_row(row) for row in rows)

    def denial_count(self) -> int:
        with self._engine.begin() as connection:
            return int(
                connection.execute(
                    text("SELECT count(*) FROM review_access_denial")
                ).scalar_one()
            )

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Expose one explicit transaction for callers that need multi-statement atomicity."""

        with self._engine.begin() as connection:
            yield connection


def _denial_row(row: Any) -> DenialRow:
    return DenialRow(
        denial_id=row.denial_id,
        sequence=int(row.sequence),
        occurred_at=row.occurred_at,
        request_id=row.request_id,
        operation=row.operation,
        reason=row.reason,
        actor_subject_id=row.actor_subject_id,
        actor_tenant_id=row.actor_tenant_id,
        case_id=row.case_id,
        target_digest=row.target_digest,
    )


def _table_exists(connection: Any, table_name: str) -> bool:
    return bool(
        connection.execute(
            text("SELECT to_regclass(:name) IS NOT NULL"),
            {"name": f"public.{table_name}"},
        ).scalar()
    )


def _lock_revision_for_update(connection: Any, tenant_id: str) -> None:
    """Take the tenant revision row lock that every authorization mutation starts with.

    Live-authority readers hold a ``FOR SHARE`` lock on this row until their transaction
    ends, so a revocation either finishes before a commit's check or waits for it; the
    two can never interleave unnoticed.
    """

    connection.execute(
        text(
            "SELECT revision FROM review_authorization_revision "
            "WHERE tenant_id = :t FOR UPDATE"
        ),
        {"t": tenant_id},
    )


def _bump_revision(connection: Any, tenant_id: str) -> None:
    connection.execute(
        text(
            "UPDATE review_authorization_revision SET revision = revision + 1 "
            "WHERE tenant_id = :t"
        ),
        {"t": tenant_id},
    )


def _insert_seed(connection: Any, plan: Any) -> None:
    for tenant in plan.tenants:
        connection.execute(
            text(
                """
                INSERT INTO review_tenant (tenant_id, display_name)
                VALUES (:t, :n) ON CONFLICT (tenant_id) DO NOTHING
                """
            ),
            {"t": tenant.tenant_id, "n": tenant.display_name},
        )
        connection.execute(
            text(
                """
                INSERT INTO review_authorization_revision (tenant_id, revision)
                VALUES (:t, 1) ON CONFLICT (tenant_id) DO NOTHING
                """
            ),
            {"t": tenant.tenant_id},
        )
    for subject in plan.subjects:
        connection.execute(
            text(
                """
                INSERT INTO review_subject (tenant_id, subject_id, display_name, disabled)
                VALUES (:t, :s, :n, :d)
                ON CONFLICT (tenant_id, subject_id) DO NOTHING
                """
            ),
            {
                "t": subject.tenant_id,
                "s": subject.subject_id,
                "n": subject.display_name,
                "d": subject.disabled,
            },
        )
        for role in subject.roles:
            connection.execute(
                text(
                    """
                    INSERT INTO review_subject_role (tenant_id, subject_id, role)
                    VALUES (:t, :s, :r) ON CONFLICT DO NOTHING
                    """
                ),
                {"t": subject.tenant_id, "s": subject.subject_id, "r": role},
            )
    for case in plan.cases:
        connection.execute(
            text(
                """
                INSERT INTO review_case_registry (
                    tenant_id, case_id, label, case_type, scenario, purchase_order_id,
                    case_version, document_version, dataset_version, index_version,
                    rules_version, checksum
                ) VALUES (
                    :t, :c, :label, :case_type, :scenario, :po, :cv, :dv, :dsv, :iv,
                    :rv, :checksum
                ) ON CONFLICT (tenant_id, case_id) DO NOTHING
                """
            ),
            {
                "t": case.tenant_id,
                "c": case.case_id,
                "label": case.label,
                "case_type": case.case_type,
                "scenario": case.scenario,
                "po": case.purchase_order_id,
                "cv": case.case_version,
                "dv": case.document_version,
                "dsv": case.dataset_version,
                "iv": case.index_version,
                "rv": case.rules_version,
                "checksum": case.checksum(),
            },
        )
        for document_id, ownership in (
            *((item, "case_document") for item in case.case_document_ids),
            *((item, "shared_fixture") for item in case.shared_document_ids),
        ):
            connection.execute(
                text(
                    """
                    INSERT INTO review_case_document (
                        tenant_id, case_id, document_id, ownership
                    ) VALUES (:t, :c, :d, :o) ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "t": case.tenant_id,
                    "c": case.case_id,
                    "d": document_id,
                    "o": ownership,
                },
            )
        for kind, ids in (
            ("order", case.order_ids),
            ("supplier", case.supplier_ids),
            ("transaction", case.transaction_ids),
        ):
            for record_id in ids:
                connection.execute(
                    text(
                        """
                        INSERT INTO review_case_procurement (
                            tenant_id, case_id, record_kind, record_id
                        ) VALUES (:t, :c, :k, :r) ON CONFLICT DO NOTHING
                        """
                    ),
                    {
                        "t": case.tenant_id,
                        "c": case.case_id,
                        "k": kind,
                        "r": record_id,
                    },
                )
        for subject_id in case.granted_subject_ids:
            connection.execute(
                text(
                    """
                    INSERT INTO review_case_grant (
                        tenant_id, case_id, subject_id, granted_at, revoked_at
                    ) VALUES (:t, :c, :s, :at, NULL)
                    ON CONFLICT (tenant_id, case_id, subject_id) DO NOTHING
                    """
                ),
                {
                    "t": case.tenant_id,
                    "c": case.case_id,
                    "s": subject_id,
                    "at": plan.seeded_at,
                },
            )
    for document in plan.documents:
        connection.execute(
            text(
                """
                INSERT INTO review_document_registry (
                    registry_id, document_id, ownership, tenant_id, case_id,
                    category, source_id, source_version_id, document_version,
                    index_version, content_checksum, test_only, chunk_payload
                ) VALUES (
                    :registry_id, :d, :o, :t, :c, :cat, :src, :sv, :dv, :iv,
                    :checksum, true, CAST(:payload AS jsonb)
                ) ON CONFLICT DO NOTHING
                """
            ),
            {
                "registry_id": (
                    f"{document.ownership}:{document.tenant_id or '-'}:"
                    f"{document.case_id or '-'}:{document.document_id}"
                ),
                "d": document.document_id,
                "o": document.ownership,
                "t": document.tenant_id,
                "c": document.case_id,
                "cat": document.category,
                "src": document.source_id,
                "sv": document.source_version_id,
                "dv": document.document_version,
                "iv": document.index_version,
                "checksum": document.content_checksum,
                "payload": _json(document.chunk_payload),
            },
        )
    for query_text, scores in plan.retrieval_scores.items():
        for chunk_id, score in scores.items():
            connection.execute(
                text(
                    """
                    INSERT INTO review_retrieval_score (
                        document_version, index_version, query_text, chunk_id, score
                    ) VALUES (:dv, :iv, :q, :c, :s) ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "dv": plan.document_version,
                    "iv": plan.index_version,
                    "q": query_text,
                    "c": chunk_id,
                    "s": score,
                },
            )
    for record in plan.records:
        connection.execute(
            text(
                """
                INSERT INTO review_procurement_record (
                    tenant_id, case_id, record_kind, record_id, dataset_version,
                    as_of, payload
                ) VALUES (
                    :t, :c, :k, :r, :dv,
                    CAST(:as_of AS date), CAST(:payload AS jsonb)
                ) ON CONFLICT DO NOTHING
                """
            ),
            {
                "t": record.tenant_id,
                "c": record.case_id,
                "k": record.kind,
                "r": record.record_id,
                "dv": record.dataset_version,
                "as_of": record.as_of,
                "payload": _json(record.payload),
            },
        )


def digest_token(token: str) -> str:
    """Return the server-side digest of one presented opaque token."""

    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "CaseRegistration",
    "CredentialRow",
    "DenialRow",
    "MigrationError",
    "MigrationReport",
    "RegisteredDocument",
    "RegisteredProcurementRecord",
    "ReviewStore",
    "ReviewStoreError",
    "STORE_SCHEMA_VERSION",
    "SeedConflictError",
    "SeedReport",
    "UnknownMigrationVersionError",
    "UnsupportedReviewStorageError",
    "digest_token",
    "utcnow",
]
