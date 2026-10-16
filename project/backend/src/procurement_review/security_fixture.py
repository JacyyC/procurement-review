"""Synthetic security and business registry fixtures for the V1-03 runtime.

The fixture file holds only non-secret tenant, subject, Case, document and synthetic
procurement definitions.  Credential values are never stored in the repository; the
maintenance CLI issues them at runtime and keeps them in a protected local file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ingestion import (
    DEFAULT_CONFIG,
    IngestedChunk,
    LocatorKind,
    SourceLocator,
    make_chunk_id,
    make_document_id,
    make_source_version_id,
)
from .procurement_store import read_fixture
from .rules import RULES_VERSION_V2

SECURITY_FIXTURE_PATH = (
    Path(__file__).parent / "resources" / "security" / "fixture.v1.json"
)
SECURITY_FIXTURE_VERSION = "security-fixture.v1"
DEMO_FIXTURE_VERSION = "local-demo-fixture.v1"
DEMO_TENANT_ID = "tenant-local-demo"
DEMO_TENANT_LABEL = "Local deterministic demo tenant"

_SHARED_OWNERSHIP = "shared_fixture"
_CASE_OWNERSHIP = "case_document"
_DEMO_SOURCE_VERSION = "Generated Retrieval Fixture, Version 1"


class SecurityFixtureError(RuntimeError):
    """Raised when the synthetic security fixture is internally inconsistent."""


def _checksum(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _content_checksum(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SeedTenant:
    tenant_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class SeedSubject:
    tenant_id: str
    subject_id: str
    display_name: str
    roles: tuple[str, ...]
    case_grants: tuple[str, ...]
    disabled: bool = False


@dataclass(frozen=True, slots=True)
class SeedDocument:
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
    chunk_payload: dict[str, Any]

    @property
    def chunk_id(self) -> str:
        return str(self.chunk_payload["chunk_id"])


@dataclass(frozen=True, slots=True)
class SeedProcurementRecord:
    tenant_id: str
    case_id: str
    kind: str
    record_id: str
    dataset_version: str
    as_of: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SeedCase:
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
    case_document_ids: tuple[str, ...]
    shared_document_ids: tuple[str, ...]
    order_ids: tuple[str, ...]
    supplier_ids: tuple[str, ...]
    transaction_ids: tuple[str, ...]
    granted_subject_ids: tuple[str, ...]

    def checksum(self) -> str:
        return _checksum(
            {
                "tenant_id": self.tenant_id,
                "case_id": self.case_id,
                "case_type": self.case_type,
                "scenario": self.scenario,
                "purchase_order_id": self.purchase_order_id,
                "case_version": self.case_version,
                "document_version": self.document_version,
                "dataset_version": self.dataset_version,
                "index_version": self.index_version,
                "rules_version": self.rules_version,
                "case_document_ids": list(self.case_document_ids),
                "shared_document_ids": list(self.shared_document_ids),
                "order_ids": list(self.order_ids),
                "supplier_ids": list(self.supplier_ids),
                "transaction_ids": list(self.transaction_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class SecuritySeedPlan:
    fixture_version: str
    document_version: str
    dataset_version: str
    index_version: str
    tenants: tuple[SeedTenant, ...]
    subjects: tuple[SeedSubject, ...]
    cases: tuple[SeedCase, ...]
    documents: tuple[SeedDocument, ...]
    records: tuple[SeedProcurementRecord, ...]
    retrieval_scores: dict[str, dict[str, float]]
    seeded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def retrieval_score_count(self) -> int:
        return sum(len(item) for item in self.retrieval_scores.values())

    def case(self, tenant_id: str, case_id: str) -> SeedCase:
        for item in self.cases:
            if item.tenant_id == tenant_id and item.case_id == case_id:
                return item
        raise SecurityFixtureError(f"unknown seeded case {tenant_id}/{case_id}")

    def document(self, document_id: str, *, tenant_id: str | None) -> SeedDocument:
        for item in self.documents:
            if item.document_id == document_id and item.tenant_id == tenant_id:
                return item
        raise SecurityFixtureError(f"unknown seeded document {tenant_id}/{document_id}")

    def checksum(self) -> str:
        return _checksum(
            {
                "fixture_version": self.fixture_version,
                "document_version": self.document_version,
                "dataset_version": self.dataset_version,
                "index_version": self.index_version,
                "tenants": [
                    {"tenant_id": item.tenant_id, "display_name": item.display_name}
                    for item in self.tenants
                ],
                "subjects": [
                    {
                        "tenant_id": item.tenant_id,
                        "subject_id": item.subject_id,
                        "roles": list(item.roles),
                        "case_grants": list(item.case_grants),
                        "disabled": item.disabled,
                    }
                    for item in self.subjects
                ],
                "cases": [item.checksum() for item in self.cases],
                "documents": [
                    {
                        "document_id": item.document_id,
                        "ownership": item.ownership,
                        "tenant_id": item.tenant_id,
                        "case_id": item.case_id,
                        "content_checksum": item.content_checksum,
                        "chunk_id": item.chunk_id,
                    }
                    for item in self.documents
                ],
                "records": [
                    {
                        "tenant_id": item.tenant_id,
                        "case_id": item.case_id,
                        "kind": item.kind,
                        "record_id": item.record_id,
                        "payload": item.payload,
                    }
                    for item in self.records
                ],
                "retrieval_scores": self.retrieval_scores,
            }
        )


def _security_document(
    raw: dict[str, Any],
    *,
    ownership: str,
    tenant_id: str | None,
    case_id: str | None,
    document_version: str,
    index_version: str,
) -> SeedDocument:
    content = str(raw["content"])
    source_id = str(raw["source_id"])
    document_id = str(raw["document_id"])
    content_checksum = _content_checksum(content)
    locator = SourceLocator(
        kind=LocatorKind.TEXT_LINES,
        line_start=1,
        line_end=max(1, content.count("\n") + 1),
    )
    chunk_id = make_chunk_id(
        document_id=document_id,
        ordinal=1,
        locator=locator,
        content_checksum=content_checksum,
        chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
        chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
    )
    payload = {
        "source_id": source_id,
        "source_version": document_version,
        "source_version_id": make_source_version_id(source_id, document_version),
        "source_checksum": content_checksum,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "ordinal": 1,
        "media_type": "text/plain",
        "locator": {
            "kind": locator.kind.value,
            "line_start": locator.line_start,
            "line_end": locator.line_end,
        },
        "content": content,
        "content_checksum": content_checksum,
        "ingestion_version": DEFAULT_CONFIG.ingestion_version,
        "chunk_rules_version": DEFAULT_CONFIG.chunk_rules_version,
        "chunk_rules_checksum": DEFAULT_CONFIG.chunk_rules_checksum,
        "index_version": index_version,
    }
    return SeedDocument(
        document_id=document_id,
        ownership=ownership,
        tenant_id=tenant_id,
        case_id=case_id,
        category=str(raw["category"]),
        source_id=source_id,
        source_version_id=payload["source_version_id"],
        document_version=document_version,
        index_version=index_version,
        content_checksum=content_checksum,
        chunk_payload=payload,
    )


def _business_shared_documents() -> tuple[SeedDocument, ...]:
    """Register the frozen V1-01 retrieval fixture as explicit shared references."""

    from .local_runtime import RETRIEVAL_FIXTURE_PATH, LOCAL_SOURCE_VERSION

    fixture = json.loads(RETRIEVAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    documents: list[SeedDocument] = []
    for document in fixture["documents"]:
        source_id = str(document["source_id"])
        content = str(document["content"])
        source_bytes = content.encode("utf-8")
        source_checksum = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
        source_version_id = make_source_version_id(source_id, LOCAL_SOURCE_VERSION)
        document_id = make_document_id(
            source_id=source_id,
            source_version=LOCAL_SOURCE_VERSION,
            source_checksum=source_checksum,
            media_type="text/plain",
            ingestion_version=DEFAULT_CONFIG.ingestion_version,
        )
        locator = SourceLocator(
            kind=LocatorKind.TEXT_LINES,
            line_start=1,
            line_end=max(1, content.count("\n") + 1),
        )
        chunk_id = make_chunk_id(
            document_id=document_id,
            ordinal=1,
            locator=locator,
            content_checksum=source_checksum,
            chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
            chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
        )
        payload = {
            "source_id": source_id,
            "source_version": LOCAL_SOURCE_VERSION,
            "source_version_id": source_version_id,
            "source_checksum": source_checksum,
            "document_id": document_id,
            "chunk_id": chunk_id,
            "ordinal": 1,
            "media_type": "text/plain",
            "locator": {
                "kind": locator.kind.value,
                "line_start": locator.line_start,
                "line_end": locator.line_end,
            },
            "content": content,
            "content_checksum": source_checksum,
            "ingestion_version": DEFAULT_CONFIG.ingestion_version,
            "chunk_rules_version": DEFAULT_CONFIG.chunk_rules_version,
            "chunk_rules_checksum": DEFAULT_CONFIG.chunk_rules_checksum,
            "index_version": DEFAULT_CONFIG.index_version,
        }
        documents.append(
            SeedDocument(
                document_id=document_id,
                ownership=_SHARED_OWNERSHIP,
                tenant_id=None,
                case_id=None,
                category="shared_reference",
                source_id=source_id,
                source_version_id=source_version_id,
                document_version="retrieval-fixture-v1",
                index_version=DEFAULT_CONFIG.index_version,
                content_checksum=source_checksum,
                chunk_payload=payload,
            )
        )
    return tuple(documents)


def _business_retrieval_scores(
    documents: tuple[SeedDocument, ...],
) -> dict[str, dict[str, float]]:
    from .local_runtime import RETRIEVAL_FIXTURE_PATH

    fixture = json.loads(RETRIEVAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    ids_by_source = {item.source_id: item.chunk_id for item in documents}
    scores: dict[str, dict[str, float]] = {}
    for query in fixture["queries"]:
        scores[str(query["query"])] = {
            ids_by_source[source_id]: float(score)
            for source_id, score in query["fake_vector_scores"].items()
        }
    return scores


def _demo_business_plan(
    documents: tuple[SeedDocument, ...],
    scores: dict[str, dict[str, float]],
) -> tuple[
    tuple[SeedSubject, ...],
    tuple[SeedCase, ...],
    tuple[SeedProcurementRecord, ...],
]:
    from .case_catalog import CASE_CATALOG

    fixture = read_fixture()
    payload = fixture.payload
    orders = {item.purchase_order_id: item for item in payload.purchase_orders}
    suppliers = {item.supplier_id: item for item in payload.suppliers}
    transactions_by_order: dict[str, list[Any]] = {}
    for item in payload.historical_transactions:
        transactions_by_order.setdefault(item.purchase_order_id, []).append(item)

    records: list[SeedProcurementRecord] = []
    cases: list[SeedCase] = []
    for option in CASE_CATALOG:
        order_id = option.purchase_order_id
        order = orders[order_id]
        order_ids = (order_id,)
        supplier_ids = (order.supplier_id,)
        transaction_ids = tuple(
            sorted(
                item.transaction_id for item in transactions_by_order.get(order_id, ())
            )
        )
        subject_ids = (
            "subject-local-demo",
            "subject-local-demo-reader",
            "subject-local-demo-auditor",
        )
        cases.append(
            SeedCase(
                tenant_id=DEMO_TENANT_ID,
                case_id=option.case_id,
                label=option.label,
                case_type=option.case_type.value,
                scenario=option.scenario.value,
                purchase_order_id=order_id,
                case_version=f"{option.case_id.lower()}.v1",
                document_version="retrieval-fixture-v1",
                dataset_version=payload.dataset_version,
                index_version=DEFAULT_CONFIG.index_version,
                rules_version=RULES_VERSION_V2,
                case_document_ids=(),
                shared_document_ids=tuple(
                    sorted(item.document_id for item in documents)
                ),
                order_ids=order_ids,
                supplier_ids=supplier_ids,
                transaction_ids=transaction_ids,
                granted_subject_ids=subject_ids,
            )
        )
        for record_id in order_ids:
            records.append(
                SeedProcurementRecord(
                    tenant_id=DEMO_TENANT_ID,
                    case_id=option.case_id,
                    kind="order",
                    record_id=record_id,
                    dataset_version=payload.dataset_version,
                    as_of=payload.as_of.isoformat(),
                    payload=order.model_dump(mode="json"),
                )
            )
        for record_id in supplier_ids:
            records.append(
                SeedProcurementRecord(
                    tenant_id=DEMO_TENANT_ID,
                    case_id=option.case_id,
                    kind="supplier",
                    record_id=record_id,
                    dataset_version=payload.dataset_version,
                    as_of=payload.as_of.isoformat(),
                    payload=suppliers[record_id].model_dump(mode="json"),
                )
            )
        for record_id in transaction_ids:
            record = next(
                item
                for item in transactions_by_order[order_id]
                if item.transaction_id == record_id
            )
            records.append(
                SeedProcurementRecord(
                    tenant_id=DEMO_TENANT_ID,
                    case_id=option.case_id,
                    kind="transaction",
                    record_id=record_id,
                    dataset_version=payload.dataset_version,
                    as_of=payload.as_of.isoformat(),
                    payload=record.model_dump(mode="json"),
                )
            )
    subjects = (
        SeedSubject(
            tenant_id=DEMO_TENANT_ID,
            subject_id="subject-local-demo",
            display_name="Local demo operator and approver",
            roles=("operator", "approver"),
            case_grants=tuple(item.case_id for item in CASE_CATALOG),
        ),
        SeedSubject(
            tenant_id=DEMO_TENANT_ID,
            subject_id="subject-local-demo-reader",
            display_name="Local demo reader",
            roles=("reader",),
            case_grants=tuple(item.case_id for item in CASE_CATALOG),
        ),
        SeedSubject(
            tenant_id=DEMO_TENANT_ID,
            subject_id="subject-local-demo-auditor",
            display_name="Local demo auditor",
            roles=("auditor",),
            case_grants=tuple(item.case_id for item in CASE_CATALOG),
        ),
        SeedSubject(
            tenant_id=DEMO_TENANT_ID,
            subject_id="subject-local-demo-nogrant",
            display_name="Local demo reader without a case grant",
            roles=("reader",),
            case_grants=(),
        ),
    )
    return subjects, tuple(cases), tuple(records)


def build_seed_plan() -> SecuritySeedPlan:
    """Build the complete non-secret registry seed for the security and demo tenants."""

    raw = json.loads(SECURITY_FIXTURE_PATH.read_text(encoding="utf-8"))
    if raw.get("test_only") is not True:
        raise SecurityFixtureError("the security fixture must be marked test_only")
    document_version = str(raw["document_version"])
    dataset_version = str(raw["dataset_version"])
    index_version = str(raw["index_version"])

    tenants: list[SeedTenant] = []
    subjects: list[SeedSubject] = []
    for tenant in raw["tenants"]:
        tenants.append(
            SeedTenant(
                tenant_id=str(tenant["tenant_id"]),
                display_name=str(tenant["display_name"]),
            )
        )
        for subject in tenant["subjects"]:
            subjects.append(
                SeedSubject(
                    tenant_id=str(tenant["tenant_id"]),
                    subject_id=str(subject["subject_id"]),
                    display_name=str(subject["display_name"]),
                    roles=tuple(str(item) for item in subject["roles"]),
                    case_grants=tuple(str(item) for item in subject["case_grants"]),
                    disabled=bool(subject.get("disabled", False)),
                )
            )

    documents: list[SeedDocument] = []
    for document in raw["shared_documents"]:
        documents.append(
            _security_document(
                document,
                ownership=_SHARED_OWNERSHIP,
                tenant_id=None,
                case_id=None,
                document_version=str(raw["document_version"]),
                index_version=index_version,
            )
        )

    cases: list[SeedCase] = []
    records: list[SeedProcurementRecord] = []
    grants_by_tenant: dict[str, dict[str, list[str]]] = {}
    for subject in subjects:
        for case_id in subject.case_grants:
            grants_by_tenant.setdefault(subject.tenant_id, {}).setdefault(
                case_id, []
            ).append(subject.subject_id)

    for case in raw["cases"]:
        tenant_id = str(case["tenant_id"])
        case_id = str(case["case_id"])
        case_documents = [
            _security_document(
                document,
                ownership=_CASE_OWNERSHIP,
                tenant_id=tenant_id,
                case_id=case_id,
                document_version=document_version,
                index_version=index_version,
            )
            for document in case["documents"]
        ]
        documents.extend(case_documents)
        for kind in ("orders", "suppliers", "transactions"):
            for record in case["procurement"][kind]:
                records.append(
                    SeedProcurementRecord(
                        tenant_id=tenant_id,
                        case_id=case_id,
                        kind={
                            "orders": "order",
                            "suppliers": "supplier",
                            "transactions": "transaction",
                        }[kind],
                        record_id=str(record["record_id"]),
                        dataset_version=dataset_version,
                        as_of=str(raw.get("as_of", "2026-06-30")),
                        payload=dict(record["payload"]),
                    )
                )
        cases.append(
            SeedCase(
                tenant_id=tenant_id,
                case_id=case_id,
                label=str(case["label"]),
                case_type=str(case["case_type"]),
                scenario=str(case["scenario"]),
                purchase_order_id=str(case["purchase_order_id"]),
                case_version=str(case["case_version"]),
                document_version=document_version,
                dataset_version=dataset_version,
                index_version=index_version,
                rules_version=RULES_VERSION_V2,
                case_document_ids=tuple(
                    sorted(item.document_id for item in case_documents)
                ),
                shared_document_ids=tuple(
                    sorted(str(item["document_id"]) for item in raw["shared_documents"])
                ),
                order_ids=tuple(
                    sorted(
                        str(item["record_id"]) for item in case["procurement"]["orders"]
                    )
                ),
                supplier_ids=tuple(
                    sorted(
                        str(item["record_id"])
                        for item in case["procurement"]["suppliers"]
                    )
                ),
                transaction_ids=tuple(
                    sorted(
                        str(item["record_id"])
                        for item in case["procurement"]["transactions"]
                    )
                ),
                granted_subject_ids=tuple(
                    sorted(grants_by_tenant.get(tenant_id, {}).get(case_id, ()))
                ),
            )
        )

    # The fixture declares retrieval scores per registered document ID; a document ID
    # may deliberately exist in two tenants, so one declared score covers every derived
    # chunk that carries that identity.
    chunk_ids_by_document: dict[str, list[str]] = {}
    for document in documents:
        chunk_ids_by_document.setdefault(document.document_id, []).append(
            document.chunk_id
        )
    security_scores: dict[str, dict[str, float]] = {}
    for query in raw["queries"]:
        scores: dict[str, float] = {}
        for document_id, score in query["fake_vector_scores"].items():
            for chunk_id in chunk_ids_by_document.get(str(document_id), ()):
                scores[chunk_id] = float(score)
        security_scores[str(query["query"])] = scores

    business_documents = _business_shared_documents()
    business_scores = _business_retrieval_scores(business_documents)
    demo_subjects, demo_cases, demo_records = _demo_business_plan(
        business_documents, business_scores
    )

    tenants.append(SeedTenant(tenant_id=DEMO_TENANT_ID, display_name=DEMO_TENANT_LABEL))
    subjects.extend(demo_subjects)
    cases.extend(demo_cases)
    documents.extend(business_documents)
    records.extend(demo_records)

    merged_scores: dict[str, dict[str, float]] = {}
    for source in (security_scores, business_scores):
        for query_text, chunk_scores in source.items():
            merged_scores.setdefault(query_text, {}).update(chunk_scores)

    return SecuritySeedPlan(
        fixture_version=SECURITY_FIXTURE_VERSION,
        document_version=document_version,
        dataset_version=dataset_version,
        index_version=index_version,
        tenants=tuple(tenants),
        subjects=tuple(subjects),
        cases=tuple(cases),
        documents=tuple(documents),
        records=tuple(records),
        retrieval_scores=merged_scores,
    )


def fixture_checksum() -> str:
    """Return the checksum of the repository fixture file itself."""

    return "sha256:" + hashlib.sha256(SECURITY_FIXTURE_PATH.read_bytes()).hexdigest()


__all__ = [
    "DEMO_FIXTURE_VERSION",
    "DEMO_TENANT_ID",
    "DEMO_TENANT_LABEL",
    "SECURITY_FIXTURE_PATH",
    "SECURITY_FIXTURE_VERSION",
    "SecurityFixtureError",
    "SecuritySeedPlan",
    "SeedCase",
    "SeedDocument",
    "SeedProcurementRecord",
    "SeedSubject",
    "SeedTenant",
    "build_seed_plan",
    "fixture_checksum",
]


def ingested_chunk(document: SeedDocument) -> IngestedChunk:
    """Rebuild the ingestion contract for a registered synthetic document."""

    return IngestedChunk.model_validate(document.chunk_payload)
