"""Real retrieval for G: register material, scope-filter it, rank it for real.

Predeclared Fake Vector scores say nothing about retrieval quality, so this module
assembles the product's own scoped port in lexical-only mode over an isolated
PostgreSQL registry: the ranking is computed from the text of exactly the chunks
the ``RuntimeScope`` authorized.  Q-08 is measured against the independent truth's
required evidence identities.  Gold labels never enter a query, and the Oracle
context is never used to backfill a retrieval miss.
"""

from __future__ import annotations

import re
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .eval_f_context import load_cases
from .eval_f_score import load_truth
from .ingestion import (
    DEFAULT_CONFIG,
    LocatorKind,
    SourceLocator,
    make_chunk_id,
    make_source_version_id,
)
from .retrieval import RetrievalRequest, RetrievalResponse
from .security_contracts import RuntimeScope
from .security_fixture import (
    SecuritySeedPlan,
    SeedCase,
    SeedDocument,
    SeedProcurementRecord,
    SeedSubject,
    SeedTenant,
    _content_checksum,
)

G_RETRIEVAL_CONTRACT = "v1-07g-retrieval.v1"
G_TENANT = "tenant-v1-07g"
G_SUBJECT = "subject-v1-07g"
G_OTHER_TENANT = "tenant-v1-07g-other"
#: The corpus is registered at the material's own declared version, so a retrieved
#: hit and the independent truth name the same source version.
G_SUPERSEDED_VERSION = "synthetic-gold-document.v3-superseded"
G_DATASET_VERSION = "synthetic-gold-procurement.v3"
G_INDEX_VERSION = "g-retrieval-index.v1"
G_RULES_VERSION = "3.0.0"
TOP_K = 5

#: Identifiers a de-identified query variant must not carry.
_IDENTIFIER = re.compile(r"V107-[A-Z]+-[A-Z]?-?\d+|V107-[A-Z0-9-]+")
_CALENDAR_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def deidentified_signature(content: str) -> str:
    """The form one document takes once every identifier, date and number is masked.

    Two documents with the same signature carry the same retrievable text for a
    query that is not allowed to echo an identifier back at the index.
    """

    masked = _CALENDAR_DATE.sub("DATE", _IDENTIFIER.sub("ID", content))
    masked = _NUMBER.sub("N", masked)
    return hashlib.sha256(masked.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SendObservation:
    """What one retrieval call actually sent, returned and assembled.

    The three counts are read back from the response the port really produced, not
    assumed: a chunk outside the authorized candidate set, a chunk at another document
    version and a request whose index version is not the scope's are named here even
    when the correct answer is zero.
    """

    requested_index_version: str
    request_matches_scope: bool
    returned_chunk_ids: tuple[str, ...]
    out_of_scope_chunk_ids: tuple[str, ...]
    version_drift_ids: tuple[tuple[str, str, str], ...]
    assembled: tuple[tuple[str, str, str], ...]


def observe_retrieval(
    response: RetrievalResponse,
    *,
    scope: RuntimeScope,
    allowed_chunk_ids: frozenset[str],
) -> SendObservation:
    """Observe one real response: returns, assembly content and the request itself."""

    out_of_scope = tuple(
        sorted(
            {
                hit.chunk_id
                for hit in response.results
                if hit.chunk_id not in allowed_chunk_ids
            }
        )
    )
    drift = tuple(
        sorted(
            {
                _identity(hit.source_id, hit.source_version, hit.content_checksum)
                for hit in response.results
                if hit.source_version != scope.document_version
            }
        )
    )
    assembled = tuple(
        _identity(hit.source_id, hit.source_version, hit.content_checksum)
        for hit in response.results
    )
    return SendObservation(
        requested_index_version=response.requested_index_version,
        request_matches_scope=response.requested_index_version == scope.index_version,
        returned_chunk_ids=tuple(hit.chunk_id for hit in response.results),
        out_of_scope_chunk_ids=out_of_scope,
        version_drift_ids=drift,
        assembled=assembled,
    )


@dataclass(frozen=True, slots=True)
class RetrievalSlice:
    """One measured (Case, variant, domain) retrieval observation."""

    case_id: str
    query_variant: str
    evidence_domain: str
    required: tuple[tuple[str, str, str], ...]
    top_five: tuple[tuple[str, str, str], ...]
    hits: int
    wrong_scope_count: int
    wrong_version_count: int
    status: str
    request_matches_scope: bool = True
    assembled: tuple[tuple[str, str, str], ...] = ()
    assembly_out_of_scope: int = 0
    assembly_checked: bool = False


def _identity(source_id: str, version: str, checksum: str) -> tuple[str, str, str]:
    return (source_id, version, checksum.removeprefix("sha256:"))


def document_row(
    *,
    source_id: str,
    content: str,
    document_id: str,
    ownership: str,
    tenant_id: str | None,
    case_id: str | None,
    category: str,
    document_version: str,
) -> SeedDocument:
    """Register one synthetic material chunk through the product ingestion contract."""

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
        "index_version": G_INDEX_VERSION,
    }
    return SeedDocument(
        document_id=document_id,
        ownership=ownership,
        tenant_id=tenant_id,
        case_id=case_id,
        category=category,
        source_id=source_id,
        source_version_id=payload["source_version_id"],
        document_version=document_version,
        index_version=G_INDEX_VERSION,
        content_checksum=content_checksum,
        chunk_payload=payload,
    )


def procurement_rows(case: Any) -> tuple[SeedProcurementRecord, ...]:
    """Register the three read-only fact records one Case's MCP boundary answers."""

    facts = case.model_input.business_facts
    data = facts.model_dump(mode="json")
    order = {
        key: data[key]
        for key in (
            "purchase_order_id",
            "supplier_id",
            "competitive_quote_count",
            "contract_reference",
            "ordered_on",
            "invoice_amount_minor",
            "invoice_currency",
        )
    }
    order.update(amount_minor=facts.order_amount_minor, currency=facts.order_currency)
    supplier = {
        key: data[key]
        for key in (
            "supplier_id",
            "registry_status",
            "procurement_status",
            "tax_identifier",
            "due_diligence_status",
            "sanctions_screening",
        )
    }
    transaction = {
        key: data[key]
        for key in (
            "transaction_id",
            "purchase_order_id",
            "supplier_id",
            "counterparty_reference",
            "booked_on",
        )
    }
    transaction.update(
        amount_minor=facts.transaction_amount_minor, currency=facts.transaction_currency
    )
    return tuple(
        SeedProcurementRecord(
            tenant_id=G_TENANT,
            case_id=case.case_id,
            kind=kind,
            record_id=record_id,
            dataset_version=G_DATASET_VERSION,
            as_of=case.as_of.isoformat(),
            payload=payload,
        )
        for kind, record_id, payload in (
            ("order", facts.purchase_order_id, order),
            ("supplier", facts.supplier_id, supplier),
            ("transaction", facts.transaction_id, transaction),
        )
    )


def build_corpus(
    *,
    measured: Sequence[Any],
    distractors: Sequence[Any],
    data_version: str,
    siblings: Sequence[Any] = (),
    fixture_version: str = "v1-07g-retrieval-corpus.v1",
) -> SecuritySeedPlan:
    """One registry plan: Case material, a distractor library, and near copies.

    The library holds other Cases' documents, one superseded version of each
    measured Case's policy and one same-source document registered to another
    tenant.  Neither near copy is in any measured scope, so the measurement proves
    the scope and version predicates exclude them rather than the ranking happening
    to lose them.

    ``siblings`` adds the adversarial configuration: Cases from the same generation
    family, whose policy text the generator reuses byte for byte.  They are kept
    apart from ``distractors`` so the two configurations can be reported separately
    and a miss can be attributed to content ambiguity instead of being hidden.
    """

    documents: list[SeedDocument] = []
    cases: list[SeedCase] = []
    records: list[SeedProcurementRecord] = []
    shared_ids: list[str] = []
    versions = {
        document.version
        for case in (*measured, *distractors)
        for document in (
            case.model_input.policy_document,
            case.model_input.contract_document,
        )
    }
    if len(versions) != 1:
        raise ValueError("one retrieval corpus registers exactly one material version")
    document_version = next(iter(versions))
    for case in (*distractors, *siblings):
        for domain, document in (
            ("policy", case.model_input.policy_document),
            ("contract", case.model_input.contract_document),
        ):
            document_id = f"DOC-LIB-{document.source_id}"
            shared_ids.append(document_id)
            documents.append(
                document_row(
                    source_id=document.source_id,
                    content=document.content,
                    document_id=document_id,
                    ownership="shared_fixture",
                    tenant_id=None,
                    case_id=None,
                    category=domain,
                    document_version=document_version,
                )
            )
    for case in measured:
        policy = case.model_input.policy_document
        contract = case.model_input.contract_document
        case_ids = []
        for domain, document in (("policy", policy), ("contract", contract)):
            document_id = f"DOC-{document.source_id}"
            case_ids.append(document_id)
            documents.append(
                document_row(
                    source_id=document.source_id,
                    content=document.content,
                    document_id=document_id,
                    ownership="case_document",
                    tenant_id=G_TENANT,
                    case_id=case.case_id,
                    category=domain,
                    document_version=document_version,
                )
            )
        # A superseded version of the same source, registered but out of scope.
        superseded_id = f"DOC-{policy.source_id}-SUPERSEDED"
        documents.append(
            document_row(
                source_id=policy.source_id,
                content=policy.content.replace("2026 年度", "2025 年度"),
                document_id=superseded_id,
                ownership="shared_fixture",
                tenant_id=None,
                case_id=None,
                category="policy",
                document_version=G_SUPERSEDED_VERSION,
            )
        )
        # The same source registered to another tenant: a cross-tenant namesake.
        documents.append(
            document_row(
                source_id=policy.source_id,
                content=policy.content,
                document_id=f"DOC-{policy.source_id}-OTHER-TENANT",
                ownership="case_document",
                tenant_id=G_OTHER_TENANT,
                case_id=f"{case.case_id}-OTHER",
                category="policy",
                document_version=document_version,
            )
        )
        facts = case.model_input.business_facts
        cases.append(
            SeedCase(
                tenant_id=G_TENANT,
                case_id=case.case_id,
                label=case.case_id,
                case_type="purchase_order",
                scenario="low",
                purchase_order_id=facts.purchase_order_id,
                case_version=f"{case.case_id.lower()}.v1",
                document_version=document_version,
                dataset_version=G_DATASET_VERSION,
                index_version=G_INDEX_VERSION,
                rules_version=G_RULES_VERSION,
                case_document_ids=tuple(case_ids),
                shared_document_ids=tuple(sorted(shared_ids)),
                order_ids=(facts.purchase_order_id,),
                supplier_ids=(facts.supplier_id,),
                transaction_ids=(facts.transaction_id,),
                granted_subject_ids=(G_SUBJECT,),
            )
        )
        records.extend(procurement_rows(case))
    return SecuritySeedPlan(
        fixture_version=fixture_version,
        document_version=document_version,
        dataset_version=G_DATASET_VERSION,
        index_version=G_INDEX_VERSION,
        tenants=(
            SeedTenant(G_TENANT, "V1-07G retrieval tenant"),
            SeedTenant(G_OTHER_TENANT, "V1-07G other tenant"),
        ),
        subjects=(
            SeedSubject(
                tenant_id=G_TENANT,
                subject_id=G_SUBJECT,
                display_name="V1-07G operator",
                roles=("operator",),
                case_grants=tuple(case.case_id for case in measured),
            ),
        ),
        cases=tuple(cases),
        documents=tuple(documents),
        records=tuple(records),
        # Lexical-only mode reads no score row, so none is declared.
        retrieval_scores={},
    )


def query_for(case: Any, domain: str, variant: str) -> str:
    """Build one query from product input only: the question and the domain asked.

    The de-identified variant strips every registered identifier, so a hit cannot
    come from echoing a Case, order or supplier id back at the index.
    """

    subject = (
        "适用政策、专项参数与供应商资格要求"
        if domain == "policy"
        else ("合同主体、标的、价款币种、交付验收质保与变更责任条款")
    )
    question = case.model_input.question
    if variant == "without_case_id":
        question = _IDENTIFIER.sub(" ", question)
        question = re.sub(r"\s+", " ", question).strip()
    return f"{question} {subject}"


def required_for(truth: Any, domain: str) -> tuple[tuple[str, str, str], ...]:
    """The truth's required evidence identities for one domain, as retrieval sees them."""

    return tuple(
        _identity(item.source_id, item.version, item.content_sha256)
        for item in truth.required_evidence
        if item.locator.startswith(f"{domain}-style-")
    )


def domain_port(store: Any, scope: RuntimeScope, domain: str) -> Any:
    """The product's scoped port narrowed to one worker's authorized category.

    A document worker reaches only its own material category, which is a registered
    property of the material and never a Gold label.  Narrowing happens before any
    query, so the other domain's documents are not candidates at all.
    """

    from .scoped_runtime import ScopedRetrievalPort

    return ScopedRetrievalPort.from_store(
        store=store,
        scope=scope,
        vector_mode="none",
        lexical_adapter="idf",
        category=domain,
    )


def measure_slice(
    port: Any,
    *,
    case: Any,
    truth: Any,
    domain: str,
    variant: str,
    scope: RuntimeScope,
) -> RetrievalSlice:
    """Retrieve once through the product's scoped port and score the top five."""

    required = required_for(truth, domain)
    request = RetrievalRequest(
        query=query_for(case, domain, variant),
        index_version=scope.index_version,
        limit=TOP_K,
        minimum_results=1,
    )
    response = port.retrieve(request)
    observation = observe_retrieval(
        response, scope=scope, allowed_chunk_ids=port.allowed_chunk_ids
    )
    assembled_out_of_scope = observe_assembly(
        observation.returned_chunk_ids, allowed_chunk_ids=port.allowed_chunk_ids
    )
    top = tuple(
        _identity(hit.source_id, hit.source_version, hit.content_checksum)
        for hit in response.results
    )
    return RetrievalSlice(
        case_id=case.case_id,
        query_variant=variant,
        evidence_domain=domain,
        required=required,
        top_five=top,
        hits=sum(item in required for item in top),
        wrong_scope_count=len(observation.out_of_scope_chunk_ids),
        wrong_version_count=len(observation.version_drift_ids),
        status=response.status.value,
        request_matches_scope=observation.request_matches_scope,
        assembled=observation.assembled,
        assembly_out_of_scope=len(assembled_out_of_scope),
        assembly_checked=bool(port.allowed_chunk_ids),
    )


def observe_assembly(
    assembled_chunk_ids: Sequence[str],
    *,
    allowed_chunk_ids: frozenset[str],
) -> tuple[str, ...]:
    """The chunks a context assembly carries that the port never authorized.

    Checked by chunk identity, not by source identity: a cross-tenant namesake of an
    authorized document shares its source id, version and checksum, so only the chunk
    id separates what the scope actually admitted.  An empty authority set means the
    caller supplied nothing to check against and the observation reports nothing
    rather than a pass.
    """

    if not allowed_chunk_ids:
        return ()
    return tuple(
        chunk_id
        for chunk_id in assembled_chunk_ids
        if chunk_id not in allowed_chunk_ids
    )


def negative_probe(port: Any, scope: RuntimeScope) -> dict[str, Any]:
    """A subject the authorized material does not carry must not fabricate a hit."""

    request = RetrievalRequest(
        query="zzz qqq nonexistent-subject-token",
        index_version=scope.index_version,
        limit=TOP_K,
        minimum_results=1,
    )
    response = port.retrieve(request)
    return {
        "status": response.status.value,
        "issue": response.issue.value if response.issue else None,
        "results": len(response.results),
    }


def q08(slices: Sequence[RetrievalSlice]) -> dict[str, Any]:
    """Recall@5 with the truth's evidence identities as the denominator."""

    def ratio(rows: Sequence[RetrievalSlice]) -> dict[str, Any]:
        denominator = sum(len(row.required) for row in rows)
        numerator = sum(row.hits for row in rows)
        return {
            "numerator": numerator,
            "denominator": denominator,
            "value": (numerator / denominator) if denominator else None,
            "meets_threshold": None
            if not denominator
            else numerator / denominator >= 0.95,
            "wrong_version_count": sum(row.wrong_version_count for row in rows),
            "wrong_scope_count": sum(row.wrong_scope_count for row in rows),
            "assembly_out_of_scope": sum(row.assembly_out_of_scope for row in rows),
            "request_version_mismatches": sum(
                1 for row in rows if not row.request_matches_scope
            ),
            "slices": len(rows),
        }

    by_slice = {
        f"{variant}:{domain}": ratio(
            [
                row
                for row in slices
                if row.query_variant == variant and row.evidence_domain == domain
            ]
        )
        for variant in ("original", "without_case_id")
        for domain in ("policy", "contract")
    }
    overall = ratio(list(slices))
    return {
        "contract": G_RETRIEVAL_CONTRACT,
        "recall_at_five": overall,
        "by_query_variant_and_domain": by_slice,
        "meets_threshold": (
            None
            if overall["value"] is None
            else overall["meets_threshold"] is True
            and all(item["meets_threshold"] is True for item in by_slice.values())
            and overall["wrong_version_count"] == 0
            and overall["wrong_scope_count"] == 0
            and overall["assembly_out_of_scope"] == 0
            and overall["request_version_mismatches"] == 0
        ),
    }


def corpus_cases(
    data_version: str, *, measured_count: int, distractor_count: int
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], Mapping[str, Any]]:
    """Split DEV into measured Cases, unrelated distractors and same-family siblings.

    The split is by the material's own registered generation family, never by a
    label or an expected outcome.
    """

    cases = load_cases("dev", data_version)
    measured = cases[:measured_count]
    families = {case.generation_family_id for case in measured}
    unrelated = []
    siblings = []
    for case in cases[measured_count:]:
        if case.generation_family_id in families:
            siblings.append(case)
        else:
            unrelated.append(case)
        if len(unrelated) >= distractor_count:
            break
    return measured, tuple(unrelated), tuple(siblings), load_truth(data_version)


__all__ = [
    "G_DATASET_VERSION",
    "G_INDEX_VERSION",
    "G_OTHER_TENANT",
    "G_RETRIEVAL_CONTRACT",
    "G_SUBJECT",
    "G_SUPERSEDED_VERSION",
    "G_TENANT",
    "RetrievalSlice",
    "SendObservation",
    "TOP_K",
    "deidentified_signature",
    "build_corpus",
    "corpus_cases",
    "document_row",
    "domain_port",
    "measure_slice",
    "negative_probe",
    "observe_assembly",
    "observe_retrieval",
    "procurement_rows",
    "q08",
    "query_for",
    "required_for",
]
