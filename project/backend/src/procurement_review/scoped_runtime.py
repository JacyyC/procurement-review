"""Scope-bound runtime construction for the V1-03 review execution.

Everything a Worker can reach is built from one immutable :class:`RuntimeScope`:
the retrieval index only contains documents registered for that tenant/Case, the
procurement reader only answers for the registered synthetic records, and every
returned reference is re-verified against the same registration before it may be
stored or shown.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from . import admitted_reads
from .contracts import BusinessFact, EvidenceRef, WorkerKind
from .ingestion import IngestedChunk, LocatorKind
from .procurement_store import (
    ProcurementDatasetIdentity,
    ProcurementRecord,
    ProcurementScopeError,
)
from .review_graph import ReviewExecution
from .review_store import RegisteredDocument, RegisteredProcurementRecord, ReviewStore
from .retrieval import (
    DeterministicFakeVectorAdapter,
    DeterministicHybridRetriever,
    DeterministicIdfLexicalIndex,
    DeterministicLexicalIndex,
    NoVectorAdapter,
    RetrievalPort,
    RetrievalRequest,
    RetrievalResponse,
)
from .security_contracts import RuntimeScope
from .worker_contracts import WorkerBranchResult

SCOPED_RUNTIME_VERSION = "scoped-runtime.v1"

#: Which vector route the scoped port assembles.  ``fake_scores`` is the frozen
#: fixture behaviour: it consumes predeclared scores and is never real retrieval
#: quality.  ``none`` reads no score row at all, so the ranking is computed only
#: from the in-scope chunks' own text.
VectorMode = str  # Literal["fake_scores", "none"]
LexicalAdapter = str  # Literal["unigram", "idf"]

#: One live-authority gate.  It is called with the connection the protected read will use,
#: or with ``None`` when the read is a port call over an already-built index and the gate
#: therefore owns its own short transaction.
AdmissionGate = Callable[..., None]


@contextmanager
def admitted_read_transaction(
    store: ReviewStore, admission: AdmissionGate | None, site: str
) -> Iterator[Any]:
    """Open the one short transaction a protected preload reads in.

    The admission runs first, on this very connection, so the share locks it takes stay
    held until the reads below are done: a revocation either commits first and is seen
    here, or waits for this short transaction.  Nothing in this block touches a network,
    a model or the MCP port.
    """

    with store.transaction() as connection:
        if admission is not None:
            admission(connection, site=site)
        yield connection


class ScopeIntegrityError(RuntimeError):
    """Raised when a resource does not match the runtime scope it was loaded for."""


@dataclass(slots=True)
class ScopeTelemetry:
    """Per-execution counters reported as Evidence; they never carry payloads."""

    retrieval_calls: int = 0
    retrieval_documents: int = 0
    scope_documents: int = 0
    tool_calls: int = 0
    denied_tool_calls: int = 0
    denied_retrieval_attempts: int = 0

    def as_summary(self) -> dict[str, int]:
        return {
            "retrieval_calls": self.retrieval_calls,
            "retrieval_documents": self.retrieval_documents,
            "scope_documents": self.scope_documents,
            "tool_calls": self.tool_calls,
            "denied_tool_calls": self.denied_tool_calls,
            "denied_retrieval_attempts": self.denied_retrieval_attempts,
        }


def ingested_chunk_from_registry(document: RegisteredDocument) -> IngestedChunk:
    """Rebuild one ingestion contract from a registered document payload."""

    payload = dict(document.chunk_payload)
    locator = dict(payload.get("locator") or {})
    locator["kind"] = LocatorKind(locator["kind"])
    payload["locator"] = locator
    return IngestedChunk.model_validate(payload)


def _validated_chunk(
    document: RegisteredDocument, *, scope: RuntimeScope
) -> IngestedChunk:
    chunk = ingested_chunk_from_registry(document)
    if chunk.index_version != scope.index_version:
        raise ScopeIntegrityError("registered document index version mismatch")
    if chunk.source_version_id != document.source_version_id:
        raise ScopeIntegrityError("registered document source version mismatch")
    if chunk.content_checksum != document.content_checksum:
        raise ScopeIntegrityError("registered document checksum mismatch")
    if document.ownership == "case_document":
        if document.tenant_id != scope.tenant_id or document.case_id != scope.case_id:
            raise ScopeIntegrityError("case document ownership mismatch")
        if document.document_id not in scope.case_document_ids:
            raise ScopeIntegrityError("case document is not registered for this scope")
    elif document.ownership == "shared_fixture":
        if document.tenant_id is not None or document.case_id is not None:
            raise ScopeIntegrityError("shared fixture documents must be tenant-free")
        if document.document_id not in scope.shared_document_ids:
            raise ScopeIntegrityError(
                "shared document is not registered for this scope"
            )
    else:
        raise ScopeIntegrityError("unknown document ownership class")
    return chunk


class ScopedRetrievalPort:
    """A retrieval port whose index only ever contains in-scope documents.

    The port is built *after* the scope predicate narrows the candidate set, so no
    out-of-scope chunk can influence recall, ranking, fusion, counts or Top-k.  The
    narrowing happens *before* the query, not after it: the only chunk ids this port
    ever asks the score table for are the ones the ``RuntimeScope`` registered, so an
    out-of-scope chunk is never read, never counted and never named by this object.
    """

    test_only = True

    def __init__(
        self,
        *,
        scope: RuntimeScope,
        retriever: RetrievalPort,
        telemetry: ScopeTelemetry,
        allowed_chunk_ids: frozenset[str],
        admission: AdmissionGate | None = None,
    ) -> None:
        self._scope = scope
        self._retriever = retriever
        self._telemetry = telemetry
        self._allowed_chunk_ids = allowed_chunk_ids
        self._admission = admission

    @property
    def scope(self) -> RuntimeScope:
        return self._scope

    @property
    def telemetry(self) -> ScopeTelemetry:
        return self._telemetry

    @classmethod
    def from_store(
        cls,
        *,
        store: ReviewStore,
        scope: RuntimeScope,
        telemetry: ScopeTelemetry | None = None,
        admission: AdmissionGate | None = None,
        vector_mode: VectorMode = "fake_scores",
        lexical_adapter: LexicalAdapter = "unigram",
        category: str | None = None,
    ) -> ScopedRetrievalPort:
        counter = telemetry or ScopeTelemetry()
        # The index this port answers from is built *inside* the admitted transaction: the
        # registry rows and the retrieval scores are the protected data, so the live check
        # and those reads share one short transaction rather than being separated by one.
        with admitted_read_transaction(
            store, admission, "retrieval_index"
        ) as connection:
            documents = admitted_reads.load_documents(
                connection,
                case_document_ids=scope.case_document_ids,
                shared_document_ids=scope.shared_document_ids,
                tenant_id=scope.tenant_id,
                case_id=scope.case_id,
                document_version=scope.document_version,
                index_version=scope.index_version,
            )
            expected = len(scope.case_document_ids) + len(scope.shared_document_ids)
            if len(documents) != expected:
                raise ScopeIntegrityError(
                    "the registry does not hold every document the scope requires"
                )
            # The whole registered scope is integrity-checked first, so narrowing to
            # one document category can never hide a missing registry row.
            validated = tuple(
                (item, _validated_chunk(item, scope=scope)) for item in documents
            )
            counter.scope_documents = len(validated)
            if category is not None:
                # A document worker reaches only its own authorized material category;
                # this narrows the scope and never widens it.
                validated = tuple(
                    (item, chunk)
                    for item, chunk in validated
                    if item.category == category
                )
                if not validated:
                    raise ScopeIntegrityError(
                        "the scope authorizes no document of the requested category"
                    )
            chunks = tuple(chunk for _item, chunk in validated)
            allowed_chunk_ids = frozenset(chunk.chunk_id for chunk in chunks)
            if vector_mode == "none":
                # No score row is read at all: a predeclared score cannot reach the
                # candidate set, the ranking, the fusion or the returned references.
                counter.retrieval_documents = len(chunks)
                return cls(
                    scope=scope,
                    retriever=DeterministicHybridRetriever(
                        lexical_index=(
                            DeterministicIdfLexicalIndex
                            if lexical_adapter == "idf"
                            else DeterministicLexicalIndex
                        )(chunks, index_version=scope.index_version),
                        vector_port=NoVectorAdapter(index_version=scope.index_version),
                    ),
                    telemetry=counter,
                    allowed_chunk_ids=allowed_chunk_ids,
                    admission=admission,
                )
            # Only the scores of the chunks this scope registered are read.  The candidate
            # set is narrowed *before* the query rather than filtered after it: no statement
            # this port issues can name a chunk outside ``allowed_chunk_ids``, so another
            # tenant's registry rows can never reach the index, the ranking, the counts or
            # any attribute of the returned port.
            all_scores = admitted_reads.retrieval_scores(
                connection,
                document_version=scope.document_version,
                index_version=scope.index_version,
                chunk_ids=sorted(allowed_chunk_ids),
            )
        counter.retrieval_documents = len(chunks)
        retriever = DeterministicHybridRetriever(
            lexical_index=DeterministicLexicalIndex(
                chunks,
                index_version=scope.index_version,
            ),
            vector_port=DeterministicFakeVectorAdapter(
                index_version=scope.index_version,
                scores_by_query=all_scores,
            ),
        )
        return cls(
            scope=scope,
            retriever=retriever,
            telemetry=counter,
            allowed_chunk_ids=allowed_chunk_ids,
            admission=admission,
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        self._telemetry.retrieval_calls += 1
        if request.index_version != self._scope.index_version:
            self._telemetry.denied_retrieval_attempts += 1
        # The last gate before the index is read.  The index is already in memory, so the
        # admission owns its own short transaction and closes it before the port call: no
        # database row lock is held across a port, network or model call.
        if self._admission is not None:
            self._admission()
        return self._retriever.retrieve(request)

    @property
    def allowed_chunk_ids(self) -> frozenset[str]:
        return self._allowed_chunk_ids


class ScopedProcurementReader:
    """Procurement reader limited to the records registered for one Case."""

    test_only = True

    def __init__(
        self,
        *,
        scope: RuntimeScope,
        records: Sequence[RegisteredProcurementRecord],
        telemetry: ScopeTelemetry,
        identity: ProcurementDatasetIdentity,
        admission: AdmissionGate | None = None,
    ) -> None:
        self._scope = scope
        self._telemetry = telemetry
        self._identity = identity
        self._admission = admission
        self._orders: dict[str, RegisteredProcurementRecord] = {}
        self._suppliers: dict[str, RegisteredProcurementRecord] = {}
        self._transactions: dict[str, RegisteredProcurementRecord] = {}
        self._transactions_by_order: dict[str, list[str]] = {}
        self._transactions_by_supplier: dict[str, list[str]] = {}
        for record in records:
            payload = record.payload_dict()
            if record.record_kind == "order":
                self._orders[record.record_id] = record
            elif record.record_kind == "supplier":
                self._suppliers[record.record_id] = record
            elif record.record_kind == "transaction":
                self._transactions[record.record_id] = record
                order_id = payload.get("purchase_order_id")
                supplier_id = payload.get("supplier_id")
                if isinstance(order_id, str):
                    self._transactions_by_order.setdefault(order_id, []).append(
                        record.record_id
                    )
                if isinstance(supplier_id, str):
                    self._transactions_by_supplier.setdefault(supplier_id, []).append(
                        record.record_id
                    )
        if set(self._orders) - set(scope.allowed_order_ids):
            raise ScopeIntegrityError("loaded orders are not registered for this scope")
        if set(self._suppliers) - set(scope.allowed_supplier_ids):
            raise ScopeIntegrityError(
                "loaded suppliers are not registered for this scope"
            )
        if set(self._transactions) - set(scope.allowed_transaction_ids):
            raise ScopeIntegrityError(
                "loaded transactions are not registered for this scope"
            )

    @classmethod
    def from_store(
        cls,
        *,
        store: ReviewStore,
        scope: RuntimeScope,
        telemetry: ScopeTelemetry | None = None,
        admission: AdmissionGate | None = None,
    ) -> ScopedProcurementReader:
        # The registered records are protected data: the live check and the read that
        # produces them share one short transaction, opened here.
        with admitted_read_transaction(
            store, admission, "procurement_records"
        ) as connection:
            records = admitted_reads.procurement_records(
                connection,
                tenant_id=scope.tenant_id,
                case_id=scope.case_id,
                dataset_version=scope.dataset_version,
            )
        as_of = date.fromisoformat(records[0].as_of) if records else date(1970, 1, 1)
        identity = ProcurementDatasetIdentity(
            dataset_version=scope.dataset_version,
            fixture_schema_version=SCOPED_RUNTIME_VERSION,
            envelope_schema_version=SCOPED_RUNTIME_VERSION,
            generator_version=SCOPED_RUNTIME_VERSION,
            seed=0,
            as_of=as_of,
            digest_algorithm="sha256",
            payload_canonicalization="json-sort-keys",
            payload_sha256=scope.case_version,
        )
        return cls(
            scope=scope,
            records=records,
            telemetry=telemetry or ScopeTelemetry(),
            identity=identity,
            admission=admission,
        )

    @property
    def identity(self) -> ProcurementDatasetIdentity:
        return self._identity

    @property
    def scope(self) -> RuntimeScope:
        return self._scope

    @property
    def telemetry(self) -> ScopeTelemetry:
        return self._telemetry

    def _check_version(self, dataset_version: str) -> None:
        if dataset_version != self._identity.dataset_version:
            from .procurement_store import DatasetVersionMismatchError

            raise DatasetVersionMismatchError("registered dataset version mismatch")

    def note_scope_denial(self) -> None:
        """Record one rejection raised by the MCP tool boundary."""

        self._telemetry.denied_tool_calls += 1

    def _admit(self) -> None:
        """Require live authority for the generation this read serves.

        Reached after argument and scope validation, so an out-of-scope tool argument
        stays the non-retryable branch rejection the contract describes and never becomes
        an authority event. The admission owns its own short transaction; the record lookup
        that follows happens after it closed.
        """

        if self._admission is not None:
            self._admission()

    def is_order_allowed(self, purchase_order_id: str) -> bool:
        return purchase_order_id in self._scope.allowed_order_ids

    def is_supplier_allowed(self, supplier_id: str) -> bool:
        return supplier_id in self._scope.allowed_supplier_ids

    def get_supplier(
        self, supplier_id: str, dataset_version: str
    ) -> ProcurementRecord | None:
        self._telemetry.tool_calls += 1
        self._check_version(dataset_version)
        if not self.is_supplier_allowed(supplier_id):
            self._telemetry.denied_tool_calls += 1
            raise ProcurementScopeError("supplier is outside the registered Case scope")
        self._admit()
        record = self._suppliers.get(supplier_id)
        if record is None:
            return None
        return ProcurementRecord(
            record_id=record.record_id,
            dataset_version=record.dataset_version,
            as_of=date.fromisoformat(record.as_of),
            payload=record.payload_dict(),
        )

    def get_purchase_order(
        self, purchase_order_id: str, dataset_version: str
    ) -> ProcurementRecord | None:
        self._telemetry.tool_calls += 1
        self._check_version(dataset_version)
        if not self.is_order_allowed(purchase_order_id):
            self._telemetry.denied_tool_calls += 1
            raise ProcurementScopeError("order is outside the registered Case scope")
        self._admit()
        record = self._orders.get(purchase_order_id)
        if record is None:
            return None
        return ProcurementRecord(
            record_id=record.record_id,
            dataset_version=record.dataset_version,
            as_of=date.fromisoformat(record.as_of),
            payload=record.payload_dict(),
        )

    def list_historical_transactions(
        self,
        *,
        dataset_version: str,
        purchase_order_id: str | None = None,
        supplier_id: str | None = None,
    ) -> tuple[ProcurementRecord, ...]:
        from .procurement_store import InvalidProcurementQueryError

        self._telemetry.tool_calls += 1
        self._check_version(dataset_version)
        if (purchase_order_id is None) == (supplier_id is None):
            raise InvalidProcurementQueryError("select exactly one history scope")
        if purchase_order_id is not None:
            if not self.is_order_allowed(purchase_order_id):
                self._telemetry.denied_tool_calls += 1
                raise ProcurementScopeError(
                    "order is outside the registered Case scope"
                )
            self._admit()
            candidates = self._transactions_by_order.get(purchase_order_id, ())
        else:
            assert supplier_id is not None
            if not self.is_supplier_allowed(supplier_id):
                self._telemetry.denied_tool_calls += 1
                raise ProcurementScopeError(
                    "supplier is outside the registered Case scope"
                )
            self._admit()
            candidates = self._transactions_by_supplier.get(supplier_id, ())

        allowed = set(self._scope.allowed_transaction_ids)
        return tuple(
            ProcurementRecord(
                record_id=self._transactions[record_id].record_id,
                dataset_version=self._transactions[record_id].dataset_version,
                as_of=date.fromisoformat(self._transactions[record_id].as_of),
                payload=self._transactions[record_id].payload_dict(),
            )
            for record_id in sorted(set(candidates) & allowed)
        )


@dataclass(frozen=True, slots=True)
class ScopedEvidenceIndex:
    """Registration-backed index used to re-verify stored references."""

    chunk_keys: Mapping[tuple[str, str, str], RegisteredDocument]
    fact_keys: frozenset[tuple[str, str]]

    @classmethod
    def from_scope(
        cls,
        *,
        scope: RuntimeScope,
        documents: Sequence[RegisteredDocument],
    ) -> ScopedEvidenceIndex:
        chunk_keys: dict[tuple[str, str, str], RegisteredDocument] = {}
        for document in documents:
            chunk = ingested_chunk_from_registry(document)
            chunk_keys[(chunk.source_id, chunk.source_version_id, chunk.chunk_id)] = (
                document
            )
        fact_keys = frozenset(
            (f"{kind}:{record_id}", kind)
            for kind, ids in (
                ("get_purchase_order", scope.allowed_order_ids),
                ("get_supplier", scope.allowed_supplier_ids),
                ("list_historical_transactions", scope.allowed_transaction_ids),
            )
            for record_id in ids
        )
        return cls(chunk_keys=chunk_keys, fact_keys=fact_keys)

    def verify_evidence_ref(self, reference: EvidenceRef) -> None:
        key = (reference.source_id, reference.version, reference.chunk_id or "")
        document = self.chunk_keys.get(key)
        if document is None:
            raise ScopeIntegrityError("evidence reference is not registered in scope")
        if reference.content_checksum != document.content_checksum:
            raise ScopeIntegrityError("evidence reference checksum mismatch")

    def verify_business_fact(self, fact: BusinessFact) -> None:
        if (f"{fact.tool_name}:{fact.record_id}", fact.tool_name) not in self.fact_keys:
            raise ScopeIntegrityError("business fact is not registered in scope")

    def verify_branch_result(self, result: WorkerBranchResult) -> None:
        for finding in result.findings:
            for reference in finding.evidence_refs:
                self.verify_evidence_ref(reference)
            for fact in finding.business_fact_refs:
                self.verify_business_fact(fact)

    def verify_execution(self, execution: ReviewExecution) -> None:
        for result in execution.branch_results:
            self.verify_branch_result(result)


@dataclass(slots=True)
class ScopedReviewRuntime:
    """One scope-bound runtime handed to exactly one review execution."""

    scope: RuntimeScope
    retrieval: ScopedRetrievalPort
    procurement: ScopedProcurementReader
    evidence: ScopedEvidenceIndex
    telemetry: ScopeTelemetry = field(default_factory=ScopeTelemetry)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def build(
        cls,
        *,
        store: ReviewStore,
        scope: RuntimeScope,
        admission: AdmissionGate | None = None,
        vector_mode: VectorMode = "fake_scores",
        lexical_adapter: LexicalAdapter = "unigram",
    ) -> ScopedReviewRuntime:
        telemetry = ScopeTelemetry()
        retrieval = ScopedRetrievalPort.from_store(
            store=store,
            scope=scope,
            telemetry=telemetry,
            admission=admission,
            vector_mode=vector_mode,
            lexical_adapter=lexical_adapter,
        )
        procurement = ScopedProcurementReader.from_store(
            store=store,
            scope=scope,
            telemetry=telemetry,
            admission=admission,
        )
        # The evidence index re-verifies every stored reference against the same registry,
        # so it is a protected read like the other two and is admitted the same way.
        with admitted_read_transaction(
            store, admission, "evidence_index"
        ) as connection:
            documents = admitted_reads.load_documents(
                connection,
                case_document_ids=scope.case_document_ids,
                shared_document_ids=scope.shared_document_ids,
                tenant_id=scope.tenant_id,
                case_id=scope.case_id,
                document_version=scope.document_version,
                index_version=scope.index_version,
            )
            evidence = ScopedEvidenceIndex.from_scope(scope=scope, documents=documents)
        return cls(
            scope=scope,
            retrieval=retrieval,
            procurement=procurement,
            evidence=evidence,
            telemetry=telemetry,
        )

    def verify_execution(self, execution: ReviewExecution) -> None:
        self.evidence.verify_execution(execution)

    def worker_scope_summary(self) -> dict[str, int]:
        return self.telemetry.as_summary()


def worker_kinds_in_scope(scope: RuntimeScope) -> tuple[WorkerKind, ...]:
    """Return the fixed three Worker kinds; the scope never changes the topology."""

    if scope.runtime_scope_version != "runtime-scope.v1":
        raise ScopeIntegrityError("unknown runtime scope version")
    return (WorkerKind.POLICY, WorkerKind.CONTRACT, WorkerKind.PROCUREMENT_DATA)


__all__ = [
    "SCOPED_RUNTIME_VERSION",
    "AdmissionGate",
    "ScopeIntegrityError",
    "ScopeTelemetry",
    "admitted_read_transaction",
    "ingested_chunk_from_registry",
    "ScopedEvidenceIndex",
    "ScopedProcurementReader",
    "ScopedRetrievalPort",
    "LexicalAdapter",
    "VectorMode",
    "ScopedReviewRuntime",
    "worker_kinds_in_scope",
]
