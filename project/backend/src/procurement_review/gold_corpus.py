"""Split-isolated lexical corpus built from Gold Case documents.

The V1-02C retrieval score must not reuse the model-visible allowlist, otherwise
"did the retriever find it" would be answered by construction. This module turns
the Gold evidence documents of one split into ordinary ``IngestedChunk`` records and
scores the existing deterministic lexical index against the independent Gold
``required_evidence_refs``, so Recall@k is measured rather than assumed.

Embedding and rerank stay ``not_supported``: no fake vector route is wired in, and
the reported adapter version says so.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .contracts import EvidenceRef
from .gold_set import GoldCase, GoldLabel, GoldSplit
from .ingestion import (
    DEFAULT_INDEX_VERSION,
    IngestedChunk,
    IngestionConfig,
    LocatorKind,
    SourceLocator,
    _checksum_bytes,
    make_chunk_id,
    make_document_id,
    make_source_version_id,
)
from .retrieval import DeterministicLexicalIndex, canonical_query_text

GOLD_CORPUS_BUILDER_VERSION = "gold-lexical-corpus.v1"
GOLD_LEXICAL_VECTOR_MODE: Literal["not_supported_lexical_only.v1"] = (
    "not_supported_lexical_only.v1"
)


@dataclass(frozen=True, slots=True)
class GoldCorpusDocument:
    """One indexed Gold document with the reference a hit maps back to."""

    chunk: IngestedChunk
    evidence_ref: EvidenceRef

    @property
    def source_id(self) -> str:
        return self.chunk.source_id


@dataclass(frozen=True, slots=True)
class GoldLexicalCorpus:
    """A deterministic lexical index over exactly one split's Gold documents."""

    split: GoldSplit
    index: DeterministicLexicalIndex
    documents: tuple[GoldCorpusDocument, ...]
    builder_version: str = GOLD_CORPUS_BUILDER_VERSION
    vector_mode: str = GOLD_LEXICAL_VECTOR_MODE

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def index_checksum(self) -> str:
        return self.index.index_checksum

    def ranked_source_ids(self, query: str, *, limit: int) -> tuple[str, ...]:
        parsed = canonical_query_text(query)
        if not parsed:
            return ()
        scores = self.index.recall(parsed, limit=limit)
        by_chunk = {doc.chunk.chunk_id: doc.source_id for doc in self.documents}
        return tuple(by_chunk[item.chunk_id] for item in scores)


def _chunk_for_evidence(
    ref: EvidenceRef, content: str, *, index_version: str
) -> IngestedChunk:
    config = IngestionConfig(index_version=index_version)
    # Ingestion checksums the chunk bytes; Gold references checksum id plus content,
    # so the index carries its own identity and maps back to Gold by source_id.
    chunk_checksum = _checksum_bytes(content.encode("utf-8"))
    locator = SourceLocator(kind=LocatorKind.TEXT_LINES, line_start=1, line_end=1)
    document_id = make_document_id(
        source_id=ref.source_id,
        source_version=ref.version,
        source_checksum=chunk_checksum,
        media_type="text/plain",
        ingestion_version=config.ingestion_version,
    )
    chunk_id = make_chunk_id(
        document_id=document_id,
        ordinal=1,
        locator=locator,
        content_checksum=chunk_checksum,
        chunk_rules_version=config.chunk_rules_version,
        chunk_rules_checksum=config.chunk_rules_checksum,
    )
    return IngestedChunk(
        source_id=ref.source_id,
        source_version=ref.version,
        source_version_id=make_source_version_id(ref.source_id, ref.version),
        source_checksum=chunk_checksum,
        document_id=document_id,
        chunk_id=chunk_id,
        ordinal=1,
        media_type="text/plain",
        locator=locator,
        content=content,
        content_checksum=chunk_checksum,
        ingestion_version=config.ingestion_version,
        chunk_rules_version=config.chunk_rules_version,
        chunk_rules_checksum=config.chunk_rules_checksum,
        index_version=config.index_version,
    )


def build_gold_lexical_corpus(
    cases: Sequence[GoldCase],
    *,
    split: GoldSplit,
    index_version: str = DEFAULT_INDEX_VERSION,
) -> GoldLexicalCorpus:
    """Index every distinct Gold document of one split; never mix splits."""

    documents: list[GoldCorpusDocument] = []
    seen: set[str] = set()
    for case in cases:
        if case.split is not split:
            continue
        for context in case.model_input.evidence_contexts:
            ref = context.evidence_ref
            if ref.source_id in seen:
                continue
            seen.add(ref.source_id)
            documents.append(
                GoldCorpusDocument(
                    chunk=_chunk_for_evidence(
                        ref, context.content, index_version=index_version
                    ),
                    evidence_ref=ref,
                )
            )
    if not documents:
        raise ValueError("Gold corpus requires at least one indexed document")
    chunks = tuple(document.chunk for document in documents)
    return GoldLexicalCorpus(
        split=split,
        index=DeterministicLexicalIndex(chunks, index_version=index_version),
        documents=tuple(documents),
    )


def gold_required_document_ids(label: GoldLabel) -> tuple[str, ...]:
    """Gold document references a retrieval run must surface for this case."""

    return tuple(
        ref.source_id
        for ref in label.required_evidence_refs
        if ref.source_id.startswith("DOC-")
    )


def mask_case_identifiers(query: str) -> str:
    """Remove every digit from a query so only topical terms remain.

    On the v1 corpus this exposes the degenerate case where documents differ from
    each other almost only by that number; on the clause-level v2 corpus it proves the
    score is not just an identifier leak, because topical matching must carry it.
    """

    masked = "".join(
        " " if char.isdigit() else char for char in query.replace("\u3000", " ")
    )
    return canonical_query_text(masked) or ""


def recall_at_k(ranked: Sequence[str], required: Sequence[str], *, k: int) -> float:
    if not required:
        raise ValueError("recall requires at least one required reference")
    top = set(ranked[:k])
    return sum(1 for ref in required if ref in top) / len(required)


def reciprocal_rank(ranked: Sequence[str], required: Sequence[str]) -> float:
    positions = [
        index + 1 for index, item in enumerate(ranked) if item in set(required)
    ]
    return 1.0 / positions[0] if positions else 0.0


def ndcg_at_k(ranked: Sequence[str], required: Sequence[str], *, k: int = 10) -> float:
    if not required:
        raise ValueError("nDCG requires at least one required reference")
    wanted = set(required)
    gains = [1.0 if item in wanted else 0.0 for item in ranked[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(wanted), k)))
    return dcg / ideal if ideal else 0.0


def query_fingerprint(query: str) -> str:
    return hashlib.sha256(
        (canonical_query_text(query) or "").encode("utf-8")
    ).hexdigest()[:16]
