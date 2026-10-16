"""Deterministic retrieval contracts and replaceable recall ports.

The vector adapter included here is deliberately a test fake.  It consumes
predeclared scores and does not provide embeddings or semantic retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field, StringConstraints, ValidationError, model_validator

from procurement_review.contracts import EvidenceRef, Identifier, StrictContract
from procurement_review.ingestion import (
    DEFAULT_CONFIG,
    CorpusArtifact,
    IngestedChunk,
    IngestionConfig,
    IngestionError,
    SourceLocator,
    ingest_manifest,
)
from procurement_review.manifest import PublicSourceManifest, load_manifest
from procurement_review.public_corpus import (
    DEFAULT_CACHE_DIR,
    PublicCorpusError,
    verify_manifest_cache,
)

RETRIEVAL_SCHEMA_VERSION = "1.0"
FUSION_VERSION = "weighted-rrf-v1"
LEXICAL_ADAPTER_VERSION = "deterministic-lexical-v1"
IDF_LEXICAL_ADAPTER_VERSION = "deterministic-lexical-idf-v1"
FAKE_VECTOR_ADAPTER_VERSION = "deterministic-fake-vector-v1"
NO_VECTOR_ADAPTER_VERSION = "no-vector-v1"

QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]
Score = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
Checksum = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")


class RetrievalStatus(StrEnum):
    """Outcome of one retrieval request."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    INVALID_INPUT = "invalid_input"
    UPSTREAM_FAILURE = "upstream_failure"


class RetrievalIssue(StrEnum):
    """Stable fail-closed reasons for unresolved retrieval."""

    NO_MATCH = "no_match"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INDEX_VERSION_MISMATCH = "index_version_mismatch"
    QUERY_PARSE_FAILURE = "query_parse_failure"
    LEXICAL_RECALL_FAILURE = "lexical_recall_failure"
    VECTOR_RECALL_FAILURE = "vector_recall_failure"
    EVIDENCE_MAPPING_FAILURE = "evidence_mapping_failure"


class RetrievalRequest(StrictContract):
    """Framework-independent retrieval request."""

    query: QueryText
    index_version: Identifier
    limit: Annotated[int, Field(strict=True, ge=1, le=50)] = 10
    minimum_results: Annotated[int, Field(strict=True, ge=1, le=50)] = 1

    @model_validator(mode="after")
    def minimum_does_not_exceed_limit(self) -> RetrievalRequest:
        if self.minimum_results > self.limit:
            raise ValueError("minimum_results cannot exceed limit")
        return self


class RecallScore(StrictContract):
    """One adapter-owned score without storage or framework objects."""

    chunk_id: Identifier
    score: Score


@runtime_checkable
class LexicalRecallPort(Protocol):
    """Narrow lexical recall boundary with a traceable chunk snapshot."""

    @property
    def index_version(self) -> str: ...

    @property
    def index_checksum(self) -> str: ...

    @property
    def chunks(self) -> Sequence[IngestedChunk]: ...

    def recall(self, query: str, *, limit: int) -> Sequence[RecallScore]: ...


@runtime_checkable
class VectorRecallPort(Protocol):
    """Narrow vector recall boundary; implementations own their index access."""

    @property
    def index_version(self) -> str: ...

    @property
    def adapter_version(self) -> str: ...

    @property
    def test_only(self) -> bool: ...

    def recall(self, query: str, *, limit: int) -> Sequence[RecallScore]: ...


@runtime_checkable
class RetrievalPort(Protocol):
    """Narrow retrieval boundary exposed to future read-only workers."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...


class FusionPolicy(StrictContract):
    """Versioned weighted reciprocal-rank fusion configuration."""

    version: Literal["weighted-rrf-v1"] = FUSION_VERSION
    rrf_k: Literal[60] = 60
    lexical_weight: Literal[1] = 1
    vector_weight: Literal[1] = 1


FUSION_POLICY = FusionPolicy()


class RetrievalHit(StrictContract):
    """Traceable retrieval result with scores from every recall route."""

    rank: Annotated[int, Field(strict=True, ge=1)]
    source_id: Identifier
    source_version: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    source_version_id: Identifier
    source_checksum: Checksum
    document_id: Identifier
    chunk_id: Identifier
    ordinal: Annotated[int, Field(strict=True, ge=1)]
    locator: SourceLocator
    content: Annotated[str, StringConstraints(min_length=1)]
    content_checksum: Checksum
    ingestion_version: Identifier
    chunk_rules_version: Identifier
    chunk_rules_checksum: Checksum
    index_version: Identifier
    lexical_score: Score
    vector_score: Score
    lexical_rank: Annotated[int, Field(strict=True, ge=1)] | None = None
    vector_rank: Annotated[int, Field(strict=True, ge=1)] | None = None
    fusion_score: Score
    evidence_ref: EvidenceRef

    @model_validator(mode="after")
    def evidence_matches_chunk(self) -> RetrievalHit:
        reference = self.evidence_ref
        if (
            reference.source_id != self.source_id
            or reference.version != self.source_version_id
            or reference.chunk_id != self.chunk_id
            or reference.content_checksum != self.content_checksum
        ):
            raise ValueError("evidence_ref does not match the retrieved chunk")
        return self


class RetrievalResponse(StrictContract):
    """A known result or an explicit unresolved outcome with no references."""

    query: QueryText
    requested_index_version: Identifier
    index_version: Identifier
    index_checksum: Checksum
    fusion_version: Literal["weighted-rrf-v1"] = FUSION_VERSION
    vector_adapter_version: Identifier
    status: RetrievalStatus
    results: tuple[RetrievalHit, ...] = ()
    issue: RetrievalIssue | None = None
    detail: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = (
        None
    )

    @model_validator(mode="after")
    def result_matches_status(self) -> RetrievalResponse:
        if self.status is RetrievalStatus.KNOWN:
            if not self.results:
                raise ValueError("known retrieval requires at least one result")
            if self.issue is not None or self.detail is not None:
                raise ValueError("known retrieval cannot carry an unresolved issue")
            if tuple(hit.rank for hit in self.results) != tuple(
                range(1, len(self.results) + 1)
            ):
                raise ValueError("retrieval result ranks must be contiguous")
        else:
            if self.results:
                raise ValueError("unresolved retrieval cannot construct references")
            if self.issue is None or self.detail is None:
                raise ValueError("unresolved retrieval requires an issue and detail")
        return self


@dataclass(frozen=True, slots=True)
class _RouteMatch:
    chunk_id: str
    score: float
    rank: int


def _query_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def canonical_query_text(value: str) -> str:
    """Return the stable lexical form shared by retrieval adapters."""

    return " ".join(_query_tokens(value))


def canonical_index_bytes(
    chunks: Sequence[IngestedChunk], *, index_version: str
) -> bytes:
    """Serialize a derived retrieval index independently of input ordering."""

    payload = {
        "index_version": index_version,
        "chunks": [
            chunk.model_dump(mode="json")
            for chunk in sorted(chunks, key=lambda item: item.chunk_id)
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def index_checksum(chunks: Sequence[IngestedChunk], *, index_version: str) -> str:
    """Return the content-addressed checksum of a derived retrieval index."""

    return (
        "sha256:"
        + hashlib.sha256(
            canonical_index_bytes(chunks, index_version=index_version)
        ).hexdigest()
    )


class DeterministicLexicalIndex:
    """Small deterministic lexical adapter used behind the retrieval port."""

    adapter_version = LEXICAL_ADAPTER_VERSION

    def __init__(self, chunks: Sequence[IngestedChunk], *, index_version: str) -> None:
        if not chunks:
            raise ValueError("retrieval index requires at least one chunk")
        if any(chunk.index_version != index_version for chunk in chunks):
            raise ValueError("chunk index version does not match the index")

        ordered = tuple(sorted(chunks, key=lambda item: item.chunk_id))
        chunk_ids = tuple(chunk.chunk_id for chunk in ordered)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("retrieval index chunk IDs must be unique")

        document_ordinals = tuple(
            (chunk.document_id, chunk.ordinal) for chunk in ordered
        )
        if len(document_ordinals) != len(set(document_ordinals)):
            raise ValueError("chunk ordinals must be unique within each document")

        self._chunks = ordered
        self._tokens = MappingProxyType(
            {chunk.chunk_id: Counter(_query_tokens(chunk.content)) for chunk in ordered}
        )
        self.index_version = index_version
        self.index_checksum = index_checksum(ordered, index_version=index_version)

    @property
    def chunks(self) -> tuple[IngestedChunk, ...]:
        return self._chunks

    def recall(self, query: str, *, limit: int) -> tuple[RecallScore, ...]:
        query_counts = Counter(_query_tokens(query))
        if not query_counts or limit < 1:
            return ()

        total_query_tokens = sum(query_counts.values())
        unique_query_terms = len(query_counts)
        scored: list[RecallScore] = []
        for chunk in self._chunks:
            chunk_counts = self._tokens[chunk.chunk_id]
            matched_occurrences = sum(
                min(count, chunk_counts.get(term, 0))
                for term, count in query_counts.items()
            )
            if matched_occurrences == 0:
                continue
            occurrence_coverage = matched_occurrences / total_query_tokens
            unique_coverage = (
                sum(term in chunk_counts for term in query_counts) / unique_query_terms
            )
            score = round((occurrence_coverage + unique_coverage) / 2, 12)
            scored.append(RecallScore(chunk_id=chunk.chunk_id, score=score))

        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return tuple(scored[:limit])


class DeterministicFakeVectorAdapter:
    """Test-only vector port backed by declared scores, not embeddings."""

    adapter_version = FAKE_VECTOR_ADAPTER_VERSION
    test_only = True

    def __init__(
        self,
        *,
        index_version: str,
        scores_by_query: Mapping[str, Mapping[str, float]],
    ) -> None:
        normalized: dict[str, tuple[RecallScore, ...]] = {}
        for query, chunk_scores in scores_by_query.items():
            canonical_query = canonical_query_text(query)
            if not canonical_query:
                raise ValueError("fake vector query must contain searchable terms")
            if canonical_query in normalized:
                raise ValueError("fake vector queries must be unique after parsing")

            scores: list[RecallScore] = []
            for chunk_id, score in chunk_scores.items():
                if not math.isfinite(score) or score < 0.0 or score > 1.0:
                    raise ValueError(
                        "fake vector scores must be finite and within 0..1"
                    )
                if score > 0.0:
                    scores.append(RecallScore(chunk_id=chunk_id, score=score))
            scores.sort(key=lambda item: (-item.score, item.chunk_id))
            normalized[canonical_query] = tuple(scores)

        self.index_version = index_version
        self._scores = MappingProxyType(normalized)

    def recall(self, query: str, *, limit: int) -> tuple[RecallScore, ...]:
        if limit < 1:
            return ()
        return self._scores.get(canonical_query_text(query), ())[:limit]


class DeterministicIdfLexicalIndex(DeterministicLexicalIndex):
    """Deterministic lexical recall weighted by inverse document frequency.

    The frozen unigram adapter scores raw coverage, so boilerplate shared by every
    document in a library counts as much as the few terms that identify one of them.
    Weighting each query term by how many indexed chunks contain it keeps the
    computation deterministic and offline while making the ranking reflect the
    evidence that actually distinguishes a chunk.  It is a real computed retrieval,
    not a predeclared score table.
    """

    adapter_version = IDF_LEXICAL_ADAPTER_VERSION

    def __init__(self, chunks: Sequence[IngestedChunk], *, index_version: str) -> None:
        super().__init__(chunks, index_version=index_version)
        documents = len(self._chunks)
        frequency: Counter[str] = Counter()
        for chunk in self._chunks:
            frequency.update(set(self._tokens[chunk.chunk_id]))
        self._idf = MappingProxyType(
            {
                term: math.log((1 + documents) / (1 + count)) + 1.0
                for term, count in frequency.items()
            }
        )
        self._default_idf = math.log(1 + documents) + 1.0

    def recall(self, query: str, *, limit: int) -> tuple[RecallScore, ...]:
        """Rank by IDF-weighted term coverage of the query, computed per chunk."""

        query_counts = Counter(_query_tokens(query))
        if not query_counts or limit < 1:
            return ()
        query_weight = {
            term: self._idf.get(term, self._default_idf) * (1 + math.log(count))
            for term, count in query_counts.items()
        }
        total_weight = sum(query_weight.values())
        scored: list[RecallScore] = []
        for chunk in self._chunks:
            chunk_counts = self._tokens[chunk.chunk_id]
            matched = sum(
                self._idf.get(term, self._default_idf) * (1 + math.log(present))
                for term, present in (
                    (term, chunk_counts.get(term, 0)) for term in query_counts
                )
                if present
            )
            if matched <= 0.0:
                continue
            score = round(min(1.0, matched / total_weight), 12)
            if score > 0.0:
                scored.append(RecallScore(chunk_id=chunk.chunk_id, score=score))
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return tuple(scored[:limit])


class NoVectorAdapter:
    """The explicit absence of a vector route.

    A real retrieval measurement must not be carried by predeclared scores, so the
    lexical-only mode names its missing route instead of substituting a fake one.
    It contributes no candidate, no score and no rank.
    """

    adapter_version = NO_VECTOR_ADAPTER_VERSION
    test_only = False

    def __init__(self, *, index_version: str) -> None:
        self.index_version = index_version

    def recall(self, query: str, *, limit: int) -> tuple[RecallScore, ...]:
        del query, limit
        return ()


class DeterministicHybridRetriever:
    """Hybrid retrieval implementation with lexical and vector port isolation."""

    def __init__(
        self,
        *,
        lexical_index: LexicalRecallPort,
        vector_port: VectorRecallPort,
        fusion_policy: FusionPolicy = FUSION_POLICY,
    ) -> None:
        self._lexical = lexical_index
        self._vector = vector_port
        self._fusion = fusion_policy
        self._chunks = MappingProxyType(
            {chunk.chunk_id: chunk for chunk in lexical_index.chunks}
        )

    def _response(
        self,
        request: RetrievalRequest,
        *,
        status: RetrievalStatus,
        results: tuple[RetrievalHit, ...] = (),
        issue: RetrievalIssue | None = None,
        detail: str | None = None,
    ) -> RetrievalResponse:
        return RetrievalResponse(
            query=request.query,
            requested_index_version=request.index_version,
            index_version=self._lexical.index_version,
            index_checksum=self._lexical.index_checksum,
            fusion_version=self._fusion.version,
            vector_adapter_version=self._vector.adapter_version,
            status=status,
            results=results,
            issue=issue,
            detail=detail,
        )

    @staticmethod
    def _rank_route(scores: Sequence[RecallScore]) -> tuple[_RouteMatch, ...]:
        ordered = sorted(scores, key=lambda item: (-item.score, item.chunk_id))
        if len({item.chunk_id for item in ordered}) != len(ordered):
            raise ValueError("recall adapter returned duplicate chunk IDs")
        return tuple(
            _RouteMatch(chunk_id=item.chunk_id, score=item.score, rank=rank)
            for rank, item in enumerate(ordered, start=1)
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        if request.index_version != self._lexical.index_version:
            return self._response(
                request,
                status=RetrievalStatus.UNKNOWN,
                issue=RetrievalIssue.INDEX_VERSION_MISMATCH,
                detail="requested index version does not match the loaded index",
            )
        if self._vector.index_version != self._lexical.index_version:
            return self._response(
                request,
                status=RetrievalStatus.UNKNOWN,
                issue=RetrievalIssue.INDEX_VERSION_MISMATCH,
                detail="vector and lexical indexes use different versions",
            )

        parsed_query = canonical_query_text(request.query)
        if not parsed_query:
            return self._response(
                request,
                status=RetrievalStatus.INVALID_INPUT,
                issue=RetrievalIssue.QUERY_PARSE_FAILURE,
                detail="query contains no searchable terms",
            )

        recall_limit = min(
            len(self._chunks), max(request.limit, request.minimum_results) * 4
        )
        try:
            lexical = self._rank_route(
                self._lexical.recall(parsed_query, limit=recall_limit)
            )
        # Adapter implementations are untrusted at this fail-closed boundary.
        except Exception:  # noqa: BLE001
            return self._response(
                request,
                status=RetrievalStatus.UPSTREAM_FAILURE,
                issue=RetrievalIssue.LEXICAL_RECALL_FAILURE,
                detail="lexical recall adapter failed",
            )
        try:
            vector = self._rank_route(
                self._vector.recall(parsed_query, limit=recall_limit)
            )
        # Adapter implementations are untrusted at this fail-closed boundary.
        except Exception:  # noqa: BLE001
            return self._response(
                request,
                status=RetrievalStatus.UPSTREAM_FAILURE,
                issue=RetrievalIssue.VECTOR_RECALL_FAILURE,
                detail="vector recall adapter failed",
            )

        lexical_by_id = {item.chunk_id: item for item in lexical}
        vector_by_id = {item.chunk_id: item for item in vector}
        unknown_lexical_ids = set(lexical_by_id) - set(self._chunks)
        if unknown_lexical_ids:
            return self._response(
                request,
                status=RetrievalStatus.UPSTREAM_FAILURE,
                issue=RetrievalIssue.LEXICAL_RECALL_FAILURE,
                detail="lexical recall returned a chunk outside the loaded index",
            )
        unknown_vector_ids = set(vector_by_id) - set(self._chunks)
        if unknown_vector_ids:
            return self._response(
                request,
                status=RetrievalStatus.UPSTREAM_FAILURE,
                issue=RetrievalIssue.VECTOR_RECALL_FAILURE,
                detail="recall adapter returned a chunk outside the loaded index",
            )
        candidate_ids = set(lexical_by_id) | set(vector_by_id)
        if not candidate_ids:
            return self._response(
                request,
                status=RetrievalStatus.UNKNOWN,
                issue=RetrievalIssue.NO_MATCH,
                detail="no indexed chunk matched the query",
            )

        fused: list[tuple[str, float, float, float, int | None, int | None]] = []
        for chunk_id in candidate_ids:
            lexical_match = lexical_by_id.get(chunk_id)
            vector_match = vector_by_id.get(chunk_id)
            fusion_score = 0.0
            if lexical_match is not None:
                fusion_score += self._fusion.lexical_weight / (
                    self._fusion.rrf_k + lexical_match.rank
                )
            if vector_match is not None:
                fusion_score += self._fusion.vector_weight / (
                    self._fusion.rrf_k + vector_match.rank
                )
            fused.append(
                (
                    chunk_id,
                    round(fusion_score, 15),
                    lexical_match.score if lexical_match else 0.0,
                    vector_match.score if vector_match else 0.0,
                    lexical_match.rank if lexical_match else None,
                    vector_match.rank if vector_match else None,
                )
            )

        fused.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        selected = fused[: request.limit]
        if len(selected) < request.minimum_results:
            return self._response(
                request,
                status=RetrievalStatus.UNKNOWN,
                issue=RetrievalIssue.INSUFFICIENT_EVIDENCE,
                detail="retrieval produced fewer results than explicitly required",
            )

        results: list[RetrievalHit] = []
        try:
            for rank, (
                chunk_id,
                fusion_score,
                lexical_score,
                vector_score,
                lexical_rank,
                vector_rank,
            ) in enumerate(selected, start=1):
                chunk = self._chunks[chunk_id]
                results.append(
                    RetrievalHit(
                        rank=rank,
                        source_id=chunk.source_id,
                        source_version=chunk.source_version,
                        source_version_id=chunk.source_version_id,
                        source_checksum=chunk.source_checksum,
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        ordinal=chunk.ordinal,
                        locator=chunk.locator,
                        content=chunk.content,
                        content_checksum=chunk.content_checksum,
                        ingestion_version=chunk.ingestion_version,
                        chunk_rules_version=chunk.chunk_rules_version,
                        chunk_rules_checksum=chunk.chunk_rules_checksum,
                        index_version=chunk.index_version,
                        lexical_score=lexical_score,
                        vector_score=vector_score,
                        lexical_rank=lexical_rank,
                        vector_rank=vector_rank,
                        fusion_score=fusion_score,
                        evidence_ref=chunk.to_evidence_ref(),
                    )
                )
        except ValidationError:
            return self._response(
                request,
                status=RetrievalStatus.UPSTREAM_FAILURE,
                issue=RetrievalIssue.EVIDENCE_MAPPING_FAILURE,
                detail="retrieved chunk could not produce a valid evidence reference",
            )

        return self._response(
            request,
            status=RetrievalStatus.KNOWN,
            results=tuple(results),
        )


def rebuild_public_lexical_index(
    manifest: PublicSourceManifest,
    source_paths: Mapping[str, str | Path],
    *,
    config: IngestionConfig = DEFAULT_CONFIG,
) -> tuple[CorpusArtifact, DeterministicLexicalIndex]:
    """Rebuild the complete public corpus and its in-memory lexical derivative."""

    corpus = ingest_manifest(manifest, source_paths, config=config)
    chunks = tuple(chunk for document in corpus.documents for chunk in document.chunks)
    return corpus, DeterministicLexicalIndex(
        chunks,
        index_version=corpus.index_version,
    )


def public_index_summary(
    corpus: CorpusArtifact,
    lexical_index: DeterministicLexicalIndex,
    *,
    rebuild_verified: bool,
) -> dict[str, bool | int | str]:
    canonical_bytes = canonical_index_bytes(
        lexical_index.chunks,
        index_version=lexical_index.index_version,
    )
    return {
        "canonical_index_bytes": len(canonical_bytes),
        "chunk_count": corpus.chunk_count,
        "corpus_checksum": corpus.corpus_checksum,
        "document_count": corpus.document_count,
        "index_checksum": lexical_index.index_checksum,
        "index_version": lexical_index.index_version,
        "lexical_adapter_version": lexical_index.adapter_version,
        "rebuild_verified": rebuild_verified,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--verify-rebuild",
        action="store_true",
        help="rebuild twice and require byte-identical canonical indexes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify cached sources and rebuild a deterministic in-memory index."""

    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = verify_manifest_cache(manifest, args.cache_dir)
        source_paths = {source.source_id: source.path for source in report.sources}
        corpus, lexical_index = rebuild_public_lexical_index(
            manifest,
            source_paths,
        )
        if args.verify_rebuild:
            rebuilt_corpus, rebuilt_index = rebuild_public_lexical_index(
                manifest,
                source_paths,
            )
            first_bytes = canonical_index_bytes(
                lexical_index.chunks,
                index_version=lexical_index.index_version,
            )
            second_bytes = canonical_index_bytes(
                rebuilt_index.chunks,
                index_version=rebuilt_index.index_version,
            )
            if (
                corpus.canonical_bytes() != rebuilt_corpus.canonical_bytes()
                or first_bytes != second_bytes
            ):
                raise ValueError("two public index rebuilds differ")
    except (
        OSError,
        ValidationError,
        PublicCorpusError,
        IngestionError,
        ValueError,
    ) as exc:
        print(f"retrieval index rebuild failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            public_index_summary(
                corpus,
                lexical_index,
                rebuild_verified=args.verify_rebuild,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
