from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from procurement_review.ingestion import (
    DEFAULT_CONFIG,
    IngestedChunk,
    IngestionConfig,
    LocatorKind,
    SourceLocator,
    make_chunk_id,
    make_document_id,
    make_source_version_id,
)
from procurement_review.retrieval import (
    FAKE_VECTOR_ADAPTER_VERSION,
    FUSION_VERSION,
    DeterministicFakeVectorAdapter,
    DeterministicHybridRetriever,
    DeterministicLexicalIndex,
    RecallScore,
    RetrievalIssue,
    RetrievalPort,
    RetrievalRequest,
    RetrievalStatus,
    VectorRecallPort,
    canonical_index_bytes,
    index_checksum,
)
from procurement_review.retrieval_postgres import (
    POSTGRES_FAKE_VECTOR_VERSION,
    IndexBuildStatus,
    IndexConsistencyError,
    IndexVersionMismatchError,
    PostgresDeterministicFakeVectorAdapter,
    PostgresLexicalRecallAdapter,
    PostgresRetrievalStore,
)

RETRIEVAL_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "procurement_review"
    / "resources"
    / "retrieval"
    / "fixture.v1.json"
)
RETRIEVAL_FIXTURE = json.loads(RETRIEVAL_FIXTURE_PATH.read_text(encoding="utf-8"))
FIXTURE_VERSION = RETRIEVAL_FIXTURE["fixture_version"]
INDEX_VERSION = DEFAULT_CONFIG.index_version
SOURCE_VERSION = RETRIEVAL_FIXTURE["source_version"]


def _checksum(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _chunk(source_id: str, content: str) -> IngestedChunk:
    source_bytes = content.encode("utf-8")
    source_checksum = _checksum(source_bytes)
    source_version_id = make_source_version_id(source_id, SOURCE_VERSION)
    document_id = make_document_id(
        source_id=source_id,
        source_version=SOURCE_VERSION,
        source_checksum=source_checksum,
        media_type="text/plain",
        ingestion_version=DEFAULT_CONFIG.ingestion_version,
    )
    locator = SourceLocator(
        kind=LocatorKind.TEXT_LINES,
        line_start=1,
        line_end=1,
    )
    content_checksum = _checksum(source_bytes)
    chunk_id = make_chunk_id(
        document_id=document_id,
        ordinal=1,
        locator=locator,
        content_checksum=content_checksum,
        chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
        chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
    )
    return IngestedChunk(
        source_id=source_id,
        source_version=SOURCE_VERSION,
        source_version_id=source_version_id,
        source_checksum=source_checksum,
        document_id=document_id,
        chunk_id=chunk_id,
        ordinal=1,
        media_type="text/plain",
        locator=locator,
        content=content,
        content_checksum=content_checksum,
        ingestion_version=DEFAULT_CONFIG.ingestion_version,
        chunk_rules_version=DEFAULT_CONFIG.chunk_rules_version,
        chunk_rules_checksum=DEFAULT_CONFIG.chunk_rules_checksum,
        index_version=INDEX_VERSION,
    )


@pytest.fixture(scope="module")
def chunks() -> tuple[IngestedChunk, ...]:
    return tuple(
        _chunk(document["source_id"], document["content"])
        for document in RETRIEVAL_FIXTURE["documents"]
    )


def _vector(
    chunks: tuple[IngestedChunk, ...], *, index_version: str = INDEX_VERSION
) -> DeterministicFakeVectorAdapter:
    chunk_ids_by_source = {chunk.source_id: chunk.chunk_id for chunk in chunks}
    return DeterministicFakeVectorAdapter(
        index_version=index_version,
        scores_by_query={
            query["query"]: {
                chunk_ids_by_source[source_id]: score
                for source_id, score in query["fake_vector_scores"].items()
            }
            for query in RETRIEVAL_FIXTURE["queries"]
        },
    )


def _retriever(
    chunks: tuple[IngestedChunk, ...],
    *,
    vector: VectorRecallPort | None = None,
) -> DeterministicHybridRetriever:
    return DeterministicHybridRetriever(
        lexical_index=DeterministicLexicalIndex(
            chunks,
            index_version=INDEX_VERSION,
        ),
        vector_port=vector or _vector(chunks),
    )


def test_versioned_fixture_and_fake_vector_are_explicitly_non_semantic(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    vector = _vector(chunks)
    retriever = _retriever(chunks, vector=vector)

    assert FIXTURE_VERSION == "retrieval-fixture-v1"
    assert RETRIEVAL_FIXTURE["schema_version"] == "retrieval-contract-fixture.v1"
    assert RETRIEVAL_FIXTURE["test_only"] is True
    assert RETRIEVAL_FIXTURE["index_version"] == INDEX_VERSION
    assert len(RETRIEVAL_FIXTURE["documents"]) == 3
    assert len(RETRIEVAL_FIXTURE["queries"]) == 2
    assert FUSION_VERSION == "weighted-rrf-v1"
    assert vector.adapter_version == FAKE_VECTOR_ADAPTER_VERSION
    assert vector.test_only is True
    assert isinstance(vector, VectorRecallPort)
    assert isinstance(retriever, RetrievalPort)


def test_lexical_recall_is_deterministic_and_uses_stable_tie_breaks(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    index = DeterministicLexicalIndex(chunks, index_version=INDEX_VERSION)

    first = index.recall("SANCTIONS supplier supplier", limit=3)
    second = index.recall("SANCTIONS supplier supplier", limit=3)

    assert first == second
    assert [item.chunk_id for item in first] == [chunks[1].chunk_id]
    assert first[0].score == 0.833333333333


def test_fusion_returns_route_scores_and_a_truthful_evidence_reference(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    response = _retriever(chunks).retrieve(
        RetrievalRequest(
            query="competitive bidding threshold",
            index_version=INDEX_VERSION,
            limit=3,
        )
    )

    assert response.status is RetrievalStatus.KNOWN
    assert response.issue is None
    assert response.fusion_version == FUSION_VERSION
    expected_order = RETRIEVAL_FIXTURE["queries"][0]["expected_source_order"]
    assert [item.source_id for item in response.results[: len(expected_order)]] == (
        expected_order
    )

    hit = response.results[0]
    chunk = chunks[0]
    assert hit.chunk_id == chunk.chunk_id
    assert hit.document_id == chunk.document_id
    assert hit.source_version == SOURCE_VERSION
    assert hit.source_version_id == chunk.source_version_id
    assert hit.source_checksum == chunk.source_checksum
    assert hit.content_checksum == chunk.content_checksum
    assert hit.locator == chunk.locator
    assert hit.content == chunk.content
    assert hit.lexical_score == 1.0
    assert hit.vector_score == 0.70
    assert hit.lexical_rank == 1
    assert hit.vector_rank == 2
    assert hit.fusion_score > response.results[1].fusion_score
    assert hit.evidence_ref == chunk.to_evidence_ref()


def test_same_index_and_query_rebuild_byte_identically_in_any_input_order(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    forward_index = DeterministicLexicalIndex(chunks, index_version=INDEX_VERSION)
    reverse_chunks = tuple(reversed(chunks))
    reverse_index = DeterministicLexicalIndex(
        reverse_chunks,
        index_version=INDEX_VERSION,
    )
    request = RetrievalRequest(query="sanctions supplier", index_version=INDEX_VERSION)

    assert canonical_index_bytes(
        chunks, index_version=INDEX_VERSION
    ) == canonical_index_bytes(reverse_chunks, index_version=INDEX_VERSION)
    assert forward_index.index_checksum == reverse_index.index_checksum
    assert index_checksum(chunks, index_version=INDEX_VERSION) == (
        forward_index.index_checksum
    )

    first = _retriever(chunks).retrieve(request)
    second = _retriever(reverse_chunks, vector=_vector(chunks)).retrieve(request)
    assert first.model_dump_json() == second.model_dump_json()


def test_content_or_index_version_changes_the_derived_index_checksum(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    changed = (
        _chunk(
            "fixture-competition",
            "Competitive bidding is required above a changed procurement threshold.",
        ),
        *chunks[1:],
    )

    assert index_checksum(changed, index_version=INDEX_VERSION) != index_checksum(
        chunks, index_version=INDEX_VERSION
    )
    assert index_checksum(chunks, index_version="retrieval-index-v2") != (
        index_checksum(chunks, index_version=INDEX_VERSION)
    )


def test_no_match_is_unknown_and_constructs_no_reference(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    response = _retriever(chunks).retrieve(
        RetrievalRequest(query="unrepresented phrase", index_version=INDEX_VERSION)
    )

    assert response.status is RetrievalStatus.UNKNOWN
    assert response.issue is RetrievalIssue.NO_MATCH
    assert response.results == ()


def test_index_version_mismatch_is_unknown_without_calling_recall(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    response = _retriever(chunks).retrieve(
        RetrievalRequest(query="sanctions", index_version="retrieval-index-v2")
    )

    assert response.status is RetrievalStatus.UNKNOWN
    assert response.issue is RetrievalIssue.INDEX_VERSION_MISMATCH
    assert response.results == ()


def test_vector_index_version_mismatch_fails_closed(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    response = _retriever(
        chunks,
        vector=_vector(chunks, index_version="retrieval-index-v2"),
    ).retrieve(RetrievalRequest(query="sanctions", index_version=INDEX_VERSION))

    assert response.status is RetrievalStatus.UNKNOWN
    assert response.issue is RetrievalIssue.INDEX_VERSION_MISMATCH
    assert response.results == ()


def test_insufficient_evidence_discards_partial_candidates(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    response = _retriever(chunks).retrieve(
        RetrievalRequest(
            query="competitive bidding threshold",
            index_version=INDEX_VERSION,
            limit=3,
            minimum_results=3,
        )
    )

    assert response.status is RetrievalStatus.UNKNOWN
    assert response.issue is RetrievalIssue.INSUFFICIENT_EVIDENCE
    assert response.results == ()


def test_query_parse_failure_is_explicit_and_constructs_no_reference(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    response = _retriever(chunks).retrieve(
        RetrievalRequest(query="!!! ???", index_version=INDEX_VERSION)
    )

    assert response.status is RetrievalStatus.INVALID_INPUT
    assert response.issue is RetrievalIssue.QUERY_PARSE_FAILURE
    assert response.results == ()


def test_vector_failure_is_explicit_and_constructs_no_reference(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    class FailingVectorPort:
        index_version = INDEX_VERSION
        adapter_version = "failing-test-vector-v1"
        test_only = True

        def recall(self, query: str, *, limit: int) -> tuple[()]:
            raise RuntimeError("fixture failure")

    response = _retriever(chunks, vector=FailingVectorPort()).retrieve(
        RetrievalRequest(query="sanctions", index_version=INDEX_VERSION)
    )

    assert response.status is RetrievalStatus.UPSTREAM_FAILURE
    assert response.issue is RetrievalIssue.VECTOR_RECALL_FAILURE
    assert response.results == ()
    assert "fixture failure" not in response.detail


def test_vector_cannot_return_a_chunk_outside_the_loaded_index(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    vector = DeterministicFakeVectorAdapter(
        index_version=INDEX_VERSION,
        scores_by_query={"sanctions": {"chunk-not-in-index": 0.99}},
    )
    response = _retriever(chunks, vector=vector).retrieve(
        RetrievalRequest(query="sanctions", index_version=INDEX_VERSION)
    )

    assert response.status is RetrievalStatus.UPSTREAM_FAILURE
    assert response.issue is RetrievalIssue.VECTOR_RECALL_FAILURE
    assert response.results == ()


def test_lexical_adapter_outside_chunk_is_attributed_to_lexical_failure(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    canonical = DeterministicLexicalIndex(chunks, index_version=INDEX_VERSION)

    class RogueLexicalPort:
        index_version = INDEX_VERSION
        index_checksum = canonical.index_checksum
        chunks = canonical.chunks

        def recall(self, query: str, *, limit: int) -> tuple[RecallScore, ...]:
            return (RecallScore(chunk_id="chunk-not-in-index", score=0.99),)

    response = DeterministicHybridRetriever(
        lexical_index=RogueLexicalPort(),
        vector_port=DeterministicFakeVectorAdapter(
            index_version=INDEX_VERSION,
            scores_by_query={},
        ),
    ).retrieve(RetrievalRequest(query="sanctions", index_version=INDEX_VERSION))

    assert response.status is RetrievalStatus.UPSTREAM_FAILURE
    assert response.issue is RetrievalIssue.LEXICAL_RECALL_FAILURE
    assert response.results == ()


def test_request_and_index_contracts_reject_inconsistent_inputs(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    with pytest.raises(ValidationError, match="minimum_results cannot exceed limit"):
        RetrievalRequest(
            query="valid query",
            index_version=INDEX_VERSION,
            limit=1,
            minimum_results=2,
        )
    with pytest.raises(ValueError, match="chunk IDs must be unique"):
        DeterministicLexicalIndex(
            (chunks[0], chunks[0]),
            index_version=INDEX_VERSION,
        )

    wrong_version = IngestionConfig(index_version="retrieval-index-v2")
    wrong_chunk = chunks[0].model_copy(
        update={"index_version": wrong_version.index_version}
    )
    with pytest.raises(ValueError, match="chunk index version"):
        DeterministicLexicalIndex(
            (wrong_chunk,),
            index_version=INDEX_VERSION,
        )


def test_postgres_fts_pgvector_rebuild_and_version_contract(
    chunks: tuple[IngestedChunk, ...],
) -> None:
    database_url = os.environ.get("TEST_POSTGRES_DSN")
    if database_url is None:
        pytest.skip("TEST_POSTGRES_DSN is required for PostgreSQL integration")

    embeddings = {
        chunks[0].chunk_id: (1.0, 0.0, 0.0),
        chunks[1].chunk_id: (0.0, 1.0, 0.0),
        chunks[2].chunk_id: (0.0, 0.0, 1.0),
    }
    with PostgresRetrievalStore(database_url, vector_dimension=3) as store:
        first = store.build_index(
            chunks=chunks,
            embeddings=embeddings,
            index_version=INDEX_VERSION,
            embedding_adapter_version=FAKE_VECTOR_ADAPTER_VERSION,
        )
        assert first.status in {
            IndexBuildStatus.CREATED,
            IndexBuildStatus.REBUILT,
            IndexBuildStatus.UNCHANGED,
        }

        second = store.build_index(
            chunks=tuple(reversed(chunks)),
            embeddings=embeddings,
            index_version=INDEX_VERSION,
            embedding_adapter_version=FAKE_VECTOR_ADAPTER_VERSION,
        )
        assert second.status is IndexBuildStatus.UNCHANGED
        assert second.index_checksum == first.index_checksum

        lexical = PostgresLexicalRecallAdapter(
            store,
            index_version=INDEX_VERSION,
        )
        vector = PostgresDeterministicFakeVectorAdapter(
            store,
            index_version=INDEX_VERSION,
            query_vectors={"competitive bidding threshold": (1.0, 0.0, 0.0)},
        )
        lexical_results = lexical.recall("competitive bidding threshold", limit=3)
        vector_results = vector.recall("competitive bidding threshold", limit=3)
        assert lexical_results[0].chunk_id == chunks[0].chunk_id
        assert lexical_results[0].score > 0.0
        assert vector_results[0].chunk_id == chunks[0].chunk_id
        assert vector_results[0].score == 1.0
        assert vector.adapter_version == POSTGRES_FAKE_VECTOR_VERSION
        assert vector.test_only is True

        response = DeterministicHybridRetriever(
            lexical_index=lexical,
            vector_port=vector,
        ).retrieve(
            RetrievalRequest(
                query="competitive bidding threshold",
                index_version=INDEX_VERSION,
            )
        )
        assert response.status is RetrievalStatus.KNOWN
        assert response.results[0].chunk_id == chunks[0].chunk_id
        assert response.index_checksum == first.index_checksum

        tampered_embeddings = dict(embeddings)
        tampered_embeddings[chunks[0].chunk_id] = (0.5, 0.5, 0.0)
        with pytest.raises(IndexConsistencyError, match="same index version"):
            store.build_index(
                chunks=chunks,
                embeddings=tampered_embeddings,
                index_version=INDEX_VERSION,
                embedding_adapter_version=FAKE_VECTOR_ADAPTER_VERSION,
            )
        with pytest.raises(IndexVersionMismatchError):
            store.snapshot(index_version="retrieval-index-v2")

        v2_chunks = tuple(
            IngestedChunk.model_validate(
                {
                    **chunk.model_dump(mode="python"),
                    "index_version": "retrieval-index-v2",
                }
            )
            for chunk in chunks
        )
        rebuilt = store.build_index(
            chunks=v2_chunks,
            embeddings=embeddings,
            index_version="retrieval-index-v2",
            embedding_adapter_version=FAKE_VECTOR_ADAPTER_VERSION,
        )
        assert rebuilt.status is IndexBuildStatus.REBUILT
        assert rebuilt.index_checksum != first.index_checksum

        stale_response = DeterministicHybridRetriever(
            lexical_index=lexical,
            vector_port=vector,
        ).retrieve(
            RetrievalRequest(query="competitive bidding", index_version=INDEX_VERSION)
        )
        assert stale_response.status is RetrievalStatus.UPSTREAM_FAILURE
        assert stale_response.issue is RetrievalIssue.LEXICAL_RECALL_FAILURE
        assert stale_response.results == ()
