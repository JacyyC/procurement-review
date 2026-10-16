"""PostgreSQL FTS and pgvector adapters for the narrow retrieval ports.

The storage layer owns SQLAlchemy and PostgreSQL objects.  Domain callers see
only ingestion chunks, versioned build results, and ``RecallScore`` values.
The pgvector query adapter in this module is explicitly test-only until an
Embedding model passes the separate G-MODEL decision gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Self

from pgvector.sqlalchemy import Vector
from pydantic import Field, TypeAdapter
from sqlalchemy import (
    CheckConstraint,
    Column,
    Computed,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from procurement_review.contracts import Identifier, StrictContract
from procurement_review.ingestion import IngestedChunk
from procurement_review.retrieval import (
    RecallScore,
    canonical_query_text,
    index_checksum,
)

POSTGRES_ADAPTER_VERSION = "postgres-retrieval-v1"
POSTGRES_FAKE_VECTOR_VERSION = "postgres-deterministic-fake-vector-v1"

Checksum = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
_identifier_adapter = TypeAdapter(Identifier)


class PostgresRetrievalError(RuntimeError):
    """Base failure raised inside the PostgreSQL retrieval boundary."""


class PostgresStorageError(PostgresRetrievalError):
    """PostgreSQL could not complete a storage or query operation."""


class IndexVersionMismatchError(PostgresRetrievalError):
    """A caller attempted to read a different index version."""


class IndexConsistencyError(PostgresRetrievalError):
    """Stored metadata, chunks, vectors, or checksums disagree."""


class IndexBuildStatus(StrEnum):
    CREATED = "created"
    REBUILT = "rebuilt"
    UNCHANGED = "unchanged"


class PostgresIndexBuildResult(StrictContract):
    adapter_version: Identifier = POSTGRES_ADAPTER_VERSION
    status: IndexBuildStatus
    index_version: Identifier
    index_checksum: Checksum
    chunk_count: Annotated[int, Field(strict=True, ge=1)]
    vector_dimension: Annotated[int, Field(strict=True, ge=1, le=4_096)]
    embedding_adapter_version: Identifier


@dataclass(frozen=True, slots=True)
class PostgresIndexSnapshot:
    index_version: str
    index_checksum: str
    embedding_adapter_version: str
    vector_dimension: int
    chunks: tuple[IngestedChunk, ...]


def _canonical_storage_bytes(
    *,
    chunks: Sequence[IngestedChunk],
    embeddings: Mapping[str, Sequence[float]],
    index_version: str,
    embedding_adapter_version: str,
    vector_dimension: int,
) -> bytes:
    payload = {
        "adapter_version": POSTGRES_ADAPTER_VERSION,
        "chunk_index_checksum": index_checksum(
            chunks,
            index_version=index_version,
        ),
        "embedding_adapter_version": embedding_adapter_version,
        "embeddings": [
            {
                "chunk_id": chunk_id,
                "vector": list(embeddings[chunk_id]),
            }
            for chunk_id in sorted(embeddings)
        ],
        "index_version": index_version,
        "vector_dimension": vector_dimension,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def postgres_index_checksum(
    *,
    chunks: Sequence[IngestedChunk],
    embeddings: Mapping[str, Sequence[float]],
    index_version: str,
    embedding_adapter_version: str,
    vector_dimension: int,
) -> str:
    """Checksum the complete persisted FTS/pgvector derivative."""

    return (
        "sha256:"
        + hashlib.sha256(
            _canonical_storage_bytes(
                chunks=chunks,
                embeddings=embeddings,
                index_version=index_version,
                embedding_adapter_version=embedding_adapter_version,
                vector_dimension=vector_dimension,
            )
        ).hexdigest()
    )


def _tables(vector_dimension: int) -> tuple[MetaData, Table, Table]:
    metadata = MetaData()
    index_metadata = Table(
        "retrieval_index_metadata",
        metadata,
        Column("singleton_id", SmallInteger, primary_key=True),
        Column("adapter_version", String(160), nullable=False),
        Column("index_version", String(160), nullable=False),
        Column("index_checksum", String(71), nullable=False),
        Column("chunk_count", Integer, nullable=False),
        Column("vector_dimension", Integer, nullable=False),
        Column("embedding_adapter_version", String(160), nullable=False),
        CheckConstraint("singleton_id = 1", name="ck_retrieval_index_singleton"),
        CheckConstraint("chunk_count > 0", name="ck_retrieval_index_chunk_count"),
        CheckConstraint(
            "vector_dimension > 0", name="ck_retrieval_index_vector_dimension"
        ),
    )
    chunks = Table(
        "retrieval_chunks",
        metadata,
        Column("chunk_id", String(160), primary_key=True),
        Column("index_version", String(160), nullable=False),
        Column("content_checksum", String(71), nullable=False),
        Column("content", Text, nullable=False),
        Column("payload", JSONB, nullable=False),
        Column(
            "search_vector",
            TSVECTOR,
            Computed("to_tsvector('simple', content)", persisted=True),
            nullable=False,
        ),
        Column("embedding", Vector(vector_dimension), nullable=False),
        CheckConstraint("char_length(content) > 0", name="ck_retrieval_chunk_content"),
    )
    Index(
        "ix_retrieval_chunks_index_version",
        chunks.c.index_version,
        chunks.c.chunk_id,
    )
    Index(
        "ix_retrieval_chunks_search_vector",
        chunks.c.search_vector,
        postgresql_using="gin",
    )
    return metadata, index_metadata, chunks


class PostgresRetrievalStore:
    """The sole persisted retrieval adapter for Stage C."""

    def __init__(self, database_url: str, *, vector_dimension: int) -> None:
        if vector_dimension < 1 or vector_dimension > 4_096:
            raise ValueError("vector_dimension must be within 1..4096")
        self.vector_dimension = vector_dimension
        self._engine: Engine = create_engine(database_url, pool_pre_ping=True)
        self._metadata, self._index_metadata, self._chunks = _tables(vector_dimension)
        try:
            with self._engine.begin() as connection:
                connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
                self._metadata.create_all(connection)
        except SQLAlchemyError as exc:
            self._engine.dispose()
            raise PostgresStorageError("failed to initialize retrieval tables") from exc

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _validated_build_inputs(
        self,
        *,
        chunks: Sequence[IngestedChunk],
        embeddings: Mapping[str, Sequence[float]],
        index_version: str,
        embedding_adapter_version: str,
    ) -> tuple[tuple[IngestedChunk, ...], dict[str, tuple[float, ...]]]:
        validated_index_version = _identifier_adapter.validate_python(index_version)
        validated_adapter_version = _identifier_adapter.validate_python(
            embedding_adapter_version
        )
        if validated_index_version != index_version:
            raise ValueError("index_version is not canonical")
        if validated_adapter_version != embedding_adapter_version:
            raise ValueError("embedding_adapter_version is not canonical")

        ordered = tuple(sorted(chunks, key=lambda item: item.chunk_id))
        if not ordered:
            raise ValueError("PostgreSQL retrieval index requires chunks")
        chunk_ids = tuple(chunk.chunk_id for chunk in ordered)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("PostgreSQL retrieval chunk IDs must be unique")
        if any(chunk.index_version != index_version for chunk in ordered):
            raise ValueError("chunk index version does not match the build")
        if set(embeddings) != set(chunk_ids):
            raise ValueError("embeddings must match the exact chunk ID set")

        normalized: dict[str, tuple[float, ...]] = {}
        for chunk_id in chunk_ids:
            vector = tuple(
                struct.unpack("!f", struct.pack("!f", float(value)))[0]
                for value in embeddings[chunk_id]
            )
            if len(vector) != self.vector_dimension:
                raise ValueError("embedding dimension does not match the store")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("embeddings must contain only finite values")
            if math.sqrt(sum(value * value for value in vector)) == 0.0:
                raise ValueError("zero vectors are not valid for cosine recall")
            normalized[chunk_id] = vector
        return ordered, normalized

    def _read_rows(
        self, connection: Connection
    ) -> tuple[tuple[IngestedChunk, ...], dict[str, tuple[float, ...]]]:
        rows = connection.execute(
            select(
                self._chunks.c.chunk_id,
                self._chunks.c.index_version,
                self._chunks.c.content_checksum,
                self._chunks.c.content,
                self._chunks.c.payload,
                self._chunks.c.embedding,
            ).order_by(self._chunks.c.chunk_id)
        ).mappings()
        chunks: list[IngestedChunk] = []
        embeddings: dict[str, tuple[float, ...]] = {}
        for row in rows:
            try:
                chunk = IngestedChunk.model_validate_json(
                    json.dumps(
                        row["payload"],
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            except Exception as exc:
                raise IndexConsistencyError("stored chunk payload is invalid") from exc
            if (
                row["chunk_id"] != chunk.chunk_id
                or row["index_version"] != chunk.index_version
                or row["content_checksum"] != chunk.content_checksum
                or row["content"] != chunk.content
            ):
                raise IndexConsistencyError(
                    "stored chunk columns do not match the canonical payload"
                )
            chunks.append(chunk)
            embeddings[chunk.chunk_id] = tuple(
                float(value) for value in row["embedding"]
            )
        return tuple(chunks), embeddings

    def build_index(
        self,
        *,
        chunks: Sequence[IngestedChunk],
        embeddings: Mapping[str, Sequence[float]],
        index_version: str,
        embedding_adapter_version: str,
    ) -> PostgresIndexBuildResult:
        """Create, atomically rebuild, or verify an identical persisted index."""

        ordered, normalized_embeddings = self._validated_build_inputs(
            chunks=chunks,
            embeddings=embeddings,
            index_version=index_version,
            embedding_adapter_version=embedding_adapter_version,
        )
        expected_checksum = postgres_index_checksum(
            chunks=ordered,
            embeddings=normalized_embeddings,
            index_version=index_version,
            embedding_adapter_version=embedding_adapter_version,
            vector_dimension=self.vector_dimension,
        )

        try:
            with self._engine.begin() as connection:
                connection.exec_driver_sql(
                    "LOCK TABLE retrieval_index_metadata IN EXCLUSIVE MODE"
                )
                metadata_row = (
                    connection.execute(
                        select(self._index_metadata).where(
                            self._index_metadata.c.singleton_id == 1
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                stored_count = connection.execute(
                    select(func.count()).select_from(self._chunks)
                ).scalar_one()

                if metadata_row is None and stored_count:
                    raise IndexConsistencyError(
                        "stored retrieval chunks have no index metadata"
                    )

                if metadata_row is not None and metadata_row["index_version"] == (
                    index_version
                ):
                    stored_chunks, stored_embeddings = self._read_rows(connection)
                    actual_checksum = postgres_index_checksum(
                        chunks=stored_chunks,
                        embeddings=stored_embeddings,
                        index_version=index_version,
                        embedding_adapter_version=metadata_row[
                            "embedding_adapter_version"
                        ],
                        vector_dimension=metadata_row["vector_dimension"],
                    )
                    consistent = (
                        metadata_row["adapter_version"] == POSTGRES_ADAPTER_VERSION
                        and metadata_row["index_checksum"] == actual_checksum
                        and metadata_row["index_checksum"] == expected_checksum
                        and metadata_row["chunk_count"] == stored_count == len(ordered)
                        and metadata_row["vector_dimension"] == self.vector_dimension
                        and metadata_row["embedding_adapter_version"]
                        == embedding_adapter_version
                    )
                    if not consistent:
                        raise IndexConsistencyError(
                            "same index version has different payload or checksum"
                        )
                    status = IndexBuildStatus.UNCHANGED
                else:
                    status = (
                        IndexBuildStatus.CREATED
                        if metadata_row is None
                        else IndexBuildStatus.REBUILT
                    )
                    connection.execute(delete(self._chunks))
                    connection.execute(delete(self._index_metadata))
                    connection.execute(
                        insert(self._chunks),
                        [
                            {
                                "chunk_id": chunk.chunk_id,
                                "index_version": chunk.index_version,
                                "content_checksum": chunk.content_checksum,
                                "content": chunk.content,
                                "payload": chunk.model_dump(mode="json"),
                                "embedding": list(
                                    normalized_embeddings[chunk.chunk_id]
                                ),
                            }
                            for chunk in ordered
                        ],
                    )
                    connection.execute(
                        insert(self._index_metadata).values(
                            singleton_id=1,
                            adapter_version=POSTGRES_ADAPTER_VERSION,
                            index_version=index_version,
                            index_checksum=expected_checksum,
                            chunk_count=len(ordered),
                            vector_dimension=self.vector_dimension,
                            embedding_adapter_version=embedding_adapter_version,
                        )
                    )
        except IndexConsistencyError:
            raise
        except SQLAlchemyError as exc:
            raise PostgresStorageError("failed to build retrieval index") from exc

        return PostgresIndexBuildResult(
            status=status,
            index_version=index_version,
            index_checksum=expected_checksum,
            chunk_count=len(ordered),
            vector_dimension=self.vector_dimension,
            embedding_adapter_version=embedding_adapter_version,
        )

    def snapshot(self, *, index_version: str) -> PostgresIndexSnapshot:
        """Load and validate the traceable chunk snapshot for one version."""

        try:
            with self._engine.connect() as connection:
                metadata_row = (
                    connection.execute(
                        select(self._index_metadata).where(
                            self._index_metadata.c.singleton_id == 1
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if (
                    metadata_row is None
                    or metadata_row["index_version"] != index_version
                ):
                    raise IndexVersionMismatchError(
                        "requested PostgreSQL retrieval index is not loaded"
                    )
                chunks, embeddings = self._read_rows(connection)
                actual_checksum = postgres_index_checksum(
                    chunks=chunks,
                    embeddings=embeddings,
                    index_version=index_version,
                    embedding_adapter_version=metadata_row["embedding_adapter_version"],
                    vector_dimension=metadata_row["vector_dimension"],
                )
                if (
                    metadata_row["adapter_version"] != POSTGRES_ADAPTER_VERSION
                    or metadata_row["index_checksum"] != actual_checksum
                    or metadata_row["chunk_count"] != len(chunks)
                    or metadata_row["vector_dimension"] != self.vector_dimension
                ):
                    raise IndexConsistencyError(
                        "PostgreSQL retrieval snapshot checksum is inconsistent"
                    )
                return PostgresIndexSnapshot(
                    index_version=index_version,
                    index_checksum=actual_checksum,
                    embedding_adapter_version=metadata_row["embedding_adapter_version"],
                    vector_dimension=metadata_row["vector_dimension"],
                    chunks=chunks,
                )
        except (IndexVersionMismatchError, IndexConsistencyError):
            raise
        except SQLAlchemyError as exc:
            raise PostgresStorageError("failed to load retrieval index") from exc

    def lexical_recall(
        self, query: str, *, index_version: str, limit: int
    ) -> tuple[RecallScore, ...]:
        if limit < 1:
            return ()
        try:
            with self._engine.connect() as connection:
                self._require_version(connection, index_version)
                parsed = func.plainto_tsquery("simple", query)
                rank = func.ts_rank_cd(self._chunks.c.search_vector, parsed)
                normalized_rank = rank / (1.0 + rank)
                rows = connection.execute(
                    select(self._chunks.c.chunk_id, normalized_rank.label("score"))
                    .where(self._chunks.c.index_version == index_version)
                    .where(self._chunks.c.search_vector.op("@@")(parsed))
                    .order_by(rank.desc(), self._chunks.c.chunk_id)
                    .limit(limit)
                )
                return tuple(
                    RecallScore(chunk_id=row.chunk_id, score=float(row.score))
                    for row in rows
                )
        except IndexVersionMismatchError:
            raise
        except SQLAlchemyError as exc:
            raise PostgresStorageError("PostgreSQL lexical recall failed") from exc

    def vector_recall(
        self,
        query_vector: Sequence[float],
        *,
        index_version: str,
        limit: int,
    ) -> tuple[RecallScore, ...]:
        if limit < 1:
            return ()
        vector = tuple(float(value) for value in query_vector)
        if len(vector) != self.vector_dimension:
            raise ValueError("query vector dimension does not match the store")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("query vector must contain only finite values")
        if math.sqrt(sum(value * value for value in vector)) == 0.0:
            raise ValueError("zero query vector is invalid for cosine recall")
        try:
            with self._engine.connect() as connection:
                self._require_version(connection, index_version)
                distance = self._chunks.c.embedding.cosine_distance(list(vector))
                similarity = func.greatest(
                    0.0,
                    func.least(1.0, (2.0 - distance) / 2.0),
                )
                rows = connection.execute(
                    select(self._chunks.c.chunk_id, similarity.label("score"))
                    .where(self._chunks.c.index_version == index_version)
                    .order_by(distance, self._chunks.c.chunk_id)
                    .limit(limit)
                )
                return tuple(
                    RecallScore(chunk_id=row.chunk_id, score=float(row.score))
                    for row in rows
                )
        except IndexVersionMismatchError:
            raise
        except SQLAlchemyError as exc:
            raise PostgresStorageError("PostgreSQL vector recall failed") from exc

    def _require_version(self, connection: Connection, index_version: str) -> None:
        stored_version = connection.execute(
            select(self._index_metadata.c.index_version).where(
                self._index_metadata.c.singleton_id == 1
            )
        ).scalar_one_or_none()
        if stored_version != index_version:
            raise IndexVersionMismatchError(
                "requested PostgreSQL retrieval index is not loaded"
            )


class PostgresLexicalRecallAdapter:
    """PostgreSQL FTS implementation of the lexical recall port."""

    def __init__(self, store: PostgresRetrievalStore, *, index_version: str) -> None:
        snapshot = store.snapshot(index_version=index_version)
        self._store = store
        self.index_version = snapshot.index_version
        self.index_checksum = snapshot.index_checksum
        self.chunks = snapshot.chunks

    def recall(self, query: str, *, limit: int) -> tuple[RecallScore, ...]:
        return self._store.lexical_recall(
            query,
            index_version=self.index_version,
            limit=limit,
        )


class PostgresDeterministicFakeVectorAdapter:
    """Test-only fake query vectors executed by PostgreSQL pgvector."""

    adapter_version = POSTGRES_FAKE_VECTOR_VERSION
    test_only = True

    def __init__(
        self,
        store: PostgresRetrievalStore,
        *,
        index_version: str,
        query_vectors: Mapping[str, Sequence[float]],
    ) -> None:
        snapshot = store.snapshot(index_version=index_version)
        normalized: dict[str, tuple[float, ...]] = {}
        for query, vector in query_vectors.items():
            canonical = canonical_query_text(query)
            if not canonical:
                raise ValueError("fake pgvector query must contain searchable terms")
            if canonical in normalized:
                raise ValueError("fake pgvector queries must be canonically unique")
            values = tuple(float(value) for value in vector)
            if len(values) != snapshot.vector_dimension:
                raise ValueError("fake query vector dimension is inconsistent")
            if not all(math.isfinite(value) for value in values):
                raise ValueError("fake query vectors must be finite")
            if math.sqrt(sum(value * value for value in values)) == 0.0:
                raise ValueError("fake query vectors cannot be zero")
            normalized[canonical] = values
        self._store = store
        self._vectors = MappingProxyType(normalized)
        self.index_version = snapshot.index_version

    def recall(self, query: str, *, limit: int) -> tuple[RecallScore, ...]:
        vector = self._vectors.get(canonical_query_text(query))
        if vector is None:
            return ()
        return self._store.vector_recall(
            vector,
            index_version=self.index_version,
            limit=limit,
        )
