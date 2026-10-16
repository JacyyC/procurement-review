"""Deterministic document ingestion with stable, traceable citations.

The public-source manifest remains authoritative for source identity, version,
media type, byte size, and original SHA-256.  This module consumes already
acquired local files, verifies them again, and produces a framework-neutral
artifact.  It never writes source documents or derived indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pypdf import PdfReader

from .contracts import EvidenceRef, SourceKind
from .manifest import PublicSourceManifest, PublicSourceRecord, load_manifest
from .public_corpus import (
    DEFAULT_CACHE_DIR,
    PublicCorpusError,
    verify_manifest_cache,
)

INGESTION_SCHEMA_VERSION = "1.0"
DEFAULT_INGESTION_VERSION = "ingestion-v1"
DEFAULT_CHUNK_RULES_VERSION = "chunk-rules-v1"
DEFAULT_INDEX_VERSION = "retrieval-index-v1"
DEFAULT_NORMALIZATION_VERSION = "unicode-nfc-lines-v1"
SUPPORTED_MEDIA_TYPES = frozenset({"application/pdf", "text/markdown", "text/plain"})

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    ),
]
Checksum = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )


class LocatorKind(StrEnum):
    PDF_PAGE = "pdf_page"
    MARKDOWN_SECTION = "markdown_section"
    TEXT_LINES = "text_lines"


class IngestionFailureCode(StrEnum):
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_SIZE_MISMATCH = "source_size_mismatch"
    SOURCE_CHECKSUM_MISMATCH = "source_checksum_mismatch"
    UNSUPPORTED_FORMAT = "unsupported_format"
    PARSE_FAILED = "parse_failed"
    NO_EXTRACTABLE_TEXT = "no_extractable_text"
    LOCATOR_MISSING = "locator_missing"
    SOURCE_SET_MISMATCH = "source_set_mismatch"
    NONDETERMINISTIC_REBUILD = "nondeterministic_rebuild"


class IngestionError(RuntimeError):
    """Base error carrying an explicit failed-closed reason code."""

    def __init__(
        self,
        code: IngestionFailureCode,
        source_id: str,
        detail: str,
    ) -> None:
        self.code = code
        self.source_id = source_id
        self.detail = detail
        super().__init__(f"{source_id}: {code.value}: {detail}")


class SourceIntegrityError(IngestionError):
    """The local bytes do not match the authoritative manifest record."""


class UnsupportedDocumentError(IngestionError):
    """The manifest media type is outside the accepted ingestion contract."""


class DocumentParseError(IngestionError):
    """The registered document cannot be decoded or parsed."""


class NoExtractableTextError(IngestionError):
    """The document contains no usable text (including scanned-only PDFs)."""


class SourceSetError(IngestionError):
    """A corpus rebuild did not receive exactly the manifest source set."""


class IngestionConfig(_StrictModel):
    """All inputs that can affect deterministic extraction and chunking."""

    ingestion_version: Identifier = DEFAULT_INGESTION_VERSION
    chunk_rules_version: Identifier = DEFAULT_CHUNK_RULES_VERSION
    index_version: Identifier = DEFAULT_INDEX_VERSION
    normalization_version: Identifier = DEFAULT_NORMALIZATION_VERSION
    max_chunk_chars: Annotated[int, Field(ge=32, le=20_000)] = 1_600

    @property
    def chunk_rules_checksum(self) -> str:
        return _checksum_json(
            {
                "chunk_rules_version": self.chunk_rules_version,
                "max_chunk_chars": self.max_chunk_chars,
                "normalization_version": self.normalization_version,
            }
        )


DEFAULT_CONFIG = IngestionConfig()


class SourceLocator(_StrictModel):
    """Serializable source coordinates; parser objects never cross this port."""

    kind: LocatorKind
    page_number: Annotated[int, Field(ge=1)] | None = None
    heading_path: tuple[Annotated[str, StringConstraints(min_length=1)], ...] = ()
    line_start: Annotated[int, Field(ge=1)] | None = None
    line_end: Annotated[int, Field(ge=1)] | None = None
    part_number: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def require_kind_specific_coordinates(self) -> SourceLocator:
        if self.kind is LocatorKind.PDF_PAGE:
            valid = (
                self.page_number is not None
                and not self.heading_path
                and self.line_start is None
                and self.line_end is None
            )
        elif self.kind is LocatorKind.MARKDOWN_SECTION:
            valid = (
                self.page_number is None
                and bool(self.heading_path)
                and self.line_start is not None
                and self.line_end is not None
                and self.line_end >= self.line_start
            )
        else:
            valid = (
                self.page_number is None
                and not self.heading_path
                and self.line_start is not None
                and self.line_end is not None
                and self.line_end >= self.line_start
            )
        if not valid:
            raise ValueError("locator coordinates do not match locator kind")
        return self

    def evidence_section(self) -> str | None:
        if self.kind is LocatorKind.PDF_PAGE:
            return None
        if self.kind is LocatorKind.TEXT_LINES:
            return f"lines:{self.line_start}-{self.line_end}:part:{self.part_number}"

        heading = " > ".join(self.heading_path)
        suffix = f" [lines:{self.line_start}-{self.line_end}:part:{self.part_number}]"
        if len(heading) + len(suffix) <= 500:
            return heading + suffix
        heading_hash = hashlib.sha256(heading.encode("utf-8")).hexdigest()[:16]
        available = 500 - len(suffix) - len(heading_hash) - 11
        return f"{heading[:available]} [heading:{heading_hash}]{suffix}"


class IngestedChunk(_StrictModel):
    """A stable retrieval unit with enough metadata to reconstruct evidence."""

    schema_version: Literal["1.0"] = INGESTION_SCHEMA_VERSION
    source_id: Identifier
    source_version: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    source_version_id: Identifier
    source_checksum: Checksum
    document_id: Identifier
    chunk_id: Identifier
    ordinal: Annotated[int, Field(ge=1)]
    media_type: Annotated[str, StringConstraints(min_length=3, max_length=127)]
    locator: SourceLocator
    content: NonEmptyText
    content_checksum: Checksum
    ingestion_version: Identifier
    chunk_rules_version: Identifier
    chunk_rules_checksum: Checksum
    index_version: Identifier

    @model_validator(mode="after")
    def validate_derived_identity(self) -> IngestedChunk:
        if self.source_version_id != make_source_version_id(
            self.source_id, self.source_version
        ):
            raise ValueError("source_version_id does not match source version")
        if self.content_checksum != _checksum_bytes(self.content.encode("utf-8")):
            raise ValueError("content_checksum does not match chunk content")
        expected = make_chunk_id(
            document_id=self.document_id,
            ordinal=self.ordinal,
            locator=self.locator,
            content_checksum=self.content_checksum,
            chunk_rules_version=self.chunk_rules_version,
            chunk_rules_checksum=self.chunk_rules_checksum,
        )
        if self.chunk_id != expected:
            raise ValueError("chunk_id does not match chunk identity")
        return self

    def to_evidence_ref(self) -> EvidenceRef:
        """Map this verified chunk onto the existing domain evidence contract."""

        return EvidenceRef(
            source_kind=SourceKind.PUBLIC_DOCUMENT,
            source_id=self.source_id,
            version=self.source_version_id,
            page_number=self.locator.page_number,
            section=self.locator.evidence_section(),
            chunk_id=self.chunk_id,
            content_checksum=self.content_checksum,
        )


class IngestedDocument(_StrictModel):
    """Canonical result for one manifest record and one verified byte stream."""

    schema_version: Literal["1.0"] = INGESTION_SCHEMA_VERSION
    source_id: Identifier
    source_version: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    source_version_id: Identifier
    source_checksum: Checksum
    source_byte_size: Annotated[int, Field(gt=0)]
    media_type: Annotated[str, StringConstraints(min_length=3, max_length=127)]
    document_id: Identifier
    document_checksum: Checksum
    ingestion_version: Identifier
    chunk_rules_version: Identifier
    chunk_rules_checksum: Checksum
    index_version: Identifier
    chunk_count: Annotated[int, Field(gt=0)]
    chunks: Annotated[tuple[IngestedChunk, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_document_consistency(self) -> IngestedDocument:
        expected_document_id = make_document_id(
            source_id=self.source_id,
            source_version=self.source_version,
            source_checksum=self.source_checksum,
            media_type=self.media_type,
            ingestion_version=self.ingestion_version,
        )
        if self.document_id != expected_document_id:
            raise ValueError("document_id does not match source identity")
        if self.source_version_id != make_source_version_id(
            self.source_id, self.source_version
        ):
            raise ValueError("source_version_id does not match source version")
        if self.chunk_count != len(self.chunks):
            raise ValueError("chunk_count does not match chunks")
        if tuple(chunk.ordinal for chunk in self.chunks) != tuple(
            range(1, len(self.chunks) + 1)
        ):
            raise ValueError("chunks must have contiguous one-based ordinals")
        for chunk in self.chunks:
            shared = (
                chunk.source_id == self.source_id
                and chunk.source_version == self.source_version
                and chunk.source_version_id == self.source_version_id
                and chunk.source_checksum == self.source_checksum
                and chunk.document_id == self.document_id
                and chunk.media_type == self.media_type
                and chunk.ingestion_version == self.ingestion_version
                and chunk.chunk_rules_version == self.chunk_rules_version
                and chunk.chunk_rules_checksum == self.chunk_rules_checksum
                and chunk.index_version == self.index_version
            )
            if not shared:
                raise ValueError("chunk metadata does not match its document")
        if self.document_checksum != _document_checksum(self.chunks):
            raise ValueError("document_checksum does not match canonical chunks")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_document_bytes(self)


class CorpusArtifact(_StrictModel):
    """Versioned, deterministic rebuild output for a complete manifest."""

    schema_version: Literal["1.0"] = INGESTION_SCHEMA_VERSION
    manifest_id: Identifier
    manifest_records_checksum: Checksum
    ingestion_version: Identifier
    chunk_rules_version: Identifier
    chunk_rules_checksum: Checksum
    index_version: Identifier
    document_count: Annotated[int, Field(gt=0)]
    chunk_count: Annotated[int, Field(gt=0)]
    corpus_checksum: Checksum
    documents: Annotated[tuple[IngestedDocument, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_corpus_consistency(self) -> CorpusArtifact:
        if self.document_count != len(self.documents):
            raise ValueError("document_count does not match documents")
        if self.chunk_count != sum(doc.chunk_count for doc in self.documents):
            raise ValueError("chunk_count does not match documents")
        source_ids = tuple(document.source_id for document in self.documents)
        if source_ids != tuple(sorted(source_ids)) or len(source_ids) != len(
            set(source_ids)
        ):
            raise ValueError("corpus documents must have unique sorted source IDs")
        for document in self.documents:
            if (
                document.ingestion_version != self.ingestion_version
                or document.chunk_rules_version != self.chunk_rules_version
                or document.chunk_rules_checksum != self.chunk_rules_checksum
                or document.index_version != self.index_version
            ):
                raise ValueError("document versions do not match corpus versions")
        if self.corpus_checksum != _corpus_checksum(
            manifest_id=self.manifest_id,
            manifest_records_checksum=self.manifest_records_checksum,
            documents=self.documents,
            ingestion_version=self.ingestion_version,
            chunk_rules_version=self.chunk_rules_version,
            chunk_rules_checksum=self.chunk_rules_checksum,
            index_version=self.index_version,
        ):
            raise ValueError("corpus_checksum does not match canonical documents")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_corpus_bytes(self)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _checksum_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _checksum_json(value: object) -> str:
    return _checksum_bytes(_canonical_json_bytes(value))


def make_source_version_id(source_id: str, source_version: str) -> str:
    digest = _checksum_json(
        {
            "namespace": "source-version-v1",
            "source_id": source_id,
            "version": source_version,
        }
    ).removeprefix("sha256:")
    return f"source-version-{digest}"


def make_document_id(
    *,
    source_id: str,
    source_version: str,
    source_checksum: str,
    media_type: str,
    ingestion_version: str,
) -> str:
    digest = _checksum_json(
        {
            "ingestion_version": ingestion_version,
            "media_type": media_type,
            "namespace": "document-id-v1",
            "source_checksum": source_checksum,
            "source_id": source_id,
            "source_version": source_version,
        }
    ).removeprefix("sha256:")
    return f"document-{digest}"


def make_chunk_id(
    *,
    document_id: str,
    ordinal: int,
    locator: SourceLocator,
    content_checksum: str,
    chunk_rules_version: str,
    chunk_rules_checksum: str,
) -> str:
    digest = _checksum_json(
        {
            "chunk_rules_checksum": chunk_rules_checksum,
            "chunk_rules_version": chunk_rules_version,
            "content_checksum": content_checksum,
            "document_id": document_id,
            "locator": locator.model_dump(mode="json"),
            "namespace": "chunk-id-v1",
            "ordinal": ordinal,
        }
    ).removeprefix("sha256:")
    return f"chunk-{digest}"


def canonical_document_bytes(document: IngestedDocument) -> bytes:
    return _canonical_json_bytes(document.model_dump(mode="json"))


def canonical_corpus_bytes(corpus: CorpusArtifact) -> bytes:
    return _canonical_json_bytes(corpus.model_dump(mode="json"))


def _document_checksum(chunks: Sequence[IngestedChunk]) -> str:
    return _checksum_json([chunk.model_dump(mode="json") for chunk in chunks])


def _corpus_checksum(
    *,
    manifest_id: str,
    manifest_records_checksum: str,
    documents: Sequence[IngestedDocument],
    ingestion_version: str,
    chunk_rules_version: str,
    chunk_rules_checksum: str,
    index_version: str,
) -> str:
    return _checksum_json(
        {
            "chunk_rules_checksum": chunk_rules_checksum,
            "chunk_rules_version": chunk_rules_version,
            "documents": [doc.model_dump(mode="json") for doc in documents],
            "index_version": index_version,
            "ingestion_version": ingestion_version,
            "manifest_id": manifest_id,
            "manifest_records_checksum": manifest_records_checksum,
        }
    )


def _normalize_lines(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in normalized or "\ufffd" in normalized:
        raise ValueError("text contains an invalid replacement or NUL character")
    return [line.rstrip() for line in normalized.split("\n")]


def _normalized_text(text: str) -> str:
    lines = _normalize_lines(text)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _split_text(text: str, max_chars: int) -> tuple[str, ...]:
    remaining = text.strip()
    parts: list[str] = []
    while remaining:
        if len(remaining) <= max_chars:
            parts.append(remaining)
            break
        boundary = remaining.rfind("\n", max_chars // 2, max_chars + 1)
        if boundary < 0:
            boundary = remaining.rfind(" ", max_chars // 2, max_chars + 1)
        if boundary < 0:
            boundary = max_chars
        part = remaining[:boundary].strip()
        if not part:
            boundary = max_chars
            part = remaining[:boundary]
        parts.append(part)
        remaining = remaining[boundary:].strip()
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class _LocatedPart:
    content: str
    line_start: int
    line_end: int


def _split_located_lines(
    records: Sequence[tuple[int, str]], max_chars: int
) -> tuple[_LocatedPart, ...]:
    fragments: list[tuple[int, str]] = []
    for line_number, line in records:
        if len(line) <= max_chars:
            fragments.append((line_number, line))
            continue
        for offset in range(0, len(line), max_chars):
            fragments.append((line_number, line[offset : offset + max_chars]))

    result: list[_LocatedPart] = []
    current: list[tuple[int, str]] = []

    def flush() -> None:
        while current and not current[0][1].strip():
            current.pop(0)
        while current and not current[-1][1].strip():
            current.pop()
        if not current:
            return
        content = "\n".join(text for _, text in current).strip()
        if content:
            result.append(_LocatedPart(content, current[0][0], current[-1][0]))
        current.clear()

    for fragment in fragments:
        candidate = "\n".join(text for _, text in (*current, fragment)).strip()
        if current and len(candidate) > max_chars:
            flush()
        current.append(fragment)
    flush()
    return tuple(result)


_ATX_HEADING = re.compile(r"^[ ]{0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_SETEXT_HEADING = re.compile(r"^[ \t]*(=+|-+)[ \t]*$")
_FENCE_OPEN = re.compile(r"^[ ]{0,3}(`{3,}|~{3,}).*$")


def _atx_heading(line: str, line_number: int) -> tuple[int, str] | None:
    match = _ATX_HEADING.match(line)
    if match is None:
        return None
    title = (match.group(2) or "").strip()
    title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
    if not title:
        title = f"[untitled heading at line {line_number}]"
    return len(match.group(1)), title


def _markdown_sections(
    lines: Sequence[str],
) -> tuple[tuple[tuple[str, ...], tuple[tuple[int, str], ...]], ...]:
    sections: list[tuple[tuple[str, ...], tuple[tuple[int, str], ...]]] = []
    heading_path: list[str] = []
    current_path: tuple[str, ...] = ("[preamble]",)
    current: list[tuple[int, str]] = []
    fence_character: str | None = None
    fence_length = 0

    def flush() -> None:
        if any(text.strip() for _, text in current):
            sections.append((current_path, tuple(current)))
        current.clear()

    def begin_heading(
        level: int, title: str, records: Sequence[tuple[int, str]]
    ) -> None:
        nonlocal current_path
        flush()
        del heading_path[max(0, level - 1) :]
        heading_path.append(title)
        current_path = tuple(heading_path)
        current.extend(records)

    index = 0
    while index < len(lines):
        line_number = index + 1
        line = lines[index]
        fence = _FENCE_OPEN.match(line)
        if fence_character is not None:
            current.append((line_number, line))
            if re.match(
                rf"^[ ]{{0,3}}{re.escape(fence_character)}{{{fence_length},}}[ \t]*$",
                line,
            ):
                fence_character = None
                fence_length = 0
            index += 1
            continue
        if fence is not None:
            marker = fence.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            current.append((line_number, line))
            index += 1
            continue
        atx = _atx_heading(line, line_number)
        if atx is not None:
            begin_heading(atx[0], atx[1], ((line_number, line),))
            index += 1
            continue
        if index + 1 < len(lines) and line.strip():
            setext = _SETEXT_HEADING.match(lines[index + 1])
            if setext:
                level = 1 if setext.group(1).startswith("=") else 2
                begin_heading(
                    level,
                    line.strip(),
                    ((line_number, line), (line_number + 1, lines[index + 1])),
                )
                index += 2
                continue
        current.append((line_number, line))
        index += 1
    flush()
    return tuple(sections)


def _extract_pdf(
    data: bytes, source_id: str, config: IngestionConfig
) -> tuple[tuple[SourceLocator, str], ...]:
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise ValueError("encrypted PDFs are not supported")
        pages = tuple(reader.pages)
    except Exception as exc:
        raise DocumentParseError(
            IngestionFailureCode.PARSE_FAILED,
            source_id,
            f"PDF structure could not be parsed: {type(exc).__name__}: {exc}",
        ) from exc

    extracted: list[tuple[SourceLocator, str]] = []
    for page_number, page in enumerate(pages, start=1):
        try:
            raw_text = page.extract_text()
            text = _normalized_text(raw_text or "")
        except Exception as exc:
            raise DocumentParseError(
                IngestionFailureCode.PARSE_FAILED,
                source_id,
                f"PDF page {page_number} text extraction failed: {type(exc).__name__}: {exc}",
            ) from exc
        if not text:
            continue
        for part_number, content in enumerate(
            _split_text(text, config.max_chunk_chars), start=1
        ):
            extracted.append(
                (
                    SourceLocator(
                        kind=LocatorKind.PDF_PAGE,
                        page_number=page_number,
                        part_number=part_number,
                    ),
                    content,
                )
            )
    return tuple(extracted)


def _decode_utf8(data: bytes, source_id: str, media_type: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError(
            IngestionFailureCode.PARSE_FAILED,
            source_id,
            f"{media_type} must be valid UTF-8: {exc}",
        ) from exc


def _extract_markdown(
    data: bytes, source_id: str, config: IngestionConfig
) -> tuple[tuple[SourceLocator, str], ...]:
    try:
        lines = _normalize_lines(_decode_utf8(data, source_id, "text/markdown"))
    except ValueError as exc:
        raise DocumentParseError(
            IngestionFailureCode.PARSE_FAILED, source_id, str(exc)
        ) from exc
    extracted: list[tuple[SourceLocator, str]] = []
    for heading_path, records in _markdown_sections(lines):
        for part_number, part in enumerate(
            _split_located_lines(records, config.max_chunk_chars), start=1
        ):
            extracted.append(
                (
                    SourceLocator(
                        kind=LocatorKind.MARKDOWN_SECTION,
                        heading_path=heading_path,
                        line_start=part.line_start,
                        line_end=part.line_end,
                        part_number=part_number,
                    ),
                    part.content,
                )
            )
    return tuple(extracted)


def _extract_text(
    data: bytes, source_id: str, config: IngestionConfig
) -> tuple[tuple[SourceLocator, str], ...]:
    try:
        lines = _normalize_lines(_decode_utf8(data, source_id, "text/plain"))
    except ValueError as exc:
        raise DocumentParseError(
            IngestionFailureCode.PARSE_FAILED, source_id, str(exc)
        ) from exc
    records = tuple((number, line) for number, line in enumerate(lines, start=1))
    return tuple(
        (
            SourceLocator(
                kind=LocatorKind.TEXT_LINES,
                line_start=part.line_start,
                line_end=part.line_end,
                part_number=part_number,
            ),
            part.content,
        )
        for part_number, part in enumerate(
            _split_located_lines(records, config.max_chunk_chars), start=1
        )
    )


def _verified_source_bytes(record: PublicSourceRecord, source_path: Path) -> bytes:
    try:
        data = source_path.read_bytes()
    except FileNotFoundError as exc:
        raise SourceIntegrityError(
            IngestionFailureCode.SOURCE_NOT_FOUND,
            record.source_id,
            f"local source does not exist: {source_path}",
        ) from exc
    if len(data) != record.byte_size:
        raise SourceIntegrityError(
            IngestionFailureCode.SOURCE_SIZE_MISMATCH,
            record.source_id,
            f"expected {record.byte_size} bytes, received {len(data)}",
        )
    actual_checksum = hashlib.sha256(data).hexdigest()
    if actual_checksum != record.sha256:
        raise SourceIntegrityError(
            IngestionFailureCode.SOURCE_CHECKSUM_MISMATCH,
            record.source_id,
            f"expected sha256:{record.sha256}, received sha256:{actual_checksum}",
        )
    return data


def ingest_document(
    record: PublicSourceRecord,
    source_path: str | Path,
    *,
    config: IngestionConfig = DEFAULT_CONFIG,
) -> IngestedDocument:
    """Verify and deterministically ingest one manifest-registered document."""

    path = Path(source_path)
    data = _verified_source_bytes(record, path)
    if record.content_media_type not in SUPPORTED_MEDIA_TYPES:
        raise UnsupportedDocumentError(
            IngestionFailureCode.UNSUPPORTED_FORMAT,
            record.source_id,
            f"unsupported media type: {record.content_media_type}",
        )

    extractors = {
        "application/pdf": _extract_pdf,
        "text/markdown": _extract_markdown,
        "text/plain": _extract_text,
    }
    extracted = extractors[record.content_media_type](data, record.source_id, config)
    if not extracted:
        raise NoExtractableTextError(
            IngestionFailureCode.NO_EXTRACTABLE_TEXT,
            record.source_id,
            "document has no extractable non-whitespace text",
        )

    source_checksum = f"sha256:{record.sha256}"
    source_version_id = make_source_version_id(record.source_id, record.version)
    document_id = make_document_id(
        source_id=record.source_id,
        source_version=record.version,
        source_checksum=source_checksum,
        media_type=record.content_media_type,
        ingestion_version=config.ingestion_version,
    )
    chunks: list[IngestedChunk] = []
    for ordinal, (locator, content) in enumerate(extracted, start=1):
        if not content.strip():
            continue
        content_checksum = _checksum_bytes(content.encode("utf-8"))
        chunk_id = make_chunk_id(
            document_id=document_id,
            ordinal=ordinal,
            locator=locator,
            content_checksum=content_checksum,
            chunk_rules_version=config.chunk_rules_version,
            chunk_rules_checksum=config.chunk_rules_checksum,
        )
        chunks.append(
            IngestedChunk(
                source_id=record.source_id,
                source_version=record.version,
                source_version_id=source_version_id,
                source_checksum=source_checksum,
                document_id=document_id,
                chunk_id=chunk_id,
                ordinal=ordinal,
                media_type=record.content_media_type,
                locator=locator,
                content=content,
                content_checksum=content_checksum,
                ingestion_version=config.ingestion_version,
                chunk_rules_version=config.chunk_rules_version,
                chunk_rules_checksum=config.chunk_rules_checksum,
                index_version=config.index_version,
            )
        )
    frozen_chunks = tuple(chunks)
    if not frozen_chunks:
        raise NoExtractableTextError(
            IngestionFailureCode.NO_EXTRACTABLE_TEXT,
            record.source_id,
            "document did not produce a retrievable chunk",
        )
    return IngestedDocument(
        source_id=record.source_id,
        source_version=record.version,
        source_version_id=source_version_id,
        source_checksum=source_checksum,
        source_byte_size=len(data),
        media_type=record.content_media_type,
        document_id=document_id,
        document_checksum=_document_checksum(frozen_chunks),
        ingestion_version=config.ingestion_version,
        chunk_rules_version=config.chunk_rules_version,
        chunk_rules_checksum=config.chunk_rules_checksum,
        index_version=config.index_version,
        chunk_count=len(frozen_chunks),
        chunks=frozen_chunks,
    )


def ingest_manifest(
    manifest: PublicSourceManifest,
    source_paths: Mapping[str, str | Path],
    *,
    config: IngestionConfig = DEFAULT_CONFIG,
) -> CorpusArtifact:
    """Rebuild a complete corpus, rejecting missing or unregistered inputs."""

    expected = {record.source_id for record in manifest.records}
    actual = set(source_paths)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SourceSetError(
            IngestionFailureCode.SOURCE_SET_MISMATCH,
            manifest.manifest_id,
            f"missing={missing}; unexpected={unexpected}",
        )
    documents = tuple(
        ingest_document(record, source_paths[record.source_id], config=config)
        for record in manifest.records
    )
    manifest_checksum = f"sha256:{manifest.records_sha256}"
    corpus_checksum = _corpus_checksum(
        manifest_id=manifest.manifest_id,
        manifest_records_checksum=manifest_checksum,
        documents=documents,
        ingestion_version=config.ingestion_version,
        chunk_rules_version=config.chunk_rules_version,
        chunk_rules_checksum=config.chunk_rules_checksum,
        index_version=config.index_version,
    )
    return CorpusArtifact(
        manifest_id=manifest.manifest_id,
        manifest_records_checksum=manifest_checksum,
        ingestion_version=config.ingestion_version,
        chunk_rules_version=config.chunk_rules_version,
        chunk_rules_checksum=config.chunk_rules_checksum,
        index_version=config.index_version,
        document_count=len(documents),
        chunk_count=sum(document.chunk_count for document in documents),
        corpus_checksum=corpus_checksum,
        documents=documents,
    )


def corpus_summary(
    corpus: CorpusArtifact, *, rebuild_verified: bool
) -> dict[str, bool | int | str]:
    canonical_bytes = corpus.canonical_bytes()
    return {
        "canonical_artifact_bytes": len(canonical_bytes),
        "canonical_artifact_checksum": _checksum_bytes(canonical_bytes),
        "chunk_count": corpus.chunk_count,
        "chunk_rules_checksum": corpus.chunk_rules_checksum,
        "chunk_rules_version": corpus.chunk_rules_version,
        "corpus_checksum": corpus.corpus_checksum,
        "document_count": corpus.document_count,
        "index_version": corpus.index_version,
        "ingestion_version": corpus.ingestion_version,
        "manifest_id": corpus.manifest_id,
        "manifest_records_checksum": corpus.manifest_records_checksum,
        "rebuild_verified": rebuild_verified,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--verify-rebuild",
        action="store_true",
        help="ingest twice and require byte-identical canonical artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify cached sources and print a deterministic in-memory build summary."""

    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = verify_manifest_cache(manifest, args.cache_dir)
        source_paths = {source.source_id: source.path for source in report.sources}
        corpus = ingest_manifest(manifest, source_paths)
        if args.verify_rebuild:
            rebuilt = ingest_manifest(manifest, source_paths)
            if corpus.canonical_bytes() != rebuilt.canonical_bytes():
                raise IngestionError(
                    IngestionFailureCode.NONDETERMINISTIC_REBUILD,
                    manifest.manifest_id,
                    "two canonical corpus rebuilds differ",
                )
    except (OSError, ValidationError, PublicCorpusError, IngestionError) as exc:
        print(f"document ingestion failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            corpus_summary(corpus, rebuild_verified=args.verify_rebuild),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
