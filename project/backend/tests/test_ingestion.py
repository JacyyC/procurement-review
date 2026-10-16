from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from procurement_review import ingestion
from procurement_review.contracts import SourceKind
from procurement_review.ingestion import (
    CorpusArtifact,
    DocumentParseError,
    IngestedDocument,
    IngestionConfig,
    IngestionFailureCode,
    LocatorKind,
    NoExtractableTextError,
    SourceIntegrityError,
    SourceLocator,
    SourceSetError,
    UnsupportedDocumentError,
    canonical_corpus_bytes,
    canonical_document_bytes,
    ingest_document,
    ingest_manifest,
    main,
)
from procurement_review.manifest import (
    Acquisition,
    AcquisitionMethod,
    LicenseStatus,
    PublicationDateStatus,
    PublicSourceManifest,
    PublicSourceRecord,
    RedistributionConclusion,
    records_checksum,
)


def _record(
    data: bytes,
    media_type: str,
    *,
    source_id: str = "fixture-source",
    version: str = "Edition 1, English PDF",
) -> PublicSourceRecord:
    url = f"https://example.invalid/{source_id}"
    return PublicSourceRecord(
        source_id=source_id,
        source_url=url,
        title=f"Generated document for {source_id}",
        publisher="Test Fixture Publisher",
        publication_date=date(2026, 8, 30),
        publication_date_status=PublicationDateStatus.KNOWN,
        license_status=LicenseStatus.OPEN_LICENSE,
        license_note="Generated test-only bytes with no external or private source data.",
        redistribution_conclusion=RedistributionConclusion.PERMITTED_NOT_INCLUDED,
        version=version,
        retrieved_at=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        acquisition=Acquisition(
            method=AcquisitionMethod.HTTPS_GET,
            command=("curl", "--output", "{output}", url),
        ),
        content_media_type=media_type,
        byte_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        repository_inclusion=False,
    )


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _pdf_bytes(pages: tuple[str | None, ...]) -> bytes:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_ingestion_preserves_page_and_evidence_location(tmp_path: Path) -> None:
    data = _pdf_bytes((None, "Competition is required above 5000."))
    record = _record(data, "application/pdf")
    document = ingest_document(record, _write(tmp_path, "policy.pdf", data))

    assert document.source_id == record.source_id
    assert document.source_version == "Edition 1, English PDF"
    assert document.source_version_id.startswith("source-version-")
    assert document.chunk_count == 1
    chunk = document.chunks[0]
    assert chunk.locator.kind is LocatorKind.PDF_PAGE
    assert chunk.locator.page_number == 2
    assert chunk.content == "Competition is required above 5000."

    evidence = chunk.to_evidence_ref()
    assert evidence.source_kind is SourceKind.PUBLIC_DOCUMENT
    assert evidence.source_id == record.source_id
    assert evidence.version == chunk.source_version_id
    assert evidence.page_number == 2
    assert evidence.chunk_id == chunk.chunk_id
    assert evidence.content_checksum == chunk.content_checksum


def test_markdown_ingestion_preserves_heading_hierarchy_and_lines(
    tmp_path: Path,
) -> None:
    text = (
        "Preface text.\n\n"
        "# Procurement Policy\n"
        "General rule.\n\n"
        "## Competition\n"
        "Three quotes are required.\n"
    )
    data = text.encode()
    document = ingest_document(
        _record(data, "text/markdown"), _write(tmp_path, "policy.md", data)
    )

    assert [chunk.locator.heading_path for chunk in document.chunks] == [
        ("[preamble]",),
        ("Procurement Policy",),
        ("Procurement Policy", "Competition"),
    ]
    source_lines = text.splitlines()
    for chunk in document.chunks:
        locator = chunk.locator
        assert locator.kind is LocatorKind.MARKDOWN_SECTION
        assert locator.line_start is not None
        assert locator.line_end is not None
        original_region = "\n".join(
            source_lines[locator.line_start - 1 : locator.line_end]
        ).strip()
        assert chunk.content == original_region
        evidence = chunk.to_evidence_ref()
        assert evidence.section is not None
        assert locator.heading_path[-1] in evidence.section


def test_markdown_setext_heading_is_stable(tmp_path: Path) -> None:
    data = b"Policy Title\n============\nRule text.\n"
    document = ingest_document(
        _record(data, "text/markdown"), _write(tmp_path, "setext.md", data)
    )
    assert document.chunks[0].locator.heading_path == ("Policy Title",)
    assert document.chunks[0].locator.line_start == 1
    assert document.chunks[0].locator.line_end == 3


def test_markdown_code_fence_hashes_do_not_create_false_sections(
    tmp_path: Path,
) -> None:
    data = (
        "# Policy\n"
        "```markdown\n"
        "# This is code, not a source heading\n"
        "```\n"
        "## Actual Section\n"
        "Actual rule.\n"
    ).encode()
    document = ingest_document(
        _record(data, "text/markdown"), _write(tmp_path, "fenced.md", data)
    )

    assert [chunk.locator.heading_path for chunk in document.chunks] == [
        ("Policy",),
        ("Policy", "Actual Section"),
    ]
    assert "This is code" in document.chunks[0].content


def test_txt_ingestion_preserves_exact_line_ranges(tmp_path: Path) -> None:
    data = "First rule.\n\nSecond rule.\nThird rule.\n".encode()
    config = IngestionConfig(max_chunk_chars=32)
    document = ingest_document(
        _record(data, "text/plain"),
        _write(tmp_path, "rules.txt", data),
        config=config,
    )

    assert document.chunk_count == 2
    source_lines = data.decode().splitlines()
    for chunk in document.chunks:
        locator = chunk.locator
        assert locator.kind is LocatorKind.TEXT_LINES
        assert locator.line_start is not None
        assert locator.line_end is not None
        region = "\n".join(
            source_lines[locator.line_start - 1 : locator.line_end]
        ).strip()
        assert chunk.content == region
        assert chunk.to_evidence_ref().section == (
            f"lines:{locator.line_start}-{locator.line_end}:part:{locator.part_number}"
        )


def test_same_input_rebuilds_byte_identical_artifact_and_stable_ids(
    tmp_path: Path,
) -> None:
    data = "alpha\nbeta\ngamma\n".encode()
    record = _record(data, "text/plain")
    path = _write(tmp_path, "stable.txt", data)

    first = ingest_document(record, path)
    second = ingest_document(record, path)

    assert canonical_document_bytes(first) == canonical_document_bytes(second)
    assert first.document_id == second.document_id
    assert first.document_checksum == second.document_checksum
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    assert IngestedDocument.model_validate_json(first.canonical_bytes()) == first


def test_content_ingestion_and_rule_versions_change_the_right_identities(
    tmp_path: Path,
) -> None:
    original = b"a stable source body\n"
    changed = b"a changed source body\n"
    original_doc = ingest_document(
        _record(original, "text/plain"), _write(tmp_path, "original.txt", original)
    )
    changed_doc = ingest_document(
        _record(changed, "text/plain"), _write(tmp_path, "changed.txt", changed)
    )
    rule_doc = ingest_document(
        _record(original, "text/plain"),
        tmp_path / "original.txt",
        config=IngestionConfig(chunk_rules_version="chunk-rules-v2"),
    )
    ingestion_doc = ingest_document(
        _record(original, "text/plain"),
        tmp_path / "original.txt",
        config=IngestionConfig(ingestion_version="ingestion-v2"),
    )

    assert changed_doc.source_id == original_doc.source_id
    assert changed_doc.document_id != original_doc.document_id
    assert changed_doc.chunks[0].chunk_id != original_doc.chunks[0].chunk_id
    assert rule_doc.document_id == original_doc.document_id
    assert rule_doc.chunks[0].chunk_id != original_doc.chunks[0].chunk_id
    assert ingestion_doc.document_id != original_doc.document_id


def test_rule_parameters_are_checksummed_even_when_version_label_is_reused(
    tmp_path: Path,
) -> None:
    data = ("word " * 40).encode()
    record = _record(data, "text/plain")
    path = _write(tmp_path, "rules.txt", data)
    short = ingest_document(record, path, config=IngestionConfig(max_chunk_chars=32))
    long = ingest_document(record, path, config=IngestionConfig(max_chunk_chars=64))

    assert short.document_id == long.document_id
    assert short.chunk_rules_version == long.chunk_rules_version
    assert short.chunk_rules_checksum != long.chunk_rules_checksum
    assert [chunk.chunk_id for chunk in short.chunks] != [
        chunk.chunk_id for chunk in long.chunks
    ]


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (lambda data: data + b"!", IngestionFailureCode.SOURCE_SIZE_MISMATCH),
        (
            lambda data: bytes([data[0] ^ 1]) + data[1:],
            IngestionFailureCode.SOURCE_CHECKSUM_MISMATCH,
        ),
    ],
)
def test_manifest_size_and_checksum_tampering_fail_closed(
    tmp_path: Path,
    mutator: object,
    expected_code: IngestionFailureCode,
) -> None:
    data = b"verified source bytes"
    record = _record(data, "text/plain")
    tampered = mutator(data)  # type: ignore[operator]
    path = _write(tmp_path, "tampered.txt", tampered)

    with pytest.raises(SourceIntegrityError) as raised:
        ingest_document(record, path)
    assert raised.value.code is expected_code


def test_missing_source_fails_closed(tmp_path: Path) -> None:
    data = b"registered source"
    with pytest.raises(SourceIntegrityError) as raised:
        ingest_document(_record(data, "text/plain"), tmp_path / "missing.txt")
    assert raised.value.code is IngestionFailureCode.SOURCE_NOT_FOUND


@pytest.mark.parametrize(
    ("name", "data", "media_type", "error_type", "expected_code"),
    [
        (
            "empty.txt",
            b" \n\t\n",
            "text/plain",
            NoExtractableTextError,
            IngestionFailureCode.NO_EXTRACTABLE_TEXT,
        ),
        (
            "broken.pdf",
            b"%PDF-1.7\nbroken",
            "application/pdf",
            DocumentParseError,
            IngestionFailureCode.PARSE_FAILED,
        ),
        (
            "unsupported.bin",
            b"opaque bytes",
            "application/octet-stream",
            UnsupportedDocumentError,
            IngestionFailureCode.UNSUPPORTED_FORMAT,
        ),
        (
            "invalid.txt",
            b"\xff\xfe",
            "text/plain",
            DocumentParseError,
            IngestionFailureCode.PARSE_FAILED,
        ),
    ],
)
def test_invalid_documents_fail_without_chunks(
    tmp_path: Path,
    name: str,
    data: bytes,
    media_type: str,
    error_type: type[Exception],
    expected_code: IngestionFailureCode,
) -> None:
    with pytest.raises(error_type) as raised:
        ingest_document(_record(data, media_type), _write(tmp_path, name, data))
    assert raised.value.code is expected_code  # type: ignore[attr-defined]


def test_textless_pdf_is_rejected_as_scanned_or_blank(tmp_path: Path) -> None:
    data = _pdf_bytes((None, None))
    with pytest.raises(NoExtractableTextError) as raised:
        ingest_document(
            _record(data, "application/pdf"), _write(tmp_path, "scan.pdf", data)
        )
    assert raised.value.code is IngestionFailureCode.NO_EXTRACTABLE_TEXT


def test_locator_models_reject_missing_or_mixed_coordinates() -> None:
    with pytest.raises(ValidationError, match="locator coordinates"):
        SourceLocator(kind=LocatorKind.PDF_PAGE)
    with pytest.raises(ValidationError, match="locator coordinates"):
        SourceLocator(
            kind=LocatorKind.TEXT_LINES,
            page_number=1,
            line_start=1,
            line_end=1,
        )
    with pytest.raises(ValidationError, match="locator coordinates"):
        SourceLocator(
            kind=LocatorKind.MARKDOWN_SECTION,
            heading_path=("Policy",),
            line_start=3,
            line_end=2,
        )


def _manifest_and_paths(tmp_path: Path) -> tuple[PublicSourceManifest, dict[str, Path]]:
    records: list[PublicSourceRecord] = []
    paths: dict[str, Path] = {}
    for number in range(1, 21):
        source_id = f"corpus-source-{number:02d}"
        data = f"Rule {number}: deterministic corpus content.\n".encode()
        record = _record(data, "text/plain", source_id=source_id, version="fixture-v1")
        records.append(record)
        paths[source_id] = _write(tmp_path, f"{source_id}.txt", data)
    checksum = records_checksum(records)
    manifest = PublicSourceManifest(
        schema_version="1.0",
        manifest_id="procurement-public-sources-v1",
        record_count=len(records),
        records_sha256=checksum,
        records=tuple(records),
    )
    return manifest, paths


def test_complete_manifest_rebuild_is_canonical_and_rejects_source_drift(
    tmp_path: Path,
) -> None:
    manifest, paths = _manifest_and_paths(tmp_path)

    first = ingest_manifest(manifest, paths)
    second = ingest_manifest(manifest, dict(reversed(tuple(paths.items()))))

    assert first.document_count == 20
    assert first.chunk_count == 20
    assert first.corpus_checksum == second.corpus_checksum
    assert canonical_corpus_bytes(first) == canonical_corpus_bytes(second)
    assert CorpusArtifact.model_validate_json(first.canonical_bytes()) == first

    incomplete = dict(paths)
    incomplete.pop("corpus-source-01")
    with pytest.raises(SourceSetError) as raised:
        ingest_manifest(manifest, incomplete)
    assert raised.value.code is IngestionFailureCode.SOURCE_SET_MISMATCH

    unexpected = {**paths, "not-in-manifest": tmp_path / "extra.txt"}
    with pytest.raises(SourceSetError):
        ingest_manifest(manifest, unexpected)


def test_cli_verifies_cache_and_byte_identical_rebuild_without_writing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, paths = _manifest_and_paths(tmp_path)
    report = SimpleNamespace(
        sources=tuple(
            SimpleNamespace(source_id=source_id, path=path)
            for source_id, path in paths.items()
        )
    )
    monkeypatch.setattr(ingestion, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(
        ingestion,
        "verify_manifest_cache",
        lambda manifest_value, cache_dir: report,
    )

    assert main(["--cache-dir", str(tmp_path), "--verify-rebuild"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["document_count"] == 20
    assert summary["chunk_count"] == 20
    assert summary["index_version"] == "retrieval-index-v1"
    assert summary["rebuild_verified"] is True
    assert summary["corpus_checksum"].startswith("sha256:")
    assert summary["canonical_artifact_checksum"].startswith("sha256:")
    assert list(tmp_path.rglob("*.json")) == []
