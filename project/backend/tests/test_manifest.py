from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from procurement_review.manifest import (
    DEFAULT_MANIFEST_PATH,
    LicenseStatus,
    PublicationDateStatus,
    PublicSourceManifest,
    PublicSourceRecord,
    RedistributionConclusion,
    load_manifest,
    main,
    records_checksum,
)

EXPECTED_RECORD_COUNT = 24
EXPECTED_RECORDS_SHA256 = (
    "e1f778899e167b8fbb6a964bab88cca49bbe8466ab444c7566aff0b2d074bcc2"
)


@pytest.fixture(scope="module")
def manifest() -> PublicSourceManifest:
    return load_manifest()


def test_manifest_schema_is_strict_and_versioned(
    manifest: PublicSourceManifest,
) -> None:
    schema = PublicSourceManifest.model_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert manifest.schema_version == "1.0"
    assert manifest.manifest_id == "procurement-public-sources-v1"

    raw = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PublicSourceManifest.model_validate_json(json.dumps(raw))


def test_manifest_has_expected_count_and_unique_sorted_keys(
    manifest: PublicSourceManifest,
) -> None:
    source_ids = [record.source_id for record in manifest.records]
    source_urls = [str(record.source_url) for record in manifest.records]

    assert manifest.record_count == EXPECTED_RECORD_COUNT
    assert len(manifest.records) == EXPECTED_RECORD_COUNT
    assert source_ids == sorted(source_ids)
    assert len(source_ids) == len(set(source_ids))
    assert len(source_urls) == len(set(source_urls))


def test_every_record_has_complete_metadata_and_date_state(
    manifest: PublicSourceManifest,
) -> None:
    for record in manifest.records:
        assert str(record.source_url).startswith("https://")
        assert record.title
        assert record.publisher
        assert record.version
        assert record.license_note
        assert record.content_media_type == "application/pdf"
        assert record.byte_size > 0
        assert len(record.sha256) == 64
        assert record.retrieved_at.utcoffset() is not None

        if record.publication_date_status is PublicationDateStatus.KNOWN:
            assert record.publication_date is not None
        else:
            assert record.publication_date_status is PublicationDateStatus.UNKNOWN
            assert record.publication_date is None


def test_manifest_contains_no_source_documents(
    manifest: PublicSourceManifest,
) -> None:
    public_sources_dir = DEFAULT_MANIFEST_PATH.parent
    embedded_documents = [
        path
        for path in public_sources_dir.rglob("*")
        if path.is_file() and path != DEFAULT_MANIFEST_PATH
    ]

    assert embedded_documents == []
    assert all(record.repository_inclusion is False for record in manifest.records)


def test_canonical_manifest_checksum_is_exact_and_order_independent(
    manifest: PublicSourceManifest,
) -> None:
    assert manifest.records_sha256 == EXPECTED_RECORDS_SHA256
    assert records_checksum(manifest.records) == EXPECTED_RECORDS_SHA256
    assert (
        records_checksum(tuple(reversed(manifest.records))) == EXPECTED_RECORDS_SHA256
    )

    raw = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["records"][0]["byte_size"] += 1
    with pytest.raises(ValidationError, match="records_sha256 does not match"):
        PublicSourceManifest.model_validate_json(json.dumps(raw))


def test_licensing_has_a_conservative_concrete_conclusion(
    manifest: PublicSourceManifest,
) -> None:
    for record in manifest.records:
        assert record.redistribution_conclusion is RedistributionConclusion.SOURCE_ONLY
        if record.license_status in {LicenseStatus.UNCLEAR, LicenseStatus.RESTRICTED}:
            assert (
                record.redistribution_conclusion is RedistributionConclusion.SOURCE_ONLY
            )


def test_acquisition_command_is_optional_but_all_records_supply_one(
    manifest: PublicSourceManifest,
) -> None:
    for record in manifest.records:
        command = record.acquisition.command
        assert command is not None
        assert command[0] == "curl"
        assert command.count("{output}") == 1
        assert command[-1] == str(record.source_url)

    raw_record = manifest.records[0].model_dump(mode="json")
    raw_record["acquisition"].pop("command")
    parsed = PublicSourceRecord.model_validate_json(json.dumps(raw_record))
    assert parsed.acquisition.command is None


def test_cli_validates_without_network(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert f"validated {EXPECTED_RECORD_COUNT} public sources" in output
    assert EXPECTED_RECORDS_SHA256 in output
