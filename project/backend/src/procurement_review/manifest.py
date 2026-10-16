"""Strict schema and offline validation for the public-source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
    model_validator,
)

SCHEMA_VERSION = "1.0"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "public_sources"
    / "manifest.v1.json"
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SourceId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=96,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
MediaType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        max_length=127,
        pattern=r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$",
    ),
]


class PublicationDateStatus(StrEnum):
    """Whether the publisher supplied a defensible publication date."""

    KNOWN = "known"
    UNKNOWN = "unknown"


class LicenseStatus(StrEnum):
    """Conservative rights assessment recorded during source registration."""

    PUBLIC_DOMAIN = "public_domain"
    OPEN_LICENSE = "open_license"
    CONDITIONAL_REUSE = "conditional_reuse"
    UNCLEAR = "unclear"
    RESTRICTED = "restricted"


class RedistributionConclusion(StrEnum):
    """Concrete repository action resulting from the rights assessment."""

    SOURCE_ONLY = (
        "Store metadata and reproducible acquisition instructions only; "
        "do not include or redistribute the source document in this repository."
    )
    PERMITTED_NOT_INCLUDED = (
        "Redistribution is permitted under the recorded terms, but this baseline "
        "stores metadata only and does not include the source document."
    )


class AcquisitionMethod(StrEnum):
    """Supported reproducible source acquisition mechanisms."""

    HTTPS_GET = "https_get"


class Acquisition(BaseModel):
    """Network method plus an optional shell-free command representation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    method: Literal[AcquisitionMethod.HTTPS_GET]
    command: tuple[NonEmptyText, ...] | None = None

    @model_validator(mode="after")
    def validate_command(self) -> Acquisition:
        if self.command is None:
            return self
        if not self.command or self.command[0] != "curl":
            raise ValueError("acquisition command must be a curl argv sequence")
        if self.command.count("{output}") != 1:
            raise ValueError(
                "acquisition command must contain one {output} placeholder"
            )
        return self


class PublicSourceRecord(BaseModel):
    """Metadata-only registration for one authoritative public document."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_id: SourceId
    source_url: HttpUrl
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5)]
    publisher: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2)]
    publication_date: date | None
    publication_date_status: PublicationDateStatus
    license_status: LicenseStatus
    license_note: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=24)
    ]
    redistribution_conclusion: RedistributionConclusion
    version: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
    ]
    retrieved_at: datetime
    acquisition: Acquisition
    content_media_type: MediaType
    byte_size: Annotated[int, Field(gt=0)]
    sha256: Sha256
    repository_inclusion: Literal[False]

    @model_validator(mode="after")
    def validate_record_consistency(self) -> PublicSourceRecord:
        if self.source_url.scheme != "https":
            raise ValueError("source_url must use HTTPS")

        if self.publication_date_status is PublicationDateStatus.KNOWN:
            if self.publication_date is None:
                raise ValueError("known publication date status requires a date")
        elif self.publication_date is not None:
            raise ValueError("unknown publication date status requires a null date")

        if self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must include a UTC offset")

        if (
            self.license_status
            in {
                LicenseStatus.UNCLEAR,
                LicenseStatus.RESTRICTED,
            }
            and self.redistribution_conclusion
            is not RedistributionConclusion.SOURCE_ONLY
        ):
            raise ValueError(
                "unclear or restricted rights require source-only handling"
            )

        if self.acquisition.command is not None and self.acquisition.command[-1] != str(
            self.source_url
        ):
            raise ValueError("acquisition command must end with the source URL")
        return self


def canonical_records_bytes(records: Sequence[PublicSourceRecord]) -> bytes:
    """Serialize records canonically, independent of manifest formatting or order."""

    ordered = sorted(records, key=lambda record: record.source_id)
    payload = [record.model_dump(mode="json") for record in ordered]
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def records_checksum(records: Sequence[PublicSourceRecord]) -> str:
    """Return the SHA-256 checksum of canonical manifest records."""

    return hashlib.sha256(canonical_records_bytes(records)).hexdigest()


class PublicSourceManifest(BaseModel):
    """Versioned collection of validated public-source metadata."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[SCHEMA_VERSION]
    manifest_id: Literal["procurement-public-sources-v1"]
    record_count: Annotated[int, Field(ge=20, le=50)]
    records_sha256: Sha256
    records: Annotated[
        tuple[PublicSourceRecord, ...], Field(min_length=20, max_length=50)
    ]

    @model_validator(mode="after")
    def validate_manifest_consistency(self) -> PublicSourceManifest:
        if self.record_count != len(self.records):
            raise ValueError("record_count does not match records")

        source_ids = [record.source_id for record in self.records]
        if source_ids != sorted(source_ids):
            raise ValueError("records must be sorted by source_id")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")

        source_urls = [str(record.source_url) for record in self.records]
        if len(source_urls) != len(set(source_urls)):
            raise ValueError("source_url values must be unique")

        expected_checksum = records_checksum(self.records)
        if self.records_sha256 != expected_checksum:
            raise ValueError(
                "records_sha256 does not match canonical records: "
                f"expected {expected_checksum}"
            )
        return self


def load_manifest(path: str | Path | None = None) -> PublicSourceManifest:
    """Load and validate a manifest without accessing any registered source URL."""

    manifest_path = DEFAULT_MANIFEST_PATH if path is None else Path(path)
    return PublicSourceManifest.model_validate_json(manifest_path.read_bytes())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="manifest JSON to validate",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="print the generated Pydantic JSON Schema instead of validating",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the checked-in manifest and report its canonical checksum."""

    args = _build_parser().parse_args(argv)
    if args.print_schema:
        print(json.dumps(PublicSourceManifest.model_json_schema(), indent=2))
        return 0

    try:
        manifest = load_manifest(args.path)
    except (OSError, ValidationError) as exc:
        print(f"manifest validation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"validated {manifest.record_count} public sources; "
        f"records_sha256={manifest.records_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
