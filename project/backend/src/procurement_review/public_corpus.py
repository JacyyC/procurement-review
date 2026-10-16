"""Acquire and verify manifest-registered public documents in a local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .manifest import PublicSourceManifest, PublicSourceRecord, load_manifest

MEDIA_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "text/markdown": ".md",
    "text/plain": ".txt",
}
DEFAULT_CACHE_DIR = Path(".local/public_sources")


class PublicCorpusError(RuntimeError):
    """Base error for a failed-closed public corpus operation."""


class SourceAcquisitionError(PublicCorpusError):
    """Raised when a registered source cannot be acquired."""


class SourceIntegrityError(PublicCorpusError):
    """Raised after unexpected bytes have been isolated from the verified cache."""


@dataclass(frozen=True, slots=True)
class CachedSource:
    source_id: str
    path: Path
    byte_size: int
    sha256: str
    reused: bool


@dataclass(frozen=True, slots=True)
class AcquisitionReport:
    manifest_id: str
    manifest_records_sha256: str
    verified_count: int
    downloaded_count: int
    reused_count: int
    total_bytes: int
    verified_set_sha256: str
    sources: tuple[CachedSource, ...]

    def summary(self) -> dict[str, int | str]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_records_sha256": self.manifest_records_sha256,
            "verified_count": self.verified_count,
            "downloaded_count": self.downloaded_count,
            "reused_count": self.reused_count,
            "total_bytes": self.total_bytes,
            "verified_set_sha256": self.verified_set_sha256,
        }


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )


def _cache_snapshot_dir(
    cache_dir: Path,
    manifest: PublicSourceManifest,
) -> Path:
    return cache_dir / manifest.manifest_id / manifest.records_sha256


def _verified_path(snapshot_dir: Path, record: PublicSourceRecord) -> Path:
    suffix = MEDIA_TYPE_SUFFIXES.get(record.content_media_type)
    if suffix is None:
        raise SourceAcquisitionError(
            f"{record.source_id}: unsupported media type {record.content_media_type}"
        )
    return snapshot_dir / "verified" / record.source_id / f"{record.sha256}{suffix}"


def _is_expected_file(path: Path, record: PublicSourceRecord) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == record.byte_size
        and _sha256_file(path) == record.sha256
    )


def _quarantine(
    path: Path,
    snapshot_dir: Path,
    record: PublicSourceRecord,
) -> Path | None:
    if not path.exists():
        return None
    actual_size = path.stat().st_size
    actual_sha256 = _sha256_file(path)
    suffix = path.suffix or ".bin"
    quarantine_dir = snapshot_dir / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / (
        f"{record.source_id}-{actual_size}-{actual_sha256}{suffix}"
    )
    if destination.exists():
        path.unlink()
    else:
        path.replace(destination)
    return destination


def acquire_source(
    record: PublicSourceRecord,
    snapshot_dir: Path,
    *,
    runner: CommandRunner = _default_runner,
) -> CachedSource:
    """Acquire one source using its validated shell-free manifest command."""

    destination = _verified_path(snapshot_dir, record)
    if destination.exists():
        if _is_expected_file(destination, record):
            return CachedSource(
                source_id=record.source_id,
                path=destination,
                byte_size=record.byte_size,
                sha256=record.sha256,
                reused=True,
            )
        quarantined = _quarantine(destination, snapshot_dir, record)
        raise SourceIntegrityError(
            f"{record.source_id}: cached content failed byte-size or SHA-256 "
            f"verification; isolated at {quarantined}"
        )

    command = record.acquisition.command
    if command is None:
        raise SourceAcquisitionError(
            f"{record.source_id}: manifest has no reproducible acquisition command"
        )

    download_dir = snapshot_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{record.source_id}-",
        suffix=".part",
        dir=download_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    argv = tuple(
        str(temporary_path) if argument == "{output}" else argument
        for argument in command
    )
    try:
        result = runner(argv)
    except OSError as exc:
        quarantined = _quarantine(temporary_path, snapshot_dir, record)
        raise SourceAcquisitionError(
            f"{record.source_id}: acquisition command could not start; "
            f"partial={quarantined}"
        ) from exc
    if result.returncode != 0:
        quarantined = _quarantine(temporary_path, snapshot_dir, record)
        detail = (result.stderr or result.stdout or "no command output").strip()
        raise SourceAcquisitionError(
            f"{record.source_id}: acquisition command failed with "
            f"exit {result.returncode}: {detail}; partial={quarantined}"
        )

    if not _is_expected_file(temporary_path, record):
        actual_size = temporary_path.stat().st_size
        actual_sha256 = _sha256_file(temporary_path)
        quarantined = _quarantine(temporary_path, snapshot_dir, record)
        raise SourceIntegrityError(
            f"{record.source_id}: acquired content mismatch; "
            f"expected bytes={record.byte_size} sha256={record.sha256}, "
            f"actual bytes={actual_size} sha256={actual_sha256}; "
            f"isolated at {quarantined}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.replace(destination)
    return CachedSource(
        source_id=record.source_id,
        path=destination,
        byte_size=record.byte_size,
        sha256=record.sha256,
        reused=False,
    )


def _verified_set_sha256(sources: Sequence[CachedSource]) -> str:
    payload = [
        {
            "byte_size": source.byte_size,
            "sha256": source.sha256,
            "source_id": source.source_id,
        }
        for source in sorted(sources, key=lambda item: item.source_id)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def acquire_manifest(
    manifest: PublicSourceManifest,
    cache_dir: Path,
    *,
    runner: CommandRunner = _default_runner,
) -> AcquisitionReport:
    snapshot_dir = _cache_snapshot_dir(cache_dir, manifest)
    sources = tuple(
        acquire_source(record, snapshot_dir, runner=runner)
        for record in manifest.records
    )
    return AcquisitionReport(
        manifest_id=manifest.manifest_id,
        manifest_records_sha256=manifest.records_sha256,
        verified_count=len(sources),
        downloaded_count=sum(not source.reused for source in sources),
        reused_count=sum(source.reused for source in sources),
        total_bytes=sum(source.byte_size for source in sources),
        verified_set_sha256=_verified_set_sha256(sources),
        sources=sources,
    )


def verify_manifest_cache(
    manifest: PublicSourceManifest,
    cache_dir: Path,
) -> AcquisitionReport:
    snapshot_dir = _cache_snapshot_dir(cache_dir, manifest)
    sources: list[CachedSource] = []
    for record in manifest.records:
        path = _verified_path(snapshot_dir, record)
        if not _is_expected_file(path, record):
            if path.exists():
                quarantined = _quarantine(path, snapshot_dir, record)
                detail = f"; isolated at {quarantined}"
            else:
                detail = ""
            raise SourceIntegrityError(
                f"{record.source_id}: verified cache entry is missing or invalid{detail}"
            )
        sources.append(
            CachedSource(
                source_id=record.source_id,
                path=path,
                byte_size=record.byte_size,
                sha256=record.sha256,
                reused=True,
            )
        )
    frozen_sources = tuple(sources)
    return AcquisitionReport(
        manifest_id=manifest.manifest_id,
        manifest_records_sha256=manifest.records_sha256,
        verified_count=len(frozen_sources),
        downloaded_count=0,
        reused_count=len(frozen_sources),
        total_bytes=sum(source.byte_size for source in frozen_sources),
        verified_set_sha256=_verified_set_sha256(frozen_sources),
        sources=frozen_sources,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "verify"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = (
            acquire_manifest(manifest, args.cache_dir)
            if args.action == "acquire"
            else verify_manifest_cache(manifest, args.cache_dir)
        )
    except (OSError, PublicCorpusError) as exc:
        print(f"public corpus {args.action} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
