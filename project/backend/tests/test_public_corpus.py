from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from procurement_review.manifest import PublicSourceRecord, load_manifest
from procurement_review.public_corpus import (
    SourceAcquisitionError,
    SourceIntegrityError,
    _cache_snapshot_dir,
    acquire_source,
)


def _record_for(content: bytes) -> PublicSourceRecord:
    original = load_manifest().records[0]
    return original.model_copy(
        update={
            "byte_size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )


def _runner_for(content: bytes, returncode: int = 0):
    def run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        output = Path(argv[argv.index("--output") + 1])
        output.write_bytes(content)
        return subprocess.CompletedProcess(
            argv,
            returncode,
            stdout="",
            stderr="download failed" if returncode else "",
        )

    return run


def test_acquisition_verifies_and_reuses_exact_bytes(tmp_path: Path) -> None:
    content = b"%PDF-1.4\nverified fixture\n%%EOF\n"
    record = _record_for(content)
    snapshot = _cache_snapshot_dir(tmp_path, load_manifest())

    first = acquire_source(record, snapshot, runner=_runner_for(content))
    second = acquire_source(record, snapshot, runner=_runner_for(b"wrong"))

    assert first.path.read_bytes() == content
    assert first.reused is False
    assert second.reused is True
    assert second.sha256 == record.sha256


def test_download_checksum_mismatch_is_quarantined(tmp_path: Path) -> None:
    record = _record_for(b"expected")
    snapshot = _cache_snapshot_dir(tmp_path, load_manifest())

    with pytest.raises(SourceIntegrityError, match="acquired content mismatch"):
        acquire_source(record, snapshot, runner=_runner_for(b"tampered"))

    assert list((snapshot / "quarantine").iterdir())
    assert not list((snapshot / "verified").rglob("*"))


def test_tampered_verified_file_is_quarantined_and_fails_visible(
    tmp_path: Path,
) -> None:
    content = b"expected"
    record = _record_for(content)
    snapshot = _cache_snapshot_dir(tmp_path, load_manifest())
    acquired = acquire_source(record, snapshot, runner=_runner_for(content))
    acquired.path.write_bytes(b"tampered")

    with pytest.raises(SourceIntegrityError, match="cached content failed"):
        acquire_source(record, snapshot, runner=_runner_for(content))

    assert not acquired.path.exists()
    assert list((snapshot / "quarantine").iterdir())


def test_acquisition_command_failure_is_explicit(tmp_path: Path) -> None:
    record = _record_for(b"expected")
    snapshot = _cache_snapshot_dir(tmp_path, load_manifest())

    with pytest.raises(SourceAcquisitionError, match="exit 22: download failed"):
        acquire_source(
            record,
            snapshot,
            runner=_runner_for(b"partial", returncode=22),
        )

    assert list((snapshot / "quarantine").iterdir())


def test_acquisition_process_start_failure_is_explicit_and_quarantined(
    tmp_path: Path,
) -> None:
    record = _record_for(b"expected")
    snapshot = _cache_snapshot_dir(tmp_path, load_manifest())

    def fail_to_start(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        raise OSError("test-only missing executable")

    with pytest.raises(SourceAcquisitionError, match="could not start"):
        acquire_source(record, snapshot, runner=fail_to_start)

    assert list((snapshot / "quarantine").iterdir())
