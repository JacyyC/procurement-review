from __future__ import annotations

import stat
from pathlib import Path

import pytest

from procurement_review.blind_seal import (
    BlindSealError,
    BlindState,
    DisposableBlindSeal,
)


def _seal(tmp_path: Path, name: str = "disposable-unit") -> DisposableBlindSeal:
    return DisposableBlindSeal.create(
        tmp_path / ".local" / "blind-tests",
        batch_id=name,
        candidate_fingerprint="a" * 64,
        seed_material=b"stage-b-disposable-seed",
        expected_arms=("multi_actual_1", "multi_oracle_1", "single_actual_1"),
    )


def _generate_and_seal(value: DisposableBlindSeal) -> None:
    generated = value.generate(
        cases={"disposable": True, "split": "test", "cases": [{"id": "T-1"}]},
        labels={"disposable": True, "labels": [{"id": "T-1", "answer": "ok"}]},
    )
    assert generated.state is BlindState.GENERATED
    assert value.seal().state is BlindState.SEALED


def test_disposable_batch_runs_the_full_sealed_state_machine_and_is_destroyed(
    tmp_path: Path,
) -> None:
    value = _seal(tmp_path)
    try:
        assert stat.S_IMODE(value.directory.stat().st_mode) == 0o700
        assert value._record().state is BlindState.CANDIDATE_FROZEN
        _generate_and_seal(value)
        assert not (value.directory / "runner" / "labels.json").exists()
        assert stat.S_IMODE(value.runner_cases_path.stat().st_mode) == 0o600
        for arm in value._record().expected_arms:
            value.complete_arm(arm)
        assert value.mark_run_complete().state is BlindState.RUN_COMPLETE
        labels = value.release_labels()
        assert b'"answer": "ok"' in labels
        record = value.mark_scored()
        assert record.state is BlindState.SCORED
        assert record.transitions == tuple(BlindState)
    finally:
        directory = value.directory
        if directory.exists():
            value.destroy()
    assert not directory.exists()


def test_labels_cannot_be_released_before_every_registered_arm_finishes(
    tmp_path: Path,
) -> None:
    value = _seal(tmp_path, "disposable-early-release")
    try:
        _generate_and_seal(value)
        with pytest.raises(BlindSealError, match="expected run_complete"):
            value.release_labels()
        value.complete_arm(value._record().expected_arms[0])
        with pytest.raises(BlindSealError, match="before every arm"):
            value.mark_run_complete()
    finally:
        value.destroy()


@pytest.mark.parametrize("target", ["cases", "labels", "key"])
def test_sealed_payload_or_commitment_tampering_is_rejected(
    tmp_path: Path, target: str
) -> None:
    value = _seal(tmp_path, f"disposable-tamper-{target}")
    try:
        _generate_and_seal(value)
        for arm in value._record().expected_arms:
            value.complete_arm(arm)
        value.mark_run_complete()
        paths = {
            "cases": value.runner_cases_path,
            "labels": value.directory / "private" / "labels.json",
            "key": value.directory / "private" / "commitment.key",
        }
        paths[target].write_bytes(paths[target].read_bytes() + b"tampered")
        with pytest.raises(BlindSealError, match="tampered"):
            value.release_labels()
    finally:
        value.destroy()


def test_stage_b_rejects_formal_names_content_and_non_disposable_batches(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".local" / "blind-tests"
    with pytest.raises(BlindSealError, match="invalid disposable"):
        DisposableBlindSeal.create(
            root,
            batch_id="blind-formal-200",
            candidate_fingerprint="b" * 64,
            seed_material=b"formal-seed",
            expected_arms=("arm",),
        )
    assert not (root / "blind-formal-200").exists()

    with pytest.raises(BlindSealError, match="reserved for Stage H"):
        DisposableBlindSeal.create(
            root,
            batch_id="disposable-not-really",
            candidate_fingerprint="b" * 64,
            seed_material=b"formal-seed",
            expected_arms=("arm",),
            disposable=False,
        )

    value = _seal(tmp_path, "disposable-formal-content")
    try:
        with pytest.raises(BlindSealError, match="formal blind content"):
            value.generate(
                cases={"disposable": True, "split": "blind"},
                labels={"disposable": True},
            )
        assert value._record().state is BlindState.CANDIDATE_FROZEN
        assert not value.runner_cases_path.exists()
    finally:
        value.destroy()


def test_invalid_batch_identifier_cannot_escape_the_disposable_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".local" / "blind-tests"
    with pytest.raises(BlindSealError, match="invalid disposable"):
        DisposableBlindSeal.create(
            root,
            batch_id="../../escape",
            candidate_fingerprint="c" * 64,
            seed_material=b"seed",
            expected_arms=("arm",),
        )
    assert not (tmp_path / "escape").exists()
