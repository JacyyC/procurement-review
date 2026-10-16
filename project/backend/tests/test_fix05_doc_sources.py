"""FIX-05B: the ADR's frozen-source bindings are checked mechanically, not by claim counts.

FIX-01D audit finding D-AUD-07 was a sha256 bound to the wrong batch inside ADR-0002's
dated correction.  The ``eval_audit cross-check`` claim set recomputes 318 numbers from
the frozen reports and never reads ADR prose, so it could not catch the drift.  These
cases drive the real ``project/scripts/check_adr_sources.py``:

* the shipped ADR must pass against the frozen reports (path ↔ sha256 ↔ batch ↔ metric ↔
  delta ↔ CI ↔ direction ↔ denominator);
* the *original* wrong binding must fail — the check is only worth something if it
  rejects the counterexample that motivated it;
* a drifted delta, a drifted denominator and a removed binding must each fail.

The check needs the frozen historical artifacts under ``project/backend/.local/runs/``,
which are Git-ignored and not distributed with the repository.  When they are absent the
tool must refuse to run (exit 2) instead of silently passing; the first case pins that
behaviour down, so this module never reports a green result it did not verify.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER = REPO_ROOT / "project" / "scripts" / "check_adr_sources.py"
DOCUMENT = (
    REPO_ROOT
    / "workspace"
    / "docs"
    / "decisions"
    / "ADR-0002-rejudge-topology-under-rules-v2.md"
)
RUNS_DIR = REPO_ROOT / "project" / "backend" / ".local" / "runs"
C2_REPORT = RUNS_DIR / "v1-02c-paired-360x3" / "eval-rescore.rules-v2.json"

C2_HASH = "e5a4b57bbea973aae5e3be73948d2b2c991267214445c0cac1e4b609e9037e8d"
C2X_HASH = "c485dc16013251aa24afaa1d25cf3866b9e623fc55322563128b6e083883908a"
FROZEN_AVAILABLE = C2_REPORT.is_file()


def run_checker(
    document: Path | None = None, runs_dir: Path | None = None
) -> subprocess.CompletedProcess:
    argv = [sys.executable, str(CHECKER), "--runs-dir", str(runs_dir or RUNS_DIR)]
    if document is not None:
        argv += ["--document", str(document)]
    return subprocess.run(
        argv, capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )


def mutated(tmp_path: Path, *, old: str, new: str) -> Path:
    assert old in DOCUMENT.read_text(encoding="utf-8"), old
    copy = tmp_path / "adr-copy.md"
    copy.write_text(
        DOCUMENT.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    return copy


def test_the_checker_refuses_without_the_frozen_artifacts(tmp_path: Path) -> None:
    """A missing artifact is a refusal, never a silent pass."""

    completed = run_checker(runs_dir=tmp_path / "absent")
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 2, combined
    assert "could not run" in completed.stderr, combined
    assert "Traceback" not in combined, combined


@pytest.mark.skipif(
    not FROZEN_AVAILABLE, reason="frozen historical runs are not present"
)
def test_the_shipped_adr_matches_the_frozen_artifacts() -> None:
    completed = run_checker()
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "source check passed" in completed.stdout
    for expected in (
        "sha256 e5a4b57bbea9",
        "sha256 c485dc160132",
        "cross-context isolation",
        "required-finding coverage",
    ):
        assert expected in completed.stdout, completed.stdout


@pytest.mark.skipif(
    not FROZEN_AVAILABLE, reason="frozen historical runs are not present"
)
def test_the_original_wrong_binding_is_rejected(tmp_path: Path) -> None:
    """D-AUD-07 itself: the C2 digest bound to the C-2 path must fail the check."""

    copy = mutated(tmp_path, old=C2_HASH, new=C2X_HASH)
    completed = run_checker(document=copy)
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 1, combined
    assert "wrong report is bound" in completed.stderr, combined


@pytest.mark.skipif(
    not FROZEN_AVAILABLE, reason="frozen historical runs are not present"
)
@pytest.mark.parametrize(
    ("old", "new", "expect"),
    [
        ("+0.0148", "+0.0149", "does not state"),
        ("518/1080", "518/1081", "does not state"),
        ("0/60", "0/61", "does not state"),
    ],
)
def test_a_drifted_number_is_rejected(
    tmp_path: Path, old: str, new: str, expect: str
) -> None:
    copy = mutated(tmp_path, old=old, new=new)
    completed = run_checker(document=copy)
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 1, combined
    assert expect in completed.stderr, combined


@pytest.mark.skipif(
    not FROZEN_AVAILABLE, reason="frozen historical runs are not present"
)
def test_a_removed_binding_is_rejected(tmp_path: Path) -> None:
    copy = mutated(
        tmp_path,
        old="`v1-02c2-resolution-120x2/eval-rescore.rules-v2.json`（sha256 `",
        new="`v1-02c2-resolution-120x2/eval-rescore.v3.json`（sha256 `",
    )
    completed = run_checker(document=copy)
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 1, combined
    assert "never binds a sha256" in completed.stderr, combined
