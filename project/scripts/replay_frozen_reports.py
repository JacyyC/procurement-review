#!/usr/bin/env python3
"""Read-only replay of the twelve frozen evaluation reports (FIX-05B AC-07).

FIX-01 A-15 and D-AUD-07 both turned on whether the ADR's numbers still match the frozen
artifacts under ``project/backend/.local/runs/``.  Those artifacts are Git-ignored, so the
repository cannot re-derive them from a clean checkout; what it *can* do is prove, on the
machine that holds them, that every report still rebuilds byte-for-byte from the frozen
segments and that nothing was rewritten in the process.

This tool therefore:

* rebuilds each of the twelve reports in memory through the read-only ``build_report``
  entry points and compares it byte-for-byte with the file on disk;
* hashes every original before and after, so "read-only" is measured rather than promised;
* matches each file against the hash and size recorded when the batch was frozen
  (``FROZEN_REPORTS`` below);
* recomputes the ``eval_audit cross-check`` claim set into the caller's exclusive output
  directory, and requires it to stay clean at ``318/318``.

Nothing under ``project/backend/.local/runs/`` is written.  The reporting CLIs that write
their output back into a runs directory are deliberately not used here.

Usage (the tool imports the backend package, so run it through the backend environment)::

    cd project/backend
    uv run --offline --frozen --python 3.12 \\
        python ../scripts/replay_frozen_reports.py

Exit codes: 0 all twelve byte-identical and claims clean; 1 a report drifted, an original
changed, or a claim mismatched; 2 the frozen artifacts are not present or unreadable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "project" / "backend"
DEFAULT_RUNS_DIR = BACKEND / ".local" / "runs"

#: sha256 and byte size recorded when each batch was frozen.  A report that no longer
#: matches its recorded digest is a broken source, not a refreshed one.
FROZEN_REPORTS: dict[str, tuple[str, int]] = {
    "v1-02c-calibration-48/eval-report.v3.json": (
        "83223382b337718d6e4d6338dcc26d248d23ea01c7f721535f3fc81ad9785ab6", 221867),
    "v1-02c-calibration-48/eval-rescore.rules-v1.json": (
        "f26a60b64857d19b7eaf1e4348275c325f5673afb39f69d8c21668e8e0687865", 223421),
    "v1-02c-calibration-48/eval-rescore.rules-v2.json": (
        "c0fa1b30a9278f464eff6a19cd4305762c1bc382481c06a355f172067a448c36", 229944),
    "v1-02c-paired-360x3/eval-report.v3.json": (
        "5d67be8dda37a3a4117cec256c4a2c6f57abc2821080561ad62194d6458b6c2f", 264690),
    "v1-02c-paired-360x3/eval-rescore.rules-v1.json": (
        "fa6c2887dcfb28df25ea2167324a3032d87b21d6dbc9f57351f5dc5b7f0f297a", 268060),
    "v1-02c-paired-360x3/eval-rescore.rules-v2.json": (
        "e5a4b57bbea973aae5e3be73948d2b2c991267214445c0cac1e4b609e9037e8d", 275720),
    "v1-02c2-resolution-120x2/eval-report.v3.json": (
        "d3983b566a4c176f809f03facb5e1fe02bb4e6adb3b30167023429a40ac50243", 267659),
    "v1-02c2-resolution-120x2/eval-rescore.rules-v1.json": (
        "e00cf7f22d219f5ff663e3873827d31fb4109f35b890f9104f974be0902df284", 270139),
    "v1-02c2-resolution-120x2/eval-rescore.rules-v2.json": (
        "c485dc16013251aa24afaa1d25cf3866b9e623fc55322563128b6e083883908a", 272774),
    "v1-05b-holdout-120x2/eval-report.v3.json": (
        "4b411e9b9bf584778247a12698eef126a60b7ab6a78e2ad341e98df37924e9dc", 266479),
    "v1-05b-holdout-120x2/eval-rescore.rules-v1.json": (
        "85bfb1c0936ad4a8f893af8b64793b5a77e9730779f96ea1a6a32535c800a8ec", 269021),
    "v1-05b-holdout-120x2/eval-rescore.rules-v2.json": (
        "09623953e2c489607a69148456fe894d3bd2932d210f0bd05f30956c9943c4fc", 272327),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rebuild(relative: str, runs_dir: Path) -> bytes:
    """Rebuild one report in memory with the read-only entry points."""

    from procurement_review import eval_rescore, eval_score

    path = Path(relative)
    batch = path.parts[0]
    if path.name == "eval-report.v3.json":
        module: Any = eval_score
        report = module.build_report(batch, runs_dir=runs_dir)
    else:
        module = eval_rescore
        contract = "v1" if path.name.endswith("rules-v1.json") else "v2"
        report = module.build_report(batch, contract, runs_dir=runs_dir)
    return json.dumps(
        report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True,
        default=module._json_default,
    ).encode()


def cross_check(runs_dir: Path, output: Path) -> dict[str, Any]:
    """Recompute the claim set into the caller's exclusive directory."""

    claims_path = output / "claims.json"
    completed = subprocess.run(
        [sys.executable, "-m", "procurement_review.eval_audit", "--runs-dir",
         str(runs_dir), "cross-check", "--output", str(claims_path)],
        cwd=BACKEND, capture_output=True, text=True, check=False,
    )
    if not claims_path.is_file():
        raise SystemExit(
            f"eval_audit cross-check wrote no record (exit {completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claims["_exit"] = completed.returncode
    return claims


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, required=True,
                        help="exclusive directory for the replay record and claims file")
    arguments = parser.parse_args(argv)

    missing = [relative for relative in FROZEN_REPORTS
               if not (arguments.runs_dir / relative).is_file()]
    if missing:
        print(
            "read-only replay could not run: the frozen artifacts are not present at "
            f"{arguments.runs_dir} (missing {len(missing)} of {len(FROZEN_REPORTS)}, "
            f"e.g. {missing[0]}). These files are Git-ignored and are not distributed "
            "with the repository, so this check cannot run from a clean checkout.",
            file=sys.stderr,
        )
        return 2
    arguments.output.mkdir(parents=True, exist_ok=True)

    before = {relative: digest((arguments.runs_dir / relative).read_bytes())
              for relative in FROZEN_REPORTS}
    rows = []
    for relative in sorted(FROZEN_REPORTS):
        recorded_hash, recorded_size = FROZEN_REPORTS[relative]
        original = (arguments.runs_dir / relative).read_bytes()
        rows.append({
            "path": relative,
            "bytes": len(original),
            "sha256": digest(original),
            "recorded_hash_matches": digest(original) == recorded_hash,
            "recorded_size_matches": len(original) == recorded_size,
            "replay_identical": rebuild(relative, arguments.runs_dir) == original,
        })
    after = {relative: digest((arguments.runs_dir / relative).read_bytes())
             for relative in FROZEN_REPORTS}
    claims = cross_check(arguments.runs_dir, arguments.output)

    result = {
        "runs_dir": str(arguments.runs_dir),
        "reports": rows,
        "replayed": len(rows),
        "identical": sum(1 for row in rows if row["replay_identical"]),
        "originals_unchanged": before == after,
        "claims_command_exit": claims.get("_exit"),
        "claims": claims.get("claims"),
        "claims_matched": claims.get("matched"),
        "claims_mismatched": claims.get("mismatched"),
    }
    (arguments.output / "replay-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)

    problems = []
    if not result["originals_unchanged"]:
        problems.append("the frozen reports changed during the replay")
    if result["identical"] != len(rows):
        problems.append("a report did not replay byte-identically")
    if not all(row["recorded_hash_matches"] and row["recorded_size_matches"] for row in rows):
        problems.append("a report does not match its recorded hash/size")
    if result["claims_command_exit"] != 0 or claims.get("mismatched"):
        problems.append("the claim cross-check is not clean")

    print(
        f"replayed {result['replayed']} reports; byte-identical {result['identical']}; "
        f"originals unchanged {result['originals_unchanged']}; claims "
        f"{result['claims_matched']}/{result['claims']} "
        f"mismatched={result['claims_mismatched']}",
        flush=True,
    )
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
