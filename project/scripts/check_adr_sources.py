#!/usr/bin/env python3
"""Mechanically check the frozen evaluation sources an ADR's correction depends on (FIX-05B).

FIX-01 A-15 corrected [ADR-0002] with a dated note, but the note first bound the C2
report's sha256 to the C-2 report's path (FIX-01D audit finding D-AUD-07).  The existing
``eval_audit cross-check`` claim set recomputes 318 numbers from the frozen reports and
never looks at ADR prose, so it cannot catch that class of error.  This tool closes the
gap for the two bindings the correction depends on:

* **path ↔ sha256** — the document must bind the *recomputed* sha256 of the file it
  names, and no other 64-hex digest may sit next to that path;
* **batch** — the batch id inside the report must match the batch the path names;
* **metric, delta, CI, direction** — the paired delta for ``recommendation_exact`` is
  recomputed from the report, converted to the direction the document states, and every
  stated number must appear in the document with the right sign;
* **denominators** — the observation counts, model-call counts, token totals, p95
  latencies, safety numerators/denominators and the cross-context isolation counts are
  recomputed from the reports and the read-only segment manifests, and each number the
  document states must match.

The tool is read-only: it never writes into ``.local/runs/``, and its only output is
stdout plus an optional JSON record in a caller-chosen directory.

Usage::

    python3 project/scripts/check_adr_sources.py
    python3 project/scripts/check_adr_sources.py --document <copy.md> --runs-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT = (ROOT / "workspace" / "docs" / "decisions"
                    / "ADR-0002-rejudge-topology-under-rules-v2.md")
DEFAULT_RUNS_DIR = ROOT / "project" / "backend" / ".local" / "runs"

C2_BATCH = "v1-02c-paired-360x3"
C2X_BATCH = "v1-02c2-resolution-120x2"
HOLDOUT_BATCH = "v1-05b-holdout-120x2"
RESCORE = "eval-rescore.rules-v2.json"

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
# A binding is the digest written next to the path it belongs to; the window is tight so
# that a digest belonging to a *different* report mentioned in the same paragraph cannot
# be mistaken for this one's.
BINDING_WINDOW = 40


class SourceRefused(RuntimeError):
    """One mechanical check failed; the caller reports every problem it collected."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SourceRefused(f"cannot read {path}: {exc}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def number_tokens(value: float, digits: int = 4) -> tuple[str, ...]:
    """The renderings a document may legitimately use for one measured number."""

    tokens = {f"{value:.{digits}f}", f"{value:+.{digits}f}", f"{value:g}", f"{value:+g}"}
    for smaller in (2, 3):
        tokens.add(f"{value:.{smaller}f}")
        tokens.add(f"{value:+.{smaller}f}")
    return tuple(sorted(tokens))


def require_number(document: str, value: float, what: str, *,
                   signed: str | None = None) -> None:
    tokens = number_tokens(value)
    if not any(token in document for token in tokens):
        raise SourceRefused(
            f"the document does not state {what} = {value} (looked for {tokens}); the "
            "document and the frozen artifacts disagree"
        )
    if signed is not None and signed not in document:
        raise SourceRefused(
            f"the document does not state {what} with the sign this direction requires "
            f"({signed!r}); a direction or binding error is exactly what this check exists "
            "to catch"
        )


def paired_entry(report: dict, metric: str) -> dict:
    for entry in report.get("paired_deltas") or []:
        if entry.get("metric") == metric:
            return entry
    raise SourceRefused(f"the report carries no paired delta for metric {metric!r}")


def segment_totals(runs_dir: Path, batch: str, topology: str) -> dict[str, int]:
    """Aggregate the branch counters ``eval_audit`` resolves under ``segments:``."""

    report = load_json(runs_dir / batch / RESCORE)
    totals: dict[str, int] = {"calls": 0, "usage_bearing_calls": 0, "not_requested": 0}
    for segment in report.get("segment_run_ids", {}).get(topology, []):
        manifest = load_json(runs_dir / segment / "eval-run-manifest.v1.json")
        for call in manifest.get("calls") or []:
            totals["calls"] += 1
            if call.get("disposition") == "not_requested":
                totals["not_requested"] += 1
            if call.get("total_tokens") is not None:
                totals["usage_bearing_calls"] += 1
            code = call.get("error_code")
            if isinstance(code, str) and code:
                totals[f"error.{code}"] = totals.get(f"error.{code}", 0) + 1
            detail = call.get("error_detail")
            if isinstance(detail, str) and detail:
                totals[f"detail.{detail}"] = totals.get(f"detail.{detail}", 0) + 1
    return totals


def check_binding(runs_dir: Path, document: str, *, key: str, relative: str,
                  sha256: str, metric: str, direction: str, observations: int,
                  delta: float, ci_low: float, ci_high: float,
                  problems: list[str], notes: list[str]) -> None:
    """One path ↔ hash ↔ batch ↔ metric ↔ delta/CI/direction ↔ denominator binding."""

    path = runs_dir / relative
    try:
        actual_hash = file_sha256(path)
    except OSError as exc:
        problems.append(f"{key}: cannot read the frozen report {path}: {exc}")
        return
    if actual_hash != sha256:
        problems.append(
            f"{key}: {relative} hashes to {actual_hash}, not the recorded {sha256}"
        )
        return
    report = load_json(path)
    batch_id = (report.get("batch") or {}).get("batch_id")
    if batch_id != relative.split("/")[0]:
        problems.append(
            f"{key}: the report at {relative} carries batch_id {batch_id!r}, which is not "
            "the batch named by its own path"
        )
    entry = paired_entry(report, metric)
    as_run_delta = float(entry["delta"])
    as_run_low = float(entry["ci_low"])
    as_run_high = float(entry["ci_high"])
    if direction == "multi_minus_single":
        expected_delta = -as_run_delta
        expected_low = -as_run_high
        expected_high = -as_run_low
        if entry.get("favors") != "multi_agent" or expected_delta <= 0:
            problems.append(
                f"{key}: the report's pairing does not support a multi-minus-single gain "
                f"(delta {as_run_delta}, favors {entry.get('favors')!r})"
            )
    else:
        expected_delta, expected_low, expected_high = as_run_delta, as_run_low, as_run_high
    for label, computed, recorded in (
        (f"{key} delta", expected_delta, delta),
        (f"{key} CI low", expected_low, ci_low),
        (f"{key} CI high", expected_high, ci_high),
    ):
        if abs(computed - recorded) > 5e-5:
            problems.append(
                f"{label}: recomputed {computed!r} but the binding records {recorded!r}"
            )
    observations_seen = int(entry.get("case_count") or 0)
    if observations_seen != observations:
        problems.append(
            f"{key} denominators: the report's paired sample is {observations_seen} cases, "
            f"not {observations}"
        )
    # document side: the path must carry exactly this hash, and the numbers must appear
    bound: list[str] = []
    for match in re.finditer(re.escape(relative), document):
        window = document[match.end():match.end() + BINDING_WINDOW + 64]
        digest = HEX64.search(window)
        if digest is not None and digest.start() <= BINDING_WINDOW:
            bound.append(digest.group(0))
    if not bound:
        problems.append(
            f"{key}: the document never binds a sha256 to {relative}; write the digest "
            f"within {BINDING_WINDOW} characters of the path it belongs to"
        )
    for digest in sorted(set(bound)):
        if digest != sha256:
            problems.append(
                f"{key}: the document binds {digest[:12]}… to {relative}, but the file "
                f"hashes to {sha256[:12]}… — the wrong report is bound"
            )
    signed = f"{delta:+.4f}" if delta else None
    require_number(document, delta, f"{key} delta", signed=signed)
    require_number(document, ci_low, f"{key} CI low")
    require_number(document, ci_high, f"{key} CI high")
    notes.append(
        f"{key}: {relative} sha256 {sha256[:12]}…, batch {batch_id}, {metric} "
        f"multi-minus-single {expected_delta:+.4f} "
        f"[{expected_low:+.4f}, {expected_high:+.4f}], sample {observations} cases"
    )


def check_derived(runs_dir: Path, document: str, reports: dict[str, dict],
                  problems: list[str], notes: list[str]) -> None:
    """The ADR's ratio, safety and isolation numbers, recomputed from the artifacts."""

    c2 = reports[C2_BATCH]
    c2x = reports[C2X_BATCH]
    holdout = reports[HOLDOUT_BATCH]

    def cost(report: dict, topology: str, field: str) -> float:
        return float(report["cost_and_reliability"][topology][field])

    def require_tokens(label: str, computed: float, tokens: tuple[str, ...]) -> None:
        for token in tokens:
            if token not in document:
                problems.append(
                    f"{label}: the document does not state {token!r} (recomputed "
                    f"{computed!r}); the document and the frozen artifacts disagree"
                )
        notes.append(f"{label}: {computed!r} (document tokens {list(tokens) or 'shared'})")

    requests = {topology: cost(holdout, topology, "model_call_count")
                for topology in ("multi_agent", "single_agent")}
    if (requests["multi_agent"], requests["single_agent"]) != (660, 240):
        problems.append(
            "holdout request counts changed: "
            f"{requests['multi_agent']}/{requests['single_agent']} (document states 660/240)"
        )
    require_tokens("holdout request ratio",
                   requests["multi_agent"] / requests["single_agent"],
                   ("660/240", "2.750"))
    tokens = {topology: cost(holdout, topology, "input_tokens")
              + cost(holdout, topology, "output_tokens")
              for topology in ("multi_agent", "single_agent")}
    if (tokens["multi_agent"], tokens["single_agent"]) != (2507592, 840235):
        problems.append(
            "holdout token totals changed: "
            f"{tokens['multi_agent']}/{tokens['single_agent']} (document states "
            "2,507,592/840,235)"
        )
    require_tokens("holdout token ratio",
                   tokens["multi_agent"] / tokens["single_agent"],
                   ("2,507,592", "840,235", "2.984"))
    p95 = {topology: float(holdout["cost_and_reliability"][topology]["latency_ms"]["p95"])
           for topology in ("multi_agent", "single_agent")}
    if (p95["multi_agent"], p95["single_agent"]) != (90004, 109047):
        problems.append(
            "holdout p95 latencies changed: "
            f"{p95['multi_agent']}/{p95['single_agent']} (document states 90,004/109,047)"
        )
    require_tokens("holdout p95 ratio", p95["multi_agent"] / p95["single_agent"],
                   ("90,004", "109,047", "0.825"))
    for key, report in (("C-2", c2), ("C2", c2x), ("holdout", holdout)):
        entry = paired_entry(report, "required_finding_coverage")
        gain = -float(entry["delta"])
        lower = -float(entry["ci_high"])
        require_tokens(f"{key} required-finding coverage gain", gain,
                       (f"{gain:+.4f}",))
        require_tokens(f"{key} required-finding coverage lower bound", lower,
                       (f"{lower:+.4f}",))
    for key, report, doc_tokens in (("C-2", c2, ("0/540",)),
                                    ("C2", c2x, ()),
                                    ("holdout", holdout, ("0/120",))):
        quality = report["quality"]
        misses = quality["multi_agent"]["high_risk_miss_rate"]
        if (int(misses["numerator"]), int(misses["denominator"])) != (0, 540 if key == "C-2" else 120):
            problems.append(
                f"{key}: the high-risk miss counter moved to "
                f"{misses['numerator']}/{misses['denominator']}"
            )
        require_tokens(f"{key} high-risk misses", float(misses["numerator"]), doc_tokens)
        recall = float(quality["multi_agent"]["unknown_recall"]["value"])
        require_tokens(f"{key} UNKNOWN recall", recall, ())
        if not any(token in document for token in number_tokens(recall)):
            problems.append(
                f"{key} UNKNOWN recall: the document does not state {recall}"
            )
        if quality["single_agent"]["unknown_recall"]["value"] != 1.0:
            problems.append(
                f"{key}: the single-agent UNKNOWN recall moved off 1.0"
            )
        support = float(quality["multi_agent"]["citation_support_precision"]["value"])
        require_tokens(f"{key} citation support precision", support, ("1.0",))
    holdout_conflict = holdout["quality"]["multi_agent"]["conflict_recall"]
    if (int(holdout_conflict["numerator"]), int(holdout_conflict["denominator"])) != (0, 60):
        problems.append(
            "the out-of-sample conflict recall moved: "
            f"{holdout_conflict['numerator']}/{holdout_conflict['denominator']}, the "
            "document states 0/60"
        )
    require_tokens("out-of-sample conflict recall (holdout)", 0.0, ("0/60",))
    isolation = (("C-2", C2_BATCH, 518, 1080),
                 ("C2", C2X_BATCH, 82, 240),
                 ("holdout", HOLDOUT_BATCH, 91, 240))
    for key, batch, expected, denominator in isolation:
        totals = segment_totals(runs_dir, batch, "single_agent")
        measured = (totals.get("detail.document_worker_cited_business_fact", 0)
                    + totals.get("detail.procurement_worker_cited_document", 0))
        if measured != expected:
            problems.append(
                f"{key} cross-context isolation: the read-only segment manifests count "
                f"{measured} single-agent out-of-context citations, the document states "
                f"{expected}"
            )
        multi = segment_totals(runs_dir, batch, "multi_agent")
        multi_count = (multi.get("detail.document_worker_cited_business_fact", 0)
                       + multi.get("detail.procurement_worker_cited_document", 0))
        if multi_count != 0:
            problems.append(
                f"{key}: the multi-agent topology now records {multi_count} out-of-context "
                "citations; the document states 0 by construction"
            )
        observations = int(report_observations(batch, reports))
        if observations != denominator:
            problems.append(
                f"{key} isolation denominator: the report's observation count is "
                f"{observations}, the document uses {denominator}"
            )
        require_tokens(f"{key} cross-context isolation", float(measured),
                       (f"{expected}/{denominator}",))
        notes.append(
            f"{key}: multi-agent out-of-context citations {multi_count} (by construction)"
        )


def report_observations(batch: str, reports: dict[str, dict]) -> float:
    return float(reports[batch]["quality"]["multi_agent"]["observation_count"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full result record to this path")
    arguments = parser.parse_args(argv)
    try:
        document = arguments.document.read_text(encoding="utf-8")
        reports = {batch: load_json(arguments.runs_dir / batch / RESCORE)
                   for batch in (C2_BATCH, C2X_BATCH, HOLDOUT_BATCH)}
    except (OSError, SourceRefused) as exc:
        print(f"source check could not run: {exc}", file=sys.stderr)
        return 2
    problems: list[str] = []
    notes: list[str] = []
    try:
        check_binding(
            arguments.runs_dir, document, key="C-2",
            relative=f"{C2_BATCH}/{RESCORE}",
            sha256=("e5a4b57bbea973aae5e3be73948d2b2c991267214445c0cac1e4b609e9037e8d"),
            metric="recommendation_exact", direction="multi_minus_single",
            observations=360, delta=0.0148, ci_low=0.0046, ci_high=0.0250,
            problems=problems, notes=notes,
        )
        check_binding(
            arguments.runs_dir, document, key="C2",
            relative=f"{C2X_BATCH}/{RESCORE}",
            sha256=("c485dc16013251aa24afaa1d25cf3866b9e623fc55322563128b6e083883908a"),
            metric="recommendation_exact", direction="as_run",
            observations=120, delta=0.0, ci_low=-0.0208, ci_high=0.0208,
            problems=problems, notes=notes,
        )
        check_derived(arguments.runs_dir, document, reports, problems, notes)
    except SourceRefused as refusal:
        problems.append(str(refusal))
    for note in notes:
        print(f"checked: {note}")
    result = {"document": str(arguments.document), "runs_dir": str(arguments.runs_dir),
              "checked": notes, "problems": problems}
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        print(
            f"source check FAILED: {len(problems)} problem(s); the document does not match "
            "the frozen artifacts",
            file=sys.stderr,
        )
        return 1
    print(
        f"source check passed: {len(notes)} mechanically recomputed claims agree with "
        f"{arguments.document.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
