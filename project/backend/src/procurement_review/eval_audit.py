"""Audit the persisted real-run artifacts for AC-09 and V1-05 AC-06.

V1-02D adds this tool. It is strictly read-only over the ignored run directory and the
corpus already committed in the repository: it never contacts the network and never sends
a model request. Nothing it prints or writes carries raw artifact text. Free-form string
leaves are reported as counts, length bounds and hashes, and an optional local credential
is only ever used as a match target and reported as a hash identifier.

The sub-commands answer the AC-09 questions separately:

``ledger``      every artifact with its checksum, permissions, load status and coverage,
                plus the cumulative local-experiment brake recomputed from the manifests.
``trace``       the end-to-end chain for one ``(segment, case)``: branch calls, projected
                findings, the deterministic aggregation and the escalation reason.
``scan``        field classification and the redaction battery, including the retention
                declaration consistency check that earlier stages asserted but never
                measured.
``cross-check`` every number the reports quote, recomputed from the manifests or a
                regenerated report and compared against the recorded expectation; any
                miss makes the exit status non-zero.
``audit``       all of the above as one ``run-artifact-audit.v1`` document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from . import smoke_run
from .hashing import sha256_file
from .eval_contract import (
    DEFAULT_RUNS_DIR,
    EVAL_BATCHES,
    EVAL_MANIFEST_SCHEMA_VERSION,
    MANIFEST_FILE_NAME,
    PROJECTION_FILE_NAME,
    EvalProjection,
    EvalRunManifest,
    gold_corpus_contract,
    guard_eval_manifest,
    load_eval_manifest,
    load_projection,
)
from .eval_metrics import observations_from_projection, quality_metrics
from .eval_rescore import CONTRACT_RULE_SETS
from .eval_rescore import build_report as build_rescore_report
from .eval_score import build_report
from .gold_set import (
    GoldCase,
    GoldCorpusVersion,
    GoldLabel,
    GoldSplit,
    load_gold_cases,
    load_gold_labels_for_split,
)
from .rules import is_generated_reason_code
from .run_guard import secret_value_paths

AUDIT_SCHEMA_VERSION = "run-artifact-audit.v1"
REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_MANIFEST_FILE_NAME = "run-manifest.v1.json"
REPORT_GLOB = "eval-report.v*.json"

# The V1-02C2 run was paused while one segment was in flight. Segment manifests are
# written as a whole, so that attempt spent budget without appearing in any ledger. These
# are the windows recorded in Evidence §44, named here so the arithmetic stays inspectable.
PAUSE_BATCH_ID = "v1-02c2-resolution-120x2"
PAUSE_SEGMENT_1_FINISHED_AT = datetime(2026, 9, 10, 9, 8, 11, tzinfo=timezone.utc)
PAUSE_REQUESTED_AT = datetime(2026, 9, 10, 9, 34, 0, tzinfo=timezone.utc)
PAUSE_IN_FLIGHT_MINUTES_RECORDED = 27.9

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "chain_of_thought",
    "cookie",
    "credential",
    "password",
    "prompt",
    "reasoning",
    "response_body",
    "secret",
    "set_cookie",
    "thought",
)
GOV_REFERENCE = re.compile(r"(?i)\.gov(\.cn)?\b|ggzy|ccgp|uncitral|eur-lex|ted\.europa")
# Applied to decoded string leaves only: over raw JSON text these would match the
# fractional digits of every confidence interval in a report.
SENSITIVE_VALUE_PATTERNS = (
    (
        "credential_assignment",
        re.compile(r"(?i)\b(?:api[_ -]?key|password|secret)\s*[:=]"),
    ),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}")),
    ("prefixed_secret_key", re.compile(r"\bsk-[A-Za-z0-9_-]{12,}")),
    ("cloud_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("hosted_vcs_token", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
)
PERSONAL_DATA_PATTERNS = (
    ("mainland_mobile_number", re.compile(r"1[3-9]\d{9}")),
    ("mainland_id_card_number", re.compile(r"\d{17}[\dXx]")),
    ("email_address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")),
)
# A personal-data shape counts only when the leaf is not a hex digest: a 64-character
# request identifier contains plenty of incidental eleven- and eighteen-digit runs.
HEX_DIGEST_LEAF = re.compile(r"^[0-9a-f]{16,64}$")
HASH_LEAF = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
# Narrower than the artifact scan on purpose: source code names credential parameters all
# the time, so only a quoted literal value or a long bearer token is treated as a leak.
REPOSITORY_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_ -]?key|password|secret|authorization)\s*[=:]\s*[\"'][^\"']{12,}[\"']"
)
REPOSITORY_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{16,}")
TRUNCATED_HASH_LEAF = re.compile(r"^(?:statement|action):[0-9a-f]{24}$")
IDENTIFIER_LEAF = re.compile(
    r"^(?:GOLD-\d{4}|[A-Za-z][A-Za-z0-9._:/+@-]{0,79}|\d{4,8}|[0-9a-f]{8,64})$"
)
# Reason codes this repository generates itself come from rules.is_generated_reason_code;
# anything else sitting in a free-text field originated at the model.
_CONTRACT_LITERALS = ("timeout_seconds", "concurrency", "max_retries", "temperature")


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One persisted artifact with what an auditor needs to locate and trust it."""

    path: str
    kind: str
    sha256: str
    file_mode: str
    container_mode: str
    schema_version: str | None
    load_status: str
    status_detail: str | None = None
    batch_id: str | None = None
    stage_id: str | None = None
    topology: str | None = None
    repeat_seed: int | None = None
    execution_mode: str | None = None
    code_revision: str | None = None
    gold_manifest_checksum: str | None = None
    case_count: int | None = None
    branch_count: int | None = None
    model_call_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    resolved_model_ids: tuple[str, ...] = ()
    outcome: str | None = None
    failure_code: str | None = None
    voided: bool | None = None


def _mode(path: Path) -> str:
    return stat.filemode(stat.S_IMODE(path.stat().st_mode))[-9:]


def _detail(exc: BaseException) -> str:
    return str(exc).strip().replace("\n", " ")[:200]


def _string_leaves(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _string_leaves(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _string_leaves(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _load_smoke(path: Path) -> tuple[dict[str, Any], str, str | None]:
    """Load a V1-02B smoke manifest, separating contract drift from real breakage.

    ``run-manifest.v1`` pins its contract values as literals, so the §37 revision that
    moved the timeout from 30 to 90 seconds left the three earlier manifests unloadable by
    the current model even though they are structurally intact history.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        smoke_run.RunManifest.model_validate(payload)
        return payload, "ok", None
    except Exception as exc:  # noqa: BLE001 - classifying the failure is the point
        text = str(exc)
        drifted = [
            name
            for name in _CONTRACT_LITERALS
            if re.search(rf"{name}\s+Input should be", text)
        ]
        absent = [name for name in ("error_reason_counts",) if name not in payload] + [
            name
            for name in ("error_reason",)
            if payload.get("calls") and name not in payload["calls"][0]
        ]
        if "Input should be" in text and drifted:
            detail = (
                "pinned contract literals no longer match the current model: "
                + ", ".join(drifted)
            )
            if absent:
                detail += f"; fields absent when written: {', '.join(absent)}"
            return payload, "contract_revision_drift", detail
        return payload, "load_failed", _detail(exc)


def _identity_entry(
    *,
    runs_dir: Path,
    path: Path,
    kind: str,
    payload: Mapping[str, Any],
    load_status: str,
    status_detail: str | None = None,
) -> LedgerEntry:
    calls = payload.get("calls") or payload.get("branches") or ()
    case_ids = {str(call.get("case_id")) for call in calls if isinstance(call, Mapping)}
    versions = payload.get("source_versions") or {}
    usage = payload.get("usage") or {}
    return LedgerEntry(
        path=str(path.relative_to(runs_dir)),
        kind=kind,
        sha256=sha256_file(path),
        file_mode=_mode(path),
        container_mode=_mode(path.parent),
        schema_version=payload.get("schema_version"),
        load_status=load_status,
        status_detail=status_detail,
        batch_id=payload.get("batch_id"),
        stage_id=payload.get("stage_id"),
        topology=payload.get("topology"),
        repeat_seed=payload.get("repeat_seed"),
        execution_mode=payload.get("execution_mode"),
        code_revision=versions.get("code"),
        gold_manifest_checksum=payload.get("gold_manifest_checksum"),
        case_count=len(case_ids) or None,
        branch_count=payload.get("branch_count") or len(calls) or None,
        model_call_count=payload.get("model_call_count"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        resolved_model_ids=tuple(payload.get("resolved_model_ids") or ()),
        outcome=payload.get("outcome"),
        failure_code=payload.get("failure_code"),
        voided=payload.get("voided"),
    )


def build_ledger(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[LedgerEntry]:
    """Describe every persisted artifact without requiring any of them to be re-run."""

    root = runs_dir.resolve()
    entries: list[LedgerEntry] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = directory / MANIFEST_FILE_NAME
        if manifest.is_file():
            entries.append(_eval_manifest_entry(root, manifest))
        projection = directory / PROJECTION_FILE_NAME
        if projection.is_file():
            entries.append(_projection_entry(root, projection))
        smoke = directory / RUN_MANIFEST_FILE_NAME
        if smoke.is_file():
            payload, status, detail = _load_smoke(smoke)
            entries.append(
                _identity_entry(
                    runs_dir=root,
                    path=smoke,
                    kind="smoke-run-manifest",
                    payload=payload,
                    load_status=status,
                    status_detail=detail,
                )
            )
        for report in sorted(directory.glob(REPORT_GLOB)):
            entries.append(
                _identity_entry(
                    runs_dir=root,
                    path=report,
                    kind="eval-report",
                    payload=json.loads(report.read_text(encoding="utf-8")),
                    load_status="ok",
                )
            )
    return entries


def _eval_manifest_entry(runs_dir: Path, path: Path) -> LedgerEntry:
    try:
        payload: Mapping[str, Any] = load_eval_manifest(path).model_dump(mode="json")
        status, detail = "ok", None
    except Exception as exc:  # noqa: BLE001 - the ledger must still record the file
        payload = json.loads(path.read_text(encoding="utf-8"))
        status, detail = "load_failed", _detail(exc)
    return _identity_entry(
        runs_dir=runs_dir,
        path=path,
        kind="eval-run-manifest",
        payload=payload,
        load_status=status,
        status_detail=detail,
    )


def _projection_entry(runs_dir: Path, path: Path) -> LedgerEntry:
    try:
        payload: Mapping[str, Any] = load_projection(path).model_dump(mode="json")
        status, detail = "ok", None
    except Exception as exc:  # noqa: BLE001
        payload = json.loads(path.read_text(encoding="utf-8"))
        status, detail = "load_failed", _detail(exc)
    branches = payload.get("branches") or ()
    findings = [
        finding
        for branch in branches
        if isinstance(branch, Mapping)
        for finding in branch.get("findings") or ()
    ]
    entry = _identity_entry(
        runs_dir=runs_dir,
        path=path,
        kind="eval-projection",
        payload=payload,
        load_status=status,
        status_detail=detail,
    )
    return replace(
        entry,
        case_count=len(
            {str(b.get("case_id")) for b in branches if isinstance(b, Mapping)}
        ),
        branch_count=len(branches),
        model_call_count=len(findings),
    )


@dataclass(frozen=True, slots=True)
class BrakeRollup:
    """Cumulative §15.4 local-experiment brake recomputed from the artifacts."""

    decision_max_requests: int
    decision_max_input_tokens: int
    decision_max_output_tokens: int
    warning_ratio: float
    manifest_requests: int
    manifest_input_tokens: int
    manifest_output_tokens: int
    request_share: float
    input_share: float
    output_share: float
    usage_bearing_requests: int
    dry_run_requests: int
    dry_run_input_tokens: int
    dry_run_output_tokens: int
    eval_segments: int
    smoke_segments: int
    warning_reached: bool
    unaccounted_requests_low: int
    unaccounted_requests_high: int
    unaccounted_output_tokens_low: int
    unaccounted_output_tokens_high: int
    adjusted_request_share: float
    adjusted_output_share: float


def cumulative_brake(
    entries: Sequence[LedgerEntry], runs_dir: Path = DEFAULT_RUNS_DIR
) -> BrakeRollup:
    """Recompute consumption from the manifests, including what no manifest can show."""

    maxima = (
        max(spec.max_requests for spec in EVAL_BATCHES.values()),
        max(spec.max_input_tokens for spec in EVAL_BATCHES.values()),
        max(spec.max_output_tokens for spec in EVAL_BATCHES.values()),
    )
    run_entries = [
        entry
        for entry in entries
        if entry.kind in {"eval-run-manifest", "smoke-run-manifest"}
    ]
    real = [entry for entry in run_entries if entry.execution_mode == "real_endpoint"]
    dry = [entry for entry in run_entries if entry.execution_mode == "dry_run"]
    requests = sum(entry.model_call_count or 0 for entry in real)
    input_tokens = sum(entry.input_tokens or 0 for entry in real)
    output_tokens = sum(entry.output_tokens or 0 for entry in real)
    low, high, out_low, out_high = _pause_consumption(runs_dir)
    warning = 0.8
    return BrakeRollup(
        decision_max_requests=maxima[0],
        decision_max_input_tokens=maxima[1],
        decision_max_output_tokens=maxima[2],
        warning_ratio=warning,
        manifest_requests=requests,
        manifest_input_tokens=input_tokens,
        manifest_output_tokens=output_tokens,
        request_share=round(requests / maxima[0], 6),
        input_share=round(input_tokens / maxima[1], 6),
        output_share=round(output_tokens / maxima[2], 6),
        usage_bearing_requests=sum(
            entry.model_call_count or 0
            for entry in real
            if (entry.input_tokens or 0) > 0
        ),
        dry_run_requests=sum(entry.model_call_count or 0 for entry in dry),
        dry_run_input_tokens=sum(entry.input_tokens or 0 for entry in dry),
        dry_run_output_tokens=sum(entry.output_tokens or 0 for entry in dry),
        eval_segments=len(
            [entry for entry in real if entry.kind == "eval-run-manifest"]
        ),
        smoke_segments=len(
            [entry for entry in real if entry.kind == "smoke-run-manifest"]
        ),
        warning_reached=(
            requests >= maxima[0] * warning
            or input_tokens >= maxima[1] * warning
            or output_tokens >= maxima[2] * warning
        ),
        unaccounted_requests_low=low,
        unaccounted_requests_high=high,
        unaccounted_output_tokens_low=out_low,
        unaccounted_output_tokens_high=out_high,
        adjusted_request_share=round((requests + high) / maxima[0], 6),
        adjusted_output_share=round((output_tokens + out_high) / maxima[2], 6),
    )


def _segment_rate(
    runs_dir: Path, *, batch_id: str, topology: str, repeat_seed: int
) -> tuple[float, float] | None:
    """Return ``(seconds per call, output tokens per call)`` for one completed segment."""

    for path in sorted(runs_dir.glob(f"*/{MANIFEST_FILE_NAME}")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("batch_id") != batch_id
            or payload.get("topology") != topology
            or payload.get("repeat_seed") != repeat_seed
        ):
            continue
        calls = payload["model_call_count"]
        span = (
            datetime.fromisoformat(payload["finished_at"])
            - datetime.fromisoformat(payload["started_at"])
        ).total_seconds()
        return span / calls, payload["usage"]["output_tokens"] / calls
    return None


def _pause_consumption(runs_dir: Path) -> tuple[int, int, int, int]:
    """Bound the in-flight V1-02C2 consumption that no manifest can show.

    The lower bound uses artifact timestamps only: the completed first segment ended, and
    the operator then asked for the pause. The upper bound uses the in-flight duration
    recorded in Evidence §44. Both divide by the measured per-call wall clock of that same
    segment, so the estimate never depends on an unrecorded assumption about latency.
    """

    measured = _segment_rate(
        runs_dir,
        batch_id=PAUSE_BATCH_ID,
        topology="multi_agent",
        repeat_seed=20260901,
    )
    if measured is None:
        return 0, 0, 0, 0
    seconds_per_call, output_per_call = measured
    low_minutes = (
        PAUSE_REQUESTED_AT - PAUSE_SEGMENT_1_FINISHED_AT
    ).total_seconds() / 60
    high_minutes = max(low_minutes, PAUSE_IN_FLIGHT_MINUTES_RECORDED)
    low_calls = int(low_minutes * 60 / seconds_per_call)
    high_calls = int(round(high_minutes * 60 / seconds_per_call))
    return (
        low_calls,
        high_calls,
        int(low_calls * output_per_call),
        int(high_calls * output_per_call),
    )


def _enumerated_vocabulary() -> frozenset[str]:
    """Every value the contracts enumerate, so classification never relies on prose."""

    from .contracts import (
        FindingAssertion,
        FindingStatus,
        RecommendationStatus,
        RiskCategory,
        RiskLevel,
        WorkerKind,
    )

    words = {
        member.value
        for enum in (
            FindingAssertion,
            FindingStatus,
            RecommendationStatus,
            RiskCategory,
            RiskLevel,
            WorkerKind,
        )
        for member in enum
    }
    words |= {
        "completed",
        "failed_closed",
        "succeeded",
        "failed",
        "not_requested",
        "not_recorded",
        "model_call",
        "skipped_empty_context",
        "dry_run",
        "real_endpoint",
        "single_agent",
        "multi_agent",
        "chat_adapter_smoke",
        "redacted",
        "dev",
        "validation",
        "holdout",
        "indistinguishable",
        "single_agent_better",
        "multi_agent_better",
        "raw_query",
        "masked_case_number_query",
    }
    return frozenset(words)


def classify_leaves(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Split every string leaf into identity, enumeration or free-form prose."""

    vocabulary = _enumerated_vocabulary()
    classes: dict[str, Counter[str]] = {}
    lengths: dict[str, int] = {}
    for path, value in _string_leaves(payload):
        if HASH_LEAF.fullmatch(value):
            shape = "hash"
        elif TRUNCATED_HASH_LEAF.fullmatch(value):
            shape = "truncated_hash"
        elif is_generated_reason_code(value):
            shape = "aggregator_code"
        elif IDENTIFIER_LEAF.fullmatch(value):
            shape = "identifier"
        elif value in vocabulary:
            shape = "enumerated"
        else:
            shape = "free_form"
        base = re.sub(r"\[\d+\]", "[]", path)
        classes.setdefault(base, Counter())[shape] += 1
        lengths[base] = max(lengths.get(base, 0), len(value))
    return {
        "field_classes": {
            base: dict(sorted(counts.items()))
            for base, counts in sorted(classes.items())
        },
        "longest_per_field": dict(sorted(lengths.items())),
        "free_form_fields": {
            base: dict(counts)["free_form"]
            for base, counts in sorted(classes.items())
            if counts.get("free_form")
        },
    }


def _pattern_hits(name: str, pattern: re.Pattern[str], value: str) -> bool:
    """Personal-data shapes must be a whole word; secret shapes may sit inside prose."""

    if not any(name.startswith(prefix) for prefix in ("mainland_", "email_")):
        return bool(pattern.search(value))
    if HEX_DIGEST_LEAF.fullmatch(value):
        return False
    return any(pattern.fullmatch(word) for word in value.split())


@dataclass(frozen=True, slots=True)
class ScanFinding:
    """One measured redaction result, always expressed as counts and hashes."""

    code: str
    severity: Literal["clean", "note", "finding"]
    subject: str
    detail: str
    count: int | None = None


def _sensitive_keys(payload: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(payload, Mapping):
        for key, child in payload.items():
            normalized = str(key).lower().replace("-", "_")
            label = f"{path}.{key}" if path else str(key)
            if any(part in normalized for part in SECRET_KEY_PARTS):
                hits.append(label)
            hits.extend(_sensitive_keys(child, label))
    elif isinstance(payload, (list, tuple)):
        for index, child in enumerate(payload):
            hits.extend(_sensitive_keys(child, f"{path}[{index}]"))
    return hits


def _splits_with_artifacts(runs_dir: Path) -> frozenset[GoldSplit]:
    """Which Gold splits the persisted segments actually cover.

    The hidden split only enters the echo and label-overlap scans once a batch that
    declares it has really been run, so auditing the historical artifacts never reads it.
    """

    covered: set[GoldSplit] = set()
    for path in sorted(runs_dir.glob(f"*/{MANIFEST_FILE_NAME}")):
        spec = EVAL_BATCHES.get(
            str(json.loads(path.read_text(encoding="utf-8")).get("batch_id"))
        )
        if spec is not None:
            covered.update(spec.splits)
    return frozenset(covered)


def _visible_corpus_text(splits: frozenset[GoldSplit]) -> dict[str, str]:
    """Serialise the non-hidden corpus the models were shown, for echo detection."""

    text: dict[str, str] = {}
    for version in GoldCorpusVersion:
        for case in load_gold_cases(corpus_version=version).cases:
            if case.split not in splits:
                continue
            blob = json.dumps(
                case.model_input.model_dump(mode="json"), ensure_ascii=False
            )
            text[case.case_id] = text.get(case.case_id, "") + blob + case.question
    return text


def _visible_label_terms(splits: frozenset[GoldSplit]) -> dict[str, set[str]]:
    terms: dict[str, set[str]] = {}
    for version in GoldCorpusVersion:
        for split in splits:
            for label in load_gold_labels_for_split(
                split, corpus_version=version, allow_holdout=True
            ):
                payload = label.model_dump(mode="json")
                bucket = terms.setdefault(label.case_id, set())
                for key in (
                    "expected_recommendation",
                    "expected_risk_level",
                    "expected_review_state",
                ):
                    bucket.add(str(payload.get(key)))
                for key in (
                    "rule_ids",
                    "acceptable_alternatives",
                    "expected_missing_information",
                ):
                    bucket.update(str(item) for item in payload.get(key) or ())
    return terms


def _retained_model_text(runs_dir: Path) -> list[tuple[str, str, str]]:
    """``(segment, case_id, text)`` for every model statement fragment kept on disk."""

    rows: list[tuple[str, str, str]] = []
    for path in sorted(runs_dir.glob(f"*/{PROJECTION_FILE_NAME}")):
        projection = load_projection(path)
        for branch in projection.branches:
            for finding in branch.findings:
                for item in finding.missing_information:
                    rows.append((path.parent.name, branch.case_id, item))
    return rows


def _permission_findings(root: Path) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for path in (root, *sorted(root.rglob("*"))):
        wanted = "rwx------" if path.is_dir() else "rw-------"
        if _mode(path) != wanted:
            findings.append(
                ScanFinding(
                    code="permission_deviation",
                    severity="finding",
                    subject=str(path.relative_to(root.parent)),
                    detail=f"mode is {_mode(path)} where the run contract says {wanted}",
                )
            )
    return findings


def _tracked_tree_findings(credential: str) -> list[ScanFinding]:
    """Prove that no secret or run payload entered the repository itself."""

    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    findings: list[ScanFinding] = []
    for relative in sorted(name for name in listing.stdout.split("\0") if name):
        path = REPO_ROOT / relative
        if not path.is_file() or path.stat().st_size > 4_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if credential and credential in text:
            findings.append(
                ScanFinding(
                    code="live_credential_in_repository",
                    severity="finding",
                    subject=relative,
                    detail="the locally configured credential appears in a tracked file",
                )
            )
        # The guard's own tests plant a placeholder secret on purpose, so placeholder
        # literals and the unambiguous prefixed-key shapes are stripped first. What is
        # left must look like a real credential: a quoted literal value or a bearer token.
        body = re.sub(
            r"(?i)[A-Za-z0-9_.-]*(?:placeholder|example|dummy|sample|not-real|fake)"
            r"[A-Za-z0-9_.-]*",
            "",
            text,
        )
        body = re.sub(
            r"sk-[A-Za-z0-9_-]{12,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}",
            "",
            body,
        )
        for code, detail, pattern in (
            (
                "repository_secret_assignment",
                "a tracked file assigns a literal secret-looking value",
                REPOSITORY_SECRET_ASSIGNMENT,
            ),
            (
                "repository_bearer_token",
                "a tracked file carries a bearer token value",
                REPOSITORY_BEARER_TOKEN,
            ),
        ):
            if pattern.search(body):
                findings.append(
                    ScanFinding(
                        code=code,
                        severity="finding",
                        subject=relative,
                        detail=detail,
                    )
                )
        if re.search(r"^MODEL_API_KEY=\S", text, flags=re.MULTILINE):
            findings.append(
                ScanFinding(
                    code="repository_inline_credential_assignment",
                    severity="finding",
                    subject=relative,
                    detail="a tracked file assigns a non-empty MODEL_API_KEY value",
                )
            )
    return findings


def scan_artifacts(
    runs_dir: Path = DEFAULT_RUNS_DIR, *, credential: str = ""
) -> dict[str, Any]:
    """Run the AC-09 redaction battery over every persisted artifact."""

    root = runs_dir.resolve()
    findings: list[ScanFinding] = []
    scanned = 0
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        scanned += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        relative = str(path.relative_to(root))
        leaves = list(_string_leaves(payload))
        if credential and credential in path.read_text(encoding="utf-8"):
            findings.append(
                ScanFinding(
                    code="live_credential_present",
                    severity="finding",
                    subject=relative,
                    detail="the locally configured credential appears in a persisted artifact",
                )
            )
        secret_paths = secret_value_paths(payload, credential=credential)
        if secret_paths:
            findings.append(
                ScanFinding(
                    code="secret_value_path_hit",
                    severity="finding",
                    subject=relative,
                    detail=(
                        "the write-time detector flags string leaves at: "
                        + ", ".join(
                            ".".join(str(part) for part in item)
                            for item in secret_paths[:5]
                        )
                    ),
                    count=len(secret_paths),
                )
            )
        for name, pattern in (*SENSITIVE_VALUE_PATTERNS, *PERSONAL_DATA_PATTERNS):
            hits = sum(1 for _, value in leaves if _pattern_hits(name, pattern, value))
            if hits:
                findings.append(
                    ScanFinding(
                        code=f"sensitive_value_shape_{name}",
                        severity="finding",
                        subject=relative,
                        detail=f"{hits} string leaves match the {name} shape",
                        count=hits,
                    )
                )
        gov = sum(1 for _, value in leaves if GOV_REFERENCE.search(value))
        if gov:
            findings.append(
                ScanFinding(
                    code="external_government_reference",
                    severity="finding",
                    subject=relative,
                    detail=f"{gov} string leaves reference a government source",
                    count=gov,
                )
            )
        key_hits = _sensitive_keys(payload)
        if key_hits:
            findings.append(
                ScanFinding(
                    code="sensitive_field_name",
                    severity="note",
                    subject=relative,
                    detail=(
                        f"{len(key_hits)} field names mention secret material; both reasons "
                        "are expected and neither carries a value - a closed manifest names "
                        "what it must not retain, and two case families are named after the "
                        "attack they probe: " + ", ".join(sorted(set(key_hits))[:8])
                    ),
                    count=len(key_hits),
                )
            )
        reasons = _manifest_guard_reasons(payload, credential)
        if reasons:
            drifted = reasons[0].startswith(
                "manifest_unloadable_under_current_contract"
            )
            findings.append(
                ScanFinding(
                    code="manifest_guard_hit",
                    severity="note" if drifted else "finding",
                    subject=relative,
                    detail=(
                        "the write-time guard re-applied to this manifest on disk: "
                        f"{reasons[0]}"
                    ),
                    count=len(reasons),
                )
            )
    retained = _retained_model_text(root)
    splits = _splits_with_artifacts(root)
    corpus = _visible_corpus_text(splits)
    labels = _visible_label_terms(splits)
    echoes = [
        row
        for row in retained
        if len(row[2]) >= 12 and row[2] in corpus.get(row[1], "")
    ]
    overlap = [row for row in retained if row[2] in labels.get(row[1], set())]
    totals = {
        "retained_occurrences": len(retained),
        "retained_distinct": len({row[2] for row in retained}),
        "retained_longest": max((len(row[2]) for row in retained), default=0),
        "retained_segments": len({row[0] for row in retained}),
        "echo_count": len(echoes),
        "label_term_overlap": len(overlap),
    }
    if retained:
        findings.append(
            ScanFinding(
                code="retained_model_free_text",
                severity="finding",
                subject="eval-projection.v1",
                detail=(
                    "branches[].findings[].missing_information[] still keeps model prose in "
                    "the scoring input (the derived eval-report.v3 no longer copies it): "
                    f"{totals['retained_occurrences']} occurrences across "
                    f"{totals['retained_segments']} segments, "
                    f"{totals['retained_distinct']} distinct, longest "
                    f"{totals['retained_longest']} characters; the projection schema "
                    "bounds the field to 12 items of at most 160 characters"
                ),
                count=totals["retained_occurrences"],
            )
        )
    if echoes:
        findings.append(
            ScanFinding(
                code="model_text_echoes_visible_corpus",
                severity="note",
                subject="eval-projection.v1",
                detail=(
                    f"{len(echoes)} retained strings are verbatim substrings of the case "
                    "context they came from; they are business field names, and the corpus "
                    "is synthetic and already committed, so no undisclosed source text left "
                    "the repository"
                ),
                count=len(echoes),
            )
        )
    if overlap:
        findings.append(
            ScanFinding(
                code="model_text_matches_gold_term",
                severity="note",
                subject="eval-projection.v1",
                detail=(
                    f"{len(overlap)} retained strings equal a Gold label term through "
                    "shared generic business wording; labels were only loaded at scoring "
                    "time, so this is not a leak direction"
                ),
                count=len(overlap),
            )
        )
    findings.append(
        ScanFinding(
            code="retention_declaration_scope",
            severity="note",
            subject="eval-manifest-retention.v1",
            detail=(
                "the manifest declares model_statement_text_retained=false and keeps only "
                "hashes for statement and recommended_action, but neither the projection "
                "schema nor the derived report carries that declaration, so the declaration "
                "covers the manifest only"
            ),
        )
    )
    findings.extend(_permission_findings(root))
    findings.extend(_tracked_tree_findings(credential))
    classification = classify_leaves(
        {
            "projections": [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(root.glob(f"eval-*/{PROJECTION_FILE_NAME}"))
            ]
        }
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifacts_scanned": scanned,
        "splits_covered": sorted(split.value for split in splits),
        "credential_identifier": (
            hashlib.sha256(credential.encode("utf-8")).hexdigest()[:12]
            if credential
            else None
        ),
        "totals": totals,
        "projection_field_classes": classification,
        "findings": [asdict(item) for item in findings],
        "finding_codes": dict(sorted(Counter(item.code for item in findings).items())),
    }


def _manifest_guard_reasons(
    payload: Mapping[str, Any], credential: str
) -> tuple[str, ...]:
    """Re-apply the write-time guard to a manifest that is already on disk."""

    schema = payload.get("schema_version")
    if schema == "run-manifest.v1":
        try:
            return smoke_run.guard_run_manifest(
                smoke_run.RunManifest.model_validate(dict(payload)),
                credential=credential,
            )
        except Exception as exc:  # noqa: BLE001 - drift is itself the audit result
            return (f"manifest_unloadable_under_current_contract:{_detail(exc)}",)
    if schema == EVAL_MANIFEST_SCHEMA_VERSION:
        try:
            return guard_eval_manifest(
                EvalRunManifest.model_validate(dict(payload)), credential=credential
            )
        except Exception as exc:  # noqa: BLE001
            return (f"manifest_unloadable_under_current_contract:{_detail(exc)}",)
    return ()


@dataclass(frozen=True, slots=True)
class TraceCall:
    worker: str | None
    disposition: str
    outcome: str
    http_status: int | None
    error_code: str | None
    error_detail: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    retry_count: int
    request_id_hash: str | None
    resolved_model_id: str | None


@dataclass(frozen=True, slots=True)
class TraceFinding:
    worker: str
    finding_key: str
    category: str
    risk_level: str
    assertion: str
    status: str
    statement_hash: str
    recommended_action_hash: str
    evidence_refs: tuple[str, ...]
    business_fact_refs: tuple[str, ...]
    missing_information_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseTrace:
    """The whole auditable chain for one ``(segment, case)``, with no raw content."""

    run_id: str
    batch_id: str
    topology: str
    repeat_seed: int
    execution_mode: str
    case_id: str
    split: str
    case_family_id: str
    allowed_context_evidence_refs: tuple[str, ...]
    calls: tuple[TraceCall, ...]
    findings: tuple[TraceFinding, ...]
    recommendation: str
    highest_risk: str
    human_review_required: bool
    branch_outcomes: Mapping[str, str]
    failed_workers: tuple[str, ...]
    conflict_keys: tuple[str, ...]
    missing_information: tuple[str, ...]
    failure_reasons: tuple[str, ...]


def _segment(run_id: str, runs_dir: Path) -> tuple[EvalRunManifest, EvalProjection]:
    directory = runs_dir.resolve() / run_id
    manifest_path = directory / MANIFEST_FILE_NAME
    if not manifest_path.is_file():
        raise ValueError("segment_not_found")
    return (
        load_eval_manifest(manifest_path),
        load_projection(directory / PROJECTION_FILE_NAME),
    )


def _gold_case(corpus_version: GoldCorpusVersion, case_id: str) -> GoldCase:
    for case in load_gold_cases(corpus_version=corpus_version).cases:
        if case.case_id == case_id:
            return case
    raise ValueError("case_not_in_corpus")


def _hash_reason(item: str) -> str:
    return (
        item
        if is_generated_reason_code(item)
        else f"sha256:{hashlib.sha256(item.encode('utf-8')).hexdigest()}"
    )


def trace_case(
    *,
    run_id: str,
    case_id: str,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    contract: str = "v1",
) -> CaseTrace:
    """Rebuild one case's end-to-end path from the persisted segment alone."""

    manifest, projection = _segment(run_id, runs_dir)
    spec = EVAL_BATCHES[manifest.batch_id]
    case = _gold_case(GoldCorpusVersion(spec.gold_corpus_version), case_id)
    branches = tuple(b for b in projection.branches if b.case_id == case_id)
    if not branches:
        raise ValueError("case_has_no_projected_branch")
    if contract not in CONTRACT_RULE_SETS:
        raise ValueError("contract_not_registered")
    rule_set = CONTRACT_RULE_SETS[contract]
    observation = observations_from_projection(
        projection.model_copy(update={"branches": branches}),
        {case_id: case},
        rule_set=rule_set,
    )[0]
    allowed = tuple(
        ref_id
        for item in case.allowed_context
        for ref_id in (item.allowed_ref_ids or ())
    )
    return CaseTrace(
        run_id=manifest.run_id,
        batch_id=manifest.batch_id,
        topology=manifest.topology,
        repeat_seed=manifest.repeat_seed,
        execution_mode=manifest.execution_mode,
        case_id=case_id,
        split=case.split.value,
        case_family_id=case.case_family_id,
        allowed_context_evidence_refs=allowed,
        calls=tuple(
            TraceCall(
                worker=call.worker.value if call.worker else None,
                disposition=call.disposition,
                outcome=call.outcome,
                http_status=call.http_status,
                error_code=call.error_code,
                error_detail=call.error_detail,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                latency_ms=call.latency_ms,
                retry_count=call.retry_count,
                request_id_hash=call.request_id_hash,
                resolved_model_id=call.resolved_model_id,
            )
            for call in manifest.calls
            if call.case_id == case_id
        ),
        findings=tuple(
            TraceFinding(
                worker=finding.worker.value,
                finding_key=finding.finding_key,
                category=finding.category.value,
                risk_level=finding.risk_level.value,
                assertion=finding.assertion.value,
                status=finding.status.value,
                statement_hash=finding.statement_hash,
                recommended_action_hash=finding.recommended_action_hash,
                evidence_refs=tuple(
                    f"{ref.source_id}@{ref.version}" for ref in finding.evidence_refs
                ),
                business_fact_refs=tuple(
                    f"{ref.source_id}@{ref.version}"
                    for ref in finding.business_fact_refs
                ),
                missing_information_hashes=tuple(
                    hashlib.sha256(item.encode("utf-8")).hexdigest()[:16]
                    for item in finding.missing_information
                ),
            )
            for branch in branches
            for finding in branch.findings
        ),
        recommendation=observation.recommendation.value,
        highest_risk=observation.highest_risk.value,
        human_review_required=observation.human_review_required,
        branch_outcomes=observation.branch_outcomes,
        failed_workers=observation.failed_workers,
        conflict_keys=observation.conflict_keys,
        missing_information=tuple(
            _hash_reason(item) for item in observation.missing_information
        ),
        failure_reasons=observation.failure_reasons,
    )


@dataclass(frozen=True, slots=True)
class Claim:
    """One number a report quotes, and where it must come from.

    ``source`` is a tiny DSL so the claim table stays one row per number:
    ``report:<batch>/<dotted.path>`` a field of the regenerated batch report;
    ``segments:<batch>.<topology>.<key>`` a counter over the segment branch entries;
    ``slice:<batch>/<topology>/<worker>`` one Worker-level coverage ratio recomputed on
    the range-aligned slice; ``brake:<field>`` a field of the brake rollup;
    ``rescore:<batch>/<contract>/<dotted.path>`` a field of the V1-05A offline rescore
    report regenerated under the named aggregation contract.
    """

    claim_id: str
    subject: str
    source: str
    expectation: float | int | str
    tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class SliceScope:
    """A narrowed observation set scored with the frozen formulas."""

    batch_id: str
    topology: str
    worker: str


def _report_default(value: Any) -> Any:
    """Serialize a regenerated report exactly as ``write_report`` persists it."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unserialisable report value: {type(value).__name__}")


_SOURCE_KIND = re.compile(
    r"^(?:report:[a-z0-9-]+/.+|segments:.+\.[a-z_]+\..+|slice:[a-z0-9-]+/[a-z_]+/[a-z_]+"
    r"|brake:[a-z_]+|rescore:[a-z0-9-]+/v[12]/.+)$"
)
_REPORT_SOURCE = re.compile(r"^report:(?P<batch>[a-z0-9-]+)/(?P<path>.+)$")
_SEGMENTS_SOURCE = re.compile(r"^segments:(?P<key>.+)$")
_SLICE_SOURCE = re.compile(
    r"^slice:(?P<batch>[a-z0-9-]+)/(?P<topology>[a-z_]+)/(?P<worker>[a-z_]+)$"
)
_BRAKE_SOURCE = re.compile(r"^brake:(?P<field>[a-z_]+)$")
_RESCORE_SOURCE = re.compile(
    r"^rescore:(?P<batch>[a-z0-9-]+)/(?P<contract>v[12])/(?P<path>.+)$"
)
# Both slices use one narrowed scope: dev's first 10 cases per family and the two seeds
# that the V1 and V2 batches share.
SLICE_CASES_PER_FAMILY = 10
SLICE_REPEAT_SEEDS = (20260901, 20260902)


_SELECTOR = re.compile(r"^(.*?)\[([A-Za-z_][A-Za-z0-9_]*)=(.+)\]$")


def _resolve_path(document: Mapping[str, Any], dotted: str) -> Any:
    """Walk a dotted claim path, allowing ``list[field=value]`` selectors."""

    current: Any = document
    for part in dotted.split("."):
        selector = _SELECTOR.fullmatch(part)
        if selector is not None:
            name, key, wanted = selector.groups()
            if name:
                current = current[name]
            matches = [
                item
                for item in current
                if isinstance(item, Mapping) and str(item.get(key)) == wanted
            ]  # exactly one element must answer to the selector
            if len(matches) != 1:
                raise KeyError(part)
            current = matches[0]
        elif part.isdigit():
            current = current[int(part)]
        else:
            current = current[part]
    return current


def _segment_counters(runs_dir: Path) -> Counter[str]:
    """Count branch outcomes straight out of the segment manifests."""

    totals: Counter[str] = Counter()
    for path in sorted(runs_dir.glob(f"*/{MANIFEST_FILE_NAME}")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        batch = payload.get("batch_id")
        topology = payload.get("topology")
        if not batch or not topology:
            continue
        for call in payload.get("calls") or ():
            prefix = f"{batch}.{topology}"
            totals[f"{prefix}.calls"] += 1
            if (call.get("input_tokens") or 0) > 0:
                totals[f"{prefix}.usage_bearing_calls"] += 1
            totals[f"{prefix}.{call.get('outcome')}"] += 1
            if call.get("error_code"):
                totals[f"{prefix}.error.{call['error_code']}"] += 1
            if call.get("error_detail"):
                totals[f"{prefix}.detail.{call['error_detail']}"] += 1
    return totals


def _slice_coverage(scope: SliceScope, *, runs_dir: Path) -> float:
    """Recompute one Worker-level coverage ratio on a narrowed case/seed scope.

    Only the observation set is narrowed. The formulas, thresholds and labels are the ones
    ``eval_score`` uses, which is what lets the V1/V2 range-aligned slice be quoted without
    a second scoring implementation.
    """

    version = gold_corpus_contract(EVAL_BATCHES[scope.batch_id]).version
    cases = load_gold_cases(corpus_version=version).cases
    labels: dict[str, GoldLabel] = {
        label.case_id: label
        for label in load_gold_labels_for_split(GoldSplit.DEV, corpus_version=version)
    }
    selected: dict[str, GoldCase] = {}
    per_family: dict[str, int] = {}
    for case in cases:
        if case.split is not GoldSplit.DEV:
            continue
        if per_family.get(case.case_family_id, 0) >= SLICE_CASES_PER_FAMILY:
            continue
        selected[case.case_id] = case
        per_family[case.case_family_id] = per_family.get(case.case_family_id, 0) + 1
    observations = []
    for path in sorted(runs_dir.glob(f"*/{MANIFEST_FILE_NAME}")):
        manifest = load_eval_manifest(path)
        if (
            manifest.batch_id != scope.batch_id
            or manifest.topology != scope.topology
            or manifest.repeat_seed not in SLICE_REPEAT_SEEDS
        ):
            continue
        projection = load_projection(path.parent / PROJECTION_FILE_NAME)
        # A batch that also covers validation must not feed cases the slice excluded.
        narrowed = projection.model_copy(
            update={
                "branches": tuple(
                    branch
                    for branch in projection.branches
                    if branch.case_id in selected
                )
            }
        )
        if narrowed.branches:
            observations.extend(observations_from_projection(narrowed, selected))
    metrics = quality_metrics(observations, selected, labels)
    entry = metrics["required_finding_coverage_by_worker"][scope.worker]
    value = getattr(entry, "value", None)
    if value is None:
        value = entry["value"]
    return float(value)


def cross_check(
    claims: Sequence[Claim], runs_dir: Path = DEFAULT_RUNS_DIR
) -> dict[str, Any]:
    """Recompute every reported number from the artifacts and compare it to the claim."""

    root = runs_dir.resolve()
    reports: dict[str, Mapping[str, Any]] = {}
    rescored: dict[tuple[str, str], Mapping[str, Any]] = {}
    counters: Counter[str] | None = None
    brake = cumulative_brake(build_ledger(root), runs_dir=root)
    rows: list[dict[str, Any]] = []
    for claim in claims:
        actual: Any = None
        report = _REPORT_SOURCE.match(claim.source)
        segments = _SEGMENTS_SOURCE.match(claim.source)
        slice_source = _SLICE_SOURCE.match(claim.source)
        brake_source = _BRAKE_SOURCE.match(claim.source)
        rescore_source = _RESCORE_SOURCE.match(claim.source)
        if report is not None:
            batch_id = report.group("batch")
            if batch_id not in reports:
                reports[batch_id] = json.loads(
                    json.dumps(
                        build_report(batch_id, runs_dir=root),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        default=_report_default,
                    )
                )
            try:
                actual = _resolve_path(reports[batch_id], report.group("path"))
            except (KeyError, IndexError, TypeError):
                actual = None
        elif segments is not None:
            counters = counters or _segment_counters(root)
            actual = counters.get(segments.group("key"), 0)
        elif slice_source is not None:
            actual = _slice_coverage(
                SliceScope(
                    batch_id=slice_source.group("batch"),
                    topology=slice_source.group("topology"),
                    worker=slice_source.group("worker"),
                ),
                runs_dir=root,
            )
        elif brake_source is not None:
            actual = getattr(brake, brake_source.group("field"), None)
        elif rescore_source is not None:
            key = (rescore_source.group("batch"), rescore_source.group("contract"))
            if key not in rescored:
                rescored[key] = json.loads(
                    json.dumps(
                        build_rescore_report(key[0], key[1], runs_dir=root),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        default=_report_default,
                    )
                )
            try:
                actual = _resolve_path(rescored[key], rescore_source.group("path"))
            except (KeyError, IndexError, TypeError):
                actual = None
        rows.append(
            {
                "claim_id": claim.claim_id,
                "subject": claim.subject,
                "source": claim.source,
                "expected": claim.expectation,
                "actual": actual,
                "match": _claim_matches(claim, actual),
            }
        )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "claims": len(rows),
        "matched": sum(1 for row in rows if row["match"]),
        "mismatched": [row["claim_id"] for row in rows if not row["match"]],
        "rows": rows,
    }


def _claim_source(claim: Claim) -> str:
    if claim.report_path is not None:
        return f"report:{claim.batch_id}.{claim.report_path}"
    if claim.segment_counter is not None:
        return f"segments:{claim.segment_counter}"
    if claim.slice_scope is not None:
        scope = claim.slice_scope
        return (
            f"slice:{scope.batch_id}/{scope.topology}/{scope.worker}"
            f"/dev{scope.cases_per_family}x{len(scope.repeat_seeds)}"
        )
    return f"ledger:{claim.ledger_field}"


def _claim_matches(claim: Claim, actual: Any) -> bool:
    if actual is None or isinstance(actual, (dict, list)):
        return False
    if isinstance(claim.expectation, str):
        return str(actual) == claim.expectation
    if not isinstance(actual, (int, float)):
        return False
    return abs(float(actual) - float(claim.expectation)) <= claim.tolerance


def run_audit(
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    credential: str = "",
    claims: Sequence[Claim] | None = None,
) -> dict[str, Any]:
    """Produce the whole AC-09 audit document in one pass."""

    entries = build_ledger(runs_dir)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ledger": [asdict(entry) for entry in entries],
        "brake": asdict(cumulative_brake(entries, runs_dir=runs_dir)),
        "scan": scan_artifacts(runs_dir, credential=credential),
        "cross_check": cross_check(
            claims if claims is not None else CLAIMS, runs_dir=runs_dir
        ),
    }


def write_audit(document: Mapping[str, Any], *, runs_dir: Path) -> Path:
    root = runs_dir.resolve()
    if ".local" not in root.parts:
        raise ValueError("artifact_dir_not_ignored")
    directory = root / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / f"{AUDIT_SCHEMA_VERSION}.json"
    path.write_text(
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def local_credential() -> str:
    """Read the locally configured credential for scanning only; never echo it."""

    return os.environ.get("MODEL_API_KEY", "")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the persisted V1-02 run artifacts (read-only, zero network)."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    ledger = commands.add_parser(
        "ledger", help="artifact inventory and cumulative brake"
    )
    ledger.add_argument("--summary-only", action="store_true")
    trace = commands.add_parser(
        "trace", help="end-to-end chain for one segment and case"
    )
    trace.add_argument("--run-id", required=True)
    trace.add_argument("--case-id", required=True)
    trace.add_argument(
        "--contract",
        choices=sorted(CONTRACT_RULE_SETS),
        default="v1",
        help="aggregation contract to replay under; v1 keeps the frozen history wording",
    )
    scan = commands.add_parser("scan", help="redaction battery over every artifact")
    scan.add_argument(
        "--use-local-credential",
        action="store_true",
        help="also search artifacts for the value of MODEL_API_KEY; only a hash "
        "identifier is ever reported",
    )
    check = commands.add_parser("cross-check", help="recompute the report numbers")
    check.add_argument(
        "--claim", action="append", default=[], help="restrict to claim ids"
    )
    audit = commands.add_parser("audit", help="ledger + brake + scan + cross-check")
    audit.add_argument(
        "--use-local-credential",
        action="store_true",
        help="pass the value of MODEL_API_KEY to the scan; only a hash identifier is kept",
    )
    for command in (ledger, trace, scan, check, audit):
        command.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "ledger":
            entries = build_ledger(args.runs_dir)
            brake = asdict(cumulative_brake(entries, runs_dir=args.runs_dir))
            document: Any = (
                brake
                if args.summary_only
                else {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "ledger": [asdict(entry) for entry in entries],
                    "brake": brake,
                }
            )
        elif args.command == "trace":
            document = asdict(
                trace_case(
                    run_id=args.run_id,
                    case_id=args.case_id,
                    runs_dir=args.runs_dir,
                    contract=args.contract,
                )
            )
        elif args.command == "scan":
            document = scan_artifacts(
                args.runs_dir,
                credential=local_credential() if args.use_local_credential else "",
            )
        elif args.command == "audit":
            document = run_audit(
                runs_dir=args.runs_dir,
                credential=local_credential() if args.use_local_credential else "",
            )
        else:
            claims = (
                CLAIMS
                if not args.claim
                else tuple(
                    claim for claim in CLAIMS if claim.claim_id in set(args.claim)
                )
            )
            document = cross_check(claims, runs_dir=args.runs_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(
            f"eval-audit-error: {type(exc).__name__}: {_detail(exc)}", file=sys.stderr
        )
        return 1
    if args.command == "cross-check" and args.output is None:
        # The gate is the exit status; the full row set goes to --output for a closer look.
        print(
            json.dumps(
                {
                    "claims": document["claims"],
                    "matched": document["matched"],
                    "mismatched": document["mismatched"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1 if document["mismatched"] else 0
    payload = json.dumps(
        document, ensure_ascii=False, indent=2, sort_keys=True, default=str
    )
    output = args.output
    if output is None and args.command == "audit":
        path = write_audit(document, runs_dir=args.runs_dir)
    elif output is None:
        print(payload)
        return 0
    else:
        path = Path(output)
        if ".local" not in path.resolve().parts:
            print("eval-audit-error: artifact_dir_not_ignored", file=sys.stderr)
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)
    print(f"wrote {path} sha256={sha256_file(path)}")
    return 0


# Every number the V1-02D, V1-05A and V1-05B reports quote, one per line as
# ``claim id|subject|artifact source|expected|tolerance``. The expectations are the values
# recorded in Evidence §42、§45 and OFF-02/03/04 or in RUN-01/AM-01, so a mismatch means the
# report and the persisted run records disagree rather than that a formula moved. Rows using
# the ``rescore:`` source carry the contract in the path: ``/v1/`` is the frozen-contract
# baseline and ``/v2/`` the revised one, so every "before → after" pair is two independent
# lookups against the same segments. The ``HB-*`` block is the V1-05B holdout batch read
# under both contracts. The trailing rows added for V1-05C cover exactly the numbers the
# §15.4 rejudgement table and the summary report quote beyond the Stage A/B set: the paired
# coverage deltas, the C2 end-to-end interval, the holdout token and latency readings the
# cost clause needs, and the holdout isolation/safety counters.
_CLAIM_ROWS = """\
V1-MULTI-VALID|C-2 多 Agent 结构化输出有效率|report:v1-02c-paired-360x3/cost_and_reliability.multi_agent.structured_output_valid_rate.value|0.5097643097643098|0
V1-MULTI-VALID-NUM|C-2 多 Agent 结构化输出有效数|report:v1-02c-paired-360x3/cost_and_reliability.multi_agent.structured_output_valid_rate.numerator|1514|0
V1-MULTI-CALLS|C-2 多 Agent 模型调用数|report:v1-02c-paired-360x3/cost_and_reliability.multi_agent.structured_output_valid_rate.denominator|2970|0
V1-SINGLE-VALID|C-2 单 Agent 结构化输出有效率|report:v1-02c-paired-360x3/cost_and_reliability.single_agent.structured_output_valid_rate.value|0.030555555555555555|0
V1-SINGLE-VALID-NUM|C-2 单 Agent 结构化输出有效数|report:v1-02c-paired-360x3/cost_and_reliability.single_agent.structured_output_valid_rate.numerator|33|0
V1-CALLS-SINGLE|C-2 单 Agent 模型调用数|report:v1-02c-paired-360x3/cost_and_reliability.single_agent.structured_output_valid_rate.denominator|1080|0
V1-ACCURACY|C-2 端到端建议准确率（两拓扑同值）|report:v1-02c-paired-360x3/quality.multi_agent.recommendation_accuracy.value|0.4166666666666667|0
V1-ACCURACY-NUM|C-2 端到端建议正确数|report:v1-02c-paired-360x3/quality.multi_agent.recommendation_accuracy.numerator|450|0
V1-ACCURACY-DEN|C-2 端到端观测数|report:v1-02c-paired-360x3/quality.multi_agent.recommendation_accuracy.denominator|1080|0
V1-MACRO-F1|C-2 建议宏平均 F1|report:v1-02c-paired-360x3/quality.multi_agent.recommendation_macro_f1.value|0.14705882352941177|0
V1-UNKNOWN|C-2 UNKNOWN 召回分子/分母|report:v1-02c-paired-360x3/quality.multi_agent.unknown_recall.numerator|720|0
V1-CONFLICT-NUM|C-2 冲突识别命中数|report:v1-02c-paired-360x3/quality.multi_agent.conflict_recall.numerator|0|0
V1-CONFLICT-DEN|C-2 应识别冲突观测数|report:v1-02c-paired-360x3/quality.multi_agent.conflict_recall.denominator|270|0
V1-HIGHRISK-DEN|C-2 高风险观测数|report:v1-02c-paired-360x3/quality.multi_agent.high_risk_miss_rate.denominator|540|0
V1-HIGHRISK-NUM|C-2 高风险漏报数|report:v1-02c-paired-360x3/quality.multi_agent.high_risk_miss_rate.numerator|0|0
V1-CITATION-MULTI|C-2 多 Agent 被支持引用数|report:v1-02c-paired-360x3/quality.multi_agent.citation_support_precision.numerator|2510|0
V1-CITATION-SINGLE|C-2 单 Agent 可评引用数|report:v1-02c-paired-360x3/quality.single_agent.citation_support_precision.denominator|65|0
V1-COVERAGE-MULTI|C-2 多 Agent 必需 Finding 覆盖数|report:v1-02c-paired-360x3/quality.multi_agent.required_finding_coverage.numerator|559|0
V1-COVERAGE-MULTI-RATE|C-2 多 Agent 必需 Finding 覆盖率|report:v1-02c-paired-360x3/quality.multi_agent.required_finding_coverage.value|0.17253086419753086|5e-05
V1-COVERAGE-DEN|C-2 必需 Finding 观测分母|report:v1-02c-paired-360x3/quality.multi_agent.required_finding_coverage.denominator|3240|0
V1-COVERAGE-SINGLE|C-2 单 Agent 必需 Finding 覆盖数|report:v1-02c-paired-360x3/quality.single_agent.required_finding_coverage.numerator|0|0
V1-TAKEOVER|C-2 人工接管观测数|report:v1-02c-paired-360x3/quality.multi_agent.human_takeover_rate.numerator|1080|0
V1-POLICY-COVERAGE|C-2 多 Agent policy 分支必需 Finding 覆盖|report:v1-02c-paired-360x3/quality.multi_agent.required_finding_coverage_by_worker.policy.numerator|1|0
V1-CONTRACT-COVERAGE|C-2 多 Agent contract 分支必需 Finding 覆盖|report:v1-02c-paired-360x3/quality.multi_agent.required_finding_coverage_by_worker.contract.numerator|24|0
V1-PROCUREMENT-COVERAGE|C-2 多 Agent procurement 分支必需 Finding 覆盖|report:v1-02c-paired-360x3/quality.multi_agent.required_finding_coverage_by_worker.procurement_data.numerator|534|0
V1-RETRIEVAL-RAW|C-2 v1 dev 检索 Recall@5 原始 query 命中|report:v1-02c-paired-360x3/retrieval.dev.variants.raw_query.recall_at_5.numerator|20|0
V1-RETRIEVAL-MASKED|C-2 v1 dev 检索 Recall@5 去 Case 号命中|report:v1-02c-paired-360x3/retrieval.dev.variants.masked_case_number_query.recall_at_5.numerator|5|0
V1-RETRIEVAL-DEN|C-2 v1 dev 检索 Gold 必需证据数|report:v1-02c-paired-360x3/retrieval.dev.variants.raw_query.recall_at_5.denominator|440|0
V1-RETRIEVAL-MRR|C-2 v1 dev 检索 MRR（原始 query）|report:v1-02c-paired-360x3/retrieval.dev.variants.raw_query.mrr|0.09090909090909091|5e-05
V1-VALIDATION-RETRIEVAL-RAW|C-2 v1 validation 检索 Recall@5 原始 query 命中|report:v1-02c-paired-360x3/retrieval.validation.variants.raw_query.recall_at_5.numerator|10|0
V1-VALIDATION-RETRIEVAL-MASKED|C-2 v1 validation 检索 Recall@5 去 Case 号命中|report:v1-02c-paired-360x3/retrieval.validation.variants.masked_case_number_query.recall_at_5.numerator|5|0
V1-DUPLICATE-DEN-DEV|C-2 v1 dev 重复查询率分母|report:v1-02c-paired-360x3/retrieval.dev.duplicate_query_rate.denominator|220|0
V1-DUPLICATE-DEN-VALIDATION|C-2 v1 validation 重复查询率分母（按 split 隔离后）|report:v1-02c-paired-360x3/retrieval.validation.duplicate_query_rate.denominator|110|0
V1-DELTA-COVERAGE|C-2 必需 Finding 覆盖配对差（单 − 多）|report:v1-02c-paired-360x3/paired_deltas[metric=required_finding_coverage].delta|-0.17253086419753086|0
V1-DELTA-COVERAGE-CI-LOW|C-2 必需 Finding 覆盖配对差区间下界|report:v1-02c-paired-360x3/paired_deltas[metric=required_finding_coverage].ci_low|-0.18425925925925923|0
V1-DELTA-COVERAGE-CI-HIGH|C-2 必需 Finding 覆盖配对差区间上界|report:v1-02c-paired-360x3/paired_deltas[metric=required_finding_coverage].ci_high|-0.16111111111111112|0
V1-DELTA-CITATION|C-2 引用支持率配对差（单 − 多）|report:v1-02c-paired-360x3/paired_deltas[metric=citation_support_precision].delta|-0.812037037037037|0
V1-DELTA-CITATION-CI-LOW|C-2 引用支持率配对差区间下界|report:v1-02c-paired-360x3/paired_deltas[metric=citation_support_precision].ci_low|-0.837962962962963|0
V1-DELTA-ENDTOEND|C-2 端到端建议完全匹配配对差|report:v1-02c-paired-360x3/paired_deltas[metric=recommendation_exact].delta|0|0
V1-DELTA-ENDTOEND-CI|C-2 端到端建议完全匹配区间上界|report:v1-02c-paired-360x3/paired_deltas[metric=recommendation_exact].ci_high|0|0
V1-DELTA-CASES|C-2 配对自助法有效样本（Case 数）|report:v1-02c-paired-360x3/paired_deltas[metric=recommendation_exact].case_count|360|0
V1-CROSS-CONTEXT-SINGLE-A|C-2 单 Agent 引用业务事实越界数|segments:v1-02c-paired-360x3.single_agent.detail.document_worker_cited_business_fact|501|0
V1-CROSS-CONTEXT-SINGLE-B|C-2 单 Agent 采购分支引用文档越界数|segments:v1-02c-paired-360x3.single_agent.detail.procurement_worker_cited_document|17|0
V1-CROSS-CONTEXT-MULTI|C-2 多 Agent 跨上下文引用越界数（按构造为 0）|segments:v1-02c-paired-360x3.multi_agent.detail.document_worker_cited_business_fact|0|0
V1-USAGE-BEARING-MULTI|C-2 多 Agent 产生用量的调用数|segments:v1-02c-paired-360x3.multi_agent.usage_bearing_calls|2769|0
V1-USAGE-BEARING-SINGLE|C-2 单 Agent 产生用量的调用数|segments:v1-02c-paired-360x3.single_agent.usage_bearing_calls|714|0
V1-TIMEOUT-SINGLE|C-2 单 Agent 超时分支数|segments:v1-02c-paired-360x3.single_agent.error.timeout|366|0
V1-TIMEOUT-MULTI|C-2 多 Agent 超时分支数|segments:v1-02c-paired-360x3.multi_agent.error.timeout|201|0
V1-TOKENS-MULTI|C-2 多 Agent 总 Token|report:v1-02c-paired-360x3/cost_and_reliability.multi_agent.total_tokens|7736414|0
V1-TOKENS-SINGLE|C-2 单 Agent 总 Token|report:v1-02c-paired-360x3/cost_and_reliability.single_agent.total_tokens|4527093|0
V1-P50-MULTI|C-2 多 Agent p50 延迟|report:v1-02c-paired-360x3/cost_and_reliability.multi_agent.latency_ms.p50|16743|0
V1-P95-MULTI|C-2 多 Agent p95 延迟|report:v1-02c-paired-360x3/cost_and_reliability.multi_agent.latency_ms.p95|90003|0
V1-P50-SINGLE|C-2 单 Agent p50 延迟|report:v1-02c-paired-360x3/cost_and_reliability.single_agent.latency_ms.p50|67053|0
V1-P95-SINGLE|C-2 单 Agent p95 延迟|report:v1-02c-paired-360x3/cost_and_reliability.single_agent.latency_ms.p95|90120|0
V1-RETRIES|C-2 真实端点重试事件数|report:v1-02c-paired-360x3/cost_and_reliability.single_agent.retry_total|1|0
V1-COMPLETENESS-MULTI|C-2 多 Agent 运行记录完整分支数|report:v1-02c-paired-360x3/cost_and_reliability.multi_agent.run_record_completeness.numerator|3240|0
V2-MULTI-VALID|C2 多 Agent 结构化输出有效率|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.structured_output_valid_rate.value|0.4727272727272727|0
V2-MULTI-VALID-NUM|C2 多 Agent 结构化输出有效数|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.structured_output_valid_rate.numerator|312|0
V2-SINGLE-VALID|C2 单 Agent 结构化输出有效率|report:v1-02c2-resolution-120x2/cost_and_reliability.single_agent.structured_output_valid_rate.value|0.004166666666666667|0
V2-SINGLE-VALID-NUM|C2 单 Agent 结构化输出有效数|report:v1-02c2-resolution-120x2/cost_and_reliability.single_agent.structured_output_valid_rate.numerator|1|0
V2-ACCURACY|C2 端到端建议准确率|report:v1-02c2-resolution-120x2/quality.multi_agent.recommendation_accuracy.value|0.4166666666666667|0
V2-ACCURACY-NUM|C2 端到端建议正确数|report:v1-02c2-resolution-120x2/quality.multi_agent.recommendation_accuracy.numerator|100|0
V2-ACCURACY-DEN|C2 端到端观测数|report:v1-02c2-resolution-120x2/quality.multi_agent.recommendation_accuracy.denominator|240|0
V2-CONFLICT-NUM|C2 冲突识别命中数|report:v1-02c2-resolution-120x2/quality.multi_agent.conflict_recall.numerator|0|0
V2-CONFLICT-DEN|C2 应识别冲突观测数|report:v1-02c2-resolution-120x2/quality.multi_agent.conflict_recall.denominator|60|0
V2-CONFLICT-CI-HIGH|C2 冲突召回区间上界|report:v1-02c2-resolution-120x2/quality.multi_agent.conflict_recall.ci_high|0.06017185214208986|5e-05
V2-HIGHRISK-DEN|C2 高风险观测数|report:v1-02c2-resolution-120x2/quality.multi_agent.high_risk_miss_rate.denominator|120|0
V2-UNKNOWN|C2 UNKNOWN 召回分子|report:v1-02c2-resolution-120x2/quality.multi_agent.unknown_recall.numerator|160|0
V2-CITATION-MULTI|C2 多 Agent 被支持引用数|report:v1-02c2-resolution-120x2/quality.multi_agent.citation_support_precision.numerator|544|0
V2-CITATION-SINGLE|C2 单 Agent 可评引用数|report:v1-02c2-resolution-120x2/quality.single_agent.citation_support_precision.denominator|3|0
V2-COVERAGE-MULTI|C2 多 Agent 必需 Finding 覆盖数|report:v1-02c2-resolution-120x2/quality.multi_agent.required_finding_coverage.numerator|155|0
V2-COVERAGE-MULTI-RATE|C2 多 Agent 必需 Finding 覆盖率|report:v1-02c2-resolution-120x2/quality.multi_agent.required_finding_coverage.value|0.2152777777777778|5e-05
V2-COVERAGE-SINGLE|C2 单 Agent 必需 Finding 覆盖数|report:v1-02c2-resolution-120x2/quality.single_agent.required_finding_coverage.numerator|1|0
V2-POLICY-COVERAGE|C2 多 Agent policy 分支必需 Finding 覆盖|report:v1-02c2-resolution-120x2/quality.multi_agent.required_finding_coverage_by_worker.policy.numerator|37|0
V2-CONTRACT-COVERAGE|C2 多 Agent contract 分支必需 Finding 覆盖|report:v1-02c2-resolution-120x2/quality.multi_agent.required_finding_coverage_by_worker.contract.numerator|90|0
V2-PROCUREMENT-COVERAGE|C2 多 Agent procurement 分支必需 Finding 覆盖|report:v1-02c2-resolution-120x2/quality.multi_agent.required_finding_coverage_by_worker.procurement_data.numerator|28|0
V2-RETRIEVAL-RAW|C2 v2 dev 检索 Recall@5 原始 query 命中|report:v1-02c2-resolution-120x2/retrieval.dev.variants.raw_query.recall_at_5.numerator|438|0
V2-RETRIEVAL-RAW-RATE|C2 v2 dev 检索 Recall@5 原始 query|report:v1-02c2-resolution-120x2/retrieval.dev.variants.raw_query.recall_at_5.value|0.9954545454545455|5e-05
V2-RETRIEVAL-MASKED|C2 v2 dev 检索 Recall@5 去 Case 号命中|report:v1-02c2-resolution-120x2/retrieval.dev.variants.masked_case_number_query.recall_at_5.numerator|432|0
V2-RETRIEVAL-MASKED-RATE|C2 v2 dev 检索 Recall@5 去 Case 号|report:v1-02c2-resolution-120x2/retrieval.dev.variants.masked_case_number_query.recall_at_5.value|0.9818181818181818|5e-05
V2-MRR-RAW|C2 v2 检索 MRR（原始 query）|report:v1-02c2-resolution-120x2/retrieval.dev.variants.raw_query.mrr|0.9977272727272727|5e-05
V2-MRR-MASKED|C2 v2 检索 MRR（去 Case 号）|report:v1-02c2-resolution-120x2/retrieval.dev.variants.masked_case_number_query.mrr|0.9954545454545455|5e-05
V2-INDEXED-DOCS|C2 v2 dev 索引文档数|report:v1-02c2-resolution-120x2/retrieval.dev.indexed_document_count|440|0
V2-DELTA-COVERAGE|C2 必需 Finding 覆盖配对差（单 − 多）|report:v1-02c2-resolution-120x2/paired_deltas[metric=required_finding_coverage].delta|-0.21388888888888888|0
V2-DELTA-COVERAGE-CI-LOW|C2 必需 Finding 覆盖配对差区间下界|report:v1-02c2-resolution-120x2/paired_deltas[metric=required_finding_coverage].ci_low|-0.25277777777777777|0
V2-DELTA-COVERAGE-CI-HIGH|C2 必需 Finding 覆盖配对差区间上界|report:v1-02c2-resolution-120x2/paired_deltas[metric=required_finding_coverage].ci_high|-0.17638888888888887|0
V2-DELTA-CITATION|C2 引用支持率配对差（单 − 多）|report:v1-02c2-resolution-120x2/paired_deltas[metric=citation_support_precision].delta|-0.7791666666666667|0
V2-DELTA-ENDTOEND|C2 端到端建议完全匹配配对差|report:v1-02c2-resolution-120x2/paired_deltas[metric=recommendation_exact].delta|0|0
V2-DELTA-ENDTOEND-CI|C2 端到端建议完全匹配区间上界|report:v1-02c2-resolution-120x2/paired_deltas[metric=recommendation_exact].ci_high|0|0
V2-DELTA-CASES|C2 配对自助法有效样本（Case 数）|report:v1-02c2-resolution-120x2/paired_deltas[metric=recommendation_exact].case_count|120|0
V2-CALLS-MULTI|C2 多 Agent 模型调用数|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.model_call_count|660|0
V2-CALLS-SINGLE|C2 单 Agent 模型调用数|report:v1-02c2-resolution-120x2/cost_and_reliability.single_agent.model_call_count|240|0
V2-TOKENS-MULTI|C2 多 Agent 总 Token|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.total_tokens|2389472|0
V2-TOKENS-SINGLE|C2 单 Agent 总 Token|report:v1-02c2-resolution-120x2/cost_and_reliability.single_agent.total_tokens|755390|0
V2-USAGE-BEARING-MULTI|C2 多 Agent 产生用量的调用数|segments:v1-02c2-resolution-120x2.multi_agent.usage_bearing_calls|548|0
V2-USAGE-BEARING-SINGLE|C2 单 Agent 产生用量的调用数|segments:v1-02c2-resolution-120x2.single_agent.usage_bearing_calls|98|0
V2-TIMEOUT-SINGLE|C2 单 Agent 超时分支数|segments:v1-02c2-resolution-120x2.single_agent.error.timeout|142|0
V2-TIMEOUT-MULTI|C2 多 Agent 超时分支数|segments:v1-02c2-resolution-120x2.multi_agent.error.timeout|112|0
V2-CROSS-CONTEXT-SINGLE-A|C2 单 Agent 引用业务事实越界数|segments:v1-02c2-resolution-120x2.single_agent.detail.document_worker_cited_business_fact|72|0
V2-CROSS-CONTEXT-SINGLE-B|C2 单 Agent 采购分支引用文档越界数|segments:v1-02c2-resolution-120x2.single_agent.detail.procurement_worker_cited_document|10|0
V2-CROSS-CONTEXT-MULTI|C2 多 Agent 跨上下文引用越界数（按构造为 0）|segments:v1-02c2-resolution-120x2.multi_agent.detail.document_worker_cited_business_fact|0|0
V2-P50-MULTI|C2 多 Agent p50 延迟|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.latency_ms.p50|47261|0
V2-P50-SINGLE|C2 单 Agent p50 延迟（恰为超时上限）|report:v1-02c2-resolution-120x2/cost_and_reliability.single_agent.latency_ms.p50|90003|0
V2-P95-MULTI|C2 多 Agent p95 延迟|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.latency_ms.p95|90020|0
V2-P95-SINGLE|C2 单 Agent p95 延迟|report:v1-02c2-resolution-120x2/cost_and_reliability.single_agent.latency_ms.p95|90118|0
V2-RETRIES|C2 真实端点重试事件数|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.retry_total|5|0
V2-COMPLETENESS-MULTI|C2 多 Agent 运行记录完整分支数|report:v1-02c2-resolution-120x2/cost_and_reliability.multi_agent.run_record_completeness.numerator|720|0
SLICE-V1-PROCUREMENT|v1 dev120×2 多 Agent procurement 分支覆盖|slice:v1-02c-paired-360x3/multi_agent/procurement_data|0.5291666666666667|5e-05
SLICE-V2-PROCUREMENT|v2 dev120×2 多 Agent procurement 分支覆盖|slice:v1-02c2-resolution-120x2/multi_agent/procurement_data|0.11666666666666667|5e-05
SLICE-V1-POLICY|v1 dev120×2 多 Agent policy 分支覆盖|slice:v1-02c-paired-360x3/multi_agent/policy|0|5e-05
SLICE-V1-CONTRACT|v1 dev120×2 多 Agent contract 分支覆盖|slice:v1-02c-paired-360x3/multi_agent/contract|0.008333333333333333|5e-05
BRAKE-REQUESTS|清单口径累计请求（含 V1-05B）|brake:manifest_requests|6822|0
BRAKE-INPUT|清单口径累计 input tokens（含 V1-05B）|brake:manifest_input_tokens|5388459|0
BRAKE-OUTPUT|清单口径累计 output tokens（含 V1-05B）|brake:manifest_output_tokens|14563168|0
BRAKE-REQUEST-SHARE|累计请求占比|brake:request_share|0.27288|1e-05
BRAKE-INPUT-SHARE|累计 input 占比|brake:input_share|0.021554|1e-05
BRAKE-OUTPUT-SHARE|累计 output 占比|brake:output_share|0.606799|1e-05
BRAKE-UNACCOUNTED-LOW|暂停期未入账调用下界|brake:unaccounted_requests_low|125|0
BRAKE-UNACCOUNTED-HIGH|暂停期未入账调用上界|brake:unaccounted_requests_high|135|0
BRAKE-ADJUSTED-REQUEST-SHARE|含未入账消耗的累计请求占比|brake:adjusted_request_share|0.27828|1e-05
BRAKE-ADJUSTED-OUTPUT-SHARE|含未入账消耗的累计 output 占比|brake:adjusted_output_share|0.621789|1e-05
BRAKE-USAGE-BEARING|产生用量的真实请求数|brake:usage_bearing_requests|6294|0
BRAKE-WARNING-REACHED|累计刹车是否触及 80% 预警（False 即未触及）|brake:warning_reached|False|0
LEDGER-EVAL-SEGMENTS|真实 eval 分段清单数（含 V1-05B 的 4 段）|brake:eval_segments|16|0
LEDGER-SMOKE-SEGMENTS|真实 Smoke 清单数|brake:smoke_segments|3|0
LEDGER-DRY-RUN-REQUESTS|干跑清单的 Mock 调用数（不计入刹车）|brake:dry_run_requests|792|0
RESCORE-ANCHOR-C2|C-2 rescore 锚定的历史报告摘要|rescore:v1-02c-paired-360x3/v2/rescore.frozen_report_sha256|5d67be8dda37a3a4117cec256c4a2c6f57abc2821080561ad62194d6458b6c2f|0
RESCORE-ANCHOR-CC|C2 rescore 锚定的历史报告摘要|rescore:v1-02c2-resolution-120x2/v2/rescore.frozen_report_sha256|d3983b566a4c176f809f03facb5e1fe02bb4e6adb3b30167023429a40ac50243|0
RESCORE-ANCHOR-C1|C-1 rescore 锚定的历史报告摘要|rescore:v1-02c-calibration-48/v2/rescore.frozen_report_sha256|83223382b337718d6e4d6338dcc26d248d23ea01c7f721535f3fc81ad9785ab6|0
RESCORE-MODEL-INPUT|M2 未改变模型可见输入|rescore:v1-02c-paired-360x3/v2/rescore.model_input_changed|False|0
RESCORE-HOLDOUT-READ|离线重评未读取 holdout|rescore:v1-02c-paired-360x3/v2/rescore.holdout_read|False|0
RESCORE-RULES2-DIGEST|rules v2 资源文件摘要|rescore:v1-02c-paired-360x3/v2/rescore.rules_resource_sha256|39c321cfefad6d70e64f910c7bca16b90f6a9ace5be0e816fcd2e091fdb373f9|0
RESCORE-C2-ACCURACY|C-2 rules2 端到端建议准确率|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.recommendation_accuracy.value|0.43148148148148147|5e-05
RESCORE-C2-ACCURACY-NUM|C-2 rules2 端到端建议正确数|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.recommendation_accuracy.numerator|466|0
RESCORE-C2-ACCURACY-R1-NUM|C-2 rules1 端到端建议正确数（对照）|rescore:v1-02c-paired-360x3/v1/quality.multi_agent.recommendation_accuracy.numerator|450|0
RESCORE-C2-CONFLICT-NUM|C-2 rules2 冲突识别命中数|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.conflict_recall.numerator|2|0
RESCORE-C2-CONFLICT-DEN|C-2 应识别冲突观测数（分母不变）|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.conflict_recall.denominator|270|0
RESCORE-C2-UNKNOWN-NUM|C-2 rules2 UNKNOWN 召回命中数|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.unknown_recall.numerator|699|0
RESCORE-C2-TAKEOVER-NUM|C-2 rules2 人工接管观测数|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.human_takeover_rate.numerator|1036|0
RESCORE-C2-TAKEOVER-R1-NUM|C-2 rules1 人工接管观测数（全部升级）|rescore:v1-02c-paired-360x3/v1/quality.multi_agent.human_takeover_rate.numerator|1080|0
RESCORE-C2-HIGHRISK-NUM|C-2 rules2 高风险漏报数（不回退）|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.high_risk_miss_rate.numerator|0|0
RESCORE-C2-CITATION-NUM|C-2 rules2 被支持引用数（计分未变）|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.citation_support_precision.numerator|2510|0
RESCORE-C2-COVERAGE-NUM|C-2 rules2 必需 Finding 覆盖数（计分未变）|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.required_finding_coverage.numerator|559|0
RESCORE-C2-APPROVE-F1|C-2 rules2 recommend_approve 类 F1（真实输出未触发）|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.recommendation_macro_f1.per_class.recommend_approve|0|0
RESCORE-C2-DELTA-ENDTOEND|C-2 rules2 端到端建议配对差（单 − 多）|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=recommendation_exact].delta|-0.014814814814814812|5e-05
RESCORE-C2-DELTA-ENDTOEND-CI-LOW|C-2 rules2 端到端配对差区间下界|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=recommendation_exact].ci_low|-0.025|5e-05
RESCORE-C2-DELTA-ENDTOEND-CI-HIGH|C-2 rules2 端到端配对差区间上界|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=recommendation_exact].ci_high|-0.004629629629629628|5e-05
RESCORE-C2-DELTA-ENDTOEND-R1|C-2 rules1 端到端配对差（退化点）|rescore:v1-02c-paired-360x3/v1/paired_deltas[metric=recommendation_exact].delta|0|0
RESCORE-C2-DELTA-ENDTOEND-R1-CI|C-2 rules1 端到端配对差区间上界（退化点）|rescore:v1-02c-paired-360x3/v1/paired_deltas[metric=recommendation_exact].ci_high|0|0
RESCORE-C2-DELTA-ESCALATED|C-2 rules2 升级人工配对差|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=escalated_to_human].delta|0.040740740740740744|5e-05
RESCORE-C2-DELTA-CONFLICT|C-2 rules2 冲突检出配对差|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=conflict_detected].delta|-0.0037037037037037034|5e-05
RESCORE-C2-DELTA-HIGHRISK|C-2 rules2 高风险漏报配对差（两侧真零）|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=high_risk_missed].delta|0|0
RESCORE-C2-CHANGED-METRICS|C-2 rules2 相对 rules1 变化的指标数（多 Agent）|rescore:v1-02c-paired-360x3/v2/contract_delta.multi_agent.changed_metric_count|4|0
RESCORE-C2-CHANGED-METRICS-SINGLE|C-2 rules2 相对 rules1 变化的指标数（单 Agent）|rescore:v1-02c-paired-360x3/v2/contract_delta.single_agent.changed_metric_count|0|0
RESCORE-C2-R1-CONFUSION-CELLS|C-2 rules1 混淆矩阵格子数（单列退化）|rescore:v1-02c-paired-360x3/v1/quality.multi_agent.confusion_matrix.insufficient_information->insufficient_information|450|0
RESCORE-C2-D1-HEALTHY|C-2 rules2 三分支全完成观测数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d1_healthy_subset.healthy_observations|114|0
RESCORE-C2-D1-OBS|C-2 多 Agent 观测总数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d1_healthy_subset.observations|1080|0
RESCORE-C2-D1-DISTINCT|C-2 rules2 健康子集不同建议类数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d1_healthy_subset.distinct_recommendations|3|0
RESCORE-C2-D1-DISTINCT-R1|C-2 rules1 健康子集不同建议类数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_1.d1_healthy_subset.distinct_recommendations|1|0
RESCORE-C2-D1-REJECT|C-2 rules2 健康子集 recommend_reject 数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d1_healthy_subset.recommendation_counts.recommend_reject|44|0
RESCORE-C2-D1-HUMAN|C-2 rules2 健康子集 needs_human_review 数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d1_healthy_subset.recommendation_counts.needs_human_review|1|0
RESCORE-C2-D2-DETECTED|C-2 rules2 冲突标注观测中检出数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d2_conflicts.observations_with_detected_conflict|2|0
RESCORE-C2-D2-FALSE-POSITIVE|C-2 rules2 无冲突标注却检出数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d2_conflicts.detections_without_expected_conflicts|2|0
RESCORE-C2-D2-BOTH-PRESENT|C-2 rules2 未检出但对侧断言在场观测数（M1 上界）|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d2_conflicts.undetected_with_both_assertions_present|2|0
RESCORE-C2-D3-CONFLICT|C-2 rules2 升级类别 conflict 观测数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.conflict|4|0
RESCORE-C2-D3-CONTENT|C-2 rules2 结局类别 content 观测数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.content|44|0
RESCORE-C2-D3-EVIDENCE|C-2 rules2 升级类别 evidence_insufficient 观测数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.evidence_insufficient|69|0
RESCORE-C2-D3-PIPELINE|C-2 rules2 升级类别 pipeline_incomplete 观测数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.pipeline_incomplete|963|0
RESCORE-C2-D3-R1-INSEPARABLE|C-2 rules1 结局不可分类观测数|rescore:v1-02c-paired-360x3/v2/decidability.multi_agent.rules_1.d3_escalation.by_class.not_directional|1080|0
RESCORE-C2-FR-CONFLICT-HEALTHY|C-2 rules2 fact_conflict 族健康观测数|rescore:v1-02c-paired-360x3/v2/family_mapping.multi_agent.rules_2.GF-fact_conflict.healthy_observations|14|0
RESCORE-C2-FR-CONFLICT-DETS|C-2 rules2 fact_conflict 族冲突检出观测数|rescore:v1-02c-paired-360x3/v2/family_mapping.multi_agent.rules_2.GF-fact_conflict.conflict_detections|2|0
RESCORE-C2-FR-CONFLICT-DETS-R1|C-2 rules1 fact_conflict 族冲突检出观测数|rescore:v1-02c-paired-360x3/v2/family_mapping.multi_agent.rules_1.GF-fact_conflict.conflict_detections|0|0
RESCORE-C2-FR-MISSING-MODE|C-2 rules2 missing_required_fields 健康众数|rescore:v1-02c-paired-360x3/v2/family_mapping.multi_agent.rules_2.GF-missing_required_fields.healthy_mode|recommend_reject|0
RESCORE-C2-FR-MISSING-MODE-R1|C-2 rules1 missing_required_fields 健康众数|rescore:v1-02c-paired-360x3/v2/family_mapping.multi_agent.rules_1.GF-missing_required_fields.healthy_mode|insufficient_information|0
RESCORE-C2-FR-HIGHRISK-REJECT|C-2 rules2 supported_high_risk 健康 reject 数|rescore:v1-02c-paired-360x3/v2/family_mapping.multi_agent.rules_2.GF-supported_high_risk.healthy_counts.recommend_reject|14|0
RESCORE-C2-FR-LOWRISK-MODE|C-2 rules2 low_risk_complete 健康众数（保守闸门）|rescore:v1-02c-paired-360x3/v2/family_mapping.multi_agent.rules_2.GF-low_risk_complete.healthy_mode|insufficient_information|0
RESCORE-CC-D1-HEALTHY|C2 rules2 三分支全完成观测数|rescore:v1-02c2-resolution-120x2/v2/decidability.multi_agent.rules_2.d1_healthy_subset.healthy_observations|27|0
RESCORE-CC-D1-DISTINCT|C2 rules2 健康子集不同建议类数|rescore:v1-02c2-resolution-120x2/v2/decidability.multi_agent.rules_2.d1_healthy_subset.distinct_recommendations|2|0
RESCORE-CC-D2-DETECTED|C2 rules2 冲突标注观测中检出数|rescore:v1-02c2-resolution-120x2/v2/decidability.multi_agent.rules_2.d2_conflicts.observations_with_detected_conflict|0|0
RESCORE-CC-D2-BOTH-PRESENT|C2 rules2 未检出但对侧断言在场观测数（M1 上界为 0）|rescore:v1-02c2-resolution-120x2/v2/decidability.multi_agent.rules_2.d2_conflicts.undetected_with_both_assertions_present|0|0
RESCORE-CC-D3-CONTENT|C2 rules2 结局类别 content 观测数|rescore:v1-02c2-resolution-120x2/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.content|9|0
RESCORE-CC-D3-PIPELINE|C2 rules2 升级类别 pipeline_incomplete 观测数|rescore:v1-02c2-resolution-120x2/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.pipeline_incomplete|213|0
RESCORE-CC-UNKNOWN-NUM|C2 rules2 UNKNOWN 召回命中数|rescore:v1-02c2-resolution-120x2/v2/quality.multi_agent.unknown_recall.numerator|154|0
RESCORE-CC-TAKEOVER-NUM|C2 rules2 人工接管观测数|rescore:v1-02c2-resolution-120x2/v2/quality.multi_agent.human_takeover_rate.numerator|231|0
RESCORE-CC-ACCURACY-NUM|C2 rules2 端到端建议正确数（不变）|rescore:v1-02c2-resolution-120x2/v2/quality.multi_agent.recommendation_accuracy.numerator|100|0
RESCORE-CC-DELTA-CI-LOW|C2 rules2 端到端配对差区间下界（非退化）|rescore:v1-02c2-resolution-120x2/v2/paired_deltas[metric=recommendation_exact].ci_low|-0.020833333333333332|5e-05
RESCORE-CC-DELTA-CI-HIGH-R1|C2 rules1 端到端配对差区间上界（退化点）|rescore:v1-02c2-resolution-120x2/v1/paired_deltas[metric=recommendation_exact].ci_high|0|0
RESCORE-C1-D2-DETECTED|C-1 rules2 冲突标注观测中检出数|rescore:v1-02c-calibration-48/v2/decidability.multi_agent.rules_2.d2_conflicts.observations_with_detected_conflict|1|0
RESCORE-C1-D2-DEN|C-1 应识别冲突观测数|rescore:v1-02c-calibration-48/v2/decidability.multi_agent.rules_2.d2_conflicts.observations_with_expected_conflicts|12|0
HB-ACCURACY-R2-NUM|holdout rules2 多 Agent recommendation_accuracy 分子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.recommendation_accuracy.numerator|96|0
HB-ACCURACY-R2-DEN|holdout rules2 多 Agent recommendation_accuracy 分母|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.recommendation_accuracy.denominator|240|0
HB-UNKNOWN-R2-NUM|holdout rules2 多 Agent unknown_recall 分子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.unknown_recall.numerator|153|0
HB-UNKNOWN-R2-DEN|holdout rules2 多 Agent unknown_recall 分母|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.unknown_recall.denominator|160|0
HB-CONFLICT-R2-NUM|holdout rules2 多 Agent conflict_recall 分子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.conflict_recall.numerator|0|0
HB-CONFLICT-R2-DEN|holdout rules2 多 Agent conflict_recall 分母|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.conflict_recall.denominator|60|0
HB-HIGHRISK-R2-NUM|holdout rules2 多 Agent high_risk_miss_rate 分子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.high_risk_miss_rate.numerator|0|0
HB-HIGHRISK-R2-DEN|holdout rules2 多 Agent high_risk_miss_rate 分母|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.high_risk_miss_rate.denominator|120|0
HB-CITATION-R2-NUM|holdout rules2 多 Agent citation_support_precision 分子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.citation_support_precision.numerator|515|0
HB-CITATION-R2-DEN|holdout rules2 多 Agent citation_support_precision 分母|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.citation_support_precision.denominator|515|0
HB-COVERAGE-R2-NUM|holdout rules2 多 Agent required_finding_coverage 分子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.required_finding_coverage.numerator|164|0
HB-COVERAGE-R2-DEN|holdout rules2 多 Agent required_finding_coverage 分母|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.required_finding_coverage.denominator|720|0
HB-TAKEOVER-R2-NUM|holdout rules2 多 Agent human_takeover_rate 分子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.human_takeover_rate.numerator|231|0
HB-TAKEOVER-R2-DEN|holdout rules2 多 Agent human_takeover_rate 分母|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.human_takeover_rate.denominator|240|0
HB-ACCURACY-R2|holdout rules2 端到端建议准确率|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.recommendation_accuracy.value|0.4|5e-05
HB-ACCURACY-R1|holdout rules1 端到端建议准确率（对照口径）|rescore:v1-05b-holdout-120x2/v1/quality.multi_agent.recommendation_accuracy.value|0.4166666666666667|5e-05
HB-MACROF1-R2|holdout rules2 建议宏平均 F1|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.recommendation_macro_f1.value|0.1564867113271159|5e-05
HB-MACROF1-R2-APPROVE|holdout rules2 recommend_approve 类 F1（样本外仍未触发）|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.recommendation_macro_f1.per_class.recommend_approve|0.0|5e-05
HB-VALID-R2|holdout rules2 多 Agent 结构化输出有效率|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.multi_agent.structured_output_valid_rate.value|0.4575757575757576|5e-05
HB-VALID-SINGLE-R2|holdout rules2 单 Agent 结构化输出有效数|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.single_agent.structured_output_valid_rate.numerator|0|0
HB-CONFUSION-REJECT-CELL|holdout rules2 混淆矩阵 reject→reject 格子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.confusion_matrix.recommend_reject->recommend_reject|2|0
HB-CONFUSION-DRIFT-CELL|holdout rules2 混淆矩阵 标签信息不足→reject 格子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.confusion_matrix.insufficient_information->recommend_reject|6|0
HB-CONFUSION-CELLS-R1|holdout rules1 混淆矩阵单列退化格子的观测数|rescore:v1-05b-holdout-120x2/v1/quality.multi_agent.confusion_matrix.insufficient_information->insufficient_information|100|0
HB-DELTA-ENDTOEND-R2|holdout rules2 端到端建议配对差（单 − 多）|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=recommendation_exact].delta|0.016666666666666666|5e-05
HB-DELTA-ENDTOEND-R2-CI-LOW|holdout rules2 端到端配对差区间下界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=recommendation_exact].ci_low|-0.008333333333333333|5e-05
HB-DELTA-ENDTOEND-R2-CI-HIGH|holdout rules2 端到端配对差区间上界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=recommendation_exact].ci_high|0.041666666666666664|5e-05
HB-DELTA-ENDTOEND-R1|holdout rules1 端到端配对差（退化点）|report:v1-05b-holdout-120x2/paired_deltas[metric=recommendation_exact].delta|0.0|5e-05
HB-DELTA-ENDTOEND-R1-CI-HIGH|holdout rules1 端到端配对差区间上界（退化点）|report:v1-05b-holdout-120x2/paired_deltas[metric=recommendation_exact].ci_high|0.0|5e-05
HB-DELTA-ESCALATED-R2|holdout rules2 升级人工配对差|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=escalated_to_human].delta|0.0375|5e-05
HB-DELTA-ESCALATED-R2-CI-LOW|holdout rules2 升级人工配对差区间下界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=escalated_to_human].ci_low|0.0125|5e-05
HB-DELTA-ESCALATED-R2-CI-HIGH|holdout rules2 升级人工配对差区间上界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=escalated_to_human].ci_high|0.06666666666666667|5e-05
HB-DELTA-ESCALATED-R1|holdout rules1 升级人工配对差（退化点）|report:v1-05b-holdout-120x2/paired_deltas[metric=escalated_to_human].delta|0.0|5e-05
HB-DELTA-ESCALATED-R1-CI-HIGH|holdout rules1 升级人工配对差区间上界（退化点）|report:v1-05b-holdout-120x2/paired_deltas[metric=escalated_to_human].ci_high|0.0|5e-05
HB-DELTA-COVERAGE-R2|holdout rules2 必需 Finding 覆盖配对差|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=required_finding_coverage].delta|-0.22777777777777777|5e-05
HB-DELTA-COVERAGE-R2-CI-LOW|holdout rules2 必需 Finding 覆盖配对差区间下界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=required_finding_coverage].ci_low|-0.26666666666666666|5e-05
HB-DELTA-COVERAGE-R2-CI-HIGH|holdout rules2 必需 Finding 覆盖配对差区间上界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=required_finding_coverage].ci_high|-0.19027777777777777|5e-05
HB-DELTA-COVERAGE-R1|holdout rules1 必需 Finding 覆盖配对差（两合同同值）|report:v1-05b-holdout-120x2/paired_deltas[metric=required_finding_coverage].delta|-0.22777777777777777|5e-05
HB-DELTA-CITATION-R2|holdout rules2 引用支持率配对差|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=citation_support_precision].delta|-0.7958333333333333|5e-05
HB-DELTA-CITATION-R2-CI-LOW|holdout rules2 引用支持率配对差区间下界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=citation_support_precision].ci_low|-0.8458333333333333|5e-05
HB-DELTA-CITATION-R1|holdout rules1 引用支持率配对差（两合同同值）|report:v1-05b-holdout-120x2/paired_deltas[metric=citation_support_precision].delta|-0.7958333333333333|5e-05
HB-DELTA-HIGHRISK-R2|holdout rules2 高风险漏报配对差（两侧真零）|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=high_risk_missed].delta|0.0|5e-05
HB-DELTA-HIGHRISK-R2-CI-HIGH|holdout rules2 高风险漏报配对差区间上界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=high_risk_missed].ci_high|0.0|5e-05
HB-DELTA-CONFLICT-R2|holdout rules2 冲突检出配对差|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=conflict_detected].delta|0.0|5e-05
HB-DELTA-CONFLICT-R2-CI-LOW|holdout rules2 冲突检出配对差区间下界|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=conflict_detected].ci_low|0.0|5e-05
HB-DELTA-CASES|holdout 配对自助法有效样本（Case 数）|rescore:v1-05b-holdout-120x2/v2/paired_deltas[metric=recommendation_exact].case_count|120|0
HB-D1-HEALTHY|holdout rules2 三分支全完成观测数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d1_healthy_subset.healthy_observations|19|0
HB-D1-OBS|holdout 多 Agent 观测总数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d1_healthy_subset.observations|240|0
HB-D1-DISTINCT|holdout rules2 健康子集不同建议类数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d1_healthy_subset.distinct_recommendations|2|0
HB-D1-DISTINCT-R1|holdout rules1 健康子集不同建议类数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_1.d1_healthy_subset.distinct_recommendations|1|0
HB-D1-REJECT|holdout rules2 健康子集 recommend_reject 数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d1_healthy_subset.recommendation_counts.recommend_reject|9|0
HB-D2-DETECTED|holdout rules2 冲突标注观测中检出数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d2_conflicts.observations_with_detected_conflict|0|0
HB-D2-DEN|holdout 应识别冲突观测数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d2_conflicts.observations_with_expected_conflicts|60|0
HB-D2-BOTH-PRESENT|holdout rules2 未检出但对侧断言在场观测数（M1 上界）|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d2_conflicts.undetected_with_both_assertions_present|0|0
HB-D2-FALSE-POSITIVE|holdout rules2 无冲突标注却检出数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d2_conflicts.detections_without_expected_conflicts|0|0
HB-D3-CONTENT|holdout rules2 结局类别 content 观测数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.content|9|0
HB-D3-EVIDENCE|holdout rules2 升级类别 evidence_insufficient 观测数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.evidence_insufficient|10|0
HB-D3-PIPELINE|holdout rules2 升级类别 pipeline_incomplete 观测数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.pipeline_incomplete|221|0
HB-D3-R1-INSEPARABLE|holdout rules1 结局不可分类观测数|rescore:v1-05b-holdout-120x2/v2/decidability.multi_agent.rules_1.d3_escalation.by_class.not_directional|240|0
HB-CHANGED-METRICS|holdout rules2 相对 rules1 变化的指标数（多 Agent）|rescore:v1-05b-holdout-120x2/v2/contract_delta.multi_agent.changed_metric_count|3|0
HB-CHANGED-METRICS-SINGLE|holdout rules2 相对 rules1 变化的指标数（单 Agent）|rescore:v1-05b-holdout-120x2/v2/contract_delta.single_agent.changed_metric_count|0|0
HB-FR-MISSING-MODE|holdout rules2 missing_required_fields 健康众数|rescore:v1-05b-holdout-120x2/v2/family_mapping.multi_agent.rules_2.GF-missing_required_fields.healthy_mode|recommend_reject|0
HB-FR-LOWRISK-MODE|holdout rules2 low_risk_complete 健康众数（保守闸门）|rescore:v1-05b-holdout-120x2/v2/family_mapping.multi_agent.rules_2.GF-low_risk_complete.healthy_mode|insufficient_information|0
HB-FR-CONFLICT-HEALTHY|holdout rules2 fact_conflict 族健康观测数|rescore:v1-05b-holdout-120x2/v2/family_mapping.multi_agent.rules_2.GF-fact_conflict.healthy_observations|1|0
HB-FR-CONFLICT-DETS|holdout rules2 fact_conflict 族冲突检出观测数|rescore:v1-05b-holdout-120x2/v2/family_mapping.multi_agent.rules_2.GF-fact_conflict.conflict_detections|0|0
HB-HOLDOUT-READ|本批次确已读取 holdout 分区|rescore:v1-05b-holdout-120x2/v2/rescore.holdout_read|True|0
HB-IN-SAMPLE|本批次属样本外确认（in_sample=False）|rescore:v1-05b-holdout-120x2/v2/rescore.in_sample|False|0
HB-MODEL-INPUT|holdout 批次模型可见输入未变|rescore:v1-05b-holdout-120x2/v2/rescore.model_input_changed|False|0
HB-SPEC-CASE-COUNT|holdout 批次冻结 Case 数|report:v1-05b-holdout-120x2/batch.case_count|120|0
HB-SPEC-CONCURRENCY|holdout 批次冻结并发数|report:v1-05b-holdout-120x2/batch.concurrency|4|0
HB-SPEC-MAX-REQUESTS|holdout 批次冻结请求硬上限|report:v1-05b-holdout-120x2/batch.max_requests|1200|0
HB-RETRIEVAL-RAW|holdout v2 检索 Recall@5 原始 query 命中|report:v1-05b-holdout-120x2/retrieval.holdout.variants.raw_query.recall_at_5.numerator|220|0
HB-RETRIEVAL-RAW-DEN|holdout v2 检索 Gold 必需证据数|report:v1-05b-holdout-120x2/retrieval.holdout.variants.raw_query.recall_at_5.denominator|220|0
HB-RETRIEVAL-MASKED|holdout v2 检索 Recall@5 去 Case 号命中|report:v1-05b-holdout-120x2/retrieval.holdout.variants.masked_case_number_query.recall_at_5.numerator|219|0
HB-INDEXED-DOCS|holdout v2 索引文档数|report:v1-05b-holdout-120x2/retrieval.holdout.indexed_document_count|220|0
HB-DUPLICATE-DEN|holdout 重复查询率分母|report:v1-05b-holdout-120x2/retrieval.holdout.duplicate_query_rate.denominator|110|0
HB-CALLS-MULTI|holdout 多 Agent 模型调用数|segments:v1-05b-holdout-120x2.multi_agent.calls|720|0
HB-SUCCEEDED-MULTI|holdout 多 Agent 成功分支数|segments:v1-05b-holdout-120x2.multi_agent.succeeded|302|0
HB-FAILED-MULTI|holdout 多 Agent 失败分支数|segments:v1-05b-holdout-120x2.multi_agent.failed|358|0
HB-NOTREQUESTED-MULTI|holdout 多 Agent 本地跳过分支数|segments:v1-05b-holdout-120x2.multi_agent.not_requested|60|0
HB-USAGE-BEARING-MULTI|holdout 多 Agent 产生用量的调用数|segments:v1-05b-holdout-120x2.multi_agent.usage_bearing_calls|565|0
HB-DRIFT-MULTI|holdout 多 Agent 模型漂移分支数（应为 0）|segments:v1-05b-holdout-120x2.multi_agent.error.model_id_drift|0|0
HB-RATE-LIMITED-MULTI|holdout 多 Agent 终态限流分支数|segments:v1-05b-holdout-120x2.multi_agent.error.rate_limited|4|0
HB-TIMEOUT-MULTI|holdout 多 Agent 超时分支数|segments:v1-05b-holdout-120x2.multi_agent.error.timeout|91|0
HB-SCHEMA-MULTI|holdout 多 Agent 输出 Schema 失败分支数|segments:v1-05b-holdout-120x2.multi_agent.error.output_schema|237|0
HB-SECURITY-MULTI|holdout 多 Agent 安全边界失败分支数|segments:v1-05b-holdout-120x2.multi_agent.error.security_boundary|14|0
HB-CROSS-CONTEXT-MULTI|holdout 多 Agent 跨上下文引用越界数（按构造为 0）|segments:v1-05b-holdout-120x2.multi_agent.detail.document_worker_cited_business_fact|0|0
HB-CALLS-SINGLE|holdout 单 Agent 模型调用数|segments:v1-05b-holdout-120x2.single_agent.calls|240|0
HB-SUCCEEDED-SINGLE|holdout 单 Agent 成功分支数|segments:v1-05b-holdout-120x2.single_agent.succeeded|0|0
HB-FAILED-SINGLE|holdout 单 Agent 失败分支数|segments:v1-05b-holdout-120x2.single_agent.failed|240|0
HB-NOTREQUESTED-SINGLE|holdout 单 Agent 本地跳过分支数|segments:v1-05b-holdout-120x2.single_agent.not_requested|0|0
HB-USAGE-BEARING-SINGLE|holdout 单 Agent 产生用量的调用数|segments:v1-05b-holdout-120x2.single_agent.usage_bearing_calls|106|0
HB-DRIFT-SINGLE|holdout 单 Agent 模型漂移分支数（应为 0）|segments:v1-05b-holdout-120x2.single_agent.error.model_id_drift|0|0
HB-RATE-LIMITED-SINGLE|holdout 单 Agent 终态限流分支数|segments:v1-05b-holdout-120x2.single_agent.error.rate_limited|9|0
HB-TIMEOUT-SINGLE|holdout 单 Agent 超时分支数|segments:v1-05b-holdout-120x2.single_agent.error.timeout|125|0
HB-SCHEMA-SINGLE|holdout 单 Agent 输出 Schema 失败分支数|segments:v1-05b-holdout-120x2.single_agent.error.output_schema|106|0
HB-SECURITY-SINGLE|holdout 单 Agent 安全边界失败分支数|segments:v1-05b-holdout-120x2.single_agent.error.security_boundary|0|0
HB-CROSS-CONTEXT-SINGLE|holdout 单 Agent 引用业务事实越界数|segments:v1-05b-holdout-120x2.single_agent.detail.document_worker_cited_business_fact|82|0
HB-CROSS-CONTEXT-SINGLE-B|holdout 单 Agent 采购分支引用文档越界数|segments:v1-05b-holdout-120x2.single_agent.detail.procurement_worker_cited_document|9|0
HB-TOKENS-MULTI|holdout 多 Agent 总 Token（主合同重算）|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.multi_agent.total_tokens|2507592|0
HB-TOKENS-SINGLE|holdout 单 Agent 总 Token（主合同重算）|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.single_agent.total_tokens|840235|0
HB-LATENCY-P95-MULTI|holdout 多 Agent p95 延迟（被 90 秒上限截断）|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.multi_agent.latency_ms.p95|90004|0
HB-LATENCY-P95-SINGLE|holdout 单 Agent p95 延迟|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.single_agent.latency_ms.p95|109047|0
HB-LATENCY-P50-MULTI|holdout 多 Agent p50 延迟（有判别力的延迟口径）|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.multi_agent.latency_ms.p50|39161|0
HB-LATENCY-P50-SINGLE|holdout 单 Agent p50 延迟（中位数即一次超时）|rescore:v1-05b-holdout-120x2/v2/cost_and_reliability.single_agent.latency_ms.p50|90003|0
HB-CITATION-OUTSIDE-MULTI|holdout 多 Agent 越出允许上下文的引用数|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.citation_outside_allowed_context.numerator|0|0
HB-UNSUPPORTED-CITATION-MULTI|holdout 多 Agent 无依据引用数|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.unsupported_citation_rate.numerator|0|0
HB-COVERAGE-RATE|holdout rules2 多 Agent 必需 Finding 覆盖率|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.required_finding_coverage.value|0.22777777777777777|5e-05
HB-CONFUSION-APPROVE-DRIFT|holdout rules2 混淆矩阵 批准标签→信息不足 格子|rescore:v1-05b-holdout-120x2/v2/quality.multi_agent.confusion_matrix.recommend_approve->insufficient_information|20|0
RESCORE-C2-DELTA-CASES|C-2 rules2 配对自助法有效样本（Case 数）|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=recommendation_exact].case_count|360|0
RESCORE-CC-DELTA-CASES|C2 rules2 配对自助法有效样本（Case 数）|rescore:v1-02c2-resolution-120x2/v2/paired_deltas[metric=recommendation_exact].case_count|120|0
RESCORE-C2-DELTA-COVERAGE|C-2 rules2 必需 Finding 覆盖配对差（单 − 多）|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=required_finding_coverage].delta|-0.17253086419753086|5e-05
RESCORE-C2-DELTA-COVERAGE-CI-LOW|C-2 rules2 必需 Finding 覆盖配对差区间下界|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=required_finding_coverage].ci_low|-0.18425925925925923|5e-05
RESCORE-C2-DELTA-COVERAGE-CI-HIGH|C-2 rules2 必需 Finding 覆盖配对差区间上界|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=required_finding_coverage].ci_high|-0.16111111111111112|5e-05
RESCORE-C2-DELTA-ESCALATED-CI-LOW|C-2 rules2 升级人工配对差区间下界|rescore:v1-02c-paired-360x3/v2/paired_deltas[metric=escalated_to_human].ci_low|0.029629629629629634|5e-05
RESCORE-C2-COVERAGE-RATE|C-2 rules2 多 Agent 必需 Finding 覆盖率|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.required_finding_coverage.value|0.17253086419753086|5e-05
RESCORE-C2-VALID-RATE|C-2 rules2 多 Agent 结构化输出有效率|rescore:v1-02c-paired-360x3/v2/cost_and_reliability.multi_agent.structured_output_valid_rate.value|0.5097643097643098|5e-05
RESCORE-C2-MACRO-F1|C-2 rules2 建议宏平均 F1|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.recommendation_macro_f1.value|0.18905926395936698|5e-05
RESCORE-C2-CITATION-OUTSIDE|C-2 rules2 多 Agent 越出允许上下文的引用数|rescore:v1-02c-paired-360x3/v2/quality.multi_agent.citation_outside_allowed_context.numerator|0|0
RESCORE-CC-ACCURACY-R2|C2 rules2 端到端建议准确率（不变）|rescore:v1-02c2-resolution-120x2/v2/quality.multi_agent.recommendation_accuracy.value|0.4166666666666667|5e-05
RESCORE-CC-CONFLICT-DEN|C2 应识别冲突观测数（分母不变）|rescore:v1-02c2-resolution-120x2/v2/quality.multi_agent.conflict_recall.denominator|60|0
RESCORE-CC-COVERAGE-RATE|C2 rules2 多 Agent 必需 Finding 覆盖率|rescore:v1-02c2-resolution-120x2/v2/quality.multi_agent.required_finding_coverage.value|0.2152777777777778|5e-05
RESCORE-CC-VALID-RATE|C2 rules2 多 Agent 结构化输出有效率|rescore:v1-02c2-resolution-120x2/v2/cost_and_reliability.multi_agent.structured_output_valid_rate.value|0.4727272727272727|5e-05
RESCORE-CC-MACRO-F1|C2 rules2 建议宏平均 F1|rescore:v1-02c2-resolution-120x2/v2/quality.multi_agent.recommendation_macro_f1.value|0.16826481019309075|5e-05
RESCORE-CC-D3-EVIDENCE|C2 rules2 升级类别 evidence_insufficient 观测数|rescore:v1-02c2-resolution-120x2/v2/decidability.multi_agent.rules_2.d3_escalation.by_class.evidence_insufficient|18|0
RESCORE-CC-DELTA-ENDTOEND|C2 rules2 端到端配对差（可解读但无差异）|rescore:v1-02c2-resolution-120x2/v2/paired_deltas[metric=recommendation_exact].delta|0.0|5e-05
RESCORE-CC-DELTA-ENDTOEND-CI-HIGH|C2 rules2 端到端配对差区间上界|rescore:v1-02c2-resolution-120x2/v2/paired_deltas[metric=recommendation_exact].ci_high|0.020833333333333332|5e-05
RESCORE-CC-DELTA-COVERAGE|C2 rules2 必需 Finding 覆盖配对差（单 − 多）|rescore:v1-02c2-resolution-120x2/v2/paired_deltas[metric=required_finding_coverage].delta|-0.21388888888888888|5e-05
RESCORE-CC-DELTA-COVERAGE-CI-LOW|C2 rules2 必需 Finding 覆盖配对差区间下界|rescore:v1-02c2-resolution-120x2/v2/paired_deltas[metric=required_finding_coverage].ci_low|-0.25277777777777777|5e-05
RESCORE-CC-DELTA-COVERAGE-CI-HIGH|C2 rules2 必需 Finding 覆盖配对差区间上界|rescore:v1-02c2-resolution-120x2/v2/paired_deltas[metric=required_finding_coverage].ci_high|-0.17638888888888887|5e-05
"""

_CLAIM_NUMBER = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?$", re.IGNORECASE)


def _parse_claims(rows: str) -> tuple[Claim, ...]:
    """Read the claim table, rejecting malformed or duplicated rows."""

    claims: list[Claim] = []
    for line in rows.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            raise ValueError(f"claim row must have 5 fields: {line[:40]}")
        claim_id, subject, source, expected, tolerance = parts
        if any(not part for part in (claim_id, subject, source)):
            raise ValueError(f"claim row has an empty field: {line[:40]}")
        if _SOURCE_KIND.match(source) is None:
            raise ValueError(f"claim source is not in the DSL: {source[:60]}")
        number = expected if _CLAIM_NUMBER.fullmatch(expected) is None else None
        claims.append(
            Claim(
                claim_id,
                subject,
                source,
                number if number is not None else float(expected),
                float(tolerance),
            )
        )
    if len({claim.claim_id for claim in claims}) != len(claims):
        raise ValueError("claim ids must be unique")
    return tuple(claims)


CLAIMS: tuple[Claim, ...] = _parse_claims(_CLAIM_ROWS)


if __name__ == "__main__":
    raise SystemExit(main())
