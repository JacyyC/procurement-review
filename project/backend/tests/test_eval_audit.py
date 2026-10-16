"""AC-09 audit tool: artifact ledger, end-to-end trace, redaction scan and cross-check.

Everything here runs offline. Segments are produced by the dry-run transport into a
temporary ``.local/runs`` directory, so the tool is exercised on real persisted artifacts
without a model request. The claim table is checked for shape only; resolving all 318
claims against the real batch artifacts is what Evidence §46, RUN-01/AM-01 and ADR-01/REP-01
record.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path

import pytest

from procurement_review.eval_audit import (
    AUDIT_SCHEMA_VERSION,
    CLAIMS,
    BrakeRollup,
    Claim,
    _pattern_hits,
    _resolve_path,
    build_ledger,
    classify_leaves,
    cross_check,
    cumulative_brake,
    main,
    run_audit,
    scan_artifacts,
    trace_case,
    write_audit,
)
from procurement_review.eval_contract import (
    MANIFEST_FILE_NAME,
    PROJECTION_FILE_NAME,
    load_projection,
)
from procurement_review.eval_run import run_batch

CALIBRATION = "v1-02c-calibration-48"
PROSE = "缺少供应商税务登记证明与有效期"


def _dry_run_runs_dir(tmp_path: Path) -> Path:
    runs_dir = tmp_path / ".local" / "runs"
    written = asyncio.run(
        run_batch(
            batch_id=CALIBRATION,
            execution_mode="dry_run",
            topologies=("multi_agent",),
            runs_dir=runs_dir,
        )
    )
    assert len(written) == 1
    return runs_dir


def _segment_run_id(runs_dir: Path) -> str:
    return next(
        path.parent.name
        for path in runs_dir.glob(f"eval-*/{MANIFEST_FILE_NAME}")
        if path.parent.name.startswith("eval-multi_agent")
    )


def test_ledger_describes_every_segment_and_keeps_mock_usage_out_of_the_brake(
    tmp_path: Path,
) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    entries = build_ledger(runs_dir)
    kinds = {entry.kind for entry in entries}
    assert kinds == {"eval-run-manifest", "eval-projection"}
    assert all(entry.load_status == "ok" for entry in entries)
    manifests = [entry for entry in entries if entry.kind == "eval-run-manifest"]
    assert len(manifests) == 1
    assert manifests[0].execution_mode == "dry_run"
    assert (manifests[0].model_call_count or 0) > 0
    assert manifests[0].case_count == 48
    assert len(manifests[0].resolved_model_ids) == 1

    brake = cumulative_brake(entries, runs_dir=runs_dir)
    assert brake.manifest_requests == 0
    assert brake.dry_run_requests == (manifests[0].model_call_count or 0)
    assert brake.warning_reached is False
    assert brake.eval_segments == 0


def test_ledger_marks_a_historical_manifest_as_contract_drift(tmp_path: Path) -> None:
    runs_dir = tmp_path / ".local" / "runs"
    directory = runs_dir / "smoke-real_endpoint-history"
    directory.mkdir(parents=True)
    payload = {
        "schema_version": "run-manifest.v1",
        "run_id": "smoke-real_endpoint-history",
        "work_item_id": "V1-02",
        "stage_id": "V1-02B",
        "topology": "chat_adapter_smoke",
        "execution_mode": "real_endpoint",
        "batch_id": "v1-02b-frozen-smoke-48",
        "case_split": "dev",
        "case_count": 48,
        "family_count": 12,
        "cases_per_family": 4,
        "repeat_seeds": [20260901],
        "concurrency": 1,
        "timeout_seconds": 30.0,
        "max_retries": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 1536,
        "requested_model_id": "deepseek-v4-flash",
        "resolved_model_ids": ["deepseek-v4-flash-ga-260731"],
        "capabilities": {},
        "source_versions": {},
        "gold_manifest_checksum": "0" * 64,
        "started_at": "2026-09-09T12:00:00+00:00",
        "finished_at": "2026-09-09T13:00:00+00:00",
        "branch_count": 1,
        "model_call_count": 1,
        "skipped_empty_context_count": 0,
        "succeeded_count": 1,
        "failed_count": 0,
        "retry_total": 0,
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "latency": {"p50_ms": 1, "p95_ms": 1, "max_ms": 1},
        "calls": [],
        "budget": {
            "max_requests": 1000,
            "max_input_tokens": 1000000,
            "max_output_tokens": 100000,
            "consumed_requests": 1,
            "consumed_input_tokens": 10,
            "consumed_output_tokens": 20,
            "warning_reached": False,
            "exceeded": False,
        },
        "outcome": "completed",
        "redaction_status": "redacted",
    }
    (directory / "run-manifest.v1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    entries = build_ledger(runs_dir)
    entry = next(item for item in entries if item.kind == "smoke-run-manifest")
    assert entry.load_status == "contract_revision_drift"
    assert "timeout_seconds" in (entry.status_detail or "")


def test_trace_rebuilds_one_case_end_to_end_with_no_free_text(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    run_id = _segment_run_id(runs_dir)
    projection = load_projection(runs_dir / run_id / PROJECTION_FILE_NAME)
    case_id = next(branch.case_id for branch in projection.branches if branch.findings)

    trace = trace_case(run_id=run_id, case_id=case_id, runs_dir=runs_dir)
    document = asdict(trace)
    assert trace.batch_id == CALIBRATION
    assert trace.recommendation in {
        "insufficient_information",
        "needs_human_review",
        "recommend_approve",
        "recommend_reject",
    }
    assert trace.findings
    assert trace.allowed_context_evidence_refs
    assert trace.branch_outcomes
    assert all(
        call.request_id_hash is None or len(call.request_id_hash) == 64
        for call in trace.calls
    )
    assert classify_leaves(document)["free_form_fields"] == {}
    assert all(
        item.startswith("critical_evidence_missing:")
        or item.startswith("unresolved_finding:")
        or item.startswith("finding_failure:")
        or item.startswith("worker_failure:")
        or item.startswith("sha256:")
        for item in trace.missing_information
    )


def test_trace_refuses_an_unknown_segment_or_case(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    with pytest.raises(ValueError, match="segment_not_found"):
        trace_case(run_id="missing", case_id="GOLD-0001", runs_dir=runs_dir)


def _plant_prose(runs_dir: Path, text: str) -> Path:
    path = next(iter(sorted(runs_dir.glob(f"eval-*/{PROJECTION_FILE_NAME}"))))
    projection = json.loads(path.read_text(encoding="utf-8"))
    branch = next(item for item in projection["branches"] if item["findings"])
    branch["findings"][0]["missing_information"] = [text]
    path.write_text(json.dumps(projection), encoding="utf-8")
    return path


def test_scan_reports_retained_model_prose_and_never_echoes_it(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    _plant_prose(runs_dir, PROSE)
    document = scan_artifacts(runs_dir, credential="")
    codes = {finding["code"] for finding in document["findings"]}
    assert "retained_model_free_text" in codes
    assert document["totals"]["retained_occurrences"] >= 1
    assert PROSE not in json.dumps(document, ensure_ascii=False)
    free_form = document["projection_field_classes"]["free_form_fields"]
    assert any(count >= 1 for count in free_form.values())


def test_scan_flags_planted_secret_shapes_and_the_live_credential(
    tmp_path: Path,
) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    path = _plant_prose(runs_dir, "Bearer sk-abcdefghijklmnopqrst")
    credential = "local-credential-value-1234567890"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_id_lock"] = f"{payload['model_id_lock']} {credential}"
    path.write_text(json.dumps(payload), encoding="utf-8")

    codes = {
        finding["code"]
        for finding in scan_artifacts(runs_dir, credential=credential)["findings"]
        if finding["severity"] == "finding"
    }
    assert "live_credential_present" in codes
    assert "secret_value_path_hit" in codes
    assert "sensitive_value_shape_prefixed_secret_key" in codes


def test_scan_flags_a_world_readable_run_directory(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    os.chmod(runs_dir, 0o755)
    findings = scan_artifacts(runs_dir, credential="")["findings"]
    assert any(finding["code"] == "permission_deviation" for finding in findings)


def test_scan_accepts_owner_only_directories(tmp_path: Path) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    # run_batch creates the parent directories without narrowing them, so the premise of
    # this test has to be established first.
    for path in (runs_dir, *runs_dir.rglob("*")):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    findings = [
        finding
        for finding in scan_artifacts(runs_dir, credential="")["findings"]
        if finding["code"] == "permission_deviation"
    ]
    assert findings == []


def test_pii_shapes_never_fire_inside_a_hex_digest() -> None:
    import re

    mobile = re.compile(r"1[3-9]\d{9}")
    digest = "0123456789abcdef1381234567890abcdef1381234567890abcdef"
    assert _pattern_hits("mainland_mobile_number", mobile, digest) is False
    assert _pattern_hits("mainland_mobile_number", mobile, "13812345678") is True
    assert _pattern_hits("mainland_mobile_number", mobile, "联系人 13812345678") is True
    assert _pattern_hits("prefixed_secret_key", re.compile(r"sk-\w+"), "sk-x") is True


def test_repository_patterns_ignore_parameter_names_but_keep_literal_secrets() -> None:
    from procurement_review.eval_audit import (
        REPOSITORY_BEARER_TOKEN,
        REPOSITORY_SECRET_ASSIGNMENT,
    )

    assert (
        REPOSITORY_SECRET_ASSIGNMENT.search("api_key=SecretStr(DRY_RUN_CREDENTIAL)")
        is None
    )
    assert (
        REPOSITORY_SECRET_ASSIGNMENT.search('api_key = "sk-live-1234567890ab"')
        is not None
    )
    assert REPOSITORY_BEARER_TOKEN.search("Bearer fake-abcdef0123456789") is not None


def test_claim_sources_all_parse_and_stay_unique() -> None:
    assert len(CLAIMS) > 180
    assert len({claim.claim_id for claim in CLAIMS}) == len(CLAIMS)
    prefixes = {claim.source.split(":", 1)[0] for claim in CLAIMS}
    assert prefixes <= {"report", "segments", "slice", "brake", "rescore"}
    assert "rescore" in prefixes
    assert all(claim.subject and isinstance(claim.tolerance, float) for claim in CLAIMS)


def test_cross_check_recomputes_a_report_claim_and_reports_a_mismatch(
    tmp_path: Path,
) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    actual = cross_check(
        (
            Claim(
                "PROBE",
                "干跑分段的多 Agent 调用数",
                f"report:{CALIBRATION}/cost_and_reliability.multi_agent.model_call_count",
                -1,
            ),
            Claim("BRAKE", "干跑不计入刹车", "brake:manifest_requests", 0),
        ),
        runs_dir=runs_dir,
    )
    assert actual["claims"] == 2
    assert actual["mismatched"] == ["PROBE"]
    assert actual["matched"] == 1
    assert json.dumps(actual, ensure_ascii=False)


def test_selector_paths_resolve_by_field_and_reject_ambiguity(tmp_path: Path) -> None:
    document = {
        "paired_deltas": [
            {"metric": "required_finding_coverage", "delta": -0.2},
            {"metric": "conflict_detected", "delta": 0.0},
        ]
    }
    assert (
        _resolve_path(document, "paired_deltas[metric=conflict_detected].delta") == 0.0
    )
    with pytest.raises(KeyError):
        _resolve_path(document, "paired_deltas[metric=missing].delta")


def test_run_audit_writes_the_document_into_an_ignored_directory(
    tmp_path: Path,
) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    document = run_audit(runs_dir=runs_dir, claims=())
    assert document["schema_version"] == AUDIT_SCHEMA_VERSION
    assert document["cross_check"]["claims"] == 0
    path = write_audit(document, runs_dir=runs_dir)
    assert path.parent.name == "audit"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    with pytest.raises(ValueError, match="artifact_dir_not_ignored"):
        write_audit(document, runs_dir=tmp_path / "runs")


def test_cli_writes_only_inside_an_ignored_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    assert main(["--runs-dir", str(runs_dir), "ledger", "--summary-only"]) == 0
    assert json.loads(capsys.readouterr().out)["manifest_requests"] == 0
    outside = tmp_path / "leak.json"
    assert main(["--runs-dir", str(runs_dir), "ledger", "--output", str(outside)]) == 1
    assert "artifact_dir_not_ignored" in capsys.readouterr().err
    assert not outside.exists()


def test_cross_check_cli_exits_non_zero_when_one_claim_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir = _dry_run_runs_dir(tmp_path)
    path = f"report:{CALIBRATION}/cost_and_reliability.multi_agent.model_call_count"
    good = Claim("PROBE-OK", "干跑分段的多 Agent 调用数", path, 132)
    bad = Claim("PROBE-BAD", "同一个数字被写错", path, 999)
    monkeypatch.setattr("procurement_review.eval_audit.CLAIMS", (good,))
    assert main(["--runs-dir", str(runs_dir), "cross-check"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "claims": 1,
        "matched": 1,
        "mismatched": [],
    }
    monkeypatch.setattr("procurement_review.eval_audit.CLAIMS", (good, bad))
    assert main(["--runs-dir", str(runs_dir), "cross-check"]) == 1
    assert json.loads(capsys.readouterr().out)["mismatched"] == ["PROBE-BAD"]


def test_brake_rollup_type_is_serialisable(tmp_path: Path) -> None:
    rollup = BrakeRollup(
        decision_max_requests=25_000,
        decision_max_input_tokens=250_000_000,
        decision_max_output_tokens=24_000_000,
        warning_ratio=0.8,
        manifest_requests=1,
        manifest_input_tokens=2,
        manifest_output_tokens=3,
        request_share=0.1,
        input_share=0.2,
        output_share=0.3,
        usage_bearing_requests=1,
        dry_run_requests=0,
        dry_run_input_tokens=0,
        dry_run_output_tokens=0,
        eval_segments=1,
        smoke_segments=0,
        warning_reached=False,
        unaccounted_requests_low=0,
        unaccounted_requests_high=1,
        unaccounted_output_tokens_low=0,
        unaccounted_output_tokens_high=1,
        adjusted_request_share=0.11,
        adjusted_output_share=0.31,
    )
    assert asdict(rollup)["manifest_requests"] == 1
