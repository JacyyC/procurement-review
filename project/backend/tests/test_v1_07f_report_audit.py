"""The read-only audit must reject report, export and observation tampering."""

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from procurement_review.eval_f_context import stable_write
from procurement_review.eval_f_registration import prepare
from procurement_review.eval_f_run import execute
from v1_07f_support import response_for

SCRIPT = Path(__file__).parents[2] / "scripts/precise_report.py"
spec = importlib.util.spec_from_file_location("precise_report", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def batch(tmp_path):
    root = tmp_path / ".local/live"
    path = tmp_path / ".local/registration.json"
    prepare(path)
    asyncio.run(
        execute(
            path,
            "minimum",
            transport_override=httpx.MockTransport(response_for),
            offline_root=root,
        )
    )
    return next(root.glob("*/minimum"))


def test_complete_batch_audit_is_read_only_and_never_claims_stage_pass(batch):
    report = audit.build_report(batch)
    assert report["audit_passed"] and not report["stage_acceptance"]
    assert report["network_calls"] == 0
    assert report["usage"]["physical_reserved"] == 12
    assert report["independent_reference"]["exact_truth_matches"] == 6
    assert report["audit_checks"]["originals_unchanged"]
    assert report["timing"]["111"]["all_planned"]["mcp_ms"]["unobserved"] == 6


def test_reference_disagreement_cannot_pass_an_otherwise_intact_audit(
    batch, monkeypatch
):
    from procurement_review import gold_v31_oracle

    original = gold_v31_oracle.derive_truth

    def disagree(case):
        truth = original(case)
        return truth.model_copy(update={"high_risk": not truth.high_risk})

    monkeypatch.setattr(gold_v31_oracle, "derive_truth", disagree)
    report = audit.build_report(batch)
    assert report["independent_reference"]["exact_truth_matches"] == 0
    assert not report["audit_passed"]
    assert not report["audit_checks"]["independent_truth_matches"]


@pytest.mark.parametrize("kind", ["report", "receipt", "missing_case"])
def test_corruption_cannot_become_a_pass(batch, kind):
    if kind == "report":
        path = batch / "result.json"
        value = json.loads(path.read_text())
        value["combined"]["counts"]["first_valid"] += 1
        stable_write(path, value)
        failed_check = "zero_network_replay_identical"
    elif kind == "receipt":
        path = batch / "ledger-export.json"
        value = json.loads(path.read_text())
        value["attempts"][0]["receipt"]["output_tokens"] += 100
        stable_write(path, value)
        failed_check = "export_and_record_hashes"
    else:
        next(batch.glob("V107-*.json")).unlink()
        failed_check = "case_set_complete"
    report = audit.build_report(batch)
    assert not report["audit_passed"] and not report["audit_checks"][failed_check]
    assert not report["stage_acceptance"]
    if kind == "missing_case":
        assert report["timing"]["111"]["all_planned"]["execution_ms"]["count"] == 6
        assert report["timing"]["111"]["all_planned"]["execution_ms"]["p95"] is None
        assert report["quality"]["combined"]["counts"]["cases"] == 6
