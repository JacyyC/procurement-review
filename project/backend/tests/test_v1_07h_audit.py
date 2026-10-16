"""Actual H and comparison CLIs, strict replay, permissions and frozen artifact sets."""

import asyncio
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from procurement_review.eval_h_audit import audit, freeze, replay, verify_freeze
from procurement_review.eval_h_cli import preflight
from procurement_review.eval_h_consumption import completion_proof, observation_failures
from procurement_review.eval_h_journal import Journal
from procurement_review.eval_h_runner import run
from procurement_review.eval_h_score import rebuild as score
from procurement_review.eval_h_report import rebuild as report
from procurement_review.eval_h_storage import write
from test_v1_07h_preparation import pg_batch as _pg_batch, sealed

pg_batch = _pg_batch


@pytest.fixture(scope="module")
def consumed(pg_batch):
    seal, requests = pg_batch
    seal.complete()
    seal.release()
    write(seal.directory / "score.json", score(seal.directory))
    seal.scored(seal.directory / "score.json")
    write(seal.directory / "result.json", report(seal.directory))
    assert len(requests) == 72
    return seal


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "procurement_review.eval_h_cli", *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_actual_h_and_comparison_clis_recompute_without_network(consumed, tmp_path):
    root = consumed.directory
    for command, expected in (("replay", "verified"), ("audit", "audit_passed")):
        result = cli(command, root)
        assert result.returncode == 0, result.stderr
        body = json.loads(result.stdout)
        assert (
            body[expected]
            and body["network_calls"] == 0
            and not body["stage_acceptance"]
        )
    script = Path(__file__).parents[2] / "scripts/comparison_report.py"
    target = tmp_path / "comparison-audit.json"
    result = subprocess.run(
        [sys.executable, str(script), str(root), "--output", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(target.read_bytes())["audit_passed"]
    frozen = tmp_path / "cli-freeze.json"
    assert cli("freeze", root, frozen).returncode == 0
    assert cli("verify-freeze", root, frozen).returncode == 0


def test_disposable_preflight_and_failure_cli_do_not_claim_formal_ready(tmp_path):
    seal = sealed(tmp_path)
    result = cli("preflight", seal.directory)
    assert result.returncode == 1 and not json.loads(result.stdout)["ready_for_formal"]
    assert preflight(seal.directory)["checks"]["lifecycle_evidence"] is False
    result = cli("report", seal.directory)
    assert result.returncode == 0
    assert json.loads(result.stdout)["comparison_complete"] is False
    assert seal.record()["state"] == "sealed"
    assert cli("replay", seal.directory).returncode == 0
    assert cli("audit", seal.directory).returncode == 1
    assert cli("complete", seal.directory).returncode != 0
    assert cli("release", seal.directory).returncode != 0
    with pytest.raises(ValueError, match="requires_mock"):
        asyncio.run(run(seal.directory))


@pytest.mark.parametrize(
    "mutation", ["boolean", "extra_field", "claim", "candidate", "score"]
)
def test_report_and_candidate_tampering_cannot_obtain_a_freeze(
    consumed, tmp_path, mutation
):
    from procurement_review.eval_h_seal import HSeal

    copied = tmp_path / ".local" / mutation
    shutil.copytree(consumed.directory, copied)
    if mutation == "score":
        path = copied / "score.json"
        body = json.loads(path.read_bytes())
        body["completion"]["complete"] = 1
    elif mutation == "candidate":
        path = copied / "candidate.json"
        body = json.loads(path.read_bytes())
        body["product_sha256"] = "0" * 64
    else:
        path = copied / "result.json"
        body = json.loads(path.read_bytes())
        if mutation == "boolean":
            body["Q-12"]["complete"] = 1
        elif mutation == "extra_field":
            body["extra"] = None
        else:
            next(iter(body["claims"].values()))["supported"] += 1
    write(path, body, replace=True)
    assert HSeal(copied).record()["state"] == "scored"
    assert not audit(copied)["audit_passed"]
    output = tmp_path / "refused-freeze.json"
    with pytest.raises(ValueError, match="freeze_audit_failed"):
        freeze(copied, output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["extra", "permission", "symlink", "unfinished"])
def test_freeze_verifies_full_safe_artifact_set(consumed, tmp_path, mutation):
    copied = tmp_path / ".local" / mutation
    shutil.copytree(consumed.directory, copied)
    output = tmp_path / "frozen.json"
    freeze(copied, output)
    assert verify_freeze(copied, output)["verified"]
    if mutation == "extra":
        write(copied / "new-observation.json", {})
    elif mutation == "permission":
        (copied / "private/seed.bin").chmod(0o644)
    elif mutation == "symlink":
        (copied / "escaping-link").symlink_to(output)
    else:
        write(copied / ".h-transaction-unfinished", {})
    with pytest.raises(
        ValueError, match="artifact_set|permissions|file_type|unfinished_artifact"
    ):
        verify_freeze(copied, output)


def test_success_without_model_identity_and_single_commit_type_confusion_are_refused(
    consumed,
):
    case = consumed.runner_cases()[0]
    body = Journal(consumed.directory).committed(case.case_id, 101, "single_agent")
    changed = copy.deepcopy(body)
    changed["attempts"][0]["receipt"]["resolved_model"] = None
    assert "attempt_model_missing" in observation_failures(
        consumed.plan(), case, changed
    )
    changed = copy.deepcopy(body)
    changed["attempts"][0]["projection"]["policy"]["blocked"] = 0
    from procurement_review.gold_v3_contracts import payload_sha256

    changed["attempts"][0]["projection_sha256"] = payload_sha256(
        changed["attempts"][0]["projection"]
    )
    errors = observation_failures(consumed.plan(), case, changed)
    assert "single_commit_contract_presence_or_type" in errors


def test_orphan_attempt_cannot_be_ignored_by_completion(consumed, tmp_path):
    copied = tmp_path / ".local" / "orphan"
    shutil.copytree(consumed.directory, copied)
    task = copied / "attempts/not-in-plan"
    task.mkdir(mode=0o700)
    write(task / "joint.1.json", dict(status="started"))
    proof = completion_proof(copied)
    assert (
        not proof["complete"]
        and "unregistered_attempt_task:not-in-plan" in proof["failures"]
    )


def test_canonical_key_order_does_not_change_replay(consumed, tmp_path):
    copied = tmp_path / ".local" / "key-order"
    shutil.copytree(consumed.directory, copied)
    path = copied / "result.json"
    body = json.loads(path.read_bytes())
    write(path, dict(reversed(list(body.items()))), replace=True)
    assert replay(copied)["verified"] and audit(copied)["audit_passed"]
