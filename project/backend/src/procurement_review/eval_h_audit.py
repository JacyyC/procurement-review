"""Zero-network H consumer audit and exclusive frozen artifact manifests."""

from pathlib import Path

from .eval_h_consumption import completion_proof
from .eval_h_identity import identity, require_identity
from .eval_h_report import rebuild
from .eval_h_seal import HSeal
from .eval_h_storage import private, read, sha, write
from .gold_v3_contracts import payload_sha256


def replay(directory):
    from .eval_h_score import rebuild as score

    directory = Path(directory)
    rebuilt = rebuild(directory)
    checks = dict(
        report=payload_sha256(read(directory / "result.json"))
        == payload_sha256(rebuilt)
    )
    if (directory / "score.json").exists():
        checks["score"] = payload_sha256(
            read(directory / "score.json")
        ) == payload_sha256(score(directory))
    return dict(
        verified=all(checks.values()),
        checks=checks,
        network_calls=0,
        comparison_complete=rebuilt["Q-12"]["complete"],
        stage_acceptance=False,
    )


def audit(directory):
    from .eval_h_score import rebuild as score

    directory = Path(directory)
    seal = HSeal(directory)
    plan = seal.plan()
    require_identity(plan)
    report = rebuild(directory)
    stored = (
        read(directory / "result.json")
        if (directory / "result.json").exists()
        else None
    )
    score_path = directory / "score.json"
    recorded = seal.record()
    checks = dict(
        complete_persistent_plan=completion_proof(directory)["complete"],
        report_recomputation=payload_sha256(stored) == payload_sha256(report),
        score_recomputation=score_path.exists()
        and payload_sha256(read(score_path)) == payload_sha256(score(directory)),
        scored_artifact=recorded["state"] == "scored"
        and score_path.exists()
        and sha(score_path) == recorded.get("score_sha256"),
        current_candidate=True,
        stored_candidate=payload_sha256(read(directory / "candidate.json"))
        == payload_sha256(identity()),
        claim_reconciliation=len(report["claims"]) == len(plan.observation_keys()),
        formal_not_claimed_for_disposable=report["stage_acceptance"] is False,
    )
    return dict(
        schema_version="v1-07h-audit.v1",
        batch=plan.batch_id,
        audit_checks=checks,
        audit_passed=all(checks.values()),
        comparison_complete=report["Q-12"]["complete"],
        incomplete_reasons=report["Q-12"]["incomplete_reasons"],
        product_gates=report["product_quality"],
        outcome=report["Q-12"]["outcome"],
        stage_acceptance=False,
        network_calls=0,
        candidate_sha256=plan.candidate_sha256,
        consumer_sha256=plan.consumer_sha256,
    )


def freeze(directory, output):
    directory = Path(directory)
    verdict = audit(directory)
    if not verdict["audit_passed"]:
        raise ValueError("h_freeze_audit_failed")
    output = Path(output)
    if output.resolve().is_relative_to(directory.resolve()):
        raise ValueError("h_freeze_output_must_be_outside_batch")
    files = artifact_manifest(directory)
    body = dict(
        schema_version="v1-07h-freeze.v1",
        candidate=identity(),
        artifacts=files,
        audit=verdict,
        mode=HSeal(directory).plan().mode,
        stage_acceptance=False,
    )
    write(output, body)
    return {
        "frozen": True,
        "artifacts": len(files),
        "candidate_sha256": body["candidate"]["candidate_sha256"],
    }


def verify_freeze(directory, frozen):
    body = read(Path(frozen))
    if body.get("schema_version") != "v1-07h-freeze.v1" or payload_sha256(
        body.get("candidate")
    ) != payload_sha256(identity()):
        raise ValueError("h_frozen_candidate_changed")
    if payload_sha256(body["artifacts"]) != payload_sha256(
        artifact_manifest(Path(directory))
    ):
        raise ValueError("h_frozen_artifact_set_changed")
    for entry, digest in body["artifacts"].items():
        path = Path(directory) / entry
        if (
            not path.resolve().is_relative_to(Path(directory).resolve())
            or sha(path) != digest
        ):
            raise ValueError("h_frozen_artifact_changed")
    if payload_sha256(audit(directory)) != payload_sha256(body["audit"]):
        raise ValueError("h_frozen_audit_changed")
    return {"verified": True, "network_calls": 0, "stage_acceptance": False}


def artifact_manifest(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.name.startswith(".h-transaction-"):
            raise ValueError("h_unfinished_artifact_transaction")
        if (
            path.relative_to(directory) == Path("private/runtime.json")
            or path.name == ".transition.lock"
        ):
            continue
        private(path, directory=path.is_dir())
        if path.is_file():
            files[str(path.relative_to(directory))] = sha(path)
    return files
