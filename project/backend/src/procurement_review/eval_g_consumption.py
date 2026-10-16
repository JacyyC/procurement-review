"""Explicit G v5 consumption; v4 records retain their historical semantics."""

import hashlib
import json
import tempfile
from pathlib import Path

from .gold_v3_contracts import payload_sha256
from .eval_g_product_summary import execution_contract_failures, projection_failures
from .eval_g_transactions import transaction_failures

CONSUMPTION_V5 = "v1-07g-consumption.v5"
SCORE_V5 = "v1-07g-dev-score.v5"
SEMANTICS_V5 = "v1-07g-score-semantics.v5"


def receipt_value_matches(
    item,
    attempt,
    receipt,
    observed_key,
    receipt_key,
    version,
    *,
    consumption_version=None,
):
    from .precise_dev_paired_metering import receipt_value_matches_v4

    if receipt_value_matches_v4(
        item, attempt, receipt, observed_key, receipt_key, version
    ):
        return True
    return (
        consumption_version == CONSUMPTION_V5
        and observed_key == "error_code"
        and receipt.get("next_correction_kind") is None
        and receipt.get("attempt_error") is None
        and attempt.get("error_code")
        in {item.get("first_error"), item.get("last_error")}
        and str(attempt.get("error_code")).startswith("citation_")
    )


def verify_versioned_binding(registration, observations, bindings, **kwargs):
    if (bindings or {}).get("consumption_version") and not consumption_version(
        registration
    ):
        raise ValueError("binding_consumption_version_mismatch")
    if consumption_version(registration):
        return verify_binding_v5(registration, observations, bindings or {}, **kwargs)
    from .eval_f_validation import _verify_sweep_binding_legacy_v4
    from .eval_f_lineage import _verify_sweep_binding_legacy_v3

    if (
        registration.get("extends", {}).get("extension_version")
        == "v1-07-truth-extension.v3"
    ):
        return _verify_sweep_binding_legacy_v4(
            registration, observations, bindings or {}, **kwargs
        )
    return _verify_sweep_binding_legacy_v3(
        registration, observations, bindings, **kwargs
    )


def consumption_version(registration):
    version = registration.get("binding", {}).get("consumption_version")
    if version is not None and version != CONSUMPTION_V5:
        raise ValueError("consumption_version_unsupported")
    if (
        version is not None
        and registration.get("extends", {}).get("extension_version")
        != "v1-07-truth-extension.v3"
    ):
        raise ValueError("consumption_extension_version")
    return version


def source_projections(source):
    if type(source) is not dict or type(source.get("runs")) is not dict:
        raise ValueError("v5_source_projection_type:runs")
    for run_id, run in source["runs"].items():
        if type(run) is not dict or type(run.get("execution")) is not dict:
            raise ValueError("v5_source_projection_type:execution:" + run_id)
        if (
            type(run.get("summary")) is not dict
            or type(run["summary"].get("summary_payload")) is not dict
        ):
            raise ValueError("v5_source_projection_type:summary:" + run_id)
    return {
        run_id: {
            "execution_sha256": payload_sha256(run["execution"]),
            "summary_sha256": payload_sha256(run["summary"]["summary_payload"]),
        }
        for run_id, run in source["runs"].items()
    }


def verify_source_v5(registration, source_path):
    from .eval_g_plan import verify_evidence_source

    if consumption_version(registration) != CONSUMPTION_V5:
        raise ValueError("v5_consumption_version_required")
    digest = verify_evidence_source(registration, source_path)
    source = json.loads(Path(source_path).read_bytes())
    if registration["binding"].get("source_projections") != source_projections(source):
        raise ValueError("v5_source_projection_binding")
    for run_id, run in source["runs"].items():
        failures = execution_contract_failures(run_id, run)
        if not failures:
            failures = projection_failures(registration, run_id, run)
        if failures:
            raise ValueError("v5_source_contract:" + ",".join(failures))
    return digest


def extend_registration_v5(base_path, source_path, output_path, inventory_path=None):
    from .eval_g_plan import extend_registration_v4

    with tempfile.TemporaryDirectory(prefix="g-v5-registration-") as folder:
        staged = Path(folder) / "registration.json"
        extend_registration_v4(base_path, source_path, staged)
        registered = json.loads(staged.read_bytes())
    registered["binding"].update(
        consumption_version=CONSUMPTION_V5,
        source_projections=source_projections(
            json.loads(Path(source_path).read_bytes())
        ),
    )
    if inventory_path is not None:
        from .eval_g_reuse import attach_product_reuse

        attach_product_reuse(
            registered, json.loads(Path(source_path).read_bytes()), inventory_path
        )
    verify_source_v5(registered, source_path)
    Path(output_path).write_text(
        json.dumps(registered, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return {"written": str(output_path), "consumption_version": CONSUMPTION_V5}


def verify_binding_v5(registration, observations, bindings, **kwargs):
    from .eval_f_validation import _verify_sweep_binding_legacy_v4

    if consumption_version(registration) != CONSUMPTION_V5:
        raise ValueError("v5_consumption_version_required")
    result = _verify_sweep_binding_legacy_v4(
        registration, observations, bindings, **kwargs
    )
    failures = []
    if bindings.get("consumption_version") != CONSUMPTION_V5:
        failures.append("binding_consumption_version:")
    proofs = registration["binding"].get("source_projections")
    if type(proofs) is not dict or set(proofs) != set(bindings.get("runs", {})):
        failures.append("source_projection_run_set:")
        proofs = {}
    for run_id, run in bindings.get("runs", {}).items():
        contract = execution_contract_failures(run_id, run)
        failures.extend(contract)
        unsafe = any(
            code.endswith(":" + run_id)
            and ("_missing:" in code or "_type" in code or "_time:" in code)
            for code in result["failures"]
        )
        if contract or unsafe:
            continue
        proof = proofs.get(run_id, {})
        for key, payload in (
            ("execution_sha256", run["execution"]),
            ("summary_sha256", run["summary"]["summary_payload"]),
        ):
            if type(proof) is not dict or proof.get(key) != payload_sha256(payload):
                failures.append(f"source_projection_content:{key}:{run_id}")
        failures.extend(projection_failures(registration, run_id, run))
        if not any(
            code.endswith(":" + run_id)
            and ("_type" in code or "_missing:" in code or "_time:" in code)
            for code in result["failures"]
        ):
            failures.extend(transaction_failures(run_id, run))
    result["failures"].extend(failures)
    result["complete"] = not result["failures"]
    result["consumption_version"] = CONSUMPTION_V5
    return result


def freeze_sweep_v5(
    registration_path,
    observations_path,
    bindings_path,
    source_path,
    score_path,
    output_path,
):
    from .precise_candidate import candidate_fingerprint
    from .precise_dev_score import score_sweep

    paths = dict(
        registration=Path(registration_path),
        observations=Path(observations_path),
        bindings=Path(bindings_path),
        source_bindings=Path(source_path),
    )
    values = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    verify_source_v5(values["registration"], source_path)
    result = score_sweep(
        values["registration"], values["observations"], values["bindings"]
    )
    hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
    }
    result["inputs"]["file_sha256"] = hashes
    if values["bindings"].get("observations_file_sha256") != hashes["observations"]:
        raise ValueError("v5_freeze_raw_file_binding")
    if payload_sha256(result) != payload_sha256(
        json.loads(Path(score_path).read_bytes())
    ):
        raise ValueError("v5_freeze_full_score_mismatch")
    if (
        result["score_version"] != SCORE_V5
        or not result["stage_acceptance"]
        or result["failed_gates"]
    ):
        raise ValueError("v5_freeze_gates_failed")
    candidate = candidate_fingerprint()
    if result["inputs"]["scorer_candidate_sha256"] != candidate["candidate_sha256"]:
        raise ValueError("v5_freeze_current_candidate")
    body = dict(
        freeze_version="v1-07g-dev-freeze.v2",
        candidate=candidate,
        consumption_version=CONSUMPTION_V5,
        score_version=SCORE_V5,
        semantics_version=SEMANTICS_V5,
        input_files=hashes,
        score_file_sha256=hashlib.sha256(Path(score_path).read_bytes()).hexdigest(),
        complete=result["completeness"]["complete"],
        stage_acceptance=True,
    )
    Path(output_path).write_text(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return {
        "written": str(output_path),
        "candidate_sha256": candidate["candidate_sha256"],
        "stage_acceptance": True,
    }
