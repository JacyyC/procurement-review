"""H product ledger and checkpoint causality, reusing G v5's independent validators."""

import json
from collections import Counter

from .eval_f_validation import (
    binding_shape_failures,
    causal_failures,
    metering_failures,
)
from .eval_g_product_summary import projection_failures
from .eval_g_transactions import transaction_failures
from .eval_h_material import documents, source_identities
from .gold_v3_contracts import payload_sha256
from .precise_run_contracts import PreciseCheckpointPayloadV1, precise_contract_identity
from .trace import canonical_checksum


def h_metering_failures(registration, run_id, run):
    from .precise_run_state import metering_from_ledger

    errors = metering_failures(registration, run_id, run)
    receipts = [
        {**event["output_summary"], "attempt": event.get("attempt")}
        for event in run["events"]
        if event.get("step") == "model_attempt"
    ]
    model_type_errors = [
        body for body in receipts if type(body.get("receipt_model")) is not str
    ]
    if model_type_errors and all(
        "receipt_model" in body
        and body["receipt_model"] is None
        and type(body.get("attempt_error")) is str
        and bool(body["attempt_error"])
        and body.get("outcome") == "failed"
        for body in model_type_errors
    ):
        errors = [error for error in errors if error != f"receipt_model_type:{run_id}"]
    if errors:
        return errors
    # G returns before ledger recomputation on its stricter model-type rule.
    # Recompute from the unmodified receipts after accepting H's failed-null case.
    expected = metering_from_ledger(
        run["attempts"],
        receipts=receipts,
        requested_model=registration["binding"]["requested_model"],
        resolved_models=[
            body["receipt_model"] for body in receipts if body["receipt_model"]
        ],
        explicit_recoveries=run["resume_count"],
        reused_branches=sum(
            cp["generation"] < len(run["generations"]) for cp in run["checkpoints"]
        ),
    ).model_dump(mode="json")
    meter = run["execution"]["metering"]
    return [
        f"ledger_mismatch:{name}:{run_id}"
        for name, value in expected.items()
        if payload_sha256(meter.get(name)) != payload_sha256(value)
    ]


def product_attempts(binding):
    receipts = {
        (event["generation"], event["worker"], event["attempt"]): event[
            "output_summary"
        ]
        for event in binding["events"]
        if event["step"] == "model_attempt"
    }
    result = []
    for attempt in binding["attempts"]:
        receipt = receipts.get(
            (attempt["generation"], attempt["worker"], attempt["attempt_no"])
        )
        projected = next(
            (
                row
                for row in binding.get("execution", {})
                .get(attempt["worker"], {})
                .get("attempts", [])
                if row["attempt_no"] == attempt["attempt_no"]
            ),
            {},
        )
        result.append(
            dict(
                branch=attempt["worker"],
                generation=attempt["generation"],
                attempt_no=attempt["attempt_no"],
                status="settled"
                if attempt["finished_at"] or attempt["abandoned"]
                else "started",
                effective_error=projected.get(
                    "error_code",
                    attempt.get("last_error")
                    or (
                        "interrupted_unknown"
                        if receipt is None
                        else receipt["attempt_error"]
                    ),
                ),
                receipt=receipt
                or dict(
                    attempt_error="interrupted_unknown",
                    input_usage=None,
                    output_usage=None,
                    usage_unknown=True,
                    receipt_model=None,
                    latency_ms=None,
                    http_status=None,
                    finish_reason=None,
                    request_sha256=None,
                ),
            )
        )
    return result


def product_failures(plan, case, observation, snapshot):
    run, run_id = observation["binding"], observation["row"]["run_id"]
    errors = binding_shape_failures(run_id, run)
    if errors:
        return errors
    if (
        run["execution"]["run_id"] != run_id
        or run["execution"]["case_id"] != case.case_id
    ):
        errors.append("product_execution_parent")
    sources = source_identities(plan, case)
    branch_sources, material_text = {}, {}
    for index, document in enumerate(case.documents):
        key = document.role + str(index)
        branch_sources[key] = sources[document.source_id]
        material_text[key] = document.content
    facts = {}
    for record in case.records:
        for field, value in {**record.payload, "identity": record.record_id}.items():
            facts[record.kind + "." + field] = dict(
                source_id=record.record_id,
                version=plan.dataset_version,
                content_sha256=payload_sha256(record.payload),
                value=value,
                effective_at=case.as_of,
            )
    registration = dict(
        truth_extension={
            case.case_id: dict(branch_sources=branch_sources, facts=facts)
        },
        material_text={case.case_id: material_text},
        binding=dict(
            expected_snapshot_fields=snapshot.model_dump(mode="json"),
            consumption_version="v1-07g-consumption.v5",
            requested_model=plan.configuration.requested_model,
            resolved_model=plan.configuration.resolved_model,
            max_attempts_per_branch=2,
            product_implementation_sha256=plan.product_sha256,
            expected_rules_version="3.1.0",
            expected_document_version=plan.document_version,
            expected_dataset_version=plan.dataset_version,
            expected_index_version=plan.index_version,
        ),
    )
    errors.extend(projection_failures(registration, run_id, run))
    errors.extend(causal_failures(registration, run_id, run, observation))
    errors.extend(h_metering_failures(registration, run_id, run))
    errors.extend(transaction_failures(run_id, run))
    if payload_sha256(observation["attempts"]) != payload_sha256(product_attempts(run)):
        errors.append("product_attempt_ledger_projection")
    keys = [
        (item["generation"], item["worker"], item["attempt_no"])
        for item in run["attempts"]
    ]
    if len(keys) != len(set(keys)):
        errors.append("product_duplicate_attempt")
    counts = Counter(item["worker"] for item in run["attempts"])
    if any(
        worker not in {"policy", "contract"} or count > 2
        for worker, count in counts.items()
    ):
        errors.append("product_shared_attempt_quota")
    expected = {
        role
        for role in ("policy", "contract")
        if run["execution"][role]["outcome"] == "completed"
    }
    if (
        len(run["checkpoints"]) != len(expected)
        or {cp["worker"] for cp in run["checkpoints"]} != expected
    ):
        errors.append("product_checkpoint_set")
    for cp in run["checkpoints"]:
        payload = cp["result_payload"]
        parsed = PreciseCheckpointPayloadV1.model_validate_json(
            json.dumps(payload), strict=True
        )
        native = run["execution"][cp["worker"]]
        commit = parsed.result.model_dump(mode="json")
        if cp["run_id"] != run_id or parsed.result.worker != cp["worker"]:
            errors.append("product_checkpoint_parent")
        contract = precise_contract_identity(snapshot, cp["worker"])
        if (
            cp["result_checksum"] != canonical_checksum(payload)
            or cp["contract_checksum"] != contract
            or parsed.contract_checksum != contract
        ):
            errors.append("product_checkpoint_digest")
        if (
            cp["input_checksum"] != parsed.input_checksum
            or cp["input_checksum"] != "sha256:" + native["request_sha256"]
        ):
            errors.append("product_checkpoint_request")
        for field in (
            "worker",
            "spans",
            "request_sha256",
            "answer_sha256",
            "support_failures",
            "attempts",
            "evidence_refs",
        ):
            if payload_sha256(commit[field]) != payload_sha256(native[field]):
                errors.append("product_checkpoint_projection:" + field)
        reported = [
            dict(
                obligation_id=duty["obligation_id"],
                status=duty["status"],
                normalized=None if duty["value"] == "UNKNOWN" else duty["value"],
            )
            for duty in commit["duties"]
        ]
        if payload_sha256(reported) != payload_sha256(native["reported"]):
            errors.append("product_checkpoint_projection:reported")
        if any(
            event not in run["execution"]["security"] for event in commit["security"]
        ):
            errors.append("product_checkpoint_security")
    own_docs = documents(plan, case)
    for role in ("policy", "contract"):
        actual = run["execution"][role]["evidence_refs"]
        allowed = [row for row in own_docs if row.category == role]
        for ref in actual:
            matches = [row for row in allowed if row.chunk_id == ref["chunk_id"]]
            if (
                len(matches) != 1
                or matches[0].content_checksum != ref["content_checksum"]
                or matches[0].source_id != ref["source_id"]
            ):
                errors.append("product_actual_source_identity")
    return errors
