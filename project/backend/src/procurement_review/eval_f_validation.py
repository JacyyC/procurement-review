"""Durable validation consumption and versioned G adoption evidence proofs."""

import json
import re
from datetime import UTC, datetime, timedelta

from pydantic import TypeAdapter, ValidationError

from .eval_attempt_ledger import LedgerError
from .eval_f_context import load_cases, stable_write
from .gold_v3_contracts import payload_sha256
from .precise_run_contracts import (
    PreciseBranchCommitV1,
    PreciseCheckpointPayloadV1,
    PreciseDocumentBranchV1,
    PreciseRunSnapshotV1,
    RunMeteringV1,
    precise_contract_identity,
)
from .trace import canonical_checksum
from .record_contracts import WorkerAttemptRecord
from .audit_contracts import CheckpointRecord
from .business_v3_security import SecurityEventV3
from .contracts import ApprovalActionType, Identifier, RunEventType
from .eval_g_plan import (
    adoption_evidence_shape_failures,
    evidence_manifest as evidence_manifest,
    extend_registration_v4 as extend_registration_v4,
    freeze_sweep_v4 as freeze_sweep_v4,
    generation_shape_failures,
    incomplete_binding_v4,
    verify_evidence_source as verify_evidence_source,
)


def group(version, case_ids):
    cases = {c.case_id: c for c in load_cases("validation", version)}
    selected = [cases[cid] for cid in case_ids]
    return dict(
        case_ids=sorted(case_ids),
        families=sorted({c.generation_family_id for c in selected}),
        entities=sorted({c.entity_root_id for c in selected}),
        source_hashes=sorted(
            {
                d.content_sha256
                for c in selected
                for d in (
                    c.model_input.policy_document,
                    c.model_input.contract_document,
                )
            }
        ),
    )


def claim_validation(root, registered, candidate_name):
    """Called under the stage lock, after prerequisites, before any network work."""
    candidate = registered["candidates"][candidate_name]
    identity = dict(
        registration=payload_sha256(registered),
        candidate=candidate_name,
        case_ids=candidate["case_ids"],
    )
    current = group(registered["versions"][0], candidate["case_ids"])
    entries = []
    legacy = root / "validation-consumed.json"
    if legacy.exists():
        old = json.loads(legacy.read_text())
        entries.append(
            dict(identity=old, group=group("gold-set.v3.1", old["case_ids"]))
        )
    folder = root / "consumed-validation"
    entries.extend(json.loads(p.read_text()) for p in folder.glob("*.json"))
    for old in entries:
        if old["identity"] == identity:
            if old["group"] != current:
                raise LedgerError("consumed validation inputs changed")
            return
        if any(set(old["group"][key]) & set(current[key]) for key in current):
            raise LedgerError(
                "validation group already consumed; a new group is required"
            )
    folder.mkdir(exist_ok=True)
    stable_write(
        folder / (payload_sha256(current) + ".json"),
        dict(identity=identity, group=current),
    )


BINDING_V4 = "v1-07g-db-binding.v4"
EXTENSION_V4 = "v1-07-truth-extension.v3"
EVENT_FIELDS_V4 = {
    "trace_id": TypeAdapter(Identifier),
    "step": TypeAdapter(Identifier),
    "event_type": TypeAdapter(RunEventType),
}


def expected_snapshot(registration):
    binding = registration["binding"]
    return {
        **binding["expected_snapshot_fields"],
        "mode": "real",
        "implementation_sha256": binding["product_implementation_sha256"],
        **{
            name: binding[name]
            for name in ("requested_model", "resolved_model", "max_attempts_per_branch")
        },
        **{
            name: binding["expected_" + name]
            for name in (
                "rules_version",
                "document_version",
                "dataset_version",
                "index_version",
            )
        },
    }


def checkpoint_failures(registration, run_id, run):
    failures = []

    def fail(code):
        failures.append(f"{code}:{run_id}")

    def required(record, fields, prefix):
        if type(record) is not dict:
            fail(prefix + "_type")
            return False
        for name in fields:
            if name not in record:
                fail(f"{prefix}_missing:{name}")
        return True

    frozen = expected_snapshot(registration)
    required(run["snapshot"], PreciseRunSnapshotV1.model_fields, "snapshot")
    try:
        PreciseRunSnapshotV1.model_validate_json(
            json.dumps(run["snapshot"]), strict=True
        )
    except ValidationError:
        fail("snapshot_contract_type")
        return failures
    if run["snapshot"] != frozen:
        fail("snapshot_frozen_identity")
    expected = PreciseRunSnapshotV1.model_validate_json(json.dumps(frozen), strict=True)
    manifest = registration["binding"].get("adoption_evidence", {})
    proof = manifest.get("runs", {}).get(run_id, {})
    if proof.get("case_id") != run.get("case_id"):
        fail("evidence_run_identity")
    for cp in run.get("checkpoints", []):
        required(
            cp,
            (
                "run_id",
                "worker",
                "generation",
                "attempt_no",
                "input_checksum",
                "contract_checksum",
                "result_checksum",
                "result_payload",
                "adopted_at",
            ),
            "checkpoint",
        )
        try:
            CheckpointRecord.model_validate_json(
                json.dumps(
                    {key: cp[key] for key in CheckpointRecord.model_fields if key in cp}
                ),
                strict=True,
            )
        except ValidationError:
            fail("checkpoint_record_type")
            continue
        payload = cp.get("result_payload", {})
        if type(payload) is not dict:
            fail("checkpoint_payload_type")
            continue
        required(payload, PreciseCheckpointPayloadV1.model_fields, "checkpoint_payload")
        commit = payload.get("result", {})
        if type(commit) is not dict:
            fail("checkpoint_commit_type")
            continue
        required(commit, PreciseBranchCommitV1.model_fields, "checkpoint_commit")
        if set(PreciseBranchCommitV1.model_fields) - set(commit):
            continue
        worker = cp.get("worker")
        if payload.get("checkpoint_version") != "precise-branch-commit.v1":
            fail("checkpoint_version")
        if commit.get("commit_version") != "precise-branch-commit.v1":
            fail("checkpoint_commit_version")
        request = (
            registration["binding"]["expected_requests"]
            .get(run["case_id"], {})
            .get(worker)
        )
        contract = precise_contract_identity(expected, worker)
        for record, prefix in ((cp, "checkpoint"), (payload, "checkpoint_payload")):
            if record.get("contract_checksum") != contract:
                fail(prefix + "_contract")
            if record.get("input_checksum") != "sha256:" + str(request):
                fail(prefix + "_request")
        if canonical_checksum(payload) != cp.get("result_checksum"):
            fail("checkpoint_result_digest")
        committed = proof.get("checkpoints", {}).get(worker, {})
        if (cp.get("generation"), cp.get("attempt_no")) != (
            committed.get("generation"),
            committed.get("attempt_no"),
        ):
            fail("checkpoint_evidence_identity")
        if payload_sha256(payload) != committed.get("payload_sha256"):
            fail("checkpoint_complete_content")
        for key in set(PreciseBranchCommitV1.model_fields) | set(commit):
            if payload_sha256(commit.get(key)) != committed.get(
                "commit_fields", {}
            ).get(key):
                fail("checkpoint_commit_content:" + key)
                if key in {"document_failures", "blocked"}:
                    fail("checkpoint_" + key)
        try:
            PreciseCheckpointPayloadV1.model_validate_json(
                json.dumps(payload), strict=True
            )
            native = run["execution"].get(worker, {})
            if not required(
                native, PreciseDocumentBranchV1.model_fields, "execution_branch"
            ):
                continue
            PreciseDocumentBranchV1.model_validate_json(json.dumps(native), strict=True)
        except ValidationError:
            fail("checkpoint_complete_contract_type")
            continue
        for key in (
            "worker",
            "spans",
            "support_failures",
            "request_sha256",
            "answer_sha256",
            "attempts",
            "evidence_refs",
        ):
            if commit[key] != native.get(key):
                fail("checkpoint_commit_projection:" + key)
        reported = [
            {
                "obligation_id": d["obligation_id"],
                "status": d["status"],
                "normalized": None if d["value"] == "UNKNOWN" else d["value"],
            }
            for d in commit["duties"]
        ]
        if reported != native.get("reported"):
            fail("checkpoint_commit_projection:reported")
        if native.get("outcome") != "completed":
            fail("checkpoint_branch_not_completed")
        if any(
            event not in run["execution"].get("security", [])
            for event in commit["security"]
        ):
            fail("checkpoint_security_projection")
        if commit["blocked"] != any(
            SecurityEventV3.model_validate(event).escalates
            for event in commit["security"]
        ):
            fail("checkpoint_blocked")
    return failures


def causal_failures(registration, run_id, run, row):
    failures = []
    frozen_window = timedelta(
        seconds=expected_snapshot(registration)["generation_deadline_seconds"]
    )

    def fail(code):
        failures.append(f"{code}:{run_id}")

    def instant(record, key, prefix, optional=False):
        if key not in record or record[key] is None:
            if not optional:
                fail(f"{prefix}_missing:{key}")
            return None
        try:
            if type(record[key]) is not str:
                raise ValueError("time_type")
            at = datetime.fromisoformat(record[key])
            if at.tzinfo is None or at.utcoffset() is None:
                raise ValueError("timezone_missing")
            return at.astimezone(UTC)
        except (ValueError, TypeError, OverflowError):
            fail(f"{prefix}_time:{key}")
            return None

    created = instant(run, "created_at", "run")
    observed = instant(row, "observed_at", "observation")
    summary = instant(run.get("summary", {}), "created_at", "summary")
    generations = {g["generation"]: g for g in run.get("generations", [])}
    count = run.get("resume_count")
    if type(count) is not int or count != len(generations) - 1 or not 0 <= count <= 2:
        fail("resume_generation_mismatch")
    if run["execution"].get("metering", {}).get("explicit_recoveries") != count:
        fail("resume_metering_mismatch")
    reused = sum(
        cp.get("generation", len(generations)) < len(generations)
        for cp in run.get("checkpoints", [])
    )
    if run["execution"].get("metering", {}).get("reused_branches") != reused:
        fail("reused_checkpoint_mismatch")
    windows = {}
    previous = None
    for number, g in sorted(generations.items()):
        if g.get("run_id") != run_id:
            fail("generation_parent_run")
        start = instant(g, "created_at", "generation")
        deadline = instant(g, "deadline_at", "generation")
        windows[number] = (start, deadline)
        if start and deadline and deadline - start != frozen_window:
            fail("generation_frozen_deadline")
        if start and (created and start < created or previous and start <= previous):
            fail("generation_creation_order")
        if number == 1 and start != created:
            fail("first_generation_created_at")
        if number < len(generations) and g.get("status") not in {
            "expired",
            "superseded",
        }:
            fail("resumed_generation_status")
        previous = start
    receipts = {}
    receipt_bounds = {}
    invalidations = {}
    previous_trace = causal_trace = None
    for event in run.get("events", []):
        if event.get("run_id") != run_id:
            fail("event_parent_run")
        if not run.get("trace_id") or event.get("trace_id") != run["trace_id"]:
            fail("event_trace_identity")
        at = instant(event, "occurred_at", "event")
        if at and previous_trace and at <= previous_trace:
            fail("event_trace_time_order")
        # Trace resolves tied clocks with exactly previous + 1us. Its causal bound
        # follows that chain; it is not a new wall-clock reading for the transaction.
        if at is not None and at != (
            previous_trace + timedelta(microseconds=1) if previous_trace else None
        ):
            causal_trace = at
        previous_trace = at
        if at and created and at < created:
            fail("event_before_run")
        number = event.get("generation")
        if number is not None and number not in generations:
            fail("event_generation_identity")
        if event.get("step") == "model_attempt":
            identity = (number, event.get("worker"), event.get("attempt"))
            receipts[identity] = at
            receipt_bounds[identity] = causal_trace
        body = event.get("output_summary", {})
        if body.get("status") in {"cancelled", "interrupted", "failed"}:
            invalidations.setdefault(number, []).append(at)
    attempts = {}
    for attempt in run.get("attempts", []):
        for key in WorkerAttemptRecord.model_fields:
            if key not in attempt:
                fail("attempt_missing:" + key)
        try:
            WorkerAttemptRecord.model_validate_json(json.dumps(attempt), strict=True)
        except ValidationError:
            fail("attempt_contract_type")
        identity = (
            attempt.get("generation"),
            attempt.get("worker"),
            attempt.get("attempt_no"),
        )
        start = instant(attempt, "started_at", "attempt")
        finish = instant(attempt, "finished_at", "attempt", optional=True)
        attempts[identity] = (attempt, start, finish)
        gen_start, deadline = windows.get(identity[0], (None, None))
        if start and gen_start and start < gen_start:
            fail("attempt_before_generation")
        if start and deadline and start >= deadline:
            fail("attempt_start_after_deadline")
        receipt = receipts.get(identity)
        if receipt and start and receipt < start:
            fail("receipt_before_request")
        if finish and start and finish < start:
            fail("attempt_finish_before_start")
        # Settlement after a deadline or interruption is legal; adoption is checked below.
        if receipt is not None and finish is None:
            fail("returned_attempt_not_settled")
        if observed and start and start > observed:
            fail("attempt_start_after_observation")
    adopted_times = []
    for cp in run.get("checkpoints", []):
        identity = (cp.get("generation"), cp.get("worker"), cp.get("attempt_no"))
        at = instant(cp, "adopted_at", "checkpoint")
        attempt, start, finish = attempts.get(identity, ({}, None, None))
        gen_start, deadline = windows.get(identity[0], (None, None))
        receipt = receipts.get(identity)
        if at:
            adopted_times.append(at)
            if start is None or start > at:
                fail("adoption_before_request")
            if receipt is None or receipt > at and receipt_bounds.get(identity) > at:
                fail("adoption_before_receipt")
            if finish != at:
                fail("adoption_transaction_time")
            if (
                gen_start is None
                or deadline is None
                or not gen_start <= at < deadline
                or at - gen_start >= frozen_window
            ):
                fail("adoption_generation_deadline")
            next_start = windows.get(identity[0] + 1, (None, None))[0]
            if next_start and at >= next_start:
                fail("adoption_superseded_generation")
            for stop in invalidations.get(identity[0], []) + invalidations.get(
                None, []
            ):
                if (
                    stop
                    and at >= stop
                    and (
                        registration["binding"].get("consumption_version")
                        != "v1-07g-consumption.v5"
                        or gen_start is not None
                        and stop >= gen_start
                    )
                ):
                    fail("adoption_after_invalidation")
        if (
            not attempt
            or attempt.get("outcome") != "completed"
            or attempt.get("abandoned")
            or attempt.get("discarded_late")
        ):
            fail("adopted_attempt_ineligible")
    if summary and any(summary < at for at in adopted_times):
        fail("summary_before_adoption")
    if summary and created and summary < created:
        fail("summary_before_run")
    if observed and (
        summary is None
        or observed < summary
        or any(observed < at for at in adopted_times)
    ):
        fail("observation_before_summary_or_adoption")
    return failures


def metering_failures(registration, run_id, run):
    from .precise_run_state import metering_from_ledger

    failures = []
    meter = run["execution"].get("metering", {})
    for key in RunMeteringV1.model_fields:
        if key not in meter:
            failures.append(f"metering_missing:{key}:{run_id}")
    try:
        RunMeteringV1.model_validate_json(json.dumps(meter), strict=True)
    except ValidationError:
        return failures + [f"metering_contract_type:{run_id}"]
    receipts = []
    for event in run["events"]:
        if event.get("step") != "model_attempt":
            continue
        body = event["output_summary"]
        for key in (
            "worker",
            "attempt_no",
            "outcome",
            "attempt_error",
            "request_sha256",
            "input_usage",
            "output_usage",
            "usage_unknown",
            "latency_ms",
            "next_correction_kind",
            "receipt_model",
        ):
            if key not in body:
                failures.append(f"receipt_missing:{key}:{run_id}")
        if (
            type(body.get("latency_ms")) is not int
            or not 0 <= body["latency_ms"] <= 3_600_000
        ):
            failures.append(f"receipt_latency_type:{run_id}")
        kind = body.get("next_correction_kind")
        if kind is not None and (
            type(kind) is not str
            or kind not in {"contract", "citation", "transport", "other"}
        ):
            failures.append(f"receipt_correction_type:{run_id}")
        if type(body.get("receipt_model")) is not str:
            failures.append(f"receipt_model_type:{run_id}")
        for key in ("input_usage", "output_usage"):
            value = body.get(key)
            if value is not None and (
                type(value) is not int or not 0 <= value <= 10**9
            ):
                failures.append(f"receipt_token_type:{key}:{run_id}")
        error = body.get("attempt_error")
        if error is not None and (type(error) is not str or not error):
            failures.append(f"receipt_error_type:{run_id}")
        if body.get("outcome") != ("completed" if error is None else "failed"):
            failures.append(f"receipt_outcome:{run_id}")
        if event.get("event_type") != (
            "step_completed" if error is None else "step_failed"
        ):
            failures.append(f"receipt_event_type:{run_id}")
        branch = run["execution"].get(event.get("worker"), {})
        if type(branch) is not dict:
            failures.append(f"execution_branch_type:{run_id}")
            continue
        for projected in branch.get("attempts", []):
            if projected.get("attempt_no") == event.get("attempt"):
                for key, source in (
                    ("latency_ms", "latency_ms"),
                    ("resolved_model", "receipt_model"),
                ):
                    if projected.get(key) != body.get(source):
                        failures.append(f"attempt_receipt_value:{key}:{run_id}")
        receipts.append({**body, "attempt": event.get("attempt")})
    if failures:
        return failures
    expected = metering_from_ledger(
        run["attempts"],
        receipts=receipts,
        requested_model=registration["binding"]["requested_model"],
        resolved_models=[
            item["receipt_model"] for item in receipts if item["receipt_model"]
        ],
        explicit_recoveries=run["resume_count"],
        reused_branches=sum(
            cp["generation"] < len(run["generations"]) for cp in run["checkpoints"]
        ),
    ).model_dump(mode="json")
    for key, value in expected.items():
        if type(meter.get(key)) is not type(value) or meter.get(key) != value:
            failures.append(f"ledger_mismatch:{key}:{run_id}")
    return failures


def binding_shape_failures(run_id, run):
    failures = []
    if type(run.get("resume_count")) is not int:
        failures.append(f"resume_count_type:{run_id}")
    if type(run.get("trace_id")) is not str or not run.get("trace_id"):
        failures.append(f"trace_identity_type:{run_id}")
    if type(run.get("status")) is not str:
        failures.append(f"run_status_type:{run_id}")
    failures.extend(generation_shape_failures(run_id, run))
    for event in run["events"]:
        for key in ("run_id", "trace_id", "step", "generation", "attempt", "worker"):
            if key not in event:
                failures.append(f"event_missing:{key}:{run_id}")
        for key, adapter in EVENT_FIELDS_V4.items():
            try:
                adapter.validate_json(json.dumps(event.get(key)), strict=True)
            except ValidationError:
                failures.append(f"event_field_type:{key}:{run_id}")
        if type(event.get("trace_id")) is not str or not event.get("trace_id"):
            failures.append(f"trace_identity_type:{run_id}")
        body = event["output_summary"]
        if type(event.get("event_type")) is str and event["event_type"] in {
            "confirmation_recorded",
            "human_action_recorded",
        }:
            try:
                TypeAdapter(ApprovalActionType).validate_json(
                    json.dumps(body.get("action")), strict=True
                )
            except ValidationError:
                failures.append(f"event_action_type:{run_id}")
        if "status" in body and type(body["status"]) is not str:
            failures.append(f"event_status_type:{run_id}")
    return failures


def verify_sweep_binding_v4(registration, observations, bindings, **kwargs):
    from .eval_g_consumption import verify_versioned_binding

    return verify_versioned_binding(registration, observations, bindings, **kwargs)


def _verify_sweep_binding_legacy_v4(registration, observations, bindings, **kwargs):
    from .eval_f_lineage import _verify_sweep_binding_legacy_v3
    from .precise_dev_paired_metering import validate_score_input_v3

    validate_score_input_v3(registration, observations, bindings)
    shape = adoption_evidence_shape_failures(registration, bindings.get("runs", {}))
    if shape:
        return incomplete_binding_v4(registration, observations, bindings, shape)
    failures = []
    manifest = registration.get("binding", {}).get("adoption_evidence", {})
    if manifest.get("manifest_version") != "v1-07g-adoption-evidence.v1":
        failures.append("adoption_evidence_manifest_version:")
    for location, record, field in (
        ("registration", manifest, "source_file_sha256"),
        ("binding", bindings, "source_evidence_file_sha256"),
    ):
        if field not in record:
            failures.append("adoption_evidence_source_missing:" + location)
        elif type(record[field]) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", record[field]
        ):
            failures.append("adoption_evidence_source_type:" + location)
    if manifest.get("source_file_sha256") != bindings.get(
        "source_evidence_file_sha256"
    ):
        failures.append("adoption_evidence_source:")
    if manifest.get("observations_payload_sha256") != payload_sha256(observations):
        failures.append("adoption_evidence_observations:")
    if set(manifest.get("runs", {})) != set(bindings.get("runs", {})):
        failures.append("adoption_evidence_run_set:")
    for row in observations["observations"]:
        run_id = row["run_id"]
        run = bindings.get("runs", {}).get(run_id)
        if run:
            missing = set(
                (
                    "case_id",
                    "snapshot",
                    "created_at",
                    "resume_count",
                    "summary",
                    "execution",
                    "attempts",
                    "events",
                    "checkpoints",
                    "generations",
                )
            ) - set(run)
            if missing:
                failures.extend(
                    f"run_missing:{key}:{run_id}" for key in sorted(missing)
                )
                continue
            shape = binding_shape_failures(run_id, run)
            failures.extend(shape)
            if shape:
                continue
            contract = checkpoint_failures(registration, run_id, run)
            failures.extend(contract)
            if any(code.startswith("checkpoint_record_type:") for code in contract):
                continue
            failures.extend(causal_failures(registration, run_id, run, row))
            failures.extend(metering_failures(registration, run_id, run))
    if any(
        "_missing:" in code or "_type" in code or "_time:" in code for code in failures
    ):
        result = incomplete_binding_v4(registration, observations, bindings, [])
    else:
        result = _verify_sweep_binding_legacy_v3(
            registration,
            observations,
            bindings,
            expected_binding_version=BINDING_V4,
            **kwargs,
        )
    failures = result["failures"] + failures
    result["failures"] = failures
    result["complete"] = not failures
    result["binding_version"] = BINDING_V4
    return result


def main_sweep_cli() -> None:
    from .eval_g_cli import main_sweep_cli as run_cli

    run_cli()
