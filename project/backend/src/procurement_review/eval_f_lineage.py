"""Explicit reuse of unchanged diagnostic evidence after runtime-only repairs."""

import json
from pathlib import Path

from .eval_attempt_ledger import LedgerError
from .eval_g_plan import validate_adoption_evidence
from .gold_v3_contracts import payload_sha256
from .eval_g_consumption import verify_versioned_binding as verify_sweep_binding_v3

ARMS = ("minimum", "baseline", "compact", "reasoning", "concurrency")
RUNTIME_ONLY = {
    "eval_f_budget.py",
    "eval_f_registration.py",
    "eval_f_run.py",
    "eval_f_selection.py",
    "eval_f_lineage.py",
}


def diagnostic_directory(root, registered, name):
    predecessor = registered.get("diagnostic_predecessor")
    if not predecessor or name not in ARMS:
        return Path(root) / payload_sha256(registered) / name
    directory = Path(root) / predecessor / name
    old = json.loads((directory / "registration.json").read_text())
    if payload_sha256(old) != predecessor:
        raise LedgerError("diagnostic predecessor registration is corrupt")
    if any(
        old[key] != registered[key]
        for key in ("data", "versions", "gold_manifest_sha256")
    ):
        raise LedgerError("diagnostic data or contracts changed")
    if old["candidates"][name] != registered["candidates"][name]:
        raise LedgerError("diagnostic candidate changed")
    keys = set(old["implementation"]) | set(registered["implementation"])
    changed = {
        k
        for k in keys
        if old["implementation"].get(k) != registered["implementation"].get(k)
    }
    if changed - RUNTIME_ONLY:
        raise LedgerError("diagnostic quality evidence invalidated by implementation")
    from .eval_f_run import replay

    previous = json.loads((directory / "result.json").read_text())
    now = replay(directory)
    if any(previous[k] != value for k, value in now.items()):
        raise LedgerError("diagnostic re-score differs; adoption refused")
    return directory


def bind_sweep_v3(
    state_path,
    registration_path,
    observations_path,
    output_path,
    *,
    binding_version="v1-07g-db-binding.v3",
):
    """Read immutable execution evidence and later event chains in one DB snapshot."""
    import hashlib
    from datetime import UTC, datetime
    from sqlalchemy import create_engine, text

    from .precise_dev_sweep import _summarise

    state = json.loads(Path(state_path).read_text())
    registration = json.loads(Path(registration_path).read_text())
    from .precise_candidate import verify_registration_v3

    errors = verify_registration_v3(registration)
    if errors:
        raise ValueError("v3_registration:" + ",".join(errors))
    original_bytes = Path(observations_path).read_bytes()
    observations = json.loads(original_bytes)
    if binding_version == "v1-07g-db-binding.v4":
        validate_adoption_evidence(
            registration, (r["run_id"] for r in observations["observations"])
        )
    engine = create_engine(
        state["database_url"].split("?", 1)[0], isolation_level="REPEATABLE READ"
    )
    runs = {}
    v4 = binding_version == "v1-07g-db-binding.v4"
    generation_parent = "run_id," if v4 else ""
    trace_parent = "run_id,trace_id," if v4 else ""

    def serial(value):
        return json.loads(json.dumps(value, default=lambda item: item.isoformat()))

    try:
        with engine.connect() as connection, connection.begin():
            connection.execute(text("SET TRANSACTION READ ONLY"))
            for observed in observations["observations"]:
                run_id = observed["run_id"]
                params = {"run": run_id}
                row = (
                    connection.execute(
                        text("SELECT * FROM review_run WHERE run_id=:run"), params
                    )
                    .mappings()
                    .one()
                )
                summary = (
                    connection.execute(
                        text("SELECT * FROM review_run_summary WHERE run_id=:run"),
                        params,
                    )
                    .mappings()
                    .one()
                )
                execution = summary["execution_payload"]
                snapshot = row["isolation"]["execution_snapshot"]
                attempts = (
                    connection.execute(
                        text(
                            "SELECT * FROM review_worker_attempt WHERE run_id=:run ORDER BY generation,worker,attempt_no"
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
                checkpoints = (
                    connection.execute(
                        text(
                            "SELECT * FROM review_branch_checkpoint WHERE run_id=:run ORDER BY worker"
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
                generations = (
                    connection.execute(
                        text(
                            f"SELECT {generation_parent}generation,status,created_at,deadline_at FROM review_run_generation WHERE run_id=:run ORDER BY generation"
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
                events = (
                    connection.execute(
                        text(
                            f"SELECT {trace_parent}event_ordinal,event_type,step,occurred_at,generation,attempt,worker,output_summary,output_checksum FROM review_trace_event WHERE run_id=:run ORDER BY event_ordinal"
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )
                projection = _summarise(
                    {
                        "execution": execution,
                        "status": observed["status"],
                        "summary": summary["summary_payload"],
                        "execution_mode": snapshot["mode"],
                        "error_code": row["error_code"],
                    }
                )
                runs[run_id] = {
                    "case_id": row["case_id"],
                    **({"trace_id": row["trace_id"]} if v4 else {}),
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "resume_count": row["resume_count"],
                    "snapshot": snapshot,
                    "projection": projection,
                    "execution": execution,
                    "summary": {
                        key: summary[key]
                        for key in ("summary_checksum", "summary_payload", "created_at")
                    },
                    "attempts": [dict(item) for item in attempts],
                    "checkpoints": [dict(item) for item in checkpoints],
                    "generations": [dict(item) for item in generations],
                    "events": [dict(item) for item in events],
                }
    finally:
        engine.dispose()
    payload = serial(
        {
            "binding_version": binding_version,
            "database": state["database"],
            "generated_at": datetime.now(UTC),
            "base_registration_sha256": registration["extends"][
                "base_registration_sha256"
            ],
            "observations_file_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "observations_payload_sha256": payload_sha256(observations),
            "runs": runs,
        }
    )
    if binding_version == "v1-07g-db-binding.v4":
        payload["source_evidence_file_sha256"] = registration["binding"][
            "adoption_evidence"
        ]["source_file_sha256"]
        if registration["binding"].get("consumption_version"):
            payload["consumption_version"] = registration["binding"][
                "consumption_version"
            ]
        verdict = verify_sweep_binding_v3(registration, observations, payload)
        if not verdict["complete"]:
            raise ValueError("v4_binding_rejected:" + ",".join(verdict["failures"]))
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return {
        "written": str(output_path),
        "runs": len(runs),
        "binding_version": payload["binding_version"],
    }


def execution_projection_v3(row, binding):
    """Add fields dropped by the old sweep, without changing any original value."""
    import copy

    enriched = copy.deepcopy(row)
    execution = binding["execution"]
    for worker, branch in row.get("branches", {}).items():
        native = execution.get(worker, {})
        enriched["branches"][worker].update(
            request_sha256=native.get("request_sha256"),
            answer_sha256=native.get("answer_sha256"),
        )
    full = {}
    for worker in ("policy", "contract"):
        for span in (execution.get(worker) or {}).get("spans", []):
            prop = span["proposition"]
            full[
                (
                    worker,
                    prop["check_id"],
                    prop["field_path"],
                    span["start"],
                    span["end"],
                )
            ] = span
    for span in enriched.get("spans", []):
        native = full.get(
            (
                span["worker"],
                span["check_id"],
                span["field_path"],
                span["start"],
                span["end"],
            )
        )
        if native is not None:
            for field in ("effective_at", "scope_id", "subject_type"):
                span.setdefault(field, native["proposition"][field])
            span.setdefault("content_sha256", native["evidence"]["content_sha256"])
    enriched.setdefault("fact_observations", native_facts_v3(execution))
    return enriched


def native_facts_v3(execution):
    return [
        item
        for item in execution.get("decision", {}).get("observations", [])
        if not item["proposition"]["field_path"].startswith(("policy.", "contract."))
    ]


def facts_to_judge_v3(extension, observed):
    """Missing registered propositions remain failed samples; extra claims remain too."""
    remaining = list(observed)
    for duty, paths in extension["required_facts"].items():
        for path in paths:
            matches = [
                item
                for item in remaining
                if item["proposition"].get("field_path") == path
                and item["proposition"].get("check_id") == duty
            ]
            if len(matches) == 1:
                yield matches[0]
                remaining.remove(matches[0])
            else:
                yield {"proposition": {"check_id": duty, "field_path": path}}
    yield from remaining


def _verify_sweep_binding_legacy_v3(
    registration,
    observations,
    bindings,
    *,
    expected_binding_version="v1-07g-db-binding.v3",
    **_,
):
    """Cross-check identities and settlement, never just equal aggregate counts."""
    import re
    from datetime import datetime
    from .precise_dev_paired_metering import (
        _planned_pairs,
        REQUIRED_METERING_KEYS,
        service_latency_v3,
        receipt_outcome_matches_v4,
    )
    from .trace import canonical_checksum
    from .eval_g_plan import terminal_generation_states

    rows = observations["observations"]
    planned = _planned_pairs(registration)
    pairs = [(row.get("case_id"), row.get("repeat")) for row in rows]
    failures, unknown_details = [], []
    from .precise_candidate import verify_registration_v3

    failures.extend(verify_registration_v3(registration))

    def fail(code, run=""):
        failures.append(f"{code}:{run}")

    for pair in planned:
        if pairs.count(pair) != 1:
            fail("planned_pair_count", str(pair))
    if set(pairs) - set(planned):
        fail("unplanned_pairs")
    run_ids = [row.get("run_id") for row in rows]
    if len(set(run_ids)) != len(run_ids) or any(not run for run in run_ids):
        fail("duplicate_or_missing_run")
    bindings = bindings or {}
    if bindings.get("binding_version") != expected_binding_version:
        fail("binding_version")
    if bindings.get("observations_payload_sha256") != payload_sha256(observations):
        fail("raw_observation_binding")
    if (
        bindings.get("base_registration_sha256")
        != registration["extends"]["base_registration_sha256"]
    ):
        fail("base_registration_binding")
    binding = registration["binding"]
    branch_names = binding["branches"]
    runs = bindings.get("runs", {})
    if set(runs) - set(run_ids):
        fail("unplanned_binding_runs")
    allowed_terminal = {
        "pending_confirmation",
        "pending_human_review",
        "confirmed",
        "rejected",
        "cancelled",
        "failed",
        "interrupted",
    }
    for row in rows:
        run_id = row.get("run_id")
        if row["case_id"] not in registration["case_ids"]:
            fail("unplanned_case", run_id)
            continue
        run = runs.get(run_id)
        if not run:
            fail("binding_missing_run", run_id)
            continue
        if (
            row.get("terminal_status") != 200
            or row.get("status") not in allowed_terminal
        ):
            fail("not_terminal", run_id)
        if run.get("case_id") != row["case_id"]:
            fail("run_case", run_id)
        for field, expected in (
            ("execution_version", binding["expected_execution_version"]),
            ("summary_schema", binding["expected_summary_schema"]),
            ("rules_version", binding["expected_rules_version"]),
        ):
            if row.get(field) != expected:
                fail(f"projection_version:{field}", run_id)
        projection = run.get("projection", {})
        for key in (
            "status",
            "recommendation",
            "findings",
            "spans",
            "branches",
            "metering",
            "conflicts",
            "security",
        ):
            if key not in projection:
                fail(f"execution_projection_missing:{key}", run_id)
        for key, expected in projection.items():
            if row.get(key) != expected:
                fail(f"execution_projection:{key}", run_id)
        summary = run.get("summary")
        if not summary:
            fail("summary_missing", run_id)
        elif summary["summary_checksum"] != canonical_checksum(
            summary["summary_payload"]
        ):
            fail("summary_digest", run_id)
        elif (
            summary["summary_payload"].get("run_id") != run_id
            or summary["summary_payload"].get("case_id") != row["case_id"]
        ):
            fail("summary_identity", run_id)
        execution = run.get("execution", {})
        if summary and execution.get("summary") != summary["summary_payload"]:
            fail("execution_summary_mismatch", run_id)
        from .precise_dev_sweep import _summarise

        recomputed = _summarise(
            {
                "execution": execution,
                "status": row.get("status"),
                "summary": (summary or {}).get("summary_payload"),
                "execution_mode": run.get("snapshot", {}).get("mode"),
                "error_code": projection.get("error_code"),
            }
        )
        if recomputed != projection:
            fail("execution_projection_recomputed", run_id)
        if (
            execution.get("run_id") != run_id
            or execution.get("case_id") != row["case_id"]
        ):
            fail("execution_identity", run_id)
        facts = native_facts_v3(execution)
        required = registration["truth_extension"][row["case_id"]]["required_facts"]
        identities = [
            (fact["proposition"].get("check_id"), fact["proposition"].get("field_path"))
            for fact in facts
        ]
        if sorted(identities) != sorted(
            (duty, path) for duty, paths in required.items() for path in paths
        ):
            fail("planned_fact_identity", run_id)
        if "fact_observations" in row and row["fact_observations"] != facts:
            fail("fact_projection_mismatch", run_id)
        if service_latency_v3(row) is None:
            fail("service_time_invalid", run_id)
        elif datetime.fromisoformat(
            row["service_created_at"]
        ) != datetime.fromisoformat(run["created_at"]):
            fail("service_created_at", run_id)
        snapshot = run.get("snapshot", {})
        for name, expected in binding.get("expected_snapshot_fields", {}).items():
            if (
                name not in snapshot
                or type(snapshot[name]) is not type(expected)
                or snapshot[name] != expected
            ):
                fail(f"run_binding:{name}", run_id)
        for name, expected in (
            ("implementation_sha256", binding["product_implementation_sha256"]),
            ("mode", "real"),
            ("rules_version", binding["expected_rules_version"]),
            ("document_version", binding["expected_document_version"]),
            ("dataset_version", binding["expected_dataset_version"]),
            ("index_version", binding["expected_index_version"]),
            ("requested_model", binding["requested_model"]),
            ("resolved_model", binding["resolved_model"]),
            ("max_attempts_per_branch", binding["max_attempts_per_branch"]),
        ):
            if snapshot.get(name) != expected:
                fail(f"run_binding:{name}", run_id)
        generations = {
            item.get("generation"): item for item in run.get("generations", [])
        }
        if (
            len(generations) != len(run.get("generations", []))
            or any(type(g) is not int for g in generations)
            or set(generations) != set(range(1, len(generations) + 1))
            or len(generations) > 3
        ):
            fail("generation_identity", run_id)
        if not generations or any(
            item.get("status")
            not in terminal_generation_states(expected_binding_version)
            for item in generations.values()
        ):
            fail("generation_not_terminal", run_id)
        for generation in generations.values():
            window = service_latency_v3(
                {
                    "service_created_at": generation.get("created_at"),
                    "observed_at": generation.get("deadline_at"),
                }
            )
            seconds = snapshot.get("generation_deadline_seconds")
            if (
                type(seconds) not in {int, float}
                or not 0 < seconds <= 120
                or window is None
                or abs(window - seconds * 1000) > 50
            ):
                fail("frozen_generation_deadline", run_id)
        attempts = run.get("attempts", [])
        keys = [
            (item.get("generation"), item.get("worker"), item.get("attempt_no"))
            for item in attempts
        ]
        if len(set(keys)) != len(keys):
            fail("duplicate_attempt", run_id)
        receipts = [
            event
            for event in run.get("events", [])
            if event.get("step") == "model_attempt"
        ]
        receipt_keys = [
            (event.get("generation"), event.get("worker"), event.get("attempt"))
            for event in receipts
        ]
        if len(set(receipt_keys)) != len(receipt_keys) or set(receipt_keys) - set(keys):
            fail("model_receipt_identity", run_id)
        checkpoints = {item["worker"]: item for item in run.get("checkpoints", [])}
        expected_checkpoints = {
            worker
            for worker in branch_names
            if expected_binding_version != "v1-07g-db-binding.v4"
            or execution.get(worker, {}).get("outcome") == "completed"
        }
        if (
            len(checkpoints) != len(run.get("checkpoints", []))
            or set(checkpoints) != expected_checkpoints
        ):
            fail("checkpoint_branches", run_id)
        proven = unknown = abandoned = opened = first = known_input = known_output = 0
        for item in attempts:
            key = (item.get("generation"), item.get("worker"), item.get("attempt_no"))
            worker = key[1]
            if item.get("run_id") != run_id:
                fail("attempt_parent_run", run_id)
            if (
                service_latency_v3(
                    {
                        "service_created_at": item.get("started_at"),
                        "observed_at": item.get("finished_at"),
                    }
                )
                is None
                and item.get("finished_at") is not None
            ):
                fail("attempt_time_invalid", run_id)
            if (
                worker not in branch_names
                or key[0] not in generations
                or type(key[2]) is not int
                or key[2] < 1
            ):
                fail("attempt_identity", run_id)
                continue
            matches = [
                event
                for event, identity in zip(receipts, receipt_keys)
                if identity == key
            ]
            if len(matches) != 1:
                if matches or not (
                    item.get("abandoned")
                    or item.get("outcome") in {"interrupted", "cancelled"}
                ):
                    fail("attempt_receipt_missing", run_id)
                unknown += 1
                unknown_details.append(
                    {
                        "run_id": run_id,
                        "generation": key[0],
                        "worker": worker,
                        "attempt_no": key[2],
                        "reason": "unreturned_abandoned"
                        if item.get("finished_at")
                        else "unreturned_open",
                    }
                )
                abandoned += int(item.get("finished_at") is not None)
                opened += int(item.get("finished_at") is None)
                continue
            event = matches[0]
            receipt = event["output_summary"]
            proven += 1
            first += key[2] == 1
            if event.get("output_checksum") != canonical_checksum(receipt):
                fail("receipt_digest", run_id)
            if (
                receipt.get("worker") != worker
                or receipt.get("attempt_no") != key[2]
                or not receipt_outcome_matches_v4(
                    item, receipt, expected_binding_version
                )
            ):
                fail("receipt_row_mismatch", run_id)
            request = receipt.get("request_sha256")
            if not isinstance(request, str) or not re.fullmatch(
                "[0-9a-f]{64}", request
            ):
                fail("request_identity", run_id)
            branch = run.get("execution", {}).get(worker, {})
            if request != branch.get("request_sha256"):
                fail("request_branch_mismatch", run_id)
            if request != binding.get("expected_requests", {}).get(
                row["case_id"], {}
            ).get(worker):
                fail("request_plan_mismatch", run_id)
            projected = [
                attempt
                for attempt in branch.get("attempts", [])
                if attempt.get("attempt_no") == key[2]
            ]
            if projected:
                attempt = projected[0]
                for observed_key, receipt_key in (
                    ("input_tokens", "input_usage"),
                    ("output_tokens", "output_usage"),
                    ("usage_unknown", "usage_unknown"),
                    ("error_code", "attempt_error"),
                ):
                    from .eval_g_consumption import receipt_value_matches

                    if not receipt_value_matches(
                        item,
                        attempt,
                        receipt,
                        observed_key,
                        receipt_key,
                        expected_binding_version,
                        consumption_version=binding.get("consumption_version"),
                    ):
                        fail(f"attempt_receipt_value:{observed_key}", run_id)
            if receipt.get("usage_unknown") is True:
                unknown += 1
                unknown_details.append(
                    {
                        "run_id": run_id,
                        "generation": key[0],
                        "worker": worker,
                        "attempt_no": key[2],
                        "reason": receipt.get("attempt_error")
                        or "model_receipt_usage_unavailable",
                    }
                )
                if (
                    receipt.get("input_usage") is not None
                    or receipt.get("output_usage") is not None
                ):
                    fail("unknown_usage_has_known_tokens", run_id)
            elif receipt.get("usage_unknown") is False:
                tokens = [receipt.get("input_usage"), receipt.get("output_usage")]
                if any(type(value) is not int or value < 0 for value in tokens):
                    fail("known_tokens_missing", run_id)
                else:
                    known_input += tokens[0]
                    known_output += tokens[1]
            else:
                fail("usage_marker_missing", run_id)
            if receipt.get("receipt_model") != snapshot.get("resolved_model"):
                fail("receipt_model", run_id)
        observed_branches = row.get("branches", {})
        if set(observed_branches) != set(branch_names):
            fail("planned_branch_missing", run_id)
        for worker in branch_names:
            own = [item for item in attempts if item.get("worker") == worker]
            numbers = sorted(item["attempt_no"] for item in own)
            if (
                numbers != list(range(1, len(numbers) + 1))
                or len(numbers) > binding["max_attempts_per_branch"]
            ):
                fail(f"cross_generation_branch_budget:{worker}", run_id)
            checkpoint = checkpoints.get(worker)
            if checkpoint:
                if checkpoint.get("run_id") != run_id:
                    fail("checkpoint_parent_run", run_id)
                identity = (checkpoint["generation"], worker, checkpoint["attempt_no"])
                if (
                    identity not in keys
                    or canonical_checksum(checkpoint["result_payload"])
                    != checkpoint["result_checksum"]
                ):
                    fail("checkpoint_identity_or_digest", run_id)
                native = run["execution"].get(worker, {})
                if checkpoint["input_checksum"] != "sha256:" + str(
                    native.get("request_sha256")
                ):
                    fail("checkpoint_request", run_id)
                result = checkpoint["result_payload"]["result"]
                if (
                    result.get("worker") != worker
                    or result.get("spans") != native.get("spans")
                    or result.get("attempts") != native.get("attempts")
                ):
                    fail("adopted_result_mismatch", run_id)
                adopted_attempt = next(
                    (
                        item
                        for item in attempts
                        if (
                            item.get("generation"),
                            item.get("worker"),
                            item.get("attempt_no"),
                        )
                        == identity
                    ),
                    None,
                )
                if (
                    adopted_attempt is None
                    or adopted_attempt.get("abandoned")
                    or adopted_attempt.get("discarded_late")
                    or adopted_attempt.get("outcome") != "completed"
                ):
                    fail("adopted_attempt_ineligible", run_id)
                if "adopted_at" in checkpoint:
                    generation = generations.get(checkpoint["generation"])
                    adopted = datetime.fromisoformat(checkpoint["adopted_at"])
                    if generation is None or not (
                        datetime.fromisoformat(generation["created_at"])
                        <= adopted
                        <= datetime.fromisoformat(generation["deadline_at"])
                    ):
                        fail("adoption_generation_deadline", run_id)
        metering = row.get("metering") or {}
        expected_counts = {
            "attempts_opened": len(attempts),
            "physical_attempts": len(attempts),
            "proven_sends": proven,
            "first_sends": first,
            "unresolved_attempts": abandoned,
            "open_attempts": opened,
            "unreturned_attempts": len(attempts) - proven,
            "unknown_usage_attempts": unknown,
            "usage_complete": unknown == 0 and bool(attempts),
            "input_tokens_known": known_input if proven else None,
            "output_tokens_known": known_output if proven else None,
        }
        for key, expected in expected_counts.items():
            if (
                key not in metering
                or type(metering[key]) is not type(expected)
                or metering[key] != expected
            ):
                fail(f"ledger_mismatch:{key}", run_id)
        for key in REQUIRED_METERING_KEYS:
            if key not in metering:
                fail(f"metering_key_missing:{key}", run_id)
        events = run.get("events", [])
        if [event["event_ordinal"] for event in events] != list(
            range(1, len(events) + 1)
        ):
            fail("event_chain_gap", run_id)
        at = (
            datetime.fromisoformat(row["observed_at"])
            if service_latency_v3(row) is not None
            else None
        )
        statuses = [
            (
                datetime.fromisoformat(event["occurred_at"]),
                event["output_summary"]["status"],
            )
            for event in events
            if "status" in event.get("output_summary", {})
        ]
        prior = [status for time, status in statuses if at is not None and time <= at]
        if not prior or prior[-1] != row.get("status"):
            fail("execution_status_at_observation", run_id)
        current = prior[-1] if prior else None
        transitions = {
            "confirm": "confirmed",
            "reject": "rejected",
            "cancel": "cancelled",
        }
        for event in events:
            if at is None or datetime.fromisoformat(event["occurred_at"]) <= at:
                continue
            body = event.get("output_summary", {})
            if event["event_type"] in {
                "confirmation_recorded",
                "human_action_recorded",
            }:
                expected = transitions.get(body.get("action"))
                if (
                    current not in {"pending_confirmation", "pending_human_review"}
                    or body.get("status") != expected
                ):
                    fail("illegal_post_observation_transition", run_id)
                current = expected
        if run.get("status") != current:
            fail("post_observation_state", run_id)
    return {
        "planned_pairs": len(planned),
        "planned_model_tasks": len(planned) * len(branch_names),
        "unknown_attempts": unknown_details,
        "observed_pairs": len(rows),
        "binding_runs": len(runs),
        "failures": failures,
        "complete": not failures,
    }
