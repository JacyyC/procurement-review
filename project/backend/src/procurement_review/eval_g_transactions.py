"""Production transaction causality and attempt settlement for G consumption v5."""

from datetime import datetime, timedelta

from .attempt_metering import TRANSPORT_RETRY_CODES
from .trace import canonical_checksum


def transaction_failures(run_id, run):
    failures = []

    def fail(code):
        failures.append(f"{code}:{run_id}")

    def at(record, key):
        value = record.get(key)
        return datetime.fromisoformat(value) if value is not None else None

    events = run["events"]
    bounds = {}
    previous = causal = None
    for event in events:
        if event["step"] not in {
            "review_run",
            "validate_input",
            "create_run",
            "model_attempt",
            "deterministic_aggregator",
            "review_workflow",
            "review_recovery",
            "human_gate",
        }:
            fail("trace_step_contract")
        kinds = {
            "review_run": {"run_started", "run_completed", "run_cancelled"},
            "validate_input": {"step_completed"},
            "create_run": {"step_completed"},
            "model_attempt": {"step_completed", "step_failed"},
            "deterministic_aggregator": {"step_completed"},
            "review_workflow": {"step_completed", "step_failed"},
            "review_recovery": {"step_completed"},
            "human_gate": {"confirmation_recorded", "human_action_recorded"},
        }
        if event["event_type"] not in kinds.get(event["step"], set()):
            fail("trace_event_contract")
        if event["step"] == "review_workflow" and event["event_type"] == "step_failed":
            if (
                event["output_summary"]
                not in (
                    {"status": "interrupted"},
                    {"status": "interrupted", "reason": "lease_expired"},
                )
                or event["worker"] is not None
                or event["attempt"] is not None
            ):
                fail("trace_event_contract")
        time = at(event, "occurred_at")
        if time != (previous + timedelta(microseconds=1) if previous else None):
            causal = time
        bounds[event["event_ordinal"]] = causal
        previous = time
        if event.get("output_checksum") != canonical_checksum(event["output_summary"]):
            fail("trace_output_digest:" + event["step"])

    def select(step, kind=None):
        return [
            event
            for event in events
            if event["step"] == step and (kind is None or event["event_type"] == kind)
        ]

    created = at(run, "created_at")
    for step, kind, body in (
        ("review_run", "run_started", {}),
        ("validate_input", "step_completed", {"status": "valid"}),
        ("create_run", "step_completed", {"status": "processing"}),
    ):
        rows = select(step, kind) if step == "review_run" else select(step)
        if len(rows) != 1:
            fail("trace_required_event:" + step)
            continue
        event = rows[0]
        if event["event_type"] != kind or event["output_summary"] != body:
            fail("trace_event_content:" + step)
        if (
            event["generation"] is not None
            or event["worker"] is not None
            or event["attempt"] is not None
        ):
            fail("trace_creation_identity:" + step)
        if bounds[event["event_ordinal"]] != created:
            fail("trace_creation_transaction:" + step)
    creation = [
        select("review_run", "run_started"),
        select("validate_input"),
        select("create_run"),
    ]
    if all(len(rows) == 1 for rows in creation) and [
        rows[0]["event_ordinal"] for rows in creation
    ] != [1, 2, 3]:
        fail("trace_creation_order")

    generations = {g["generation"]: g for g in run["generations"]}
    summary_at = at(run["summary"], "created_at")
    summary = run["summary"]["summary_payload"]
    aggregates = select("deterministic_aggregator")
    workflow = select("review_workflow", "step_completed")
    if len(aggregates) != 1:
        fail("trace_required_event:deterministic_aggregator")
    if len(workflow) != 1:
        fail("trace_required_event:summary_workflow")
    if len(aggregates) == 1:
        event = aggregates[0]
        number = event["generation"]
        if (
            type(number) is not int
            or number not in generations
            or number != max(generations)
        ):
            fail("summary_generation_identity")
        else:
            generation = generations[number]
            if (
                not at(generation, "created_at")
                <= summary_at
                < at(generation, "deadline_at")
            ):
                fail("summary_generation_window")
            for stop in events:
                if stop["generation"] in {number, None} and stop["output_summary"].get(
                    "status"
                ) in {"interrupted", "cancelled", "failed"}:
                    if (
                        at(generation, "created_at")
                        <= bounds[stop["event_ordinal"]]
                        <= summary_at
                    ):
                        fail("summary_after_invalidation")
        expected = {
            "recommendation": summary["recommendation"],
            "highest_risk": summary["highest_risk"],
            "finding_count": len(summary["findings"]),
            "missing_count": len(summary["missing_information"]),
            "conflict_count": len(summary["conflicts"]),
            "failure_count": len(summary["failure_reasons"]),
        }
        if event["event_type"] != "step_completed" or canonical_checksum(
            event["output_summary"]
        ) != canonical_checksum(expected):
            fail("trace_aggregator_content")
        if event["worker"] is not None or event["attempt"] is not None:
            fail("trace_aggregator_identity")
        ordinal = event["event_ordinal"]
        previous_at = (
            at(events[ordinal - 2], "occurred_at") if ordinal > 1 else summary_at
        )
        if at(event, "occurred_at") != max(
            summary_at, previous_at + timedelta(microseconds=1)
        ):
            fail("trace_summary_transaction")
        for cp in run["checkpoints"]:
            matching = [
                e
                for e in select("model_attempt")
                if (e["generation"], e["worker"], e["attempt"])
                == (cp["generation"], cp["worker"], cp["attempt_no"])
            ]
            if len(matching) != 1 or matching[0]["event_ordinal"] >= ordinal:
                fail("summary_before_adopted_receipt")
        if len(workflow) == 1:
            next_event = workflow[0]
            status = (
                "pending_confirmation"
                if summary["human_review_status"] == "pending_confirmation"
                else "pending_human_review"
            )
            if (
                next_event["event_ordinal"] != ordinal + 1
                or next_event["generation"] != number
                or next_event["worker"] is not None
                or next_event["attempt"] is not None
                or next_event["output_summary"] != {"status": status}
                or at(next_event, "occurred_at")
                != at(event, "occurred_at") + timedelta(microseconds=1)
            ):
                fail("trace_summary_workflow_transaction")
    recovery = select("review_recovery")
    if len(recovery) != len(generations) - 1:
        fail("trace_recovery_count")
    interruptions = select("review_workflow", "step_failed")
    if len(interruptions) != len(generations) - 1:
        fail("trace_interruption_count")
    for number, generation in sorted(generations.items()):
        if number == 1:
            continue
        start = at(generation, "created_at")
        rows = [e for e in recovery if e["generation"] == number]
        if len(rows) != 1:
            fail("trace_recovery_identity")
            continue
        event = rows[0]
        if event["worker"] is not None or event["attempt"] is not None:
            fail("trace_recovery_identity")
        if (
            event["event_type"] != "step_completed"
            or event["output_summary"] != {"status": "processing"}
            or bounds[event["event_ordinal"]] != start
        ):
            fail("trace_recovery_transaction")
        stops = [
            e
            for e in events
            if e["generation"] in {number - 1, None}
            and e["step"] == "review_workflow"
            and e["event_type"] == "step_failed"
            and e["output_summary"].get("status") == "interrupted"
            and e["event_ordinal"] < event["event_ordinal"]
            and bounds[e["event_ordinal"]] <= start
            and bounds[e["event_ordinal"]] >= at(generations[number - 1], "created_at")
        ]
        if len(stops) != 1:
            fail("trace_recovery_precondition")
    failures.extend(human_transaction_failures(run_id, run, bounds))
    failures.extend(attempt_settlement_failures(run_id, run, bounds))
    return failures


def human_transaction_failures(run_id, run, bounds):
    failures = []
    events = run["events"]
    actions = [event for event in events if event["step"] == "human_gate"]
    finals = [
        event
        for event in events
        if event["step"] == "review_run"
        and event["event_type"] in {"run_cancelled", "run_completed"}
    ]
    if len(actions) != len(finals) or len(actions) > 1:
        failures.append("trace_human_pair_count:" + run_id)
    for event in actions:
        action = event["output_summary"].get("action")
        status = {
            "confirm": "confirmed",
            "reject": "rejected",
            "cancel": "cancelled",
        }.get(action)
        ordinal = event["event_ordinal"]
        expected_type = (
            "confirmation_recorded" if action == "confirm" else "human_action_recorded"
        )
        final = events[ordinal] if ordinal < len(events) else {}
        if (
            status is None
            or event["event_type"] != expected_type
            or event["output_summary"] != {"action": action, "status": status}
            or event["generation"] is not None
            or event["worker"] is not None
            or event["attempt"] is not None
            or final.get("step") != "review_run"
            or final.get("event_type")
            != ("run_cancelled" if action == "cancel" else "run_completed")
            or final.get("output_summary") != {"status": status}
            or final.get("generation") is not None
            or final.get("worker") is not None
            or final.get("attempt") is not None
            or datetime.fromisoformat(final["occurred_at"])
            != datetime.fromisoformat(event["occurred_at"]) + timedelta(microseconds=1)
        ):
            failures.append("trace_human_transaction:" + run_id)
        if bounds[ordinal] < datetime.fromisoformat(run["summary"]["created_at"]):
            failures.append("human_before_summary:" + run_id)
    return failures


def attempt_settlement_failures(run_id, run, bounds):
    failures = []

    def fail(code):
        failures.append(f"{code}:{run_id}")

    def time(value):
        return datetime.fromisoformat(value) if value is not None else None

    receipts = {
        (e["generation"], e["worker"], e["attempt"]): e
        for e in run["events"]
        if e["step"] == "model_attempt"
    }
    prior_by_worker = {}
    for attempt in sorted(
        run["attempts"], key=lambda a: (a["worker"], a["attempt_no"])
    ):
        worker = attempt["worker"]
        key = (attempt["generation"], worker, attempt["attempt_no"])
        event = receipts.get(key)
        previous = prior_by_worker.get(worker)
        if previous and previous["generation"] == attempt["generation"]:
            previous_event = receipts.get(
                (previous["generation"], worker, previous["attempt_no"])
            )
            start = time(attempt["started_at"])
            if (
                previous_event is None
                or previous_event["output_summary"].get("next_correction_kind") is None
            ):
                fail("correction_precondition")
            else:
                if start < bounds[previous_event["event_ordinal"]]:
                    fail("correction_before_diagnostic")
                if previous["finished_at"] is None or start < time(
                    previous["finished_at"]
                ):
                    fail("correction_before_settlement")
        prior_by_worker[worker] = attempt
        if event is None:
            expected = "abandoned" if attempt["abandoned"] else None
            if (
                attempt["outcome"] != expected
                or attempt["first_error"] is not None
                or attempt["last_error"] is not None
                or attempt["abandoned"] != (attempt["finished_at"] is not None)
            ):
                fail("ledger_unreturned_state")
            continue
        body = event["output_summary"]
        error = body["attempt_error"]
        kind = body["next_correction_kind"]
        if kind not in {None, "contract", "citation", "transport"}:
            fail("ledger_correction_kind")
        if kind == "transport" and error not in TRANSPORT_RETRY_CODES:
            # A transport retry label is only valid for one of the known recoverable
            # transport outcomes; anything else is a mislabelled correction.
            fail("ledger_transport_kind")
        branch = run["execution"][worker]
        projected = [
            a for a in branch["attempts"] if a["attempt_no"] == attempt["attempt_no"]
        ]
        if kind == "citation":
            if (
                error is not None
                or body["outcome"] != "completed"
                or not str(attempt["last_error"]).startswith("citation_")
            ):
                fail("ledger_citation_error")
            expected_error = attempt["first_error"]
        else:
            expected_error = error
            if error is None and kind is None and projected:
                grounded_error = projected[0]["error_code"]
                support = branch["support_failures"]
                if grounded_error is not None:
                    if (
                        not str(grounded_error).startswith("citation_")
                        or not support
                        or grounded_error != support[0]
                    ):
                        fail("ledger_grounding_error")
                    expected_error = grounded_error
        if kind == "contract" and error not in {
            "invalid_json",
            "invalid_schema",
            "required_check_missing_or_duplicate",
            "duplicate_citation",
        }:
            fail("ledger_contract_error")
        expected_outcome = (
            "format_corrected"
            if kind in {"contract", "citation"}
            else "completed"
            if expected_error is None
            else "failed"
        )
        # A grounded, partially supported branch may be adopted after a refusal:
        # adoption atomically makes the ledger completed and clears last_error.
        adopted = any(
            (cp["generation"], cp["worker"], cp["attempt_no"]) == key
            for cp in run["checkpoints"]
        )
        if adopted and (error is not None or kind is not None):
            fail("ledger_adoption_receipt")
        if not adopted:
            ordinal = event["event_ordinal"]
            previous_at = (
                time(run["events"][ordinal - 2]["occurred_at"]) if ordinal > 1 else None
            )
            finished = time(attempt["finished_at"])
            if finished is None or time(event["occurred_at"]) != max(
                finished,
                previous_at + timedelta(microseconds=1) if previous_at else finished,
            ):
                fail("trace_settlement_transaction")
        if attempt["outcome"] != ("completed" if adopted else expected_outcome):
            fail("ledger_settlement_outcome")
        last_error = None if adopted else expected_error
        if attempt["last_error"] != last_error:
            fail("ledger_last_error")
        if attempt["first_error"] != expected_error:
            fail("ledger_first_error")
        if projected:
            if projected[0]["error_code"] != expected_error:
                fail("ledger_branch_error")
            if projected[0]["correctable"] != (kind in {"contract", "citation"}):
                fail("ledger_branch_correction")
    return failures
