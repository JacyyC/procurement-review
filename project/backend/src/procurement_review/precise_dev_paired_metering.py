"""The paired diagnostic's per-attempt metering, from the batch's own durable ledger.

A row without a receipt is not a returned attempt and not a free one: it is reported as
unreturned with unknown usage, so the diagnostic can never read a lost or mocked answer as
a cheap one.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .attempt_metering import classify, facts_from_receipts
from .eval_attempt_ledger import AttemptLedger


def _attempt_receipts(ledger: AttemptLedger, task_key: str) -> tuple[dict, ...]:
    """The bounded per-attempt facts of one branch: state, code and location tokens."""

    with sqlite3.connect(ledger.path) as connection:
        rows = connection.execute(
            "SELECT state, receipt_json FROM attempts WHERE task_key=? ORDER BY"
            " attempt_no",
            (task_key,),
        ).fetchall()
    out: list[dict] = []
    for state, receipt in rows:
        body = {}
        if receipt:
            try:
                loaded = json.loads(receipt)
                body = loaded if isinstance(loaded, dict) else {}
            except ValueError:
                body = {}
        error = body.get("error_code") if isinstance(body, dict) else None
        out.append(
            {
                "state": str(state),
                "error": str(error) if error else None,
                "format_diagnostics": list(body.get("format_diagnostics") or ()),
            }
        )
    return tuple(out)


def _with_kind(row: dict, correction_kinds: tuple[str, ...]) -> str | None:
    """The branch's own correction order, annotated onto the attempt it followed.

    The eval ledger's receipts predate this field, so the diagnostic's own projection order
    is the authority for *what* its extra attempts were for.
    """

    if not row.get("receipt_json"):
        return None
    body = json.loads(row["receipt_json"])
    index = int(row.get("attempt_no") or 1) - 1
    if 0 <= index < len(correction_kinds):
        body["next_correction_kind"] = correction_kinds[index]
    return json.dumps(body)


def branch_metering(
    ledger: AttemptLedger, task_key: str, correction_kinds: tuple[str, ...] = ()
) -> dict[str, int | bool | None]:
    """The same send/return split the product Run reports, from this branch's ledger rows.

    A row without a receipt is not a returned attempt and not a free one: it is reported as
    unreturned with unknown usage, so the diagnostic can never read a mocked or lost answer
    as a cheap one.
    """

    with sqlite3.connect(ledger.path) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM attempts WHERE task_key=? ORDER BY attempt_no",
                (task_key,),
            )
        ]
    if correction_kinds:
        rows = [
            {**row, "receipt_json": _with_kind(row, correction_kinds)} for row in rows
        ]
    metering = classify(
        facts_from_receipts(
            rows,
            receipt_of=lambda row: (
                json.loads(row["receipt_json"]) if row.get("receipt_json") else None
            ),
            # The rows are already filtered to this task, so this branch is the whole set;
            # naming it keeps the branch identity the same as the product line's.
            branch_of=lambda row: str(row.get("task_key") or ""),
        )
    )
    return {
        "attempts_opened": metering.attempts_opened,
        "proven_sends": metering.proven_sends,
        "unresolved_attempts": metering.unresolved_attempts,
        "first_sends": metering.first_sends,
        "contract_corrections": metering.contract_corrections,
        "citation_corrections": metering.citation_corrections,
        "transport_retries": metering.transport_retries,
        "other_extra_attempts": metering.other_extra_attempts,
        "input_tokens_known": metering.input_tokens_known,
        "output_tokens_known": metering.output_tokens_known,
        "usage_complete": metering.usage_complete,
        "unknown_usage_attempts": metering.unknown_usage_attempts,
    }


def bind_runs(state_path: Path, registration_path: Path, output_path: Path) -> dict:
    """Read every observed Run's own binding and ledger rows from the product database.

    No model call is involved: this reads the frozen execution snapshot and the attempt
    ledger rows the product already persisted, so the scorer can cross-check the observed
    metering against the ledger instead of trusting the projection alone.
    """

    from sqlalchemy import create_engine, text as sql_text

    state = json.loads(state_path.read_text())
    body = json.loads(registration_path.read_text())
    chosen = list(body["case_ids"])
    base = state["database_url"].split("?", 1)[0]
    engine = create_engine(base if "://" in base else state["database_url"])
    runs: dict[str, Any] = {}
    try:
        with engine.connect() as connection:
            for case_id in chosen:
                rows = connection.execute(
                    sql_text(
                        "SELECT run_id, status, isolation FROM review_run "
                        "WHERE case_id = :case_id ORDER BY created_at"
                    ),
                    {"case_id": case_id},
                ).all()
                for run_id, status, isolation in rows:
                    snapshot = (isolation or {}).get("execution_snapshot") or {}
                    attempts = connection.execute(
                        sql_text(
                            "SELECT generation, worker, attempt_no, outcome, abandoned, "
                            "discarded_late FROM review_worker_attempt "
                            "WHERE run_id = :run_id ORDER BY generation, worker, attempt_no"
                        ),
                        {"run_id": run_id},
                    ).all()
                    receipts = connection.execute(
                        sql_text(
                            "SELECT count(*) FROM review_receipt WHERE run_id = :run_id"
                        ),
                        {"run_id": run_id},
                    ).scalar_one()
                    runs[run_id] = {
                        "case_id": case_id,
                        "status": status,
                        "implementation_sha256": snapshot.get("implementation_sha256"),
                        "rules_version": snapshot.get("rules_version"),
                        "document_version": snapshot.get("document_version"),
                        "dataset_version": snapshot.get("dataset_version"),
                        "index_version": snapshot.get("index_version"),
                        "resolved_model": snapshot.get("resolved_model"),
                        "requested_model": snapshot.get("requested_model"),
                        "attempts": [
                            {
                                "generation": generation,
                                "worker": worker,
                                "attempt_no": attempt_no,
                                "outcome": outcome,
                                "abandoned": abandoned,
                                "discarded_late": discarded_late,
                            }
                            for (
                                generation,
                                worker,
                                attempt_no,
                                outcome,
                                abandoned,
                                discarded_late,
                            ) in attempts
                        ],
                        "receipt_count": receipts,
                    }
    finally:
        engine.dispose()
    payload = {
        "database": state["database"],
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": runs,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return {"written": str(output_path), "runs": len(runs)}


# ------------------------------------------ the DEV sweep's completeness and timing v2
#
# The two readers below judge a sweep's observation file against its own pre-registration
# and against the product database's ledger rows: every planned Case x repeat has to be
# present exactly once with its own Run, every Run's binding has to be the registered
# build, and the observed metering has to close against the ledger.  A missing sample, a
# duplicate, an extra or a field a row does not carry is a named failure, never a silent
# exclusion.

REQUIRED_METERING_KEYS = (
    "attempts_opened",
    "physical_attempts",
    "proven_sends",
    "unresolved_attempts",
    "open_attempts",
    "unreturned_attempts",
    "first_sends",
    "contract_corrections",
    "citation_corrections",
    "transport_retries",
    "other_extra_attempts",
    "unknown_usage_attempts",
    "usage_complete",
    "requested_model",
    "resolved_models",
)


def _planned_pairs(registration: dict) -> list[tuple[str, int]]:
    return [
        (case_id, repeat)
        for case_id in registration["case_ids"]
        for repeat in range(1, int(registration["repeats"]) + 1)
    ]


def completeness_report(
    registration: dict,
    observations: dict,
    bindings: dict | None,
    *,
    observed_registration_sha256: str,
) -> dict:
    """Every planned observation, its Run's binding and its ledger, checked one by one."""

    rows = observations["observations"]
    binding = registration.get("binding") or {}
    failures: list[str] = []
    pairs: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        key = (row.get("case_id"), row.get("repeat"))
        pairs.setdefault(key, []).append(row)
    planned = _planned_pairs(registration)
    missing = [list(key) for key in planned if key not in pairs]
    duplicates = [
        [list(key), len(group)] for key, group in pairs.items() if len(group) > 1
    ]
    extra = [list(key) for key in pairs if key not in planned]
    for name, entries in (
        ("missing", missing),
        ("duplicate", duplicates),
        ("extra", extra),
    ):
        for entry in entries:
            failures.append(f"{name}_pair:{entry}")
    run_ids = [row.get("run_id") for row in rows]
    duplicate_runs = sorted(
        {run_id for run_id in run_ids if run_ids.count(run_id) > 1 and run_id}
    )
    for run_id in duplicate_runs:
        failures.append(f"duplicate_run_id:{run_id}")
    for row in rows:
        if not row.get("run_id"):
            failures.append(f"missing_run_id:{row.get('case_id')}:{row.get('repeat')}")
        if row.get("terminal_status") != 200 or row.get("status") in {None, "failed"}:
            failures.append(
                f"not_terminal:{row.get('case_id')}:{row.get('repeat')}:{row.get('status')}"
            )
        if row.get("execution_mode") != "real":
            failures.append(f"wrong_execution_mode:{row.get('run_id')}")
        for key, expected in (
            ("execution_version", binding.get("expected_execution_version")),
            ("rules_version", binding.get("expected_rules_version")),
            ("summary_schema", binding.get("expected_summary_schema")),
        ):
            if row.get(key) != expected:
                failures.append(
                    f"row_binding:{row.get('run_id')}:{key}:{row.get(key)!r}!={expected!r}"
                )
        metering = row.get("metering")
        if not isinstance(metering, dict):
            failures.append(f"metering_missing:{row.get('run_id')}")
            continue
        for key in REQUIRED_METERING_KEYS:
            if key not in metering:
                failures.append(f"metering_key_missing:{row.get('run_id')}:{key}")
        if metering.get("resolved_models") not in ([binding.get("resolved_model")], []):
            failures.append(f"model_drift:{row.get('run_id')}")
        if metering.get("requested_model") not in (
            binding.get("requested_model"),
            None,
        ):
            failures.append(f"requested_model:{row.get('run_id')}")
    binding_runs = (bindings or {}).get("runs") or {}
    observed_runs = {row.get("run_id") for row in rows}
    for run_id in sorted(observed_runs - set(binding_runs)):
        failures.append(f"binding_missing_run:{run_id}")
    # The database holds other batches' Runs as well, so an unobserved binding row is
    # reported rather than failed; an observed Run without one is a failure above.
    extra_binding_runs = sorted(set(binding_runs) - observed_runs)
    for row in rows:
        run_id = row.get("run_id")
        ledger = binding_runs.get(run_id)
        if ledger is None:
            continue
        expected_identity = {
            "implementation_sha256": binding.get("product_implementation_sha256"),
            "rules_version": binding.get("expected_rules_version"),
            "document_version": binding.get("expected_document_version"),
            "dataset_version": binding.get("expected_dataset_version"),
            "index_version": binding.get("expected_index_version"),
            "resolved_model": binding.get("resolved_model"),
            "requested_model": binding.get("requested_model"),
        }
        for key, expected in expected_identity.items():
            if ledger.get(key) != expected:
                failures.append(
                    f"run_binding:{run_id}:{key}:{ledger.get(key)!r}!={expected!r}"
                )
        if ledger.get("case_id") != row.get("case_id"):
            failures.append(f"run_case:{run_id}")
        attempts = ledger.get("attempts") or []
        metering = row.get("metering") or {}
        abandoned = sum(1 for item in attempts if item.get("abandoned"))
        open_rows = sum(1 for item in attempts if item.get("outcome") is None)
        proven = sum(
            1
            for item in attempts
            if item.get("outcome") is not None and not item.get("abandoned")
        )
        firsts = sum(
            1
            for item in attempts
            if item.get("attempt_no") == 1
            and item.get("outcome") is not None
            and not item.get("abandoned")
        )
        checks = (
            ("attempts_opened", len(attempts)),
            ("physical_attempts", len(attempts)),
            ("proven_sends", proven),
            ("unresolved_attempts", abandoned),
            ("open_attempts", open_rows),
            ("first_sends", firsts),
        )
        for key, expected in checks:
            if metering.get(key) != expected:
                failures.append(
                    f"ledger_mismatch:{run_id}:{key}:{metering.get(key)!r}!={expected!r}"
                )
        unknown = metering.get("unknown_usage_attempts")
        if isinstance(unknown, int) and unknown < abandoned + open_rows:
            failures.append(f"unknown_usage_understated:{run_id}")
        complete_flag = metering.get("usage_complete")
        if complete_flag != (
            isinstance(unknown, int) and unknown == 0 and attempts != []
        ):
            failures.append(f"usage_complete_inconsistent:{run_id}")
        limit = int(binding.get("max_attempts_per_branch") or 0)
        for item in attempts:
            if limit and int(item.get("attempt_no") or 0) > limit:
                failures.append(f"plan_attempt_bound:{run_id}:{item.get('worker')}")
        if limit and len(attempts) > limit * len(binding.get("branches") or []):
            failures.append(f"plan_run_bound:{run_id}")
    return {
        "planned_pairs": len(planned),
        "missing_pairs": missing,
        "duplicate_pairs": duplicates,
        "extra_pairs": extra,
        "duplicate_run_ids": duplicate_runs,
        "binding_runs": len(binding_runs),
        "extra_binding_runs": extra_binding_runs,
        "observed_registration_sha256": observed_registration_sha256,
        "base_registration_sha256": (registration.get("extends") or {}).get(
            "base_registration_sha256"
        ),
        "failures": failures,
        "complete": not failures,
    }


def latency_report(
    registration: dict,
    observations: dict,
    *,
    p95_limit_ms: int,
    deadline_ms: int,
) -> dict:
    """The full planned sample's timing: a missing sample is a failure, not an omission."""

    rows = observations["observations"]
    by_pair = {(row.get("case_id"), row.get("repeat")): row for row in rows}
    planned = _planned_pairs(registration)
    latencies: list[int] = []
    missing: list[str] = []
    for case_id, repeat in planned:
        row = by_pair.get((case_id, repeat))
        if row is None:
            missing.append(f"{case_id}:r{repeat}:absent")
            continue
        latency = _service_latency_ms(row)
        if latency is None:
            missing.append(f"{case_id}:r{repeat}:no_service_timing")
            continue
        latencies.append(latency)
    ordered = sorted(latencies)
    p95 = None
    if ordered:
        rank = max(0, min(len(ordered) - 1, -(-len(ordered) * 95 // 100) - 1))
        p95 = ordered[rank]
    censored = [
        run_id
        for run_id in (
            row.get("run_id") for row in rows if _service_latency_ms(row) is not None
        )
        if _service_latency_ms(next(row for row in rows if row.get("run_id") == run_id))
        > deadline_ms
    ]
    return {
        "planned_samples": len(planned),
        "samples": len(latencies),
        "missing_samples": missing,
        "p95_ms": p95,
        "max_ms": max(latencies) if latencies else None,
        "censored_over_deadline": censored,
        "p95_limit_ms": p95_limit_ms,
        "deadline_ms": deadline_ms,
    }


def _service_latency_ms(row: dict) -> int | None:
    created = row.get("service_created_at")
    observed = row.get("observed_at")
    if not isinstance(created, str) or not isinstance(observed, str):
        return None
    try:
        begin = datetime.fromisoformat(created)
        end = datetime.fromisoformat(observed)
    except ValueError:
        return None
    return max(0, int((end - begin).total_seconds() * 1000))


def service_latency_v3(row):
    created, observed = row.get("service_created_at"), row.get("observed_at")
    try:
        begin, end = datetime.fromisoformat(created), datetime.fromisoformat(observed)
        if begin.tzinfo is None or end.tzinfo is None or end < begin:
            return None
        return int((end - begin).total_seconds() * 1000)
    except (ValueError, TypeError):
        return None


def latency_report_v3(
    registration, observations, *, p95_limit_ms, deadline_ms, repeat=None
):
    planned = [
        pair
        for pair in _planned_pairs(registration)
        if repeat is None or pair[1] == repeat
    ]
    values, missing, censored = [], [], []
    for case_id, repetition in planned:
        rows = [
            row
            for row in observations["observations"]
            if (row.get("case_id"), row.get("repeat")) == (case_id, repetition)
        ]
        latency = service_latency_v3(rows[0]) if len(rows) == 1 else None
        if latency is None:
            missing.append(f"{case_id}:r{repetition}:invalid_or_missing_timing")
        else:
            values.append(latency)
            if latency > deadline_ms:
                censored.append(rows[0].get("run_id"))
    ordered = sorted(values)
    complete = len(values) == len(planned) and not missing
    return {
        "planned_samples": len(planned),
        "samples": len(values),
        "missing_samples": missing,
        "p95_ms": ordered[-(-len(ordered) * 95 // 100) - 1]
        if ordered and complete
        else None,
        "max_ms": max(values) if values and complete else None,
        "complete": complete,
        "censored_over_deadline": censored,
        "p95_limit_ms": p95_limit_ms,
        "deadline_ms": deadline_ms,
    }


def repeat_consistency(
    observations: dict, registration: dict | None = None
) -> dict[str, Any]:
    """Same-Case agreement between the pre-registered repeats, obligations included."""

    by_case: dict[str, list[dict]] = {}
    for row in observations["observations"]:
        by_case.setdefault(row.get("case_id"), []).append(row)
    if registration is not None:
        by_case = {case: by_case.get(case, []) for case in registration["case_ids"]}
    rows = []
    for case_id, group in sorted(by_case.items()):
        if len(group) != 2 or (
            registration is not None and {row.get("repeat") for row in group} != {1, 2}
        ):
            rows.append(
                {
                    "case_id": case_id,
                    "recommendations": [row.get("recommendation") for row in group],
                    "recommendation_consistent": False,
                    "obligations_consistent": False,
                    "repeats": len(group),
                }
            )
            continue

        def states(row: dict) -> dict:
            return {
                item.get("obligation_id"): (item.get("state"), item.get("reason"))
                for item in row.get("findings") or []
            }

        recommendations = [row.get("recommendation") for row in group]
        rows.append(
            {
                "case_id": case_id,
                "recommendations": recommendations,
                "recommendation_consistent": len(set(recommendations)) == 1,
                "obligations_consistent": states(group[0]) == states(group[1]),
                "repeats": len(group),
            }
        )
    consistent = sum(1 for row in rows if row["recommendation_consistent"])
    return {
        "cases": rows,
        "numerator": consistent,
        "denominator": len(rows),
        "value": consistent / len(rows) if rows else None,
    }


def completeness_report_v3(registration, observations, bindings, **kwargs):
    from .eval_f_lineage import verify_sweep_binding_v3

    validate_score_input_v3(registration, observations, bindings)
    return verify_sweep_binding_v3(registration, observations, bindings, **kwargs)


def validate_score_input_v3(registration, observations, bindings):
    """Strict JSON containers and identity types, without changing historical readers."""

    def require(value, kind, path):
        if type(value) is not kind:
            raise ValueError(f"v3_input_type:{path}:{kind.__name__}")

    def required(value, fields, path):
        for field in fields:
            if field not in value:
                raise ValueError(f"v3_input_missing:{path}.{field}")

    require(registration, dict, "registration")
    require(registration.get("repeats"), int, "registration.repeats")
    require(registration.get("case_ids"), list, "registration.case_ids")
    for case in registration["case_ids"]:
        require(case, str, "registration.case_ids[]")
    if (
        registration["repeats"] < 1
        or not registration["case_ids"]
        or len(set(registration["case_ids"])) != len(registration["case_ids"])
    ):
        raise ValueError("v3_input_plan:repeats_or_case_ids")
    arrays = {
        "observations",
        "findings",
        "spans",
        "conflicts",
        "security",
        "fact_observations",
        "attempts",
        "checkpoints",
        "generations",
        "events",
    }
    objects = {
        "branches",
        "metering",
        "runs",
        "projection",
        "execution",
        "snapshot",
        "proposition",
        "output_summary",
        "summary_payload",
        "result_payload",
    }
    integers = {
        "repeat",
        "attempt_no",
        "generation",
        "attempt",
        "event_ordinal",
        "start",
        "end",
    }
    booleans = {
        "blocking",
        "abandoned",
        "discarded_late",
        "usage_unknown",
        "usage_complete",
    }

    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                where = f"{path}.{key}"
                if key in arrays:
                    require(item, list, where)
                    for child in item:
                        require(child, dict, where + "[]")
                        if key == "spans":
                            required(child, ("start", "end"), where)
                            if "proposition" not in child:
                                required(
                                    child, ("worker", "check_id", "field_path"), where
                                )
                elif key in objects:
                    require(item, dict, where)
                    if key == "proposition":
                        required(item, ("check_id", "field_path"), where)
                elif key in {"field_path", "worker", "check_id", "obligation_id"} and (
                    item is not None or key == "field_path"
                ):
                    require(item, str, where)
                elif key == "evidence":
                    if type(item) not in {dict, list}:
                        raise ValueError(f"v3_input_type:{where}:evidence")
                    if isinstance(item, list):
                        for child in item:
                            require(child, dict, where + "[]")
                elif key in integers and item is not None:
                    require(item, int, where)
                elif key in booleans:
                    require(item, bool, where)
                if (
                    path.startswith("bindings")
                    and key
                    in {
                        "created_at",
                        "deadline_at",
                        "adopted_at",
                        "started_at",
                        "finished_at",
                        "occurred_at",
                    }
                    and item is not None
                ):
                    try:
                        timestamp = datetime.fromisoformat(item)
                        if timestamp.tzinfo is None:
                            raise ValueError("timezone_missing")
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"v3_input_time:{where}") from error
                visit(item, where)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    require(observations, dict, "observations")
    require(observations.get("observations"), list, "observations.observations")
    for row in observations["observations"]:
        require(row, dict, "observations.observations[]")
        for key, kind in (("case_id", str), ("run_id", str), ("repeat", int)):
            require(row.get(key), kind, f"observations.observations[].{key}")
    visit(observations, "observations")
    if (
        registration.get("extends", {}).get("extension_version")
        == "v1-07-truth-extension.v3"
    ):
        from pydantic import TypeAdapter, ValidationError
        from .contracts import RecommendationStatus, ReviewLifecycleStatus

        for row in observations["observations"]:
            for key, contract in (
                ("status", ReviewLifecycleStatus),
                ("recommendation", RecommendationStatus | None),
            ):
                try:
                    TypeAdapter(contract).validate_json(
                        json.dumps(row.get(key)), strict=True
                    )
                except ValidationError as error:
                    raise ValueError("v4_input_state:" + key) from error
    if bindings is not None:
        require(bindings, dict, "bindings")
        visit(bindings, "bindings")
        for run_id, run in bindings.get("runs", {}).items():
            path = f"bindings.runs.{run_id}"
            require(run, dict, path)
            containers = {
                "summary": ("summary_checksum", "summary_payload"),
                "checkpoints": (
                    "worker",
                    "generation",
                    "attempt_no",
                    "result_payload",
                    "result_checksum",
                    "input_checksum",
                ),
                "events": (
                    "event_ordinal",
                    "occurred_at",
                    "event_type",
                    "output_summary",
                ),
            }
            for container, fields in containers.items():
                if container not in run:
                    continue
                records = [run[container]] if container == "summary" else run[container]
                for record in records:
                    require(record, dict, path + "." + container)
                    required(record, fields, path + "." + container)


def receipt_outcome_matches_v4(item, receipt, version):
    if receipt.get("outcome") == item.get("outcome"):
        return True
    return (
        version.endswith(".v4")
        and item.get("outcome") == "format_corrected"
        and (
            receipt.get("outcome") == "failed"
            and receipt.get("next_correction_kind") == "contract"
            or receipt.get("outcome") == "completed"
            and receipt.get("next_correction_kind") == "citation"
        )
    )


def receipt_value_matches_v4(
    item, attempt, receipt, observed_key, receipt_key, version
):
    if attempt.get(observed_key) == receipt.get(receipt_key):
        return True
    return (
        version.endswith(".v4")
        and observed_key == "error_code"
        and receipt.get("next_correction_kind") == "citation"
        and receipt.get("attempt_error") is None
        and attempt.get("error_code")
        in {item.get("first_error"), item.get("last_error")}
        and str(attempt.get("error_code")).startswith("citation_")
    )


def gate(
    value: float | None,
    threshold: float,
    *,
    direction: str = "at_least",
    integer_boundary: bool = False,
) -> dict[str, Any]:
    """One verdict, with a zero denominator reported as undecidable rather than passing."""

    if value is None:
        return {
            "value": None,
            "threshold": threshold,
            "meets": None,
            "undecidable": True,
        }
    meets = value >= threshold if direction == "at_least" else value <= threshold
    return {
        "value": value,
        "threshold": threshold,
        "meets": meets,
        "undecidable": False,
        "integer_boundary": integer_boundary,
    }
