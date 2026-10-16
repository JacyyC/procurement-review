"""Independently expected metering and lifecycle field checks for complete G contracts."""

import copy
import json
from datetime import datetime, timedelta

import pytest

from procurement_review.precise_dev_paired_metering import completeness_report_v3
from procurement_review.precise_dev_score import score_sweep
from procurement_review.trace import canonical_checksum
from test_v1_07g_binding_v4 import (
    complete_fixture,
    first_run,
    refresh,
    format_correction_fixture,
    write_inputs,
    score_cli,
)


@pytest.fixture(scope="module")
def baseline():
    return complete_fixture()


@pytest.mark.parametrize(
    "field,value",
    [
        ("corrections", 1),
        ("contract_corrections", 1),
        ("citation_corrections", 1),
        ("transport_retries", 1),
        ("other_extra_attempts", 1),
        ("branch_latency_ms", [["contract", 999], ["policy", 1001]]),
        ("requested_model", "another-model"),
        ("resolved_models", ["another-model"]),
    ],
)
def test_equal_totals_cannot_hide_wrong_metering(baseline, field, value):
    data = copy.deepcopy(baseline)
    first_run(data)["execution"]["metering"][field] = value
    refresh(data)
    result = score_sweep(*data)
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert result["completeness"]["planned_model_tasks"] == 48
    assert any(
        "ledger_mismatch:" + field + ":" in f
        for f in result["completeness"]["failures"]
    )
    for repeat in (1, 2):
        assert result["repeats"][repeat]["q01_first_answer"]["numerator"] == 24
        assert result["repeats"][repeat]["q01_first_answer"]["denominator"] == 24


@pytest.mark.parametrize(
    "field", ["abandoned", "discarded_late", "outcome", "started_at", "finished_at"]
)
def test_ledger_fields_cannot_be_omitted(baseline, field):
    data = copy.deepcopy(baseline)
    first_run(data)["attempts"][0].pop(field)
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert result["planned_pairs"] == 24 and result["planned_model_tasks"] == 48
    assert any("attempt_missing:" + field + ":" in f for f in result["failures"])


def test_trace_clock_tie_does_not_invent_a_future_receipt(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    # append_trace uses previous occurred_at + 1us when a transaction clock ties it.
    adopted = datetime.fromisoformat(run["checkpoints"][0]["adopted_at"])
    for number, event in enumerate(run["events"]):
        event["occurred_at"] = (adopted + timedelta(microseconds=number)).isoformat()
    row = next(
        row for row in data[1]["observations"] if data[2]["runs"][row["run_id"]] is run
    )
    row["observed_at"] = run["events"][-1]["occurred_at"]
    refresh(data)
    result = completeness_report_v3(*data)
    assert result["complete"], result["failures"]


def test_execution_branch_wrong_type_is_named(baseline):
    data = copy.deepcopy(baseline)
    first_run(data)["execution"]["policy"] = None
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert any("execution_branch_type:" in f for f in result["failures"])


def test_legal_citation_correction_keeps_transport_outcome(baseline):
    # Build the independently specified 3-attempt contract correction, then replace
    # only the facts that distinguish source correction from wire correction.
    data = format_correction_fixture(baseline)
    run = first_run(data)
    run["attempts"][0].update(
        first_error="citation_missing", last_error="citation_missing"
    )
    projected = run["execution"]["policy"]["attempts"][0]
    projected["error_code"] = "citation_missing"
    run["events"][0]["output_summary"].update(
        outcome="completed", attempt_error=None, next_correction_kind="citation"
    )
    run["events"][0]["event_type"] = "step_completed"
    from procurement_review.trace import canonical_checksum

    run["events"][0]["output_checksum"] = canonical_checksum(
        run["events"][0]["output_summary"]
    )
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        run["execution"]["policy"]["attempts"]
    )
    run["execution"]["metering"].update(contract_corrections=0, citation_corrections=1)
    refresh(data)
    result = completeness_report_v3(*data)
    assert result["complete"], result["failures"]
    assert run["execution"]["metering"]["proven_sends"] == 3
    assert run["execution"]["metering"]["citation_corrections"] == 1


@pytest.mark.parametrize("mutation", ["both_contracts", "missing_adoption"])
def test_cli_replay_and_freeze_cannot_promote_invalid_adoption(
    tmp_path, baseline, mutation
):
    from procurement_review.eval_f_validation import freeze_sweep_v4
    import test_v1_07g_binding_v4 as fixtures

    paths = write_inputs(tmp_path, baseline)
    data = [
        json.loads(paths[key].read_text())
        for key in ("registration", "observations", "bindings")
    ]
    getattr(fixtures, mutation)(data)
    paths["bindings"].write_text(json.dumps(data[2]))
    score = tmp_path / "score.json"
    assert score_cli(paths, score).returncode == 0
    result = json.loads(score.read_text())
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert result["completeness"]["planned_pairs"] == 24
    assert result["completeness"]["planned_model_tasks"] == 48
    replayed = score_cli(paths, score, "recheck")
    assert replayed.returncode == 0
    assert json.loads(replayed.stdout)["recheck"] == "identical"
    assert json.loads(replayed.stdout)["stage_acceptance"] is False
    output = tmp_path / "freeze.json"
    with pytest.raises(ValueError, match="v4_freeze_gates_failed"):
        freeze_sweep_v4(
            paths["registration"],
            paths["observations"],
            paths["bindings"],
            paths["source_bindings"],
            score,
            output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "field", ["attempt_error", "next_correction_kind", "latency_ms"]
)
def test_receipt_fields_cannot_disappear(baseline, field):
    data = copy.deepcopy(baseline)
    first_run(data)["events"][0]["output_summary"].pop(field)
    from procurement_review.trace import canonical_checksum

    event = first_run(data)["events"][0]
    event["output_checksum"] = canonical_checksum(event["output_summary"])
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert any("receipt_missing:" + field + ":" in f for f in result["failures"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("next_correction_kind", []),
        ("input_usage", "not-a-token-count"),
        ("receipt_model", 17),
    ],
)
def test_receipt_invalid_types_are_named(baseline, field, value):
    data = copy.deepcopy(baseline)
    first_run(data)["events"][0]["output_summary"][field] = value
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert any("receipt_" in f and "type" in f for f in result["failures"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("resume_type", "resume_count_type"),
        ("generation_missing", "generation_missing:generation"),
        ("checkpoint_null_generation", "checkpoint_record_type"),
        ("trace_type", "trace_identity_type"),
        ("status_type", "generation_status_type"),
    ],
)
def test_binding_identity_shape_is_required(baseline, mutation, reason):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    if mutation == "resume_type":
        run["resume_count"] = "0"
    elif mutation == "generation_missing":
        run["generations"][0].pop("generation")
    elif mutation == "checkpoint_null_generation":
        run["checkpoints"][0]["generation"] = None
    elif mutation == "trace_type":
        run["trace_id"] = 17
        for event in run["events"]:
            event["trace_id"] = 17
    else:
        run["generations"][0]["status"] = []
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert result["planned_pairs"] == 24 and result["planned_model_tasks"] == 48
    assert any(reason + ":" in f for f in result["failures"])


def test_trace_order_cannot_reverse_receipt_clock(baseline):
    data = copy.deepcopy(baseline)
    events = first_run(data)["events"]
    events[0]["occurred_at"], events[1]["occurred_at"] = (
        events[1]["occurred_at"],
        events[0]["occurred_at"],
    )
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert any("event_trace_time_order:" in f for f in result["failures"])


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("event_type", [], "event_field_type:event_type"),
        ("event_type", "unknown_event", "event_field_type:event_type"),
        ("step", None, "event_field_type:step"),
        ("step", "", "event_field_type:step"),
        ("step", 17, "event_field_type:step"),
        ("event_type", "run_cancelled", "receipt_event_type"),
        ("event_type", "step_failed", "receipt_event_type"),
    ],
)
def test_trace_event_shape_and_receipt_kind_are_required(
    baseline, field, value, reason
):
    data = copy.deepcopy(baseline)
    first_run(data)["events"][0][field] = value
    result = score_sweep(*data)
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert result["completeness"]["planned_pairs"] == 24
    assert result["completeness"]["planned_model_tasks"] == 48
    assert any(reason + ":" in f for f in result["completeness"]["failures"])


@pytest.mark.parametrize("field", ["step", "generation", "attempt", "worker"])
def test_trace_nullable_columns_are_explicit(baseline, field):
    data = copy.deepcopy(baseline)
    first_run(data)["events"][-1].pop(field)
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert result["planned_pairs"] == 24 and result["planned_model_tasks"] == 48
    assert any("event_missing:" + field + ":" in f for f in result["failures"])


def test_failed_receipt_cannot_claim_completed_trace(baseline):
    data = format_correction_fixture(baseline)
    first_run(data)["events"][0]["event_type"] = "step_completed"
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert any("receipt_event_type:" in f for f in result["failures"])


@pytest.mark.parametrize("field", ["status", "recommendation"])
def test_observation_state_types_fail_with_named_input_error(baseline, field):
    data = copy.deepcopy(baseline)
    data[1]["observations"][0][field] = []
    with pytest.raises(ValueError, match="v4_input_state:" + field):
        score_sweep(*data)


def test_post_observation_human_action_type_is_named(baseline):
    from procurement_review.trace import canonical_checksum

    data = copy.deepcopy(baseline)
    run = first_run(data)
    last = run["events"][-1]
    event = {
        **copy.deepcopy(last),
        "event_ordinal": last["event_ordinal"] + 1,
        "event_type": "confirmation_recorded",
        "step": "human_gate",
        "occurred_at": (
            datetime.fromisoformat(last["occurred_at"]) + timedelta(seconds=1)
        ).isoformat(),
        "generation": None,
        "attempt": None,
        "worker": None,
        "output_summary": {"action": "confirm", "status": "confirmed"},
    }
    event["output_checksum"] = canonical_checksum(event["output_summary"])
    run["events"].append(event)
    run["status"] = "confirmed"
    assert completeness_report_v3(*data)["complete"]
    event["output_summary"]["action"] = []
    event["output_checksum"] = canonical_checksum(event["output_summary"])
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert any("event_action_type:" in f for f in result["failures"])


@pytest.mark.parametrize(
    "location,value,reason",
    [
        ("registration", "missing", "adoption_evidence_source_missing:registration"),
        ("binding", "missing", "adoption_evidence_source_missing:binding"),
        ("both", "missing", "adoption_evidence_source_missing:registration"),
        ("both", None, "adoption_evidence_source_type:registration"),
        ("both", [], "adoption_evidence_source_type:registration"),
        ("both", "", "adoption_evidence_source_type:registration"),
        ("both", "z" * 64, "adoption_evidence_source_type:registration"),
    ],
)
def test_source_digest_proof_cannot_be_omitted_or_invalid(
    baseline, location, value, reason
):
    data = copy.deepcopy(baseline)
    for name, record, key in (
        ("registration", data[0]["binding"]["adoption_evidence"], "source_file_sha256"),
        ("binding", data[2], "source_evidence_file_sha256"),
    ):
        if location in {name, "both"}:
            if value == "missing":
                record.pop(key)
            else:
                record[key] = value
    result = score_sweep(*data)
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert reason in result["completeness"]["failures"]
    assert result["completeness"]["planned_pairs"] == 24
    assert result["completeness"]["planned_model_tasks"] == 48
    for repeat in (1, 2):
        assert result["repeats"][repeat]["q01_first_answer"]["numerator"] == 24
        assert result["repeats"][repeat]["q01_first_answer"]["denominator"] == 24
        assert result["repeats"][repeat]["q02_obligations"]["denominator"] == 144


@pytest.mark.parametrize(
    "delta_us,late_adoption",
    [(1, False), (-1, False), (20000, True)],
    ids=["extended-1us", "shortened-1us", "late-adoption-in-extended-window"],
)
def test_generation_window_matches_frozen_deadline(baseline, delta_us, late_adoption):
    data = copy.deepcopy(baseline)
    for row in data[1]["observations"]:
        run = data[2]["runs"][row["run_id"]]
        deadline = datetime.fromisoformat(run["generations"][0]["deadline_at"])
        run["generations"][0]["deadline_at"] = (
            deadline + timedelta(microseconds=delta_us)
        ).isoformat()
        if late_adoption:
            adopted = (deadline + timedelta(milliseconds=5)).isoformat()
            for checkpoint in run["checkpoints"]:
                checkpoint["adopted_at"] = adopted
            for attempt in run["attempts"]:
                attempt["finished_at"] = adopted
            run["summary"]["created_at"] = (
                deadline + timedelta(milliseconds=10)
            ).isoformat()
            run["events"][-1]["occurred_at"] = run["summary"]["created_at"]
            row["observed_at"] = (deadline + timedelta(milliseconds=15)).isoformat()
            row["service_elapsed_ms"] = 90015
    refresh(data)
    result = score_sweep(*data)
    assert not result["completeness"]["complete"]
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert result["completeness"]["planned_pairs"] == 24
    assert result["completeness"]["planned_model_tasks"] == 48
    assert any(
        "generation_frozen_deadline:" in failure
        for failure in result["completeness"]["failures"]
    )
    if late_adoption:
        assert any(
            "adoption_generation_deadline:" in failure
            for failure in result["completeness"]["failures"]
        )
    for repeat in (1, 2):
        assert result["repeats"][repeat]["q01_first_answer"]["numerator"] == 24
        assert result["repeats"][repeat]["q01_first_answer"]["denominator"] == 24
        assert result["repeats"][repeat]["q02_obligations"]["denominator"] == 144


def test_generation_window_allows_equivalent_timezone(baseline):
    from datetime import UTC

    data = copy.deepcopy(baseline)
    for run in data[2]["runs"].values():
        for generation in run["generations"]:
            for field in ("created_at", "deadline_at"):
                generation[field] = (
                    datetime.fromisoformat(generation[field])
                    .astimezone(UTC)
                    .isoformat()
                )
    result = score_sweep(*data)
    assert result["stage_acceptance"], result["completeness"]["failures"]


def replace_evidence(data, depth, value):
    parent, key = data[0]["binding"], "adoption_evidence"
    if depth != "manifest":
        parent, key = parent[key], "runs"
    if depth not in {"manifest", "runs"}:
        parent = parent[key]
        key = next(iter(parent))
    if depth not in {"manifest", "runs", "run"}:
        parent, key = parent[key], "checkpoints"
    if depth in {"checkpoint", "commit_fields"}:
        parent, key = parent[key], "policy"
    if depth == "commit_fields":
        parent, key = parent[key], "commit_fields"
    if value == "missing":
        parent.pop(key)
    else:
        parent[key] = value
    run_path = ".runs." + next(iter(data[2]["runs"]))
    return (
        "registration.binding.adoption_evidence"
        + {
            "manifest": "",
            "runs": ".runs",
            "run": run_path,
            "checkpoints": run_path + ".checkpoints",
            "checkpoint": run_path + ".checkpoints.policy",
            "commit_fields": run_path + ".checkpoints.policy.commit_fields",
        }[depth]
    )


@pytest.mark.parametrize(
    "depth", ["manifest", "runs", "run", "checkpoints", "checkpoint", "commit_fields"]
)
@pytest.mark.parametrize("value", [None, [], "missing"])
def test_evidence_objects_have_named_failures_and_fixed_plan(baseline, depth, value):
    from procurement_review.eval_f_lineage import verify_sweep_binding_v3

    data = copy.deepcopy(baseline)
    path = replace_evidence(data, depth, value)
    kind = "missing" if value == "missing" else "type"
    reason = "adoption_evidence_" + kind + ":" + path
    for reader in (verify_sweep_binding_v3, completeness_report_v3):
        result = reader(*data)
        assert not result["complete"]
        assert reason in result["failures"]
        assert result["planned_pairs"] == 24
        assert result["planned_model_tasks"] == 48
    result = score_sweep(*data)
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert reason in result["completeness"]["failures"]
    for repeat in (1, 2):
        assert result["repeats"][repeat]["q01_first_answer"]["numerator"] == 24
        assert result["repeats"][repeat]["q01_first_answer"]["denominator"] == 24
        assert result["repeats"][repeat]["q02_obligations"]["denominator"] == 144


@pytest.mark.parametrize("depth,value", [("manifest", None), ("commit_fields", [])])
def test_cli_recheck_freeze_share_evidence_shape_contract(
    tmp_path, baseline, depth, value
):
    from procurement_review.eval_f_validation import freeze_sweep_v4

    paths = write_inputs(tmp_path, baseline)
    data = [
        json.loads(paths[name].read_text())
        for name in ("registration", "observations", "bindings")
    ]
    output = tmp_path / "score.json"
    assert score_cli(paths, output).returncode == 0
    path = replace_evidence(data, depth, value)
    reason = "adoption_evidence_type:" + path
    paths["registration"].write_text(json.dumps(data[0]))
    for command, target in (("score", tmp_path / "refused.json"), ("recheck", output)):
        result = score_cli(paths, target, command)
        assert result.returncode != 0
        assert reason in result.stderr
        assert "AttributeError" not in result.stderr
    assert not (tmp_path / "refused.json").exists()
    frozen = tmp_path / "refused-freeze.json"
    with pytest.raises(ValueError, match=reason.replace(".", r"\.")):
        freeze_sweep_v4(
            paths["registration"],
            paths["observations"],
            paths["bindings"],
            paths["source_bindings"],
            output,
            frozen,
        )
    assert not frozen.exists()


@pytest.mark.parametrize(
    "location,field",
    [
        ("envelope", name)
        for name in (
            "checkpoint_version",
            "input_checksum",
            "contract_checksum",
            "result",
        )
    ]
    + [
        ("commit", name)
        for name in (
            "commit_version",
            "worker",
            "duties",
            "spans",
            "document_failures",
            "support_failures",
            "security",
            "blocked",
            "request_sha256",
            "answer_sha256",
            "attempts",
            "evidence_refs",
        )
    ],
)
def test_evidence_source_never_supplies_missing_contract_defaults(
    baseline, location, field
):
    from procurement_review.eval_f_validation import evidence_manifest

    source = copy.deepcopy(baseline[2])
    run_id = next(iter(source["runs"]))
    payload = source["runs"][run_id]["checkpoints"][0]["result_payload"]
    record = payload if location == "envelope" else payload["result"]
    record.pop(field)
    suffix = ".result" if location == "commit" else ""
    reason = (
        "v4_source_input_missing:source.runs."
        + run_id
        + ".checkpoints[0].result_payload"
        + suffix
        + "."
        + field
    )
    with pytest.raises(
        ValueError,
        match=reason.replace(".", r"\.").replace("[", r"\[").replace("]", r"\]"),
    ):
        evidence_manifest(source, "f" * 64)


@pytest.mark.parametrize(
    "depth", ["source", "runs", "run", "checkpoints", "payload", "commit"]
)
@pytest.mark.parametrize("value", [None, []])
def test_evidence_generation_names_malformed_source(baseline, depth, value):
    from procurement_review.eval_f_validation import evidence_manifest

    source = copy.deepcopy(baseline[2])
    run_id = next(iter(source["runs"]))
    path = "source"
    if depth == "source":
        source = value
    else:
        parent, key, path = source, "runs", "source.runs"
        if depth != "runs":
            parent, key, path = parent[key], run_id, path + "." + run_id
        if depth not in {"runs", "run"}:
            parent, key, path = parent[key], "checkpoints", path + ".checkpoints"
        if depth in {"payload", "commit"}:
            parent, key, path = (
                parent[key][0],
                "result_payload",
                path + "[0].result_payload",
            )
        if depth == "commit":
            parent, key, path = parent[key], "result", path + ".result"
        parent[key] = value
    kind = "missing" if depth == "checkpoints" and value == [] else "type"
    with pytest.raises(
        ValueError,
        match=("v4_source_input_" + kind + ":" + path)
        .replace(".", r"\.")
        .replace("[", r"\[")
        .replace("]", r"\]"),
    ):
        evidence_manifest(source, "f" * 64)


@pytest.mark.parametrize(
    "status,reason",
    [(s, "generation_status_type") for s in ("interrupted", "failed", "unsupported")]
    + [
        (s, "generation_run_state")
        for s in ("active", "cancelled", "expired", "superseded")
    ],
)
def test_generation_status_keeps_production_contract(baseline, status, reason):
    data = copy.deepcopy(baseline)
    for run in data[2]["runs"].values():
        run["generations"][-1]["status"] = status
    result = score_sweep(*data)
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    complete = result["completeness"]
    assert not complete["complete"]
    assert complete["planned_pairs"] == 24 and complete["planned_model_tasks"] == 48
    assert sum(code.startswith(reason + ":") for code in complete["failures"]) == 24
    for repeat in result["repeats"].values():
        assert repeat["q01_first_answer"]["numerator"] == 24
        assert repeat["q01_first_answer"]["denominator"] == 24
        assert repeat["q02_obligations"]["denominator"] == 144


@pytest.mark.parametrize(
    "generation_status,valid",
    [("completed", True), ("cancelled", True)]
    + [(s, False) for s in ("active", "expired", "superseded")],
)
def test_post_summary_cancel_keeps_production_generation(
    baseline, generation_status, valid
):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    run_id = run["generations"][0]["run_id"]
    row = next(row for row in data[1]["observations"] if row["run_id"] == run_id)
    run["status"] = "cancelled"
    run["generations"][-1]["status"] = generation_status
    body = {"action": "cancel", "status": "cancelled"}
    run["events"].append(
        {
            "run_id": run_id,
            "trace_id": run["trace_id"],
            "event_ordinal": len(run["events"]) + 1,
            "event_type": "human_action_recorded",
            "step": "human_action",
            "generation": None,
            "attempt": None,
            "worker": None,
            "occurred_at": (
                datetime.fromisoformat(row["observed_at"]) + timedelta(seconds=1)
            ).isoformat(),
            "output_summary": body,
            "output_checksum": canonical_checksum(body),
        }
    )
    result = completeness_report_v3(*data)
    assert result["planned_pairs"] == 24 and result["planned_model_tasks"] == 48
    if valid:
        assert result["complete"], result["failures"]
    else:
        assert not result["complete"]
        assert "generation_run_state:" + run_id in result["failures"]
        score = score_sweep(*data)
        assert not score["stage_acceptance"]
        assert "cross-repeat/Q-10-ledger-complete" in score["failed_gates"]


@pytest.mark.parametrize("value", [[], {}, None, 7])
def test_generation_consumer_preserves_run_state_type_rejection(baseline, value):
    data = copy.deepcopy(baseline)
    for run in data[2]["runs"].values():
        run["status"] = value
    result = score_sweep(*data)
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert result["completeness"]["planned_pairs"] == 24
    assert result["completeness"]["planned_model_tasks"] == 48
    assert (
        sum(
            code.startswith("run_status_type:")
            for code in result["completeness"]["failures"]
        )
        == 24
    )
