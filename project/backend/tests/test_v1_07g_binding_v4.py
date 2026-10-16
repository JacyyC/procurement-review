"""Complete checkpoint contracts and independently specified causal counterexamples."""

import copy
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from procurement_review.gold_v3_contracts import payload_sha256
from procurement_review.business_v31_documents import VALUE_MODELS_V31
from procurement_review.precise_dev_paired_metering import completeness_report_v3
from procurement_review.precise_dev_score import score_sweep
from procurement_review.precise_run_contracts import PreciseCheckpointPayloadV1
from procurement_review.trace import canonical_checksum
from test_v1_07g_score_v3 import fixture_v3


def complete_fixture():
    registered, observed, bindings = fixture_v3()
    profile = registered["binding"]
    for run in bindings["runs"].values():
        run["trace_id"] = "trace-" + run["case_id"]
        for generation in run["generations"]:
            generation["run_id"] = next(
                key for key, value in bindings["runs"].items() if value is run
            )
        for event in run["events"]:
            event["run_id"] = run["generations"][0]["run_id"]
            event["trace_id"] = run["trace_id"]
            for field in ("generation", "attempt", "worker"):
                event.setdefault(field, None)
        for attempt in run["attempts"]:
            attempt.update(first_error=None, last_error=None)
        run["resume_count"] = 0
        run["execution"]["metering"]["explicit_recoveries"] = 0
        run["execution"]["metering"]["reused_branches"] = 0
        run["execution"]["metering"].update(
            corrections=0, branch_latency_ms=[["contract", 1000], ["policy", 1000]]
        )
        snapshot = run["snapshot"]
        snapshot.pop("product_implementation_sha256", None)
        snapshot.pop("execution_version", None)
        snapshot.pop("summary_schema", None)
        snapshot.pop("model", None)
        for checkpoint in run["checkpoints"]:
            worker = checkpoint["worker"]
            native = run["execution"][worker]
            native["attempts"][0]["resolved_model"] = profile["resolved_model"]
            receipt = next(e for e in run["events"] if e.get("worker") == worker)
            receipt["output_summary"].update(
                latency_ms=1000, next_correction_kind=None, attempt_error=None
            )
            receipt["output_checksum"] = canonical_checksum(receipt["output_summary"])
            receipt["occurred_at"] = (
                datetime.fromisoformat(checkpoint["adopted_at"])
                - timedelta(seconds=1)
                + timedelta(microseconds=0 if worker == "policy" else 1)
            ).isoformat()
            text = registered["material_text"][run["case_id"]][worker]
            for span in native["spans"]:
                if type(span["value"]) is list:
                    span["value"] = json.dumps(
                        span["value"], ensure_ascii=False, separators=(",", ":")
                    )
                span["span_sha256"] = hashlib.sha256(
                    text[span["start"] : span["end"]].encode()
                ).hexdigest()
            profile_source = registered["truth_extension"][run["case_id"]][
                "branch_sources"
            ][worker]
            refs = [
                {
                    "source_kind": "public_document"
                    if worker == "policy"
                    else "case_document",
                    "source_id": profile_source["source_id"],
                    "version": profile_source["version"],
                    "page_number": None,
                    "section": None,
                    "chunk_id": profile_source["locator"],
                    "content_checksum": "sha256:" + profile_source["content_sha256"],
                }
            ]
            native.update(answer_sha256="a" * 64, evidence_refs=refs, reported=[])
            duties = []
            for duty in VALUE_MODELS_V31[worker]:
                values = {}
                for span in native["spans"]:
                    if span["proposition"]["check_id"] == duty:
                        value = span["value"]
                        try:
                            value = json.loads(value)
                        except (ValueError, TypeError):
                            pass
                        values[span["proposition"]["field_path"].rsplit(".", 1)[-1]] = (
                            value
                        )
                normalized = (
                    json.dumps(values, ensure_ascii=False, separators=(",", ":"))
                    if values
                    else None
                )
                status = "supported" if values else "unknown"
                native["reported"].append(
                    {"obligation_id": duty, "status": status, "normalized": normalized}
                )
                duties.append(
                    {
                        "obligation_id": duty,
                        "status": status,
                        "risk_level": "none" if values else "unknown",
                        "value": normalized or "UNKNOWN",
                        "evidence_refs": ["ref-" + payload_sha256(profile_source)]
                        if values
                        else [],
                    }
                )
            # The identity formula is written independently from the production helper.
            contract = "sha256:" + payload_sha256(
                {"snapshot": payload_sha256(snapshot), "worker": worker}
            )
            checkpoint["contract_checksum"] = contract
            checkpoint["result_payload"] = {
                "checkpoint_version": "precise-branch-commit.v1",
                "input_checksum": "sha256:"
                + profile["expected_requests"][run["case_id"]][worker],
                "contract_checksum": contract,
                "result": {
                    "commit_version": "precise-branch-commit.v1",
                    "worker": worker,
                    "duties": duties,
                    "spans": copy.deepcopy(native["spans"]),
                    "document_failures": [],
                    "support_failures": [],
                    "security": [],
                    "blocked": False,
                    "request_sha256": native["request_sha256"],
                    "answer_sha256": native["answer_sha256"],
                    "attempts": copy.deepcopy(native["attempts"]),
                    "evidence_refs": copy.deepcopy(refs),
                },
            }
            checkpoint["result_checksum"] = canonical_checksum(
                checkpoint["result_payload"]
            )
            PreciseCheckpointPayloadV1.model_validate_json(
                json.dumps(checkpoint["result_payload"]), strict=True
            )
    from procurement_review.precise_dev_sweep import _summarise

    for row in observed["observations"]:
        run = bindings["runs"][row["run_id"]]
        run["projection"] = _summarise(
            {
                "execution": run["execution"],
                "status": row["status"],
                "summary": run["summary"]["summary_payload"],
                "execution_mode": "real",
                "error_code": None,
            }
        )
        row.update(copy.deepcopy(run["projection"]))
    bindings["observations_payload_sha256"] = payload_sha256(observed)
    from procurement_review.eval_f_validation import evidence_manifest

    registered["extends"]["extension_version"] = "v1-07-truth-extension.v3"
    registered["binding"]["adoption_evidence"] = evidence_manifest(bindings, "f" * 64)
    bindings["binding_version"] = "v1-07g-db-binding.v4"
    bindings["source_evidence_file_sha256"] = "f" * 64
    return registered, observed, bindings


@pytest.fixture(scope="module")
def baseline():
    return complete_fixture()


def first_run(data):
    return next(iter(data[2]["runs"].values()))


def change_checkpoint(data, action):
    for run in data[2]["runs"].values():
        for checkpoint in run["checkpoints"]:
            action(checkpoint)
            checkpoint["result_checksum"] = canonical_checksum(
                checkpoint["result_payload"]
            )


def delete_contract(data):
    change_checkpoint(data, lambda cp: cp.pop("contract_checksum"))


def outer_contract(data):
    change_checkpoint(
        data, lambda cp: cp.update(contract_checksum="sha256:" + "0" * 64)
    )


def both_contracts(data):
    def change(cp):
        cp["contract_checksum"] = cp["result_payload"]["contract_checksum"] = (
            "sha256:" + "0" * 64
        )

    change_checkpoint(data, change)


def payload_request(data):
    change_checkpoint(
        data,
        lambda cp: cp["result_payload"].update(input_checksum="sha256:" + "0" * 64),
    )


def envelope_version(data):
    change_checkpoint(
        data,
        lambda cp: cp["result_payload"].update(
            checkpoint_version="precise-branch-commit.v9"
        ),
    )


def commit_version(data):
    change_checkpoint(
        data,
        lambda cp: cp["result_payload"]["result"].update(
            commit_version="precise-branch-commit.v9"
        ),
    )


def commit_answer(data):
    change_checkpoint(
        data, lambda cp: cp["result_payload"]["result"].update(answer_sha256="b" * 64)
    )


def commit_failures(data):
    change_checkpoint(
        data,
        lambda cp: cp["result_payload"]["result"].update(
            document_failures=["document_citation_invalid"]
        ),
    )


def commit_blocked(data):
    change_checkpoint(
        data, lambda cp: cp["result_payload"]["result"].update(blocked=True)
    )


def missing_adoption(data):
    change_checkpoint(data, lambda cp: cp.pop("adopted_at"))


def future_attempts(data):
    for run in data[2]["runs"].values():
        for attempt in run["attempts"]:
            for field in ("started_at", "finished_at"):
                attempt[field] = (
                    datetime.fromisoformat(attempt[field])
                    .replace(year=2027)
                    .isoformat()
                )


def receipt_before_request(data):
    for run in data[2]["runs"].values():
        for event in run["events"]:
            if event.get("step") == "model_attempt":
                event["occurred_at"] = (
                    datetime.fromisoformat(run["created_at"]) - timedelta(seconds=1)
                ).isoformat()


def wrong_transaction(data):
    for run in data[2]["runs"].values():
        for attempt in run["attempts"]:
            attempt["finished_at"] = (
                datetime.fromisoformat(attempt["finished_at"])
                + timedelta(microseconds=1)
            ).isoformat()


def before_generation(data):
    for run in data[2]["runs"].values():
        for attempt in run["attempts"]:
            attempt["started_at"] = (
                datetime.fromisoformat(run["created_at"]) - timedelta(seconds=1)
            ).isoformat()


def summary_before_adoption(data):
    for run in data[2]["runs"].values():
        run["summary"]["created_at"] = run["created_at"]


def recovery_mismatch(data):
    for run in data[2]["runs"].values():
        run["resume_count"] = 1


def payload_contract_missing(data):
    change_checkpoint(data, lambda cp: cp["result_payload"].pop("contract_checksum"))


def internal_field_missing(data):
    change_checkpoint(data, lambda cp: cp["result_payload"]["result"].pop("blocked"))


def internal_duty_value(data):
    def change(cp):
        cp["result_payload"]["result"]["duties"][0]["value"] = "UNKNOWN"

    change_checkpoint(data, change)


def internal_request(data):
    change_checkpoint(
        data, lambda cp: cp["result_payload"]["result"].update(request_sha256="b" * 64)
    )


def wrong_worker_contract(data):
    for run in data[2]["runs"].values():
        left, right = run["checkpoints"]
        left["contract_checksum"] = right["contract_checksum"]
        left["result_payload"]["contract_checksum"] = right["contract_checksum"]
        left["result_checksum"] = canonical_checksum(left["result_payload"])


def adoption_at_deadline(data):
    for run in data[2]["runs"].values():
        for cp in run["checkpoints"]:
            cp["adopted_at"] = run["generations"][0]["deadline_at"]
            next(a for a in run["attempts"] if a["worker"] == cp["worker"])[
                "finished_at"
            ] = cp["adopted_at"]


def observation_before_adoption(data):
    for row in data[1]["observations"]:
        row["observed_at"] = row["service_created_at"]
    data[2]["observations_payload_sha256"] = payload_sha256(data[1])


def adoption_before_request(data):
    for run in data[2]["runs"].values():
        for attempt in run["attempts"]:
            attempt["started_at"] = (
                datetime.fromisoformat(attempt["finished_at"]) + timedelta(seconds=1)
            ).isoformat()
            attempt["finished_at"] = (
                datetime.fromisoformat(attempt["finished_at"]) + timedelta(seconds=2)
            ).isoformat()


CASES = [
    (delete_contract, "checkpoint_missing:contract_checksum"),
    (outer_contract, "checkpoint_contract"),
    (both_contracts, "checkpoint_contract"),
    (payload_request, "checkpoint_payload_request"),
    (envelope_version, "checkpoint_version"),
    (commit_version, "checkpoint_commit_version"),
    (commit_answer, "checkpoint_commit_projection:answer_sha256"),
    (commit_failures, "checkpoint_document_failures"),
    (commit_blocked, "checkpoint_blocked"),
    (missing_adoption, "checkpoint_missing:adopted_at"),
    (future_attempts, "receipt_before_request"),
    (receipt_before_request, "receipt_before_request"),
    (wrong_transaction, "adoption_transaction_time"),
    (before_generation, "attempt_before_generation"),
    (summary_before_adoption, "summary_before_adoption"),
    (recovery_mismatch, "resume_generation_mismatch"),
    (payload_contract_missing, "checkpoint_payload_missing:contract_checksum"),
    (internal_field_missing, "checkpoint_commit_missing:blocked"),
    (internal_duty_value, "checkpoint_commit_content:duties"),
    (internal_request, "checkpoint_commit_projection:request_sha256"),
    (wrong_worker_contract, "checkpoint_contract"),
    (adoption_at_deadline, "adoption_generation_deadline"),
    (observation_before_adoption, "observation_before_summary_or_adoption"),
    (adoption_before_request, "adoption_before_request"),
]


@pytest.mark.parametrize("mutation,reason", CASES, ids=[fn.__name__ for fn, _ in CASES])
def test_complete_contract_counterexamples(baseline, mutation, reason):
    data = copy.deepcopy(baseline)
    mutation(data)
    registered, observed, bindings = data
    complete = completeness_report_v3(*data)
    assert complete["planned_pairs"] == complete["observed_pairs"] == 24
    assert complete["planned_model_tasks"] == 48
    assert not complete["complete"]
    assert any(reason + ":" in code for code in complete["failures"]), complete[
        "failures"
    ]
    result = score_sweep(registered, observed, bindings)
    assert not result["stage_acceptance"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    for repeat in (1, 2):
        assert result["repeats"][repeat]["q01_first_answer"]["denominator"] == 24
        assert result["repeats"][repeat]["q02_obligations"]["denominator"] == 144


def test_complete_normal_contract(baseline):
    result = score_sweep(*copy.deepcopy(baseline))
    assert result["stage_acceptance"], result["failed_gates"]
    assert result["score_version"] == "v1-07g-dev-score.v4"


def refresh(data):
    from procurement_review.precise_dev_sweep import _summarise
    from procurement_review.eval_f_validation import evidence_manifest

    for row in data[1]["observations"]:
        run = data[2]["runs"][row["run_id"]]
        for cp in run["checkpoints"]:
            cp["result_checksum"] = canonical_checksum(cp["result_payload"])
        run["projection"] = _summarise(
            {
                "execution": run["execution"],
                "status": row["status"],
                "summary": run["summary"]["summary_payload"],
                "execution_mode": "real",
                "error_code": None,
            }
        )
        row.update(copy.deepcopy(run["projection"]))
    data[2]["observations_payload_sha256"] = payload_sha256(data[1])
    data[0]["binding"]["adoption_evidence"] = evidence_manifest(data[2], "f" * 64)


@pytest.mark.parametrize("previous_status", ["expired", "superseded"])
def test_legitimate_recovery_reuses_adopted_generation(baseline, previous_status):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    run["resume_count"] = 1
    run["execution"]["metering"].update(explicit_recoveries=1, reused_branches=2)
    start = datetime.fromisoformat(run["created_at"]) + timedelta(seconds=20)
    run["generations"][0]["status"] = previous_status
    run["generations"].append(
        {
            "generation": 2,
            "run_id": run["generations"][0]["run_id"],
            "status": "completed",
            "created_at": start.isoformat(),
            "deadline_at": (start + timedelta(seconds=90)).isoformat(),
        }
    )
    finish = (start + timedelta(seconds=5)).isoformat()
    run["summary"]["created_at"] = finish
    run["events"][-1]["occurred_at"] = finish
    run["events"][-1]["generation"] = 2
    next(
        row
        for row in data[1]["observations"]
        if row["run_id"] in data[2]["runs"] and data[2]["runs"][row["run_id"]] is run
    )["observed_at"] = finish
    refresh(data)
    assert completeness_report_v3(*data)["complete"]
    assert score_sweep(*data)["stage_acceptance"]


def test_unknown_usage_is_kept_as_unknown(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    event = run["events"][0]
    event["output_summary"].update(
        usage_unknown=True, input_usage=None, output_usage=None
    )
    event["output_checksum"] = canonical_checksum(event["output_summary"])
    run["execution"]["policy"]["attempts"][0].update(
        usage_unknown=True, input_tokens=None, output_tokens=None
    )
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        run["execution"]["policy"]["attempts"]
    )
    run["execution"]["metering"].update(
        unknown_usage_attempts=1,
        usage_complete=False,
        input_tokens_known=100,
        output_tokens_known=40,
    )
    refresh(data)
    result = score_sweep(*data)
    assert result["stage_acceptance"], result["completeness"]["failures"]
    assert len(result["completeness"]["unknown_attempts"]) == 1


@pytest.mark.parametrize("workers", [("policy",), ("policy", "contract")])
def test_late_completed_receipt_is_recorded_without_adoption(baseline, workers):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    run["checkpoints"] = [
        cp for cp in run["checkpoints"] if cp["worker"] not in workers
    ]
    deadline = datetime.fromisoformat(run["generations"][0]["deadline_at"])
    for number, worker in enumerate(workers, 1):
        run["execution"][worker].update(
            outcome="failed", failure_code="deadline_exhausted", reported=[], spans=[]
        )
        attempt = next(a for a in run["attempts"] if a["worker"] == worker)
        attempt.update(
            discarded_late=True,
            finished_at=(deadline + timedelta(seconds=number)).isoformat(),
        )
        event = next(e for e in run["events"] if e.get("worker") == worker)
        event["occurred_at"] = attempt["finished_at"]
    run["events"].sort(key=lambda event: datetime.fromisoformat(event["occurred_at"]))
    for number, item in enumerate(run["events"], 1):
        item["event_ordinal"] = number
    refresh(data)
    result = completeness_report_v3(*data)
    assert result["complete"], result["failures"]
    assert result["planned_pairs"] == 24
    assert not score_sweep(*data)["stage_acceptance"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("adopted_at", None),
        ("adopted_at", "2026-10-08T00:00:00"),
        ("generation", True),
        ("attempt_no", "1"),
    ],
)
def test_current_binding_rejects_bad_types_and_clocks(baseline, field, value):
    data = copy.deepcopy(baseline)
    first_run(data)["checkpoints"][0][field] = value
    if field == "adopted_at" and value is None:
        result = completeness_report_v3(*data)
        assert any("checkpoint_missing:adopted_at:" in f for f in result["failures"])
    else:
        with pytest.raises(ValueError, match="v3_input_(time|type):"):
            completeness_report_v3(*data)


def test_equivalent_timezone_instants(baseline):
    data = copy.deepcopy(baseline)
    for run in data[2]["runs"].values():
        for cp in run["checkpoints"]:
            cp["adopted_at"] = (
                datetime.fromisoformat(cp["adopted_at"]).astimezone(UTC).isoformat()
            )
        for attempt in run["attempts"]:
            for key in ("started_at", "finished_at"):
                attempt[key] = (
                    datetime.fromisoformat(attempt[key]).astimezone(UTC).isoformat()
                )
    assert score_sweep(*data)["stage_acceptance"]


def write_inputs(tmp_path, baseline):
    from procurement_review.eval_f_validation import evidence_manifest

    data = copy.deepcopy(baseline)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {**data[2], "binding_version": "v1-07g-db-binding.v3"}, sort_keys=True
        )
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    data[0]["binding"]["adoption_evidence"] = evidence_manifest(data[2], source_hash)
    data[2]["source_evidence_file_sha256"] = source_hash
    paths = {"source_bindings": source}
    for name, value in zip(("registration", "observations", "bindings"), data):
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_text(json.dumps(value, sort_keys=True))
    data[2]["observations_file_sha256"] = hashlib.sha256(
        paths["observations"].read_bytes()
    ).hexdigest()
    paths["bindings"].write_text(json.dumps(data[2], sort_keys=True))
    return paths


def score_cli(paths, output, command="score", include_source=True):
    args = [
        sys.executable,
        "-m",
        "procurement_review.precise_dev_score",
        command,
        str(paths["registration"]),
        str(paths["observations"]),
        "--bindings",
        str(paths["bindings"]),
        "--output",
        str(output),
    ]
    if include_source:
        args.extend(("--source-bindings", str(paths["source_bindings"])))
    return subprocess.run(args, text=True, capture_output=True, check=False)


def test_score_recheck_freeze_and_full_artifact_rejection(tmp_path, baseline):
    from procurement_review.eval_f_validation import freeze_sweep_v4

    paths = write_inputs(tmp_path, baseline)
    score = tmp_path / "score.json"
    completed = score_cli(paths, score)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(score.read_text())
    assert result["stage_acceptance"]
    assert result["inputs"]["file_sha256"] == {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
    }
    for repeat in ("1", "2"):
        assert result["repeats"][repeat]["q01_first_answer"]["numerator"] == 24
        assert result["repeats"][repeat]["q02_obligations"]["numerator"] == 144
    assert score_cli(paths, score, "recheck").returncode == 0
    freeze = tmp_path / "freeze.json"
    with pytest.raises(ValueError, match="v4_freeze_retired_use_v5"):
        freeze_sweep_v4(
            paths["registration"],
            paths["observations"],
            paths["bindings"],
            paths["source_bindings"],
            score,
            freeze,
        )
    assert not freeze.exists()
    result["repeats"]["1"]["q01_first_answer"]["numerator"] = 23
    score.write_text(json.dumps(result))
    assert score_cli(paths, score, "recheck").returncode != 0
    refused = tmp_path / "refused-freeze.json"
    with pytest.raises(ValueError, match="v4_freeze_full_score_mismatch"):
        freeze_sweep_v4(
            paths["registration"],
            paths["observations"],
            paths["bindings"],
            paths["source_bindings"],
            score,
            refused,
        )
    assert not refused.exists()


def test_cli_requires_unchanged_independent_source(tmp_path, baseline):
    paths = write_inputs(tmp_path, baseline)
    output = tmp_path / "score.json"
    missing = score_cli(paths, output, include_source=False)
    assert missing.returncode != 0 and "needs --source-bindings" in missing.stderr
    paths["source_bindings"].write_text(paths["source_bindings"].read_text() + "\n")
    changed = score_cli(paths, output)
    assert (
        changed.returncode != 0 and "v4_source_evidence_file_binding" in changed.stderr
    )
    assert not output.exists()


def test_full_historical_v3_contract_uses_its_own_reader(baseline):
    data = copy.deepcopy(baseline)
    data[0]["extends"]["extension_version"] = "v1-07-truth-extension.v2"
    data[0]["binding"].pop("adoption_evidence")
    data[2]["binding_version"] = "v1-07g-db-binding.v3"
    result = score_sweep(*data)
    assert result["stage_acceptance"]
    assert result["score_version"] == "v1-07g-dev-score.v3"


@pytest.mark.parametrize(
    "container,field,reason",
    [
        ("generations", "run_id", "generation_parent_run"),
        ("events", "run_id", "event_parent_run"),
        ("events", "trace_id", "event_trace_identity"),
    ],
)
def test_generation_and_trace_cannot_change_parent(baseline, container, field, reason):
    data = copy.deepcopy(baseline)
    first_run(data)[container][0][field] = "other-run-or-trace"
    result = completeness_report_v3(*data)
    assert not result["complete"]
    assert result["planned_model_tasks"] == 48
    assert any(reason + ":" in f for f in result["failures"])


def format_correction_fixture(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    first = copy.deepcopy(run["attempts"][0])
    start = datetime.fromisoformat(run["created_at"])
    first.update(
        outcome="format_corrected",
        first_error="invalid_schema",
        last_error="invalid_schema",
        finished_at=(start + timedelta(seconds=1)).isoformat(),
    )
    run["attempts"][0].update(
        attempt_no=2, started_at=(start + timedelta(seconds=2)).isoformat()
    )
    run["attempts"].insert(0, first)
    projected = copy.deepcopy(run["execution"]["policy"]["attempts"][0])
    projected.update(
        error_code="invalid_schema", correctable=True, input_tokens=10, output_tokens=4
    )
    run["execution"]["policy"]["attempts"][0]["attempt_no"] = 2
    run["execution"]["policy"]["attempts"].insert(0, projected)
    run["checkpoints"][0]["attempt_no"] = 2
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        run["execution"]["policy"]["attempts"]
    )
    failed = copy.deepcopy(run["events"][0])
    failed["event_type"] = "step_failed"
    failed["occurred_at"] = first["finished_at"]
    failed["output_summary"].update(
        outcome="failed",
        attempt_error="invalid_schema",
        input_usage=10,
        output_usage=4,
        next_correction_kind="contract",
    )
    failed["output_checksum"] = canonical_checksum(failed["output_summary"])
    run["events"][0]["attempt"] = 2
    run["events"][0]["output_summary"]["attempt_no"] = 2
    run["events"][0]["output_checksum"] = canonical_checksum(
        run["events"][0]["output_summary"]
    )
    run["events"].insert(0, failed)
    for number, event in enumerate(run["events"], 1):
        event["event_ordinal"] = number
    run["execution"]["metering"].update(
        physical_attempts=3,
        attempts_opened=3,
        proven_sends=3,
        corrections=1,
        contract_corrections=1,
        input_tokens_known=210,
        output_tokens_known=84,
        branch_latency_ms=[["contract", 1000], ["policy", 2000]],
    )
    refresh(data)
    return data


def test_legal_format_correction_settles_failed_receipt(baseline):
    data = format_correction_fixture(baseline)
    result = completeness_report_v3(*data)
    assert result["complete"], result["failures"]
