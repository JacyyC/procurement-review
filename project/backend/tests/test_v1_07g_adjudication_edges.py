"""Additional independent-audit and JSON-boundary regressions for G semantics v3."""

import copy
from datetime import datetime, timedelta

import pytest

from procurement_review.eval_f_obligations import judge_obligation_v3, judge_span_v3
from procurement_review.eval_f_text import semantic_value_v3
from procurement_review.eval_g_report import build_report_v6
from procurement_review.gold_v3_contracts import payload_sha256
from procurement_review.precise_dev_score import score_sweep
from test_v1_07g_score_v3 import fixture_v3, comparison_v3


@pytest.fixture(scope="module")
def baseline():
    return fixture_v3()


@pytest.mark.parametrize(
    "mutation,path",
    [
        ("repeat", "repeat"),
        ("spans", "spans"),
        ("events", "events"),
        ("attempts", "attempts"),
    ],
)
def test_json_types_are_controlled_rejections(baseline, mutation, path):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    run = bindings["runs"][row["run_id"]]
    if mutation == "repeat":
        row["repeat"] = True
    elif mutation == "spans":
        row["spans"] = {"silently_skipped": True}
    else:
        run[mutation] = "not-a-record-array"
    with pytest.raises(ValueError, match="v3_input_type:.*" + path):
        score_sweep(registered, observed, bindings)


def test_null_source_version_and_integer_blocking_are_not_equal(baseline):
    registered, observed, _ = copy.deepcopy(baseline)
    row = observed["observations"][0]
    span = row["spans"][0]
    span["source_version"] = None
    verdict = judge_span_v3(
        registered, row["case_id"], span, registered["material_text"]
    )
    assert not verdict["supported"]
    assert "wrong_source_version" in verdict["failures"]
    truth = registered["truth"][row["case_id"]]["expectations"][0]
    finding = row["findings"][0]
    finding["blocking"] = int(finding["blocking"])
    verdict = judge_obligation_v3(
        truth, [], finding, registered["truth_extension"][row["case_id"]]
    )
    assert "wrong_blocking" in verdict["evidence_failures"]


@pytest.mark.parametrize("dimension", ["scope_id", "subject_type", "content_sha256"])
def test_conflict_source_content_and_scope_are_bound(baseline, dimension):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = next(
        row
        for row in observed["observations"]
        if row["repeat"] == 1 and row["conflicts"]
    )
    for name in ("left", "right"):
        side = row["conflicts"][0][name]
        if dimension == "content_sha256":
            side["evidence"][0][dimension] = "0" * 64
        else:
            side["proposition"][dimension] = (
                "tenant-outside" if dimension == "scope_id" else "supplier"
            )
    score = score_sweep(registered, observed, bindings)
    assert score["repeats"][1]["q06_events"]["conflict_matched"] == 0
    assert score["repeats"][1]["q06_events"]["conflict_expected"] == 1
    assert score["repeats"][1]["q07_citations"]["out_of_scope_citations"] == 2
    assert "repeat-1/Q-06-conflict-recall" in score["failed_gates"]
    assert "repeat-1/Q-07-out-of-scope" in score["failed_gates"]


def test_frozen_generation_window_cannot_expand(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    run = bindings["runs"][row["run_id"]]
    generation = run["generations"][0]
    generation["deadline_at"] = (
        datetime.fromisoformat(generation["created_at"]) + timedelta(minutes=55)
    ).isoformat()
    result = score_sweep(registered, observed, bindings)
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert (
        "frozen_generation_deadline:" + row["run_id"]
        in result["completeness"]["failures"]
    )


@pytest.mark.parametrize("amount", ["1,001.37", "1001.370", 1001.37])
def test_amount_formats_share_one_complete_decimal_meaning(amount):
    assert semantic_value_v3("amount", amount) == semantic_value_v3("amount", "1001.37")


@pytest.mark.parametrize("amount", ["10,01.37", "1,,001.37", "1,001.37美元", True])
def test_amount_normalization_never_drops_invalid_content(amount):
    assert semantic_value_v3("amount", amount) is None


@pytest.mark.parametrize("mutation", ["cross_case_tokens", "open_receipt"])
def test_q12_per_observation_metering_cannot_hide_in_arm_totals(mutation):
    plan, observed, kwargs, ledgers = comparison_v3()
    arm = next(iter(ledgers))
    if mutation == "open_receipt":
        ledgers[arm][0]["state"] = "open"
        reason = "receipt_attempt_not_settled"
    else:
        first = ledgers[arm][0]
        second = next(
            row
            for row in ledgers[arm]
            if row["task_key"].split(".")[0] != first["task_key"].split(".")[0]
        )
        first["receipt"]["input_tokens"] += 1
        second["receipt"]["input_tokens"] -= 1
        for row in (first, second):
            row["receipt_sha256"] = payload_sha256(row["receipt"])
        reason = "ledger_observation"
    report = build_report_v6(plan, observed, attempts_by_arm=ledgers, **kwargs)
    assert not report["Q-12"]["complete"]
    assert any(reason in failure for failure in report["Q-12"]["incomplete_reasons"])


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("summary_digest", "summary_digest"),
        ("summary_identity", "summary_identity"),
        ("summary_execution", "execution_summary_mismatch"),
        ("projection", "execution_projection_missing:spans"),
        ("checkpoint_digest", "checkpoint_identity_or_digest"),
        ("checkpoint_request", "checkpoint_request"),
        ("receipt_digest", "receipt_digest"),
        ("receipt_model", "receipt_model"),
        ("tokens", "known_tokens_missing"),
        ("usage_marker", "usage_marker_missing"),
        ("unknown_tokens", "unknown_usage_has_known_tokens"),
        ("snapshot", "run_binding:temperature"),
        ("event_chain", "event_chain_gap"),
        ("illegal_human", "illegal_post_observation_transition"),
    ],
)
def test_each_ledger_failure_reports_its_actual_cause(baseline, mutation, code):
    from procurement_review.trace import canonical_checksum

    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    run = bindings["runs"][row["run_id"]]
    receipt_event = run["events"][0]
    receipt = receipt_event["output_summary"]
    if mutation == "summary_digest":
        run["summary"]["summary_checksum"] = "sha256:" + "0" * 64
    elif mutation == "summary_identity":
        run["summary"]["summary_payload"]["run_id"] = "run-wrong"
        run["summary"]["summary_checksum"] = canonical_checksum(
            run["summary"]["summary_payload"]
        )
    elif mutation == "summary_execution":
        run["execution"]["summary"] = {}
    elif mutation == "projection":
        del run["projection"]["spans"]
    elif mutation == "checkpoint_digest":
        run["checkpoints"][0]["result_checksum"] = "sha256:" + "0" * 64
    elif mutation == "checkpoint_request":
        run["checkpoints"][0]["input_checksum"] = "sha256:" + "0" * 64
    elif mutation == "receipt_digest":
        receipt_event["output_checksum"] = "sha256:" + "0" * 64
    elif mutation == "receipt_model":
        receipt["receipt_model"] = "other-model"
    elif mutation == "tokens":
        del receipt["output_usage"]
    elif mutation == "usage_marker":
        del receipt["usage_unknown"]
    elif mutation == "unknown_tokens":
        receipt["usage_unknown"] = True
    elif mutation == "snapshot":
        run["snapshot"]["temperature"] = 0.5
    elif mutation == "event_chain":
        run["events"][0]["event_ordinal"] = 90
    else:
        run["events"].append(
            {
                "event_ordinal": 4,
                "event_type": "confirmation_recorded",
                "occurred_at": "2026-10-07T04:00:00+00:00",
                "output_summary": {"action": "confirm", "status": "cancelled"},
            }
        )
    result = score_sweep(registered, observed, bindings)
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert any(
        failure.startswith(code + ":") for failure in result["completeness"]["failures"]
    )


@pytest.mark.parametrize("status", ["interrupted", "cancelled"])
def test_nonbusiness_terminal_failures_keep_the_planned_pipeline_denominator(
    baseline, status
):
    registered, observed, bindings = copy.deepcopy(baseline)
    observed["observations"][0]["status"] = status
    result = score_sweep(registered, observed, bindings)
    assert result["repeats"][1]["q09_pipeline"] == {
        "non_expected_pipeline_failures": 1,
        "planned_runs": 12,
    }
    assert "repeat-1/Q-09-pipeline-failures" in result["failed_gates"]


def test_registration_marks_optional_and_inapplicable_without_output(baseline):
    registered, _, _ = baseline
    disposition = registered["truth_extension"]["V107-D-9001"]["field_disposition"]
    assert disposition["O-05"]["contract.subject"] == "required"
    assert disposition["O-04"]["policy.exception_basis"] == "optional"
    assert disposition["O-04"]["contract.superseding_amendment"] == "not_applicable"


@pytest.mark.parametrize(
    "field",
    ["effective_at", "scope_id", "subject_type", "content_sha256", "source_version"],
)
def test_direct_span_adjudication_requires_full_identity(baseline, field):
    from procurement_review.eval_f_lineage import execution_projection_v3

    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    enriched = execution_projection_v3(row, bindings["runs"][row["run_id"]])
    span = enriched["spans"][0]
    assert judge_span_v3(registered, row["case_id"], span, registered["material_text"])[
        "supported"
    ]
    span.pop(field)
    verdict = judge_span_v3(
        registered, row["case_id"], span, registered["material_text"]
    )
    assert not verdict["supported"]
    assert "wrong_" + field in verdict["failures"]


@pytest.mark.parametrize("shape", ["attempt_array", "attempt_row", "task_key"])
def test_q12_ledger_shapes_are_controlled_rejections(shape):
    plan, observed, kwargs, ledgers = comparison_v3()
    arm = next(iter(ledgers))
    if shape == "attempt_array":
        ledgers[arm] = {"not": "an array"}
    elif shape == "attempt_row":
        ledgers[arm][0] = []
    else:
        ledgers[arm][0]["task_key"] = ["unhashable"]
    with pytest.raises(ValueError, match="q12_ledger_shape"):
        build_report_v6(plan, observed, attempts_by_arm=ledgers, **kwargs)


@pytest.mark.parametrize(
    "container,field",
    [
        ("summary", "summary_checksum"),
        ("checkpoints", "result_checksum"),
        ("events", "event_ordinal"),
    ],
)
def test_binding_missing_structural_fields_are_controlled(baseline, container, field):
    registered, observed, bindings = copy.deepcopy(baseline)
    run = bindings["runs"][observed["observations"][0]["run_id"]]
    target = run[container] if container == "summary" else run[container][0]
    target.pop(field)
    with pytest.raises(ValueError, match="v3_input_missing:.*" + field):
        score_sweep(registered, observed, bindings)


def test_v3_cli_without_bindings_writes_failed_verdict(
    baseline, tmp_path, monkeypatch, capsys
):
    import json
    from procurement_review.precise_dev_score import main

    registered, observed, _ = baseline
    registration_path = tmp_path / "registration.json"
    observation_path = tmp_path / "observations.json"
    output_path = tmp_path / "score.json"
    registration_path.write_text(json.dumps(registered))
    observation_path.write_text(json.dumps(observed))
    monkeypatch.setattr(
        "sys.argv",
        [
            "score",
            "score",
            str(registration_path),
            str(observation_path),
            "--output",
            str(output_path),
        ],
    )
    main()
    result = json.loads(output_path.read_text())
    assert result["stage_acceptance"] is False
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert "binding_version:" in result["completeness"]["failures"]
    assert json.loads(capsys.readouterr().out)["stage_acceptance"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("branches", ["policy"]),
        ("max_attempts_per_branch", 20),
        ("requested_model", "other"),
        ("resolved_model", "other"),
        ("expected_rules_version", "other"),
        ("expected_summary_schema", "other"),
        ("product_implementation_sha256", "0" * 64),
    ],
)
def test_all_registration_bindings_are_independently_pinned(baseline, field, value):
    from procurement_review.precise_candidate import verify_registration_v3

    registered, _, _ = copy.deepcopy(baseline)
    registered["binding"][field] = value
    assert "binding_registration_mismatch:" + field in verify_registration_v3(
        registered
    )


@pytest.mark.parametrize(
    "mutation", ["native_path_null", "native_path_missing", "span_worker_missing"]
)
def test_nested_proposition_shapes_are_controlled(baseline, mutation):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    if mutation == "span_worker_missing":
        row["spans"][0].pop("worker")
    else:
        proposition = bindings["runs"][row["run_id"]]["execution"]["policy"]["spans"][
            0
        ]["proposition"]
        if mutation == "native_path_null":
            proposition["field_path"] = None
        else:
            proposition.pop("field_path")
    with pytest.raises(ValueError, match="v3_input_(missing|type)"):
        score_sweep(registered, observed, bindings)


def test_unplanned_binding_run_cannot_be_ignored(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    bindings["runs"]["unplanned-run"] = copy.deepcopy(
        next(iter(bindings["runs"].values()))
    )
    result = score_sweep(registered, observed, bindings)
    assert not result["stage_acceptance"]
    assert "unplanned_binding_runs:" in result["completeness"]["failures"]


@pytest.mark.parametrize(
    "field", ["execution_version", "summary_schema", "rules_version"]
)
def test_self_consistent_projection_versions_still_match_the_plan(baseline, field):
    from procurement_review.precise_dev_sweep import _summarise
    from procurement_review.trace import canonical_checksum

    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    run = bindings["runs"][row["run_id"]]
    row[field] = "other-version"
    if field == "execution_version":
        run["execution"][field] = row[field]
    else:
        key = "schema_version" if field == "summary_schema" else field
        run["summary"]["summary_payload"][key] = row[field]
        run["execution"]["summary"][key] = row[field]
        run["summary"]["summary_checksum"] = canonical_checksum(
            run["summary"]["summary_payload"]
        )
    run["projection"] = _summarise(
        {
            "execution": run["execution"],
            "summary": run["summary"]["summary_payload"],
            "status": row["status"],
            "execution_mode": "real",
            "error_code": row.get("error_code"),
        }
    )
    bindings["observations_payload_sha256"] = payload_sha256(observed)
    result = score_sweep(registered, observed, bindings)
    assert not result["stage_acceptance"]
    assert (
        f"projection_version:{field}:" + row["run_id"]
        in result["completeness"]["failures"]
    )


def test_contract_claim_failure_keeps_the_policy_first_answer(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    span = next(
        span
        for span in observed["observations"][0]["spans"]
        if span["field"] == "subject"
    )
    span["value"] += "及十艘航空母舰"
    result = score_sweep(registered, observed, bindings)
    report = result["repeats"][1]
    assert report["q01_first_answer"]["numerator"] == 23
    assert report["q01_first_answer"]["denominator"] == 24
    assert report["q02_obligations"]["numerator"] == 143
    assert report["q07_citations"]["unsupported_propositions"] == 1
    assert report["q07_citations"]["adopted_propositions"] == 551
    assert result["gates"]["repeat-1"]["Q-02"]["meets"] is True
    assert result["gates"]["repeat-1"]["Q-07-semantic-support"]["meets"] is True
    assert result["gates"]["repeat-1"]["Q-01"]["meets"] is False
