"""Independent summary, transaction and settlement counterexamples for G v5."""

import copy
from datetime import datetime, timedelta

import pytest

from procurement_review.eval_f_lineage import verify_sweep_binding_v3
from procurement_review.precise_dev_paired_metering import completeness_report_v3
from procurement_review.precise_dev_score import score_sweep
from procurement_review.trace import canonical_checksum
from v1_07g_consumption_support import fixture_v5, recovery_fixture, seal, sort_events


@pytest.fixture(scope="module")
def baseline():
    return fixture_v5()


def first_run(data):
    return next(iter(data[2]["runs"].values()))


def rejected(data, reason):
    for result in (verify_sweep_binding_v3(*data), completeness_report_v3(*data)):
        assert not result["complete"]
        assert any(reason in item for item in result["failures"]), result["failures"]
        assert (result["planned_pairs"], result["planned_model_tasks"]) == (24, 48)
    score = score_sweep(*data)
    assert not score["stage_acceptance"]
    assert not score["gates"]["cross-repeat"]["Q-10-ledger-complete"]["meets"]
    for repeat in score["repeats"].values():
        assert repeat["q01_first_answer"]["denominator"] == 24
        assert repeat["q02_obligations"]["denominator"] == 144


def test_full_summary_positive(baseline):
    assert completeness_report_v3(*baseline)["complete"]
    score = score_sweep(*baseline)
    assert score["score_version"] == "v1-07g-dev-score.v5"
    run = first_run(baseline)
    assert len(run["execution"]["decision"]["findings"]) == 12
    assert len(run["summary"]["summary_payload"]["findings"]) == 10
    assert run["summary"]["summary_payload"]["highest_risk"] == "low"


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("recommendation", "recommend_reject", "summary_recommendation"),
        ("highest_risk", "critical", "summary_highest_risk"),
        ("human_review_status", "required", "execution_contract_type:summary"),
        ("findings", [], "execution_contract_type:summary"),
        (
            "missing_information",
            ["invented_missing"],
            "execution_contract_type:summary",
        ),
        ("failure_reasons", ["invented_failure"], "execution_contract_type:summary"),
        (
            "is_final_decision",
            True,
            "execution_contract_type:summary.is_final_decision",
        ),
    ],
)
def test_synchronized_wrong_public_summary(baseline, field, value, reason):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    summary = run["summary"]["summary_payload"]
    summary[field] = value
    run["summary"]["summary_checksum"] = canonical_checksum(summary)
    run["execution"]["summary"] = copy.deepcopy(summary)
    # Even an internally consistent source digest must not legitimize a wrong projection.
    seal(data)
    rejected(data, reason)


@pytest.mark.parametrize(
    "field,value",
    [
        ("statement", "rules 3.1.0 O-01: invented_reason"),
        ("worker", "contract"),
        ("category", "contract_terms"),
        ("risk_level", "high"),
        ("assertion", "risk_present"),
        ("recommended_action", "invented_action"),
    ],
)
def test_finding_projection_fields(baseline, field, value):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    run["summary"]["summary_payload"]["findings"][0][field] = value
    run["execution"]["summary"] = copy.deepcopy(run["summary"]["summary_payload"])
    run["summary"]["summary_checksum"] = canonical_checksum(
        run["summary"]["summary_payload"]
    )
    seal(data)
    rejected(data, "summary_finding_projection:O-01." + field)


@pytest.mark.parametrize(
    "field", ["rules_version", "procurement", "stop_reason", "summary"]
)
def test_execution_required_fields(baseline, field):
    data = copy.deepcopy(baseline)
    first_run(data)["execution"].pop(field)
    rejected(data, "execution_contract_")


def test_deterministic_procurement_calls_no_model(baseline):
    data = copy.deepcopy(baseline)
    first_run(data)["execution"]["procurement"]["model_calls"] = 1
    rejected(data, "execution_contract_type:procurement.model_calls")


@pytest.mark.parametrize("value", [False, 0.0])
def test_procurement_literal_zero_keeps_its_integer_type(baseline, value):
    data = copy.deepcopy(baseline)
    first_run(data)["execution"]["procurement"]["model_calls"] = value
    seal(data)
    rejected(data, "execution_contract_type:procurement.model_calls")


@pytest.mark.parametrize("value", [False, 0, 0.0])
def test_security_literal_false_keeps_its_boolean_type(baseline, value):
    data = copy.deepcopy(baseline)
    first_run(data)["execution"]["security"].append(
        dict(
            schema_version="security-event.v3",
            code="benign_refusal",
            origin="output",
            location_sha256="0" * 64,
            forbidden_export_succeeded=value,
        )
    )
    seal(data)
    if type(value) is bool:
        assert completeness_report_v3(*data)["complete"]
    else:
        rejected(data, "execution_contract_type:security.0.forbidden_export_succeeded")


@pytest.mark.parametrize(
    "step",
    [
        "review_run",
        "validate_input",
        "create_run",
        "deterministic_aggregator",
        "review_workflow",
    ],
)
@pytest.mark.parametrize(
    "mutation", ["delete", "duplicate", "failed", "digest", "content", "generation"]
)
def test_every_required_event(baseline, step, mutation):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    selected = next(e for e in run["events"] if e["step"] == step)
    if mutation == "delete":
        run["events"].remove(selected)
    elif mutation == "duplicate":
        duplicate = copy.deepcopy(selected)
        duplicate["occurred_at"] = (
            datetime.fromisoformat(selected["occurred_at"]) + timedelta(microseconds=4)
        ).isoformat()
        run["events"].append(duplicate)
    elif mutation == "failed":
        selected["event_type"] = "step_failed"
    elif mutation == "digest":
        selected["output_checksum"] = "sha256:" + "0" * 64
    elif mutation == "content":
        selected["output_summary"] = {"status": "invented"}
        selected["output_checksum"] = canonical_checksum(selected["output_summary"])
    else:
        selected["generation"] = 1 if selected["generation"] is None else None
    sort_events(run)
    rejected(
        data,
        "trace_"
        if not (step == "deterministic_aggregator" and mutation == "generation")
        else "summary_generation_identity",
    )


def test_aggregator_cannot_precede_adoption(baseline):
    data = copy.deepcopy(baseline)
    run = first_run(data)
    next(e for e in run["events"] if e["step"] == "deterministic_aggregator")[
        "occurred_at"
    ] = (
        datetime.fromisoformat(run["created_at"]) + timedelta(milliseconds=2)
    ).isoformat()
    sort_events(run)
    rejected(data, "summary_before_adopted_receipt")


def test_aggregator_counts_keep_their_integer_type(baseline):
    data = copy.deepcopy(baseline)
    event = next(
        e for e in first_run(data)["events"] if e["step"] == "deterministic_aggregator"
    )
    event["output_summary"]["finding_count"] = float(
        event["output_summary"]["finding_count"]
    )
    event["output_checksum"] = canonical_checksum(event["output_summary"])
    rejected(data, "trace_aggregator_content")


def test_recovery_positive(baseline):
    data = recovery_fixture(baseline)
    assert completeness_report_v3(*data)["complete"]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("future", "summary_generation_window"),
        ("recovery_missing", "trace_recovery_count"),
        ("interruption_missing", "trace_recovery_precondition"),
        ("recovery_content", "trace_recovery_transaction"),
    ],
)
def test_recovery_transactions(baseline, mutation, reason):
    data = recovery_fixture(baseline)
    run = first_run(data)
    if mutation == "future":
        start = datetime.fromisoformat(run["created_at"]) + timedelta(days=365)
        run["generations"][-1].update(
            created_at=start.isoformat(),
            deadline_at=(start + timedelta(seconds=90)).isoformat(),
        )
    elif mutation == "recovery_content":
        next(e for e in run["events"] if e["step"] == "review_recovery")[
            "output_summary"
        ] = {"status": "failed"}
    else:
        run["events"] = [
            e
            for e in run["events"]
            if not (
                e["step"] == "review_recovery"
                if mutation == "recovery_missing"
                else e["output_summary"].get("status") == "interrupted"
            )
        ]
    sort_events(run)
    rejected(data, reason)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("before_diagnostic", "correction_before_diagnostic"),
        ("before_settlement", "correction_before_settlement"),
        ("no_kind", "correction_precondition"),
        ("last_error", "ledger_last_error"),
        ("first_error", "ledger_first_error"),
        ("outcome", "ledger_settlement_outcome"),
    ],
)
def test_correction_causality_and_errors(mutation, reason):
    data = fixture_v5(correction=True)
    run = first_run(data)
    first, second = run["attempts"][:2]
    start = datetime.fromisoformat(run["created_at"])
    if mutation == "before_diagnostic":
        second["started_at"] = (start + timedelta(milliseconds=500)).isoformat()
    elif mutation == "before_settlement":
        first["finished_at"] = (start + timedelta(seconds=3)).isoformat()
    elif mutation == "no_kind":
        next(e for e in run["events"] if e["step"] == "model_attempt")[
            "output_summary"
        ]["next_correction_kind"] = None
    elif mutation in {"last_error", "first_error"}:
        first[mutation] = "wrong_error"
    else:
        first["outcome"] = "failed"
    rejected(data, reason)
    assert "repeat-1/Q-01" in score_sweep(*data)["failed_gates"]


def test_legal_correction_keeps_q01_failure():
    data = fixture_v5(correction=True)
    score = score_sweep(*data)
    assert score["completeness"]["complete"]
    assert "repeat-1/Q-01" in score["failed_gates"]
    assert score["gates"]["cross-repeat"]["Q-10-ledger-complete"]["meets"]


@pytest.mark.parametrize("field", ["first_error", "last_error"])
def test_successful_attempt_has_no_error(baseline, field):
    data = copy.deepcopy(baseline)
    first_run(data)["attempts"][0][field] = "invalid_schema"
    rejected(data, "ledger_" + field)


def test_unknown_consumer_version_fails_closed(baseline):
    data = copy.deepcopy(baseline)
    data[0]["binding"]["consumption_version"] = "v1-07g-consumption.v999"
    with pytest.raises(ValueError, match="consumption_version_unsupported"):
        score_sweep(*data)


def test_direct_v4_entry_cannot_skip_v5_projection(baseline):
    from procurement_review.eval_f_validation import verify_sweep_binding_v4

    data = copy.deepcopy(baseline)
    run = first_run(data)
    run["execution"]["summary"]["recommendation"] = "recommend_reject"
    result = verify_sweep_binding_v4(*data)
    assert not result["complete"]
    assert any("execution_summary_mismatch" in code for code in result["failures"])


def test_one_sided_consumer_downgrade_fails_closed(baseline):
    data = copy.deepcopy(baseline)
    del data[0]["binding"]["consumption_version"]
    with pytest.raises(ValueError, match="binding_consumption_version_mismatch"):
        verify_sweep_binding_v3(*data)


def test_contract_correction_needs_a_correctable_contract_error():
    data = fixture_v5(correction=True)
    run = first_run(data)
    run["attempts"][0].update(first_error="model_timeout", last_error="model_timeout")
    run["execution"]["policy"]["attempts"][0]["error_code"] = "model_timeout"
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        run["execution"]["policy"]["attempts"]
    )
    receipt = next(e for e in run["events"] if e["step"] == "model_attempt")
    receipt["output_summary"]["attempt_error"] = "model_timeout"
    receipt["output_checksum"] = canonical_checksum(receipt["output_summary"])
    seal(data)
    rejected(data, "ledger_contract_error")


def test_recovery_event_has_no_worker_attempt_identity(baseline):
    data = recovery_fixture(baseline)
    run = first_run(data)
    next(e for e in run["events"] if e["step"] == "review_recovery")["worker"] = (
        "policy"
    )
    rejected(data, "trace_recovery_identity")


def test_unregistered_case_has_named_rejection(baseline):
    data = copy.deepcopy(baseline)
    first_run(data)["case_id"] = "V107-G-UNREGISTERED"
    rejected(data, "execution_case_unregistered")


@pytest.mark.parametrize("part", ["execution", "summary"])
def test_source_projection_requires_named_complete_containers(baseline, part):
    from procurement_review.eval_g_consumption import source_projections

    data = copy.deepcopy(baseline)
    del first_run(data)[part]
    with pytest.raises(ValueError, match="v5_source_projection_type:" + part):
        source_projections(data[2])


def test_explicit_reuse_requires_unchanged_product_bytes(baseline):
    from procurement_review.eval_g_reuse import reuse_implementation
    from procurement_review.precise_dependencies import PACKAGE

    data = copy.deepcopy(baseline)
    data[0]["binding"]["product_reuse"] = {
        "contract": "v1-07g-product-reuse.v1",
        "inventory_source": (PACKAGE / "precise_dependencies.py").read_text(),
    }
    assert (
        reuse_implementation(data[0])
        == data[0]["binding"]["product_implementation_sha256"]
    )
    data[0]["implementation"]["api"] = "0" * 64
    with pytest.raises(ValueError, match="product_reuse_dependency_changed:api"):
        reuse_implementation(data[0])


def test_reuse_inventory_semantics_cannot_change(baseline):
    import hashlib
    from procurement_review.eval_g_reuse import reuse_implementation
    from procurement_review.precise_dependencies import PACKAGE

    data = copy.deepcopy(baseline)
    text = (
        (PACKAGE / "precise_dependencies.py")
        .read_text()
        .replace('"precise_run_runner.py",', '"unapproved_runner.py",', 1)
    )
    data[0]["binding"]["product_reuse"] = {
        "contract": "v1-07g-product-reuse.v1",
        "inventory_source": text,
    }
    data[0]["implementation"]["precise_dependencies"] = hashlib.sha256(
        text.encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="product_reuse_inventory_semantics"):
        reuse_implementation(data[0])


@pytest.mark.parametrize("field", ["summary", "events", "generations", "attempts"])
def test_missing_run_container_is_a_named_gate_failure(baseline, field):
    data = copy.deepcopy(baseline)
    del first_run(data)[field]
    rejected(data, "run_missing:" + field)


def test_unknown_trace_step_is_not_a_product_event(baseline):
    from v1_07g_consumption_support import event

    data = copy.deepcopy(baseline)
    run = first_run(data)
    at = datetime.fromisoformat(run["created_at"]) + timedelta(milliseconds=500)
    run["events"].append(event(run, "unregistered_product_step", {}, at))
    sort_events(run)
    rejected(data, "trace_step_contract")


@pytest.mark.parametrize(
    "step,kind,body",
    [
        ("review_run", "step_completed", {}),
        ("review_workflow", "run_started", {}),
        ("review_workflow", "step_failed", {"status": "invented"}),
        (
            "review_workflow",
            "step_failed",
            {"status": "interrupted", "reason": "invented"},
        ),
    ],
)
def test_known_step_requires_a_production_event(baseline, step, kind, body):
    from v1_07g_consumption_support import event

    data = copy.deepcopy(baseline)
    run = first_run(data)
    at = datetime.fromisoformat(run["created_at"]) + timedelta(milliseconds=500)
    run["events"].append(event(run, step, body, at, kind=kind))
    sort_events(run)
    rejected(data, "trace_event_contract")
