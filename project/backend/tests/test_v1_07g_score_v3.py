"""Whole-field and identity counterexamples for the versioned G repair.

The fixtures use authored DEV materials and an independently specified ledger. No
network or formal blind data is used. Mutations assert exact metrics and named failures.
"""

import copy
import json
from datetime import datetime, timedelta

import pytest

from procurement_review.gold_v3_contracts import payload_sha256
from procurement_review.gold_v38_dev_quality import dev_quality_cases
from procurement_review.precise_candidate import (
    truth_extension_v3,
    request_identities_v3,
)
from procurement_review.precise_dev_paired_metering import completeness_report_v3
from procurement_review.precise_dev_score import score_sweep
from procurement_review.trace import canonical_checksum
from test_v1_07g_dev_quality import _extended_registration, _fixture_observations


def fixture_v3():
    registered = _extended_registration()
    observations = _fixture_observations(registered)
    registered["extends"]["extension_version"] = "v1-07-truth-extension.v2"
    registered["truth_extension"] = {
        case.case_id: truth_extension_v3(case) for case in dev_quality_cases()
    }
    registered["binding"]["expected_requests"] = {
        case.case_id: request_identities_v3(case) for case in dev_quality_cases()
    }
    from procurement_review.precise_candidate import expected_snapshot_fields_v3

    registered["binding"]["expected_snapshot_fields"] = expected_snapshot_fields_v3()
    runs = {}
    for row in observations["observations"]:
        extension = registered["truth_extension"][row["case_id"]]
        for span in row["spans"]:
            span["text"] = None
            descriptor = extension["support"][span["check_id"]][
                span["worker"] + "." + span["field"]
            ]
            span["value"] = descriptor["value"]
            span["field_path"] = (
                f"{span['worker']}.{span['check_id'].lower().replace('-', '_')}.{span['field']}"
            )
            text = registered["material_text"][row["case_id"]][span["worker"]]
            spans = [
                (text.index(clause), text.index(clause) + len(clause))
                for clause in descriptor["clauses"]
            ]
            span["start"], span["end"] = (
                min(a for a, _ in spans),
                max(b for _, b in spans),
            )
        for finding in row["findings"]:
            finding["evidence"] = [
                {
                    "locator": evidence["locator"],
                    **{
                        key: extension["facts"][evidence["locator"]][key]
                        for key in ("source_id", "version")
                    },
                }
                for evidence in finding["evidence"]
            ]
        row["metering"].update(input_tokens_known=200, output_tokens_known=80)
        for pair in row.get("conflicts", []):
            for name in ("left", "right"):
                side = pair[name]
                side["proposition"].update(
                    scope_id=extension["scope_id"], subject_type="purchase_order"
                )
                for evidence in side["evidence"]:
                    evidence["content_sha256"] = extension["facts"][
                        evidence["locator"]
                    ]["content_sha256"]
        snapshot = {
            key.removeprefix("expected_"): value
            for key, value in registered["binding"].items()
            if key not in {"branches", "expected_requests", "expected_snapshot_fields"}
        }
        snapshot.update(registered["binding"]["expected_snapshot_fields"])
        snapshot.update(
            implementation_sha256=registered["binding"][
                "product_implementation_sha256"
            ],
            mode="real",
            generation_deadline_seconds=90.0,
        )
        fact_fields = {
            "purchase_order": (
                "currency",
                "invoice_currency",
                "amount_minor",
                "invoice_amount_minor",
                "ordered_on",
            ),
            "supplier": ("registry_status", "procurement_status"),
            "transaction": ("currency", "amount_minor", "booked_on"),
        }
        facts = []
        for kind, fields in fact_fields.items():
            for field in fields:
                path = f"{kind}.{field}"
                locator = path.replace("transaction.", "historical_transaction.")
                fact = extension["facts"][locator]
                duty = (
                    "O-07"
                    if field in {"ordered_on", "booked_on"}
                    else "O-03"
                    if kind == "supplier"
                    else "O-06"
                )
                facts.append(
                    {
                        "proposition": {
                            "check_id": duty,
                            "field_path": path,
                            "scope_id": extension["scope_id"],
                            "source_version": fact["version"],
                            **{
                                key: fact[key]
                                for key in (
                                    "subject_id",
                                    "subject_type",
                                    "effective_at",
                                )
                            },
                        },
                        "value": fact["value"],
                        "evidence": [
                            {
                                "locator": locator,
                                **{
                                    key: fact[key]
                                    for key in (
                                        "source_id",
                                        "version",
                                        "content_sha256",
                                    )
                                },
                            }
                        ],
                    }
                )
        execution = {
            "run_id": row["run_id"],
            "case_id": row["case_id"],
            "decision": {"observations": facts},
        }
        attempts, events, checkpoints = [], [], []
        for worker in ("policy", "contract"):
            branch = row["branches"][worker]
            branch["failure_code"] = None
            branch["attempts"][0].update(
                input_tokens=100, output_tokens=40, latency_ms=1000, usage_unknown=False
            )
            native_spans = [
                {
                    "proposition": {
                        "check_id": span["check_id"],
                        "field_path": span["field_path"],
                        "effective_at": extension["as_of"],
                        "scope_id": extension["scope_id"],
                        "subject_type": "purchase_order",
                        "subject_id": span["subject_id"],
                        "source_version": span.get("source_version", span["version"]),
                    },
                    "value": span["value"],
                    "start": span["start"],
                    "end": span["end"],
                    "evidence": extension["branch_sources"][worker],
                }
                for span in row["spans"]
                if span["worker"] == worker
            ]
            native = {
                "worker": worker,
                "request_sha256": registered["binding"]["expected_requests"][
                    row["case_id"]
                ][worker],
                "spans": native_spans,
                "attempts": copy.deepcopy(branch["attempts"]),
                "outcome": branch.get("outcome"),
                "support_failures": branch.get("support_failures"),
                "failure_code": branch.get("failure_code"),
            }
            execution[worker] = native
            attempts.append(
                {
                    "run_id": row["run_id"],
                    "generation": 1,
                    "worker": worker,
                    "attempt_no": 1,
                    "outcome": "completed",
                    "abandoned": False,
                    "discarded_late": False,
                    "started_at": row["service_created_at"],
                    "finished_at": row["observed_at"],
                }
            )
            receipt = {
                "worker": worker,
                "attempt_no": 1,
                "outcome": "completed",
                "usage_unknown": False,
                "input_usage": 100,
                "output_usage": 40,
                "request_sha256": native["request_sha256"],
                "receipt_model": snapshot["resolved_model"],
            }
            events.append(
                {
                    "event_ordinal": len(events) + 1,
                    "generation": 1,
                    "worker": worker,
                    "attempt": 1,
                    "step": "model_attempt",
                    "event_type": "step_completed",
                    "occurred_at": row["observed_at"],
                    "output_summary": receipt,
                    "output_checksum": canonical_checksum(receipt),
                }
            )
            payload = {"result": native}
            checkpoints.append(
                {
                    "run_id": row["run_id"],
                    "worker": worker,
                    "generation": 1,
                    "attempt_no": 1,
                    "adopted_at": row["observed_at"],
                    "input_checksum": "sha256:" + native["request_sha256"],
                    "result_payload": payload,
                    "result_checksum": canonical_checksum(payload),
                }
            )
        events.append(
            {
                "event_ordinal": 3,
                "step": "review_workflow",
                "event_type": "step_completed",
                "occurred_at": row["observed_at"],
                "output_summary": {"status": row["status"]},
            }
        )
        summary = {
            "run_id": row["run_id"],
            "case_id": row["case_id"],
            "schema_version": row.get("summary_schema"),
            "rules_version": row.get("rules_version"),
        }
        execution.update(
            summary=summary,
            metering=copy.deepcopy(row["metering"]),
            conflicts=copy.deepcopy(row.get("conflicts", [])),
            security=copy.deepcopy(row.get("security", [])),
            execution_version=row.get("execution_version"),
            stop_reason=row.get("stop_reason"),
        )
        execution["decision"].update(
            findings=copy.deepcopy(row["findings"]),
            recommendation=row.get("recommendation"),
        )
        from procurement_review.precise_dev_sweep import _summarise

        projection = _summarise(
            {
                "execution": execution,
                "summary": summary,
                "status": row["status"],
                "execution_mode": snapshot["mode"],
                "error_code": row.get("error_code"),
            }
        )
        runs[row["run_id"]] = {
            "case_id": row["case_id"],
            "status": row["status"],
            "created_at": row["service_created_at"],
            "snapshot": snapshot,
            "projection": projection,
            "execution": execution,
            "attempts": attempts,
            "events": events,
            "summary": {
                "summary_payload": summary,
                "summary_checksum": canonical_checksum(summary),
                "created_at": row["observed_at"],
            },
            "checkpoints": checkpoints,
            "generations": [
                {
                    "generation": 1,
                    "status": "completed",
                    "created_at": row["service_created_at"],
                    "deadline_at": (
                        datetime.fromisoformat(row["service_created_at"])
                        + timedelta(seconds=90)
                    ).isoformat(),
                }
            ],
        }
    bindings = {
        "binding_version": "v1-07g-db-binding.v3",
        "database": "independent-fixture",
        "base_registration_sha256": registered["extends"]["base_registration_sha256"],
        "observations_payload_sha256": payload_sha256(observations),
        "runs": runs,
    }
    return registered, observations, bindings


@pytest.fixture(scope="module")
def baseline():
    return fixture_v3()


def test_normal_v3_and_explicit_historical_v2(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    result = score_sweep(registered, observed, bindings)
    assert result["stage_acceptance"], result["failed_gates"]
    assert result["score_version"] == "v1-07g-dev-score.v3"
    assert (
        result["repeats"][1]["q02_obligations"]["supported_subset_denominator"] == 129
    )
    assert result["repeats"][1]["q01_first_answer"]["denominator"] == 24
    from test_v1_07g_dev_quality import _derived_bindings

    old = _extended_registration()
    rows = _fixture_observations(old)
    assert (
        score_sweep(old, rows, _derived_bindings(old, rows))["semantics_version"]
        == "v1-07g-score-semantics.v2"
    )


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("responsibility", lambda _: "卖方无需承担维修责任，由买方自费维修故障设备"),
        ("subject", lambda value: value + "及十艘航空母舰"),
        ("responsibility", lambda value: "如买方支付费用，" + value),
        ("damages", lambda value: value.replace("每日", "每年")),
        ("special", lambda value: value.replace("<=", ">=", 1)),
        ("cap", lambda value: value.replace("CNY", "USD")),
    ],
)
def test_keywords_do_not_prove_full_semantics(baseline, field, mutation):
    registered, observed, bindings = copy.deepcopy(baseline)
    for row in observed["observations"]:
        for span in row["spans"]:
            if span["field"] == field:
                span["value"] = mutation(span["value"])
    result = score_sweep(registered, observed, bindings)
    assert result["repeats"][1]["q02_obligations"]["numerator"] == 132
    assert result["repeats"][1]["q07_citations"]["unsupported_propositions"] == 12
    assert result["repeats"][1]["q01_first_answer"]["final_effective_branches"] == 12
    assert result["repeats"][1]["q01_first_answer"]["final_effective_value"] == 0.5
    assert "repeat-1/Q-02" in result["failed_gates"]
    assert "repeat-1/Q-07-semantic-support" in result["failed_gates"]
    assert any(
        "semantic_value_mismatch" in code
        for duty in result["repeats"][1]["obligations"]
        for code in duty.get("claim_failures", [])
    )


@pytest.mark.parametrize(
    "field,alias",
    [
        ("responsibility", "供货方自担费用修理损坏设备。"),
        ("standard", "所有设备通过约定检测。"),
        ("special", "验收≤7日；质保≥90天；交付≤15日"),
        ("cap", "84,000.00 人民币"),
        ("applies_to", '["服务","采购"]'),
    ],
)
def test_whole_semantic_aliases_remain_supported(baseline, field, alias):
    registered, observed, bindings = copy.deepcopy(baseline)
    span = next(s for s in observed["observations"][0]["spans"] if s["field"] == field)
    span["value"] = alias
    result = score_sweep(registered, observed, bindings)
    assert result["repeats"][1]["q02_obligations"]["numerator"] == 144
    assert result["repeats"][1]["q07_citations"]["unsupported_propositions"] == 0


def test_required_subject_and_missing_supported_keep_fixed_denominators(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    for row in observed["observations"]:
        row["spans"] = [span for span in row["spans"] if span["field"] != "subject"]
        row["findings"] = [
            duty for duty in row["findings"] if duty["obligation_id"] != "O-02"
        ]
    result = score_sweep(registered, observed, bindings)
    q02 = result["repeats"][1]["q02_obligations"]
    assert (q02["numerator"], q02["denominator"]) == (120, 144)
    assert (q02["supported_subset_numerator"], q02["supported_subset_denominator"]) == (
        105,
        129,
    )
    assert "repeat-1/Q-02-supported-subset" in result["failed_gates"]
    assert any(
        duty.get("missing_claims") == ["contract.subject"]
        for duty in result["repeats"][1]["obligations"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "branch",
        "tokens",
        "processing",
        "duplicate",
        "worker",
        "generation",
        "request",
        "unknown",
        "receipt",
    ],
)
def test_planned_branches_and_attempt_identities_are_closed(baseline, mutation):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    run = bindings["runs"][row["run_id"]]
    if mutation == "branch":
        for item in observed["observations"]:
            del item["branches"]["contract"]
    elif mutation == "tokens":
        del row["metering"]["input_tokens_known"]
        del row["branches"]["policy"]["attempts"][0]["input_tokens"]
    elif mutation == "processing":
        row["status"] = "processing"
    elif mutation == "duplicate":
        run["attempts"][1] = copy.deepcopy(run["attempts"][0])
    elif mutation in {"worker", "generation"}:
        run["attempts"][0][mutation] = "wrong" if mutation == "worker" else 99
    elif mutation == "request":
        run["events"][0]["output_summary"]["request_sha256"] = "0" * 64
    elif mutation == "unknown":
        row["metering"].update(unknown_usage_attempts=1, usage_complete=False)
    else:
        run["events"][0]["worker"] = "contract"
    result = score_sweep(registered, observed, bindings)
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert any(
        code.startswith(
            {
                "branch": "planned_branch_missing",
                "tokens": "ledger_mismatch:input_tokens_known",
                "processing": "not_terminal",
                "duplicate": "duplicate_attempt",
                "worker": "attempt_identity",
                "generation": "attempt_identity",
                "request": "request_branch_mismatch",
                "unknown": "ledger_mismatch:unknown_usage_attempts",
                "receipt": "model_receipt_identity",
            }[mutation]
        )
        for code in result["completeness"]["failures"]
    )
    if mutation == "branch":
        assert result["repeats"][1]["q01_first_answer"]["numerator"] == 12
        assert result["repeats"][1]["q01_first_answer"]["denominator"] == 24
        assert "repeat-1/Q-01" in result["failed_gates"]


@pytest.mark.parametrize("timing", ["slow", "missing", "reverse", "invalid"])
def test_each_repeat_has_its_own_complete_latency_gate(baseline, timing):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    if timing == "slow":
        row["observed_at"] = (
            datetime.fromisoformat(row["service_created_at"]) + timedelta(seconds=100)
        ).isoformat()
    elif timing == "missing":
        row["service_created_at"] = None
    elif timing == "reverse":
        row["observed_at"] = "2026-10-07T02:59:59+00:00"
    else:
        row["observed_at"] = "not-a-date"
    result = score_sweep(registered, observed, bindings)
    assert result["repeats"][1]["latency"]["planned_samples"] == 12
    assert result["repeats"][1]["latency"]["p95_ms"] == (
        100000 if timing == "slow" else None
    )
    assert result["repeats"][2]["latency"]["p95_ms"] == 10000
    assert "repeat-1/Q-09-p95" in result["failed_gates"]
    assert "repeat-2/Q-09-p95" not in result["failed_gates"]


def test_post_observation_human_chain_is_legal_and_unknown_usage_is_honest(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    run = bindings["runs"][row["run_id"]]
    run["events"].append(
        {
            "event_ordinal": 4,
            "step": "human_gate",
            "event_type": "confirmation_recorded",
            "occurred_at": "2026-10-07T04:00:00+00:00",
            "output_summary": {"action": "confirm", "status": "confirmed"},
        }
    )
    run["status"] = "confirmed"
    assert completeness_report_v3(registered, observed, bindings)["complete"]
    receipt = run["events"][0]["output_summary"]
    receipt.update(usage_unknown=True, input_usage=None, output_usage=None)
    run["events"][0]["output_checksum"] = canonical_checksum(receipt)
    row["metering"].update(
        input_tokens_known=100,
        output_tokens_known=40,
        unknown_usage_attempts=1,
        usage_complete=False,
    )
    for branch in (row["branches"]["policy"], run["execution"]["policy"]):
        branch["attempts"][0].update(
            input_tokens=None, output_tokens=None, usage_unknown=True
        )
    run["checkpoints"][0]["result_payload"]["result"]["attempts"] = copy.deepcopy(
        run["execution"]["policy"]["attempts"]
    )
    run["checkpoints"][0]["result_checksum"] = canonical_checksum(
        run["checkpoints"][0]["result_payload"]
    )
    run["projection"]["branches"] = copy.deepcopy(row["branches"])
    run["projection"]["metering"] = copy.deepcopy(row["metering"])
    run["execution"]["metering"] = copy.deepcopy(row["metering"])
    bindings["observations_payload_sha256"] = payload_sha256(observed)
    assert completeness_report_v3(registered, observed, bindings)["complete"]


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("attempt_parent", "attempt_parent_run"),
        ("checkpoint_parent", "checkpoint_parent_run"),
        ("summary", "summary_missing"),
        ("processing_generation", "generation_not_terminal"),
        ("late_adoption", "adopted_attempt_ineligible"),
        ("reverse_attempt", "attempt_time_invalid"),
        ("facts", "planned_fact_identity"),
        ("raw_facts", "fact_projection_mismatch"),
    ],
)
def test_independent_audit_binding_counterexamples(baseline, mutation, code):
    registered, observed, bindings = copy.deepcopy(baseline)
    row = observed["observations"][0]
    run = bindings["runs"][row["run_id"]]
    if mutation == "attempt_parent":
        run["attempts"][0]["run_id"] = "run-other-case"
    elif mutation == "checkpoint_parent":
        run["checkpoints"][0]["run_id"] = "run-other-case"
    elif mutation == "summary":
        del run["summary"]
    elif mutation == "processing_generation":
        run["generations"][0]["status"] = "processing"
    elif mutation == "late_adoption":
        run["attempts"][0]["discarded_late"] = True
    elif mutation == "reverse_attempt":
        run["attempts"][0]["started_at"] = "2026-10-08T03:00:00+00:00"
    elif mutation == "facts":
        run["execution"]["decision"]["observations"] = []
    else:
        row["fact_observations"] = []
        bindings["observations_payload_sha256"] = payload_sha256(observed)
    result = score_sweep(registered, observed, bindings)
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert any(
        item.startswith(code + ":") for item in result["completeness"]["failures"]
    )
    if mutation == "facts":
        assert result["repeats"][1]["q07_citations"]["adopted_propositions"] == 551
        assert result["repeats"][1]["q07_citations"]["unsupported_propositions"] == 10


def test_recheck_rejects_semantics_and_final_verdict_tampering(
    baseline, tmp_path, monkeypatch
):
    from procurement_review.precise_dev_score import main
    import sys

    registered, observed, bindings = copy.deepcopy(baseline)
    paths = [tmp_path / name for name in ("r.json", "o.json", "b.json", "s.json")]
    for path, value in zip(paths, (registered, observed, bindings)):
        path.write_text(json.dumps(value))
    import hashlib

    bindings["observations_file_sha256"] = hashlib.sha256(
        paths[1].read_bytes()
    ).hexdigest()
    paths[2].write_text(json.dumps(bindings))
    argv = [
        "score",
        "score",
        str(paths[0]),
        str(paths[1]),
        "--bindings",
        str(paths[2]),
        "--output",
        str(paths[3]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    stored = json.loads(paths[3].read_text())
    for field, value in (
        ("semantics_version", "wrong"),
        ("stage_acceptance", False),
        ("failed_gates", ["invented"]),
    ):
        paths[3].write_text(json.dumps({**stored, field: value}))
        argv[1] = "recheck"
        with pytest.raises(SystemExit, match="recheck mismatch"):
            main()


@pytest.mark.parametrize(
    "identity,value",
    [
        ("subject_id", "PO-WRONG"),
        ("field_path", "contract.o_09.subject"),
        ("source_id", "wrong"),
        ("version", "not-a-version"),
        ("effective_at", "2027-06-30"),
        ("end", 99999),
        ("start", -1),
    ],
)
def test_complete_proposition_identity_and_bounds_are_required(
    baseline, identity, value
):
    registered, observed, bindings = copy.deepcopy(baseline)
    for row in observed["observations"]:
        span = next(s for s in row["spans"] if s["field"] == "subject")
        span[identity] = value
    result = score_sweep(registered, observed, bindings)
    assert result["repeats"][1]["q02_obligations"]["numerator"] == 132
    assert result["repeats"][1]["q07_citations"]["unsupported_propositions"] == 12


def test_fact_locator_cannot_hide_wrong_source_or_version(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    for row in observed["observations"]:
        duty = next(item for item in row["findings"] if item["obligation_id"] == "O-03")
        duty["evidence"][0].update(source_id="PO-WRONG", version="missing-version")
    result = score_sweep(registered, observed, bindings)
    assert result["repeats"][1]["q02_obligations"]["numerator"] == 132
    assert any(
        "fact_identity_mismatch" in code
        for duty in result["repeats"][1]["obligations"]
        for code in duty.get("evidence_failures", [])
    )


def test_token_enum_needs_its_own_supporting_clause(baseline):
    registered, observed, bindings = copy.deepcopy(baseline)
    for row in observed["observations"]:
        span = next(s for s in row["spans"] if s["field"] == "supplier")
        span["start"], span["end"] = (
            0,
            len(registered["material_text"][row["case_id"]]["policy"].splitlines()[0]),
        )
    result = score_sweep(registered, observed, bindings)
    assert result["repeats"][1]["q07_citations"]["unsupported_propositions"] == 12
    assert any(
        "excerpt_does_not_prove_field" in code
        for duty in result["repeats"][1]["obligations"]
        for code in duty.get("claim_failures", [])
    )


def comparison_v3(multi="recommend_approve", single="recommend_approve"):
    from test_v1_07g_comparison import (
        _plan,
        _pair,
        _resources,
        _expected,
        _ledger_matching,
    )

    plan = _plan()
    observed = _pair(plan, multi=multi, single=single)
    resources = _ledger_matching(plan, observed, _resources(plan))
    ledgers = {}
    for arm, arm_plan in plan.arm.items():
        ledger = []
        for task in arm_plan.task_keys():
            case, seed, _ = task.split(".")
            observation = next(
                row
                for row in observed
                if row.arm == arm
                and row.case_id == case
                and row.repeat_seed == int(seed)
            )
            branches = len(arm_plan.shape.model_branches)
            request_sha = payload_sha256(task)
            receipt = {
                "task_key": task,
                "attempt_no": 1,
                "request_sha256": request_sha,
                "usage_unknown": False,
                "input_tokens": observation.known_input_tokens // branches,
                "output_tokens": observation.known_output_tokens // branches,
            }
            ledger.append(
                {
                    "task_key": task,
                    "attempt_no": 1,
                    "request_sha256": request_sha,
                    "state": "committed",
                    "receipt": receipt,
                    "receipt_sha256": payload_sha256(receipt),
                }
            )
        ledgers[arm] = ledger
    return (
        plan,
        observed,
        {"resources": resources, "expected": _expected(plan), "arm_quality": {}},
        ledgers,
    )


@pytest.mark.parametrize(
    "multi,single",
    [
        ("recommend_approve", "recommend_reject"),
        ("recommend_approve", "recommend_approve"),
        ("recommend_reject", "recommend_approve"),
    ],
)
def test_q12_v3_win_tie_loss_have_no_direction_gate(multi, single):
    from procurement_review.eval_g_report import build_report_v6, verify_report_v6

    plan, observed, kwargs, ledgers = comparison_v3(multi, single)
    report = build_report_v6(plan, observed, attempts_by_arm=ledgers, **kwargs)
    assert report["Q-12"]["complete"], report["Q-12"]["incomplete_reasons"]
    assert report["metrics_version"] == "v1-07-eval-metrics.v3"
    assert report["Q-12"]["direction_gate"] is None
    assert report["stage_acceptance"] is False
    assert verify_report_v6(report, plan, observed, attempts_by_arm=ledgers, **kwargs)[
        "verified"
    ]
    with pytest.raises(ValueError, match="report_replay_mismatch"):
        verify_report_v6(
            {**report, "stage_acceptance": True},
            plan,
            observed,
            attempts_by_arm=ledgers,
            **kwargs,
        )


def test_q12_v3_rejects_deleted_known_token_and_same_count_task_swap():
    from procurement_review.eval_g_report import build_report_v6

    plan, observed, kwargs, ledgers = comparison_v3()
    arm = next(iter(ledgers))
    row = ledgers[arm][0]
    del row["receipt"]["output_tokens"]
    row["receipt_sha256"] = payload_sha256(row["receipt"])
    report = build_report_v6(plan, observed, attempts_by_arm=ledgers, **kwargs)
    assert any(
        "known_token_usage_missing" in reason
        for reason in report["Q-12"]["incomplete_reasons"]
    )
    ledgers[arm][0] = copy.deepcopy(ledgers[arm][1])
    report = build_report_v6(plan, observed, attempts_by_arm=ledgers, **kwargs)
    assert any(
        "planned_task_identity" in reason
        for reason in report["Q-12"]["incomplete_reasons"]
    )
    assert report["Q-12"]["complete"] is False
