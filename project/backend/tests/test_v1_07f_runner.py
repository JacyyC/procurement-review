import asyncio
import json
from datetime import timedelta

import httpx
import pytest

from procurement_review.eval_attempt_ledger import AttemptLedger, LedgerError
from procurement_review.eval_f_budget import FStageBudget
from procurement_review.eval_f_context import stable_write
from procurement_review.eval_f_registration import prepare, registration
from procurement_review.eval_f_run import execute, replay
from procurement_review.eval_f_transport import PreciseAttemptDriver, evidence_payload
from procurement_review.model_attempt_v2 import AdmissionDenied
from v1_07f_support import setup, response_for


def run(driver, request, deadline):
    return asyncio.run(
        driver.run(
            "case.one.policy",
            request,
            deadline=deadline,
            reserve_input=100,
            reserve_output=200,
        )
    )


def test_f_precise_receipt_commit_reopen_and_zero_resend(tmp_path):
    sent = []
    driver, request, deadline = setup(
        tmp_path, lambda req: sent.append(req) or response_for(req)
    )
    result = run(driver, request, deadline)
    assert not result.pipeline_failure and result.evidence_output
    again = PreciseAttemptDriver(
        AttemptLedger(driver.ledger.path), driver.transport, driver.authority
    )
    assert run(again, request, deadline) == result and len(sent) == 1
    assert driver.ledger.manifest().stage_id == "V1-07F"
    assert driver.ledger.budget_snapshot().known_output_tokens == 200


@pytest.mark.parametrize(
    "sequence",
    [
        ("format", "transport"),
        ("transport", "format"),
        ("format", "valid"),
        ("rate", "valid"),
        ("server", "valid"),
    ],
)
def test_f_extra_attempts_share_allowance(tmp_path, sequence):
    sent = []

    def handler(req):
        action = sequence[len(sent)]
        sent.append(action)
        if action == "transport":
            raise httpx.ReadError("must not escape")
        return response_for(
            req,
            body={"bad": True} if action == "format" else None,
            status=429 if action == "rate" else 503 if action == "server" else 200,
        )

    driver, request, deadline = setup(tmp_path, handler)
    result = run(driver, request, deadline)
    assert result.model_request_count == 2 and len(sent) == 2
    assert not result.first_pass_valid
    assert result.pipeline_failure == (sequence[1] != "valid")
    assert (
        "must not escape"
        not in driver.ledger.export(tmp_path / ".local/export.json").read_text()
    )


def test_f_concurrent_retries_keep_their_own_first_body_observation(tmp_path):
    async def scenario():
        other_retry_finished = asyncio.Event()
        first_retry_started = asyncio.Event()
        sends = []
        counts = {"case.a": 0, "case.b": 0}

        async def handler(req):
            sends.append(req)
            key = json.loads(json.loads(req.content)["messages"][1]["content"])[
                "question"
            ]
            counts[key] += 1
            if key == "case.a" and counts[key] == 1:
                return response_for(req, body={"bad": True})
            if key == "case.a":
                first_retry_started.set()
                await other_retry_finished.wait()
                return response_for(req)
            if counts[key] == 1:
                await first_retry_started.wait()
                raise httpx.ReadError("first body unavailable")
            other_retry_finished.set()
            return response_for(req)

        driver, request, deadline = setup(tmp_path, handler)

        async def branch(key):
            return await driver.run(
                key,
                request.model_copy(update={"question": key}),
                deadline=deadline,
                reserve_input=100,
                reserve_output=200,
            )

        a, b = await asyncio.gather(branch("case.a"), branch("case.b"))
        assert len(sends) == 4
        assert not a.pipeline_failure and not b.pipeline_failure
        assert a.first_received_nonempty_body is True
        assert b.first_received_nonempty_body is False
        assert await branch("case.a") == a and await branch("case.b") == b
        assert len(sends) == 4

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["authentication", "configuration", "drift", "unsafe"]
)
def test_f_forbidden_retry_never_sent(tmp_path, failure):
    sent = []

    def handler(req):
        sent.append(1)
        body = (
            {"forbidden": "SYNTHETIC_SECRET_TO_EXTERNAL_SINK"}
            if failure == "unsafe"
            else None
        )
        return response_for(
            req,
            body=body,
            status=401
            if failure == "authentication"
            else 400
            if failure == "configuration"
            else 200,
            model="drift-model"
            if failure == "drift"
            else "deepseek-v4-flash-ga-260731",
        )

    driver, request, deadline = setup(tmp_path, handler)
    result = run(driver, request, deadline)
    assert result.pipeline_failure and len(sent) == 1 and result.evidence_output is None


@pytest.mark.parametrize("event", ["cancel", "generation", "deadline"])
def test_f_late_response_never_adopted(tmp_path, event):
    def handler(req):
        if event == "cancel":
            stable_write(
                driver.authority.execution_path, dict(generation=1, cancelled=True)
            )
        elif event == "generation":
            stable_write(
                driver.authority.execution_path, dict(generation=2, cancelled=False)
            )
        else:
            driver.authority.now = lambda: deadline + timedelta(seconds=1)
        return response_for(req)

    driver, request, deadline = setup(tmp_path, handler)
    result = run(driver, request, deadline)
    assert (
        result.pipeline_failure and not result.claims and result.evidence_output is None
    )
    output = json.loads(
        driver.ledger.export(tmp_path / ".local/export.json").read_text()
    )
    assert output["attempts"][0]["receipt"]["error_code"] == "late_result_discarded"
    assert output["attempts"][0]["receipt"]["output_tokens"] == 200


def test_f_cancelled_extra_attempt_checks_gate(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        stable_write(
            driver.authority.execution_path, dict(generation=1, cancelled=True)
        )
        return response_for(req, body={"bad": True})

    driver, request, deadline = setup(tmp_path, handler)
    result = run(driver, request, deadline)
    assert len(calls) == 1 and result.pipeline_failure


def test_f_generation_denies_first_transport(tmp_path):
    driver, request, deadline = setup(
        tmp_path, lambda req: pytest.fail("network forbidden")
    )
    stable_write(driver.authority.execution_path, dict(generation=2, cancelled=False))
    with pytest.raises(AdmissionDenied):
        run(driver, request, deadline)
    assert driver.ledger.manifest().attempt_count == 0


def test_f_payload_excludes_case_scoring_metadata_and_different_arms_one_factor(
    tmp_path,
):
    driver, request, _ = setup(tmp_path, response_for)
    body = evidence_payload(driver.transport.config, request)
    visible = json.dumps(body["messages"], ensure_ascii=False)
    for name in (
        "expected_recommendation",
        "generation_family_id",
        "near_duplicate",
        "case_id",
        "truth",
        "label",
    ):
        assert name not in visible
    r = registration()
    b = r["candidates"]["baseline"]
    c = r["candidates"]["compact"]
    assert {
        k for k in b["configuration"] if b["configuration"][k] != c["configuration"][k]
    } == {"prompt_style"}
    assert (
        r["reserve"]["complete_requests"] == 1440
        and r["reserve"]["registered_diagnostics"] == 108
    )


def test_f_inventory_reopen_config_changes_and_deleted_ledger_fail_closed(tmp_path):
    root = tmp_path / ".local/live"
    budget = FStageBudget(root)
    for name in ("one", "two"):
        path = root / name / "ledger.sqlite3"
        ledger = AttemptLedger.create(
            path,
            run_id="v1-07-f-" + name,
            gold_manifest_sha256="b" * 64,
            configuration_sha256="a" * 64,
            max_requests=4,
            max_input_tokens=None,
            max_output_tokens=None,
            stage_id="V1-07F",
        )
        budget.add(path)
        ledger.register_task("one", "c" * 64)
        ledger.reserve_attempt(
            "one",
            request_sha256="d" * 64,
            reserve_input_tokens=100,
            reserve_output_tokens=200,
        )
    assert FStageBudget(root).summary()["reserved_attempts"] == 2
    with pytest.raises(LedgerError):
        FStageBudget(root, prior_c=0)
    (root / "one/ledger.sqlite3").rename(root / "one/moved.sqlite3")
    with pytest.raises(LedgerError):
        FStageBudget(root).require()


def test_full_minimum_offline_runs_cases_scores_and_replays(tmp_path):
    path = tmp_path / ".local/registration.json"
    prepare(path)
    root = tmp_path / ".local/live"
    sent = []
    transport = httpx.MockTransport(lambda req: sent.append(req) or response_for(req))
    result = asyncio.run(
        execute(path, "minimum", transport_override=transport, offline_root=root)
    )
    assert result["diagnostic_passed"], result["rows"]
    assert not result["stage_acceptance"] and len(sent) == 12
    directory = next(root.glob("*/minimum"))
    again = replay(directory)
    assert (
        again["rows"] == result["rows"]
        and again["combined"] == result["combined"]
        and again["network_calls"] == 0
    )
    assert (
        asyncio.run(
            execute(path, "minimum", transport_override=transport, offline_root=root)
        )["combined"]
        == result["combined"]
    )
    assert len(sent) == 12
    # Correct ID with wrong values is not independently supported.
    from procurement_review.eval_f_contracts import FCaseProjection
    from procurement_review.eval_f_score import score_batch

    p = FCaseProjection.model_validate_json(
        (
            directory
            / f"V107-D-0001.{registration()['candidates']['minimum']['repeat_seeds'][0]}.json"
        ).read_bytes()
    )
    broken = p.model_copy(
        update={
            "decision": p.decision.model_copy(
                update={
                    "spans": tuple(
                        s.model_copy(update={"value": 900})
                        if s.proposition.field_path.endswith(".warranty")
                        else s
                        for s in p.decision.spans
                    )
                }
            )
        }
    )
    report = score_batch([broken], [p.case_id], [p.repeat_seed])
    assert report["combined"]["metrics"]["Q07"]["value"] < 0.98
    assert not report["diagnostic_passed"]

    # Whole-UNKNOWN has no semantic claim denominator and cannot win by abstention.
    from procurement_review.business_v3_types import finding

    unknown_rows = tuple(
        finding(f"O-{i:02}", "unknown", "missing_source", missing=("source",))
        for i in range(1, 13)
    )
    abstain = p.model_copy(
        update={
            "decision": p.decision.model_copy(
                update={
                    "business": p.decision.business.model_copy(
                        update={
                            "findings": unknown_rows,
                            "recommendation": "insufficient_information",
                        }
                    ),
                    "spans": (),
                }
            )
        }
    )
    abstain_report = score_batch([abstain], [p.case_id], [p.repeat_seed])
    assert abstain_report["combined"]["metrics"]["Q02"]["numerator"] == 0
    # Stale observations remain claims and must be scored even after summaries abstain.
    assert abstain_report["combined"]["metrics"]["Q07"]["value"] < 0.98
    assert not abstain_report["diagnostic_passed"]
    abstain = abstain.model_copy(
        update={
            "decision": abstain.decision.model_copy(
                update={
                    "business": abstain.decision.business.model_copy(
                        update={"observations": ()}
                    )
                }
            )
        }
    )
    abstain_report = score_batch([abstain], [p.case_id], [p.repeat_seed])
    assert abstain_report["combined"]["metrics"]["Q07"]["value"] is None
    assert not abstain_report["diagnostic_passed"]
    # A high-risk wrong approval is independently counted even if formatting and IDs pass.
    high = FCaseProjection.model_validate_json(
        (directory / f"V107-D-0003.{p.repeat_seed}.json").read_bytes()
    )
    released = high.model_copy(
        update={
            "decision": high.decision.model_copy(
                update={
                    "business": high.decision.business.model_copy(
                        update={"recommendation": "recommend_approve"}
                    )
                }
            )
        }
    )
    released_report = score_batch([released], [high.case_id], [high.repeat_seed])
    assert released_report["combined"]["counts"]["dangerous_release"] == 1
    assert not released_report["diagnostic_passed"]


def test_real_run_cannot_redirect_counter_to_new_root(tmp_path):
    with pytest.raises(LedgerError, match="fixed live root"):
        asyncio.run(execute(tmp_path / "absent", "minimum", offline_root=tmp_path))


def test_f_process_restart_marks_inflight_unknown_and_never_resends(tmp_path):
    import subprocess
    import sys
    from procurement_review.gold_v3_contracts import payload_sha256

    sent = []
    driver, request, deadline = setup(
        tmp_path, lambda req: sent.append(req) or response_for(req)
    )
    identity = dict(
        request=request.model_dump(mode="json"),
        deadline=deadline.isoformat(),
        reserve_input=100,
        reserve_output=200,
        max_attempts=2,
    )
    driver.ledger.register_task("case.one.policy", payload_sha256(identity))
    attempt = driver.ledger.reserve_attempt(
        "case.one.policy",
        request_sha256="d" * 64,
        reserve_input_tokens=100,
        reserve_output_tokens=200,
    )
    driver.ledger.mark_sent(attempt.attempt_id)
    script = "from pathlib import Path; from procurement_review.eval_attempt_ledger import AttemptLedger; import sys; l=AttemptLedger(Path(sys.argv[1])); print(l.recover().sent_marked_remote_unknown)"
    result = subprocess.run(
        [sys.executable, "-c", script, str(driver.ledger.path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "1"
    reopened = PreciseAttemptDriver(
        AttemptLedger(driver.ledger.path), driver.transport, driver.authority
    )
    with pytest.raises(LedgerError, match="unknown"):
        run(reopened, request, deadline)
    assert not sent and reopened.ledger.manifest().attempt_count == 1
    assert reopened.ledger.budget_snapshot().unknown_usage_attempts == 1


def test_f_configuration_change_invalidates_recovery(tmp_path):
    driver, request, deadline = setup(tmp_path, response_for)
    run(driver, request, deadline)
    driver.transport.config = driver.transport.config.model_copy(
        update={"prompt_style": "compact"}
    )
    with pytest.raises(LedgerError, match="configuration"):
        run(driver, request, deadline)


def test_f_precise_receipt_and_sidecar_rollback_together(tmp_path):
    from procurement_review.eval_attempt_ledger import InjectedCrash
    from procurement_review.eval_v2_contracts import AttemptReceiptV1
    from procurement_review.gold_v3_contracts import payload_sha256

    driver, request, deadline = setup(tmp_path, response_for)
    projection = run(driver, request, deadline)
    driver.ledger.register_task("rollback", "b" * 64)
    attempt = driver.ledger.reserve_attempt(
        "rollback",
        request_sha256=payload_sha256({"x": 1}),
        reserve_input_tokens=100,
        reserve_output_tokens=200,
    )
    driver.ledger.mark_sent(attempt.attempt_id)
    receipt = AttemptReceiptV1(
        task_key="rollback",
        attempt_no=1,
        outcome="succeeded",
        request_sha256=attempt.request_sha256,
        input_tokens=100,
        output_tokens=200,
        usage_unknown=False,
    )

    def crash(point):
        if point == "after_projection_write":
            raise InjectedCrash()

    with pytest.raises(InjectedCrash):
        driver.ledger.commit_result(
            attempt.attempt_id, receipt=receipt, projection=projection, fault=crash
        )
    assert driver.ledger.next_action("rollback") == "recover_sent_attempt"
    assert driver.ledger.budget_snapshot().known_output_tokens == 200
    export = json.loads(
        driver.ledger.export(tmp_path / ".local/export.json").read_text()
    )
    assert (
        export["attempts"][-1]["receipt"] is None
        and export["attempts"][-1]["projection"] is None
    )


def test_f_witness_schema_error_retains_only_fixed_field_diagnostic(tmp_path):
    from v1_07f_support import body_for_payload

    def handler(req):
        body = body_for_payload(json.loads(req.content))
        body["witnesses"][0]["value"] = ["untrusted-diagnostic-value"]
        return response_for(req, body=body)

    driver, request, deadline = setup(tmp_path, handler)
    result = run(driver, request, deadline)
    exported = driver.ledger.export(tmp_path / ".local/export.json").read_text()
    assert result.pipeline_failure and "untrusted-diagnostic-value" not in exported
    receipts = [a["receipt"] for a in json.loads(exported)["attempts"]]
    assert all(r["format_fields"] == ["value"] for r in receipts)


def test_f_missing_typed_witness_identifies_the_fixed_field(tmp_path):
    from v1_07f_support import body_for_payload

    def handler(req):
        body = body_for_payload(json.loads(req.content))
        body["witnesses"] = [w for w in body["witnesses"] if w["field"] != "exempt"]
        return response_for(req, body=body)

    driver, request, deadline = setup(tmp_path, handler)
    result = run(driver, request, deadline)
    exported = json.loads(
        driver.ledger.export(tmp_path / ".local/export.json").read_text()
    )
    assert result.pipeline_failure and result.model_request_count == 1
    assert (
        exported["attempts"][0]["receipt"]["error_code"]
        == "citation_value_mismatch_o_01_exempt"
    )


@pytest.mark.parametrize("late", [False, True])
def test_f_rejected_field_audit_is_not_adopted_and_late_diagnostics_are_dropped(
    tmp_path, late
):
    from v1_07f_support import body_for_payload

    def handler(req):
        body = body_for_payload(json.loads(req.content))
        finding = next(f for f in body["findings"] if f["obligation_id"] == "O-02")
        values = json.loads(finding["normalized_value"])
        values["delivery"] = 900
        finding["normalized_value"] = json.dumps(
            values, ensure_ascii=False, separators=(",", ":")
        )
        next(w for w in body["witnesses"] if w["field"] == "delivery")["value"] = 900
        if late:
            driver.authority.cancel()
        return response_for(req, body=body)

    driver, request, deadline = setup(tmp_path, handler)
    result = run(driver, request, deadline)
    assert (
        result.pipeline_failure
        and result.claims == ()
        and result.evidence_output is None
    )
    if late:
        assert result.rejected_witnesses == ()
    else:
        assert len(result.rejected_witnesses) == 1
        assert (
            result.rejected_witnesses[0].field == "delivery"
            and result.rejected_witnesses[0].value == 900
        )


def test_f_position_options_bind_unicode_offsets_to_visible_text():
    from procurement_review.eval_f_transport import source_spans

    text = "买方：甲方。\n金额：100.10 CNY。\n验收：全部完好。"
    options = source_spans(text)
    assert options[0] == dict(start=0, end=len(text), text=text)
    for option in options:
        assert text[option["start"] : option["end"]] == option["text"]
    assert any(o["text"] == "金额：100.10 CNY。" for o in options)


def test_independent_conflict_score_rejects_correct_values_with_wrong_field_evidence(
    tmp_path,
):
    from procurement_review.eval_f_contracts import FCaseProjection
    from procurement_review.eval_f_context import load_cases, DATA
    from procurement_review.eval_f_score import score_case
    from procurement_review.gold_v31_contracts import TruthFileV11

    path = tmp_path / ".local/registration.json"
    prepare(path)
    root = tmp_path / ".local/live"
    asyncio.run(
        execute(
            path,
            "minimum",
            transport_override=httpx.MockTransport(response_for),
            offline_root=root,
        )
    )
    directory = next(root.glob("*/minimum"))
    seed = registration()["candidates"]["minimum"]["repeat_seeds"][0]
    p = FCaseProjection.model_validate_json(
        (directory / f"V107-D-0005.{seed}.json").read_bytes()
    )
    pair = p.decision.conflicts[0]
    unrelated = next(
        e
        for f in p.decision.business.findings
        for e in f.evidence
        if e.locator == "supplier.registry_status"
    )
    forged = pair.model_copy(
        update={"left": pair.left.model_copy(update={"evidence": (unrelated,)})}
    )
    altered = p.model_copy(
        update={"decision": p.decision.model_copy(update={"conflicts": (forged,)})}
    )
    case = next(c for c in load_cases() if c.case_id == p.case_id)
    truth = next(
        t
        for t in TruthFileV11.model_validate_json(
            (DATA / "truth.json").read_bytes()
        ).truths
        if t.case_id == p.case_id
    )
    assert score_case(case, truth, altered)["counts"]["matched_conflicts"] == 0
