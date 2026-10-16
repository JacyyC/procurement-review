import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx

from procurement_review.chat_transport_v2 import decode_response, request_payload
from procurement_review.model_probe_v2 import registration
from procurement_review.model_probe_run import execute
from test_v1_07c_authorized_run import mock_handler
from test_v1_07c_attempts import export, run
from v1_07c_support import branch, response, setup_driver


def test_final_text_registration_keeps_all_dev_branches_and_historical_consumption():
    plan = registration(
        no_balance_limit_authorized=True,
        extended_deadline=True,
        non_reasoning_modes=True,
        final_text=True,
    )
    assert plan["probe_revision"] == 5
    assert plan["physical_request_limit"] == 42
    assert 167 + plan["physical_request_limit"] <= 240
    assert [len(c["tasks"]) for c in plan["candidates"]] == [3, 18]
    assert plan["candidates"][1]["dependency"] == "minimum:text"
    assert all(c["configuration"]["thinking"] == "disabled" for c in plan["candidates"])


def test_last_schema_candidate_keeps_full_screen_and_fixed_sampling():
    plan = registration(
        no_balance_limit_authorized=True,
        extended_deadline=True,
        non_reasoning_modes=True,
        schema_zero=True,
    )
    assert plan["probe_revision"] == 6
    assert plan["physical_request_limit"] == 42
    assert 190 + plan["physical_request_limit"] <= 240
    assert [len(c["tasks"]) for c in plan["candidates"]] == [3, 18]
    assert all(c["configuration"]["temperature"] == 0 for c in plan["candidates"])
    assert all(c["configuration"]["mode"] == "json_schema" for c in plan["candidates"])


def test_empty_source_revision_preserves_full_screen_without_extra_attempts():
    plan = registration(
        no_balance_limit_authorized=True,
        extended_deadline=True,
        non_reasoning_modes=True,
        schema_zero=True,
        empty_source_revision=True,
    )
    minimum, screen = plan["candidates"]
    assert plan["physical_request_limit"] == 20 and 213 + 20 <= 240
    assert len(screen["tasks"]) == 18 and screen["max_attempts_per_branch"] == 1
    assert minimum["tasks"][0]["input_variant"] == "missing_authorized_document"
    assert minimum["max_attempts_per_branch"] == 2


def test_non_scalar_finish_reason_is_a_controlled_envelope_error(tmp_path):
    driver, _ = setup_driver(tmp_path, lambda req: response())
    for finish in ([], {}, 7, None):
        result = decode_response(
            response(finish=finish), branch(), driver.transport.config, ""
        )
        assert result.error_code == "response_envelope_invalid"
        assert not result.correctable and not result.findings


def test_revision_two_is_bounded_and_thinking_is_explicit():
    plan = registration(no_balance_limit_authorized=True, extended_deadline=True)
    assert plan["physical_request_limit"] == 168
    assert 32 + plan["physical_request_limit"] <= 240
    assert plan["logical_deadline_seconds"] == 120
    assert all(
        c["configuration"]["single_attempt_seconds"] == 90 for c in plan["candidates"]
    )
    configs = {c["candidate"]: c["configuration"] for c in plan["candidates"]}
    assert configs["minimum:json_schema"]["thinking"] is None
    assert configs["minimum:thinking_disabled"]["thinking"] == "disabled"


def test_reasoning_body_never_enters_receipt_or_projection(tmp_path):
    raw = response().json()
    raw["choices"][0]["message"]["reasoning_content"] = (
        "private reasoning stays transient"
    )
    raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": 17}
    driver, deadline = setup_driver(tmp_path, lambda req: httpx.Response(200, json=raw))
    run(driver, deadline)
    document = export(driver, tmp_path)
    receipt = document["attempts"][0]["receipt"]
    assert receipt["reasoning_tokens"] == 17
    assert receipt["reasoning_bytes"] == len("private reasoning stays transient")
    assert "private reasoning" not in json.dumps(document)
    config = driver.transport.config.model_copy(update={"thinking": "disabled"})
    assert request_payload(config, branch())["thinking"] == {"type": "disabled"}
    assert "thinking" not in request_payload(driver.transport.config, branch())
    assert (
        decode_response(
            httpx.Response(200, json=raw), branch(), config, ""
        ).reasoning_tokens
        == 17
    )


def test_late_timeout_retains_both_deadline_and_transport_failure(tmp_path):
    current = [datetime.now(UTC)]

    def handler(req):
        current[0] += timedelta(seconds=100)
        raise httpx.ReadTimeout("untrusted transport message")

    driver, deadline = setup_driver(tmp_path, handler, now=lambda: current[0])
    assert run(driver, deadline).pipeline_failure
    receipt = export(driver, tmp_path)["attempts"][0]["receipt"]
    assert receipt["error_code"] == "late_result_discarded"
    assert receipt["underlying_error_code"] == "timeout"
    assert receipt["usage_unknown"]


def test_capability_support_does_not_relax_screening_threshold(tmp_path):
    path = tmp_path / "registration.json"
    path.write_text(
        json.dumps(
            registration(no_balance_limit_authorized=True, extended_deadline=True)
        )
    )
    root = tmp_path / ".local/live"
    calls = []
    valid = mock_handler(calls)
    failures = []

    def handler(req):
        if not failures:
            failures.append(True)
            raise httpx.ReadTimeout("usage unknown")
        return valid(req)

    result = asyncio.run(
        execute(
            path,
            "minimum:json_schema",
            root=root,
            transport_override=httpx.MockTransport(handler),
        )
    )
    assert result["capability_supported"] and not result["screen_passed"]
    assert result["first_valid"] == 2 and result["actual_attempts"] == 4
    assert result["replay_network_calls"] == 0
    result2 = asyncio.run(
        execute(
            path,
            "minimum:thinking_disabled",
            root=root,
            transport_override=httpx.MockTransport(valid),
        )
    )
    assert result2["capability_supported"] and result2["screen_passed"]


def test_format_diagnostics_are_fixed_codes_without_original_text(tmp_path):
    from procurement_review.chat_transport_v2 import format_diagnostic

    assert format_diagnostic("```json\n{}\n```") == "markdown_fence"
    assert format_diagnostic('{"x":1,"x":2}') == "duplicate_key"
    assert format_diagnostic('{"x":NaN}') == "invalid_number"
    assert format_diagnostic("{broken") == "json_syntax"
    assert format_diagnostic("some untrusted explanation") == "non_json_text"
    raw = response().json()
    raw["choices"][0]["message"]["content"] = "```json\n{}\n```"
    driver, deadline = setup_driver(tmp_path, lambda req: httpx.Response(200, json=raw))
    assert run(driver, deadline).pipeline_failure
    document = export(driver, tmp_path)
    assert len(document["attempts"]) == 2
    assert all(
        a["receipt"]["format_diagnostic"] == "markdown_fence"
        for a in document["attempts"]
    )
    assert "```" not in json.dumps(document)


def test_revision_three_same_modes_and_small_parameter_probes():
    plan = registration(
        no_balance_limit_authorized=True,
        extended_deadline=True,
        non_reasoning_modes=True,
    )
    assert plan["physical_request_limit"] == 144
    assert 41 + plan["physical_request_limit"] <= 240
    assert all(c["configuration"]["thinking"] == "disabled" for c in plan["candidates"])
    assert {c["configuration"]["mode"] for c in plan["candidates"]} == {
        "json_schema",
        "json_object",
        "text",
    }


def test_semantic_review_retains_bounded_conclusions_not_only_hashes(tmp_path):
    from v1_07c_support import answer

    body = answer()
    driver, deadline = setup_driver(tmp_path, lambda req: response(body))
    result = run(driver, deadline)
    assert result.details[0].statement == body["findings"][0]["statement"]
    assert (
        result.details[0].recommended_action
        == body["findings"][0]["recommended_action"]
    )
    assert len(result.details[0].statement) <= 256
    assert "choices" not in export(driver, tmp_path)


def test_error_fields_are_safe_and_survive_correction_receipts(tmp_path):
    from v1_07c_support import answer

    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        bad = answer()
        bad["findings"][0]["risk_level"] = "unknown"
        return response(bad) if len(calls) == 1 else response()

    driver, deadline = setup_driver(tmp_path, handler)
    assert run(driver, deadline).corrected_valid
    receipt = export(driver, tmp_path)["attempts"][0]["receipt"]
    assert receipt["format_fields"] == ["risk_level"]
    assert "固定错误字段=risk_level" in calls[1]["messages"][0]["content"]
    assert "test-credential-not-real" not in calls[1]["messages"][0]["content"]


def test_unhashable_status_remains_controlled_schema_error():
    from v1_07c_support import answer
    from procurement_review.model_wire_v2 import parse_findings, WireIssue, messages
    import pytest

    bad = answer()
    bad["findings"][0]["status"] = {"untrusted": "payload"}
    with pytest.raises(WireIssue, match="invalid_schema"):
        parse_findings(json.dumps(bad), branch())
    with pytest.raises(WireIssue, match="invalid_correction_reason"):
        messages(branch(), "invalid_schema", ("forged instruction",))


def test_focused_registration_fits_current_stage_remainder():
    plan = registration(
        no_balance_limit_authorized=True,
        extended_deadline=True,
        non_reasoning_modes=True,
        focused=True,
    )
    assert plan["physical_request_limit"] == 102
    assert 114 + plan["physical_request_limit"] <= 240
    assert {c["configuration"]["mode"] for c in plan["candidates"]} == {
        "json_schema",
        "text",
    }
