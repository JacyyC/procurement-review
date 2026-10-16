from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from procurement_review import smoke_run
from procurement_review.chat_adapter import ChatAdapterConfig, ChatCallRecord
from procurement_review.contracts import WorkerKind
from procurement_review.gold_set import GoldSplit, load_gold_cases
from procurement_review.run_guard import REDACTED_MARKER
from procurement_review.smoke_run import (
    DRY_RUN_CREDENTIAL,
    SMOKE_CASE_COUNT,
    SMOKE_FAMILY_COUNT,
    ManifestRetention,
    RunManifest,
    RunManifestError,
    SmokeCallEntry,
    finalize_run_manifest,
    guard_run_manifest,
    model_request_for,
    select_frozen_batch,
    write_run_manifest,
)


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture(scope="module")
def dry_run_manifest() -> RunManifest:
    batch = select_frozen_batch(load_gold_cases().cases)
    return asyncio.run(
        smoke_run.run_frozen_smoke(
            execution_mode="dry_run", batch=batch, sleep=_no_sleep
        )
    )


def test_frozen_batch_is_dev_only_and_covers_every_family() -> None:
    cases = select_frozen_batch(load_gold_cases().cases)
    assert len(cases) == SMOKE_CASE_COUNT
    assert {case.split for case in cases} == {GoldSplit.DEV}
    families = [case.case_family_id for case in cases]
    assert len(set(families)) == SMOKE_FAMILY_COUNT
    assert all(families.count(family) == 4 for family in set(families))
    assert select_frozen_batch(load_gold_cases().cases) == cases
    assert not any(case.split is not GoldSplit.DEV for case in cases)


def test_frozen_batch_rejects_any_other_shape() -> None:
    cases = select_frozen_batch(load_gold_cases().cases)
    with pytest.raises(RunManifestError, match="batch_family_size"):
        select_frozen_batch(cases[1:])
    single_family = tuple(
        case for case in cases if case.case_family_id == cases[0].case_family_id
    )
    with pytest.raises(RunManifestError, match="batch_family_count"):
        select_frozen_batch(single_family)
    six_families = tuple(
        case
        for case in cases
        if case.case_family_id in sorted({c.case_family_id for c in cases})[:6]
    )
    with pytest.raises(RunManifestError, match="batch_family_count"):
        select_frozen_batch(six_families)


def test_request_context_carries_only_allowlisted_synthetic_input() -> None:
    for case in select_frozen_batch(load_gold_cases().cases):
        for worker in WorkerKind:
            request = model_request_for(case, worker)
            if request is None:
                continue
            assert request.case_id == case.case_id
            assert request.worker is worker
            assert request.question == case.question
            permitted = set(smoke_run.allowed_refs(case, worker))
            if worker is WorkerKind.PROCUREMENT_DATA:
                assert request.evidence_contexts == ()
                assert {fact.record_id for fact in request.business_facts} <= permitted
            else:
                assert request.business_facts == ()
                assert {
                    context.evidence_ref.source_id
                    for context in request.evidence_contexts
                } <= permitted


def test_dry_run_reports_the_frozen_branch_contract(
    dry_run_manifest: RunManifest,
) -> None:
    assert dry_run_manifest.execution_mode == "dry_run"
    assert dry_run_manifest.outcome == "completed"
    assert dry_run_manifest.failure_code is None
    assert dry_run_manifest.branch_count == 288
    assert dry_run_manifest.model_call_count == 264
    assert dry_run_manifest.skipped_empty_context_count == 24
    assert dry_run_manifest.succeeded_count == 264
    assert dry_run_manifest.failed_count == 0
    assert dry_run_manifest.repeat_seeds == (20260901, 20260902)
    assert dry_run_manifest.concurrency == 1
    assert dry_run_manifest.timeout_seconds == 90.0
    assert dry_run_manifest.max_retries == 1
    assert dry_run_manifest.case_split == "dev"
    assert dry_run_manifest.capabilities == {
        "chat": "supported",
        "embedding": "not_supported",
        "rerank": "not_supported",
    }
    assert sorted({call.repeat_seed for call in dry_run_manifest.calls}) == [
        20260901,
        20260902,
    ]
    assert len({call.case_id for call in dry_run_manifest.calls}) == SMOKE_CASE_COUNT
    assert dry_run_manifest.usage.total_tokens == 15_840
    assert dry_run_manifest.budget.exceeded is False
    assert dry_run_manifest.resolved_model_ids == (smoke_run.DRY_RUN_RESOLVED_MODEL_ID,)
    assert dry_run_manifest.gold_manifest_checksum == (
        "871be1687f3ddcb25080bdba9947ae935062a2f4a28f8258293542c9d54b916f"
    )


def test_guard_accepts_its_own_retention_declarations(
    dry_run_manifest: RunManifest,
) -> None:
    """Regression for the Smoke closure: declaring non-retention is not a secret."""

    names = {
        field for field in ManifestRetention.model_fields if field.endswith("_retained")
    }
    assert names == {
        "prompt_content_retained",
        "response_content_retained",
        "reasoning_content_retained",
        "authorization_header_retained",
        "api_key_retained",
        "gold_label_retained",
    }
    payload = dry_run_manifest.model_dump(mode="json")
    assert all(payload["retention"][name] is False for name in names)
    assert guard_run_manifest(dry_run_manifest, credential=DRY_RUN_CREDENTIAL) == ()


def test_guard_still_catches_a_planted_secret_value(
    dry_run_manifest: RunManifest,
) -> None:
    leaked = list(dry_run_manifest.calls)
    leaked[0] = leaked[0].model_copy(
        update={"error_detail": "leaked sk-abcdefghijklmnopqrst"}
    )
    poisoned = dry_run_manifest.model_copy(update={"calls": tuple(leaked)})
    assert guard_run_manifest(poisoned, credential=DRY_RUN_CREDENTIAL)

    cleaned = finalize_run_manifest(poisoned, credential=DRY_RUN_CREDENTIAL)
    assert cleaned.outcome == "failed_closed"
    assert cleaned.failure_code == "secret_value"
    assert cleaned.calls[0].error_detail == REDACTED_MARKER
    assert "sk-" not in json.dumps(cleaned.model_dump(mode="json"))
    assert cleaned.model_call_count == poisoned.model_call_count
    assert guard_run_manifest(cleaned, credential=DRY_RUN_CREDENTIAL) == ()


def test_guard_catches_the_live_credential_without_a_pattern(
    dry_run_manifest: RunManifest,
) -> None:
    leaked = list(dry_run_manifest.calls)
    leaked[0] = leaked[0].model_copy(update={"resolved_model_id": DRY_RUN_CREDENTIAL})
    poisoned = dry_run_manifest.model_copy(update={"calls": tuple(leaked)})
    assert guard_run_manifest(poisoned, credential=DRY_RUN_CREDENTIAL)

    cleaned = finalize_run_manifest(poisoned, credential=DRY_RUN_CREDENTIAL)
    assert cleaned.failure_code == "secret_value"
    assert DRY_RUN_CREDENTIAL not in json.dumps(cleaned.model_dump(mode="json"))


def test_guard_catches_sensitive_free_form_mapping_keys(
    dry_run_manifest: RunManifest,
) -> None:
    poisoned = dry_run_manifest.model_copy(
        update={"error_code_counts": {"authorization_header": 1}}
    )
    reasons = guard_run_manifest(poisoned, credential=DRY_RUN_CREDENTIAL)
    assert any(reason.startswith("free_form_key:") for reason in reasons)

    cleaned = finalize_run_manifest(poisoned, credential=DRY_RUN_CREDENTIAL)
    assert cleaned.outcome == "failed_closed"
    assert cleaned.failure_code == "free_form_key"
    assert cleaned.error_code_counts == {}


def test_manifest_schema_cannot_carry_content_or_admit_new_fields(
    dry_run_manifest: RunManifest,
) -> None:
    payload = dry_run_manifest.model_dump(mode="json")
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        RunManifest.model_validate({**payload, "response_body": "full model output"})

    retention = dict(payload["retention"])
    retention["response_content_retained"] = True
    with pytest.raises(Exception):
        RunManifest.model_validate({**payload, "retention": retention})

    oversized = dry_run_manifest.calls[0].model_copy(update={"error_detail": "x" * 161})
    with pytest.raises(Exception, match="at most 160"):
        SmokeCallEntry.model_validate(oversized.model_dump(mode="json"))

    for field, value in (
        ("temperature", 0.9),
        ("top_p", 0.5),
        ("max_output_tokens", 64),
        ("timeout_seconds", 5.0),
        ("concurrency", 4),
        ("max_retries", 0),
    ):
        with pytest.raises(Exception):
            RunManifest.model_validate({**payload, field: value})

    skipped = next(
        call
        for call in dry_run_manifest.calls
        if call.disposition == "skipped_empty_context"
    )
    poisoned_skip = skipped.model_copy(update={"error_code": "http_error"}).model_dump(
        mode="json"
    )
    with pytest.raises(Exception, match="must not report an error"):
        SmokeCallEntry.model_validate(poisoned_skip)


def test_manifest_aggregates_cannot_drift_from_recorded_calls(
    dry_run_manifest: RunManifest,
) -> None:
    payload = dry_run_manifest.model_dump(mode="json")
    payload["succeeded_count"] -= 1
    with pytest.raises(Exception, match="must match recorded calls"):
        RunManifest.model_validate(payload)

    payload = dry_run_manifest.model_dump(mode="json")
    payload["usage"]["input_tokens"] += 1
    with pytest.raises(Exception, match="usage must match recorded calls"):
        RunManifest.model_validate(payload)

    payload = dry_run_manifest.model_dump(mode="json")
    payload["repeat_seeds"] = [20260901]
    with pytest.raises(Exception, match="frozen"):
        RunManifest.model_validate(payload)


def test_audit_record_survives_a_failing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Smoke stays auditable when every model call fails closed."""

    def failing_transport_factory() -> httpx.AsyncBaseTransport:
        return httpx.MockTransport(
            lambda request: httpx.Response(
                500, json={"error": {"type": "server_error"}}
            )
        )

    monkeypatch.setattr(smoke_run, "_dry_run_transport", failing_transport_factory)
    batch = select_frozen_batch(load_gold_cases().cases)
    manifest = asyncio.run(
        smoke_run.run_frozen_smoke(
            execution_mode="dry_run", batch=batch, sleep=_no_sleep
        )
    )
    assert manifest.outcome == "failed_closed"
    assert manifest.failure_code == "model_call_failed"
    assert manifest.model_call_count == 264
    assert manifest.failed_count == 264
    assert manifest.succeeded_count == 0
    assert manifest.error_code_counts == {"server_error": 264}
    assert manifest.http_status_counts == {"500": 264}
    assert manifest.retry_total == 264
    assert manifest.resolved_model_ids == ()
    assert manifest.error_reason_counts == {"server_error": 264}
    assert all(call.finding_count is None for call in manifest.calls)


def test_untrusted_server_reason_keys_are_screened_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory() -> httpx.AsyncBaseTransport:
        return httpx.MockTransport(
            lambda request: httpx.Response(
                400,
                json={
                    "error": {
                        "type": "authorization_token",
                        "param": "seed",
                        "message": "not recorded",
                    }
                },
            )
        )

    monkeypatch.setattr(smoke_run, "_dry_run_transport", factory)
    batch = select_frozen_batch(load_gold_cases().cases)
    manifest = asyncio.run(
        smoke_run.run_frozen_smoke(
            execution_mode="dry_run", batch=batch, sleep=_no_sleep
        )
    )
    assert manifest.outcome == "failed_closed"
    assert manifest.failure_code == "free_form_key"
    assert manifest.error_reason_counts == {REDACTED_MARKER: 264}
    assert all(
        call.error_reason == REDACTED_MARKER
        for call in manifest.calls
        if call.disposition == "model_call"
    )
    assert manifest.model_call_count == 264
    assert manifest.failed_count == 264
    assert guard_run_manifest(manifest, credential=DRY_RUN_CREDENTIAL) == ()


def test_run_stops_at_the_frozen_request_brake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_run, "SMOKE_MAX_REQUESTS", 3)
    batch = select_frozen_batch(load_gold_cases().cases)
    manifest = asyncio.run(
        smoke_run.run_frozen_smoke(
            execution_mode="dry_run", batch=batch, sleep=_no_sleep
        )
    )
    assert manifest.outcome == "failed_closed"
    assert manifest.failure_code == "budget_exceeded"
    assert manifest.model_call_count == 3


def test_outgoing_requests_never_carry_labels_split_or_tools(
    monkeypatch: pytest.MonkeyPatch,
    dry_run_manifest: RunManifest,
) -> None:
    captured: list[dict] = []
    inner = smoke_run._dry_run_transport()

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return await inner.handle_async_request(request)

    monkeypatch.setattr(
        smoke_run, "_dry_run_transport", lambda: httpx.MockTransport(handler)
    )
    batch = select_frozen_batch(load_gold_cases().cases)
    asyncio.run(
        smoke_run.run_frozen_smoke(
            execution_mode="dry_run", batch=batch, sleep=_no_sleep
        )
    )
    assert len(captured) == dry_run_manifest.model_call_count
    markers = (
        "expected_recommendation",
        "expected_risk_level",
        "rule_ids",
        "label_id",
        "label_checksum",
        "case_family_id",
        "scenario_tags",
        "group_id",
        "split",
        "checksum",
        "holdout",
        DRY_RUN_CREDENTIAL,
    )
    offenders = sorted(
        {
            marker
            for marker in markers
            for body in captured
            if marker in json.dumps(body, ensure_ascii=False)
        }
    )
    assert offenders == []
    assert not any(key in body for body in captured for key in ("tools", "tool_choice"))
    assert {body["model"] for body in captured} == {smoke_run.DRY_RUN_MODEL_ALIAS}
    assert {body["seed"] for body in captured} == {20260901, 20260902}


def test_manifest_is_written_once_outside_the_tracked_tree(
    dry_run_manifest: RunManifest, tmp_path: Path
) -> None:
    runs_dir = tmp_path / ".local" / "runs"
    path = write_run_manifest(dry_run_manifest, runs_dir=runs_dir)
    assert path.parent.name == dry_run_manifest.run_id
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == (
        "run-manifest.v1"
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(RunManifestError, match="already_exists"):
        write_run_manifest(dry_run_manifest, runs_dir=runs_dir)
    with pytest.raises(RunManifestError, match="not_ignored"):
        write_run_manifest(dry_run_manifest, runs_dir=tmp_path / "tracked")


def test_cli_refuses_a_real_run_without_the_frozen_batch_authorization(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SMOKE_REAL_AUTHORIZATION", raising=False)
    assert smoke_run.main(["--real"]) == 1
    assert "run_not_authorized" in capsys.readouterr().err


def test_new_adapter_pins_the_frozen_sampling_contract() -> None:
    deviating = ChatAdapterConfig(
        base_url="https://real.invalid/api",
        model_name="deepseek-v4-pro",
        api_key=SecretStr("operator-local-key-never-printed"),
        temperature=0.9,
        top_p=0.5,
        seed=1,
        max_output_tokens=64,
        timeout_seconds=5.0,
        max_retries=0,
    )

    async def build(mode: str) -> tuple:
        adapter = smoke_run._new_adapter(
            execution_mode=mode,
            seed=20260902,
            real_config=deviating,
            sleep=_no_sleep,
            clock=lambda: 0.0,
        )
        config = adapter.config
        await adapter.aclose()
        return (
            config.seed,
            config.temperature,
            config.top_p,
            config.max_output_tokens,
            config.timeout_seconds,
            config.max_retries,
        )

    pinned = (20260902, 0.0, 1.0, 1_536, 90.0, 1)
    assert asyncio.run(build("real_endpoint")) == pinned
    assert asyncio.run(build("dry_run")) == pinned


def test_cli_dry_run_writes_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs_dir = tmp_path / ".local" / "runs"
    assert smoke_run.main(["--dry-run", "--runs-dir", str(runs_dir)]) == 0
    out = capsys.readouterr().out
    assert "outcome=completed" in out
    assert "calls=264" in out
    assert len(list(runs_dir.glob("*/run-manifest.v1.json"))) == 1


def test_real_endpoint_mode_requires_an_explicit_configuration() -> None:
    batch = select_frozen_batch(load_gold_cases().cases)
    with pytest.raises(RunManifestError, match="real_configuration_missing"):
        asyncio.run(
            smoke_run.run_frozen_smoke(
                execution_mode="real_endpoint", batch=batch, real_config=None
            )
        )


def test_real_endpoint_keeps_the_frozen_batch_and_pins_each_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int | None] = []
    sleep_calls: Callable[[float], Awaitable[None]] = _no_sleep

    class RecordingAdapter:
        def __init__(self, config: ChatAdapterConfig) -> None:
            seeds.append(config.seed)
            self.config = config
            self.records: list[ChatCallRecord] = []
            self.calls = 0

        async def generate(self, request: object) -> tuple:
            self.calls += 1
            self.records.append(
                ChatCallRecord(
                    case_id=request.case_id,  # type: ignore[attr-defined]
                    worker=request.worker,  # type: ignore[attr-defined]
                    requested_model_id=self.config.model_name,
                    resolved_model_id="resolved-model-v1",
                    request_id_hash="a" * 64,
                    http_status=200,
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    latency_ms=42,
                    retry_count=0,
                )
            )
            return (object(),)

        async def aclose(self) -> None:
            return None

    created: list[RecordingAdapter] = []

    def factory(**kwargs: object) -> RecordingAdapter:
        adapter = RecordingAdapter(
            replace(kwargs["real_config"], seed=kwargs["seed"])  # type: ignore[type-var]
        )
        created.append(adapter)
        return adapter

    monkeypatch.setattr(smoke_run, "_new_adapter", factory)
    config = ChatAdapterConfig(
        base_url="https://real.invalid/api",
        model_name="deepseek-v4-pro",
        api_key=SecretStr("operator-local-key-never-printed"),
        timeout_seconds=30.0,
        max_retries=1,
    )
    batch = select_frozen_batch(load_gold_cases().cases)
    manifest = asyncio.run(
        smoke_run.run_frozen_smoke(
            execution_mode="real_endpoint",
            batch=batch,
            real_config=config,
            sleep=sleep_calls,
        )
    )
    assert seeds == [20260901, 20260902]
    assert sum(adapter.calls for adapter in created) == 264
    assert manifest.execution_mode == "real_endpoint"
    assert manifest.requested_model_id == "deepseek-v4-pro"
    assert manifest.resolved_model_ids == ("resolved-model-v1",)
    assert manifest.model_call_count == 264
    assert manifest.outcome == "completed"
    assert "operator-local-key-never-printed" not in json.dumps(
        manifest.model_dump(mode="json")
    )
