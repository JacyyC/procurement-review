"""Versioned run manifest and frozen Smoke runner for the V1-02B local experiment.

The manifest is a closed field set, so the privacy guard never inspects the
manifest's own declared field names; it screens free-form mapping keys and every
string value. One manifest is therefore always produced, including when the run
fails closed, which keeps request counts, usage and errors auditable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from .chat_adapter import (
    CHAT_ADAPTER_VERSION,
    CHAT_FINDING_CONTRACT_VERSION,
    ChatAdapterConfig,
    ChatAdapterError,
    OpenAICompatibleChatAdapter,
)
from .contracts import BusinessFact, WorkerKind
from .gold_set import (
    GOLD_CORPUS_SNAPSHOT_ID,
    GOLD_DATASET_VERSION,
    GOLD_GENERATOR_VERSION,
    GOLD_POLICY_VERSION,
    GOLD_SET_VERSION,
    GoldCase,
    GoldSplit,
    load_gold_cases,
    load_gold_manifest,
    manifest_checksum,
)
from .rules import RULES_VERSION
from .run_guard import (
    finalize_payload,
    guard_payload,
)
from .worker_contracts import WorkerModelRequest

RUN_MANIFEST_SCHEMA_VERSION = "run-manifest.v1"
SMOKE_RUNNER_VERSION = "v1-02b-smoke-runner.v1"
BUDGET_VERSION = "v1-02b-local-brake.v1"
RETRIEVAL_MODE = "precomputed_gold_context.v1"

SMOKE_BATCH_ID = "v1-02b-frozen-smoke-48"
SMOKE_CASE_COUNT = 48
SMOKE_FAMILY_COUNT = 12
SMOKE_CASES_PER_FAMILY = 4
SMOKE_REPEAT_SEEDS: tuple[int, ...] = (20260901, 20260902)
SMOKE_CONCURRENCY = 1
SMOKE_TIMEOUT_SECONDS = 90.0
SMOKE_MAX_RETRIES = 1
SMOKE_MAX_REQUESTS = 1_000
SMOKE_MAX_INPUT_TOKENS = 10_000_000
SMOKE_MAX_OUTPUT_TOKENS = 1_500_000
SMOKE_TEMPERATURE = 0.0
SMOKE_TOP_P = 1.0
SMOKE_MAX_OUTPUT_TOKENS_PER_REQUEST = 1_536

DRY_RUN_MODEL_ALIAS = "dry-run-model-alias"
DRY_RUN_RESOLVED_MODEL_ID = "dry-run-model-resolved-local"
DRY_RUN_CREDENTIAL = "dry-run-placeholder-credential-not-real"

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / ".local" / "runs"
MANIFEST_FILE_NAME = "run-manifest.v1.json"
_NAME_SCREENED_FIELDS = ("error_code", "error_reason")
OPEN_MAPPING_FIELDS = (
    "capabilities",
    "http_status_counts",
    "error_code_counts",
    "error_reason_counts",
)
EXPECTED_CAPABILITIES = {
    "chat": "supported",
    "embedding": "not_supported",
    "rerank": "not_supported",
}
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


class RunManifestError(RuntimeError):
    """A bounded reason code for a run that failed closed; carries no payload."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ManifestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ManifestRetention(ManifestModel):
    """Retention declarations; the schema keeps retaining anything unrepresentable."""

    schema_version: Literal["run-manifest-retention.v1"] = "run-manifest-retention.v1"
    prompt_content_retained: Literal[False] = False
    response_content_retained: Literal[False] = False
    reasoning_content_retained: Literal[False] = False
    authorization_header_retained: Literal[False] = False
    api_key_retained: Literal[False] = False
    gold_label_retained: Literal[False] = False


class SmokeCallEntry(ManifestModel):
    """One frozen-batch branch outcome, including branches that never call."""

    case_id: str = Field(min_length=1, max_length=160)
    worker: WorkerKind
    repeat_seed: int = Field(gt=0)
    disposition: Literal["model_call", "skipped_empty_context"]
    outcome: Literal["succeeded", "failed", "not_requested"]
    requested_model_id: str | None = Field(default=None, min_length=1, max_length=300)
    resolved_model_id: str | None = Field(default=None, min_length=1, max_length=300)
    request_id_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    http_status: int | None = Field(default=None, ge=100, le=599)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0, le=1)
    finding_count: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, min_length=1, max_length=80)
    error_detail: str | None = Field(default=None, min_length=1, max_length=160)
    error_reason: str | None = Field(
        default=None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:/$\[\]-]+$"
    )

    @model_validator(mode="after")
    def disposition_matches_outcome(self) -> SmokeCallEntry:
        if self.disposition == "skipped_empty_context":
            if self.outcome != "not_requested" or self.request_id_hash is not None:
                raise ValueError("skipped branches must not report a request")
            if self.error_code or self.error_detail or self.error_reason:
                raise ValueError("skipped branches must not report an error")
        elif self.outcome == "not_requested":
            raise ValueError("called branches must report an outcome")
        if self.outcome == "failed" and self.error_code is None:
            raise ValueError("failed branches require an error code")
        return self


class ManifestSourceVersions(ManifestModel):
    adapter: str = Field(min_length=1, max_length=160)
    finding_contract: str = Field(min_length=1, max_length=160)
    gold_set: str = Field(min_length=1, max_length=160)
    gold_generator: str = Field(min_length=1, max_length=160)
    corpus_snapshot: str = Field(min_length=1, max_length=160)
    dataset: str = Field(min_length=1, max_length=160)
    policy: str = Field(min_length=1, max_length=160)
    rules: str = Field(min_length=1, max_length=160)
    retrieval: str = Field(min_length=1, max_length=160)
    budget: str = Field(min_length=1, max_length=160)
    runner: str = Field(min_length=1, max_length=160)
    code: str = Field(min_length=1, max_length=160)


class ManifestUsage(ManifestModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ManifestLatency(ManifestModel):
    observed_count: int = Field(ge=0)
    p50_ms: int | None = Field(default=None, ge=0)
    p95_ms: int | None = Field(default=None, ge=0)
    max_ms: int | None = Field(default=None, ge=0)


class ManifestBudget(ManifestModel):
    max_requests: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    consumed_requests: int = Field(ge=0)
    consumed_input_tokens: int = Field(ge=0)
    consumed_output_tokens: int = Field(ge=0)
    exceeded: bool = False

    @model_validator(mode="after")
    def exceeded_matches_consumed(self) -> ManifestBudget:
        over = (
            self.consumed_requests > self.max_requests
            or self.consumed_input_tokens > self.max_input_tokens
            or self.consumed_output_tokens > self.max_output_tokens
        )
        if over != self.exceeded:
            raise ValueError("budget exceeded flag must match consumed values")
        return self


class RunManifest(ManifestModel):
    """The only shape allowed for an auditable Smoke run."""

    schema_version: Literal["run-manifest.v1"] = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:/-]+$")
    work_item_id: Literal["V1-02"] = "V1-02"
    stage_id: Literal["V1-02B"] = "V1-02B"
    topology: Literal["chat_adapter_smoke"] = "chat_adapter_smoke"
    execution_mode: Literal["dry_run", "real_endpoint"]
    batch_id: Literal["v1-02b-frozen-smoke-48"] = SMOKE_BATCH_ID
    case_split: Literal["dev"] = "dev"
    case_count: Literal[48] = SMOKE_CASE_COUNT
    family_count: Literal[12] = SMOKE_FAMILY_COUNT
    cases_per_family: Literal[4] = SMOKE_CASES_PER_FAMILY
    repeat_seeds: tuple[int, ...]
    concurrency: Literal[1] = SMOKE_CONCURRENCY
    timeout_seconds: Literal[90.0] = SMOKE_TIMEOUT_SECONDS
    max_retries: Literal[1] = SMOKE_MAX_RETRIES
    temperature: Literal[0.0] = SMOKE_TEMPERATURE
    top_p: Literal[1.0] = SMOKE_TOP_P
    max_output_tokens: Literal[1536] = SMOKE_MAX_OUTPUT_TOKENS_PER_REQUEST
    requested_model_id: str = Field(min_length=1, max_length=300)
    resolved_model_ids: tuple[str, ...] = ()
    capabilities: dict[str, str]
    source_versions: ManifestSourceVersions
    gold_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    finished_at: datetime
    branch_count: int = Field(gt=0)
    model_call_count: int = Field(ge=0)
    skipped_empty_context_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    retry_total: int = Field(ge=0)
    usage: ManifestUsage
    latency: ManifestLatency
    http_status_counts: dict[str, int] = Field(default_factory=dict)
    error_code_counts: dict[str, int] = Field(default_factory=dict)
    error_reason_counts: dict[str, int] = Field(default_factory=dict)
    calls: tuple[SmokeCallEntry, ...]
    budget: ManifestBudget
    outcome: Literal["completed", "failed_closed"]
    failure_code: str | None = Field(default=None, min_length=1, max_length=80)
    redaction_status: Literal["redacted"] = "redacted"
    retention: ManifestRetention = Field(default_factory=ManifestRetention)

    @model_validator(mode="after")
    def aggregates_match_calls(self) -> RunManifest:
        if self.repeat_seeds != SMOKE_REPEAT_SEEDS:
            raise ValueError("Smoke repeat seeds are frozen")
        if self.capabilities != EXPECTED_CAPABILITIES:
            raise ValueError(
                "capabilities must state chat supported and others not_supported"
            )
        if self.source_versions.adapter != CHAT_ADAPTER_VERSION:
            raise ValueError("adapter version must match the running adapter")
        if self.source_versions.finding_contract != CHAT_FINDING_CONTRACT_VERSION:
            raise ValueError("finding contract version must match the running adapter")
        if not _MODEL_ID.fullmatch(self.requested_model_id):
            raise ValueError("requested_model_id must be a safe identifier")
        if any(not _MODEL_ID.fullmatch(value) for value in self.resolved_model_ids):
            raise ValueError("resolved model IDs must be safe identifiers")
        derived = derived_manifest_fields(self.calls)
        for name, value in derived.items():
            if name == "usage":
                if self.usage != value:
                    raise ValueError("usage must match recorded calls")
            elif name == "latency":
                if self.latency != value:
                    raise ValueError("latency must match recorded calls")
            elif name == "budget":
                if self.budget.model_dump() != value.model_dump():
                    raise ValueError("budget must match recorded calls")
            elif getattr(self, name) != value:
                raise ValueError(f"{name} must match recorded calls")
        if self.outcome == "completed" and (
            self.failure_code is not None or self.failed_count or self.budget.exceeded
        ):
            raise ValueError("completed runs cannot carry a failure or budget breach")
        if self.outcome == "failed_closed" and self.failure_code is None:
            raise ValueError("failed_closed runs require a failure code")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("manifest timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("manifest must finish after it starts")
        return self


def derived_manifest_fields(calls: Sequence[SmokeCallEntry]) -> dict[str, Any]:
    """Recompute every aggregate from branch entries so the manifest cannot drift."""

    entries = tuple(calls)
    model_calls = [call for call in entries if call.disposition == "model_call"]
    latencies = [call.latency_ms for call in model_calls if call.latency_ms is not None]
    input_tokens = sum(call.input_tokens or 0 for call in entries)
    output_tokens = sum(call.output_tokens or 0 for call in entries)
    status_counts: Counter[str] = Counter(
        str(call.http_status) for call in entries if call.http_status is not None
    )
    error_counts: Counter[str] = Counter(
        str(call.error_code) for call in entries if call.error_code is not None
    )
    reason_counts: Counter[str] = Counter(
        str(call.error_reason) for call in entries if call.error_reason is not None
    )
    exceeded = (
        len(model_calls) > SMOKE_MAX_REQUESTS
        or input_tokens > SMOKE_MAX_INPUT_TOKENS
        or output_tokens > SMOKE_MAX_OUTPUT_TOKENS
    )
    failed = sum(1 for call in entries if call.outcome == "failed")
    return {
        "branch_count": len(entries),
        "model_call_count": len(model_calls),
        "skipped_empty_context_count": len(entries) - len(model_calls),
        "succeeded_count": sum(1 for call in entries if call.outcome == "succeeded"),
        "failed_count": failed,
        "retry_total": sum(call.retry_count for call in entries),
        "resolved_model_ids": tuple(
            sorted({c.resolved_model_id for c in entries if c.resolved_model_id})
        ),
        "http_status_counts": dict(status_counts),
        "error_code_counts": dict(error_counts),
        "error_reason_counts": dict(reason_counts),
        "usage": ManifestUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=sum(call.total_tokens or 0 for call in entries),
        ),
        "latency": ManifestLatency(
            observed_count=len(latencies),
            p50_ms=_percentile(latencies, 0.50),
            p95_ms=_percentile(latencies, 0.95),
            max_ms=max(latencies) if latencies else None,
        ),
        "budget": ManifestBudget(
            max_requests=SMOKE_MAX_REQUESTS,
            max_input_tokens=SMOKE_MAX_INPUT_TOKENS,
            max_output_tokens=SMOKE_MAX_OUTPUT_TOKENS,
            consumed_requests=len(model_calls),
            consumed_input_tokens=input_tokens,
            consumed_output_tokens=output_tokens,
            exceeded=exceeded,
        ),
    }


def _percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def guard_run_manifest(
    manifest: RunManifest,
    *,
    credential: str,
) -> tuple[str, ...]:
    """Return bounded guard reasons; empty means the manifest is safe to write.

    Declared manifest fields are never name-screened: the closed schema is what
    makes a retention declaration such as ``response_body_retained`` addressable
    without tripping the guard.
    """

    return guard_payload(
        manifest.model_dump(mode="json"),
        credential=credential,
        open_mapping_fields=OPEN_MAPPING_FIELDS,
        calls_field="calls",
        screened_fields=_NAME_SCREENED_FIELDS,
    )


def finalize_run_manifest(
    manifest: RunManifest,
    *,
    credential: str,
) -> RunManifest:
    """Screen the manifest, scrub what it should never carry, and fail closed.

    A guard hit never discards the run: aggregates stay auditable and the outcome
    records why the run closed.
    """

    reasons = guard_run_manifest(manifest, credential=credential)
    if not reasons:
        return manifest
    cleaned = finalize_payload(
        manifest,
        credential=credential,
        open_mapping_fields=OPEN_MAPPING_FIELDS,
        calls_field="calls",
        screened_fields=_NAME_SCREENED_FIELDS,
        entry_model=SmokeCallEntry,
        derive=derived_manifest_fields,
        failure_reason="failed_closed",
        close_code=reasons[0].split(":", maxsplit=1)[0][:80],
    )
    rebuilt = RunManifest.model_validate(cleaned)
    if guard_run_manifest(rebuilt, credential=credential):
        raise RunManifestError("manifest_guard_unresolved")
    return rebuilt


def select_frozen_batch(
    cases: Sequence[GoldCase],
    *,
    split: GoldSplit = GoldSplit.DEV,
) -> tuple[GoldCase, ...]:
    """Take the first ``SMOKE_CASES_PER_FAMILY`` cases of each family in file order."""

    by_family: dict[str, list[GoldCase]] = {}
    for case in cases:
        if case.split is split:
            by_family.setdefault(case.case_family_id, []).append(case)
    if len(by_family) != SMOKE_FAMILY_COUNT:
        raise RunManifestError("batch_family_count")
    selected: list[GoldCase] = []
    for family in sorted(by_family):
        pool = by_family[family]
        if len(pool) < SMOKE_CASES_PER_FAMILY:
            raise RunManifestError("batch_family_size")
        selected += pool[:SMOKE_CASES_PER_FAMILY]
    if len(selected) != SMOKE_CASE_COUNT:
        raise RunManifestError("batch_case_count")
    return tuple(selected)


def allowed_refs(case: GoldCase, worker: WorkerKind) -> tuple[str, ...]:
    return next(
        item.allowed_ref_ids for item in case.allowed_context if item.worker is worker
    )


def model_request_for(case: GoldCase, worker: WorkerKind) -> WorkerModelRequest | None:
    """Build the model-visible request from the Gold Case allowlist, or None."""

    permitted = set(allowed_refs(case, worker))
    if worker in {WorkerKind.POLICY, WorkerKind.CONTRACT}:
        contexts = tuple(
            context
            for context in case.model_input.evidence_contexts
            if context.evidence_ref.source_id in permitted
        )
        if not contexts:
            return None
        return WorkerModelRequest(
            worker=worker,
            case_id=case.case_id,
            model_fixture_id=CHAT_FINDING_CONTRACT_VERSION,
            question=case.question,
            evidence_contexts=contexts,
        )
    facts: tuple[BusinessFact, ...] = tuple(
        fact for fact in case.model_input.business_facts if fact.record_id in permitted
    )
    if not facts:
        return None
    return WorkerModelRequest(
        worker=WorkerKind.PROCUREMENT_DATA,
        case_id=case.case_id,
        model_fixture_id=CHAT_FINDING_CONTRACT_VERSION,
        question=case.question,
        business_facts=facts,
    )


def _branch_entries(
    adapter: OpenAICompatibleChatAdapter,
    request: WorkerModelRequest,
    seed: int,
    *,
    finding_count: int | None,
    unexpected_error: str | None = None,
) -> SmokeCallEntry:
    if unexpected_error is not None:
        return SmokeCallEntry(
            case_id=request.case_id,
            worker=request.worker,
            repeat_seed=seed,
            disposition="model_call",
            outcome="failed",
            error_code="unexpected_branch_failure",
            error_detail=unexpected_error[:160],
        )
    record = adapter.records[-1]
    return SmokeCallEntry(
        case_id=request.case_id,
        worker=request.worker,
        repeat_seed=seed,
        disposition="model_call",
        outcome="failed" if record.error_code else "succeeded",
        requested_model_id=record.requested_model_id,
        resolved_model_id=record.resolved_model_id,
        request_id_hash=record.request_id_hash,
        http_status=record.http_status,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.total_tokens,
        latency_ms=record.latency_ms,
        retry_count=record.retry_count,
        finding_count=finding_count,
        error_code=record.error_code,
        error_detail=record.error_detail,
        error_reason=record.error_reason,
    )


async def _run_branch(
    adapter: OpenAICompatibleChatAdapter,
    request: WorkerModelRequest,
    *,
    seed: int,
) -> SmokeCallEntry:
    findings: object = None
    unexpected_error: str | None = None
    try:
        findings = await adapter.generate(request)
    except ChatAdapterError:
        # The adapter always emits a record for this path; it is read below.
        pass
    except Exception as exc:  # noqa: BLE001 - an unhandled error must still be audited
        unexpected_error = type(exc).__name__
    count = len(findings) if isinstance(findings, tuple) else None
    return _branch_entries(
        adapter,
        request,
        seed,
        finding_count=count,
        unexpected_error=unexpected_error,
    )


def _dry_run_transport() -> httpx.AsyncBaseTransport:
    counter = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal counter
        counter += 1
        body = json.loads(request.content)
        context = json.loads(body["messages"][1]["content"])
        worker = context["worker"]
        if worker == WorkerKind.PROCUREMENT_DATA.value:
            refs = [fact["record_id"] for fact in context["business_facts"]]
            category = "procurement_anomaly"
            finding_key = "procurement.required_business_facts"
            citation = {"business_fact_refs": refs[:1]}
        else:
            refs = [item["ref_id"] for item in context["evidence_contexts"]]
            category = (
                "policy_compliance"
                if worker == WorkerKind.POLICY.value
                else "contract_terms"
            )
            finding_key = f"{worker}.required_evidence"
            citation = {"evidence_refs": refs[:1]}
        finding = {
            "schema_version": "1.0",
            "worker": worker,
            "finding_key": finding_key,
            "category": category,
            "risk_level": "low",
            "assertion": "risk_absent",
            "statement": "The synthetic context supports the required finding.",
            "status": "supported",
            "recommended_action": "Continue with deterministic review.",
            **citation,
        }
        return httpx.Response(
            200,
            json={
                "id": f"dry-run-request-{counter:04d}",
                "model": DRY_RUN_RESOLVED_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"findings": [finding]}, ensure_ascii=False
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
            },
        )

    return httpx.MockTransport(handler)


def _new_adapter(
    *,
    execution_mode: Literal["dry_run", "real_endpoint"],
    seed: int,
    real_config: ChatAdapterConfig | None,
    sleep: Callable[[float], Any],
    clock: Callable[[], float],
) -> OpenAICompatibleChatAdapter:
    if execution_mode == "dry_run":
        config = ChatAdapterConfig(
            base_url="https://dry-run.invalid/api",
            model_name=DRY_RUN_MODEL_ALIAS,
            api_key=SecretStr(DRY_RUN_CREDENTIAL),
            temperature=SMOKE_TEMPERATURE,
            top_p=SMOKE_TOP_P,
            max_output_tokens=SMOKE_MAX_OUTPUT_TOKENS_PER_REQUEST,
            seed=seed,
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
            max_retries=SMOKE_MAX_RETRIES,
        )
        return OpenAICompatibleChatAdapter(
            config,
            transport=_dry_run_transport(),
            sleep=sleep,
            clock=clock,
        )
    if real_config is None:
        raise RunManifestError("real_configuration_missing")
    return OpenAICompatibleChatAdapter(
        replace(
            real_config,
            seed=seed,
            temperature=SMOKE_TEMPERATURE,
            top_p=SMOKE_TOP_P,
            max_output_tokens=SMOKE_MAX_OUTPUT_TOKENS_PER_REQUEST,
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
            max_retries=SMOKE_MAX_RETRIES,
        ),
        sleep=sleep,
        clock=clock,
    )


def _run_id(execution_mode: str, at: datetime) -> str:
    stamp = at.strftime("%Y%m%dT%H%M%SZ")
    salt = hashlib.sha256(
        f"{stamp}-{at.microsecond}-{os.getpid()}-{time.monotonic()}".encode()
    ).hexdigest()
    return f"smoke-{execution_mode}-{stamp}-{salt[:8]}"


async def run_frozen_smoke(
    *,
    execution_mode: Literal["dry_run", "real_endpoint"],
    batch: Sequence[GoldCase],
    real_config: ChatAdapterConfig | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RunManifest:
    """Execute the frozen Smoke batch; the audit record survives any failure."""

    started_at = now()
    if execution_mode == "real_endpoint" and real_config is None:
        raise RunManifestError("real_configuration_missing")
    config = None if execution_mode == "dry_run" else real_config
    credential = (
        DRY_RUN_CREDENTIAL if config is None else config.api_key.get_secret_value()
    )
    requested_model_id = DRY_RUN_MODEL_ALIAS if config is None else config.model_name
    entries: list[SmokeCallEntry] = []
    aborted_code: str | None = None

    for seed in SMOKE_REPEAT_SEEDS:
        if aborted_code:
            break
        adapter = _new_adapter(
            execution_mode=execution_mode,
            seed=seed,
            real_config=real_config,
            sleep=sleep,
            clock=clock,
        )
        try:
            for case in batch:
                if aborted_code:
                    break
                for worker in WorkerKind:
                    request = model_request_for(case, worker)
                    if request is None:
                        entries.append(
                            SmokeCallEntry(
                                case_id=case.case_id,
                                worker=worker,
                                repeat_seed=seed,
                                disposition="skipped_empty_context",
                                outcome="not_requested",
                            )
                        )
                        continue
                    if _budget_breached(entries):
                        aborted_code = "budget_exceeded"
                        break
                    entries.append(await _run_branch(adapter, request, seed=seed))
        finally:
            await adapter.aclose()

    finished_at = now()
    calls = tuple(entries)
    derived = derived_manifest_fields(calls)
    payload = {
        "run_id": _run_id(execution_mode, started_at),
        "execution_mode": execution_mode,
        "repeat_seeds": SMOKE_REPEAT_SEEDS,
        "requested_model_id": requested_model_id,
        "capabilities": dict(EXPECTED_CAPABILITIES),
        "source_versions": ManifestSourceVersions(
            adapter=CHAT_ADAPTER_VERSION,
            finding_contract=CHAT_FINDING_CONTRACT_VERSION,
            gold_set=GOLD_SET_VERSION,
            gold_generator=GOLD_GENERATOR_VERSION,
            corpus_snapshot=GOLD_CORPUS_SNAPSHOT_ID,
            dataset=GOLD_DATASET_VERSION,
            policy=GOLD_POLICY_VERSION,
            rules=RULES_VERSION,
            retrieval=RETRIEVAL_MODE,
            budget=BUDGET_VERSION,
            runner=SMOKE_RUNNER_VERSION,
            code=_git_revision(),
        ),
        "gold_manifest_checksum": manifest_checksum(load_gold_manifest()),
        "started_at": started_at,
        "finished_at": finished_at,
        "calls": calls,
        **derived,
    }
    if aborted_code:
        outcome, failure_code = "failed_closed", aborted_code
    elif derived["failed_count"] or derived["budget"].exceeded:
        outcome, failure_code = (
            "failed_closed",
            ("budget_exceeded" if derived["budget"].exceeded else "model_call_failed"),
        )
    else:
        outcome, failure_code = "completed", None
    payload.update({"outcome": outcome, "failure_code": failure_code})
    return finalize_run_manifest(
        RunManifest.model_validate(payload), credential=credential
    )


def _budget_breached(entries: Sequence[SmokeCallEntry]) -> bool:
    return (
        sum(1 for entry in entries if entry.disposition == "model_call")
        >= SMOKE_MAX_REQUESTS
        or sum(entry.input_tokens or 0 for entry in entries) >= SMOKE_MAX_INPUT_TOKENS
        or sum(entry.output_tokens or 0 for entry in entries) >= SMOKE_MAX_OUTPUT_TOKENS
    )


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return "unknown"
    value = result.stdout.strip()
    return value if re.fullmatch(r"[0-9a-f]{7,40}", value) else "unknown"


def write_run_manifest(
    manifest: RunManifest,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Persist one manifest outside the tracked tree with owner-only permissions."""

    resolved = runs_dir.resolve()
    if ".local" not in resolved.parts:
        raise RunManifestError("manifest_dir_not_ignored")
    directory = resolved / manifest.run_id
    path = directory / MANIFEST_FILE_NAME
    if path.exists():
        raise RunManifestError("manifest_already_exists")
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    directory.mkdir(parents=True, exist_ok=False)
    resolved.chmod(0o700)
    directory.chmod(0o700)
    path.write_bytes(payload)
    path.chmod(0o600)
    return path


def summarize(manifest: RunManifest) -> str:
    """A bounded summary containing no request or response content."""

    failure = f" failure_code={manifest.failure_code}" if manifest.failure_code else ""
    return (
        f"run_id={manifest.run_id} mode={manifest.execution_mode} "
        f"outcome={manifest.outcome}{failure} "
        f"branches={manifest.branch_count} calls={manifest.model_call_count} "
        f"skipped={manifest.skipped_empty_context_count} "
        f"succeeded={manifest.succeeded_count} failed={manifest.failed_count} "
        f"retries={manifest.retry_total} "
        f"tokens=in:{manifest.usage.input_tokens} "
        f"out:{manifest.usage.output_tokens} total:{manifest.usage.total_tokens} "
        f"latency_p50={manifest.latency.p50_ms} p95={manifest.latency.p95_ms} "
        f"max={manifest.latency.max_ms} "
        f"http={dict(sorted(manifest.http_status_counts.items()))} "
        f"errors={dict(sorted(manifest.error_code_counts.items()))} "
        f"error_reasons={dict(sorted(manifest.error_reason_counts.items()))} "
        f"resolved_model_ids={list(manifest.resolved_model_ids)}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen V1-02B synthetic Chat Smoke with a run manifest."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="use an offline mock transport and a placeholder credential; no network",
    )
    mode.add_argument(
        "--real",
        action="store_true",
        help=(
            "run the frozen batch against the locally configured endpoint; requires "
            f"SMOKE_REAL_AUTHORIZATION={SMOKE_BATCH_ID}"
        ),
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    execution_mode: Literal["dry_run", "real_endpoint"] = (
        "real_endpoint" if args.real else "dry_run"
    )
    try:
        if execution_mode == "real_endpoint" and (
            os.environ.get("SMOKE_REAL_AUTHORIZATION", "") != SMOKE_BATCH_ID
        ):
            raise RunManifestError("run_not_authorized")
        config = None if args.dry_run else ChatAdapterConfig.from_env()
        batch = select_frozen_batch(load_gold_cases().cases)
        manifest = asyncio.run(
            run_frozen_smoke(
                execution_mode=execution_mode, batch=batch, real_config=config
            )
        )
        path = write_run_manifest(manifest, runs_dir=args.runs_dir)
    except (RunManifestError, OSError, ValueError) as exc:
        code = getattr(exc, "code", None) or type(exc).__name__
        print(f"smoke-error: {code}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    print(summarize(manifest))
    return 0 if manifest.outcome == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
