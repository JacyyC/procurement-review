"""Run contract for the paired single-agent / multi-agent evaluation batches.

One manifest is written per ``(batch, topology, repeat_seed)`` segment so a
multi-hour batch stays auditable and resumable, while ``batch_id`` binds the
segments back into one paired experiment. As in V1-02B, the manifest is a closed
field set and every aggregate is recomputed from the branch entries, so a run can
never restate its own results.

Findings survive the run in a separate ``eval-projection.v1`` document. Statement and
recommended-action text become hashes, and no prompt or response body is retained; what
that document does keep verbatim is each Finding's ``missing_information`` list, bounded by
schema to twelve fragments of at most 160 characters, because the deterministic aggregator
treats a model's own statement of what is missing as evidence of incompleteness. The
derived report no longer copies that prose (see ``eval-report.v3``).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .chat_adapter import CHAT_ADAPTER_VERSION, CHAT_FINDING_CONTRACT_VERSION
from .contracts import (
    FindingAssertion,
    FindingStatus,
    RiskCategory,
    RiskLevel,
    WorkerKind,
)
from .gold_set import (
    GOLD_CORPUS_CONTRACTS,
    GoldCorpusContract,
    GoldCorpusVersion,
    GoldSplit,
)
from .run_guard import (
    REDACTED_MARKER,
    finalize_payload,
    guard_payload,
    is_safe_model_id,
    secret_value_paths,
)

EVAL_MANIFEST_SCHEMA_VERSION = "eval-run-manifest.v1"
EVAL_PROJECTION_SCHEMA_VERSION = "eval-projection.v1"
EVAL_RUNNER_VERSION = "v1-02c-eval-runner.v1"
EVAL_BUDGET_VERSION = "v1-02c-local-brake.v1"
EVAL_FINDING_CONTRACT_VERSION = "chat-finding-envelope.v1"
EVAL_RETRIEVAL_MODE = "precomputed_gold_context.v1"

EVAL_TEMPERATURE = 0.0
EVAL_TOP_P = 1.0
EVAL_MAX_OUTPUT_TOKENS_PER_REQUEST = 1_536
EVAL_TIMEOUT_SECONDS = 90.0
EVAL_MAX_RETRIES = 1

# Locked by the user for the whole V1-02C comparison: an alias may be requested,
# but every call must resolve to this exact server-side model identifier.
MODEL_ID_LOCK = "deepseek-v4-flash-ga-260731"

BUDGET_WARNING_RATIO = 0.8
# Failure codes that void a run's conclusions rather than merely closing it: the
# data is still auditable, but it can no longer support a quality statement.
_VOIDING_CODES = frozenset(
    {"model_drift", "gold_label_leak_detected", "record_incomplete"}
)
DRY_RUN_MODEL_ALIAS = "dry-run-model-alias"
DRY_RUN_RESOLVED_MODEL_ID = "dry-run-model-resolved-local"
DRY_RUN_CREDENTIAL = "dry-run-placeholder-credential-not-real"
# The offline transport resolves to the same identifier it is locked to, so the
# drift check is exercised rather than trivially bypassed.
DRY_RUN_MODEL_ID_LOCK = DRY_RUN_RESOLVED_MODEL_ID

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / ".local" / "runs"
MANIFEST_FILE_NAME = "eval-run-manifest.v1.json"
PROJECTION_FILE_NAME = "eval-projection.v1.json"

EVAL_TOPOLOGIES: tuple[str, ...] = ("single_agent", "multi_agent")
# ``holdout`` is only reachable through a batch that declares it (Spec §6.2); the runner
# gate in select_batch_cases stays the single place that decides whether it may be read.
EvalSplitName = Literal["dev", "validation", "holdout"]


class EvalBatchSpec(BaseModel):
    """A frozen batch: case selection, repeats, concurrency and local brakes."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: Literal["eval-batch-spec.v1"] = "eval-batch-spec.v1"
    batch_id: str = Field(min_length=1, max_length=160, pattern=r"^[a-z0-9-]+$")
    splits: tuple[GoldSplit, ...]
    cases_per_family: int = Field(gt=0)
    family_count: Literal[12] = 12
    case_count: int = Field(gt=0)
    repeat_seeds: tuple[int, ...] = Field(min_length=1)
    concurrency: int = Field(gt=0, le=8)
    max_requests: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    model_id_lock: str = Field(min_length=1, max_length=300)
    gold_corpus_version: Literal["v1", "v2"] = "v1"
    stage_id: Literal["V1-02C", "V1-02C2", "V1-05B"] = "V1-02C"
    note: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def counts_match_selection(self) -> EvalBatchSpec:
        if self.case_count != self.cases_per_family * self.family_count:
            raise ValueError("batch case count must be family_count x cases_per_family")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("batch splits must be unique")
        if len(set(self.repeat_seeds)) != len(self.repeat_seeds):
            raise ValueError("batch repeat seeds must be unique")
        if not is_safe_model_id(self.model_id_lock):
            raise ValueError("model_id_lock must be a safe identifier")
        return self


# C-1 calibrates the single-agent path against the real endpoint before any
# conclusion-bearing batch; C-2 is the paired source for AC-06..AC-08.
CALIBRATION_BATCH = EvalBatchSpec(
    batch_id="v1-02c-calibration-48",
    splits=(GoldSplit.DEV,),
    cases_per_family=4,
    case_count=48,
    repeat_seeds=(20260901,),
    concurrency=1,
    max_requests=1_000,
    max_input_tokens=10_000_000,
    max_output_tokens=1_500_000,
    model_id_lock=MODEL_ID_LOCK,
    note="calibration only; not used for any AC conclusion",
)
PAIRED_BATCH = EvalBatchSpec(
    batch_id="v1-02c-paired-360x3",
    splits=(GoldSplit.DEV, GoldSplit.VALIDATION),
    cases_per_family=30,
    case_count=360,
    repeat_seeds=(20260901, 20260902, 20260903),
    concurrency=4,
    max_requests=25_000,
    max_input_tokens=250_000_000,
    max_output_tokens=24_000_000,
    model_id_lock=MODEL_ID_LOCK,
    note="dev 240 + validation 120; holdout is never read in V1-02C",
)
# V1-02C2 re-runs the same contract on the clause-level corpus v2, where a
# recommendation, a conflict and a retrieval hit can actually be distinguished.
RESOLUTION_BATCH = EvalBatchSpec(
    batch_id="v1-02c2-resolution-120x2",
    splits=(GoldSplit.DEV,),
    cases_per_family=10,
    case_count=120,
    repeat_seeds=(20260901, 20260902),
    concurrency=4,
    max_requests=25_000,
    max_input_tokens=250_000_000,
    max_output_tokens=24_000_000,
    model_id_lock=MODEL_ID_LOCK,
    gold_corpus_version="v2",
    stage_id="V1-02C2",
    note=(
        "Gold corpus v2 documents on dev 120; validation/holdout not read; brakes "
        "are enforced cumulatively across every real batch of §15.4 Decision tier"
    ),
)
# V1-05B is the one pre-registered real paired batch of Spec §6.2: the clause-level
# corpus's hidden holdout split, unsealed once and only for this batch. Its caps are the
# §6.2 hard limits, narrower than the §15.4 Decision tier the historical batches declared.
HOLDOUT_REJUDGEMENT_BATCH = EvalBatchSpec(
    batch_id="v1-05b-holdout-120x2",
    splits=(GoldSplit.HOLDOUT,),
    cases_per_family=10,
    case_count=120,
    repeat_seeds=(20260901, 20260902),
    concurrency=4,
    max_requests=1_200,
    max_input_tokens=20_000_000,
    max_output_tokens=4_000_000,
    model_id_lock=MODEL_ID_LOCK,
    gold_corpus_version="v2",
    stage_id="V1-05B",
    note=(
        "holdout 120 on Gold corpus v2; the split is unsealed once for this batch only "
        "and never read again for tuning; brakes also apply cumulatively to §15.4"
    ),
)
EVAL_BATCHES: dict[str, EvalBatchSpec] = {
    spec.batch_id: spec
    for spec in (
        CALIBRATION_BATCH,
        PAIRED_BATCH,
        RESOLUTION_BATCH,
        HOLDOUT_REJUDGEMENT_BATCH,
    )
}
# Which work item owns each registered stage, so a manifest never restates it by default.
STAGE_WORK_ITEM_IDS: dict[str, Literal["V1-02", "V1-05"]] = {
    "V1-02C": "V1-02",
    "V1-02C2": "V1-02",
    "V1-05B": "V1-05",
}


def gold_corpus_contract(spec: EvalBatchSpec) -> GoldCorpusContract:
    """The frozen Gold corpus this batch must run and be scored against."""

    return GOLD_CORPUS_CONTRACTS[GoldCorpusVersion(spec.gold_corpus_version)]


_EVAL_NAME_SCREENED_FIELDS = ("error_code", "error_reason")
_EVAL_OPEN_MAPPING_FIELDS = (
    "capabilities",
    "http_status_counts",
    "error_code_counts",
    "error_reason_counts",
)


class EvalContractError(RuntimeError):
    """A bounded reason code for an eval run that cannot proceed."""

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


class EvalRetention(ManifestModel):
    """Retention declarations; the schema keeps retaining anything unrepresentable."""

    schema_version: Literal["eval-manifest-retention.v1"] = "eval-manifest-retention.v1"
    prompt_content_retained: Literal[False] = False
    response_content_retained: Literal[False] = False
    reasoning_content_retained: Literal[False] = False
    model_statement_text_retained: Literal[False] = False
    authorization_header_retained: Literal[False] = False
    api_key_retained: Literal[False] = False
    gold_label_retained: Literal[False] = False


class EvalSourceVersions(ManifestModel):
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


class EvalUsage(ManifestModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class EvalLatency(ManifestModel):
    observed_count: int = Field(ge=0)
    p50_ms: int | None = Field(default=None, ge=0)
    p95_ms: int | None = Field(default=None, ge=0)
    max_ms: int | None = Field(default=None, ge=0)


class EvalBudget(ManifestModel):
    max_requests: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    consumed_requests: int = Field(ge=0)
    consumed_input_tokens: int = Field(ge=0)
    consumed_output_tokens: int = Field(ge=0)
    warning_reached: bool = False
    exceeded: bool = False

    @model_validator(mode="after")
    def flags_match_consumed(self) -> EvalBudget:
        warning = (
            self.consumed_requests >= self.max_requests * BUDGET_WARNING_RATIO
            or self.consumed_input_tokens
            >= self.max_input_tokens * BUDGET_WARNING_RATIO
            or self.consumed_output_tokens
            >= self.max_output_tokens * BUDGET_WARNING_RATIO
        )
        if warning != self.warning_reached:
            raise ValueError("budget warning flag must match consumed values")
        over = (
            self.consumed_requests > self.max_requests
            or self.consumed_input_tokens > self.max_input_tokens
            or self.consumed_output_tokens > self.max_output_tokens
        )
        if over != self.exceeded:
            raise ValueError("budget exceeded flag must match consumed values")
        return self


class EvalBranchEntry(ManifestModel):
    """One ``(case, worker)`` branch of a segment, including branches that never call."""

    case_id: str = Field(min_length=1, max_length=160)
    split: EvalSplitName
    worker: WorkerKind | None
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
    def shape_matches_disposition(self) -> EvalBranchEntry:
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


class EvalRunManifest(ManifestModel):
    """The only shape allowed for one auditable segment of a paired eval batch."""

    schema_version: Literal["eval-run-manifest.v1"] = EVAL_MANIFEST_SCHEMA_VERSION
    run_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/-]+$")
    work_item_id: Literal["V1-02", "V1-05"] = "V1-02"
    stage_id: Literal["V1-02C", "V1-02C2", "V1-05B"] = "V1-02C"
    batch_id: str = Field(min_length=1, max_length=160, pattern=r"^[a-z0-9-]+$")
    topology: Literal["single_agent", "multi_agent"]
    repeat_seed: int = Field(gt=0)
    execution_mode: Literal["dry_run", "real_endpoint"]
    case_split_scope: str = Field(min_length=1, max_length=80)
    case_count: int = Field(gt=0)
    family_count: Literal[12] = 12
    cases_per_family: int = Field(gt=0)
    concurrency: int = Field(gt=0, le=8)
    timeout_seconds: Literal[90.0] = EVAL_TIMEOUT_SECONDS
    max_retries: Literal[1] = EVAL_MAX_RETRIES
    temperature: Literal[0.0] = EVAL_TEMPERATURE
    top_p: Literal[1.0] = EVAL_TOP_P
    max_output_tokens: Literal[1536] = EVAL_MAX_OUTPUT_TOKENS_PER_REQUEST
    model_id_lock: str = Field(min_length=1, max_length=300)
    required_findings_included: Literal[True] = True
    requested_model_id: str = Field(min_length=1, max_length=300)
    resolved_model_ids: tuple[str, ...] = ()
    capabilities: dict[str, str]
    source_versions: EvalSourceVersions
    gold_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: datetime
    finished_at: datetime
    branch_count: int = Field(gt=0)
    model_call_count: int = Field(ge=0)
    skipped_empty_context_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    retry_total: int = Field(ge=0)
    usage: EvalUsage
    latency: EvalLatency
    http_status_counts: dict[str, int] = Field(default_factory=dict)
    error_code_counts: dict[str, int] = Field(default_factory=dict)
    error_reason_counts: dict[str, int] = Field(default_factory=dict)
    calls: tuple[EvalBranchEntry, ...]
    budget: EvalBudget
    outcome: Literal["completed", "failed_closed"]
    failure_code: str | None = Field(default=None, min_length=1, max_length=80)
    voided: bool = False
    redaction_status: Literal["redacted"] = "redacted"
    retention: EvalRetention = Field(default_factory=EvalRetention)

    @model_validator(mode="after")
    def aggregates_match_calls(self) -> EvalRunManifest:
        spec = EVAL_BATCHES.get(self.batch_id)
        if spec is None:
            raise ValueError("batch_id must be a registered frozen batch")
        if spec.concurrency != self.concurrency:
            raise ValueError("concurrency is frozen per batch")
        if spec.repeat_seeds and self.repeat_seed not in spec.repeat_seeds:
            raise ValueError("repeat seed is not part of the frozen batch")
        if self.case_count != spec.case_count:
            raise ValueError("case count is frozen per batch")
        if self.cases_per_family != spec.cases_per_family:
            raise ValueError("cases per family is frozen per batch")
        if self.case_split_scope != "+".join(split.value for split in spec.splits):
            raise ValueError("case split scope is frozen per batch")
        scope = {split.value for split in spec.splits}
        if any(call.split not in scope for call in self.calls):
            raise ValueError("branch split outside the frozen batch scope")
        if self.stage_id != spec.stage_id:
            raise ValueError("stage id is frozen per batch")
        if self.work_item_id != STAGE_WORK_ITEM_IDS[self.stage_id]:
            raise ValueError("work item id must follow the registered stage")
        expected_corpus = gold_corpus_contract(spec)
        if self.source_versions.corpus_snapshot != expected_corpus.corpus_snapshot_id:
            raise ValueError("corpus snapshot does not match the batch")
        if self.source_versions.gold_set != expected_corpus.gold_set_version:
            raise ValueError("gold set version does not match the batch")
        if self.topology not in EVAL_TOPOLOGIES:
            raise ValueError("topology must be single_agent or multi_agent")
        locked = (
            DRY_RUN_MODEL_ID_LOCK
            if self.execution_mode == "dry_run"
            else spec.model_id_lock
        )
        if self.model_id_lock != locked:
            raise ValueError("model id lock is frozen per batch")
        if any(not is_safe_model_id(value) for value in self.resolved_model_ids):
            raise ValueError("resolved model IDs must be safe identifiers")
        if not is_safe_model_id(self.requested_model_id):
            raise ValueError("requested_model_id must be a safe identifier")
        if self.capabilities != expected_capabilities():
            raise ValueError(
                "capabilities must state chat supported and others not_supported"
            )
        if self.source_versions.adapter != CHAT_ADAPTER_VERSION:
            raise ValueError("adapter version must match the running adapter")
        if self.source_versions.finding_contract != CHAT_FINDING_CONTRACT_VERSION:
            raise ValueError("finding contract version must match the running adapter")
        derived = derived_eval_fields(self.calls, spec=spec)
        for name, value in derived.items():
            if name in {"usage", "latency", "budget"}:
                if getattr(self, name) != value:
                    raise ValueError(f"{name} must match recorded calls")
            elif getattr(self, name) != value:
                raise ValueError(f"{name} must match recorded calls")
        if self.outcome == "completed" and (
            self.failure_code is not None or self.failed_count or self.budget.exceeded
        ):
            raise ValueError("completed runs cannot carry a failure or budget breach")
        if self.outcome == "failed_closed" and self.failure_code is None:
            raise ValueError("failed_closed runs require a failure code")
        if self.voided != (self.failure_code in _VOIDING_CODES):
            raise ValueError("voided flag must match a voiding failure code")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("manifest timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("manifest must finish after it starts")
        return self


def expected_capabilities() -> dict[str, str]:
    return {
        "chat": "supported",
        "embedding": "not_supported",
        "rerank": "not_supported",
    }


def percentile(values: Sequence[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def derived_eval_fields(
    calls: Sequence[EvalBranchEntry],
    *,
    spec: EvalBatchSpec,
) -> dict[str, Any]:
    """Recompute every aggregate from branch entries so the manifest cannot drift."""

    entries = tuple(calls)
    model_calls = [call for call in entries if call.disposition == "model_call"]
    latencies = [call.latency_ms for call in model_calls if call.latency_ms is not None]
    input_tokens = sum(call.input_tokens or 0 for call in entries)
    output_tokens = sum(call.output_tokens or 0 for call in entries)
    consumed_requests = len(model_calls)
    exceeded = (
        consumed_requests > spec.max_requests
        or input_tokens > spec.max_input_tokens
        or output_tokens > spec.max_output_tokens
    )
    warning = (
        consumed_requests >= spec.max_requests * BUDGET_WARNING_RATIO
        or input_tokens >= spec.max_input_tokens * BUDGET_WARNING_RATIO
        or output_tokens >= spec.max_output_tokens * BUDGET_WARNING_RATIO
    )
    return {
        "branch_count": len(entries),
        "model_call_count": consumed_requests,
        "skipped_empty_context_count": len(entries) - len(model_calls),
        "succeeded_count": sum(1 for call in entries if call.outcome == "succeeded"),
        "failed_count": sum(1 for call in entries if call.outcome == "failed"),
        "retry_total": sum(call.retry_count for call in entries),
        "resolved_model_ids": tuple(
            sorted(
                {call.resolved_model_id for call in entries if call.resolved_model_id}
            )
        ),
        "http_status_counts": dict(
            Counter(str(call.http_status) for call in entries if call.http_status)
        ),
        "error_code_counts": dict(
            Counter(str(call.error_code) for call in entries if call.error_code)
        ),
        "error_reason_counts": dict(
            Counter(str(call.error_reason) for call in entries if call.error_reason)
        ),
        "usage": EvalUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=sum(call.total_tokens or 0 for call in entries),
        ),
        "latency": EvalLatency(
            observed_count=len(latencies),
            p50_ms=percentile(latencies, 0.50),
            p95_ms=percentile(latencies, 0.95),
            max_ms=max(latencies) if latencies else None,
        ),
        "budget": EvalBudget(
            max_requests=spec.max_requests,
            max_input_tokens=spec.max_input_tokens,
            max_output_tokens=spec.max_output_tokens,
            consumed_requests=consumed_requests,
            consumed_input_tokens=input_tokens,
            consumed_output_tokens=output_tokens,
            warning_reached=warning,
            exceeded=exceeded,
        ),
    }


def guard_eval_manifest(
    manifest: EvalRunManifest, *, credential: str
) -> tuple[str, ...]:
    return guard_payload(
        manifest.model_dump(mode="json"),
        credential=credential,
        open_mapping_fields=_EVAL_OPEN_MAPPING_FIELDS,
        calls_field="calls",
        screened_fields=_EVAL_NAME_SCREENED_FIELDS,
    )


def finalize_eval_manifest(
    manifest: EvalRunManifest,
    *,
    credential: str,
    spec: EvalBatchSpec,
) -> EvalRunManifest:
    """Screen and scrub a segment manifest, keeping the run auditable when it hits."""

    reasons = guard_eval_manifest(manifest, credential=credential)
    if not reasons:
        return manifest
    cleaned = finalize_payload(
        manifest,
        credential=credential,
        open_mapping_fields=_EVAL_OPEN_MAPPING_FIELDS,
        calls_field="calls",
        screened_fields=_EVAL_NAME_SCREENED_FIELDS,
        entry_model=EvalBranchEntry,
        derive=lambda entries: derived_eval_fields(entries, spec=spec),
        failure_reason="failed_closed",
        close_code=reasons[0].split(":", maxsplit=1)[0][:80],
    )
    # The scrubbed failure code replaces the original reason, so the void flag has
    # to be re-derived or a guarded manifest could not be written at all.
    cleaned["voided"] = cleaned.get("failure_code") in _VOIDING_CODES
    rebuilt = EvalRunManifest.model_validate(cleaned)
    if guard_eval_manifest(rebuilt, credential=credential):
        raise EvalContractError("manifest_guard_unresolved")
    return rebuilt


class ProjectedRef(ManifestModel):
    """The identity fields needed to re-resolve a reference from its Gold Case."""

    source_id: str = Field(min_length=1, max_length=160)
    version: str = Field(min_length=1, max_length=160)
    content_checksum: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")


class ProjectedFinding(ManifestModel):
    """Identity-only view of one Finding; model narrative text is hashed, not stored."""

    schema_version: Literal["projected-finding.v1"] = "projected-finding.v1"
    worker: WorkerKind
    finding_key: str = Field(min_length=1, max_length=160)
    category: RiskCategory
    risk_level: RiskLevel
    assertion: FindingAssertion
    status: FindingStatus
    evidence_refs: tuple[ProjectedRef, ...] = ()
    business_fact_refs: tuple[ProjectedRef, ...] = ()
    # ``project_findings`` keeps at most twelve model-authored fragments of at most 160
    # characters; the bound is declared here so it cannot widen by accident.
    missing_information: tuple[Annotated[str, Field(max_length=160)], ...] = Field(
        default=(), max_length=12
    )
    statement_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recommended_action_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProjectedBranch(BaseModel):
    """Scoreable result of one branch: the outcome plus its projected findings."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: Literal["projected-branch.v1"] = "projected-branch.v1"
    case_id: str = Field(min_length=1, max_length=160)
    split: EvalSplitName
    worker: WorkerKind | None
    outcome: Literal["succeeded", "failed", "not_requested"]
    error_code: str | None = Field(default=None, min_length=1, max_length=80)
    findings: tuple[ProjectedFinding, ...] = ()


class EvalProjection(BaseModel):
    """The post-run scoring input for one segment; carries no Gold Label."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: Literal["eval-projection.v1"] = EVAL_PROJECTION_SCHEMA_VERSION
    run_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/-]+$")
    batch_id: str = Field(min_length=1, max_length=160, pattern=r"^[a-z0-9-]+$")
    topology: Literal["single_agent", "multi_agent"]
    repeat_seed: int = Field(gt=0)
    execution_mode: Literal["dry_run", "real_endpoint"]
    model_id_lock: str = Field(min_length=1, max_length=300)
    redaction_count: int = Field(default=0, ge=0)
    branches: tuple[ProjectedBranch, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def branch_shape_matches_topology(self) -> EvalProjection:
        for branch in self.branches:
            if self.topology == "multi_agent" and branch.worker is None:
                raise ValueError("multi-agent branches must name their worker")
            if self.topology == "single_agent" and branch.worker is not None:
                raise ValueError("single-agent branches must not name a worker")
        return self


def _screen_projection(projection: EvalProjection) -> tuple[EvalProjection, int]:
    """Redact any screened string in a projection without dropping the branch."""

    payload = projection.model_dump(mode="json")
    paths = frozenset(secret_value_paths(payload, credential=""))
    if not paths:
        return projection, 0
    hits = 0

    def walk(value: Any, path: tuple[str | int, ...]) -> Any:
        nonlocal hits
        if isinstance(value, dict):
            return {key: walk(child, (*path, str(key))) for key, child in value.items()}
        if isinstance(value, list):
            return [walk(child, (*path, index)) for index, child in enumerate(value)]
        if isinstance(value, str) and path in paths:
            hits += 1
            return REDACTED_MARKER
        return value

    return EvalProjection.model_validate(walk(payload, ())), hits


def build_projection(
    *,
    run_id: str,
    spec: EvalBatchSpec,
    topology: Literal["single_agent", "multi_agent"],
    repeat_seed: int,
    execution_mode: Literal["dry_run", "real_endpoint"],
    branches: Sequence[ProjectedBranch],
) -> EvalProjection:
    projection = EvalProjection(
        run_id=run_id,
        batch_id=spec.batch_id,
        topology=topology,
        repeat_seed=repeat_seed,
        execution_mode=execution_mode,
        model_id_lock=(
            DRY_RUN_MODEL_ID_LOCK if execution_mode == "dry_run" else spec.model_id_lock
        ),
        branches=tuple(branches),
    )
    scope = {split.value for split in spec.splits}
    if any(branch.split not in scope for branch in projection.branches):
        raise EvalContractError("projection_scope_violation")
    screened, hits = _screen_projection(projection)
    if not hits:
        return screened
    return screened.model_copy(update={"redaction_count": hits})


def private_runs_root(path: Path, *, create: bool = False) -> Path:
    """Resolve a directory that is allowed to hold run material, and narrow it on write.

    Run artifacts live outside the tracked tree precisely because they are not shareable,
    so the directory that holds them is set owner-only when something is written into it.
    Read paths never create or re-permission anything.
    """

    root = path.resolve()
    if ".local" not in root.parts:
        raise EvalContractError("artifact_dir_not_ignored")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if root.is_dir():
        root.chmod(0o700)
    return root


def write_segment_artifacts(
    manifest: EvalRunManifest,
    projection: EvalProjection,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> tuple[Path, Path]:
    """Persist one segment outside the tracked tree with owner-only permissions."""

    root = private_runs_root(runs_dir, create=True)
    directory = root / manifest.run_id
    if directory.exists():
        raise EvalContractError("segment_already_exists")
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    written: list[Path] = []
    for name, model in (
        (MANIFEST_FILE_NAME, manifest),
        (PROJECTION_FILE_NAME, projection),
    ):
        path = directory / name
        payload = json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        path.write_bytes(payload)
        path.chmod(0o600)
        written.append(path)
    return written[0], written[1]


def find_existing_segment(
    *,
    batch_id: str,
    topology: str,
    repeat_seed: int,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> Path | None:
    """Locate a previously written segment so a long batch can resume safely."""

    root = private_runs_root(runs_dir)
    if not root.is_dir():
        return None
    pattern = re.compile(
        rf"^eval-{re.escape(topology)}-{re.escape(batch_id)}-{repeat_seed}-[0-9a-f]{{8}}$"
    )
    matches = sorted(
        path.name for path in root.iterdir() if pattern.fullmatch(path.name)
    )
    if len(matches) > 1:
        raise EvalContractError("duplicate_segment_runs")
    return root / matches[0] / MANIFEST_FILE_NAME if matches else None


def load_eval_manifest(path: Path) -> EvalRunManifest:
    return EvalRunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def summarize_segment(manifest: EvalRunManifest) -> str:
    """A bounded summary containing no request or response content."""

    failure = f" failure_code={manifest.failure_code}" if manifest.failure_code else ""
    return (
        f"run_id={manifest.run_id} batch={manifest.batch_id} "
        f"topology={manifest.topology} seed={manifest.repeat_seed} "
        f"mode={manifest.execution_mode} outcome={manifest.outcome}{failure} "
        f"branches={manifest.branch_count} calls={manifest.model_call_count} "
        f"skipped={manifest.skipped_empty_context_count} "
        f"succeeded={manifest.succeeded_count} failed={manifest.failed_count} "
        f"retries={manifest.retry_total} "
        f"tokens=in:{manifest.usage.input_tokens} "
        f"out:{manifest.usage.output_tokens} total:{manifest.usage.total_tokens} "
        f"budget_warning={manifest.budget.warning_reached} "
        f"latency_p50={manifest.latency.p50_ms} p95={manifest.latency.p95_ms} "
        f"max={manifest.latency.max_ms} "
        f"http={dict(sorted(manifest.http_status_counts.items()))} "
        f"errors={dict(sorted(manifest.error_code_counts.items()))} "
        f"error_reasons={dict(sorted(manifest.error_reason_counts.items()))} "
        f"resolved_model_ids={list(manifest.resolved_model_ids)}"
    )


def load_segment_paths(
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
) -> list[tuple[Path, Path]]:
    """Return ``(manifest, projection)`` pairs for every persisted segment."""

    root = private_runs_root(Path(runs_dir))
    if not root.is_dir():
        return []
    segments: list[tuple[Path, Path]] = []
    for directory in sorted(root.iterdir()):
        manifest = directory / MANIFEST_FILE_NAME
        projection = directory / PROJECTION_FILE_NAME
        if not directory.is_dir() or not manifest.is_file():
            continue
        if not manifest.name.startswith("eval-") or not projection.is_file():
            continue
        segments.append((manifest, projection))
    return segments


def load_projection(path: Path) -> EvalProjection:
    return EvalProjection.model_validate_json(path.read_text(encoding="utf-8"))


def load_batch_segments(
    batch_id: str,
    *,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
) -> list[tuple[EvalRunManifest, EvalProjection]]:
    """Load every segment of one batch so a report can be rebuilt from disk."""

    segments: list[tuple[EvalRunManifest, EvalProjection]] = []
    for manifest_path, projection_path in load_segment_paths(runs_dir):
        manifest = load_eval_manifest(manifest_path)
        if manifest.batch_id != batch_id:
            continue
        segments.append((manifest, load_projection(projection_path)))
    return sorted(segments, key=lambda item: (item[0].topology, item[0].repeat_seed))
