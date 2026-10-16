"""Frozen-batch Eval Runner for the paired single-agent / multi-agent comparisons.

Both topologies read the same frozen Gold allowlist context, use the same adapter,
sampling, timeout, retry and deterministic aggregation; the only independent
variable is how many model calls produce the findings. Every segment writes one
closed ``eval-run-manifest.v1`` plus its scoring projection even when it fails
closed, and a real run needs ``EVAL_REAL_AUTHORIZATION`` to name the batch.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from .chat_adapter import (
    CHAT_ADAPTER_VERSION,
    required_findings_for,
    CHAT_FINDING_CONTRACT_VERSION,
    ChatAdapterConfig,
    ChatAdapterError,
    ChatCallRecord,
    OpenAICompatibleChatAdapter,
)
from .contracts import BusinessFact, Finding, WorkerKind
from .eval_contract import (
    BUDGET_WARNING_RATIO,
    DEFAULT_RUNS_DIR,
    DRY_RUN_CREDENTIAL,
    DRY_RUN_MODEL_ALIAS,
    DRY_RUN_MODEL_ID_LOCK,
    DRY_RUN_RESOLVED_MODEL_ID,
    EVAL_BATCHES,
    EVAL_BUDGET_VERSION,
    EVAL_MAX_OUTPUT_TOKENS_PER_REQUEST,
    EVAL_MAX_RETRIES,
    EVAL_RETRIEVAL_MODE,
    EVAL_RUNNER_VERSION,
    EVAL_TEMPERATURE,
    EVAL_TIMEOUT_SECONDS,
    EVAL_TOP_P,
    STAGE_WORK_ITEM_IDS,
    EvalBatchSpec,
    EvalBranchEntry,
    EvalContractError,
    EvalProjection,
    EvalRunManifest,
    EvalSourceVersions,
    EvalSplitName,
    ProjectedBranch,
    ProjectedFinding,
    ProjectedRef,
    build_projection,
    derived_eval_fields,
    expected_capabilities,
    find_existing_segment,
    finalize_eval_manifest,
    gold_corpus_contract,
    load_eval_manifest,
    summarize_segment,
    write_segment_artifacts,
)
from .gold_set import (
    GOLD_DATASET_VERSION,
    GOLD_POLICY_VERSION,
    GoldCase,
    GoldSplit,
    load_gold_cases,
    load_gold_manifest,
    manifest_checksum,
)
from .rules import RULES_VERSION
from .smoke_run import model_request_for
from .worker_contracts import SingleAgentModelRequest, WorkerModelRequest

Topology = Literal["single_agent", "multi_agent"]
ExecutionMode = Literal["dry_run", "real_endpoint"]
REAL_AUTHORIZATION_ENV = "EVAL_REAL_AUTHORIZATION"
WORKERS: tuple[WorkerKind, ...] = tuple(WorkerKind)
# The only registered stages whose batch may read the hidden split. Adding a member here
# is a scope decision for the user to confirm, never something a run can infer.
_HOLDOUT_STAGES: frozenset[str] = frozenset({"V1-05B"})

# Static strings that would name a Gold decision. Checked against the outbound
# payload without ever loading a label, so the evaluator stays independent.
_FORBIDDEN_OUTBOUND_NEEDLES = (
    "acceptable_alternatives",
    "case_family_id",
    "expected_conflicts",
    "expected_missing_information",
    "expected_recommendation",
    "expected_risk_level",
    "gold_label",
    "label_checksum",
    "must_fail_closed",
    "rule_ids",
    "scenario_tags",
    "adjudication_status",
)


@dataclass(frozen=True, slots=True)
class EvalTask:
    """One model-call slot: the case, the worker it feeds, and the built request."""

    case: GoldCase
    worker: WorkerKind | None
    request: WorkerModelRequest | SingleAgentModelRequest


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    """One branch: what the manifest records and what the scorer replays."""

    entry: EvalBranchEntry
    branch: ProjectedBranch
    void_reason: str | None = None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def project_findings(findings: Sequence[Finding]) -> tuple[ProjectedFinding, ...]:
    """Turn validated Findings into the identity-only scoring projection."""

    return tuple(
        ProjectedFinding(
            worker=finding.worker,
            finding_key=finding.finding_key,
            category=finding.category,
            risk_level=finding.risk_level,
            assertion=finding.assertion,
            status=finding.status,
            evidence_refs=tuple(
                ProjectedRef(
                    source_id=reference.source_id,
                    version=reference.version,
                    content_checksum=reference.content_checksum,
                )
                for reference in finding.evidence_refs
            ),
            business_fact_refs=tuple(
                ProjectedRef(
                    source_id=fact.record_id,
                    version=fact.dataset_version,
                    content_checksum=None,
                )
                for fact in finding.business_fact_refs
            ),
            missing_information=tuple(
                item[:160] for item in finding.missing_information[:12]
            ),
            statement_hash=_hash_text(finding.statement),
            recommended_action_hash=_hash_text(finding.recommended_action),
        )
        for finding in findings
    )


def single_agent_request_for(case: GoldCase) -> SingleAgentModelRequest | None:
    """Build the union context; it carries no evidence the workers would not see."""

    document_refs = {
        ref_id
        for entry in case.allowed_context
        if entry.worker in {WorkerKind.POLICY, WorkerKind.CONTRACT}
        for ref_id in entry.allowed_ref_ids
    }
    contexts = tuple(
        context
        for context in case.model_input.evidence_contexts
        if context.evidence_ref.source_id in document_refs
    )
    facts: tuple[BusinessFact, ...] = tuple(
        fact
        for fact in case.model_input.business_facts
        if fact.record_id
        in {
            ref_id
            for entry in case.allowed_context
            if entry.worker is WorkerKind.PROCUREMENT_DATA
            for ref_id in entry.allowed_ref_ids
        }
    )
    if not contexts and not facts:
        return None
    return SingleAgentModelRequest(
        case_id=case.case_id,
        model_fixture_id=CHAT_FINDING_CONTRACT_VERSION,
        question=case.question,
        workers=WORKERS,
        evidence_contexts=contexts,
        business_facts=facts,
    )


def assert_no_gold_label_leak(payload: Mapping[str, Any]) -> None:
    """Refuse to send anything that names a Gold decision to the model."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    hit = next(
        (needle for needle in _FORBIDDEN_OUTBOUND_NEEDLES if needle in encoded), None
    )
    if hit is not None:
        raise EvalContractError("gold_label_leak_detected")


def select_batch_cases(
    cases: Sequence[GoldCase], spec: EvalBatchSpec
) -> tuple[GoldCase, ...]:
    """Take the first ``cases_per_family`` cases of each family inside the frozen scope."""

    by_family: dict[str, list[GoldCase]] = {}
    for case in cases:
        if case.split not in spec.splits:
            continue
        by_family.setdefault(case.case_family_id, []).append(case)
    if len(by_family) != spec.family_count:
        raise EvalContractError("batch_family_count")
    selected: list[GoldCase] = []
    for family in sorted(by_family):
        pool = by_family[family]
        if len(pool) < spec.cases_per_family:
            raise EvalContractError("batch_family_size")
        selected += pool[: spec.cases_per_family]
    if len(selected) != spec.case_count:
        raise EvalContractError("batch_case_count")
    if any(case.split is GoldSplit.HOLDOUT for case in selected):
        # The hidden split stays sealed for every batch but the one pre-registered
        # holdout judgement run, whose unseal the user confirmed as one-time and
        # irreversible (V1-05 Spec §6.2).
        if spec.stage_id not in _HOLDOUT_STAGES:
            raise EvalContractError("batch_scope_violation")
    return tuple(selected)


def build_tasks(
    batch: Sequence[GoldCase], *, topology: Topology
) -> tuple[list[EvalTask], list[EvalBranchEntry]]:
    """Fan a batch out into model-call tasks plus the branches that never call."""

    tasks: list[EvalTask] = []
    skipped: list[EvalBranchEntry] = []
    for case in batch:
        if topology == "single_agent":
            request = single_agent_request_for(case)
            if request is None:
                skipped.append(_skipped_entry(case, None))
                continue
            _add_task(tasks, EvalTask(case=case, worker=None, request=request))
            continue
        for worker in WORKERS:
            worker_request = model_request_for(case, worker)
            if worker_request is None:
                skipped.append(_skipped_entry(case, worker))
                continue
            _add_task(tasks, EvalTask(case=case, worker=worker, request=worker_request))
    return tasks, skipped


def _add_task(tasks: list[EvalTask], task: EvalTask) -> None:
    assert_no_gold_label_leak(task.request.model_dump(mode="json"))
    tasks.append(task)


def _skipped_entry(case: GoldCase, worker: WorkerKind | None) -> EvalBranchEntry:
    return EvalBranchEntry(
        case_id=case.case_id,
        split=_split_name(case),
        worker=worker,
        disposition="skipped_empty_context",
        outcome="not_requested",
    )


def _split_name(case: GoldCase) -> EvalSplitName:
    if case.split is GoldSplit.DEV:
        return "dev"
    if case.split is GoldSplit.VALIDATION:
        return "validation"
    if case.split is GoldSplit.HOLDOUT:
        return "holdout"
    raise EvalContractError("batch_scope_violation")


def _failed_projection(
    case: GoldCase, worker: WorkerKind | None, entry: EvalBranchEntry
) -> ProjectedBranch:
    return ProjectedBranch(
        case_id=case.case_id,
        split=_split_name(case),
        worker=worker,
        outcome="failed",
        error_code=entry.error_code,
    )


def _entry_from_record(
    record: ChatCallRecord,
    *,
    case: GoldCase,
    worker: WorkerKind | None,
    findings: Sequence[Finding] | None,
    unexpected_error: str | None,
) -> EvalBranchEntry:
    if unexpected_error is not None:
        return EvalBranchEntry(
            case_id=case.case_id,
            split=_split_name(case),
            worker=worker,
            disposition="model_call",
            outcome="failed",
            error_code="unexpected_branch_failure",
            error_detail=unexpected_error[:160],
        )
    return EvalBranchEntry(
        case_id=case.case_id,
        split=_split_name(case),
        worker=worker,
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
        finding_count=len(findings or ()),
        error_code=record.error_code,
        error_detail=record.error_detail,
        error_reason=record.error_reason,
    )


async def _run_task(
    adapter: OpenAICompatibleChatAdapter,
    task: EvalTask,
    *,
    model_id_lock: str,
) -> BranchOutcome:
    findings: Sequence[Finding] | None = None
    error: str | None = None
    try:
        produced = await adapter.generate(task.request)
        if isinstance(produced, tuple):
            findings = produced
    except ChatAdapterError:
        # The adapter always emits a record for this path; it is read below.
        pass
    except Exception as exc:  # noqa: BLE001 - an unhandled error must still be audited
        error = type(exc).__name__
    if not adapter.records:
        entry = EvalBranchEntry(
            case_id=task.case.case_id,
            split=_split_name(task.case),
            worker=task.worker,
            disposition="model_call",
            outcome="failed",
            error_code="missing_call_record",
            error_detail="adapter_returned_no_record",
        )
        return BranchOutcome(
            entry=entry,
            branch=_failed_projection(task.case, task.worker, entry),
            void_reason="record_incomplete",
        )
    entry = _entry_from_record(
        adapter.records[-1],
        case=task.case,
        worker=task.worker,
        findings=findings,
        unexpected_error=error,
    )
    if entry.resolved_model_id and entry.resolved_model_id != model_id_lock:
        entry = entry.model_copy(
            update={
                "outcome": "failed",
                "error_code": "model_id_drift",
                "error_detail": "resolved_model_id_outside_lock",
            }
        )
        return BranchOutcome(
            entry=entry,
            branch=_failed_projection(task.case, task.worker, entry),
            void_reason="model_drift",
        )
    if findings is None or entry.outcome != "succeeded":
        return BranchOutcome(
            entry=entry, branch=_failed_projection(task.case, task.worker, entry)
        )
    return BranchOutcome(
        entry=entry,
        branch=ProjectedBranch(
            case_id=task.case.case_id,
            split=_split_name(task.case),
            worker=task.worker,
            outcome="succeeded",
            findings=project_findings(findings),
        ),
    )


def _consumed(entries: Sequence[EvalBranchEntry]) -> tuple[int, int, int]:
    calls = [entry for entry in entries if entry.disposition == "model_call"]
    return (
        len(calls),
        sum(entry.input_tokens or 0 for entry in entries),
        sum(entry.output_tokens or 0 for entry in entries),
    )


def _budget_breached(entries: Sequence[EvalBranchEntry], spec: EvalBatchSpec) -> bool:
    requests, input_tokens, output_tokens = _consumed(entries)
    return (
        requests > spec.max_requests
        or input_tokens > spec.max_input_tokens
        or output_tokens > spec.max_output_tokens
    )


def _budget_warning(entries: Sequence[EvalBranchEntry], spec: EvalBatchSpec) -> bool:
    requests, input_tokens, output_tokens = _consumed(entries)
    return (
        requests >= spec.max_requests * BUDGET_WARNING_RATIO
        or input_tokens >= spec.max_input_tokens * BUDGET_WARNING_RATIO
        or output_tokens >= spec.max_output_tokens * BUDGET_WARNING_RATIO
    )


def _new_adapter(
    *,
    execution_mode: ExecutionMode,
    seed: int,
    real_config: ChatAdapterConfig | None,
    transport: httpx.AsyncBaseTransport | None,
    sleep: Callable[[float], Any],
    clock: Callable[[], float],
) -> OpenAICompatibleChatAdapter:
    if execution_mode == "dry_run":
        config = ChatAdapterConfig(
            base_url="https://dry-run.invalid/api",
            model_name=DRY_RUN_MODEL_ALIAS,
            api_key=SecretStr(DRY_RUN_CREDENTIAL),
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            max_output_tokens=EVAL_MAX_OUTPUT_TOKENS_PER_REQUEST,
            seed=seed,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
            max_retries=EVAL_MAX_RETRIES,
            include_required_findings=True,
        )
        return OpenAICompatibleChatAdapter(
            config,
            transport=transport or _dry_run_transport(),
            sleep=sleep,
            clock=clock,
        )
    if real_config is None:
        raise EvalContractError("real_configuration_missing")
    return OpenAICompatibleChatAdapter(
        replace(
            real_config,
            seed=seed,
            temperature=EVAL_TEMPERATURE,
            top_p=EVAL_TOP_P,
            max_output_tokens=EVAL_MAX_OUTPUT_TOKENS_PER_REQUEST,
            timeout_seconds=EVAL_TIMEOUT_SECONDS,
            max_retries=EVAL_MAX_RETRIES,
            include_required_findings=True,
        ),
        transport=transport,
        sleep=sleep,
        clock=clock,
    )


def _finding_for(
    vocabulary: Mapping[str, str], context: Mapping[str, Any]
) -> dict[str, Any]:
    """A schema-valid Finding for one required contract; unresolved with nothing to cite."""

    worker = vocabulary["worker"]
    key = vocabulary["finding_key"]
    category = vocabulary["category"]
    if worker == WorkerKind.PROCUREMENT_DATA.value:
        refs = [fact["record_id"] for fact in context.get("business_facts", ())][:1]
        citation: dict[str, Any] = {"business_fact_refs": refs}
    else:
        refs = [item["ref_id"] for item in context.get("evidence_contexts", ())][:1]
        citation = {"evidence_refs": refs}
    if not refs:
        return {
            "schema_version": "1.0",
            "worker": worker,
            "finding_key": key,
            "category": category,
            "risk_level": "unknown",
            "assertion": "unknown",
            "statement": "The offline synthetic context is too thin to decide.",
            "status": "insufficient_evidence",
            "missing_information": (f"offline_transport_no_context:{worker}",),
            "recommended_action": "Require human review.",
        }
    return {
        "schema_version": "1.0",
        "worker": worker,
        "finding_key": key,
        "category": category,
        "risk_level": "low",
        "assertion": "risk_absent",
        "statement": "The synthetic context supports the required finding.",
        "status": "supported",
        "recommended_action": "Continue with deterministic review.",
        **citation,
    }


def _offline_vocabulary(context: Mapping[str, Any]) -> list[Mapping[str, str]]:
    """Read back the required-finding vocabulary the adapter just sent out."""

    entries = list(context.get("required_findings") or [])
    if entries:
        return entries
    workers = (
        context["workers"]
        if context.get("agent_topology") == "single_agent"
        else [context["worker"]]
    )
    return required_findings_for([WorkerKind(worker) for worker in workers])


def _scoped_context(
    vocabulary: Mapping[str, str], context: Mapping[str, Any]
) -> dict[str, Any]:
    """Limit one offline worker to the same items its own branch would have seen."""

    worker = vocabulary["worker"]
    if worker == WorkerKind.PROCUREMENT_DATA.value:
        return {"business_facts": context.get("business_facts", ())}
    return {
        "evidence_contexts": [
            item
            for item in context.get("evidence_contexts", ())
            if item["ref_id"].startswith("DOC-GOLD-POLICY")
            is (worker == WorkerKind.POLICY.value)
        ]
    }


def _dry_run_transport() -> httpx.AsyncBaseTransport:
    """Offline transport that answers both topologies with valid Findings."""

    counter = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal counter
        counter += 1
        body = json.loads(request.content)
        context = json.loads(body["messages"][1]["content"])
        findings = [
            _finding_for(vocabulary, _scoped_context(vocabulary, context))
            for vocabulary in _offline_vocabulary(context)
        ]
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
                                {"findings": findings}, ensure_ascii=False
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


async def run_segment(
    *,
    spec: EvalBatchSpec,
    topology: Topology,
    repeat_seed: int,
    execution_mode: ExecutionMode,
    batch: Sequence[GoldCase] | None = None,
    real_config: ChatAdapterConfig | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Any] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[EvalRunManifest, EvalProjection]:
    """Execute one ``(topology, seed)`` segment; the audit record survives failure."""

    started_at = now()
    if repeat_seed not in spec.repeat_seeds:
        raise EvalContractError("seed_outside_batch")
    if execution_mode == "real_endpoint" and real_config is None:
        raise EvalContractError("real_configuration_missing")
    if EVAL_BATCHES.get(spec.batch_id) != spec:
        raise EvalContractError("batch_spec_not_frozen")
    credential = (
        DRY_RUN_CREDENTIAL
        if execution_mode == "dry_run"
        else real_config.api_key.get_secret_value()
    )
    corpus = gold_corpus_contract(spec)
    requested_model_id = (
        DRY_RUN_MODEL_ALIAS if execution_mode == "dry_run" else real_config.model_name
    )
    model_id_lock = (
        DRY_RUN_MODEL_ID_LOCK if execution_mode == "dry_run" else spec.model_id_lock
    )
    selected = (
        select_batch_cases(load_gold_cases(corpus_version=corpus.version).cases, spec)
        if batch is None
        else tuple(batch)
    )
    tasks, skipped = build_tasks(selected, topology=topology)
    entries: list[EvalBranchEntry] = list(skipped)
    branches: list[ProjectedBranch] = []
    void_reason: str | None = None
    aborted: str | None = None
    cursor = 0

    def _pull() -> EvalTask | None:
        nonlocal cursor, aborted
        if aborted is not None or cursor >= len(tasks):
            return None
        if _budget_breached(entries, spec):
            aborted = "budget_exceeded"
            return None
        task = tasks[cursor]
        cursor += 1
        return task

    slot_count = max(1, min(spec.concurrency, len(tasks) or 1))
    adapters = [
        _new_adapter(
            execution_mode=execution_mode,
            seed=repeat_seed,
            real_config=real_config,
            transport=transport,
            sleep=sleep,
            clock=clock,
        )
        for _ in range(slot_count)
    ]

    warned = False

    async def _slot(adapter: OpenAICompatibleChatAdapter) -> None:
        nonlocal void_reason, aborted, warned
        while True:
            task = _pull()
            if task is None:
                return
            outcome = await _run_task(adapter, task, model_id_lock=model_id_lock)
            entries.append(outcome.entry)
            branches.append(outcome.branch)
            if outcome.void_reason and void_reason is None:
                void_reason = outcome.void_reason
                aborted = outcome.void_reason
            if not warned and _budget_warning(entries, spec):
                warned = True
                print(
                    f"eval-warning: batch={spec.batch_id} topology={topology} "
                    f"seed={repeat_seed} reached {BUDGET_WARNING_RATIO:.0%} of the "
                    "declared local budget",
                    file=sys.stderr,
                )

    try:
        await asyncio.gather(*(_slot(adapter) for adapter in adapters))
    finally:
        await asyncio.gather(*(adapter.aclose() for adapter in adapters))

    finished_at = now()
    calls = tuple(entries)
    derived = derived_eval_fields(calls, spec=spec)
    run_id = _run_id(topology=topology, spec=spec, seed=repeat_seed, at=started_at)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "batch_id": spec.batch_id,
        "work_item_id": STAGE_WORK_ITEM_IDS[spec.stage_id],
        "stage_id": spec.stage_id,
        "topology": topology,
        "repeat_seed": repeat_seed,
        "execution_mode": execution_mode,
        "case_split_scope": "+".join(split.value for split in spec.splits),
        "case_count": spec.case_count,
        "cases_per_family": spec.cases_per_family,
        "concurrency": spec.concurrency,
        "model_id_lock": model_id_lock,
        "required_findings_included": True,
        "requested_model_id": requested_model_id,
        "capabilities": expected_capabilities(),
        "source_versions": EvalSourceVersions(
            adapter=CHAT_ADAPTER_VERSION,
            finding_contract=CHAT_FINDING_CONTRACT_VERSION,
            gold_set=corpus.gold_set_version,
            gold_generator=corpus.generator_version,
            corpus_snapshot=corpus.corpus_snapshot_id,
            dataset=GOLD_DATASET_VERSION,
            policy=GOLD_POLICY_VERSION,
            rules=RULES_VERSION,
            retrieval=EVAL_RETRIEVAL_MODE,
            budget=EVAL_BUDGET_VERSION,
            runner=EVAL_RUNNER_VERSION,
            code=_git_revision(),
        ),
        "gold_manifest_checksum": manifest_checksum(
            load_gold_manifest(corpus_version=corpus.version)
        ),
        "started_at": started_at,
        "finished_at": finished_at,
        "calls": calls,
        "voided": void_reason is not None,
        **derived,
    }
    failure_code = aborted or ("model_drift" if void_reason else None)
    if failure_code is None and (derived["failed_count"] or derived["budget"].exceeded):
        failure_code = (
            "budget_exceeded" if derived["budget"].exceeded else "model_call_failed"
        )
    payload["outcome"] = "completed" if failure_code is None else "failed_closed"
    payload["failure_code"] = failure_code
    manifest = finalize_eval_manifest(
        EvalRunManifest.model_validate(payload), credential=credential, spec=spec
    )
    projection = build_projection(
        run_id=run_id,
        spec=spec,
        topology=topology,
        repeat_seed=repeat_seed,
        execution_mode=execution_mode,
        branches=branches,
    )
    return manifest, projection


def _run_id(*, topology: str, spec: EvalBatchSpec, seed: int, at: datetime) -> str:
    salt = hashlib.sha256(
        f"{topology}-{spec.batch_id}-{seed}-{at.isoformat()}-{os.getpid()}-"
        f"{time.monotonic()}".encode()
    ).hexdigest()
    return f"eval-{topology}-{spec.batch_id}-{seed}-{salt[:8]}"


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


async def run_batch(
    *,
    batch_id: str,
    execution_mode: ExecutionMode,
    topologies: Sequence[Topology] = ("multi_agent", "single_agent"),
    real_config: ChatAdapterConfig | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    skip_existing: bool = True,
) -> list[tuple[Path, Path]]:
    """Run every requested segment of a frozen batch and persist its artifacts."""

    spec = EVAL_BATCHES.get(batch_id)
    if spec is None:
        raise EvalContractError("batch_not_registered")
    batch = select_batch_cases(
        load_gold_cases(corpus_version=gold_corpus_contract(spec).version).cases, spec
    )
    written: list[tuple[Path, Path]] = []
    for topology in topologies:
        for seed in spec.repeat_seeds:
            existing = (
                find_existing_segment(
                    batch_id=batch_id,
                    topology=topology,
                    repeat_seed=seed,
                    runs_dir=runs_dir,
                )
                if skip_existing
                else None
            )
            if existing is not None:
                continue
            manifest, projection = await run_segment(
                spec=spec,
                topology=topology,
                repeat_seed=seed,
                execution_mode=execution_mode,
                batch=batch,
                real_config=real_config,
            )
            written.append(
                write_segment_artifacts(manifest, projection, runs_dir=runs_dir)
            )
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen V1-02C paired evaluation batch with run manifests."
    )
    parser.add_argument("--batch", required=True, choices=sorted(EVAL_BATCHES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="offline mock transport")
    mode.add_argument(
        "--real",
        action="store_true",
        help=f"requires {REAL_AUTHORIZATION_ENV} to equal the batch id",
    )
    parser.add_argument(
        "--topology",
        choices=["both", *sorted({"single_agent", "multi_agent"})],
        default="both",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip segments whose manifest already exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    execution_mode: ExecutionMode = "real_endpoint" if args.real else "dry_run"
    topologies: Sequence[Topology] = (
        ("multi_agent", "single_agent") if args.topology == "both" else (args.topology,)
    )
    try:
        if execution_mode == "real_endpoint" and (
            os.environ.get(REAL_AUTHORIZATION_ENV, "") != args.batch
        ):
            raise EvalContractError("run_not_authorized")
        config = None if args.dry_run else ChatAdapterConfig.from_env()
        segments = asyncio.run(
            run_batch(
                batch_id=args.batch,
                execution_mode=execution_mode,
                topologies=topologies,
                real_config=config,
                runs_dir=args.runs_dir,
                skip_existing=args.resume,
            )
        )
    except (EvalContractError, OSError, ValueError) as exc:
        code = getattr(exc, "code", None) or type(exc).__name__
        print(f"eval-error: {code}", file=sys.stderr)
        return 1
    exit_code = 0
    for manifest_path, _ in segments:
        manifest = load_eval_manifest(manifest_path)
        print(f"wrote {manifest_path}")
        print(summarize_segment(manifest))
        if manifest.outcome != "completed" or manifest.voided:
            exit_code = 1
    if not segments:
        print("eval: no segments to run (all already written)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
