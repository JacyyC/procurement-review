"""Offline tests for the V1-02C paired Eval Runner.

Everything here runs against a mock transport with a placeholder credential, so a
real endpoint is never contacted. The fairness assertions are the point of the file:
the two topologies must differ only in how many calls they make.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from procurement_review.chat_adapter import ChatAdapterConfig
from procurement_review.contracts import (
    BusinessFact,
    EvidenceRef,
    FindingAssertion,
    FindingStatus,
    RiskCategory,
    RiskLevel,
    WorkerKind,
)
from procurement_review.eval_contract import (
    CALIBRATION_BATCH,
    HOLDOUT_REJUDGEMENT_BATCH,
    ProjectedBranch,
    ProjectedFinding,
    ProjectedRef,
    build_projection,
    EVAL_BATCHES,
    DRY_RUN_CREDENTIAL,
    DRY_RUN_MODEL_ID_LOCK,
    PAIRED_BATCH,
    RESOLUTION_BATCH,
    EvalContractError,
    EvalProjection,
    EvalRunManifest,
    derived_eval_fields,
)
from procurement_review.chat_adapter import required_findings_for
from procurement_review.eval_metrics import observations_from_projection
from procurement_review.eval_run import (
    REAL_AUTHORIZATION_ENV,
    EvalTask,
    WORKERS,
    assert_no_gold_label_leak,
    build_tasks,
    main,
    run_batch,
    run_segment,
    select_batch_cases,
    single_agent_request_for,
)
from procurement_review.gold_set import (
    GoldCase,
    GoldCorpusVersion,
    GoldSplit,
    load_gold_cases,
)

from procurement_review.smoke_run import model_request_for

CASES = load_gold_cases().cases
CASES_V2 = load_gold_cases(corpus_version=GoldCorpusVersion.V2).cases
CASE_INDEX = {case.case_id: case for case in CASES}


async def _no_sleep(_: float) -> None:
    return None


def _transport(
    model_id: str = DRY_RUN_MODEL_ID_LOCK,
    *,
    status: int = 200,
    sink: list[dict] | None = None,
) -> httpx.AsyncBaseTransport:
    """A mock transport that answers with valid envelopes for both topologies."""

    counter = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal counter
        counter += 1
        if sink is not None:
            sink.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, json={"error": {"type": "ServerOverloaded"}})
        body = json.loads(request.content)
        context = json.loads(body["messages"][1]["content"])
        workers = (
            context["workers"]
            if context.get("agent_topology") == "single_agent"
            else [context["worker"]]
        )
        findings = [
            _finding(worker, context) for worker in workers if _refs(worker, context)
        ]
        if not findings:
            findings = [_unresolved(workers[0])]
        return httpx.Response(
            200,
            json={
                "id": f"probe-{counter:04d}",
                "model": model_id,
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
                    "prompt_tokens": 30,
                    "completion_tokens": 12,
                    "total_tokens": 42,
                },
            },
        )

    return httpx.MockTransport(handler)


def _refs(worker: str, context: dict) -> list[str]:
    if worker == WorkerKind.PROCUREMENT_DATA.value:
        return [fact["record_id"] for fact in context.get("business_facts", ())][:1]
    return [
        item["ref_id"]
        for item in context.get("evidence_contexts", ())
        if item["ref_id"].startswith("DOC-GOLD-POLICY")
        is (worker == WorkerKind.POLICY.value)
    ][:1]


def _finding(worker: str, context: dict) -> dict:
    vocabulary = next(
        item for item in context["required_findings"] if item["worker"] == worker
    )
    return {
        "schema_version": "1.0",
        "worker": worker,
        "finding_key": vocabulary["finding_key"],
        "category": vocabulary["category"],
        "risk_level": "low",
        "assertion": "risk_absent",
        "status": "supported",
        "statement": "The synthetic context supports the required finding.",
        "recommended_action": "Continue with deterministic review.",
        **(
            {"business_fact_refs": _refs(worker, context)}
            if worker == WorkerKind.PROCUREMENT_DATA.value
            else {"evidence_refs": _refs(worker, context)}
        ),
    }


def _unresolved(worker: str) -> dict:
    return {
        "schema_version": "1.0",
        "worker": worker,
        "finding_key": f"{worker}.required",
        "category": "policy_compliance",
        "risk_level": "unknown",
        "assertion": "unknown",
        "status": "insufficient_evidence",
        "statement": "Nothing to cite.",
        "missing_information": ("probe_no_context",),
        "recommended_action": "Require human review.",
    }


def _run(coroutine: Awaitable[object]) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_calibration_batch_is_the_same_48_dev_cases_as_the_frozen_smoke() -> None:
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)
    assert len(batch) == 48
    assert {case.split for case in batch} == {GoldSplit.DEV}
    per_family = {
        family: sum(1 for case in batch if case.case_family_id == family)
        for family in {case.case_family_id for case in batch}
    }
    assert len(per_family) == 12 and set(per_family.values()) == {4}


def test_paired_batch_covers_dev_and_validation_and_never_holdout() -> None:
    batch = select_batch_cases(CASES, PAIRED_BATCH)
    assert len(batch) == 360
    counts = {
        split: sum(1 for case in batch if case.split is split)
        for split in (GoldSplit.DEV, GoldSplit.VALIDATION, GoldSplit.HOLDOUT)
    }
    assert counts == {
        GoldSplit.DEV: 240,
        GoldSplit.VALIDATION: 120,
        GoldSplit.HOLDOUT: 0,
    }


def test_resolution_batch_takes_the_first_ten_dev_cases_of_every_family() -> None:
    batch = select_batch_cases(CASES_V2, RESOLUTION_BATCH)
    assert len(batch) == 120
    assert {case.split for case in batch} == {GoldSplit.DEV}
    per_family = {
        family: sum(1 for case in batch if case.case_family_id == family)
        for family in {case.case_family_id for case in batch}
    }
    assert len(per_family) == 12 and set(per_family.values()) == {10}
    assert all(case.corpus_snapshot_id == "synthetic-gold-corpus.v2" for case in batch)
    dev = [case for case in CASES_V2 if case.split is GoldSplit.DEV]
    expected = tuple(
        case
        for family in sorted({case.case_family_id for case in dev})
        for case in [item for item in dev if item.case_family_id == family][:10]
    )
    assert batch == expected


def test_the_holdout_batch_selects_the_hidden_split_for_one_stage_only() -> None:
    batch = select_batch_cases(CASES_V2, HOLDOUT_REJUDGEMENT_BATCH)
    assert len(batch) == 120
    assert {case.split for case in batch} == {GoldSplit.HOLDOUT}
    per_family = {
        family: sum(1 for case in batch if case.case_family_id == family)
        for family in {case.case_family_id for case in batch}
    }
    assert len(per_family) == 12 and set(per_family.values()) == {10}
    assert all(case.corpus_snapshot_id == "synthetic-gold-corpus.v2" for case in batch)
    # The same split scope under any other registered stage stays sealed.
    sealed = HOLDOUT_REJUDGEMENT_BATCH.model_copy(update={"stage_id": "V1-02C"})
    with pytest.raises(EvalContractError, match="batch_scope_violation"):
        select_batch_cases(CASES_V2, sealed)


def test_a_dry_run_holdout_segment_records_every_branch_as_holdout() -> None:
    batch = select_batch_cases(CASES_V2, HOLDOUT_REJUDGEMENT_BATCH)[:8]

    async def scenario() -> tuple[EvalRunManifest, EvalProjection]:
        return await run_segment(
            spec=HOLDOUT_REJUDGEMENT_BATCH,
            topology="multi_agent",
            repeat_seed=20260901,
            execution_mode="dry_run",
            batch=batch,
            sleep=_no_sleep,
        )

    manifest, projection = _run(scenario())
    assert manifest.outcome == "completed"
    assert manifest.work_item_id == "V1-05"
    assert manifest.stage_id == "V1-05B"
    assert manifest.case_split_scope == "holdout"
    assert {entry.split for entry in manifest.calls} == {"holdout"}
    assert {branch.split for branch in projection.branches} == {"holdout"}
    _, skips = build_tasks(batch, topology="multi_agent")
    assert manifest.model_call_count == len(batch) * 3 - len(skips)


def test_both_topologies_see_the_same_clause_level_context_on_v2() -> None:
    batch = select_batch_cases(CASES_V2, RESOLUTION_BATCH)
    multi_tasks, multi_skips = build_tasks(batch, topology="multi_agent")
    single_tasks, single_skips = build_tasks(batch, topology="single_agent")
    assert (len(multi_tasks), len(multi_skips)) == (330, 30)
    assert (len(single_tasks), len(single_skips)) == (120, 0)
    for case in batch:
        single = single_agent_request_for(case)
        assert single is not None
        union_evidence = {
            context.evidence_ref.source_id for context in single.evidence_contexts
        }
        union_facts = {fact.record_id for fact in single.business_facts}
        worker_evidence: set[str] = set()
        worker_facts: set[str] = set()
        for worker in WORKERS:
            request = model_request_for(case, worker)
            if request is None:
                continue
            worker_evidence.update(
                context.evidence_ref.source_id for context in request.evidence_contexts
            )
            worker_facts.update(fact.record_id for fact in request.business_facts)
        assert union_evidence == worker_evidence
        assert union_facts == worker_facts
        assert_no_gold_label_leak(single.model_dump(mode="json"))
    payload = json.dumps(
        [task.request.model_dump(mode="json") for task in multi_tasks],
        ensure_ascii=False,
    )
    for needle in (
        "expected_",
        "rule_ids",
        "must_fail_closed",
        "scenario_tags",
        "split",
    ):
        assert needle not in payload


def test_v2_outbound_documents_are_materially_thicker_than_v1() -> None:
    def median_chars(cases) -> float:
        lengths = sorted(
            len(context.content)
            for case in cases
            for context in case.model_input.evidence_contexts
        )
        return lengths[len(lengths) // 2]

    v1_batch = select_batch_cases(CASES, RESOLUTION_BATCH)
    v2_batch = select_batch_cases(CASES_V2, RESOLUTION_BATCH)
    assert median_chars(v2_batch) >= 8 * median_chars(v1_batch)


def test_selection_refuses_a_batch_that_cannot_be_filled_from_scope() -> None:
    trimmed = [case for case in CASES if case.case_family_id != "GF-amount_anomaly"]
    with pytest.raises(EvalContractError, match="batch_family_count"):
        select_batch_cases(trimmed, CALIBRATION_BATCH)
    dev = [case for case in CASES if case.split is GoldSplit.DEV]
    families = sorted({case.case_family_id for case in dev})
    assert len(families) == 12
    thin: list[GoldCase] = []
    for index, family in enumerate(families):
        pool = [case for case in dev if case.case_family_id == family]
        thin += pool[: 2 if index == 0 else 4]
    with pytest.raises(EvalContractError, match="batch_family_size"):
        select_batch_cases(thin, CALIBRATION_BATCH)


def test_task_fan_out_matches_the_frozen_smoke_branch_shape() -> None:
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)
    multi_tasks, multi_skips = build_tasks(batch, topology="multi_agent")
    assert (len(multi_tasks), len(multi_skips)) == (132, 12)
    single_tasks, single_skips = build_tasks(batch, topology="single_agent")
    assert (len(single_tasks), len(single_skips)) == (48, 0)
    assert {task.worker for task in single_tasks} == {None}
    assert {task.worker for task in multi_tasks} == set(WorkerKind)


def test_both_topologies_see_exactly_the_same_context_items() -> None:
    for case in select_batch_cases(CASES, PAIRED_BATCH)[:40]:
        single = single_agent_request_for(case)
        assert single is not None
        union_evidence = {
            context.evidence_ref.source_id for context in single.evidence_contexts
        }
        union_facts = {fact.record_id for fact in single.business_facts}
        worker_evidence: set[str] = set()
        worker_facts: set[str] = set()
        for worker in WORKERS:
            request = model_request_for(case, worker)
            if request is None:
                continue
            worker_evidence.update(
                context.evidence_ref.source_id for context in request.evidence_contexts
            )
            worker_facts.update(fact.record_id for fact in request.business_facts)
        assert union_evidence == worker_evidence
        assert union_facts == worker_facts
        assert single.question == case.question
        assert [entry.worker for entry in case.allowed_context] == list(WORKERS)


def test_outbound_requests_never_carry_labels_tools_or_credentials() -> None:
    captured: list[dict] = []
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)[:6]

    async def scenario() -> None:
        for topology in ("multi_agent", "single_agent"):
            await run_segment(
                spec=CALIBRATION_BATCH,
                topology=topology,
                repeat_seed=CALIBRATION_BATCH.repeat_seeds[0],
                execution_mode="dry_run",
                batch=batch,
                transport=_transport(sink=captured),
                sleep=_no_sleep,
            )

    _run(scenario())
    assert len(captured) == 24
    needles = (
        "expected_",
        "label_id",
        "label_checksum",
        "rule_ids",
        "case_family_id",
        "scenario_tags",
        "group_id",
        "split",
        "holdout",
        "must_fail_closed",
        DRY_RUN_CREDENTIAL,
    )
    for body in captured:
        assert "tools" not in body and "tool_choice" not in body
        assert body["temperature"] == 0.0 and body["top_p"] == 1.0
        assert body["max_tokens"] == 1536
        assert body["seed"] == 20260901
        assert {message["role"] for message in body["messages"]} == {"system", "user"}
        encoded = json.dumps(body, ensure_ascii=False)
        for needle in needles:
            assert needle not in encoded, needle
    vocabularies = {
        len(json.loads(body["messages"][1]["content"])["required_findings"])
        for body in captured
    }
    assert vocabularies == {1, 3}


def test_label_leak_guard_rejects_a_context_naming_a_gold_decision() -> None:
    assert_no_gold_label_leak({"question": "审查合成采购 Case 0001"})
    with pytest.raises(EvalContractError, match="gold_label_leak_detected"):
        assert_no_gold_label_leak({"expected_recommendation": "recommend_approve"})
    case = CASES[0]
    poisoned = case.model_input.model_copy(
        update={
            "evidence_contexts": tuple(
                context.model_copy(
                    update={"content": context.content + " must_fail_closed true"}
                )
                for context in case.model_input.evidence_contexts
            )
        }
    )
    tampered = EvalTask(
        case=case,
        worker=None,
        request=_single_request_with(poisoned),
    )
    with pytest.raises(EvalContractError, match="gold_label_leak_detected"):
        assert_no_gold_label_leak(tampered.request.model_dump(mode="json"))


def _single_request_with(model_input: object) -> object:
    case = CASES[0]
    tampered = case.model_copy(update={"model_input": model_input})
    request = single_agent_request_for(tampered)
    assert request is not None
    return request


def test_dry_run_segments_complete_for_both_topologies_and_agree() -> None:
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)[:8]

    async def scenario() -> tuple[
        EvalRunManifest, EvalProjection, EvalRunManifest, EvalProjection
    ]:
        multi = await run_segment(
            spec=CALIBRATION_BATCH,
            topology="multi_agent",
            repeat_seed=20260901,
            execution_mode="dry_run",
            batch=batch,
            sleep=_no_sleep,
        )
        single = await run_segment(
            spec=CALIBRATION_BATCH,
            topology="single_agent",
            repeat_seed=20260901,
            execution_mode="dry_run",
            batch=batch,
            sleep=_no_sleep,
        )
        return multi[0], multi[1], single[0], single[1]

    multi_manifest, multi_projection, single_manifest, single_projection = _run(
        scenario()
    )
    assert multi_manifest.outcome == "completed"
    assert single_manifest.outcome == "completed"
    _, skips = build_tasks(batch, topology="multi_agent")
    assert multi_manifest.model_call_count == len(batch) * 3 - len(skips)
    assert single_manifest.model_call_count == len(batch)
    index = {case.case_id: case for case in batch}
    multi_rows = observations_from_projection(multi_projection, index)
    single_rows = observations_from_projection(single_projection, index)
    assert {row.case_id: row.recommendation for row in multi_rows} == {
        row.case_id: row.recommendation for row in single_rows
    }


def test_a_different_resolved_model_voids_the_segment_and_stops_consuming() -> None:
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)[:6]

    async def scenario() -> EvalRunManifest:
        manifest, _ = await run_segment(
            spec=CALIBRATION_BATCH,
            topology="multi_agent",
            repeat_seed=20260901,
            execution_mode="real_endpoint",
            batch=batch,
            real_config=_offline_real_config(),
            transport=_transport(model_id="some-other-model-20260901"),
            sleep=_no_sleep,
        )
        return manifest

    manifest = _run(scenario())
    assert isinstance(manifest, EvalRunManifest)
    assert manifest.voided is True
    assert manifest.failure_code == "model_drift"
    assert manifest.error_code_counts == {"model_id_drift": 1}
    assert manifest.model_call_count == 1


def test_the_local_budget_brake_stops_before_the_next_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)[:6]
    tight = CALIBRATION_BATCH.model_copy(update={"max_requests": 2})
    monkeypatch.setitem(EVAL_BATCHES, tight.batch_id, tight)

    async def scenario() -> EvalRunManifest:
        manifest, _ = await run_segment(
            spec=tight,
            topology="multi_agent",
            repeat_seed=20260901,
            execution_mode="dry_run",
            batch=batch,
            sleep=_no_sleep,
        )
        return manifest

    manifest = _run(scenario())
    assert isinstance(manifest, EvalRunManifest)
    assert manifest.failure_code == "budget_exceeded"
    assert manifest.model_call_count <= 3
    assert manifest.budget.consumed_requests == manifest.model_call_count


def test_a_server_error_is_classified_without_losing_the_segment() -> None:
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)[:2]

    async def scenario() -> EvalRunManifest:
        manifest, _ = await run_segment(
            spec=CALIBRATION_BATCH,
            topology="single_agent",
            repeat_seed=20260901,
            execution_mode="real_endpoint",
            batch=batch,
            real_config=_offline_real_config(),
            transport=_transport(status=503),
            sleep=_no_sleep,
        )
        return manifest

    manifest = _run(scenario())
    assert isinstance(manifest, EvalRunManifest)
    assert manifest.outcome == "failed_closed"
    assert manifest.error_code_counts == {"server_error": 2}
    assert manifest.error_reason_counts == {"ServerOverloaded": 2}
    assert manifest.retry_total == 2
    assert manifest.http_status_counts == {"503": 2}


def _offline_real_config() -> ChatAdapterConfig:
    return ChatAdapterConfig(
        base_url="https://real-endpoint.invalid/api",
        model_name="deepseek-v4-flash",
        api_key=SecretStr(DRY_RUN_CREDENTIAL),
    )


def test_a_real_run_requires_the_batch_id_in_the_authorization_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(REAL_AUTHORIZATION_ENV, raising=False)
    assert main(["--batch", CALIBRATION_BATCH.batch_id, "--real"]) == 1
    monkeypatch.setenv(REAL_AUTHORIZATION_ENV, "another-batch")
    assert main(["--batch", CALIBRATION_BATCH.batch_id, "--real"]) == 1


def test_dry_run_cli_writes_one_segment_per_topology_and_seed(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / ".local" / "runs"
    (runs_dir).mkdir(parents=True)
    assert (
        main(
            [
                "--batch",
                CALIBRATION_BATCH.batch_id,
                "--dry-run",
                "--runs-dir",
                str(runs_dir),
            ]
        )
        == 0
    )
    written = sorted(path.name for path in runs_dir.iterdir())
    assert len(written) == 2
    assert any("multi_agent" in name for name in written)
    assert any("single_agent" in name for name in written)
    assert (
        main(
            [
                "--batch",
                CALIBRATION_BATCH.batch_id,
                "--dry-run",
                "--runs-dir",
                str(runs_dir),
                "--resume",
            ]
        )
        == 0
    )


def test_run_batch_skips_segments_that_already_exist(tmp_path: Path) -> None:
    runs_dir = tmp_path / ".local" / "runs"
    runs_dir.mkdir(parents=True)
    first = _run(
        run_batch(
            batch_id=CALIBRATION_BATCH.batch_id,
            execution_mode="dry_run",
            topologies=("single_agent",),
            runs_dir=runs_dir,
        )
    )
    assert first
    again = _run(
        run_batch(
            batch_id=CALIBRATION_BATCH.batch_id,
            execution_mode="dry_run",
            topologies=("single_agent",),
            runs_dir=runs_dir,
            skip_existing=True,
        )
    )
    assert again == []


def test_manifest_aggregates_are_recomputed_from_branches() -> None:
    batch = select_batch_cases(CASES, CALIBRATION_BATCH)[:3]

    async def scenario() -> EvalRunManifest:
        manifest, _ = await run_segment(
            spec=CALIBRATION_BATCH,
            topology="single_agent",
            repeat_seed=20260901,
            execution_mode="dry_run",
            batch=batch,
            sleep=_no_sleep,
        )
        return manifest

    manifest = _run(scenario())
    assert isinstance(manifest, EvalRunManifest)
    derived = derived_eval_fields(manifest.calls, spec=CALIBRATION_BATCH)
    assert manifest.usage == derived["usage"]
    assert manifest.latency == derived["latency"]
    assert manifest.resolved_model_ids == (DRY_RUN_MODEL_ID_LOCK,)


def test_the_single_agent_request_keeps_the_same_fixture_id_as_the_workers() -> None:
    case = next(
        item
        for item in select_batch_cases(CASES, CALIBRATION_BATCH)
        if item.model_input.evidence_contexts and item.model_input.business_facts
    )
    single = single_agent_request_for(case)
    worker_request = model_request_for(case, WorkerKind.POLICY)
    assert single is not None and worker_request is not None
    assert single.model_fixture_id == worker_request.model_fixture_id
    assert tuple(single.workers) == tuple(WorkerKind)
    assert single.question == worker_request.question


def test_a_mutated_batch_spec_cannot_be_used_to_move_the_brakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(EVAL_BATCHES, CALIBRATION_BATCH.batch_id, raising=False)
    with pytest.raises(EvalContractError, match="batch_spec_not_frozen"):
        _run(
            run_segment(
                spec=CALIBRATION_BATCH,
                topology="single_agent",
                repeat_seed=20260901,
                execution_mode="dry_run",
                batch=select_batch_cases(CASES, CALIBRATION_BATCH)[:1],
                sleep=_no_sleep,
            )
        )


def test_a_single_agent_finding_citing_another_workers_document_fails_that_branch() -> (
    None
):
    """Cross-worker citations survive the adapter but not the per-worker validation."""

    case = next(
        item
        for item in select_batch_cases(CASES, PAIRED_BATCH)
        if len(item.model_input.evidence_contexts) == 2
        and item.model_input.business_facts
    )
    policy_ref, contract_ref = (
        context.evidence_ref for context in case.model_input.evidence_contexts
    )
    fact = case.model_input.business_facts[0]
    projection = build_projection(
        run_id="eval-single_agent-x-1-abcdef01",
        spec=PAIRED_BATCH,
        topology="single_agent",
        repeat_seed=20260901,
        execution_mode="dry_run",
        branches=[
            ProjectedBranch(
                case_id=case.case_id,
                split=case.split.value,
                worker=None,
                outcome="succeeded",
                findings=(
                    _projected(WorkerKind.POLICY, contract_ref),
                    _projected(WorkerKind.CONTRACT, contract_ref),
                    _projected_fact(WorkerKind.PROCUREMENT_DATA, fact),
                ),
            )
        ],
    )
    rows = observations_from_projection(projection, {case.case_id: case})
    assert len(rows) == 1
    outcomes = rows[0].branch_outcomes
    assert set(outcomes.values()) == {"succeeded"}
    assert rows[0].failed_workers == ("policy",)
    assert any("policy" in reason for reason in rows[0].failure_reasons)
    assert all(
        finding.worker is not WorkerKind.POLICY or finding.status.value != "supported"
        for finding in rows[0].findings
    )


def _projected(worker: WorkerKind, reference: EvidenceRef) -> ProjectedFinding:
    vocabulary = required_findings_for([worker])[0]
    return ProjectedFinding(
        worker=worker,
        finding_key=vocabulary["finding_key"],
        category=RiskCategory(vocabulary["category"]),
        risk_level=RiskLevel.LOW,
        assertion=FindingAssertion.RISK_ABSENT,
        status=FindingStatus.SUPPORTED,
        evidence_refs=(
            ProjectedRef(
                source_id=reference.source_id,
                version=reference.version,
                content_checksum=reference.content_checksum,
            ),
        ),
        statement_hash="a" * 64,
        recommended_action_hash="b" * 64,
    )


def _projected_fact(worker: WorkerKind, fact: BusinessFact) -> ProjectedFinding:
    vocabulary = required_findings_for([worker])[0]
    return ProjectedFinding(
        worker=worker,
        finding_key=vocabulary["finding_key"],
        category=RiskCategory(vocabulary["category"]),
        risk_level=RiskLevel.LOW,
        assertion=FindingAssertion.RISK_ABSENT,
        status=FindingStatus.SUPPORTED,
        business_fact_refs=(
            ProjectedRef(
                source_id=fact.record_id,
                version=fact.dataset_version,
                content_checksum=None,
            ),
        ),
        statement_hash="a" * 64,
        recommended_action_hash="b" * 64,
    )
