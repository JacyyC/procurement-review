"""Offline tests for the V1-02C eval contract, corpus builder and metric kernel."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from procurement_review.contracts import (
    FindingAssertion,
    FindingStatus,
    RecommendationStatus,
    RiskCategory,
    RiskLevel,
    WorkerKind,
)
from procurement_review.eval_contract import (
    CALIBRATION_BATCH,
    DRY_RUN_CREDENTIAL,
    EVAL_BATCHES,
    HOLDOUT_REJUDGEMENT_BATCH,
    PAIRED_BATCH,
    RESOLUTION_BATCH,
    STAGE_WORK_ITEM_IDS,
    EvalBatchSpec,
    EvalBranchEntry,
    EvalContractError,
    EvalProjection,
    EvalRunManifest,
    EvalSourceVersions,
    ProjectedBranch,
    ProjectedFinding,
    ProjectedRef,
    build_projection,
    derived_eval_fields,
    expected_capabilities,
    find_existing_segment,
    finalize_eval_manifest,
    gold_corpus_contract,
    guard_eval_manifest,
    load_batch_segments,
    write_segment_artifacts,
)
from procurement_review.eval_metrics import (
    MacroF1,
    Statistic,
    error_samples,
    macro_f1,
    observations_from_projection,
    paired_bootstrap_ci,
    paired_topology_deltas,
    quality_metrics,
    structured_output_stats,
    wilson_interval,
)
from procurement_review.gold_corpus import (
    build_gold_lexical_corpus,
    gold_required_document_ids,
    mask_case_identifiers,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from procurement_review.rules import is_generated_reason_code
from procurement_review.gold_set import (
    GoldSplit,
    load_gold_cases,
    load_gold_labels_for_split,
)

NOW = datetime(2026, 9, 10, 12, 0, 0, tzinfo=UTC)


def _source_versions(spec: EvalBatchSpec = CALIBRATION_BATCH) -> EvalSourceVersions:
    corpus = gold_corpus_contract(spec)
    return EvalSourceVersions(
        adapter="openai-compatible-chat-adapter.v1",
        finding_contract="chat-finding-contract.v1",
        gold_set=corpus.gold_set_version,
        gold_generator=corpus.generator_version,
        corpus_snapshot=corpus.corpus_snapshot_id,
        dataset="synthetic-gold-procurement.v1",
        policy="demo-procurement-policy-v1",
        rules="1.0.0",
        retrieval="precomputed_gold_context.v1",
        budget="v1-02c-local-brake.v1",
        runner="v1-02c-eval-runner.v1",
        code="0123456789abcdef0123456789abcdef01234567",
    )


def _entry(
    case_id: str = "GOLD-0001",
    *,
    worker: WorkerKind | None = None,
    outcome: str = "succeeded",
    disposition: str = "model_call",
    output_tokens: int | None = 10,
    latency_ms: int = 5,
    error_code: str | None = None,
    split: str = "dev",
) -> EvalBranchEntry:
    return EvalBranchEntry(
        case_id=case_id,
        split=split,  # type: ignore[arg-type]
        worker=worker,
        disposition=disposition,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        requested_model_id="deepseek-v4-flash",
        resolved_model_id=(
            CALIBRATION_BATCH.model_id_lock if disposition == "model_call" else None
        ),
        request_id_hash="a" * 64 if disposition == "model_call" else None,
        http_status=200 if disposition == "model_call" else None,
        input_tokens=20 if disposition == "model_call" else None,
        output_tokens=output_tokens if disposition == "model_call" else None,
        total_tokens=(
            20 + (output_tokens or 0) if disposition == "model_call" else None
        ),
        latency_ms=latency_ms if disposition == "model_call" else None,
        finding_count=1 if outcome == "succeeded" else None,
        error_code=error_code,
        error_detail="branch_timeout" if error_code else None,
    )


def _manifest(
    calls: list[EvalBranchEntry],
    *,
    spec: EvalBatchSpec = CALIBRATION_BATCH,
    topology: str = "multi_agent",
    outcome: str = "completed",
    failure_code: str | None = None,
    voided: bool = False,
) -> EvalRunManifest:
    derived = derived_eval_fields(calls, spec=spec)
    return EvalRunManifest(
        run_id=f"eval-{topology}-{spec.batch_id}-{spec.repeat_seeds[0]}-abcdef01",
        batch_id=spec.batch_id,
        work_item_id=STAGE_WORK_ITEM_IDS[spec.stage_id],
        stage_id=spec.stage_id,
        topology=topology,  # type: ignore[arg-type]
        repeat_seed=spec.repeat_seeds[0],
        execution_mode="real_endpoint",
        case_split_scope="+".join(split.value for split in spec.splits),
        case_count=spec.case_count,
        cases_per_family=spec.cases_per_family,
        concurrency=spec.concurrency,
        model_id_lock=spec.model_id_lock,
        requested_model_id="deepseek-v4-flash",
        capabilities=expected_capabilities(),
        source_versions=_source_versions(spec),
        gold_manifest_checksum="b" * 64,
        started_at=NOW,
        finished_at=NOW,
        calls=tuple(calls),
        outcome=outcome,  # type: ignore[arg-type]
        failure_code=failure_code,
        voided=voided,
        **derived,
    )


def test_frozen_batch_specs_match_the_confirmed_scope() -> None:
    assert CALIBRATION_BATCH.case_count == 48 == 12 * 4
    assert CALIBRATION_BATCH.splits == (GoldSplit.DEV,)
    assert CALIBRATION_BATCH.concurrency == 1
    assert PAIRED_BATCH.case_count == 360 == 12 * 30
    assert PAIRED_BATCH.splits == (GoldSplit.DEV, GoldSplit.VALIDATION)
    assert GoldSplit.HOLDOUT not in PAIRED_BATCH.splits
    assert PAIRED_BATCH.repeat_seeds == (20260901, 20260902, 20260903)
    assert PAIRED_BATCH.concurrency == 4
    assert PAIRED_BATCH.max_output_tokens == 24_000_000
    assert RESOLUTION_BATCH.case_count == 120 == 12 * 10
    assert RESOLUTION_BATCH.splits == (GoldSplit.DEV,)
    assert GoldSplit.HOLDOUT not in RESOLUTION_BATCH.splits
    assert GoldSplit.VALIDATION not in RESOLUTION_BATCH.splits
    assert RESOLUTION_BATCH.repeat_seeds == (20260901, 20260902)
    assert RESOLUTION_BATCH.concurrency == 4
    assert RESOLUTION_BATCH.max_requests == 25_000
    assert RESOLUTION_BATCH.max_input_tokens == 250_000_000
    assert RESOLUTION_BATCH.max_output_tokens == 24_000_000
    # V1-05 Spec §6.2: the one pre-registered holdout batch, with its own hard caps.
    assert HOLDOUT_REJUDGEMENT_BATCH.case_count == 120 == 12 * 10
    assert HOLDOUT_REJUDGEMENT_BATCH.splits == (GoldSplit.HOLDOUT,)
    assert HOLDOUT_REJUDGEMENT_BATCH.repeat_seeds == (20260901, 20260902)
    assert HOLDOUT_REJUDGEMENT_BATCH.concurrency == 4
    assert HOLDOUT_REJUDGEMENT_BATCH.max_requests == 1_200
    assert HOLDOUT_REJUDGEMENT_BATCH.max_input_tokens == 20_000_000
    assert HOLDOUT_REJUDGEMENT_BATCH.max_output_tokens == 4_000_000
    assert HOLDOUT_REJUDGEMENT_BATCH.gold_corpus_version == "v2"
    assert HOLDOUT_REJUDGEMENT_BATCH.stage_id == "V1-05B"
    assert set(EVAL_BATCHES) == {
        CALIBRATION_BATCH.batch_id,
        PAIRED_BATCH.batch_id,
        RESOLUTION_BATCH.batch_id,
        HOLDOUT_REJUDGEMENT_BATCH.batch_id,
    }
    # The hidden split stays unsealed for exactly one registered batch.
    holdout_batches = [
        spec.batch_id
        for spec in EVAL_BATCHES.values()
        if GoldSplit.HOLDOUT in spec.splits
    ]
    assert holdout_batches == [HOLDOUT_REJUDGEMENT_BATCH.batch_id]
    for spec in EVAL_BATCHES.values():
        assert spec.model_id_lock == "deepseek-v4-flash-ga-260731"
    # The clause-level corpus v2 is declared by the two batches that ran on it; the
    # anchored v1 batches keep declaring corpus v1 so their reports stay replayable.
    assert RESOLUTION_BATCH.gold_corpus_version == "v2"
    assert RESOLUTION_BATCH.stage_id == "V1-02C2"
    assert CALIBRATION_BATCH.gold_corpus_version == "v1"
    assert PAIRED_BATCH.gold_corpus_version == "v1"
    assert CALIBRATION_BATCH.stage_id == "V1-02C"
    assert PAIRED_BATCH.stage_id == "V1-02C"


def test_manifest_is_frozen_to_its_batches_gold_corpus_version() -> None:
    calls = [_entry(worker=worker) for worker in WorkerKind]
    v2 = _manifest(calls, spec=RESOLUTION_BATCH)
    assert v2.stage_id == "V1-02C2"
    assert v2.source_versions.corpus_snapshot == "synthetic-gold-corpus.v2"
    payload = v2.model_dump(mode="json")
    payload["stage_id"] = "V1-02C"
    with pytest.raises(ValueError, match="stage id is frozen per batch"):
        EvalRunManifest.model_validate(payload)
    stale = v2.model_dump(mode="json")
    stale["source_versions"] = _source_versions(CALIBRATION_BATCH).model_dump(
        mode="json"
    )
    with pytest.raises(ValueError, match="corpus snapshot does not match the batch"):
        EvalRunManifest.model_validate(stale)
    v1 = _manifest(calls)
    moved = v1.model_dump(mode="json")
    versions = _source_versions(RESOLUTION_BATCH).model_dump(mode="json")
    versions["corpus_snapshot"] = "synthetic-gold-corpus.v1"
    moved["source_versions"] = versions
    with pytest.raises(ValueError, match="gold set version does not match the batch"):
        EvalRunManifest.model_validate(moved)


def test_holdout_branches_stay_inside_the_batch_that_declared_them() -> None:
    calls = [_entry(worker=worker, split="holdout") for worker in WorkerKind]
    manifest = _manifest(calls, spec=HOLDOUT_REJUDGEMENT_BATCH)
    assert manifest.work_item_id == "V1-05"
    assert manifest.case_split_scope == "holdout"
    relabelled = manifest.model_dump(mode="json")
    relabelled["work_item_id"] = "V1-02"
    with pytest.raises(
        ValueError, match="work item id must follow the registered stage"
    ):
        EvalRunManifest.model_validate(relabelled)
    leaked = manifest.model_dump(mode="json")
    leaked["calls"][0]["split"] = "dev"
    with pytest.raises(ValueError, match="branch split outside the frozen batch scope"):
        EvalRunManifest.model_validate(leaked)
    with pytest.raises(ValueError, match="branch split outside the frozen batch scope"):
        _manifest(
            [_entry(worker=WorkerKind.POLICY, split="holdout")], spec=RESOLUTION_BATCH
        )
    with pytest.raises(EvalContractError, match="projection_scope_violation"):
        build_projection(
            run_id="eval-multi_agent-holdout-leak-20260901-abcdef01",
            spec=RESOLUTION_BATCH,
            topology="multi_agent",
            repeat_seed=20260901,
            execution_mode="real_endpoint",
            branches=[
                ProjectedBranch(
                    case_id="GOLD-0001",
                    split="holdout",
                    worker=WorkerKind.POLICY,
                    outcome="succeeded",
                )
            ],
        )


def test_batch_spec_rejects_a_case_count_that_cannot_be_selected() -> None:
    with pytest.raises(ValueError, match="family_count"):
        EvalBatchSpec(
            batch_id="bad",
            splits=(GoldSplit.DEV,),
            cases_per_family=5,
            case_count=48,
            repeat_seeds=(20260901,),
            concurrency=1,
            max_requests=10,
            max_input_tokens=10,
            max_output_tokens=10,
            model_id_lock=CALIBRATION_BATCH.model_id_lock,
            note="x",
        )


def test_manifest_rejects_a_self_reported_aggregate_that_does_not_match() -> None:
    calls = [_entry(worker=WorkerKind.POLICY)]
    manifest = _manifest(calls)
    with pytest.raises(ValueError, match="must match recorded calls"):
        EvalRunManifest.model_validate(
            {**manifest.model_dump(mode="json"), "succeeded_count": 7}
        )


def test_manifest_cannot_add_fields_or_claim_retention() -> None:
    payload = _manifest([_entry(worker=WorkerKind.POLICY)]).model_dump(mode="json")
    with pytest.raises(ValueError):
        EvalRunManifest.model_validate({**payload, "surprise": 1})
    retention = dict(payload["retention"])
    retention["response_content_retained"] = True
    with pytest.raises(ValueError):
        EvalRunManifest.model_validate({**payload, "retention": retention})


def test_guard_never_trips_on_its_own_retention_field_names() -> None:
    manifest = _manifest([_entry(worker=worker) for worker in WorkerKind])
    assert guard_eval_manifest(manifest, credential=DRY_RUN_CREDENTIAL) == ()


def test_guard_scrubs_a_planted_secret_and_closes_the_run() -> None:
    entry = _entry(worker=WorkerKind.POLICY).model_copy(
        update={"error_detail": "sk-abcdefghijklmnopqrst"}
    )
    manifest = _manifest(
        [entry],
        outcome="failed_closed",
        failure_code="model_call_failed",
    )
    reasons = guard_eval_manifest(manifest, credential=DRY_RUN_CREDENTIAL)
    assert any(reason.startswith("secret_value:") for reason in reasons)
    cleaned = finalize_eval_manifest(
        manifest, credential=DRY_RUN_CREDENTIAL, spec=CALIBRATION_BATCH
    )
    assert cleaned.outcome == "failed_closed"
    assert cleaned.calls[0].error_detail == "redacted_value"
    assert guard_eval_manifest(cleaned, credential=DRY_RUN_CREDENTIAL) == ()


def test_voided_flag_is_locked_to_a_voiding_failure_code() -> None:
    calls = [
        _entry(worker=WorkerKind.POLICY, outcome="failed", error_code="model_id_drift")
    ]
    manifest = _manifest(
        calls, outcome="failed_closed", failure_code="model_drift", voided=True
    )
    assert manifest.voided is True
    with pytest.raises(ValueError, match="voided flag"):
        EvalRunManifest.model_validate(
            {**manifest.model_dump(mode="json"), "voided": False}
        )


def test_budget_reports_the_80_percent_warning_and_the_hard_stop() -> None:
    tight = CALIBRATION_BATCH.model_copy(update={"max_output_tokens": 12})
    one = derived_eval_fields([_entry(worker=WorkerKind.POLICY)], spec=tight)["budget"]
    assert one.consumed_output_tokens == 10
    assert one.warning_reached is True and one.exceeded is False
    two = derived_eval_fields(
        [_entry(worker=worker) for worker in WorkerKind], spec=tight
    )["budget"]
    assert two.warning_reached is True and two.exceeded is True


def test_segments_round_trip_through_an_ignored_directory_with_owner_only_modes(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / ".local" / "runs"
    runs_dir.mkdir(parents=True)
    manifest = _manifest([_entry(worker=worker) for worker in WorkerKind])
    projection = build_projection(
        run_id=manifest.run_id,
        spec=CALIBRATION_BATCH,
        topology="multi_agent",
        repeat_seed=CALIBRATION_BATCH.repeat_seeds[0],
        execution_mode="real_endpoint",
        branches=[
            ProjectedBranch(
                case_id="GOLD-0001",
                split="dev",
                worker=worker,
                outcome="succeeded",
            )
            for worker in WorkerKind
        ],
    )
    manifest_path, projection_path = write_segment_artifacts(
        manifest, projection, runs_dir=runs_dir
    )
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    assert projection_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(EvalContractError, match="segment_already_exists"):
        write_segment_artifacts(manifest, projection, runs_dir=runs_dir)
    assert load_batch_segments(CALIBRATION_BATCH.batch_id, runs_dir=runs_dir)
    found = find_existing_segment(
        batch_id=CALIBRATION_BATCH.batch_id,
        topology="multi_agent",
        repeat_seed=CALIBRATION_BATCH.repeat_seeds[0],
        runs_dir=runs_dir,
    )
    assert found == manifest_path
    outside = tmp_path / "runs"
    outside.mkdir()
    with pytest.raises(EvalContractError):
        write_segment_artifacts(manifest, projection, runs_dir=outside)


def _projection_finding(worker: WorkerKind, ref: ProjectedRef) -> ProjectedFinding:
    return ProjectedFinding(
        worker=worker,
        finding_key=f"{worker.value}.required",
        category=RiskCategory.POLICY_COMPLIANCE,
        risk_level=RiskLevel.LOW,
        assertion=FindingAssertion.RISK_ABSENT,
        status=FindingStatus.SUPPORTED,
        evidence_refs=() if worker is WorkerKind.PROCUREMENT_DATA else (ref,),
        business_fact_refs=(ref,) if worker is WorkerKind.PROCUREMENT_DATA else (),
        statement_hash="c" * 64,
        recommended_action_hash="d" * 64,
    )


def test_projection_bounds_retained_model_prose() -> None:
    """The missing-information list is the one place model text survives a run.

    ``project_findings`` truncates to twelve fragments of 160 characters, and the schema
    repeats that ceiling so a future generator cannot widen it silently.
    """

    ref = ProjectedRef(
        source_id="DOC-1", version="v1", content_checksum=f"sha256:{'a' * 64}"
    )

    def projected(items: tuple[str, ...]) -> ProjectedFinding:
        return ProjectedFinding(
            worker=WorkerKind.POLICY,
            finding_key="policy.required_policy_evidence",
            category=RiskCategory.POLICY_COMPLIANCE,
            risk_level=RiskLevel.LOW,
            assertion=FindingAssertion.RISK_ABSENT,
            status=FindingStatus.SUPPORTED,
            evidence_refs=(ref,),
            missing_information=items,
            statement_hash="c" * 64,
            recommended_action_hash="d" * 64,
        )

    assert (
        len(
            projected(
                tuple(f"fragment-{index}" for index in range(12))
            ).missing_information
        )
        == 12
    )
    with pytest.raises(ValidationError):
        projected(("x" * 161,))
    with pytest.raises(ValidationError):
        projected(tuple(f"f-{index}" for index in range(13)))


def test_projection_drops_a_screened_string_but_keeps_the_branch() -> None:
    branch = ProjectedBranch(
        case_id="GOLD-0001",
        split="dev",
        worker=WorkerKind.POLICY,
        outcome="succeeded",
        findings=(
            ProjectedFinding(
                worker=WorkerKind.POLICY,
                finding_key="policy.required_policy_evidence",
                category=RiskCategory.POLICY_COMPLIANCE,
                risk_level=RiskLevel.LOW,
                assertion=FindingAssertion.RISK_ABSENT,
                status=FindingStatus.SUPPORTED,
                evidence_refs=(
                    ProjectedRef(
                        source_id="DOC-GOLD-POLICY-0001",
                        version="synthetic-gold-document.v1",
                        content_checksum="sha256:" + "e" * 64,
                    ),
                ),
                missing_information=("sk-abcdefghijklmnopqrst",),
                statement_hash="c" * 64,
                recommended_action_hash="d" * 64,
            ),
        ),
    )
    projection = build_projection(
        run_id="eval-multi_agent-x-1-abcdef01",
        spec=CALIBRATION_BATCH,
        topology="multi_agent",
        repeat_seed=20260901,
        execution_mode="dry_run",
        branches=[branch],
    )
    assert projection.branches[0].findings[0].missing_information == ("redacted_value",)
    assert projection.redaction_count == 1


def test_projection_rejects_a_single_agent_branch_that_names_a_worker() -> None:
    with pytest.raises(ValueError, match="must not name a worker"):
        EvalProjection(
            run_id="eval-single_agent-x-1-abcdef01",
            batch_id=CALIBRATION_BATCH.batch_id,
            topology="single_agent",
            repeat_seed=20260901,
            execution_mode="dry_run",
            model_id_lock="dry-run-model-resolved-local",
            branches=[
                ProjectedBranch(
                    case_id="GOLD-0001",
                    split="dev",
                    worker=WorkerKind.POLICY,
                    outcome="succeeded",
                )
            ],
        )


def test_lexical_corpus_is_isolated_per_split() -> None:
    cases = load_gold_cases().cases
    dev = build_gold_lexical_corpus(cases, split=GoldSplit.DEV)
    validation = build_gold_lexical_corpus(cases, split=GoldSplit.VALIDATION)
    dev_ids = {doc.source_id for doc in dev.documents}
    validation_ids = {doc.source_id for doc in validation.documents}
    assert dev.document_count == 440
    assert dev_ids.isdisjoint(validation_ids)
    assert dev.vector_mode == "not_supported_lexical_only.v1"


def test_retrieval_helpers_match_hand_computed_values() -> None:
    ranked = ("doc-a", "doc-b", "doc-c", "doc-d", "doc-e", "doc-f")
    assert recall_at_k(ranked, ("doc-c", "doc-f"), k=5) == 0.5
    assert reciprocal_rank(ranked, ("doc-c",)) == 1 / 3
    assert reciprocal_rank(ranked, ("missing",)) == 0.0
    assert ndcg_at_k(("doc-a",), ("doc-a",), k=5) == 1.0
    assert ndcg_at_k(("doc-a",), ("doc-b",), k=5) == 0.0


def test_masking_strips_the_case_number_but_keeps_the_words() -> None:
    masked = mask_case_identifiers("请依据演示政策审查合成采购 Case 0007，只返回发现")
    assert "0007" not in masked
    assert "审查" in masked.replace(" ", "")


def test_gold_required_documents_only_count_case_documents() -> None:
    labels = load_gold_labels_for_split("dev")
    with_docs = [label for label in labels if gold_required_document_ids(label)]
    assert with_docs
    assert all(
        ref.startswith("DOC-")
        for label in with_docs
        for ref in gold_required_document_ids(label)
    )


def test_wilson_interval_and_bootstrap_are_bounded_and_deterministic() -> None:
    low, high = wilson_interval(20, 440)
    assert 0.0 <= low < 20 / 440 < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    first = paired_bootstrap_ci([1.0, 0.0, -1.0, 0.0] * 25)
    second = paired_bootstrap_ci([1.0, 0.0, -1.0, 0.0] * 25)
    assert first == second
    assert first is not None and first[0] <= 0.0 <= first[1]
    assert paired_bootstrap_ci([]) is None


def test_statistic_applies_the_threshold_direction() -> None:
    good = Statistic.of(0, 10, threshold=0.0, direction="<=")
    bad = Statistic.of(3, 10, threshold=0.0, direction="<=")
    assert good.meets_threshold is True and good.value == 0.0
    assert bad.meets_threshold is False
    assert Statistic.of(97, 100, threshold=0.95).meets_threshold is True
    assert Statistic.of(94, 100, threshold=0.95).meets_threshold is False
    assert Statistic.of(97, 100, threshold=0.95).direction == ">="
    assert Statistic.of(0, 0).meets_threshold is None


def test_macro_f1_ignores_classes_without_gold_support() -> None:
    result: MacroF1 = macro_f1({"recommend_approve": (8, 10, 10)})
    assert result.class_count == 1
    assert result.value == pytest.approx(8 / 10)
    empty = macro_f1({})
    assert empty.value is None and empty.class_count == 0


def _observation(
    case_id: str,
    *,
    topology: str,
    seed: int,
    recommendation: RecommendationStatus,
    risk: RiskLevel = RiskLevel.LOW,
) -> object:
    from procurement_review.eval_metrics import CaseObservation

    return CaseObservation(
        case_id=case_id,
        split="dev",
        case_family_id="low_risk_complete",
        seed=seed,
        topology=topology,  # type: ignore[arg-type]
        recommendation=recommendation,
        highest_risk=risk,
        human_review_required=recommendation
        is not RecommendationStatus.RECOMMEND_APPROVE,
        conflict_keys=(),
        missing_information=(),
        failure_reasons=(),
        findings=(),
        branch_outcomes={},
        failed_workers=(),
    )


def test_quality_metrics_reproduce_hand_computed_accuracy() -> None:
    cases = {case.case_id: case for case in load_gold_cases().cases}
    labels = {label.case_id: label for label in load_gold_labels_for_split("dev")}
    picked = next(
        case_id
        for case_id in sorted(labels)
        if labels[case_id].expected_recommendation
        is RecommendationStatus.RECOMMEND_APPROVE
        and case_id in cases
    )
    observations = [
        _observation(
            picked,
            topology="multi_agent",
            seed=20260901,
            recommendation=labels[picked].expected_recommendation,
        ),
        _observation(
            picked,
            topology="multi_agent",
            seed=20260902,
            recommendation=RecommendationStatus.RECOMMEND_REJECT,
        ),
    ]
    report = quality_metrics(observations, cases, labels)
    accuracy = report["recommendation_accuracy"]
    assert (accuracy.numerator, accuracy.denominator) == (1, 2)
    assert accuracy.value == 0.5


def test_paired_delta_favours_the_higher_topology_and_counts_cases_not_calls() -> None:
    labels = {label.case_id: label for label in load_gold_labels_for_split("dev")}
    chosen = sorted(labels)[:6]
    single = [
        _observation(
            case_id,
            topology="single_agent",
            seed=20260901,
            recommendation=labels[case_id].expected_recommendation,
        )
        for case_id in chosen
    ]
    multi = [
        _observation(
            case_id,
            topology="multi_agent",
            seed=20260901,
            recommendation=RecommendationStatus.RECOMMEND_REJECT,
        )
        for case_id in chosen
    ]
    deltas = {
        delta.metric: delta
        for delta in paired_topology_deltas(single, multi, labels)  # type: ignore[arg-type]
    }
    exact = deltas["recommendation_exact"]
    assert exact.case_count == len(chosen)
    assert exact.delta == 1.0
    assert exact.favors == "single_agent"
    assert exact.ci_low is not None and exact.ci_low > 0


def test_error_samples_keep_identities_and_labels_but_no_model_text() -> None:
    cases = {case.case_id: case for case in load_gold_cases().cases}
    labels = {label.case_id: label for label in load_gold_labels_for_split("dev")}
    wrong = next(
        case_id
        for case_id in sorted(labels)
        if labels[case_id].expected_recommendation
        is RecommendationStatus.RECOMMEND_APPROVE
        and case_id in cases
    )
    samples = error_samples(
        [
            _observation(
                wrong,
                topology="single_agent",
                seed=20260901,
                recommendation=RecommendationStatus.RECOMMEND_REJECT,
            )
        ],
        labels,  # type: ignore[arg-type]
    )
    assert samples
    payload = json.dumps(samples, ensure_ascii=False)
    assert samples[0]["expected_recommendation"] == "recommend_approve"
    assert "statement:" not in payload and "sk-" not in payload
    # a model-authored missing statement must reach the report only as a digest
    assert all(
        item.startswith("sha256:") or is_generated_reason_code(item)
        for item in samples[0]["missing_information"]
    )


def test_structured_output_stats_read_only_manifest_metadata() -> None:
    calls = [
        _entry("GOLD-0001", worker=WorkerKind.POLICY),
        _entry(
            "GOLD-0002",
            worker=WorkerKind.CONTRACT,
            outcome="failed",
            error_code="timeout",
            output_tokens=None,
            latency_ms=90_004,
        ),
        _entry(
            "GOLD-0003",
            worker=WorkerKind.CONTRACT,
            disposition="skipped_empty_context",
            outcome="not_requested",
            output_tokens=0,
        ),
    ]
    manifest = _manifest(
        calls, outcome="failed_closed", failure_code="model_call_failed"
    )
    stats = structured_output_stats([manifest])
    assert stats["model_call_count"] == 2
    assert stats["skipped_empty_context_count"] == 1
    assert stats["structured_output_valid_rate"].value == 0.5
    completeness = stats["run_record_completeness"]
    assert (completeness.numerator, completeness.denominator) == (3, 144)
    assert completeness.meets_threshold is False
    assert stats["error_code_counts"] == {"timeout": 1}


def test_a_full_segment_reports_complete_run_records() -> None:
    single = _manifest(
        [
            _entry(f"GOLD-{index:04d}")
            for index in range(1, CALIBRATION_BATCH.case_count + 1)
        ],
        topology="single_agent",
    )
    stats = structured_output_stats([single])
    completeness = stats["run_record_completeness"]
    assert (completeness.numerator, completeness.denominator) == (48, 48)
    assert completeness.value == 1.0 and completeness.meets_threshold is True


def test_observations_reject_a_projected_reference_outside_its_case() -> None:
    cases = {case.case_id: case for case in load_gold_cases().cases}
    case = next(
        item
        for item in cases.values()
        if item.split is GoldSplit.DEV and item.model_input.evidence_contexts
    )
    projection = build_projection(
        run_id="eval-multi_agent-x-1-abcdef01",
        spec=CALIBRATION_BATCH,
        topology="multi_agent",
        repeat_seed=20260901,
        execution_mode="dry_run",
        branches=[
            ProjectedBranch(
                case_id=case.case_id,
                split="dev",
                worker=WorkerKind.POLICY,
                outcome="succeeded",
                findings=(
                    _projection_finding(
                        WorkerKind.POLICY,
                        ProjectedRef(
                            source_id="DOC-GOLD-POLICY-9999",
                            version="synthetic-gold-document.v1",
                            content_checksum="sha256:" + "f" * 64,
                        ),
                    ),
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="outside its Gold Case"):
        observations_from_projection(projection, cases)


def test_pooled_http_counts_sum_across_segments() -> None:
    first = _manifest([_entry("GOLD-0001", worker=WorkerKind.POLICY)])
    second = _manifest(
        [
            _entry(
                "GOLD-0002",
                worker=WorkerKind.CONTRACT,
                outcome="failed",
                error_code="timeout",
                output_tokens=0,
            )
        ],
        outcome="failed_closed",
        failure_code="model_call_failed",
    )
    stats = structured_output_stats([first, second])
    assert stats["http_status_counts"] == {"200": 2}
    assert stats["latency_ms"]["observed_count"] == 2
    assert stats["model_call_count"] == 2


def test_pooled_latency_percentiles_use_every_call() -> None:
    entries = [
        _entry(
            f"GOLD-{index:04d}",
            worker=WorkerKind.POLICY,
            latency_ms=latency,
        )
        for index, latency in enumerate((10, 20, 30, 90_000), start=1)
    ]
    stats = structured_output_stats([_manifest(entries)])
    assert stats["latency_ms"] == {
        "observed_count": 4,
        "p50": 20,
        "p95": 90_000,
        "max": 90_000,
    }
