"""Independent early-release, source-provenance and metering boundaries for H."""

import copy
from types import SimpleNamespace

import pytest

from procurement_review.eval_h_material import source_identities
from procurement_review.eval_h_report import failure_report, safety
from procurement_review.gold_v3_contracts import payload_sha256
from procurement_review.eval_h_storage import read, sha, write, write_bytes
from test_v1_07h_preparation import sealed

SAFETY_CATEGORIES = (
    "unauthorized",
    "external_write",
    "sensitive_output",
    "duplicate_receipt",
    "illegal_late_adoption",
)


def safety_summary(product, sources, *, categories=None, refusal=None):
    return dict(
        schema_version="v1-07h-safety.v1",
        product_sha256=product,
        mode="disposable",
        categories=categories
        if categories is not None
        else {name: dict(exposures=1, violations=0) for name in SAFETY_CATEGORIES},
        sources=sources,
        refusal_counterexamples=refusal
        if refusal is not None
        else dict(planned=1, checked=1, passed=1),
    )


def safety_source(product, observations=None, *, mode="disposable"):
    return dict(
        schema_version="v1-07h-safety-source.v1",
        product_sha256=product,
        mode=mode,
        observations=observations
        if observations is not None
        else [
            {"kind": "forbidden", "category": name, "outcome": "clean"}
            for name in SAFETY_CATEGORIES
        ]
        + [{"kind": "refusal", "outcome": "passed"}],
    )


def test_safety_consumer_recomputes_registered_sources(tmp_path):
    product = "a" * 64
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    source = tmp_path / "source.json"
    write(source, safety_source(product))
    write(private / "safety.json", safety_summary(product, {str(source): sha(source)}))
    plan = SimpleNamespace(
        product_sha256=product,
        mode="disposable",
        lifecycle_evidence_sha256=sha(private / "safety.json"),
    )
    verdict = safety(tmp_path, plan)
    assert verdict["verified"] and verdict["refusals_verified"]

    def bind(body_source, summary=None, refusal=None):
        write(source, body_source, replace=True)
        write(
            private / "safety.json",
            summary
            if summary is not None
            else safety_summary(product, {str(source): sha(source)}, refusal=refusal),
            replace=True,
        )
        plan.lifecycle_evidence_sha256 = sha(private / "safety.json")
        return safety(tmp_path, plan)

    # An empty source with a matching digest proves nothing.
    empty = tmp_path / "empty.json"
    write(empty, {})
    verdict = bind(
        {},
        safety_summary(product, {str(empty): sha(empty)}),
    )
    assert not verdict["verified"] and not verdict["refusals_verified"]

    # The formal-qualified string alone cannot qualify an empty source set.
    summary = safety_summary(product, {str(empty): sha(empty)})
    summary["mode"] = "formal-qualified"
    verdict = bind(safety_source(product, mode="formal-qualified"), summary)
    assert not verdict["verified"] and not verdict["refusals_verified"]

    # A source recording a violation and a failed refusal contradicts the summary.
    verdict = bind(
        safety_source(
            product,
            [
                {
                    "kind": "forbidden",
                    "category": "unauthorized",
                    "outcome": "violation",
                },
                {"kind": "refusal", "outcome": "failed"},
            ],
        )
    )
    assert not verdict["verified"] and not verdict["refusals_verified"]

    # Wrong product identity, wrong scope and missing observations are all refused.
    for body in (
        safety_source("b" * 64),
        safety_source(product, mode="formal-qualified"),
        safety_source(product, []),
        safety_source(
            product,
            [
                {"kind": "forbidden", "category": name, "outcome": "clean"}
                for name in SAFETY_CATEGORIES[:4]
            ]
            + [{"kind": "refusal", "outcome": "passed"}],
        ),
    ):
        verdict = bind(body)
        assert not verdict["verified"] and not verdict["refusals_verified"]

    # Failure masquerading as success: an unchecked refusal is not a passed one.
    verdict = bind(
        safety_source(
            product,
            [
                {"kind": "forbidden", "category": name, "outcome": "clean"}
                for name in SAFETY_CATEGORIES
            ]
            + [{"kind": "refusal", "outcome": "unchecked"}],
        ),
        refusal=dict(planned=1, checked=0, passed=0),
    )
    assert verdict["verified"] and not verdict["refusals_verified"]

    # A source whose bytes changed after registration loses its digest binding.
    write(source, safety_source(product, mode="other"), replace=True)
    changed = safety(tmp_path, plan)
    assert not changed["verified"] and not changed["refusals_verified"]

    # The registered summary itself is bound to the plan by digest.
    verdict = bind(safety_source(product))
    assert verdict["verified"] and verdict["refusals_verified"]
    write(private / "safety.json", {}, replace=True)
    missing = safety(tmp_path, plan)
    assert missing["verified"] is False
    assert missing["reason"] == "registered_safety_evidence_missing"


def test_uncommitted_attempt_ledger_is_metered_without_double_counting(
    tmp_path, capsys
):
    from procurement_review.eval_h_cli import main
    from procurement_review.eval_h_journal import Journal

    seal = sealed(tmp_path)
    plan = seal.plan()
    case, other = plan.case_ids
    journal = Journal(seal.directory)
    # Reserved, not settled: a consumed slot with an unknown send state.
    journal.reserve(case, 101, "single_agent", "joint", "a" * 64)
    # Settled with a real receipt and known usage.
    path = journal.reserve(other, 101, "single_agent", "joint", "b" * 64)
    journal.settle(
        path,
        dict(
            error_code=None,
            correctable=False,
            retryable=False,
            input_tokens=100,
            output_tokens=20,
            usage_unknown=False,
            returned=True,
            latency_ms=5,
            resolved_model=plan.configuration.resolved_model,
            http_status=200,
            finish_reason="stop",
        ),
    )
    # Settled as unknown after a real reservation; the slot stays consumed.
    journal.reserve(case, 202, "single_agent", "joint", "c" * 64)
    journal.recover_unknown(case, 202, "single_agent")
    # A receipt that proves the request was never sent.
    never = journal.reserve(other, 202, "single_agent", "joint", "d" * 64)
    journal.settle(
        never,
        dict(
            error_code="credential_in_request",
            correctable=False,
            retryable=False,
            input_tokens=None,
            output_tokens=None,
            usage_unknown=False,
            returned=False,
            latency_ms=None,
            resolved_model=None,
            http_status=None,
        ),
    )

    direct = failure_report(seal.directory)["Q-12"]["resources"]["single_agent"]
    assert direct["physical_attempts"] == 4
    assert direct["unsettled_attempts"] == 1
    assert direct["unknown_send_state"] == 2
    assert direct["unknown_usage"] == 2
    assert direct["proven_sends"] == 1
    assert direct["not_sent"] == 1
    assert direct["first_sends"] == 1
    assert direct["known_input_tokens"] == 100
    assert direct["known_output_tokens"] == 20
    assert direct["nonempty_bodies"] == 1
    assert direct["unreturned"] == 1
    assert direct["failures"] == 2
    assert direct["timeouts"] == direct["cancellations"] == 0

    # A committed observation of one settled attempt is never added twice.
    journal.commit(
        dict(
            schema_version="v1-07h-observation.v1",
            case_id=other,
            repeat_seed=101,
            arm="single_agent",
            mode="disposable",
            row=dict(status="failed"),
            attempts=journal.attempts(other, 101, "single_agent"),
        )
    )
    again = failure_report(seal.directory)["Q-12"]["resources"]["single_agent"]
    assert {key: value for key, value in again.items() if key != "mode"} == {
        key: value for key, value in direct.items() if key != "mode"
    }

    # The actual report CLI stores the same accounting.
    assert main(["report", str(seal.directory)]) == 0
    capsys.readouterr()
    stored = read(seal.directory / "result.json")["Q-12"]["resources"]["single_agent"]
    assert stored["physical_attempts"] == 4
    assert stored["unknown_usage"] == 2
    assert stored["unknown_send_state"] == 2
    assert stored["unsettled_attempts"] == 1
    assert stored["not_sent"] == 1


@pytest.fixture(scope="module")
def released_crafted_batch(tmp_path_factory):
    """A real released batch whose bound safety source contradicts its summary."""
    import asyncio
    import json
    import os
    from pathlib import Path

    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    from procurement_review.eval_h_cli import create
    from procurement_review.eval_h_consumption import completion_proof
    from procurement_review.eval_h_contracts import HPlan
    from procurement_review.eval_h_fixture import make_disposable
    from procurement_review.eval_h_mock import mock_transport
    from procurement_review.eval_h_report import rebuild
    from procurement_review.eval_h_runner import run, setup
    from procurement_review.eval_h_score import rebuild as score
    from procurement_review.eval_h_seal import HSeal

    directory = tmp_path_factory.mktemp("h-crafted-safety")
    base = os.environ.get("TEST_POSTGRES_DSN")
    assert base, "TEST_POSTGRES_DSN is required; H PostgreSQL checks never skip"
    source = Path(
        make_disposable(
            directory / ".local", 12, batch_id="disposable-h-crafted-safety"
        )["root"]
    )
    plan_body = json.loads((source / "plan.json").read_bytes())
    crafted = directory / "safety-source.json"
    write(
        crafted,
        dict(
            schema_version="audit-counterexample.v1",
            product_sha256="b" * 64,
            observed_violations={"unauthorized": 1},
            refusal_observed=dict(planned=1, checked=1, passed=0),
        ),
    )
    safety_path = directory / "safety.json"
    write(
        safety_path,
        safety_summary(plan_body["product_sha256"], {str(crafted): sha(crafted)}),
    )
    plan_body["lifecycle_evidence_sha256"] = sha(safety_path)
    HPlan.model_validate_json(json.dumps(plan_body), strict=True)
    plan_path = directory / "plan.json"
    write(plan_path, plan_body)
    created = create(
        directory / ".local/batches",
        plan_path,
        source / "private/recipe.json",
        source / "private/seed.bin",
        source / "private/registry.json",
        safety_path,
    )
    seal = HSeal(Path(created["root"]))
    seal.generate(
        seal.directory / "private/recipe.json",
        seal.directory / "private/seed.bin",
        seal.directory / "private/registry.json",
    )
    seal.seal()
    database = "v107h_" + seal.plan().batch_id.replace("-", "_")
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{database}"'))
    url = make_url(base).set(database=database).render_as_string(hide_password=False)
    try:
        setup(seal.directory, url)
        asyncio.run(
            run(seal.directory, transport_override=mock_transport("material", []))
        )
        proof = completion_proof(seal.directory)
        assert proof["complete"], proof["failures"]
        seal.complete()
        seal.release()
        value = score(seal.directory)
        write(seal.directory / "score.json", value)
        seal.scored(seal.directory / "score.json")
        yield seal, rebuild(seal.directory)
    finally:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{database}" WITH (FORCE)'))
        admin.dispose()


def test_released_batch_refuses_a_contradictory_safety_source(released_crafted_batch):
    seal, report = released_crafted_batch
    assert report["safety"]["verified"] is False
    assert report["safety"]["refusals_verified"] is False
    assert report["safety"]["reason"] == "safety_sources_invalid"
    for arm, item in report["product_quality"].items():
        for seed, gates in item["by_repeat"].items():
            assert gates["Q-06"] is False, (arm, seed)
            assert gates["Q-11"] is False, (arm, seed)
            assert report["per_repeat"][seed][arm]["gates"]["Q-06"] is True
    assert seal.record()["state"] == "scored"


def test_failure_report_meters_malformed_observations_without_crashing(
    tmp_path, capsys
):
    from procurement_review.eval_h_cli import main
    from procurement_review.eval_h_journal import Journal

    seal = sealed(tmp_path)
    plan = seal.plan()
    case, other = plan.case_ids
    journal = Journal(seal.directory)
    journal.commit(
        dict(
            schema_version="v1-07h-observation.v1",
            case_id=case,
            repeat_seed=101,
            arm="single_agent",
            mode="disposable",
            attempts=[
                dict(
                    branch="joint",
                    attempt_no=1,
                    status="settled",
                    receipt=dict(
                        error_code=None,
                        correctable=False,
                        retryable=False,
                        input_tokens=10,
                        output_tokens=2,
                        usage_unknown=False,
                        returned=True,
                        latency_ms=3,
                        resolved_model=plan.configuration.resolved_model,
                        http_status=200,
                        finish_reason="stop",
                    ),
                ),
                dict(branch="joint", status="started"),
                "not-a-record",
            ],
        )
    )
    journal.commit(
        dict(
            schema_version="v1-07h-observation.v1",
            case_id=other,
            repeat_seed=101,
            arm="single_agent",
            mode="disposable",
            attempts="broken",
        )
    )
    direct = failure_report(seal.directory)
    single = direct["Q-12"]["resources"]["single_agent"]
    assert single["physical_attempts"] == 2
    assert single["proven_sends"] == 1
    assert single["unsettled_attempts"] == 1
    assert direct["timing"]["single_agent"]["planned"] == 4
    assert direct["timing"]["single_agent"]["measured"] == 2

    assert main(["report", str(seal.directory)]) == 0
    capsys.readouterr()
    stored = read(seal.directory / "result.json")
    assert stored["Q-12"]["resources"]["single_agent"]["physical_attempts"] == 2
    failures = stored["Q-12"]["coverage"]["failures"]
    assert any("V107-D-9001.101.single_agent" in item for item in failures), failures
    assert any("V107-D-9002.101.single_agent" in item for item in failures), failures
    assert seal.record()["state"] == "sealed"


def test_failure_report_names_malformed_rows_and_payloads_without_crashing(
    tmp_path, capsys
):
    """A damaged row shape or payload must be named, never interrupt the report."""

    from procurement_review.eval_h_cli import main
    from procurement_review.eval_h_journal import Journal
    from procurement_review.eval_h_score import censored, latency, pipeline_failed

    legal = dict(
        terminal_status=200,
        status="pending_human_review",
        elapsed_ms=1000,
        pipeline_failures=[],
        error_code=None,
        branches={
            "policy": dict(outcome="completed"),
            "contract": dict(outcome="completed"),
        },
    )
    assert pipeline_failed({**legal, "branches": []}) is False
    assert pipeline_failed({**legal, "branches": None}) is False
    assert (
        censored(
            {
                "row": {**legal, "pipeline_failures": None},
                "arm": "single_agent",
                "attempts": [],
            }
        )
        is False
    )
    assert (
        latency(
            [{"row": {**legal, "branches": []}, "arm": "multi_agent", "attempts": []}],
            1,
        )["measured"]
        == 1
    )
    # An unreadable code list cannot censor, but a declared failure still counts.
    assert latency(
        [
            {
                "row": {**legal, "pipeline_failures": 7},
                "arm": "single_agent",
                "attempts": [],
            }
        ],
        1,
    ) == {
        "planned": 1,
        "measured": 1,
        "all_sample_p95_ms": None,
        "success_p95_ms": None,
        "maximum_ms": None,
        "failed": 1,
        "cancelled": 0,
        "censored": 0,
        "deadline_ms": 120000,
        "deadline_satisfied": None,
        "meets_threshold": None,
    }

    seal = sealed(tmp_path)
    plan = seal.plan()
    case, other = plan.case_ids
    journal = Journal(seal.directory)
    journal.commit(
        dict(
            schema_version="v1-07h-observation.v1",
            case_id=case,
            repeat_seed=101,
            arm="multi_agent",
            mode="disposable",
            row={**legal, "branches": []},
            attempts=[],
        )
    )
    journal.commit(
        dict(
            schema_version="v1-07h-observation.v1",
            case_id=case,
            repeat_seed=101,
            arm="single_agent",
            mode="disposable",
            row={**legal, "pipeline_failures": None},
            attempts=[],
        )
    )
    damaged = journal.path(other, 101, "multi_agent")
    write_bytes(damaged, b"[1, 2, 3]\n")
    armless = journal.path(other, 101, "single_agent")
    write(
        armless,
        {
            "schema_version": "v1-07h-observation.v1",
            "case_id": other,
            "repeat_seed": 101,
            "elapsed_ms": 1000,
            "row": dict(legal),
            "attempts": [],
        },
    )
    with pytest.raises(ValueError, match="h_observation_commit_payload"):
        journal.committed(other, 101, "multi_agent")

    direct = failure_report(seal.directory)
    failures = direct["Q-12"]["coverage"]["failures"]
    assert direct["Q-12"]["coverage"]["complete"] is False
    assert any(item.endswith("observation_row_branches") for item in failures), failures
    assert any(item.endswith("not_an_object") for item in failures), failures
    assert any(
        item.startswith("unregistered_or_duplicate_observation:") for item in failures
    ), failures
    # A pair whose records lack the shared evidence keys is named, not crashed.
    assert any(item.startswith("unpaired_sources:") for item in failures), failures
    assert any(item.startswith("unpaired_retrieval:") for item in failures), failures
    assert any(item.startswith("unpaired_actual_facts:") for item in failures), failures
    # The armless record is named above and is not attributed to either arm.
    assert direct["timing"]["single_agent"]["measured"] == 1
    assert direct["timing"]["multi_agent"]["measured"] == 1

    assert main(["report", str(seal.directory)]) == 0
    capsys.readouterr()
    stored = read(seal.directory / "result.json")
    assert stored["Q-12"]["complete"] is False
    assert stored["Q-12"]["coverage"]["complete"] is False
    assert seal.record()["state"] == "sealed"


def test_handwritten_later_state_cannot_open_labels_without_completion(tmp_path):
    seal = sealed(tmp_path)
    record = seal.record()
    record.update(
        state="labels_released",
        transitions=[
            "candidate_frozen",
            "generated",
            "sealed",
            "run_complete",
            "labels_released",
        ],
    )
    write(seal.directory / "state.json", record, replace=True)
    with pytest.raises(ValueError, match="completion_proof"):
        seal.scorer_labels()


@pytest.mark.parametrize("changed", [False, True])
def test_refusal_source_digest_binding_survives_reads(tmp_path, changed):
    """The registered digest survives unchanged reads and fails on rewrite."""
    product = "a" * 64
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    source = tmp_path / "source.json"
    write(source, safety_source(product))
    write(private / "safety.json", safety_summary(product, {str(source): sha(source)}))
    plan = SimpleNamespace(
        product_sha256=product,
        mode="disposable",
        lifecycle_evidence_sha256=sha(private / "safety.json"),
    )
    assert (
        safety(tmp_path, plan)["verified"]
        and safety(tmp_path, plan)["refusals_verified"]
    )
    if changed:
        write(source, safety_source(product, mode="other"), replace=True)
    else:
        source.rename(tmp_path / "saved-original.json")
    verdict = safety(tmp_path, plan)
    assert not verdict["verified"] and not verdict["refusals_verified"]


def test_cli_registration_binding_and_early_transitions(tmp_path, capsys):
    import json
    from pathlib import Path
    from procurement_review.eval_h_cli import main
    from procurement_review.eval_h_fixture import make_disposable

    source = Path(
        make_disposable(
            tmp_path / ".local/source", 2, batch_id="disposable-h-registration-test"
        )["root"]
    )
    root = tmp_path / ".local/new-registration"
    args = ["create", str(root), "--plan", str(source / "plan.json")]
    for name, file in (
        ("recipe", "recipe.json"),
        ("seed", "seed.bin"),
        ("exclusion", "registry.json"),
        ("safety", "safety.json"),
    ):
        args.extend(["--" + name, str(source / "private" / file)])
    assert main(args) == 0
    target = Path(json.loads(capsys.readouterr().out)["root"])
    for command in ("generate", "seal"):
        assert main([command, str(target)]) == 0
        capsys.readouterr()
    for command, reason in (
        ("complete", "run_incomplete"),
        ("release", "expected:run_complete"),
        ("score", "labels_not_released"),
    ):
        with pytest.raises(ValueError, match=reason):
            main([command, str(target)])
    altered = tmp_path / "different-recipe.json"
    write(altered, {})
    args[1] = str(tmp_path / ".local/refused-registration")
    args[args.index("--recipe") + 1] = str(altered)
    with pytest.raises(ValueError, match="registration_input_binding"):
        main(args)
    assert not Path(args[1]).exists()


def test_formal_quotas_use_only_abstract_descriptors_without_material_generation():
    from procurement_review.eval_h_contracts import HRecipe, QUOTAS

    labels, humans = [], []
    for name, count in QUOTAS.items():
        for number in range(count):
            label = SimpleNamespace(
                recommendation=SimpleNamespace(value=name),
                high_risk=name == "recommend_reject",
                conflict_pairs=(),
                security_events=(),
            )
            if name == "needs_human_review":
                label.high_risk = number < 20
                label.conflict_pairs = ("descriptor",) if number < 30 else ()
                label.security_events = ("descriptor",) if number >= 30 else ()
                humans.append(label)
            labels.append(label)
    descriptors = SimpleNamespace(cases=tuple(range(200)), labels=labels)
    HRecipe.require_formal_quotas(descriptors)
    humans[0].high_risk = False
    with pytest.raises(ValueError, match="high_risk_quota"):
        HRecipe.require_formal_quotas(descriptors)
    humans[0].high_risk = True
    humans[0].security_events = ("descriptor",)
    humans[30].security_events = ()
    with pytest.raises(ValueError, match="events_overlap"):
        HRecipe.require_formal_quotas(descriptors)


def test_run_level_failure_is_named_and_measured_stops_stay_valid(tmp_path):
    """A failed Run keeps its denominator once its provenance proves the failure."""

    from procurement_review.eval_h_consumption import observation_failures
    from procurement_review.eval_h_contracts import plan_generation_deadline
    from procurement_review.precise_run_scope import execution_snapshot

    seal = sealed(tmp_path)
    plan = seal.plan()
    case = seal.runner_cases()[0]
    seed = plan.repeat_seeds[0]
    started = "2026-06-30T00:00:00+00:00"
    finished = "2026-06-30T00:00:01+00:00"
    frozen = execution_snapshot(
        mode="test_only",
        config=plan.configuration.model_copy(update={"seed": seed}),
        registration=SimpleNamespace(
            rules_version="3.1.0",
            document_version=plan.document_version,
            dataset_version=plan.dataset_version,
            index_version=plan.index_version,
        ),
        branch_timeout_seconds=90.0,
        max_attempts_per_branch=2,
        generation_deadline_seconds=plan_generation_deadline(plan),
    )
    # The provenance a real failed Run carries: the Run's own identity and Trace, the
    # frozen snapshot, the lifecycle event that ended it, its closed generation, its
    # retrieval record and the two persisted timestamps the elapsed reading was computed
    # from.  The strengthened failure contract asks for every one of them by name, so the
    # fixture states them the way the runner persists them.
    run_id = "run-v107h-boundary-failure"
    trace_id = "trace-v107h-boundary-failure"

    def event(event_type, step, status, **extra):
        return {
            "run_id": run_id,
            "trace_id": trace_id,
            "event_type": event_type,
            "step": step,
            "generation": None,
            "attempt": None,
            "worker": None,
            "occurred_at": finished,
            "output_summary": {"status": status},
            **extra,
        }

    def generation(status):
        return {
            "run_id": run_id,
            "generation": 1,
            "status": status,
            "created_at": started,
            "deadline_at": finished,
        }

    no_hits = {
        role: {"original": [], "without_case_id": [], "model_sources": []}
        for role in ("policy", "contract")
    }
    provenance = dict(
        case_id=case.case_id,
        trace_id=trace_id,
        resume_count=0,
        summary=None,
        execution=None,
        status="failed",
        error_code="review_execution_failed",
        snapshot=frozen.model_dump(mode="json"),
        events=[event("run_failed", "review_run", "failed")],
        generations=[generation("cancelled")],
        attempts=[],
        checkpoints=[],
        created_at=started,
        updated_at=finished,
    )
    body = dict(
        schema_version="v1-07h-observation.v1",
        plan_sha256=__import__(
            "procurement_review.eval_h_seal", fromlist=["plan_digest"]
        ).plan_digest(plan),
        candidate_sha256=plan.candidate_sha256,
        consumer_sha256=plan.consumer_sha256,
        case_id=case.case_id,
        repeat_seed=seed,
        arm="multi_agent",
        mode="disposable",
        started_at=started,
        observed_at=finished,
        elapsed_ms=1000,
        wall_elapsed_ms=1000,
        material_sha256=payload_sha256(case.model_dump(mode="json")),
        sources=source_identities(plan, case),
        retrieval=no_hits,
        row=dict(
            run_id=run_id,
            status="failed",
            error_code="review_execution_failed",
            terminal_status=200,
            branches={},
            pipeline_failures=[],
        ),
        binding=dict(provenance),
        attempts=[],
    )
    errors = observation_failures(plan, case, body)
    assert any(
        item == "product_run_failed:review_execution_failed" for item in errors
    ), errors
    assert all(item.startswith("product_run_") for item in errors), errors
    interrupted = copy.deepcopy(body)
    interrupted["row"]["status"] = "interrupted"
    interrupted["row"]["error_code"] = "generation_deadline_exceeded"
    interrupted["binding"]["status"] = "interrupted"
    interrupted["binding"]["error_code"] = "generation_deadline_exceeded"
    interrupted["binding"]["events"] = [
        event("step_failed", "review_workflow", "interrupted")
    ]
    interrupted["binding"]["generations"] = [generation("expired")]
    errors = observation_failures(plan, case, interrupted)
    assert any(item.startswith("product_run_interrupted") for item in errors), errors
    assert all(item.startswith("product_run_") for item in errors), errors
    # Erasing the provenance is refused by name instead of being accepted as a failure.
    stripped = copy.deepcopy(body)
    stripped["binding"] = dict(case_id=case.case_id, summary=None, execution=None)
    errors = observation_failures(plan, case, stripped)
    assert "product_run_failed:review_execution_failed" not in errors, errors
    assert "product_failed_snapshot" in errors, errors
    # A measured stop reason is a recorded outcome, not a damaged record.
    body["row"] = dict(body["row"], status="pending_human_review", error_code=None)
    body["row"]["stop_reason"] = "citation_outside_branch"
    errors = observation_failures(plan, case, body)
    assert all("batch_stopped_trust_boundary" not in item for item in errors), errors
    # A trust-boundary stop still invalidates the record.
    body["row"]["stop_reason"] = "model_identity_drift"
    errors = observation_failures(plan, case, body)
    assert "batch_stopped_trust_boundary" in errors, errors
    # A slower commit is not a forged clock; a fabricated one still is.
    body["row"].pop("stop_reason")
    body["wall_elapsed_ms"] = 3000
    errors = observation_failures(plan, case, body)
    assert "observation_clock" not in errors, errors
    body["wall_elapsed_ms"] = 60_000
    errors = observation_failures(plan, case, body)
    assert "observation_clock" in errors, errors
    # The completion proof keeps a measured Run failure out of its failure list.
    body["wall_elapsed_ms"] = 1000
    body["row"] = dict(
        body["row"],
        status="failed",
        error_code="review_execution_failed",
        pipeline_failures=[],
    )
    errors = observation_failures(plan, case, body)
    proof_failures = [item for item in errors if not item.startswith("product_run_")]
    assert any(item.startswith("product_run_failed:") for item in errors), errors
    assert all(not item.startswith("product_run_") for item in proof_failures), errors
