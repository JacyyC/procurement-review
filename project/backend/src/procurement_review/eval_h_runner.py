"""H paired execution. The runner never opens labels, recipe, seed or seal key."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from pydantic import SecretStr

from .authorization import generate_credential
from .business_v31_review import finalize_review_v31
from .business_v3_procurement import review_procurement_v3
from .business_v3_security import ProtectedOutputV3
from .eval_h_contracts import plan_generation_deadline
from .eval_h_projection import project
from .eval_h_identity import require_identity
from .eval_h_journal import Journal
from .eval_h_joint import execute
from .eval_h_material import seed_plan, source_identities
from .eval_h_product import admission, multi
from .eval_h_seal import HSeal, plan_digest
from .eval_h_storage import locked, private, read, write
from .gold_v3_contracts import payload_sha256
from .mcp_server import create_procurement_mcp_server
from .precise_branch_records import failed_document
from .precise_dev_sweep import _summarise
from .precise_run_scope import (
    branch_inputs,
    procurement_worker_input,
    PRECISE_QUESTIONS,
)
from .precise_run_summary import business_context
from .precise_run_transport import PreciseRunChatTransport
from .procurement_mcp_client import OfficialProcurementMCPClient
from .retrieval import RetrievalRequest
from .review_store import ReviewStore
from .eval_h_batch_stop import (
    committed_batch_stop,
    read_batch_stop,
    write_batch_stop,
)
from .run_stop import BatchStopController, BatchStoppedSignal
from .scoped_runtime import ScopedReviewRuntime, ScopedRetrievalPort


def setup(directory, database_url):
    seal = HSeal(directory)
    plan, cases = seal.plan(), seal.runner_cases()
    require_identity(plan)
    path = seal.directory / "private/runtime.json"
    if path.exists():
        private(path)
        return {"reused": True}
    # A supplied database must be explicitly dedicated to this batch.
    from sqlalchemy.engine import make_url

    url = make_url(database_url)
    if url.database != "v107h_" + plan.batch_id.replace("-", "_"):
        raise ValueError("h_dedicated_database_name_required")
    store = ReviewStore.from_url(database_url)
    try:
        store.migrate()
        store.seed(seed_plan(plan, cases))
        token, credential_id, digest = generate_credential()
        now = datetime.now(UTC)
        store.issue_credential(
            credential_id=credential_id,
            tenant_id=plan.tenant_id,
            subject_id=plan.subject_id,
            token_digest=digest,
            issued_at=now,
            expires_at=now + timedelta(hours=12),
        )
        write(
            path,
            dict(database_url=database_url, token=token, plan_sha256=plan_digest(plan)),
        )
    finally:
        store.close()
    return {"seeded": len(cases), "mode": plan.mode}


async def assembled(store, token, plan, case):
    def admit(*args, **kwargs):
        admission(store, token, plan, case.case_id)

    _context, registration, scope = admission(store, token, plan, case.case_id)
    runtime = ScopedReviewRuntime.build(
        store=store,
        scope=scope,
        admission=admit,
        vector_mode="none",
        lexical_adapter="idf",
    )
    from mcp import Client

    async with Client(
        create_procurement_mcp_server(runtime.procurement, scope=scope),
        raise_exceptions=True,
    ) as client:
        facts = await OfficialProcurementMCPClient(client).collect_review_facts(
            procurement_worker_input(registration, case.case_id)
        )
    context = business_context(scope, facts)
    inputs = branch_inputs(
        context=context,
        case_id=case.case_id,
        store=store,
        scope=scope,
        admission=admit,
        runtime=runtime,
    )
    retrieval = {}
    for role, branch in inputs.items():
        port = ScopedRetrievalPort.from_store(
            store=store,
            scope=scope,
            admission=admit,
            vector_mode="none",
            lexical_adapter="idf",
            category=role,
        )
        # Product questions contain no Case identifier. Both registered variants
        # are nevertheless executed and recorded separately.
        retrieval[role] = {}
        for variant in ("original", "without_case_id"):
            query = (
                PRECISE_QUESTIONS[role].replace(case.case_id, "")
                if variant == "without_case_id"
                else PRECISE_QUESTIONS[role]
            )
            hits = port.retrieve(
                RetrievalRequest(
                    query=query,
                    index_version=scope.index_version,
                    limit=5,
                    minimum_results=1,
                )
            ).results
            retrieval[role][variant] = [
                dict(
                    source_id=hit.source_id,
                    version=hit.source_version,
                    locator=hit.chunk_id,
                    content_sha256=hit.content_checksum[7:],
                )
                for hit in hits
            ]
        retrieval[role]["model_sources"] = [
            source.identity.model_dump(mode="json") for source in branch.slices
        ]
    admit()
    return context, facts, inputs, retrieval, admit


async def single(store, token, plan, case, seed, transport, journal, *, stop=None):
    context, facts, inputs, retrieval, admit = await assembled(store, token, plan, case)
    grounded, attempts = await execute(
        journal,
        plan,
        case,
        seed,
        transport,
        inputs,
        facts,
        admit,
        stop=stop,
        deadline_seconds=plan_generation_deadline(plan),
    )
    failed = grounded is None
    grounded = grounded or {role: failed_document(role) for role in inputs}
    decision = finalize_review_v31(
        context,
        grounded["policy"],
        grounded["contract"],
        review_procurement_v3(context, facts, authorized_scope_id=context.scope_id),
        sources=tuple(source for branch in inputs.values() for source in branch.slices),
        protection=ProtectedOutputV3(),
        pipeline_failures=("document_model_failed",) if failed else (),
        control_events=(),
    )
    # Reuse the bounded product projection, without claiming a product Run or transaction.
    execution = dict(
        decision=decision.business.model_dump(mode="json"),
        conflicts=[pair.model_dump(mode="json") for pair in decision.conflicts],
        security=[event.model_dump(mode="json") for event in decision.security],
    )
    for role in inputs:
        execution[role] = dict(
            spans=[span.model_dump(mode="json") for span in grounded[role].spans]
        )
    status = (
        "failed"
        if failed
        else "pending_human_review"
        if decision.business.recommendation
        in {"needs_human_review", "insufficient_information"}
        else "pending_confirmation"
    )
    row = _summarise(dict(execution=execution, status=status))
    row["fact_observations"] = []
    from .eval_f_lineage import native_facts_v3

    row["fact_observations"] = native_facts_v3(execution)
    native = {"execution": execution}
    row = project(row, native)
    row.update(
        terminal_status=200,
        branches={
            role: dict(outcome="failed" if failed else "completed", attempts=[])
            for role in inputs
        },
    )
    return (
        row,
        dict(
            admission_contract="h-evaluation-live-scope.v1",
            product_transactions_claimed=False,
            execution=execution,
            sources={role: retrieval[role]["model_sources"] for role in inputs},
            facts=[fact.model_dump(mode="json") for fact in facts.facts],
        ),
        retrieval,
        attempts,
    )


async def run(
    directory: Path,
    *,
    transport_override=None,
    resume=False,
    real_authorization=None,
    fault=None,
):
    """Execute one sealed batch; ``fault`` is the test-only product fault controller."""
    seal = HSeal(directory)
    plan = seal.plan()
    require_identity(plan)
    if seal.record()["state"] != "sealed":
        raise ValueError("h_run_requires_sealed")
    if plan.mode == "disposable" and not isinstance(
        transport_override, httpx.MockTransport
    ):
        raise ValueError("h_disposable_requires_mock_transport")
    if plan.mode == "formal":
        from .eval_h_cli import preflight

        if transport_override is not None or real_authorization != plan.authorization:
            raise ValueError("h_explicit_formal_send_authorization_required")
        if not preflight(directory, runner=True)["ready_for_formal"]:
            raise ValueError("h_formal_preflight_incomplete")
    elif plan.mode != "disposable" and transport_override is None:
        # A probe batch that would talk to the real endpoint needs its own explicit
        # authorization string; a mock transport is the offline validation path.
        if real_authorization != plan.authorization:
            raise ValueError("h_explicit_probe_send_authorization_required")
    private(seal.directory / "private/runtime.json")
    state = read(seal.directory / "private/runtime.json")
    if state["plan_sha256"] != plan_digest(plan):
        raise ValueError("h_runtime_plan_identity")
    if read_batch_stop(seal.directory, plan) is not None or (
        committed_batch_stop(seal.directory, plan) is not None
    ):
        # The batch stopped sending in an earlier process and the reason is durable — in
        # the batch's own stop record and in the observation that recorded the stop.  A
        # new controller, a new process and a repeated CLI call all read the same state:
        # the stop is not cleared by re-entering, and no queue restarts underneath it.
        # This is where a batch-level stop differs from a normal Run interruption — the
        # latter is recovered inside its own batch (committed branches reused, remaining
        # budget spent) — and continuing a stopped batch would need a contract-registered
        # new pre-registration, which this contract does not define.
        raise ValueError("h_batch_stopped_explicit_continuation_required")
    journal = Journal(directory)
    cases = {case.case_id: case for case in seal.runner_cases()}
    with locked(seal.directory / "runner", blocking=False):
        store = ReviewStore.from_url(state["database_url"])
        stop = asyncio.Event()
        capacity = asyncio.Semaphore(2)
        controller = BatchStopController(
            # The stop policy comes from the plan's own version contract: every version
            # answers for itself, so a plan that registered the dense-429 window is never
            # run without it, and a version that never registered one stays readable.
            policy=plan.stop_policy(),
            on_stop=lambda stopping: write_batch_stop(seal.directory, plan, stopping),
        )

        async def case_run(case_id, seed):
            async with capacity:
                if controller.stopped:
                    # A batch that already stopped never starts another queued task; the
                    # observation stays absent instead of being sent late.
                    return
                try:
                    await arms(case_id, seed)
                except BaseException:
                    stop.set()
                    raise

        async def arms(case_id, seed):
            for arm in plan.arms():
                if stop.is_set() or controller.stopped:
                    return
                if journal.committed(case_id, seed, arm):
                    continue
                case = cases[case_id]
                config = plan.configuration.model_copy(update={"seed": seed})
                from .model_probe_run import credential

                transport = PreciseRunChatTransport(
                    config,
                    SecretStr("h-offline-only") if transport_override else credential(),
                    catalog=None,
                    transport=transport_override,
                )
                begin = time.monotonic()
                started = datetime.now(UTC).isoformat()
                if arm == "multi_agent":
                    row, binding = await multi(
                        store,
                        state["token"],
                        plan,
                        case,
                        seed,
                        transport,
                        resume=resume,
                        fault=fault,
                        stop=controller,
                    )
                    _context, facts, _inputs, retrieval, _admit = await assembled(
                        store, state["token"], plan, case
                    )
                    from .eval_h_product_checks import product_attempts

                    attempts = product_attempts(binding)
                    binding["retrieval"] = retrieval
                    binding["facts"] = [
                        fact.model_dump(mode="json") for fact in facts.facts
                    ]
                else:
                    try:
                        row, binding, retrieval, attempts = await single(
                            store,
                            state["token"],
                            plan,
                            case,
                            seed,
                            transport,
                            journal,
                            stop=controller,
                        )
                    except BatchStoppedSignal:
                        # The batch stopped before this arm's next physical attempt: the
                        # observation stays absent instead of being sent late, and the
                        # batch continues winding down without cancelling other cases.
                        return
                wall_elapsed = int((time.monotonic() - begin) * 1000)
                elapsed = (
                    int(
                        (
                            datetime.fromisoformat(
                                binding["summary"]["created_at"]
                                if binding["summary"]
                                else binding["updated_at"]
                            )
                            - datetime.fromisoformat(binding["created_at"])
                        ).total_seconds()
                        * 1000
                    )
                    if arm == "multi_agent"
                    else wall_elapsed
                )
                journal.commit(
                    dict(
                        schema_version="v1-07h-observation.v1",
                        plan_sha256=plan_digest(plan),
                        candidate_sha256=plan.candidate_sha256,
                        consumer_sha256=plan.consumer_sha256,
                        case_id=case_id,
                        repeat_seed=seed,
                        arm=arm,
                        mode=plan.mode,
                        started_at=started,
                        observed_at=datetime.now(UTC).isoformat(),
                        elapsed_ms=elapsed,
                        wall_elapsed_ms=wall_elapsed,
                        material_sha256=payload_sha256(case.model_dump(mode="json")),
                        sources=source_identities(plan, case),
                        retrieval=retrieval,
                        row=row,
                        binding=binding,
                        attempts=attempts,
                    )
                )
                if row.get("stop_reason") or row.get("status") in {
                    "interrupted",
                    "failed",
                    "cancelled",
                }:
                    # The committed observation is reported to the one batch decision:
                    # a measured stop reason keeps its place, anything else — including
                    # an unregistered reason — stops the batch at the next gate.
                    controller.note_stop_reason(row.get("stop_reason"))
                    controller.note_run(row.get("status"), row.get("error_code"))
                    if controller.stopped:
                        # The batch winds down instead of cancelling anything: this
                        # case's remaining arms are skipped by the check above, other
                        # in-flight cases finish their admitted attempt and end at the
                        # gate, and the queued cases never start.
                        stop.set()
                        return

        async def batch():
            for seed in plan.repeat_seeds:
                async with asyncio.TaskGroup() as tasks:
                    for case_id in plan.case_ids:
                        tasks.create_task(case_run(case_id, seed))

        try:
            await batch()
        finally:
            store.close()
            if controller.stopped:
                # The transition already persisted the stop; rewriting it here records the
                # most severe reason the whole batch reached, and re-writes the record if
                # the very first write was lost with a killed process.
                write_batch_stop(seal.directory, plan, controller)
    policy = plan.stop_policy()
    return {
        "committed": sum(
            journal.committed(*key) is not None for key in plan.observation_keys()
        ),
        "mode": plan.mode,
        "stopped": controller.stopped,
        "stop_reason": controller.reason,
        "stop_summary": controller.summary(),
        "stop_policy": (
            None
            if policy is None
            else {
                "rate_limit_window_seconds": policy.rate_limit_window_seconds,
                "rate_limit_burst_threshold": policy.rate_limit_burst_threshold,
            }
        ),
        "stage_acceptance": False,
    }
