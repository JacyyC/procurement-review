"""H uses the protected product service and exports a fresh read-only DB snapshot."""

import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy import text

from .api_contracts import ReviewCreateRequestV2, ResumeRequestV2
from .authorization import AuthorizationService
from .eval_h_contracts import plan_generation_deadline
from .precise_dev_sweep import _summarise
from .precise_run_runner import PreciseScopedRunner
from .review_repository import ReviewRepository
from .review_service import ReviewService
from .review_store import ReviewStore
from .security_contracts import SecurityOperation


def admission(store, token, plan, case_id, *, deadline_seconds=None):
    auth = AuthorizationService(store=store)
    context = auth.resolve(
        token, request_id="h-evaluation", operation=SecurityOperation.REVIEW_CREATE
    )
    if (context.tenant_id, context.subject_id) != (plan.tenant_id, plan.subject_id):
        raise ValueError("h_runtime_authority_identity")
    registration = auth.require_case(
        context,
        case_id,
        request_id="h-evaluation",
        operation=SecurityOperation.REVIEW_CREATE,
    )
    scope = auth.build_runtime_scope(
        context,
        registration,
        generation=1,
        deadline_at=auth.runtime_scope_deadline(
            plan_generation_deadline(plan)
            if deadline_seconds is None
            else deadline_seconds
        ),
    )
    return context, registration, scope


def source_snapshot(store, run_id):
    def rows(connection, table, ordering):
        return [
            dict(row)
            for row in connection.execute(
                text(f"SELECT * FROM {table} WHERE run_id=:run ORDER BY {ordering}"),
                {"run": run_id},
            ).mappings()
        ]

    with (
        store.engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        ) as connection,
        connection.begin(),
    ):
        connection.execute(text("SET TRANSACTION READ ONLY"))
        header = rows(connection, "review_run", "run_id")[0]
        summaries = rows(connection, "review_run_summary", "run_id")
        summary = summaries[0] if summaries else None
        run = dict(
            case_id=header["case_id"],
            trace_id=header["trace_id"],
            status=header["status"],
            created_at=header["created_at"],
            updated_at=header["updated_at"],
            error_code=header["error_code"],
            resume_count=header["resume_count"],
            snapshot=header["isolation"]["execution_snapshot"],
            execution=summary["execution_payload"] if summary else {},
            summary={
                name: summary[name]
                for name in ("summary_checksum", "summary_payload", "created_at")
            }
            if summary
            else None,
            attempts=rows(
                connection, "review_worker_attempt", "generation,worker,attempt_no"
            ),
            checkpoints=rows(connection, "review_branch_checkpoint", "worker"),
            generations=rows(connection, "review_run_generation", "generation"),
            events=rows(connection, "review_trace_event", "event_ordinal"),
        )
        run["projection"] = _summarise(
            dict(
                execution=run["execution"],
                status=run["status"],
                summary=run["summary"]["summary_payload"] if summary else None,
                execution_mode=run["snapshot"]["mode"],
                error_code=header["error_code"],
            )
        )
    return json.loads(json.dumps(run, default=lambda value: value.isoformat()))


async def multi(
    store, token, plan, case, seed, transport, *, resume=False, fault=None, stop=None
):
    deadline_seconds = plan_generation_deadline(plan)
    context, registration, scope = admission(
        store, token, plan, case.case_id, deadline_seconds=deadline_seconds
    )
    service = ReviewService(
        repository=ReviewRepository(store, fault=fault),
        fault=fault,
        runner=PreciseScopedRunner(
            store=store,
            transport=transport,
            generation_deadline_seconds=deadline_seconds,
            mode="test_only" if plan.mode == "disposable" else "real",
            stop=stop,
        ),
        # The service creates the Run's generation and its lease; it must freeze the same
        # deadline the runner and the expected snapshot are built with, or every V1-08
        # Run stops as checkpoint_incompatible before its first send.
        generation_deadline_seconds=deadline_seconds,
    )
    try:
        created, _ = await service.create_review(
            ReviewCreateRequestV2(
                api_version="review-api.v2",
                case_id=case.case_id,
                case_version=registration.case_version,
            ),
            registration=registration,
            scope=scope,
            context=context,
            idempotency_key=f"{plan.batch_id}-{case.case_id}-{seed}",
        )
        resumed = False
        while True:
            detail = service.get_review(created.run_id, context=context).model_dump(
                mode="json"
            )
            if detail["status"] == "interrupted" and resume and not resumed:
                auth = AuthorizationService(store=store)
                resume_context = auth.resolve(
                    token,
                    request_id="h-resume",
                    operation=SecurityOperation.REVIEW_RESUME,
                )
                auth.require_case(
                    resume_context,
                    case.case_id,
                    request_id="h-resume",
                    operation=SecurityOperation.REVIEW_RESUME,
                )
                await service.resume_review(
                    created.run_id,
                    ResumeRequestV2(
                        api_version="review-api.v2", expected_version=detail["revision"]
                    ),
                    context=resume_context,
                    idempotency_key=f"{plan.batch_id}-{case.case_id}-{seed}-resume-{detail['revision']}",
                )
                resumed = True
                continue
            if detail["status"] != "processing":
                break
            await asyncio.sleep(0.02)
        # A Run that expired its generation window is a normal timeout: the product
        # recorded it as interrupted and its remaining branches are not resent.  It keeps
        # its place in the denominator as a measured failure instead of stopping the
        # batch; only the registered trust-boundary stops halt sending.
        fresh = ReviewStore.from_url(
            store.engine.url.render_as_string(hide_password=False)
        )
        try:
            binding = source_snapshot(fresh, created.run_id)
            reread = ReviewService(
                repository=ReviewRepository(fresh), runner=service._runner
            )
            fresh_detail = reread.get_review(
                created.run_id, context=context
            ).model_dump(mode="json")
            if _summarise(fresh_detail) != binding["projection"]:
                raise ValueError("h_fresh_product_projection_mismatch")
        finally:
            fresh.close()
        from .eval_h_projection import project

        row = project(binding["projection"], binding)
        row.update(
            terminal_status=200,
            run_id=created.run_id,
            service_created_at=binding["created_at"],
            observed_at=datetime.now(UTC).isoformat(),
        )
        return row, binding
    finally:
        await service.shutdown()
