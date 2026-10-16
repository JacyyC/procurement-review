"""The API view of one persisted review run.

The projection reads only committed facts and rebuilds the immutable Case input a
 Run was created with, so a new process answers the same request with the same view and
 no in-process execution state is ever part of an answer.
"""

from __future__ import annotations

from .api_contracts import (
    BranchFailure,
    ReceiptViewV2,
    RecoveryView,
    ReviewRunResponseV2,
)
from .contracts import (
    BusinessFact,
    CaseType,
    EvidenceRef,
    ReviewLifecycleStatus,
    ReviewRequest,
    ReviewSummary,
)
from .precise_run_contracts import ExecutionPayload, ExecutionRecord
from .record_guard import validate_stored_model
from .review_repository import PersistedRun
from .review_store import CaseRegistration
from .run_errors import RecordIncompleteError
from .trace import canonical_checksum
from .worker_contracts import WorkerOutcome


def build_run_response(
    run: PersistedRun,
    *,
    resume_limit: int,
) -> ReviewRunResponseV2:
    header = run.header
    isolation = dict(header.isolation)
    request = ReviewRequest(
        case_id=header.case_id,
        case_type=CaseType(str(isolation.get("case_type", "purchase_order"))),
        document_version=header.document_version,
        purchase_order_id=str(isolation.get("purchase_order_id", "")),
        dataset_version=header.dataset_version,
        initiator=header.created_by_subject,
    )
    execution: ExecutionPayload | None = None
    summary: ReviewSummary | None = None
    if run.execution_payload is not None:
        execution = validate_stored_model(ExecutionRecord, run.execution_payload).root
        summary = execution.summary
    elif run.summary_payload is not None:
        summary = validate_stored_model(ReviewSummary, run.summary_payload)
    failures: list[BranchFailure] = []
    if execution is not None and execution.execution_version == "review-execution.v1":
        for result in execution.branch_results:
            if result.outcome is WorkerOutcome.FAILED:
                failures.append(
                    BranchFailure(
                        worker=result.worker,
                        code=(
                            result.failure_code.value
                            if result.failure_code is not None
                            else "unknown_failure"
                        ),
                        detail=result.failure_detail or "Worker failed closed.",
                    )
                )
    evidence: dict[str, EvidenceRef] = {}
    facts: dict[str, BusinessFact] = {}
    if summary is not None:
        for finding in summary.findings:
            for item in finding.evidence_refs:
                evidence.setdefault(_stable_key(item), item)
            for item in finding.business_fact_refs:
                facts.setdefault(_stable_key(item), item)
    receipt = (
        validate_stored_model(ReceiptViewV2, run.receipt_payload)
        if run.receipt_payload is not None
        else None
    )
    trace_id = run.persisted_trace_id
    if trace_id is None:
        # A Run with no committed Trace identity is incomplete; the projection never
        # invents a second identity from the Run ID (A-12).
        raise RecordIncompleteError(
            f"run {header.run_id} has no persisted trace identity"
        )
    return ReviewRunResponseV2(
        run_id=header.run_id,
        trace_id=trace_id,
        status=header.status,
        version=header.revision,
        revision=header.revision,
        request=request,
        summary=summary,
        summary_checksum=run.summary_checksum,
        execution=execution,
        branch_failures=tuple(
            sorted(failures, key=lambda item: (item.worker.value, item.code))
        ),
        evidence_refs=tuple(evidence[key] for key in sorted(evidence)),
        business_facts=tuple(facts[key] for key in sorted(facts)),
        receipt=receipt,
        created_at=header.created_at,
        updated_at=header.updated_at,
        trace_event_count=run.trace_event_count,
        audit_event_count=run.audit_event_count,
        error_code=header.error_code,
        error_message=header.error_message,
        recovery=RecoveryView(
            resume_count=header.resume_count,
            resume_limit=resume_limit,
            resumable=(
                header.status is ReviewLifecycleStatus.INTERRUPTED
                and header.resume_count < resume_limit
            ),
            generation=run.generation.generation if run.generation else 0,
            lease_expires_at=(
                run.generation.lease_expires_at if run.generation is not None else None
            ),
            adopted_workers=tuple(sorted(run.checkpoints)),
            attempt_totals=_attempt_totals(run),
            interrupted_reason=(
                header.error_code
                if header.status is ReviewLifecycleStatus.INTERRUPTED
                else None
            ),
        ),
    )


def registration_from_run(run: PersistedRun) -> CaseRegistration:
    """Rebuild the immutable input registration a Run was created with."""

    isolation = dict(run.header.isolation)
    return CaseRegistration(
        tenant_id=run.header.tenant_id,
        case_id=run.header.case_id,
        label=run.header.case_id,
        case_type=str(isolation.get("case_type", CaseType.PURCHASE_ORDER.value)),
        scenario=str(isolation.get("scenario", "low")),
        purchase_order_id=str(isolation.get("purchase_order_id", "")),
        case_version=run.header.case_version,
        document_version=run.header.document_version,
        dataset_version=run.header.dataset_version,
        index_version=run.header.index_version,
        rules_version=run.header.rules_version,
        checksum=run.header.request_checksum,
        case_document_ids=tuple(isolation.get("case_document_ids", ())),
        shared_document_ids=tuple(isolation.get("shared_document_ids", ())),
        allowed_order_ids=tuple(isolation.get("allowed_order_ids", ())),
        allowed_supplier_ids=tuple(isolation.get("allowed_supplier_ids", ())),
        allowed_transaction_ids=tuple(isolation.get("allowed_transaction_ids", ())),
        granted_subject_ids=(run.header.current_subject_id,),
    )


def _attempt_totals(run: PersistedRun) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in run.attempts:
        totals[item.worker] = totals.get(item.worker, 0) + 1
    return dict(sorted(totals.items()))


COMPLETE_STATUSES = {
    ReviewLifecycleStatus.CONFIRMED,
    ReviewLifecycleStatus.REJECTED,
    ReviewLifecycleStatus.CANCELLED,
    ReviewLifecycleStatus.FAILED,
}


def _stable_key(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return canonical_checksum(value)


__all__ = [
    "COMPLETE_STATUSES",
    "build_run_response",
    "registration_from_run",
]
