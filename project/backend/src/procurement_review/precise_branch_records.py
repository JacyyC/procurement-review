"""What one precise branch records, and what a refused commit means for the execution.

The runner owns *when* an attempt may happen; this module owns what is written down
about it: the bounded attempt record, the branch record a completed, failed or stopped
branch produces, the record an adopted commit restates, the control-failure vocabulary
the summary speaks, and the decision a commit that was *not* adopted forces on the
execution.

That last rule is the reason this module exists apart.  Two different facts produce one
refusal: a generation that no longer owns the Run stops sending with the guard's own
reason, and a commit that lost to another attempt of the same branch adopts the committed
branch instead of failing.  Both are explicit paths; neither is an incidental exception,
and neither is a checkpoint story told about a late answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .business_v3_documents import DocumentValuesV3
from .business_v3_evidence import GroundedDocumentV3
from .checkpointing import IncompatibleCheckpointError
from .contracts import EvidenceRef
from .execution_guards import (
    INVALIDATION_NO_ACTIVE_GENERATION,
    INVALIDATION_RUN_NOT_PROCESSING,
    INVALIDATION_SUPERSEDED,
    ExecutionInvalidated,
)
from .precise_run_contracts import (
    PreciseBranchOutcome,
    PreciseDocumentBranchV1,
    PreciseDutyValueV31,
    PreciseModelAttemptV1,
)
from .precise_run_scope import BranchInputs
from .precise_run_state import adopted_branch_commit, restore_document
from .precise_run_summary import sha256_of_text


@dataclass(slots=True)
class BranchOutcome:
    """What one branch produced: its record, its document, its projection and any stop."""

    record: PreciseDocumentBranchV1
    document: GroundedDocumentV3
    security: tuple[Any, ...] = ()
    blocked: bool = False
    stop_reason: str | None = None
    #: True when the branch reused its own persisted commit instead of being sent again.
    reused: bool = False


#: The registered control-failure vocabulary of the review.  A branch error travels as the
#: control code that owns it, never as an unregistered string.
CONTROL_FAILURES = {
    "timeout": "timeout",
    "transport": "transport",
    "rate_limited": "transport",
    "server_error": "transport",
    "authentication": "transport",
    "model_identity_drift": "transport",
    "credential_in_request": "transport",
    "account_quota_exhausted": "transport",
    "unregistered_provider_error": "transport",
    "invalid_json": "evidence_envelope_invalid",
    "invalid_schema": "evidence_envelope_invalid",
    "evidence_envelope_invalid": "evidence_envelope_invalid",
    "security_boundary": "security_boundary",
    "check_outside_branch": "security_boundary",
    "citation_outside_branch": "security_boundary",
    "tool_call_forbidden": "tool_call_forbidden",
    "safety_refusal": "safety_refusal",
    "deadline_exhausted": "deadline",
    "attempt_budget_exhausted": "document_model_failed",
}

#: Failures after which no further attempt of this generation may be sent.  They are
#: authentication, configuration and security facts: retrying them would re-ask a model
#: that was never entitled to answer, so the stop is an explicit path and never an
#: incidental exception.  The V1-08 stop contract adds the account-quota refusal and the
#: versioned unregistered provider error for the same reason: neither may keep a batch
#: sending while it is unclassified or unpaid for.
STOP_ERROR_CODES = frozenset(
    {
        "authentication",
        "credential_in_request",
        "model_identity_drift",
        "security_boundary",
        "check_outside_branch",
        "citation_outside_branch",
        "account_quota_exhausted",
        "unregistered_provider_error",
    }
)

#: The reasons a branch commit is refused because this generation no longer owns the Run.
#: They are the guard's own words, and they travel as the explicit stop path: a commit
#: refused for one of them is not a checkpoint problem and not a late answer.
NON_OWNER_REASONS = frozenset(
    {
        INVALIDATION_SUPERSEDED,
        INVALIDATION_NO_ACTIVE_GENERATION,
        INVALIDATION_RUN_NOT_PROCESSING,
    }
)


def guard_failure_code(categories: tuple[str, ...] | None) -> str:
    """The branch failure code for a payload the stored-record guard refused.

    It names the guard's own fixed categories — never a value — so the branch record says
    what the product refused to persist.
    """

    return "record_guard_refused:" + ",".join(categories or ("record_contract",))


def attempt_record(
    attempt_no: int,
    observed: Any,
    latency_ms: int,
    *,
    correctable: bool | None = None,
    error_code: str | None = None,
) -> PreciseModelAttemptV1:
    """The bounded record of one physical attempt.

    ``error_code`` and ``correctable`` state what the *branch* decided about this
    attempt: a citation the source boundary refused has its own code, and it is
    correctable exactly when another attempt is available — neither is visible on the
    wire receipt alone.
    """

    return PreciseModelAttemptV1(
        attempt_no=attempt_no,
        error_code=observed.error_code if error_code is None else error_code,
        correctable=bool(observed.correctable) if correctable is None else correctable,
        latency_ms=latency_ms,
        input_tokens=observed.input_tokens,
        output_tokens=observed.output_tokens,
        resolved_model=observed.resolved_model_id,
        usage_unknown=(observed.input_tokens is None or observed.output_tokens is None),
    )


def reported_from(
    document: GroundedDocumentV3 | None,
) -> tuple[PreciseDutyValueV31, ...]:
    """The duty values one branch reported, or nothing when it reported none."""

    if document is None:
        return ()
    reported = []
    for observed in document.document.findings:
        item = observed.finding
        reported.append(
            PreciseDutyValueV31(
                obligation_id=item.obligation_id,
                status=item.status,
                normalized=(
                    None
                    if item.normalized_value == "UNKNOWN"
                    else item.normalized_value
                ),
            )
        )
    return tuple(reported)


def branch_record(
    *,
    worker: str,
    document: GroundedDocumentV3 | None,
    request_sha: str,
    answer_sha: str | None,
    attempts: tuple,
    hits: tuple,
    outcome: PreciseBranchOutcome = PreciseBranchOutcome.COMPLETED,
    failure_code: str | None = None,
) -> PreciseDocumentBranchV1:
    """The bounded record of one branch: what it reported and what it cost."""

    return PreciseDocumentBranchV1(
        worker=worker,
        outcome=outcome,
        reported=reported_from(document),
        spans=document.spans if document is not None else (),
        support_failures=document.support_failures if document is not None else (),
        request_sha256=request_sha,
        answer_sha256=answer_sha,
        attempts=attempts,
        evidence_refs=tuple(hit.evidence_ref for hit in hits),
        failure_code=failure_code,
    )


def commit_record(commit: Any, request_sha: str) -> PreciseDocumentBranchV1:
    """The branch record of an adopted commit, restated from its bounded projection."""

    return PreciseDocumentBranchV1(
        worker=commit.worker,
        outcome=PreciseBranchOutcome.COMPLETED,
        reported=tuple(
            PreciseDutyValueV31(
                obligation_id=duty.obligation_id,
                status=duty.status,
                normalized=None if duty.value == "UNKNOWN" else duty.value,
            )
            for duty in commit.duties
        ),
        spans=commit.spans,
        support_failures=commit.support_failures,
        request_sha256=commit.request_sha256,
        answer_sha256=commit.answer_sha256,
        attempts=commit.attempts,
        evidence_refs=commit.evidence_refs,
    )


def skipped_record(*, worker: str, branch: BranchInputs) -> PreciseDocumentBranchV1:
    """The record of a branch whose sends an explicit control path stopped."""

    return PreciseDocumentBranchV1(
        worker=worker,
        outcome=PreciseBranchOutcome.STOPPED,
        reported=(),
        spans=(),
        support_failures=(),
        attempts=(),
        request_sha256=sha256_of_text(branch.request.model_dump_json()),
        evidence_refs=tuple(
            EvidenceRef(**hit.evidence_ref.model_dump()) for hit in branch.hits
        ),
    )


def failed_document(worker: str) -> GroundedDocumentV3:
    """The empty document one failed branch stands for: no values, one failure."""

    return GroundedDocumentV3(
        DocumentValuesV3(worker, {}, (), ("document_branch_failed",)), (), ()
    )


def pipeline_failures(
    records: dict[str, PreciseDocumentBranchV1],
    procurement_read: Any,
    stop_reason: str | None,
) -> tuple[str, ...]:
    """The control failures the summary records, each reason exactly once."""

    failures = []
    for record in records.values():
        if record.outcome in {
            PreciseBranchOutcome.COMPLETED,
            PreciseBranchOutcome.STOPPED,
        }:
            continue
        failures.append(
            CONTROL_FAILURES.get(record.failure_code or "", "document_model_failed")
        )
    if stop_reason is not None:
        failures.append(CONTROL_FAILURES.get(stop_reason, "document_model_failed"))
    if procurement_read.status.value != "known":
        failures.append("transport")
    return tuple(dict.fromkeys(failures))


def adopted_outcome(committed: Any, request: Any, request_sha: str) -> BranchOutcome:
    """The outcome of a branch whose result is the persisted commit, never a new answer."""

    return BranchOutcome(
        record=commit_record(committed, request_sha),
        document=restore_document(committed, request),
        security=tuple(committed.security),
        blocked=committed.blocked,
        reused=True,
    )


def resume_refused_commit(
    refused: Any,
    *,
    repository: Any,
    run_id: str,
    worker: str,
    request: Any,
    request_sha: str,
) -> BranchOutcome:
    """What a branch commit that was not adopted means for the execution.

    A generation that no longer owns the Run stops sending and reports the guard's own
    reason.  A commit that lost to another attempt of the same branch adopts that
    committed branch: the late answer is never this execution's result, and the execution
    does not fail over a race it lost.
    """

    reason = str(getattr(refused, "reason", "") or "")
    if reason in NON_OWNER_REASONS:
        raise ExecutionInvalidated(reason)
    committed = adopted_branch_commit(repository, run_id, worker)
    if (
        committed is None
        or committed.request_sha256 != request_sha
        or committed.worker != worker
    ):
        raise IncompatibleCheckpointError(
            "the branch result was not adopted and no committed branch matches this "
            "execution"
        )
    return adopted_outcome(committed, request, request_sha)


__all__ = [
    "BranchOutcome",
    "guard_failure_code",
    "CONTROL_FAILURES",
    "NON_OWNER_REASONS",
    "STOP_ERROR_CODES",
    "adopted_outcome",
    "attempt_record",
    "branch_record",
    "commit_record",
    "failed_document",
    "pipeline_failures",
    "reported_from",
    "resume_refused_commit",
    "skipped_record",
]
