"""What a precise Run commits and reuses: its frozen snapshot and its branch commits.

The stored shapes live in :mod:`.precise_run_contracts`; this module owns the behaviour
around them — writing the snapshot into the Run's frozen Case input, committing one branch
atomically with its attempt row, adopting a committed branch back into a grounded document,
and recomputing the Run's metering from the persisted attempt ledger and receipts.

The frozen 3.0.0 line and this line share the product's checkpoint table: one row per
Worker per Run, judged by the version its own payload declares.  A row that belongs to the
other line, or that no longer satisfies its record, is a controlled incompatibility rather
than a silent re-run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .gold_v3_contracts import payload_sha256
from .precise_run_contracts import (
    PRECISE_BRANCH_COMMIT_VERSION,
    PRECISE_RUN_SNAPSHOT_VERSION,
    PreciseBranchCommitV1,
    PreciseCheckpointPayloadV1,
    PreciseRunSnapshotV1,
    precise_contract_identity,
    snapshot_mismatch,
)

#: The fixed phrases a restored branch restates its findings with.  They are the two
#: bounded words the 3.1.0 task vocabulary already asks for, never a re-invented answer.
_STATEMENTS = {
    "supported": "原文提取",
    "violated": "原文提取",
    "not_applicable": "原文提取",
    "unknown": "材料不足",
}
_ACTIONS = {
    "supported": "人工核对来源",
    "violated": "人工核对来源",
    "not_applicable": "人工核对来源",
    "unknown": "补充所缺证据",
}
_ASSERTIONS = {
    "supported": "compliant",
    "violated": "non_compliant",
    "unknown": "unknown",
    "not_applicable": "not_applicable",
}


def _incompatible() -> Any:
    from .checkpointing import IncompatibleCheckpointError

    return IncompatibleCheckpointError


def run_isolation(registration: Any, runner: Any = None) -> dict[str, object]:
    """The frozen Case input of one Run, including the runner's execution snapshot.

    A runner that declares no snapshot keeps the frozen shape; one that does writes its
    mode, model, parameters, digests, versions, budgets and deadline here, immutably, so
    every later generation of the Run reuses exactly that configuration.
    """

    return {
        "case_type": registration.case_type,
        "scenario": registration.scenario,
        "purchase_order_id": registration.purchase_order_id,
        "case_document_ids": list(registration.case_document_ids),
        "shared_document_ids": list(registration.shared_document_ids),
        "allowed_order_ids": list(registration.allowed_order_ids),
        "allowed_supplier_ids": list(registration.allowed_supplier_ids),
        "allowed_transaction_ids": list(registration.allowed_transaction_ids),
        "execution_snapshot": _snapshot_of(runner, registration),
    }


def _snapshot_of(runner: Any, registration: Any) -> Any:
    provider = getattr(runner, "execution_snapshot", None)
    return provider(registration) if callable(provider) else None


def snapshot_from_isolation(isolation: Any) -> PreciseRunSnapshotV1 | None:
    """The frozen snapshot of one Run's Case input, or ``None`` when it carries none."""

    if not isinstance(isolation, Mapping):
        return None
    stored = isolation.get("execution_snapshot")
    if not isinstance(stored, Mapping):
        return None
    try:
        return PreciseRunSnapshotV1.model_validate(dict(stored), strict=True)
    except Exception:  # noqa: BLE001 - a damaged snapshot is reported by its caller
        return None


def checkpoint_reusable(
    payload: Mapping[str, Any],
    scope: Any,
    worker: str,
    *,
    snapshot: PreciseRunSnapshotV1 | None = None,
) -> bool:
    """Whether one stored checkpoint row may be reused by this Run's next execution.

    The frozen 3.0.0 line keeps its own contract identity; the precise line matches the
    identity derived from the Run's frozen snapshot — the snapshot the Run carries, not the
    scope, which has no execution snapshot of its own.  A row that matches neither is a
    controlled incompatibility — never a silent re-run under different parameters.
    """

    from .checkpointing import contract_checksum

    stored = str(payload.get("contract_checksum", ""))
    if stored == contract_checksum(scope, worker):
        return True
    if snapshot is None:
        return False
    return stored == precise_contract_identity(snapshot, worker)


def commit_from_branch(
    *,
    worker: str,
    branch: Any,
    document: Any,
    security: Sequence[Any],
    blocked: bool,
) -> PreciseBranchCommitV1:
    """Project one completed branch into its bounded, versioned commit record."""

    cited = {
        item.finding.obligation_id: item.finding for item in document.document.findings
    }
    duties = []
    for reported in branch.reported:
        finding = cited.get(reported.obligation_id)
        duties.append(
            {
                "obligation_id": reported.obligation_id,
                "status": reported.status,
                "risk_level": "unknown" if finding is None else finding.risk_level,
                "value": "UNKNOWN" if finding is None else finding.normalized_value,
                "evidence_refs": () if finding is None else finding.evidence_refs,
            }
        )
    return PreciseBranchCommitV1(
        worker=worker,
        duties=tuple(duties),
        spans=branch.spans,
        document_failures=document.document.failures,
        support_failures=branch.support_failures,
        security=tuple(security),
        blocked=blocked,
        request_sha256=branch.request_sha256 or payload_sha256({}),
        answer_sha256=branch.answer_sha256 or payload_sha256({}),
        attempts=branch.attempts,
        evidence_refs=branch.evidence_refs,
    )


def restore_document(commit: PreciseBranchCommitV1, request: Any) -> Any:
    """Rebuild the grounded document one committed branch stands for.

    The committed duties are re-canonicalized and re-decoded through the same closed
    boundary the first execution used, with the citations resolved against the request's
    own authorized sources.  A branch that no longer covers the requested duties, or that
    cites a source this execution is not authorized for, is a contract mismatch — never a
    repaired record.
    """

    from .business_v3_documents import DocumentValuesV3
    from .business_v3_evidence import GroundedDocumentV3
    from .precise_answer_v3 import canonical_answer, decode_canonical_answer

    incompatible = _incompatible()
    raw = {
        "schema_version": "finding-evidence.v3",
        "findings": [
            {
                "obligation_id": duty.obligation_id,
                "status": duty.status,
                "statement": _STATEMENTS[duty.status],
                "recommended_action": _ACTIONS[duty.status],
                "normalized_value": duty.value,
                "risk_level": (
                    duty.risk_level
                    if duty.risk_level in {"low", "medium", "high", "critical"}
                    else None
                ),
                "evidence_refs": [ref for ref in duty.evidence_refs],
                "missing_information": (
                    ["材料不足"] if duty.status == "unknown" else []
                ),
            }
            for duty in commit.duties
        ],
    }
    witnesses = [
        {
            "obligation_id": span.proposition.check_id,
            "field": span.proposition.field_path.rsplit(".", 1)[-1],
            "value": span.value,
            "source_ref": _ref_for(request, span),
            "start": span.start,
            "end": span.end,
        }
        for span in commit.spans
    ]
    try:
        canonical, witnesses = canonical_answer(json.dumps(raw), request)
        decoded, _witnesses = decode_canonical_answer(canonical, request, witnesses)
    except Exception as error:  # noqa: BLE001 - a damaged commit is a controlled stop
        raise incompatible(
            "the committed branch no longer satisfies the branch contract"
        ) from error
    if decoded.failures:
        raise incompatible(
            "the committed branch no longer satisfies its value registry"
        )
    return GroundedDocumentV3(
        DocumentValuesV3(
            request.worker,
            decoded.values,
            decoded.findings,
            tuple(commit.document_failures),
        ),
        tuple(commit.spans),
        tuple(commit.support_failures),
    )


def _ref_for(request: Any, span: Any) -> str:
    """The authorized reference one committed span points at, by its own identity."""

    for source in request.sources:
        identity = source.identity
        if (
            identity.source_id == span.evidence.source_id
            and identity.version == span.evidence.version
            and identity.locator == span.evidence.locator
        ):
            return source.ref
    raise _incompatible()("the committed citation is outside this branch")


def _mark_late(
    connection: Any, run_id: str, generation: int, worker: str, attempt_no: int
) -> None:
    """Mark one attempt whose commit lost the race to a later attempt of its branch."""

    from sqlalchemy import text

    connection.execute(
        text(
            "UPDATE review_worker_attempt SET discarded_late = true "
            "WHERE run_id = :run_id AND generation = :generation "
            "AND worker = :worker AND attempt_no = :attempt"
        ),
        {
            "run_id": run_id,
            "generation": generation,
            "worker": worker,
            "attempt": attempt_no,
        },
    )


def commit_precise_branch(
    repository: Any,
    *,
    run_id: str,
    generation: int,
    worker: str,
    attempt_no: int,
    input_checksum: str,
    contract_checksum: str,
    payload: PreciseBranchCommitV1,
    now: Any,
    owner_id: str | None = None,
    events: Sequence[Any] = (),
) -> Any:
    """Adopt one committed branch atomically with its attempt row and Trace events.

    The row is the product's existing checkpoint row — the same table the frozen line
    adopts its branches in — so the offline inventory and the complete-record reader judge
    it under the version the payload itself declares.
    """

    from sqlalchemy import text

    from .audit_contracts import CheckpointRecord
    from .record_guard import guard_record
    from .review_repository import current_time
    from .run_records import BranchCommit
    from .run_trace import append_trace as append_trace_events, trace_identity
    from .trace import canonical_checksum

    clock = now
    payload_document = {
        "checkpoint_version": PRECISE_BRANCH_COMMIT_VERSION,
        "input_checksum": input_checksum,
        "contract_checksum": contract_checksum,
        "result": payload.model_dump(mode="json"),
    }
    with repository.store.transaction() as connection:
        guard = repository._guards.lock_active_generation(
            connection,
            run_id=run_id,
            generation=generation,
            now=now,
            owner_id=owner_id,
        )
        if guard is not None:
            # The attempt was sent and answered, and this generation may not adopt it:
            # the ledger says "settled, not adopted" instead of leaving it looking used.
            _mark_late(connection, run_id, generation, worker, attempt_no)
            return BranchCommit(adopted=False, reason=guard)
        now = current_time(clock)
        latest = connection.execute(
            text(
                "SELECT coalesce(max(attempt_no), 0) FROM review_worker_attempt "
                "WHERE run_id = :run_id AND worker = :worker"
            ),
            {"run_id": run_id, "worker": worker},
        ).scalar_one()
        if int(latest) > attempt_no:
            _mark_late(connection, run_id, generation, worker, attempt_no)
            return BranchCommit(adopted=False, reason="superseded_attempt")
        adapted = guard_record(
            "checkpoint",
            {
                "run_id": run_id,
                "worker": worker,
                "generation": generation,
                "attempt_no": attempt_no,
                "input_checksum": input_checksum,
                "contract_checksum": contract_checksum,
                "result_checksum": canonical_checksum(payload_document),
            },
            location="checkpoint",
        )
        assert isinstance(adapted, CheckpointRecord)
        stored = guard_record(
            "checkpoint_payload",
            payload_document,
            location="checkpoint_payload",
            strict=False,
        )
        inserted = connection.execute(
            text(
                """
                INSERT INTO review_branch_checkpoint (
                    run_id, worker, generation, attempt_no, input_checksum,
                    contract_checksum, result_checksum, result_payload, adopted_at
                ) VALUES (
                    :run_id, :worker, :generation, :attempt, :input, :contract,
                    :checksum, CAST(:payload AS jsonb), :now
                )
                ON CONFLICT (run_id, worker) DO NOTHING
                """
            ),
            {
                "run_id": adapted.run_id,
                "worker": adapted.worker,
                "generation": adapted.generation,
                "attempt": adapted.attempt_no,
                "input": adapted.input_checksum,
                "contract": adapted.contract_checksum,
                "checksum": adapted.result_checksum,
                "payload": json.dumps(
                    stored.model_dump(mode="json"), ensure_ascii=False
                ),
                "now": now,
            },
        )
        if not inserted.rowcount:
            _mark_late(connection, run_id, generation, worker, attempt_no)
            return BranchCommit(adopted=False, reason="already_adopted")
        connection.execute(
            text(
                "UPDATE review_worker_attempt SET finished_at = :now, "
                "outcome = 'completed', last_error = NULL "
                "WHERE run_id = :run_id AND generation = :generation "
                "AND worker = :worker AND attempt_no = :attempt"
            ),
            {
                "run_id": run_id,
                "generation": generation,
                "worker": worker,
                "attempt": attempt_no,
                "now": now,
            },
        )
        append_trace_events(
            connection,
            run_id=run_id,
            trace_id=trace_identity(connection, run_id),
            drafts=events,
            now=now,
        )
        repository._guards.require_execution_time(
            connection, run_id=run_id, generation=generation, now=clock
        )
    return BranchCommit(adopted=True, reason="committed")


def adopted_branch_commit(
    repository: Any, run_id: str, worker: str
) -> PreciseBranchCommitV1 | None:
    """The commit of one branch, or ``None`` when this Run has not adopted it yet.

    A stored row that belongs to another execution contract, or that no longer satisfies
    the record its version declares, stops the Run with a recorded reason instead of being
    interpreted as a success.
    """

    from sqlalchemy import text

    from .record_reads import verify_checkpoint_rows
    from .record_rows import json_value, row_mapping
    from .run_errors import RecordUnavailableError

    incompatible = _incompatible()
    with repository.store.transaction() as connection:
        row = connection.execute(
            text(
                "SELECT * FROM review_branch_checkpoint "
                "WHERE run_id = :run_id AND worker = :worker"
            ),
            {"run_id": run_id, "worker": worker},
        ).first()
    if row is None:
        return None
    stored = row_mapping(row)
    try:
        verify_checkpoint_rows((stored,))
    except RecordUnavailableError as error:
        raise incompatible(
            "the adopted checkpoint no longer satisfies its record"
        ) from error
    payload = json_value(stored.get("result_payload"))
    if not isinstance(payload, Mapping):
        raise incompatible("the adopted checkpoint carries no record")
    if payload.get("checkpoint_version") != PRECISE_BRANCH_COMMIT_VERSION:
        raise incompatible(
            "the adopted checkpoint belongs to another execution contract"
        )
    try:
        # The stored document is JSON: arrays are the stored spelling of the tuples the
        # record declares, exactly as the write path serialized them.
        return PreciseCheckpointPayloadV1.model_validate_json(
            json.dumps(dict(payload)), strict=True
        ).result
    except Exception as error:  # noqa: BLE001 - a damaged commit is a controlled stop
        raise incompatible("the adopted branch commit is not readable") from error


def finish_attempt_with_receipt(
    repository: Any,
    *,
    run_id: str,
    generation: int,
    worker: str,
    attempt_no: int,
    outcome: str,
    error_code: str | None,
    now: Any,
    receipt: Any,
) -> None:
    """Settle one attempt and persist its receipt in the same transaction.

    The ledger row keeps the attempt's own lifecycle; the receipt (request hash, tokens,
    resolved model, error category, unknown-result marker) is appended to the Run's Trace
    beside it.  Both come from one checked write, so an interrupted execution can still
    say what it sent and what came back.
    """

    from sqlalchemy import text

    from .record_guard import guard_record
    from .run_trace import append_trace as append_trace_events, trace_identity

    guard_record(
        "attempt_outcome",
        {
            "outcome": outcome,
            "error_code": error_code,
            "abandoned": False,
            "discarded_late": False,
        },
        location="attempt_outcome",
    )
    with repository.store.transaction() as connection:
        connection.execute(
            text(
                """
                UPDATE review_worker_attempt
                SET finished_at = :now,
                    outcome = :outcome,
                    last_error = :error,
                    first_error = coalesce(first_error, :error),
                    abandoned = abandoned OR false,
                    discarded_late = discarded_late OR false
                WHERE run_id = :run_id AND generation = :generation
                  AND worker = :worker AND attempt_no = :attempt
                """
            ),
            {
                "run_id": run_id,
                "generation": generation,
                "worker": worker,
                "attempt": attempt_no,
                "outcome": outcome,
                "error": error_code,
                "now": now,
            },
        )
        append_trace_events(
            connection,
            run_id=run_id,
            trace_id=trace_identity(connection, run_id),
            drafts=(receipt,),
            now=now,
        )


def attempt_ledger_rows(repository: Any, run_id: str) -> tuple[Mapping[str, Any], ...]:
    """Every attempt row of one Run, across all of its generations."""

    from sqlalchemy import text

    from .record_rows import row_mapping

    with repository.store.transaction() as connection:
        rows = connection.execute(
            text(
                "SELECT run_id, generation, worker, attempt_no, started_at, finished_at,"
                " outcome, first_error, last_error, abandoned, discarded_late"
                " FROM review_worker_attempt WHERE run_id = :run_id"
                " ORDER BY generation, worker, attempt_no"
            ),
            {"run_id": run_id},
        ).all()
    return tuple(row_mapping(row) for row in rows)


def attempt_receipts(repository: Any, run_id: str) -> tuple[Mapping[str, Any], ...]:
    """The persisted receipts of every attempt of one Run, in event order.

    They come from the Run's own Trace rows: the attempt ledger has no column for a
    request hash, a token count or a resolved model, and an interrupted execution must not
    lose them.
    """

    run = repository.load_run(run_id)
    if run is None:
        return ()
    receipts = []
    for event in run.trace_events:
        stored = (
            event
            if isinstance(event, Mapping)
            else getattr(event, "model_dump", lambda mode=None: {})(mode="json")
        )
        if stored.get("step") != "model_attempt":
            continue
        summary = dict(stored.get("output_summary") or {})
        summary["generation"] = stored.get("generation") or summary.get("generation")
        summary["attempt"] = summary.get("attempt_no", stored.get("attempt"))
        receipts.append(summary)
    return tuple(receipts)


def attempt_receipt_draft(
    *,
    worker: str,
    generation: int,
    attempt_no: int,
    request_sha256: str,
    observed: Any,
    latency_ms: int,
    next_correction_kind: str | None = None,
) -> Any:
    """The bounded Trace event one settled attempt persists.

    It carries the receipt metadata the ledger row has no column for — request hash,
    tokens, resolved model, error category and the unknown-result marker — and never an
    answer text or a prompt.  ``next_correction_kind`` names what the following attempt of
    the same branch was for, so the Run's metering can tell a contract correction from a
    citation correction and from a transport retry instead of calling every extra attempt
    a format correction.
    """

    from .contracts import RunEventType
    from .run_records import TraceDraft

    return TraceDraft(
        event_type=(
            RunEventType.STEP_COMPLETED.value
            if observed.error_code is None
            else RunEventType.STEP_FAILED.value
        ),
        step="model_attempt",
        error=observed.error_code,
        output_summary={
            "status": "ok",
            "worker": worker,
            "attempt_no": attempt_no,
            "outcome": "completed" if observed.error_code is None else "failed",
            "attempt_error": observed.error_code,
            "request_sha256": request_sha256,
            "receipt_model": observed.resolved_model_id,
            "http_status": observed.http_status,
            "response_bytes": observed.response_bytes,
            "finish_reason": observed.finish_reason,
            "format_diagnostic": observed.format_diagnostic,
            "format_fields": list(observed.format_fields),
            "format_diagnostics": list(observed.format_diagnostics),
            "input_usage": observed.input_tokens,
            "output_usage": observed.output_tokens,
            "usage_unknown": bool(observed.usage_unknown),
            "next_correction_kind": next_correction_kind,
            "latency_ms": max(0, latency_ms),
        },
        source_versions={
            "snapshot": PRECISE_RUN_SNAPSHOT_VERSION,
            "execution": PRECISE_BRANCH_COMMIT_VERSION,
        },
        generation=generation,
        attempt=attempt_no,
        worker=worker,
    )


def metering_from_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    receipts: Sequence[Mapping[str, Any]],
    requested_model: str | None,
    resolved_models: Sequence[str],
    reused_branches: int = 0,
    explicit_recoveries: int = 0,
) -> Any:
    """Recompute the Run's metering from its persisted ledger and attempt receipts.

    A ledger row is not a send.  Only an attempt with its own persisted receipt proves that
    a request was sent and answered; a row closed without one may or may not have reached
    the endpoint and is reported as unresolved, and an attempt closed by an interruption is
    unknown usage too because whether it was physically sent is not decidable from
    persisted facts.  Nothing is ever reported as a free attempt, and the known token
    figures stay a subtotal unless every opened attempt's usage is known.

    Every generation of the Run is included, so a resumed execution does not start the
    count over, and the extra attempts are split by what they corrected rather than being
    pooled as format corrections.
    """

    from .attempt_metering import AttemptFact, classify
    from .precise_run_contracts import RunMeteringV1

    by_attempt = {
        (str(item.get("worker")), item.get("attempt")): item for item in receipts
    }
    facts = []
    for row in rows:
        worker = str(row.get("worker"))
        key = (worker, row.get("attempt_no"))
        receipt = by_attempt.get(key)
        body = receipt or {}
        facts.append(
            AttemptFact(
                attempt_no=int(row.get("attempt_no") or 1),
                has_receipt=receipt is not None,
                closed=row.get("finished_at") is not None,
                usage_unknown=bool(body.get("usage_unknown", True)),
                input_tokens=body.get("input_usage"),
                output_tokens=body.get("output_usage"),
                next_kind=body.get("next_correction_kind"),
                # One Run holds several independent branches whose attempt numbers
                # restart at one, so the correction order is only meaningful per worker.
                branch_key=worker,
                discarded=bool(row.get("discarded_late") or row.get("abandoned")),
            )
        )
    metering = classify(facts)
    latency: dict[str, int] = {}
    for item in receipts:
        worker = str(item.get("worker"))
        latency[worker] = latency.get(worker, 0) + int(item.get("latency_ms") or 0)
    return RunMeteringV1(
        physical_attempts=metering.attempts_opened,
        first_sends=metering.first_sends,
        corrections=metering.extra_attempts,
        input_tokens_known=metering.input_tokens_known,
        output_tokens_known=metering.output_tokens_known,
        unknown_usage_attempts=metering.unknown_usage_attempts,
        unreturned_attempts=metering.unresolved_attempts + metering.open_attempts,
        branch_latency_ms=tuple(sorted(latency.items())),
        requested_model=requested_model,
        resolved_models=tuple(sorted(set(resolved_models))),
        attempts_opened=metering.attempts_opened,
        proven_sends=metering.proven_sends,
        unresolved_attempts=metering.unresolved_attempts,
        open_attempts=metering.open_attempts,
        contract_corrections=metering.contract_corrections,
        citation_corrections=metering.citation_corrections,
        transport_retries=metering.transport_retries,
        other_extra_attempts=metering.other_extra_attempts,
        usage_complete=metering.usage_complete,
        reused_branches=reused_branches,
        explicit_recoveries=explicit_recoveries,
    )


__all__ = [
    "PRECISE_BRANCH_COMMIT_VERSION",
    "PRECISE_RUN_SNAPSHOT_VERSION",
    "adopted_branch_commit",
    "attempt_ledger_rows",
    "attempt_receipt_draft",
    "attempt_receipts",
    "checkpoint_reusable",
    "commit_from_branch",
    "commit_precise_branch",
    "finish_attempt_with_receipt",
    "metering_from_ledger",
    "precise_contract_identity",
    "restore_document",
    "run_isolation",
    "snapshot_from_isolation",
    "snapshot_mismatch",
]
