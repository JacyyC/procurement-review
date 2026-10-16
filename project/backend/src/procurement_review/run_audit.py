"""The approval-audit stream: database-assigned numbering and the write path.

One allocator row per stream is taken with a row lock that is held to commit, so the
number a transaction receives always follows the *committed* order of the events and
never the order in which two transactions happened to read a maximum.  The same value
builds both the relation column and the persisted JSON, and the complete audit record
passes the closed, type-aware guard before it is written.  Gaps are allowed and are never
filled in.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from .audit_contracts import ApprovalAuditEventV2
from .record_guard import guard_record
from .run_errors import RecordIncompleteError
from .run_records import AuditDraft


def allocate_sequence(connection: Any, stream: str) -> int:
    """Assign the next number of one stream inside the caller's transaction.

    The row lock is held until the transaction ends, so a second writer cannot be handed
    a number before the first one's event is committed.  Taking this lock is the last step
    of a state transaction: nothing after it acquires an authorization, Run or generation
    lock, so the order cannot be reversed.
    """

    row = connection.execute(
        text(
            "SELECT next_value FROM review_sequence_allocator "
            "WHERE stream = :stream FOR UPDATE"
        ),
        {"stream": stream},
    ).first()
    if row is None:
        raise RecordIncompleteError(
            f"the {stream} sequence allocator row is missing; run the migrations"
        )
    value = int(row.next_value)
    connection.execute(
        text(
            "UPDATE review_sequence_allocator SET next_value = next_value + 1 "
            "WHERE stream = :stream"
        ),
        {"stream": stream},
    )
    return value


def append_audit(
    connection: Any,
    *,
    run_id: str,
    draft: AuditDraft,
    fault: Any | None = None,
) -> None:
    """Persist one audit event with the number the database assigned it.

    ``fault`` is the inert test-only hook: a scenario that wants to observe the wait for
    a row lock parks the allocating transaction between the allocation and the insert.
    """

    row = connection.execute(
        text("SELECT tenant_id, case_id, status FROM review_run WHERE run_id = :r"),
        {"r": run_id},
    ).first()
    tenant_id = row.tenant_id if row is not None else None
    case_id = row.case_id if row is not None else None
    sequence = allocate_sequence(connection, "audit")
    if fault is not None:
        fault.barrier("during_audit_sequence")
    event = ApprovalAuditEventV2(
        audit_id=draft.audit_id,
        sequence=sequence,
        tenant_id=tenant_id,
        case_id=case_id,
        run_id=run_id,
        event_class=draft.event_class,
        action_type=draft.action_type,
        actor=draft.actor,
        authorization_revision=draft.authorization_revision,
        previous_status=draft.previous_status,
        new_status=draft.new_status,
        previous_revision=draft.previous_revision,
        new_revision=draft.new_revision,
        result_checksum=draft.result_checksum,
        idempotency_digest=draft.idempotency_digest,
        receipt_id=draft.receipt_id,
        occurred_at=draft.occurred_at,
        detail_code=draft.detail_code,
    )
    guard_record("audit", event, location="audit")
    connection.execute(
        text(
            """
            INSERT INTO review_audit_event (
                sequence, audit_id, run_id, case_id, event_class, action_type,
                actor, authorization_revision, previous_status, new_status,
                previous_revision, new_revision, result_checksum,
                idempotency_digest, receipt_id, api_version, occurred_at,
                event_payload
            ) VALUES (
                :sequence, :audit_id, :run_id, :case_id, :class, :action, :actor,
                :revision, :previous_status, :new_status, :previous_revision,
                :new_revision, :result_checksum, :idempotency_digest, :receipt_id,
                :api_version, :occurred_at, CAST(:payload AS jsonb)
            )
            """
        ),
        {
            "sequence": sequence,
            "audit_id": draft.audit_id,
            "run_id": run_id,
            "case_id": case_id,
            "class": draft.event_class.value,
            "action": draft.action_type.value if draft.action_type else None,
            "actor": draft.actor,
            "revision": draft.authorization_revision,
            "previous_status": draft.previous_status.value
            if draft.previous_status
            else None,
            "new_status": draft.new_status.value if draft.new_status else None,
            "previous_revision": draft.previous_revision,
            "new_revision": draft.new_revision,
            "result_checksum": draft.result_checksum,
            "idempotency_digest": draft.idempotency_digest,
            "receipt_id": draft.receipt_id,
            "api_version": "review-api.v2",
            "occurred_at": draft.occurred_at,
            "payload": event.model_dump_json(),
        },
    )


__all__ = ["allocate_sequence", "append_audit"]
