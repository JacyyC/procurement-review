"""Shared harness for the FIX-03A recursive-contract and full-write verifications.

Two things live here and nothing else:

* a complete ledger snapshot of one Run — every table a state transaction can touch, the
  audit allocator and the row counts — so "the refused transaction wrote nothing" is a
  comparison of committed facts rather than an assertion that something raised;
* the *pre-fix* guard, which re-applies the exact defect the FIX-01 audit reproduced, so
  every new counterexample can show that its own assertion would fail without the fix.

The pre-fix behaviour is reconstructed in-process instead of by checking out old code:
``model_construct`` builds the closed type without running any validator (the old write
paths bound their parameters straight from the caller's plain values), and dropping a
mapping's registry entry restores the old "open mapping, untyped values" walk.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text

from procurement_review import record_contracts, record_guard
from procurement_review.contracts import (
    ApprovalActionType,
    ReviewLifecycleStatus,
)
from procurement_review.trace import canonical_checksum

from fix01a_support import Fix01aFixture

__all__ = [
    "LEDGER_TABLES",
    "create_arguments",
    "ledger_snapshot",
    "open_mapping_reversion",
    "unchecked_guard",
]


def create_arguments(
    ident: Any,
    run_id: str,
    *,
    trace_id: str | None = None,
    request_digest: str | None = None,
    events: Any = None,
) -> dict[str, Any]:
    """The arguments of one real create, overridable where a scenario poisons one."""

    import secrets
    from datetime import UTC, datetime

    registration = ident.registration
    return {
        "run_id": run_id,
        "trace_id": trace_id or f"trace-{secrets.token_hex(16)}",
        "tenant_id": registration.tenant_id,
        "case_id": registration.case_id,
        "case_version": registration.case_version,
        "subject_id": ident.context.subject_id,
        "credential_id": ident.context.credential_id,
        "authorization_revision": ident.context.authorization_revision,
        "request_checksum": registration.checksum,
        "document_version": registration.document_version,
        "dataset_version": registration.dataset_version,
        "index_version": registration.index_version,
        "rules_version": registration.rules_version,
        "isolation": {
            "case_type": registration.case_type,
            "scenario": registration.scenario,
            "purchase_order_id": registration.purchase_order_id,
            "case_document_ids": list(registration.case_document_ids),
            "shared_document_ids": list(registration.shared_document_ids),
            "allowed_order_ids": list(registration.allowed_order_ids),
            "allowed_supplier_ids": list(registration.allowed_supplier_ids),
            "allowed_transaction_ids": list(registration.allowed_transaction_ids),
        },
        "idempotency_key": f"v-03a-{secrets.token_hex(4)}",
        "request_digest": request_digest or canonical_checksum({"create": run_id}),
        "owner_id": "owner-" + secrets.token_hex(16),
        "now": datetime.now(UTC),
        "lease_seconds": 1000.0,
        "deadline_seconds": 1000.0,
        "events": () if events is None else events,
    }


def processing() -> ReviewLifecycleStatus:
    return ReviewLifecycleStatus.PROCESSING


def pending_confirmation() -> ReviewLifecycleStatus:
    return ReviewLifecycleStatus.PENDING_CONFIRMATION


def confirmed() -> ReviewLifecycleStatus:
    return ReviewLifecycleStatus.CONFIRMED


def confirm_action() -> ApprovalActionType:
    return ApprovalActionType.CONFIRM


#: Every table a persisted-review state transaction can write, in a fixed order.
LEDGER_TABLES = (
    "review_run",
    "review_run_generation",
    "review_worker_attempt",
    "review_branch_checkpoint",
    "review_run_summary",
    "review_idempotency",
    "review_receipt",
    "review_trace_event",
    "review_audit_event",
)

_RUN_SCOPED_TABLES = (
    "review_run_generation",
    "review_worker_attempt",
    "review_branch_checkpoint",
    "review_run_summary",
    "review_idempotency",
    "review_receipt",
    "review_trace_event",
    "review_audit_event",
)


def _rows(connection: Any, table: str, run_id: str) -> list[Any]:
    """Every committed row of one table, as JSON text, in a stable order."""

    result = connection.execute(
        text(f"SELECT to_jsonb(t) AS body FROM {table} t WHERE run_id = :r ORDER BY 1"),
        {"r": run_id},
    ).all()
    return [json.dumps(row.body, sort_keys=True, default=str) for row in result]


def ledger_snapshot(fixture: Fix01aFixture, run_id: str) -> dict[str, Any]:
    """The committed facts of one Run plus the audit allocator, as one comparable value.

    Rows are compared as their JSON projection, so a refused transaction that left a
    partially written row behind cannot hide behind an equal row count.
    """

    with fixture.driver.transaction() as connection:
        snapshot: dict[str, Any] = {
            table: _rows(connection, table, run_id) for table in LEDGER_TABLES
        }
        snapshot["allocator"] = [
            (str(row.stream), int(row.next_value))
            for row in connection.execute(
                text(
                    "SELECT stream, next_value FROM review_sequence_allocator "
                    "ORDER BY stream"
                )
            ).all()
        ]
    snapshot["digest"] = canonical_checksum(snapshot)
    return snapshot


def run_ledger_body(fixture: Fix01aFixture, run_id: str) -> dict[str, Any]:
    """The scalar facts of the Run row and its counter tables, for readable asserts."""

    with fixture.driver.transaction() as connection:
        run = connection.execute(
            text(
                "SELECT status, revision, resume_count, trace_id, error_code "
                "FROM review_run WHERE run_id = :r"
            ),
            {"r": run_id},
        ).one()
        counts = {
            table: int(
                connection.execute(
                    text(f"SELECT count(*) FROM {table} WHERE run_id = :r"),
                    {"r": run_id},
                ).scalar_one()
            )
            for table in _RUN_SCOPED_TABLES
        }
    return {
        "status": run.status,
        "revision": int(run.revision),
        "resume_count": int(run.resume_count),
        "trace_id": run.trace_id,
        "error_code": run.error_code,
        "counts": counts,
    }


@contextmanager
def open_mapping_reversion() -> Iterator[None]:
    """Restore the pre-FIX-03A walk of an open mapping, for one negative control.

    The old guard recursed into a registered mapping as an ordinary ``dict[str, JsonValue]``
    and exempted a value in an ``Any`` slot from the value patterns whenever it looked like
    a minted identity.  Both halves are re-applied here and undone on exit.
    """

    saved = dict(record_contracts.MAPPING_KEY_REGISTRIES)
    saved_kind = record_guard._annotation_kind
    for key in list(record_contracts.MAPPING_KEY_REGISTRIES):
        del record_contracts.MAPPING_KEY_REGISTRIES[key]

    def pre_fix_annotation_kind(annotation: Any) -> str:
        resolved = record_guard._unwrap_optional(annotation)
        if resolved is Any or getattr(resolved, "__name__", "") in {
            "JsonValue",
            "JsonValueType",
        }:
            return "any"
        return saved_kind(annotation)

    record_guard._annotation_kind = pre_fix_annotation_kind
    try:
        yield
    finally:
        record_contracts.MAPPING_KEY_REGISTRIES.clear()
        record_contracts.MAPPING_KEY_REGISTRIES.update(saved)
        record_guard._annotation_kind = saved_kind


class _UncheckedRecord:
    """A closed record built without validation, standing in for the pre-fix write path.

    The old entry points bound their SQL parameters straight from the caller's plain
    values; ``model_construct`` is the same thing expressed through the current type, so
    the surrounding code (including its ``isinstance`` checks) still runs unchanged.
    """

    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = tuple(fields)

    def __call__(self, kind: str, record: Any, **kwargs: Any) -> Any:
        if kind not in self._fields:
            return record_guard.guard_record(kind, record, **kwargs)
        payload = dict(record) if isinstance(record, Mapping) else _fields_of(record)
        model = record_guard.RECORD_MODELS[kind]
        return model.model_construct(**payload)


def _fields_of(record: BaseModel) -> dict[str, Any]:
    return {name: getattr(record, name) for name in type(record).model_fields}


def unchecked_guard(*kinds: str) -> _UncheckedRecord:
    """A guard that skips validation for exactly the named kinds (negative controls)."""

    return _UncheckedRecord(kinds)
