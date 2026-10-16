"""Protected local reads: one live admission and the data read it guards.

A read that touches the database is admitted on the caller's connection. Authorization
revision and Case grant share locks, then Run and active-generation update locks, stay
held through that same short transaction. Revocation and cancellation either commit
first and are observed, or wait for the protected local step. No caller holds these
connections while waiting for a network, port or model response.

This module is the caller-connection half of the FIX-06A invariant: ``execution_guards``
owns the authorization vocabulary and applies it inside its own short transaction, and
this module applies it to a connection the caller already opened, together with the reads
that must be covered by it.  FIX-06D split the two halves into their own registrable
modules so the guarded files stay inside the size ceilings their baseline records.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .review_store import RegisteredDocument, RegisteredProcurementRecord


def require_generation_live(
    guards: Any,
    connection: Any,
    *,
    run_id: str,
    generation: int,
    now: datetime | Callable[[], datetime],
    owner_id: str | None = None,
    on_authorized: Any | None = None,
) -> None:
    """Lock and re-read execution state on the transaction that performs the step.

    The shared guard takes revision, grant, Run and active generation in that order.
    The caller holds these locks until its local read or synchronous model receipt ends.
    Network/port/model response waits belong outside this transaction.
    """
    from .execution_guards import ExecutionInvalidated

    reason = guards.lock_active_generation(
        connection, run_id=run_id, generation=generation, now=now, owner_id=owner_id
    )
    if reason is not None:
        raise ExecutionInvalidated(reason)
    if on_authorized is not None:
        on_authorized()
    guards.require_execution_time(
        connection, run_id=run_id, generation=generation, now=now
    )


def _document_rows(result: Iterable[Any]) -> list[RegisteredDocument]:
    # The read models stay with the store they describe; the local import breaks the
    # ``review_store`` -> ``admitted_reads`` -> ``review_store`` cycle.
    from .review_store import RegisteredDocument

    return [
        RegisteredDocument(
            document_id=row.document_id,
            ownership=row.ownership,
            tenant_id=row.tenant_id,
            case_id=row.case_id,
            category=row.category,
            source_id=row.source_id,
            source_version_id=row.source_version_id,
            document_version=row.document_version,
            index_version=row.index_version,
            content_checksum=row.content_checksum,
            chunk_payload=row.chunk_payload,
        )
        for row in result
    ]


def load_documents(
    connection: Any,
    *,
    case_document_ids: Sequence[str],
    shared_document_ids: Sequence[str],
    tenant_id: str,
    case_id: str,
    document_version: str,
    index_version: str,
) -> tuple[RegisteredDocument, ...]:
    """Read the registered documents of one scope on a caller-supplied connection.

    A protected preload admits the execution and reads its data in *one* transaction: the
    caller opens that transaction, runs its live-authority check on this same connection
    and only then calls this function.  Opening a second transaction here would give a
    revocation somewhere to land in between.
    """

    rows: list[RegisteredDocument] = []
    if case_document_ids:
        result = connection.execute(
            text(
                """
                SELECT * FROM review_document_registry
                WHERE ownership = 'case_document' AND tenant_id = :t
                  AND case_id = :c AND document_id = ANY(:ids)
                  AND document_version = :dv AND index_version = :iv
                ORDER BY document_id
                """
            ),
            {
                "t": tenant_id,
                "c": case_id,
                "ids": list(case_document_ids),
                "dv": document_version,
                "iv": index_version,
            },
        )
        rows.extend(_document_rows(result))
    if shared_document_ids:
        result = connection.execute(
            text(
                """
                SELECT * FROM review_document_registry
                WHERE ownership = 'shared_fixture' AND tenant_id IS NULL
                  AND document_id = ANY(:ids)
                  AND document_version = :dv AND index_version = :iv
                ORDER BY document_id
                """
            ),
            {
                "ids": list(shared_document_ids),
                "dv": document_version,
                "iv": index_version,
            },
        )
        rows.extend(_document_rows(result))
    return tuple(rows)


def retrieval_scores(
    connection: Any,
    *,
    document_version: str,
    index_version: str,
    chunk_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Read the retrieval scores of one index on a caller-supplied connection."""

    if not chunk_ids:
        return {}
    rows = connection.execute(
        text(
            """
            SELECT query_text, chunk_id, score FROM review_retrieval_score
            WHERE document_version = :dv AND index_version = :iv
              AND chunk_id = ANY(:ids)
            ORDER BY query_text, chunk_id
            """
        ),
        {
            "dv": document_version,
            "iv": index_version,
            "ids": list(chunk_ids),
        },
    ).all()
    scores: dict[str, dict[str, float]] = {}
    for query_text, chunk_id, score in rows:
        scores.setdefault(query_text, {})[chunk_id] = float(score)
    return scores


def procurement_records(
    connection: Any,
    *,
    tenant_id: str,
    case_id: str,
    dataset_version: str,
) -> tuple[RegisteredProcurementRecord, ...]:
    """Read the registered procurement records on a caller-supplied connection."""

    from .review_store import RegisteredProcurementRecord

    rows = connection.execute(
        text(
            """
            SELECT tenant_id, case_id, record_kind, record_id,
                   dataset_version, as_of, payload
            FROM review_procurement_record
            WHERE tenant_id = :t AND case_id = :c
              AND dataset_version = :d
            ORDER BY record_kind, record_id
            """
        ),
        {"t": tenant_id, "c": case_id, "d": dataset_version},
    ).all()
    return tuple(
        RegisteredProcurementRecord(
            tenant_id=row.tenant_id,
            case_id=row.case_id,
            record_kind=row.record_kind,
            record_id=row.record_id,
            dataset_version=row.dataset_version,
            as_of=row.as_of.isoformat(),
            payload=row.payload,
        )
        for row in rows
    )


__all__ = [
    "load_documents",
    "procurement_records",
    "require_generation_live",
    "retrieval_scores",
]
