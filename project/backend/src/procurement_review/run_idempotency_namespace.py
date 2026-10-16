"""The idempotency namespace key itself: one scoped lookup and one conflict-safe claim.

An operation is identified by ``(tenant, subject, operation family, key digest)``.  The
lookup is always scoped to the calling subject's own namespace, so one subject's committed
operation can never answer another's request.  The claim is the single statement through
which every write entry point registers an operation: the primary key is the final
arbiter, so two transactions that both read "no operation" cannot both proceed — the
second one waits for the first to commit and then writes nothing, and the loser has
written nothing it would have to undo.

FIX-06D moved these two primitives next to each other and out of ``run_idempotency``,
which keeps the replay/conflict decisions and re-exports both names, so no caller and no
structural guard had to change.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text


def load_operation(
    connection: Any,
    *,
    tenant_id: str,
    subject_id: str,
    family: str,
    key_digest: str,
) -> Any:
    """Read one committed operation of the calling subject's own namespace."""

    return connection.execute(
        text(
            "SELECT * FROM review_idempotency WHERE tenant_id = :t "
            "AND subject_id = :s AND operation_family = :family "
            "AND key_digest = :k"
        ),
        {"t": tenant_id, "s": subject_id, "family": family, "k": key_digest},
    ).first()


#: The one statement that claims a namespace key.  The primary key
#: ``(tenant, subject, operation family, key digest)`` is the final arbiter, so two
#: transactions that both read "no operation" cannot both proceed: PostgreSQL makes the
#: second one wait for the first to commit and then writes nothing.
_CLAIM_STATEMENT = text(
    """
    INSERT INTO review_idempotency (
        tenant_id, subject_id, operation_family, key_digest,
        request_digest, run_id, action, receipt_id, first_revision, outcome,
        failure_code, created_at
    ) VALUES (
        :tenant_id, :subject_id, :operation_family, :key_digest,
        :request_digest, :run_id, :action, :receipt_id, :first_revision, :outcome,
        :failure_code, :created_at
    )
    ON CONFLICT (tenant_id, subject_id, operation_family, key_digest)
    DO NOTHING
    RETURNING key_digest
    """
)


def claim_operation(connection: Any, record: Any) -> bool:
    """Claim one namespace key inside the caller's transaction.

    Every write entry point that registers an operation — a new create, a new resume and a
    permanent rejection alike — claims its key through this one statement before it writes
    anything else.  A lookup followed by a plain ``INSERT`` is not enough: two transactions
    can both read "no operation" (a Run lock only serializes the *same* Run), and the losing
    one would surface the constraint violation as a dependency failure instead of the
    contract's replay-or-conflict answer.  A loser has written nothing and can answer from
    the committed operation.
    """

    claimed = connection.execute(
        _CLAIM_STATEMENT,
        {
            "tenant_id": record.tenant_id,
            "subject_id": record.subject_id,
            "operation_family": record.operation_family,
            "key_digest": record.key_digest,
            "request_digest": record.request_digest,
            "run_id": record.run_id,
            "action": record.action,
            "receipt_id": record.receipt_id,
            "first_revision": record.first_revision,
            "outcome": record.outcome,
            "failure_code": record.failure_code,
            "created_at": record.created_at,
        },
    ).first()
    return claimed is not None


__all__ = [
    "claim_operation",
    "load_operation",
]
