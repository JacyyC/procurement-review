"""The durable batch-level stop of one H batch (Spec P-05).

A batch that stopped sending must stay stopped: the reason lives in one versioned record
inside the batch directory, written the moment the shared controller records it and read
by every later admission.  A new controller, a new process and a repeated CLI call all see
the same record — the stop is not something a fresh run clears by starting over — while a
normal Run-level interruption is recovered inside its own batch exactly as before.

The record binds the registered plan: a batch that carries a stop it cannot verify (a
missing plan binding, another schema, an unreadable file) closes sending instead of being
treated as a batch that never stopped.  Nothing here sends, opens a label or reads a
credential.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .eval_h_journal import Journal
from .eval_h_seal import plan_digest
from .eval_h_storage import private, read, write
from .run_stop import (
    BATCH,
    BATCH_STOPPED,
    BATCH_STOP_RECORD,
    BatchStopController,
    stop_level,
)


def stop_record_path(directory: Path) -> Path:
    return Path(directory) / "runner/stop.json"


def read_batch_stop(directory: Path, plan) -> dict | None:
    """The durable batch-level stop of this batch, or ``None`` when it never stopped."""

    path = stop_record_path(directory)
    if not path.exists():
        return None
    try:
        private(path)
        body = read(path)
    except (OSError, ValueError) as error:
        raise ValueError("h_batch_stop_record_unreadable") from error
    if (
        type(body) is not dict
        or body.get("schema_version") != BATCH_STOP_RECORD
        or body.get("plan_sha256") != plan_digest(plan)
        or type(body.get("reason")) is not str
        or not body["reason"]
        or type(body.get("stopped_at")) is not str
        or not body["stopped_at"]
    ):
        raise ValueError("h_batch_stop_record_unreadable")
    return body


def committed_batch_stop(directory: Path, plan) -> dict | None:
    """The batch-level stop this batch's own committed observations already record.

    The record file is the first witness; the committed observations are the second.  A
    batch whose own observation says sending ended at a batch level is preserved as a whole
    (Spec P-05), so a lost or deleted record file cannot reopen it — and an observation
    that cannot be read at all closes sending too, because the stop state is then not
    verifiable either.
    """

    journal = Journal(directory)
    for key in plan.observation_keys():
        try:
            body = journal.committed(*key)
        except (OSError, ValueError) as error:
            raise ValueError("h_batch_stop_observation_unreadable") from error
        if type(body) is not dict:
            continue
        row = body.get("row")
        row = row if type(row) is dict else {}
        reason = row.get("stop_reason")
        if type(reason) is str and reason and stop_level(reason) == BATCH:
            return dict(reason=reason, case_id=key[0], source="observation")
        if row.get("error_code") == BATCH_STOPPED:
            return dict(reason=BATCH_STOPPED, case_id=key[0], source="observation")
    return None


def write_batch_stop(directory: Path, plan, controller: BatchStopController) -> None:
    """Persist why this batch stopped sending, bound to the registered plan."""

    path = stop_record_path(directory)
    body = dict(
        schema_version=BATCH_STOP_RECORD,
        plan_sha256=plan_digest(plan),
        reason=controller.reason,
        priority=controller.priority,
        trigger=dict(controller.trigger),
        stopped_at=datetime.now(UTC).isoformat(),
    )
    if path.exists():
        private(path)
        write(path, body, replace=True)
    else:
        write(path, body)


__all__ = [
    "committed_batch_stop",
    "read_batch_stop",
    "stop_record_path",
    "write_batch_stop",
]
