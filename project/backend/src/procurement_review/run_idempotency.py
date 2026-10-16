"""The idempotency namespace of one subject: replay decision and conflict.

The namespace key itself — the scoped lookup and the conflict-safe claim every write
entry point goes through — lives in :mod:`.run_idempotency_namespace` and is re-exported
here, so the replay rules below and their readers keep one import site.  A replay never
writes: the committed operation is the whole answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .run_errors import IdempotencyConflictError, RunNotFoundError
from .run_idempotency_namespace import claim_operation, load_operation
from .run_records import (
    CreateOutcome,
    IDEMPOTENT_FAILED,
    IDEMPOTENT_REPLAY,
    PersistedRun,
    ResumeOutcome,
    load_run,
)
from .trace import canonical_checksum


def create_replay(connection: Any, existing: Any, request_digest: str) -> CreateOutcome:
    """Answer a duplicate create from the committed operation, never by writing."""

    if existing.request_digest != request_digest:
        raise IdempotencyConflictError(
            "the create key was already used with a different request"
        )
    run = load_run(connection, existing.run_id)
    if run is None:
        raise RunNotFoundError("the replayed run no longer exists")
    return CreateOutcome(run=run, created=False)


def resume_replay(
    connection: Any, run_id: str, existing: Any, request_digest: str
) -> ResumeOutcome:
    """Answer a duplicate resume from the committed operation, never by writing."""

    if existing.request_digest != request_digest:
        raise IdempotencyConflictError(
            "the resume key was already used with a different request"
        )
    current = load_run(connection, run_id)
    assert current is not None
    if existing.outcome == IDEMPOTENT_FAILED:
        return ResumeOutcome(
            run=current,
            kind=IDEMPOTENT_FAILED,
            failure_code=existing.failure_code,
        )
    # The committed operation is the answer; a replay never re-decides, re-counts or
    # re-dispatches anything.
    return ResumeOutcome(
        run=current,
        kind=IDEMPOTENT_REPLAY,
        generation=applied_generation(current),
    )


def applied_generation(run: PersistedRun) -> int | None:
    """The generation one resumed Run currently holds, if it has one."""

    if run.generation is None:
        return None
    return run.generation.generation


__all__ = [
    "applied_generation",
    "claim_operation",
    "create_replay",
    "load_operation",
    "resume_replay",
]


def idempotency_key_digest(key: str) -> str:
    """Return the stored digest of one raw idempotency key."""

    return canonical_checksum({"idempotency_key": key})


def checkpoints_digest(checkpoints: Mapping[str, Mapping[str, Any]]) -> str:
    """Reduce the identity of the committed checkpoint set to one checksum.

    Only the re-usable identity of each branch takes part: worker, input checksum and
    contract checksum.  A resume decision computed outside a transaction can therefore be
    re-verified against the same rows under the Run lock.
    """

    return canonical_checksum(
        [
            {
                "worker": worker,
                "input_checksum": str(item.get("input_checksum", "")),
                "contract_checksum": str(item.get("contract_checksum", "")),
            }
            for worker, item in sorted(checkpoints.items())
        ]
    )
