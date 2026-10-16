"""The read-side half of the record contract (FIX-03B, closed for FIX-01B).

A record that was committed by an older build, or by a connection the product does not
own, is still judged by the rule the current build applies to its own writes.  Every read
and export path calls one of the functions here *after* its authorization decision and
before it projects anything, so a stored record that no longer satisfies its closed type,
or that carries a sensitive value outside its declared slot, is refused instead of being
republished.

Four properties matter and are deliberately kept:

* the check runs on the **complete** record.  A reader refuses from the raw rows it loaded
  (:mod:`.record_rows`), so a value that only lives in a relation column, in a field the
  component did not need, or in the shape of a stored payload is still seen;
* the refusal is a controlled one.  :class:`RecordUnavailableError` carries the record kind
  and the fixed finding categories — never the value, the JSON path, the validation input
  or the database error — so an API answer and a log line cannot leak what they refused.
  Anything a damaged record can make a reader raise on the way out is translated into that
  same answer here, because a parse error is a fact about the record, not about the caller;
* a dependency failure is never reported as a damaged record.  Only the record-shape
  failures below are translated, so a store that is down keeps its own answer;
* the check is pure.  It reads what the caller already loaded and performs no write of its
  own, so refusing a record cannot change a byte of history or a Run's state.

Compatibility is explicit rather than lenient: a historical record is read through a
declared version (see :mod:`.record_rows`), and the *complete* record of that version is
what gets verified.  No reader drops a field to make an old record pass.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .record_rows import isolation_kind, row_findings
from .run_errors import (
    FieldGuardError,
    RecordIncompleteError,
    RecordUnavailableError,
)
from .service_errors import ReviewServiceError

logger = logging.getLogger("procurement_review.record_reads")

READ_PROJECTION_VERSION = "review-record-read.v2"

#: The fixed answer for a record that cannot be read.  Kept as constants so no path can
#: vary the message and turn it into a channel for what was refused.
RECORD_UNAVAILABLE_CODE = "review_record_unavailable"
RECORD_UNAVAILABLE_MESSAGE = "The stored review record is not available for reading."

#: Everything a damaged stored record can make a reader raise while it parses, adapts a
#: declared version, or projects an answer: the refusal is the same fixed one either way.
#:
#: Two families are deliberately **not** in this set.  A storage failure keeps its own
#: dependency answer, so an outage is never reported as a damaged record.  And an
#: *incomplete* record — one that carries no Trace identity at all — stays the typed
#: :class:`RecordIncompleteError` a repository caller reads directly (FIX-01B V-B-03); the
#: HTTP surface gives it the very same 503 envelope, so no caller can see a traceback for it
#: while a state transaction can still tell the two conditions apart.
RECORD_FAILURES: tuple[type[BaseException], ...] = (
    RecordUnavailableError,
    FieldGuardError,
    ValidationError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

#: The category a refusal reports when the failure was an escaping exception rather than a
#: finding the closed contract named.
PARSE_FAILURE = "record_parse_failure"


def record_unavailable() -> ReviewServiceError:
    """The one answer for a committed record that no longer satisfies its contract.

    The code, message and details are fixed strings: neither the value that failed, nor
    the validation input, nor the SQL or the original exception travels with it (AC-04.2).
    """

    return ReviewServiceError(
        status_code=503,
        code=RECORD_UNAVAILABLE_CODE,
        message=RECORD_UNAVAILABLE_MESSAGE,
        details={},
    )


def _refusal_tokens(error: BaseException) -> tuple[str, str]:
    """The only two facts a read refusal may name: a record kind and a fixed category."""

    if isinstance(error, RecordUnavailableError):
        return error.record, ",".join(error.categories) or PARSE_FAILURE
    return "review_run", PARSE_FAILURE


def record_refusal(*, location: str = "review_run") -> Any:
    """Translate a read-side record failure into that one controlled answer.

    The original exception is not chained: a record that made a reader raise keeps its
    contents out of any traceback the server prints, and the log line names only the fixed
    record kind and category.
    """

    @contextlib.contextmanager
    def translate() -> Iterator[None]:
        try:
            yield
        except RECORD_FAILURES as error:
            record, categories = _refusal_tokens(error)
            logger.info(
                "review read refused: code=%s record=%s categories=%s",
                RECORD_UNAVAILABLE_CODE,
                record or location,
                categories,
            )
            raise record_unavailable() from None

    return translate()


def _require_clean(payload: Any, *, kind: str, location: str) -> None:
    """Refuse one stored payload that no longer satisfies its closed type."""

    if payload is None:
        return
    categories = row_findings_for(kind, payload, location=location)
    if categories:
        raise RecordUnavailableError(record=location, categories=categories)


def row_findings_for(kind: str, payload: Any, *, location: str) -> tuple[str, ...]:
    from .record_guard import scan_persisted

    try:
        report = scan_persisted(payload, kind=kind, location=location)
    except Exception as error:  # noqa: BLE001 - judging a record never raises
        return (f"{PARSE_FAILURE}:{type(error).__name__}",)
    return tuple(sorted({item.category for item in report.findings}))


def verify_run_record(run: Any, *, location: str = "review_run") -> None:
    """Verify every record one committed Run is made of, or refuse the whole Run.

    The Run header and its Trace identity, the immutable Case isolation input, the
    execution generation, the Worker attempt ledger, the adopted checkpoints, the fan-in
    summary and execution, the idempotent operations that created them and the local
    receipt are all judged as complete stored rows — and so are the Trace and audit events
    whenever the reader loaded them.  A reader calls this before it projects anything, so a
    Run with one unusable record answers with a single controlled refusal instead of a
    partial view.
    """

    if run.header.trace_id is None:
        # A Run with no committed Trace identity at all is the incomplete record FIX-01B
        # already defined; the reader reports it as such instead of inventing an identity
        # or reporting it as a damaged one.
        raise RecordIncompleteError(
            f"run {run.header.run_id} has no persisted trace identity"
        )
    refusal = run.raw.refusal
    if refusal is not None:
        raise RecordUnavailableError(
            record=refusal.record or location, categories=refusal.categories
        )


def verify_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Verify the complete adopted checkpoint rows before anything is reused from them."""

    for row in rows:
        categories = row_findings("review_branch_checkpoint", row)
        if categories:
            raise RecordUnavailableError(
                record="review_checkpoint", categories=tuple(sorted(set(categories)))
            )


def verify_checkpoint_row(row: Mapping[str, Any]) -> None:
    """Verify one committed checkpoint row on its own."""

    verify_checkpoint_rows((row,))


def verify_trace_records(
    events: Sequence[Any], *, location: str = "review_trace"
) -> None:
    """Verify every committed Trace event of one Run."""

    for event in events:
        _require_clean(event, kind="trace", location=location)


def verify_audit_records(
    events: Sequence[Any], *, location: str = "review_audit"
) -> None:
    """Verify every committed audit event of one Run, after its sequence projection."""

    for event in events:
        _require_clean(event, kind="audit", location=location)


__all__ = [
    "PARSE_FAILURE",
    "READ_PROJECTION_VERSION",
    "RECORD_FAILURES",
    "RECORD_UNAVAILABLE_CODE",
    "RECORD_UNAVAILABLE_MESSAGE",
    "isolation_kind",
    "record_refusal",
    "record_unavailable",
    "row_findings_for",
    "verify_audit_records",
    "verify_checkpoint_row",
    "verify_checkpoint_rows",
    "verify_run_record",
    "verify_trace_records",
]
