"""Classify persisted model attempts into provable send, receipt and terminal facts.

A ledger row is not a send.  The product opens the attempt row before it reads anything,
so a row whose process died before the transport was entered was never sent; a row closed
by an interruption may or may not have reached the endpoint; only a persisted receipt
proves that a request was sent *and* answered.  The counts here keep those three states
apart instead of collapsing them into one "physical attempt" number, and they never turn
a missing receipt into a free attempt: an attempt whose result is unknown counts as
unknown usage, not as zero cost.

Every count is taken **inside one logical branch**.  Two branches of the same Run number
their attempts independently, so a policy attempt 2 and a contract attempt 2 are not a
"previous" and a "next" of each other; the correction kinds are decided by pairing each
attempt with the predecessor of its own branch, never by sorting the whole set and
pairing neighbours.  The correction a refusal selects is the one the *refused* attempt's
receipt names for the attempt that followed it in the same branch, so a citation refusal
plus an unrelated branch's first send cannot be read as an unlabelled extra attempt.

The classifier is shared by the product Run and by the evaluation line so both report the
same distinctions from the same persisted facts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: The outcome codes that make the *next* physical attempt a transport retry rather than
#: a content correction.  They are the same codes the drivers treat as retryable.
TRANSPORT_RETRY_CODES = frozenset(
    {"timeout", "transport", "rate_limited", "server_error"}
)

#: The correction kinds an extra attempt can follow.
CORRECTION_KINDS = ("contract", "citation", "transport", "other")


def _branch_of(row: Mapping[str, Any]) -> str:
    """The logical branch one ledger row belongs to, from the row's own identity."""

    for key in ("worker", "task_key", "branch"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


@dataclass(frozen=True, slots=True)
class AttemptFact:
    """One persisted attempt, in the terms the counts below need."""

    attempt_no: int
    has_receipt: bool
    closed: bool
    usage_unknown: bool
    input_tokens: int | None
    output_tokens: int | None
    #: The kind of correction or retry the following attempt of the same branch used.
    next_kind: str | None = None
    #: The logical branch this attempt belongs to.  Empty means "one unnamed branch": a
    #: single-branch caller keeps its old reading, and two branches are never mixed.
    branch_key: str = ""
    #: The line's own local markers, when it records them: an attempt the Run adopted as
    #: its branch commit, and an attempt the ledger marked as superseded/discarded.
    adopted: bool = False
    discarded: bool = False


@dataclass(frozen=True, slots=True)
class PlanFacts:
    """The planned and reserved numbers a ledger reports beside its attempt rows.

    They are inputs, not inferences: a planned task is not an attempt, and a reservation
    is a budget claim, so neither may be reported as a send.  ``None`` means the line does
    not record the number at all.
    """

    planned_tasks: int | None = None
    reserved_requests: int | None = None


@dataclass(frozen=True, slots=True)
class SendMetering:
    """What one Run's or one arm's persisted attempts prove."""

    attempts_opened: int
    proven_sends: int
    unresolved_attempts: int
    open_attempts: int
    first_sends: int
    contract_corrections: int
    citation_corrections: int
    transport_retries: int
    other_extra_attempts: int
    input_tokens_known: int | None
    output_tokens_known: int | None
    usage_complete: bool
    unknown_usage_attempts: int
    #: The plan/reservation figures, passed through unchanged (never derived from rows).
    planned_tasks: int | None = None
    reserved_requests: int | None = None
    #: Rows that reached a local terminal marker, whatever their send state.
    terminal_attempts: int = 0
    adopted_attempts: int = 0
    discarded_attempts: int = 0
    branches: int = 0

    @property
    def extra_attempts(self) -> int:
        return (
            self.contract_corrections
            + self.citation_corrections
            + self.transport_retries
            + self.other_extra_attempts
        )


def kind_of(error_code: str | None, correction: str | None) -> str | None:
    """The correction kind one settled attempt selects for the attempt after it."""

    if correction is None:
        return "transport" if error_code in TRANSPORT_RETRY_CODES else None
    if correction in {"citation", "citation_proposition_unsupported"}:
        return "citation"
    if correction == "transport":
        return "transport"
    return "contract" if error_code else None


def _extra_kinds(items: Iterable[AttemptFact]) -> dict[str, int]:
    """Count the extra attempts of every branch by the correction their predecessor named.

    The order inside one branch is the attempt number the line itself assigned; each
    attempt with a number above one is charged to the attempt immediately before it *in
    its own branch*, and only when that predecessor's row is actually present.  A row
    whose predecessor is missing (a legible gap in the persisted rows) or that ties
    another row on the same number is reported as an unlabelled extra attempt instead of
    being paired with another branch's row or with a correction it did not follow.
    """

    kinds: dict[str, int] = {kind: 0 for kind in CORRECTION_KINDS}
    by_branch: dict[str, list[AttemptFact]] = {}
    for item in items:
        by_branch.setdefault(item.branch_key, []).append(item)
    for group in by_branch.values():
        by_number: dict[int, AttemptFact] = {}
        for item in group:
            # Two rows claiming the same number of one branch: the tie is not a
            # predecessor, so neither row is charged to the other.
            by_number[item.attempt_no] = None if item.attempt_no in by_number else item
        for current in group:
            if current.attempt_no <= 1:
                continue
            previous = by_number.get(current.attempt_no - 1)
            kind = "other" if previous is None else (previous.next_kind or "other")
            kinds[kind] += 1
    return kinds


def classify(
    facts: Iterable[AttemptFact], *, plan: PlanFacts | None = None
) -> SendMetering:
    """The counts one branch-set's attempts support, with nothing inferred."""

    items = list(facts)
    proven = [item for item in items if item.has_receipt]
    unresolved = [item for item in items if not item.has_receipt and item.closed]
    open_rows = [item for item in items if not item.has_receipt and not item.closed]
    kinds = _extra_kinds(items)
    known_input = sum(
        int(item.input_tokens)
        for item in proven
        if not item.usage_unknown and item.input_tokens is not None
    )
    known_output = sum(
        int(item.output_tokens)
        for item in proven
        if not item.usage_unknown and item.output_tokens is not None
    )
    unknown = (
        len(unresolved)
        + len(open_rows)
        + sum(1 for item in proven if item.usage_unknown)
    )
    return SendMetering(
        attempts_opened=len(items),
        proven_sends=len(proven),
        unresolved_attempts=len(unresolved),
        open_attempts=len(open_rows),
        first_sends=sum(1 for item in proven if item.attempt_no == 1),
        contract_corrections=kinds["contract"],
        citation_corrections=kinds["citation"],
        transport_retries=kinds["transport"],
        other_extra_attempts=kinds["other"],
        input_tokens_known=known_input if proven else None,
        output_tokens_known=known_output if proven else None,
        usage_complete=bool(items) and unknown == 0,
        unknown_usage_attempts=unknown,
        planned_tasks=None if plan is None else plan.planned_tasks,
        reserved_requests=None if plan is None else plan.reserved_requests,
        terminal_attempts=sum(1 for item in items if item.closed),
        adopted_attempts=sum(1 for item in items if item.adopted),
        discarded_attempts=sum(1 for item in items if item.discarded),
        branches=len({item.branch_key for item in items}),
    )


def facts_from_receipts(
    rows: Iterable[Mapping[str, Any]],
    *,
    receipt_of: Any,
    closed_of: Any | None = None,
    branch_of: Callable[[Mapping[str, Any]], str] | None = None,
    adopted_of: Any | None = None,
    discarded_of: Any | None = None,
) -> tuple[AttemptFact, ...]:
    """Build facts from ledger rows plus a lookup for their own receipts.

    ``receipt_of(row)`` returns the persisted receipt mapping of one attempt, or ``None``
    when no receipt was ever written; ``closed_of(row)`` says whether the attempt carries a
    terminal marker (default: every row is closed); ``branch_of(row)`` names the logical
    branch the row belongs to (default: the row's own ``worker``/``task_key``), so rows of
    two workers are never paired as if they were consecutive attempts of one branch.
    ``adopted_of``/``discarded_of`` read the line's local markers when it records them.
    Both the SQLite evaluation ledger and the product's PostgreSQL attempt table use this,
    so the two lines classify alike, and the result does not depend on the order in which
    the rows came back.
    """

    reader = branch_of or _branch_of
    facts: list[AttemptFact] = []
    for row in rows:
        receipt = receipt_of(row)
        body = dict(receipt) if isinstance(receipt, Mapping) else {}
        facts.append(
            AttemptFact(
                attempt_no=int(row.get("attempt_no") or 1),
                has_receipt=bool(receipt),
                closed=bool(closed_of(row)) if closed_of else True,
                usage_unknown=bool(body.get("usage_unknown", True)),
                input_tokens=body.get("input_usage", body.get("input_tokens")),
                output_tokens=body.get("output_usage", body.get("output_tokens")),
                next_kind=body.get("next_correction_kind"),
                branch_key=reader(row),
                adopted=bool(adopted_of(row)) if adopted_of else False,
                discarded=bool(discarded_of(row)) if discarded_of else False,
            )
        )
    return tuple(facts)


__all__ = [
    "CORRECTION_KINDS",
    "TRANSPORT_RETRY_CODES",
    "AttemptFact",
    "PlanFacts",
    "SendMetering",
    "classify",
    "facts_from_receipts",
    "kind_of",
]
