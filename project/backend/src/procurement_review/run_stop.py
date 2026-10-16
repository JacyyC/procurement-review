"""The frozen V1-08 stop contract: one classification, one batch send gate (Spec P-05).

The old H consumer knew two lists and only stopped the batch after a stop reason had
already been recorded.  The frozen contract needs more than that: an unregistered reason
or an unrecognized provider error stops the batch until it is classified, an account or
subscription exhaustion stops it immediately, a dense 429 window is a machine-decidable
stop, authority or storage moving underneath the measurement stops the batch, while a
measured Run failure and a controlled per-Case refusal keep their places.

The classification therefore lives in one module with the Spec P-05 priority order, and
every sender consults the gate *before* a physical attempt instead of checking a field
after it.  An event that matches more than one condition stops at the most severe level
while every reason it carried stays in the protected notes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

#: Which level one recorded stop reason stops at.
BATCH = "batch"
CASE = "case"

#: The versioned shape of the durable batch-level stop a batch writes when it stops
#: sending (Spec P-05).  It is the record later admissions read: a batch that stopped stays
#: stopped across a new controller, a new process and a repeated CLI call, and only a
#: contract-required new pre-registration may open its sending again.
BATCH_STOP_RECORD = "v1-08h-batch-stop.v1"

#: Stop reasons that end the whole batch: the product's identity, credential or authority
#: moved, so nothing measured after them can be trusted.  Spec P-05 priority 1.
H_STOP_UNTRUSTED = frozenset(
    {"authentication", "credential_in_request", "model_identity_drift"}
)
#: Stop reasons that are measured per-observation outcomes: the product failed closed at
#: its own boundary (a controlled refusal or an unauthorized citation) and recorded why, so
#: the observation keeps its place and the batch continues.  Spec P-05 priority 8.
H_STOP_MEASURED = frozenset(
    {"security_boundary", "check_outside_branch", "citation_outside_branch"}
)

#: The fixed classification codes this contract itself produces.  A provider signal or a
#: failure code outside the registered vocabulary is never echoed into the records; it is
#: reported under one of these versioned names instead.
UNREGISTERED_PROVIDER_ERROR = "unregistered_provider_error"
ACCOUNT_QUOTA_EXHAUSTED = "account_quota_exhausted"
RATE_LIMITED_DENSE = "rate_limited_dense"
UNRECEIPTED_UNKNOWN = "interrupted_unknown"
#: The code a Run interrupted *because the batch had already stopped* records.  It
#: acknowledges an existing stop and is never a new or more severe reason by itself.
BATCH_STOPPED = "batch_stopped"

#: The Spec P-05 priority order of stop conditions (0 is never a valid rank).
TRUST_BOUNDARY = (
    "trust_boundary"  # 1 authority, credential, identity, configuration drift
)
UNREGISTERED = "unregistered"  # 2 unregistered stop reason / unknown provider error
UNRECEIPTED = "unreceipted"  # 3 no trusted receipt / unknown send status / unreconciled
ACCOUNT_QUOTA = "account_quota"  # 4 account or subscription exhausted
RATE_LIMITED_DENSE_CLASS = "rate_limited_dense"  # 5 registered dense-429 window
RECOVERABLE = "recoverable"  # 6 ordinary 429 / 5xx / transport timeout
USAGE_UNKNOWN_CLASS = "usage_unknown"  # 7 one locatable unknown usage reading
CONTROLLED_REFUSAL = (
    "controlled_refusal"  # 8 controlled per-Case refusal / recorded branch outcome
)

#: The priority of every class, exactly as Spec P-05 numbers them.
STOP_PRIORITY: dict[str, int] = {
    TRUST_BOUNDARY: 1,
    UNREGISTERED: 2,
    UNRECEIPTED: 3,
    ACCOUNT_QUOTA: 4,
    RATE_LIMITED_DENSE_CLASS: 5,
    RECOVERABLE: 6,
    USAGE_UNKNOWN_CLASS: 7,
    CONTROLLED_REFUSAL: 8,
}
#: The ranks at which the batch stops sending: priorities 1 through 5.  Ranks 6 to 8 are
#: recorded, retried or measured outcomes that never end the batch on their own.
BATCH_STOP_RANKS = frozenset({1, 2, 3, 4, 5})

#: Run-level interruption reasons that keep the batch running: the Run expired its own
#: window or lease and the product recorded which one.
RUN_INTERRUPT_CONTINUES = frozenset({"generation_deadline_exceeded", "lease_expired"})
#: Run-level interruption reasons that end the batch: the generation lost its live
#: authority, so no later send can be trusted to belong to this measurement.
RUN_INTERRUPT_AUTHORITY = frozenset({"authorization_revoked"})
#: Run-level interruption reasons about persistence and settlement: the record of what was
#: already sent can no longer be trusted or closed.
RUN_INTERRUPT_PERSISTENCE = frozenset({"storage_unavailable", "batch_stopped"})

#: Model-attempt outcomes a branch may retry inside its own shared two-attempt budget.
#: They are the same codes the persisted receipts label as transport retries.
RETRYABLE_ATTEMPT_ERRORS = frozenset(
    {"timeout", "transport", "rate_limited", "server_error"}
)

#: Registered attempt outcomes that are recorded as this observation's or branch's own
#: bounded failure and never stop the batch: the controlled refusals the product raised at
#: its own boundary plus the registered format, citation and budget outcomes.  A code
#: outside this set and the sets above is unregistered content and stops the batch.
MEASURED_ATTEMPT_ERRORS = H_STOP_MEASURED | frozenset(
    {
        "attempt_budget_exhausted",
        "citation_field_value_unsupported",
        "citation_proposition_unsupported",
        "deadline_exhausted",
        "duplicate_citation",
        "evidence_envelope_invalid",
        "invalid_json",
        "invalid_schema",
        "length_truncated",
        "required_check_missing_or_duplicate",
        "response_envelope_invalid",
        "safety_refusal",
        "tool_call_forbidden",
        "unsupported_request",
    }
)
#: Registered outcome families whose codes carry their own bounded category suffix.
MEASURED_ATTEMPT_PREFIXES = ("record_guard_refused:",)

#: The failure codes the product itself records on a Run it ended under its own control
#: (the ``mark_failed`` callers).  A recorded failure outside this set cannot be read as a
#: measured product outcome and is refused by name instead.
RUN_FAILURE_CODES = frozenset(
    {
        "generation_missing",
        "checkpoint_incompatible",
        "review_execution_failed",
        "material_scope_incomplete",
    }
)

#: The most a single retry waits before the next physical attempt of the same branch.
MAX_RETRY_BACKOFF_SECONDS = 10.0


class BatchStoppedSignal(BaseException):
    """Raised before a new physical attempt once the batch has stopped sending.

    It is a ``BaseException`` like the other controlled stop signals, so a broad
    ``except Exception`` cannot silently turn "the batch stopped" into an ordinary
    execution failure.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or f"batch stopped: {reason}")
        self.reason = reason
        self.message = message


@dataclass(frozen=True, slots=True)
class StopPolicy:
    """The registered dense-429 window and threshold of one batch.

    They are batch admission parameters (Spec P-05): a batch without them is refused
    admission rather than run with an unstated policy.
    """

    rate_limit_window_seconds: int
    rate_limit_burst_threshold: int

    def check(self) -> None:
        if not 1 <= self.rate_limit_window_seconds <= 3_600:
            raise ValueError("h_rate_limit_window_out_of_range")
        if not 2 <= self.rate_limit_burst_threshold <= 64:
            raise ValueError("h_rate_limit_burst_threshold_out_of_range")


def stop_level(code: str | None) -> str:
    """Which level one recorded stop reason stops at.

    A reason in the measured set is a per-case outcome; everything else — including an
    unregistered code — stops the batch, because an unclassified stop cannot be read as
    "the product failed closed at its own boundary".
    """

    if code in H_STOP_MEASURED:
        return CASE
    return BATCH


def priority_of(code: str) -> str:
    """The Spec P-05 priority class of one stop code, whatever event produced it."""

    if code in H_STOP_UNTRUSTED or code in RUN_INTERRUPT_AUTHORITY:
        return TRUST_BOUNDARY
    if code == ACCOUNT_QUOTA_EXHAUSTED:
        return ACCOUNT_QUOTA
    if code == RATE_LIMITED_DENSE:
        return RATE_LIMITED_DENSE_CLASS
    if code == UNRECEIPTED_UNKNOWN or code in RUN_INTERRUPT_PERSISTENCE:
        return UNRECEIPTED
    if code == UNREGISTERED_PROVIDER_ERROR:
        return UNREGISTERED
    if code in H_STOP_MEASURED:
        return CONTROLLED_REFUSAL
    if code in RETRYABLE_ATTEMPT_ERRORS:
        return RECOVERABLE
    if code in MEASURED_ATTEMPT_ERRORS or code.startswith(MEASURED_ATTEMPT_PREFIXES):
        return CONTROLLED_REFUSAL
    return UNREGISTERED


def attempt_priority(code: str) -> str:
    """The Spec P-05 priority class of one settled model attempt outcome.

    Everything outside the registered vocabulary is unregistered content, not an ordinary
    business failure: the batch stops until the code has been classified and registered.
    """

    if code in H_STOP_UNTRUSTED:
        return TRUST_BOUNDARY
    if code == ACCOUNT_QUOTA_EXHAUSTED:
        return ACCOUNT_QUOTA
    if code == UNRECEIPTED_UNKNOWN:
        return UNRECEIPTED
    if code in RETRYABLE_ATTEMPT_ERRORS:
        return RECOVERABLE
    if code in MEASURED_ATTEMPT_ERRORS or code.startswith(MEASURED_ATTEMPT_PREFIXES):
        return CONTROLLED_REFUSAL
    return UNREGISTERED


def run_priority(status: str | None, error_code: str | None) -> str | None:
    """The Spec P-05 priority class of one committed Run terminal state, if any.

    A Run that reached its gate with a recorded failure, or a cancelled Run, is a measured
    outcome: its branch-level causes were already reported through ``note_attempt``.  A Run
    the product ended under its own control carries one of the registered failure codes;
    an interruption keeps the batch running only for the registered batch-neutral reasons.
    Anything else — including a code-less failure or an interruption — is unregistered
    content and stops the batch.
    """

    if status == "failed":
        if error_code is None or error_code in RUN_FAILURE_CODES:
            return None
        return UNREGISTERED
    if status != "interrupted":
        return None
    if error_code in RUN_INTERRUPT_CONTINUES:
        return RECOVERABLE
    if error_code in RUN_INTERRUPT_AUTHORITY:
        return TRUST_BOUNDARY
    if error_code in RUN_INTERRUPT_PERSISTENCE:
        return UNRECEIPTED
    return UNREGISTERED


@dataclass
class BatchStopController:
    """The shared stop decision of one batch.

    The scheduler asks ``stopped`` before it starts a case or an arm, every sender calls
    ``gate`` before a physical attempt, and every settled attempt, committed Run and
    recorded stop reason is reported through ``note_attempt``/``note_run``/
    ``note_stop_reason`` so the dense-429 window, the trust-boundary stops, the account-quota
    refusal and the run-level interruptions all reach the same decision point.

    The recorded stop is the most severe reason seen so far (Spec P-05: one event may
    match several conditions, the highest level decides and every reason stays recorded);
    ``notes`` keeps the whole classification trail for the protected record and
    ``summary`` returns the counts the batch evidence records.

    ``on_stop`` is called once, at the moment the first batch-level reason is recorded, so
    the batch can persist why sending ended before another admission can look at it.  A
    controller without the callback keeps the historical in-memory behaviour; a controller
    with it announces the transition exactly once per process.
    """

    policy: StopPolicy | None = None
    reason: str | None = None
    trigger: dict = field(default_factory=dict)
    priority: str | None = None
    notes: list[dict] = field(default_factory=list)
    attempts_noted: int = 0
    rate_limited_events: list[datetime] = field(default_factory=list)
    on_stop: Callable[[BatchStopController], None] | None = None
    announced: bool = field(default=False, repr=False)

    @property
    def stopped(self) -> bool:
        return self.reason is not None

    def _note(
        self, kind: str, code: str, priority: str, *, stops: bool = True, **facts
    ) -> None:
        """Record one classified event; stop the batch when its rank says so.

        ``stops=False`` records a classification that must never decide or rewrite the
        recorded stop — used for the acknowledgement a Run writes when it was interrupted
        only because the batch had already stopped.
        """

        rank = STOP_PRIORITY[priority]
        self.notes.append(
            {"kind": kind, "code": code, "priority": priority, "rank": rank, **facts}
        )
        if not stops or rank not in BATCH_STOP_RANKS:
            return
        current = STOP_PRIORITY[self.priority] if self.priority is not None else 99
        if self.reason is None or rank < current:
            self.reason = code
            self.trigger = dict(facts)
            self.priority = priority
            if self.on_stop is not None and not self.announced:
                # The transition itself is announced once; the callback persists it.
                self.announced = True
                self.on_stop(self)

    def stop(self, reason: str, **trigger) -> None:
        """Record one explicit stop through the same single classification."""

        self._note("stop", reason, priority_of(reason), **trigger)

    def gate(self) -> None:
        if self.reason is not None:
            raise BatchStoppedSignal(self.reason)

    def note_attempt(self, error_code: str | None, at: datetime) -> None:
        """Report one settled model attempt to the batch decision.

        The dense-429 window counts physical 429 responses inside the registered rolling
        window; an account-quota refusal is classified on its own and never waits for that
        window, and an unregistered outcome stops the batch instead of continuing as an
        ordinary business failure.
        """

        self.attempts_noted += 1
        if error_code is None:
            return
        if error_code == "rate_limited":
            self.rate_limited_events.append(at)
            self._note("attempt", error_code, RECOVERABLE, at=at.isoformat())
            if self.policy is not None:
                window = [
                    moment
                    for moment in self.rate_limited_events
                    if at - moment
                    <= timedelta(seconds=self.policy.rate_limit_window_seconds)
                ]
                self.rate_limited_events = window
                if len(window) >= self.policy.rate_limit_burst_threshold:
                    self._note(
                        "dense",
                        RATE_LIMITED_DENSE,
                        RATE_LIMITED_DENSE_CLASS,
                        window_seconds=self.policy.rate_limit_window_seconds,
                        threshold=self.policy.rate_limit_burst_threshold,
                        observed=len(window),
                    )
            return
        if error_code == ACCOUNT_QUOTA_EXHAUSTED:
            self._note("attempt", error_code, ACCOUNT_QUOTA, at=at.isoformat())
            return
        if error_code == UNRECEIPTED_UNKNOWN:
            self._note("attempt", error_code, UNRECEIPTED, at=at.isoformat())
            return
        priority = attempt_priority(error_code)
        if priority == UNREGISTERED:
            # The observed code stays in the protected note; the stop itself carries the
            # fixed versioned classification so no unknown string becomes a stop reason.
            self._note(
                "attempt",
                UNREGISTERED_PROVIDER_ERROR,
                UNREGISTERED,
                attempt_error=error_code,
                at=at.isoformat(),
            )
            return
        self._note("attempt", error_code, priority, at=at.isoformat())

    def note_stop_reason(self, reason: str | None) -> None:
        """Report the reason one committed observation recorded for ending its sends.

        A reason in the measured set is a per-case outcome; anything else — including an
        unregistered code — stops the batch, because an unclassified stop cannot be read
        as "the product failed closed at its own boundary".
        """

        if not reason:
            return
        self._note("stop_reason", reason, priority_of(reason))

    def note_run(self, status: str | None, error_code: str | None) -> None:
        """Report one committed Run's terminal state to the batch decision."""

        priority = run_priority(status, error_code)
        if priority is None:
            return
        if priority == UNREGISTERED:
            self._note(
                "run",
                UNREGISTERED_PROVIDER_ERROR,
                UNREGISTERED,
                run_status=status,
                run_error=error_code,
            )
            return
        # A Run interrupted because the batch had already stopped records that fact; once a
        # reason is recorded, this acknowledgement never rewrites why sending ended (the
        # same way it never *is* why the batch stopped).
        acknowledged = (
            status == "interrupted"
            and error_code == BATCH_STOPPED
            and self.reason is not None
        )
        self._note(
            "run",
            error_code,
            priority,
            stops=not acknowledged,
            run_status=status,
        )

    def summary(self) -> dict:
        return {
            "stopped": self.stopped,
            "reason": self.reason,
            "priority": self.priority,
            "trigger": dict(self.trigger),
            "attempts_noted": self.attempts_noted,
            "rate_limited_events": len(self.rate_limited_events),
            "notes": [dict(note) for note in self.notes],
        }


__all__ = [
    "ACCOUNT_QUOTA",
    "ACCOUNT_QUOTA_EXHAUSTED",
    "BATCH",
    "BATCH_STOP_RANKS",
    "CASE",
    "CONTROLLED_REFUSAL",
    "H_STOP_MEASURED",
    "H_STOP_UNTRUSTED",
    "MAX_RETRY_BACKOFF_SECONDS",
    "MEASURED_ATTEMPT_ERRORS",
    "MEASURED_ATTEMPT_PREFIXES",
    "RATE_LIMITED_DENSE",
    "RATE_LIMITED_DENSE_CLASS",
    "RECOVERABLE",
    "RETRYABLE_ATTEMPT_ERRORS",
    "RUN_FAILURE_CODES",
    "RUN_INTERRUPT_AUTHORITY",
    "RUN_INTERRUPT_CONTINUES",
    "RUN_INTERRUPT_PERSISTENCE",
    "STOP_PRIORITY",
    "TRUST_BOUNDARY",
    "UNRECEIPTED",
    "UNRECEIPTED_UNKNOWN",
    "UNREGISTERED",
    "UNREGISTERED_PROVIDER_ERROR",
    "USAGE_UNKNOWN_CLASS",
    "BatchStopController",
    "BatchStoppedSignal",
    "StopPolicy",
    "attempt_priority",
    "priority_of",
    "run_priority",
    "stop_level",
]
