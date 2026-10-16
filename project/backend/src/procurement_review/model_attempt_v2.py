"""Independent evaluation admission and durable shared retry/correction budget.

Not a product Run adapter: G must use the database-linearized invocation port.
The batch authority here is explicit and every physical send re-enters its gate.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Callable, Literal

from pydantic import Field, model_validator

from .chat_transport_v2 import ChatTransportV2, AttemptResult, request_payload
from .eval_attempt_ledger import AttemptLedger, LedgerError
from .eval_v2_contracts import AttemptReceiptV1, EvalV2Model
from .gold_v3_contracts import HexSha256, payload_sha256, canonical_json_bytes
from .model_attempt_projection import BranchProjectionV2, ClaimDetail
from .model_wire_v2 import BranchRequestV2


class BudgetProof(EvalV2Model):
    """Trusted operator inputs backed by evidence, never inferred from max_tokens."""

    evidence_sha256: HexSha256
    cumulative_request_upper: int = Field(ge=0)
    cumulative_input_upper: int | None = Field(ge=0)
    cumulative_output_upper: int | None = Field(ge=0)
    limit_policy: Literal["cumulative_brakes", "user_no_balance_limit"] = (
        "cumulative_brakes"
    )
    authorization_basis: Literal["user-explicit-no-balance-limit"] | None = None
    # Unknown stays None even when the operator explicitly removes balance limits.
    historical_unknown_input_reserve: int | None = Field(ge=0)
    historical_unknown_output_reserve: int | None = Field(ge=0)
    allocated_requests: int = Field(gt=0, le=240)
    allocated_input: int | None = Field(gt=0)
    allocated_output: int | None = Field(gt=0)

    @model_validator(mode="after")
    def explicit_unlimited_policy(self):
        if self.limit_policy == "user_no_balance_limit":
            if (
                self.authorization_basis is None
                or self.allocated_input is not None
                or self.allocated_output is not None
            ):
                raise ValueError(
                    "unlimited balance requires explicit authorization and null token caps"
                )
        elif self.allocated_input is None or self.allocated_output is None:
            raise ValueError("bounded policy requires token caps")
        return self

    def require_capacity(self):
        if self.limit_policy == "user_no_balance_limit":
            return
        if (
            self.cumulative_input_upper is None
            or self.cumulative_output_upper is None
            or self.historical_unknown_input_reserve is None
            or self.historical_unknown_output_reserve is None
        ):
            raise LedgerError("historical usage has no defensible reserve")
        if (
            self.cumulative_request_upper + self.allocated_requests > 25_000
            or self.cumulative_input_upper
            + self.historical_unknown_input_reserve
            + self.allocated_input
            > 250_000_000
            or self.cumulative_output_upper
            + self.historical_unknown_output_reserve
            + self.allocated_output
            > 24_000_000
        ):
            raise LedgerError("cumulative experimental brake exceeded")


class AdmissionDenied(RuntimeError):
    pass


class BatchAuthority:
    def __init__(
        self,
        *,
        authorization_id: str,
        scope_ids: frozenset[str],
        expires_at: datetime,
        proof: BudgetProof,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        before_attempt: Callable[[], None] | None = None,
    ):
        if expires_at.utcoffset() is None:
            raise ValueError("batch expiry requires offset")
        self.authorization_id = authorization_id
        self.scope_ids = scope_ids
        self.expires_at = expires_at
        self.proof = proof
        self.now = now
        self.before_attempt = before_attempt
        self._active = True
        self._lock = threading.RLock()

    def cancel(self):
        with self._lock:
            self._active = False

    @contextmanager
    def enter(self, request: BranchRequestV2, deadline: datetime):
        with self._lock:
            self.proof.require_capacity()
            if (
                not self._active
                or request.scope_id not in self.scope_ids
                or self.now() >= min(deadline, self.expires_at)
            ):
                raise AdmissionDenied("batch_authority_or_deadline")
            yield


def configuration_hash(transport: ChatTransportV2, authority: BatchAuthority) -> str:
    return payload_sha256(
        {
            "endpoint": transport.configuration_fingerprint,
            "authority": authority.authorization_id,
            "scope_ids": sorted(authority.scope_ids),
            "expires_at": authority.expires_at.isoformat(),
            "budget": authority.proof.model_dump(mode="json"),
        }
    )


def _history(ledger: AttemptLedger, task_key: str) -> list[dict]:
    with sqlite3.connect(ledger.path) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM attempts WHERE task_key=? ORDER BY attempt_no",
                (task_key,),
            )
        ]


class BranchAttemptDriver:
    stage_id = "V1-07C"
    projection_type = BranchProjectionV2

    def correctable_code(self, code):
        return code in {
            "invalid_json",
            "invalid_schema",
            "required_check_missing_or_duplicate",
            "duplicate_citation",
        }

    def refusal_of(self, observed, request, task_key):
        """A content refusal this line finds after the wire accepted the answer.

        Returns ``(code, location_tokens)`` or ``None``.  The base driver does not ground,
        so its default is no refusal; a line that grounds overrides this and gets its
        remaining attempt spent on the refusal under the same shared budget.
        """

        return None

    def on_correction(self, task_key, correction, correction_fields, refusal):
        """Hook called once the next attempt is committed to a correction of this one."""

        return None

    def make_payload(self, request, correction, correction_fields):
        return request_payload(
            self.transport.config, request, correction, correction_fields
        )

    def make_projection(self, observed, success, *, task_key, **fields):
        return self.projection_type(**fields)

    def discarded_projection(self, projection):
        return projection.model_copy(
            update={
                "claims": (),
                "details": (),
                "first_pass_valid": False,
                "corrected_valid": False,
                "pipeline_failure": True,
            }
        )

    def __init__(
        self,
        ledger: AttemptLedger,
        transport: ChatTransportV2,
        authority: BatchAuthority,
    ):
        self.ledger, self.transport, self.authority = ledger, transport, authority
        manifest = ledger.manifest()
        if (
            manifest.stage_id != self.stage_id
            or manifest.configuration_sha256 != configuration_hash(transport, authority)
        ):
            raise LedgerError("fixed stage configuration mismatch")
        budget = manifest.budget
        proof = authority.proof
        if (budget.max_requests, budget.max_input_tokens, budget.max_output_tokens) != (
            proof.allocated_requests,
            proof.allocated_input,
            proof.allocated_output,
        ):
            raise LedgerError("ledger differs from authorized allocation")

    async def run(
        self,
        task_key: str,
        request: BranchRequestV2,
        *,
        deadline: datetime,
        reserve_input: int,
        reserve_output: int,
        max_attempts: int = 2,
    ) -> BranchProjectionV2:
        if type(max_attempts) is not int or max_attempts not in {1, 2}:
            raise ValueError("physical attempt limit must be one or two")
        if self.ledger.manifest().configuration_sha256 != configuration_hash(
            self.transport, self.authority
        ):
            raise LedgerError("fixed stage configuration mismatch")
        if deadline.utcoffset() is None or reserve_input <= 0 or reserve_output <= 0:
            raise ValueError(
                "explicit deadline and positive token reservations required"
            )
        if (deadline - self.authority.now()).total_seconds() > 120:
            raise ValueError("logical branch deadline exceeds 120 seconds")
        # Immutable task identity binds deadline and reserves across process recovery.
        identity = {
            "request": request.model_dump(mode="json"),
            "deadline": deadline.isoformat(),
            "reserve_input": reserve_input,
            "reserve_output": reserve_output,
            "max_attempts": max_attempts,
        }
        self.ledger.register_task(task_key, payload_sha256(identity))
        if self.ledger.next_action(task_key) == "reuse_committed":
            result = self.ledger.load_committed_projection(task_key)
            if not isinstance(result, self.projection_type):
                raise LedgerError("incompatible branch projection")
            with self.authority.enter(request, deadline):
                return result
        if self.ledger.next_action(task_key) in {
            "recover_sent_attempt",
            "blocked_remote_unknown",
        }:
            raise LedgerError("remote unknown requires explicit recovery; never resend")
        history = _history(self.ledger, task_key)
        correction = None
        correction_fields = ()
        retry_reason = None
        if history:
            completed = [row for row in history if row["receipt_json"]]
            last = completed[-1] if completed else history[-1]
            if last["receipt_json"]:
                receipt = AttemptReceiptV1.model_validate_json(last["receipt_json"])
                if self.correctable_code(receipt.error_code):
                    correction = receipt.error_code
                    correction_fields = receipt.format_fields
                elif receipt.error_code not in {
                    "timeout",
                    "transport",
                    "rate_limited",
                    "server_error",
                }:
                    raise LedgerError("terminal failure cannot be retried")
                retry_reason = receipt.error_code
                if len(history) >= max_attempts and history[-1]["state"] != "reserved":
                    raise LedgerError("physical attempt budget exhausted")
                if receipt.retry_not_before is not None:
                    delay = (
                        receipt.retry_not_before - self.authority.now()
                    ).total_seconds()
                    if delay > 0:
                        with self.authority.enter(request, deadline):
                            if (
                                delay > 10
                                or self.authority.now() + timedelta(seconds=delay + 1)
                                >= deadline
                            ):
                                raise AdmissionDenied("retry_deadline")
                        await asyncio.sleep(delay)
        started = time.monotonic()
        first_started = (
            datetime.fromisoformat(history[0]["created_at"])
            if history
            else self.authority.now()
        )
        while True:
            if self.ledger.manifest().configuration_sha256 != configuration_hash(
                self.transport, self.authority
            ):
                raise LedgerError("fixed stage configuration mismatch")
            payload = self.make_payload(request, correction, correction_fields)
            # Admission, reservation and synchronous scheduling form one local batch
            # gate. Network await happens after releasing the gate.
            with self.authority.enter(request, deadline):
                if self.authority.before_attempt is not None:
                    self.authority.before_attempt()
                attempt = self.ledger.reserve_attempt(
                    task_key,
                    request_sha256=payload_sha256(payload),
                    reserve_input_tokens=reserve_input,
                    reserve_output_tokens=reserve_output,
                    max_attempts=max_attempts,
                )
                self.ledger.mark_sent(attempt.attempt_id)
                began = self.authority.now()
                remaining = min(
                    120.0,
                    (min(deadline, self.authority.expires_at) - began).total_seconds(),
                )

                # Recheck at actual transport entry as well: cancelled queued work
                # cannot reach the socket merely because a coroutine was created.
                async def send():
                    with self.authority.enter(request, deadline):
                        remaining_now = min(
                            remaining,
                            (
                                min(deadline, self.authority.expires_at)
                                - self.authority.now()
                            ).total_seconds(),
                        )
                    return await self.transport.send(payload, request, remaining_now)

                pending = asyncio.create_task(send())
            try:
                observed = await pending
            except asyncio.CancelledError:
                self.ledger.commit_failure(
                    attempt.attempt_id,
                    AttemptReceiptV1(
                        task_key=task_key,
                        attempt_no=attempt.attempt_no,
                        outcome="cancelled",
                        request_sha256=attempt.request_sha256,
                        usage_unknown=True,
                        error_code="cancelled",
                        started_at=began,
                        completed_at=self.authority.now(),
                    ),
                )
                raise
            except AdmissionDenied:
                observed = AttemptResult(error_code="cancelled_before_transport")
            live = True
            try:
                with self.authority.enter(request, deadline):
                    pass
            except AdmissionDenied:
                live = False
            code = observed.error_code if live else "late_result_discarded"
            success = live and code is None
            # A driver whose line grounds the answer may find a content refusal *after* the
            # wire accepted it.  The hook lets such a line spend its remaining attempt on
            # that refusal under exactly the same shared budget as a format correction; a
            # driver that does not need it returns None and keeps the frozen behaviour.
            refusal = self.refusal_of(observed, request, task_key) if success else None
            if refusal is not None:
                code = refusal[0]
                success = False
            receipt = AttemptReceiptV1(
                task_key=task_key,
                attempt_no=attempt.attempt_no,
                outcome="succeeded" if success else "failed",
                request_sha256=attempt.request_sha256,
                requested_model_id=self.transport.config.requested_model,
                resolved_model_id=observed.resolved_model_id,
                http_status=observed.http_status,
                input_tokens=None if observed.usage_unknown else observed.input_tokens,
                output_tokens=None
                if observed.usage_unknown
                else observed.output_tokens,
                usage_unknown=observed.usage_unknown,
                request_bytes=len(canonical_json_bytes(payload)),
                response_bytes=observed.response_bytes,
                received_nonempty_body=observed.received_body,
                underlying_error_code=observed.error_code
                if code != observed.error_code
                else None,
                reasoning_tokens=observed.reasoning_tokens,
                reasoning_bytes=observed.reasoning_bytes,
                format_diagnostic=observed.format_diagnostic,
                format_fields=observed.format_fields,
                format_diagnostics=(
                    (*observed.format_diagnostics, *refusal[1])
                    if refusal is not None
                    else observed.format_diagnostics
                ),
                finish_reason=observed.finish_reason,
                retry_reason=retry_reason,
                retry_not_before=self.authority.now()
                + timedelta(seconds=observed.retry_after)
                if observed.retryable
                else None,
                started_at=began,
                completed_at=self.authority.now(),
                error_code=code,
            )
            can_retry = (
                live
                and not success
                and attempt.attempt_no < max_attempts
                and (observed.correctable or observed.retryable or refusal is not None)
                and observed.retry_after <= 10
                and (deadline - self.authority.now()).total_seconds()
                > observed.retry_after + 1
            )
            if can_retry:
                self.ledger.commit_failure(attempt.attempt_id, receipt)
                correction = (
                    refusal[0]
                    if refusal is not None
                    else code
                    if observed.correctable
                    else None
                )
                correction_fields = (
                    observed.format_fields if observed.correctable else ()
                )
                retry_reason = correction or code
                self.on_correction(
                    task_key,
                    correction,
                    correction_fields,
                    refusal,
                )
                if observed.retry_after:
                    await asyncio.sleep(observed.retry_after)
                continue
            projection = self.make_projection(
                observed,
                success,
                task_key=task_key,
                case_id=request.case_id,
                worker=request.worker,
                claims=tuple(f.projection() for f in observed.findings)
                if success
                else (),
                details=tuple(
                    ClaimDetail(
                        obligation_id=f.finding.obligation_id,
                        risk_level=f.finding.risk_level,
                        missing_information=f.finding.missing_information,
                        statement=f.finding.statement,
                        recommended_action=f.finding.recommended_action,
                    )
                    for f in observed.findings
                )
                if success
                else (),
                first_pass_valid=success and attempt.attempt_no == 1,
                received_nonempty_body=observed.received_body,
                corrected_valid=success and correction is not None,
                pipeline_failure=not success,
                model_request_count=attempt.attempt_no,
                latency_ms=int(
                    max(
                        time.monotonic() - started,
                        (self.authority.now() - first_started).total_seconds(),
                    )
                    * 1000
                ),
            )
            if not live:
                projection = self.discarded_projection(projection)
            if live:
                # Final adoption must still be inside the same current batch gate.
                try:
                    with self.authority.enter(request, deadline):
                        self.ledger.commit_result(
                            attempt.attempt_id, receipt=receipt, projection=projection
                        )
                except AdmissionDenied:
                    receipt = receipt.model_copy(
                        update={
                            "outcome": "failed",
                            "error_code": "late_result_discarded",
                            "underlying_error_code": observed.error_code,
                        }
                    )
                    projection = self.discarded_projection(projection)
                    self.ledger.commit_result(
                        attempt.attempt_id, receipt=receipt, projection=projection
                    )
            else:
                self.ledger.commit_result(
                    attempt.attempt_id, receipt=receipt, projection=projection
                )
            return projection
