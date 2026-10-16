"""Small in-process Trace contract for the V1-01E review boundary.

Trace events deliberately contain execution metadata and bounded summaries only.
Domain evidence and approval decisions are stored in their own contracts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import RLock
from typing import Literal

from pydantic import Field, model_validator

from .contracts import (
    Identifier,
    JsonValue,
    RedactionStatus,
    RunEvent,
    RunEventType,
    StrictContract,
)

TRACE_SCHEMA_VERSION = "trace.v1"


def canonical_checksum(value: object) -> str:
    """Return a stable checksum without retaining the value itself."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class TraceEnvelope(StrictContract):
    """Serializable Trace view, independent from evidence and audit records."""

    trace_schema_version: Literal["trace.v1"] = TRACE_SCHEMA_VERSION
    trace_id: Identifier
    run_id: Identifier
    events: tuple[RunEvent, ...] = Field(min_length=1)
    complete: bool = True

    @model_validator(mode="after")
    def events_are_correlated_and_ordered(self) -> TraceEnvelope:
        if any(
            event.trace_id != self.trace_id or event.run_id != self.run_id
            for event in self.events
        ):
            raise ValueError("trace events must share trace and run IDs")
        event_ids = tuple(event.event_id for event in self.events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("trace event IDs must be unique")
        if event_ids != tuple(sorted(event_ids)):
            raise ValueError("trace events must be stably sorted")
        occurred_at = tuple(event.occurred_at for event in self.events)
        if occurred_at != tuple(sorted(occurred_at)):
            raise ValueError("trace events must be ordered by occurrence time")
        if self.events[0].event_type is not RunEventType.RUN_STARTED:
            raise ValueError("trace must begin with run_started")
        if self.complete and self.events[-1].event_type not in {
            RunEventType.RUN_COMPLETED,
            RunEventType.RUN_FAILED,
            RunEventType.RUN_CANCELLED,
        }:
            raise ValueError("complete trace must end with a terminal run event")
        return self


class TraceRecorder:
    """Append-only local recorder with deterministic event IDs and ordering."""

    def __init__(
        self,
        *,
        run_id: str,
        trace_id: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.run_id = run_id
        self.trace_id = trace_id
        self._clock = clock or (lambda: datetime.now(UTC))
        self._events: list[RunEvent] = []
        self._sequence = 0
        self._last_time: datetime | None = None

    @property
    def events(self) -> tuple[RunEvent, ...]:
        return tuple(self._events)

    def append(
        self,
        event_type: RunEventType,
        *,
        step: str,
        input_summary: Mapping[str, JsonValue] | None = None,
        output_summary: Mapping[str, JsonValue] | None = None,
        source_versions: Mapping[str, str] | None = None,
        duration_ms: int = 0,
        error: str | None = None,
    ) -> RunEvent:
        """Record one bounded execution event; no raw prompt or model reasoning."""

        if duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        self._sequence += 1
        occurred_at = self._clock()
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("trace clock must return a timezone-aware datetime")
        if self._last_time is not None and occurred_at <= self._last_time:
            occurred_at = self._last_time + timedelta(microseconds=1)
        self._last_time = occurred_at

        input_value = dict(input_summary or {})
        output_value = dict(output_summary or {})
        versions = dict(source_versions or {"trace": TRACE_SCHEMA_VERSION})
        # Checksums are contracts, while summaries remain intentionally tiny.
        event = RunEvent(
            event_id=f"event-{self.run_id}-{self._sequence:04d}",
            run_id=self.run_id,
            trace_id=self.trace_id,
            event_type=event_type,
            step=step,
            occurred_at=occurred_at,
            input_summary=input_value,
            output_summary=output_value,
            input_checksum=canonical_checksum(input_value),
            output_checksum=canonical_checksum(output_value),
            source_versions=versions,
            duration_ms=duration_ms,
            token_count=0,
            cost_amount=Decimal("0"),
            cost_currency="CNY",
            redaction_status=RedactionStatus.NOT_REQUIRED,
            error=error,
        )
        self._events.append(event)
        return event

    def snapshot(self, *, complete: bool = True) -> TraceEnvelope:
        return TraceEnvelope(
            trace_id=self.trace_id,
            run_id=self.run_id,
            events=tuple(self._events),
            complete=complete,
        )


class TraceStore:
    """Thread-safe local index of completed and in-flight trace envelopes."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._traces: dict[str, TraceEnvelope] = {}

    def put(self, trace: TraceEnvelope) -> TraceEnvelope:
        with self._lock:
            self._traces[trace.trace_id] = trace
            return trace

    def get(self, trace_id: str) -> TraceEnvelope | None:
        with self._lock:
            return self._traces.get(trace_id)

    def get_for_run(self, run_id: str) -> TraceEnvelope | None:
        with self._lock:
            for trace in self._traces.values():
                if trace.run_id == run_id:
                    return trace
        return None
