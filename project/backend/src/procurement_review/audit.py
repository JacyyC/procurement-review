"""Local approval-audit storage kept separate from execution Trace."""

from __future__ import annotations

from threading import RLock

from .contracts import ApprovalAuditEvent, Identifier


class ApprovalAuditStore:
    """In-memory append-only audit store for explicit human actions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._events: dict[str, list[ApprovalAuditEvent]] = {}

    def append(self, event: ApprovalAuditEvent) -> ApprovalAuditEvent:
        with self._lock:
            events = self._events.setdefault(event.run_id, [])
            if any(item.audit_id == event.audit_id for item in events):
                return next(item for item in events if item.audit_id == event.audit_id)
            events.append(event)
            events.sort(key=lambda item: (item.occurred_at, item.audit_id))
            return event

    def list(self, run_id: Identifier) -> tuple[ApprovalAuditEvent, ...]:
        with self._lock:
            return tuple(self._events.get(run_id, ()))

    def all(self) -> tuple[ApprovalAuditEvent, ...]:
        with self._lock:
            return tuple(
                event
                for run_id in sorted(self._events)
                for event in sorted(
                    self._events[run_id],
                    key=lambda item: (item.occurred_at, item.audit_id),
                )
            )

    def count(self, run_id: Identifier) -> int:
        return len(self.list(run_id))
