"""Test-only fault injection for the V1-03 recovery matrix.

The hooks are inert unless a fault plan is supplied through the environment by the
`security_recovery_check` runner.  They are never reachable from an HTTP request, a
model output or a tool argument, and no product code branches on them when no plan is
configured.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAULT_PLAN_ENV = "V1_03_FAULT_PLAN"
FAULT_DIR_ENV = "V1_03_FAULT_DIR"
FAULT_ADMIN_DSN_ENV = "V1_03_FAULT_ADMIN_DSN"
FAULT_APPLICATION_NAME = "v1_03_security_recovery"
FAULT_APPLICATION_NAME_ENV = "V1_03_FAULT_APPLICATION_NAME"
FAULT_EXIT_CODE = 70
FAULT_PLAN_VERSION = "fault-plan.v1"


class FaultConfigurationError(RuntimeError):
    """Raised when a plan is unreadable or asks for an unsupported action."""


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    action: str
    timeout_seconds: float = 60.0
    release: str | None = None
    once: bool = False


@dataclass(slots=True)
class FaultController:
    """One parsed fault plan; the default instance does nothing at all."""

    enabled: bool = False
    directory: Path | None = None
    barriers: dict[str, BarrierSpec] = field(default_factory=dict)
    transient: dict[str, dict[int, int]] = field(default_factory=dict)
    slow: dict[str, dict[int, float]] = field(default_factory=dict)
    admin_dsn: str | None = None
    reached: list[str] = field(default_factory=list)
    fired: set[str] = field(default_factory=set)
    transient_fired: list[str] = field(default_factory=list)
    slow_fired: list[str] = field(default_factory=list)

    @classmethod
    def from_environment(cls) -> FaultController:
        raw = os.environ.get(FAULT_PLAN_ENV)
        if not raw:
            return cls()
        path = Path(raw)
        if not path.is_file():
            raise FaultConfigurationError("the fault plan file does not exist")
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise FaultConfigurationError("the fault plan is unreadable") from error
        if not isinstance(payload, dict):
            raise FaultConfigurationError("the fault plan must be a JSON object")
        if payload.get("fault_plan_version") != FAULT_PLAN_VERSION:
            raise FaultConfigurationError("unknown fault plan version")
        directory = os.environ.get(FAULT_DIR_ENV)
        barriers = {
            str(name): BarrierSpec(
                action=str(spec.get("action", "wait")),
                timeout_seconds=float(spec.get("timeout_seconds", 60.0)),
                release=spec.get("release"),
                once=bool(spec.get("once", False)),
            )
            for name, spec in (payload.get("barriers") or {}).items()
        }
        for spec in barriers.values():
            if spec.action not in {"wait", "exit", "terminate"}:
                raise FaultConfigurationError(
                    f"unsupported barrier action {spec.action}"
                )
        return cls(
            enabled=True,
            directory=Path(directory) if directory else None,
            barriers=barriers,
            transient={
                str(worker): {
                    int(attempt): int(count) for attempt, count in plan.items()
                }
                for worker, plan in (payload.get("transient") or {}).items()
            },
            slow={
                str(worker): {
                    int(attempt): float(seconds) for attempt, seconds in plan.items()
                }
                for worker, plan in (payload.get("slow") or {}).items()
            },
            admin_dsn=os.environ.get(FAULT_ADMIN_DSN_ENV),
        )

    # ---------------------------------------------------------------- barriers

    def barrier(self, name: str) -> None:
        """Reach one named commit boundary; the plan decides what happens next."""

        if not self.enabled:
            return
        spec = self.barriers.get(name)
        if spec is not None and spec.once and name in self.fired:
            return
        self.reached.append(name)
        if spec is None or self.directory is None:
            return
        self.fired.add(name)
        marker = self.directory / f"{name.replace(':', '_')}.reached"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(time.time()), encoding="utf-8")
        if spec.action == "exit":
            os._exit(FAULT_EXIT_CODE)
        if spec.action == "terminate":
            # Break the connection the caller is about to use and keep breaking it for a
            # short window, so the transaction under test really cannot complete.
            self._terminate_application_backends()
            if spec.timeout_seconds > 0:
                thread = threading.Thread(
                    target=self._keep_terminating,
                    args=(spec.timeout_seconds,),
                    daemon=True,
                )
                thread.start()
            return
        release = self.directory / f"{(spec.release or name).replace(':', '_')}.release"
        deadline = time.monotonic() + spec.timeout_seconds
        while time.monotonic() < deadline:
            if release.exists():
                return
            time.sleep(0.01)
        raise FaultConfigurationError(f"the barrier {name} was never released")

    def _keep_terminating(self, duration_seconds: float) -> None:
        deadline = time.monotonic() + duration_seconds
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                self._terminate_application_backends()
            time.sleep(0.05)

    @staticmethod
    def application_name() -> str:
        """Return the connection tag the terminate action targets.

        A scenario can narrow the target so two independently running test servers do
        not kill each other's sessions.
        """

        return os.environ.get(FAULT_APPLICATION_NAME_ENV) or FAULT_APPLICATION_NAME

    def _terminate_application_backends(self) -> None:
        if not self.admin_dsn:
            raise FaultConfigurationError("the terminate action needs an admin DSN")
        from sqlalchemy import create_engine, text

        engine = create_engine(self.admin_dsn, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as connection:
                connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE application_name = :name AND pid <> pg_backend_pid()"
                    ),
                    {"name": self.application_name()},
                )
        finally:
            engine.dispose()

    # ------------------------------------------------- transient / slow faults

    def transient_count(self, worker: str, attempt_no: int) -> int:
        if not self.enabled:
            return 0
        return self.transient.get(worker, {}).get(attempt_no, 0)

    def note_transient(self, worker: str, attempt_no: int) -> None:
        self.transient_fired.append(f"{worker}:{attempt_no}")

    def slow_seconds(self, worker: str, attempt_no: int) -> float:
        if not self.enabled:
            return 0.0
        return self.slow.get(worker, {}).get(attempt_no, 0.0)

    def note_slow(self, worker: str, attempt_no: int) -> None:
        self.slow_fired.append(f"{worker}:{attempt_no}")

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reached": list(self.reached),
            "transient_fired": list(self.transient_fired),
            "slow_fired": list(self.slow_fired),
        }


__all__ = [
    "BarrierSpec",
    "FAULT_ADMIN_DSN_ENV",
    "FAULT_APPLICATION_NAME_ENV",
    "FAULT_APPLICATION_NAME",
    "FAULT_DIR_ENV",
    "FAULT_EXIT_CODE",
    "FAULT_PLAN_ENV",
    "FAULT_PLAN_VERSION",
    "FaultConfigurationError",
    "FaultController",
]
