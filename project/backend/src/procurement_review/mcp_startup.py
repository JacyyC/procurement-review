"""Protected launch context for the stdio MCP entry point.

The stdio process cannot ask a browser or a model for its scope.  It reads exactly one
protected startup file produced by the trusted composition root, verifies the scope
against the same authorization store the API uses, and refuses to start otherwise.
There is no implicit unrestricted default and no fixture loading.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .procurement_store import ProcurementReader
from .review_store import ReviewStore
from .scoped_runtime import ScopeTelemetry, ScopedProcurementReader
from .security_contracts import RuntimeScope

MCP_STARTUP_VERSION = "mcp-startup.v1"
STARTUP_FILE_ENV = "PROCUREMENT_MCP_STARTUP_FILE"
_REQUIRED_DIRECTORY_MODE = 0o700
_REQUIRED_FILE_MODE = 0o600


class McpStartupError(RuntimeError):
    """Raised when the protected startup context is missing or untrusted."""


@dataclass(slots=True)
class StdioStartupContext:
    scope: RuntimeScope
    reader: ProcurementReader
    telemetry: ScopeTelemetry
    _store: ReviewStore

    def close(self) -> None:
        self._store.close()


def _permission_bits(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def load_startup_context(path: Path) -> StdioStartupContext:
    """Load, validate and authorize one protected stdio startup context."""

    from .authorization import scope_admission, verify_scope_authorization

    if not path.is_absolute():
        raise McpStartupError("the startup context path must be absolute")
    if not path.is_file():
        raise McpStartupError("the startup context file does not exist")
    if _permission_bits(path) != _REQUIRED_FILE_MODE:
        raise McpStartupError("the startup context file must be mode 0600")
    parent_mode = _permission_bits(path.parent)
    if parent_mode != _REQUIRED_DIRECTORY_MODE:
        raise McpStartupError("the startup context directory must be mode 0700")

    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise McpStartupError("the startup context file is unreadable") from error
    if not isinstance(payload, dict):
        raise McpStartupError("the startup context must be a JSON object")
    if payload.get("startup_context_version") != MCP_STARTUP_VERSION:
        raise McpStartupError("unknown MCP startup context version")
    database_url = payload.get("database_url")
    if not isinstance(database_url, str) or not database_url:
        raise McpStartupError("the startup context must carry a database URL")
    raw_scope = payload.get("scope")
    if not isinstance(raw_scope, dict):
        raise McpStartupError("the startup context must carry one runtime scope")
    try:
        scope = RuntimeScope.model_validate(raw_scope, strict=False)
    except ValueError as error:
        raise McpStartupError("the startup context scope is invalid") from error

    store = ReviewStore.from_url(database_url)
    try:
        store.ensure_migrated()
        verify_scope_authorization(store, scope)
        telemetry = ScopeTelemetry()
        reader = ScopedProcurementReader.from_store(
            store=store,
            scope=scope,
            telemetry=telemetry,
            # The startup check above happens before any tool is called; each read has to
            # re-check authority for itself.
            admission=scope_admission(store, scope),
        )
    except Exception:
        store.close()
        raise
    return StdioStartupContext(
        scope=scope,
        reader=reader,
        telemetry=telemetry,
        _store=store,
    )


def load_stdio_startup_context() -> StdioStartupContext:
    """Return the startup context named by the protected environment variable."""

    raw = os.environ.get(STARTUP_FILE_ENV)
    if not raw:
        raise McpStartupError(
            f"{STARTUP_FILE_ENV} is required; the MCP server has no default scope"
        )
    return load_startup_context(Path(raw))


__all__ = [
    "MCP_STARTUP_VERSION",
    "McpStartupError",
    "STARTUP_FILE_ENV",
    "StdioStartupContext",
    "load_startup_context",
    "load_stdio_startup_context",
]
