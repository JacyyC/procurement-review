#!/usr/bin/env python3
"""Local quality gate for the procurement-review repository (V1-06; FIX-01D/FIX-05 hardening).

One command runs every check the commit discipline requires, in a fixed order:

1. GOV — workspace views check, document health check, markdown link check and
   ``git diff --check``;
2. RUFF — ``ruff check`` and ``ruff format --check`` under the explicit committed
   configuration;
3. BACKEND — the full pytest suite.  The database scenarios never skip: the gate
   provisions its own one-shot PostgreSQL container (digest-pinned pgvector image,
   random loopback port, destroyed afterwards) and injects ``TEST_POSTGRES_DSN``.
   When Docker is unavailable the section fails with a clear reason instead of
   degrading to a skip;
4. FRONTEND — eslint, the vitest suite with v8 coverage, and the production build;
5. RATCHET — coverage and file-size baselines only improve: a coverage drop
   beyond the noise floor (``COVERAGE_TOLERANCE_PP``), a guarded file growing,
   a guarded path disappearing without an explicit migration declaration, a
   migration whose declared or freshly measured targets exceed the inherited
   budget, a new >800-line file without a registered exemption, or an
   assembly-root factory passing its recorded line ceiling (V1-06 AC-07) fails
   the gate.

The first failing section stops the run with a non-zero exit and names itself.

Measurement lifecycle (FIX-01D A-10, hardened by FIX-05A).  A run is an identified
event, not a file that happens to exist: it starts by publishing a ``running`` state
with a fresh run ID, a monotonic *generation* and the owner token of the process that
started it, so an older run can never pass for the latest one and can never publish
over a newer state.  A failure writes a separated diagnostic record and publishes
nothing else.  The measured content is fingerprinted before and after the sections,
and a run whose inputs changed underneath it publishes no promotable success.  Only a
complete green five-section run publishes a structured success record carrying the
per-section status, the measurement tool version, both timestamps, the full input file
list with content fingerprints, the coverage/factory measurement and the hash of the
baseline it measured against.

Run, update and bootstrap share one work-tree OS lock (``fcntl.flock`` on
``gate.lock``).  The lock is released by closing the descriptor — never by deleting
the file — and the operating system releases it when a process dies, so a signal or a
hard kill can never leave the work tree permanently locked.  The lock coordinates
admission and publication; it is not a content lock, because ordinary editors do not
obey it.

The lifecycle records are closed-form.  ``run.json``, ``success.json``,
``failure.json``, ``baseline-update.json`` and the publish intent are parsed by
validators that reject duplicate JSON keys, non-standard numbers, unknown or missing
fields, empty run IDs, unusable timestamps, non-finite numbers, wrong versions,
malformed hashes, escaping or duplicated manifest paths and inconsistent
cross-record identities; every internal decoding error becomes a controlled refusal,
never a traceback, and every damaged record is refused *before* anything is written.

``--update-baseline`` is the only writer of the baseline contract.  It requires the
latest run to be a complete green one, re-derives the fingerprints from the current
tree, refuses when the measured content changed since that run, re-runs the governance
section so the newest documentation state is covered, then performs a final review
under the same lock — content, run identity, generation and old baseline bytes — before
writing an intent record and atomically replacing the baseline as a single commit
point.  A failure before the commit point leaves the baseline byte-identical; a
failure after it is reported as *committed* with a recovery entry point, never as
"unchanged".  ``--recover-baseline`` only accepts the expected old or new hash and
stops for manual inspection on anything else.

``--bootstrap-baseline`` is the separate, explicit first-time path.  It is reachable
from the real command line in a repository that never had a contract (the first run
self-checks the candidate contract instead of failing on a missing baseline), and it
refuses whenever Git history, the index or local lifecycle evidence proves that a
contract existed — deleting the file is not a reset.

Guarded paths are protected by a general migration contract: a deletion, rename,
split, merge or retirement of a registered path must be declared under
``guarded_migrations`` with ``source``, ``kind``, ``targets``, ``responsibility``,
``source_ceiling``, ``target_ceilings`` and ``reason``.  One budget rule covers every
kind that keeps a budget: the recorded source ceiling is the authority, so a migration
may never declare its targets above it — a rename names *exactly one* target whose
ceiling is at most the recorded budget, and a split or merge shares one budget between
its targets instead of giving each of them its own copy.  ``retire`` names no target
and releases only its own registration.  The same judgement runs when the declaration
is parsed, when a candidate contract is built and when a written contract is enforced,
and the freshly measured paths are re-checked against the inherited ceiling on every
run.  A target that is itself migrating on is a chain hop: the budget is followed to
the paths that exist now, so successive renames or a split after a rename keep their
provenance without widening the protection.  Targets must live inside the managed
trees and stay registered even below 800 lines, and replaying a declaration can only
tighten, never re-consume, the recorded budget.

Measurements, lifecycle records and a copy of the backend coverage JSON go to the
Git-ignored ``project/.local/quality-gate/``; raw coverage work files stay in fresh
per-run temp directories so the gate never delete-then-writes under the repo path (host
sandbox bulk-delete guards reject that).  The baseline contract lives in
``project/quality-baselines.v1.json`` and is only rewritten deliberately.

``--install-hook`` idempotently installs a pre-push hook that runs this gate
(about ten minutes; ``git push --no-verify`` is the documented escape hatch).
The gate is pure standard library, POSIX-only by design (``fcntl``), and equally
runnable from a future CI job.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import json
import math
import os
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "project" / "backend"
FRONTEND = ROOT / "project" / "frontend"
GATE_DIR = ROOT / "project" / ".local" / "quality-gate"
BASELINE_PATH = ROOT / "project" / "quality-baselines.v1.json"
MEASURED_PATH = GATE_DIR / "measured.json"
RUN_PATH = GATE_DIR / "run.json"
SUCCESS_PATH = GATE_DIR / "success.json"
FAILURE_PATH = GATE_DIR / "failure.json"
UPDATE_PATH = GATE_DIR / "baseline-update.json"
INTENT_PATH = GATE_DIR / "baseline-intent.json"
LOCK_PATH = GATE_DIR / "gate.lock"

GATE_NAME = "quality-gate"
# The measurement tool version a record is bound to: a record written by another
# contract revision never promotes a baseline.
GATE_VERSION = 3
RUN_RECORD_VERSION = 2
SUCCESS_RECORD_VERSION = 2
FAILURE_RECORD_VERSION = 2
UPDATE_RECORD_VERSION = 2
INTENT_RECORD_VERSION = 1

BASELINE_VERSION = 1
#: How one run's failure ended up in the lifecycle records (FIX-06E).
FINALIZED_PUBLISHED = "published"
FINALIZED_SUPERSEDED = "superseded"
FINALIZED_BUSY = "busy"
FINALIZED_UNWRITABLE = "unwritable"

BASELINE_STATE_NEVER_CREATED = "never_created"
BASELINE_STATE_PRESENT = "present"
BASELINE_STATES = (BASELINE_STATE_NEVER_CREATED, BASELINE_STATE_PRESENT)
SIZE_LIMIT = 800
# Coverage of a timing-sensitive suite wobbles between identical-code runs
# (observed run-to-run noise: 0.01-0.03pp on the backend).  The ratchet fails
# only drops larger than this noise floor; file sizes compare as exact integers.
COVERAGE_TOLERANCE_PP = 0.1
PG_IMAGE = (
    "pgvector/pgvector:pg17"
    "@sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f"
)
PG_CONTAINER_PREFIX = "quality-gate-pg-"

# The backend section runs the entire suite under ``coverage run --branch``, so it is
# the one section whose runtime is measured in tens of minutes rather than seconds.
# On a quiet developer host it took 1689.1s (run 20260915T133535Z-1edd2ca5,
# gate_version 3); the same unmodified tree has measured 1.8x-4.9x slower on a host
# shared with other workspaces (2026-09-16/17, FIX-06 work item, EV-FIX06E-02), where
# the old 3600s cap cut a working run off mid-suite and could not be distinguished from
# a real test failure by anything but the timeout text.
#
# This cap is therefore a hang detector, not a statement about how fast a host has to
# be: it sits far above the quiet-host time so that being slow is never itself the
# verdict.  It stays finite on purpose - a section that genuinely hangs still ends the
# run - and a run that reaches it is finalized as a failure of that run only (FIX-06E),
# can never publish a success record, and can never move the baseline.
BACKEND_SECTION_TIMEOUT_SECONDS = 10_800

# Modules that were split out of a guarded file.  A split must keep an explicit
# responsibility mapping and its own ceiling, otherwise the parent's ceiling could be
# evaded by moving code into an unguarded sibling (FIX-01D step 3 / FIX-02 AF-10).
# This registry is the record of *already performed* migrations; the general
# mechanism is ``guarded_migrations`` in the baseline (FIX-05A), which covers every
# other guarded path and is the only way to declare a new deletion, rename or split.
SPLIT_MODULE_REGISTRY: dict[str, str] = {
    "project/backend/src/procurement_review/review_schema.py":
        "版本化 PostgreSQL 迁移声明（FIX-01A 从 review_store 拆出）",
    "project/backend/src/procurement_review/run_records.py":
        "已提交 Run 读模型与整行装配（FIX-01A 从 review_repository 拆出）",
    "project/backend/src/procurement_review/execution_guards.py":
        "live 授权与租约守卫（FIX-01A 从 review_repository 拆出）",
    "project/backend/src/procurement_review/run_errors.py":
        "持久化运行的失败词汇表（FIX-01A）",
    "project/backend/src/procurement_review/run_projection.py":
        "Run 的 API 投影（FIX-01A 从 review_service 拆出）",
    "project/backend/src/procurement_review/record_guard.py":
        "分记录闭式类型与类型感知敏感扫描（FIX-01B）",
    "project/backend/src/procurement_review/run_trace.py":
        "每 Run 的 Trace 身份、序号与 envelope（FIX-01B 从 review_repository 拆出）",
    "project/backend/src/procurement_review/run_audit.py":
        "审计序号分配与事件写入口（FIX-01B 从 review_repository 拆出）",
    "project/backend/src/procurement_review/run_idempotency.py":
        "幂等命名空间查询、重放与冲突判定（FIX-01B 从 review_repository 拆出）",
    "project/backend/src/procurement_review/record_inventory.py":
        "持久化记录清单与列归属声明（FIX-01B 返修）",
    "project/backend/src/procurement_review/record_rows.py":
        "每类记录整行判定与兼容投影（FIX-01B 返修）",
}

# Input fingerprinting (FIX-01D step 1-2).  The measured content and the governance
# content are fingerprinted separately: a measurement may be reused after a
# documentation-only change, but never after a change to code, tests, locks, tools or
# read fixtures.  Nothing is excluded by directory *name* other than generated trees, so
# a fixture or document the run actually reads cannot fall out of the manifest.
MEASUREMENT_TREES = (
    "project/backend",
    "project/frontend",
    "project/scripts",
    "project/design-references",
)
MANIFEST_SKIP_DIRS = frozenset(
    {
        ".git", ".local", ".venv", ".workbuddy-ai", "__pycache__", "node_modules",
        "dist", "build", "coverage", "test-results", "playwright-report",
        ".pytest_cache", ".ruff_cache", ".mypy_cache",
    }
)
MANIFEST_SKIP_SUFFIXES = frozenset({".pyc", ".pyo"})
MANIFEST_RULE = (
    "SHA-256 of the UTF-8 JSON.stringify(entries) of one group's [{path, sha256}] "
    "array sorted by path"
)
MANIFEST_GROUPS = ("measurement", "governance")

# The split assembly root (V1-06 AC-07): the baseline records a per-function line
# ceiling, so a factory may keep shrinking without penalty but re-inflating past the
# recorded ceiling fails the gate by name.
API_FACTORY_FILE = BACKEND / "src" / "procurement_review" / "api.py"
API_FACTORY_FUNCTIONS = ("create_app",)
API_FACTORY_LIMIT = 200

HOOK_MARKER = "quality-gate pre-push hook"
HOOK_CONTENT = f"""#!/bin/sh
# {HOOK_MARKER} (installed by project/scripts/quality_gate.py --install-hook, V1-06)
# Runs the full local quality gate; expect about ten minutes.
# Explicit escape hatch: git push --no-verify
exec python3 "$(git rev-parse --show-toplevel)/project/scripts/quality_gate.py"
"""

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")
# A host shell may inject proxy variables (sandboxes do).  Two separate failures follow:
# a loopback request is sent to the proxy and answered 404, and httpx parses every
# ``no_proxy`` entry as a URL pattern, so a bracketed IPv6 entry such as ``[::1]`` raises
# ``InvalidURL`` before any request is made.  Both are host-environment facts, so the gate
# runs every child process with one sanitized view: no proxy variables at all, and both
# spellings of the bypass list set to the loopback origins the suites talk to.
GATE_BYPASS = "127.0.0.1,localhost"

# Trees the size ratchet watches for both guarded files and new oversized files.
GUARDED_TREES = (
    (ROOT / "project" / "backend" / "src", (".py",)),
    (ROOT / "project" / "frontend" / "src", (".ts", ".tsx", ".css")),
)

# Sections and the lifecycle steps that may stop a run outside them.
LIFECYCLE_SECTIONS = ("inputs", "publish")
FAILING_SECTIONS = ("gov", "ruff", "backend", "frontend", "ratchet") + LIFECYCLE_SECTIONS

MIGRATION_KINDS = ("rename", "split", "merge", "retire")


class GateFailure(RuntimeError):
    """One section failed; carries the section name for attribution."""

    def __init__(self, section: str, message: str) -> None:
        super().__init__(message)
        self.section = section


class UpdateRefused(RuntimeError):
    """A baseline write was refused; the baseline file keeps its exact bytes."""


class GateBusy(RuntimeError):
    """Another quality-gate process holds the work-tree lock."""


class CommittedIncomplete(RuntimeError):
    """The baseline was committed atomically, but an auxiliary record failed afterwards."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def output_tail(completed: subprocess.CompletedProcess, limit: int = 1500) -> str:
    return f"{completed.stdout}\n{completed.stderr}".strip()[-limit:]


def require_ok(section: str, completed: subprocess.CompletedProcess, what: str) -> None:
    if completed.returncode != 0:
        raise GateFailure(
            section,
            f"{what} failed with exit {completed.returncode}:\n"
            f"{output_tail(completed)}",
        )


def clean_environment() -> dict[str, str]:
    """The one environment every gate child process sees (FIX-01D dependency/tool work)."""

    env = {key: value for key, value in os.environ.items() if key not in PROXY_VARS}
    env["NO_PROXY"] = GATE_BYPASS
    env["no_proxy"] = GATE_BYPASS
    return env


def free_loopback_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def git_head() -> str:
    completed = run(["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=30)
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


# ------------------------------------------------------------------ artifact plumbing


def ensure_gate_dir() -> None:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(GATE_DIR, 0o700)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict, *, private: bool = True) -> None:
    """Write JSON through a same-directory temp file and an atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        mode = 0o600
    else:
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            mode = 0o644
    handle_fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


# ------------------------------------------------------------- closed-form records
#
# Every lifecycle record is decoded and validated here before it is used or written.
# The validators are closed-form: unknown fields, missing fields, duplicate JSON keys,
# non-standard numbers, empty identities, unusable timestamps, non-finite numbers,
# malformed hashes, escaping or duplicated manifest paths and inconsistent status
# combinations are all refused.  Nothing below ever lets a KeyError, TypeError or
# JSONDecodeError escape as a traceback.


class _DamagedText(ValueError):
    """Raised while decoding JSON text; converted into ``UpdateRefused``."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise _DamagedText(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def _reject_non_standard_number(name: str) -> Any:
    raise _DamagedText(f"non-standard number {name}")


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_standard_number,
    )


def load_record_text(text: str, what: str) -> dict:
    """Decode record text strictly; damage becomes a controlled refusal."""

    try:
        payload = strict_json_loads(text)
    except (ValueError, TypeError, RecursionError) as exc:
        raise UpdateRefused(
            f"the {what} is unreadable or damaged ({exc}); re-run the gate"
        ) from exc
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} is not a JSON object; re-run the gate")
    return payload


def read_record_file(path: Path, what: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UpdateRefused(
            f"the {what} at {path} is unreadable ({exc}); re-run the gate"
        ) from exc
    return load_record_text(text, f"{what} at {path}")


def freeze(value: Any) -> Any:
    """Recursively make a parsed record read-only."""

    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    return value


def to_plain(value: Any) -> Any:
    """The mutable copy internal builders work with."""

    if isinstance(value, MappingProxyType):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def _closed(payload: dict, what: str, required: frozenset[str],
            optional: frozenset[str] = frozenset()) -> None:
    keys = set(payload)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise UpdateRefused(f"the {what} is missing required fields {missing}")
    if unknown:
        raise UpdateRefused(f"the {what} carries unknown fields {unknown}")


def _text_field(payload: dict, key: str, what: str, *,
                allow_empty: bool = False) -> str:
    if key not in payload:
        raise UpdateRefused(f"the {what} is missing required field {key!r}")
    value = payload[key]
    if not isinstance(value, str):
        raise UpdateRefused(f"the {what} field {key!r} must be a string, not {value!r}")
    if not allow_empty and not value.strip():
        raise UpdateRefused(f"the {what} field {key!r} must not be empty")
    return value


def _optional_text(payload: dict, key: str, what: str, *, allow_none: bool = False) -> str | None:
    if key not in payload:
        raise UpdateRefused(f"the {what} is missing required field {key!r}")
    value = payload[key]
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise UpdateRefused(f"the {what} field {key!r} must be a string, not {value!r}")
    return value


def _int_field(payload: dict, key: str, what: str, *, minimum: int | None = None) -> int:
    if key not in payload:
        raise UpdateRefused(f"the {what} is missing required field {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise UpdateRefused(f"the {what} field {key!r} must be an integer, not {value!r}")
    if minimum is not None and value < minimum:
        raise UpdateRefused(
            f"the {what} field {key!r} must be at least {minimum}, not {value!r}"
        )
    return value


def _number_field(payload: dict, key: str, what: str, *, minimum: float | None = None,
                  maximum: float | None = None) -> float:
    if key not in payload:
        raise UpdateRefused(f"the {what} is missing required field {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpdateRefused(f"the {what} field {key!r} must be a number, not {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise UpdateRefused(
            f"the {what} field {key!r} must be a finite number, not {value!r}"
        )
    if minimum is not None and number < minimum:
        raise UpdateRefused(f"the {what} field {key!r} must be >= {minimum}, not {number}")
    if maximum is not None and number > maximum:
        raise UpdateRefused(f"the {what} field {key!r} must be <= {maximum}, not {number}")
    return number


def _timestamp(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UpdateRefused(f"the {what} must be an ISO-8601 timestamp string, not {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise UpdateRefused(
            f"the {what} is not a valid ISO-8601 timestamp ({value!r})"
        ) from exc
    if parsed.tzinfo is None:
        raise UpdateRefused(f"the {what} must carry a timezone offset ({value!r})")
    return value


def _relative_path(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise UpdateRefused(f"the {what} must be a repository-relative path, not {value!r}")
    if value.startswith("/") or "\\" in value or "\x00" in value:
        raise UpdateRefused(f"the {what} must be a repository-relative path, not {value!r}")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise UpdateRefused(
            f"the {what} is not a normalised repository-relative path ({value!r}): "
            "no '.', '..' or empty segment is allowed"
        )
    return value


def _sha256(value: Any, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise UpdateRefused(f"the {what} must be a sha256 hex digest, not {value!r}")
    if any(character not in "0123456789abcdef" for character in value):
        raise UpdateRefused(f"the {what} must be a lowercase sha256 hex digest")
    return value


def _object(payload: dict, key: str, what: str) -> dict:
    if key not in payload:
        raise UpdateRefused(f"the {what} is missing required field {key!r}")
    value = payload[key]
    if not isinstance(value, dict):
        raise UpdateRefused(f"the {what} field {key!r} must be a JSON object")
    return value


def _list(payload: dict, key: str, what: str) -> list:
    if key not in payload:
        raise UpdateRefused(f"the {what} is missing required field {key!r}")
    value = payload[key]
    if not isinstance(value, list):
        raise UpdateRefused(f"the {what} field {key!r} must be a JSON array")
    return value


def _mapping(payload: dict, key: str, what: str,
             key_kind: str, value_kind: str) -> dict:
    values = _object(payload, key, what)
    for name, item in values.items():
        if key_kind == "path":
            _relative_path(name, f"{what} field {key!r} key")
        elif not isinstance(name, str) or not name.strip():
            raise UpdateRefused(
                f"the {what} field {key!r} keys must be non-empty names"
            )
        if value_kind == "int":
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise UpdateRefused(
                    f"the {what} field {key!r} entry {name!r} must be a positive integer"
                )
        elif value_kind == "text":
            if not isinstance(item, str) or not item.strip():
                raise UpdateRefused(
                    f"the {what} field {key!r} entry {name!r} must be non-empty text"
                )
        else:  # pragma: no cover - guards the call sites
            raise AssertionError(value_kind)
    return values


def parse_manifest_summary(payload: dict | None, what: str) -> dict:
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} must be a JSON object")
    _closed(payload, what, frozenset({"rule", "groups"}))
    rule = _text_field(payload, "rule", what)
    if rule != MANIFEST_RULE:
        raise UpdateRefused(f"the {what} was recorded under an unknown fingerprint rule")
    groups = _object(payload, "groups", what)
    if set(groups) != set(MANIFEST_GROUPS):
        raise UpdateRefused(
            f"the {what} must describe exactly the {sorted(MANIFEST_GROUPS)} groups"
        )
    for name, values in sorted(groups.items()):
        if not isinstance(values, dict):
            raise UpdateRefused(f"the {what} group {name!r} must be a JSON object")
        _closed(values, f"{what} group {name!r}", frozenset({"file_count", "fingerprint"}))
        _int_field(values, "file_count", f"{what} group {name!r}", minimum=0)
        _sha256(values["fingerprint"], f"{what} group {name!r} fingerprint")
    return {key: to_plain(value) for key, value in payload.items()}


def parse_manifest(payload: dict | None, what: str) -> dict:
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} must be a JSON object")
    _closed(payload, what, frozenset({"rule", "groups", "files"}))
    parse_manifest_summary({"rule": payload["rule"], "groups": payload["groups"]}, what)
    files = _list(payload, "files", what)
    counts = {name: 0 for name in MANIFEST_GROUPS}
    paths: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise UpdateRefused(f"the {what} entry {index} must be a JSON object")
        _closed(entry, f"{what} entry {index}", frozenset({"path", "group", "sha256"}))
        relative = _relative_path(entry["path"], f"{what} entry {index} path")
        group = entry["group"]
        if group not in MANIFEST_GROUPS:
            raise UpdateRefused(
                f"the {what} entry {index} names unknown group {group!r}"
            )
        expected = "measurement" if is_measurement_input(relative) else "governance"
        if group != expected:
            raise UpdateRefused(
                f"the {what} entry {index} files {relative!r} under {group!r} but the "
                f"tool classifies it as {expected!r}"
            )
        _sha256(entry["sha256"], f"{what} entry {index} sha256")
        counts[group] += 1
        paths.append(relative)
    if len(set(paths)) != len(paths):
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        raise UpdateRefused(f"the {what} lists the same path more than once: {duplicates}")
    if paths != sorted(paths):
        raise UpdateRefused(f"the {what} entries must be sorted by path")
    for name in MANIFEST_GROUPS:
        declared = payload["groups"][name]["file_count"]
        if int(declared) != counts[name]:
            raise UpdateRefused(
                f"the {what} declares {declared} {name} files but lists {counts[name]}"
            )
    return {key: to_plain(value) for key, value in payload.items()}


def parse_owner(payload: dict | None, what: str) -> dict:
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} must be a JSON object")
    _closed(payload, what, frozenset({"token", "pid", "purpose", "started_at"}))
    token = _text_field(payload, "token", what)
    if len(token) < 8 or any(character not in "0123456789abcdef" for character in token):
        raise UpdateRefused(f"the {what} token must be a hex owner token, not {token!r}")
    _int_field(payload, "pid", what, minimum=1)
    _text_field(payload, "purpose", what)
    _timestamp(payload["started_at"], f"{what} started_at")
    return {key: to_plain(value) for key, value in payload.items()}


def parse_sections(entries: Any, what: str, *, status: str) -> list[dict]:
    if not isinstance(entries, list):
        raise UpdateRefused(f"the {what} must be a JSON array")
    parsed: list[dict] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise UpdateRefused(f"the {what} entry {index} must be a JSON object")
        _closed(entry, f"{what} entry {index}", frozenset({"name", "status", "seconds"}))
        name = entry["name"]
        if name not in FAILING_SECTIONS:
            raise UpdateRefused(f"the {what} entry {index} names unknown section {name!r}")
        if entry["status"] not in ("ok", "failed"):
            raise UpdateRefused(
                f"the {what} entry {index} carries status {entry['status']!r}"
            )
        _number_field(entry, "seconds", f"{what} entry {index}", minimum=0)
        parsed.append({"name": name, "status": entry["status"],
                       "seconds": float(entry["seconds"])})
    names = [entry["name"] for entry in parsed]
    if any(name not in SECTION_NAMES for name in names):
        raise UpdateRefused(
            f"the {what} may only list the fixed five sections {SECTION_NAMES}"
        )
    if names != list(SECTION_NAMES[: len(names)]):
        raise UpdateRefused(
            f"the {what} must list a prefix of {SECTION_NAMES} in order; recorded {names}"
        )
    if len(set(names)) != len(names):
        raise UpdateRefused(f"the {what} lists the same section more than once")
    statuses = [entry["status"] for entry in parsed]
    if status == "running" and parsed:
        raise UpdateRefused("a running run has no finished section yet")
    if status != "running" and names and statuses[:-1] != ["ok"] * (len(names) - 1):
        raise UpdateRefused(f"the {what} must not mark a section failed before the last one")
    return parsed


def parse_measurement_fields(payload: dict, what: str, *, require_all: bool) -> dict:
    required = {"backend", "frontend", "sizes", "api_factory", "measured_at", "git_head",
                "gate_version"}
    _closed(payload, what, required if require_all else frozenset(),
            frozenset() if require_all else required)
    if require_all or "backend" in payload:
        backend = _object(payload, "backend", what)
        _closed(backend, f"{what} backend", frozenset({"line_percent", "branch_percent"}))
        _number_field(backend, "line_percent", f"{what} backend", minimum=0, maximum=100)
        _number_field(backend, "branch_percent", f"{what} backend", minimum=0, maximum=100)
    if require_all or "frontend" in payload:
        frontend = _object(payload, "frontend", what)
        _closed(frontend, f"{what} frontend", frozenset({"line_percent", "branch_percent"}))
        _number_field(frontend, "line_percent", f"{what} frontend", minimum=0, maximum=100)
        _number_field(frontend, "branch_percent", f"{what} frontend", minimum=0, maximum=100)
    if require_all or "sizes" in payload:
        _mapping(payload, "sizes", what, "path", "int")
    if require_all or "api_factory" in payload:
        factory = _object(payload, "api_factory", what)
        for name, lines in factory.items():
            if not isinstance(name, str) or not name:
                raise UpdateRefused(f"the {what} api_factory names must be non-empty")
            if isinstance(lines, bool) or not isinstance(lines, int) or lines <= 0:
                raise UpdateRefused(
                    f"the {what} api_factory entry {name!r} must be a positive integer"
                )
    if require_all or "measured_at" in payload:
        _timestamp(payload["measured_at"], f"{what} measured_at")
    if require_all or "git_head" in payload:
        _text_field(payload, "git_head", what, allow_empty=True)
    if require_all or "gate_version" in payload:
        version = _int_field(payload, "gate_version", what)
        if version != GATE_VERSION:
            raise UpdateRefused(
                "the embedded measurement was not produced by this gate version"
            )
    return {key: to_plain(value) for key, value in payload.items()}


def parse_measurement(payload: dict | None, what: str) -> dict:
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} must be a JSON object")
    return parse_measurement_fields(payload, what, require_all=True)


def parse_baseline_reference(payload: dict | None, what: str) -> dict:
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} must be a JSON object")
    _closed(payload, what, frozenset({"path", "present", "sha256", "baseline_version"}))
    relative = _relative_path(payload["path"], f"{what} path")
    if relative != str(BASELINE_PATH.relative_to(ROOT)):
        raise UpdateRefused(
            f"the {what} names {relative!r}, not the contract at "
            f"{BASELINE_PATH.relative_to(ROOT)!r}"
        )
    present = payload["present"]
    if not isinstance(present, bool):
        raise UpdateRefused(f"the {what} field 'present' must be a boolean")
    if present:
        _sha256(payload["sha256"], f"{what} sha256")
        if isinstance(payload["baseline_version"], bool) or not isinstance(
            payload["baseline_version"], int
        ):
            raise UpdateRefused(f"the {what} baseline_version must be an integer")
    else:
        if payload["sha256"] is not None or payload["baseline_version"] is not None:
            raise UpdateRefused(
                f"the {what} must record null hash and version when no contract existed"
            )
    return {key: to_plain(value) for key, value in payload.items()}


def parse_run_record(payload: dict | None, *, source: str = "run state") -> dict:
    what = source
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} is not a JSON object; re-run the gate")
    _closed(payload, what, frozenset({
        "record_version", "gate_version", "run_id", "generation", "status", "started_at",
        "finished_at", "failed_section", "sections", "inputs", "baseline_state", "owner",
    }))
    version = _int_field(payload, "record_version", what)
    if version != RUN_RECORD_VERSION:
        raise UpdateRefused(
            f"unsupported run record version {version!r}; re-run the gate"
        )
    gate_version = _int_field(payload, "gate_version", what)
    if gate_version != GATE_VERSION:
        raise UpdateRefused(
            f"run {payload.get('run_id')} was measured by gate_version "
            f"{gate_version!r}, not {GATE_VERSION}; re-run the gate"
        )
    _text_field(payload, "run_id", what)
    _int_field(payload, "generation", what, minimum=1)
    status = _text_field(payload, "status", what)
    if status not in ("running", "success", "failed"):
        raise UpdateRefused(f"the {what} carries unknown status {status!r}")
    started_at = _timestamp(payload["started_at"], f"{what} started_at")
    finished_at = payload["finished_at"]
    if finished_at is not None:
        finished_at = _timestamp(finished_at, f"{what} finished_at")
        if finished_at < started_at:
            raise UpdateRefused(
                f"the {what} finished before it started ({started_at} -> {finished_at})"
            )
    failed_section = payload["failed_section"]
    if failed_section is not None:
        if not isinstance(failed_section, str) or failed_section not in FAILING_SECTIONS:
            raise UpdateRefused(
                f"the {what} names unknown failed section {failed_section!r}"
            )
    baseline_state = _text_field(payload, "baseline_state", what)
    if baseline_state not in BASELINE_STATES:
        raise UpdateRefused(
            f"the {what} carries unknown baseline_state {baseline_state!r}"
        )
    parse_owner(payload["owner"], f"{what} owner")
    sections = parse_sections(payload["sections"], f"{what} sections", status=status)
    parse_manifest_summary(payload["inputs"], f"{what} inputs")
    if status == "running":
        if finished_at is not None or failed_section is not None:
            raise UpdateRefused(f"the {what} is running but records an end state")
    else:
        if finished_at is None:
            raise UpdateRefused(f"the {what} is {status} but records no finished_at")
        if status == "success":
            if failed_section is not None:
                raise UpdateRefused(f"the {what} reports success and a failed section")
            if len(sections) != len(SECTION_NAMES) or any(
                entry["status"] != "ok" for entry in sections
            ):
                raise UpdateRefused(
                    "the success record does not describe one complete green run of "
                    f"the five sections {SECTION_NAMES}"
                )
        else:
            if failed_section in SECTION_NAMES:
                index = SECTION_NAMES.index(failed_section)
                if len(sections) != index + 1 or sections[-1]["status"] != "failed":
                    raise UpdateRefused(
                        f"the {what} names {failed_section!r} as failed but its section "
                        "list does not end there"
                    )
            elif failed_section is None:
                raise UpdateRefused(f"the {what} failed without naming a section")
    return freeze({key: to_plain(value) for key, value in payload.items()})


def parse_success_record(payload: dict | None, *, source: str = "success record") -> dict:
    what = source
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} is not a JSON object")
    _closed(payload, what, frozenset({
        "record_version", "gate_version", "status", "run_id", "generation", "started_at",
        "finished_at", "sections", "measurement", "baseline", "baseline_state", "inputs",
        "owner",
    }))
    version = _int_field(payload, "record_version", what)
    if version != SUCCESS_RECORD_VERSION:
        raise UpdateRefused(
            f"unsupported success record version {version!r} (old or damaged format); "
            "re-run the gate"
        )
    gate_version = _int_field(payload, "gate_version", what)
    if gate_version != GATE_VERSION:
        raise UpdateRefused(
            f"the success record was written by gate_version {gate_version!r}, not "
            f"{GATE_VERSION}; re-run the gate"
        )
    status = _text_field(payload, "status", what)
    if status != "success":
        raise UpdateRefused(f"the success record carries status {status!r}")
    _text_field(payload, "run_id", what)
    _int_field(payload, "generation", what, minimum=1)
    started_at = _timestamp(payload["started_at"], f"{what} started_at")
    finished_at = _timestamp(payload["finished_at"], f"{what} finished_at")
    if finished_at < started_at:
        raise UpdateRefused(f"the {what} finished before it started")
    parse_owner(payload["owner"], f"{what} owner")
    sections = parse_sections(payload["sections"], f"{what} sections", status="success")
    recorded = [(entry["name"], entry["status"]) for entry in sections]
    if recorded != [(name, "ok") for name in SECTION_NAMES]:
        raise UpdateRefused(
            "the success record does not describe one complete green run of the five "
            f"sections {SECTION_NAMES}; recorded {recorded!r}"
        )
    measurement = payload["measurement"]
    if not isinstance(measurement, dict) or not {"backend", "frontend"} <= set(measurement):
        raise UpdateRefused(
            "the success record's coverage measurement is missing or malformed"
        )
    parse_measurement(measurement, f"{what} measurement")
    baseline = parse_baseline_reference(payload["baseline"], f"{what} baseline")
    baseline_state = _text_field(payload, "baseline_state", what)
    if baseline_state not in BASELINE_STATES:
        raise UpdateRefused(
            f"the {what} carries unknown baseline_state {baseline_state!r}"
        )
    if bool(baseline["present"]) != (baseline_state == BASELINE_STATE_PRESENT):
        raise UpdateRefused(
            f"the {what} records baseline.present={baseline['present']!r} with "
            f"baseline_state={baseline_state!r}; the two must agree"
        )
    parse_manifest(payload["inputs"], f"{what} inputs")
    return freeze({key: to_plain(value) for key, value in payload.items()})


def parse_failure_record(payload: dict | None, *, source: str = "failure record") -> dict:
    what = source
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} is not a JSON object")
    _closed(payload, what, frozenset({
        "record_version", "gate_version", "status", "run_id", "generation", "started_at",
        "failed_at", "section", "message", "sections", "measurement", "inputs", "owner",
    }))
    version = _int_field(payload, "record_version", what)
    if version != FAILURE_RECORD_VERSION:
        raise UpdateRefused(f"unsupported failure record version {version!r}; re-run the gate")
    gate_version = _int_field(payload, "gate_version", what)
    if gate_version != GATE_VERSION:
        raise UpdateRefused(
            f"the failure record was written by gate_version {gate_version!r}, not "
            f"{GATE_VERSION}; re-run the gate"
        )
    if _text_field(payload, "status", what) != "failed":
        raise UpdateRefused("the failure record must carry status 'failed'")
    _text_field(payload, "run_id", what)
    _int_field(payload, "generation", what, minimum=1)
    started_at = _timestamp(payload["started_at"], f"{what} started_at")
    failed_at = _timestamp(payload["failed_at"], f"{what} failed_at")
    if failed_at < started_at:
        raise UpdateRefused(f"the {what} failed before it started")
    section = _text_field(payload, "section", what)
    if section not in FAILING_SECTIONS:
        raise UpdateRefused(f"the {what} names unknown section {section!r}")
    _text_field(payload, "message", what, allow_empty=True)
    parse_owner(payload["owner"], f"{what} owner")
    parse_sections(payload["sections"], f"{what} sections", status="failed")
    measurement = payload["measurement"]
    if not isinstance(measurement, dict):
        raise UpdateRefused(f"the {what} measurement must be a JSON object")
    parse_measurement_fields(measurement, f"{what} measurement", require_all=False)
    if "backend" in measurement and "frontend" in measurement:
        parse_measurement({"backend": measurement["backend"],
                           "frontend": measurement["frontend"],
                           "sizes": measurement.get("sizes", {}),
                           "api_factory": measurement.get("api_factory", {}),
                           "measured_at": measurement.get("measured_at", now()),
                           "git_head": measurement.get("git_head", "unknown"),
                           "gate_version": measurement.get("gate_version", GATE_VERSION)},
                          f"{what} measurement")
    parse_manifest_summary(payload["inputs"], f"{what} inputs")
    return freeze({key: to_plain(value) for key, value in payload.items()})


def parse_update_record(payload: dict | None, *, source: str = "baseline update record") -> dict:
    what = source
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} is not a JSON object")
    _closed(payload, what, frozenset({
        "record_version", "gate_version", "mode", "run_id", "generation", "at", "baseline",
        "governance_check", "measurement_inputs",
    }))
    version = _int_field(payload, "record_version", what)
    if version != UPDATE_RECORD_VERSION:
        raise UpdateRefused(
            f"unsupported baseline update record version {version!r}; re-run the update"
        )
    gate_version = _int_field(payload, "gate_version", what)
    if gate_version != GATE_VERSION:
        raise UpdateRefused(
            f"the baseline update record was written by gate_version {gate_version!r}, "
            f"not {GATE_VERSION}"
        )
    mode = _text_field(payload, "mode", what)
    if mode not in ("update", "bootstrap", "recovery"):
        raise UpdateRefused(f"the {what} carries unknown mode {mode!r}")
    _text_field(payload, "run_id", what)
    _int_field(payload, "generation", what, minimum=1)
    _timestamp(payload["at"], f"{what} at")
    baseline = _object(payload, "baseline", what)
    _closed(baseline, f"{what} baseline", frozenset({
        "path", "sha256_before", "sha256_after", "baseline_version_before",
        "baseline_version_after",
    }))
    _relative_path(baseline["path"], f"{what} baseline path")
    before = baseline["sha256_before"]
    if before is not None:
        _sha256(before, f"{what} baseline sha256_before")
    _sha256(baseline["sha256_after"], f"{what} baseline sha256_after")
    for key in ("baseline_version_before", "baseline_version_after"):
        value = baseline[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise UpdateRefused(f"the {what} {key} must be an integer or null")
    if (before is None) != (baseline["baseline_version_before"] is None):
        raise UpdateRefused(
            f"the {what} must record the previous hash and version together"
        )
    governance = _object(payload, "governance_check", what)
    _closed(governance, f"{what} governance_check", frozenset({
        "checked_at", "fingerprint", "file_count", "result",
    }))
    _timestamp(governance["checked_at"], f"{what} governance_check checked_at")
    _sha256(governance["fingerprint"], f"{what} governance_check fingerprint")
    _int_field(governance, "file_count", f"{what} governance_check", minimum=0)
    if governance["result"] != "ok":
        raise UpdateRefused(
            f"the {what} records a non-green governance check "
            f"({governance['result']!r})"
        )
    inputs = _object(payload, "measurement_inputs", what)
    _closed(inputs, f"{what} measurement_inputs", frozenset({"file_count", "fingerprint"}))
    _int_field(inputs, "file_count", f"{what} measurement_inputs", minimum=0)
    _sha256(inputs["fingerprint"], f"{what} measurement_inputs fingerprint")
    return freeze({key: to_plain(value) for key, value in payload.items()})


def parse_intent_record(payload: dict | None, *, source: str = "publish intent") -> dict:
    what = source
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} is not a JSON object")
    _closed(payload, what, frozenset({
        "record_version", "gate_version", "mode", "run_id", "generation", "created_at",
        "old", "new", "candidate", "update_record",
    }))
    version = _int_field(payload, "record_version", what)
    if version != INTENT_RECORD_VERSION:
        raise UpdateRefused(f"unsupported publish intent version {version!r}")
    gate_version = _int_field(payload, "gate_version", what)
    if gate_version != GATE_VERSION:
        raise UpdateRefused(
            f"the publish intent was written by gate_version {gate_version!r}, not "
            f"{GATE_VERSION}"
        )
    mode = _text_field(payload, "mode", what)
    if mode not in ("update", "bootstrap"):
        raise UpdateRefused(f"the {what} carries unknown mode {mode!r}")
    _text_field(payload, "run_id", what)
    _int_field(payload, "generation", what, minimum=1)
    _timestamp(payload["created_at"], f"{what} created_at")
    old = _object(payload, "old", what)
    _closed(old, f"{what} old", frozenset({"path", "sha256", "present", "baseline_version"}))
    _relative_path(old["path"], f"{what} old path")
    if not isinstance(old["present"], bool):
        raise UpdateRefused(f"the {what} old.present must be a boolean")
    if old["present"]:
        _sha256(old["sha256"], f"{what} old sha256")
        if isinstance(old["baseline_version"], bool) or not isinstance(
            old["baseline_version"], int
        ):
            raise UpdateRefused(f"the {what} old baseline_version must be an integer")
    elif old["sha256"] is not None or old["baseline_version"] is not None:
        raise UpdateRefused(
            f"the {what} must record null old hash/version for a first baseline"
        )
    if mode == "bootstrap" and old["present"]:
        raise UpdateRefused(f"the {what} is a bootstrap intent but records an old contract")
    if mode == "update" and not old["present"]:
        raise UpdateRefused(f"the {what} is an update intent but records no old contract")
    new = _object(payload, "new", what)
    _closed(new, f"{what} new", frozenset({"sha256", "baseline_version"}))
    _sha256(new["sha256"], f"{what} new sha256")
    if isinstance(new["baseline_version"], bool) or not isinstance(
        new["baseline_version"], int
    ):
        raise UpdateRefused(f"the {what} new baseline_version must be an integer")
    parse_baseline(payload["candidate"], f"{what} candidate")
    parse_update_record(payload["update_record"], source=f"{what} update_record")
    return freeze({key: to_plain(value) for key, value in payload.items()})


def _live_line_of(relative: str) -> int | None:
    """The measured size of a path that is present, else ``None``."""

    path = ROOT / relative
    return line_count(path) if path.is_file() else None


def _target_endpoints(target: str, migrations: dict, seen: frozenset[str]) -> list[str] | None:
    """Where a declared target's budget lives now, following declared hops.

    A target that is still a file holds its own budget.  A target that is gone but is
    itself a declared migration is a chain hop: a rename of a renamed path, a split of a
    renamed path and so on keep their provenance, and the budget is followed to the
    paths that exist now.  ``None`` means the chain cannot be resolved — the path is
    absent and is not migrating on, a hop is malformed or retired, or the chain is
    cyclic — and the caller refuses the declaration instead of guessing.
    """

    if _live_line_of(target) is not None:
        return [target]
    hop = migrations.get(target)
    if target in seen or not isinstance(hop, dict):
        return None
    hop_targets = hop.get("targets")
    if (hop.get("kind") == "retire" or not isinstance(hop_targets, list)
            or not hop_targets):
        return None
    endpoints: list[str] = []
    for step in hop_targets:
        if not isinstance(step, str):
            return None
        resolved = _target_endpoints(step, migrations, seen | {target})
        if resolved is None:
            return None
        endpoints.extend(resolved)
    return endpoints


def resolve_migration_endpoints(source: str, declaration: dict, migrations: dict,
                                ) -> list[str] | None:
    """The live paths one declaration covers, chain hops included.

    ``retire`` covers nothing, and ``None`` reports a chain that does not resolve.
    """

    if declaration.get("kind") == "retire":
        return []
    targets = declaration.get("targets")
    if not isinstance(targets, list) or not targets:
        return None
    endpoints: list[str] = []
    for target in targets:
        if not isinstance(target, str):
            return None
        resolved = _target_endpoints(target, migrations, frozenset({source}))
        if resolved is None:
            return None
        endpoints.extend(resolved)
    return endpoints


def parse_migration_declaration(source: str, declaration: dict, what: str, *,
                                previous_ceiling: int | None,
                                line_of: Callable[[str], int | None],
                                migrations: dict | None = None) -> dict:
    """One explicit migration of a guarded path: deletion, rename, split, merge, retire.

    The invariant is the same for every kind that keeps a budget: the trade can only
    tighten.  A rename names exactly one target, and no kind may declare its targets
    above the recorded source ceiling.
    """

    if not isinstance(declaration, dict):
        raise UpdateRefused(f"the {what} must be a JSON object")
    _closed(declaration, what,
            frozenset({"kind", "targets", "responsibility", "source_ceiling",
                       "target_ceilings", "reason"}),
            frozenset({"declared_at"}))
    kind = _text_field(declaration, "kind", what)
    if kind not in MIGRATION_KINDS:
        raise UpdateRefused(
            f"the {what} declares unknown migration kind {kind!r}; known kinds are "
            f"{list(MIGRATION_KINDS)}"
        )
    _text_field(declaration, "responsibility", what)
    _text_field(declaration, "reason", what)
    targets = _list(declaration, "targets", what)
    parsed_targets: list[str] = []
    for index, target in enumerate(targets):
        parsed_targets.append(_relative_path(target, f"{what} target {index}"))
    if len(set(parsed_targets)) != len(parsed_targets):
        raise UpdateRefused(f"the {what} lists the same target more than once")
    if kind == "retire":
        if parsed_targets:
            raise UpdateRefused(f"the {what} retires {source!r} but also names targets")
    elif not parsed_targets:
        raise UpdateRefused(f"the {what} ({kind}) must name at least one target")
    if kind == "rename" and len(parsed_targets) != 1:
        raise UpdateRefused(
            f"the {what} is a rename but names {len(parsed_targets)} targets; a rename "
            "moves one guarded path onto exactly one target — declare a split or a merge "
            "when one budget has to cover several files"
        )
    ceilings = _mapping(declaration, "target_ceilings", what, "path", "int")
    if set(ceilings) != set(parsed_targets):
        raise UpdateRefused(
            f"the {what} must declare target_ceilings for exactly its targets "
            f"({sorted(set(ceilings) ^ set(parsed_targets))} differ)"
        )
    source_ceiling = _int_field(declaration, "source_ceiling", what, minimum=1)
    if previous_ceiling is not None and source_ceiling != int(previous_ceiling):
        raise UpdateRefused(
            f"the {what} declares source_ceiling {source_ceiling} but the recorded "
            f"ceiling for {source!r} is {int(previous_ceiling)}: a migration inherits "
            "the recorded budget instead of restating it"
        )
    if "declared_at" in declaration:
        _timestamp(declaration["declared_at"], f"{what} declared_at")
    if kind in ("rename", "split", "merge"):
        total = sum(int(value) for value in ceilings.values())
        if total > source_ceiling:
            raise UpdateRefused(
                f"the {what} would expand the budget: its targets declare {total} lines "
                f"in total but the source ceiling is {source_ceiling}; a {kind} inherits "
                "the recorded budget instead of giving every target its own copy"
            )
    for target in parsed_targets:
        if not _inside_guarded_trees(target):
            raise UpdateRefused(
                f"the {what} target {target!r} is not inside a managed tree "
                f"({[str(tree.relative_to(ROOT)) for tree, _ in GUARDED_TREES]}); a "
                "migration may only move guard coverage inside the guarded trees"
            )
    hop_map = migrations if isinstance(migrations, dict) else {}
    for target in parsed_targets:
        if line_of(target) is not None:
            continue
        if _target_endpoints(target, hop_map, frozenset({source})) is None:
            raise UpdateRefused(
                f"the {what} target {target!r} does not exist and is not itself a "
                "declared migration; a migration must point at files that are present "
                "when the contract is written, or at a path that is migrating on in the "
                "same change"
            )
    for target in parsed_targets:
        current = line_of(target)
        if current is not None and current > int(ceilings[target]):
            raise UpdateRefused(
                f"the {what} target {target!r} already has {current} lines, above its "
                f"declared ceiling {ceilings[target]}; declare a ceiling at or above the "
                "measured size of the files the migration registers"
            )
    return {"kind": kind, "targets": parsed_targets,
            "responsibility": declaration["responsibility"],
            "source_ceiling": source_ceiling,
            "target_ceilings": {target: int(value) for target, value in ceilings.items()},
            "reason": declaration["reason"], **(
                {"declared_at": declaration["declared_at"]} if "declared_at" in declaration
                else {}
            )}


def _inside_guarded_trees(relative: str) -> bool:
    path = ROOT / relative
    return any(path.is_relative_to(tree) and path.suffix in suffixes
               for tree, suffixes in GUARDED_TREES)


def parse_guarded_migrations(payload: Any, what: str, *,
                             guarded: dict[str, int] | None,
                             line_of: Callable[[str], int | None] | None) -> dict:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} must be a JSON object")
    parsed: dict = {}
    for source, declaration in sorted(payload.items()):
        _relative_path(source, f"{what} source")
        previous = None
        if guarded is not None and source in guarded:
            previous = int(guarded[source])
        detail = f"{what} declaration for {source!r}"
        parsed[source] = parse_migration_declaration(
            source, declaration, detail, previous_ceiling=previous,
            line_of=line_of or (lambda _relative: None),
            migrations=payload,
        )
    return parsed


BASELINE_REQUIRED = frozenset(
    {"baseline_version", "coverage", "guarded_files", "api_factory_max_lines"}
)
BASELINE_OPTIONAL = frozenset({
    "coverage_tolerance_pp", "size_exemptions", "guarded_registry", "measured_at",
    "updated_at", "git_head", "guarded_migrations",
})


def parse_baseline(payload: dict | None, what: str = "baseline contract") -> dict:
    if not isinstance(payload, dict):
        raise UpdateRefused(f"the {what} is not a JSON object")
    _closed(payload, what, BASELINE_REQUIRED, BASELINE_OPTIONAL)
    version = _int_field(payload, "baseline_version", what)
    if version != BASELINE_VERSION:
        raise UpdateRefused(f"unsupported baseline_version {version!r}")
    coverage = _object(payload, "coverage", what)
    if set(coverage) != {"backend", "frontend"}:
        raise UpdateRefused(
            f"the {what} coverage must describe exactly backend and frontend"
        )
    for side in ("backend", "frontend"):
        values = coverage[side]
        if not isinstance(values, dict):
            raise UpdateRefused(f"the {what} coverage for {side} must be a JSON object")
        _closed(values, f"{what} coverage {side}",
                frozenset({"line_percent", "branch_percent"}))
        _number_field(values, "line_percent", f"{what} coverage {side}",
                      minimum=0, maximum=100)
        _number_field(values, "branch_percent", f"{what} coverage {side}",
                      minimum=0, maximum=100)
    guarded = _mapping(payload, "guarded_files", what, "path", "int")
    factory = _mapping(payload, "api_factory_max_lines", what, "label", "int")
    if "size_exemptions" in payload:
        _mapping(payload, "size_exemptions", what, "path", "text")
    if "coverage_tolerance_pp" in payload:
        _number_field(payload, "coverage_tolerance_pp", what, minimum=0, maximum=100)
    if "guarded_registry" in payload:
        value = payload["guarded_registry"]
        if not isinstance(value, int) or isinstance(value, bool):
            raise UpdateRefused(f"the {what} guarded_registry must be an integer")
    for key in ("measured_at", "updated_at"):
        if key in payload and payload[key] is not None:
            _timestamp(payload[key], f"{what} {key}")
    if "git_head" in payload and payload["git_head"] is not None:
        _text_field(payload, "git_head", what, allow_empty=True)
    if "guarded_migrations" in payload:
        parse_guarded_migrations(
            payload["guarded_migrations"],
            f"{what} guarded_migrations",
            guarded={name: int(value) for name, value in guarded.items()},
            line_of=lambda relative: (
                line_count(ROOT / relative) if (ROOT / relative).is_file() else None
            ),
        )
    return {key: to_plain(value) for key, value in payload.items()}


RECORD_PARSERS: dict[str, Callable[..., dict]] = {
    "run": parse_run_record,
    "success": parse_success_record,
    "failure": parse_failure_record,
    "update": parse_update_record,
    "intent": parse_intent_record,
    "measurement": parse_measurement,
    "baseline": parse_baseline,
}
RECORD_SOURCES = {
    "run": "run state",
    "success": "success record",
    "failure": "failure record",
    "update": "baseline update record",
    "intent": "publish intent",
}


def validate_record(payload: dict | None, kind: str) -> None:
    """Run the closed-form validator for one record kind; damage raises UpdateRefused."""

    parser = RECORD_PARSERS[kind]
    what = RECORD_SOURCES.get(kind, kind)
    if kind in RECORD_SOURCES:
        parser(payload, source=what)
    else:
        parser(payload, what)


def write_record(path: Path, payload: dict, kind: str, *, private: bool = True) -> None:
    """Validate a record completely, then write it atomically (never a half record)."""

    validate_record(to_plain(payload), kind)
    atomic_write_json(path, to_plain(payload), private=private)


# ------------------------------------------------------------------- input manifest


def is_measurement_input(relative: str) -> bool:
    return any(
        relative == tree or relative.startswith(f"{tree}/")
        for tree in MEASUREMENT_TREES
    )


def fingerprint_entries(entries: list[dict[str, str]]) -> str:
    serialized = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_manifest() -> dict:
    """Every input the gate reads, split into measured content and governance content.

    The walk is over the working tree, not the Git index: a new, still-untracked test or
    fixture is an input exactly like a committed one.  The run fingerprints this twice —
    once before the sections and once at publication — and the update fingerprints it
    again immediately before its commit point, so content that changes while the gate is
    reviewing a promotion cannot be promoted as measured.
    """

    groups: dict[str, list[dict[str, str]]] = {"measurement": [], "governance": []}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(name for name in dirnames
                             if name not in MANIFEST_SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix in MANIFEST_SKIP_SUFFIXES or not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            group = "measurement" if is_measurement_input(relative) else "governance"
            groups[group].append({"path": relative, "sha256": file_sha256(path)})
    manifest: dict = {"rule": MANIFEST_RULE, "files": [], "groups": {}}
    for group, entries in groups.items():
        entries.sort(key=lambda entry: entry["path"])
        manifest["groups"][group] = {
            "file_count": len(entries),
            "fingerprint": fingerprint_entries(entries),
        }
        manifest["files"].extend({**entry, "group": group} for entry in entries)
    manifest["files"].sort(key=lambda entry: entry["path"])
    return manifest


def manifest_summary(manifest: dict) -> dict:
    return {
        "rule": manifest["rule"],
        "groups": {name: dict(values)
                   for name, values in sorted(manifest["groups"].items())},
    }


def describe_manifest(manifest: dict) -> str:
    groups = manifest["groups"]
    return (
        f"{groups['measurement']['file_count']} measurement inputs "
        f"({groups['measurement']['fingerprint'][:12]}), "
        f"{groups['governance']['file_count']} governance inputs "
        f"({groups['governance']['fingerprint'][:12]})"
    )


def measurement_fingerprint(manifest: dict) -> str:
    return manifest["groups"]["measurement"]["fingerprint"]


# ---------------------------------------------------------------- the work-tree lock


def _read_lock_state() -> dict | None:
    try:
        text = LOCK_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UpdateRefused(
            f"the work-tree lock at {LOCK_PATH} is unreadable ({exc}); inspect it by hand"
        ) from exc
    if not text.strip():
        return None
    payload = load_record_text(text, f"work-tree lock at {LOCK_PATH}")
    _closed(payload, f"work-tree lock at {LOCK_PATH}",
            frozenset({"gate_version", "acquisitions", "owner"}))
    _int_field(payload, "gate_version", "work-tree lock")
    _int_field(payload, "acquisitions", "work-tree lock", minimum=0)
    parse_owner(payload["owner"], "work-tree lock owner")
    return payload


def describe_lock_owner() -> str:
    try:
        payload = _read_lock_state()
    except UpdateRefused as refusal:
        return str(refusal)
    if payload is None:
        return "no owner recorded"
    owner = payload["owner"]
    return (f"pid {owner['pid']}, purpose {owner['purpose']!r}, token "
            f"{owner['token'][:8]}, acquisitions {payload['acquisitions']}")


@contextlib.contextmanager
def gate_lock(purpose: str) -> Iterator[dict]:
    """One work-tree-level OS lock shared by the run, the updater and bootstrap.

    The lock is released by closing the descriptor — never by deleting the file — and the
    operating system releases it for us when a process dies, so a signal, a crash or a
    hard kill cannot leave the work tree locked.  Descriptors opened here are not
    inheritable, so a child process cannot inherit and forge the parent's ownership; a
    child that opens the same file gets its own description and is refused while the
    parent holds the lock.
    """

    ensure_gate_dir()
    handle_fd = os.open(str(LOCK_PATH), os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(handle_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise GateBusy(
                "another quality-gate process holds the work-tree lock "
                f"({describe_lock_owner()}); run, update and bootstrap share this lock, "
                "so retry after that process finishes"
            ) from exc
        previous = _read_lock_state()
        state = {
            "gate_version": GATE_VERSION,
            "acquisitions": int((previous or {}).get("acquisitions", 0)) + 1,
            "owner": {
                "token": secrets.token_hex(8),
                "pid": os.getpid(),
                "purpose": purpose,
                "started_at": now(),
            },
        }
        os.lseek(handle_fd, 0, os.SEEK_SET)
        os.ftruncate(handle_fd, 0)
        os.write(handle_fd, (
            json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"))
        os.fsync(handle_fd)
        yield state
    finally:
        if acquired:
            try:
                fcntl.flock(handle_fd, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - defence in depth
                pass
        os.close(handle_fd)


# ------------------------------------------------------------------- record builders


def owner_of(lock_state: dict, purpose: str) -> dict:
    owner = dict(lock_state["owner"])
    owner["purpose"] = purpose
    return owner


def run_record_payload(*, run_id: str, generation: int, status: str, started_at: str,
                       finished_at: str | None, failed_section: str | None,
                       sections: list[dict], manifest: dict, baseline_state: str,
                       owner: dict) -> dict:
    return {
        "record_version": RUN_RECORD_VERSION,
        "gate_version": GATE_VERSION,
        "run_id": run_id,
        "generation": generation,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "failed_section": failed_section,
        "sections": sections,
        "inputs": manifest_summary(manifest),
        "baseline_state": baseline_state,
        "owner": owner,
    }


def success_record_payload(*, run_id: str, generation: int, started_at: str,
                           finished_at: str, sections: list[dict], measurement: dict,
                           baseline: dict, baseline_state: str, manifest: dict,
                           owner: dict) -> dict:
    return {
        "record_version": SUCCESS_RECORD_VERSION,
        "gate_version": GATE_VERSION,
        "status": "success",
        "run_id": run_id,
        "generation": generation,
        "started_at": started_at,
        "finished_at": finished_at,
        "sections": sections,
        "measurement": measurement,
        "baseline": baseline,
        "baseline_state": baseline_state,
        "inputs": {
            "rule": manifest["rule"],
            "groups": manifest["groups"],
            "files": manifest["files"],
        },
        "owner": owner,
    }


def failure_record_payload(*, run_id: str, generation: int, failure: GateFailure,
                           started_at: str, sections: list[dict], measured: dict,
                           manifest: dict, owner: dict) -> dict:
    """Diagnostics never share a file with the evidence that can move a baseline."""

    return {
        "record_version": FAILURE_RECORD_VERSION,
        "gate_version": GATE_VERSION,
        "status": "failed",
        "run_id": run_id,
        "generation": generation,
        "started_at": started_at,
        "failed_at": now(),
        "section": failure.section,
        "message": str(failure),
        "sections": to_plain(sections),
        "measurement": to_plain(measured),
        "inputs": manifest_summary(manifest),
        "owner": owner,
    }


def baseline_reference() -> dict:
    if BASELINE_PATH.is_file():
        payload = read_record_file(BASELINE_PATH, "baseline contract")
        return {
            "path": str(BASELINE_PATH.relative_to(ROOT)),
            "present": True,
            "sha256": file_sha256(BASELINE_PATH),
            "baseline_version": payload.get("baseline_version"),
        }
    return {
        "path": str(BASELINE_PATH.relative_to(ROOT)),
        "present": False,
        "sha256": None,
        "baseline_version": None,
    }


def current_baseline_state() -> str:
    if BASELINE_PATH.is_file():
        return BASELINE_STATE_PRESENT
    try:
        if prior_contract_evidence() is not None:
            return BASELINE_STATE_PRESENT
    except UpdateRefused:
        return BASELINE_STATE_PRESENT
    return BASELINE_STATE_NEVER_CREATED


def next_generation() -> int:
    if not RUN_PATH.is_file():
        return 1
    try:
        record = parse_run_record(read_record_file(RUN_PATH, "run state"))
    except UpdateRefused:
        return 1  # a damaged record has no standing; this run replaces it
    return int(record["generation"]) + 1


def prior_contract_evidence() -> str | None:
    """Why this work tree must be treated as a repository that *had* a contract.

    Deleting the file is not a reset: Git history, the index and the local lifecycle
    records all count.  When the tool cannot decide (no usable Git), it refuses instead
    of assuming "never had one".
    """

    relative = str(BASELINE_PATH.relative_to(ROOT))
    completed = run(["git", "log", "--all", "--format=%H", "--", relative], cwd=ROOT,
                    timeout=120)
    if completed.returncode != 0:
        raise UpdateRefused(
            "cannot decide whether this repository ever had a baseline contract: "
            f"`git log -- {relative}` failed ({output_tail(completed, 300)}); restore "
            "the committed contract, or bootstrap from a clean work tree that Git can "
            "answer for"
        )
    revisions = completed.stdout.split()
    if revisions:
        return f"{relative} appears in Git history (commit {revisions[0][:12]})"
    tracked = run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT,
                  timeout=60)
    if tracked.returncode == 0:
        return f"{relative} is still tracked in the Git index"
    for path, what in ((SUCCESS_PATH, "success record"), (RUN_PATH, "run state"),
                       (UPDATE_PATH, "baseline update record")):
        if not path.is_file():
            continue
        try:
            payload = load_record_text(path.read_text(encoding="utf-8"), what)
        except (OSError, UpdateRefused):
            continue
        if payload.get("baseline_state") == BASELINE_STATE_PRESENT:
            return f"the local {what} records baseline_state='present'"
        if isinstance(payload.get("baseline"), dict) and payload["baseline"].get("present"):
            return f"the local {what} records baseline.present=true"
    return None


# --------------------------------------------------------------------- sections


def section_gov() -> None:
    section = "gov"
    # The public distribution uses its own release checks. Private-workspace
    # fixtures retain the original governance path when that checker is absent.
    public_check = ROOT / "scripts" / "check_public_release.py"
    commands = (
        ["python3", "workspace/scripts/workspace_views.py", "--check"],
        ["python3", "workspace/scripts/document_health_check.py", "--full"],
        ["python3", "workspace/scripts/check_markdown_links.py"],
        ["git", "diff", "--check"],
    )
    if public_check.is_file():
        commands = (
            ["python3", str(public_check)],
            ["git", "diff", "--check"],
        )
    for argv in commands:
        completed = run(argv, cwd=ROOT, timeout=600)
        require_ok(section, completed, " ".join(argv))


def section_ruff() -> None:
    section = "ruff"
    for arguments in (("check", "src", "tests"), ("format", "--check", "src", "tests")):
        completed = run(
            ["uv", "run", "--offline", "--frozen", "--python", "3.12", "ruff",
             *arguments],
            cwd=BACKEND,
            timeout=600,
        )
        require_ok(section, completed, f"ruff {' '.join(arguments)}")


def section_backend(measured: dict) -> None:
    section = "backend"
    info = run(["docker", "info", "--format", "{{.ServerVersion}}"], cwd=ROOT, timeout=60)
    if info.returncode != 0:
        raise GateFailure(
            section,
            "Docker is required for the backend section: the gate provisions its own "
            "one-shot PostgreSQL and the database scenarios must never be skipped. "
            f"docker info failed:\n{output_tail(info)}",
        )
    port = free_loopback_port()
    name = f"{PG_CONTAINER_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
    started = run(
        [
            "docker", "run", "-d", "--rm", "--name", name,
            "-p", f"127.0.0.1:{port}:5432",
            "-e", "POSTGRES_PASSWORD=postgres",
            # The V1-03 scenarios create passwordless read-only roles on the fly and
            # connect as them; trust is required and safe on this loopback-only,
            # auto-removed container.
            "-e", "POSTGRES_HOST_AUTH_METHOD=trust",
            PG_IMAGE,
        ],
        cwd=ROOT,
        timeout=300,
    )
    require_ok(section, started, "docker run (one-shot PostgreSQL)")
    basetemp = Path(tempfile.mkdtemp(prefix="quality-gate-pytest-"))
    try:
        deadline = time.time() + 90
        while True:
            ready = run(["docker", "exec", name, "pg_isready", "-U", "postgres"],
                        cwd=ROOT, timeout=30)
            if ready.returncode == 0:
                break
            if time.time() > deadline:
                logs = run(["docker", "logs", "--tail", "30", name], cwd=ROOT,
                           timeout=30)
                raise GateFailure(
                    section,
                    "the one-shot PostgreSQL never became ready:\n"
                    f"{output_tail(logs)}",
                )
            time.sleep(1)

        env = clean_environment()
        env["TEST_POSTGRES_DSN"] = (
            f"postgresql+psycopg://postgres:postgres@127.0.0.1:{port}/postgres"
        )
        ensure_gate_dir()
        # Coverage files live in a fresh per-run temp directory: coverage erases a
        # pre-existing data file before writing, and any delete-then-write under
        # the repo path trips host sandbox bulk-delete guards.  Only the parsed
        # JSON summary is copied back (a pure write) for human inspection.
        workdir = Path(tempfile.mkdtemp(prefix="quality-gate-coverage-"))
        coverage_data = workdir / "backend.coverage"
        coverage_json = workdir / "backend-coverage.json"
        # Measure with `coverage run` rather than `pytest --cov`: pytest-cov's
        # subprocess instrumentation (.pth hook) makes every real REC subprocess
        # write its own parallel data file, and combining them bulk-deletes
        # hundreds of files, which host sandbox bulk-delete guards reject.
        # coverage's direct runner yields identical line/branch totals with a
        # single data file. pytest-cov remains the dev-surface tool (V1-06 AC-06);
        # the gate deliberately drives its pinned coverage engine directly.
        completed = run(
            [
                "uv", "run", "--offline", "--frozen", "--python", "3.12",
                "python", "-m", "coverage", "run",
                f"--data-file={coverage_data}",
                "--source=src/procurement_review",
                "--branch",
                "-m", "pytest", "-q",
                "--basetemp", str(basetemp),
            ],
            cwd=BACKEND,
            env=env,
            timeout=BACKEND_SECTION_TIMEOUT_SECONDS,
        )
        require_ok(section, completed, "full pytest (with one-shot PostgreSQL)")
        reported = run(
            [
                "uv", "run", "--offline", "--frozen", "--python", "3.12",
                "python", "-m", "coverage", "json",
                f"--data-file={coverage_data}",
                "-o", str(coverage_json),
            ],
            cwd=BACKEND,
            env=env,
            timeout=120,
        )
        require_ok(section, reported, "coverage json report")
        shutil.copyfile(coverage_json, GATE_DIR / "backend-coverage.json")

        report = json.loads(coverage_json.read_text(encoding="utf-8"))
        totals = report["totals"]
        lines = totals["covered_lines"] + totals["missing_lines"]
        branches = totals["num_branches"]
        measured["backend"] = {
            "line_percent": round(100.0 * totals["covered_lines"] / lines, 2)
            if lines else 100.0,
            "branch_percent": round(100.0 * totals["covered_branches"] / branches, 2)
            if branches else 100.0,
        }
    finally:
        run(["docker", "rm", "-f", name], cwd=ROOT, timeout=120)
        # basetemp is left to the OS temp janitor on purpose: bulk-deleting
        # thousands of pytest files trips host sandbox bulk-delete guards.


def section_frontend(measured: dict) -> None:
    section = "frontend"
    # Reports go to a fresh per-run temp directory: vitest's coverage provider
    # delete-then-writes its raw temp files, and deletions under the repo path
    # trip host sandbox bulk-delete guards.  Only the parsed summary values are
    # carried forward; the raw reports are left to the OS temp janitor.
    reports_dir = Path(tempfile.mkdtemp(prefix="quality-gate-frontend-"))
    for argv, timeout in (
        (["npm", "run", "lint"], 600),
        (["npm", "run", "test:coverage", "--",
          f"--coverage.reportsDirectory={reports_dir}"], 1200),
        (["npm", "run", "build"], 1200),
    ):
        completed = run(argv, cwd=FRONTEND, env=clean_environment(), timeout=timeout)
        require_ok(section, completed, " ".join(argv))
    summary_path = reports_dir / "coverage-summary.json"
    if not summary_path.is_file():
        raise GateFailure(
            section,
            f"frontend coverage summary missing at {summary_path}",
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["total"]
    measured["frontend"] = {
        "line_percent": round(float(summary["lines"]["pct"]), 2),
        "branch_percent": round(float(summary["branches"]["pct"]), 2),
    }


def measure_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for tree, suffixes in GUARDED_TREES:
        for path in sorted(tree.rglob("*")):
            if path.is_file() and path.suffix in suffixes:
                lines = line_count(path)
                if lines > SIZE_LIMIT:
                    sizes[str(path.relative_to(ROOT))] = lines
    return sizes


def measure_api_factory() -> dict[str, int]:
    """Line spans of the assembly-root factories the ratchet guards (AC-07)."""

    tree = ast.parse(API_FACTORY_FILE.read_text(encoding="utf-8"))
    return {
        node.name: int(node.end_lineno or node.lineno) - node.lineno + 1
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in API_FACTORY_FUNCTIONS
    }


def _migration_problems(source: str, declaration: Any, guarded: dict[str, int],
                        migrations: dict) -> list[str]:
    detail = f"guarded migration for {source!r}"
    problems: list[str] = []
    if isinstance(declaration, dict):
        # The inherited budget is checked against the freshly measured endpoints even
        # when the declaration itself is malformed, so no migration can look green while
        # the paths it covers have grown past the budget they inherited.
        kind = declaration.get("kind")
        ceiling = declaration.get("source_ceiling")
        if (kind in ("rename", "split", "merge")
                and isinstance(ceiling, int) and not isinstance(ceiling, bool)):
            endpoints = resolve_migration_endpoints(source, declaration, migrations)
            if endpoints is not None:
                total = sum(
                    line_count(ROOT / endpoint) for endpoint in endpoints
                    if (ROOT / endpoint).is_file() and not (ROOT / endpoint).is_dir()
                )
                if total > ceiling:
                    problems.append(
                        f"{detail} exceeds its inherited budget: targets total {total} > "
                        f"source ceiling {ceiling} lines — a {kind} inherits one budget "
                        "instead of expanding it"
                    )
    try:
        parse_migration_declaration(
            source, declaration if isinstance(declaration, dict) else {},
            detail,
            previous_ceiling=(int(guarded[source]) if source in guarded else None),
            line_of=_live_line_of,
            migrations=migrations,
        )
    except UpdateRefused as refusal:
        problems.append(f"{detail}: {refusal}")
    if (ROOT / source).exists() and not (ROOT / source).is_file():
        problems.append(f"{detail}: {source!r} exists but is not a file")
    return problems


def ratchet_problems(measured: dict, baseline: dict) -> list[str]:
    """Every way a measured tree may not be promoted to ``baseline``.

    Shared by the gate's ratchet section, the updater's candidate self-check and the
    bootstrap path so a candidate can never be written that this same code would reject.
    """

    problems: list[str] = []
    tolerance = float(baseline.get("coverage_tolerance_pp", COVERAGE_TOLERANCE_PP))
    coverage = baseline.get("coverage", {}) or {}
    for side in ("backend", "frontend"):
        values = measured.get(side)
        floors = coverage.get(side, {}) or {}
        if not isinstance(values, dict) or not isinstance(floors, dict):
            problems.append(
                f"the {side} coverage measurement or floor is missing; the ratchet "
                "cannot compare it"
            )
            continue
        for key in ("line_percent", "branch_percent"):
            if key not in values or key not in floors:
                problems.append(
                    f"the {side} {key} measurement or floor is missing; the ratchet "
                    "cannot compare it"
                )
                continue
            base = float(floors[key])
            current = float(values[key])
            if current < base - tolerance:
                problems.append(
                    f"{side} {key} regressed: {current:.2f} < baseline {base:.2f} "
                    f"by more than the {tolerance:.2f}pp noise floor"
                )

    guarded: dict[str, int] = dict(baseline.get("guarded_files", {}) or {})
    exemptions: dict[str, str] = baseline.get("size_exemptions", {}) or {}
    migrations: dict = baseline.get("guarded_migrations", {}) or {}
    for relative, base_lines in sorted(guarded.items()):
        path = ROOT / relative
        if path.is_file():
            current_lines = line_count(path)
            if current_lines > base_lines:
                problems.append(
                    f"guarded file grew: {relative} {current_lines} > baseline "
                    f"{base_lines}"
                )
            continue
        if relative in migrations:
            continue  # the declaration below accounts for this path's budget
        problems.append(
            f"guarded file disappeared without a migration declaration: {relative} "
            f"({base_lines} lines recorded and no file present); a deletion, rename or "
            "split must be declared under guarded_migrations with source, kind, targets, "
            "responsibility, source_ceiling, target_ceilings and reason (FIX-05A)"
        )
    for source, declaration in sorted(migrations.items()):
        problems.extend(_migration_problems(source, declaration, guarded, migrations))
    declared_targets: set[str] = set()
    for declaration in migrations.values():
        targets = declaration.get("targets") if isinstance(declaration, dict) else None
        if isinstance(targets, list):
            declared_targets.update(
                target for target in targets if isinstance(target, str)
            )
    if baseline.get("guarded_registry"):
        for relative in sorted(SPLIT_MODULE_REGISTRY):
            if relative not in guarded:
                problems.append(
                    f"split module is declared in the tool registry but not registered "
                    f"in guarded_files: {relative} — a rename or split must keep the "
                    "registry and the baseline in step (FIX-01D step 3 / FIX-02 AF-10)"
                )
    for relative, lines in sorted(measured.get("sizes", {}).items()):
        if relative in guarded or relative in exemptions or relative in declared_targets:
            continue
        problems.append(
            f"new file over {SIZE_LIMIT} lines without a registered exemption: "
            f"{relative} ({lines} lines); split it or register a reason under "
            "size_exemptions via --update-baseline"
        )
    ceilings: dict[str, int] = baseline.get("api_factory_max_lines", {})
    if not ceilings:
        problems.append(
            "baseline records no api_factory_max_lines ceiling; the split assembly "
            f"root is unguarded — refresh it with --update-baseline (limit is "
            f"{API_FACTORY_LIMIT} lines)"
        )
    for name, ceiling in sorted(ceilings.items()):
        current = measured.get("api_factory", {}).get(name)
        if current is None:
            problems.append(
                f"baseline-guarded API factory {name} is no longer measurable in "
                f"{API_FACTORY_FILE.relative_to(ROOT)} (V1-06 AC-07)"
            )
        elif current > ceiling:
            problems.append(
                f"API factory grew past its ceiling: {name} {current} > {ceiling} "
                "lines — move endpoints back into project/backend/src/"
                "procurement_review/api_routers/ (V1-06 AC-07)"
            )
    return problems


def section_ratchet(measured: dict) -> None:
    section = "ratchet"
    measured["sizes"] = measure_sizes()
    try:
        measured["api_factory"] = measure_api_factory()
    except (OSError, SyntaxError) as exc:
        raise GateFailure(
            section, f"cannot measure the API factory in {API_FACTORY_FILE}: {exc}"
        ) from exc

    if not BASELINE_PATH.is_file():
        try:
            evidence = prior_contract_evidence()
        except UpdateRefused as refusal:
            raise GateFailure(section, str(refusal)) from refusal
        if evidence is not None:
            raise GateFailure(
                section,
                f"baseline contract missing at {BASELINE_PATH}; this work tree shows a "
                f"prior contract ({evidence}) — restore the committed contract with "
                f"`git checkout -- {BASELINE_PATH.relative_to(ROOT)}` instead of "
                "measuring without one (a deleted contract is never a reset)",
            )
        # A genuine first-time repository: the ratchet judges the candidate contract
        # this run would create, so the first baseline stays reachable from the real
        # command line without a hand-written success record (FIX-05A / D-AUD-01).
        try:
            candidate = candidate_baseline(measured, {})
        except UpdateRefused as refusal:
            raise GateFailure(
                section, f"the first baseline cannot be built from this run: {refusal}"
            ) from refusal
        first_time = ratchet_problems(measured, candidate)
        if first_time:
            raise GateFailure(
                section,
                "this repository has no baseline contract and no record of one, so this "
                "run cannot establish one:\n" + "\n".join(first_time),
            )
        print(
            "[gate] ratchet: no contract exists and none is recorded in Git; this run "
            "measured the tree that a first baseline may be created from with "
            "--bootstrap-baseline (FIX-05A)",
            flush=True,
        )
        return

    try:
        baseline = parse_baseline(read_record_file(BASELINE_PATH, "baseline contract"))
    except UpdateRefused as refusal:
        raise GateFailure(
            section, f"the baseline contract at {BASELINE_PATH} is damaged: {refusal}"
        ) from refusal
    if not baseline.get("guarded_registry"):
        print(
            "[gate] ratchet: this baseline predates the split-module registry; the next "
            f"--update-baseline registers {len(SPLIT_MODULE_REGISTRY)} modules and turns "
            "the check on (FIX-01D step 3 / AF-10)",
            flush=True,
        )
    problems = ratchet_problems(measured, baseline)
    if problems:
        raise GateFailure(section, "\n".join(problems))


SECTIONS = (
    ("gov", section_gov),
    ("ruff", section_ruff),
    ("backend", section_backend),
    ("frontend", section_frontend),
    ("ratchet", section_ratchet),
)
SECTION_NAMES = tuple(name for name, _ in SECTIONS)


# ------------------------------------------------------------------------- the run


def superseded_run_diagnostic(run_id: str) -> Path:
    """The per-run diagnostic a run keeps when it may not rewrite the shared state.

    A failure is published into ``failure.json``/``run.json`` only by the run that still
    owns the latest state.  Any other run — superseded, or unable to take the work-tree
    lock — keeps its own record here, so its failure stays attributable without touching
    what the newest run published.
    """

    return GATE_DIR / "diagnostics" / f"{run_id}.failure.json"


def _latest_run_identity() -> dict | None:
    """The identity of the run that currently owns the shared state, if it is readable.

    A missing or damaged run record yields ``None``: the finalization then treats the
    shared state as somebody else's and writes only its own diagnostic.
    """

    if not RUN_PATH.is_file():
        return None
    try:
        return parse_run_record(read_record_file(RUN_PATH, "run state"))
    except UpdateRefused:
        return None


def _keep_superseded_diagnostic(run_id: str, payload: dict) -> tuple[Path | None, list[str]]:
    path = superseded_run_diagnostic(run_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        write_record(path, to_plain(payload), "failure")
    except (OSError, UpdateRefused) as error:
        return None, [f"the diagnostic for run {run_id} could not be written: {error}"]
    return path, []


def finalize_failure(*, run_id: str, generation: int, started_at: str,
                     failure: GateFailure, sections: list[dict], measured: dict,
                     manifest: dict, baseline_state: str, owner: dict,
                     ) -> tuple[str, list[str]]:
    """Publish one run's failure, or keep it in that run's own diagnostic.

    Finalizing a failure is a shared-state write exactly like publishing a success: it
    takes the same work-tree lock and it re-checks the identity that owns the latest
    state — the run id and its generation, never a wall clock, a PID or the mere
    presence of a success record.  A run that may not write leaves the newer state
    untouched and keeps its own run-identity diagnostic; a lock that somebody else holds
    is reported as busy rather than bypassed, and no path here ever reports success.
    """

    failure_payload = failure_record_payload(
        run_id=run_id, generation=generation, failure=failure, started_at=started_at,
        sections=sections, measured=dict(measured), manifest=manifest, owner=owner,
    )
    run_payload = run_record_payload(
        run_id=run_id, generation=generation, status="failed", started_at=started_at,
        finished_at=now(), failed_section=failure.section, sections=sections,
        manifest=manifest, baseline_state=baseline_state, owner=owner,
    )
    try:
        with gate_lock("failure"):
            latest = _latest_run_identity()
            if (latest is None or latest["run_id"] != run_id
                    or latest["generation"] != generation):
                _path, problems = _keep_superseded_diagnostic(run_id, failure_payload)
                return FINALIZED_SUPERSEDED, problems
            problems = []
            for path, payload, kind in ((FAILURE_PATH, failure_payload, "failure"),
                                        (RUN_PATH, run_payload, "run")):
                try:
                    write_record(path, payload, kind)
                except (OSError, UpdateRefused) as error:
                    problems.append(
                        f"the {kind} record at {path} could not be written: {error}"
                    )
            return (FINALIZED_PUBLISHED if not problems else FINALIZED_UNWRITABLE), problems
    except GateBusy as busy:
        _path, problems = _keep_superseded_diagnostic(run_id, failure_payload)
        return FINALIZED_BUSY, [str(busy), *problems]


def run_gate() -> int:
    measured: dict = {}
    section_results: list[dict] = []
    started_all = time.time()
    try:
        with gate_lock("run") as lock:
            manifest = build_manifest()
            run_id = (
                f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"
            )
            started_at = now()
            generation = next_generation()
            baseline_state = current_baseline_state()
            owner = owner_of(lock, "run")
            # A run is an event with an identity: from this moment the previous success
            # record belongs to another run and can no longer be mistaken for the latest
            # result, and this run holds the generation that lets it publish.
            write_record(RUN_PATH, run_record_payload(
                run_id=run_id, generation=generation, status="running",
                started_at=started_at, finished_at=None, failed_section=None,
                sections=[], manifest=manifest, baseline_state=baseline_state,
                owner=owner,
            ), "run")
    except (GateBusy, UpdateRefused) as refused:
        print(f"[gate] cannot start a run: {refused}", file=sys.stderr)
        print("[gate] nothing was measured and no state was written", file=sys.stderr)
        return 1
    print(
        f"[gate] run {run_id}: gate_version {GATE_VERSION}, generation {generation}, "
        f"baseline {baseline_state}, {describe_manifest(manifest)}",
        flush=True,
    )

    def stop(failure: GateFailure, results: list[dict]) -> int:
        outcome, problems = finalize_failure(
            run_id=run_id, generation=generation, started_at=started_at, failure=failure,
            sections=results, measured=dict(measured), manifest=manifest,
            baseline_state=baseline_state, owner=owner,
        )
        if outcome == FINALIZED_PUBLISHED:
            print(
                f"[gate] diagnostics at {FAILURE_PATH}; no success record was published, "
                "so no measurement of this run can move the baseline",
                file=sys.stderr,
            )
        elif outcome == FINALIZED_SUPERSEDED:
            print(
                f"[gate] run {run_id} failed after a newer run took the shared state; it "
                "did not touch that state and kept its own diagnostic at "
                f"{superseded_run_diagnostic(run_id)}",
                file=sys.stderr,
            )
        elif outcome == FINALIZED_BUSY:
            print(
                f"[gate] run {run_id} failed while another quality-gate process held the "
                "work-tree lock; the shared state keeps its place and this run's own "
                f"diagnostic is at {superseded_run_diagnostic(run_id)}",
                file=sys.stderr,
            )
        else:
            print(
                f"[gate] run {run_id} failed but its failure records could not all be "
                "written; the shared state may be incomplete and no success record was "
                "published",
                file=sys.stderr,
            )
        for problem in problems:
            print(f"[gate] {problem}", file=sys.stderr)
        print(str(failure), file=sys.stderr)
        print(f"[gate] gate FAILED in section {failure.section}", file=sys.stderr)
        return 1

    for name, handler in SECTIONS:
        started = time.time()
        print(f"[gate] {name}: running...", flush=True)
        try:
            if name in {"backend", "frontend", "ratchet"}:
                handler(measured)  # type: ignore[arg-type]
            else:
                handler()  # type: ignore[call-arg]
        except GateFailure as failure:
            section_results.append(
                {"name": name, "status": "failed",
                 "seconds": round(time.time() - started, 1)}
            )
            print(f"[gate] {name}: FAILED ({time.time() - started:.1f}s)", flush=True)
            return stop(failure, section_results)
        except Exception as error:
            # An escaping exception is a failure of *this* run, not a run that never
            # ended: without this the shared state would stay ``running`` with no
            # diagnostic at all.  Observed when a section's own subprocess hit its
            # timeout (FIX-06E).  ``KeyboardInterrupt``/``SystemExit`` stay outside.
            section_results.append(
                {"name": name, "status": "failed",
                 "seconds": round(time.time() - started, 1)}
            )
            print(f"[gate] {name}: FAILED ({time.time() - started:.1f}s)", flush=True)
            return stop(
                GateFailure(
                    name, f"the {name} section raised {type(error).__name__}: {error}"
                ),
                section_results,
            )
        section_results.append(
            {"name": name, "status": "ok", "seconds": round(time.time() - started, 1)}
        )
        print(f"[gate] {name}: ok ({time.time() - started:.1f}s)", flush=True)

    finished_at = now()
    measurement = dict(measured)
    measurement["gate_version"] = GATE_VERSION
    measurement["measured_at"] = finished_at
    measurement["git_head"] = git_head()
    changed_manifest: dict | None = None
    try:
        with gate_lock("publish") as lock:
            latest = parse_run_record(read_record_file(RUN_PATH, "run state"))
            if latest["run_id"] != run_id or latest["generation"] != generation:
                print(
                    f"[gate] run {run_id} was superseded by run {latest['run_id']} "
                    f"({latest['status']}); nothing was published for it and the newer "
                    "state keeps its place",
                    file=sys.stderr,
                )
                return 1
            final_manifest = build_manifest()
            if measurement_fingerprint(final_manifest) != measurement_fingerprint(manifest):
                # The finalization below takes the work-tree lock itself, so it has to
                # run after this ``with`` block: nesting it here would ask a
                # non-reentrant ``flock`` for a lock this process already holds.
                changed_manifest = final_manifest
            else:
                write_record(MEASURED_PATH, to_plain(measurement), "measurement")
                write_record(SUCCESS_PATH, success_record_payload(
                    run_id=run_id, generation=generation, started_at=started_at,
                    finished_at=finished_at, sections=section_results,
                    measurement=measurement, baseline=baseline_reference(),
                    baseline_state=baseline_state, manifest=final_manifest, owner=owner,
                ), "success")
                write_record(RUN_PATH, run_record_payload(
                    run_id=run_id, generation=generation, status="success",
                    started_at=started_at, finished_at=finished_at, failed_section=None,
                    sections=section_results, manifest=final_manifest,
                    baseline_state=baseline_state, owner=owner,
                ), "run")
    except (GateBusy, UpdateRefused) as refused:
        print(
            f"[gate] run {run_id} finished its sections but could not publish: {refused}",
            file=sys.stderr,
        )
        print(
            "[gate] no complete success record exists for this run, so it cannot move "
            "the baseline",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(
            f"[gate] run {run_id} finished its sections but the publish step failed: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return stop(
            GateFailure(
                "publish", f"the publish step raised {type(error).__name__}: {error}"
            ),
            section_results,
        )
    if changed_manifest is not None:
        print(
            "[gate] the measured content changed while this run was measuring "
            f"(recorded {measurement_fingerprint(manifest)[:12]}, now "
            f"{measurement_fingerprint(changed_manifest)[:12]}); no success record was "
            "published, so this run can never move the baseline",
            file=sys.stderr,
        )
        return stop(
            GateFailure(
                "inputs",
                "the measured content changed during the run; the result is not a "
                "measurement of any single tree",
            ),
            section_results,
        )
    print(
        f"[gate] all sections green ({time.time() - started_all:.1f}s); success record "
        f"{SUCCESS_PATH} for run {run_id}",
        flush=True,
    )
    return 0


# ------------------------------------------------------------- baseline write paths


def load_success_evidence() -> tuple[dict, dict, dict]:
    """The current run state, its success record and today's input manifest.

    Everything that must hold before a measurement may be promoted is checked here, so
    the two write paths (update and bootstrap) cannot disagree about what "a complete
    green run of the current content" means.
    """

    if not RUN_PATH.is_file():
        raise UpdateRefused(
            f"no run state at {RUN_PATH}: run "
            "`python3 project/scripts/quality_gate.py` first"
        )
    run_state = parse_run_record(read_record_file(RUN_PATH, "run state"))
    status = run_state["status"]
    if status != "success":
        raise UpdateRefused(
            f"the latest run ({run_state['run_id']}) is {status!r}, not 'success'"
            + (f" (section {run_state['failed_section']!r} failed)"
               if run_state["failed_section"] else "")
            + "; a failed, interrupted or still-running measurement never changes the "
            "baseline"
        )
    if not SUCCESS_PATH.is_file():
        raise UpdateRefused(
            f"run {run_state['run_id']} reports success but no success record exists "
            f"at {SUCCESS_PATH}"
        )
    success = parse_success_record(read_record_file(SUCCESS_PATH, "success record"))
    if success["run_id"] != run_state["run_id"]:
        raise UpdateRefused(
            f"the success record belongs to run {success['run_id']} but the latest run "
            f"is {run_state['run_id']}; a leftover success record never promotes a "
            "baseline"
        )
    if success["generation"] != run_state["generation"]:
        raise UpdateRefused(
            f"the success record records generation {success['generation']} but the "
            f"latest run is generation {run_state['generation']}: a stale generation "
            "never promotes a baseline"
        )
    if (success["owner"]["token"] != run_state["owner"]["token"]
            or int(success["owner"]["pid"]) != int(run_state["owner"]["pid"])):
        raise UpdateRefused(
            "the run state and the success record were published by different owners "
            f"({run_state['owner']['token'][:8]} vs {success['owner']['token'][:8]}); a "
            "record pair from two runs never promotes a baseline"
        )
    manifest = build_manifest()
    current = manifest["groups"]["measurement"]
    recorded = success["inputs"]["groups"]["measurement"]
    if recorded["fingerprint"] != current["fingerprint"]:
        raise UpdateRefused(
            "the measured content changed since run "
            f"{success['run_id']}: recorded {recorded['file_count']} files / "
            f"{str(recorded['fingerprint'])[:12]}, now {current['file_count']} files / "
            f"{current['fingerprint'][:12]} ({MANIFEST_RULE}); re-run the gate"
        )
    return run_state, success, manifest


def candidate_baseline(measured: dict, previous: dict) -> dict:
    """The only shape a baseline may be written in: everything can tighten, nothing can loosen."""

    coverage: dict[str, dict[str, float]] = {}
    for side in ("backend", "frontend"):
        coverage[side] = {}
        for key in ("line_percent", "branch_percent"):
            floor = float(previous.get("coverage", {}).get(side, {}).get(key, 0.0))
            observed = float(measured[side][key])
            # A drop inside the noise floor keeps the older, higher floor.
            coverage[side][key] = max(floor, observed)

    guarded: dict[str, int] = {}
    migrations: dict = {
        source: to_plain(declaration)
        for source, declaration in (previous.get("guarded_migrations", {}) or {}).items()
    }
    for relative, ceiling in sorted(previous.get("guarded_files", {}).items()):
        path = ROOT / relative
        if path.is_file():
            guarded[relative] = min(int(ceiling), line_count(path))
            continue
        declaration = migrations.get(relative)
        if declaration is None:
            raise UpdateRefused(
                f"guarded path {relative} is registered at {int(ceiling)} lines but no "
                "longer exists and carries no guarded_migrations declaration; a "
                "deletion, rename, split, merge or retirement must be declared with "
                "source, kind, targets, responsibility, source_ceiling, target_ceilings "
                "and reason (FIX-05A)"
            )
        parsed = parse_migration_declaration(
            relative, declaration, f"guarded migration for {relative!r}",
            previous_ceiling=int(ceiling),
            line_of=_live_line_of,
            migrations=migrations,
        )
        if parsed["kind"] == "retire":
            migrations[relative] = {**declaration, "kind": "retire"}
            continue
        endpoints = resolve_migration_endpoints(relative, declaration, migrations)
        if endpoints is None:
            raise UpdateRefused(
                f"guarded migration {relative!r} no longer resolves to a path that "
                "exists; declare where the budget moved on instead of leaving the "
                "contract pointing at a path that is gone (FIX-06D)"
            )
        for target in endpoints:
            current = line_count(ROOT / target)
            ceiling_for_target = min(
                int(parsed["target_ceilings"].get(target, current)), current
            )
            guarded[target] = min(guarded.get(target, ceiling_for_target),
                                  ceiling_for_target)
        recorded = {
            target: int(value)
            for target, value in (declaration.get("target_ceilings", {}) or {}).items()
        }
        for target in parsed["targets"]:
            if target in guarded:
                recorded[target] = guarded[target]
        migrations[relative] = {
            **declaration,
            "target_ceilings": {
                target: recorded[target] for target in parsed["targets"]
            },
        }
    for relative, lines in sorted(measured["sizes"].items()):
        path = ROOT / relative
        guarded[relative] = min(guarded.get(relative, int(lines)), line_count(path))
    for relative, responsibility in sorted(SPLIT_MODULE_REGISTRY.items()):
        path = ROOT / relative
        if not path.is_file():
            raise UpdateRefused(
                f"the tool registry declares {relative} ({responsibility}) but the file "
                "is gone; a rename or split must update SPLIT_MODULE_REGISTRY in "
                "project/scripts/quality_gate.py deliberately (FIX-01D step 3 / AF-10)"
            )
        current = line_count(path)
        guarded[relative] = min(guarded.get(relative, current), current)

    ceilings: dict[str, int] = {
        name: int(value)
        for name, value in previous.get("api_factory_max_lines", {}).items()
    }
    for name in measured["api_factory"]:
        ceilings[name] = min(ceilings.get(name, API_FACTORY_LIMIT), API_FACTORY_LIMIT)
    missing = sorted(
        name for name in previous.get("api_factory_max_lines", {})
        if name not in measured["api_factory"]
    )
    if missing:
        raise UpdateRefused(
            "baseline-guarded API factories are no longer measurable: "
            f"{missing}; a rename must be an explicit registry change (V1-06 AC-07)"
        )

    candidate = {
        "baseline_version": BASELINE_VERSION,
        "coverage_tolerance_pp": COVERAGE_TOLERANCE_PP,
        "updated_at": now(),
        "measured_at": measured.get("measured_at"),
        "git_head": measured.get("git_head"),
        "coverage": coverage,
        "guarded_files": dict(sorted(guarded.items())),
        "api_factory_max_lines": dict(sorted(ceilings.items())),
        "guarded_registry": 1,
        "size_exemptions": dict(sorted(previous.get("size_exemptions", {}).items())),
    }
    if migrations:
        candidate["guarded_migrations"] = {
            source: migrations[source] for source in sorted(migrations)
        }
    return candidate


def assert_not_weaker(previous: dict, candidate: dict, measured: dict) -> None:
    """The candidate's own self-check: it may never lower an existing protection."""

    problems: list[str] = []
    for side in ("backend", "frontend"):
        for key in ("line_percent", "branch_percent"):
            old = float(previous.get("coverage", {}).get(side, {}).get(key, 0.0))
            new = float(candidate["coverage"][side][key])
            if new < old:
                problems.append(f"coverage floor lowered: {side} {key} {old} -> {new}")
    old_guarded = previous.get("guarded_files", {})
    new_guarded = candidate["guarded_files"]
    for relative, ceiling in sorted(old_guarded.items()):
        if not (ROOT / relative).is_file():
            continue
        if relative not in new_guarded:
            problems.append(f"guarded file lost its registration: {relative}")
        elif new_guarded[relative] > int(ceiling):
            problems.append(
                f"guarded file ceiling raised: {relative} {ceiling} -> "
                f"{new_guarded[relative]}"
            )
    previous_migrations = previous.get("guarded_migrations", {}) or {}
    for source, declaration in sorted(previous_migrations.items()):
        if not isinstance(declaration, dict):
            continue
        endpoints = resolve_migration_endpoints(source, declaration, previous_migrations)
        targets = endpoints if endpoints is not None else declaration.get("targets")
        if not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, str) or not (ROOT / target).is_file():
                continue
            if target not in new_guarded:
                problems.append(
                    f"migration target lost its registration: {target} (from {source})"
                )
    old_ceilings = previous.get("api_factory_max_lines", {})
    for name, ceiling in sorted(old_ceilings.items()):
        if name not in candidate["api_factory_max_lines"]:
            problems.append(f"API factory ceiling dropped: {name}")
        elif candidate["api_factory_max_lines"][name] > int(ceiling):
            problems.append(
                f"API factory ceiling raised: {name} {ceiling} -> "
                f"{candidate['api_factory_max_lines'][name]}"
            )
    added_exemptions = sorted(
        set(candidate["size_exemptions"]) - set(previous.get("size_exemptions", {}))
    )
    if added_exemptions:
        problems.append(f"size exemptions added: {added_exemptions}")
    if previous.get("guarded_registry") and not candidate.get("guarded_registry"):
        problems.append("the split-module registry flag was dropped")
    problems.extend(ratchet_problems(measured, candidate))
    if problems:
        raise UpdateRefused(
            "refusing to write a weaker or self-inconsistent baseline:\n"
            + "\n".join(problems)
        )


def build_update_record(*, mode: str, run_id: str, generation: int, previous: dict | None,
                        candidate: dict, manifest: dict,
                        baseline_sha_before: str | None) -> dict:
    return {
        "record_version": UPDATE_RECORD_VERSION,
        "gate_version": GATE_VERSION,
        "mode": mode,
        "run_id": run_id,
        "generation": generation,
        "at": now(),
        "baseline": {
            "path": str(BASELINE_PATH.relative_to(ROOT)),
            "sha256_before": baseline_sha_before,
            "sha256_after": hashlib.sha256(
                (json.dumps(candidate, ensure_ascii=False, sort_keys=True, indent=2)
                 + "\n").encode("utf-8")
            ).hexdigest(),
            "baseline_version_before": (previous or {}).get("baseline_version"),
            "baseline_version_after": candidate["baseline_version"],
        },
        "governance_check": {
            "checked_at": now(),
            "fingerprint": manifest["groups"]["governance"]["fingerprint"],
            "file_count": manifest["groups"]["governance"]["file_count"],
            "result": "ok",
        },
        "measurement_inputs": dict(manifest["groups"]["measurement"]),
    }


def build_intent(*, mode: str, run_id: str, generation: int, previous: dict | None,
                 candidate: dict, manifest: dict,
                 baseline_sha_before: str | None) -> dict:
    return {
        "record_version": INTENT_RECORD_VERSION,
        "gate_version": GATE_VERSION,
        "mode": mode,
        "run_id": run_id,
        "generation": generation,
        "created_at": now(),
        "old": {
            "path": str(BASELINE_PATH.relative_to(ROOT)),
            "sha256": baseline_sha_before,
            "present": baseline_sha_before is not None,
            "baseline_version": (previous or {}).get("baseline_version"),
        },
        "new": {
            "sha256": hashlib.sha256(
                (json.dumps(candidate, ensure_ascii=False, sort_keys=True, indent=2)
                 + "\n").encode("utf-8")
            ).hexdigest(),
            "baseline_version": candidate["baseline_version"],
        },
        "candidate": candidate,
        "update_record": build_update_record(
            mode=mode, run_id=run_id, generation=generation, previous=previous,
            candidate=candidate, manifest=manifest,
            baseline_sha_before=baseline_sha_before,
        ),
    }


def commit_baseline(intent: dict) -> None:
    """The single commit point: write the intent, then replace the baseline atomically.

    Anything that fails before the replace raises ``UpdateRefused`` and leaves the
    contract byte-identical.  Anything that fails *after* it raises
    ``CommittedIncomplete``: the caller must report a committed baseline with a recovery
    entry point, never "unchanged".
    """

    write_record(INTENT_PATH, intent, "intent")
    try:
        atomic_write_json(BASELINE_PATH, to_plain(intent["candidate"]), private=False)
    except BaseException as exc:  # noqa: BLE001 - reported as a refusal below
        raise UpdateRefused(
            f"the baseline was not written ({exc}); the contract keeps its exact bytes"
        ) from exc
    try:
        write_record(UPDATE_PATH, to_plain(intent["update_record"]), "update")
    except BaseException as exc:  # noqa: BLE001 - the commit already happened
        raise CommittedIncomplete(
            "baseline already committed: the contract was replaced atomically and now "
            f"carries the intended bytes (sha256 {intent['new']['sha256'][:12]}), but the "
            f"auxiliary update record could not be written ({exc}); finish the records "
            "with `python3 project/scripts/quality_gate.py --recover-baseline`"
        ) from exc


def verify_governance_now() -> None:
    """Re-run the governance checks so the newest documentation state is covered."""

    print("[gate] update: re-running the governance section against the current tree...",
          flush=True)
    try:
        section_gov()
    except GateFailure as failure:
        raise UpdateRefused(
            "the current tree fails the governance checks, so this measurement may not "
            f"be promoted even though the measured content is unchanged:\n{failure}"
        ) from failure


def review_before_write(run_state: dict, success: dict) -> dict:
    """The final review performed under the lock, immediately before the commit point.

    Ordinary editors do not obey any lock, so the recorded measurement is re-bound to the
    current content, the current run identity and the current baseline bytes here.  Only
    the review that happens directly before the atomic replace counts.
    """

    latest = parse_run_record(read_record_file(RUN_PATH, "run state"))
    if latest["run_id"] != run_state["run_id"]:
        raise UpdateRefused(
            f"a newer run ({latest['run_id']}, {latest['status']}, generation "
            f"{latest['generation']}) arrived while this measurement was being reviewed; "
            f"run {success['run_id']} is no longer the latest run and never publishes"
        )
    if latest["generation"] != run_state["generation"]:
        raise UpdateRefused(
            f"the latest run generation is now {latest['generation']} but this "
            f"measurement was taken in generation {run_state['generation']}: a stale "
            "generation never publishes"
        )
    manifest = build_manifest()
    recorded = success["inputs"]["groups"]["measurement"]
    current = manifest["groups"]["measurement"]
    if recorded["fingerprint"] != current["fingerprint"]:
        raise UpdateRefused(
            "the measured content changed while this promotion was being reviewed "
            f"(recorded {recorded['file_count']} files / "
            f"{str(recorded['fingerprint'])[:12]}, now {current['file_count']} files / "
            f"{current['fingerprint'][:12]}); the measurement no longer describes this "
            "work tree, so it is not promoted"
        )
    return manifest


def update_baseline() -> int:
    run_state: dict | None = None
    candidate: dict | None = None
    baseline_sha_before: str | None = None
    manifest: dict | None = None
    try:
        with gate_lock("update"):
            run_state, success, manifest = load_success_evidence()
            if not BASELINE_PATH.is_file():
                raise UpdateRefused(
                    f"no baseline at {BASELINE_PATH}; a repository that never had a "
                    "contract uses `--bootstrap-baseline "
                    "--coverage-floor backend=<line>/<branch>,frontend=<line>/<branch>` "
                    "instead (in this repository a missing baseline means the committed "
                    "contract was deleted: restore it)"
                )
            baseline_bytes_before = BASELINE_PATH.read_bytes()
            previous = parse_baseline(
                load_record_text(baseline_bytes_before.decode("utf-8"),
                                 f"existing baseline at {BASELINE_PATH}")
            )
            baseline_sha_before = hashlib.sha256(baseline_bytes_before).hexdigest()
            recorded_baseline = success["baseline"]
            if recorded_baseline["sha256"] != baseline_sha_before:
                raise UpdateRefused(
                    "the baseline already moved after run "
                    f"{success['run_id']} (recorded "
                    f"{str(recorded_baseline['sha256'])[:12]}, now "
                    f"{baseline_sha_before[:12]}); either this measurement was already "
                    "promoted or the contract was edited by hand — re-run the gate and "
                    "update once"
                )
            verify_governance_now()
            measurement = success["measurement"]
            candidate = candidate_baseline(to_plain(measurement), previous)
            assert_not_weaker(previous, candidate, to_plain(measurement))
            manifest = review_before_write(run_state, success)
            final_bytes = BASELINE_PATH.read_bytes()
            if hashlib.sha256(final_bytes).hexdigest() != baseline_sha_before:
                raise UpdateRefused(
                    "the baseline was replaced while this promotion was being reviewed; "
                    "the reviewed measurement is bound to different bytes and is not "
                    "promoted"
                )
            intent = build_intent(
                mode="update", run_id=run_state["run_id"],
                generation=run_state["generation"], previous=previous, candidate=candidate,
                manifest=manifest, baseline_sha_before=baseline_sha_before,
            )
            commit_baseline(intent)
    except CommittedIncomplete as partial:
        print(str(partial), file=sys.stderr)
        return 1
    except (UpdateRefused, GateBusy) as refusal:
        print(f"baseline unchanged: {refusal}", file=sys.stderr)
        return 1
    print(
        f"baseline rewritten from the complete green run {run_state['run_id']} "
        f"(measured {success['measurement']['measured_at']}, "
        f"HEAD {success['measurement']['git_head']})"
    )
    print("coverage:", json.dumps(candidate["coverage"], sort_keys=True))
    print(
        f"guarded files: {len(candidate['guarded_files'])} "
        f"(registry modules: {len(SPLIT_MODULE_REGISTRY)}); "
        f"exemptions preserved: {len(candidate['size_exemptions'])}"
    )
    print("api factory ceilings:",
          json.dumps(candidate["api_factory_max_lines"], sort_keys=True))
    print(
        f"governance re-checked now and green ({manifest['groups']['governance']['file_count']}"
        f" files, {manifest['groups']['governance']['fingerprint'][:12]}); "
        f"audit record at {UPDATE_PATH}"
    )
    print("review the diff and commit it deliberately")
    return 0


def parse_coverage_floors(specification: str | None) -> dict[str, dict[str, float]]:
    if not specification:
        raise UpdateRefused(
            "--bootstrap-baseline requires explicit initial floors: "
            "--coverage-floor backend=<line>/<branch>,frontend=<line>/<branch>"
        )
    floors: dict[str, dict[str, float]] = {}
    for part in specification.split(","):
        part = part.strip()
        if not part or "=" not in part or "/" not in part:
            raise UpdateRefused(
                f"malformed coverage floor {part!r}; expected side=<line>/<branch>"
            )
        side, values = part.split("=", 1)
        line, branch = values.split("/", 1)
        side = side.strip()
        if side not in ("backend", "frontend"):
            raise UpdateRefused(f"unknown coverage floor side {side!r}")
        if side in floors:
            raise UpdateRefused(
                f"the coverage floor for {side} was declared twice; declare each side "
                "exactly once"
            )
        floors[side] = {
            "line_percent": _floor_value(line, part),
            "branch_percent": _floor_value(branch, part),
        }
    for side in ("backend", "frontend"):
        if side not in floors:
            raise UpdateRefused(f"no initial floor declared for {side}")
    return floors


def _floor_value(text: str, part: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise UpdateRefused(
            f"malformed coverage floor {part!r}: {text!r} is not a number"
        ) from exc
    if not math.isfinite(value):
        raise UpdateRefused(
            f"malformed coverage floor {part!r}: {text!r} is not a finite number "
            "between 0 and 100"
        )
    if value < 0 or value > 100:
        raise UpdateRefused(
            f"malformed coverage floor {part!r}: {text!r} must be between 0 and 100"
        )
    return value


def bootstrap_baseline(floors_specification: str | None) -> int:
    run_state: dict | None = None
    candidate: dict | None = None
    manifest: dict | None = None
    try:
        with gate_lock("bootstrap"):
            if BASELINE_PATH.is_file():
                raise UpdateRefused(
                    f"a baseline already exists at {BASELINE_PATH}; bootstrap never "
                    "replaces an existing contract — a measurable change goes through "
                    "--update-baseline, and a deleted contract is restored from Git"
                )
            floors = parse_coverage_floors(floors_specification)
            run_state, success, manifest = load_success_evidence()
            if (success["baseline_state"] != BASELINE_STATE_NEVER_CREATED
                    or success["baseline"]["present"]):
                raise UpdateRefused(
                    f"the latest run ({success['run_id']}) recorded "
                    f"baseline_state={success['baseline_state']!r} with "
                    f"baseline.present={success['baseline']['present']!r}: its "
                    "measurement was taken while a prior contract existed, and a first "
                    "baseline may only come from a run that measured a repository with "
                    "no prior contract — restore the committed contract instead"
                )
            evidence = prior_contract_evidence()
            if evidence is not None:
                raise UpdateRefused(
                    "bootstrap is only for a repository that never had a contract, but "
                    f"this work tree shows a prior contract ({evidence}); restore the "
                    f"committed contract with `git checkout -- "
                    f"{BASELINE_PATH.relative_to(ROOT)}`"
                )
            measurement = success["measurement"]
            for side in ("backend", "frontend"):
                for key in ("line_percent", "branch_percent"):
                    if floors[side][key] > float(measurement[side][key]):
                        raise UpdateRefused(
                            f"the declared initial floor {side} {key} "
                            f"{floors[side][key]} is above the measured "
                            f"{measurement[side][key]}; a first baseline cannot promise "
                            "more than the run that establishes it"
                        )
            candidate = candidate_baseline(to_plain(measurement), {})
            candidate["coverage"] = floors
            assert_not_weaker({}, candidate, to_plain(measurement))
            manifest = review_before_write(run_state, success)
            intent = build_intent(
                mode="bootstrap", run_id=run_state["run_id"],
                generation=run_state["generation"], previous=None, candidate=candidate,
                manifest=manifest, baseline_sha_before=None,
            )
            commit_baseline(intent)
    except CommittedIncomplete as partial:
        print(str(partial), file=sys.stderr)
        return 1
    except (UpdateRefused, GateBusy) as refusal:
        print(f"baseline unchanged: {refusal}", file=sys.stderr)
        return 1
    print(
        f"baseline created from the complete green run {run_state['run_id']} with the "
        "pre-declared initial floors"
    )
    print("coverage floors:", json.dumps(candidate["coverage"], sort_keys=True))
    print(f"guarded files: {len(candidate['guarded_files'])}")
    print("review the diff and commit it deliberately")
    return 0


def recover_baseline() -> int:
    """Finish or classify an interrupted publication from its intent record."""

    try:
        with gate_lock("recovery"):
            if not INTENT_PATH.is_file():
                raise UpdateRefused(
                    f"no publish intent at {INTENT_PATH}: there is nothing to recover"
                )
            intent = parse_intent_record(read_record_file(INTENT_PATH, "publish intent"))
            expected_old = intent["old"]["sha256"]
            expected_new = intent["new"]["sha256"]
            current = file_sha256(BASELINE_PATH) if BASELINE_PATH.is_file() else None
            if current == expected_new:
                if UPDATE_PATH.is_file():
                    print(
                        f"the baseline already carries the intended new bytes "
                        f"({expected_new[:12]}) and the auxiliary record exists; nothing "
                        "to recover"
                    )
                    return 0
                write_record(UPDATE_PATH, to_plain(intent["update_record"]), "update")
                print(
                    "baseline already committed: the contract carries the intended new "
                    f"bytes ({expected_new[:12]}); the auxiliary records were completed "
                    "from the intent"
                )
                return 0
            if current == expected_old:
                print(
                    "no commit happened: the baseline still carries the reviewed old "
                    f"bytes ({str(expected_old)[:12]}); re-run --update-baseline when "
                    "the work tree is ready"
                )
                return 0
            raise UpdateRefused(
                f"the baseline hash {str(current)[:12] if current else 'absent'} is "
                f"neither the expected old ({str(expected_old)[:12]}) nor the expected "
                f"new ({expected_new[:12]}) hash of the intent at {INTENT_PATH}; stop "
                "and inspect the contract and the intent manually — this tool will not "
                "roll back over another legitimate commit"
            )
    except (UpdateRefused, GateBusy) as refusal:
        print(f"baseline unchanged: {refusal}", file=sys.stderr)
        return 1


def install_hook() -> int:
    hooks_dir = ROOT / ".git" / "hooks"
    if not hooks_dir.is_dir():
        print(f"no git hooks directory at {hooks_dir}", file=sys.stderr)
        return 1
    hook = hooks_dir / "pre-push"
    if hook.is_file():
        existing = hook.read_text(encoding="utf-8")
        if HOOK_MARKER not in existing:
            print(
                f"refusing to clobber an existing pre-push hook without the "
                f"{HOOK_MARKER!r} marker; merge it manually",
                file=sys.stderr,
            )
            return 1
        if existing == HOOK_CONTENT:
            print(f"pre-push hook already installed at {hook} (idempotent no-op)")
            return 0
    hook.write_text(HOOK_CONTENT, encoding="utf-8")
    os.chmod(hook, 0o755)
    print(f"pre-push hook installed at {hook}")
    print("it runs the full quality gate (about ten minutes); "
          "escape hatch: git push --no-verify")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "rewrite the baseline contract from the latest complete green run, after "
            "verifying its success record, its input fingerprints and the old baseline"
        ),
    )
    parser.add_argument(
        "--bootstrap-baseline",
        action="store_true",
        help=(
            "create the baseline contract for a repository that never had one; requires "
            "--coverage-floor and refuses to replace an existing contract"
        ),
    )
    parser.add_argument(
        "--recover-baseline",
        action="store_true",
        help=(
            "classify an interrupted publication from its intent record: report the "
            "baseline as committed or unchanged, and stop on any other hash"
        ),
    )
    parser.add_argument(
        "--coverage-floor",
        metavar="SPEC",
        help="initial floors for --bootstrap-baseline: backend=<line>/<branch>,frontend=<line>/<branch>",
    )
    parser.add_argument(
        "--install-hook",
        action="store_true",
        help="idempotently install the pre-push hook that runs this gate",
    )
    arguments = parser.parse_args(argv)
    selected = [
        arguments.update_baseline,
        arguments.bootstrap_baseline,
        arguments.recover_baseline,
        arguments.install_hook,
    ]
    if sum(bool(value) for value in selected) > 1:
        print(
            "choose one of --update-baseline, --bootstrap-baseline, "
            "--recover-baseline, --install-hook",
            file=sys.stderr,
        )
        return 2
    if arguments.update_baseline:
        return update_baseline()
    if arguments.bootstrap_baseline:
        return bootstrap_baseline(arguments.coverage_floor)
    if arguments.recover_baseline:
        return recover_baseline()
    if arguments.install_hook:
        return install_hook()
    return run_gate()


if __name__ == "__main__":
    raise SystemExit(main())
