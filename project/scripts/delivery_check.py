#!/usr/bin/env python3
"""Cold-start delivery check for the V1-04 Compose workbench.

Stage scope: V1-04B.  It carries the full predeclared verification matrix of the V1-04
Plan §5 that can be driven automatically:

* ENV-01/ENV-02 — cold start, offline restart and the three-level lock check;
* BOOT-01/BOOT-02 — one command to an operable workbench, and the fail-closed paths;
* ROLE-01 — the least-privilege database boundary of the delivered roles;
* GOLD-01 — the accepted golden flow and the fixed nine-Case business expectations on
  the persisted `rules 2.0.0` contract, including receipt/audit uniqueness;
* SECS-01 — the security-gate subset (anonymous, insufficient scope, foreign object,
  unknown object, retired v1, denial ledger);
* RECS-01 — the recovery subset REC-01/02/04/05 executed against the delivered
  topology by really terminating an `api` container at the V1-03 controlled injection
  points;
* EVD-01 — the eval pipeline proven offline with the mock transport, plus a replay of
  the scored report;
* LIM-01 — every limitation the V1-04 Spec §7 registers is present in the delivered
  operations page;
* TEAR-01 — `down -v` leaves no orchestration residual behind.

The browser walkthrough (WALK-01) and the formal cold-start acceptance run belong to
V1-04C; this script is the carrier they execute.

Everything runs against one checkout and one Docker project prefix.  The script only
touches resources carrying that prefix (`v104cs` by default) and never writes a secret,
a token or a DSN into its report.

Two artifacts are written into `--output`: `delivery-observations.json` holds the raw
per-scenario rows, and `delivery-report.json` holds the counters derived from them.
`--verify-report` recomputes every reported counter from the raw rows without touching
Docker, the network or the database, and fails on a missing row, a duplicated row or a
drifted count.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_API_PORT = 8000
DEFAULT_WEB_PORT = 5173
WEB_TITLE_MARKER = "Review Desk"

STAGE = "V1-04B"
CHECK_NAME = "v104cs-delivery"
DEFAULT_PREFIX = "v104cs"
CREDENTIAL_RELATIVE = Path(".local/delivery/credentials/demo-operator.token")
CREDENTIAL_DIRECTORY_RELATIVE = Path(".local/delivery/credentials")
UNMIGRATED_DATABASE = "review_workbench_unmigrated"
GOLDEN_CASE = "CASE-LOW-001"
SERVICE_WAIT_SECONDS = 420
RUN_WAIT_SECONDS = 180
INTERRUPT_WAIT_SECONDS = 150
CONTAINER_EXIT_SECONDS = 180
DATABASE_DOWN_SECONDS = 150
EVD_BATCH = "v1-02c-calibration-48"
EVD_TOPOLOGY = "single_agent"
EVD_RUNS_DIR = "/app/.local/runs"
EVD_DRY_RUN_ALIAS = "dry-run-model-alias"
EVD_DRY_RUN_RESOLVED_ID = "dry-run-model-resolved-local"
FAULT_PLAN_VERSION = "fault-plan.v1"
FAULT_EXIT_CODE = 70
DELIVERY_DOC_RELATIVE = Path("project/docs/delivery.md")
OBSERVATIONS_FILENAME = "delivery-observations.json"
REPORT_FILENAME = "delivery-report.json"
REPORT_TEXT_FILENAME = "delivery-report.txt"
DEMO_SUBJECT = "subject-local-demo"
READER_SUBJECT = "subject-local-demo-reader"
FOREIGN_SUBJECT = "subject-alpha-operator"
ACCEPTANCE_READER_NAME = "acceptance-reader"
ACCEPTANCE_FOREIGN_NAME = "acceptance-foreign"
TERMINAL_RUN_STATUSES = frozenset(
    {
        "pending_confirmation",
        "pending_human_review",
        "confirmed",
        "rejected",
        "cancelled",
        "failed",
        "interrupted",
    }
)
SCENARIOS = (
    "preflight",
    "lim01",
    "env01",
    "env02",
    "boot01",
    "gold",
    "secs",
    "recs",
    "evd",
    "boot02",
    "role01",
    "teardown",
)

# The accepted V1-01 nine-Case business expectations, carried on the persisted
# `rules 2.0.0` contract.  The table is frozen exactly as V1-05 measured it (REG-01),
# including the one registered deviation: `CASE-CONFLICT-001` reports two conflict
# entries under rules 2.0.0 (the shared EvidenceRef is an explicit conflict anchor).
GOLD_RULES_VERSION = "2.0.0"
GOLD_EXPECTATIONS: dict[str, dict[str, object]] = {
    "CASE-LOW-001": {
        "status": "pending_confirmation",
        "recommendation": "recommend_approve",
        "conflicts": 0,
        "branch_failures": 0,
        "action": "confirm",
    },
    "CASE-HIGH-001": {
        "status": "pending_confirmation",
        "recommendation": "recommend_reject",
        "conflicts": 0,
        "branch_failures": 0,
        "action": "reject",
    },
    "CASE-CONFLICT-001": {
        "status": "pending_human_review",
        "recommendation": "needs_human_review",
        "conflicts": 2,
        "branch_failures": 0,
        "action": None,
    },
    "CASE-UNKNOWN-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 1,
        "action": None,
    },
    "CASE-PARTIAL-FAILURE-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 1,
        "action": None,
    },
    "CASE-ALL-FAILURE-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 3,
        "action": None,
    },
    "CASE-VERSION-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 3,
        "action": None,
    },
    "CASE-CANCEL-001": {
        "status": "cancelled",
        "recommendation": None,
        "conflicts": 0,
        "branch_failures": None,
        "action": "cancel",
    },
    "CASE-RUNTIME-FAILURE-001": {
        "status": "failed",
        "recommendation": None,
        "conflicts": 0,
        "branch_failures": None,
        "action": None,
    },
}
GOLD_DECISIONS = {"confirm": "confirmed", "reject": "rejected"}

# The recovery subset: a minimum representative set of the V1-03 matrix, reusing its
# controlled injection points unchanged.  One plan per scenario, each crashing the
# `api` process at a named commit boundary.
RECS_SCENARIOS = ("REC-01", "REC-02", "REC-04", "REC-05")
RECS_PLANS: dict[str, dict[str, object]] = {
    "REC-01": {"barriers": {"after_create_commit": {"action": "exit"}}},
    "REC-02": {"barriers": {"branch_committed:policy": {"action": "exit"}}},
    "REC-04": {"barriers": {"after_all_branches_committed": {"action": "exit"}}},
    "REC-05": {"barriers": {"after_summary_commit": {"action": "exit"}}},
}
RECS_RESUME = {"REC-01": True, "REC-02": True, "REC-04": True, "REC-05": False}

# LIM-01: the limitations V1-04 Spec §7 requires the delivery page to register, one
# stable id each, plus the phrases that show each one is actually described.
REQUIRED_LIMITATION_IDS = (
    "L-1",
    "L-2",
    "NG-01",
    "NG-02",
    "NG-03",
    "NG-04",
    "NG-05",
    "Q-1",
    "H-1",
    "A-1",
)
REQUIRED_DOC_PHRASES = (
    ("L-1 role separation", "review_runtime"),
    ("L-2 controlled fault contract", "受控故障"),
    ("Q-1 measured quality pointer", "V1-02"),
    ("H-1 holdout consumption", "holdout"),
    ("A-1 artifact locality", ".local/runs"),
)
DELIVERY_DOC_LINKS = {
    "README.md": ("docs/delivery.md",),
    "project/docs/index.md": ("delivery.md",),
    "docs/release-preparation.md": ("../project/docs/delivery.md",),
}

_DSN_PATTERN = re.compile(r"postgresql(?:\+psycopg)?://[^\s\"']+")

# Every HTTP call goes straight to the loopback port this check published.  Python reads
# the machine's system proxy settings on macOS, and a proxy answers `502 Bad Gateway`
# for a port where nothing is listening yet — which would turn "the container is not
# ready" into a spurious HTTP answer, let a stray local server masquerade as the
# workbench, and route local acceptance traffic off the machine.
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def redact(text: str) -> str:
    """Remove anything that looks like a DSN or a token before it is recorded."""

    return _DSN_PATTERN.sub("<redacted-dsn>", text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def now() -> str:
    return datetime.now(UTC).isoformat()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def business_summary_checksum(body: dict) -> str | None:
    """Canonical digest of one Run's business result, independent of its run id.

    Mirrors `security_recovery_check.business_summary` + `trace.canonical_checksum`, so
    "the recovered result equals the no-fault baseline" means the same thing here as it
    does in the V1-03 recovery matrix.
    """

    summary = body.get("summary") if isinstance(body, dict) else None
    if summary is None:
        return None
    payload = dict(summary)
    payload["run_id"] = "run-placeholder"
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def without_request_id(payload: object) -> object:
    """Drop the per-request correlation id before comparing two error bodies."""

    clone = json.loads(json.dumps(payload))
    if isinstance(clone, dict):
        clone.get("error", {}).pop("request_id", None)
    return clone


def token_echo_count(token: str, observed: object) -> int:
    """How often a credential value appears in the recorded projections."""

    return json.dumps(observed, ensure_ascii=False).count(token)


def gold_case_mismatches(case_id: str, observed: dict) -> list[str]:
    """Compare one raw Case outcome with the frozen rules 2.0.0 expectation.

    Used both while the Run is live and by `--verify-report`, so the recomputation does
    not depend on any counter the run itself recorded.
    """

    expected = GOLD_EXPECTATIONS.get(case_id)
    if expected is None:
        return ["not_a_registered_case"]
    if observed.get("error"):
        return [str(observed["error"])]
    problems: list[str] = []
    if observed.get("status") != expected["status"]:
        problems.append("status")
    if observed.get("recommendation") != expected["recommendation"]:
        problems.append("recommendation")
    if observed.get("conflicts") != expected["conflicts"]:
        problems.append("conflicts")
    if (
        expected["branch_failures"] is not None
        and observed.get("branch_failures") != expected["branch_failures"]
    ):
        problems.append("branch_failures")
    if observed.get("rules_version") not in {None, GOLD_RULES_VERSION}:
        problems.append("rules_version")
    if observed.get("receipt") is not None:
        problems.append("receipt_before_action")
    action = expected["action"]
    if action == "cancel":
        if observed.get("audit_event_count") != 1:
            problems.append("cancel_audit_event_count")
    elif action is None:
        if observed.get("audit_event_count") != 0:
            problems.append("audit_event_count")
    if action in GOLD_DECISIONS:
        after = observed.get("after_action") or {}
        if after.get("status") != GOLD_DECISIONS[action]:
            problems.append("decided_status")
        if after.get("receipt_count") != 1:
            problems.append("decided_receipt_count")
        if after.get("audit_event_count") != 1:
            problems.append("decided_audit_event_count")
        if after.get("external_write_performed") is not False:
            problems.append("external_write_performed")
        if after.get("simulation_mode") != "local_test_only":
            problems.append("simulation_mode")
        if after.get("human_actor") != DEMO_SUBJECT:
            problems.append("human_actor")
    return problems


def recs_invariant_failures(rec: str, observed: dict) -> list[str]:
    """The V1-03 recovery invariants for the four delivered subset scenarios.

    Same counting rules as `security_recovery_check._rec_invariants`: committed branches
    are reused instead of being re-run, nothing is committed twice, and the recovered
    business result equals the no-fault baseline.
    """

    if observed.get("error"):
        return [str(observed["error"])]
    ledger = observed.get("ledger") or {}
    attempts = ledger.get("attempts") or []
    checkpoints = ledger.get("checkpoints") or []
    workers = {item.get("worker") for item in attempts}
    failures: list[str] = []
    if observed.get("summary_matches_baseline") is not True:
        failures.append("summary_matches_baseline")
    if rec == "REC-01":
        if ledger.get("resume_count") != 1:
            failures.append("resume_count_is_one")
        if not all(item.get("generation") == 2 for item in checkpoints):
            failures.append("nothing_committed_before_the_crash")
        if workers != {"policy", "contract", "procurement_data"}:
            failures.append("all_workers_attempted")
    elif rec == "REC-02":
        if ledger.get("resume_count") != 1:
            failures.append("resume_count_is_one")
        if not any(item.get("worker") == "policy" for item in checkpoints):
            failures.append("policy_reused")
        if sum(1 for item in attempts if item.get("worker") == "policy") != 1:
            failures.append("policy_attempted_once")
        if not {"contract", "procurement_data"} <= workers:
            failures.append("other_branches_ran")
    elif rec == "REC-04":
        if len({item.get("worker") for item in checkpoints}) != 3:
            failures.append("three_checkpoints")
        if not all(item.get("attempt_no") == 1 for item in attempts):
            failures.append("no_second_attempt")
        if len(attempts) != 3:
            failures.append("no_repeat_calls")
        if ledger.get("resume_count") != 1:
            failures.append("resume_count_is_one")
    elif rec == "REC-05":
        if observed.get("final_status") not in {
            "pending_confirmation",
            "pending_human_review",
        }:
            failures.append("status_pending")
        if ledger.get("receipts") != 0:
            failures.append("no_receipt")
        if len(ledger.get("generations") or []) != 1:
            failures.append("no_second_generation")
        if ledger.get("resume_count") != 0:
            failures.append("no_resume")
    else:
        failures.append("unknown_scenario")
    return failures


# ---------------------------------------------------------------- derived counters
#
# Every counter in `delivery-report.json` is a pure function of the raw observation row
# written into `delivery-observations.json`.  `--verify-report` recomputes them from the
# raw rows, so a missing row, a duplicated row or an edited measurement shows up as a
# mismatch instead of being reported as fact.


def _env01_counters(observations: dict) -> dict:
    return {
        "first_start_seconds": observations.get("first_start_seconds"),
        "offline_restart_seconds": observations.get("offline_restart_seconds"),
        "rebuild_identical": observations.get("rebuild_identical"),
    }


def _env02_counters(observations: dict) -> dict:
    return {
        "pinned_base_images": len(observations.get("pinned_base_images") or {}),
        "pinned_digest_mismatches": len(
            observations.get("pinned_digest_mismatches") or []
        ),
    }


def _boot01_counters(observations: dict) -> dict:
    golden = observations.get("golden_run") or {}
    return {
        "cases_visible": (observations.get("cases") or {}).get("count"),
        "golden_status": golden.get("status"),
        "golden_rules_version": golden.get("rules_version"),
        "demo_token_unchanged": observations.get("demo_token_unchanged", {}).get(
            "sha256_before"
        )
        == observations.get("demo_token_unchanged", {}).get("sha256_after"),
    }


def _boot02_counters(observations: dict) -> dict:
    database_down = observations.get("database_down") or {}
    attempts = database_down.get("attempts") or []
    return {
        "no_dsn_cases_status": (observations.get("no_dsn") or {}).get("cases_status"),
        "unmigrated_cases_status": observations.get("unmigrated_cases_status"),
        "database_down_status": database_down.get("during_status"),
        "database_down_attempts": len(attempts),
        "database_down_served_data": sum(
            1 for item in attempts if item.get("status") == 200
        ),
        "recovered_status": observations.get("recovered_status"),
    }


def _role01_counters(observations: dict) -> dict:
    counts = (observations.get("role_probe") or {}).get("counts") or {}
    return {
        "unauthorized_writes_succeeded": counts.get("unauthorized_writes_succeeded"),
        "denials_are_insufficient_privilege": counts.get(
            "denials_are_insufficient_privilege"
        ),
        "procurement_facts_unchanged": counts.get("procurement_facts_unchanged"),
        "lock_reads_ok": counts.get("lock_reads_ok"),
        "registry_guards": counts.get("registry_guards"),
        "escalation_effective": counts.get("escalation_effective"),
    }


def _gold_counters(observations: dict) -> dict:
    cases = {
        case_id: observations.get(case_id)
        for case_id in GOLD_EXPECTATIONS
        if observations.get(case_id) is not None
    }
    mismatches = {
        case_id: gold_case_mismatches(case_id, row) for case_id, row in cases.items()
    }
    mismatches = {case_id: item for case_id, item in mismatches.items() if item}
    ledger = observations.get("case_ledgers") or {}
    decided = [
        case_id
        for case_id, row in cases.items()
        if (row or {}).get("after_action") is not None
    ]
    return {
        "cases_expected": len(GOLD_EXPECTATIONS),
        "cases_observed": len(cases),
        "expectation_mismatches": len(mismatches),
        "decided_cases": len(decided),
        "receipts_recorded": sum(
            1 for item in ledger.values() if (item or {}).get("receipts") == 1
        ),
        "replays_identical": sum(
            1
            for case_id in decided
            if (cases[case_id] or {}).get("receipt_id")
            and (cases[case_id] or {}).get("receipt_id")
            == (cases[case_id] or {}).get("replay_receipt_id")
        ),
        "token_echoes": observations.get("token_echoes"),
    }


def _secs_counters(observations: dict) -> dict:
    return {
        "anonymous_status": (observations.get("anonymous_create") or {}).get("status"),
        "reader_status": (observations.get("reader_confirm") or {}).get("status"),
        "unknown_object_status": observations.get("unknown_object_status"),
        "foreign_status": (observations.get("foreign_object") or {}).get("status"),
        "foreign_matches_unknown": (observations.get("foreign_object") or {}).get(
            "same_shape_as_unknown"
        ),
        "retired_v1_status": (observations.get("retired_v1") or {}).get("status"),
        "denial_events_added": observations.get("denial_events_added"),
        "token_echoes": observations.get("token_echoes"),
    }


def _recs_counters(observations: dict) -> dict:
    rows = {
        rec: observations.get(rec)
        for rec in RECS_SCENARIOS
        if observations.get(rec) is not None
    }
    failures = {rec: recs_invariant_failures(rec, row) for rec, row in rows.items()}
    return {
        "scenarios_expected": len(RECS_SCENARIOS),
        "scenarios_observed": len(rows),
        "recovery_success": {
            "numerator": sum(1 for item in failures.values() if not item),
            "denominator": len(rows),
        },
        "crashes_with_expected_exit": sum(
            1 for row in rows.values() if row.get("crash_exit_code") == FAULT_EXIT_CODE
        ),
        "barriers_reached": sum(
            1 for row in rows.values() if row.get("barrier_reached") is True
        ),
        "baseline_matches": sum(
            1 for row in rows.values() if row.get("summary_matches_baseline") is True
        ),
    }


def _evd_counters(observations: dict) -> dict:
    segment = observations.get("segment") or {}
    replays = observations.get("replay") or []
    return {
        "execution_mode": segment.get("execution_mode"),
        "outcome": segment.get("outcome"),
        "branch_count": segment.get("branch_count"),
        "model_call_count": segment.get("model_call_count"),
        "resolved_model_ids": segment.get("resolved_model_ids"),
        "replays": len(replays),
        "replays_identical": len(replays) == 2
        and replays[0].get("stdout") == replays[1].get("stdout"),
        "replays_succeeded": sum(1 for item in replays if item.get("exit") == 0),
    }


def _lim01_counters(observations: dict) -> dict:
    presence = observations.get("limitations_present") or {}
    phrases = observations.get("phrases_present") or {}
    links = observations.get("entry_links") or {}
    return {
        "limitations_expected": len(REQUIRED_LIMITATION_IDS),
        "limitations_registered": sum(1 for item in presence.values() if item),
        "missing_limitations": sorted(
            item for item, present in presence.items() if not present
        ),
        "phrases_missing": sorted(
            item for item, present in phrases.items() if not present
        ),
        "entry_links_ok": all(links.get(item) for item in DELIVERY_DOC_LINKS),
    }


def _teardown_counters(observations: dict) -> dict:
    residuals = observations.get("residuals") or {}
    return {
        "residual_containers": len(residuals.get("containers") or []),
        "residual_volumes": len(residuals.get("volumes") or []),
        "residual_networks": len(residuals.get("networks") or []),
        "host_leftovers": observations.get("host_local_leftovers"),
    }


COUNTER_DERIVERS = {
    "env01": _env01_counters,
    "env02": _env02_counters,
    "boot01": _boot01_counters,
    "boot02": _boot02_counters,
    "role01": _role01_counters,
    "gold": _gold_counters,
    "secs": _secs_counters,
    "recs": _recs_counters,
    "evd": _evd_counters,
    "lim01": _lim01_counters,
    "teardown": _teardown_counters,
}


def derive_counters(row: dict) -> dict:
    deriver = COUNTER_DERIVERS.get(str(row.get("scenario")))
    if deriver is None:
        return {}
    try:
        return deriver(row.get("observations") or {})
    except Exception as error:  # noqa: BLE001 - a broken row is a mismatch, not a crash
        return {"derivation_error": f"{type(error).__name__}: {error}"[:200]}


def row_status(row: dict) -> str:
    assertions = row.get("assertions") or []
    return "failed" if any(not item.get("passed") for item in assertions) else "passed"


def row_failures(row: dict) -> list[str]:
    return [
        f"{row.get('scenario')}: {item.get('check')}"
        for item in row.get("assertions") or []
        if not item.get("passed")
    ]


def build_report(
    rows: list[dict], *, environment: dict, limitations: list[str], duration_s: float
) -> dict:
    failures: list[str] = []
    for row in rows:
        failures.extend(row_failures(row))
    passed = sum(1 for row in rows if row_status(row) == "passed")
    verification_notes = rows_limitations(rows)
    return {
        "check": CHECK_NAME,
        "stage": STAGE,
        "reported_scenarios": [row.get("scenario") for row in rows],
        "status": "passed" if not failures else "failed",
        "totals": {
            "scenarios": len(rows),
            "passed": passed,
            "failed": len(rows) - passed,
            "assertions": sum(len(row.get("assertions") or []) for row in rows),
            "failed_assertions": len(failures),
        },
        "counters": {str(row.get("scenario")): derive_counters(row) for row in rows},
        "failures": failures,
        "environment": environment,
        "verification_notes": verification_notes,
        "limitations": list(limitations),
        "total_duration_s": round(duration_s, 2),
        "generated_at": now(),
    }


def rows_limitations(rows: list[dict]) -> list[str]:
    """Boundaries that stay true whatever the run measured."""

    notes = [
        "Browsers are not driven here: the golden flow is executed against the delivered "
        "API and the served workbench page; the user/developer/auditor walkthrough "
        "(WALK-01) belongs to V1-04C.",
        "Recovery is exercised through the local controlled fault contract of V1-04 "
        "Spec §2.2 L-2; it is not a production recovery guarantee.",
        "The lock check proves the committed lock files resolve and that the images were "
        "built from them; it does not claim byte-identical no-cache rebuilds.",
        "The eval proof uses the mock dry-run transport with networking disabled; it is "
        "not a model-quality result.",
    ]
    if any(row.get("scenario") == "evd" for row in rows):
        notes.append(
            "Dry-run eval artifacts live under the private output directory and are "
            "machine-local; a real batch replay needs the V1-02/V1-05 products copied "
            "first (see the delivery page)."
        )
    return notes


def verify_report(directory: Path) -> dict:
    """Recompute the report from the raw observation rows, without any side effect."""

    observations_path = directory / OBSERVATIONS_FILENAME
    report_path = directory / REPORT_FILENAME
    if not observations_path.is_file() or not report_path.is_file():
        raise RuntimeError("the report directory is incomplete")
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = observations.get("rows") or []
    names = [row.get("scenario") for row in rows]
    duplicates = sorted({str(name) for name in names if names.count(name) > 1})
    unknown = sorted({str(name) for name in names if name not in SCENARIOS})
    reported = list(report.get("reported_scenarios") or [])
    missing = [name for name in reported if name not in names]
    status_mismatches = sorted(
        str(row.get("scenario")) for row in rows if row.get("status") != row_status(row)
    )
    recomputed_counters = {
        str(row.get("scenario")): derive_counters(row) for row in rows
    }
    recorded_counters = report.get("counters") or {}
    counter_mismatches = sorted(
        name
        for name in set(recomputed_counters) | set(recorded_counters)
        if recomputed_counters.get(name) != recorded_counters.get(name)
    )
    recomputed_failures: list[str] = []
    for row in rows:
        recomputed_failures.extend(row_failures(row))
    recomputed_totals = {
        "scenarios": len(rows),
        "passed": sum(1 for row in rows if row_status(row) == "passed"),
        "failed": sum(1 for row in rows if row_status(row) != "passed"),
        "assertions": sum(len(row.get("assertions") or []) for row in rows),
        "failed_assertions": len(recomputed_failures),
    }
    recorded_totals = report.get("totals") or {}
    totals_mismatch = sorted(
        key
        for key in set(recomputed_totals) | set(recorded_totals)
        if recomputed_totals.get(key) != recorded_totals.get(key)
    )
    recomputed_status = "passed" if not recomputed_failures else "failed"
    status_mismatch = report.get("status") != recomputed_status
    failures_match = list(report.get("failures") or []) == recomputed_failures
    verified = not (
        duplicates
        or unknown
        or missing
        or status_mismatches
        or counter_mismatches
        or totals_mismatch
        or status_mismatch
        or not failures_match
    )
    return {
        "verified": verified,
        "check": report.get("check"),
        "stage": report.get("stage"),
        "observations": len(rows),
        "scenarios": sorted(str(name) for name in names),
        "missing_rows": missing,
        "duplicate_rows": duplicates,
        "unknown_rows": unknown,
        "status_mismatches": status_mismatches,
        "counter_mismatches": counter_mismatches,
        "totals_mismatch": totals_mismatch,
        "report_status": report.get("status"),
        "recomputed_status": recomputed_status,
        "failures_match": failures_match,
        "recomputed_totals": recomputed_totals,
        "recomputed_counters": recomputed_counters,
        "network_access": False,
    }


def require_absolute_mount_source(path: Path, *, context: str) -> Path:
    """Fail closed unless a docker `-v` source is an absolute host path.

    A relative or otherwise non-absolute source reaches the daemon resolved against
    nothing this check controls, which is exactly how the audit's never-ready recs
    containers and the exit-125 evd runs were produced (V1-06 AC-01).
    """

    if not path.is_absolute():
        raise RuntimeError(
            f"{context}: docker mount source {path} is not an absolute host path; "
            "refusing to start the container (mount sources are normalized from "
            "--checkout/--output before this point)"
        )
    return path


def merged_output_tail(completed: subprocess.CompletedProcess, limit: int = 1200) -> str:
    """The tail of stdout+stderr combined, for failure observations."""

    return redact(f"{completed.stdout}\n{completed.stderr}".strip()[-limit:])


class Delivery:
    """One delivery-check run against one checkout and one Docker prefix."""

    def __init__(
        self,
        *,
        checkout: Path,
        prefix: str,
        output: Path,
        scenarios: tuple[str, ...],
        teardown: bool,
        api_port: int | None = None,
        web_port: int | None = None,
    ) -> None:
        self.checkout = checkout.resolve()
        self.prefix = prefix
        # Normalize like checkout: every docker bind-mount source derived from the
        # output directory must reach the daemon as an absolute host path, no matter
        # how --output was spelled (V1-06 AC-01).
        self.output = output.resolve()
        self.scenarios = scenarios
        self.do_teardown = teardown
        self.compose_file = self.checkout / "project" / "docker-compose.yml"
        self.compose_dir = self.compose_file.parent
        self.credential_file = self.checkout / "project" / CREDENTIAL_RELATIVE
        self.credential_dir = self.checkout / "project" / CREDENTIAL_DIRECTORY_RELATIVE
        self.port_substitutions: dict[str, dict[str, object]] = {}
        self.fault_starts: dict[str, dict[str, object]] = {}
        self.api_port = self._resolve_port("api", DEFAULT_API_PORT, api_port)
        self.web_port = self._resolve_port("web", DEFAULT_WEB_PORT, web_port)
        self.api_base = f"http://127.0.0.1:{self.api_port}"
        self.web_base = f"http://127.0.0.1:{self.web_port}"
        self.results: dict[str, dict] = {}
        self.commands: list[dict] = []
        self.failures: list[str] = []
        self.environment: dict[str, object] = {}
        self.started = time.time()

    def _resolve_port(self, name: str, default: int, requested: int | None) -> int:
        """Use the delivered default port, or the first free alternative if it is taken.

        A delivery check must not fight another process for a loopback port, and it must
        not silently talk to somebody else's server.  The substitution is recorded.
        """

        candidate = requested or default
        if port_is_free(candidate):
            return candidate
        if requested:
            raise RuntimeError(f"requested {name} port {candidate} is already in use")
        for alternative in range(default + 10000, default + 10020):
            if port_is_free(alternative):
                self.port_substitutions[name] = {
                    "default": default,
                    "used": alternative,
                    "reason": f"127.0.0.1:{default} was already in use",
                }
                return alternative
        raise RuntimeError(f"no free {name} port near {default}")

    def compose_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "V104CS_API_PORT": str(self.api_port),
            "V104CS_WEB_PORT": str(self.web_port),
        }

    # ------------------------------------------------------------- bookkeeping

    def row(self, scenario: str) -> dict:
        """One raw observation row per scenario; the report is derived from these."""

        return self.results.setdefault(
            scenario,
            {
                "scenario": scenario,
                "status": "failed",
                "duration_s": 0.0,
                "assertions": [],
                "observations": {},
                "notes": [],
            },
        )

    def expect(self, scenario: str, condition: bool, message: str) -> bool:
        self.row(scenario)["assertions"].append(
            {"check": message, "passed": bool(condition)}
        )
        if not condition:
            self.failures.append(f"{scenario}: {message}")
        return bool(condition)

    def observe(self, scenario: str, key: str, value: object) -> None:
        self.row(scenario)["observations"][key] = value

    def note(self, scenario: str, message: str) -> None:
        self.row(scenario)["notes"].append(message)

    def scenario_failed(self, scenario: str) -> bool:
        return any(item.startswith(f"{scenario}:") for item in self.failures)

    # ---------------------------------------------------------------- commands

    def _record_command(
        self, argv: list[str], exit_code: int, duration: float, output: str
    ) -> None:
        self.commands.append(
            {
                "command": redact(" ".join(argv)),
                "exit": exit_code,
                "duration_s": round(duration, 3),
                "output_tail": redact(output[-1500:]),
            }
        )

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 900,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        start = time.time()
        completed = subprocess.run(
            argv,
            cwd=str(cwd or self.compose_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=self.compose_env(),
        )
        output = f"{completed.stdout}\n{completed.stderr}".strip()
        self._record_command(argv, completed.returncode, time.time() - start, output)
        if check and completed.returncode != 0:
            raise RuntimeError(
                redact(f"{' '.join(argv)} failed with exit {completed.returncode}")
            )
        return completed

    def compose(self, *arguments: str, timeout: int = 900, check: bool = False):
        return self.run(
            [
                "docker-compose",
                "-f",
                str(self.compose_file),
                "-p",
                self.prefix,
                *arguments,
            ],
            timeout=timeout,
            check=check,
        )

    def docker(self, *arguments: str, timeout: int = 300, check: bool = False):
        return self.run(["docker", *arguments], timeout=timeout, check=check)

    def oneoff(
        self,
        *command: str,
        service: str = "init",
        extra_env: dict[str, str] | None = None,
        detach: bool = False,
        name: str | None = None,
        service_ports: bool = False,
        entrypoint: str | None = None,
        timeout: int = 600,
    ):
        argv = [
            "docker-compose",
            "-f",
            str(self.compose_file),
            "-p",
            self.prefix,
            "run",
            "--rm",
            "-T",
            "--no-deps",
        ]
        if detach:
            argv.append("-d")
        if service_ports:
            argv.append("--service-ports")
        if name:
            argv += ["--name", name]
        if entrypoint:
            argv += ["--entrypoint", entrypoint]
        for key, value in (extra_env or {}).items():
            argv += ["-e", f"{key}={value}"]
        argv += [service, *command]
        return self.run(argv, timeout=timeout)

    # -------------------------------------------------------------------- http

    def http(
        self,
        method: str,
        url: str,
        *,
        token: str | None = None,
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict | str]:
        body = None
        request_headers = dict(headers or {})
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        if token:
            request_headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            url, data=body, headers=request_headers, method=method
        )
        try:
            with DIRECT_OPENER.open(request, timeout=30) as response:
                raw = response.read().decode("utf-8", "replace")
                status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")
            status = error.code
        except urllib.error.URLError as error:
            return 0, f"transport error: {type(error.reason).__name__}"
        except (http.client.HTTPException, OSError) as error:
            # A container that is still binding its port accepts and then drops the
            # connection; that is a "not ready yet" signal, never a check failure.
            return 0, f"transport error: {type(error).__name__}"
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, raw

    def api(self, method: str, path: str, **kwargs):
        return self.http(method, f"{self.api_base}{path}", **kwargs)

    def wait_for_api_ready(
        self, base: str | None = None, *, attempts: int = 60
    ) -> bool:
        """Wait until the API on `base` answers *as this workbench*.

        The identity marker matters: a machine proxy or an unrelated local server must
        never be mistaken for the container under test, and a port where nothing listens
        yet must stay "not ready" instead of producing a spurious HTTP status.
        """

        target = base or self.api_base
        for _ in range(attempts):
            status, body = self.http("GET", f"{target}/api/v2/health")
            if (
                status == 200
                and isinstance(body, dict)
                and body.get("service") == "review-api"
            ):
                return True
            time.sleep(2)
        return False

    # ---------------------------------------------------------------- waiting

    def service_states(self) -> dict[str, dict]:
        listed = self.docker(
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={self.prefix}",
            "--format",
            "{{.Names}}",
        )
        names = [line for line in listed.stdout.split() if line]
        if not names:
            return {}
        inspected = self.docker(
            "inspect",
            "--format",
            '{{.Name}}|{{index .Config.Labels "com.docker.compose.service"}}|'
            "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|"
            "{{.State.ExitCode}}",
            *names,
        )
        states: dict[str, dict] = {}
        for line in inspected.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) != 5:
                continue
            states[parts[1] or parts[0].lstrip("/")] = {
                "name": parts[0].lstrip("/"),
                "status": parts[2],
                "health": parts[3],
                "exit_code": int(parts[4] or 0),
            }
        return states

    def wait_for_stack(self, scenario: str, *, require_init: bool = True) -> bool:
        deadline = time.time() + SERVICE_WAIT_SECONDS
        states: dict[str, dict] = {}
        while time.time() < deadline:
            states = self.service_states()
            ready = all(
                states.get(service, {}).get("status") == "running"
                and states.get(service, {}).get("health") == "healthy"
                for service in ("db", "api", "web")
            )
            if require_init:
                init = states.get("init", {})
                ready = (
                    ready
                    and init.get("status") == "exited"
                    and init.get("exit_code", 1) == 0
                )
            if ready:
                self.observe(scenario, "service_states", states)
                return True
            time.sleep(3)
        self.observe(scenario, "service_states", states)
        return False

    # ----------------------------------------------------------------- checks

    def scenario_preflight(self) -> None:
        scenario = "preflight"
        status = self.run(["git", "status", "--porcelain"], cwd=self.checkout)
        head = self.run(["git", "rev-parse", "HEAD"], cwd=self.checkout)
        self.expect(
            scenario,
            status.stdout.strip() == "",
            "the checkout has uncommitted changes",
        )
        self.observe(scenario, "git_status_clean", status.stdout.strip() == "")
        self.observe(scenario, "git_head", head.stdout.strip())
        # Record the measuring instrument itself: the report is only reproducible if the
        # checker version that produced it is known.
        self.observe(scenario, "checker_sha256", sha256_file(Path(__file__).resolve()))
        for forbidden in (
            self.checkout / "project/backend/.env.local",
            self.checkout / "project/backend/.local",
            self.checkout / "project/.local",
        ):
            self.expect(
                scenario,
                not forbidden.exists(),
                f"the checkout still carries local state at {forbidden}",
            )
            self.observe(scenario, f"absent::{forbidden.name}", not forbidden.exists())

        docker = self.run(["docker", "--version"], cwd=self.checkout)
        compose = self.run(["docker-compose", "--version"], cwd=self.checkout)
        info = self.run(
            ["docker", "info", "--format", "{{.ServerVersion}}|{{.OperatingSystem}}"],
            cwd=self.checkout,
        )
        self.environment = {
            "host_platform": platform.platform(),
            "host_python": sys.version.split()[0],
            "docker_client": docker.stdout.strip(),
            "compose": compose.stdout.strip(),
            "docker_server": info.stdout.strip(),
            "checkout": str(self.checkout),
            "prefix": self.prefix,
            "api_port": self.api_port,
            "web_port": self.web_port,
            "port_substitutions": self.port_substitutions,
            "started_at": now(),
        }
        self.observe(scenario, "environment", self.environment)

        rendered = self.compose("config", timeout=120)
        self.expect(scenario, rendered.returncode == 0, "compose config did not render")
        self.observe(
            scenario,
            "compose_config_sha256",
            hashlib.sha256(rendered.stdout.encode("utf-8")).hexdigest(),
        )
        self.observe(
            scenario,
            "compose_services",
            sorted(self.service_states().keys()) or "not running yet",
        )
        self.observe(
            scenario,
            "docker_state_before",
            {
                "containers": self.docker(
                    "ps", "-a", "--format", "{{.Names}}"
                ).stdout.split(),
                "volumes": self.docker(
                    "volume", "ls", "--format", "{{.Name}}"
                ).stdout.split(),
                "networks": self.docker(
                    "network", "ls", "--format", "{{.Name}}"
                ).stdout.split(),
            },
        )

    def scenario_env01(self) -> None:
        scenario = "env01"
        start = time.time()
        up = self.compose("up", "--build", "-d", timeout=2400)
        self.expect(scenario, up.returncode == 0, "the one-key start command failed")
        ready = self.wait_for_stack(scenario)
        self.observe(scenario, "first_start_seconds", round(time.time() - start, 2))
        self.expect(scenario, ready, "the stack did not become healthy")

        images = self.docker(
            "inspect",
            "--format",
            "{{.RepoTags}}|{{.Id}}|{{.Created}}",
            "v104cs-backend:local",
            "v104cs-frontend:local",
        )
        self.observe(scenario, "built_images_first", images.stdout.strip().splitlines())
        self.observe(scenario, "build_log_tail", redact(up.stdout[-800:]))

        rebuild = self.compose("build", timeout=2400)
        self.expect(scenario, rebuild.returncode == 0, "the second build failed")
        images_again = self.docker(
            "inspect",
            "--format",
            "{{.RepoTags}}|{{.Id}}|{{.Created}}",
            "v104cs-backend:local",
            "v104cs-frontend:local",
        )
        first = images.stdout.strip().splitlines()
        second = images_again.stdout.strip().splitlines()
        self.observe(scenario, "built_images_second", second)
        self.expect(
            scenario,
            first == second,
            "two builds of the same checkout produced different images",
        )
        self.observe(scenario, "rebuild_identical", first == second)

        # Offline restart: tear the containers down, keep the volumes, start again from
        # the images that are already local.  Any registry access would show up as a
        # pull in the captured output.
        down = self.compose("down", timeout=600)
        self.expect(scenario, down.returncode == 0, "down before the restart failed")
        restart_start = time.time()
        again = self.compose("up", "-d", "--pull", "never", timeout=1200)
        restarted = self.wait_for_stack(scenario)
        self.observe(
            scenario, "offline_restart_seconds", round(time.time() - restart_start, 2)
        )
        self.observe(scenario, "offline_restart_output", redact(again.stdout[-600:]))
        self.expect(scenario, again.returncode == 0, "the offline restart failed")
        self.expect(scenario, restarted, "the stack did not recover after the restart")
        self.expect(
            scenario,
            "Pulling" not in again.stdout and "Building" not in again.stdout,
            "the restart contacted a registry or rebuilt an image",
        )

    def scenario_env02(self) -> None:
        scenario = "env02"
        backend = self.checkout / "project/backend"
        frontend = self.checkout / "project/frontend"
        self.observe(scenario, "uv_lock_sha256", sha256_file(backend / "uv.lock"))
        self.observe(
            scenario,
            "package_lock_sha256",
            sha256_file(frontend / "package-lock.json"),
        )

        lock_check = self.oneoff("lock", "--check", "--offline", entrypoint="uv")
        self.observe(scenario, "uv_lock_check_exit", lock_check.returncode)
        self.observe(
            scenario, "uv_lock_check_output", redact(lock_check.stderr.strip())
        )
        self.expect(
            scenario, lock_check.returncode == 0, "uv lock --check did not pass offline"
        )

        node_lock = self.oneoff(
            "node",
            "-e",
            "console.log(JSON.stringify({node:process.version,"
            "vite:require('/app/node_modules/vite/package.json').version,"
            "react:require('/app/node_modules/react/package.json').version,"
            "typescript:require('/app/node_modules/typescript/package.json').version,"
            "lockPackages:Object.keys(require('/app/package-lock.json').packages).length}))",
            service="web",
        )
        self.observe(scenario, "node_toolchain", redact(node_lock.stdout.strip()))
        self.observe(scenario, "node_lock_probe_exit", node_lock.returncode)
        self.expect(
            scenario,
            node_lock.returncode == 0 and "lockPackages" in node_lock.stdout,
            "the frontend image could not read its own lock file",
        )

        pins = self.pinned_images()
        self.observe(scenario, "pinned_base_images", pins)
        mismatches = []
        for name, digest in pins.items():
            inspected = self.docker(
                "image", "inspect", "--format", "{{json .RepoDigests}}", name
            )
            if digest not in inspected.stdout:
                mismatches.append(name)
        self.expect(
            scenario,
            not mismatches,
            f"pinned digests do not match the local images: {mismatches}",
        )
        self.observe(scenario, "pinned_digest_mismatches", mismatches)
        self.observe(scenario, "pulled_image_digests", self.local_digests(pins))

    def pinned_images(self) -> dict[str, str]:
        """Every `name@sha256:...` reference pinned in the delivery files."""

        pattern = re.compile(r"([\w./-]+):([\w.-]+)@(sha256:[0-9a-f]{64})")
        pins: dict[str, str] = {}
        sources = (
            self.checkout / "project/backend/Dockerfile",
            self.checkout / "project/frontend/Dockerfile",
            self.compose_file,
        )
        for source in sources:
            for match in pattern.finditer(source.read_text(encoding="utf-8")):
                pins[f"{match.group(1)}:{match.group(2)}"] = match.group(3)
        return pins

    def local_digests(self, pins: dict[str, str]) -> dict[str, str]:
        digests: dict[str, str] = {}
        for name in pins:
            inspected = self.docker(
                "image", "inspect", "--format", "{{json .RepoDigests}}", name
            )
            digests[name] = inspected.stdout.strip()
        return digests

    def scenario_boot01(self) -> None:
        scenario = "boot01"
        health_status, health = self.api("GET", "/api/v2/health")
        self.expect(scenario, health_status == 200, "health endpoint is not reachable")
        self.expect(
            scenario,
            isinstance(health, dict) and health.get("security_store") == "ready",
            "health does not report a ready security store",
        )
        self.observe(scenario, "health", health)

        anonymous_status, anonymous = self.api("GET", "/api/v2/cases")
        self.expect(
            scenario, anonymous_status == 401, "an anonymous caller was not refused"
        )
        self.observe(
            scenario,
            "anonymous_cases",
            {"status": anonymous_status, "body": redact(str(anonymous)[:200])},
        )

        token = self.credential_file.read_text(encoding="utf-8").strip()
        session_status, session = self.api("GET", "/api/v2/session", token=token)
        self.expect(scenario, session_status == 200, "the demo credential was refused")
        self.observe(
            scenario,
            "session",
            {
                "status": session_status,
                "subject_id": session.get("subject_id")
                if isinstance(session, dict)
                else None,
                "roles": session.get("roles") if isinstance(session, dict) else None,
            },
        )

        cases_status, cases = self.api("GET", "/api/v2/cases", token=token)
        entries = cases.get("cases", []) if isinstance(cases, dict) else []
        self.expect(
            scenario, cases_status == 200 and entries, "no granted Case is visible"
        )
        self.observe(
            scenario,
            "cases",
            {
                "status": cases_status,
                "count": len(entries),
                "case_ids": [item.get("case_id") for item in entries],
            },
        )

        golden = next(
            (item for item in entries if item.get("case_id") == GOLDEN_CASE), None
        )
        if golden is None:
            self.expect(
                scenario, False, f"{GOLDEN_CASE} is not granted to the demo subject"
            )
            return

        created_status, created = self.api(
            "POST",
            "/api/v2/reviews",
            token=token,
            payload={
                "api_version": "review-api.v2",
                "case_id": golden["case_id"],
                "case_version": golden["case_version"],
            },
            headers={"Idempotency-Key": f"{self.prefix}-boot01-golden"},
        )
        self.expect(scenario, created_status == 202, "the Run was not accepted")
        if created_status != 202 or not isinstance(created, dict):
            self.observe(
                scenario,
                "create_run",
                {"status": created_status, "body": redact(str(created)[:300])},
            )
            return
        run_id = created["run_id"]

        deadline = time.time() + RUN_WAIT_SECONDS
        run: dict | str = {}
        run_status = 0
        while time.time() < deadline:
            run_status, run = self.api("GET", f"/api/v2/reviews/{run_id}", token=token)
            if isinstance(run, dict) and run.get("status") in TERMINAL_RUN_STATUSES:
                break
            time.sleep(2)
        summary = run.get("summary") if isinstance(run, dict) else None
        self.observe(
            scenario,
            "golden_run",
            {
                "run_id": run_id,
                "status": run.get("status") if isinstance(run, dict) else None,
                "rules_version": (summary or {}).get("rules_version"),
                "finding_count": len((summary or {}).get("findings", [])),
                "recommendation": (summary or {}).get("recommendation"),
                "human_review_status": (summary or {}).get("human_review_status"),
            },
        )
        self.expect(scenario, run_status == 200, "the Run could not be read back")
        self.expect(
            scenario,
            isinstance(run, dict) and run.get("status") in TERMINAL_RUN_STATUSES,
            "the Run never reached a terminal status",
        )
        self.expect(
            scenario,
            (summary or {}).get("rules_version") == "2.0.0",
            "the Run summary is not on the accepted rules 2.0.0 contract",
        )

        for suffix in ("trace", "audit"):
            endpoint_status, _body = self.api(
                "GET", f"/api/v2/reviews/{run_id}/{suffix}", token=token
            )
            self.observe(scenario, f"{suffix}_status", endpoint_status)
            self.expect(
                scenario,
                endpoint_status == 200,
                f"the {suffix} projection is not readable",
            )

        web_status, web_body = self.http("GET", f"{self.web_base}/")
        self.observe(scenario, "web_root_status", web_status)
        self.expect(scenario, web_status == 200, "the web workbench is not served")
        proxied_status, proxied = self.http("GET", f"{self.web_base}/api/v2/health")
        self.observe(scenario, "web_proxy_status", proxied_status)
        self.expect(
            scenario,
            proxied_status == 200 and isinstance(proxied, dict),
            "the same-origin /api proxy does not reach the API",
        )
        # The port may have been substituted, so prove the page really is this
        # workbench and not another local server that happens to hold the port.
        self.observe(
            scenario,
            "web_root_is_workbench",
            isinstance(web_body, str) and WEB_TITLE_MARKER in web_body,
        )
        self.expect(
            scenario,
            isinstance(web_body, str) and WEB_TITLE_MARKER in web_body,
            "the frontend port is not serving this workbench",
        )

        self.expect(
            scenario,
            token not in json.dumps(cases) and token not in json.dumps(session),
            "the credential value was echoed back by the API",
        )
        self.observe(scenario, "credential_echoes", 0)

        self.observe(scenario, "credential_file", self.credential_mode_report())

        # Idempotency: the same bootstrap again must not duplicate seed data and must
        # not rotate the still-valid credential.
        token_before = sha256_file(self.credential_file)
        before = self.oneoff("python", "/app/scripts/count_facts.py")
        first_report = self.oneoff()
        after = self.oneoff("python", "/app/scripts/count_facts.py")
        token_after = sha256_file(self.credential_file)
        self.observe(scenario, "init_first_replay", redact(first_report.stdout.strip()))
        self.observe(
            scenario,
            "counts_before_replay",
            redact(before.stdout.strip()),
        )
        self.observe(scenario, "counts_after_replay", redact(after.stdout.strip()))
        replay = self.parse_json(first_report.stdout)
        self.expect(scenario, replay is not None, "the init replay printed no report")
        if replay:
            self.observe(
                scenario,
                "replay_summary",
                {
                    "seed_already_present": replay.get("seed", {}).get(
                        "already_present"
                    ),
                    "credential_action": replay.get("credential", {}).get("action"),
                    "roles": replay.get("roles"),
                },
            )
            self.expect(
                scenario,
                replay.get("seed", {}).get("already_present") is True,
                "the second init re-seeded the registry",
            )
            self.expect(
                scenario,
                replay.get("credential", {}).get("action") == "reused",
                "the second init rotated the demo credential",
            )
        self.expect(
            scenario,
            self.parse_json(before.stdout) == self.parse_json(after.stdout),
            "registry counts changed across the init replay",
        )
        self.observe(
            scenario,
            "demo_token_unchanged",
            {"sha256_before": token_before, "sha256_after": token_after},
        )
        self.expect(
            scenario,
            token_before == token_after,
            "the demo credential file changed across the init replay",
        )

    def credential_mode_report(self) -> dict[str, object]:
        directory = self.credential_dir.stat()
        token = self.credential_file.stat()
        return {
            "path": str(self.credential_file.relative_to(self.checkout)),
            "directory_mode": oct(directory.st_mode & 0o777),
            "file_mode": oct(token.st_mode & 0o777),
            "token_sha256": sha256_file(self.credential_file),
        }

    @staticmethod
    def parse_json(text: str) -> dict | None:
        stripped = text.strip()
        start = stripped.find("{")
        if start < 0:
            return None
        try:
            return json.loads(stripped[start:])
        except json.JSONDecodeError:
            return None

    def scenario_boot02(self) -> None:
        """Fail-closed behaviour.  Each phase is independent so one failure still

        leaves the other two as evidence instead of hiding them behind an abort.
        """

        scenario = "boot02"
        self.compose("stop", "api", timeout=300)
        for phase in (
            self._boot02_without_dsn,
            self._boot02_unmigrated,
            self._boot02_database_down,
        ):
            try:
                phase(scenario)
            except Exception as error:  # noqa: BLE001 - recorded per phase
                self.expect(
                    scenario,
                    False,
                    f"{phase.__name__}: {type(error).__name__}: {redact(str(error))}",
                )
                for leftover in (
                    f"{self.prefix}-boot02-nodsn",
                    f"{self.prefix}-boot02-unmigrated",
                ):
                    self.docker("rm", "-f", leftover)
                self.compose("start", "api", timeout=300)

    def _boot02_without_dsn(self, scenario: str) -> None:
        # (a) no runtime DSN at all: the API must start and refuse every business call.
        probe = f"{self.prefix}-boot02-nodsn"
        self.oneoff(
            detach=True,
            service_ports=True,
            name=probe,
            extra_env={"REVIEW_DATABASE_URL": ""},
            service="api",
        )
        status = self.wait_for_api_ready()
        health = self.api("GET", "/api/v2/health")
        cases = self.api("GET", "/api/v2/cases")
        create = self.api(
            "POST",
            "/api/v2/reviews",
            payload={
                "api_version": "review-api.v2",
                "case_id": GOLDEN_CASE,
                "case_version": "case-low-001.v1",
            },
            headers={"Idempotency-Key": f"{self.prefix}-boot02-nodsn"},
        )
        health_body = health[1] if isinstance(health[1], dict) else {}
        self.observe(
            scenario,
            "no_dsn",
            {
                "health": health_body,
                "cases_status": cases[0],
                "create_status": create[0],
            },
        )
        self.expect(
            scenario,
            health_body.get("security_store") == "not_configured",
            "health did not report an unconfigured store",
        )
        self.expect(scenario, cases[0] == 503, "an unconfigured API served Case data")
        self.expect(scenario, create[0] == 503, "an unconfigured API accepted a Run")
        self.expect(
            scenario,
            not self.leaks_secret(str(cases[1]))
            and not self.leaks_secret(str(create[1])),
            "an unconfigured error body leaked internal detail",
        )
        self.observe(scenario, "no_dsn_http_ready_status", status)
        self.docker("rm", "-f", probe)

    def _boot02_unmigrated(self, scenario: str) -> None:
        # (b) database present but not migrated.
        self.compose(
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-c",
            f"DROP DATABASE IF EXISTS {UNMIGRATED_DATABASE}",
            "-c",
            f"CREATE DATABASE {UNMIGRATED_DATABASE} OWNER review_migrator",
        )
        self.compose(
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            "postgres",
            "-d",
            UNMIGRATED_DATABASE,
            "-c",
            "REVOKE ALL ON SCHEMA public FROM PUBLIC",
            "-c",
            f"GRANT CONNECT ON DATABASE {UNMIGRATED_DATABASE} TO review_runtime",
            "-c",
            "GRANT USAGE ON SCHEMA public TO review_runtime",
        )
        runtime_dsn = self.runtime_dsn()
        unmigrated_dsn = runtime_dsn.replace(
            "/review_workbench", f"/{UNMIGRATED_DATABASE}"
        )
        refusal = self.oneoff(
            "python",
            "-c",
            "from procurement_review.review_store import ReviewStore\n"
            f"store = ReviewStore.from_url({unmigrated_dsn!r})\n"
            "try:\n"
            "    store.ensure_migrated()\n"
            "except Exception as error:\n"
            "    print(type(error).__name__, error)\n"
            "else:\n"
            "    print('NO_ERROR')",
            extra_env={"REVIEW_DATABASE_URL": unmigrated_dsn},
        )
        self.observe(
            scenario, "unmigrated_store_refusal", redact(refusal.stdout.strip())
        )
        self.expect(
            scenario,
            "MigrationError" in refusal.stdout,
            "an unmigrated store was not refused",
        )

        unmigrated_probe = f"{self.prefix}-boot02-unmigrated"
        self.oneoff(
            detach=True,
            service_ports=True,
            name=unmigrated_probe,
            extra_env={"REVIEW_DATABASE_URL": unmigrated_dsn},
            service="api",
        )
        self.wait_for_api_ready()
        unmigrated_status = self.api("GET", "/api/v2/cases")[0]
        self.observe(scenario, "unmigrated_cases_status", unmigrated_status)
        self.expect(
            scenario, unmigrated_status == 503, "an unmigrated API served Case data"
        )
        self.docker("rm", "-f", unmigrated_probe)
        self.compose(
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-c",
            f"DROP DATABASE IF EXISTS {UNMIGRATED_DATABASE}",
        )

    def _boot02_database_down(self, scenario: str) -> None:
        # (c) database unavailable while the API keeps running.
        token = self.credential_file.read_text(encoding="utf-8").strip()
        self.compose("start", "api", timeout=300)
        self.wait_for_api_ready()
        warm_status = self.api("GET", "/api/v2/cases", token=token)[0]
        self.compose("stop", "db", timeout=300)
        time.sleep(5)
        # The first request after the outage can wait for a pooled connection that was
        # opened before the database went away (the lease scanner keeps touching the
        # store every second), so the check is bounded rather than single-shot: it fails
        # only if the endpoint never answers 503, or if it ever serves Case data.
        attempts: list[dict] = []
        during_status = 0
        during_body: dict | str = {}
        deadline = time.time() + DATABASE_DOWN_SECONDS
        while time.time() < deadline:
            attempt_started = time.time()
            code, body = self.api("GET", "/api/v2/cases", token=token)
            attempts.append(
                {"status": code, "elapsed_s": round(time.time() - attempt_started, 2)}
            )
            if code:
                during_status, during_body = code, body
                if code == 503:
                    break
            time.sleep(2)
        self.observe(
            scenario,
            "database_down",
            {
                "before_status": warm_status,
                "during_status": during_status,
                "attempts": attempts,
                "body": redact(str(during_body)[:200]),
            },
        )
        self.expect(
            scenario, warm_status == 200, "the API was not serving before the stop"
        )
        self.expect(
            scenario,
            during_status == 503,
            "the API did not fail closed without a database",
        )
        self.expect(
            scenario,
            all(item["status"] != 200 for item in attempts),
            "the API served Case data while the database was unavailable",
        )
        self.expect(
            scenario,
            not self.leaks_secret(str(during_body)),
            "the fail-closed response leaked internal detail",
        )
        self.compose("start", "db", timeout=300)
        if not self.wait_for_stack(scenario, require_init=False):
            self.note(scenario, "the stack did not recover after the database restart")
        recovered_status = self.api("GET", "/api/v2/cases", token=token)[0]
        if recovered_status != 200:
            self.note(
                scenario,
                "the API needed an explicit restart after the database returned",
            )
            self.compose("restart", "api", timeout=300)
            self.wait_for_api_ready()
            recovered_status = self.api("GET", "/api/v2/cases", token=token)[0]
        self.observe(scenario, "recovered_status", recovered_status)
        self.expect(scenario, recovered_status == 200, "the API did not recover")

    @staticmethod
    def leaks_secret(text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in ("postgresql", "password", "traceback", "review_workbench")
        )

    def runtime_dsn(self) -> str:
        completed = self.oneoff(
            "python",
            "-c",
            "print(open('/run/v104cs/runtime/runtime_dsn').read().strip())",
        )
        return completed.stdout.strip().splitlines()[-1]

    # ------------------------------------------------- delivery/scenario helpers

    def demo_token(self) -> str:
        return self.credential_file.read_text(encoding="utf-8").strip()

    def read_token_file(self, name: str) -> str | None:
        path = self.credential_dir / f"{name}.token"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8").strip()

    def ledger(self, run_id: str) -> dict | None:
        """Read one Run's persisted ledger with the least-privilege runtime DSN."""

        probe = self.oneoff(
            "python",
            "/app/scripts/run_ledger.py",
            "run",
            "--run-id",
            run_id,
            service="api",
        )
        return self.parse_json(probe.stdout)

    def denial_count(self) -> int | None:
        probe = self.oneoff(
            "python", "/app/scripts/run_ledger.py", "denials", service="api"
        )
        report = self.parse_json(probe.stdout)
        return None if report is None else report.get("denial_events")

    def wait_run(
        self,
        token: str,
        run_id: str,
        statuses: tuple[str, ...],
        *,
        timeout: int = RUN_WAIT_SECONDS,
        tolerate_unavailable: bool = False,
    ) -> tuple[int, dict | str]:
        """Poll one Run until it reaches a status (or the deadline expires)."""

        deadline = time.time() + timeout
        body: dict | str = {}
        code = 0
        while time.time() < deadline:
            code, body = self.api("GET", f"/api/v2/reviews/{run_id}", token=token)
            if tolerate_unavailable and code in {500, 503}:
                time.sleep(0.5)
                continue
            if isinstance(body, dict) and body.get("status") in statuses:
                return code, body
            time.sleep(2)
        return code, body

    def start_fault_api(
        self,
        *,
        name: str,
        port: int,
        fault_dir: Path,
        plan: dict,
        scenario: str,
    ) -> None:
        """Start one extra `api` container with a controlled fault plan mounted.

        The delivered `api` service never carries a fault plan; this overlay exists only
        for the acceptance run and is the V1-03 controlled injection point (Spec §2.2
        L-2) applied to the delivered topology.
        """

        fault_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(fault_dir, 0o777)
        plan_path = fault_dir / "plan.json"
        payload = {"fault_plan_version": FAULT_PLAN_VERSION, **plan}
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(plan_path, 0o644)
        self.observe(scenario, f"{name}_plan", payload)
        require_absolute_mount_source(fault_dir, context=f"{scenario}/{name}")
        started = self.run(
            [
                "docker-compose",
                "-f",
                str(self.compose_file),
                "-p",
                self.prefix,
                "run",
                "-d",
                "--no-deps",
                "--name",
                name,
                "-p",
                f"127.0.0.1:{port}:8000",
                "-v",
                f"{fault_dir}:/run/v104cs/fault",
                "-e",
                "V1_03_FAULT_PLAN=/run/v104cs/fault/plan.json",
                "-e",
                "V1_03_FAULT_DIR=/run/v104cs/fault",
                "api",
            ],
            timeout=600,
        )
        # Kept off the passing-path observation keys; surfaced only when the
        # fault-injected container fails so the docker stderr tail is at hand.
        self.fault_starts[name] = {
            "exit": started.returncode,
            "output_tail": merged_output_tail(started),
        }

    def wait_for_container_exit(self, name: str, *, timeout: int) -> int | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            inspected = self.docker(
                "inspect", "--format", "{{.State.Status}}|{{.State.ExitCode}}", name
            )
            if inspected.returncode == 0:
                status, _, code = inspected.stdout.strip().partition("|")
                if status == "exited":
                    return int(code or 0)
            time.sleep(1)
        return None

    def container_log_tail(self, name: str) -> str:
        return redact(self.docker("logs", "--tail", "40", name).stdout[-1200:])

    def remove_container(self, name: str) -> None:
        self.docker("rm", "-f", name)

    # ------------------------------------------------------------------- GOLD-01

    def scenario_gold(self) -> None:
        """The accepted golden flow over all nine registered Cases (AC-04 carrier)."""

        scenario = "gold"
        token = self.demo_token()
        listed_status, listed = self.api("GET", "/api/v2/cases", token=token)
        entries = (
            {item["case_id"]: item for item in listed.get("cases", [])}
            if isinstance(listed, dict)
            else {}
        )
        self.expect(scenario, listed_status == 200, "the Case list is not readable")
        self.observe(scenario, "cases_registered", sorted(entries))
        self.observe(
            scenario,
            "case_rules_versions",
            {case_id: entry.get("rules_version") for case_id, entry in entries.items()},
        )

        observed: dict[str, dict] = {}
        for case_id, expected in GOLD_EXPECTATIONS.items():
            entry = entries.get(case_id)
            if entry is None:
                observed[case_id] = {"error": "case_not_registered"}
                self.observe(scenario, case_id, observed[case_id])
                continue
            observed[case_id] = self._gold_case(
                scenario, token, case_id, entry, expected
            )
            self.observe(scenario, case_id, observed[case_id])

        mismatches = {
            case_id: gold_case_mismatches(case_id, row)
            for case_id, row in observed.items()
        }
        mismatches = {case_id: item for case_id, item in mismatches.items() if item}
        for case_id, problems in mismatches.items():
            self.expect(
                scenario,
                False,
                f"{case_id} deviates from the frozen rules 2.0.0 expectation: "
                f"{', '.join(problems)}",
            )
        self.observe(scenario, "case_expectation_mismatches", mismatches)
        self.observe(scenario, "cases_observed", len(observed))

        # Receipt and audit uniqueness are judged from the database, not from the
        # transport projection that produced them.
        ledger_rows = {
            case_id: self.ledger(row["run_id"])
            for case_id, row in observed.items()
            if row.get("run_id")
        }
        self.observe(
            scenario,
            "case_ledgers",
            {
                case_id: {
                    "status": (item or {}).get("status"),
                    "receipts": (item or {}).get("receipts"),
                    "audit_events": (item or {}).get("audit_events"),
                    "resume_count": (item or {}).get("resume_count"),
                }
                for case_id, item in ledger_rows.items()
            },
        )
        for case_id, row in observed.items():
            if not row.get("run_id"):
                continue
            item = ledger_rows.get(case_id) or {}
            if row.get("after_action") is None:
                self.expect(
                    scenario,
                    item.get("receipts") == 0,
                    f"{case_id} recorded a receipt without a human action",
                )
                continue
            self.expect(
                scenario,
                item.get("receipts") == 1,
                f"{case_id} does not carry exactly one receipt",
            )
            self.expect(
                scenario,
                item.get("audit_events") == 1,
                f"{case_id} does not carry exactly one audit event",
            )
            self.expect(
                scenario,
                row.get("replay_receipt_id") == row.get("receipt_id"),
                f"{case_id} replayed the same key into a different receipt",
            )

        # Evidence and audit projections must be readable for a decided Run, and the
        # browser workbench must be the one serving the published web port.
        readable = next(
            (row for row in observed.values() if row.get("after_action") is not None),
            None,
        )
        if readable:
            for suffix in ("trace", "audit"):
                code, _body = self.api(
                    "GET",
                    f"/api/v2/reviews/{readable['run_id']}/{suffix}",
                    token=token,
                )
                self.observe(scenario, f"{suffix}_status", code)
                self.expect(
                    scenario, code == 200, f"the {suffix} projection is not readable"
                )
        web_status, web_body = self.http("GET", f"{self.web_base}/")
        self.observe(scenario, "web_root_status", web_status)
        self.observe(
            scenario,
            "web_root_is_workbench",
            isinstance(web_body, str) and WEB_TITLE_MARKER in web_body,
        )
        self.expect(scenario, web_status == 200, "the web workbench is not served")
        self.expect(
            scenario,
            isinstance(web_body, str) and WEB_TITLE_MARKER in web_body,
            "the frontend port is not serving this workbench",
        )
        self.observe(scenario, "token_echoes", token_echo_count(token, observed))
        self.expect(
            scenario,
            token_echo_count(token, observed) == 0,
            "the credential value was echoed back by the API",
        )

    def _gold_case(
        self,
        scenario: str,
        token: str,
        case_id: str,
        entry: dict,
        expected: dict,
    ) -> dict:
        """Drive one fixed Case through the delivered API and return its raw outcome."""

        key = f"{self.prefix}-gold-{case_id.lower()}"
        created_status, created = self.api(
            "POST",
            "/api/v2/reviews",
            token=token,
            payload={
                "api_version": "review-api.v2",
                "case_id": case_id,
                "case_version": entry.get("case_version"),
            },
            headers={"Idempotency-Key": key},
        )
        if created_status not in {200, 202} or not isinstance(created, dict):
            return {"error": f"create_status={created_status}"}
        run_id = created["run_id"]
        action = expected.get("action")

        if action == "cancel":
            # The cancellable fixture holds the Run in `processing` for a short window;
            # cancelling with the version the create response returned is the same
            # contract the frozen regression uses.
            body = created
            if body.get("status") != "processing":
                return {"run_id": run_id, "error": f"status={body.get('status')}"}
            code, body = self.api(
                "POST",
                f"/api/v2/reviews/{run_id}/cancel",
                token=token,
                payload={
                    "api_version": "review-api.v2",
                    "expected_version": body["version"],
                },
                headers={"Idempotency-Key": f"{key}-cancel"},
            )
            if code != 200 or not isinstance(body, dict):
                return {"run_id": run_id, "error": f"cancel_status={code}"}
        else:
            code, body = self.wait_run(token, run_id, (str(expected["status"]),))
            if code != 200 or not isinstance(body, dict):
                return {"run_id": run_id, "error": f"read_status={code}"}

        summary = body.get("summary")
        row = {
            "run_id": run_id,
            "status": body.get("status"),
            "execution_mode": body.get("execution_mode"),
            "recommendation": None
            if summary is None
            else summary.get("recommendation"),
            "rules_version": None if summary is None else summary.get("rules_version"),
            "conflicts": 0 if summary is None else len(summary.get("conflicts") or []),
            "branch_failures": len(body.get("branch_failures") or []),
            "receipt": body.get("receipt"),
            "audit_event_count": body.get("audit_event_count"),
            "after_action": None,
            "receipt_id": None,
            "replay_receipt_id": None,
        }
        if action in GOLD_DECISIONS:
            payload = {
                "api_version": "review-api.v2",
                "expected_version": body["version"],
                "expected_summary_checksum": body["summary_checksum"],
            }
            headers = {"Idempotency-Key": f"{key}-action"}
            code, decided = self.api(
                "POST",
                f"/api/v2/reviews/{run_id}/{action}",
                token=token,
                payload=payload,
                headers=headers,
            )
            if code != 200 or not isinstance(decided, dict):
                row["error"] = f"action_status={code}"
                return row
            receipt = decided.get("receipt") or {}
            row["after_action"] = {
                "status": decided.get("status"),
                "receipt_count": 0 if decided.get("receipt") is None else 1,
                "audit_event_count": decided.get("audit_event_count"),
                "external_write_performed": receipt.get("external_write_performed"),
                "simulation_mode": receipt.get("simulation_mode"),
                "human_actor": receipt.get("human_actor"),
            }
            row["receipt_id"] = receipt.get("receipt_id")
            # Same key, same body: the committed decision must be read back instead of
            # being applied twice.
            replay_code, replayed = self.api(
                "POST",
                f"/api/v2/reviews/{run_id}/{action}",
                token=token,
                payload=payload,
                headers=headers,
            )
            if replay_code == 200 and isinstance(replayed, dict):
                row["replay_receipt_id"] = (replayed.get("receipt") or {}).get(
                    "receipt_id"
                )
            else:
                row["replay_receipt_id"] = f"status={replay_code}"
        return row

    # ------------------------------------------------------------------- SECS-01

    def scenario_secs(self) -> None:
        """The security-gate subset on the delivered topology (AC-05 carrier)."""

        scenario = "secs"
        token = self.demo_token()
        before = self.denial_count()
        self.observe(scenario, "denials_before", before)

        anonymous_status, anonymous_body = self.http(
            "POST",
            f"{self.api_base}/api/v2/reviews",
            token="not-a-credential",
            payload={
                "api_version": "review-api.v2",
                "case_id": GOLDEN_CASE,
                "case_version": "case-low-001.v1",
            },
        )
        anonymous_code = (
            (anonymous_body or {}).get("error", {}).get("code")
            if isinstance(anonymous_body, dict)
            else None
        )
        self.observe(
            scenario,
            "anonymous_create",
            {"status": anonymous_status, "error_code": anonymous_code},
        )
        self.expect(
            scenario,
            anonymous_status == 401 and anonymous_code == "unauthenticated",
            "an anonymous caller was not refused with 401 unauthenticated",
        )

        created_status, created = self.api(
            "POST",
            "/api/v2/reviews",
            token=token,
            payload={
                "api_version": "review-api.v2",
                "case_id": GOLDEN_CASE,
                "case_version": "case-low-001.v1",
            },
            headers={"Idempotency-Key": f"{self.prefix}-secs-anchor"},
        )
        if created_status not in {200, 202} or not isinstance(created, dict):
            self.expect(scenario, False, "the security anchor Run was not created")
            return
        run_id = created["run_id"]

        unknown_status, unknown_body = self.api(
            "GET", "/api/v2/reviews/run-does-not-exist", token=token
        )
        self.observe(scenario, "unknown_object_status", unknown_status)
        self.expect(scenario, unknown_status == 404, "an unknown object was not a 404")

        reader_token = self._issue_acceptance_credential(
            scenario, READER_SUBJECT, ACCEPTANCE_READER_NAME
        )
        foreign_token = self._issue_acceptance_credential(
            scenario, FOREIGN_SUBJECT, ACCEPTANCE_FOREIGN_NAME
        )
        bodies = [str(anonymous_body), str(unknown_body)]
        reader_status = 0
        foreign_status = 0
        if reader_token:
            reader_status, reader_body = self.api(
                "POST",
                f"/api/v2/reviews/{run_id}/confirm",
                token=reader_token,
                payload={
                    "api_version": "review-api.v2",
                    "expected_version": 1,
                    "expected_summary_checksum": "sha256:" + "0" * 64,
                },
                headers={"Idempotency-Key": f"{self.prefix}-secs-reader"},
            )
            reader_code = (
                reader_body.get("error", {}).get("code")
                if isinstance(reader_body, dict)
                else None
            )
            bodies.append(str(reader_body))
            self.observe(
                scenario,
                "reader_confirm",
                {"status": reader_status, "error_code": reader_code},
            )
            self.expect(
                scenario,
                reader_status == 403 and reader_code == "insufficient_scope",
                "a reader credential was not refused with 403 insufficient_scope",
            )
        if foreign_token:
            foreign_status, foreign_body = self.api(
                "GET", f"/api/v2/reviews/{run_id}", token=foreign_token
            )
            bodies.append(str(foreign_body))
            self.observe(
                scenario,
                "foreign_object",
                {
                    "status": foreign_status,
                    "same_shape_as_unknown": without_request_id(foreign_body)
                    == without_request_id(unknown_body),
                },
            )
            self.expect(
                scenario,
                foreign_status == 404,
                "a foreign-tenant object was not hidden behind a 404",
            )
            self.expect(
                scenario,
                without_request_id(foreign_body) == without_request_id(unknown_body),
                "a foreign-tenant object is distinguishable from an unknown object",
            )

        retired_status, retired_body = self.api("GET", "/api/v1/reviews", token=token)
        retired_code = (
            retired_body.get("error", {}).get("code")
            if isinstance(retired_body, dict)
            else None
        )
        retired_carries_business = bool(
            isinstance(retired_body, dict)
            and {"run_id", "runs", "summary", "cases", "receipt"} & set(retired_body)
        )
        self.observe(
            scenario,
            "retired_v1",
            {
                "status": retired_status,
                "error_code": retired_code,
                "carries_business_data": retired_carries_business,
            },
        )
        self.expect(
            scenario,
            retired_status == 410
            and retired_code == "api_version_retired"
            and not retired_carries_business,
            "the retired v1 surface did not answer with a business-free 410",
        )

        after = self.denial_count()
        added = None if before is None or after is None else after - before
        self.observe(scenario, "denials_after", after)
        self.observe(scenario, "denial_events_added", added)
        self.expect(
            scenario,
            added is not None and added >= 3,
            "the denial ledger did not record every rejection",
        )
        echoes = sum(
            body.count(item)
            for body in bodies
            for item in (token, reader_token or "", foreign_token or "")
            if item
        )
        self.observe(scenario, "token_echoes", echoes)
        self.expect(scenario, echoes == 0, "a presented credential was echoed back")

    def _issue_acceptance_credential(
        self, scenario: str, subject: str, name: str
    ) -> str | None:
        """Issue one short-lived acceptance credential through the delivered init path."""

        issued = self.oneoff(
            "python",
            "/app/scripts/credential_admin.py",
            "issue",
            "--subject",
            subject,
            "--name",
            name,
            "--ttl-seconds",
            "7200",
            service="init",
        )
        report = self.parse_json(issued.stdout)
        self.observe(
            scenario,
            f"credential::{name}",
            {
                "subject_id": (report or {}).get("subject_id"),
                "tenant_id": (report or {}).get("tenant_id"),
                "expires_at": (report or {}).get("expires_at"),
                "token_printed": (report or {}).get("token_printed"),
            },
        )
        token = self.read_token_file(name)
        self.expect(
            scenario,
            bool(token),
            f"the acceptance credential {name} was not written to the protected path",
        )
        return token

    # ------------------------------------------------------------------- RECS-01

    def scenario_recs(self) -> None:
        """The recovery subset over the delivered topology (AC-05 carrier)."""

        scenario = "recs"
        token = self.demo_token()
        baseline = self._recs_run(
            scenario, token, key=f"{self.prefix}-recs-baseline", label="baseline"
        )
        self.observe(scenario, "baseline", baseline)
        self.expect(
            scenario,
            baseline.get("status") in {"pending_confirmation", "pending_human_review"},
            "the no-fault baseline Run did not reach a reviewable status",
        )
        baseline_summary = baseline.get("business_summary_checksum")
        observed_rows: dict[str, dict] = {}

        for rec in RECS_SCENARIOS:
            observed = self._recs_scenario(scenario, token, rec, baseline_summary)
            observed_rows[rec] = observed
            self.observe(scenario, rec, observed)
            for problem in recs_invariant_failures(rec, observed):
                self.expect(scenario, False, f"{rec}: {problem}")
            self.expect(
                scenario,
                observed.get("crash_exit_code") == FAULT_EXIT_CODE,
                f"{rec}: the injected fault did not terminate the api container",
            )
            self.expect(
                scenario,
                observed.get("barrier_reached") is True,
                f"{rec}: the controlled barrier was never reached",
            )
            self.expect(
                scenario,
                observed.get("summary_matches_baseline") is True,
                f"{rec}: the recovered result differs from the no-fault baseline",
            )

        counts = {
            rec: recs_invariant_failures(rec, observed_rows[rec])
            for rec in RECS_SCENARIOS
        }
        self.observe(
            scenario,
            "recovery_success",
            {
                "numerator": sum(1 for rec in RECS_SCENARIOS if not counts[rec]),
                "denominator": len(RECS_SCENARIOS),
            },
        )
        self.note(
            scenario,
            "recovery is exercised through the local controlled fault contract "
            "(V1-04 Spec §2.2 L-2); it is not a production recovery guarantee",
        )

    def _recs_run(
        self,
        scenario: str,
        token: str,
        *,
        key: str,
        label: str,
        base: str | None = None,
    ) -> dict:
        """Create one Run and read it back with its business summary digest."""

        target = base or self.api_base
        code, created = self.http(
            "POST",
            f"{target}/api/v2/reviews",
            token=token,
            payload={
                "api_version": "review-api.v2",
                "case_id": GOLDEN_CASE,
                "case_version": "case-low-001.v1",
            },
            headers={"Idempotency-Key": key},
        )
        if code not in {200, 202} or not isinstance(created, dict):
            return {"label": label, "error": f"create_status={code}"}
        run_id = created["run_id"]
        read_code, body = self.wait_run(
            token,
            run_id,
            ("pending_confirmation", "pending_human_review"),
            tolerate_unavailable=True,
        )
        if read_code != 200 or not isinstance(body, dict):
            return {
                "label": label,
                "run_id": run_id,
                "error": f"read_status={read_code}",
            }
        return {
            "label": label,
            "run_id": run_id,
            "status": body.get("status"),
            "business_summary_checksum": business_summary_checksum(body),
        }

    def _recs_scenario(
        self, scenario: str, token: str, rec: str, baseline_summary: str | None
    ) -> dict:
        """Terminate a real `api` container at one controlled barrier and recover."""

        name = f"{self.prefix}-{rec.lower()}"
        port = free_port()
        fault_dir = self.output / "recs" / rec
        plan = RECS_PLANS[rec]
        barrier = next(iter(plan["barriers"]))  # type: ignore[index]
        marker = fault_dir / f"{barrier.replace(':', '_')}.reached"
        if marker.exists():
            marker.unlink()
        self.start_fault_api(
            name=name, port=port, fault_dir=fault_dir, plan=plan, scenario=scenario
        )
        base = f"http://127.0.0.1:{port}"
        key = f"{self.prefix}-{rec}"
        run_id: str | None = None
        create_lost = False
        try:
            if not self.wait_for_api_ready(base, attempts=45):
                return {
                    "error": "the fault-injected api container never became ready",
                    "container_log_tail": self.container_log_tail(name),
                    "start": self.fault_starts.get(name, {}),
                }
            code, created = self.http(
                "POST",
                f"{base}/api/v2/reviews",
                token=token,
                payload={
                    "api_version": "review-api.v2",
                    "case_id": GOLDEN_CASE,
                    "case_version": "case-low-001.v1",
                },
                headers={"Idempotency-Key": key},
            )
            if isinstance(created, dict) and created.get("run_id"):
                run_id = created["run_id"]
            else:
                # The process may die before answering: that is the documented
                # "result unknown" boundary, and the same key is the only safe retry.
                create_lost = True
            exit_code = self.wait_for_container_exit(
                name, timeout=CONTAINER_EXIT_SECONDS
            )
            barrier_reached = marker.exists()
            log_tail = self.container_log_tail(name)
        finally:
            self.remove_container(name)

        observed: dict = {
            "run_id": run_id,
            "create_response_lost": create_lost,
            "crash_exit_code": exit_code,
            "barrier_reached": barrier_reached,
            "barrier": barrier,
            "container_log_tail": log_tail,
            "resumed": RECS_RESUME[rec],
        }
        if exit_code != FAULT_EXIT_CODE:
            observed["error"] = "the injected fault did not kill the api container"
            return observed

        if run_id is None:
            replay_status, replayed = self.api(
                "POST",
                "/api/v2/reviews",
                token=token,
                payload={
                    "api_version": "review-api.v2",
                    "case_id": GOLDEN_CASE,
                    "case_version": "case-low-001.v1",
                },
                headers={"Idempotency-Key": key},
            )
            if isinstance(replayed, dict) and replayed.get("run_id"):
                run_id = replayed["run_id"]
                observed["run_id"] = run_id
                observed["create_replayed"] = True
            else:
                observed["error"] = f"create_replay_status={replay_status}"
                return observed

        if RECS_RESUME[rec]:
            # The delivered API keeps running while the injected container died, so it
            # is the process that identifies the expired lease and performs the
            # authorised resume.
            code, interrupted = self.wait_run(
                token, run_id, ("interrupted",), timeout=INTERRUPT_WAIT_SECONDS
            )
            if (
                not isinstance(interrupted, dict)
                or interrupted.get("status") != "interrupted"
            ):
                observed["error"] = f"interrupt_wait_status={code}"
                return observed
            observed["interrupted_version"] = interrupted.get("version")
            resume_code, resumed = self.api(
                "POST",
                f"/api/v2/reviews/{run_id}/resume",
                token=token,
                payload={
                    "api_version": "review-api.v2",
                    "expected_version": interrupted.get("version"),
                },
                headers={"Idempotency-Key": f"{key}-resume"},
            )
            observed["resume_status"] = resume_code
            if resume_code not in {200, 202}:
                observed["error"] = f"resume_status={resume_code}"
                return observed
        read_code, final = self.wait_run(
            token,
            run_id,
            ("pending_confirmation", "pending_human_review"),
            timeout=RUN_WAIT_SECONDS,
        )
        if read_code != 200 or not isinstance(final, dict):
            observed["error"] = f"final_read_status={read_code}"
            return observed
        observed["final_status"] = final.get("status")
        observed["business_summary_checksum"] = business_summary_checksum(final)
        observed["summary_matches_baseline"] = (
            observed["business_summary_checksum"] == baseline_summary
        )
        ledger = self.ledger(run_id) or {}
        observed["ledger"] = {
            "status": ledger.get("status"),
            "resume_count": ledger.get("resume_count"),
            "receipts": ledger.get("receipts"),
            "attempts": ledger.get("attempts"),
            "checkpoints": ledger.get("checkpoints"),
            "generations": ledger.get("generations"),
        }
        return observed

    # -------------------------------------------------------------------- EVD-01

    def scenario_evd(self) -> None:
        """The offline eval pipeline proof and its replay (AC-06 carrier)."""

        scenario = "evd"
        landing = self.output / "evd"
        landing.mkdir(parents=True, exist_ok=True)
        os.chmod(landing, 0o777)
        require_absolute_mount_source(landing, context="evd")
        run = self.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{landing}:/app/.local",
                "v104cs-backend:local",
                "python",
                "-m",
                "procurement_review.eval_run",
                "--batch",
                EVD_BATCH,
                "--dry-run",
                "--topology",
                EVD_TOPOLOGY,
                "--runs-dir",
                EVD_RUNS_DIR,
            ],
            timeout=2400,
        )
        self.observe(scenario, "eval_run", redact(run.stdout[-600:]))
        if run.returncode != 0:
            # Failure-only key: the docker stderr tail that explains the exit.
            self.observe(scenario, "eval_run_output_tail", merged_output_tail(run))
        self.expect(
            scenario,
            run.returncode == 0,
            "the dry-run eval pipeline did not complete",
        )
        manifest_path = next(
            (landing / "runs").glob("*/eval-run-manifest.v1.json"), None
        )
        self.expect(
            scenario, manifest_path is not None, "no eval segment manifest was written"
        )
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path is not None
            else {}
        )
        summary = {
            "run_id": manifest.get("run_id"),
            "batch_id": manifest.get("batch_id"),
            "execution_mode": manifest.get("execution_mode"),
            "outcome": manifest.get("outcome"),
            "voided": manifest.get("voided"),
            "branch_count": manifest.get("branch_count"),
            "model_call_count": manifest.get("model_call_count"),
            "requested_model_id": manifest.get("requested_model_id"),
            "resolved_model_ids": manifest.get("resolved_model_ids"),
            "http_status_counts": manifest.get("http_status_counts"),
            "failure_code": manifest.get("failure_code"),
        }
        self.observe(scenario, "segment", summary)
        self.expect(
            scenario,
            summary["execution_mode"] == "dry_run"
            and summary["requested_model_id"] == EVD_DRY_RUN_ALIAS
            and summary["resolved_model_ids"] == [EVD_DRY_RUN_RESOLVED_ID],
            "the segment was not produced by the mock dry-run transport",
        )
        self.expect(
            scenario,
            summary["outcome"] == "completed" and summary["voided"] is False,
            "the dry-run segment did not complete",
        )
        self.expect(
            scenario,
            summary["batch_id"] == EVD_BATCH and summary["branch_count"] == 48,
            "the dry-run segment does not cover the frozen batch",
        )

        # Replay: the same artifacts are scored twice in a row and must agree.
        replays = []
        for _ in range(2):
            scored = self.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "-v",
                    f"{landing}:/app/.local",
                    "v104cs-backend:local",
                    "python",
                    "-m",
                    "procurement_review.eval_score",
                    "--batch",
                    EVD_BATCH,
                    "--runs-dir",
                    EVD_RUNS_DIR,
                    "--print-summary",
                ],
                timeout=1800,
            )
            replay_entry: dict[str, object] = {
                "exit": scored.returncode,
                "stdout": redact(scored.stdout.strip()),
            }
            if scored.returncode != 0:
                # Failure-only key: merged stdout+stderr tail for diagnosis.
                replay_entry["output_tail"] = merged_output_tail(scored)
            replays.append(replay_entry)
        self.observe(scenario, "replay", replays)
        self.expect(
            scenario,
            all(item["exit"] == 0 for item in replays),
            "the scored report could not be replayed from the artifacts",
        )
        self.expect(
            scenario,
            len(replays) == 2 and replays[0]["stdout"] == replays[1]["stdout"],
            "two replays of the same artifacts produced different numbers",
        )

    # -------------------------------------------------------------------- LIM-01

    def scenario_lim01(self) -> None:
        """Every Spec §7 limitation is registered in the delivered page (AC-07)."""

        scenario = "lim01"
        doc = self.checkout / DELIVERY_DOC_RELATIVE
        self.expect(
            scenario, doc.is_file(), f"{DELIVERY_DOC_RELATIVE} is not delivered"
        )
        text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
        presence: dict[str, bool] = {}
        for item in REQUIRED_LIMITATION_IDS:
            found = (
                re.search(rf"(?<![\w-]){re.escape(item)}(?![\w-])", text) is not None
            )
            presence[item] = found
            self.expect(scenario, found, f"the delivery page does not register {item}")
        self.observe(scenario, "limitations_present", presence)
        self.observe(
            scenario,
            "limitations_registered",
            sum(1 for item in presence.values() if item),
        )
        phrase_presence = {}
        for label, phrase in REQUIRED_DOC_PHRASES:
            found = phrase in text
            phrase_presence[label] = found
            self.expect(scenario, found, f"the delivery page does not describe {label}")
        self.observe(scenario, "phrases_present", phrase_presence)

        link_presence: dict[str, bool] = {}
        for relative, needles in DELIVERY_DOC_LINKS.items():
            source = self.checkout / relative
            body = source.read_text(encoding="utf-8") if source.is_file() else ""
            found = all(needle in body for needle in needles)
            link_presence[relative] = found
            self.expect(
                scenario,
                found,
                f"{relative} does not link the delivery page",
            )
        self.observe(scenario, "entry_links", link_presence)

    def scenario_role01(self) -> None:
        scenario = "role01"
        probe = self.oneoff("python", "/app/scripts/role_probe.py")
        report = self.parse_json(probe.stdout)
        self.expect(scenario, probe.returncode == 0, "the role probe failed")
        self.expect(scenario, report is not None, "the role probe printed no report")
        if not report:
            return
        counts = report.get("counts", {})
        self.observe(scenario, "role_probe", report)
        self.expect(
            scenario,
            counts.get("unauthorized_writes_succeeded") == 0,
            "an unauthorized write succeeded",
        )
        self.expect(
            scenario,
            counts.get("denials_are_insufficient_privilege") is True,
            "a refusal was not an insufficient-privilege refusal",
        )
        self.expect(
            scenario,
            counts.get("procurement_facts_unchanged") is True,
            "the procurement fact checksum changed",
        )
        self.expect(
            scenario,
            report.get("positive_control", {}).get("create_table_denied") is False,
            "the migrator positive control could not create a table",
        )
        self.expect(
            scenario,
            report.get("roles", [{}])[0].get("superuser") is False,
            "the runtime role is a superuser",
        )
        # The accepted security layer locks two registry tables on every human action;
        # those locking reads must work under least privilege, and the delivered guard
        # must keep the same privilege unusable for real writes.
        self.expect(
            scenario,
            counts.get("lock_reads_ok") == counts.get("lock_reads_expected") == 2,
            "the accepted locking reads do not work for the runtime role",
        )
        self.expect(
            scenario,
            counts.get("registry_guards") == 2,
            "the authorization-registry guard is not installed",
        )
        self.expect(
            scenario,
            counts.get("escalation_effective") is False,
            "a privilege escalation attempt took effect",
        )

    def scenario_teardown(self) -> None:
        scenario = "teardown"
        down = self.compose("down", "-v", "--remove-orphans", timeout=600)
        self.expect(scenario, down.returncode == 0, "the teardown command failed")
        # Residuals are judged by the orchestration label, so only resources this
        # project actually created can fail the check; anything else that happens to
        # carry a similar name is reported, not blamed.
        containers = self.docker(
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={self.prefix}",
            "--format",
            "{{.Names}}",
        ).stdout.split()
        volumes = self.docker(
            "volume",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={self.prefix}",
            "--format",
            "{{.Name}}",
        ).stdout.split()
        networks = self.docker(
            "network",
            "ls",
            "--filter",
            f"label=com.docker.compose.project={self.prefix}",
            "--format",
            "{{.Name}}",
        ).stdout.split()
        self.observe(
            scenario,
            "residuals",
            {"containers": containers, "volumes": volumes, "networks": networks},
        )
        self.expect(
            scenario,
            not containers and not volumes and not networks,
            "teardown left orchestration resources behind",
        )
        self.observe(
            scenario,
            "prefixed_containers",
            self.docker(
                "ps", "-a", "--filter", f"name={self.prefix}", "--format", "{{.Names}}"
            ).stdout.split(),
        )
        self.observe(
            scenario,
            "credential_directory_kept",
            {
                "path": str(self.credential_dir),
                "exists": self.credential_dir.exists(),
                "mode": oct(self.credential_dir.stat().st_mode & 0o777)
                if self.credential_dir.exists()
                else None,
            },
        )
        leftovers = (
            sorted(
                str(item.relative_to(self.checkout / "project"))
                for item in (self.checkout / "project/.local").rglob("*")
            )
            if (self.checkout / "project/.local").exists()
            else []
        )
        self.observe(scenario, "host_local_leftovers", leftovers)
        self.expect(
            scenario,
            all(item.startswith(".local/delivery") for item in leftovers),
            "the delivery left host artifacts outside the declared credential path",
        )
        before = (
            self.results.get("preflight", {})
            .get("observations", {})
            .get("docker_state_before", {})
        )
        after = {
            "containers": self.docker(
                "ps", "-a", "--format", "{{.Names}}"
            ).stdout.split(),
            "volumes": self.docker(
                "volume", "ls", "--format", "{{.Name}}"
            ).stdout.split(),
            "networks": self.docker(
                "network", "ls", "--format", "{{.Name}}"
            ).stdout.split(),
        }
        self.observe(scenario, "docker_state_after", after)
        if before:
            # Other actors may legitimately create containers during the run; what must
            # never happen is that this check removes something it did not create.
            for kind in ("containers", "volumes", "networks"):
                missing = sorted(set(before.get(kind, [])) - set(after.get(kind, [])))
                added = sorted(set(after.get(kind, [])) - set(before.get(kind, [])))
                self.observe(scenario, f"{kind}_removed_by_check", missing)
                if added:
                    self.note(
                        scenario,
                        f"{len(added)} {kind} appeared during the check (not created by "
                        "this project)",
                    )
                self.expect(
                    scenario,
                    not missing,
                    f"the check removed {kind} it did not create: {missing}",
                )

    # ------------------------------------------------------------------- run

    def run_all(self) -> dict:
        handlers = {
            "preflight": self.scenario_preflight,
            "lim01": self.scenario_lim01,
            "env01": self.scenario_env01,
            "env02": self.scenario_env02,
            "boot01": self.scenario_boot01,
            "gold": self.scenario_gold,
            "secs": self.scenario_secs,
            "recs": self.scenario_recs,
            "evd": self.scenario_evd,
            "boot02": self.scenario_boot02,
            "role01": self.scenario_role01,
            "teardown": self.scenario_teardown,
        }
        try:
            for name in self.scenarios:
                if name == "teardown" and not self.do_teardown:
                    continue
                started = time.time()
                try:
                    handlers[name]()
                except Exception as error:  # noqa: BLE001 - recorded, never silent
                    self.failures.append(
                        f"{name}: {type(error).__name__}: {redact(str(error))}"
                    )
                row = self.row(name)
                row["duration_s"] = round(time.time() - started, 2)
                row["status"] = row_status(row)
                print(
                    f"[{name}] {row['status']} ({row['duration_s']}s)",
                    flush=True,
                )
        finally:
            if self.do_teardown and "teardown" not in self.results:
                try:
                    self.scenario_teardown()
                except Exception as error:  # noqa: BLE001
                    self.failures.append(f"teardown: {type(error).__name__}")
                self.row("teardown")["status"] = row_status(self.results["teardown"])

        rows = [self.results[name] for name in SCENARIOS if name in self.results]
        return build_report(
            rows,
            environment=self.environment,
            limitations=rows_limitations(rows),
            duration_s=time.time() - self.started,
        )

    def write_outputs(self, report: dict) -> Path:
        """Write the raw rows and the derived report; they are verified separately."""

        self.output.mkdir(parents=True, exist_ok=True)
        os.chmod(self.output, 0o700)
        rows = [self.results[name] for name in SCENARIOS if name in self.results]
        observations = {
            "check": CHECK_NAME,
            "stage": STAGE,
            "checkout": str(self.checkout),
            "prefix": self.prefix,
            "environment": self.environment,
            "rows": rows,
            "commands": self.commands,
            "generated_at": now(),
        }
        observations_path = self.output / OBSERVATIONS_FILENAME
        observations_path.write_text(
            json.dumps(observations, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.chmod(observations_path, 0o600)
        target = self.output / REPORT_FILENAME
        target.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(target, 0o600)

        lines = [
            f"v104cs delivery check — stage {STAGE} — {report['status']}",
            f"checkout: {report['environment'].get('checkout')}",
            f"prefix:   {report['environment'].get('prefix')}",
            f"total:    {report['total_duration_s']}s",
            "",
        ]
        for row in rows:
            lines.append(
                f"- {row.get('scenario')}: {row.get('status')} ({row.get('duration_s')}s)"
            )
            for note in row.get("notes", []):
                lines.append(f"    note: {note}")
        lines.append("")
        lines.append(
            f"counters: {json.dumps(report['counters'], ensure_ascii=False, sort_keys=True)}"
        )
        if report["failures"]:
            lines.append("")
            lines.append("failures:")
            lines.extend(f"  - {item}" for item in report["failures"])
        summary = self.output / REPORT_TEXT_FILENAME
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(summary, 0o600)
        return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        default=str(Path(__file__).resolve().parents[2]),
        help="checkout to run against (defaults to this repository)",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--api-port", type=int, default=None)
    parser.add_argument("--web-port", type=int, default=None)
    parser.add_argument(
        "--output",
        help="private report directory (relative and absolute spellings are "
        "equivalent: it is normalized to an absolute host path internally)",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="comma-separated subset of: " + ", ".join(SCENARIOS),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the orchestration running instead of tearing it down",
    )
    parser.add_argument(
        "--verify-report",
        help="recompute an existing report from its raw rows, without Docker or network",
    )
    arguments = parser.parse_args(argv)

    if arguments.verify_report:
        try:
            result = verify_report(Path(arguments.verify_report))
        except (OSError, ValueError, RuntimeError) as error:
            print(f"verify-error: {redact(str(error))}")
            return 1
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["verified"] else 1

    if not arguments.output:
        parser.error("--output is required unless --verify-report is used")

    checkout = Path(arguments.checkout)
    if not (checkout / "project/docker-compose.yml").is_file():
        parser.error(f"{checkout} does not look like the delivery checkout")

    scenarios = tuple(
        item.strip() for item in arguments.scenarios.split(",") if item.strip()
    )
    unknown = [item for item in scenarios if item not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {unknown}")

    delivery = Delivery(
        checkout=checkout,
        prefix=arguments.prefix,
        output=Path(arguments.output),
        scenarios=scenarios,
        teardown=not arguments.keep,
        api_port=arguments.api_port,
        web_port=arguments.web_port,
    )
    report = delivery.run_all()
    target = delivery.write_outputs(report)
    print(f"report: {target}")
    print(f"status: {report['status']}")
    for failure in report["failures"]:
        print(f"  - {failure}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
