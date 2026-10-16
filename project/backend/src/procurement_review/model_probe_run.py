"""Run one registered C candidate; persist every attempt, never raw responses."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import SecretStr

from .chat_transport_v2 import ChatTransportV2, EndpointConfigV2
from .eval_attempt_ledger import AttemptLedger, LedgerError, _atomic_write
from .gold_v3_contracts import canonical_json_bytes, payload_sha256
from .model_attempt_v2 import (
    BatchAuthority,
    BudgetProof,
    BranchAttemptDriver,
    configuration_hash,
)
from .model_probe_v2 import dev_requests, registration, replay

BACKEND = Path(__file__).resolve().parents[2]
LIVE_ROOT = BACKEND / ".local/v1-07c/live"


def write(path: Path, value: dict):
    _atomic_write(path, canonical_json_bytes(value))


def stage_observations(root: Path) -> tuple[int, list[dict]]:
    count, receipts = 0, []
    for path in root.rglob("ledger.sqlite3"):
        ledger = AttemptLedger(path)
        count += ledger.manifest().attempt_count
        with sqlite3.connect(path) as connection:
            for (body,) in connection.execute(
                "SELECT receipt_json FROM attempts WHERE receipt_json IS NOT NULL"
            ):
                receipts.append(json.loads(body))
    return count, sorted(receipts, key=lambda r: r.get("completed_at") or "")


def require_stage_capacity(root: Path, additional: int = 2):
    if (root / "stopped.json").exists():
        raise LedgerError(
            "stage stopped; review its fixed event code before continuing"
        )
    count, receipts = stage_observations(root)
    for receipt in receipts:
        if {receipt.get("error_code"), receipt.get("underlying_error_code")} & {
            "model_identity_drift",
            "security_boundary",
            "tool_call_forbidden",
            "credential_in_request",
        }:
            raise LedgerError("stage safety stop recorded in durable receipt")
    if count + additional > 240:
        raise LedgerError("C stage physical request ceiling")
    consecutive = 0
    for receipt in reversed(receipts):
        if (receipt.get("underlying_error_code") or receipt.get("error_code")) not in {
            "timeout",
            "transport",
        }:
            break
        consecutive += 1
    if consecutive >= 12:
        raise LedgerError("twelve consecutive transport failures")


def credential() -> SecretStr:
    values = {}
    for line in (BACKEND / ".env.local").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().removeprefix("export ")] = value.strip().strip("\"'")
    key = values.get("MODEL_API_KEY")
    if not key:
        raise LedgerError("local model credential unavailable")
    return SecretStr(key)


async def execute(
    path: Path, candidate_name: str, *, root: Path = LIVE_ROOT, transport_override=None
):
    registered = json.loads(path.read_text())
    expected = registration(
        no_balance_limit_authorized=True,
        extended_deadline=registered.get("probe_revision") in {2, 3, 4, 5, 6, 7},
        non_reasoning_modes=registered.get("probe_revision") in {3, 4, 5, 6, 7},
        focused=registered.get("probe_revision") == 4,
        final_text=registered.get("probe_revision") == 5,
        schema_zero=registered.get("probe_revision") in {6, 7},
        empty_source_revision=registered.get("probe_revision") == 7,
    )
    if (
        registered != expected
        or registered["execution_status"] != "authorized_no_balance_limit"
    ):
        raise LedgerError("registration stale or lacks explicit authorization")
    candidate = next(
        (c for c in registered["candidates"] if c["candidate"] == candidate_name), None
    )
    if candidate is None:
        raise LedgerError("candidate not registered")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = payload_sha256(registered)
    directory = root / digest
    directory.mkdir(exist_ok=True, mode=0o700)
    with (root / "stage.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidate_dir = directory / candidate_name.replace(":", "-")
        candidate_dir.mkdir(exist_ok=True, mode=0o700)
        result_path = candidate_dir / "result.json"
        if result_path.exists():
            return replay(candidate_dir / "ledger-export.json")
        if candidate["dependency"]:
            dependency = (
                directory / candidate["dependency"].replace(":", "-") / "result.json"
            )
            if not dependency.exists() or not json.loads(dependency.read_text()).get(
                "capability_supported"
                if candidate["dependency"].startswith("minimum:")
                else "screen_passed",
                False,
            ):
                raise LedgerError("candidate dependency has not passed")
        require_stage_capacity(root)
        execution_path = candidate_dir / "execution.json"
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
        else:
            execution = {
                "registration_sha256": digest,
                "expires_at": (
                    datetime.now(UTC)
                    + timedelta(seconds=registered["batch_deadline_seconds"])
                ).isoformat(),
                "deadlines": {},
            }
            write(execution_path, execution)
        requests = {(r.case_id, r.worker): r for r in dev_requests()}
        selected = [
            (task, requests[(task["case_id"], task["worker"])])
            for task in candidate["tasks"]
        ]
        proof = BudgetProof(
            evidence_sha256=payload_sha256(
                {
                    "quote": "不设余额上限，这个问题不再应该成为阻断项。",
                    "stage": "V1-07C",
                }
            ),
            cumulative_request_upper=7639,
            cumulative_input_upper=None,
            cumulative_output_upper=None,
            historical_unknown_input_reserve=None,
            historical_unknown_output_reserve=None,
            limit_policy="user_no_balance_limit",
            authorization_basis="user-explicit-no-balance-limit",
            allocated_requests=candidate["physical_request_limit"],
            allocated_input=None,
            allocated_output=None,
        )
        authority = BatchAuthority(
            authorization_id="V1-07-user-no-balance-limit",
            scope_ids=frozenset(r.scope_id for _, r in selected),
            expires_at=datetime.fromisoformat(execution["expires_at"]),
            proof=proof,
            before_attempt=lambda: require_stage_capacity(root, additional=1),
        )
        config = EndpointConfigV2.model_validate(candidate["configuration"])
        transport = ChatTransportV2(
            config,
            SecretStr("offline-only")
            if transport_override is not None
            else credential(),
            transport=transport_override,
        )
        ledger_path = candidate_dir / "ledger.sqlite3"
        if ledger_path.exists():
            ledger = AttemptLedger(ledger_path)
            ledger.recover()
        else:
            ledger = AttemptLedger.create(
                ledger_path,
                run_id="v1-07-c-" + candidate_name.replace(":", "-").replace("_", "-"),
                gold_manifest_sha256=registered["gold_manifest_sha256"],
                gold_set_version=registered["gold_set_version"],
                configuration_sha256=configuration_hash(transport, authority),
                max_requests=proof.allocated_requests,
                max_input_tokens=None,
                max_output_tokens=None,
                stage_id="V1-07C",
            )
        driver = BranchAttemptDriver(ledger, transport, authority)
        for task, request in selected:
            key = task["task_key"]
            if key not in execution["deadlines"]:
                execution["deadlines"][key] = (
                    datetime.now(UTC)
                    + timedelta(seconds=registered["logical_deadline_seconds"])
                ).isoformat()
                write(execution_path, execution)
            require_stage_capacity(root)
            try:
                await driver.run(
                    key,
                    request,
                    deadline=datetime.fromisoformat(execution["deadlines"][key]),
                    reserve_input=task["planning_input_utf8_bytes"],
                    reserve_output=7290,
                    max_attempts=candidate.get("max_attempts_per_branch", 2),
                )
            finally:
                ledger.export(candidate_dir / "ledger-export.json")
            _, receipts = stage_observations(root)
            last = receipts[-1] if receipts else {}
            code = last.get("underlying_error_code") or last.get("error_code")
            print(
                json.dumps(
                    {
                        "task": key,
                        "outcome": last.get("outcome"),
                        "error_code": code,
                        "input": last.get("input_tokens"),
                        "output": last.get("output_tokens"),
                        "unknown": last.get("usage_unknown"),
                    }
                ),
                flush=True,
            )
            if code in {
                "model_identity_drift",
                "security_boundary",
                "tool_call_forbidden",
            }:
                write(
                    root / "stopped.json",
                    {"event_code": code, "candidate": candidate_name},
                )
                break
            if code in {"unsupported_request", "authentication"}:
                break
        result = replay(candidate_dir / "ledger-export.json")
        result["replay_network_calls"] = result.pop("network_calls")
        result["actual_attempts"] = result["attempts"]
        result["capability_supported"] = result["final_valid"] == len(selected)
        result["registered_first_attempts"] = len(selected)
        result["screen_passed"] = result["first_valid"] / len(selected) >= 0.95
        result["semantic_adjudication"] = "pending_independent_source_review"
        result["quality_acceptance"] = False
        write(result_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(execute(args.registration, args.candidate))))
    except Exception as exc:
        print(json.dumps({"status": "stopped", "exception_type": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
