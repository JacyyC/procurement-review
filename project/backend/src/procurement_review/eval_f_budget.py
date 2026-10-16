"""Durable F inventory under one fixed root; config changes never reset counts."""

import json
import sqlite3
from pathlib import Path

from .eval_attempt_ledger import AttemptLedger, LedgerError
from .eval_f_context import stable_write
from .gold_v3_contracts import payload_sha256


class FStageBudget:
    def __init__(self, root, *, prior_c=232, d_e=0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "inventory.json"
        baseline = {"prior_c": prior_c, "d_e": d_e, "ledgers": []}
        if not self.path.exists():
            if list(self.root.rglob("ledger.sqlite3")):
                raise LedgerError("unregistered F ledgers require reconciliation")
            stable_write(self.path, baseline)
        self.data = json.loads(self.path.read_text())
        if self.data["prior_c"] != prior_c or self.data["d_e"] != d_e:
            raise LedgerError("stage baseline cannot be reset")

    def add(self, path):
        relative = str(Path(path).resolve().relative_to(self.root.resolve()))
        if relative not in self.data["ledgers"]:
            self.data["ledgers"].append(relative)
            stable_write(self.path, self.data)

    def observations(self):
        found = {
            str(p.relative_to(self.root)) for p in self.root.rglob("ledger.sqlite3")
        }
        if found != set(self.data["ledgers"]):
            raise LedgerError("ledger inventory changed; fail closed")
        count, receipts = 0, []
        for relative in self.data["ledgers"]:
            path = self.root / relative
            ledger = AttemptLedger(path)
            if ledger.manifest().stage_id != "V1-07F":
                raise LedgerError("foreign stage cannot masquerade as F")
            count += ledger.manifest().attempt_count
            with sqlite3.connect(path) as conn:
                receipts.extend(
                    json.loads(row[0])
                    for row in conn.execute(
                        "SELECT receipt_json FROM attempts WHERE receipt_json IS NOT NULL"
                    )
                )
        return count, sorted(receipts, key=lambda r: r.get("completed_at") or "")

    def require(self, *, diagnostic=True, configuration=None):
        count, receipts = self.observations()
        if (self.root / "stopped.json").exists():
            raise LedgerError("durable stage stop requires reviewed disposition")
        permit = self.root / "transport-recovery.json"
        reviewed = json.loads(permit.read_text()) if permit.exists() else None
        if reviewed is not None:
            if reviewed.get("contract") != "v1-07f-transport-recovery.v1":
                raise LedgerError("invalid reviewed transport recovery")
            if reviewed.get("status") == "released":
                matched = [
                    r
                    for r in receipts
                    if payload_sha256(r) == reviewed.get("success_receipt_sha256")
                ]
                if (
                    len(matched) != 1
                    or matched[0].get("outcome") != "succeeded"
                    or matched[0].get("error_code") is not None
                ):
                    raise LedgerError("recovery has no actual successful receipt")
            elif reviewed.get("status") != "probe_pending" or count != reviewed.get(
                "at_request_count"
            ):
                raise LedgerError(
                    "recovery probe consumed; reviewed successful receipt required"
                )
        # 2026-09-20: explicit user authorization removed cumulative request
        # ceilings. The inventory, branch allowance, immutable batch plan and
        # all service/safety/lifecycle stops still apply; history is not reset.
        for r in receipts:
            if {r.get("error_code"), r.get("underlying_error_code")} & {
                "model_identity_drift",
                "security_boundary",
                "tool_call_forbidden",
                "credential_in_request",
                "authentication",
                "unsupported_request",
            }:
                raise LedgerError("durable non-retryable stop")
        failures = 0
        for r in reversed(receipts):
            if (r.get("underlying_error_code") or r.get("error_code")) not in {
                "timeout",
                "transport",
            }:
                break
            failures += 1
        if failures >= 12:
            if reviewed is None or reviewed.get("status") != "probe_pending":
                raise LedgerError("twelve consecutive transport failures")
            # One reviewed probe can establish a new observed success. It never
            # erases history, permits retries, or bypasses another stop category.
            if (
                reviewed.get("contract") != "v1-07f-transport-recovery.v1"
                or reviewed.get("max_requests") != 1
                or reviewed.get("at_request_count") != count
                or reviewed.get("last_twelve_sha256") != payload_sha256(receipts[-12:])
                or configuration is None
                or reviewed.get("configuration_sha256") != payload_sha256(configuration)
                or not reviewed.get("review_evidence_sha256")
            ):
                raise LedgerError("reviewed transport recovery permit does not apply")

    def summary(self):
        count, receipts = self.observations()
        return dict(
            reserved_attempts=count,
            receipts=len(receipts),
            known_input=sum(r.get("input_tokens") or 0 for r in receipts),
            known_output=sum(r.get("output_tokens") or 0 for r in receipts),
            unknown=sum(bool(r["usage_unknown"]) for r in receipts),
            unreturned=count - len(receipts),
            d_e_f_remaining=None,
            g_remaining=None,
            item_remaining=None,
            request_limit_policy="user_no_cumulative_request_limit_20260920",
        )
