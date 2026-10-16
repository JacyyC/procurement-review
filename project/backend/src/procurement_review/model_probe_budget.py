"""Read-only reconciliation of the pre-C local model cohorts; never sends requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(root: Path) -> dict:
    hashes = {}

    def read(relative):
        path = root / relative
        raw = path.read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    v4 = "model-accuracy-v4-flash-return-20260919/"
    baseline = read(v4 + "ledger-before.json")
    read(v4 + "resume-registration.json")
    path = root / v4 / "call-receipts.jsonl"
    hashes[v4 + "call-receipts.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipts = [json.loads(line)["entry"] for line in path.read_text().splitlines()]
    usage_rows = []
    seen = set()

    def entry(value):
        identity = value.get("request_id_hash")
        if identity:
            if identity in seen:
                raise ValueError("duplicate physical request evidence")
            seen.add(identity)
        if value.get("retry_count", 0) != 0:
            raise ValueError("retry usage requires explicit reconciliation")
        usage_rows.append((value.get("input_tokens"), value.get("output_tokens")))

    def usage(value):
        tokens = value.get("usage") or {}
        usage_rows.append(
            (tokens.get("prompt_tokens"), tokens.get("completion_tokens"))
        )

    for row in receipts:
        entry(row)
    usage(read(v4 + "connectivity.json"))
    entry(read(v4 + "business-probe.json")["entry"])
    initial_count = len(usage_rows)
    initial_input = sum(x or 0 for x, y in usage_rows)
    initial_output = sum(y or 0 for x, y in usage_rows)
    old = "model-accuracy-20260919/"
    usage(read(old + "minimal-generation.json"))
    entry(read(old + "real-preflight.json")["entry"])
    ds = "model-accuracy-deepseek-v4-1-20260919/"
    usage(read(ds + "connectivity.json"))
    for row in read(ds + "diagnosis.json"):
        if row["kind"] == "generation":
            usage(row)
    glm = "model-accuracy-glm-5-3-flash-20260919/"
    usage(read(glm + "connectivity.json"))
    entry(read(glm + "business-probe.json")["entry"])
    for row in read(glm + "diagnostic.json"):
        entry(row["entry"])
    entry(read(glm + "budget-probe-8192.json")["entry"])
    for name in ("nonthinking-probe.json", "nonthinking-error-detail.json"):
        for row in read(glm + name):
            entry(row["entry"])
    inp = baseline["manifest_input_tokens"] + sum(x or 0 for x, y in usage_rows)
    out = baseline["manifest_output_tokens"] + sum(y or 0 for x, y in usage_rows)
    optimistic = (
        baseline["decision_max_output_tokens"]
        - out
        - baseline["unaccounted_output_tokens_high"]
    )
    return {
        "schema_version": "v1-07c-budget-audit.v1",
        "network_calls": 0,
        "source_sha256": hashes,
        "cumulative_request_lower": baseline["manifest_requests"]
        + baseline["unaccounted_requests_low"]
        + len(usage_rows),
        "cumulative_request_upper": baseline["manifest_requests"]
        + baseline["unaccounted_requests_high"]
        + len(usage_rows)
        + 4,
        "known_input_lower": inp,
        "known_output_lower": out,
        "additional_requests_vs_A": len(usage_rows) - initial_count,
        "additional_known_input_vs_A": sum(x or 0 for x, y in usage_rows)
        - initial_input,
        "additional_known_output_vs_A": sum(y or 0 for x, y in usage_rows)
        - initial_output,
        "no_usage_receipts": sum(x is None or y is None for x, y in usage_rows),
        "interrupted_inflight_unknown_upper": 4,
        "historical_unledgered_requests": [
            baseline["unaccounted_requests_low"],
            baseline["unaccounted_requests_high"],
        ],
        "historical_unledgered_output": [
            baseline["unaccounted_output_tokens_low"],
            baseline["unaccounted_output_tokens_high"],
        ],
        "optimistic_output_remainder_upper": optimistic,
        "provable_executable_requests": 0,
        "historical_unknown_input_reserve": None,
        "historical_unknown_output_reserve": None,
        "H_first_output_p50_deficit": 2800 * 3033 - optimistic,
        "C_234_output_p50_estimate": 234 * 3033,
        "C_234_output_p95_estimate": 234 * 7290,
        "authorization_limit": "receipt audit cannot establish new spending authorization",
        "limits": {"requests": 25000, "input": 250000000, "output": 24000000},
        "limitation": "local receipts are usage lower bounds, not provider billing or hard caps; HTTP 400/404 unknown is not zero",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".local"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ".local" not in args.output.resolve().parts or args.output.exists():
        parser.error("output must be a new .local file")
    result = audit(args.root)
    from .eval_attempt_ledger import _atomic_write
    from .gold_v3_contracts import canonical_json_bytes

    _atomic_write(args.output, canonical_json_bytes(result))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "source_sha256"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
