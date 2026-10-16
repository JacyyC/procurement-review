"""Finite DEV repair probes at the already selected settings; no ablations.

Every send uses the F inventory, authority and B ledger. Stop new Cases on the
first failed branch; active receipts settle normally. This is diagnostic evidence
only and cannot establish full DEV, validation or Stage acceptance.
"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import fcntl
import json
from pathlib import Path

from pydantic import SecretStr

from .business_v3_documents import document_request_v3
from .business_v3_security import ProtectedOutputV3
from .business_v3_support import SourceCatalogV3
from .eval_attempt_ledger import AttemptLedger, LedgerError
from .eval_f_budget import FStageBudget
from .eval_f_context import context_for, located_sources, stable_write
from .eval_f_contracts import FBudgetProof, FEndpointConfig
from .eval_f_registration import LIVE_ROOT, configuration, implementation_fingerprints
from .eval_f_repair_cases import diagnostic_cases
from .eval_f_run import FAuthority
from .eval_f_transport import PreciseAttemptDriver, PreciseChatTransport
from .gold_v3_contracts import payload_sha256
from .model_attempt_v2 import configuration_hash
from .model_probe_run import credential


def preregistration():
    cases = diagnostic_cases()
    return {
        "contract": "v1-07f-focused-dev.v1",
        "stage": "V1-07F",
        "revision": 4,
        "selection_basis": "user fixed baseline/disabled/Case concurrency 2 after completed comparisons",
        "configuration": configuration().model_dump(mode="json"),
        "implementation": implementation_fingerprints(),
        "case_inputs": [c.model_input.model_dump(mode="json") for c in cases],
        "case_ids": [c.case_id for c in cases],
        "concurrency": 2,
        "max_requests": len(cases) * 2,
        "max_attempts": 1,
        "per_case_deadline_seconds": 120,
        "batch_deadline_seconds": 1800,
        "stop_after_first_failed_branch": True,
        "purpose": "verify pure amount and separate currency under the fixed prompt plus independent untrusted English input; no quality or Stage acceptance",
    }


async def run(path, *, transport_override=None, offline_root=None):
    if offline_root is not None and transport_override is None:
        raise LedgerError("real F repair requires fixed live root")
    registered = json.loads(Path(path).read_text())
    if registered != preregistration():
        raise LedgerError("repair inputs changed; new reviewed registration required")
    digest = payload_sha256(registered)
    root = Path(offline_root) if offline_root is not None else LIVE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stage.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if transport_override is None:
            from .model_probe_run import stage_observations, LIVE_ROOT as C_ROOT

            if stage_observations(C_ROOT)[0] != 232:
                raise LedgerError("prior C inventory changed; reconciliation required")
        budget = FStageBudget(root)
        directory = root / digest / "focused-dev"
        directory.mkdir(parents=True, exist_ok=True)
        stable_write(directory / "registration.json", registered)
        if (directory / "result.json").exists():
            return json.loads((directory / "result.json").read_text())
        config = FEndpointConfig.model_validate(registered["configuration"])
        budget.require(configuration=config.model_dump(mode="json"))
        execution_path = directory / "execution.json"
        if not execution_path.exists():
            stable_write(
                execution_path,
                dict(
                    generation=1,
                    cancelled=False,
                    registration_sha256=digest,
                    expires_at=(
                        datetime.now(UTC) + timedelta(seconds=1800)
                    ).isoformat(),
                    cases={},
                    failed=False,
                ),
            )
        execution = json.loads(execution_path.read_text())
        cases = diagnostic_cases()
        authority = FAuthority(
            authorization_id="V1-07F-user-authorized-focused-repair",
            scope_ids=frozenset(context_for(c).scope_id for c in cases),
            expires_at=datetime.fromisoformat(execution["expires_at"]),
            proof=FBudgetProof(
                evidence_sha256=digest,
                cumulative_request_upper=7871,
                cumulative_input_upper=None,
                cumulative_output_upper=None,
                historical_unknown_input_reserve=None,
                historical_unknown_output_reserve=None,
                allocated_requests=registered["max_requests"],
                allocated_input=None,
                allocated_output=None,
                limit_policy="user_no_balance_limit",
                authorization_basis="user-explicit-no-balance-limit",
            ),
            before_attempt=lambda: budget.require(
                configuration=config.model_dump(mode="json")
            ),
            execution_path=execution_path,
            generation=execution["generation"],
        )
        transport = PreciseChatTransport(
            config,
            SecretStr("offline-only") if transport_override else credential(),
            protection=ProtectedOutputV3(),
            transport=transport_override,
            catalog=SourceCatalogV3(
                tuple(s for c in cases for s in located_sources(c))
            ),
        )
        ledger_path = directory / "ledger.sqlite3"
        if ledger_path.exists():
            ledger = AttemptLedger(ledger_path)
            ledger.recover()
        else:
            ledger = AttemptLedger.create(
                ledger_path,
                run_id="v1-07-f-focused-dev",
                gold_manifest_sha256=payload_sha256(registered["case_inputs"]),
                gold_set_version="gold-set.v3.2",
                stage_id="V1-07F",
                configuration_sha256=configuration_hash(transport, authority),
                max_requests=registered["max_requests"],
                max_input_tokens=None,
                max_output_tokens=None,
            )
            budget.add(ledger_path)
        driver = PreciseAttemptDriver(ledger, transport, authority)
        semaphore = asyncio.Semaphore(2)

        async def case_probe(case):
            async with semaphore:
                state = json.loads(execution_path.read_text())
                if state["failed"]:
                    return
                if case.case_id not in state["cases"]:
                    state["cases"][case.case_id] = (
                        datetime.now(UTC) + timedelta(seconds=120)
                    ).isoformat()
                    stable_write(execution_path, state)
                deadline = datetime.fromisoformat(state["cases"][case.case_id])
                for worker in ("policy", "contract"):
                    if json.loads(execution_path.read_text())["failed"]:
                        return
                    request = document_request_v3(
                        context_for(case),
                        case_id=case.case_id,
                        worker=worker,
                        question=case.model_input.question,
                        sources=tuple(
                            s.source
                            for s in located_sources(case)
                            if s.source.worker == worker
                        ),
                    )
                    try:
                        result = await driver.run(
                            f"{case.case_id}.repair.{worker}",
                            request,
                            deadline=deadline,
                            reserve_input=len(request.model_dump_json().encode()),
                            reserve_output=12000,
                            max_attempts=1,
                        )
                        stable_write(
                            directory / f"{case.case_id}.{worker}.json",
                            result.model_dump(mode="json"),
                        )
                        if result.pipeline_failure:
                            state = json.loads(execution_path.read_text())
                            state["failed"] = True
                            stable_write(execution_path, state)
                    finally:
                        ledger.export(directory / "ledger-export.json")

        try:
            await asyncio.gather(*(case_probe(c) for c in cases))
        finally:
            ledger.export(directory / "ledger-export.json")
        result = {
            "registration": digest,
            "failed": json.loads(execution_path.read_text())["failed"],
            "ledger": ledger.audit_summary(),
            "stage_usage": budget.summary(),
            "stage_acceptance": False,
            "validation_calls": 0,
        }
        stable_write(directory / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.path.exists():
            parser.error("registration must be new")
        stable_write(args.path, preregistration())
        result = {"registration": payload_sha256(preregistration()), "network_calls": 0}
    else:
        result = asyncio.run(run(args.path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
