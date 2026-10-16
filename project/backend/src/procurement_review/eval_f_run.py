"""Explicit F-only Oracle Case runner. Independent batch authority, not product PG.

prepare/replay never access credentials or the network. Real sends use the fixed
live root and durable stage inventory; each physical attempt is B-ledgered.
"""

import argparse
import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import fcntl
import json
from pathlib import Path
import time

from pydantic import SecretStr

from .business_v3_documents import document_request_v3, DocumentValuesV3
from .business_v3_evidence import (
    parse_evidence_v3,
    ground_document_v3,
    GroundedDocumentV3,
)
from .business_v3_procurement import review_procurement_v3
from .business_v3_review import finalize_review_v3, PreciseDecisionV3
from .business_v3_types import BusinessDecisionV3, finding
from .business_v3_security import ProtectedOutputV3
from .business_v3_support import SourceCatalogV3
from .eval_attempt_ledger import AttemptLedger, LedgerError
from .eval_f_budget import FStageBudget
from .eval_f_lineage import diagnostic_directory, ARMS
from .eval_f_context import (
    load_cases,
    context_for,
    located_sources,
    oracle_procurement,
    stable_write,
)
from .eval_f_contracts import FBudgetProof, FEndpointConfig, FCaseProjection, CaseTiming
from .eval_f_registration import (
    LIVE_ROOT,
    registration,
    prepare,
    implementation_fingerprints,
    fixed_registration,
)
from .eval_f_transport import PreciseAttemptDriver, PreciseChatTransport
from .gold_v3_contracts import payload_sha256
from .model_attempt_v2 import (
    BatchAuthority,
    AdmissionDenied,
    configuration_hash,
    _history,
)
from .model_probe_run import credential


class FAuthority(BatchAuthority):
    def __init__(self, *args, execution_path, generation, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_path = execution_path
        self.generation = generation
        self.registration_sha256 = json.loads(execution_path.read_text()).get(
            "registration_sha256"
        )
        self.implementation = implementation_fingerprints()

    @contextmanager
    def enter(self, request, deadline):
        with super().enter(request, deadline):
            if reason := self.stopped_reason(request.scope_id, deadline):
                raise AdmissionDenied(reason)
            yield

    def stopped_reason(self, scope_id, deadline):
        state = json.loads(self.execution_path.read_text())
        if state.get("cancelled") or (self.execution_path.parent / "cancel").exists():
            return "cancelled"
        if state["generation"] != self.generation:
            return "execution_generation_changed"
        if (
            state.get("registration_sha256") != self.registration_sha256
            or implementation_fingerprints() != self.implementation
        ):
            return "configuration_changed"
        if not self._active or scope_id not in self.scope_ids:
            return "authorization_revoked"
        if self.now() >= min(deadline, self.expires_at):
            return "deadline"
        return None


def stopped_decision(reason):
    """Control-only terminal projection; no context is adopted after the gate closes."""
    return PreciseDecisionV3(
        business=BusinessDecisionV3(
            recommendation="insufficient_information",
            findings=tuple(
                finding(f"O-{i:02}", "unknown", reason, missing=("admitted_result",))
                for i in range(1, 13)
            ),
            violations=(),
            unknown_obligations=tuple(f"O-{i:02}" for i in range(1, 13)),
            pipeline_failures=(reason,),
            security_events=(),
            conflict_pairs=(),
        ),
        conflicts=(),
        security=(),
        spans=(),
        support_failures=(),
    )


def failed_document(worker, reason):
    return GroundedDocumentV3(DocumentValuesV3(worker, {}, (), (reason,)), (), ())


async def execute(path, candidate_name, *, transport_override=None, offline_root=None):
    if offline_root is not None and transport_override is None:
        raise LedgerError("real F execution requires the fixed live root")
    registered = json.loads(Path(path).read_text())
    expected = (
        fixed_registration(
            registered["versions"][0],
            split=registered.get("confirmation_split", "validation"),
        )
        if registered.get("fixed_selection")
        else registration(
            registered.get("diagnostic_predecessor"), registered["versions"][0]
        )
    )
    if registered != expected:
        raise LedgerError("registration does not match current immutable inputs")
    if candidate_name not in registered["candidates"]:
        raise LedgerError("candidate not registered")
    candidate = registered["candidates"][candidate_name]
    if registered.get("diagnostic_predecessor") and candidate_name in ARMS:
        raise LedgerError(
            "adopted diagnostic is immutable; use its zero-network replay"
        )
    diagnostic = "selected_arm" not in candidate
    root = Path(offline_root) if offline_root is not None else LIVE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root / "stage.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if transport_override is None:
            from .model_probe_run import stage_observations, LIVE_ROOT as C_ROOT

            actual_c, _ = stage_observations(C_ROOT)
            if actual_c != registered["prior"]["c_requests"]:
                raise LedgerError("C usage differs from registered budget evidence")
        budget = FStageBudget(root)
        digest = payload_sha256(registered)
        directory = root / digest / candidate_name
        directory.mkdir(parents=True, exist_ok=True)
        stable_write(directory / "registration.json", registered)
        if (directory / "result.json").exists():
            return replay(directory)
        if not diagnostic and not registered.get("fixed_selection"):
            selection = root / digest / "selection.json"
            if (
                not selection.exists()
                or json.loads(selection.read_text())["arm"] != candidate["selected_arm"]
            ):
                raise LedgerError(
                    "formal evaluation requires a locked DEV-selected arm"
                )
        if candidate["dependency"]:
            dependency = (
                diagnostic_directory(root, registered, candidate["dependency"])
                / "result.json"
            )
            if not dependency.exists():
                raise LedgerError("registered diagnostic dependency is incomplete")
            if not diagnostic and not json.loads(dependency.read_text()).get(
                "diagnostic_passed"
            ):
                raise LedgerError(
                    "formal evaluation requires a quality-qualified dependency"
                )
        budget.require(diagnostic=diagnostic, configuration=candidate["configuration"])
        if candidate_name.startswith("validation-"):
            from .eval_f_validation import claim_validation

            claim_validation(root, registered, candidate_name)
        execution_path = directory / "execution.json"
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
            if execution["registration_sha256"] != digest:
                raise LedgerError("execution configuration changed")
        else:
            execution = dict(
                registration_sha256=digest,
                generation=1,
                cancelled=False,
                expires_at=(
                    datetime.now(UTC)
                    + timedelta(seconds=registered["batch_deadline_seconds"])
                ).isoformat(),
                cases={},
            )
            stable_write(execution_path, execution)
        split = "validation" if candidate_name.startswith("validation-") else "dev"
        cases = {c.case_id: c for c in load_cases(split, registered["versions"][0])}
        selected = [cases[cid] for cid in candidate["case_ids"]]
        proof = FBudgetProof(
            evidence_sha256=digest,
            cumulative_request_upper=7871,
            cumulative_input_upper=None,
            cumulative_output_upper=None,
            historical_unknown_input_reserve=None,
            historical_unknown_output_reserve=None,
            limit_policy="user_no_balance_limit",
            authorization_basis="user-explicit-no-balance-limit",
            allocated_requests=candidate["max_requests"],
            allocated_input=None,
            allocated_output=None,
        )
        authority = FAuthority(
            authorization_id="V1-07F-user-authorized-third-wave",
            scope_ids=frozenset(
                c.model_input.business_facts.scope_id for c in selected
            ),
            expires_at=datetime.fromisoformat(execution["expires_at"]),
            proof=proof,
            before_attempt=lambda: budget.require(
                diagnostic=diagnostic, configuration=candidate["configuration"]
            ),
            execution_path=execution_path,
            generation=execution["generation"],
        )
        config = FEndpointConfig.model_validate(candidate["configuration"])
        protection = ProtectedOutputV3()
        transport = PreciseChatTransport(
            config,
            SecretStr("offline-only") if transport_override else credential(),
            protection=protection,
            transport=transport_override,
            catalog=SourceCatalogV3(
                tuple(s for c in selected for s in located_sources(c))
            ),
        )
        ledger_path = directory / "ledger.sqlite3"
        if ledger_path.exists():
            ledger = AttemptLedger(ledger_path)
            ledger.recover()
        else:
            ledger = AttemptLedger.create(
                ledger_path,
                run_id="v1-07-f-" + candidate_name,
                gold_manifest_sha256=registered["gold_manifest_sha256"],
                gold_set_version=registered["versions"][0],
                configuration_sha256=configuration_hash(transport, authority),
                max_requests=candidate["max_requests"],
                max_input_tokens=None,
                max_output_tokens=None,
                stage_id="V1-07F",
            )
            budget.add(ledger_path)
        driver = PreciseAttemptDriver(ledger, transport, authority)
        semaphore = asyncio.Semaphore(candidate["concurrency"])

        async def run_case(case, seed):
            key = f"{case.case_id}.{seed}"
            output = directory / (key + ".json")
            if output.exists():
                # Committed Case reads do not trigger a new authorization or send.
                return FCaseProjection.model_validate_json(output.read_bytes())
            queued = time.monotonic()
            async with semaphore:
                if registered.get("stop_when_first_pass_unreachable"):
                    from .eval_f_stop import quality_stop

                    if quality_stop(directory, candidate):
                        return None
                current = json.loads(execution_path.read_text())
                if key not in current["cases"]:
                    began = datetime.now(UTC)
                    state = dict(
                        started_at=began.isoformat(),
                        deadline=(began + timedelta(seconds=120)).isoformat(),
                        queue_ms=int((time.monotonic() - queued) * 1000),
                    )
                    if (
                        current["generation"] == authority.generation
                        and not current.get("cancelled")
                        and not (directory / "cancel").exists()
                    ):
                        current["cases"][key] = state
                        stable_write(execution_path, current)
                    else:
                        state["deadline"] = began.isoformat()
                else:
                    state = current["cases"][key]
                deadline = datetime.fromisoformat(state["deadline"])
                scope_id = case.model_input.business_facts.scope_id
                if reason := authority.stopped_reason(scope_id, deadline):
                    result = FCaseProjection(
                        case_id=case.case_id,
                        repeat_seed=seed,
                        configuration_sha256=digest,
                        decision=stopped_decision(reason),
                        branches=(),
                        timing=CaseTiming(
                            queue_ms=state["queue_ms"],
                            context_ms=0,
                            model_ms=0,
                            extra_attempt_ms=0,
                            aggregate_ms=0,
                            total_ms=state["queue_ms"],
                        ),
                    )
                    stable_write(output, result.model_dump(mode="json"))
                    return result
                context_start = time.monotonic()
                context = context_for(case)
                located = located_sources(case)
                sources = tuple(s.source for s in located)
                catalog = SourceCatalogV3(located)
                procurement = review_procurement_v3(
                    context,
                    oracle_procurement(case),
                    authorized_scope_id=context.scope_id,
                )
                context_ms = int((time.monotonic() - context_start) * 1000)
                branches = []
                grounded = []
                failures = []
                model_start = time.monotonic()
                for worker in ("policy", "contract"):
                    if reason := authority.stopped_reason(scope_id, deadline):
                        break
                    request = document_request_v3(
                        context,
                        case_id=case.case_id,
                        question=case.model_input.question,
                        worker=worker,
                        sources=tuple(s for s in sources if s.worker == worker),
                    )
                    task_key = key + "." + worker
                    try:
                        if ledger.next_action(task_key) == "reuse_committed":
                            branch = ledger.load_committed_projection(task_key)
                        else:
                            branch = await driver.run(
                                task_key,
                                request,
                                deadline=deadline,
                                reserve_input=len(request.model_dump_json().encode()),
                                reserve_output=12000,
                                max_attempts=candidate["max_attempts"],
                            )
                        branches.append(branch)
                        if branch.evidence_output is None:
                            failures.append("document_model_failed")
                            grounded.append(
                                failed_document(worker, "document_model_failed")
                            )
                        else:
                            doc, witnesses = parse_evidence_v3(
                                branch.evidence_output.model_dump_json(),
                                request,
                                protection=protection,
                            )
                            grounded.append(
                                ground_document_v3(request, doc, witnesses, catalog)
                            )
                    except (AdmissionDenied, LedgerError):
                        failures.append(
                            "deadline"
                            if datetime.now(UTC) >= deadline
                            else "document_model_failed"
                        )
                        grounded.append(
                            failed_document(worker, "document_model_failed")
                        )
                    finally:
                        ledger.export(directory / "ledger-export.json")
                model_ms = int((time.monotonic() - model_start) * 1000)
                aggregate_start = time.monotonic()
                if reason := authority.stopped_reason(scope_id, deadline):
                    decision = stopped_decision(reason)
                    branches = []
                else:
                    decision = finalize_review_v3(
                        context,
                        *grounded,
                        procurement,
                        sources=sources,
                        protection=protection,
                        pipeline_failures=failures,
                        control_events=tuple(
                            e for branch in branches for e in branch.security_events
                        ),
                    )
                aggregate_ms = int((time.monotonic() - aggregate_start) * 1000)
                receipts = [
                    json.loads(a["receipt_json"])
                    for worker in ("policy", "contract")
                    for a in _history(ledger, key + "." + worker)
                    if a["receipt_json"]
                ]
                extra_ms = sum(
                    max(
                        0,
                        int(
                            (
                                datetime.fromisoformat(r["completed_at"])
                                - datetime.fromisoformat(r["started_at"])
                            ).total_seconds()
                            * 1000
                        ),
                    )
                    for r in receipts
                    if r["task_key"].startswith(key + ".")
                    and r["attempt_no"] > 1
                    and r.get("completed_at")
                    and r.get("started_at")
                )
                result = FCaseProjection(
                    case_id=case.case_id,
                    repeat_seed=seed,
                    configuration_sha256=digest,
                    decision=decision,
                    branches=tuple(branches),
                    timing=CaseTiming(
                        queue_ms=state["queue_ms"],
                        context_ms=context_ms,
                        model_ms=model_ms,
                        extra_attempt_ms=extra_ms,
                        aggregate_ms=aggregate_ms,
                        total_ms=state["queue_ms"]
                        + int(
                            (
                                datetime.now(UTC)
                                - datetime.fromisoformat(state["started_at"])
                            ).total_seconds()
                            * 1000
                        ),
                    ),
                )
                if reason := authority.stopped_reason(scope_id, deadline):
                    result = result.model_copy(
                        update={
                            "decision": stopped_decision(reason),
                            "branches": (),
                        }
                    )
                stable_write(output, result.model_dump(mode="json"))
                print(
                    json.dumps(
                        dict(
                            case=case.case_id,
                            repeat=seed,
                            recommendation=decision.business.recommendation,
                            branches=len(branches),
                            failures=decision.business.pipeline_failures,
                            budget=budget.summary(),
                        )
                    ),
                    flush=True,
                )
                return result

        try:
            await asyncio.gather(
                *(run_case(c, s) for s in candidate["repeat_seeds"] for c in selected)
            )
        finally:
            ledger.export(directory / "ledger-export.json")
        result = replay(directory)
        result["stage_usage"] = budget.summary()
        if (directory / "quality-stop.json").exists():
            result["quality_stop"] = json.loads(
                (directory / "quality-stop.json").read_text()
            )
        stable_write(directory / "result.json", result)
        return result


def replay(directory):
    from .eval_f_score import score_batch

    directory = Path(directory)
    registered = json.loads((directory / "registration.json").read_text())
    candidate = registered["candidates"][directory.name]
    projections = [
        FCaseProjection.model_validate_json(p.read_bytes())
        for p in sorted(directory.glob("V107-*.json"))
    ]
    result = score_batch(
        projections,
        candidate["case_ids"],
        candidate["repeat_seeds"],
        registered["versions"][0],
    ) | {"network_calls": 0}
    if (directory / "quality-stop.json").exists():
        result["quality_stop"] = json.loads(
            (directory / "quality-stop.json").read_text()
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("path", type=Path)
    p.add_argument("--diagnostic-predecessor")
    p.add_argument(
        "--dataset", choices=("gold-set.v3.1", "gold-set.v3.2"), default="gold-set.v3.2"
    )
    p = sub.add_parser("run")
    p.add_argument("path", type=Path)
    p.add_argument("candidate")
    p = sub.add_parser("prepare-fixed")
    p.add_argument("path", type=Path)
    p.add_argument("--split", choices=("dev", "validation"), default="validation")
    p.add_argument(
        "--dataset",
        required=True,
        choices=(
            "gold-set.v3.1",
            "gold-set.v3.2",
            "gold-set.v3.3",
            "gold-set.v3.4",
            "gold-set.v3.5",
            "gold-set.v3.6",
        ),
    )
    p = sub.add_parser("replay")
    p.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-fixed":
        if args.path.exists():
            parser.error("registration must be new")
        value = fixed_registration(args.dataset, split=args.split)
        stable_write(args.path, value)
        result = {"registration_sha256": payload_sha256(value), "network_calls": 0}
    elif args.command == "prepare":
        result = prepare(args.path, args.diagnostic_predecessor, args.dataset)
    elif args.command == "replay":
        result = replay(args.directory)
    else:
        result = asyncio.run(execute(args.path, args.candidate))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
