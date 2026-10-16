"""The paired diagnostic's execution half: wire, driver, measurement and report.

The registration, the DEV cases and the corpus plan live beside this module; this file
owns what is sent and what is measured — one versioned 3.1.0 attempt per (Case, arm,
branch), the grounded document each successful attempt produced, the product pipeline's
own finalization of the pair, and the attribution of every conditional duty between the
two arms.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr
from sqlalchemy import text

from .eval_attempt_ledger import AttemptLedger, LedgerError
from .business_v31_review import finalize_review_v31
from .business_v3_procurement import review_procurement_v3
from .business_v3_security import ProtectedOutputV3
from .business_v31_documents import document_request_v31
from .eval_f_contracts import FBudgetProof, FEndpointConfig
from .eval_f_run import FAuthority
from .model_attempt_v2 import configuration_hash
from .gold_v37_conditional import (
    ConditionalCaseV1,
    as_gold_case,
    derive_conditional_truth,
)
from .model_probe_run import credential
from .precise_dev_paired_measure import (
    _actual_sources,
    _branch_result,
    _catalog,
    _oracle_sources,
    _own_rank,
    _slices,
    attribution_for_case,
)
from .precise_dev_paired_metering import _attempt_receipts, branch_metering
from .precise_dev_paired_wire import PairedDriver, PairedProjectionV2
from .precise_dev_paired import (
    ARMS,
    CONFIGURATIONS,
    CORPUS_DSN_ENV,
    DATASET_VERSION,
    MAX_REQUESTS_PER_BRANCH,
    PAIRED_CONTRACT,
    PAIR_SUBJECT,
    PAIR_TENANT,
    WORKERS,
    _admin_engine,
    _database_url,
    _order_id,
    corpus_plan,
    distractor_documents,
    paired_cases,
    registration,
)
from .precise_run_summary import business_context
from .precise_run_transport import PreciseRunChatTransport
from .mcp_server import create_procurement_mcp_server
from .procurement_mcp_client import OfficialProcurementMCPClient
from .review_store import ReviewStore
from .scoped_runtime import ScopedReviewRuntime
from .worker_contracts import ProcurementWorkerInput
from .security_contracts import SecurityOperation

#: Where the batch's own outputs live when no explicit root is given.
PAIRED_RUN_ROOT = (
    Path(__file__).resolve().parent.parent.parent / ".local" / "v1-07g" / "paired"
)

# ------------------------------------------------------------------ the projection


async def run(path: Path, *, transport_override=None, root: Path | None = None) -> dict:
    registered = json.loads(Path(path).read_text())
    if registered != registration():
        raise LedgerError(
            "paired diagnostic inputs changed; a new reviewed registration is required"
        )
    digest = hashlib.sha256(
        json.dumps(registered, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    base = Path(root) if root is not None else PAIRED_RUN_ROOT
    base.mkdir(parents=True, exist_ok=True)
    directory = base / digest
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "registration.json").write_text(
        json.dumps(registered, ensure_ascii=False, sort_keys=True, indent=2)
    )
    config = FEndpointConfig.model_validate(registered["configuration"])
    execution_path = directory / "execution.json"
    if not execution_path.exists():
        execution_path.write_text(
            json.dumps(
                {
                    "generation": 1,
                    "cancelled": False,
                    "registration_sha256": digest,
                    "expires_at": (
                        datetime.now(UTC) + timedelta(seconds=7200)
                    ).isoformat(),
                }
            )
        )
    authority = FAuthority(
        authorization_id="V1-07G-user-authorized-paired-diagnostic",
        # One scope id per Case, the identifier the branches carry.
        scope_ids=frozenset(case.case_id for case in paired_cases()),
        expires_at=datetime.fromisoformat(
            json.loads(execution_path.read_text())["expires_at"]
        ),
        proof=FBudgetProof(
            evidence_sha256=digest,
            cumulative_request_upper=2796,
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
        execution_path=execution_path,
        generation=1,
    )
    ledger_path = directory / "ledger.sqlite3"
    if ledger_path.exists():
        ledger = AttemptLedger(ledger_path)
        ledger.recover()
    else:
        ledger = AttemptLedger.create(
            ledger_path,
            run_id="v1-07-g-dev-paired",
            gold_manifest_sha256=digest,
            # The ledger records the shipped DEV dataset the authored conditional cases
            # derive their facts from; the conditional set itself is hashed and named in
            # this batch's registration.
            gold_set_version="gold-set.v3.1",
            stage_id="V1-07G",
            configuration_sha256=configuration_hash(
                PreciseRunChatTransport(
                    config,
                    SecretStr("offline-only"),
                    catalog=None,
                    transport=httpx.MockTransport(lambda request: None),
                ),
                authority,
            ),
            max_requests=registered["max_requests"],
            max_input_tokens=None,
            max_output_tokens=None,
        )
    cases = paired_cases()
    corpus_dsn = os.environ.get(CORPUS_DSN_ENV)
    if not corpus_dsn:
        raise LedgerError(f"{CORPUS_DSN_ENV} is required for the paired batch")
    distractors = distractor_documents()
    results: list[dict] = []
    for configuration in CONFIGURATIONS:
        database = f"v1_07g_paired_{secrets.token_hex(4)}"
        admin = _admin_engine(corpus_dsn)
        store = None
        try:
            with admin.connect() as connection:
                connection.execute(text(f'CREATE DATABASE "{database}"'))
            store = ReviewStore.from_url(_database_url(corpus_dsn, database))
            store.migrate()
            store.seed(
                corpus_plan(cases, distractors, with_library=configuration == "library")
            )
            results.extend(
                await _measure_cases(
                    cases=cases,
                    configuration=configuration,
                    store=store,
                    config=config,
                    ledger=ledger,
                    authority=authority,
                    directory=directory,
                    transport_override=transport_override,
                )
            )
        finally:
            if store is not None:
                store.close()
            with admin.connect() as connection:
                connection.execute(
                    text(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
                )
            admin.dispose()
    summary = {
        "registration": digest,
        "contract": PAIRED_CONTRACT,
        "mode": "offline-mock" if transport_override else "real-endpoint",
        "stage_acceptance": False,
        "cases": results,
        "ledger": ledger.audit_summary(),
    }
    (directory / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return summary


async def _measure_cases(
    *,
    cases: tuple[ConditionalCaseV1, ...],
    configuration: str,
    store: Any,
    config: FEndpointConfig,
    ledger: AttemptLedger,
    authority: FAuthority,
    directory: Path,
    transport_override,
) -> list[dict]:
    from .authorization import AuthorizationService, generate_credential

    transport = PreciseRunChatTransport(
        config,
        SecretStr("offline-only") if transport_override else credential(),
        catalog=None,
        transport=transport_override,
    )
    driver = PairedDriver(ledger, transport, authority)
    results: list[dict] = []
    authorization = AuthorizationService(store=store)
    scope_by_case: dict[str, Any] = {}
    token, credential_id, digest = generate_credential()
    issued_at = datetime.now(UTC)
    store.issue_credential(
        credential_id=credential_id,
        tenant_id=PAIR_TENANT,
        subject_id=PAIR_SUBJECT,
        token_digest=digest,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(seconds=7_200),
    )
    context = authorization.resolve(
        token, request_id="g-paired", operation=SecurityOperation.REVIEW_CREATE
    )
    for case in cases:
        registration = authorization.require_case(
            context,
            case.case_id,
            request_id="g-paired",
            operation=SecurityOperation.REVIEW_CREATE,
        )
        scope_by_case[case.case_id] = authorization.build_runtime_scope(
            context,
            registration,
            generation=1,
            deadline_at=authorization.runtime_scope_deadline(120.0),
        )
    for case in cases:
        results.append(
            await _measure_case(
                case=case,
                configuration=configuration,
                store=store,
                scope=scope_by_case[case.case_id],
                driver=driver,
                directory=directory,
                ledger=ledger,
            )
        )
    return results


async def _measure_case(
    *,
    case: ConditionalCaseV1,
    configuration: str,
    store: Any,
    scope: Any,
    driver: PairedDriver,
    directory: Path,
    ledger: AttemptLedger,
) -> dict:
    from mcp import Client

    truth = derive_conditional_truth(case)
    expected = dict(truth.obligations)
    procurement_read = None
    if store is not None:
        runtime = ScopedReviewRuntime.build(
            store=store,
            scope=scope,
            admission=None,
            vector_mode="none",
            lexical_adapter="idf",
        )
        server = create_procurement_mcp_server(runtime.procurement, scope=scope)
        async with Client(server, raise_exceptions=True) as client:
            procurement_read = await OfficialProcurementMCPClient(
                client
            ).collect_review_facts(
                ProcurementWorkerInput(
                    case_id=case.case_id,
                    purchase_order_id=_order_id(case),
                    dataset_version=DATASET_VERSION,
                    model_fixture_id=f"paired-{PAIRED_CONTRACT}",
                )
            )
    context = business_context(scope, procurement_read).model_copy(
        # The eval-side claims carry the Case's own scope id, the identifier the
        # comparison contracts were written against.
        update={"scope_id": case.case_id}
    )
    procurement = review_procurement_v3(
        context, procurement_read, authorized_scope_id=context.scope_id
    )
    arms: dict[str, dict] = {}
    for arm in ARMS:
        branches: dict[str, dict] = {}
        sources_by_worker: dict[str, tuple] = {}
        catalogs: dict[str, Any] = {}
        for worker in WORKERS:
            task_key = f"{case.case_id}.{configuration}.{arm}.{worker}"
            if arm == "oracle":
                sources = _oracle_sources(case, worker, context.scope_id)
                driver.note_rank(task_key, 0)
            else:
                hits, _allowed = _actual_sources(store, scope, worker)
                sources = _slices(context.scope_id, worker, hits)
                own_document = (
                    as_gold_case(case).model_input.policy_document
                    if worker == "policy"
                    else as_gold_case(case).model_input.contract_document
                )
                driver.note_rank(task_key, _own_rank(hits, own_document.content_sha256))
            request = document_request_v31(
                context,
                case_id=case.case_id,
                worker=worker,
                question=case.question,
                sources=sources,
            )
            catalog = _catalog(context, worker, sources)
            catalogs[worker] = catalog
            driver.register(task_key, request, catalog)
            projection = await driver.run(
                task_key,
                request,
                deadline=datetime.now(UTC) + timedelta(seconds=110),
                reserve_input=len(request.model_dump_json().encode()),
                reserve_output=12000,
                max_attempts=MAX_REQUESTS_PER_BRANCH,
            )
            branches[worker] = {
                **_branch_result(projection),
                "metering": branch_metering(
                    ledger, task_key, projection.correction_kinds
                ),
                "attempts": list(_attempt_receipts(ledger, task_key)),
            }
            sources_by_worker[worker] = sources

        # A branch feeds the decision with the duties it actually established: one whose
        # source check refused every field leaves nothing behind, and one that lost a
        # single duty still contributes the rest, exactly as the product line's branch does.
        def _usable(worker: str):
            document = driver.grounded.get(
                f"{case.case_id}.{configuration}.{arm}.{worker}"
            )
            if document is None or not document.document.values:
                return None
            return document

        grounded = {worker: _usable(worker) for worker in WORKERS}
        states: dict[str, str] = {}
        reasons: dict[str, str] = {}
        pipeline_failures: tuple[str, ...] = ()
        if all(item is not None for item in grounded.values()):
            decision = finalize_review_v31(
                context,
                grounded["policy"],
                grounded["contract"],
                procurement,
                sources=(*sources_by_worker["policy"], *sources_by_worker["contract"]),
                protection=ProtectedOutputV3(),
                pipeline_failures=(),
                control_events=(),
            )
            states = {
                row.obligation_id: row.state for row in decision.business.findings
            }
            reasons = {
                row.obligation_id: row.reason for row in decision.business.findings
            }
            pipeline_failures = tuple(decision.business.pipeline_failures)
        arms[arm] = {
            "branches": branches,
            "states": states,
            "reasons": reasons,
            "pipeline_failures": list(pipeline_failures),
            "sources": {
                worker: [
                    {
                        "source_id": item.identity.source_id,
                        "locator": item.identity.locator,
                        "content_sha256": item.identity.content_sha256,
                    }
                    for item in sources_by_worker[worker]
                ]
                for worker in WORKERS
            },
        }
    oracle_states = arms["oracle"]["states"]
    actual_states = arms["actual"]["states"]
    actual_grounded = all(
        arms["actual"]["branches"][worker]["qualified"] for worker in WORKERS
    )
    own_ranks = {
        worker: arms["actual"]["branches"][worker]["own_document_rank"]
        for worker in WORKERS
    }
    attribution = attribution_for_case(
        truth=expected,
        oracle_states=oracle_states,
        actual_states=actual_states,
        own_ranks=own_ranks,
        actual_grounded=actual_grounded,
    )
    result = {
        "case_id": case.case_id,
        "configuration": configuration,
        "truth": expected,
        "truth_reasons": dict(truth.reasons),
        "attribution": attribution,
        "arms": arms,
    }
    (directory / f"{case.case_id}.{configuration}.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.path.exists():
            parser.error("registration must be new")
        args.path.write_text(
            json.dumps(registration(), ensure_ascii=False, sort_keys=True, indent=2)
        )
        print(json.dumps({"prepared": str(args.path)}, ensure_ascii=False))
    else:
        print(json.dumps(asyncio.run(run(args.path)), ensure_ascii=False, indent=2))


__all__ = [
    "PAIRED_RUN_ROOT",
    "PairedDriver",
    "PairedProjectionV2",
    "attribution_for_case",
    "branch_metering",
    "main",
    "run",
]


if __name__ == "__main__":
    main()
