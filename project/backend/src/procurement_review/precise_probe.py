"""The independent rules 3.1.0 capability probe: a bounded, pre-registered batch.

This is the smallest real path that can answer whether the current endpoint can
produce the versioned ``finding-evidence.v2`` answer for the 3.1.0 schema at all:
two authored cases, two document branches each, the already selected endpoint
configuration, and the verified evaluation attempt machinery for the batch
authorization, the per-branch two-attempt cap, the persistent ledger and the
deadline.  It is *not* the product path and never its evidence: the product chain
is verified separately on an isolated PostgreSQL with a Mock transport, and this
probe claims nothing about retrieval, the Run lifecycle or quality.

Every send is metered into a G-namespaced ledger under ``.local/v1-07g/probe``;
the inherited F and earlier totals are declared, never reset or re-spent.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, SecretStr

from .business_v31_documents import document_request_v31
from .business_v3_types import BusinessContextV3
from .eval_attempt_ledger import AttemptLedger, LedgerError
from .eval_f_context import located_sources, oracle_procurement
from .eval_f_contracts import BranchProjectionV2, FBudgetProof, FEndpointConfig
from .eval_v2_contracts import SemanticClaimProjection
from .precise_answer_v3 import PreciseClaimProjection
from .eval_f_registration import LIVE_ROOT, configuration
from .eval_f_run import FAuthority
from .eval_f_transport import PreciseAttemptDriver
from .gold_v37_conditional import as_gold_case, conditional_cases
from .model_attempt_v2 import configuration_hash
from .model_probe_run import credential
from .precise_run_transport import PreciseRunChatTransport, precise_payload

PROBE_CONTRACT = "v1-07g-precise-probe.v1"
PROBE_ROOT = Path(LIVE_ROOT).parent / "v1-07g" / "probe"
CASE_INDEXES = (0, 3)
#: Which branches this batch sends.  The first real batch covered both and the
#: Contract branch passed while the Policy branch refused its schema twice; this
#: repair batch re-asks only the failing branch, under its own registration.
PROBE_WORKERS = ("policy",)
MAX_REQUESTS_PER_BRANCH = 2
MAX_REQUESTS = len(CASE_INDEXES) * len(PROBE_WORKERS) * MAX_REQUESTS_PER_BRANCH


#: The G model identity the Spec registers for this stage; the F fingerprint does
#: not carry over, and a resolved identity other than this one stops the batch.
G_REQUESTED_MODEL = "deepseek-v4.1-flash"
G_RESOLVED_MODEL = "deepseek-v4-1-flash"


def probe_configuration() -> FEndpointConfig:
    """The stage's endpoint configuration: the G model under the fixed wire settings."""

    base = configuration().model_dump(mode="json")
    return FEndpointConfig(
        **{
            **base,
            "requested_model": G_REQUESTED_MODEL,
            "resolved_model": G_RESOLVED_MODEL,
            "single_attempt_seconds": 90.0,
        }
    )


#: The G implementation files one batch depends on.  Their hashes are part of the
#: registration, so a prompt or schema change is a new batch instead of a silent
#: reuse of the previous ledger.
PROBE_IMPLEMENTATION_FILES = (
    "precise_run_transport.py",
    "business_v31_documents.py",
    "business_v31_review.py",
    "business_v3_conditional.py",
)


def probe_implementation() -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in PROBE_IMPLEMENTATION_FILES
    }


def probe_cases():
    chosen = conditional_cases()
    return tuple(chosen[index] for index in CASE_INDEXES)


def preregistration() -> dict:
    """The frozen manifest: what is sent, under which limits, and why."""

    cases = probe_cases()
    return {
        "contract": PROBE_CONTRACT,
        "stage": "V1-07G",
        "revision": 4,
        "question": "policy 分支在 finding-evidence.v3 下能否通过边界校验并产出可接地的义务值？",
        "information_gain": "诊断批次把两案两次拒绝都定位到 O-04 规范化值超长（315/271/260 字符，上限 256）；revision 2 的一次真实请求已按 v3 收到可解析正文（进程在投影阶段中断，账本按远端未知保留，不重发）。本批在修好投影上界后复验 policy 分支，确认 v3 是否恢复该分支。",
        "protection_set": "仅使用自撰合成材料（gold-set.v3.7 条件用例），不含真实企业数据、凭据或个人信息；结果只保留有界投影与计量。",
        "case_ids": [case.case_id for case in cases],
        "workers": list(PROBE_WORKERS),
        "max_attempts_per_branch": MAX_REQUESTS_PER_BRANCH,
        "max_requests": MAX_REQUESTS,
        "stop_conditions": [
            "任一分支两次尝试仍无有效 envelope 即停止该 Case，不重发",
            "认证、模型身份漂移或安全事件立即停止整个批次",
            "批次期限或请求上限到达即停止，已发请求照常结算",
        ],
        "configuration": probe_configuration().model_dump(mode="json"),
        "base_dataset": "gold-set.v3.1",
        "implementation": probe_implementation(),
        "case_inputs": [
            {
                "case_id": case.case_id,
                "policy_sha256": hashlib.sha256(case.policy_text.encode()).hexdigest(),
                "contract_sha256": hashlib.sha256(
                    case.contract_text.encode()
                ).hexdigest(),
            }
            for case in cases
        ],
    }


class PreciseV31Projection(BranchProjectionV2):
    """The ledger projection of one 3.1.0 attempt: bounded, no answer text."""

    projection_kind: Literal["precise_v31_branch"] = "precise_v31_branch"
    claims: tuple[SemanticClaimProjection | PreciseClaimProjection, ...] = Field(
        max_length=12
    )
    envelope_version: str | None = None
    witnesses: int = 0
    duties_reported: tuple[str, ...] = ()
    #: The refusal's own bounded location tokens: slot/index, authorized duty, validation
    #: category, expected word, basic JSON type and a bounded size — never a value.
    format_diagnostics: tuple[str, ...] = ()
    format_fields: tuple[str, ...] = ()


class PreciseV31Driver(PreciseAttemptDriver):
    """The same attempt machinery, sending the versioned 3.1.0 wire."""

    stage_id = "V1-07G"
    projection_type = PreciseV31Projection

    def make_payload(self, request, correction, correction_fields):
        return precise_payload(
            self.transport.config, request, correction, correction_fields
        )

    def make_projection(self, observed, success, *, task_key, **fields):
        content = getattr(observed, "evidence_content", None)
        diagnostics = tuple(getattr(observed, "format_diagnostics", ()) or ())
        fields_names = tuple(getattr(observed, "format_fields", ()) or ())
        version = None
        witnesses = 0
        duties: tuple[str, ...] = ()
        if content:
            try:
                raw = json.loads(content)
            except ValueError:
                raw = None
            if isinstance(raw, dict):
                version = raw.get("schema_version")
                witnesses = len(raw.get("witnesses") or ())
                duties = tuple(
                    sorted(
                        {
                            item.get("obligation_id")
                            for item in (raw.get("findings") or ())
                            if isinstance(item, dict)
                            and isinstance(item.get("obligation_id"), str)
                        }
                    )
                )
        projection = PreciseV31Projection(
            **{key: value for key, value in fields.items() if key != "evidence_output"},
            envelope_version=version,
            witnesses=witnesses,
            duties_reported=duties,
            format_diagnostics=diagnostics,
            format_fields=fields_names,
        )
        return PreciseV31Projection.model_validate_json(projection.model_dump_json())


def _attempt_codes(ledger: AttemptLedger, task_key: str) -> tuple[str, ...]:
    """The recorded error codes of one logical branch, in attempt order."""

    return tuple(
        item["error"] or item["state"] for item in _attempt_receipts(ledger, task_key)
    )


def _attempt_receipts(ledger: AttemptLedger, task_key: str) -> tuple[dict, ...]:
    """The bounded per-attempt facts of one logical branch, in attempt order.

    Error code, declared fields and the refusal's bounded location tokens: the receipt's
    own answer for *why* an attempt was refused, with no value or answer text.
    """

    import sqlite3

    with sqlite3.connect(ledger.path) as connection:
        rows = connection.execute(
            "SELECT state, receipt_json FROM attempts WHERE task_key=? ORDER BY"
            " attempt_no",
            (task_key,),
        ).fetchall()
    out: list[dict] = []
    for state, receipt in rows:
        body = {}
        if receipt:
            try:
                loaded = json.loads(receipt)
                body = loaded if isinstance(loaded, dict) else {}
            except ValueError:
                body = {}
        out.append(
            {
                "state": str(state),
                "error": _receipt_error(receipt),
                "format_diagnostic": body.get("format_diagnostic"),
                "format_fields": list(body.get("format_fields") or ()),
                "format_diagnostics": list(body.get("format_diagnostics") or ()),
            }
        )
    return tuple(out)


def _receipt_error(receipt: str | None) -> str | None:
    """The recorded error code of one attempt receipt, when it failed."""

    if not receipt:
        return None
    try:
        body = json.loads(receipt)
    except ValueError:
        return None
    code = body.get("error_code") if isinstance(body, dict) else None
    return str(code) if code else None


def _context(case) -> BusinessContextV3:
    gold = as_gold_case(case)
    facts = gold.model_input.business_facts
    read = oracle_procurement(gold)
    order = next(
        (fact for fact in read.facts if fact.tool_name == "get_purchase_order"), None
    )
    body = dict(order.result) if order is not None else {}
    return BusinessContextV3(
        scope_id=facts.scope_id,
        purchase_order_id=str(body.get("purchase_order_id", facts.purchase_order_id)),
        supplier_id=str(body.get("supplier_id", facts.supplier_id)),
        contract_id=str(body.get("contract_reference", facts.purchase_order_id)),
        as_of=gold.as_of,
        dataset_version=facts.source_version,
    )


def _sources(case):
    return located_sources(as_gold_case(case))


async def run(path: Path, *, transport_override=None, root: Path | None = None) -> dict:
    registered = json.loads(Path(path).read_text())
    if registered != preregistration():
        raise LedgerError(
            "probe inputs changed; a new reviewed registration is required"
        )
    digest = hashlib.sha256(
        json.dumps(registered, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    base = Path(root) if root is not None else PROBE_ROOT
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
                        datetime.now(UTC) + timedelta(seconds=1800)
                    ).isoformat(),
                }
            )
        )
    authority = FAuthority(
        authorization_id="V1-07G-user-authorized-precise-probe",
        scope_ids=frozenset(_context(case).scope_id for case in probe_cases()),
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
            run_id="v1-07-g-precise-probe",
            gold_manifest_sha256=digest,
            # The ledger records the shipped dataset the authored conditional cases
            # derive their facts from; their own texts are hashed in the registration.
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
    cases = probe_cases()
    transport = PreciseRunChatTransport(
        config,
        SecretStr("offline-only") if transport_override else credential(),
        catalog=None,
        transport=transport_override,
    )
    driver = PreciseV31Driver(ledger, transport, authority)
    results: list[dict] = []
    for case in cases:
        context = _context(case)
        located = _sources(case)
        for worker in PROBE_WORKERS:
            sources = tuple(
                item.source for item in located if item.source.worker == worker
            )
            request = document_request_v31(
                context,
                case_id=case.case_id,
                worker=worker,
                question=case.question,
                sources=sources,
            )
            result = await driver.run(
                f"{case.case_id}.probe.{worker}",
                request,
                deadline=datetime.now(UTC) + timedelta(seconds=110),
                reserve_input=len(request.model_dump_json().encode()),
                reserve_output=12000,
                max_attempts=MAX_REQUESTS_PER_BRANCH,
            )
            answer = directory / f"{case.case_id}.{worker}.json"
            answer.write_text(
                json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            receipts = _attempt_receipts(ledger, f"{case.case_id}.probe.{worker}")
            codes = tuple(item["error"] or item["state"] for item in receipts)
            results.append(
                {
                    "case_id": case.case_id,
                    "worker": worker,
                    "pipeline_failure": result.pipeline_failure,
                    "first_pass_valid": result.first_pass_valid,
                    "corrected_valid": result.corrected_valid,
                    "model_request_count": result.model_request_count,
                    "error_codes": list(codes),
                    "attempts": list(receipts),
                    "envelope_version": result.envelope_version,
                    "witnesses": result.witnesses,
                    "duties_reported": list(result.duties_reported),
                }
            )
            if any(
                code in {"authentication", "model_identity_drift", "security_boundary"}
                for code in codes
            ):
                break
    summary = {
        "registration": digest,
        "contract": PROBE_CONTRACT,
        "stage_acceptance": False,
        "mode": "offline-mock" if transport_override else "real-endpoint",
        "results": results,
        "ledger": ledger.audit_summary(),
    }
    (directory / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.path.exists():
            parser.error("registration must be new")
        args.path.write_text(
            json.dumps(preregistration(), ensure_ascii=False, sort_keys=True, indent=2)
        )
        print(json.dumps({"prepared": str(args.path)}, ensure_ascii=False))
    else:
        print(json.dumps(asyncio.run(run(args.path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
