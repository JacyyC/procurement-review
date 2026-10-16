"""Prepare/replay C registrations without credentials, labels, or network access.

This CLI deliberately has no run switch. A reviewed registration plus defensible
budget evidence is required by BranchAttemptDriver before using ChatTransportV2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .chat_transport_v2 import EndpointConfigV2, request_payload
from .gold_v3_contracts import (
    EvidenceIdentity,
    PropositionIdentity,
    canonical_json_bytes,
    payload_sha256,
)
from .gold_v31 import DIRECTORY as CORPUS_DIRECTORY
from .gold_v31_contracts import CasesFileV31, ManifestV31
from .model_wire_v2 import BranchRequestV2, CheckSlot, SourceSlice, response_schema

DEV_IDS = (
    "V107-D-0001",
    "V107-D-0002",
    "V107-D-0003",
    "V107-D-0004",
    "V107-D-0005",
    "V107-D-0007",
)
DEFAULT_ENDPOINT = "https://model.example.invalid/api/coding/v3/chat/completions"


def dev_requests() -> tuple[BranchRequestV2, ...]:
    bundle = CasesFileV31.model_validate_json(
        (CORPUS_DIRECTORY / "cases.json").read_bytes()
    )
    manifest = ManifestV31.model_validate_json(
        (CORPUS_DIRECTORY / "manifest.json").read_bytes()
    )
    if bundle.payload_sha256 != manifest.cases_payload_sha256:
        raise ValueError("case bundle differs from frozen manifest")
    cases = bundle.cases
    selected = [case for case in cases if case.case_id in DEV_IDS]
    if tuple(c.case_id for c in selected) != DEV_IDS or any(
        c.split.value != "dev" for c in selected
    ):
        raise ValueError("pre-registered DEV inputs drifted")
    requests = []
    for case in selected:
        facts = case.model_input.business_facts
        for worker, checks in (
            (
                "policy",
                (
                    (
                        "O-01",
                        "policy.applicability",
                        "仅依据政策正文提取适用范围、时点及已说明的例外；有明确范围与时点即可supported。未声明例外不新增缺失要求，不索取采购事实；不判整个Case",
                    ),
                    (
                        "O-02",
                        "policy.special_terms",
                        "本检查明确要求提取政策中的交付时限、验收时限、质保期限及上限/下限关系；材料中有这些具体值就必须提取，只有标题或实际缺值才unknown",
                    ),
                ),
            ),
            (
                "contract",
                (
                    (
                        "O-05",
                        "contract.core_terms",
                        "提取合同买方、卖方、标的和订单/供应商/合同/交易对手引用；这些必需内容实际缺失才unknown。不把其它检查的金额、状态、天数塞入本项，空白不能补齐",
                    ),
                    (
                        "O-08",
                        "contract.delivery_acceptance_warranty",
                        "提取交付、验收与质保条款，分别说明具体内容或真实缺失",
                    ),
                    (
                        "O-09",
                        "contract.change_liability",
                        "提取变更、违约及权利限制的具体约定（如责任上限具体值）；仅标题或声称条款已列明不构成充分证据。任一适用必需内容缺少具体约定即unknown并列缺失，禁止常识补齐",
                    ),
                ),
            ),
            (
                "procurement_data",
                (
                    (
                        "O-03",
                        "business_facts.procurement_status",
                        "仅核对采购状态；本探针冻结规则为 approved 满足、blocked 为高风险违规；缺失或 unknown 为未知。其它资格字段由后续完整业务审查负责，不索取其它分支正文",
                    ),
                    (
                        "O-10",
                        "business_facts.source_integrity",
                        "仅核对输入source_status=known、非空source_version和scope_id；source_integrity是本检查名而不是需补交的字段。不要求其它分支的合同正文",
                    ),
                ),
            ),
        ):
            if worker == "procurement_data":
                content = canonical_json_bytes(facts.model_dump(mode="json")).decode()
                identity = EvidenceIdentity(
                    source_id="V107-FACT-" + case.case_id.removeprefix("V107-"),
                    version=facts.source_version,
                    locator="business_facts",
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                )
            else:
                document = (
                    case.model_input.policy_document
                    if worker == "policy"
                    else case.model_input.contract_document
                )
                content = document.content
                identity = EvidenceIdentity(
                    source_id=document.source_id,
                    version=document.version,
                    locator=document.section,
                    content_sha256=document.content_sha256,
                )
            slots = tuple(
                CheckSlot(
                    proposition=PropositionIdentity(
                        scope_id=facts.scope_id,
                        subject_type="purchase_order",
                        subject_id=facts.purchase_order_id,
                        check_id=check,
                        field_path=field,
                        effective_at=case.as_of,
                        source_version=identity.version,
                    ),
                    instruction=instruction,
                )
                for check, field, instruction in checks
            )
            requests.append(
                BranchRequestV2(
                    case_id=case.case_id,
                    worker=worker,
                    scope_id=facts.scope_id,
                    question=case.model_input.question,
                    checks=slots,
                    sources=()
                    if case.case_id == "V107-D-0004" and worker != "procurement_data"
                    else (
                        SourceSlice(
                            scope_id=facts.scope_id,
                            worker=worker,
                            identity=identity,
                            content=content,
                        ),
                    ),
                )
            )
    return tuple(requests)


def registration(
    *,
    no_balance_limit_authorized: bool = False,
    extended_deadline: bool = False,
    non_reasoning_modes: bool = False,
    focused: bool = False,
    final_text: bool = False,
    schema_zero: bool = False,
    empty_source_revision: bool = False,
) -> dict:
    if empty_source_revision and not schema_zero:
        raise ValueError("empty-source revision requires measured schema-zero mode")
    if (final_text or schema_zero) and (
        not extended_deadline or not non_reasoning_modes or focused
    ):
        raise ValueError("final revision requires fixed non-reasoning deadline")
    if final_text and schema_zero:
        raise ValueError("choose one final candidate")
    requests = dev_requests()
    candidates = []
    for mode in (
        ("json_schema",)
        if schema_zero
        else ("text",)
        if final_text
        else ("json_schema", "text")
        if focused
        else ("json_schema", "json_object", "text")
    ):
        config = EndpointConfigV2(
            endpoint=DEFAULT_ENDPOINT,
            requested_model="deepseek-v4-flash",
            resolved_model="deepseek-v4-flash-ga-260731",
            mode=mode,
            thinking="disabled" if non_reasoning_modes else None,
            temperature=0.0 if schema_zero else None,
            single_attempt_seconds=90.0 if extended_deadline else 45.0,
        )
        candidates.append(("minimum:" + mode, config, requests[:3], None))
        candidates.append(("screen:" + mode, config, requests, "minimum:" + mode))
    if empty_source_revision:
        # Previous endpoint/role minima are retained. Probe the one changed schema,
        # then run all 18 registered first attempts without spending on correction.
        empty_contract = next(
            r for r in requests if r.worker == "contract" and not r.sources
        )
        candidates[0] = (
            "minimum:json_schema",
            candidates[0][1],
            (empty_contract,),
            None,
        )
    # Exactly one factor changes, only after a mode passes its minimum. The JSON
    # mode is an explicit dependency, never an automatic fallback to another mode.
    base = EndpointConfigV2(
        endpoint=DEFAULT_ENDPOINT,
        requested_model="deepseek-v4-flash",
        resolved_model="deepseek-v4-flash-ga-260731",
        mode="json_schema",
        single_attempt_seconds=90.0 if extended_deadline else 45.0,
    )
    if final_text or schema_zero:
        pass  # Already measured capabilities; this revision screens one new prompt.
    elif non_reasoning_modes:
        base = base.model_copy(update={"thinking": "disabled"})
        for field, value in (("temperature", 0.0), ("top_p", 1.0), ("max_tokens", 128)):
            candidates.append(
                (
                    "parameter:" + field,
                    base.model_copy(update={field: value}),
                    requests[:3],
                    "minimum:json_schema" if focused else "screen:json_schema",
                )
            )
    elif extended_deadline:
        disabled = base.model_copy(update={"thinking": "disabled"})
        candidates.extend(
            [
                (
                    "minimum:thinking_disabled",
                    disabled,
                    requests[:3],
                    None,
                ),
                (
                    "parameter:thinking_disabled",
                    disabled,
                    requests,
                    "minimum:thinking_disabled",
                ),
            ]
        )
    else:
        for field, value in (
            ("temperature", 0.0),
            ("top_p", 1.0),
            ("max_tokens", 1536),
        ):
            candidates.append(
                (
                    "parameter:" + field,
                    base.model_copy(update={field: value}),
                    requests,
                    "screen:json_schema",
                )
            )
    entries = []
    for name, config, subset, dependency in candidates:
        tasks = []
        for request in subset:
            payload = request_payload(config, request)
            tasks.append(
                {
                    "case_id": request.case_id,
                    "worker": request.worker,
                    "input_variant": "missing_authorized_document"
                    if not request.sources
                    else "original_dev_branch",
                    "task_key": name.replace(":", ".")
                    + ":"
                    + request.case_id
                    + ":"
                    + request.worker,
                    "request_sha256": payload_sha256(payload),
                    "schema_sha256": payload_sha256(response_schema(request)),
                    "planning_input_utf8_bytes": len(canonical_json_bytes(payload)),
                }
            )
        entries.append(
            {
                "candidate": name,
                "configuration": config.model_dump(mode="json"),
                "configuration_sha256": config.fingerprint,
                "dependency": dependency,
                "tasks": tasks,
                "physical_request_limit": len(tasks) * 2,
            }
        )
        if empty_source_revision:
            entries[-1]["max_attempts_per_branch"] = (
                1 if name.startswith("screen:") else 2
            )
            entries[-1]["physical_request_limit"] = (
                len(tasks) * entries[-1]["max_attempts_per_branch"]
            )
    maximum = sum(c["physical_request_limit"] for c in entries)
    if maximum > 240:
        raise ValueError("C cap exceeded")
    return {
        "schema_version": "v1-07c-probe-registration.v1",
        "stage": "V1-07C",
        "gold_set_version": "gold-set.v3.1",
        "gold_manifest_sha256": hashlib.sha256(
            (CORPUS_DIRECTORY / "manifest.json").read_bytes()
        ).hexdigest(),
        "probe_revision": 7
        if empty_source_revision
        else 6
        if schema_zero
        else 5
        if final_text
        else 4
        if focused
        else 3
        if non_reasoning_modes
        else 2
        if extended_deadline
        else 1,
        "network_calls": 0,
        "execution_status": "authorized_no_balance_limit"
        if no_balance_limit_authorized
        else "blocked_budget_unproven",
        "limit_policy": "user_no_balance_limit"
        if no_balance_limit_authorized
        else "cumulative_brakes",
        "authorization_basis": "user-explicit-no-balance-limit"
        if no_balance_limit_authorized
        else "2026-09-19 second-wave instruction; Spec 6.4 cumulative brakes unchanged",
        "dev_case_ids": list(DEV_IDS),
        "candidates": entries,
        "physical_request_limit": maximum,
        "stage_ceiling": 240,
        "concurrency": 1,
        "logical_deadline_seconds": 120 if extended_deadline else 90,
        "batch_deadline_seconds": 7200,
        "planning_output_p50": maximum * 3033,
        "planning_output_p95": maximum * 7290,
        "planning_input_utf8_bytes": 2
        * sum(t["planning_input_utf8_bytes"] for c in entries for t in c["tasks"]),
        "historical_unknown_input_reserve": None,
        "historical_unknown_output_reserve": None,
        "new_attempt_unknown_reserve": None,
        "estimates_are_provider_limits": False,
        "stop_conditions": [
            *(
                []
                if no_balance_limit_authorized
                else ["unproven_budget", "projected_cumulative_overrun"]
            ),
            "stage_request_limit",
            "model_identity_drift",
            "security_boundary",
            "12_consecutive_timeout_or_transport",
            "batch_deadline",
            "cancelled_or_revoked",
            "candidate_dependency_failed",
        ],
        "warning_ratio": 0.8,
        "first_pass_screening_threshold": 0.95,
        "formal_quality_threshold": 0.99,
        "formal_quality_measured": False,
        "scope": "independent DEV branch capability only; no Case recommendation or topology selection",
    }


def replay(path: Path) -> dict:
    from .model_attempt_projection import load_projection
    from .eval_v2_contracts import AttemptReceiptV1

    from .eval_v2_contracts import EvalRunManifestV2
    from .model_wire_v2 import strict_json

    document = strict_json(path.read_text())
    if document["schema_version"] != "eval-attempt-ledger.v1":
        raise ValueError("incompatible ledger")
    manifest = EvalRunManifestV2.model_validate_json(json.dumps(document["manifest"]))
    attempts = document["attempts"]
    if len(attempts) != manifest.attempt_count:
        raise ValueError("attempt count mismatch")
    identities = set()
    receipts, committed = [], []
    committed_tasks = set()
    for row in attempts:
        identity = (row["task_key"], row["attempt_no"])
        if identity in identities or row["attempt_no"] not in {1, 2}:
            raise ValueError("duplicate or excess physical attempt")
        identities.add(identity)
        raw_receipt = row["receipt"]
        if raw_receipt is None:
            if (
                row["state"] not in {"reserved", "sent"}
                or row["projection"] is not None
            ):
                raise ValueError("terminal attempt without receipt")
            continue
        receipt = AttemptReceiptV1.model_validate_json(json.dumps(raw_receipt))
        if (receipt.task_key, receipt.attempt_no, receipt.request_sha256) != (
            row["task_key"],
            row["attempt_no"],
            row["request_sha256"],
        ) or payload_sha256(raw_receipt) != row["receipt_sha256"]:
            raise ValueError("receipt identity or checksum mismatch")
        receipts.append(receipt)
        projection = row["projection"]
        if row["state"] == "committed":
            if projection is None or row["task_key"] in committed_tasks:
                raise ValueError("missing or duplicate committed projection")
            if payload_sha256(projection) != row["projection_sha256"]:
                raise ValueError("projection checksum mismatch")
            parsed = load_projection(json.dumps(projection))
            if parsed.pipeline_failure != (receipt.outcome != "succeeded"):
                raise ValueError("outcome/projection disagreement")
            if parsed.model_request_count != receipt.attempt_no:
                raise ValueError("attempt count reset")
            committed.append(parsed)
            committed_tasks.add(row["task_key"])
        elif projection is not None or receipt.outcome == "succeeded":
            raise ValueError("uncommitted success is not scoreable")
    if len(committed) != manifest.committed_task_count:
        raise ValueError("commit count mismatch")
    observed = (
        sum(r.input_tokens or 0 for r in receipts),
        sum(r.output_tokens or 0 for r in receipts),
        sum(r.usage_unknown for r in receipts),
    )
    if observed != (
        manifest.budget.known_input_tokens,
        manifest.budget.known_output_tokens,
        manifest.budget.unknown_usage_attempts,
    ):
        raise ValueError("usage does not match manifest")
    # No accuracy claim: this is only a branch contract/physical-attempt audit.
    return {
        "network_calls": 0,
        "attempts": len(attempts),
        "planned_first_attempts": manifest.task_count,
        "first_attempts": sum(row["attempt_no"] == 1 for row in attempts),
        "first_valid": sum(
            r.attempt_no == 1 and r.outcome == "succeeded" for r in receipts
        ),
        "final_valid": sum(not p.pipeline_failure for p in committed),
        "corrected_valid": sum(p.corrected_valid for p in committed),
        "known_input": sum(r.input_tokens or 0 for r in receipts),
        "known_output": sum(r.output_tokens or 0 for r in receipts),
        "unknown_usage": sum(r.usage_unknown for r in receipts),
        "unreturned": sum(row["receipt"] is None for row in attempts),
        "quality_acceptance": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--no-balance-limit-authorized", action="store_true")
    parser.add_argument("--extended-deadline", action="store_true")
    parser.add_argument("--non-reasoning-modes", action="store_true")
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--final-text", action="store_true")
    parser.add_argument("--schema-zero", action="store_true")
    parser.add_argument("--empty-source-revision", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if bool(args.prepare) == bool(args.replay):
        parser.error("choose --prepare or --replay")
    if args.prepare:
        from .eval_attempt_ledger import _atomic_write

        if ".local" not in args.prepare.resolve().parts or args.prepare.exists():
            parser.error("prepare needs a new .local output file")
        result = registration(
            no_balance_limit_authorized=args.no_balance_limit_authorized,
            extended_deadline=args.extended_deadline,
            non_reasoning_modes=args.non_reasoning_modes,
            focused=args.focused,
            final_text=args.final_text,
            schema_zero=args.schema_zero,
            empty_source_revision=args.empty_source_revision,
        )
        _atomic_write(args.prepare, canonical_json_bytes(result))
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "candidates"},
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(replay(args.replay), ensure_ascii=False))


if __name__ == "__main__":
    main()
