"""Complete product contracts and semantic projection checks for G consumption v5."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from .business_v3_procurement import canonical_json_bytes
from .business_fact_contracts import BUSINESS_FACT_TOOL_KINDS
from .gold_v3_contracts import payload_sha256
from .ingestion import LocatorKind, SourceLocator, make_source_version_id
from .precise_run_contracts import PreciseRunExecutionV1
from .precise_run_summary import DUTY_CATEGORIES, DUTY_OWNERS, PRODUCT_RISK


def execution_contract_failures(run_id, run):
    failures = []
    try:
        execution = PreciseRunExecutionV1.model_validate_json(
            json.dumps(run.get("execution")), strict=True
        ).model_dump(mode="json")
    except ValidationError as error:
        return [
            "execution_contract_type:" + ".".join(map(str, item["loc"])) + ":" + run_id
            for item in error.errors()
        ]

    # Persisted contracts carry even nullable/default fields. Model defaults are
    # historical read conveniences, not evidence that a stored field existed.
    def missing(raw, parsed, path):
        if type(parsed) is dict and type(raw) is dict:
            for key, value in parsed.items():
                if key not in raw:
                    failures.append(f"execution_contract_missing:{path}.{key}:{run_id}")
                else:
                    missing(raw[key], value, path + "." + key)
        elif type(parsed) is list and type(raw) is list:
            for index, (left, right) in enumerate(zip(raw, parsed)):
                missing(left, right, path + "." + str(index))
        elif type(parsed) in (int, bool) and type(raw) is not type(parsed):
            # Strict JSON Literal validation still coerces equal numeric/bool values.
            failures.append(
                f"execution_contract_type:{path.removeprefix('execution.')}:{run_id}"
            )

    missing(run["execution"], execution, "execution")
    return failures


def projection_failures(registration, run_id, run):
    failures = []

    def fail(code):
        failures.append(f"{code}:{run_id}")

    execution = run["execution"]
    summary = run["summary"]["summary_payload"]
    decision = execution["decision"]
    if execution["run_id"] != run_id or execution["case_id"] != run["case_id"]:
        fail("execution_parent_identity")
    if run["case_id"] not in registration["truth_extension"]:
        fail("execution_case_unregistered")
        return failures
    if payload_sha256(summary) != payload_sha256(execution["summary"]):
        fail("execution_summary_mismatch")
    if summary["recommendation"] != decision["recommendation"]:
        fail("summary_recommendation")
    if summary["rules_version"] != decision["rules_version"]:
        fail("summary_rules_version")
    human = (
        "pending_confirmation"
        if decision["recommendation"]
        in {"recommend_approve", "recommend_conditional", "recommend_reject"}
        else "required"
    )
    if summary["human_review_status"] != human:
        fail("summary_human_status")
    if summary["is_final_decision"] is not False or summary["conflicts"] != []:
        fail("summary_gate_contract")
    expected_risk = "unknown"
    for level in ("critical", "high", "medium", "low", "none"):
        if any(row["risk"] == level for row in decision["findings"]):
            expected_risk = PRODUCT_RISK[level].value
            break
    if summary["highest_risk"] != expected_risk:
        fail("summary_highest_risk")
    causes = [
        item
        for row in decision["findings"]
        if row["state"] == "unknown"
        for item in row["missing"]
    ]
    causes.extend("conflict:" + pair["pair_id"] for pair in execution["conflicts"])
    causes.extend(
        "security:" + event["code"]
        for event in execution["security"]
        if event["code"] != "benign_refusal"
    )
    if decision["recommendation"] == "needs_human_review" and not causes:
        causes.append("human_review_required:non_remediable_violation")
    if summary["missing_information"] != list(dict.fromkeys(causes)):
        fail("summary_missing_information")
    if summary["failure_reasons"] != decision["pipeline_failures"]:
        fail("summary_failure_reasons")
    expected_rows = [
        row for row in decision["findings"] if row["obligation_id"] in DUTY_OWNERS
    ]
    actual_rows = summary["findings"]
    if [r["finding_key"] for r in actual_rows] != [
        r["obligation_id"] for r in expected_rows
    ]:
        fail("summary_duty_set")
        return failures
    sources = registration["truth_extension"][run["case_id"]]["branch_sources"]
    for actual, expected in zip(actual_rows, expected_rows):
        duty = expected["obligation_id"]
        supported = expected["state"] in {"supported", "violated", "not_applicable"}
        values = {
            "worker": DUTY_OWNERS[duty].value,
            "category": DUTY_CATEGORIES[duty].value,
            "risk_level": PRODUCT_RISK[expected["risk"]].value,
            "assertion": "risk_present"
            if expected["state"] == "violated"
            else "risk_absent"
            if supported
            else "unknown",
            "status": "supported" if supported else "unknown",
            "statement": f"rules 3.1.0 {duty}: {expected['reason']}",
            "missing_information": expected["missing"]
            if expected["state"] == "unknown"
            else [],
            "recommended_action": "\u4eba\u5de5\u6838\u5bf9\u6765\u6e90"
            if supported
            else "\u8865\u5145\u6240\u7f3a\u8bc1\u636e",
        }
        for key, value in values.items():
            if actual[key] != value:
                fail("summary_finding_projection:" + duty + "." + key)
        refs = actual["evidence_refs"]
        facts = actual["business_fact_refs"]
        mapped_refs, mapped_facts = [], []
        for identity in expected["evidence"]:
            documents = [
                ref
                for worker in ("policy", "contract")
                for ref in execution[worker]["evidence_refs"]
                if ref["source_id"] == identity["source_id"]
                and ref["content_checksum"] == "sha256:" + identity["content_sha256"]
            ]
            if documents:
                ref = documents[0]
                frozen = [
                    s for s in sources.values() if s["source_id"] == ref["source_id"]
                ]
                if not any(
                    s["content_sha256"] == identity["content_sha256"]
                    and s["locator"] == identity["locator"] == ref["chunk_id"]
                    and s["version"] == identity["version"]
                    for s in frozen
                ):
                    fail("summary_document_identity:" + duty)
                source = next(
                    (
                        s
                        for s in frozen
                        if s["content_sha256"] == identity["content_sha256"]
                    ),
                    None,
                )
                material = next(
                    (
                        registration["material_text"][run["case_id"]][worker]
                        for worker, s in sources.items()
                        if s == source
                    ),
                    None,
                )
                if (
                    source is None
                    or material is None
                    or ref["version"]
                    != make_source_version_id(source["source_id"], source["version"])
                    or ref["source_kind"] != "public_document"
                    or ref["page_number"] is not None
                    or ref["section"]
                    != SourceLocator(
                        kind=LocatorKind.TEXT_LINES,
                        line_start=1,
                        line_end=material.count("\n") + 1,
                    ).evidence_section()
                ):
                    fail("summary_document_identity:" + duty)
                mapped_refs.append(ref)
                continue
            matches = [
                fact
                for fact in facts
                if fact["record_id"] == identity["source_id"]
                and payload_sha256(fact["result"]) == identity["content_sha256"]
                and fact["dataset_version"] == identity["version"]
            ]
            if not matches or len({payload_sha256(fact) for fact in matches}) != 1:
                fail("summary_source_unmapped:" + duty)
            else:
                fact = matches[0]
                frozen = registration["truth_extension"][run["case_id"]]["facts"].get(
                    identity["locator"], {}
                )
                if (
                    BUSINESS_FACT_TOOL_KINDS.get(fact["tool_name"])
                    != identity["locator"].split(".", 1)[0]
                    or any(
                        frozen.get(key) != identity[key]
                        for key in ("source_id", "version", "content_sha256")
                    )
                    or datetime.fromisoformat(fact["as_of"])
                    .astimezone(ZoneInfo("Asia/Shanghai"))
                    .date()
                    .isoformat()
                    != frozen.get("effective_at")
                ):
                    fail("summary_fact_identity:" + duty)
                mapped_facts.append(fact)
        if (
            refs != mapped_refs
            or facts != mapped_facts
            or (supported and not refs and not facts)
        ):
            fail("summary_evidence_projection:" + duty)
        for fact in facts:
            if canonical_json_bytes(fact["result"]) != canonical_json_bytes(
                next(
                    (
                        other["result"]
                        for other in mapped_facts
                        if other["record_id"] == fact["record_id"]
                    ),
                    {},
                )
            ):
                fail("summary_fact_content:" + duty)
    return failures
