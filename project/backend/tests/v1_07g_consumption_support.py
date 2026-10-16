"""Authored complete records; expected summary semantics do not call its producer."""

import copy
import hashlib
import json
from datetime import datetime, timedelta

from procurement_review.gold_v3_contracts import payload_sha256
from procurement_review.trace import canonical_checksum
from test_v1_07g_binding_v4 import complete_fixture, format_correction_fixture, refresh

OWNERS = {
    "O-01": "policy",
    "O-02": "policy",
    "O-04": "policy",
    "O-05": "contract",
    "O-06": "contract",
    "O-08": "contract",
    "O-09": "contract",
    "O-03": "procurement_data",
    "O-07": "procurement_data",
    "O-10": "procurement_data",
}
CATEGORIES = {"policy": "policy_compliance", "contract": "contract_terms"}


def source_manifest(data):
    return {
        run_id: {
            "execution_sha256": payload_sha256(run["execution"]),
            "summary_sha256": payload_sha256(run["summary"]["summary_payload"]),
        }
        for run_id, run in data[2]["runs"].items()
    }


def seal(data):
    refresh(data)
    data[0]["binding"].update(
        consumption_version="v1-07g-consumption.v5",
        source_projections=source_manifest(data),
    )
    data[2]["consumption_version"] = "v1-07g-consumption.v5"
    return data


def event(run, step, body, time, generation=None, kind="step_completed"):
    return dict(
        step=step,
        output_summary=body,
        output_checksum=canonical_checksum(body),
        occurred_at=time.isoformat(),
        generation=generation,
        event_type=kind,
        worker=None,
        attempt=None,
        run_id=run["generations"][0]["run_id"],
        trace_id=run["trace_id"],
        event_ordinal=0,
    )


def sort_events(run):
    run["events"].sort(key=lambda e: datetime.fromisoformat(e["occurred_at"]))
    for number, entry in enumerate(run["events"], 1):
        entry["event_ordinal"] = number


def fixture_v5(correction=False):
    data = complete_fixture()
    if correction:
        data = format_correction_fixture(data)
    r, o, b = data
    for run_id, run in b["runs"].items():
        execution = run["execution"]
        execution["rules_version"] = "3.1.0"
        execution["procurement"] = dict(
            worker="procurement_data",
            status="known",
            dataset_version=r["binding"]["expected_dataset_version"],
            tool_calls=3,
            record_count=3,
            model_calls=0,
            detail=None,
        )
        for branch in ("policy", "contract"):
            execution[branch].setdefault("failure_code", None)
            for ref in execution[branch]["evidence_refs"]:
                ref["schema_version"] = "1.0"
                identity = r["truth_extension"][run["case_id"]]["branch_sources"][
                    branch
                ]
                version = dict(
                    namespace="source-version-v1",
                    source_id=identity["source_id"],
                    version=identity["version"],
                )
                ref.update(
                    version="source-version-"
                    + hashlib.sha256(
                        json.dumps(
                            version,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    source_kind="public_document",
                    page_number=None,
                    section="lines:1-"
                    + str(r["material_text"][run["case_id"]][branch].count("\n") + 1)
                    + ":part:1",
                )
        for cp in run["checkpoints"]:
            cp["result_payload"]["result"]["evidence_refs"] = copy.deepcopy(
                execution[cp["worker"]]["evidence_refs"]
            )
        for entry in execution["security"]:
            entry.update(
                schema_version="security-event.v3", forbidden_export_succeeded=False
            )
        decision = execution["decision"]
        decision.update(
            rules_version="3.1.0",
            violations=[],
            unknown_obligations=[],
            pipeline_failures=[],
            security_events=[],
            conflict_pairs=[],
        )
        findings = []
        for row in decision["findings"]:
            row.update(rules_version="3.1.0", critical=True, remediable=False)
            duty = row["obligation_id"]
            if row["state"] == "unknown":
                row.update(
                    risk="unknown",
                    evidence=[],
                    missing=row["missing"] or ["missing_source"],
                )
            if row["state"] != "unknown":
                worker = "contract" if OWNERS.get(duty) == "contract" else "policy"
                row["evidence"] = [
                    copy.deepcopy(
                        r["truth_extension"][run["case_id"]]["branch_sources"][worker]
                    )
                ]
            if duty not in OWNERS:
                continue
            owner = OWNERS[duty]
            supported = row["state"] != "unknown"
            refs = (
                [
                    copy.deepcopy(
                        execution["contract" if owner == "contract" else "policy"][
                            "evidence_refs"
                        ][0]
                    )
                ]
                if supported
                else []
            )
            findings.append(
                dict(
                    schema_version="1.0",
                    worker=owner,
                    finding_key=duty,
                    category="supplier_integrity"
                    if duty == "O-03"
                    else CATEGORIES.get(owner, "data_quality"),
                    risk_level="low" if row["risk"] == "none" else row["risk"],
                    assertion="risk_present"
                    if row["state"] == "violated"
                    else "risk_absent"
                    if supported
                    else "unknown",
                    statement=f"rules 3.1.0 {duty}: {row['reason']}",
                    status="supported" if supported else "unknown",
                    evidence_refs=refs,
                    business_fact_refs=[],
                    missing_information=row["missing"] if not supported else [],
                    recommended_action="\u4eba\u5de5\u6838\u5bf9\u6765\u6e90"
                    if supported
                    else "\u8865\u5145\u6240\u7f3a\u8bc1\u636e",
                )
            )
        causes = [
            item
            for row in decision["findings"]
            if row["state"] == "unknown"
            for item in row["missing"]
        ]
        causes.extend("conflict:" + pair["pair_id"] for pair in execution["conflicts"])
        causes.extend(
            "security:" + entry["code"]
            for entry in execution["security"]
            if entry["code"] != "benign_refusal"
        )
        if decision["recommendation"] == "needs_human_review" and not causes:
            causes.append("human_review_required:non_remediable_violation")
        risk = next(
            (
                level
                for level in ("critical", "high", "medium", "low", "none")
                if any(row["risk"] == level for row in decision["findings"])
            ),
            "unknown",
        )
        summary = dict(
            schema_version="1.0",
            run_id=run_id,
            case_id=run["case_id"],
            rules_version="3.1.0",
            recommendation=decision["recommendation"],
            highest_risk="low" if risk == "none" else risk,
            human_review_status="pending_confirmation"
            if decision["recommendation"]
            in {"recommend_approve", "recommend_conditional", "recommend_reject"}
            else "required",
            findings=findings,
            conflicts=[],
            missing_information=list(dict.fromkeys(causes)),
            failure_reasons=[],
            is_final_decision=False,
        )
        run["summary"].update(
            summary_payload=summary, summary_checksum=canonical_checksum(summary)
        )
        execution["summary"] = copy.deepcopy(summary)
        start = datetime.fromisoformat(run["created_at"])
        finish = datetime.fromisoformat(run["summary"]["created_at"])
        run["events"] = [e for e in run["events"] if e["step"] == "model_attempt"]
        run["events"].extend(
            [
                event(run, "review_run", {}, start, kind="run_started"),
                event(
                    run,
                    "validate_input",
                    {"status": "valid"},
                    start + timedelta(microseconds=1),
                ),
                event(
                    run,
                    "create_run",
                    {"status": "processing"},
                    start + timedelta(microseconds=2),
                ),
                event(
                    run,
                    "deterministic_aggregator",
                    dict(
                        recommendation=summary["recommendation"],
                        highest_risk=summary["highest_risk"],
                        finding_count=10,
                        missing_count=len(summary["missing_information"]),
                        conflict_count=0,
                        failure_count=0,
                    ),
                    finish,
                    1,
                ),
                event(
                    run,
                    "review_workflow",
                    {
                        "status": "pending_confirmation"
                        if summary["human_review_status"] == "pending_confirmation"
                        else "pending_human_review"
                    },
                    finish + timedelta(microseconds=1),
                    1,
                ),
            ]
        )
        sort_events(run)
        obs = next(row for row in o["observations"] if row["run_id"] == run_id)
        run["status"] = (
            "pending_confirmation"
            if summary["human_review_status"] == "pending_confirmation"
            else "pending_human_review"
        )
        obs["status"] = run["status"]
        obs["observed_at"] = (finish + timedelta(microseconds=3)).isoformat()
    return seal(data)


def recovery_fixture(data):
    data = copy.deepcopy(data)
    run = next(iter(data[2]["runs"].values()))
    created = datetime.fromisoformat(run["created_at"])
    start, finish = created + timedelta(seconds=20), created + timedelta(seconds=25)
    run["resume_count"] = 1
    run["generations"][0]["status"] = "expired"
    run["generations"].append(
        dict(
            generation=2,
            run_id=run["generations"][0]["run_id"],
            status="completed",
            created_at=start.isoformat(),
            deadline_at=(start + timedelta(seconds=90)).isoformat(),
        )
    )
    run["execution"]["metering"].update(explicit_recoveries=1, reused_branches=2)
    run["summary"]["created_at"] = finish.isoformat()
    for e in run["events"]:
        if e["step"] in {"deterministic_aggregator", "review_workflow"}:
            e.update(
                generation=2,
                occurred_at=(
                    finish + timedelta(microseconds=e["step"] == "review_workflow")
                ).isoformat(),
            )
    run["events"].extend(
        [
            event(
                run,
                "review_workflow",
                {"status": "interrupted", "reason": "lease_expired"},
                created + timedelta(seconds=15),
                1,
                "step_failed",
            ),
            event(run, "review_recovery", {"status": "processing"}, start, 2),
        ]
    )
    sort_events(run)
    obs = next(
        row for row in data[1]["observations"] if data[2]["runs"][row["run_id"]] is run
    )
    obs.update(
        observed_at=(finish + timedelta(microseconds=3)).isoformat(),
        service_elapsed_ms=25000,
    )
    return seal(data)
