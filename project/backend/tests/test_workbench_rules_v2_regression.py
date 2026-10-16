"""REG-01 / BASE-01: the fixed 9 Case regression.

V1-01 accepted these nine predeclared synthetic scenarios under the frozen aggregation
contract, and V1-05 Spec §8 requires that any deviation under the revised contract is
attributed case by case rather than silently re-baselined.  V1-05 measured the accepted
expectations under `rules 2.0.0`; the sole deviation (`CASE-CONFLICT-001` conflict count
`1 -> 2`) was approved for re-baselining on 2026-09-11.

V1-03 keeps the expectation table and the registered deviation exactly as V1-05 froze
them and re-expresses the carrier on the authenticated `review-api.v2` transport: every
Case now runs as a registered Case under a verified local credential, and the workbench
product default is the persisted `rules 2.0.0` contract.  The historical `rules 1.0.0`
comparison is preserved as a transport-free aggregation replay, so the frozen V1-01
expectations stay independently replayable without the retired v1 API.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from procurement_review.api_contracts import ReviewCreateRequest
from procurement_review.case_catalog import CASE_CATALOG, CASE_CATALOG_BY_ID
from procurement_review.local_runtime import LocalWorkflowRunner
from procurement_review.rules import RULE_SET, RULE_SET_V2

from v1_03_support import DEMO_SUBJECT, SecurityHarness

# The accepted V1-01 business expectations: lifecycle, recommendation, escalation shape and
# the human-gate result. Nothing here is derived from rules 2.0.
EXPECTED: dict[str, dict[str, Any]] = {
    "CASE-LOW-001": {
        "status": "pending_confirmation",
        "recommendation": "recommend_approve",
        "conflicts": 0,
        "branch_failures": 0,
        "action": "confirm",
    },
    "CASE-HIGH-001": {
        "status": "pending_confirmation",
        "recommendation": "recommend_reject",
        "conflicts": 0,
        "branch_failures": 0,
        "action": "reject",
    },
    "CASE-CONFLICT-001": {
        "status": "pending_human_review",
        "recommendation": "needs_human_review",
        "conflicts": 1,
        "branch_failures": 0,
        "action": None,
    },
    "CASE-UNKNOWN-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 1,
        "action": None,
    },
    "CASE-PARTIAL-FAILURE-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 1,
        "action": None,
    },
    "CASE-ALL-FAILURE-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 3,
        "action": None,
    },
    "CASE-VERSION-001": {
        "status": "pending_human_review",
        "recommendation": "insufficient_information",
        "conflicts": 0,
        "branch_failures": 3,
        "action": None,
    },
    "CASE-CANCEL-001": {
        "status": "cancelled",
        "recommendation": None,
        "conflicts": 0,
        "branch_failures": None,
        "action": "cancel",
    },
    "CASE-RUNTIME-FAILURE-001": {
        "status": "failed",
        "recommendation": None,
        "conflicts": 0,
        "branch_failures": None,
        "action": None,
    },
}
CASE_IDS = tuple(option.case_id for option in CASE_CATALOG)
DEFAULT_CASE_VERSION = {
    option.case_id: f"{option.case_id.lower()}.v1" for option in CASE_CATALOG
}

# The one deviation found by this regression, attributed case by case instead of being
# folded into the expectation table. ``CASE-CONFLICT-001``'s policy fixture reports the same
# EvidenceRef as both ``risk_absent`` (its required finding) and ``risk_present`` (the shared
# conflict key), which rules 1.0 could only see through the shared key. Spec §4 R-3 makes a
# shared reference an explicit conflict anchor, so rules 2.0 reports both disagreements.
# Everything else in the accepted V1-01 row must stay identical; re-baselining it is the
# user's call (Spec §8), so this stays a registered deviation, not a new expectation.
V2_REGISTERED_DEVIATION: dict[str, dict[str, tuple[int, int]]] = {
    "CASE-CONFLICT-001": {"conflicts": (1, 2)}
}


def wait_for(
    client: TestClient,
    headers: dict[str, str],
    run_id: str,
    *statuses: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/v2/reviews/{run_id}", headers=headers).json()
        if body["status"] in statuses:
            return body
        time.sleep(0.01)
    raise AssertionError(f"review did not reach {statuses}: {body.get('status')}")


def run_case(
    client: TestClient,
    headers: dict[str, str],
    case_id: str,
    round_tag: str,
) -> dict[str, Any]:
    """Drive one fixed case through the authenticated workbench and record its outcome."""

    expected = EXPECTED[case_id]
    created = client.post(
        "/api/v2/reviews",
        headers={**headers, "Idempotency-Key": f"{round_tag}-{case_id}"},
        json={
            "api_version": "review-api.v2",
            "case_id": case_id,
            "case_version": DEFAULT_CASE_VERSION[case_id],
        },
    )
    assert created.status_code == 202, created.text
    run_id = created.json()["run_id"]

    if expected["action"] == "cancel":
        # Cancellable scenario: cancel while it is still processing, then keep the outcome.
        pending = client.get(f"/api/v2/reviews/{run_id}", headers=headers).json()
        assert pending["status"] == "processing"
        cancelled = client.post(
            f"/api/v2/reviews/{run_id}/cancel",
            headers={**headers, "Idempotency-Key": f"{round_tag}-cancel"},
            json={
                "api_version": "review-api.v2",
                "expected_version": pending["version"],
            },
        )
        body = cancelled.json()
        return {
            "status": body["status"],
            "recommendation": None,
            "rules_version": None,
            "conflicts": 0,
            "branch_failures": None,
            "receipt": body["receipt"],
            "audit_event_count": body["audit_event_count"],
        }

    body = wait_for(client, headers, run_id, expected["status"])
    outcome = {
        "status": body["status"],
        "recommendation": (
            None if body["summary"] is None else body["summary"]["recommendation"]
        ),
        "rules_version": (
            None if body["summary"] is None else body["summary"]["rules_version"]
        ),
        "conflicts": (
            0 if body["summary"] is None else len(body["summary"]["conflicts"])
        ),
        "branch_failures": len(body["branch_failures"]),
        "receipt": body["receipt"],
        "audit_event_count": body["audit_event_count"],
    }
    if expected["action"] is not None:
        acted = client.post(
            f"/api/v2/reviews/{run_id}/{expected['action']}",
            headers={**headers, "Idempotency-Key": f"{round_tag}-{case_id}-action"},
            json={
                "api_version": "review-api.v2",
                "expected_version": body["version"],
                "expected_summary_checksum": body["summary_checksum"],
            },
        )
        assert acted.status_code == 200, acted.text
        after = acted.json()
        outcome["after_action"] = {
            "status": after["status"],
            "receipt_count": 0 if after["receipt"] is None else 1,
            "audit_event_count": after["audit_event_count"],
            "external_write_performed": after["receipt"]["external_write_performed"],
            "simulation_mode": after["receipt"]["simulation_mode"],
            "human_actor": after["receipt"]["human_actor"],
        }
    return outcome


def assert_expectations(
    case_id: str,
    outcome: dict[str, Any],
    rules_version: str,
    *,
    contract: str = "rules_1",
) -> None:
    expected = dict(EXPECTED[case_id])
    deviations = (
        V2_REGISTERED_DEVIATION.get(case_id, {}) if contract == "rules_2" else {}
    )
    for field, (_, target) in deviations.items():
        expected[field] = target
    assert outcome["status"] == expected["status"], case_id
    assert outcome["recommendation"] == expected["recommendation"], case_id
    assert outcome["conflicts"] == expected["conflicts"], case_id
    if expected["branch_failures"] is not None:
        assert outcome["branch_failures"] == expected["branch_failures"], case_id
    assert outcome["receipt"] is None, case_id
    if outcome["rules_version"] is not None:
        assert outcome["rules_version"] == rules_version, case_id
    if expected["action"] in {"confirm", "reject"}:
        after = outcome["after_action"]
        assert after["status"] == (
            "confirmed" if expected["action"] == "confirm" else "rejected"
        )
        assert after["receipt_count"] == 1
        assert after["audit_event_count"] == 1
        assert after["external_write_performed"] is False
        assert after["simulation_mode"] == "local_test_only"
        assert after["human_actor"] == DEMO_SUBJECT
    elif expected["action"] == "cancel":
        assert outcome["audit_event_count"] == 1
    else:
        assert outcome["audit_event_count"] == 0


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_rules_2_keeps_every_accepted_business_expectation_twice(
    fresh_harness: SecurityHarness, case_id: str
) -> None:
    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = fresh_harness.authorization_header(token)
    with fresh_harness.client() as client:
        first = run_case(client, headers, case_id, "v2-round-1")
        second = run_case(client, headers, case_id, "v2-round-2")
    assert_expectations(case_id, first, "2.0.0", contract="rules_2")
    assert_expectations(case_id, second, "2.0.0", contract="rules_2")
    # Two rounds must be reproducible: repeat seeds contributed no variance in V1-01F.
    assert first == second


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_the_registered_case_carries_the_rules_2_contract(
    fresh_harness: SecurityHarness, case_id: str
) -> None:
    """The product persists exactly one rules contract per registered Case."""

    token = fresh_harness.issue(DEMO_SUBJECT)
    headers = fresh_harness.authorization_header(token)
    with fresh_harness.client() as client:
        listed = client.get("/api/v2/cases", headers=headers).json()
    entries = {item["case_id"]: item for item in listed["cases"]}
    assert entries[case_id]["rules_version"] == RULE_SET_V2.rules_version == "2.0.0"
    assert entries[case_id]["case_version"] == DEFAULT_CASE_VERSION[case_id]


def test_every_case_registration_binds_the_frozen_catalog() -> None:
    for option in CASE_CATALOG:
        registered = CASE_CATALOG_BY_ID[option.case_id]
        assert registered.scenario is option.scenario
        assert registered.case_type is option.case_type
        assert registered.purchase_order_id == option.purchase_order_id
    assert RULE_SET.rules_version == "1.0.0"


def _offline_outcome(case_id: str, rule_set: Any) -> dict[str, Any]:  # noqa: ANN401
    """Replay one case through the frozen aggregation path without any transport."""

    option = CASE_CATALOG_BY_ID[case_id]
    request = ReviewCreateRequest(
        case_id=case_id,
        case_type=option.case_type,
        scenario=option.scenario,
    )
    execution = asyncio.run(
        LocalWorkflowRunner(rule_set=rule_set).run(
            run_id=f"offline-{case_id}",
            request=request,
        )
    )
    summary = execution.summary
    return {
        "recommendation": summary.recommendation.value,
        "highest_risk": summary.highest_risk.value,
        "human_review_status": summary.human_review_status.value,
        "conflicts": len(summary.conflicts),
        "branch_failures": len(summary.failure_reasons),
    }


def test_the_only_cross_contract_difference_is_the_registered_conflict_count() -> None:
    """No business expectation moved except the one deviation attributed in Spec §8 terms."""

    observed: dict[str, dict[str, Any]] = {}
    for case_id in CASE_IDS:
        if EXPECTED[case_id]["action"] == "cancel":
            # Cancellation is a lifecycle action, not an aggregation outcome; the
            # authenticated transport test above covers it end to end.
            continue
        if case_id == "CASE-RUNTIME-FAILURE-001":
            # This scenario injects a runtime failure before the graph runs, so it has
            # no aggregation outcome to compare; the transport test above covers it.
            continue
        before = _offline_outcome(case_id, RULE_SET)
        after = _offline_outcome(case_id, RULE_SET_V2)
        differing = {
            field: (before_value, after[field])
            for field, before_value in before.items()
            if after[field] != before_value
        }
        if differing:
            observed[case_id] = differing
    assert observed == {
        "CASE-CONFLICT-001": {
            "conflicts": V2_REGISTERED_DEVIATION["CASE-CONFLICT-001"]["conflicts"]
        }
    }
