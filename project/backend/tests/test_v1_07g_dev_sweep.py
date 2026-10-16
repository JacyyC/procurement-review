"""The DEV sweep driver: what it sends, what it persists and how the whole stack joins.

The driver is validated without a network: an ``httpx.MockTransport`` answers the three
API calls the workbench makes, with a payload in the product's own projection shape.  The
case below then walks the whole chain — registration → driver → observations → score — so
the real sweep only has to be judged on its readings, not on whether the plumbing works.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from procurement_review.precise_dev_score import score_sweep
from test_v1_07g_dev_quality import _derived_bindings, _extended_registration
from procurement_review.precise_dev_sweep import (
    DEV_SWEEP_VERSION,
    _summarise,
    registration,
    run_sweep,
)


def _execution_payload(case_id: str) -> dict:
    """One Run's projection, in the shape the workbench API returns it."""

    return {
        "execution_version": "review-execution.v2",
        "run_id": f"run-{case_id}",
        "case_id": case_id,
        "rules_version": "3.1.0",
        "decision": {
            "recommendation": "recommend_approve",
            "findings": [
                {
                    "obligation_id": "O-05",
                    "state": "supported",
                    "reason": "contract_identity_complete",
                    "risk": "none",
                    "blocking": False,
                    "missing": [],
                    "evidence": [
                        {
                            "source_id": "PO-901",
                            "version": "synthetic-gold-procurement.v3",
                            "locator": "purchase_order.contract_reference",
                        }
                    ],
                }
            ],
            "violations": [],
            "unknown_obligations": [],
            "pipeline_failures": [],
            "security_events": [],
            "conflict_pairs": [],
        },
        "policy": {
            "worker": "policy",
            "outcome": "completed",
            "reported": [],
            "spans": [],
            "support_failures": [],
            "attempts": [{"attempt_no": 1, "error_code": None, "correctable": False}],
            "evidence_refs": [],
            "failure_code": None,
        },
        "contract": {
            "worker": "contract",
            "outcome": "completed",
            "reported": [],
            "spans": [
                {
                    "evidence": {
                        "source_id": "DOC-V107-D-9007-CONTRACT",
                        "version": "synthetic-gold-document.v3.1",
                        "locator": "chunk-1",
                    },
                    "proposition": {
                        "check_id": "O-05",
                        "field_path": "contract.o_05.contract",
                        "subject_id": "PO-907",
                        "source_version": "synthetic-gold-document.v3.1",
                    },
                    "value": "CONTRACT-907",
                    "start": 10,
                    "end": 22,
                    "span_sha256": "a" * 64,
                }
            ],
            "support_failures": [],
            "attempts": [
                {
                    "attempt_no": 1,
                    "error_code": "citation_proposition_unsupported",
                    "correctable": True,
                },
                {"attempt_no": 2, "error_code": None, "correctable": False},
            ],
            "evidence_refs": [],
            "failure_code": None,
        },
        "procurement": {
            "worker": "procurement_data",
            "status": "known",
            "dataset_version": "synthetic-gold-procurement.v3",
            "tool_calls": 3,
            "record_count": 3,
            "model_calls": 0,
            "detail": None,
        },
        "metering": {
            "physical_attempts": 3,
            "first_sends": 2,
            "corrections": 1,
            "attempts_opened": 3,
            "proven_sends": 3,
            "citation_corrections": 1,
            "other_extra_attempts": 0,
            "usage_complete": True,
        },
        "conflicts": [
            {
                "pair_id": "pair-1",
                "kind": "rule_inconsistency",
                "relation": "order_invoice_currency_equal",
                "left": {
                    "proposition": {
                        "subject_type": "purchase_order",
                        "subject_id": "PO-907",
                        "check_id": "O-06",
                        "field_path": "purchase_order.currency",
                        "effective_at": "2026-06-30",
                        "source_version": "synthetic-gold-procurement.v3",
                    },
                    "value": "CNY",
                    "evidence": [
                        {
                            "source_id": "PO-907",
                            "version": "synthetic-gold-procurement.v3",
                        }
                    ],
                },
                "right": {
                    "proposition": {
                        "subject_type": "purchase_order",
                        "subject_id": "PO-907",
                        "check_id": "O-06",
                        "field_path": "purchase_order.invoice_currency",
                        "effective_at": "2026-06-30",
                        "source_version": "synthetic-gold-procurement.v3",
                    },
                    "value": "USD",
                    "evidence": [
                        {
                            "source_id": "PO-907",
                            "version": "synthetic-gold-procurement.v3",
                        }
                    ],
                },
            }
        ],
        "security": [],
        "summary": {
            "schema_version": "review-summary.v2",
            "rules_version": "3.1.0",
        },
        "stop_reason": None,
    }


def test_the_summary_keeps_the_facts_the_first_answer_reading_needs():
    run = {
        "status": "pending_confirmation",
        "execution_mode": "real",
        "execution": _execution_payload("V107-D-9007"),
        "summary": {"schema_version": "review-summary.v2", "rules_version": "3.1.0"},
    }
    summary = _summarise(run)
    assert summary["recommendation"] == "recommend_approve"
    assert summary["execution_version"] == "review-execution.v2"
    assert summary["branches"]["contract"]["attempts"][0]["correctable"] is True
    assert summary["branches"]["policy"]["attempts"][0]["error_code"] is None
    assert summary["spans"][0]["check_id"] == "O-05"
    assert summary["spans"][0]["source_id"] == "DOC-V107-D-9007-CONTRACT"
    assert summary["conflicts"][0]["relation"] == "order_invoice_currency_equal"
    assert summary["findings"][0]["evidence"][0]["locator"] == (
        "purchase_order.contract_reference"
    )


def test_the_driver_walks_the_api_and_persists_every_observation(tmp_path: Path):
    """Registration → driver → observations → score, with no network and no repository."""

    registration_path = tmp_path / "registration.json"
    registration_path.write_text(
        json.dumps(registration(repeats=1), ensure_ascii=False)
    )
    body = json.loads(registration_path.read_text())
    case_ids = body["case_ids"]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/cases":
            return httpx.Response(
                200,
                json={
                    "cases": [
                        {
                            "case_id": case_id,
                            "case_version": f"{case_id.lower()}.v1",
                        }
                        for case_id in case_ids
                    ]
                },
            )
        if path == "/api/v2/reviews" and request.method == "POST":
            payload = json.loads(request.content)
            return httpx.Response(
                202, json={"run_id": f"run-{payload['case_id']}-1", "status": "queued"}
            )
        if path.startswith("/api/v2/reviews/"):
            run_id = path.rsplit("/", 1)[-1]
            case_id = run_id.split("-", 1)[1].rsplit("-", 1)[0]
            return httpx.Response(
                200,
                json={
                    "status": "pending_confirmation",
                    "execution_mode": "real",
                    "execution": _execution_payload(case_id),
                    "summary": {
                        "schema_version": "review-summary.v2",
                        "rules_version": "3.1.0",
                    },
                },
            )
        return httpx.Response(404, json={"error": "unexpected"})

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"token": "rvk1.local", "api_base": "http://127.0.0.1:8011"})
    )
    output = tmp_path / "observations.json"
    session = httpx.Client(
        base_url="http://127.0.0.1:8011", transport=httpx.MockTransport(handler)
    )
    result = run_sweep(
        registration_path, state_path, output, client=session, timeout_seconds=1.0
    )
    assert result["runs"] == len(case_ids)
    observations = json.loads(output.read_text())
    assert observations["sweep_version"] == DEV_SWEEP_VERSION
    assert len(observations["observations"]) == len(case_ids)
    assert observations["registration_sha256"]
    for row in observations["observations"]:
        assert row["terminal_status"] == 200
        assert row["recommendation"] == "recommend_approve"
    # The persisted observations are enough to produce a verdict.  The mock reading is
    # deliberately *not* a passing sweep: it carries identities another Case would have
    # to authorize and a metering block without the closure fields, so the v2 verdict
    # must name those failures instead of accepting the shape.
    extended = _extended_registration()
    scored = score_sweep(
        extended, observations, _derived_bindings(extended, observations)
    )
    assert scored["stage_acceptance"] is False
    assert set(scored["repeats"]) == {1}
    first = scored["repeats"][1]["q01_first_answer"]
    assert first["denominator"] == 2 * len(case_ids)
    assert first["corrected_after_refusal"] == len(case_ids)
    assert first["value"] < 1.0
    # The duty the driver observed is judged on the reading, and a duty the decision never
    # reported fails closed instead of silently leaving the denominator.
    duties = scored["repeats"][1]["obligations"]
    assert any(row.get("judged") for row in duties)
    missing = [row for row in duties if row.get("judged") is False]
    assert missing and all(row["correct"] is False for row in missing)
    assert "cross-repeat/Q-10-ledger-complete" in scored["failed_gates"]
    assert any(
        "metering_key_missing" in item for item in scored["completeness"]["failures"]
    )
