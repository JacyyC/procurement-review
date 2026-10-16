"""Every branch of the send/return classification and of the paired attribution.

The classification decides what a Run is allowed to claim it spent, and the attribution
decides where a difference between the two arms came from, so every branch of both is
exercised here on hand-written inputs with a hand-written expectation: a reserved attempt
that never returned, an interrupted one whose send state is undecidable, a legacy row with
no correction kind, and each attribution outcome including the two that must not be
confused with a judgment loss.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from pydantic import SecretStr

from procurement_review.attempt_metering import (
    AttemptFact,
    PlanFacts,
    classify,
    facts_from_receipts,
    kind_of,
)
from procurement_review.eval_attempt_ledger import AttemptLedger
from procurement_review.eval_f_context import context_for, located_sources
from procurement_review.eval_f_contracts import FBudgetProof
from procurement_review.eval_f_registration import configuration
from procurement_review.eval_f_run import FAuthority
from procurement_review.gold_v37_conditional import as_gold_case, conditional_cases
from procurement_review.model_attempt_v2 import configuration_hash
from procurement_review.precise_dev_paired_measure import (
    _catalog,
    _oracle_sources,
    _own_rank,
    _slices,
    attribution_for_case,
)
from procurement_review.precise_dev_paired_metering import (
    _attempt_receipts,
    _with_kind,
    branch_metering,
)
from procurement_review.precise_run_transport import PreciseRunChatTransport


def test_every_correction_kind_is_named_by_its_own_evidence():
    # A citation refusal, in either spelling the line uses, is a citation correction.
    assert kind_of("citation_proposition_unsupported", "citation") == "citation"
    assert kind_of("citation_field_value_unsupported", "citation") == "citation"
    # A transport outcome with no content correction is a transport retry, and a content
    # failure with a correction is a contract correction.
    assert kind_of("timeout", None) == "transport"
    assert kind_of("server_error", None) == "transport"
    assert kind_of("invalid_schema", "invalid_schema") == "contract"
    # Nothing to follow means no kind at all.
    assert kind_of(None, None) is None
    assert kind_of("timeout", "transport") == "transport"


def test_an_empty_branch_is_not_complete_and_invents_no_attempt():
    metering = classify([])
    assert metering.attempts_opened == 0
    assert metering.proven_sends == 0
    assert metering.usage_complete is False
    assert metering.input_tokens_known is None


def test_the_classifier_separates_proven_sends_from_undecidable_ones():
    # One branch that settled twice, the first attempt selecting a citation correction.
    settled = classify(
        (
            AttemptFact(1, True, True, False, 100, 10, next_kind="citation"),
            AttemptFact(2, True, True, False, 200, 20),
        )
    )
    assert settled.attempts_opened == 2
    assert settled.proven_sends == 2
    assert settled.first_sends == 1
    assert settled.citation_corrections == 1
    assert settled.extra_attempts == 1
    assert settled.input_tokens_known == 300
    assert settled.output_tokens_known == 30
    assert settled.unknown_usage_attempts == 0
    assert settled.usage_complete is True
    # A branch whose attempt was interrupted: not a proof of a send, and not free.
    interrupted = classify((AttemptFact(1, False, True, True, None, None),))
    assert interrupted.attempts_opened == 1
    assert interrupted.proven_sends == 0
    assert interrupted.unresolved_attempts == 1
    assert interrupted.open_attempts == 0
    assert interrupted.unknown_usage_attempts == 1
    assert interrupted.usage_complete is False
    # A branch whose attempt row is still open: the same answer, reported as open.
    open_row = classify((AttemptFact(1, False, False, True, None, None),))
    assert open_row.open_attempts == 1
    assert open_row.unresolved_attempts == 0
    assert open_row.unknown_usage_attempts == 1


def test_a_legacy_row_is_reported_as_an_unlabelled_extra_attempt():
    facts = (
        AttemptFact(1, True, True, False, 10, 1),
        AttemptFact(2, True, True, False, 10, 1),
    )
    metering = classify(facts)
    assert metering.other_extra_attempts == 1
    assert metering.contract_corrections == 0


def test_two_branches_never_share_a_correction_order():
    # The audit's repro: one policy citation correction next to one contract first send.
    # Both rows are attempt 1 of their own branch, so neither is an extra attempt.
    mixed = classify(
        (
            AttemptFact(
                1, True, True, False, 10, 1, next_kind="citation", branch_key="policy"
            ),
            AttemptFact(2, True, True, False, 10, 1, branch_key="policy"),
            AttemptFact(1, True, True, False, 10, 1, branch_key="contract"),
        )
    )
    assert mixed.citation_corrections == 1
    assert mixed.contract_corrections == 0
    assert mixed.other_extra_attempts == 0
    assert mixed.first_sends == 2
    assert mixed.branches == 2
    # Interleaved the other way and in a different row order: same reading.
    interleaved = classify(
        (
            AttemptFact(2, True, True, False, 10, 1, branch_key="contract"),
            AttemptFact(
                1, True, True, False, 10, 1, next_kind="contract", branch_key="contract"
            ),
            AttemptFact(2, True, True, False, 10, 1, branch_key="policy"),
            AttemptFact(
                1, True, True, False, 10, 1, next_kind="citation", branch_key="policy"
            ),
        )
    )
    assert interleaved.citation_corrections == 1
    assert interleaved.contract_corrections == 1
    assert interleaved.other_extra_attempts == 0
    assert interleaved.transport_retries == 0
    # A second attempt whose own branch has no first row is not charged to another branch.
    gapped = classify(
        (
            AttemptFact(
                1, True, True, False, 10, 1, next_kind="citation", branch_key="policy"
            ),
            AttemptFact(2, True, True, False, 10, 1, branch_key="contract"),
        )
    )
    assert gapped.citation_corrections == 0
    assert gapped.other_extra_attempts == 1


def test_plan_reservations_and_local_markers_stay_apart_from_sends():
    facts = (
        AttemptFact(1, True, True, False, 10, 1, branch_key="policy", adopted=True),
        AttemptFact(
            1, False, True, True, None, None, branch_key="contract", discarded=True
        ),
        AttemptFact(2, False, False, True, None, None, branch_key="contract"),
    )
    metering = classify(facts, plan=PlanFacts(planned_tasks=4, reserved_requests=6))
    # A reservation is not a send: only the row with its own receipt counts as one.
    assert metering.planned_tasks == 4
    assert metering.reserved_requests == 6
    assert metering.proven_sends == 1
    assert metering.reserved_requests > metering.proven_sends
    assert metering.terminal_attempts == 2
    assert metering.adopted_attempts == 1
    assert metering.discarded_attempts == 1
    assert metering.unknown_usage_attempts == 2
    assert metering.usage_complete is False


def test_facts_from_receipts_reads_both_receipt_spellings():
    rows = [
        {"attempt_no": 1},
        {"attempt_no": 2},
        {"attempt_no": 3},
    ]
    bodies = {
        1: None,
        2: {
            "usage_unknown": False,
            "input_usage": 5,
            "output_usage": 6,
            "next_correction_kind": "citation",
        },
        3: {"usage_unknown": True},
    }
    facts = facts_from_receipts(
        rows,
        receipt_of=lambda row: bodies[row["attempt_no"]],
        closed_of=lambda row: row["attempt_no"] != 1,
    )
    assert facts[0].has_receipt is False and facts[0].closed is False
    assert facts[1].input_tokens == 5 and facts[1].next_kind == "citation"
    assert facts[2].usage_unknown is True


def _ledger(tmp_path: Path) -> AttemptLedger:
    root = tmp_path / ".local" / "metering"
    root.mkdir(parents=True, exist_ok=True)
    execution = root / "execution.json"
    execution.write_text(json.dumps({"generation": 1, "cancelled": False}))
    authority = FAuthority(
        authorization_id="test-metering",
        scope_ids=frozenset(["scope-test"]),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        proof=FBudgetProof(
            evidence_sha256="e" * 64,
            cumulative_request_upper=2796,
            cumulative_input_upper=None,
            cumulative_output_upper=None,
            historical_unknown_input_reserve=None,
            historical_unknown_output_reserve=None,
            allocated_requests=8,
            allocated_input=None,
            allocated_output=None,
            limit_policy="user_no_balance_limit",
            authorization_basis="user-explicit-no-balance-limit",
        ),
        execution_path=execution,
        generation=1,
    )
    transport = PreciseRunChatTransport(
        configuration(),
        SecretStr("offline-only"),
        catalog=None,
        transport=httpx.MockTransport(lambda request: None),
    )
    return AttemptLedger.create(
        root / "ledger.sqlite3",
        run_id="v1-07-g-metering",
        gold_manifest_sha256="f" * 64,
        configuration_sha256=configuration_hash(transport, authority),
        max_requests=8,
        max_input_tokens=None,
        max_output_tokens=None,
        stage_id="V1-07G",
        gold_set_version="gold-set.v3.1",
    )


def _insert(
    ledger: AttemptLedger, task: str, attempt_no: int, state: str, receipt: dict | None
):
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "INSERT INTO tasks(task_key, payload_sha256, state, created_at, updated_at)"
            " VALUES (?, ?, 'pending', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')"
            " ON CONFLICT(task_key) DO NOTHING",
            (task, "a" * 64),
        )
        connection.execute(
            "INSERT INTO attempts(task_key, attempt_no, request_sha256, state,"
            " reserved_input_tokens, reserved_output_tokens, receipt_json, created_at)"
            " VALUES (?, ?, ?, ?, 1, 1, ?, '2026-01-01T00:00:00Z')",
            (
                task,
                attempt_no,
                "b" * 64,
                state,
                json.dumps(receipt) if receipt else None,
            ),
        )


def test_the_diagnostic_metering_reads_the_branch_own_correction_order(tmp_path: Path):
    ledger = _ledger(tmp_path)
    task = "V107-D-9001.own.oracle.policy"
    _insert(
        ledger,
        task,
        1,
        "failed",
        {
            "error_code": "citation_proposition_unsupported",
            "usage_unknown": False,
            "input_tokens": 10,
            "output_tokens": 2,
        },
    )
    _insert(
        ledger,
        task,
        2,
        "committed",
        {
            "error_code": None,
            "usage_unknown": False,
            "input_tokens": 20,
            "output_tokens": 4,
        },
    )
    metering = branch_metering(ledger, task, ("citation",))
    assert metering["attempts_opened"] == 2
    assert metering["proven_sends"] == 2
    assert metering["citation_corrections"] == 1
    assert metering["contract_corrections"] == 0
    assert metering["input_tokens_known"] == 30
    assert metering["usage_complete"] is True
    # A row with no receipt at all stays unreturned and unknown.
    ledger2 = _ledger(tmp_path / "second")
    _insert(ledger2, task, 1, "sent", None)
    metering2 = branch_metering(ledger2, task)
    assert metering2["proven_sends"] == 0
    assert metering2["unresolved_attempts"] == 1
    assert metering2["unknown_usage_attempts"] == 1
    assert metering2["usage_complete"] is False
    receipts = _attempt_receipts(ledger, task)
    assert [item["state"] for item in receipts] == ["failed", "committed"]
    assert receipts[0]["error"] == "citation_proposition_unsupported"
    assert (
        _with_kind({"attempt_no": 1, "receipt_json": "{}"}, ("contract",))
        == '{"next_correction_kind": "contract"}'
    )
    assert _with_kind({"attempt_no": 1, "receipt_json": None}, ("contract",)) is None
    assert _with_kind({"attempt_no": 1, "receipt_json": "{}"}, ()) == "{}"


def test_every_attribution_outcome_including_the_retrieval_ones():
    truth = {
        "O-04": "violated",
        "O-08": "unknown",
        "O-01": "supported",
        "O-02": "supported",
    }
    cases = {
        "both correct": (
            {
                "O-04": "violated",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {
                "O-04": "violated",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {"policy": 1, "contract": 1},
            True,
        ),
        # The Oracle was right and the actual arm was not, with its own document missing.
        "retrieval loss": (
            {
                "O-04": "violated",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {
                "O-04": "unknown",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {"policy": -1, "contract": -1},
            True,
        ),
        # The document was there but that branch never produced a usable document.
        "assembly loss": (
            {
                "O-04": "violated",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {
                "O-04": "unknown",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {"policy": 1, "contract": 1},
            False,
        ),
        # The document was there and the branch grounded, so the difference is judgment.
        "judgment loss": (
            {
                "O-04": "violated",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {
                "O-04": "supported",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {"policy": 1, "contract": 1},
            True,
        ),
        # The actual arm was right where the Oracle was not, on that same duty.
        "oracle only error": (
            {
                "O-04": "supported",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {
                "O-04": "violated",
                "O-08": "unknown",
                "O-01": "supported",
                "O-02": "supported",
            },
            {"policy": 1, "contract": 1},
            True,
        ),
    }
    for label, (oracle, actual, ranks, grounded) in cases.items():
        expected = label.replace(" ", "_")
        attributed = attribution_for_case(
            truth=truth,
            oracle_states=oracle,
            actual_states=actual,
            own_ranks=ranks,
            actual_grounded=grounded,
        )
        duty = "O-04"
        assert attributed[duty] == expected, (label, attributed)


def test_the_diagnostic_arm_helpers_bind_sources_to_the_case():
    case = conditional_cases()[0]
    gold = as_gold_case(case)
    context = context_for(gold)
    located = located_sources(gold)
    sources = tuple(item.source for item in located)
    oracle = _oracle_sources(case, "policy", case.case_id)
    assert oracle and all(item.worker == "policy" for item in oracle)
    slices = _slices(case.case_id, "contract", ())
    assert slices == ()
    catalog = _catalog(context, "policy", oracle)
    assert catalog is not None

    # The rank helper reports where the Case's own document was retrieved, and -1 when it
    # was not among the returned hits.
    class Hit:
        def __init__(self, checksum: str, rank: int) -> None:
            self.content_checksum = checksum
            self.rank = rank

    hits = (Hit("sha256:" + "a" * 64, 1), Hit("sha256:" + "b" * 64, 2))
    assert _own_rank(hits, "a" * 64) == 1
    assert _own_rank(hits, "c" * 64) == -1
    assert any(item.worker == "policy" for item in sources)
