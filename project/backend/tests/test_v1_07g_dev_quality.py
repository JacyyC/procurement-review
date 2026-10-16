"""The complete DEV quality set and the per-repeat verdict it produces.

The truth cases below check that the independent oracle answers every applicable obligation
for every authored Case — states, reasons, the deterministic values, the clause positions
and the fact fields — and that the verdict of a product sweep is judged *per repeat*, with
a zero denominator reported as undecidable.  The scorer cases build one observation set
from the truth itself, so a passing verdict and a failing one differ only in the field the
case is about.
"""

from __future__ import annotations

import json
from pathlib import Path


from procurement_review.gold_v38_dev_quality import (
    CASE_IDS,
    DEV_QUALITY_SET_VERSION,
    DEV_QUALITY_TRUTH_VERSION,
    dev_quality_cases,
    dev_quality_truth,
    dev_quality_truth_file,
)
from procurement_review.precise_dev_score import (
    CLASSES,
    score_sweep,
)
from procurement_review.gold_v38_dev_quality import id_map
from procurement_review.precise_dev_sweep import (
    mapped_truth,
    registration,
)

SET = {case.case_id: case for case in dev_quality_cases()}


def test_the_truth_covers_every_case_and_every_applicable_obligation():
    truths = {case_id: dev_quality_truth(case) for case_id, case in SET.items()}
    assert set(truths) == set(CASE_IDS)
    assert len(CASE_IDS) == 12
    for truth in truths.values():
        duties = {row.obligation_id for row in truth.expectations}
        assert duties == {f"O-{index:02}" for index in range(1, 13)}, truth.case_id
        for row in truth.expectations:
            assert row.state in {"supported", "violated", "unknown", "not_applicable"}
            assert row.reason
        # The material is what the truth is about: the digest is recorded for binding.
        assert len(truth.material_sha256) == 64


def test_the_truth_reaches_all_five_recommendation_classes():
    counts = {}
    for case in SET.values():
        truth = dev_quality_truth(case)
        counts[truth.recommendation] = counts.get(truth.recommendation, 0) + 1
    assert set(counts) == set(CLASSES)
    assert counts["recommend_approve"] == 3
    assert counts["recommend_conditional"] == 1
    assert counts["recommend_reject"] == 2
    assert counts["insufficient_information"] == 4
    assert counts["needs_human_review"] == 2


def test_the_authored_variants_state_what_they_were_authored_for():
    amount = dev_quality_truth(SET["V107-D-8001"])
    o06 = next(row for row in amount.expectations if row.obligation_id == "O-06")
    assert (o06.state, o06.reason, o06.blocking, o06.risk) == (
        "violated",
        "amount_rule_violated",
        True,
        "high",
    )
    assert amount.recommendation == "recommend_reject"
    injected = dev_quality_truth(SET["V107-D-8002"])
    assert injected.security_codes == ("input_privilege_attack",)
    assert injected.recommendation == "needs_human_review"
    currency = dev_quality_truth(SET["V107-D-8003"])
    assert currency.conflict_relations == ("order_invoice_currency_equal",)
    assert currency.recommendation == "needs_human_review"
    o11 = next(row for row in currency.expectations if row.obligation_id == "O-11")
    assert (o11.state, o11.reason) == ("violated", "source_conflict_proven")
    blocked = dev_quality_truth(SET["V107-D-8004"])
    o03 = next(row for row in blocked.expectations if row.obligation_id == "O-03")
    assert (o03.state, o03.reason, o03.blocking) == (
        "violated",
        "supplier_blocking_fact",
        True,
    )
    # The benign look-alike: an active registry beside a blocked procurement is a sourced
    # violation, not a conflict, so nothing escalates on the conflict channel.
    assert blocked.conflict_relations == ()
    assert blocked.security_codes == ()
    assert blocked.recommendation == "recommend_reject"


def test_the_truth_file_is_versioned_and_self_describing():
    body = dev_quality_truth_file()
    assert body["schema_version"] == DEV_QUALITY_TRUTH_VERSION
    assert body["gold_set_version"] == DEV_QUALITY_SET_VERSION
    assert len(body["truths"]) == 12


# ------------------------------------------------------------------ v2 fixture builders
#
# The fixture is a *complete* perfect reading: every claim a branch would make is cited
# from the passage that carries it, with the authorized source identity and the case's own
# object, the security/conflict events carry their expected locations and identities, and
# the metering/ledger pair is internally consistent.  Every negative case below changes
# exactly one thing, so a failing verdict is attributable to that thing alone.


def _extended_registration() -> dict:
    from hashlib import sha256

    from procurement_review.precise_dev_sweep import (
        EXTENSION_KEYS,
        SWEEP_TRUTH_EXTENSION_VERSION,
        binding_block,
        project_base,
    )
    from procurement_review.precise_candidate import truth_extension

    body = registration()
    chosen = tuple(
        case for case in dev_quality_cases() if case.case_id in body["case_ids"]
    )
    extended = dict(body)
    extended["truth_extension"] = {
        case.case_id: truth_extension(case) for case in chosen
    }
    extended["binding"] = binding_block(chosen)
    base_sha = sha256(
        json.dumps(
            project_base(extended), ensure_ascii=False, sort_keys=True, indent=2
        ).encode()
    ).hexdigest()
    extended["extends"] = {
        "base_registration_sha256": base_sha,
        "extension_version": SWEEP_TRUTH_EXTENSION_VERSION,
    }
    assert set(extended) - set(body) == set(EXTENSION_KEYS)
    return extended


def _line_bounds(text: str, needle: str) -> tuple[int, int] | None:
    """The line that carries a needle, as the [start, end) a span would cite."""

    for line in text.split("\n"):
        if needle and needle in line:
            where = text.find(line)
            return where, where + len(line)
    return None


def _bounds_cover(*ranges: tuple[int, int]) -> tuple[int, int]:
    return min(item[0] for item in ranges), max(item[1] for item in ranges)


def _spans_for(registration_body: dict, case_id: str, expectation: dict) -> list[dict]:
    """One span per registered claim, citing the passage that states it."""

    extension = registration_body["truth_extension"][case_id]
    material = registration_body["material_text"][case_id]
    sources = extension["branch_sources"]
    obligations = expectation["obligation_id"]
    fields = extension["support"].get(obligations, {})
    by_worker: dict[str, list[tuple[str, object]]] = {}
    for key, descriptor in fields.items():
        worker, field = key.split(".", 1)
        by_worker.setdefault(worker, []).append((field, descriptor))
    spans = []
    for worker, entries in by_worker.items():
        text = material[worker]
        ranges = []
        for field, descriptor in entries:
            if descriptor["kind"] == "literal":
                found = _line_bounds(text, str(descriptor["value"]))
            else:
                found = None
                for needle in descriptor.get("needles") or ():
                    found = _line_bounds(text, needle)
                    if found:
                        break
                if found is None:
                    found = (0, len(text.split("\n", 1)[0]))
            assert found is not None, (case_id, worker, field)
            ranges.append(found)
        clause_ranges = []
        for clause in expectation.get("document_clauses", ()):
            if clause in text:
                where = text.find(clause)
                clause_ranges.append((where, where + len(clause)))
        covered = (
            _bounds_cover(*ranges, *clause_ranges)
            if clause_ranges
            else _bounds_cover(*ranges)
        )
        start, end = covered
        for field, descriptor in entries:
            if descriptor["kind"] == "literal":
                value = descriptor["value"]
            elif descriptor["kind"] == "token":
                value = descriptor["value"]
            else:
                value = " ".join(descriptor.get("needles") or ())
            spans.append(
                {
                    "worker": worker,
                    "check_id": obligations,
                    "field": field,
                    "field_path": f"{worker}.{obligations.lower()}.{field}",
                    "subject_id": extension["subject_id"],
                    "source_version": sources[worker]["version"],
                    "value": value,
                    "start": start,
                    "end": end,
                    "source_id": sources[worker]["source_id"],
                    "version": sources[worker]["version"],
                    "locator": sources[worker]["locator"],
                }
            )
    return spans


def _observed_pair(pair: dict) -> dict:
    sides = {}
    for name in ("left", "right"):
        side = pair[name]
        sides[name] = {
            "proposition": {
                "subject_type": "purchase_order",
                "subject_id": side["subject_id"],
                "check_id": side["check_id"],
                "field_path": side["field_path"],
                "effective_at": side["effective_at"],
                "source_version": side["source_version"],
            },
            "value": side["value"],
            "evidence": [
                {
                    "source_id": side["source_id"],
                    "version": side["source_version"],
                    "locator": side["locator"],
                }
            ],
        }
    return {
        "pair_id": f"pair-{pair['relation']}",
        "kind": "rule_inconsistency",
        "relation": pair["relation"],
        **sides,
    }


def _fixture_observations(registration_body: dict) -> dict:
    from hashlib import sha256

    from procurement_review.precise_dev_sweep import project_base

    observations = []
    for truth_case in registration_body["truth"].values():
        case_id = truth_case["case_id"]
        extension = registration_body["truth_extension"][case_id]
        for repeat in (1, 2):
            spans = []
            findings = []
            for expectation in truth_case["expectations"]:
                spans.extend(_spans_for(registration_body, case_id, expectation))
                findings.append(
                    {
                        "obligation_id": expectation["obligation_id"],
                        "state": expectation["state"],
                        "reason": expectation["reason"],
                        "risk": expectation["risk"],
                        "blocking": expectation["blocking"],
                        "missing": [],
                        "evidence": [
                            {"source_id": locator.split(".")[0], "locator": locator}
                            for locator in expectation["fact_locators"]
                        ],
                    }
                )
            run_id = f"run-{case_id}-{repeat}"
            observations.append(
                {
                    "case_id": case_id,
                    "repeat": repeat,
                    "run_id": run_id,
                    "terminal_status": 200,
                    "status": "pending_confirmation",
                    "execution_mode": "real",
                    "execution_version": "review-execution.v2",
                    "rules_version": "3.1.0",
                    "summary_schema": "1.0",
                    "elapsed_ms": 10_000,
                    "service_created_at": "2026-10-07T03:00:00+00:00",
                    "observed_at": "2026-10-07T03:00:10+00:00",
                    "recommendation": truth_case["recommendation"],
                    "findings": findings,
                    "conflicts": [
                        _observed_pair(pair) for pair in extension["conflict_pairs"]
                    ],
                    "security": [
                        {
                            "code": event["code"],
                            "origin": event["origin"],
                            "location_sha256": event["location_sha256"],
                        }
                        for event in extension["security_events"]
                    ],
                    "branches": {
                        worker: {
                            "outcome": "completed",
                            "support_failures": [],
                            "attempts": [
                                {
                                    "attempt_no": 1,
                                    "error_code": None,
                                    "correctable": False,
                                }
                            ],
                        }
                        for worker in ("policy", "contract")
                    },
                    "metering": {
                        "attempts_opened": 2,
                        "physical_attempts": 2,
                        "proven_sends": 2,
                        "unresolved_attempts": 0,
                        "open_attempts": 0,
                        "unreturned_attempts": 0,
                        "first_sends": 2,
                        "contract_corrections": 0,
                        "citation_corrections": 0,
                        "transport_retries": 0,
                        "other_extra_attempts": 0,
                        "unknown_usage_attempts": 0,
                        "usage_complete": True,
                        "requested_model": registration_body["binding"][
                            "requested_model"
                        ],
                        "resolved_models": [
                            registration_body["binding"]["resolved_model"]
                        ],
                    },
                    "spans": spans,
                }
            )
    base_sha = sha256(
        json.dumps(
            project_base(registration_body),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode()
    ).hexdigest()
    return {
        "sweep_version": "v1-07g-dev-sweep.v1",
        "registration_sha256": base_sha,
        "observations": observations,
    }


def _derived_bindings(registration_body: dict, observations: dict) -> dict:
    """The ledger rows one observation's own metering describes, Run by Run."""

    binding = registration_body["binding"]
    runs = {}
    for row in observations["observations"]:
        attempts = []
        for worker, branch in (row.get("branches") or {}).items():
            for attempt in branch.get("attempts") or []:
                attempts.append(
                    {
                        "generation": 1,
                        "worker": worker,
                        "attempt_no": attempt.get("attempt_no"),
                        "outcome": (
                            "completed"
                            if attempt.get("error_code") is None
                            else "format_corrected"
                        ),
                        "abandoned": False,
                        "discarded_late": False,
                    }
                )
        metering = row.get("metering") or {}
        extra_open = int(metering.get("open_attempts") or 0)
        for index in range(extra_open):
            attempts.append(
                {
                    "generation": 1,
                    "worker": "policy",
                    "attempt_no": 90 + index,
                    "outcome": None,
                    "abandoned": False,
                    "discarded_late": False,
                }
            )
        runs[row["run_id"]] = {
            "case_id": row["case_id"],
            "status": row["status"],
            "implementation_sha256": binding["product_implementation_sha256"],
            "rules_version": binding["expected_rules_version"],
            "document_version": binding["expected_document_version"],
            "dataset_version": binding["expected_dataset_version"],
            "index_version": binding["expected_index_version"],
            "resolved_model": binding["resolved_model"],
            "requested_model": binding["requested_model"],
            "attempts": attempts,
            "receipt_count": 0,
        }
    return {
        "database": "fixture",
        "generated_at": "2026-10-07T03:01:00+00:00",
        "runs": runs,
    }


def _score(body: dict, observations: dict, bindings: dict | None = None) -> dict:
    if bindings is None:
        bindings = _derived_bindings(body, observations)
    return score_sweep(body, observations, bindings)


def _mutate(observations: dict, case_id: str, repeat: int, **changes) -> dict:
    rows = observations["observations"]
    for index, row in enumerate(rows):
        if row["case_id"] == case_id and row["repeat"] == repeat:
            rows[index] = {**row, **changes}
    return observations


# ------------------------------------------------------------------------ the readings


def test_a_perfect_sweep_passes_every_gate():
    body = _extended_registration()
    observations = _fixture_observations(body)
    result = _score(body, observations)
    assert result["failed_gates"] == [], result["failed_gates"]
    assert result["stage_acceptance"] is True
    assert set(result["repeats"]) == {1, 2}
    for report in result["repeats"].values():
        assert report["q01_first_answer"]["value"] == 1.0
        assert report["q02_obligations"]["value"] == 1.0
        assert report["q02_obligations"]["supported_subset_value"] == 1.0
        assert report["q03_recommendation"]["macro_f1"] == 1.0
        assert all(report["q03_recommendation"]["class_recall"].values())
        assert report["q06_events"]["conflict_precision"] == 1.0
        assert report["q06_events"]["security_recall"] == 1.0
        assert report["q07_citations"]["semantic_support"] == 1.0
        assert report["q07_citations"]["adopted_propositions"] > 0
    assert result["completeness"]["complete"] is True
    assert result["latency"]["samples"] == result["latency"]["planned_samples"] == 24
    assert result["inputs"]["matched"] is True


def test_the_registration_binds_the_mapped_material_and_truth():
    body = _extended_registration()
    assert body["plan"] == {"cases": 12, "repeats": 2, "runs": 24}
    assert body["candidate_sha256"]
    for case_id in CASE_IDS:
        truth_case = body["truth"][case_id]
        mapped = mapped_truth(SET[case_id])
        assert truth_case["recommendation"] == mapped["recommendation"]
        assert truth_case["material"] == mapped["material"]
        text = body["material_text"][case_id]["contract"]
        assert body["truth"][case_id]["expectations"][4]["values"]["order"] in text
        extension = body["truth_extension"][case_id]
        # The extension names the product-shaped object the Case's spans must carry.
        mapped_order = id_map(SET[case_id])[
            str(SET[case_id].facts["purchase_order_id"])
        ]
        assert extension["subject_id"] == mapped_order
        assert set(extension["branch_sources"]) == {"policy", "contract"}
        for source in extension["branch_sources"].values():
            assert source["locator"].startswith("chunk-")
        assert extension["support"]["O-05"]["contract.contract"]["kind"] == "literal"
    assert body["binding"]["product_implementation_sha256"]
    # The extension is additive: the pre-registered base is reproduced byte for byte.
    from procurement_review.precise_dev_sweep import project_base

    assert json.dumps(
        project_base(body), ensure_ascii=False, sort_keys=True
    ) == json.dumps(registration(), ensure_ascii=False, sort_keys=True)


def test_a_wrong_recommendation_fails_only_the_repeat_it_happened_in():
    body = _extended_registration()
    observations = _mutate(
        _fixture_observations(body), "V107-D-9001", 1, recommendation="recommend_reject"
    )
    result = _score(body, observations)
    assert "repeat-1/Q-03-class-recall/recommend_approve" in result["failed_gates"]
    assert not [name for name in result["failed_gates"] if name.startswith("repeat-2/")]
    assert (
        result["repeats"][1]["q03_recommendation"]["class_counts"]["recommend_approve"][
            "fn"
        ]
        == 1
    )
    assert result["repeats"][2]["q03_recommendation"]["value"] == 1.0


def test_a_spent_correction_is_not_reported_as_a_valid_first_answer():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9007" and row["repeat"] == 1:
            row["branches"]["contract"]["attempts"] = [
                {
                    "attempt_no": 1,
                    "error_code": "citation_proposition_unsupported",
                    "correctable": True,
                },
                {"attempt_no": 2, "error_code": None},
            ]
            row["metering"] = {
                **row["metering"],
                "attempts_opened": 3,
                "physical_attempts": 3,
                "proven_sends": 3,
                "first_sends": 2,
                "citation_corrections": 1,
            }
    result = _score(body, observations)
    assert "repeat-1/Q-01" in result["failed_gates"]
    assert result["repeats"][1]["q01_first_answer"]["corrected_after_refusal"] == 1
    assert result["repeats"][2]["q01_first_answer"]["value"] == 1.0


def test_a_verdict_with_no_denominator_is_undecidable_not_passing():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for truth_case in body["truth"].values():
        if truth_case["recommendation"] == "recommend_approve":
            truth_case["recommendation"] = "recommend_reject"
    for row in observations["observations"]:
        if row["recommendation"] == "recommend_approve":
            row["recommendation"] = "recommend_reject"
    result = _score(body, observations)
    verdict = result["gates"]["repeat-1"]["Q-04-precision"]
    assert verdict["value"] is None
    assert verdict["undecidable"] is True
    assert verdict["meets"] is None
    assert "repeat-1/Q-04-precision" in result["failed_gates"]


def test_a_conflict_pair_that_does_not_match_the_truth_fails_recall():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-8003":
            row["conflicts"] = []
    result = _score(body, observations)
    assert "repeat-1/Q-06-conflict-recall" in result["failed_gates"]
    report = result["repeats"][1]["q06_events"]
    assert report["conflict_expected"] == 1
    assert report["conflict_matched"] == 0
    assert result["gates"]["repeat-1"]["Q-06-conflict-precision"]["undecidable"] is True


def test_a_conflict_pair_with_another_object_or_source_does_not_match():
    """The same relation on the wrong order, value or source is not a match."""

    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-8003" and row["repeat"] == 1:
            for side in ("left", "right"):
                item = row["conflicts"][0][side]
                item["value"] = "XXX"
                item["proposition"]["subject_id"] = "PO-999"
                item["evidence"][0]["source_id"] = "PO-999"
                item["evidence"][0]["version"] = "bogus.version"
    result = _score(body, observations)
    assert "repeat-1/Q-06-conflict-precision" in result["failed_gates"]
    assert "repeat-1/Q-06-conflict-recall" in result["failed_gates"]
    assert result["repeats"][1]["q06_events"]["conflict_matched"] == 0
    # The untouched repeat still passes.
    assert result["repeats"][2]["q06_events"]["conflict_matched"] == 1


def test_an_additional_predicted_conflict_pair_costs_precision():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-8003" and row["repeat"] == 1:
            extra = json.loads(json.dumps(row["conflicts"][0]))
            extra["relation"] = "invented_relation"
            extra["pair_id"] = "pair-bogus"
            row["conflicts"].append(extra)
    result = _score(body, observations)
    assert "repeat-1/Q-06-conflict-precision" in result["failed_gates"]
    assert result["repeats"][1]["q06_events"]["conflict_predicted"] == 2
    assert result["repeats"][1]["q06_events"]["conflict_matched"] == 1


def test_a_security_event_at_the_wrong_location_or_origin_fails_recall():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-8002" and row["repeat"] == 1:
            row["security"][0]["location_sha256"] = "00" * 32
        if row["case_id"] == "V107-D-8002" and row["repeat"] == 2:
            row["security"][0]["origin"] = "output"
    result = _score(body, observations)
    assert "repeat-1/Q-06-security-recall" in result["failed_gates"]
    assert "repeat-2/Q-06-security-recall" in result["failed_gates"]
    report = result["repeats"][1]["q06_events"]
    # The denominator is the case that really carries an event, and the benign rate's
    # denominator is the cases that do not.
    assert report["security_events_expected"] == 1
    assert report["security_cases"] == 1
    assert report["benign_cases"] == 11


def test_an_escalating_event_on_a_case_without_truth_events_fails_benign_rate():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9001":
            row["security"] = [
                {
                    "code": "pii_output_blocked",
                    "origin": "output",
                    "location_sha256": "ab" * 32,
                }
            ]
    result = _score(body, observations)
    assert "repeat-1/Q-06-benign-events" in result["failed_gates"]
    assert result["repeats"][1]["q06_events"]["benign_events"] == 1
    assert result["repeats"][1]["q06_events"]["benign_cases"] == 11


def test_a_fabricated_claim_is_reported_and_a_pattern_of_them_fails_the_gates():
    """A wrong subject is judged per claim; when it repeats, Q-02 and Q-07 fail."""

    body = _extended_registration()
    single = _fixture_observations(body)
    for row in single["observations"]:
        if row["case_id"] == "V107-D-9007" and row["repeat"] == 1:
            for span in row["spans"]:
                if span["check_id"] == "O-05" and span["field"] == "subject":
                    span["value"] = "采购十艘航空母舰"
    result = _score(body, single)
    report = result["repeats"][1]
    assert report["q02_obligations"]["numerator"] == 143
    assert report["q02_obligations"]["denominator"] == 144
    assert report["q07_citations"]["semantic_support"] < 1.0
    assert report["q07_citations"]["unsupported_propositions"] == 1
    wrong = [
        item
        for item in report["obligations"]
        if item.get("case_id") == "V107-D-9007" and item.get("obligation_id") == "O-05"
    ]
    assert wrong and wrong[0]["correct"] is False
    assert "contract.subject:claim_missing_needle" in wrong[0]["claim_failures"]
    multi = _fixture_observations(body)
    for row in multi["observations"]:
        if row["repeat"] == 1:
            for span in row["spans"]:
                if span["check_id"] == "O-05" and span["field"] == "subject":
                    span["value"] = "采购十艘航空母舰"
    result = _score(body, multi)
    assert "repeat-1/Q-02" in result["failed_gates"]
    assert "repeat-1/Q-02-supported-subset" in result["failed_gates"]
    assert "repeat-1/Q-07-semantic-support" in result["failed_gates"]
    assert not [name for name in result["failed_gates"] if name.startswith("repeat-2/")]


def test_a_claim_about_another_object_is_an_out_of_scope_citation():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9007" and row["repeat"] == 1:
            for span in row["spans"]:
                if span["check_id"] == "O-05" and span["field"] == "contract":
                    span["value"] = "CONTRACT-909"
                    span["subject_id"] = "PO-909"
    result = _score(body, observations)
    assert "repeat-1/Q-07-out-of-scope" in result["failed_gates"]
    assert result["repeats"][1]["q07_citations"]["out_of_scope_citations"] == 1
    assert result["repeats"][2]["q07_citations"]["out_of_scope_citations"] == 0


def test_a_claim_cited_from_an_unauthorized_source_or_version_is_out_of_scope():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9007" and row["repeat"] == 1:
            for span in row["spans"]:
                if span["check_id"] == "O-05" and span["field"] == "contract":
                    span["source_id"] = "UNAUTHORIZED-SOURCE"
                    span["version"] = "wrong.version"
                    span["source_version"] = "wrong.version"
        if row["case_id"] == "V107-D-9008" and row["repeat"] == 1:
            for span in row["spans"]:
                if span["check_id"] == "O-09" and span["field"] == "cap":
                    span["locator"] = "chunk-" + "0" * 64
    result = _score(body, observations)
    assert "repeat-1/Q-07-out-of-scope" in result["failed_gates"]
    assert result["repeats"][1]["q07_citations"]["out_of_scope_citations"] == 2
    assert not [name for name in result["failed_gates"] if name.startswith("repeat-2/")]


def test_a_conclusion_cited_from_the_wrong_clause_fails_the_obligation():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["repeat"] == 1 and row["case_id"] in {
            "V107-D-9001",
            "V107-D-9007",
            "V107-D-8001",
        }:
            for span in row["spans"]:
                if span["check_id"] == "O-05":
                    span["start"], span["end"] = 0, 4
    result = _score(body, observations)
    rows = [
        item
        for item in result["repeats"][1]["obligations"]
        if item.get("obligation_id") == "O-05" and item.get("case_id") == "V107-D-9001"
    ]
    assert rows and rows[0]["correct"] is False
    assert rows[0]["claim_failures"]
    assert "repeat-1/Q-02" in result["failed_gates"]
    assert not [name for name in result["failed_gates"] if name.startswith("repeat-2/")]


def test_an_unregistered_claim_field_is_never_a_silent_pass():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9007" and row["repeat"] == 1:
            span = dict(row["spans"][0])
            span["field"] = "not_a_registered_field"
            span["field_path"] = "policy.not_a_registered_field"
            row["spans"].append(span)
    result = _score(body, observations)
    report = result["repeats"][1]
    assert report["q07_citations"]["unsupported_propositions"] == 1
    assert report["q07_citations"]["semantic_support"] < 1.0


def test_a_replaced_observation_is_not_a_complete_ledger():
    body = _extended_registration()
    observations = _fixture_observations(body)
    source = next(
        row
        for row in observations["observations"]
        if row["case_id"] == "V107-D-9001" and row["repeat"] == 1
    )
    rows = observations["observations"]
    for index, row in enumerate(rows):
        if row["case_id"] == "V107-D-9007" and row["repeat"] == 1:
            rows[index] = json.loads(json.dumps(source))
    result = _score(body, observations)
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
    assert result["completeness"]["complete"] is False
    assert ["V107-D-9007", 1] in result["completeness"]["missing_pairs"]
    assert any(
        entry[0] == ["V107-D-9001", 1]
        for entry in result["completeness"]["duplicate_pairs"]
    )


def test_a_missing_observation_is_not_a_complete_ledger():
    body = _extended_registration()
    observations = _fixture_observations(body)
    observations["observations"] = [
        row
        for row in observations["observations"]
        if not (row["case_id"] == "V107-D-9003" and row["repeat"] == 2)
    ]
    result = _score(body, observations)
    assert result["completeness"]["complete"] is False
    assert ["V107-D-9003", 2] in result["completeness"]["missing_pairs"]


def test_missing_metering_is_a_named_failure():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        row.pop("metering", None)
    bindings = _derived_bindings(body, _fixture_observations(body))
    result = _score(body, observations, bindings)
    assert result["completeness"]["complete"] is False
    assert any(
        "metering_missing" in item for item in result["completeness"]["failures"]
    )
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]


def test_a_missing_timing_sample_is_not_silently_dropped():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9003" and row["repeat"] == 1:
            row["service_created_at"] = None
    result = _score(body, observations)
    latency = result["latency"]
    assert latency["samples"] == 23 and latency["planned_samples"] == 24
    assert latency["missing_samples"]
    for repeat in (1, 2):
        assert f"repeat-{repeat}/Q-09-p95" in result["failed_gates"]
        assert f"repeat-{repeat}/Q-09-deadline" in result["failed_gates"]
        assert result["gates"][f"repeat-{repeat}"]["Q-09-p95"]["undecidable"] is True


def test_a_run_bound_to_another_build_is_not_the_registered_candidate():
    body = _extended_registration()
    observations = _fixture_observations(body)
    bindings = _derived_bindings(body, observations)
    first = bindings["runs"]["run-V107-D-9001-1"]
    first["implementation_sha256"] = "0" * 64
    result = _score(body, observations, bindings)
    assert result["completeness"]["complete"] is False
    assert any("run_binding" in item for item in result["completeness"]["failures"])
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]


def test_a_run_without_its_ledger_rows_is_not_a_complete_ledger():
    body = _extended_registration()
    observations = _fixture_observations(body)
    bindings = _derived_bindings(body, observations)
    del bindings["runs"]["run-V107-D-9001-1"]
    result = _score(body, observations, bindings)
    assert any(
        "binding_missing_run" in item for item in result["completeness"]["failures"]
    )


def test_metering_that_disagrees_with_the_ledger_is_a_failure():
    body = _extended_registration()
    observations = _fixture_observations(body)
    bindings = _derived_bindings(body, observations)
    bindings["runs"]["run-V107-D-9001-1"]["attempts"] = bindings["runs"][
        "run-V107-D-9001-1"
    ]["attempts"][:1]
    result = _score(body, observations, bindings)
    assert any("ledger_mismatch" in item for item in result["completeness"]["failures"])


def test_an_attempt_over_the_plan_bound_is_a_failure():
    body = _extended_registration()
    observations = _fixture_observations(body)
    bindings = _derived_bindings(body, observations)
    run = bindings["runs"]["run-V107-D-9001-1"]
    run["attempts"].append(dict(run["attempts"][0], attempt_no=3))
    row = next(
        item
        for item in observations["observations"]
        if item["run_id"] == "run-V107-D-9001-1"
    )
    row["metering"] = {
        **row["metering"],
        "attempts_opened": 3,
        "physical_attempts": 3,
        "proven_sends": 3,
    }
    result = _score(body, observations, bindings)
    assert any(
        "plan_attempt_bound" in item for item in result["completeness"]["failures"]
    )


def test_a_registration_that_does_not_match_the_observations_fails_closed():
    body = _extended_registration()
    observations = _fixture_observations(body)
    observations["registration_sha256"] = "1" * 64
    result = _score(body, observations)
    assert result["stage_acceptance"] is False
    assert "input_binding" in result["failed_gates"]


def test_an_inconsistent_repeat_pair_fails_the_cross_repeat_gate():
    body = _extended_registration()
    observations = _mutate(
        _fixture_observations(body),
        "V107-D-9006",
        2,
        recommendation="recommend_approve",
    )
    result = _score(body, observations)
    assert "cross-repeat/Q-10-consistency" in result["failed_gates"]
    assert result["repeat_consistency"]["numerator"] == 11


def test_a_non_zero_refusal_count_fails_its_zero_tolerance_gate():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9001" and row["repeat"] == 1:
            row["branches"]["contract"]["attempts"] = [
                {
                    "attempt_no": 1,
                    "error_code": "citation_outside_branch",
                    "correctable": False,
                }
            ]
            row["metering"] = {
                **row["metering"],
                "attempts_opened": 2,
                "physical_attempts": 2,
                "proven_sends": 2,
                "first_sends": 2,
            }
    result = _score(body, observations)
    assert result["repeats"][1]["q07_citations"]["out_of_scope_citations"] == 1
    assert "repeat-1/Q-07-out-of-scope" in result["failed_gates"]
    assert result["gates"]["repeat-1"]["Q-07-out-of-scope"]["meets"] is False
    assert result["gates"]["repeat-2"]["Q-07-out-of-scope"]["meets"] is True


def test_a_released_high_risk_case_fails_its_zero_tolerance_gate():
    body = _extended_registration()
    observations = _fixture_observations(body)
    result = _score(
        body,
        _mutate(observations, "V107-D-8001", 1, recommendation="recommend_approve"),
    )
    assert result["repeats"][1]["q04_approval"]["high_risk_release"] == 1
    assert "repeat-1/Q-04-high-risk-release" in result["failed_gates"]
    assert result["gates"]["repeat-1"]["Q-04-high-risk-release"]["meets"] is False


def test_a_wrong_duty_value_refusal_is_counted_and_reported():
    body = _extended_registration()
    observations = _fixture_observations(body)
    for row in observations["observations"]:
        if row["case_id"] == "V107-D-9005" and row["repeat"] == 2:
            row["branches"]["policy"]["support_failures"] = [
                "citation_duty_value_invalid"
            ]
            row["branches"]["policy"]["attempts"] = []
            row["metering"] = {
                **row["metering"],
                "attempts_opened": 1,
                "physical_attempts": 1,
                "proven_sends": 1,
                "first_sends": 1,
            }
    result = _score(body, observations)
    assert result["repeats"][2]["q07_citations"]["duty_value_invalid_refusals"] == 1
    assert result["repeats"][1]["q07_citations"]["duty_value_invalid_refusals"] == 0


def test_the_score_recomputes_from_its_own_inputs(tmp_path: Path):
    body = _extended_registration()
    observations = _fixture_observations(body)
    bindings = _derived_bindings(body, observations)
    first = score_sweep(body, observations, bindings)
    again = score_sweep(body, observations, bindings)
    assert first == again
    stored = tmp_path / "score.json"
    stored.write_text(json.dumps(first, sort_keys=True))
    tampered = json.loads(stored.read_text())
    tampered["gates"]["repeat-1"]["Q-02"] = {"value": 1.0, "meets": True}
    stored.write_text(json.dumps(tampered, sort_keys=True))
    reread = json.loads(stored.read_text())
    assert reread["gates"] != again["gates"]


def test_an_extra_or_wrongly_repeated_observation_is_not_a_complete_ledger():
    body = _extended_registration()
    observations = _fixture_observations(body)
    extra = json.loads(json.dumps(observations["observations"][0]))
    extra["run_id"] = "run-extra"
    extra["repeat"] = 3
    observations["observations"].append(extra)
    result = _score(body, observations)
    assert result["completeness"]["complete"] is False
    assert ["V107-D-9001", 3] in result["completeness"]["extra_pairs"]
    assert "cross-repeat/Q-10-ledger-complete" in result["failed_gates"]
