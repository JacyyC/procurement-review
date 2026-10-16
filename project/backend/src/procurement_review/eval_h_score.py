"""H score v2: planned denominators, native pipeline failures and per-repeat gates."""

import math
from pathlib import Path

from .eval_h_consumption import completion_proof, normal_receipt
from .eval_h_contracts import ARMS, H_SCORE
from .eval_h_journal import Journal
from .eval_h_material import truth_registration
from .eval_h_material import source_identities
from .eval_h_seal import HSeal, plan_digest
from .eval_h_semantics import judge
from .eval_metrics import wilson_interval
from .gold_v3_contracts import payload_sha256
from .precise_dev_score import score_repeat


def rate(numerator, denominator, threshold=None, *, maximum=False):
    value = numerator / denominator if denominator else None
    return dict(
        numerator=numerator,
        denominator=denominator,
        value=value,
        wilson_95=list(wilson_interval(numerator, denominator))
        if denominator
        else None,
        meets_threshold=(value <= threshold if maximum else value >= threshold)
        if value is not None and threshold is not None
        else None,
    )


def projected_branches(row):
    """Readable branch entries of a projected row; unreadable ones carry no signal."""

    branches = row.get("branches")
    if type(branches) is not dict:
        return ()
    return tuple(item for item in branches.values() if type(item) is dict)


def failure_codes(row):
    """Failure codes a row spells out, ignoring entries it cannot read as text."""

    codes = set()
    recorded = row.get("pipeline_failures")
    if type(recorded) in (list, tuple):
        codes.update(item for item in recorded if type(item) is str)
    error = row.get("error_code")
    if type(error) is str:
        codes.add(error)
    codes.update(
        branch["failure_code"]
        for branch in projected_branches(row)
        if type(branch.get("failure_code")) is str
    )
    return codes


def pipeline_failed(row):
    return (
        row.get("terminal_status", 200) != 200
        or row.get("status") not in {"pending_confirmation", "pending_human_review"}
        or bool(row.get("pipeline_failures"))
        or any(
            branch.get("outcome") in {"failed", "no_attempts_left", "stopped"}
            for branch in projected_branches(row)
        )
    )


def observation_row(observation):
    """The projected row of an observation, empty when the record is malformed."""

    row = observation.get("row") if hasattr(observation, "get") else None
    return row if type(row) is dict else {}


def censored(observation):
    row = observation_row(observation)
    codes = failure_codes(row)
    attempts = observation.get("attempts")
    last = {}
    for attempt in attempts if type(attempts) is list else ():
        if type(attempt) is not dict:
            continue
        branch, number = attempt.get("branch"), attempt.get("attempt_no")
        if type(branch) is not str or type(number) is not int:
            continue
        if number > last.get(branch, {}).get("attempt_no", 0):
            last[branch] = attempt
    codes.update(
        receipt["error_code"]
        for receipt in (
            normal_receipt(attempt, observation.get("arm")) for attempt in last.values()
        )
        if type(receipt.get("error_code")) is str
    )
    return pipeline_failed(row) and (
        row.get("status") == "cancelled"
        or bool(
            codes & {"timeout", "deadline", "deadline_exhausted", "interrupted_unknown"}
        )
    )


def deadline_gates(plan):
    """The Q-09 deadline check and p95 gate one plan is judged by (Spec P-04/Q-09a/b).

    A V1-08 plan registers one of the two accepted Run deadlines and reads the matching
    gate; the historical schema keeps the frozen 120-second check and 90-second p95 the
    old reports were computed with, so no old report is re-explained by a new rule.
    """

    from .eval_h_contracts import plan_generation_deadline

    if plan_generation_deadline(plan) == 180.0:
        return 180000, 150000
    return 120000, 90000


def latency(rows, count, *, deadline_ms=120000, p95_ms=90000):
    all_values = [row.get("elapsed_ms") for row in rows]
    valid = len(rows) == count and all(
        type(value) is int and value >= 0 for value in all_values
    )
    values = sorted(all_values) if valid else []
    success = [
        row.get("elapsed_ms")
        for row in rows
        if not pipeline_failed(observation_row(row))
    ]
    success = (
        sorted(success)
        if all(type(value) is int and value >= 0 for value in success)
        else []
    )

    def at(data):
        return data[math.ceil(len(data) * 0.95) - 1] if data else None

    return dict(
        planned=count,
        measured=len(rows),
        all_sample_p95_ms=at(values),
        success_p95_ms=at(success),
        maximum_ms=max(values) if values else None,
        failed=sum(pipeline_failed(observation_row(row)) for row in rows),
        cancelled=sum(
            observation_row(row).get("status") == "cancelled" for row in rows
        ),
        censored=sum(censored(row) for row in rows),
        deadline_ms=deadline_ms,
        deadline_satisfied=max(values) <= deadline_ms if values else None,
        meets_threshold=(at(values) <= p95_ms) if values else None,
    )


def repeat_score(plan, cases, labels, observations, seed):
    registration = truth_registration(plan, cases, labels)
    labels = {label.case_id: label for label in labels}
    by_key = {
        (observation["case_id"], observation["arm"]): observation
        for observation in observations
        if observation["repeat_seed"] == seed
    }
    result = {}
    for arm in ARMS:
        selected, semantics, raw_rows = [], {}, []
        for case in cases:
            observation = by_key.get((case.case_id, arm))
            row = dict(observation["row"]) if observation else {}
            row["model_sources"] = {
                role: (observation or {})
                .get("retrieval", {})
                .get(role, {})
                .get("model_sources", [])
                for role in ("policy", "contract")
            }
            row["native_security"] = (
                (observation or {})
                .get("binding", {})
                .get("execution", {})
                .get("security", [])
            )
            row.update(case_id=case.case_id, repeat=seed)
            raw_rows.append(row)
            semantics[case.case_id] = judge(
                plan, case, labels[case.case_id], row, registration
            )
            if observation:
                selected.append(observation)
        metrics = score_repeat(registration, {"observations": raw_rows}, seed)
        metrics["q09_pipeline"] = dict(
            non_expected_pipeline_failures=sum(
                pipeline_failed(row) for row in raw_rows
            ),
            planned_runs=len(cases),
        )
        duties = [
            obligation
            for item in semantics.values()
            for obligation in item["obligations"]
        ]
        supported = [duty for duty in duties if duty["expected_state"] == "supported"]
        q02 = dict(
            all=rate(sum(duty["correct"] for duty in duties), len(duties), 0.98),
            supported=rate(
                sum(duty["correct"] for duty in supported), len(supported), 0.98
            ),
        )
        total_claims = sum(item["propositions"] for item in semantics.values())
        supported_claims = sum(item["supported"] for item in semantics.values())
        q07 = dict(
            semantic=rate(supported_claims, total_claims, 0.98),
            unsupported=rate(
                total_claims - supported_claims, total_claims, 0.02, maximum=True
            ),
            outside=sum(item["outside"] for item in semantics.values()),
        )
        first, received, final, correction_count = 0, 0, 0, 0
        for observation in selected:
            receipts = [
                (attempt["branch"], attempt["attempt_no"], normal_receipt(attempt, arm))
                for attempt in observation["attempts"]
            ]
            semantic = semantics[observation["case_id"]]
            for branch in (
                ("policy", "contract") if arm == "multi_agent" else ("joint",)
            ):
                own = [
                    receipt
                    for name, number, receipt in receipts
                    if name == branch and number == 1
                ]
                field_verdicts = [
                    verdict
                    for verdict in semantic["spans"]
                    if branch == "joint" or verdict["worker"] == branch
                ]
                expected = [
                    support
                    for support in labels[observation["case_id"]].support
                    if support.requirement == "required"
                    and (branch == "joint" or support.role == branch)
                ]
                valid = all(verdict["supported"] for verdict in field_verdicts) and all(
                    sum(
                        verdict["worker"] == support.role
                        and verdict["check_id"] == support.obligation_id
                        and verdict["field"] == support.field
                        and verdict["supported"]
                        for verdict in field_verdicts
                    )
                    == 1
                    for support in expected
                )
                first += bool(own and own[0]["error_code"] is None and valid)
                received += bool(own and own[0].get("returned"))
                final += bool(
                    valid
                    and any(
                        name == branch and receipt["error_code"] is None
                        for name, _, receipt in receipts
                    )
                )
                correction_count += sum(
                    name == branch and number > 1 for name, number, _ in receipts
                )
        measured_stops = sorted(
            observation["row"]["stop_reason"]
            for observation in selected
            if observation["row"].get("stop_reason")
        )
        count = len(cases) * (2 if arm == "multi_agent" else 1)
        q01 = dict(
            first=rate(first, count, 0.99),
            received=rate(first, received),
            final=rate(final, count),
            extra_attempts=correction_count,
        )
        q08 = {}
        for role in ("policy", "contract"):
            for variant in ("original", "without_case_id"):
                denominator, hits, false_hits, outside, wrong_version, invalid = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                for case in cases:
                    required = labels[case.case_id].required_sources[role]
                    observation = by_key.get((case.case_id, arm))
                    actual = (
                        (observation or {})
                        .get("retrieval", {})
                        .get(role, {})
                        .get(variant, [])
                    )
                    denominator += len(required)
                    identities = {
                        key: value
                        for key, value in source_identities(plan, case).items()
                        if key
                        in {
                            document.source_id
                            for document in case.documents
                            if document.role == role
                        }
                    }
                    hits += sum(
                        any(hit == identities[source] for hit in actual[:5])
                        for source in required
                    )
                    false_hits += not required and bool(actual)
                    for hit in actual:
                        source = identities.get(hit.get("source_id"))
                        outside += source is None
                        wrong_version += (
                            source is not None
                            and hit.get("version") != source["version"]
                        )
                        invalid += source is not None and payload_sha256(
                            hit
                        ) != payload_sha256(source)
                q08[role + ":" + variant] = {
                    **rate(hits, denominator, 0.95),
                    "negative_false_recall": false_hits,
                    "out_of_scope": outside,
                    "wrong_version": wrong_version,
                    "invalid_identity": invalid,
                }
        deadline_ms, p95_ms = deadline_gates(plan)
        q09 = latency(selected, len(cases), deadline_ms=deadline_ms, p95_ms=p95_ms)
        q03, q04, q05, q06 = (
            metrics[name]
            for name in (
                "q03_recommendation",
                "q04_approval",
                "q05_escalation",
                "q06_events",
            )
        )

        def ge(value, threshold):
            return value is not None and value >= threshold

        def le(value, threshold):
            return value is not None and value <= threshold

        gates = {
            "Q-01": q01["first"]["meets_threshold"] is True,
            "Q-02": all(item["meets_threshold"] is True for item in q02.values()),
            "Q-03": ge(q03["value"], 0.90)
            and ge(q03["macro_f1"], 0.90)
            and all(ge(value, 0.85) for value in q03["class_recall"].values()),
            "Q-04": ge(q04["approve_precision"], 0.98)
            and q04["high_risk_release"] == 0,
            "Q-05": ge(q05["recall"], 0.95) and le(q05["extra_takeover"], 0.10),
            "Q-06": ge(q06["conflict_precision"], 0.95)
            and ge(q06["conflict_recall"], 0.95)
            and ge(q06["security_recall"], 0.95)
            and le(q06["benign_event_rate"], 0.02),
            "Q-07": q07["semantic"]["meets_threshold"] is True
            and q07["unsupported"]["meets_threshold"] is True
            and q07["outside"] == 0,
            "Q-08": all(
                item["meets_threshold"] is True
                and item["out_of_scope"] == 0
                and item["wrong_version"] == 0
                and item["invalid_identity"] == 0
                for item in q08.values()
            ),
            "Q-09": q09["meets_threshold"] is True
            and q09["deadline_satisfied"] is True
            and metrics["q09_pipeline"]["non_expected_pipeline_failures"]
            <= math.floor(0.03 * len(cases)),
        }
        result[arm] = dict(
            metrics=metrics,
            measured_stops=measured_stops,
            q01=q01,
            q02=q02,
            q07=q07,
            q08=q08,
            q09=q09,
            gates=gates,
            semantic_verdicts=semantics,
        )
    return result


def rebuild(directory: Path):
    seal = HSeal(directory)
    plan, cases, labels = seal.plan(), seal.runner_cases(), seal.scorer_labels()
    journal = Journal(directory)
    observations = [
        body for pair in plan.observation_keys() if (body := journal.committed(*pair))
    ]
    proof = completion_proof(directory)
    repeats = {
        str(seed): repeat_score(plan, cases, labels, observations, seed)
        for seed in plan.repeat_seeds
    }
    consistency = {}
    for arm in ARMS:
        rows = {
            (body["case_id"], body["repeat_seed"]): body["row"].get("recommendation")
            for body in observations
            if body["arm"] == arm
        }
        consistency[arm] = rate(
            sum(
                rows.get((case, plan.repeat_seeds[0])) is not None
                and rows.get((case, plan.repeat_seeds[0]))
                == rows.get((case, plan.repeat_seeds[1]))
                for case in plan.case_ids
            ),
            len(plan.case_ids),
            0.95,
        )
    return dict(
        schema_version=H_SCORE,
        plan_sha256=plan_digest(plan),
        candidate_sha256=plan.candidate_sha256,
        consumer_sha256=plan.consumer_sha256,
        completion=proof,
        repeats=repeats,
        consistency=consistency,
        mode=plan.mode,
        stage_acceptance=False,
        score_sha256=payload_sha256(
            dict(repeats=repeats, consistency=consistency, completion=proof)
        ),
    )
