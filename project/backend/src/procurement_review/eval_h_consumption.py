"""H completion is recomputed from every registered durable observation, before labels."""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from .eval_h_batch_stop import read_batch_stop
from .eval_h_projection import project
from .eval_g_product_summary import execution_contract_failures
from .eval_h_contracts import (
    H_CONSUMPTION,
    plan_generation_deadline,
)
from .eval_h_identity import require_identity
from .eval_h_journal import Journal
from .eval_h_material import source_identities
from .eval_h_seal import HSeal, plan_digest
from .gold_v3_contracts import payload_sha256
from .precise_dev_sweep import _summarise
from .precise_run_contracts import PreciseBranchCommitV1, PreciseCheckpointPayloadV1
from .precise_run_runner import (
    DEFAULT_BRANCH_TIMEOUT_SECONDS,
    MAX_MODEL_ATTEMPTS_PER_BRANCH,
)
from .precise_run_scope import execution_snapshot
from .run_stop import BATCH, RUN_FAILURE_CODES, stop_level
from .trace import canonical_checksum


#: How far the two wall-clock readings of one observation may differ from the measured
#: wall time before the record is treated as a forged clock rather than a slow commit.
CLOCK_TOLERANCE_MS = 5_000
#: How far the measured elapsed time of an observation may differ from the two persisted
#: Run timestamps it must have been computed from.  A forged or erased reading is refused
#: by name instead of being accepted as a fast failure.
ELAPSED_TOLERANCE_MS = 1
#: The product statuses one Run can legitimately end a measured observation in.
TERMINAL_RUN_STATUSES = frozenset({"failed", "cancelled", "interrupted"})
#: The generation closure `mark_failed`/`mark_interrupted` write.
TERMINAL_GENERATION_STATUS = {
    "failed": "cancelled",
    "interrupted": "expired",
}


def same(left, right):
    return payload_sha256(left) == payload_sha256(right)


def normal_receipt(attempt, arm):
    raw = attempt.get("receipt")
    if type(raw) is not dict:
        raw = {}
    if arm == "multi_agent":
        return dict(
            error_code=attempt.get("effective_error", raw.get("attempt_error")),
            input_tokens=raw.get("input_usage"),
            output_tokens=raw.get("output_usage"),
            usage_unknown=raw.get("usage_unknown"),
            resolved_model=raw.get("receipt_model"),
            latency_ms=raw.get("latency_ms"),
            returned=type(raw.get("response_bytes")) is int
            and raw["response_bytes"] > 0,
            finish_reason=raw.get("finish_reason"),
            http_status=raw.get("http_status"),
            request_sha256=raw.get("request_sha256"),
            has_receipt=raw.get("attempt_error") != "interrupted_unknown",
        )
    return {**raw, "has_receipt": raw.get("error_code") != "interrupted_unknown"}


def legally_skipped(row, arm, branch):
    """Whether a branch with no attempt is a recorded fact instead of a missing one.

    The execution order is policy before contract, and the runner marks the contract
    branch as stopped exactly when a recorded stop reason ended sending before it.  A
    branch with no attempt and no recorded reason is an omission, not a skip.
    """

    if arm != "multi_agent" or branch != "contract":
        return False
    stop = row.get("stop_reason")
    if type(stop) is not str or not stop:
        return False
    recorded = (row.get("branches") or {}).get(branch)
    return type(recorded) is not dict or recorded.get("outcome") == "stopped"


def _moment(value):
    """One persisted ISO timestamp as a comparable moment, or ``None`` when unreadable."""

    if type(value) is not str:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _attempt_key(row):
    return (row.get("generation"), row.get("worker"), row.get("attempt_no"))


def _failed_run_identity_failures(run_id, binding):
    """Every stored row must belong to this Run and to this Run's one Trace identity."""

    failures = []
    for label, rows in (
        ("attempt", binding.get("attempts")),
        ("checkpoint", binding.get("checkpoints")),
        ("generation", binding.get("generations")),
        ("event", binding.get("events")),
    ):
        if type(rows) is not list:
            continue
        for row in rows:
            if type(row) is not dict or row.get("run_id") != run_id:
                failures.append("product_failed_" + label + "_parent")
                break
    events = binding.get("events")
    if type(events) is list:
        trace = binding.get("trace_id")
        for event in events:
            if type(event) is dict and event.get("trace_id") != trace:
                failures.append("product_failed_event_trace")
                break
    return failures


def _failed_run_ledger_failures(binding):
    """The adoption ledger of a failed Run, checked against its own durable rows.

    A branch commit is a two-part fact: the checkpoint that carries the adopted result and
    the ledger attempt it adopted, written in one transaction under one timestamp.  Reading
    only the checkpoints (or only the attempts) lets a deleted half pass, and comparing the
    observation's attempt projection with a projection recomputed from the same damaged
    binding proves nothing.  The relations verified here are the ones the write path
    itself establishes:

    * every checkpoint names an existing ledger attempt of the same generation whose own
      ``finished_at`` is the checkpoint's ``adopted_at`` — the same instant, not a close
      one;
    * an attempt whose row carries a later timestamp than the receipt that settled it was
      rewritten by the commit that adopted it, so its checkpoint must exist;
    * a receipt event naming no ledger attempt is an orphan: the attempt it proves was
      sent has been erased;
    * the generations are one contiguous closure, and every attempt and checkpoint lives
      inside it.
    """

    failures = []
    attempts = binding.get("attempts")
    attempts = (
        [item for item in attempts if type(item) is dict]
        if type(attempts) is list
        else []
    )
    checkpoints = binding.get("checkpoints")
    checkpoints = (
        [item for item in checkpoints if type(item) is dict]
        if type(checkpoints) is list
        else []
    )
    events = binding.get("events")
    events = (
        [item for item in events if type(item) is dict] if type(events) is list else []
    )
    generations = binding.get("generations")
    generations = (
        [item for item in generations if type(item) is dict]
        if type(generations) is list
        else []
    )
    numbers = [item.get("generation") for item in generations]
    if numbers != list(range(1, len(numbers) + 1)):
        failures.append("product_failed_generation_sequence")
    known = set(numbers)
    ledger = {_attempt_key(item): item for item in attempts}
    adopted = {_attempt_key(item) for item in checkpoints}
    for item in [*attempts, *checkpoints]:
        if _attempt_key(item)[0] not in known:
            failures.append("product_failed_generation_membership")
            break
    for checkpoint in checkpoints:
        attempt = ledger.get(_attempt_key(checkpoint))
        if attempt is None or attempt.get("finished_at") != checkpoint.get(
            "adopted_at"
        ):
            failures.append("product_failed_checkpoint_attempt")
            break
    receipts = {}
    for event in events:
        if event.get("step") == "model_attempt":
            receipts.setdefault(
                (event.get("generation"), event.get("worker"), event.get("attempt")),
                event,
            )
    for key in receipts:
        if key not in ledger:
            failures.append("product_failed_receipt_without_attempt")
            break
    for key, attempt in ledger.items():
        started, finished = (
            _moment(attempt.get("started_at")),
            _moment(attempt.get("finished_at")),
        )
        if started is None or (finished is not None and finished < started):
            failures.append("product_failed_attempt_timing")
            break
        receipt = receipts.get(key)
        occurred = _moment(receipt.get("occurred_at")) if receipt else None
        if (
            finished is not None
            and occurred is not None
            and key not in adopted
            and finished > occurred
        ):
            # The row was rewritten after the receipt that settled it: that rewrite is the
            # commit's own adoption, and the commit always leaves its checkpoint behind.
            failures.append("product_failed_adoption_without_checkpoint")
            break
    return failures


def _lifecycle_event_failure(status, events):
    """Which required lifecycle event is missing for one terminal Run status."""

    wanted = {
        "failed": ({"run_failed"}, "failed"),
        "interrupted": ({"step_failed"}, "interrupted"),
        "cancelled": ({"step_cancelled", "human_action_recorded"}, "cancelled"),
    }[status]
    for event in events:
        summary = event.get("output_summary")
        summary = summary if type(summary) is dict else {}
        if event.get("event_type") in wanted[0] and summary.get("status") == wanted[1]:
            return None
    return "product_failed_lifecycle_event:" + status


def failed_run_failures(
    row, binding, frozen, observed_attempts, elapsed_ms, run_id=None
):
    """The failure projection of a Run that ended without a committed summary.

    A legitimately failed Run needs no success summary and no checkpoint it never wrote.
    What it must keep is its own durable provenance: the frozen snapshot it was created
    with, the lifecycle event that ended it, its closed generation, its own attempt
    ledger, the adoption relation between its checkpoints and attempts, and an elapsed
    reading that matches the two persisted timestamps.  Every one of those is verified
    here, so erasing the provenance, contradicting the status, deleting recorded attempts
    or their receipts, reparenting a row to another Run or Trace, deleting a branch's
    checkpoint, or fabricating the timing is refused by name instead of being accepted as
    "a product failure"; a refusal-before-send, a zero-model run and a failure after a
    completed branch all stay legitimate terminal records.

    Returns ``(outcome_code, failures)``: a clean record yields its measured outcome and
    no failures, a damaged one yields failures and no outcome claim.
    """

    failures = []

    def fail(code):
        failures.append(code)

    status, code = row.get("status"), row.get("error_code")
    if status not in TERMINAL_RUN_STATUSES:
        return None, ["product_terminal_without_execution:" + str(status)]
    if binding.get("status") != status or binding.get("error_code") != code:
        # The row is the projection of the binding; a mismatch means one of the two was
        # rewritten to disagree with the other.
        fail("product_failed_status_binding")
    if type(code) is not str or not code:
        fail("product_failed_code_type")
    elif status == "failed" and code not in RUN_FAILURE_CODES:
        fail("product_failed_code_unregistered:" + code)
    if run_id is not None:
        # Every stored row belongs to this Run and its one Trace identity; a row that
        # names another parent is another Run's record, not this one's.
        failures.extend(_failed_run_identity_failures(run_id, binding))
        failures.extend(_failed_run_ledger_failures(binding))
        from .eval_f_validation import binding_shape_failures

        try:
            failures.extend(binding_shape_failures(run_id, binding))
        except (KeyError, TypeError, IndexError):
            fail("product_failed_binding_shape")
    if not same(binding.get("snapshot"), frozen.model_dump(mode="json")):
        fail("product_failed_snapshot")
    events = binding.get("events")
    if type(events) is not list or any(type(event) is not dict for event in events):
        fail("product_failed_events")
        events = []
    else:
        missing = _lifecycle_event_failure(status, events)
        if missing is not None:
            fail(missing)
    generations = binding.get("generations")
    if type(generations) is not list or not generations:
        fail("product_failed_generations")
    else:
        latest = generations[-1]
        expected = TERMINAL_GENERATION_STATUS.get(status)
        if (
            type(latest) is not dict
            or latest.get("status") == "active"
            or (expected is not None and latest.get("status") != expected)
        ):
            fail("product_failed_generation_closure")
    attempts = binding.get("attempts")
    if type(attempts) is not list or any(type(item) is not dict for item in attempts):
        fail("product_failed_attempt_rows")
    else:
        keys = [
            (item.get("generation"), item.get("worker"), item.get("attempt_no"))
            for item in attempts
        ]
        if len(set(keys)) != len(keys):
            fail("product_failed_duplicate_attempt")
    from .eval_h_product_checks import product_attempts

    try:
        projected_attempts = product_attempts(binding)
    except (KeyError, TypeError):
        projected_attempts = None
    if projected_attempts is None or payload_sha256(
        observed_attempts
    ) != payload_sha256(projected_attempts):
        # The observation's own attempt projection was computed from the same binding at
        # commit time; a disagreement means the persisted attempts or events were edited,
        # and a binding whose rows cannot be projected at all is damaged the same way.
        fail("product_failed_attempt_ledger")
    for checkpoint in binding.get("checkpoints") or []:
        if type(checkpoint) is not dict:
            fail("product_failed_checkpoint")
            continue
        try:
            payload = checkpoint["result_payload"]
            parsed = PreciseCheckpointPayloadV1.model_validate_json(
                json.dumps(payload), strict=True
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            fail("product_failed_checkpoint")
            continue
        from .precise_run_contracts import precise_contract_identity

        if (
            checkpoint.get("result_checksum") != canonical_checksum(payload)
            or checkpoint.get("contract_checksum")
            != precise_contract_identity(frozen, checkpoint.get("worker"))
            or parsed.result.worker != checkpoint.get("worker")
        ):
            fail("product_failed_checkpoint_identity")
    elapsed = elapsed_ms
    created, updated = binding.get("created_at"), binding.get("updated_at")
    if type(elapsed) is not int or elapsed < 0:
        fail("product_failed_elapsed")
    else:
        try:
            delta = (
                datetime.fromisoformat(updated) - datetime.fromisoformat(created)
            ).total_seconds() * 1000
            if abs(delta - elapsed) > ELAPSED_TOLERANCE_MS:
                fail("product_failed_elapsed")
        except (ValueError, TypeError):
            fail("product_failed_elapsed")
    if failures:
        return None, failures
    return "product_run_" + status + ":" + code, []


def observation_failures(plan, case, observation):
    errors = []

    def fail(code):
        errors.append(code)

    arm, seed = observation.get("arm"), observation.get("repeat_seed")
    if (
        type(seed) is not int
        or (case.case_id, seed, arm) not in plan.observation_keys()
    ):
        return ["observation_key"]
    for name, expected in (
        ("schema_version", "v1-07h-observation.v1"),
        ("case_id", case.case_id),
        ("plan_sha256", plan_digest(plan)),
        ("candidate_sha256", plan.candidate_sha256),
        ("consumer_sha256", plan.consumer_sha256),
        ("material_sha256", payload_sha256(case.model_dump(mode="json"))),
        ("sources", source_identities(plan, case)),
    ):
        if not same(observation.get(name), expected):
            fail("observation_binding:" + name)
    if type(observation.get("elapsed_ms")) is not int or observation["elapsed_ms"] < 0:
        fail("observation_elapsed")
    try:
        elapsed = (
            datetime.fromisoformat(observation["observed_at"])
            - datetime.fromisoformat(observation["started_at"])
        ).total_seconds() * 1000
        # The bound tolerates the product's own I/O between the measured run and the
        # durable commit; a fabricated clock still moves the two readings apart by far
        # more than this.
        if (
            elapsed < 0
            or abs(
                elapsed - observation.get("wall_elapsed_ms", observation["elapsed_ms"])
            )
            > CLOCK_TOLERANCE_MS
        ):
            fail("observation_clock")
    except (ValueError, KeyError, TypeError):
        fail("observation_clock")
    row = observation.get("row")
    if type(row) is not dict:
        row = {}
    binding = observation.get("binding")
    if type(binding) is not dict:
        binding = {}
    recorded_branches = row.get("branches")
    if type(recorded_branches) is not dict or any(
        type(branch) is not dict for branch in recorded_branches.values()
    ):
        fail("observation_row_branches")
    if row.get("status") not in {
        "pending_confirmation",
        "pending_human_review",
        "failed",
        "cancelled",
        "interrupted",
    }:
        fail("observation_not_terminal")
    recorded_stop = row.get("stop_reason")
    if recorded_stop is not None and stop_level(recorded_stop) == BATCH:
        # A batch-level stop — including an unregistered reason, which defaults to the
        # batch level — keeps the whole batch out of scoring instead of being read as a
        # measured per-Case outcome.
        fail("batch_stopped_trust_boundary")
    attempts = observation.get("attempts", [])
    if type(attempts) is not list or any(
        type(attempt) is not dict for attempt in attempts
    ):
        fail("observation_attempt_records")
        attempts = (
            [attempt for attempt in attempts if type(attempt) is dict]
            if type(attempts) is list
            else []
        )
    branches = ("policy", "contract") if arm == "multi_agent" else ("joint",)
    # A Run that ended without a committed summary legitimately may have no attempt at
    # all (refused before its first send); its attempt ledger is verified against its own
    # persisted rows by the failure projection instead of being required to be complete.
    failed_path = arm == "multi_agent" and not (
        binding.get("summary") and binding.get("execution")
    )
    keys = [(attempt.get("branch"), attempt.get("attempt_no")) for attempt in attempts]
    if len(set(keys)) != len(keys) or set(branch for branch, _ in keys) - set(branches):
        fail("attempt_identity")
    for branch in branches:
        numbers = [number for name, number in keys if name == branch]
        if (
            any(type(number) is not int for number in numbers)
            or sorted(numbers) != list(range(1, len(numbers) + 1))
            or len(numbers) > 2
        ):
            fail("attempt_quota:" + branch)
        if not numbers and not failed_path and not legally_skipped(row, arm, branch):
            fail("planned_task_unsettled:" + branch)
    for attempt in attempts:
        if attempt.get("status") != "settled":
            fail("attempt_unsettled")
        receipt = normal_receipt(attempt, arm)
        error = receipt.get("error_code")
        if error is not None and (type(error) is not str or not error):
            fail("attempt_error_type")
        if type(receipt.get("returned")) is not bool:
            fail("attempt_returned_type")
        unknown = receipt.get("usage_unknown")
        tokens = [receipt.get("input_tokens"), receipt.get("output_tokens")]
        if (
            type(unknown) is not bool
            or (unknown and all(token is not None for token in tokens))
            or (
                not unknown
                and any(type(token) is not int or token < 0 for token in tokens)
            )
        ):
            fail("attempt_usage_type")
        if any(
            token is not None and (type(token) is not int or token < 0)
            for token in tokens
        ):
            fail("attempt_usage_type")
        model = receipt.get("resolved_model")
        if model is not None and (
            type(model) is not str or model != plan.configuration.resolved_model
        ):
            fail("attempt_model_identity")
        if (
            receipt.get("error_code") is None
            and receipt.get("resolved_model") != plan.configuration.resolved_model
        ):
            fail("attempt_model_missing")
        latency = receipt.get("latency_ms")
        if latency is not None and (type(latency) is not int or latency < 0):
            fail("attempt_latency_type")
        if error is None and (receipt.get("returned") is not True or latency is None):
            fail("successful_attempt_without_response")
        if arm == "single_agent":
            if (
                type(attempt.get("request_sha256")) is not str
                or len(attempt["request_sha256"]) != 64
            ):
                fail("single_attempt_request_digest")
            if (attempt.get("projection") is not None) != (error is None):
                fail("single_attempt_success_projection")
    if arm == "multi_agent":
        run_id = row.get("run_id", "")
        if binding.get("case_id") != case.case_id:
            fail("product_case_identity")
        from types import SimpleNamespace

        frozen = execution_snapshot(
            mode="test_only" if plan.mode == "disposable" else "real",
            config=plan.configuration.model_copy(update={"seed": seed}),
            registration=SimpleNamespace(
                rules_version="3.1.0",
                document_version=plan.document_version,
                dataset_version=plan.dataset_version,
                index_version=plan.index_version,
            ),
            branch_timeout_seconds=DEFAULT_BRANCH_TIMEOUT_SECONDS,
            max_attempts_per_branch=MAX_MODEL_ATTEMPTS_PER_BRANCH,
            generation_deadline_seconds=plan_generation_deadline(plan),
        )
        failed_path = not binding.get("summary") or not binding.get("execution")
        if failed_path:
            # A Run the product ended without a committed summary is a legitimate
            # terminal outcome — but only when its own provenance still proves that is
            # what happened.  The dedicated projection verifies the identity, frozen
            # configuration, lifecycle event, closed generation, adoption ledger and
            # timing; erasing any of them is refused by name instead of being accepted
            # as "a product failure".  The source and retrieval consistency checks below
            # run for this path too: a failure record is not exempt from proving which
            # sources its executed branches used.
            outcome, rejected = failed_run_failures(
                row,
                binding,
                frozen,
                observation.get("attempts", []),
                observation.get("elapsed_ms"),
                run_id,
            )
            errors.extend(rejected)
            if outcome is not None:
                fail(outcome)
        else:
            if row.get("status") in TERMINAL_RUN_STATUSES:
                # A summary cannot travel with a terminal failure status: the product only
                # commits a summary while the Run is processing.
                fail("product_terminal_with_summary:" + str(row.get("status")))
            errors.extend(execution_contract_failures(run_id, binding))
            projection = binding.get("projection", {})
            expected_row = project(projection, binding)
            for field, value in expected_row.items():
                if not same(row.get(field), value):
                    fail("product_projection:" + field)
            if not same(
                binding["summary"]["summary_payload"], binding["execution"]["summary"]
            ) or binding["summary"]["summary_checksum"] != canonical_checksum(
                binding["summary"]["summary_payload"]
            ):
                fail("product_summary")
            if not same(binding.get("snapshot"), frozen.model_dump(mode="json")):
                fail("product_frozen_snapshot")
            from .eval_h_product_checks import product_failures

            errors.extend(product_failures(plan, case, observation, frozen))
            from datetime import UTC

            delta = (
                datetime.fromisoformat(binding["summary"]["created_at"]).astimezone(UTC)
                - datetime.fromisoformat(binding["created_at"]).astimezone(UTC)
            ).total_seconds() * 1000
            if abs(delta - observation["elapsed_ms"]) > 1:
                fail("product_service_latency")
            if binding["execution"]["procurement"]["model_calls"] != 0:
                fail("deterministic_model_call")
            from .precise_run_contracts import precise_contract_identity

            for checkpoint in binding.get("checkpoints", []):
                worker = checkpoint["worker"]
                payload = checkpoint["result_payload"]
                if checkpoint["result_checksum"] != canonical_checksum(
                    payload
                ) or checkpoint["contract_checksum"] != precise_contract_identity(
                    frozen, worker
                ):
                    fail("product_checkpoint_identity")
                commit = PreciseBranchCommitV1.model_validate_json(
                    json.dumps(payload["result"]), strict=True
                )
                native = binding["execution"][worker]
                for field in (
                    "spans",
                    "request_sha256",
                    "answer_sha256",
                    "support_failures",
                    "evidence_refs",
                ):
                    if not same(commit.model_dump(mode="json")[field], native[field]):
                        fail("product_checkpoint_projection:" + field)
    else:
        if (
            binding.get("admission_contract") != "h-evaluation-live-scope.v1"
            or binding.get("product_transactions_claimed") is not False
        ):
            fail("single_admission_contract")
        recorded_failures = (
            binding.get("execution", {}).get("decision", {}).get("pipeline_failures")
            or []
        )
        if row.get("status") == "failed" and not recorded_failures:
            # A failed joint record must keep the cause its own execution recorded; a
            # successful record relabelled as an empty failure has none.
            fail("single_failed_without_recorded_cause")
        expected_row = project(
            _summarise(
                dict(execution=binding.get("execution", {}), status=row.get("status"))
            ),
            binding,
        )
        for field in (
            "findings",
            "recommendation",
            "pipeline_failures",
            "spans",
            "conflicts",
            "security",
            "fact_observations",
        ):
            if not same(expected_row.get(field), row.get(field)):
                fail("single_projection:" + field)
        for attempt in attempts:
            projection = attempt.get("projection")
            if attempt.get("projection_sha256") != payload_sha256(projection):
                fail("single_attempt_projection_digest")
            if projection:
                for role, commit in projection.items():
                    if set(projection) != {"policy", "contract"}:
                        fail("single_attempt_projection_roles")
                    try:
                        parsed = PreciseBranchCommitV1.model_validate_json(
                            json.dumps(commit), strict=True
                        )
                    except ValidationError:
                        fail("single_commit_contract_presence_or_type")
                        continue
                    if not same(parsed.model_dump(mode="json"), commit):
                        fail("single_commit_contract_presence_or_type")
                    if (
                        parsed.worker != role
                        or parsed.request_sha256 != attempt["request_sha256"]
                    ):
                        fail("single_commit_identity")
                    spans = [span.model_dump(mode="json") for span in parsed.spans]
                    if not same(spans, binding["execution"].get(role, {}).get("spans")):
                        fail("single_commit_spans")
    sources = source_identities(plan, case)
    retrieval_all = observation.get("retrieval")
    if type(retrieval_all) is not dict:
        retrieval_all = {}
    for role in ("policy", "contract"):
        retrieval = retrieval_all.get(role)
        if type(retrieval) is not dict:
            retrieval = {}
        for variant in ("original", "without_case_id", "model_sources"):
            hits = retrieval.get(variant)
            if (
                type(hits) is not list
                or len(hits) > 5
                or any(not same(hit, sources.get(hit.get("source_id"))) for hit in hits)
            ):
                fail("retrieval_source_identity:" + role + ":" + variant)
        if not same(retrieval.get("original"), retrieval.get("model_sources")):
            fail("retrieval_model_context:" + role)
        if arm == "multi_agent":
            # Which sources the executed branches actually used: the committed execution
            # owns the answer for a Run that reached its summary, and the adopted
            # checkpoint of each completed branch owns it for a Run that failed earlier.
            # A branch that never produced either proves nothing here — its own absence is
            # what the adoption ledger above checks — but the retrieval record it would
            # have used must still be present and identity-valid.
            actual = _branch_source_locators(binding, role)
            if actual is not None and actual != [
                hit["locator"] for hit in retrieval.get("model_sources", [])
            ]:
                fail("product_actual_model_sources:" + role)
    return sorted(set(errors))


def _branch_source_locators(binding, role):
    """The chunk locators one branch actually used, or ``None`` when nothing proves them."""

    execution = binding.get("execution")
    if type(execution) is dict and execution:
        refs = execution.get(role, {}).get("evidence_refs")
        if type(refs) is list:
            return [ref["chunk_id"] for ref in refs if type(ref) is dict]
    for checkpoint in binding.get("checkpoints") or []:
        if type(checkpoint) is not dict or checkpoint.get("worker") != role:
            continue
        payload = checkpoint.get("result_payload")
        result = payload.get("result") if type(payload) is dict else None
        refs = result.get("evidence_refs") if type(result) is dict else None
        if type(refs) is list:
            return [ref.get("chunk_id") for ref in refs if type(ref) is dict]
    return None


def completion_proof(directory: Path):
    seal = HSeal(directory)
    plan = seal.plan()
    require_identity(plan)
    cases = {case.case_id: case for case in seal.runner_cases()}
    journal = Journal(directory)
    expected = set(plan.observation_keys())
    loaded, failures, outcomes = {}, [], []
    for path in journal.root.glob("*.json"):
        try:
            body = json.loads(path.read_bytes())
            if type(body) is not dict:
                failures.append("invalid_observation:" + path.name + ":not_an_object")
                continue
            pair = (body.get("case_id"), body.get("repeat_seed"), body.get("arm"))
            if pair not in expected or type(pair[1]) is not int or pair in loaded:
                failures.append("unregistered_or_duplicate_observation:" + path.name)
                continue
            original = journal.committed(*pair)
            if not same(original, body) or path != journal.path(*pair):
                failures.append("observation_storage_identity:" + path.name)
            loaded[pair] = body
            if pair[2] == "single_agent" and not same(
                journal.attempts(*pair), body["attempts"]
            ):
                failures.append("single_persisted_attempts:" + path.name)
            for error in observation_failures(plan, cases[pair[0]], body):
                # A Run the product failed closed is a measured outcome that keeps its
                # place in the denominator; only an unreadable record makes the batch
                # incomplete.
                if error.startswith("product_run_"):
                    outcomes.append(f"{path.name}:{error}")
                else:
                    failures.append(f"{path.name}:{error}")
        except (ValueError, KeyError, TypeError, IndexError) as error:
            failures.append(
                "invalid_observation:" + path.name + ":" + type(error).__name__
            )
    failures.extend(
        "missing_observation:" + str(pair) for pair in sorted(expected - set(loaded))
    )
    try:
        stop = read_batch_stop(directory, plan)
    except ValueError:
        # The persisted stop cannot be verified; the batch cannot be read as one that
        # never stopped, so the proof stays incomplete with the damage named.
        failures.append("batch_stopped_record_unreadable")
        stop = None
    if stop is not None:
        # A batch that stopped sending is preserved as a whole instead of being read from
        # its remaining successes (Spec P-05).  The durable record is what keeps that true
        # across a new process, so the completion proof names it rather than presenting a
        # stopped batch as a completed one.
        failures.append("batch_stopped_record:" + stop["reason"])
    attempt_root = Path(directory) / "attempts"
    registered = {
        ".".join(map(str, pair)) for pair in expected if pair[2] == "single_agent"
    }
    if attempt_root.exists():
        for task in attempt_root.iterdir():
            if task.name not in registered or not task.is_dir():
                failures.append("unregistered_attempt_task:" + task.name)
                continue
            for path in task.iterdir():
                if path.name != ".transition.lock" and path.name not in {
                    "joint.1.json",
                    "joint.2.json",
                }:
                    failures.append("unregistered_attempt_artifact:" + path.name)
    # Pair the actual source and MCP sets, including failed business observations.
    for case_id in plan.case_ids:
        for seed in plan.repeat_seeds:
            multi, single = (
                loaded.get((case_id, seed, arm))
                for arm in ("multi_agent", "single_agent")
            )
            if multi and single:
                for field in ("sources", "retrieval"):
                    if field not in multi or field not in single:
                        failures.append("unpaired_" + field + ":" + case_id)
                    elif not same(multi[field], single[field]):
                        failures.append("unfair_" + field + ":" + case_id)
                multi_binding, single_binding = (
                    multi.get("binding"),
                    single.get("binding"),
                )
                if type(multi_binding) is not dict or type(single_binding) is not dict:
                    failures.append("unpaired_actual_facts:" + case_id)
                elif not same(multi_binding.get("facts"), single_binding.get("facts")):
                    failures.append("unfair_actual_facts:" + case_id)
    return dict(
        schema_version=H_CONSUMPTION,
        plan_sha256=plan_digest(plan),
        complete=not failures,
        failures=sorted(set(failures)),
        product_outcomes=sorted(set(outcomes)),
        planned=len(expected),
        observed=len(loaded),
        by_arm=dict(Counter(pair[2] for pair in loaded)),
        observation_sha256={
            ".".join(map(str, pair)): body["commit_sha256"]
            for pair, body in sorted(loaded.items())
        },
    )
