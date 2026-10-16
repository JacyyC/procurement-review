"""Pre-registration for the G/H two-arm comparison.

No truth file and no production aggregator is read here.  Coverage denominators
come from this plan, and the historical metering scopes are read back from the
artifacts that actually recorded them so that a new namespace never silently
absorbs or erases an earlier one.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .eval_f_context import input_fingerprints, load_cases, stable_write
from .eval_g_contracts import (
    DETERMINISTIC_OBLIGATIONS,
    DOCUMENT_OBLIGATIONS,
    ArmPlan,
    ArmShape,
    ComparisonArm,
    ComparisonPlanV1,
    FairnessContract,
    HistoryMetering,
    ModelBranch,
    PriorLedger,
)
from .eval_v2_contracts import EvalTopology
from .gold_v3_contracts import payload_sha256

BACKEND = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = BACKEND.parents[1]
RESTART_HISTORY = (
    REPOSITORY_ROOT / "workspace/docs/work-items/records/v1-07/restart-history.json"
)
G_LIVE_ROOT = BACKEND / ".local/v1-07g/live"
G_NAMESPACE = "v1-07g-comparison"
C_LIVE_ROOT = BACKEND / ".local/v1-07c/live"
F_LIVE_ROOT = BACKEND / ".local/v1-07f/live"

#: Every module and script a frozen candidate actually depends on.  Scoring and
#: independent-audit consumers are included on purpose: freezing only the model
#: side would let a later scorer change reinterpret an already reported batch.
_CONSUMER_PATTERNS = (
    "business_v3_*.py",
    "chat_transport_v*.py",
    "eval_attempt_*.py",
    "eval_f_contracts.py",
    "eval_f_context.py",
    "eval_f_obligations.py",
    "eval_f_score.py",
    "eval_f_semantics.py",
    "eval_g_*.py",
    "eval_v2_contracts.py",
    "eval_v2_metrics.py",
    "gold_v3*.py",
    "model_attempt_*.py",
    "model_wire_v2.py",
)
_PRODUCT_PATTERNS = (
    "admitted_reads.py",
    "api*.py",
    "checkpointing.py",
    "ingestion.py",
    "mcp_server.py",
    "model_invocation.py",
    "procurement_mcp_client.py",
    "record_*.py",
    "result_admission.py",
    "retrieval*.py",
    "review_*.py",
    "run_*.py",
    "scoped_runtime.py",
    "security_contracts.py",
    "worker_contracts.py",
    "workers.py",
)
_CONSUMER_SCRIPTS = ("precise_report.py", "comparison_report.py")


def comparison_implementation_fingerprints() -> dict[str, str]:
    """Fingerprint the full frozen candidate, not only its model side."""

    package = Path(__file__).parent
    project = package.parents[1]
    paths: dict[str, Path] = {}
    for pattern in (*_CONSUMER_PATTERNS, *_PRODUCT_PATTERNS):
        for path in package.glob(pattern):
            paths[path.name] = path
    for path in (package / "api_routers").glob("*.py"):
        paths[f"api_routers/{path.name}"] = path
    for name in _CONSUMER_SCRIPTS:
        if (project / "scripts" / name).exists():
            paths[f"scripts/{name}"] = project / "scripts" / name
    for name in ("apiContract.ts", "api.ts"):
        if (project / "frontend" / "src" / name).exists():
            paths[f"frontend/{name}"] = project / "frontend" / "src" / name
    return {
        name: hashlib.sha256(paths[name].read_bytes()).hexdigest()
        for name in sorted(paths)
    }


def terminal_generation_states(binding_version):
    if binding_version == "v1-07g-db-binding.v4":
        return {"completed", "expired", "cancelled", "superseded"}
    return {"completed", "expired", "interrupted", "cancelled", "failed"}


def generation_shape_failures(run_id, run):
    from pydantic import ValidationError
    from .record_contracts import GenerationStatusPatch

    failures = []
    for generation in run["generations"]:
        for key in ("run_id", "generation", "status", "created_at", "deadline_at"):
            if key not in generation:
                failures.append(f"generation_missing:{key}:{run_id}")
        if type(generation.get("generation")) is not int:
            failures.append(f"generation_type:{run_id}")
        try:
            GenerationStatusPatch.model_validate_json(
                json.dumps({"status": generation.get("status")}), strict=True
            )
        except ValidationError:
            failures.append(f"generation_status_type:{run_id}")
    if failures:
        return failures
    latest = max(run["generations"], key=lambda g: g["generation"], default=None)
    if latest and (
        latest["status"] == "superseded"
        or type(run.get("status")) is str
        and run["status"]
        in {"pending_confirmation", "pending_human_review", "confirmed", "rejected"}
        and latest["status"] != "completed"
        or run.get("status") == "cancelled"
        and latest["status"] not in {"completed", "cancelled"}
    ):
        failures.append(f"generation_run_state:{run_id}")
    return failures


def adoption_evidence_shape_failures(registration, expected_runs=()):
    """Validate frozen proof containers before any consumer dereferences them."""
    from .precise_run_contracts import PreciseBranchCommitV1

    failures = []

    def field(record, name, kind, path):
        where = path + "." + name
        if name not in record:
            failures.append("adoption_evidence_missing:" + where)
        elif type(record[name]) is not kind:
            failures.append("adoption_evidence_type:" + where)
        elif kind is dict and any(type(key) is not str for key in record[name]):
            failures.append("adoption_evidence_type:" + where + ".key")
        else:
            return record[name]
        return None

    path = "registration.binding.adoption_evidence"
    manifest = field(
        registration["binding"], "adoption_evidence", dict, "registration.binding"
    )
    if manifest is None:
        return failures
    runs = field(manifest, "runs", dict, path)
    if runs is None:
        return failures
    for run_id in sorted(set(runs) | set(expected_runs)):
        run_path = path + ".runs." + run_id
        proof = field(runs, run_id, dict, path + ".runs")
        if proof is None:
            continue
        field(proof, "case_id", str, run_path)
        checkpoints = field(proof, "checkpoints", dict, run_path)
        if checkpoints is None:
            continue
        adopted = (
            {
                cp.get("worker")
                for cp in expected_runs[run_id].get("checkpoints", [])
                if type(cp.get("worker")) is str
            }
            if isinstance(expected_runs, Mapping) and run_id in expected_runs
            else set()
        )
        for worker in sorted(set(checkpoints) | adopted):
            cp_path = run_path + ".checkpoints." + worker
            checkpoint = field(checkpoints, worker, dict, run_path + ".checkpoints")
            if checkpoint is None:
                continue
            for name, kind in (
                ("generation", int),
                ("attempt_no", int),
                ("payload_sha256", str),
            ):
                field(checkpoint, name, kind, cp_path)
            committed = field(checkpoint, "commit_fields", dict, cp_path)
            if committed is not None:
                for name in sorted(
                    set(committed) | set(PreciseBranchCommitV1.model_fields)
                ):
                    field(committed, name, str, cp_path + ".commit_fields")
    return failures


def validate_adoption_evidence(registration, expected_runs=()):
    failures = adoption_evidence_shape_failures(registration, expected_runs)
    if failures:
        raise ValueError(",".join(failures))


def incomplete_binding_v4(registration, observations, bindings, failures):
    return {
        "planned_pairs": len(registration["case_ids"]) * registration["repeats"],
        "planned_model_tasks": len(registration["case_ids"])
        * registration["repeats"]
        * len(registration["binding"]["branches"]),
        "observed_pairs": len(observations["observations"]),
        "binding_runs": len(bindings.get("runs", {})),
        "unknown_attempts": [],
        "failures": failures,
        "complete": not failures,
        "binding_version": "v1-07g-db-binding.v4",
    }


def validate_evidence_source_shape(source):
    from pydantic import ValidationError
    from .precise_run_contracts import PreciseBranchCommitV1, PreciseCheckpointPayloadV1

    def require(value, kind, path):
        if type(value) is not kind:
            raise ValueError("v4_source_input_type:" + path)
        return value

    def field(record, name, kind, path):
        where = path + "." + name
        if name not in record:
            raise ValueError("v4_source_input_missing:" + where)
        return require(record[name], kind, where)

    require(source, dict, "source")
    field(source, "observations_payload_sha256", str, "source")
    for run_id, run in field(source, "runs", dict, "source").items():
        require(run_id, str, "source.runs.key")
        path = "source.runs." + run_id
        require(run, dict, path)
        field(run, "case_id", str, path)
        execution = field(run, "execution", dict, path)
        for worker in ("policy", "contract"):
            if execution.get(worker) is not None:
                require(execution[worker], dict, path + ".execution." + worker)
        checkpoints = field(run, "checkpoints", list, path)
        if not checkpoints and any(
            (execution.get(worker) or {}).get("outcome") == "completed"
            for worker in ("policy", "contract")
        ):
            raise ValueError("v4_source_input_missing:" + path + ".checkpoints")
        for index, checkpoint in enumerate(checkpoints):
            cp_path = f"{path}.checkpoints[{index}]"
            require(checkpoint, dict, cp_path)
            for name, kind in (
                ("worker", str),
                ("generation", int),
                ("attempt_no", int),
            ):
                field(checkpoint, name, kind, cp_path)
            payload = field(checkpoint, "result_payload", dict, cp_path)
            committed = field(payload, "result", dict, cp_path + ".result_payload")
            for record, model, record_path in (
                (payload, PreciseCheckpointPayloadV1, cp_path + ".result_payload"),
                (committed, PreciseBranchCommitV1, cp_path + ".result_payload.result"),
            ):
                for name in model.model_fields:
                    if name not in record:
                        raise ValueError(
                            "v4_source_input_missing:" + record_path + "." + name
                        )
            try:
                PreciseCheckpointPayloadV1.model_validate_json(
                    json.dumps(payload), strict=True
                )
            except ValidationError as error:
                locations = [".".join(map(str, e["loc"])) for e in error.errors()]
                raise ValueError(
                    "v4_source_input_contract:" + cp_path + ":" + ",".join(locations)
                ) from None
    return source


def evidence_manifest(source, source_file_sha256):
    """Freeze full DEV committed content before scoring or mutating a binding."""
    validate_evidence_source_shape(source)
    return {
        "manifest_version": "v1-07g-adoption-evidence.v1",
        "source_file_sha256": source_file_sha256,
        "observations_payload_sha256": source["observations_payload_sha256"],
        "runs": {
            run_id: {
                "case_id": run["case_id"],
                "checkpoints": {
                    cp["worker"]: {
                        "generation": cp["generation"],
                        "attempt_no": cp["attempt_no"],
                        "payload_sha256": payload_sha256(cp["result_payload"]),
                        "commit_fields": {
                            key: payload_sha256(value)
                            for key, value in cp["result_payload"]["result"].items()
                        },
                    }
                    for cp in run["checkpoints"]
                },
            }
            for run_id, run in source["runs"].items()
        },
    }


def extend_registration_v4(base_path, source_path, output_path):
    from .precise_candidate import extend_registration_v3
    from .eval_f_validation import EXTENSION_V4

    output = Path(output_path)
    raw = Path(source_path).read_bytes()
    source = validate_evidence_source_shape(json.loads(raw))
    if source.get("binding_version") != "v1-07g-db-binding.v3":
        raise ValueError("v4_evidence_source_version")
    with tempfile.TemporaryDirectory(prefix="g-dev-registration-") as folder:
        staged = Path(folder) / "registration.json"
        extend_registration_v3(Path(base_path), staged)
        body = json.loads(staged.read_text())
    body["extends"]["extension_version"] = EXTENSION_V4
    body["binding"]["adoption_evidence"] = evidence_manifest(
        source, hashlib.sha256(raw).hexdigest()
    )
    verify_evidence_source(body, source_path)
    output.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2))
    return {"written": str(output), "extension_version": EXTENSION_V4}


def verify_evidence_source(registration, source_path):
    raw = Path(source_path).read_bytes()
    validate_adoption_evidence(registration)
    manifest = registration["binding"].get("adoption_evidence", {})
    if hashlib.sha256(raw).hexdigest() != manifest.get("source_file_sha256"):
        raise ValueError("v4_source_evidence_file_binding")
    source = validate_evidence_source_shape(json.loads(raw))
    validate_adoption_evidence(registration, source.get("runs", {}))
    if source.get("binding_version") != "v1-07g-db-binding.v3":
        raise ValueError("v4_evidence_source_version")
    if (
        source.get("base_registration_sha256")
        != registration["extends"]["base_registration_sha256"]
    ):
        raise ValueError("v4_source_base_registration_binding")
    if evidence_manifest(source, manifest["source_file_sha256"]) != manifest:
        raise ValueError("v4_source_evidence_manifest_binding")
    return manifest["source_file_sha256"]


def freeze_sweep_v4(
    registration_path,
    observations_path,
    bindings_path,
    source_path,
    score_path,
    output_path,
):
    from .precise_candidate import candidate_fingerprint
    from .precise_dev_score import score_sweep
    from .eval_f_validation import EXTENSION_V4

    paths = {
        "registration": Path(registration_path),
        "observations": Path(observations_path),
        "bindings": Path(bindings_path),
        "source_bindings": Path(source_path),
    }
    values = {name: json.loads(path.read_bytes()) for name, path in paths.items()}
    if (
        values["registration"].get("extends", {}).get("extension_version")
        != EXTENSION_V4
    ):
        raise ValueError("v4_freeze_extension_version")
    verify_evidence_source(values["registration"], source_path)
    result = score_sweep(
        values["registration"], values["observations"], values["bindings"]
    )
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    result["inputs"]["file_sha256"] = hashes
    if values["bindings"].get("observations_file_sha256") != hashes["observations"]:
        raise ValueError("v4_freeze_raw_file_binding")
    stored = json.loads(Path(score_path).read_bytes())
    if json.loads(json.dumps(result)) != stored:
        raise ValueError("v4_freeze_full_score_mismatch")
    if (
        result["score_version"] != "v1-07g-dev-score.v4"
        or not result["stage_acceptance"]
        or result["failed_gates"]
    ):
        raise ValueError("v4_freeze_gates_failed")
    candidate = candidate_fingerprint()
    if result["inputs"]["scorer_candidate_sha256"] != candidate["candidate_sha256"]:
        raise ValueError("v4_freeze_current_candidate")
    # Historical scores and existing freezes remain readable by their version.
    # New qualification requires the corrected, explicit consumption contract.
    raise ValueError("v4_freeze_retired_use_v5")


def _ledger_scope(root: Path, namespace: str, entry: str) -> PriorLedger | None:
    """Read one durable stage inventory exactly as its receipts recorded it."""

    import sqlite3

    from .eval_attempt_ledger import AttemptLedger

    if not root.exists():
        return None
    sent = receipts = unknown = 0
    known_input = known_output = 0
    digests: list[str] = []
    for path in sorted(root.rglob("ledger.sqlite3")):
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        sent += AttemptLedger(path).manifest().attempt_count
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            for (body,) in conn.execute(
                "SELECT receipt_json FROM attempts WHERE receipt_json IS NOT NULL"
            ):
                row = json.loads(body)
                receipts += 1
                unknown += int(bool(row.get("usage_unknown")))
                known_input += row.get("input_tokens") or 0
                known_output += row.get("output_tokens") or 0
    if not digests:
        return None
    return PriorLedger(
        namespace=namespace,
        source_entry=entry,
        source_status="read",
        count_basis="exact",
        physical_sent=sent,
        receipts=receipts,
        unreturned=sent - receipts,
        usage_unknown=unknown,
        known_input_tokens=known_input,
        known_output_tokens=known_output,
        cost_status="unknown",
        source_sha256=hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest(),
    )


def _restart_history_scope() -> PriorLedger:
    """The retired G/H inventory, read from its sealed registration."""

    registered = json.loads(RESTART_HISTORY.read_text(encoding="utf-8"))
    inventory = registered["historical_g_h_inventory"]
    return PriorLedger(
        namespace="v1-07-retired-g-h",
        source_entry=str(RESTART_HISTORY.relative_to(REPOSITORY_ROOT)),
        source_status="registered_only",
        count_basis="exact",
        physical_sent=inventory["physical_sent"],
        receipts=inventory["physical_receipted"],
        unreturned=inventory["unreturned"],
        usage_unknown=inventory["usage_unknown"],
        known_input_tokens=inventory["known_input_tokens"],
        known_output_tokens=inventory["known_output_tokens"],
        cost_status="unknown",
        source_sha256=inventory["consistent_snapshot_sha256"],
    )


def historical_prior_scope() -> PriorLedger:
    """The V1-02/V1-05 real-line scope, kept at its own declared upper bound."""

    from .eval_f_registration import registration

    prior: Mapping[str, Any] = registration()["prior"]
    sent = int(max(prior["historical_requests"]))
    receipts = sent - int(prior["historical_inflight_upper"])
    return PriorLedger(
        namespace="v1-02-v1-05-real-line",
        source_entry="eval_f_registration.registration()['prior']",
        source_status="registered_only",
        count_basis="upper_bound",
        physical_sent=sent,
        receipts=receipts,
        unreturned=int(prior["historical_inflight_upper"]),
        usage_unknown=int(prior["historical_unknown"]),
        known_input_tokens=0,
        known_output_tokens=0,
        cost_status="unknown",
        source_sha256=None,
    )


def history_metering(namespace: str = G_NAMESPACE) -> HistoryMetering:
    """Register the new namespace beside every prior scope without summing them."""

    scopes: list[PriorLedger] = [_restart_history_scope(), historical_prior_scope()]
    for root, name, entry in (
        (C_LIVE_ROOT, "v1-07c-probe", ".local/v1-07c/live"),
        (F_LIVE_ROOT, "v1-07f-oracle-runner", ".local/v1-07f/live"),
    ):
        scope = _ledger_scope(root, name, entry)
        if scope is not None:
            scopes.append(scope)
    return HistoryMetering(namespace=namespace, prior_ledgers=tuple(scopes))


def arm_shapes() -> tuple[ArmShape, ArmShape]:
    """The frozen call structures; only these differ between the two arms."""

    return (
        ArmShape(
            arm_id=ComparisonArm.MULTI_AGENT,
            topology=EvalTopology.MULTI_AGENT,
            model_branches=(
                ModelBranch(branch_id="policy", obligation_ids=("O-01", "O-02")),
                ModelBranch(
                    branch_id="contract",
                    obligation_ids=("O-05", "O-06", "O-08", "O-09"),
                ),
            ),
            deterministic_branches=("procurement", "control"),
        ),
        ArmShape(
            arm_id=ComparisonArm.SINGLE_AGENT,
            topology=EvalTopology.SINGLE_AGENT,
            model_branches=(
                ModelBranch(branch_id="joint", obligation_ids=DOCUMENT_OBLIGATIONS),
            ),
            deterministic_branches=("procurement", "control"),
        ),
    )


def fairness(
    *,
    materials_sha256: str,
    fact_tools_sha256: str,
    retrieval_index_sha256: str,
    truth_contract: str,
    requested_model: str = "deepseek-v4.1-flash",
    expected_resolved_model: str = "deepseek-v4-1-flash",
    case_concurrency: int = 2,
    single_attempt_seconds: float = 90.0,
    run_deadline_seconds: float = 120.0,
) -> FairnessContract:
    return FairnessContract(
        requested_model=requested_model,
        expected_resolved_model=expected_resolved_model,
        case_concurrency=case_concurrency,
        single_attempt_seconds=single_attempt_seconds,
        run_deadline_seconds=run_deadline_seconds,
        materials_sha256=materials_sha256,
        fact_tools_sha256=fact_tools_sha256,
        retrieval_index_sha256=retrieval_index_sha256,
        truth_contract=truth_contract,
        scoring_version="v1-07g-score.v1",
    )


def _arm_plan(
    shape: ArmShape, case_ids: Sequence[str], seeds: Sequence[int]
) -> ArmPlan:
    return ArmPlan(
        shape=shape,
        case_ids=tuple(case_ids),
        repeat_seeds=tuple(seeds),
        planned_observations=len(case_ids) * len(seeds),
        planned_first_sends=len(case_ids) * len(seeds) * len(shape.model_branches),
        max_physical_requests=len(case_ids)
        * len(seeds)
        * len(shape.model_branches)
        * 2,
    )


def plan(
    *,
    plan_id: str,
    stage_id: str,
    split: str,
    gold_set_version: str,
    fairness_contract: FairnessContract,
    case_ids: Sequence[str] | None = None,
    repeat_seeds: Sequence[int] = (601, 602),
    history: HistoryMetering | None = None,
    authorization: str = "user-explicit-unbounded-with-registered-batch-limits",
    stop_conditions: Sequence[str] = (
        "model_identity_drift",
        "security_boundary",
        "tool_call_forbidden",
        "authentication",
        "unsupported_request",
        "cancelled",
        "deadline",
        "configuration_changed",
        "ledger_untrustworthy",
    ),
) -> ComparisonPlanV1:
    """Freeze one two-arm plan; the caller supplies the material fingerprints."""

    registered = case_ids
    if registered is None:
        registered = [
            case.case_id
            for case in load_cases(
                "validation" if split == "validation" else "dev", gold_set_version
            )
        ]
    multi, single = arm_shapes()
    draft = ComparisonPlanV1(
        plan_id=plan_id,
        stage_id=stage_id,
        split=split,
        gold_set_version=gold_set_version,
        fairness=fairness_contract,
        history=history or history_metering(),
        arms=(
            _arm_plan(multi, registered, repeat_seeds),
            _arm_plan(single, registered, repeat_seeds),
        ),
        authorization=authorization,
        stop_conditions=tuple(stop_conditions),
    )
    digest = payload_sha256(draft.model_dump(mode="json", exclude={"plan_sha256"}))
    return draft.model_copy(update={"plan_sha256": digest})


def prepare(
    path: Path,
    *,
    plan_id: str,
    stage_id: str = "V1-07G",
    split: str = "dev",
    gold_set_version: str,
    fairness_contract: FairnessContract,
    case_ids: Sequence[str] | None = None,
    repeat_seeds: Sequence[int] = (601, 602),
) -> dict[str, Any]:
    """Write one frozen plan; zero network and zero model requests."""

    value = plan(
        plan_id=plan_id,
        stage_id=stage_id,
        split=split,
        gold_set_version=gold_set_version,
        fairness_contract=fairness_contract,
        case_ids=case_ids,
        repeat_seeds=repeat_seeds,
    )
    stable_write(Path(path), value.model_dump(mode="json"))
    return {
        "plan_sha256": value.plan_sha256,
        "planned_observations": value.planned_observations(),
        "planned_first_sends": value.planned_first_sends(),
        "planned_physical_capacity": value.planned_physical_capacity(),
        "network_calls": 0,
    }


def load_plan(path: Path) -> ComparisonPlanV1:
    """Read a frozen plan back and require its digest to still match."""

    loaded = ComparisonPlanV1.model_validate_json(Path(path).read_bytes())
    expected = payload_sha256(loaded.model_dump(mode="json", exclude={"plan_sha256"}))
    if loaded.plan_sha256 != expected:
        raise ValueError("comparison plan digest does not match its content")
    return loaded


def planned_task_index(plan: ComparisonPlanV1) -> dict[str, tuple[str, str, int, str]]:
    """Every logical first-send task the plan pre-registers, keyed by task key."""

    index: dict[str, tuple[str, str, int, str]] = {}
    for arm, arm_plan in plan.arm.items():
        for seed in arm_plan.repeat_seeds:
            for case_id in arm_plan.case_ids:
                for branch in arm_plan.shape.model_branches:
                    key = f"{case_id}.{seed}.{branch.branch_id}"
                    index[key] = (arm.value, case_id, seed, branch.branch_id)
    return index


class UnregisteredTask(ValueError):
    """A logical task the frozen plan never pre-registered."""


def assert_within_plan(
    plan: ComparisonPlanV1, task_keys: Iterable[str]
) -> dict[str, int]:
    """Refuse any first send outside the frozen plan; never widen it silently."""

    index = planned_task_index(plan)
    outside = sorted({key for key in task_keys if key not in index})
    if outside:
        raise UnregisteredTask(f"unregistered first-send tasks: {outside[:8]}")
    return {"planned": len(index), "checked": len(set(task_keys))}


def deterministic_obligations() -> tuple[str, ...]:
    return DETERMINISTIC_OBLIGATIONS


def data_fingerprints(gold_set_version: str) -> dict[str, str]:
    return input_fingerprints(gold_set_version)


__all__ = [
    "ArmPlan",
    "ComparisonPlanV1",
    "F_LIVE_ROOT",
    "G_LIVE_ROOT",
    "G_NAMESPACE",
    "arm_shapes",
    "comparison_implementation_fingerprints",
    "data_fingerprints",
    "deterministic_obligations",
    "fairness",
    "UnregisteredTask",
    "assert_within_plan",
    "historical_prior_scope",
    "history_metering",
    "planned_task_index",
    "load_plan",
    "plan",
    "prepare",
]
