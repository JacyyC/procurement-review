"""Explicit H registration, seal, run and zero-network consumption entrances."""

import argparse
import asyncio
import json
from pathlib import Path

from .eval_h_audit import audit, freeze, replay, verify_freeze
from .eval_h_contracts import HPlan
from .eval_h_exclusion import register
from .eval_h_fixture import make_disposable
from .eval_h_identity import identity
from .eval_h_mock import mock_transport
from .eval_h_runner import run, setup
from .eval_h_score import rebuild as score
from .eval_h_report import rebuild as report
from .eval_h_seal import HSeal
from .eval_h_storage import read, sha, write, write_bytes


def create(root, plan_path, recipe, seed, exclusion, safety):
    plan = HPlan.model_validate_json(plan_path.read_bytes(), strict=True)
    if (sha(recipe), sha(seed), read(exclusion)["registry_sha256"], sha(safety)) != (
        plan.generator_spec_sha256,
        plan.seed_sha256,
        plan.exclusion_sha256,
        plan.lifecycle_evidence_sha256,
    ):
        raise ValueError("h_registration_input_binding")
    seal = HSeal.create(root, plan)
    for source, target in (
        (recipe, "recipe.json"),
        (seed, "seed.bin"),
        (exclusion, "registry.json"),
        (safety, "safety.json"),
    ):
        write_bytes(seal.directory / "private" / target, source.read_bytes())
    return {
        "root": str(seal.directory),
        "state": "candidate_frozen",
        "network_calls": 0,
    }


def preflight(root, *, runner=False):
    seal = HSeal(root)
    plan = seal.plan()
    from .eval_h_identity import require_identity
    from .eval_h_exclusion import verify

    require_identity(plan)
    if runner:
        from .eval_h_seal import plan_digest
        from .eval_h_storage import private

        path = seal.directory / "runner/preflight.json"
        private(path)
        body = read(path)
        record = seal.record()
        if (
            record.get("preflight_sha256") != sha(path)
            or body.get("schema_version") != "v1-07h-send-preflight.v1"
            or body.get("plan_sha256") != plan_digest(plan)
            or type(body.get("checks")) is not dict
            or any(type(value) is not bool for value in body["checks"].values())
        ):
            raise ValueError("h_send_preflight_binding")
        return dict(
            checks=body["checks"],
            ready_for_formal=bool(body["checks"]) and all(body["checks"].values()),
            capacity=plan.capacity(),
            state=record["state"],
            network_calls=0,
            stage_acceptance=False,
        )
    registry = read(seal.directory / "private/registry.json")
    verify(registry)
    from .eval_h_generation import HGenerationSpec

    try:
        spec = HGenerationSpec.model_validate_json(
            (seal.directory / "private/recipe.json").read_bytes(), strict=True
        )
        spec.require_formal_quotas()
        generator_ready = set(slot.case_id for slot in spec.slots) == set(plan.case_ids)
    except ValueError:
        generator_ready = False
    from .eval_h_report import safety

    checks = dict(
        current_identity=True,
        history_complete=registry["history_complete"],
        formal_plan=plan.mode == "formal",
        explicit_authorization=bool(plan.authorization),
        seeded_formal_generator=generator_ready,
        lifecycle_evidence=safety(root, plan)["verified"],
        exclusion_binding=registry["registry_sha256"] == plan.exclusion_sha256,
        registered_material=sha(seal.directory / "private/recipe.json")
        == plan.generator_spec_sha256,
        registered_seed=sha(seal.directory / "private/seed.bin") == plan.seed_sha256,
    )
    return dict(
        checks=checks,
        ready_for_formal=all(checks.values()),
        capacity=plan.capacity(),
        state=seal.record()["state"],
        network_calls=0,
        stage_acceptance=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("identity")
    p = sub.add_parser("prepare-disposable")
    p.add_argument("root", type=Path)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--batch-id", default="disposable-h-prep")
    p = sub.add_parser("register-history")
    p.add_argument("seen", type=Path)
    p.add_argument("sources", type=Path, help="explicit path-to-SHA JSON mapping")
    p.add_argument("output", type=Path)
    p = sub.add_parser("create")
    p.add_argument("root", type=Path)
    for name in ("plan", "recipe", "seed", "exclusion", "safety"):
        p.add_argument("--" + name, required=True, type=Path)
    for name in (
        "preflight",
        "generate",
        "seal",
        "complete",
        "release",
        "score",
        "report",
        "replay",
        "audit",
    ):
        sub.add_parser(name).add_argument("root", type=Path)
    p = sub.add_parser("setup")
    p.add_argument("root", type=Path)
    p.add_argument("--database-url-env", required=True)
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    p.add_argument("--mock", choices=("material", "http500", "correction"))
    p.add_argument("--real-authorization")
    p.add_argument(
        "--resume",
        action="store_true",
        help="explicit product resume; committed branches are reused",
    )
    p = sub.add_parser("settle-unknown")
    p.add_argument("root", type=Path)
    p.add_argument("--case", required=True)
    p.add_argument("--seed", required=True, type=int)
    for name, arg in (("freeze", "output"), ("verify-freeze", "frozen")):
        p = sub.add_parser(name)
        p.add_argument("root", type=Path)
        p.add_argument(arg, type=Path)
    args = parser.parse_args(argv)
    if args.command == "identity":
        result = identity()
    elif args.command == "register-history":
        result = register(
            args.seen,
            {Path(path): digest for path, digest in read(args.sources).items()},
            output=args.output,
        )
        result = {
            key: value
            for key, value in result.items()
            if key not in {"materials", "keys", "source_sha256"}
        }
    elif args.command == "prepare-disposable":
        result = make_disposable(args.root, args.count, batch_id=args.batch_id)
    elif args.command == "create":
        result = create(
            args.root, args.plan, args.recipe, args.seed, args.exclusion, args.safety
        )
    elif args.command == "preflight":
        result = preflight(args.root)
    else:
        seal = HSeal(args.root)
        if args.command == "generate":
            result = seal.generate(
                seal.directory / "private/recipe.json",
                seal.directory / "private/seed.bin",
                seal.directory / "private/registry.json",
            )
        elif args.command == "seal":
            record = seal.seal()
            result = {"state": record["state"], "plan_sha256": record["plan_sha256"]}
        elif args.command == "setup":
            import os

            result = setup(args.root, os.environ[args.database_url_env])
        elif args.command == "run":
            plan = seal.plan()
            if not args.mock and args.real_authorization != plan.authorization:
                raise ValueError("h_explicit_formal_send_authorization_required")
            result = asyncio.run(
                run(
                    args.root,
                    transport_override=mock_transport(args.mock) if args.mock else None,
                    resume=args.resume,
                    real_authorization=args.real_authorization,
                )
            )
        elif args.command == "settle-unknown":
            from .eval_h_journal import Journal

            plan = seal.plan()
            if (
                args.case,
                args.seed,
                "single_agent",
            ) not in plan.observation_keys() or seal.record()["state"] != "sealed":
                raise ValueError("h_unknown_reconciliation_key_or_state")
            journal = Journal(args.root)
            journal.recover_unknown(args.case, args.seed, "single_agent")
            result = {
                "attempts": len(journal.attempts(args.case, args.seed, "single_agent")),
                "network_calls": 0,
                "outcome": "unknown_kept_as_failure_no_resend",
            }
        elif args.command == "complete":
            record = seal.complete()
            result = {
                "state": record["state"],
                "completion_sha256": record["completion_sha256"],
            }
        elif args.command == "release":
            result = {"labels": len(seal.release())}
        elif args.command == "score":
            value = score(args.root)
            write(seal.directory / "score.json", value)
            seal.scored(seal.directory / "score.json")
            result = {
                "score_sha256": sha(seal.directory / "score.json"),
                "stage_acceptance": False,
            }
        elif args.command == "report":
            value = report(args.root)
            write(seal.directory / "result.json", value)
            result = {
                "comparison_complete": value["Q-12"]["complete"],
                "stage_acceptance": False,
            }
        elif args.command == "replay":
            result = replay(args.root)
        elif args.command == "audit":
            result = audit(args.root)
        elif args.command == "freeze":
            result = freeze(args.root, args.output)
        else:
            result = verify_freeze(args.root, args.frozen)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return (
        0
        if args.command not in {"audit", "replay", "preflight"}
        or result.get(
            {
                "audit": "audit_passed",
                "replay": "verified",
                "preflight": "ready_for_formal",
            }[args.command]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
