"""Versioned offline G score, replay, binding and freeze entry points."""

import argparse
import hashlib
import json
from pathlib import Path

from .gold_v3_contracts import payload_sha256


def main_score_cli() -> None:
    from .precise_dev_score import score_sweep, DEV_SCORE_V3
    from .eval_g_consumption import consumption_version, verify_source_v5

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "recheck"))
    parser.add_argument("registration", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, default=None)
    parser.add_argument("--source-bindings", type=Path, default=None)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    observations = json.loads(args.observations.read_text())
    bindings = json.loads(args.bindings.read_text()) if args.bindings else None
    if (
        registration.get("extends", {}).get("extension_version")
        == "v1-07-truth-extension.v3"
    ):
        from .eval_f_validation import verify_evidence_source

        if args.source_bindings is None:
            raise SystemExit("v4 score/recheck needs --source-bindings")
        (
            verify_source_v5
            if consumption_version(registration)
            else verify_evidence_source
        )(registration, args.source_bindings)
    result = score_sweep(registration, observations, bindings)
    if result["score_version"] in {
        DEV_SCORE_V3,
        "v1-07g-dev-score.v4",
        "v1-07g-dev-score.v5",
    }:
        hashes = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("registration", args.registration),
                ("observations", args.observations),
                ("bindings", args.bindings),
                ("source_bindings", args.source_bindings),
            )
            if path
        }
        result["inputs"]["file_sha256"] = hashes
        if (bindings or {}).get("observations_file_sha256") != hashes["observations"]:
            result["failed_gates"].append("raw_file_binding")
            result["stage_acceptance"] = False
    if args.command == "recheck":
        stored = json.loads(args.output.read_text())
        recomputed = json.loads(json.dumps(result))
        current = result["score_version"] == "v1-07g-dev-score.v5"
        if current and type(stored) is not dict:
            raise SystemExit("recheck mismatch at score_contract")
        for key in set(stored) | set(recomputed):
            different = stored.get(key) != recomputed.get(key)
            if current:
                different = (
                    key not in stored
                    or key not in recomputed
                    or payload_sha256(stored[key]) != payload_sha256(recomputed[key])
                )
            if different:
                raise SystemExit(f"recheck mismatch at {key}")
        print(
            json.dumps(
                {
                    "recheck": "identical",
                    "semantics_version": recomputed["semantics_version"],
                    "stage_acceptance": recomputed["stage_acceptance"],
                    "failed_gates": recomputed["failed_gates"],
                },
                ensure_ascii=False,
            )
        )
        return
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    )
    print(
        json.dumps(
            {
                "written": str(args.output),
                "stage_acceptance": result["stage_acceptance"],
                "failed_gates": result["failed_gates"],
            },
            ensure_ascii=False,
        )
    )


def main_sweep_cli() -> None:
    from .eval_f_validation import freeze_sweep_v4, verify_evidence_source
    from .eval_g_consumption import (
        extend_registration_v5,
        freeze_sweep_v5,
        verify_source_v5,
    )
    from .precise_dev_sweep import (
        registration,
        extend_registration,
        bind_runs,
        seed,
        run_sweep,
    )
    from .precise_candidate import extend_registration_v3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "seed",
            "run",
            "extend",
            "bind",
            "extend-v3",
            "bind-v3",
            "extend-v4",
            "bind-v4",
            "freeze-v4",
            "extend-v5",
            "bind-v5",
            "freeze-v5",
        ),
    )
    parser.add_argument("path", type=Path, help="registration, state or output path")
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--observations", type=Path, default=None)
    parser.add_argument("--source-bindings", type=Path, default=None)
    parser.add_argument("--bindings", type=Path, default=None)
    parser.add_argument("--score", type=Path, default=None)
    parser.add_argument("--source-inventory", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "prepare":
        args.path.write_text(
            json.dumps(registration(), ensure_ascii=False, sort_keys=True, indent=2)
        )
        print(json.dumps({"prepared": str(args.path)}, ensure_ascii=False))
    elif args.command in {"extend-v4", "extend-v5"}:
        from .eval_f_validation import extend_registration_v4

        if args.base is None or args.source_bindings is None:
            raise SystemExit("extend-v4 needs --base and --source-bindings")
        print(
            json.dumps(
                extend_registration_v5(
                    args.base, args.source_bindings, args.path, args.source_inventory
                )
                if args.command == "extend-v5"
                else extend_registration_v4(args.base, args.source_bindings, args.path)
            )
        )
    elif args.command in {"freeze-v4", "freeze-v5"}:
        if any(
            value is None
            for value in (
                args.observations,
                args.bindings,
                args.source_bindings,
                args.score,
                args.output,
            )
        ):
            raise SystemExit(
                "freeze-v4 needs --observations, --bindings, --source-bindings, --score and --output"
            )
        print(
            json.dumps(
                (freeze_sweep_v5 if args.command == "freeze-v5" else freeze_sweep_v4)(
                    args.path,
                    args.observations,
                    args.bindings,
                    args.source_bindings,
                    args.score,
                    args.output,
                )
            )
        )
    elif args.command in {"extend", "extend-v3"}:
        if args.base is None:
            raise SystemExit("extend needs --base")
        extend = (
            extend_registration_v3
            if args.command == "extend-v3"
            else extend_registration
        )
        print(json.dumps(extend(args.base, args.path), ensure_ascii=False))
    elif args.command in {"bind-v3", "bind-v4", "bind-v5"}:
        from .eval_f_lineage import bind_sweep_v3

        if args.command in {"bind-v4", "bind-v5"}:
            if args.source_bindings is None:
                raise SystemExit("bind-v4 needs --source-bindings")
            (verify_source_v5 if args.command == "bind-v5" else verify_evidence_source)(
                json.loads(args.path.read_bytes()), args.source_bindings
            )
        if args.state is None or args.output is None or args.observations is None:
            raise SystemExit("bind-v3 needs --state, --output and --observations")
        print(
            json.dumps(
                bind_sweep_v3(
                    args.state,
                    args.path,
                    args.observations,
                    args.output,
                    binding_version="v1-07g-db-binding.v4"
                    if args.command in {"bind-v4", "bind-v5"}
                    else "v1-07g-db-binding.v3",
                ),
                ensure_ascii=False,
            )
        )
    elif args.command == "bind":
        if args.state is None or args.output is None:
            raise SystemExit("bind needs --state and --output")
        print(
            json.dumps(
                bind_runs(args.state, args.path, args.output), ensure_ascii=False
            )
        )
    elif args.command == "seed":
        print(json.dumps(seed(args.path), ensure_ascii=False))
    else:
        if args.state is None or args.output is None:
            raise SystemExit("run needs --state and --output")
        print(
            json.dumps(
                run_sweep(args.path, args.state, args.output, repeats=args.repeats),
                ensure_ascii=False,
            )
        )
