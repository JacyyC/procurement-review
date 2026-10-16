"""Formal H seal v2; the old disposable v1 protocol remains readable."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

from .blind_seal import BlindState
from .eval_h_contracts import (
    H_CASES,
    H_GENERATOR,
    H_SEEDED_GENERATOR,
    H_LABELS,
    H_SEAL,
    HCase,
    HLabel,
    HPlanLike,
    HRecipe,
    load_h_plan,
)
from .eval_h_exclusion import check, verify
from .eval_h_storage import locked, private, read, sha, write
from .gold_v3_contracts import payload_sha256


def plan_digest(plan: HPlanLike) -> str:
    return payload_sha256(plan.model_dump(mode="json"))


class HSeal:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @classmethod
    def create(cls, root: Path, plan: HPlanLike):
        from .eval_h_identity import require_identity

        candidate = require_identity(plan)
        root = Path(root).absolute()
        if ".local" not in root.parts or root.is_symlink():
            raise ValueError("h_private_local_root_required")
        root.mkdir(parents=True, exist_ok=True)
        directory = root / plan.batch_id
        directory.mkdir(mode=0o700)
        for name in ("runner", "private"):
            (directory / name).mkdir(mode=0o700)
        write(directory / "plan.json", plan.model_dump(mode="json"))
        write(directory / "candidate.json", candidate)
        write(
            directory / "state.json",
            dict(
                schema_version=H_SEAL,
                state=BlindState.CANDIDATE_FROZEN.value,
                plan_sha256=plan_digest(plan),
                transitions=[BlindState.CANDIDATE_FROZEN.value],
            ),
        )
        return cls(directory)

    def plan(self) -> HPlanLike:
        private(self.directory, directory=True)
        private(self.directory / "plan.json")
        return load_h_plan((self.directory / "plan.json").read_bytes())

    def record(self) -> dict:
        private(self.directory / "state.json")
        record = read(self.directory / "state.json")
        if record.get("schema_version") != H_SEAL or record.get(
            "plan_sha256"
        ) != plan_digest(self.plan()):
            raise ValueError("h_seal_plan_identity")
        states = [state.value for state in BlindState]
        if record.get("state") not in states:
            raise ValueError("h_seal_state")
        if record.get("transitions") != states[: states.index(record["state"]) + 1]:
            raise ValueError("h_seal_transition_history")
        return record

    def _require(self, state: BlindState) -> dict:
        record = self.record()
        if record["state"] != state.value:
            raise ValueError("h_seal_expected:" + state.value)
        return record

    def _advance(self, record, state, **fields):
        record = {
            **record,
            **fields,
            "state": state.value,
            "transitions": [*record["transitions"], state.value],
        }
        write(self.directory / "state.json", record, replace=True)
        return record

    def generate(
        self, recipe_path: Path, seed_path: Path, exclusion_path: Path
    ) -> dict:
        from .eval_h_identity import require_identity

        with locked(self.directory):
            record = self._require(BlindState.CANDIDATE_FROZEN)
            plan = self.plan()
            require_identity(plan)
            if (
                sha(recipe_path) != plan.generator_spec_sha256
                or sha(seed_path) != plan.seed_sha256
            ):
                raise ValueError("h_generation_spec_seed_identity")
            registry = read(exclusion_path)
            verify(registry)
            if registry["registry_sha256"] != plan.exclusion_sha256:
                raise ValueError("h_generation_exclusion_identity")
            definition = read(recipe_path)
            seed = Path(seed_path).read_bytes()
            if definition.get("generator_version") == H_SEEDED_GENERATOR:
                from .eval_h_generation import HGenerationSpec, generate

                spec = HGenerationSpec.model_validate_json(
                    Path(recipe_path).read_bytes(), strict=True
                )
                if plan.mode == "formal":
                    spec.require_formal_quotas()
                recipe, selection = generate(spec, seed)
            else:
                if plan.mode == "formal":
                    raise ValueError("h_formal_requires_seeded_generation_spec")
                recipe = HRecipe.model_validate_json(
                    Path(recipe_path).read_bytes(), strict=True
                )
                selection = dict(
                    generator_version=H_GENERATOR, mode="disposable_static"
                )
            if set(case.case_id for case in recipe.cases) != set(plan.case_ids):
                raise ValueError("h_generation_plan_case_set")
            if plan.mode == "formal":
                recipe.require_formal_quotas()
            exclusion = check(registry, recipe.cases, formal=plan.mode == "formal")
            if len(seed) < 16:
                raise ValueError("h_generation_seed_too_short")
            ordered = sorted(
                recipe.cases,
                key=lambda case: hmac.new(
                    seed, case.case_id.encode(), hashlib.sha256
                ).digest(),
            )
            labels = {label.case_id: label for label in recipe.labels}
            cases = dict(
                schema_version=H_CASES,
                generator_version=recipe.generator_version,
                plan_sha256=record["plan_sha256"],
                mode=plan.mode,
                cases=[case.model_dump(mode="json") for case in ordered],
            )
            truth = dict(
                schema_version=H_LABELS,
                generator_version=recipe.generator_version,
                plan_sha256=record["plan_sha256"],
                mode=plan.mode,
                labels=[
                    labels[case.case_id].model_dump(mode="json") for case in ordered
                ],
            )
            # A retry after an interrupted generation can reuse only identical bytes.
            for path, body in (
                (self.directory / "runner/cases.json", cases),
                (self.directory / "private/labels.json", truth),
                (self.directory / "private/generation.json", selection),
                (self.directory / "private/exclusion.json", exclusion),
            ):
                if path.exists():
                    private(path)
                    if payload_sha256(read(path)) != payload_sha256(body):
                        raise ValueError("h_interrupted_generation_payload_changed")
                else:
                    write(path, body)
            return self._advance(
                record,
                BlindState.GENERATED,
                generator_version=recipe.generator_version,
                cases_sha256=sha(self.directory / "runner/cases.json"),
                labels_sha256=sha(self.directory / "private/labels.json"),
                generation_sha256=sha(self.directory / "private/generation.json"),
                exclusion_result_sha256=sha(self.directory / "private/exclusion.json"),
            )

    def _commitment(self, record, key):
        message = {
            name: record[name]
            for name in (
                "schema_version",
                "plan_sha256",
                "generator_version",
                "cases_sha256",
                "labels_sha256",
                "generation_sha256",
                "exclusion_result_sha256",
                "preflight_sha256",
            )
        }
        return hmac.new(
            key, payload_sha256(message).encode(), hashlib.sha256
        ).hexdigest()

    def seal(self) -> dict:
        with locked(self.directory):
            record = self._require(BlindState.GENERATED)
            self._check_payloads(record)
            from .eval_h_cli import preflight

            readiness = preflight(self.directory)
            preflight_path = self.directory / "runner/preflight.json"
            body = dict(
                schema_version="v1-07h-send-preflight.v1",
                plan_sha256=record["plan_sha256"],
                checks=readiness["checks"],
            )
            if preflight_path.exists():
                private(preflight_path)
                if payload_sha256(read(preflight_path)) != payload_sha256(body):
                    raise ValueError("h_interrupted_send_preflight_changed")
            else:
                write(preflight_path, body)
            record = {**record, "preflight_sha256": sha(preflight_path)}
            key_path = self.directory / "private/commitment.key"
            if not key_path.exists():
                descriptor = os.open(
                    key_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as output:
                    output.write(secrets.token_bytes(32))
                    output.flush()
                    os.fsync(output.fileno())
            private(key_path)
            key = key_path.read_bytes()
            if len(key) != 32:
                raise ValueError("h_commitment_key_size")
            return self._advance(
                record,
                BlindState.SEALED,
                key_sha256=sha(key_path),
                commitment=self._commitment(record, key),
            )

    def _check_payloads(self, record):
        for name, path in (
            ("cases_sha256", "runner/cases.json"),
            ("labels_sha256", "private/labels.json"),
            ("exclusion_result_sha256", "private/exclusion.json"),
            ("generation_sha256", "private/generation.json"),
        ):
            private(self.directory / path)
            if sha(self.directory / path) != record.get(name):
                raise ValueError("h_sealed_payload_changed:" + name)
        if "preflight_sha256" in record:
            private(self.directory / "runner/preflight.json")
            if (
                sha(self.directory / "runner/preflight.json")
                != record["preflight_sha256"]
            ):
                raise ValueError("h_sealed_send_preflight_changed")

    def _verify_commitment(self, record):
        self._check_payloads(record)
        key_path = self.directory / "private/commitment.key"
        private(key_path)
        if sha(key_path) != record.get("key_sha256") or not hmac.compare_digest(
            self._commitment(record, key_path.read_bytes()),
            record.get("commitment", ""),
        ):
            raise ValueError("h_sealed_commitment_changed")

    def runner_cases(self) -> tuple[HCase, ...]:
        # This accessor never opens the Label, seed, key or exclusion-private files.
        record = self.record()
        if record["state"] not in {
            state.value
            for state in BlindState
            if state not in (BlindState.CANDIDATE_FROZEN, BlindState.GENERATED)
        }:
            raise ValueError("h_runner_cases_not_sealed")
        path = self.directory / "runner/cases.json"
        private(path)
        if sha(path) != record["cases_sha256"]:
            raise ValueError("h_runner_material_checksum")
        body = read(path)
        if (
            set(body)
            != {"schema_version", "generator_version", "plan_sha256", "mode", "cases"}
            or body["schema_version"] != H_CASES
        ):
            raise ValueError("h_runner_case_package_contract")
        plan = self.plan()
        if (
            body["mode"] != plan.mode
            or body["plan_sha256"] != plan_digest(plan)
            or body["generator_version"] != record["generator_version"]
        ):
            raise ValueError("h_runner_case_package_binding")
        cases = tuple(
            HCase.model_validate_json(json.dumps(row), strict=True)
            for row in body["cases"]
        )
        if len(cases) != len(plan.case_ids) or set(
            case.case_id for case in cases
        ) != set(plan.case_ids):
            raise ValueError("h_runner_case_package_set")
        return cases

    def complete(self) -> dict:
        from .eval_h_consumption import completion_proof

        with locked(self.directory):
            record = self._require(BlindState.SEALED)
            self._verify_commitment(record)
            proof = completion_proof(self.directory)
            if not proof["complete"]:
                raise ValueError("h_run_incomplete:" + ",".join(proof["failures"]))
            path = self.directory / "private/completion.json"
            if path.exists():
                if payload_sha256(read(path)) != payload_sha256(proof):
                    raise ValueError("h_completion_proof_changed")
            else:
                write(path, proof)
            return self._advance(
                record, BlindState.RUN_COMPLETE, completion_sha256=sha(path)
            )

    def release(self) -> tuple[HLabel, ...]:
        with locked(self.directory):
            record = self._require(BlindState.RUN_COMPLETE)
            self._verify_commitment(record)
            self._verify_completion(record)
            labels = self._labels(record)
            self._advance(record, BlindState.LABELS_RELEASED)
            return labels

    def _verify_completion(self, record):
        from .eval_h_consumption import completion_proof

        path = self.directory / "private/completion.json"
        if not path.exists() or sha(path) != record.get("completion_sha256"):
            raise ValueError("h_completion_proof_missing_or_changed")
        private(path)
        proof = completion_proof(self.directory)
        if proof["complete"] is not True or payload_sha256(proof) != payload_sha256(
            read(path)
        ):
            raise ValueError("h_completion_proof_no_longer_matches")

    def _labels(self, record):
        body = read(self.directory / "private/labels.json")
        if (
            set(body)
            != {"schema_version", "generator_version", "plan_sha256", "mode", "labels"}
            or body["schema_version"] != H_LABELS
        ):
            raise ValueError("h_label_package_contract")
        plan = self.plan()
        if (
            body["plan_sha256"] != record["plan_sha256"]
            or body["mode"] != plan.mode
            or body["generator_version"] != record["generator_version"]
        ):
            raise ValueError("h_label_package_binding")
        labels = tuple(
            HLabel.model_validate_json(json.dumps(row), strict=True)
            for row in body["labels"]
        )
        if len(labels) != len(plan.case_ids) or set(
            label.case_id for label in labels
        ) != set(plan.case_ids):
            raise ValueError("h_label_package_case_set")
        return labels

    def scorer_labels(self) -> tuple[HLabel, ...]:
        record = self.record()
        if record["state"] not in {
            BlindState.LABELS_RELEASED.value,
            BlindState.SCORED.value,
        }:
            raise ValueError("h_labels_not_released")
        self._verify_commitment(record)
        self._verify_completion(record)
        return self._labels(record)

    def scored(self, score_path: Path) -> dict:
        from .eval_h_score import rebuild

        with locked(self.directory):
            record = self._require(BlindState.LABELS_RELEASED)
            if (
                Path(score_path).absolute()
                != (self.directory / "score.json").absolute()
            ):
                raise ValueError("h_score_registered_path_required")
            if payload_sha256(read(score_path)) != payload_sha256(
                rebuild(self.directory)
            ):
                raise ValueError("h_full_score_mismatch")
            return self._advance(
                record, BlindState.SCORED, score_sha256=sha(score_path)
            )
