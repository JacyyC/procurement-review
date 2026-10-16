"""Disposable blind-batch seal used to verify the V1-07 protocol in Stage B.

Formal blind content is intentionally out of scope until Stage H.  This implementation
therefore accepts only explicitly disposable batches under a ``.local`` directory while
exercising the same commitment, permission, completion and release state transitions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

BLIND_SEAL_VERSION = "blind-seal.v1"
STATE_FILE = "state.json"
RUNNER_DIR = "runner"
PRIVATE_DIR = "private"
CASES_FILE = "cases.json"
LABELS_FILE = "labels.json"
KEY_FILE = "commitment.key"


class BlindSealError(RuntimeError):
    """A transition or integrity check violates the frozen seal protocol."""


class BlindState(StrEnum):
    CANDIDATE_FROZEN = "candidate_frozen"
    GENERATED = "generated"
    SEALED = "sealed"
    RUN_COMPLETE = "run_complete"
    LABELS_RELEASED = "labels_released"
    SCORED = "scored"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )


class SealRecord(_FrozenModel):
    schema_version: Literal["blind-seal.v1"] = BLIND_SEAL_VERSION
    batch_id: Annotated[str, Field(pattern=r"^disposable-[a-z0-9][a-z0-9-]{2,63}$")]
    disposable: Literal[True] = True
    candidate_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    seed_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    state: BlindState
    expected_arms: tuple[Annotated[str, Field(pattern=r"^[a-z0-9_-]+$")], ...]
    completed_arms: tuple[Annotated[str, Field(pattern=r"^[a-z0-9_-]+$")], ...] = ()
    cases_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    labels_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    commitment: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    key_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    transitions: tuple[BlindState, ...]

    @model_validator(mode="after")
    def state_is_closed(self) -> Self:
        if not self.expected_arms or len(self.expected_arms) != len(
            set(self.expected_arms)
        ):
            raise ValueError("blind seal expected arms must be non-empty and unique")
        if tuple(sorted(self.expected_arms)) != self.expected_arms:
            raise ValueError("blind seal expected arms must be sorted")
        if len(self.completed_arms) != len(set(self.completed_arms)):
            raise ValueError("blind seal completed arms must be unique")
        if not set(self.completed_arms).issubset(self.expected_arms):
            raise ValueError("blind seal completed an undeclared arm")
        if not self.transitions or self.transitions[-1] is not self.state:
            raise ValueError("blind seal transition history must end at current state")
        states = list(BlindState)
        positions = [states.index(item) for item in self.transitions]
        if positions != list(range(len(positions))):
            raise ValueError("blind seal transitions must be contiguous")
        if self.state.value in {
            BlindState.CANDIDATE_FROZEN.value,
            BlindState.GENERATED.value,
        }:
            if self.commitment is not None or self.key_sha256 is not None:
                raise ValueError("unsealed state cannot carry a commitment")
        if self.state is not BlindState.CANDIDATE_FROZEN:
            if self.cases_sha256 is None or self.labels_sha256 is None:
                raise ValueError("generated state requires both payload hashes")
        if states.index(self.state) >= states.index(BlindState.SEALED):
            if self.commitment is None or self.key_sha256 is None:
                raise ValueError("sealed state requires commitment identity")
        if states.index(self.state) >= states.index(BlindState.RUN_COMPLETE):
            if set(self.completed_arms) != set(self.expected_arms):
                raise ValueError("run_complete requires every pre-registered arm")
        return self


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, body: bytes, *, mode: int = 0o600) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _contains_formal_blind(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (str(key).lower() == "split" and str(item).lower() == "blind")
            or (str(key).lower() == "disposable" and item is not True)
            or _contains_formal_blind(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_formal_blind(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class DisposableBlindSeal:
    directory: Path

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        batch_id: str,
        candidate_fingerprint: str,
        seed_material: bytes,
        expected_arms: tuple[str, ...],
        disposable: bool = True,
    ) -> DisposableBlindSeal:
        if not disposable:
            raise BlindSealError("formal blind generation is reserved for Stage H")
        try:
            record = SealRecord(
                batch_id=batch_id,
                candidate_fingerprint=candidate_fingerprint,
                seed_fingerprint=_digest(seed_material),
                state=BlindState.CANDIDATE_FROZEN,
                expected_arms=tuple(sorted(expected_arms)),
                transitions=(BlindState.CANDIDATE_FROZEN,),
            )
        except ValueError as error:
            raise BlindSealError("invalid disposable blind batch contract") from error
        resolved = root.resolve()
        if ".local" not in resolved.parts:
            raise BlindSealError("disposable blind batches must live under .local")
        resolved.mkdir(parents=True, exist_ok=True)
        resolved.chmod(0o700)
        directory = resolved / batch_id
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError as error:
            raise BlindSealError("blind batch already exists") from error
        (directory / RUNNER_DIR).mkdir(mode=0o700)
        (directory / PRIVATE_DIR).mkdir(mode=0o700)
        _atomic_write(
            directory / STATE_FILE, _json_bytes(record.model_dump(mode="json"))
        )
        return cls(directory=directory)

    @property
    def runner_cases_path(self) -> Path:
        return self.directory / RUNNER_DIR / CASES_FILE

    def _record(self) -> SealRecord:
        try:
            return SealRecord.model_validate_json(
                (self.directory / STATE_FILE).read_bytes()
            )
        except (OSError, ValueError) as error:
            raise BlindSealError("blind seal state is unreadable") from error

    def _write_record(self, record: SealRecord) -> None:
        _atomic_write(
            self.directory / STATE_FILE,
            _json_bytes(record.model_dump(mode="json")),
        )

    def _require(self, state: BlindState) -> SealRecord:
        record = self._record()
        if record.state is not state:
            raise BlindSealError(
                f"blind seal state is {record.state.value}, expected {state.value}"
            )
        return record

    def generate(self, *, cases: Any, labels: Any) -> SealRecord:
        record = self._require(BlindState.CANDIDATE_FROZEN)
        if _contains_formal_blind(cases) or _contains_formal_blind(labels):
            raise BlindSealError("Stage B cannot create formal blind content")
        cases_body = _json_bytes(cases)
        labels_body = _json_bytes(labels)
        staging = self.directory / f".generated-{secrets.token_hex(8)}"
        staging.mkdir(mode=0o700)
        try:
            runner = staging / RUNNER_DIR
            private = staging / PRIVATE_DIR
            runner.mkdir(mode=0o700)
            private.mkdir(mode=0o700)
            _atomic_write(runner / CASES_FILE, cases_body)
            _atomic_write(private / LABELS_FILE, labels_body)
            os.replace(runner / CASES_FILE, self.runner_cases_path)
            os.replace(
                private / LABELS_FILE,
                self.directory / PRIVATE_DIR / LABELS_FILE,
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        generated = record.model_copy(
            update={
                "state": BlindState.GENERATED,
                "cases_sha256": _digest(cases_body),
                "labels_sha256": _digest(labels_body),
                "transitions": record.transitions + (BlindState.GENERATED,),
            }
        )
        validated = SealRecord.model_validate(generated.model_dump(mode="python"))
        self._write_record(validated)
        return validated

    def _commitment_message(self, record: SealRecord) -> bytes:
        return "\n".join(
            (
                record.batch_id,
                record.candidate_fingerprint,
                record.seed_fingerprint,
                record.cases_sha256 or "",
                record.labels_sha256 or "",
            )
        ).encode("utf-8")

    def seal(self) -> SealRecord:
        record = self._require(BlindState.GENERATED)
        cases = self.runner_cases_path.read_bytes()
        labels_path = self.directory / PRIVATE_DIR / LABELS_FILE
        labels = labels_path.read_bytes()
        if (
            _digest(cases) != record.cases_sha256
            or _digest(labels) != record.labels_sha256
        ):
            raise BlindSealError("generated blind payload changed before sealing")
        key = secrets.token_bytes(32)
        commitment = hmac.new(
            key, self._commitment_message(record), hashlib.sha256
        ).hexdigest()
        _atomic_write(self.directory / PRIVATE_DIR / KEY_FILE, key)
        sealed = record.model_copy(
            update={
                "state": BlindState.SEALED,
                "commitment": commitment,
                "key_sha256": _digest(key),
                "transitions": record.transitions + (BlindState.SEALED,),
            }
        )
        validated = SealRecord.model_validate(sealed.model_dump(mode="python"))
        self._write_record(validated)
        return validated

    def complete_arm(self, arm: str) -> SealRecord:
        record = self._require(BlindState.SEALED)
        if arm not in record.expected_arms:
            raise BlindSealError("arm was not pre-registered")
        if arm in record.completed_arms:
            return record
        updated = record.model_copy(
            update={"completed_arms": tuple(sorted((*record.completed_arms, arm)))}
        )
        validated = SealRecord.model_validate(updated.model_dump(mode="python"))
        self._write_record(validated)
        return validated

    def mark_run_complete(self) -> SealRecord:
        record = self._require(BlindState.SEALED)
        if set(record.completed_arms) != set(record.expected_arms):
            raise BlindSealError(
                "cannot complete blind run before every arm terminates"
            )
        updated = record.model_copy(
            update={
                "state": BlindState.RUN_COMPLETE,
                "transitions": record.transitions + (BlindState.RUN_COMPLETE,),
            }
        )
        validated = SealRecord.model_validate(updated.model_dump(mode="python"))
        self._write_record(validated)
        return validated

    def release_labels(self) -> bytes:
        record = self._require(BlindState.RUN_COMPLETE)
        cases = self.runner_cases_path.read_bytes()
        labels = (self.directory / PRIVATE_DIR / LABELS_FILE).read_bytes()
        key = (self.directory / PRIVATE_DIR / KEY_FILE).read_bytes()
        if _digest(cases) != record.cases_sha256:
            raise BlindSealError("sealed cases were tampered with")
        if _digest(labels) != record.labels_sha256:
            raise BlindSealError("sealed labels were tampered with")
        if _digest(key) != record.key_sha256:
            raise BlindSealError("blind commitment key was tampered with")
        observed = hmac.new(
            key,
            self._commitment_message(record),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(observed, record.commitment or ""):
            raise BlindSealError("blind commitment mismatch")
        updated = record.model_copy(
            update={
                "state": BlindState.LABELS_RELEASED,
                "transitions": record.transitions + (BlindState.LABELS_RELEASED,),
            }
        )
        validated = SealRecord.model_validate(updated.model_dump(mode="python"))
        self._write_record(validated)
        return labels

    def mark_scored(self) -> SealRecord:
        record = self._require(BlindState.LABELS_RELEASED)
        updated = record.model_copy(
            update={
                "state": BlindState.SCORED,
                "transitions": record.transitions + (BlindState.SCORED,),
            }
        )
        validated = SealRecord.model_validate(updated.model_dump(mode="python"))
        self._write_record(validated)
        return validated

    def destroy(self) -> None:
        record = self._record()
        resolved = self.directory.resolve()
        if not record.disposable or ".local" not in resolved.parts:
            raise BlindSealError("refusing to remove a non-disposable blind directory")
        if (
            not record.batch_id.startswith("disposable-")
            or resolved.name != record.batch_id
        ):
            raise BlindSealError("blind directory identity mismatch")
        shutil.rmtree(resolved)


__all__ = [
    "BLIND_SEAL_VERSION",
    "BlindSealError",
    "BlindState",
    "DisposableBlindSeal",
    "SealRecord",
]
