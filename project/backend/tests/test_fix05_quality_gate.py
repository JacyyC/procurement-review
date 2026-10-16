"""FIX-05A: the quality gate's measurement lifecycle is one coordinated contract.

Every case here started as an independent reproduction of a FIX-01D audit finding
(D-AUD-01 .. D-AUD-05) and is now a regression test over the *real* tool:

* bootstrap is an explicit first-time path that a repository with a provable
  contract can never take, and the first-time path is reachable from the real
  command line without hand-written success records;
* the lifecycle records are closed-form: every damaged shape is refused before
  anything is written, and no ``KeyError``/``TypeError``/``JSONDecodeError``
  escapes as a traceback;
* run, update and bootstrap share one work-tree lock, and an older run can never
  publish over a newer one;
* a measurement is bound to the exact content, run identity and old baseline
  reviewed at the final write, not just at the start;
* publication has one commit point, an intent record, and a recovery mode that
  only accepts the expected old/new hash;
* deleting, renaming, splitting or merging a guarded path needs an explicit
  migration declaration, and a split inherits the *same* budget.

The environment is synthetic (``Sandbox`` from the FIX-01D suite, copied gate
source, stub governance scripts, real Git repository); the lifecycle, manifest,
ratchet, publication, update, bootstrap and migration logic under test is the
shipped code.  Only the expensive external sections are replaced, and only inside
the sandbox copy of the gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_SOURCE = REPO_ROOT / "project" / "scripts" / "quality_gate.py"
D_SUITE = REPO_ROOT / "project" / "backend" / "tests" / "test_fix01d_quality_gate.py"

GUARDED = "project/backend/src/procurement_review/review_repository.py"
PART_A = "project/backend/src/procurement_review/repo_part_a.py"
PART_B = "project/backend/src/procurement_review/repo_part_b.py"


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


GATE = _load(GATE_SOURCE, "fix05_gate_contract")
SUPPORT = _load(D_SUITE, "fix05_sandbox_support")
Sandbox = SUPPORT.Sandbox

CHILD_DRIVER = '''"""Run one sandbox gate command in its own process, optionally behind a barrier."""

import importlib.util
import sys
import time
from pathlib import Path


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def wait(path, seconds=90.0):
    deadline = time.monotonic() + seconds
    while not Path(path).exists():
        if time.monotonic() > deadline:
            raise SystemExit("barrier timeout")
        time.sleep(0.01)


gate = load(Path(sys.argv[1]), "child_gate")
mode = sys.argv[2]
entered = Path(sys.argv[3])
release = Path(sys.argv[4])
failure = sys.argv[5]


def park():
    if str(entered) != "-":
        entered.touch()
        wait(release)


def section_gov():
    park()
    gate.section_gov()


def section_backend(measured):
    park()
    if failure == "backend":
        raise gate.GateFailure("backend", "injected external test failure")
    measured["backend"] = {"line_percent": 87.5, "branch_percent": 72.5}


def section_frontend(measured):
    measured["frontend"] = {"line_percent": 96.0, "branch_percent": 90.0}


gate.SECTIONS = (
    ("gov", section_gov),
    ("ruff", lambda: None),
    ("backend", section_backend),
    ("frontend", section_frontend),
    ("ratchet", gate.section_ratchet),
)

if mode == "run":
    raise SystemExit(gate.run_gate())
if mode == "lock-crash":
    with gate.gate_lock("crash"):
        raise RuntimeError("injected crash while holding the work-tree lock")
raise SystemExit(gate.main([mode]))
'''

PARKING_GOVERNANCE_STUB = """#!/usr/bin/env python3
import sys
import time
from pathlib import Path

marker = Path(__file__).resolve().parents[2] / "governance-failure"
if marker.exists():
    sys.exit(1)
entered = Path({entered!r})
release = Path({release!r})
entered.touch()
deadline = time.monotonic() + 90
while not release.exists():
    if time.monotonic() > deadline:
        sys.exit(3)
    time.sleep(0.01)
sys.exit(0)
"""

LOCK_PROBE = '''"""Ask the operating system whether this process can take the gate lock."""

import fcntl
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
handle = open(path, "a+")
try:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as exc:
    print(json.dumps({{"acquired": False, "error": str(exc)}}))
    raise SystemExit(0)
print(json.dumps({{"acquired": True}}))
'''


def _exact_lines(path: Path, lines: int, header: str = '"""Split part."""') -> None:
    body = [header] + [
        f"def helper_{index}():  # padding" for index in range(lines - 1)
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    assert len(path.read_text(encoding="utf-8").splitlines()) == lines


class Harness:
    """The lifecycle facts a test needs to build or damage one synthetic state."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self.module = sandbox.module
        self.scratch = sandbox.root.parent
        self.driver = self.scratch / "child-driver.py"
        self.driver.write_text(CHILD_DRIVER, encoding="utf-8")
        self.lock_probe = self.scratch / "lock-probe.py"
        self.lock_probe.write_text(LOCK_PROBE.format(), encoding="utf-8")

    # ------------------------------------------------------------------ workloads

    def install_workload(self, *, failure: str | None = None) -> None:
        module = self.module

        def backend(measured: dict) -> None:
            if failure == "backend":
                raise module.GateFailure("backend", "injected external test failure")
            measured["backend"] = {"line_percent": 87.5, "branch_percent": 72.5}

        def frontend(measured: dict) -> None:
            measured["frontend"] = {"line_percent": 96.0, "branch_percent": 90.0}

        module.SECTIONS = (
            ("gov", module.section_gov),
            ("ruff", lambda: None),
            ("backend", backend),
            ("frontend", frontend),
            ("ratchet", module.section_ratchet),
        )

    def green_run(self) -> dict:
        self.install_workload()
        assert self.module.run_gate() == 0
        return self.read_state("run.json")

    def park_governance(self, entered: Path, release: Path) -> None:
        """Make the sandbox's governance stubs stop where a real editor could run."""

        for script in (
            "workspace_views.py",
            "document_health_check.py",
            "check_markdown_links.py",
        ):
            path = self.sandbox.root / "workspace" / "scripts" / script
            path.write_text(
                PARKING_GOVERNANCE_STUB.format(
                    entered=str(entered), release=str(release)
                ),
                encoding="utf-8",
            )

    @staticmethod
    def wait(marker: Path | None, seconds: float = 90.0) -> None:
        deadline = time.monotonic() + seconds
        while marker is not None and not marker.exists():
            if time.monotonic() > deadline:
                raise AssertionError(f"barrier never reached: {marker}")
            time.sleep(0.01)

    def child(
        self,
        mode: str,
        *,
        entered: Path | None = None,
        release: Path | None = None,
        failure: str | None = None,
    ) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                str(self.driver),
                str(self.sandbox.gate_path),
                mode,
                str(entered) if entered is not None else "-",
                str(release) if release is not None else "-",
                failure or "-",
            ],
            cwd=self.sandbox.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def cli_popen(self, *arguments: str) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, str(self.sandbox.gate_path), *arguments],
            cwd=self.sandbox.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def finish(
        process: subprocess.Popen, *, timeout: float = 90.0
    ) -> tuple[int, str, str]:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        return process.returncode, stdout, stderr

    def lock_visibility(self) -> bool:
        """True when an independent process can take the work-tree lock."""

        completed = subprocess.run(
            [sys.executable, str(self.lock_probe), str(self.module.LOCK_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(json.loads(completed.stdout)["acquired"])

    # --------------------------------------------------------------------- records

    def manifest(self) -> dict:
        return self.module.build_manifest()

    def owner(self, **overrides: object) -> dict:
        payload = {
            "token": "0f0f0f0f0f0f0f0f",
            "pid": 4321,
            "purpose": "run",
            "started_at": "2026-09-15T12:00:00+00:00",
        }
        payload.update(overrides)
        return payload

    def run_record(self, **overrides: object) -> dict:
        module = self.module
        record: dict = {
            "record_version": module.RUN_RECORD_VERSION,
            "gate_version": module.GATE_VERSION,
            "run_id": "20260915T120000Z-aaaabbbb",
            "generation": 1,
            "status": "success",
            "started_at": "2026-09-15T12:00:00+00:00",
            "finished_at": "2026-09-15T12:10:00+00:00",
            "failed_section": None,
            "sections": [
                {"name": name, "status": "ok", "seconds": 0.1}
                for name in module.SECTION_NAMES
            ],
            "inputs": module.manifest_summary(self.manifest()),
            "baseline_state": (
                "present" if self.sandbox.baseline_path.is_file() else "never_created"
            ),
            "owner": self.owner(),
        }
        record.update(overrides)
        return record

    def success_record(self, **overrides: object) -> dict:
        module = self.module
        manifest = self.manifest()
        record: dict = {
            "record_version": module.SUCCESS_RECORD_VERSION,
            "gate_version": module.GATE_VERSION,
            "status": "success",
            "run_id": "20260915T120000Z-aaaabbbb",
            "generation": 1,
            "started_at": "2026-09-15T12:00:00+00:00",
            "finished_at": "2026-09-15T12:10:00+00:00",
            "sections": [
                {"name": name, "status": "ok", "seconds": 0.1}
                for name in module.SECTION_NAMES
            ],
            "measurement": self.measurement(),
            "baseline": self.baseline_reference(),
            "baseline_state": (
                "present" if self.sandbox.baseline_path.is_file() else "never_created"
            ),
            "inputs": {
                "rule": module.MANIFEST_RULE,
                "groups": json.loads(json.dumps(manifest["groups"])),
                "files": manifest["files"],
            },
            "owner": self.owner(),
        }
        record.update(overrides)
        return record

    def measurement(
        self,
        *,
        backend: tuple[float, float] = (87.5, 72.5),
        frontend: tuple[float, float] = (96.0, 90.0),
    ) -> dict:
        module = self.module
        return {
            "backend": {"line_percent": backend[0], "branch_percent": backend[1]},
            "frontend": {"line_percent": frontend[0], "branch_percent": frontend[1]},
            "sizes": dict(module.measure_sizes()),
            "api_factory": dict(module.measure_api_factory()),
            "measured_at": "2026-09-15T12:10:00+00:00",
            "git_head": self.sandbox.head,
            "gate_version": module.GATE_VERSION,
        }

    def baseline_reference(self) -> dict:
        if not self.sandbox.baseline_path.is_file():
            return {
                "path": "project/quality-baselines.v1.json",
                "present": False,
                "sha256": None,
                "baseline_version": None,
            }
        return {
            "path": "project/quality-baselines.v1.json",
            "present": True,
            "sha256": hashlib.sha256(self.sandbox.baseline_bytes()).hexdigest(),
            "baseline_version": 1,
        }

    def write(self, name: str, payload: dict) -> Path:
        path = self.state(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_text(payload), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.state(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read_state(self, name: str) -> dict:
        return json.loads(self.state(name).read_text(encoding="utf-8"))

    def state(self, name: str) -> Path:
        return self.sandbox.state / name

    def write_promotable_state(
        self, *, run: dict | None = None, success: dict | None = None
    ) -> tuple[dict, dict]:
        run_record = self.run_record() if run is None else run
        if success is None:
            success = self.success_record(
                run_id=run_record["run_id"], generation=run_record["generation"]
            )
        self.write("run.json", run_record)
        self.write("success.json", success)
        return run_record, success

    # ------------------------------------------------------------------- assertions

    def refuse(self, *arguments: str, expect: str) -> None:
        before = self.sandbox.snapshot()
        completed = self.sandbox.run_cli(*arguments)
        combined = completed.stdout + completed.stderr
        assert completed.returncode != 0, combined
        assert "Traceback" not in combined, combined
        assert "baseline unchanged" in completed.stderr, combined
        assert expect in combined, combined
        assert self.sandbox.snapshot() == before

    def refuse_any(self, *arguments: str) -> None:
        before = self.sandbox.snapshot()
        completed = self.sandbox.run_cli(*arguments)
        combined = completed.stdout + completed.stderr
        assert completed.returncode != 0, combined
        assert "Traceback" not in combined, combined
        assert "baseline unchanged" in completed.stderr, combined
        assert self.sandbox.snapshot() == before

    def promote(self, *arguments: str) -> subprocess.CompletedProcess:
        completed = self.sandbox.run_cli(*arguments)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed

    def expect_run_failure(self, expect: str, capsys: pytest.CaptureFixture) -> None:
        """A broken contract must stop the gate itself, by name."""

        self.install_workload()
        assert self.module.run_gate() == 1
        captured = capsys.readouterr()
        assert expect in captured.err, captured.err


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    return Harness(Sandbox(tmp_path / "repo"))


@pytest.fixture()
def fresh(tmp_path: Path) -> Harness:
    """A repository that never had a contract: no baseline, no history of one."""

    return Harness(Sandbox(tmp_path / "fresh", with_baseline=False))


# ===================================================================== bootstrap


def test_the_first_run_of_a_repo_without_a_contract_reaches_a_green_measurement(
    fresh: Harness,
) -> None:
    """D-AUD-01, forward direction: the real first-time path must be reachable."""

    fresh.install_workload()
    assert fresh.module.run_gate() == 0
    run_state = fresh.read_state("run.json")
    assert run_state["status"] == "success"
    assert run_state["baseline_state"] == "never_created"
    success = fresh.read_state("success.json")
    assert success["baseline"]["present"] is False
    assert success["baseline_state"] == "never_created"


def test_the_first_baseline_is_built_by_the_real_cli_from_that_run(
    fresh: Harness,
) -> None:
    """No hand-written success record: the real run above is the only source."""

    fresh.install_workload()
    assert fresh.module.run_gate() == 0
    fresh.promote(
        "--bootstrap-baseline", "--coverage-floor", "backend=80/70,frontend=90/85"
    )
    created = json.loads(fresh.sandbox.baseline_path.read_text(encoding="utf-8"))
    assert created["coverage"] == {
        "backend": {"line_percent": 80.0, "branch_percent": 70.0},
        "frontend": {"line_percent": 90.0, "branch_percent": 85.0},
    }
    assert created["guarded_registry"] == 1
    assert set(GATE.SPLIT_MODULE_REGISTRY) <= set(created["guarded_files"])
    record = fresh.read_state("baseline-update.json")
    assert record["mode"] == "bootstrap"
    assert record["baseline"]["sha256_before"] is None


def test_a_repo_with_a_versioned_contract_cannot_bootstrap(harness: Harness) -> None:
    harness.write_promotable_state()
    harness.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=80/70,frontend=90/85",
        expect="already exists",
    )


def test_a_deleted_contract_cannot_be_rebuilt_as_a_first_time_baseline(
    harness: Harness,
) -> None:
    """D-AUD-01, the original counterexample: deleting the contract is not a reset."""

    harness.green_run()
    previous = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    harness.sandbox.baseline_path.unlink()
    harness.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=1/1,frontend=1/1",
        expect="prior",
    )
    assert not harness.sandbox.baseline_path.exists()
    assert previous["coverage"]["backend"]["line_percent"] == 80.0


def test_bootstrap_rejects_a_stale_success_record_from_before_the_deletion(
    harness: Harness,
) -> None:
    harness.green_run()
    harness.sandbox.baseline_path.unlink()
    harness.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=80/70,frontend=90/85",
        expect="baseline_state",
    )


def test_a_second_bootstrap_is_refused(fresh: Harness) -> None:
    """Repeated initialization must not re-establish or widen anything."""

    fresh.install_workload()
    assert fresh.module.run_gate() == 0
    fresh.promote(
        "--bootstrap-baseline", "--coverage-floor", "backend=80/70,frontend=90/85"
    )
    fresh.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=1/1,frontend=1/1",
        expect="already exists",
    )


def test_a_hand_written_never_created_record_is_not_enough(harness: Harness) -> None:
    """The claim must be backed by the work tree: a committed contract wins."""

    harness.sandbox.baseline_path.unlink()
    harness.write("run.json", harness.run_record(baseline_state="never_created"))
    harness.write(
        "success.json",
        harness.success_record(
            baseline={
                "path": "project/quality-baselines.v1.json",
                "present": False,
                "sha256": None,
                "baseline_version": None,
            },
            baseline_state="never_created",
        ),
    )
    harness.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=80/70,frontend=90/85",
        expect="prior",
    )


@pytest.mark.parametrize(
    ("specification", "expect"),
    [
        (None, "requires explicit initial floors"),
        ("backend=80/70", "no initial floor declared for frontend"),
        ("backend=80/70,frontend=90", "malformed coverage floor"),
        ("backend=80/nan,frontend=90/85", "finite"),
        ("backend=inf/70,frontend=90/85", "finite"),
        ("backend=-1/70,frontend=90/85", "0"),
        ("backend=101/70,frontend=90/85", "100"),
        ("side=80/70,frontend=90/85", "unknown coverage floor side"),
        ("backend=80/70,backend=81/71,frontend=90/85", "declared twice"),
        ("backend=true/70,frontend=90/85", "malformed coverage floor"),
    ],
)
def test_the_declared_first_floors_are_validated(
    fresh: Harness, specification: str | None, expect: str
) -> None:
    fresh.install_workload()
    assert fresh.module.run_gate() == 0
    arguments = ["--bootstrap-baseline"]
    if specification is not None:
        arguments += ["--coverage-floor", specification]
    fresh.refuse(*arguments, expect=expect)


# ================================================================ damaged records


DAMAGES = (
    "missing_field",
    "unknown_field",
    "duplicate_key",
    "empty_run_id",
    "bad_time",
    "wrong_version",
    "wrong_hash",
    "path_escape",
    "duplicate_manifest_path",
)
PER_KIND_DAMAGES = {
    "run": tuple(d for d in DAMAGES if d != "duplicate_manifest_path"),
    "success": DAMAGES,
    "failure": DAMAGES,
    "update": DAMAGES,
}
FIRST_FIELD = {
    "run": "baseline_state",
    "success": "baseline",
    "failure": "section",
    "update": "measurement_inputs",
}


def _damaged(record: dict, damage: str, kind: str) -> dict:
    """Apply one closed-form violation to a record-shaped payload."""

    record = json.loads(json.dumps(record))
    if damage == "missing_field":
        record.pop(FIRST_FIELD[kind])
    elif damage == "unknown_field":
        record["surprise"] = True
    elif damage == "empty_run_id":
        record["run_id"] = ""
    elif damage == "bad_time":
        record["started_at"] = "yesterday-ish"
    elif damage == "wrong_version":
        record["record_version"] = 0
    elif damage == "wrong_hash":
        if kind == "success":
            record["baseline"]["sha256"] = "z" * 64
        elif kind == "update":
            record["measurement_inputs"]["fingerprint"] = "z" * 64
        elif kind == "run":
            record["inputs"]["groups"]["measurement"]["fingerprint"] = "not-a-hash"
        else:
            record["inputs"]["groups"]["governance"]["fingerprint"] = "not-a-hash"
    elif damage == "path_escape":
        if kind == "success":
            record["inputs"]["files"][0]["path"] = "../outside.py"
        elif kind == "run":
            record["inputs"]["groups"]["measurement"]["fingerprint"] = ".."
        elif kind == "failure":
            record["section"] = "../../etc/passwd"
        else:
            record["baseline"]["path"] = "../quality-baselines.v1.json"
    elif damage == "duplicate_manifest_path":
        if kind == "success":
            files = record["inputs"]["files"]
            files.append(dict(files[0]))
        elif kind == "update":
            record["measurement_inputs"]["file_count"] = -1
        elif kind == "failure":
            record["sections"] = [
                {"name": "gov", "status": "ok", "seconds": 0.1},
                {"name": "gov", "status": "ok", "seconds": 0.1},
            ]
        else:
            record["sections"] = [
                {"name": "gov", "status": "ok", "seconds": 0.1},
                {"name": "gov", "status": "ok", "seconds": 0.1},
            ]
    elif damage == "duplicate_key":
        raise AssertionError("duplicate keys are produced as raw text")
    else:  # pragma: no cover - guards the parametrisation itself
        raise AssertionError(damage)
    return record


def _text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


NUMERIC_TOKEN = {
    "run": '"seconds": 0.1',
    "success": '"seconds": 0.1',
    "failure": '"seconds": 0.1',
    "update": '"file_count": 21',
}


def _raw_damage(text: str, damage: str, kind: str) -> str:
    if damage == "duplicate_key":
        lines = text.splitlines()
        return "\n".join([lines[0], '  "run_id": "duplicate",', *lines[1:]]) + "\n"
    if damage == "non_finite":
        token = NUMERIC_TOKEN[kind]
        assert token in text, text
        return text.replace(token, token.split(":")[0] + ": NaN", 1)
    raise AssertionError(damage)


PARSERS = {
    "run": "parse_run_record",
    "success": "parse_success_record",
    "failure": "parse_failure_record",
    "update": "parse_update_record",
}


def _record_payloads(harness: Harness) -> dict[str, dict]:
    run = harness.run_record()
    failure = {
        "record_version": harness.module.FAILURE_RECORD_VERSION,
        "gate_version": harness.module.GATE_VERSION,
        "status": "failed",
        "run_id": run["run_id"],
        "generation": run["generation"],
        "started_at": "2026-09-15T12:00:00+00:00",
        "failed_at": "2026-09-15T12:05:00+00:00",
        "section": "backend",
        "message": "injected",
        "sections": [
            {"name": "gov", "status": "ok", "seconds": 0.1},
            {"name": "backend", "status": "failed", "seconds": 0.1},
        ],
        "measurement": harness.measurement(),
        "inputs": harness.module.manifest_summary(harness.manifest()),
        "owner": harness.owner(),
    }
    update = {
        "record_version": harness.module.UPDATE_RECORD_VERSION,
        "gate_version": harness.module.GATE_VERSION,
        "mode": "update",
        "run_id": run["run_id"],
        "generation": run["generation"],
        "at": "2026-09-15T12:11:00+00:00",
        "baseline": {
            "path": "project/quality-baselines.v1.json",
            "sha256_before": "1" * 64,
            "sha256_after": "2" * 64,
            "baseline_version_before": 1,
            "baseline_version_after": 1,
        },
        "governance_check": {
            "checked_at": "2026-09-15T12:11:00+00:00",
            "fingerprint": "3" * 64,
            "file_count": 7,
            "result": "ok",
        },
        "measurement_inputs": {
            "file_count": 21,
            "fingerprint": harness.manifest()["groups"]["measurement"]["fingerprint"],
        },
    }
    return {
        "run": run,
        "success": harness.success_record(),
        "failure": failure,
        "update": update,
    }


@pytest.mark.parametrize(
    ("kind", "damage"),
    [(kind, damage) for kind in PARSERS for damage in PER_KIND_DAMAGES[kind]],
)
def test_every_damaged_record_is_refused_by_the_closed_form_parser(
    harness: Harness, kind: str, damage: str
) -> None:
    """No damaged lifecycle record may pass, and no internal error may escape."""

    payload = _record_payloads(harness)[kind]
    if damage == "duplicate_key":
        text = _raw_damage(_text(payload), damage, kind)
        with pytest.raises(harness.module.UpdateRefused):
            harness.module.load_record_text(text, f"{kind} record")
        return
    parser = getattr(harness.module, PARSERS[kind])
    with pytest.raises(harness.module.UpdateRefused):
        parser(_damaged(payload, damage, kind))


@pytest.mark.parametrize("kind", ("run", "success", "failure", "update"))
def test_raw_text_damage_is_refused_before_anything_is_parsed(
    harness: Harness, kind: str
) -> None:
    """Non-standard numbers never reach the record layer."""

    text = _raw_damage(_text(_record_payloads(harness)[kind]), "non_finite", kind)
    with pytest.raises(harness.module.UpdateRefused):
        harness.module.load_record_text(text, f"{kind} record")


@pytest.mark.parametrize("kind", ("run", "success", "failure", "update"))
def test_a_damaged_record_is_never_written(harness: Harness, kind: str) -> None:
    """The writers validate first: a producer bug becomes a refusal, not a bad file."""

    payload = _damaged(_record_payloads(harness)[kind], "unknown_field", kind)
    path = harness.state(f"{kind}.json")
    with pytest.raises(harness.module.UpdateRefused):
        harness.module.write_record(path, payload, kind)
    assert not path.exists()


@pytest.mark.parametrize(
    ("kind", "damage"),
    [
        ("run", "missing_field"),
        ("run", "duplicate_key"),
        ("run", "non_finite"),
        ("run", "empty_run_id"),
        ("run", "bad_time"),
        ("run", "wrong_version"),
        ("run", "wrong_hash"),
        ("run", "path_escape"),
        ("success", "missing_field"),
        ("success", "duplicate_key"),
        ("success", "non_finite"),
        ("success", "wrong_hash"),
        ("success", "duplicate_manifest_path"),
        ("success", "unknown_field"),
        ("success", "bad_time"),
    ],
)
def test_the_update_path_refuses_a_damaged_state_without_touching_the_baseline(
    harness: Harness, kind: str, damage: str
) -> None:
    """D-AUD-02: a refusal may never leave a rewritten baseline behind."""

    harness.write_promotable_state()
    payload = _record_payloads(harness)[kind]
    if damage in ("duplicate_key", "non_finite"):
        harness.write_text(f"{kind}.json", _raw_damage(_text(payload), damage, kind))
    else:
        harness.write(f"{kind}.json", _damaged(payload, damage, kind))
    harness.refuse_any("--update-baseline")


def test_a_record_that_is_not_an_object_is_refused(harness: Harness) -> None:
    harness.write_promotable_state()
    harness.write_text("run.json", "[]\n")
    harness.refuse("--update-baseline", expect="not a JSON object")


# =========================================================== concurrency and locks


def test_one_work_tree_lock_serves_run_update_and_bootstrap(harness: Harness) -> None:
    """The lock is visible to independent processes and shared by all writers."""

    assert harness.lock_visibility() is True
    with harness.module.gate_lock("test-holder"):
        assert harness.lock_visibility() is False
        harness.write_promotable_state()
        harness.refuse("--update-baseline", expect="lock")
        harness.refuse(
            "--bootstrap-baseline",
            "--coverage-floor",
            "backend=80/70,frontend=90/85",
            expect="lock",
        )


def test_a_busy_refusal_does_not_record_a_new_failure(harness: Harness) -> None:
    """A refused concurrent attempt is not a new round and writes nothing."""

    harness.write_promotable_state()
    before = harness.state("run.json").read_bytes()
    with harness.module.gate_lock("test-holder"):
        assert harness.sandbox.run_cli("--update-baseline").returncode != 0
    assert harness.state("run.json").read_bytes() == before
    assert not harness.state("failure.json").exists()


def test_an_older_run_cannot_publish_over_a_newer_failed_run(harness: Harness) -> None:
    """D-AUD-03: run identity, not wall clock, decides who may publish."""

    entered = harness.scratch / "older-entered"
    release = harness.scratch / "older-release"
    older = harness.child("run", entered=entered, release=release)
    harness.wait(entered)
    older_id = harness.read_state("run.json")["run_id"]

    harness.install_workload(failure="backend")
    assert harness.module.run_gate() == 1
    newer = harness.read_state("run.json")
    assert newer["status"] == "failed" and newer["run_id"] != older_id

    release.touch()
    code, stdout, stderr = harness.finish(older)
    assert code != 0
    assert "superseded" in (stdout + stderr)
    final = harness.read_state("run.json")
    assert final["run_id"] == newer["run_id"] and final["status"] == "failed"
    success = harness.state("success.json")
    assert not success.exists() or (
        json.loads(success.read_text(encoding="utf-8"))["run_id"] != older_id
    )
    harness.refuse("--update-baseline", expect="not 'success'")


def test_an_update_is_refused_while_a_run_is_still_running(harness: Harness) -> None:
    harness.write_promotable_state()
    entered = harness.scratch / "holder-entered"
    release = harness.scratch / "holder-release"
    holder = harness.child("run", entered=entered, release=release)
    harness.wait(entered)
    harness.refuse("--update-baseline", expect="'running', not 'success'")
    release.touch()
    harness.finish(holder)


def test_a_run_cannot_start_while_the_work_tree_lock_is_held(harness: Harness) -> None:
    """The run's own admission is behind the same lock."""

    assert harness.lock_visibility() is True
    with harness.module.gate_lock("test-holder"):
        completed = harness.sandbox.run_cli()
        combined = completed.stdout + completed.stderr
        assert completed.returncode != 0, combined
        assert "lock" in combined, combined
        assert not harness.state("run.json").exists()
        assert not harness.state("failure.json").exists()


def test_a_superseded_run_never_publishes_over_a_newer_green_run(
    harness: Harness,
) -> None:
    """Two runs may overlap; only the latest identity may publish (D-AUD-03)."""

    entered = harness.scratch / "older-entered"
    release = harness.scratch / "older-release"
    older = harness.child("run", entered=entered, release=release)
    harness.wait(entered)
    older_id = harness.read_state("run.json")["run_id"]

    harness.install_workload()
    assert harness.module.run_gate() == 0
    newer_id = harness.read_state("run.json")["run_id"]
    assert newer_id != older_id
    assert (
        json.loads(harness.state("success.json").read_text("utf-8"))["run_id"]
        == newer_id
    )

    release.touch()
    code, stdout, stderr = harness.finish(older)
    assert code != 0
    assert "superseded" in (stdout + stderr)
    assert harness.read_state("run.json")["run_id"] == newer_id
    assert (
        json.loads(harness.state("success.json").read_text("utf-8"))["run_id"]
        == newer_id
    )


def test_sigterm_during_the_update_barrier_does_not_lock_the_work_tree(
    harness: Harness,
) -> None:
    """The lock is released by the operating system, not by a cleanup step."""

    harness.write_promotable_state()
    entered = harness.scratch / "gov-entered"
    release = harness.scratch / "gov-release"
    harness.park_governance(entered, release)
    update = harness.cli_popen("--update-baseline")
    harness.wait(entered)
    assert harness.lock_visibility() is False, "the update holds the lock here"
    update.send_signal(signal.SIGTERM)
    code, _, _ = harness.finish(update)
    assert code != 0
    assert harness.lock_visibility() is True
    release.touch()
    harness.install_workload()
    assert harness.module.run_gate() == 0


def test_a_hard_kill_during_the_update_barrier_does_not_lock_the_work_tree(
    harness: Harness,
) -> None:
    harness.write_promotable_state()
    entered = harness.scratch / "kill-entered"
    release = harness.scratch / "kill-release"
    harness.park_governance(entered, release)
    update = harness.cli_popen("--update-baseline")
    harness.wait(entered)
    os.kill(update.pid, signal.SIGKILL)
    harness.finish(update)
    assert harness.lock_visibility() is True
    release.touch()


def test_a_child_process_cannot_forge_the_parent_lock(harness: Harness) -> None:
    """An inherited descriptor is not ownership: a fresh open is still refused."""

    with harness.module.gate_lock("test-holder"):
        completed = subprocess.run(
            [sys.executable, str(harness.lock_probe), str(harness.module.LOCK_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert json.loads(completed.stdout)["acquired"] is False


def test_an_unexpected_exit_releases_the_work_tree_lock(harness: Harness) -> None:
    """An exception inside the critical section must not leave a stale owner."""

    crashed = harness.child("lock-crash")
    code, _, stderr = harness.finish(crashed)
    assert code != 0
    assert "injected crash" in stderr
    assert harness.lock_visibility() is True
    harness.write_promotable_state()
    harness.promote("--update-baseline")


def test_update_and_bootstrap_interleave_on_the_same_lock(harness: Harness) -> None:
    """A bootstrap never slips in while an update holds the work tree."""

    harness.write_promotable_state()
    entered = harness.scratch / "gov-entered"
    release = harness.scratch / "gov-release"
    harness.park_governance(entered, release)
    update = harness.cli_popen("--update-baseline")
    harness.wait(entered)
    harness.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=80/70,frontend=90/85",
        expect="lock",
    )
    release.touch()
    code, stdout, stderr = harness.finish(update)
    assert code == 0, stdout + stderr


def test_the_lock_file_is_not_deleted_when_the_lock_is_released(
    harness: Harness,
) -> None:
    harness.write_promotable_state()
    harness.promote("--update-baseline")
    lock = harness.module.LOCK_PATH
    assert lock.is_file()
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["owner"]["token"]
    assert isinstance(payload["acquisitions"], int)


def test_a_record_pair_from_different_owners_is_refused(harness: Harness) -> None:
    run = harness.run_record()
    success = harness.success_record(
        run_id=run["run_id"],
        generation=run["generation"],
        owner=harness.owner(token="ffffffffffffffff"),
    )
    harness.write_promotable_state(run=run, success=success)
    harness.refuse("--update-baseline", expect="owner")


def test_a_stale_generation_in_the_success_record_is_refused(harness: Harness) -> None:
    run = harness.run_record(generation=7)
    success = harness.success_record(run_id=run["run_id"], generation=3)
    harness.write_promotable_state(run=run, success=success)
    harness.refuse("--update-baseline", expect="generation")


def test_a_newer_failed_run_invalidates_an_older_success_record(
    harness: Harness,
) -> None:
    harness.write_promotable_state()
    harness.write(
        "run.json",
        harness.run_record(
            run_id="20260915T130000Z-ccccdddd",
            generation=4,
            status="failed",
            failed_section="backend",
            finished_at="2026-09-15T13:10:00+00:00",
            sections=[
                {"name": "gov", "status": "ok", "seconds": 0.1},
                {"name": "ruff", "status": "ok", "seconds": 0.1},
                {"name": "backend", "status": "failed", "seconds": 0.1},
            ],
        ),
    )
    harness.refuse("--update-baseline", expect="'failed', not 'success'")


# ======================================================== input content binding


@pytest.mark.parametrize(
    "mutated",
    [
        "project/backend/src/procurement_review/run_projection.py",
        "project/backend/tests/test_sample.py",
        "project/backend/uv.lock",
        "project/frontend/package-lock.json",
        "project/scripts/quality_gate.py",
        "project/design-references/README.md",
    ],
)
def test_editing_an_input_during_the_governance_rest_is_refused(
    harness: Harness, mutated: str
) -> None:
    """D-AUD-04: the final review before the write sees the current content."""

    harness.write_promotable_state()
    entered = harness.scratch / "gov-entered"
    release = harness.scratch / "gov-release"
    harness.park_governance(entered, release)
    update = harness.cli_popen("--update-baseline")
    harness.wait(entered)
    path = harness.sandbox.root / mutated
    path.write_text(path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    release.touch()
    code, stdout, stderr = harness.finish(update)
    combined = stdout + stderr
    assert code != 0, combined
    assert "baseline unchanged" in combined, combined
    assert "changed" in combined, combined
    unchanged = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    assert unchanged["coverage"]["backend"]["line_percent"] == 80.0


def test_a_baseline_replaced_during_the_governance_rest_is_refused(
    harness: Harness,
) -> None:
    harness.write_promotable_state()
    entered = harness.scratch / "gov-entered"
    release = harness.scratch / "gov-release"
    harness.park_governance(entered, release)
    update = harness.cli_popen("--update-baseline")
    harness.wait(entered)
    harness.sandbox.write_baseline(updated_at="2026-09-15T13:00:00+00:00")
    replaced = harness.sandbox.snapshot()
    release.touch()
    code, stdout, stderr = harness.finish(update)
    assert code != 0
    assert "baseline unchanged" in stderr
    assert harness.sandbox.snapshot() == replaced


def test_a_newer_run_makes_the_reviewed_measurement_stale(harness: Harness) -> None:
    """A run that arrives after the success record invalidates it."""

    harness.write_promotable_state()
    entered = harness.scratch / "gov-entered"
    release = harness.scratch / "gov-release"
    harness.park_governance(entered, release)
    update = harness.cli_popen("--update-baseline")
    harness.wait(entered)
    harness.write(
        "run.json",
        harness.run_record(
            run_id="20260915T140000Z-eeeeffff",
            generation=9,
            status="running",
            finished_at=None,
            sections=[],
        ),
    )
    release.touch()
    code, stdout, stderr = harness.finish(update)
    assert code != 0
    assert "baseline unchanged" in stderr
    assert "latest run" in stderr or "generation" in stderr


def test_editing_an_input_after_the_success_record_is_refused(harness: Harness) -> None:
    harness.write_promotable_state()
    note = harness.sandbox.root / "project/backend/tests/test_sample.py"
    note.write_text("def test_new_failure():\n    assert False\n", encoding="utf-8")
    harness.refuse("--update-baseline", expect="changed")


# ============================================================= publication risk


def test_a_successful_update_records_its_intent_before_the_commit_point(
    harness: Harness,
) -> None:
    harness.write_promotable_state()
    before = hashlib.sha256(harness.sandbox.baseline_bytes()).hexdigest()
    harness.promote("--update-baseline")
    after = hashlib.sha256(harness.sandbox.baseline_bytes()).hexdigest()
    intent = harness.read_state("baseline-intent.json")
    assert intent["old"]["sha256"] == before
    assert intent["new"]["sha256"] == after
    assert intent["update_record"]["mode"] == "update"
    record = harness.read_state("baseline-update.json")
    assert record["baseline"]["sha256_before"] == before
    assert record["baseline"]["sha256_after"] == after


def test_a_refusal_before_the_commit_point_leaves_no_intent_and_no_record(
    harness: Harness,
) -> None:
    guarded = harness.sandbox.root / GUARDED
    guarded.write_text(
        SUPPORT.GUARDED_MODULE.format(body=SUPPORT._body(960)), encoding="utf-8"
    )
    harness.write_promotable_state()
    harness.refuse("--update-baseline", expect="refusing to write a weaker")
    assert not harness.state("baseline-intent.json").exists()
    assert not harness.state("baseline-update.json").exists()


def test_a_failure_after_the_commit_point_reports_a_committed_baseline(
    harness: Harness,
) -> None:
    """A committed baseline is never reported as "unchanged" (D-AUD-02)."""

    harness.write_promotable_state()
    harness.state("baseline-update.json").mkdir()  # the auxiliary write must fail
    completed = harness.sandbox.run_cli("--update-baseline")
    combined = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "already committed" in combined, combined
    assert "baseline unchanged" not in combined, combined
    rewritten = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    assert rewritten["coverage"]["backend"]["line_percent"] == 87.5
    intent = harness.read_state("baseline-intent.json")
    assert (
        intent["new"]["sha256"]
        == hashlib.sha256(harness.sandbox.baseline_bytes()).hexdigest()
    )
    harness.state("baseline-update.json").rmdir()


def test_recovery_accepts_the_expected_new_hash_and_finishes_the_records(
    harness: Harness,
) -> None:
    harness.write_promotable_state()
    harness.state("baseline-update.json").mkdir()
    assert harness.sandbox.run_cli("--update-baseline").returncode != 0
    harness.state("baseline-update.json").rmdir()
    harness.promote("--recover-baseline")
    record = harness.read_state("baseline-update.json")
    assert record["mode"] == "update"
    assert (
        record["baseline"]["sha256_after"]
        == hashlib.sha256(harness.sandbox.baseline_bytes()).hexdigest()
    )


def test_recovery_accepts_the_expected_old_hash_and_reports_no_commit(
    harness: Harness,
) -> None:
    """An intent written before a crash that never reached the commit point."""

    harness.write_promotable_state()
    previous = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    candidate = harness.module.candidate_baseline(harness.measurement(), previous)
    intent = harness.module.build_intent(
        mode="update",
        run_id="20260915T120000Z-aaaabbbb",
        generation=1,
        previous=previous,
        candidate=candidate,
        manifest=harness.manifest(),
        baseline_sha_before=hashlib.sha256(
            harness.sandbox.baseline_bytes()
        ).hexdigest(),
    )
    harness.write("baseline-intent.json", intent)
    before = harness.sandbox.snapshot()
    completed = harness.sandbox.run_cli("--recover-baseline")
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined
    assert "no commit" in combined, combined
    assert harness.sandbox.snapshot() == before


def test_a_kill_before_the_commit_point_leaves_no_intent_and_no_change(
    harness: Harness,
) -> None:
    """The reviewed promotion is abandoned in full: nothing is half-written."""

    harness.write_promotable_state()
    before = harness.sandbox.snapshot()
    entered = harness.scratch / "gov-entered"
    release = harness.scratch / "gov-release"
    harness.park_governance(entered, release)
    update = harness.cli_popen("--update-baseline")
    harness.wait(entered)
    os.kill(update.pid, signal.SIGKILL)
    harness.finish(update)
    release.touch()
    assert harness.sandbox.snapshot() == before
    assert not harness.state("baseline-intent.json").exists()
    assert not harness.state("baseline-update.json").exists()


def test_recovery_stops_on_a_third_hash(harness: Harness) -> None:
    harness.write_promotable_state()
    harness.state("baseline-update.json").mkdir()
    assert harness.sandbox.run_cli("--update-baseline").returncode != 0
    harness.state("baseline-update.json").rmdir()
    harness.sandbox.write_baseline(updated_at="2026-09-15T15:00:00+00:00")
    replaced = harness.sandbox.snapshot()
    harness.refuse("--recover-baseline", expect="manual")
    assert harness.sandbox.snapshot() == replaced


def test_recovery_refuses_a_damaged_intent(harness: Harness) -> None:
    harness.write_promotable_state()
    harness.promote("--update-baseline")
    harness.write_text("baseline-intent.json", '{"record_version": 1,\n')
    harness.refuse("--recover-baseline", expect="intent")


# ============================================================ guarded migrations


def _declaration(**fields: object) -> dict:
    declaration: dict = {
        "kind": "split",
        "targets": [PART_A, PART_B],
        "responsibility": "拆分后的两个职责组",
        "source_ceiling": 832,
        "target_ceilings": {PART_A: 420, PART_B: 412},
        "reason": "FIX-05A 反例：显式迁移声明",
        "declared_at": "2026-09-15T00:00:00+00:00",
    }
    declaration.update(fields)
    return {GUARDED: declaration}


def _split_source(harness: Harness, *, sizes: tuple[int, int] = (420, 412)) -> None:
    _exact_lines(harness.sandbox.root / PART_A, sizes[0])
    _exact_lines(harness.sandbox.root / PART_B, sizes[1])
    (harness.sandbox.root / GUARDED).unlink()


def _guarded_baseline(harness: Harness, **overrides: object) -> None:
    fields: dict = {"guarded_files": {GUARDED: 832}}
    fields.update(overrides)
    harness.sandbox.write_baseline(**fields)


def test_a_deleted_guarded_path_without_a_declaration_stops_the_gate(
    harness: Harness,
    capsys: pytest.CaptureFixture,
) -> None:
    """D-AUD-05: dropping a guarded registration is never silent."""

    _guarded_baseline(harness)
    _split_source(harness)
    harness.expect_run_failure("migration", capsys)
    assert json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))[
        "guarded_files"
    ] == {GUARDED: 832}


def test_a_renamed_guarded_path_without_a_declaration_stops_the_gate(
    harness: Harness,
    capsys: pytest.CaptureFixture,
) -> None:
    _guarded_baseline(harness)
    target = "project/backend/src/procurement_review/renamed_repository.py"
    (harness.sandbox.root / GUARDED).rename(harness.sandbox.root / target)
    harness.expect_run_failure("migration", capsys)


def test_a_declared_split_inherits_one_budget_for_all_targets(harness: Harness) -> None:
    _guarded_baseline(
        harness,
        guarded_migrations=_declaration(target_ceilings={PART_A: 420, PART_B: 412}),
    )
    _split_source(harness)
    harness.green_run()
    harness.promote("--update-baseline")
    updated = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    assert GUARDED not in updated["guarded_files"]
    assert updated["guarded_files"][PART_A] == 420
    assert updated["guarded_files"][PART_B] == 412
    assert sum(updated["guarded_files"][t] for t in (PART_A, PART_B)) <= 832
    assert updated["guarded_migrations"][GUARDED]["targets"] == [PART_A, PART_B]


def test_a_split_that_would_expand_the_budget_stops_the_gate(
    harness: Harness,
    capsys: pytest.CaptureFixture,
) -> None:
    """A split inherits the recorded budget; it never multiplies it."""

    _guarded_baseline(
        harness,
        guarded_migrations=_declaration(target_ceilings={PART_A: 700, PART_B: 700}),
    )
    _split_source(harness, sizes=(700, 700))
    harness.expect_run_failure("budget", capsys)


def test_a_declared_split_keeps_small_targets_registered_and_guarded(
    harness: Harness,
) -> None:
    """Targets below 800 lines stay registered (D-AUD-05)."""

    _guarded_baseline(
        harness,
        guarded_migrations=_declaration(target_ceilings={PART_A: 420, PART_B: 412}),
    )
    _split_source(harness)
    harness.green_run()
    harness.promote("--update-baseline")
    updated = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    assert updated["guarded_files"][PART_A] == 420
    assert updated["guarded_files"][PART_B] == 412
    assert harness.module.ratchet_problems(harness.measurement(), updated) == []


def test_the_split_budget_is_enforced_after_the_migration(harness: Harness) -> None:
    test_a_declared_split_keeps_small_targets_registered_and_guarded(harness)
    baseline = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    for relative in (PART_A, PART_B):
        _exact_lines(harness.sandbox.root / relative, 799)
    problems = harness.module.ratchet_problems(harness.measurement(), baseline)
    assert any("budget" in problem for problem in problems), problems
    assert any(PART_A in problem for problem in problems), problems


def test_a_replayed_declaration_does_not_consume_the_budget_twice(
    harness: Harness,
) -> None:
    _guarded_baseline(
        harness,
        guarded_migrations=_declaration(target_ceilings={PART_A: 420, PART_B: 412}),
    )
    _split_source(harness)
    harness.green_run()
    harness.promote("--update-baseline")
    first = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    harness.green_run()
    harness.promote("--update-baseline")
    second = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    for target in (PART_A, PART_B):
        assert second["guarded_files"][target] <= first["guarded_files"][target]
    assert sum(second["guarded_files"][t] for t in (PART_A, PART_B)) <= 832
    assert second["guarded_migrations"] == first["guarded_migrations"]


def test_a_retirement_without_a_reason_stops_the_gate(
    harness: Harness,
    capsys: pytest.CaptureFixture,
) -> None:
    _guarded_baseline(
        harness,
        guarded_migrations=_declaration(
            kind="retire", targets=[], target_ceilings={}, reason=""
        ),
    )
    (harness.sandbox.root / GUARDED).unlink()
    harness.expect_run_failure("reason", capsys)


def test_a_declared_retirement_drops_the_registration(harness: Harness) -> None:
    _guarded_baseline(
        harness,
        guarded_migrations=_declaration(
            kind="retire",
            targets=[],
            target_ceilings={},
            reason="文件职责已完全并入其它模块",
        ),
    )
    (harness.sandbox.root / GUARDED).unlink()
    harness.green_run()
    harness.promote("--update-baseline")
    updated = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    assert GUARDED not in updated["guarded_files"]
    assert updated["guarded_migrations"][GUARDED]["kind"] == "retire"


@pytest.mark.parametrize(
    ("fields", "expect"),
    [
        (
            {
                "targets": ["project/docs/delivery.md"],
                "target_ceilings": {"project/docs/delivery.md": 10},
            },
            "managed",
        ),
        (
            {
                "targets": [
                    PART_A,
                    "project/backend/src/procurement_review/missing.py",
                ],
                "target_ceilings": {
                    PART_A: 400,
                    "project/backend/src/procurement_review/missing.py": 400,
                },
            },
            "exist",
        ),
        ({"kind": "rebase"}, "kind"),
        ({"responsibility": ""}, "responsibility"),
        ({"source_ceiling": 4000}, "ceiling"),
        ({"target_ceilings": {PART_A: 400}}, "target_ceilings"),
    ],
)
def test_a_malformed_migration_declaration_stops_the_gate(
    harness: Harness,
    capsys: pytest.CaptureFixture,
    fields: dict,
    expect: str,
) -> None:
    _guarded_baseline(harness, guarded_migrations=_declaration(**fields))
    _exact_lines(harness.sandbox.root / PART_A, 400)
    _exact_lines(harness.sandbox.root / PART_B, 400)
    (harness.sandbox.root / GUARDED).unlink()
    harness.expect_run_failure(expect, capsys)


def test_a_declared_merge_shares_one_budget_and_may_retire_a_sibling(
    harness: Harness,
) -> None:
    """A merge inherits the source budget; a sibling retirement stays declared."""

    legacy = "project/backend/src/procurement_review/legacy_module.py"
    merged = "project/backend/src/procurement_review/merged_module.py"
    _exact_lines(harness.sandbox.root / legacy, 100)
    harness.sandbox.write_baseline(
        guarded_files={GUARDED: 832, legacy: 100},
        guarded_migrations={
            GUARDED: {
                "kind": "merge",
                "targets": [merged],
                "responsibility": "两条守卫路径合并为一个模块",
                "source_ceiling": 832,
                "target_ceilings": {merged: 830},
                "reason": "FIX-05A 反例：合并声明",
            },
            legacy: {
                "kind": "retire",
                "targets": [],
                "responsibility": "旧模块职责并入合并模块",
                "source_ceiling": 100,
                "target_ceilings": {},
                "reason": "职责已完全合并，不再单独守卫",
            },
        },
    )
    _exact_lines(harness.sandbox.root / merged, 830)
    (harness.sandbox.root / GUARDED).unlink()
    (harness.sandbox.root / legacy).unlink()
    harness.green_run()
    harness.promote("--update-baseline")
    updated = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    assert GUARDED not in updated["guarded_files"]
    assert legacy not in updated["guarded_files"]
    assert updated["guarded_files"][merged] == 830
    _exact_lines(harness.sandbox.root / merged, 900)
    problems = harness.module.ratchet_problems(harness.measurement(), updated)
    assert any("budget" in problem for problem in problems), problems


def test_the_fixed_module_registry_is_not_a_substitute_for_a_declaration(
    harness: Harness,
    capsys: pytest.CaptureFixture,
) -> None:
    """The eleven registered modules do not generalise to other guarded files."""

    assert GUARDED not in GATE.SPLIT_MODULE_REGISTRY
    _guarded_baseline(harness)
    _split_source(harness, sizes=(700, 700))
    harness.expect_run_failure("migration", capsys)


def test_a_new_oversized_file_still_needs_a_registration(harness: Harness) -> None:
    harness.write_promotable_state()
    relative = "project/backend/src/procurement_review/fresh_big_module.py"
    (harness.sandbox.root / relative).write_text(
        SUPPORT.GUARDED_MODULE.format(body=SUPPORT._body(900)), encoding="utf-8"
    )
    problems = harness.module.ratchet_problems(
        harness.measurement(),
        json.loads(harness.sandbox.baseline_bytes().decode("utf-8")),
    )
    assert any("without a registered exemption" in problem for problem in problems)
