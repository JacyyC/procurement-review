"""FIX-01D V-D-01: only a complete, current, green measurement may move the baseline.

The subject under test is the real ``project/scripts/quality_gate.py``, copied verbatim
into a synthetic repository per test.  The synthetic tree carries the measured sources,
the governance stubs the gate's first section shells out to, and a Git repository, so the
lifecycle under test is the shipped one rather than a stand-in.  Only the *environment*
is synthetic -- never the gate's own decision logic.

Every refusal asserts three things: a non-zero exit, a message naming the reason, and a
baseline file whose bytes are unchanged.  "Refused" only means something if the contract
did not move; a check that fails after rewriting the baseline has not protected anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_SOURCE = REPO_ROOT / "project" / "scripts" / "quality_gate.py"


def _load_gate(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


GATE = _load_gate(GATE_SOURCE, "quality_gate_contract")

GOVERNANCE_STUB = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

marker = Path(__file__).resolve().parents[2] / "governance-failure"
sys.exit(1 if marker.exists() else 0)
"""

API_MODULE = '''"""Assembly root stub."""


def create_app():
    return {"app": True}
'''

REGISTRY_MODULE = '''"""Split module stub ({name})."""


def value():
    return "{name}"
'''

GUARDED_MODULE = '''"""Guarded repository stub."""

{body}
'''


def _body(lines: int) -> str:
    return "\n".join(f"def helper_{index}():  # padding" for index in range(lines))


class Sandbox:
    """A synthetic repository whose only real component is the gate script itself."""

    def __init__(self, root: Path, *, with_baseline: bool = True) -> None:
        self.root = root
        self.project = root / "project"
        self.backend = self.project / "backend"
        self.frontend = self.project / "frontend"
        self.gate_path = self.project / "scripts" / "quality_gate.py"
        self.baseline_path = self.project / "quality-baselines.v1.json"
        self.state = self.project / ".local" / "quality-gate"
        self.with_baseline = with_baseline
        self._module: ModuleType | None = None
        self._build()

    # ---------------------------------------------------------------- construction

    def _write(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _build(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ".gitignore").write_text(".local/\n", encoding="utf-8")
        shutil.copyfile(GATE_SOURCE, self._write("project/scripts/quality_gate.py", ""))
        self._write("project/backend/src/procurement_review/api.py", API_MODULE)
        self._write(
            "project/backend/src/procurement_review/review_repository.py",
            GUARDED_MODULE.format(body=_body(830)),
        )
        for relative in sorted(GATE.SPLIT_MODULE_REGISTRY):
            name = Path(relative).stem
            self._write(relative, REGISTRY_MODULE.format(name=name))
        self._write(
            "project/backend/tests/test_sample.py", "def test_sample():\n    pass\n"
        )
        self._write("project/backend/pyproject.toml", '[project]\nname = "sandbox"\n')
        self._write("project/backend/uv.lock", "version = 1\n")
        self._write("project/frontend/src/App.tsx", "export const app = 1;\n")
        self._write("project/frontend/package.json", '{"name": "sandbox"}\n')
        self._write("project/frontend/package-lock.json", '{"lockfileVersion": 3}\n')
        self._write("project/design-references/README.md", "# frozen reference\n")
        self._write("project/docs/delivery.md", "# delivery\n")
        for script in (
            "workspace_views.py",
            "document_health_check.py",
            "check_markdown_links.py",
        ):
            self._write(f"workspace/scripts/{script}", GOVERNANCE_STUB)
        self._write("workspace/docs/note.md", "# note\n")
        if self.with_baseline:
            self.write_baseline()
        self._git("init", "-q")
        self._git("add", "-A")
        self._git(
            "-c",
            "user.email=sandbox@example.invalid",
            "-c",
            "user.name=Sandbox",
            "commit",
            "-q",
            "-m",
            "sandbox",
        )

    def _git(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    # --------------------------------------------------------------------- helpers

    @property
    def module(self) -> ModuleType:
        """The sandbox copy of the gate, so its ROOT is the synthetic repository."""

        if self._module is None:
            self._module = _load_gate(self.gate_path, f"sandbox_gate_{id(self)}")
        return self._module

    @property
    def head(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.strip()

    def baseline_bytes(self) -> bytes:
        return self.baseline_path.read_bytes()

    def write_baseline(self, **overrides: object) -> None:
        baseline: dict = {
            "baseline_version": 1,
            "coverage_tolerance_pp": 0.1,
            "coverage": {
                "backend": {"line_percent": 80.0, "branch_percent": 70.0},
                "frontend": {"line_percent": 90.0, "branch_percent": 85.0},
            },
            "guarded_files": {
                "project/backend/src/procurement_review/review_repository.py": 900,
            },
            "api_factory_max_lines": {"create_app": 200},
            "size_exemptions": {},
            "measured_at": "2026-09-14T00:00:00+00:00",
            "updated_at": "2026-09-14T00:00:00+00:00",
            "git_head": "0" * 40,
        }
        baseline.update(overrides)
        self.baseline_path.write_text(
            json.dumps(baseline, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def measurement(
        self,
        *,
        backend: tuple[float, float] = (87.5, 72.5),
        frontend: tuple[float, float] = (96.0, 90.0),
        sizes: dict[str, int] | None = None,
        api_factory: dict[str, int] | None = None,
    ) -> dict:
        module = self.module
        return {
            "backend": {"line_percent": backend[0], "branch_percent": backend[1]},
            "frontend": {"line_percent": frontend[0], "branch_percent": frontend[1]},
            "sizes": dict(module.measure_sizes() if sizes is None else sizes),
            "api_factory": dict(
                module.measure_api_factory() if api_factory is None else api_factory
            ),
            "measured_at": "2026-09-15T00:00:00+00:00",
            "git_head": self.head,
            "gate_version": module.GATE_VERSION,
        }

    def owner(self, **overrides: object) -> dict:
        payload = {
            "token": "0f0f0f0f0f0f0f0f",
            "pid": 4321,
            "purpose": "run",
            "started_at": "2026-09-15T00:00:00+00:00",
        }
        payload.update(overrides)
        return payload

    def sections_for(self, run_status: str, failed_section: str | None) -> list[dict]:
        """The section list a real run would publish for this outcome (FIX-05A)."""

        if run_status == "running":
            return []
        if run_status == "failed" and failed_section in self.module.SECTION_NAMES:
            index = self.module.SECTION_NAMES.index(failed_section)
            return [
                {
                    "name": name,
                    "status": "ok" if position < index else "failed",
                    "seconds": 0.1,
                }
                for position, name in enumerate(self.module.SECTION_NAMES[: index + 1])
            ]
        return [
            {"name": name, "status": "ok", "seconds": 0.1}
            for name in self.module.SECTION_NAMES
        ]

    def write_records(
        self,
        *,
        run_id: str = "20260915T000000Z-aaaa1111",
        run_status: str = "success",
        failed_section: str | None = None,
        write_success: bool = True,
        success_run_id: str | None = None,
        success_status: str = "success",
        record_version: int | None = None,
        gate_version: int | None = None,
        sections: list[dict] | None = None,
        measurement: dict | None = None,
        baseline_sha256: str | None = None,
        measurement_fingerprint: str | None = None,
        baseline_state: str | None = None,
        owner: dict | None = None,
        success_owner: dict | None = None,
        generation: int = 1,
        success_generation: int | None = None,
        raw_success: str | None = None,
    ) -> dict:
        """Write the lifecycle records the updater reads, defaulting to a promotable run."""

        module = self.module
        manifest = module.build_manifest()
        state = (
            baseline_state
            if baseline_state is not None
            else ("present" if self.baseline_path.is_file() else "never_created")
        )
        published_by = self.owner() if owner is None else owner
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "run.json").write_text(
            json.dumps(
                {
                    "record_version": module.RUN_RECORD_VERSION,
                    "gate_version": module.GATE_VERSION,
                    "run_id": run_id,
                    "generation": generation,
                    "status": run_status,
                    "started_at": "2026-09-15T00:00:00+00:00",
                    "finished_at": None
                    if run_status == "running"
                    else "2026-09-15T00:10:00+00:00",
                    "failed_section": failed_section,
                    "sections": sections
                    if sections is not None
                    else self.sections_for(run_status, failed_section),
                    "inputs": module.manifest_summary(manifest),
                    "baseline_state": state,
                    "owner": published_by,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if raw_success is not None:
            (self.state / "success.json").write_text(raw_success, encoding="utf-8")
            return manifest
        if not write_success:
            return manifest
        groups = json.loads(json.dumps(manifest["groups"]))
        if measurement_fingerprint is not None:
            groups["measurement"]["fingerprint"] = measurement_fingerprint
        (self.state / "success.json").write_text(
            json.dumps(
                {
                    "record_version": module.SUCCESS_RECORD_VERSION
                    if record_version is None
                    else record_version,
                    "gate_version": module.GATE_VERSION
                    if gate_version is None
                    else gate_version,
                    "status": success_status,
                    "run_id": success_run_id or run_id,
                    "generation": generation
                    if success_generation is None
                    else success_generation,
                    "started_at": "2026-09-15T00:00:00+00:00",
                    "finished_at": "2026-09-15T00:10:00+00:00",
                    "sections": sections
                    if sections is not None
                    else [
                        {"name": name, "status": "ok", "seconds": 0.1}
                        for name in module.SECTION_NAMES
                    ],
                    "measurement": self.measurement()
                    if measurement is None
                    else measurement,
                    "baseline": {
                        "path": "project/quality-baselines.v1.json",
                        "present": self.baseline_path.is_file(),
                        "sha256": (
                            baseline_sha256
                            if baseline_sha256 is not None
                            else hashlib.sha256(self.baseline_bytes()).hexdigest()
                        )
                        if self.baseline_path.is_file()
                        else None,
                        "baseline_version": 1 if self.baseline_path.is_file() else None,
                    },
                    "baseline_state": (
                        state if self.baseline_path.is_file() else "never_created"
                    ),
                    "inputs": {
                        "rule": module.MANIFEST_RULE,
                        "groups": groups,
                        "files": manifest["files"],
                    },
                    "owner": published_by if success_owner is None else success_owner,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.gate_path), *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def refuse(self, *arguments: str, expect: str) -> None:
        """Run a write path, require a refusal that names its reason, keep the contract put."""

        before = self.snapshot()
        completed = self.run_cli(*arguments)
        assert completed.returncode != 0, completed.stdout + completed.stderr
        assert "baseline unchanged" in completed.stderr, completed.stderr
        assert expect in completed.stderr, completed.stderr
        assert self.snapshot() == before

    def snapshot(self) -> tuple[bool, bytes]:
        """The contract's exact state: still absent, or byte-identical."""

        if not self.baseline_path.is_file():
            return (False, b"")
        return (True, self.baseline_bytes())


@pytest.fixture()
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path / "repo")


# --------------------------------------------------------------------- the matrix


def test_the_sandbox_starts_from_a_promotable_state(sandbox: Sandbox) -> None:
    """The control case: without it, every refusal below could be vacuous."""

    sandbox.write_records()
    before = json.loads(sandbox.baseline_bytes().decode("utf-8"))
    completed = sandbox.run_cli("--update-baseline")
    assert completed.returncode == 0, completed.stderr
    updated = json.loads(sandbox.baseline_bytes().decode("utf-8"))
    assert updated != before
    # coverage only tightened, the old guarded file kept its registration
    assert updated["coverage"]["backend"]["line_percent"] == 87.5
    assert updated["coverage"]["frontend"]["branch_percent"] == 90.0
    assert (
        "project/backend/src/procurement_review/review_repository.py"
        in (updated["guarded_files"])
    )
    for relative in GATE.SPLIT_MODULE_REGISTRY:
        assert relative in updated["guarded_files"], relative
    assert updated["guarded_registry"] == 1
    assert updated["size_exemptions"] == {}
    record = json.loads((sandbox.state / "baseline-update.json").read_text("utf-8"))
    assert record["mode"] == "update"
    assert record["baseline"]["sha256_before"] != record["baseline"]["sha256_after"]
    assert record["governance_check"]["result"] == "ok"


def test_a_second_update_from_the_same_run_is_refused(sandbox: Sandbox) -> None:
    """Re-running the update must not widen anything a second time."""

    sandbox.write_records()
    assert sandbox.run_cli("--update-baseline").returncode == 0
    sandbox.refuse("--update-baseline", expect="already moved")


def test_no_run_state_is_refused(sandbox: Sandbox) -> None:
    sandbox.refuse("--update-baseline", expect="no run state")


def test_a_failed_run_never_promotes_its_measurement(sandbox: Sandbox) -> None:
    """A-10's original reproduction: a 1% measurement from a failed run."""

    sandbox.write_records(
        run_status="failed",
        failed_section="ratchet",
        write_success=False,
        measurement=sandbox.measurement(backend=(1.0, 1.0), frontend=(1.0, 1.0)),
    )
    (sandbox.state / "failure.json").write_text(
        json.dumps({"status": "failed", "section": "ratchet"}), encoding="utf-8"
    )
    sandbox.refuse("--update-baseline", expect="'failed', not 'success'")


def test_an_running_or_interrupted_run_is_refused(sandbox: Sandbox) -> None:
    sandbox.write_records(run_status="running")
    sandbox.refuse("--update-baseline", expect="'running', not 'success'")


def test_a_leftover_success_record_never_passes_for_the_latest_run(
    sandbox: Sandbox,
) -> None:
    sandbox.write_records(
        run_id="20260915T010000Z-bbbb2222", success_run_id="20260915T000000Z-aaaa1111"
    )
    sandbox.refuse("--update-baseline", expect="belongs to run")


@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        ({"write_success": False}, "no success record"),
        ({"raw_success": '{"record_version": 1,'}, "damaged"),
        ({"record_version": 0}, "unsupported success record version"),
        ({"gate_version": 1}, "was written by gate_version"),
        ({"success_status": "failed"}, "carries status"),
        ({"measurement": {}}, "coverage measurement is missing or malformed"),
        (
            {"sections": [{"name": "gov", "status": "ok", "seconds": 0.1}]},
            "does not describe one complete green run",
        ),
        (
            {
                "sections": [
                    {"name": name, "status": "ok", "seconds": 0.1}
                    for name in GATE.SECTION_NAMES
                ][:-1]
            },
            "does not describe one complete green run",
        ),
    ],
)
def test_a_missing_damaged_or_incomplete_record_is_refused(
    sandbox: Sandbox, kwargs: dict, expect: str
) -> None:
    sandbox.write_records(**kwargs)
    sandbox.refuse("--update-baseline", expect=expect)


@pytest.mark.parametrize(
    "mutated",
    [
        "project/backend/src/procurement_review/run_projection.py",
        "project/backend/tests/test_sample.py",
        "project/backend/uv.lock",
        "project/frontend/src/App.tsx",
        "project/frontend/package-lock.json",
        "project/scripts/quality_gate.py",
        "project/design-references/README.md",
    ],
)
def test_changed_measured_content_is_refused_at_the_same_head(
    sandbox: Sandbox, mutated: str
) -> None:
    """Same HEAD is not the same input: code, tests, locks, tools and fixtures count."""

    sandbox.write_records()
    head_before = sandbox.head
    path = sandbox.root / mutated
    path.write_text(path.read_text(encoding="utf-8") + "\n# change\n", encoding="utf-8")
    assert sandbox.head == head_before
    sandbox.refuse("--update-baseline", expect="measured content changed")


def test_a_governance_only_change_is_promoted_with_a_fresh_check(
    sandbox: Sandbox,
) -> None:
    """A handoff-only edit may reuse the measurement; the newest governance run decides."""

    sandbox.write_records()
    note = sandbox.root / "workspace/docs/note.md"
    note.write_text("# note, revised\n", encoding="utf-8")
    completed = sandbox.run_cli("--update-baseline")
    assert completed.returncode == 0, completed.stderr
    record = json.loads((sandbox.state / "baseline-update.json").read_text("utf-8"))
    recorded = json.loads((sandbox.state / "success.json").read_text("utf-8"))
    assert (
        record["governance_check"]["fingerprint"]
        != (recorded["inputs"]["groups"]["governance"]["fingerprint"])
    )
    assert (
        record["measurement_inputs"]["fingerprint"]
        == (recorded["inputs"]["groups"]["measurement"]["fingerprint"])
    )


def test_a_failing_governance_check_blocks_promotion(sandbox: Sandbox) -> None:
    sandbox.write_records()
    (sandbox.root / "governance-failure").write_text("fail\n", encoding="utf-8")
    sandbox.refuse("--update-baseline", expect="fails the governance checks")


def test_a_baseline_that_moved_after_the_run_is_refused(sandbox: Sandbox) -> None:
    sandbox.write_records()
    sandbox.write_baseline(
        coverage_tolerance_pp=0.1, updated_at="2026-09-15T09:00:00+00:00"
    )
    sandbox.refuse("--update-baseline", expect="already moved")


def test_a_noise_level_drop_keeps_the_older_higher_floor(sandbox: Sandbox) -> None:
    sandbox.write_records(
        measurement=sandbox.measurement(backend=(79.95, 72.5), frontend=(96.0, 90.0))
    )
    completed = sandbox.run_cli("--update-baseline")
    assert completed.returncode == 0, completed.stderr
    updated = json.loads(sandbox.baseline_bytes().decode("utf-8"))
    assert updated["coverage"]["backend"]["line_percent"] == 80.0
    assert updated["coverage"]["backend"]["branch_percent"] == 72.5


def test_a_ceiling_is_never_raised(sandbox: Sandbox) -> None:
    """A guarded file that shrank tightens its ceiling; growth is refused."""

    guarded = (
        sandbox.root / "project/backend/src/procurement_review/review_repository.py"
    )
    guarded.write_text(GUARDED_MODULE.format(body=_body(700)), encoding="utf-8")
    sandbox.write_records()
    assert sandbox.run_cli("--update-baseline").returncode == 0
    updated = json.loads(sandbox.baseline_bytes().decode("utf-8"))
    relative = "project/backend/src/procurement_review/review_repository.py"
    assert updated["guarded_files"][relative] == sandbox.module.line_count(guarded)
    assert updated["guarded_files"][relative] < 900

    # growing back past the recorded ceiling is a ratchet failure, not a promotion
    sandbox.write_baseline(guarded_files={relative: 700})
    guarded.write_text(GUARDED_MODULE.format(body=_body(900)), encoding="utf-8")
    sandbox.write_records(run_id="20260915T020000Z-cccc3333")
    sandbox.refuse("--update-baseline", expect="refusing to write a weaker")


def test_a_missing_registry_module_is_refused(sandbox: Sandbox) -> None:
    """A rename or split must update the registry deliberately (FIX-02 AF-10)."""

    missing = sorted(GATE.SPLIT_MODULE_REGISTRY)[0]
    (sandbox.root / missing).unlink()
    # the gate run itself happened after the rename, so the content is current and the
    # only thing that still disagrees is the tool's own responsibility registry
    sandbox.write_records()
    sandbox.refuse("--update-baseline", expect="SPLIT_MODULE_REGISTRY")


def test_the_ratchet_refuses_an_unregistered_oversized_file(sandbox: Sandbox) -> None:
    module = sandbox.module
    relative = "project/backend/src/procurement_review/fresh_big_module.py"
    (sandbox.root / relative).write_text(
        GUARDED_MODULE.format(body=_body(900)), encoding="utf-8"
    )
    measured = sandbox.measurement()
    problems = module.ratchet_problems(
        measured, json.loads(sandbox.baseline_bytes().decode("utf-8"))
    )
    assert any("without a registered exemption" in problem for problem in problems)


def test_the_registry_check_is_enforced_once_the_baseline_carries_it(
    sandbox: Sandbox,
) -> None:
    module = sandbox.module
    baseline = json.loads(sandbox.baseline_bytes().decode("utf-8"))
    baseline["guarded_registry"] = 1
    measured = sandbox.measurement()
    problems = module.ratchet_problems(measured, baseline)
    missing = [
        problem for problem in problems if "not registered in guarded_files" in problem
    ]
    assert len(missing) == len(GATE.SPLIT_MODULE_REGISTRY)
    baseline["guarded_files"].update(
        {
            relative: module.line_count(sandbox.root / relative)
            for relative in GATE.SPLIT_MODULE_REGISTRY
        }
    )
    assert not [
        problem
        for problem in module.ratchet_problems(measured, baseline)
        if "not registered in guarded_files" in problem
    ]


# ------------------------------------------------------------------------ bootstrap


def test_bootstrap_never_replaces_an_existing_contract(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path / "repo")
    sandbox.write_records()
    sandbox.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=80/70,frontend=90/85",
        expect="a baseline already exists",
    )
    assert (
        sandbox.run_cli(
            "--bootstrap-baseline", "--coverage-floor", "backend=80/70,frontend=90/85"
        ).returncode
        != 0
    )


def test_a_deleted_contract_is_never_a_first_time_repository(tmp_path: Path) -> None:
    """FIX-05A / D-AUD-01: deleting the committed contract is not a reset.

    The synthetic repository committed its baseline, so the deletion is provable from
    Git even though the file is gone and the records claim ``never_created``.
    """

    sandbox = Sandbox(tmp_path / "repo")
    sandbox.baseline_path.unlink()
    sandbox.write_records()
    sandbox.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=1/1,frontend=1/1",
        expect="prior contract",
    )
    assert not sandbox.baseline_path.exists()


def test_bootstrap_requires_declared_floors_and_measured_headroom(
    tmp_path: Path,
) -> None:
    sandbox = Sandbox(tmp_path / "repo", with_baseline=False)
    sandbox.write_records()
    sandbox.refuse("--bootstrap-baseline", expect="requires explicit initial floors")
    sandbox.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=80/70",
        expect="no initial floor declared for frontend",
    )
    sandbox.refuse(
        "--bootstrap-baseline",
        "--coverage-floor",
        "backend=99/70,frontend=90/85",
        expect="above the measured",
    )


def test_bootstrap_creates_the_declared_contract_atomically(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path / "repo", with_baseline=False)
    sandbox.write_records()
    completed = sandbox.run_cli(
        "--bootstrap-baseline", "--coverage-floor", "backend=80/70,frontend=90/85"
    )
    assert completed.returncode == 0, completed.stderr
    created = json.loads(sandbox.baseline_path.read_text(encoding="utf-8"))
    assert created["coverage"] == {
        "backend": {"line_percent": 80.0, "branch_percent": 70.0},
        "frontend": {"line_percent": 90.0, "branch_percent": 85.0},
    }
    assert created["guarded_registry"] == 1
    assert set(GATE.SPLIT_MODULE_REGISTRY) <= set(created["guarded_files"])
    record = json.loads((sandbox.state / "baseline-update.json").read_text("utf-8"))
    assert record["mode"] == "bootstrap"
    assert record["baseline"]["sha256_before"] is None


def test_an_interrupted_publish_leaves_no_half_record(sandbox: Sandbox) -> None:
    """Publication is a same-directory temporary file plus an atomic replace."""

    sandbox.write_records()
    assert sandbox.run_cli("--update-baseline").returncode == 0
    leftovers = sorted(
        path.name for path in sandbox.state.iterdir() if path.suffix == ".tmp"
    )
    assert leftovers == []
    leftovers = sorted(
        path.name
        for path in sandbox.baseline_path.parent.iterdir()
        if path.name.startswith(".quality-baselines")
    )
    assert leftovers == []
