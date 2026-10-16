"""FIX-09D: the coverage entry's contract, and the gate semantics around a failure.

The quality gate's frontend section is the one place where a final acceptance decision is
taken, so the resource budget it uses has to be a stated contract rather than a preference:

* only the *coverage* entry may be pinned to one worker and to file-serial execution — the
  ordinary ``npm test`` must stay parallel, because it is what developers run all day;
* the default single-test timeout, the coverage thresholds and the baseline ratchet must be
  untouched: the fix is a resource budget, never a weaker assertion;
* no retry, skip, bail or global timeout knob may appear in either entry or in the test
  configuration;
* the gate must drive the coverage entry by name, and a failing frontend section must
  publish no success record and leave the baseline byte-identical.

The last point is exercised through the real gate module on a synthetic repository, so the
lifecycle logic under test is the shipped code; only the expensive external sections are
replaced, and only inside the sandbox copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_fix01d_quality_gate import Sandbox
from test_fix05_quality_gate import Harness

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_JSON = REPO_ROOT / "project" / "frontend" / "package.json"
VITEST_CONFIG = REPO_ROOT / "project" / "frontend" / "vitest.config.ts"
GATE_SOURCE = REPO_ROOT / "project" / "scripts" / "quality_gate.py"
BASELINE = REPO_ROOT / "project" / "quality-baselines.v1.json"

SERIAL_FLAGS = ("--maxWorkers=1", "--fileParallelism=false")

#: Knobs that would soften a failure into a pass.  None of them may appear anywhere in the
#: frontend test surface or in the gate's frontend invocation.
FORBIDDEN_KNOBS = (
    "--retry",
    "--bail",
    "testTimeout",
    "hookTimeout",
    "teardownTimeout",
    "--exclude",
    "--testNamePattern",
    "it.skip",
    "describe.skip",
    "it.only",
)


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    return Harness(Sandbox(tmp_path / "repo"))


def _scripts() -> dict[str, str]:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["scripts"]


def test_only_the_coverage_entry_is_pinned_to_one_worker(tmp_path: Path) -> None:
    """AC-06: the coverage budget is fixed; the ordinary test entry stays parallel."""

    scripts = _scripts()
    coverage = scripts["test:coverage"]
    for flag in SERIAL_FLAGS:
        assert flag in coverage, coverage
    # The ordinary entries must not inherit the serial budget: developers keep the fast
    # parallel run, and the default per-test timeout applies to both.
    for entry in ("test", "test:watch"):
        for flag in SERIAL_FLAGS:
            assert flag not in scripts[entry], (entry, scripts[entry])
    assert scripts["test"] == "vitest run", scripts["test"]
    assert scripts["test:watch"] == "vitest", scripts["test:watch"]


def test_no_entry_or_test_configuration_softens_a_failure() -> None:
    """AC-06: no retry, skip, bail or global timeout was added anywhere."""

    surfaces = {
        "package.json": PACKAGE_JSON.read_text(encoding="utf-8"),
        "vitest.config.ts": VITEST_CONFIG.read_text(encoding="utf-8"),
    }
    for name, content in surfaces.items():
        for knob in FORBIDDEN_KNOBS:
            assert knob not in content, (name, knob)
    config = VITEST_CONFIG.read_text(encoding="utf-8")
    # The declared test selection and environment are unchanged, so "23 files / 304 tests"
    # remains the measured surface rather than a narrowed one.
    assert "include: ['src/**/*.{test,spec}.{ts,tsx}']" in config, config
    assert "exclude: ['e2e/**']" in config, config
    assert "environment: 'jsdom'" in config, config


def test_the_gate_drives_the_coverage_entry_by_name() -> None:
    """AC-06: the gate owns no second copy of the flags and never runs the parallel entry."""

    source = GATE_SOURCE.read_text(encoding="utf-8")
    assert '"run", "test:coverage"' in source, "the gate must call the coverage entry"
    for flag in SERIAL_FLAGS:
        # The budget lives in package.json alone; duplicating it in Python would let the
        # two drift apart, which is exactly what this contract forbids.
        assert flag not in source, (flag, "the gate hard-codes the serial flags")
    assert '["npm", "run", "test"]' not in source, (
        "the gate must not run the parallel entry"
    )


def test_the_coverage_ratchet_is_not_lowered() -> None:
    """AC-06: the recorded floors only move up; this fix does not touch them."""

    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))["coverage"]["frontend"]
    assert recorded == {"branch_percent": 90.15, "line_percent": 96.27}, recorded


def test_a_failing_frontend_section_publishes_nothing_and_moves_nothing(
    harness: Harness,
) -> None:
    """AC-06: a frontend failure keeps the gate red and leaves the contract untouched.

    The sandbox publishes one complete green run first, so a success record and a baseline
    both exist.  The frontend section is then replaced by the shipped gate's own failure
    signal, and the run must end non-zero: no new success record, no baseline write, and
    the recorded failure names the frontend section.
    """

    assert harness.green_run()["status"] == "success"
    success_path = harness.sandbox.state / "success.json"
    assert success_path.is_file(), "the green run published no success record"
    success_before = success_path.read_bytes()
    baseline_before = harness.sandbox.baseline_path.read_bytes()

    module = harness.module

    def frontend(measured: dict) -> None:
        raise module.GateFailure(
            "frontend",
            "injected real hang: npm run test:coverage exited non-zero",
        )

    def backend(measured: dict) -> None:
        measured["backend"] = {"line_percent": 87.5, "branch_percent": 72.5}

    module.SECTIONS = (
        ("gov", module.section_gov),
        ("ruff", lambda: None),
        ("backend", backend),
        ("frontend", frontend),
        ("ratchet", module.section_ratchet),
    )

    assert module.run_gate() != 0, "a failing frontend section must fail the gate"
    run = harness.read_state("run.json")
    assert run["status"] == "failed", run
    assert run["failed_section"] == "frontend", run
    failure = harness.read_state("failure.json")
    assert failure["section"] == "frontend", failure
    assert success_path.read_bytes() == success_before, (
        "a failed run published over the previous success record"
    )
    assert harness.sandbox.baseline_path.read_bytes() == baseline_before, (
        "a failed run moved the baseline"
    )
    assert not (harness.sandbox.state / "baseline-update.json").exists(), (
        "a failed run wrote a baseline update"
    )
