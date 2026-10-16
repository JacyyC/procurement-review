"""FIX-06D: one budget invariant for every guarded-path migration.

R-05 (the D-review counterexample) was that ``parse_migration_declaration`` compared a
migration's declared ceilings against the source ceiling only for ``split`` and
``merge``.  A ``rename`` therefore moved a guarded path onto a *larger* ceiling: with the
recorded budget at 832, a declaration naming one 1000-line target at ceiling 1000 passed
the gate and the updater, and the protection of that path rose from 832 to 1000.

Everything here runs the *real* tool: the shipped gate source is copied into a synthetic
repository together with stub governance scripts (``Sandbox`` from the FIX-01D suite),
the real lifecycle, ratchet, candidate construction and CLI write paths are the ones
under test, and only the expensive external sections are replaced inside the copy.

The cases follow the FIX-06D verification matrix:

* V-D-01 — the original window and its parameterised boundaries: an illegal migration is
  refused *before* anything is written and the real baseline keeps its exact bytes;
* V-D-02 — legal rename/split/merge/retire, chained and composed migrations, two sources
  sharing one target, an already guarded target, targets below the size limit and growth
  after the migration stay protected.

832/1000 are counterexample data, never a product branch: the parameterised cases below
use several unrelated budgets and paths.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_SOURCE = REPO_ROOT / "project" / "scripts" / "quality_gate.py"
BASELINE_PATH = REPO_ROOT / "project" / "quality-baselines.v1.json"
# EV-AUDIT-BASELINE: the committed contract the 2026-09-15 review measured against.
BASELINE_SHA256 = "e0d5b711fbce5eb15380574d58844d2e13ef26f44fba7e6a87a908c41d2c84cb"
FIX05_SUITE = REPO_ROOT / "project" / "backend" / "tests" / "test_fix05_quality_gate.py"


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


FIX05 = _load(FIX05_SUITE, "fix06d_fix05_suite")
Sandbox = FIX05.Sandbox
Harness = FIX05.Harness
_exact_lines = FIX05._exact_lines

GUARDED = FIX05.GUARDED
PART_A = FIX05.PART_A
PART_B = FIX05.PART_B
FINAL = "project/backend/src/procurement_review/repo_part_final.py"
LEGACY = "project/backend/src/procurement_review/legacy_repository.py"


# ------------------------------------------------------------------ building blocks


def _declaration(
    kind: str,
    targets: list[str],
    ceilings: dict[str, int],
    *,
    source_ceiling: int,
    reason: str = "FIX-06D 反例：显式迁移声明",
) -> dict:
    return {
        "kind": kind,
        "targets": targets,
        "responsibility": "迁移后的职责组",
        "source_ceiling": source_ceiling,
        "target_ceilings": ceilings,
        "reason": reason,
        "declared_at": "2026-09-16T00:00:00+00:00",
    }


def _contract(
    harness: Harness,
    *,
    kind: str = "rename",
    targets: list[str],
    ceilings: dict[str, int],
    source: str = GUARDED,
    source_ceiling: int = 832,
    extra_files: dict[str, int] | None = None,
    extra_migrations: dict[str, dict] | None = None,
) -> None:
    """Register ``source`` at ``source_ceiling`` and declare one migration for it."""

    guarded_files = {source: source_ceiling}
    migrations = {
        source: _declaration(kind, targets, ceilings, source_ceiling=source_ceiling)
    }
    if extra_files:
        guarded_files.update(extra_files)
    if extra_migrations:
        migrations.update(extra_migrations)
    harness.sandbox.write_baseline(
        guarded_files=guarded_files, guarded_migrations=migrations
    )


def _move(harness: Harness, source: str, target: str, lines: int) -> None:
    _exact_lines(harness.sandbox.root / target, lines)
    (harness.sandbox.root / source).unlink()


def _baseline(harness: Harness) -> dict:
    return json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))


def _rewrite_baseline(harness: Harness, payload: dict) -> None:
    harness.sandbox.baseline_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _declare(harness: Harness, source: str, declaration: dict) -> dict:
    """Record one further migration on top of the baseline the tool just wrote."""

    current = _baseline(harness)
    current["guarded_migrations"][source] = declaration
    _rewrite_baseline(harness, current)
    return current


def _legal_rename(harness: Harness, *, ceiling: int = 830, lines: int = 830) -> dict:
    """Drive one complete, legal rename through the real gate and updater."""

    _contract(harness, targets=[PART_A], ceilings={PART_A: ceiling}, source_ceiling=832)
    _move(harness, GUARDED, PART_A, lines)
    harness.green_run()
    harness.promote("--update-baseline")
    return _baseline(harness)


def _sizes(harness: Harness) -> dict[str, int]:
    return harness.module.measure_sizes()


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    return Harness(Sandbox(tmp_path / "repo"))


# ======================================================= V-D-01 the original R-05 window


def test_a_rename_above_the_source_ceiling_stops_the_gate(
    harness: Harness, capsys: pytest.CaptureFixture
) -> None:
    """R-05: rename used to skip the budget comparison that split and merge obey."""

    _contract(harness, targets=[PART_A], ceilings={PART_A: 1000}, source_ceiling=832)
    _move(harness, GUARDED, PART_A, 1000)
    before = harness.sandbox.baseline_bytes()
    harness.expect_run_failure("budget", capsys)
    assert harness.sandbox.baseline_bytes() == before


def test_the_updater_refuses_the_rename_and_keeps_the_real_baseline_bytes(
    harness: Harness,
) -> None:
    """The write path is not a second opinion: it refuses and changes nothing."""

    _contract(harness, targets=[PART_A], ceilings={PART_A: 1000}, source_ceiling=832)
    _move(harness, GUARDED, PART_A, 1000)
    harness.write_promotable_state()
    harness.refuse("--update-baseline", expect="budget")
    assert _baseline(harness)["guarded_files"] == {GUARDED: 832}


@pytest.mark.parametrize(
    ("source_ceiling", "declared", "lines"),
    [(832, 1000, 1000), (700, 701, 700), (1200, 1201, 1200), (540, 900, 860)],
)
def test_a_rename_ceiling_above_the_source_is_refused_for_any_budget(
    harness: Harness,
    capsys: pytest.CaptureFixture,
    source_ceiling: int,
    declared: int,
    lines: int,
) -> None:
    """Not a 832/1000 special case: the relation holds for every recorded budget."""

    _contract(
        harness,
        targets=[PART_A],
        ceilings={PART_A: declared},
        source_ceiling=source_ceiling,
    )
    _move(harness, GUARDED, PART_A, lines)
    harness.expect_run_failure("budget", capsys)


@pytest.mark.parametrize("targets", [(PART_A, PART_B), (PART_A, PART_B, FINAL)])
def test_a_rename_that_names_several_targets_is_refused(
    harness: Harness, capsys: pytest.CaptureFixture, targets: tuple[str, ...]
) -> None:
    """A rename moves one path onto exactly one target; several targets is a split."""

    declared = 832 // len(targets)
    _contract(
        harness,
        targets=list(targets),
        ceilings={target: declared for target in targets},
        source_ceiling=832,
    )
    for target in targets:
        _exact_lines(harness.sandbox.root / target, declared)
    (harness.sandbox.root / GUARDED).unlink()
    before = harness.sandbox.baseline_bytes()
    harness.expect_run_failure("exactly one", capsys)
    assert harness.sandbox.baseline_bytes() == before


def test_a_chain_hop_that_raises_the_next_budget_is_refused(harness: Harness) -> None:
    """A second hop inherits the budget the first hop recorded, it never raises it."""

    _legal_rename(harness)
    _declare(
        harness,
        PART_A,
        _declaration(
            "rename",
            [FINAL],
            {FINAL: 900},
            source_ceiling=830,
            reason="FIX-06D：越界的第二跳",
        ),
    )
    _move(harness, PART_A, FINAL, 880)
    harness.write_promotable_state()
    harness.refuse("--update-baseline", expect="budget")
    assert _baseline(harness)["guarded_files"][PART_A] == 830


def test_a_target_that_is_neither_present_nor_migrating_is_refused(
    harness: Harness, capsys: pytest.CaptureFixture
) -> None:
    """A declaration may not point at a phantom path to escape registration."""

    phantom = "project/backend/src/procurement_review/phantom_module.py"
    _contract(harness, targets=[phantom], ceilings={phantom: 800}, source_ceiling=832)
    (harness.sandbox.root / GUARDED).unlink()
    before = harness.sandbox.baseline_bytes()
    harness.expect_run_failure("exist", capsys)
    assert harness.sandbox.baseline_bytes() == before


# ============================================================ V-D-02 the legal migrations


def test_a_legal_rename_inherits_the_budget_and_keeps_the_target_guarded(
    harness: Harness,
) -> None:
    """The positive control: the same path with a ceiling inside the budget succeeds."""

    updated = _legal_rename(harness)
    assert GUARDED not in updated["guarded_files"]
    assert updated["guarded_files"][PART_A] == 830
    assert updated["guarded_migrations"][GUARDED]["targets"] == [PART_A]

    _exact_lines(harness.sandbox.root / PART_A, 900)
    problems = harness.module.ratchet_problems(harness.measurement(), updated)
    assert any(PART_A in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("source_ceiling", "declared", "lines"),
    [(832, 832, 832), (700, 700, 690), (1200, 1200, 1199), (540, 1, 1)],
)
def test_a_rename_at_or_below_the_source_ceiling_still_succeeds(
    harness: Harness, source_ceiling: int, declared: int, lines: int
) -> None:
    """The boundary is inclusive: equal ceilings are a legal rename, not a widening."""

    _contract(
        harness,
        targets=[PART_A],
        ceilings={PART_A: declared},
        source_ceiling=source_ceiling,
    )
    _move(harness, GUARDED, PART_A, lines)
    harness.green_run()
    harness.promote("--update-baseline")
    updated = _baseline(harness)
    assert updated["guarded_files"][PART_A] == min(declared, lines)
    assert updated["guarded_files"][PART_A] <= source_ceiling


def test_a_chained_rename_keeps_the_budget_on_the_live_path(harness: Harness) -> None:
    """A -> PART_A -> FINAL: the second hop follows the budget to the live file."""

    _legal_rename(harness)
    _declare(
        harness,
        PART_A,
        _declaration(
            "rename",
            [FINAL],
            {FINAL: 820},
            source_ceiling=830,
            reason="FIX-06D：连续迁移的第二跳",
        ),
    )
    _move(harness, PART_A, FINAL, 820)
    harness.green_run()
    harness.promote("--update-baseline")
    updated = _baseline(harness)
    assert PART_A not in updated["guarded_files"]
    assert updated["guarded_files"][FINAL] == 820
    assert updated["guarded_files"][FINAL] <= 832
    assert set(updated["guarded_migrations"]) == {GUARDED, PART_A}
    assert updated["guarded_migrations"][GUARDED]["targets"] == [PART_A]

    _exact_lines(harness.sandbox.root / FINAL, 900)
    problems = harness.module.ratchet_problems(harness.measurement(), updated)
    assert any(FINAL in problem for problem in problems), problems


def test_a_rename_followed_by_a_split_stays_inside_the_original_budget(
    harness: Harness,
) -> None:
    """A composed migration: the split shares the budget the rename handed over."""

    _legal_rename(harness)
    _declare(
        harness,
        PART_A,
        _declaration(
            "split",
            [PART_B, FINAL],
            {PART_B: 410, FINAL: 410},
            source_ceiling=830,
            reason="FIX-06D：rename 之后的拆分",
        ),
    )
    _exact_lines(harness.sandbox.root / PART_B, 410)
    _exact_lines(harness.sandbox.root / FINAL, 410)
    (harness.sandbox.root / PART_A).unlink()
    harness.green_run()
    harness.promote("--update-baseline")
    updated = _baseline(harness)
    assert PART_A not in updated["guarded_files"]
    assert updated["guarded_files"][PART_B] == 410
    assert updated["guarded_files"][FINAL] == 410
    assert sum(updated["guarded_files"][n] for n in (PART_B, FINAL)) <= 832

    _exact_lines(harness.sandbox.root / PART_B, 700)
    problems = harness.module.ratchet_problems(harness.measurement(), updated)
    assert any(PART_B in problem for problem in problems), problems


def test_a_split_after_a_rename_may_not_re_expand_the_budget(
    harness: Harness, capsys: pytest.CaptureFixture
) -> None:
    """The negative half of the composed migration above."""

    _legal_rename(harness)
    _declare(
        harness,
        PART_A,
        _declaration(
            "split",
            [PART_B, FINAL],
            {PART_B: 600, FINAL: 600},
            source_ceiling=830,
            reason="FIX-06D：rename 之后的越界拆分",
        ),
    )
    _exact_lines(harness.sandbox.root / PART_B, 600)
    _exact_lines(harness.sandbox.root / FINAL, 600)
    (harness.sandbox.root / PART_A).unlink()
    before = harness.sandbox.baseline_bytes()
    harness.expect_run_failure("budget", capsys)
    assert harness.sandbox.baseline_bytes() == before


def test_two_sources_mapping_to_one_target_do_not_multiply_the_budget(
    harness: Harness,
) -> None:
    """Two renames onto a shared target tighten to the smaller budget."""

    _exact_lines(harness.sandbox.root / LEGACY, 800)
    _contract(
        harness,
        targets=[PART_A],
        ceilings={PART_A: 832},
        source_ceiling=832,
        extra_files={LEGACY: 800},
        extra_migrations={
            LEGACY: _declaration(
                "rename",
                [PART_A],
                {PART_A: 800},
                source_ceiling=800,
                reason="FIX-06D：第二个来源映射到同一目标",
            )
        },
    )
    _exact_lines(harness.sandbox.root / PART_A, 700)
    (harness.sandbox.root / GUARDED).unlink()
    (harness.sandbox.root / LEGACY).unlink()
    harness.green_run()
    harness.promote("--update-baseline")
    updated = _baseline(harness)
    assert updated["guarded_files"][PART_A] <= 800
    assert updated["guarded_files"][PART_A] <= 832

    _exact_lines(harness.sandbox.root / PART_A, 900)
    problems = harness.module.ratchet_problems(harness.measurement(), updated)
    assert any(PART_A in problem for problem in problems), problems


def test_a_rename_onto_an_already_guarded_target_tightens_it(harness: Harness) -> None:
    """An existing target keeps the tighter of the two ceilings."""

    _exact_lines(harness.sandbox.root / PART_B, 400)
    _contract(
        harness,
        targets=[PART_B],
        ceilings={PART_B: 800},
        source_ceiling=832,
        extra_files={PART_B: 600},
    )
    (harness.sandbox.root / GUARDED).unlink()
    harness.green_run()
    harness.promote("--update-baseline")
    updated = _baseline(harness)
    assert GUARDED not in updated["guarded_files"]
    assert updated["guarded_files"][PART_B] <= 600
    assert updated["guarded_files"][PART_B] == 400


def test_a_rename_target_below_the_size_limit_stays_registered_and_guarded(
    harness: Harness,
) -> None:
    """Small targets never fall out of the contract (D-AUD-05 at rename)."""

    _contract(harness, targets=[PART_A], ceilings={PART_A: 500}, source_ceiling=832)
    _move(harness, GUARDED, PART_A, 500)
    harness.green_run()
    harness.promote("--update-baseline")
    updated = _baseline(harness)
    assert updated["guarded_files"][PART_A] == 500
    assert _sizes(harness).get(PART_A) is None

    _exact_lines(harness.sandbox.root / PART_A, 900)
    problems = harness.module.ratchet_problems(harness.measurement(), updated)
    assert any(PART_A in problem for problem in problems), problems


def test_a_legal_retirement_still_releases_only_its_own_path(harness: Harness) -> None:
    """Retire is unchanged: no targets, its own registration drops, others stay."""

    _exact_lines(harness.sandbox.root / PART_B, 700)
    _contract(
        harness,
        kind="retire",
        targets=[],
        ceilings={},
        source_ceiling=832,
        extra_files={PART_B: 700},
    )
    (harness.sandbox.root / GUARDED).unlink()
    harness.green_run()
    harness.promote("--update-baseline")
    updated = _baseline(harness)
    assert GUARDED not in updated["guarded_files"]
    assert updated["guarded_files"][PART_B] == 700
    assert updated["guarded_migrations"][GUARDED]["kind"] == "retire"


def test_the_shipped_baseline_is_readable_and_untouched_by_this_stage() -> None:
    """The repository's own contract is the input this change must not break."""

    before = BASELINE_PATH.read_bytes()
    assert hashlib.sha256(before).hexdigest() == BASELINE_SHA256
    gate = _load(GATE_SOURCE, "fix06d_real_gate")
    parsed = gate.parse_baseline(json.loads(before.decode("utf-8")))
    assert parsed["guarded_files"]
    assert BASELINE_PATH.read_bytes() == before


# ============================================ V-D-01 one invariant, every entry point


def test_the_same_budget_judgement_runs_in_declaration_candidate_and_ratchet(
    harness: Harness,
) -> None:
    """No entry point may be the lenient one: all three answer from one rule."""

    _contract(harness, targets=[PART_A], ceilings={PART_A: 1000}, source_ceiling=832)
    _move(harness, GUARDED, PART_A, 1000)
    previous = json.loads(harness.sandbox.baseline_bytes().decode("utf-8"))
    gate = harness.module

    with pytest.raises(gate.UpdateRefused) as declared:
        gate.parse_baseline(previous)
    assert "budget" in str(declared.value), declared.value

    with pytest.raises(gate.UpdateRefused) as candidate:
        gate.candidate_baseline(harness.measurement(), previous)
    assert "budget" in str(candidate.value), candidate.value

    problems = gate.ratchet_problems(harness.measurement(), previous)
    assert any("budget" in problem for problem in problems), problems


@pytest.mark.parametrize("kind", ["rename", "split", "merge"])
def test_every_kind_that_keeps_a_budget_obeys_the_same_ceiling(
    harness: Harness, capsys: pytest.CaptureFixture, kind: str
) -> None:
    """Rename stopped being the exception: the rule is not kind-dependent."""

    targets = [PART_A] if kind == "rename" else [PART_A, PART_B]
    declared = 1000 if kind == "rename" else 500
    _contract(
        harness,
        kind=kind,
        targets=targets,
        ceilings={target: declared for target in targets},
        source_ceiling=832,
    )
    for target in targets:
        _exact_lines(harness.sandbox.root / target, declared)
    (harness.sandbox.root / GUARDED).unlink()
    before = harness.sandbox.baseline_bytes()
    harness.expect_run_failure("budget", capsys)
    assert harness.sandbox.baseline_bytes() == before


def test_the_original_window_is_refused_by_a_separate_gate_process(
    harness: Harness,
) -> None:
    """A real child process exits non-zero: the reading is not in-process bookkeeping."""

    _contract(harness, targets=[PART_A], ceilings={PART_A: 1000}, source_ceiling=832)
    _move(harness, GUARDED, PART_A, 1000)
    before = harness.sandbox.baseline_bytes()
    process = harness.child("run")
    returncode, stdout, stderr = harness.finish(process)
    assert returncode == 1, stdout + stderr
    assert "budget" in stderr, stderr
    assert "ratchet" in stderr, stderr
    assert harness.sandbox.baseline_bytes() == before
