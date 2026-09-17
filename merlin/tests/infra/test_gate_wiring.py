"""Every repo gate must run somewhere, or say in writing why it does not.

Measured 2026-09-16: 15 of the 34 `build_tools/scripts/check_*.py` ran in no git hook, no Stop-hook
and no workflow. That is worse than not having them, because a directory of check scripts reads as
coverage. `check_no_holdout_names` was one of the fifteen, and it was not merely idle -- it was
FAILING, on a tracked ratchet that named two held-out capsules outright.
"""
from __future__ import annotations

from merlin.common.paths import repo_root

ROOT = repo_root()
SCRIPTS = ROOT / "build_tools" / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"
HOOKS = ROOT / "build_tools" / "git-hooks"
SETTINGS = ROOT / ".claude" / "settings.json"


def _gates() -> list[str]:
    return sorted(p.name for p in SCRIPTS.glob("check_*.py"))


def _callers(exclude: str) -> str:
    """Everything that could invoke a gate: workflows, git hooks, the Stop hook, this README, and
    the gates that call other gates (check_docs.py runs four generators and three linters).

    The gate's OWN source is excluded, because a script rarely spells its own filename -- counting
    it made every genuinely wired gate look like an orphan on the first cut of this test.
    """
    parts = [p.read_text(encoding="utf-8") for p in sorted(WORKFLOWS.glob("*.yml"))]
    parts += [p.read_text(encoding="utf-8") for p in sorted(HOOKS.iterdir()) if p.is_file()]
    parts.append((WORKFLOWS / "README.md").read_text(encoding="utf-8"))
    if SETTINGS.is_file():
        parts.append(SETTINGS.read_text(encoding="utf-8"))
    parts += [p.read_text(encoding="utf-8") for p in SCRIPTS.glob("check_*.py") if p.name != exclude]
    return "\n".join(parts)


def test_every_gate_is_wired_or_explained():
    orphans = [g for g in _gates() if g not in _callers(exclude=g)]
    assert not orphans, (
        "these gates run nowhere and are not explained in .github/workflows/README.md: "
        + ", ".join(orphans))


def test_the_readme_explains_the_ones_that_stay_out():
    """An omission and a decision look identical in a diff; only one of them is reviewable."""
    readme = (WORKFLOWS / "README.md").read_text(encoding="utf-8")
    assert "Gates that deliberately run nowhere" in readme
    for named in ("check_standalone_install.py", "check_repro_env.py", "check_generated_target.py"):
        assert named in readme, named
