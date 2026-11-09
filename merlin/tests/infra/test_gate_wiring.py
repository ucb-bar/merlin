"""Every repo gate must run somewhere, or say in writing why it does not.

Measured 2026-09-16: 15 of the 34 `build_tools/scripts/check_*.py` ran in no git hook, no Stop-hook
and no workflow. That is worse than not having them, because a directory of check scripts reads as
coverage. `check_no_holdout_names` was one of the fifteen, and it was not merely idle -- it was
FAILING, on a tracked ratchet that named two held-out capsules outright.
"""

from __future__ import annotations

import json
import os
import subprocess

import yaml

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
    assert not orphans, "these gates run nowhere and are not explained in .github/workflows/README.md: " + ", ".join(
        orphans
    )


def test_the_readme_explains_the_ones_that_stay_out():
    """An omission and a decision look identical in a diff; only one of them is reviewable."""
    readme = (WORKFLOWS / "README.md").read_text(encoding="utf-8")
    assert "Gates that deliberately run nowhere" in readme
    for named in ("check_standalone_install.py", "check_repro_env.py", "check_generated_target.py"):
        assert named in readme, named


def test_required_ci_status_rejects_each_failed_cancelled_or_skipped_dependency():
    workflow = yaml.load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push", "pull_request", "merge_group"}
    gate = workflow["jobs"]["fast"]
    assert gate["if"] == "${{ always() }}"
    assert set(gate["needs"]) == {"checks", "ruff", "integration"}
    command = gate["steps"][0]["run"]
    cases = [dict.fromkeys(gate["needs"], "success")]
    for name in gate["needs"]:
        cases.extend({**cases[0], name: state} for state in ("failure", "cancelled", "skipped"))
    for index, states in enumerate(cases):
        result = subprocess.run(
            ["bash", "-e", "-c", command],
            env={**os.environ, "RESULTS": json.dumps({name: {"result": state} for name, state in states.items()})},
            text=True,
            capture_output=True,
        )
        assert (result.returncode == 0) == (index == 0), (states, result.stderr)
    docs = workflow["jobs"]["docs-freshness"]
    assert docs["if"] == "${{ always() }}" and docs["needs"] == "checks"


def test_ci_reuses_required_installation_and_retains_action_and_job_boundaries():
    documents = [
        yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader) for path in sorted(WORKFLOWS.glob("*.yml"))
    ]
    structure_calls = 0
    for document in documents:
        assert document["permissions"] == {"contents": "read"}
        for job in document["jobs"].values():
            for step in job.get("steps", []):
                structure_calls += "build_tools/scripts/check_structure.py" in step.get("run", "")
                action = step.get("uses", "")
                if not action or action.startswith("./"):
                    continue
                _, separator, commit = action.partition("@")
                assert separator and len(commit) == 40 and all(char in "0123456789abcdef" for char in commit)
                if action.startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] == "false"
    assert structure_calls == 1
    integration = yaml.load((WORKFLOWS / "integration.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert {"workflow_call", "workflow_dispatch", "schedule"} == set(integration["on"])
    commands = "\n".join(step.get("run", "") for step in integration["jobs"]["hosted-boundaries"]["steps"])
    assert "check_standalone_install.py --wheel" in commands and "--require-tools" in commands
    assert "--junitxml=" in commands
    for name in ("ci.yml", "release.yml"):
        caller = yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        assert any(job.get("uses") == "./.github/workflows/integration.yml" for job in caller["jobs"].values())
