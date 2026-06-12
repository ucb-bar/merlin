"""Validate the shape of a generated target repository.

A generated ``merlin-target-<name>/`` repo is considered structurally valid when it carries
its five plan artifacts, an evidence report, the per-layer directories (xdsl/runtime/zephyr/
llvm/tests), and AGENT.md coverage. This is the logic behind
``build_tools/scripts/check_generated_target.py`` and the ``targetgen inspect`` command.
"""
from __future__ import annotations

from pathlib import Path

from ..contracts.validate import validate_target_repo

# Files/dirs every generated target repo must contain.
REQUIRED_TARGET_PATHS: list[str] = [
    "AGENT.md",
    "contracts/target_contract.yaml",
    "contracts/dialect_plan.yaml",
    "contracts/runtime_adapter_plan.yaml",
    "contracts/zephyr_plan.yaml",
    "contracts/llvm_extension_plan.yaml",
    "docs/evidence_report.md",
    "xdsl",
    "runtime",
    "zephyr",
    "llvm",
    "tests",
]


def _missing_agent_md(repo: Path) -> list[str]:
    """Return repo-relative dirs that lack an AGENT.md (skipping hidden/pycache dirs)."""
    problems: list[str] = []
    for d in sorted(p for p in repo.rglob("*") if p.is_dir()):
        name = d.name
        if name.startswith(".") or name == "__pycache__":
            continue
        if not (d / "AGENT.md").is_file():
            problems.append(f"missing AGENT.md: {d.relative_to(repo)}")
    return problems


def check_generated_target(target_repo: str | Path, validate_contracts: bool = True) -> list[str]:
    """Return a list of structural problems for a generated target repo (empty == ok)."""
    repo = Path(target_repo)
    problems: list[str] = []
    if not repo.is_dir():
        return [f"target repo not found: {repo}"]

    for rel in REQUIRED_TARGET_PATHS:
        if not (repo / rel).exists():
            problems.append(f"missing: {rel}")

    if not (repo / "AGENT.md").is_file():
        problems.append("missing AGENT.md: (repo root)")
    problems.extend(_missing_agent_md(repo))

    if validate_contracts:
        problems.extend(validate_target_repo(repo))

    return problems
