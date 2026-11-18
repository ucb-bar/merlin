"""Load, validate and report generated-target contracts and repository structure.

These read-only checks return diagnostics. They establish neither numerical
correctness nor compiler, simulator or hardware qualification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.common import record_formats as schemas
from merlin.common.record_formats import load_yaml


PLAN_FILES: dict[str, tuple[str, str]] = {
    "target_contract": ("target_contract.yaml", "target_contract"),
    "dialect_plan": ("dialect_plan.yaml", "dialect_plan"),
    "runtime_adapter_plan": ("runtime_adapter_plan.yaml", "runtime_adapter_plan"),
    "zephyr_plan": ("zephyr_plan.yaml", "zephyr_plan"),
    "llvm_extension_plan": ("llvm_extension_plan.yaml", "llvm_extension_plan"),
}


OPTIONAL_PLANS: frozenset[str] = frozenset({"dialect_plan"})


REQUIRED_PLANS: tuple[str, ...] = tuple(p for p in PLAN_FILES if p not in OPTIONAL_PLANS)


def contracts_dir(target_repo: str | Path) -> Path:
    """Return the ``contracts/`` directory inside a generated target repo."""
    return Path(target_repo) / "contracts"


def load_plan(target_repo: str | Path, plan: str) -> Any:
    """Load a single plan (by stem, e.g. ``"target_contract"``) from a target repo.

    Raises ``KeyError`` for an unknown plan name and ``FileNotFoundError`` if the file is
    absent.
    """
    if plan not in PLAN_FILES:
        raise KeyError(f"unknown plan '{plan}'; known: {sorted(PLAN_FILES)}")
    filename, _ = PLAN_FILES[plan]
    path = contracts_dir(target_repo) / filename
    if not path.is_file():
        raise FileNotFoundError(f"plan not found: {path}")
    return load_yaml(path)


def load_all_plans(target_repo: str | Path) -> dict[str, Any]:
    """Load every plan present in a target repo's ``contracts/`` dir.

    Missing files are skipped (the caller can detect them by absence). Returns a mapping of
    plan stem -> parsed YAML.
    """
    out: dict[str, Any] = {}
    for plan in PLAN_FILES:
        try:
            out[plan] = load_plan(target_repo, plan)
        except FileNotFoundError:
            continue
    return out


def validate_plan(obj: Any, plan: str) -> list[str]:
    """Validate one loaded plan object against its schema. Empty list == valid."""
    if plan not in PLAN_FILES:
        return [f"unknown plan '{plan}'"]
    _, schema_name = PLAN_FILES[plan]
    return [f"{plan}: {p}" for p in schemas.validate(obj, schema_name)]


def validate_target_repo(target_repo: str | Path) -> list[str]:
    """Load and validate all five plans in a target repo.

    Returns a flat list of diagnostics (empty == fully valid). A missing REQUIRED plan file is itself
    a diagnostic; an absent OPTIONAL plan (e.g. dialect_plan) is not — it is validated only when present.
    """
    problems: list[str] = []
    for plan in PLAN_FILES:
        try:
            obj = load_plan(target_repo, plan)
        except FileNotFoundError as exc:
            if plan not in OPTIONAL_PLANS:
                problems.append(f"{plan}: missing ({exc})")
            continue
        problems.extend(validate_plan(obj, plan))
    return problems


PLAN_SCHEMAS: dict[str, str] = {name: schema for name, (_, schema) in PLAN_FILES.items()}


def validate_plans(plans: dict[str, Any]) -> list[str]:
    """Validate every plan in ``plans`` against its schema. Empty list == valid."""
    problems: list[str] = []
    for key, obj in plans.items():
        schema_name = PLAN_SCHEMAS.get(key)
        if schema_name is None:
            problems.append(f"{key}: no schema mapping")
            continue
        problems.extend(f"{key}: {p}" for p in schemas.validate(obj, schema_name))
    return problems


REQUIRED_TARGET_PATHS: list[str] = [
    "AGENT.md",
    "contracts/target_contract.yaml",
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


def render_validation_report(
    target: str,
    plans: dict[str, Any],
    schema_problems: list[str],
    emit: list[str],
) -> str:
    """Render a human-readable validation report summarizing plan validity + review flags."""
    lines: list[str] = []
    lines.append(f"# Validation report — {target}")
    lines.append("")
    status = "PASS" if not schema_problems else "FAIL"
    lines.append(f"- schema validation: **{status}**")
    lines.append(f"- emitted layers: {', '.join(emit) if emit else '(none)'}")
    lines.append("")
    lines.append("## Plans")
    lines.append("")
    lines.append("| plan | confidence | requires_human_review |")
    lines.append("| --- | --- | --- |")
    for key, obj in plans.items():
        conf = obj.get("confidence", "?") if isinstance(obj, dict) else "?"
        rev = obj.get("requires_human_review", "?") if isinstance(obj, dict) else "?"
        lines.append(f"| {key} | {conf} | {rev} |")
    lines.append("")
    if schema_problems:
        lines.append("## Schema problems")
        lines.append("")
        for p in schema_problems:
            lines.append(f"- {p}")
        lines.append("")
    any_review = any(isinstance(o, dict) and o.get("requires_human_review") for o in plans.values())
    if any_review:
        lines.append("## Note")
        lines.append("")
        lines.append(
            "One or more plans are flagged `requires_human_review: true`. TargetGen does not "
            "claim these are correct; review against the evidence report before relying on them."
        )
        lines.append("")
    return "\n".join(lines)
