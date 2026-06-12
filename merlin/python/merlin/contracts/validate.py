"""Validate the five plan artifacts against the shared schemas.

Thin layer over :mod:`merlin.common.schemas`: it checks required top-level keys and returns
readable, path-prefixed diagnostics. This is intentionally lightweight (the schemas are not
formal JSON Schema yet) but enough to keep the cross-workstream artifacts honest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import schemas
from .load import PLAN_FILES, load_plan


def validate_plan(obj: Any, plan: str) -> list[str]:
    """Validate one loaded plan object against its schema. Empty list == valid."""
    if plan not in PLAN_FILES:
        return [f"unknown plan '{plan}'"]
    _, schema_name = PLAN_FILES[plan]
    return [f"{plan}: {p}" for p in schemas.validate(obj, schema_name)]


def validate_target_repo(target_repo: str | Path) -> list[str]:
    """Load and validate all five plans in a target repo.

    Returns a flat list of diagnostics (empty == fully valid). A missing plan file is itself
    a diagnostic.
    """
    problems: list[str] = []
    for plan in PLAN_FILES:
        try:
            obj = load_plan(target_repo, plan)
        except FileNotFoundError as exc:
            problems.append(f"{plan}: missing ({exc})")
            continue
        problems.extend(validate_plan(obj, plan))
    return problems
