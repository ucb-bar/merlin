"""Structural/artifact validation for generated target repos + contract plans.

The implementation lives in
``merlin.development.targetgen.generation.generated_target``. This package retains
the existing public validation exports; generation and inspection use that same
read-only owner. These checks do not establish numerical or hardware correctness.
"""

from __future__ import annotations

from merlin.development.targetgen.generation.generated_target import (
    REQUIRED_TARGET_PATHS,
    check_generated_target,
    PLAN_FILES,
    load_all_plans,
    validate_plan,
    validate_target_repo,
)

__all__ = [
    "check_generated_target",
    "REQUIRED_TARGET_PATHS",
    "PLAN_FILES",
    "load_all_plans",
    "validate_plan",
    "validate_target_repo",
]
