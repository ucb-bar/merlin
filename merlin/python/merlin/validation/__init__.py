"""Structural/artifact validation for generated target repos + contract plans.

Reusable predicates and loaders so build_tools check scripts + tests stay declarative:
- ``generated_target`` — does a generated target repo have its required files?
- ``load`` / ``validate`` — load and validate the target-contract plan artifacts
  (``target_contract.yaml``, ``dialect_plan.yaml``, …), merged here from the former
  ``merlin.contracts``. Pure stdlib + :mod:`merlin.common`.
"""
from __future__ import annotations

from .generated_target import check_generated_target, REQUIRED_TARGET_PATHS
from .load import PLAN_FILES, load_all_plans
from .validate import validate_plan, validate_target_repo

__all__ = [
    "check_generated_target", "REQUIRED_TARGET_PATHS",
    "PLAN_FILES", "load_all_plans", "validate_plan", "validate_target_repo",
]
