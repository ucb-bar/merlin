"""Structural/artifact validation helpers shared by the build_tools check scripts.

This package holds reusable predicates (does a generated target repo have its required
files? are the required schemas present and non-empty?) so the thin scripts under
``build_tools/scripts/`` stay declarative. Pure stdlib + :mod:`merlin.common`.
"""
from __future__ import annotations

from .generated_target import check_generated_target, REQUIRED_TARGET_PATHS

__all__ = ["check_generated_target", "REQUIRED_TARGET_PATHS"]
