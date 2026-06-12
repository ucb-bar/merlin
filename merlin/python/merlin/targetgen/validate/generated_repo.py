"""Structural validation of a generated target repo.

Re-exports :func:`merlin.validation.generated_target.check_generated_target` so the TargetGen
``inspect`` command and the build_tools script share one implementation.
"""
from __future__ import annotations

from ...validation.generated_target import check_generated_target

__all__ = ["check_generated_target"]
