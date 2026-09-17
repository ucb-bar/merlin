"""Validation layer: check synthesized plans + the generated repo, render a report."""

from __future__ import annotations

from .generated_repo import check_generated_target
from .report import render_validation_report
from .schemas import validate_plans

__all__ = ["validate_plans", "check_generated_target", "render_validation_report"]
