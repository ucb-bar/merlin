"""Validation layer: check synthesized plans + the generated repo, render a report."""

from __future__ import annotations

from merlin.development.targetgen.validate.generated_repo import check_generated_target
from merlin.development.targetgen.validate.report import render_validation_report
from merlin.development.targetgen.validate.schemas import validate_plans

__all__ = ["validate_plans", "check_generated_target", "render_validation_report"]
