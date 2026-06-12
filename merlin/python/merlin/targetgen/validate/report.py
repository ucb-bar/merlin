"""Render docs/validation_report.md for a generated target."""
from __future__ import annotations

from typing import Any


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
    any_review = any(
        isinstance(o, dict) and o.get("requires_human_review") for o in plans.values()
    )
    if any_review:
        lines.append("## Note")
        lines.append("")
        lines.append(
            "One or more plans are flagged `requires_human_review: true`. TargetGen does not "
            "claim these are correct; review against the evidence report before relying on them."
        )
        lines.append("")
    return "\n".join(lines)
