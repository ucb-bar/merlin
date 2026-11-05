"""Instructions for the final report-only authoring turn."""

from __future__ import annotations


def finalization_instructions(model_report: bool, pass_line: str) -> str:
    return (
        (
            "All six visible programs pass; the sealed official grade has not run yet.\n\n"
            if model_report
            else "All required public/dev pilot capsules now PASS (see qa/verdict.json: all_pass=true).\n\n"
        )
        + "Do ONLY this, then stop:\n"
        + (
            "1. Lead `submission/REPORT.md` with a table of the six programs and their visible statuses. "
            "Then make its final status line read EXACTLY:\n"
            if model_report
            else "1. Update `submission/REPORT.md` so its final status line reads EXACTLY:\n"
        )
        + f'   "{pass_line}"\n'
        + (
            "   State only measured visible results; do not claim sealed results.\n"
            if model_report
            else "   and its body honestly reflects the verified per-capsule pass + oracle tiers.\n"
        )
        + "2. Update `ITERATION_NOTES.md` (workspace root) with the final converged state.\n"
        + "DO NOT modify manifest.yaml, mlir_oot/, or any code — the package is frozen-pending and must\n"
        + "keep passing. Touch only REPORT.md, docs/ and ITERATION_NOTES.md.\n"
    )
