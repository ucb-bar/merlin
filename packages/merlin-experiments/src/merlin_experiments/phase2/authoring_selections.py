"""Keep optional original-host selections within their ordinary workflow."""

from . import broker_policy as BP
from .contracts import StageGateError


def admit_workflow(workflow_id: str, *, component_launch=None) -> None:
    """Keep workflow selection distinct from fresh-session launch authority."""
    if workflow_id == BP.COMPONENT_ONLY_V1:
        if component_launch is not None:
            from .component_launch import QualifiedComponentLaunch  # noqa: PLC0415

            if type(component_launch) is not QualifiedComponentLaunch:
                raise StageGateError("component authoring requires independently evaluated typed launch prerequisites")
            component_launch.verify()
            return
        raise StageGateError(
            "component-only authoring is unavailable: approved minimal view, verified runtime "
            "isolation and zero-history session transport are required; the broker profile "
            "does not grant launch authority"
        )
    if workflow_id != BP.CORPUS_FEEDBACK_V1:
        raise StageGateError(f"unsupported authoring workflow: {workflow_id!r}")


def validate(workflow_id, search_configuration, compiler_library_selection, archive_screen_selection):
    if compiler_library_selection is not None and workflow_id != BP.CORPUS_FEEDBACK_V1:
        raise StageGateError("additional compiler library requires the ordinary authoring workflow")
    if archive_screen_selection is not None and (workflow_id != BP.CORPUS_FEEDBACK_V1 or search_configuration is None):
        raise StageGateError("archive screening requires the original ordinary queued authoring workflow")


def admit_authoring_workflow(workflow_id: str, *, component_launch=None) -> None:
    """Keep the broker action profile distinct from fresh-session launch authority."""
    if workflow_id == BP.COMPONENT_ONLY_V1:
        if component_launch is not None:
            from .component_launch import QualifiedComponentLaunch  # noqa: PLC0415

            if type(component_launch) is not QualifiedComponentLaunch:
                raise StageGateError("component authoring requires independently evaluated typed launch prerequisites")
            component_launch.verify()
            return
        raise StageGateError(
            "component-only authoring is unavailable: approved minimal view, verified runtime "
            "isolation and zero-history session transport are required; the broker profile "
            "does not grant launch authority"
        )
    if workflow_id != BP.CORPUS_FEEDBACK_V1:
        raise StageGateError(f"unsupported authoring workflow: {workflow_id!r}")


def validate_stage_budgets(suite, model, budgets, *, tool_timeout_seconds, iteration_max_seconds):
    """Validate the original stage limits before any executable or workspace lookup."""
    if not isinstance(suite, str) or not suite.strip():
        raise StageGateError("an explicit telemetry suite is required")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in budgets):
        raise StageGateError("all performance stage budgets must be positive integers")
    if tool_timeout_seconds > iteration_max_seconds:
        raise StageGateError(
            f"tool_timeout_seconds exceeds the {iteration_max_seconds:g}s reduced-witness "
            "iteration limit; reduce the witness"
        )
    if not model.strip():
        raise StageGateError("an explicit Codex model is required")
