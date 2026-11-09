"""Describe one selected grouping before compiling the model.

These reports describe source grouping and its outlined dispatch representation.
They do not establish the final image's placement, runtime synchronization, or
observed execution. Those claims remain with the build and independent evaluator.
"""

from __future__ import annotations

from .groups import GroupPlan


def group_reports(grouping: GroupPlan, offload=None) -> dict:
    """Reuse the selected groups and, when present, their actual outlined module."""
    reports = {}
    try:
        report = grouping.report()
        reports["compute_groups"] = {
            key: report[key] for key in ("schema", "summary", "absorption_refusals", "groups")
        }
    except Exception as error:  # noqa: BLE001 -- diagnostics do not authorize or block emission
        reports["compute_groups"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        from merlin.xdsl_dialects.lowering import stream_plan
        from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
        from merlin.xdsl_dialects.lowering.outline import OutlineResult, outline_dispatches

        grouping.require_current(grouping.source, grouping.target, weight_args=grouping.weight_args)
        if offload is not None and offload.module is not None:
            offload.require_grouping(grouping)
            outlined = OutlineResult(module=offload.module, dispatches=list(offload.dispatches))
        else:
            outlined = outline_dispatches(grouping.source, groups=grouping.groups)
        report = stream_plan.plan(
            build_dispatch_program(outlined), outlined.dispatches, weight_args=grouping.weight_args or ()
        )
        report["stored_argument_binding"] = {
            "status": "selected_manifest" if grouping.weight_args is not None else "unknown",
            "argument_indices": sorted(grouping.weight_args or ()),
        }
        reports["stream_plan"] = report
    except Exception as error:  # noqa: BLE001 -- unavailable analysis remains unavailable
        reports["stream_plan"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        from merlin.targetgen import group_capsule_entries

        grouping.require_current(grouping.source, grouping.target, weight_args=grouping.weight_args)
        report = group_capsule_entries.entries(
            grouping.target, grouping.source, weight_args=grouping.weight_args,
            with_raw=False, oracle=grouping.oracle, groups=grouping.groups,
        )
        reports["group_programs"] = {
            key: report[key] for key in ("schema", "accelerator_groups", "stated", "distinct", "unstated", "entries")
        }
    except Exception as error:  # noqa: BLE001 -- no fabricated capsule or acceptance claim
        reports["group_programs"] = {"error": f"{type(error).__name__}: {error}"}
    return reports
