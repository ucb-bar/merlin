"""Coarse original-host profile projections; counter ticks are not performance scores."""

import json


def group_profile(record, console):
    """Coarse original-host diagnostics, never a functional-tick performance score."""
    if not record["build"]["inputs"].get("group_profile", False):
        return None
    from merlin.runtime.out_bin import binary_console_diagnostics
    from merlin.runtime.whole_model_readback import ReadbackError, parse_group_profile

    base = {"unit": "counter_ticks", "timing_qualified": False,
            "performance_ranking_eligible": False, "instrumented": True}
    try:
        profile = parse_group_profile(binary_console_diagnostics(console).decode("utf-8", errors="replace"))
        if profile is None or profile["dropped"]:
            raise ValueError("missing or incomplete group observations")
    except (ReadbackError, ValueError) as exc:
        return {**base, "status": "unavailable", "detail": str(exc)}
    groups = profile["groups"]
    grouped = sum(row["cycles"] for row in groups)
    between = profile["tail_gap"] + sum(row["gap_before"] for row in groups)
    return {**base, "status": "observed", "executions": profile["calls"],
            "group_counter_ticks": grouped, "between_group_counter_ticks": between,
            "total_counter_ticks": grouped + between,
            "largest_groups": [{"index": row["index"], "counter_ticks": row["cycles"]}
                               for row in sorted(groups, key=lambda row: row["cycles"], reverse=True)[:8]],
            "scope": "Instrumented forward groups and gaps; gaps include orchestration and profiler overhead. "
                     "Group membership does not itself establish CPU/device placement. "
                     "Compare an uninstrumented control."}


def operation_profile(record, console, *, selected_file):
    """Compact optional diagnostics; never turn functional ticks into cycles."""
    from merlin.runtime.operation_profile import OperationProfileError, summarize
    from merlin.runtime.out_bin import binary_console_diagnostics

    build = record["build"]
    if not build["inputs"].get("op_profile", False):
        return None
    table = selected_file(build["output"], "op_profile_table")
    payload = json.loads(table.read_text())
    text = binary_console_diagnostics(console).decode("utf-8", errors="replace")
    try:
        profile = summarize(payload, text)
    except OperationProfileError as exc:
        # Missing diagnostic observations do not erase independently checked
        # numerical results. Changed table identities still refuse above.
        return {"status": "unavailable", "detail": str(exc), "timing_qualified": False}
    return {
        "status": "observed",
        **{key: profile[key] for key in (
            "schema", "unit", "timing_qualified", "executions", "category_counter_ticks",
            "operation_counter_ticks", "inter_invocation_counter_ticks", "total_counter_ticks",
            "marks", "limitations",
        )},
        "table": build["output"]["op_profile_table"],
        "instrumented": True,
        "performance_ranking_eligible": False,
        "scope": "Instrumented operation intervals; functional Spike counters are not elapsed cycles. "
        "Compare an uninstrumented control and independently bind the timing engine.",
    }
