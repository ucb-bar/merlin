"""Closed single-function operation observations, without assigning counter units.

The existing marker runtime credits each interval to the previous marker. A
sentinel closes the last operation and accumulates gaps BETWEEN invocations.
Those gaps are recorded separately and never assigned to an operation category.
This reader establishes trace consistency, not timing or placement authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from merlin.llvmlower.op_profile import resolve_category


class OperationProfileError(ValueError):
    """An operation trace is incomplete, inconsistent or outside the selected scope."""


def summarize(table: Sequence[Mapping], console: str) -> dict:
    """Join one typed, single-function table to a complete marker-runtime dump.

    All operations must execute equally often. The runtime must print each
    observation exactly once, with matching total/mark counters and terminal
    DONE. Nested multi-function tables, overflow and partial traces refuse.
    Hit consistency does not establish nonrecursive ownership. ``counter_ticks``
    are raw observations: there is no Hz, time or
    accelerator-cycle conversion. Semantic categories retain their derivation;
    neither category nor callee name establishes CPU/device ownership.
    """
    if not table or any(
        not isinstance(row, Mapping)
        or type(row.get("id")) is not int
        or row["id"] != i
        or not isinstance(row.get("mlir_op"), str)
        or not row["mlir_op"]
        or "function" in row
        for i, row in enumerate(table)
    ):
        raise OperationProfileError("expected a nonempty contiguous single-function operation table")
    observations: dict[int, tuple[int, int]] = {}
    metrics: dict[str, int] = {}
    done = False
    for line in console.splitlines():
        parts = line.split()
        if not parts:
            continue
        selected = parts[0] == "PROF" or (len(parts) > 1 and parts[:2] in (
            ["METRIC", "prof_total_ticks"], ["METRIC", "prof_marks"],
        ))
        if done and (selected or parts[0] == "DONE"):
            raise OperationProfileError("operation observations after terminal DONE")
        if parts[0] == "DONE":
            if len(parts) != 1:
                raise OperationProfileError("malformed terminal DONE")
            done = True
        elif parts[0] == "PROF":
            if len(parts) != 4:
                raise OperationProfileError("malformed PROF observation")
            try:
                index, ticks, hits = map(int, parts[1:])
            except ValueError as exc:
                raise OperationProfileError("noninteger PROF observation") from exc
            if index in observations:
                raise OperationProfileError("duplicate PROF observation")
            if not 0 <= index <= len(table) or not 0 <= ticks < 2**64 or not 0 < hits < 2**32:
                raise OperationProfileError("overflow, foreign id or invalid PROF counter")
            observations[index] = ticks, hits
        elif selected:
            if len(parts) != 3 or parts[1] in metrics:
                raise OperationProfileError("malformed or duplicate profiler metric")
            try:
                value = int(parts[2])
            except ValueError as exc:
                raise OperationProfileError("noninteger profiler metric") from exc
            if not 0 <= value < 2**64:
                raise OperationProfileError("invalid profiler metric")
            metrics[parts[1]] = value
    if not done or set(metrics) != {"prof_marks", "prof_total_ticks"}:
        raise OperationProfileError("missing terminal DONE or profiler metrics")
    if not set(range(len(table))) <= observations.keys():
        raise OperationProfileError("missing operation observations")
    executions = observations[0][1]
    if any(observations[i][1] != executions for i in range(len(table))):
        raise OperationProfileError("inconsistent operation execution counts")
    gap_ticks, gap_hits = observations.get(len(table), (0, 0))
    if gap_hits != executions - 1:
        raise OperationProfileError("inconsistent sentinel inter-invocation gaps")
    total = sum(ticks for ticks, _hits in observations.values())
    if metrics["prof_marks"] != executions * (len(table) + 1) or metrics["prof_total_ticks"] != total:
        raise OperationProfileError("profiler total or marker count mismatch")
    operations, categories = [], {}
    for record in table:
        ticks, hits = observations[record["id"]]
        category, source = resolve_category(record)
        operations.append({
            **record, "counter_ticks": ticks, "hits": hits,
            "category": category, "category_source": source, "execution_owner": "unknown",
        })
        categories[category] = categories.get(category, 0) + ticks
    return {
        "schema": "merlin.operation_profile.v1",
        "unit": "raw_counter_ticks",
        "timing_qualified": False,
        "executions": executions,
        "operations": operations,
        "category_counter_ticks": dict(sorted(categories.items())),
        "operation_counter_ticks": total - gap_ticks,
        "inter_invocation_counter_ticks": gap_ticks,
        "total_counter_ticks": total,
        "marks": metrics["prof_marks"],
        "limitations": [
            "Markers may change optimization, allocation and scheduling; pair an uninstrumented control.",
            "Categories describe source semantics, not CPU/device placement or instruction effects.",
            "Nested packing, copies, dispatch, allocator and accelerator work are not separately resolved.",
            "Marker bookkeeping, entry/return, startup, output and final dump are outside operation intervals.",
            "Raw ticks establish no counter calibration, hardware latency or numerical agreement.",
        ],
    }
