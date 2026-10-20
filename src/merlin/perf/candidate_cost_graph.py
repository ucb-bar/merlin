"""Candidate-owned work graph and conditional arithmetic, never price authority."""

from __future__ import annotations

import math

from merlin.perf.activity_schedule import ActivityEvent, ActivityTimeline, _union_length, schedule_activity
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

SCHEMA = "merlin.candidate_cost_graph.v1"


def observe_candidate_cost_graph(diagnostics, *, command_buffer=None):
    """Retain complete original ownership, or the exact missing terms.

    Source tensor dependencies are value dependencies. They do not specify a
    hardware service interval, transfer multiplicity or permission to overlap.
    Host categories describe actual emitted CFG, not inferred source placement.
    """
    joined = diagnostics.get("source_emission_join", {})
    binding = diagnostics.get("dependency_stages", {})
    missing = []
    if joined.get("status") != "verified" or binding.get("status") != "verified_source_emission":
        missing.append("complete original source-to-emission task ownership")
    source = diagnostics.get("source", {})
    source_ops = {row["source_op_index"]: row for row in source.get("operations", [])
                  if row.get("source_op_index") is not None}
    owner = {}
    for task in binding.get("stages", []):
        for index in task["source_op_indices"]:
            if index in owner:
                missing.append("duplicate source operation ownership")
            owner[index] = task["task_index"]
    ordinal_owner = {row["ordinal"]: owner[index] for index, row in source_ops.items() if index in owner}
    host = joined.get("host_activity", {})
    if host.get("status") != "derived" or host.get("task_activity_coverage") != "complete":
        missing.append("complete emitted host CFG activity")
    host_rows = {str(row["task"]): row for row in host.get("tasks", [])}
    stages = []
    for task in binding.get("stages", []):
        ident = task["task_index"]
        owned = [source_ops[index] for index in task["source_op_indices"] if index in source_ops]
        if len(owned) != len(task["source_op_indices"]):
            missing.append(f"source operation metadata for task:{ident}")
        dependencies = sorted({f"task:{ordinal_owner[dep]}" for op in owned for dep in op.get("dependencies", [])
                               if dep in ordinal_owner and ordinal_owner[dep] != ident})
        activity = host_rows.get(str(ident))
        if activity is None or activity.get("dynamic_operations") is None:
            missing.append(f"emitted host multiplicity for task:{ident}")
        operations = (activity or {}).get("static_operations") or {}
        if operations.get("opaque_inline_asm", 0) or operations.get("other", 0):
            missing.append(f"opaque calls or assembly or unsupported emitted effects for task:{ident}")
        stages.append({"id": f"task:{ident}", "depends_on": dependencies,
                       "source_op_indices": task["source_op_indices"], "source_operations": owned,
                       "host_activity": activity, "tensor_reads": task["reads"], "tensor_writes": task["writes"],
                       "emission_range": [task["instruction_start"], task["instruction_end"]],
                       "execution_lane": "UNKNOWN", "service_interval": "UNKNOWN",
                       "host_effects": {key: (activity or {}).get(key) for key in
                                        ("dynamic_operations", "load_payload_bytes", "store_payload_bytes")}})
    unowned = host_rows.get("unowned")
    if unowned and any(unowned.get("static_operations", {}).values()):
        missing.append("unowned emitted host instructions")
    cb = command_buffer or {}
    tensors = cb.get("tensors") or {}
    requests = []
    for index, command in enumerate(cb.get("commands") or []):
        if not isinstance(command, dict):
            missing.append("malformed command declaration")
            continue
        requests.append({"command_index": index, "opcode": command.get("opcode"),
                         "operands": {name: {"tensor": value, "shape": tensors.get(value, {}).get("shape"),
                                             "dtype": tensors.get(value, {}).get("dtype")}
                                      for name, value in (command.get("operands") or {}).items()
                                      if isinstance(value, str)},
                         "attributes": {key: value for key, value in (command.get("attributes") or {}).items()
                                        if key in {"shape", "rows", "cols", "m", "k", "n", "stride", "layout"}},
                         "executed_multiplicity": None, "task_owner": "UNKNOWN",
                         "physical_bytes": None, "service_cycles": None})
    if requests:
        missing.append("issued device, transfer and synchronization effects and their exact task ownership")
    return {"schema": SCHEMA, "scope": "emitted candidate kernel work only", "authority": "none",
            "status": "observed", "stages": stages, "unowned_host_activity": unowned,
            "requested_commands": requests, "representation_transitions": diagnostics.get("representations", {}),
            "provenance": {key: joined.get(key) for key in
                           ("candidate_sha256", "source_sha256", "candidate_lowered_sha256",
                            "candidate_command_buffer_sha256", "plan_digest")},
            "completeness": {"status": "complete" if not missing and stages else "UNKNOWN",
                             "missing": sorted(set(missing)) or ([] if stages else ["no complete owned stages"])},
            "excluded_costs": ["harness startup and input preparation outside kernel",
                               "parent readback and publication", "cold and warm state"],
            "composition": "conditional serial premise; overlap requires independent observed intervals"}


def compose_candidate_cost_graph(graph, *, stage_prices, observed_timeline=None):
    """Pure conditional CycleInterval arithmetic over externally qualified prices.

    This function does not qualify any price or timeline. The caller must bind
    them to the exact graph/source/configuration and independently measured
    clock/effect domain. Serial addition is a selected hypothetical premise,
    not a claim that the actual hardware cannot overlap. Excluded graph costs
    never become a complete program objective through this computation.
    """
    if graph.get("schema") != SCHEMA or graph.get("completeness", {}).get("status") != "complete":
        return CycleInterval.unknown(*(graph.get("completeness", {}).get("missing") or ["incomplete graph ownership"]))
    stages = graph["stages"]
    ids = [row["id"] for row in stages]
    if not stages or len(ids) != len(set(ids)) or set(stage_prices) != set(ids):
        return CycleInterval.unknown("missing, duplicate or extra stage prices")
    if any(type(price) is not CycleInterval for price in stage_prices.values()):
        raise TypeError("stage prices require exact CycleInterval values")
    unresolved = [ident for ident, price in stage_prices.items() if not price.resolved]
    if unresolved:
        return CycleInterval.unknown(*(f"unpriced {ident}" for ident in unresolved))
    seen = set()
    for row in stages:
        if row["id"] in row["depends_on"] or set(row["depends_on"]) - seen:
            return CycleInterval.unknown("absent, cyclic or non-topological stage dependency")
        seen.add(row["id"])
    provenance = tuple(value for price in stage_prices.values() for value in price.provenance)
    if observed_timeline is None:
        endpoints = []
        for endpoint in ("lo", "hi"):
            events = [ActivityEvent(row["id"], row["id"], "priced_stage", getattr(stage_prices[row["id"]], endpoint),
                                    depends_on=tuple(row["depends_on"]), serial_group="conditional_serial",
                                    provenance="externally qualified stage-price premise") for row in stages]
            endpoints.append(schedule_activity(events).total_cycles)
        return CycleInterval(*endpoints, provenance=provenance + ("conditional serial stage addition",))
    if type(observed_timeline) is not ActivityTimeline:
        raise TypeError("overlap requires an exact externally observed ActivityTimeline")
    scheduled = {row.event.id: row for row in observed_timeline.events}
    if len(scheduled) != len(observed_timeline.events) or set(scheduled) != set(ids):
        return CycleInterval.unknown("observed timeline omitted or duplicated stage effects")
    for row in stages:
        price, event = stage_prices[row["id"]], scheduled[row["id"]]
        if (price.lo != price.hi or not event.event.provenance or event.event.cycles != price.lo
                or any(not math.isfinite(value) for value in (event.start, event.end, event.event.cycles))
                or event.start < 0 or event.end - event.start != event.event.cycles
                or any(scheduled[dep].end > event.start for dep in row["depends_on"])):
            return CycleInterval.unknown("observed intervals do not cover exact stage prices and dependencies")
    if (min(row.start for row in scheduled.values()) != 0
            or max(row.end for row in scheduled.values()) != observed_timeline.total_cycles
            or _union_length([(row.start, row.end) for row in scheduled.values()]) != observed_timeline.total_cycles):
        return CycleInterval.unknown("unpriced prefix or suffix or changed observed total interval")
    return CycleInterval.point(observed_timeline.total_cycles,
                               provenance="conditional exact independently observed stage timeline")
