"""Command-graph / control-interface analysis — honest about what a flat capture can show.

A DSE engine wants to know how many commands run per replan, which form a repeated subgraph, where
the syncs and dependencies are, and what allocations happen inside the loop — to decide whether to
expose a command buffer, a bounded device-side loop, or event-token dependency tracking.

**Honest limit, stated up front:** `torch.export` unrolls the host loop, so a true per-step
dependency / sync / allocation graph is NOT recoverable from the flat capture. What this module can
ground: the matmul-count *proxy* for commands per step (a lower bound — the measured dispatch leg
shows real dispatch granularity is ~12-14× higher, see `dispatch_coupling_report.md`), the
dispatches-per-replan proxy scaled by K, and the structural fact that the K-step head is a repeated
subgraph. Everything the capture erases is emitted as `unavailable`. No speedup is claimed.
"""
from __future__ import annotations

from merlin.dse_guidance.design_envelope import E_DERIVED, E_IR, E_NA

# The HW/SW abstractions a repeated, batchable command subgraph implies.
_IMPLIED_ABSTRACTIONS = ["command_buffer", "bounded_loop_command + loop_carried_state_handle",
                         "persistent_command_buffer / region_level_dispatch", "event_token"]


def command_graph(topo, attribution) -> dict:
    head = attribution.role("repeated_head") if attribution else None
    head_ok = head is not None and head.attribution_status == "attributed"
    mm = int(head.facts.get("matmul_count", 0)) if head_ok else 0
    K = int(topo.K)
    return {
        "workload": topo.workload,
        "commands_per_step_matmul_proxy": {
            "value": mm, "evidence": E_IR,
            "note": "matmul count of the repeated head — a LOWER BOUND on commands/step; measured "
                    "dispatch granularity is ~12-14x higher (dispatch_coupling_report.md)"},
        "dispatches_per_replan_proxy": {
            "value": mm * K, "evidence": E_DERIVED,
            "note": "commands_per_step_matmul_proxy * K (K is assumed_reference)"},
        "repeated_subgraph": {
            "region": "repeated_head", "matmuls": mm, "reused_times": K,
            "evidence": E_IR, "note": "the K-step head is a repeated, batchable subgraph"},
        "batchable": {"value": bool(head_ok), "evidence": E_IR,
                      "note": "the repeated head is a static subgraph; batchable into one buffer "
                              "IF the runtime exposes loop-carried state + dependency tracking"},
        "syncs_per_step": {"value": None, "evidence": E_NA,
                           "note": "host loop unrolled by torch.export; sync points not recoverable"},
        "dependency_graph": {"value": None, "evidence": E_NA,
                             "note": "per-step dependencies erased by loop unrolling"},
        "allocations_in_loop": {"value": None, "evidence": E_NA,
                                "note": "in-loop allocations not recoverable from the flat capture"},
        "implied_abstractions": _IMPLIED_ABSTRACTIONS,
        "what_is_not_claimed": ["speedup", "cycle_reduction", "real dispatch/sync latency"],
    }


def dispatch_granularity_csv(packages) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for p in packages:
        g = p.get("cmd")
        if not g:
            continue
        rows.append({
            "workload": g["workload"],
            "commands_per_step_matmul_proxy": g["commands_per_step_matmul_proxy"]["value"],
            "dispatches_per_replan_proxy": g["dispatches_per_replan_proxy"]["value"],
            "repeated_subgraph": g["repeated_subgraph"]["region"],
            "batchable": g["batchable"]["value"],
            "syncs_per_step": "unavailable",
            "dependency_graph": "unavailable",
        })
    return _csv(rows, ["workload", "commands_per_step_matmul_proxy", "dispatches_per_replan_proxy",
                       "repeated_subgraph", "batchable", "syncs_per_step", "dependency_graph"])


def to_yaml_obj(graph: dict) -> dict:
    return {"command_graph": graph}
