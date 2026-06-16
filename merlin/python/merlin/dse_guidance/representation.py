"""Flat vs multi-rate representation of one workload.

A flat capture collapses the K-step action/denoise loop to a single pass: weight reuse looks
like 1, there is no visible K-loop, and the replan deadline is invisible. The multi-rate
representation re-exposes the temporal structure: loop-invariant weights are reused K times,
the K submits are visible, and the deadline is in view.

The whole point of the workstream is that this difference *changes the DSE recommendation*.
We therefore build both representations from one region + temporal metadata and let the diff
and the axis triage be computed under each.

Structural facts (immutable loop-invariant weights, epilogue presence, dispatch counts) come
from the region pressure vector (:func:`merlin.design_pressure.pressure_vector.compute_rpv`)
when a region is available, and from the temporal metadata's ``loop_invariant_state`` /
``loop_carried_state`` otherwise. The region is optional: a headline VLA action head may be
described purely by its temporal metadata + measured cost breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.design_pressure import region as R
from merlin.dse_guidance.temporal import TemporalMetadata

# The abstraction-exposure axes whose recommendation flips between flat and multi-rate.
# (The hardware-scaling axes — PE_count_2x etc. — are always legal knobs and are ranked
# quantitatively by the triage rather than recommended/deprioritized structurally here.)
_ABSTRACTION_AXES = (
    "resident_packed_weights",
    "resident_prefix_kv",
    "command_batching",
    "autonomous_K_loop",
    "accumulator_commit",
    "event_tokens",
)

COULD_BE_WRONG_IF: dict[str, list[str]] = {
    "resident_packed_weights": [
        "weights do not fit resident capacity",
        "packing is already hoisted",
        "DMA/memory traffic is not a limiting component",
        "host dispatch dominates total latency anyway",
    ],
    "resident_prefix_kv": [
        "prefix/KV does not fit resident capacity",
        "prefix/KV is cheap to recompute",
        "KV traffic is not a limiting component",
    ],
    "command_batching": [
        "host dispatch is not on the critical path",
        "batching raises per-submit latency",
        "the runtime already coalesces submits",
    ],
    "autonomous_K_loop": [
        "K is data-dependent / unbounded",
        "per-step host overhead is already negligible",
        "on-device control costs area not justified by the saving",
    ],
    "accumulator_commit": [
        "intermediate materialization is already fused",
        "the epilogue is memory-bound elsewhere",
        "the accumulator does not fit",
    ],
    "event_tokens": [
        "the dependency chain is already overlapped",
        "sync wait is not a limiting component",
    ],
}


@dataclass
class Representation:
    name: str                               # "flat" | "multirate"
    workload: str
    K: int
    H: int | None
    control_rate_hz: float | None
    replan_deadline_ms: float | None
    deadline_visible: bool
    visible_weight_reuse: int
    visible_prefix_kv_reuse: int
    dispatches_per_step: int
    dispatches_per_replan: int
    work_per_dispatch: float | None
    facts: dict = field(default_factory=dict)
    recommended_axes: list[dict] = field(default_factory=list)
    deprioritized_axes: list[dict] = field(default_factory=list)

    @property
    def recommended_axis_names(self) -> list[str]:
        return [a["axis"] for a in self.recommended_axes]


def _base_facts(temporal: TemporalMetadata, region: dict | None) -> dict:
    """Structure shared by both representations, before applying the flat/multirate view."""
    loop_inv = {s.lower() for s in temporal.loop_invariant_state()}
    carried = {s.lower() for r in temporal.regions for s in r.loop_carried_state}

    weights_in_loop_inv = any("weight" in s for s in loop_inv)
    prefix_kv_in_loop_inv = any(("prefix" in s) or ("kv" in s) for s in loop_inv)

    macs = None
    dispatches_per_step = 1
    has_epilogue = False
    weights_immutable = weights_in_loop_inv  # loop-invariant state is, by definition, not mutated
    region_reuse = 1
    dram_reducible_fraction = None  # fraction of DRAM traffic removable by residency (region-derived)

    if region is not None:
        from merlin.design_pressure.pressure_vector import compute_rpv
        rpv = compute_rpv(region)
        m = rpv["metrics"]
        macs = int(m.get("macs", 0)) or None
        op_seq = R.op_sequence(region)
        dispatches_per_step = max(len(op_seq), 1)
        has_epilogue = R.has_epilogue(region)
        weights_immutable = (not R.rhs_mutable(region)) or weights_in_loop_inv
        region_reuse = R.rhs_reuse_count(region)
        base_traffic = float(m.get("dram_traffic_bytes_baseline", 0) or 0)
        resident_traffic = float(m.get("dram_traffic_bytes_resident", 0) or 0)
        if base_traffic > 0:
            dram_reducible_fraction = max(0.0, (base_traffic - resident_traffic) / base_traffic)

    # Head-specific: are the weights REUSED BY THE REPEATED HEAD immutable? (Backbone weights,
    # used once per replan, are a separate cross-replan question — not this axis.)
    head_weights_immutable = weights_immutable and (weights_in_loop_inv or region_reuse > 1)
    has_dependency_chain = dispatches_per_step > 1 or has_epilogue or bool(carried)

    return {
        "weights_immutable": weights_immutable,
        "head_weights_immutable": head_weights_immutable,
        "has_repeated_head_arch": temporal.has_repeated_head(),
        "prefix_kv_loop_invariant": prefix_kv_in_loop_inv,
        "has_epilogue": has_epilogue,
        "has_dependency_chain": has_dependency_chain,
        "dispatches_per_step": dispatches_per_step,
        "macs_per_step": macs,
        "dram_reducible_fraction": dram_reducible_fraction,
    }


def _legal(axis: str, facts: dict) -> bool:
    """Structural legality of an abstraction-exposure axis under a representation's facts."""
    if axis == "resident_packed_weights":
        # Action-head weights reused across the repeated head — NOT the once-per-replan backbone.
        return bool(facts["has_repeated_head"] and facts["head_weights_immutable"])
    if axis == "resident_prefix_kv":
        return bool(facts["has_repeated_head"] and facts["prefix_kv_loop_invariant"])
    if axis == "command_batching":
        return int(facts["dispatches_per_replan"]) > 1
    if axis == "autonomous_K_loop":
        return bool(facts["has_k_loop"])
    if axis == "accumulator_commit":
        return bool(facts["has_epilogue"])
    if axis == "event_tokens":
        return bool(facts["has_dependency_chain"])
    return False


def _reason(axis: str, facts: dict, legal: bool) -> str:
    K = facts["K"]
    if axis == "resident_packed_weights":
        return (f"action head reuses immutable weights across its K={K} loop (backbone excluded)"
                if legal else "no repeated-head reuse of immutable weights is visible")
    if axis == "resident_prefix_kv":
        return (f"prefix/KV is loop-invariant across the K={K} loop" if legal
                else "prefix/KV is not listed as loop-invariant state")
    if axis == "command_batching":
        return (f"{facts['dispatches_per_replan']} dispatches per replan collapse to a batch"
                if legal else "only one dispatch per replan is visible")
    if axis == "autonomous_K_loop":
        return (f"bounded K={K} loop can run on-device" if legal
                else "no bounded K-loop is visible (K<=1)")
    if axis == "accumulator_commit":
        return ("matmul/conv + epilogue keeps the i32 accumulator live"
                if legal else "no contraction+epilogue pattern present")
    if axis == "event_tokens":
        return ("a producer/consumer dependency chain exists" if legal
                else "no dependency chain to overlap")
    return ""


def _build(name: str, temporal: TemporalMetadata, base: dict) -> Representation:
    flat = name == "flat"
    K_eff = 1 if flat else int(temporal.K)
    has_k_loop = (not flat) and K_eff > 1
    # The flat capture collapses the repeated head (this is the whole point); only the multi-rate
    # view exposes it.
    has_repeated_head = (not flat) and bool(base["has_repeated_head_arch"])
    dps = int(base["dispatches_per_step"])
    dispatches_per_replan = dps if flat else dps * K_eff

    head_reuse = has_repeated_head and base["head_weights_immutable"]
    visible_weight_reuse = K_eff if head_reuse else 1
    visible_prefix_kv_reuse = (K_eff if (has_repeated_head and base["prefix_kv_loop_invariant"])
                               else 1)
    macs = base["macs_per_step"]
    work_per_dispatch = (macs / dps) if macs else None

    facts = dict(base)
    facts.update({
        "K": K_eff,
        "has_k_loop": has_k_loop,
        "has_repeated_head": has_repeated_head,
        "prefix_kv_loop_invariant": base["prefix_kv_loop_invariant"] and has_repeated_head,
        "dispatches_per_replan": dispatches_per_replan,
        "visible_weight_reuse": visible_weight_reuse,
        "visible_prefix_kv_reuse": visible_prefix_kv_reuse,
        "work_per_dispatch": work_per_dispatch,
    })

    recommended, deprioritized = [], []
    for axis in _ABSTRACTION_AXES:
        legal = _legal(axis, facts)
        entry = {"axis": axis, "reason": _reason(axis, facts, legal)}
        if legal:
            recommended.append({**entry, "evidence_type": "structural_bound",
                                "could_be_wrong_if": COULD_BE_WRONG_IF.get(axis, [])})
        else:
            deprioritized.append(entry)

    return Representation(
        name=name,
        workload=temporal.workload,
        K=K_eff,
        H=None if flat else temporal.H,
        control_rate_hz=None if flat else temporal.control_rate_hz,
        replan_deadline_ms=None if flat else temporal.replan_deadline_ms,
        deadline_visible=not flat,
        visible_weight_reuse=visible_weight_reuse,
        visible_prefix_kv_reuse=visible_prefix_kv_reuse,
        dispatches_per_step=dps,
        dispatches_per_replan=dispatches_per_replan,
        work_per_dispatch=work_per_dispatch,
        facts=facts,
        recommended_axes=recommended,
        deprioritized_axes=deprioritized,
    )


def build_representations(temporal: TemporalMetadata,
                          region: dict | None = None,
                          overrides: dict | None = None) -> dict[str, Representation]:
    """Return ``{"flat": Representation, "multirate": Representation}`` for the workload.

    ``overrides`` may supply base structural facts directly (``has_epilogue``,
    ``dispatches_per_step``, ``dram_reducible_fraction``, ``weights_immutable``,
    ``has_dependency_chain``, ``macs_per_step``) for workloads described without a region — e.g.
    a whole-model capture whose epilogue presence and matmul count are read from its MLIR.
    """
    base = _base_facts(temporal, region)
    if overrides:
        base.update({k: v for k, v in overrides.items() if v is not None})
        # Re-derive head weight immutability if immutability was overridden.
        if "weights_immutable" in overrides and "head_weights_immutable" not in overrides:
            loop_inv = {s.lower() for s in temporal.loop_invariant_state()}
            base["head_weights_immutable"] = base["weights_immutable"] and (
                any("weight" in s for s in loop_inv) or temporal.K > 1)
    return {"flat": _build("flat", temporal, base),
            "multirate": _build("multirate", temporal, base)}


def to_report_dict(rep: Representation) -> dict:
    """Schema-friendly mapping for ``flat_report.yaml`` / ``multirate_report.yaml``."""
    return {
        "workload": rep.workload,
        "representation": rep.name,
        "K": rep.K,
        "H": rep.H,
        "control_rate_hz": rep.control_rate_hz,
        "replan_deadline_ms": rep.replan_deadline_ms,
        "deadline_visible": rep.deadline_visible,
        "visible_weight_reuse": rep.visible_weight_reuse,
        "visible_prefix_kv_reuse": rep.visible_prefix_kv_reuse,
        "dispatches_per_step": rep.dispatches_per_step,
        "dispatches_per_replan": rep.dispatches_per_replan,
        "work_per_dispatch": rep.work_per_dispatch,
        "recommended_axes": rep.recommended_axes,
        "deprioritized_axes": rep.deprioritized_axes,
    }
