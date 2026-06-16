"""The DSE axis catalog: what each axis can reduce, and by how much.

Each axis declares exactly which baseline cost components its intervention can reduce, a
grounded model of the reduction, a legality predicate, a build-cost tier, and the evidence
quality of the *intervention model* itself. The triage (:mod:`merlin.dse_guidance.triage`)
turns the per-axis benefit into a gap-closure and a priority score.

Grounding rules (so no number is invented):

  * A benefit is clamped to the sum of the components it touches — an axis can never remove
    more time than those components contain.
  * Residency/packing reductions scale with the *visible* reuse (1 under a flat capture, K
    under the multi-rate view) and, when a region is available, with the region-derived
    fraction of DRAM traffic that residency actually removes.
  * The dispatch axes (``command_batching``, ``autonomous_K_loop``) use the measured
    op-level-vs-batched saving from the ``aet`` coupling data when it is present; only when no
    measurement exists do they fall back to a structural bound, and they say so via the
    evidence tag.
  * The axis's evidence tag is the weakest of its model evidence and the evidence of every
    baseline component it touches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import evidence as E
from merlin.dse_guidance.baseline_cost import BaselineCost
from merlin.dse_guidance.representation import COULD_BE_WRONG_IF

AXIS_FAMILY: dict[str, str] = {
    "PE_count_2x": "hardware",
    "SRAM_capacity_increase": "hardware",
    "DMA_bandwidth_2x": "hardware",
    "resident_packed_weights": "memory_residency",
    "resident_prefix_kv": "memory_residency",
    "command_batching": "dispatch",
    "autonomous_K_loop": "dispatch",
    "accumulator_commit": "datapath",
    "event_tokens": "datapath",
}

REQUIRED_AXES: tuple[str, ...] = tuple(AXIS_FAMILY.keys())

# Hardware-scaling could-be-wrong-if (the abstraction axes get theirs from representation).
_HW_CAVEATS: dict[str, list[str]] = {
    "PE_count_2x": [
        "compute is not on the critical path",
        "ideal 2x PE scaling does not hold (utilization, tail effects)",
    ],
    "DMA_bandwidth_2x": [
        "memory traffic is not bandwidth-bound",
        "doubling bandwidth does not halve the measured DMA time",
    ],
    "SRAM_capacity_increase": [
        "there is no measured capacity spill to remove",
        "the spilled tensors would not fit even at the larger capacity",
    ],
}


@dataclass
class AxisResult:
    axis: str
    family: str
    affected_components: list[str]
    benefit_ms: float
    evidence_type: str
    legality: int
    cost_tier: int
    reason: str
    could_be_wrong_if: list[str] = field(default_factory=list)
    # False when the axis is structurally legal but its benefit cannot be grounded (e.g. a real
    # whole-model capture that does not separate action-head cost from the backbone). The triage
    # then reports gap_closure = null rather than a fabricated magnitude.
    quantified: bool = True


def _affected_total(baseline: BaselineCost, components: list[str]) -> float:
    # A component may live in the whole-model breakdown or only in a role sub-breakdown
    # (repeated_head / loop_invariant, e.g. weight_memory, prefix_kv_memory). Cap against the
    # largest place it appears so a sub-breakdown benefit is not clamped to a missing whole.
    return sum(max(baseline.component(c), baseline.head_component(c),
                   baseline.loop_invariant_component(c)) for c in components)


def _combined_evidence(model_evidence: str, baseline: BaselineCost,
                       components: list[str]) -> str:
    tags = [model_evidence] + [baseline.evidence_for(c) for c in components]
    return E.weakest_evidence(tags)


def evaluate_axis(axis: str, facts: dict, baseline: BaselineCost,
                  coupling_per_replan: dict | None = None) -> AxisResult:
    """Evaluate one axis: benefit (ms), legality, evidence, cost tier, reason."""
    family = AXIS_FAMILY[axis]
    cost_tier = E.COST_TIERS[axis]
    caveats = COULD_BE_WRONG_IF.get(axis, _HW_CAVEATS.get(axis, []))

    components: list[str] = []
    benefit = 0.0
    legal = True
    quantified = True
    model_evidence = "structural_bound"
    reason = ""

    if axis == "PE_count_2x":
        components = ["compute"]
        benefit = 0.5 * baseline.component("compute")
        model_evidence = "analytical"
        reason = "doubling PEs halves the compute component (ideal scaling)"

    elif axis == "DMA_bandwidth_2x":
        components = ["dma_memory"]
        benefit = 0.5 * baseline.component("dma_memory")
        model_evidence = "analytical"
        reason = "doubling DMA bandwidth halves the dma_memory component (ideal scaling)"

    elif axis == "SRAM_capacity_increase":
        components = ["capacity_spill"]
        spill = baseline.component("capacity_spill")
        legal = spill > 0
        benefit = spill if legal else 0.0
        model_evidence = "structural_bound"
        reason = (f"removes the {spill:g} ms capacity spill" if legal
                  else "no capacity spill in the baseline to remove")

    elif axis == "resident_packed_weights":
        # ACTION-HEAD weights only — reused across the repeated head. The once-per-replan
        # backbone is a separate cross-replan residency question and is NOT counted here.
        reuse = max(int(facts.get("visible_weight_reuse", 1)), 1)
        legal = bool(facts.get("has_repeated_head") and facts.get("head_weights_immutable")
                     and reuse > 1)
        if not legal:
            components = ["packing"]
            reason = "no repeated-head reuse of immutable weights (backbone-only or reuse<=1)"
        elif baseline.has_head_breakdown:
            # Charge residency to the head's own packing + weight traffic — backbone untouched.
            # Evidence tags come from the parent whole components (packing, dma_memory); the
            # head weight traffic is a slice of dma_memory.
            components = ["packing", "dma_memory"]
            pack_b = baseline.head_component("packing") * (1 - 1 / reuse)
            wmem = baseline.head_component("weight_memory") or baseline.head_component("dma_memory")
            benefit = pack_b + wmem * (1 - 1 / reuse)
            reason = (f"action-head weights reused {reuse}x; resident head weights remove "
                      f"repeated head packing + weight DMA (backbone excluded)")
        elif facts.get("dram_reducible_fraction") is not None:
            # Single-region workload: the region IS the repeated head (no separate backbone).
            frac = float(facts["dram_reducible_fraction"])
            components = ["packing", "dma_memory"]
            benefit = baseline.component("packing") * (1 - 1 / reuse) \
                + baseline.component("dma_memory") * frac
            reason = (f"weights reused {reuse}x; one-time pack + resident load removes repeated "
                      f"packing and {frac:.0%} of DMA traffic (region-derived)")
        else:
            # Real whole-model capture: structurally legal, but we cannot separate action-head
            # cost from the backbone, so the benefit is NOT quantified (no fabricated magnitude).
            components = []
            quantified = False
            model_evidence = "assumed"
            reason = ("action head reuses weights across the K-loop, but this flat capture does "
                      "not separate action-head cost from the backbone; residency benefit not "
                      "quantified (structural legality only — needs a backbone/head split)")

    elif axis == "resident_prefix_kv":
        # Prefix/KV produced once by the backbone, reused across the K-step head. Reduce its
        # reload traffic only when its own cost is provided (loop_invariant sub-breakdown or a
        # prefix_kv_memory component); otherwise legal-but-unquantified.
        reuse = max(int(facts.get("visible_prefix_kv_reuse", 1)), 1)
        legal = bool(facts.get("has_repeated_head") and facts.get("prefix_kv_loop_invariant")
                     and reuse > 1)
        li_cost = baseline.loop_invariant_component("prefix_kv_memory")
        whole_cost = baseline.component("prefix_kv_memory") if "prefix_kv_memory" \
            in baseline.components else 0.0
        kv_cost = li_cost or whole_cost
        if not legal:
            reason = "prefix/KV is not loop-invariant across a repeated head"
        elif kv_cost > 0:
            components = ["prefix_kv_memory"]
            benefit = kv_cost * (1 - 1 / reuse)
            reason = f"prefix/KV reused {reuse}x by the head; resident KV removes repeated reloads"
        else:
            quantified = False
            model_evidence = "assumed"
            reason = ("prefix/KV is loop-invariant but no prefix_kv_memory cost was provided; "
                      "benefit not quantified")

    elif axis == "command_batching":
        components = ["cpu_dispatch", "sync"]
        legal = int(facts.get("dispatches_per_replan", 1)) > 1
        cap = _affected_total(baseline, components)
        if legal and coupling_per_replan and coupling_per_replan.get("op_level") \
                and coupling_per_replan.get("batched"):
            saved = (coupling_per_replan.get("cpu_dispatch_ms_saved", 0.0)
                     + coupling_per_replan.get("sync_ms_saved", 0.0))
            benefit = max(0.0, min(saved, cap))
            model_evidence = coupling_per_replan.get("source", "measured")
            reason = ("measured op-level vs batched submit saving "
                      f"({coupling_per_replan['op_level']['num_dispatches']} -> "
                      f"{coupling_per_replan['batched']['num_dispatches']} dispatches)")
        elif legal:
            n = int(facts["dispatches_per_replan"])
            benefit = cap * (1 - 1 / n)
            model_evidence = "structural_bound"
            reason = f"{n} dispatches/replan collapse to a batch (no measurement supplied)"
        else:
            reason = "only one dispatch per replan is visible"

    elif axis == "autonomous_K_loop":
        # The per-step host launches removed are the ones INSIDE the K-loop — i.e. the repeated
        # head's dispatch/sync, NOT the once-per-replan backbone's. Charge the head slice when a
        # breakdown exists.
        components = ["cpu_dispatch", "sync"]
        K = int(facts.get("K", 1))
        legal = bool(facts.get("has_k_loop") and K > 1)
        if baseline.has_head_breakdown:
            cap = baseline.head_component("cpu_dispatch") + baseline.head_component("sync")
            scope = "head"
        else:
            cap = _affected_total(baseline, components)
            scope = "loop"
        if legal and coupling_per_replan and coupling_per_replan.get("op_level"):
            op = coupling_per_replan["op_level"]
            dps = max(int(facts.get("dispatches_per_step", 1)), 1)
            per_step = (op["cpu_dispatch_ms"] + op["sync_ms"]) / max(op["num_dispatches"], 1) * dps
            benefit = max(0.0, min(per_step * (K - 1), cap))
            model_evidence = op.get("source", "measured")
            reason = f"on-device K={K} loop removes {K-1} per-step host launches (measured)"
        elif legal:
            benefit = cap * (1 - 1 / K)
            model_evidence = "structural_bound"
            reason = (f"on-device K={K} loop removes per-step host dispatch/sync inside the "
                      f"{scope} (backbone dispatch excluded)")
        else:
            reason = "no bounded K-loop (K<=1)"

    elif axis == "accumulator_commit":
        components = ["intermediate_materialization"]
        legal = bool(facts.get("has_epilogue"))
        benefit = baseline.component("intermediate_materialization") if legal else 0.0
        model_evidence = "structural_bound"
        reason = ("commit-in-place removes the i32 intermediate materialization"
                  if legal else "no contraction+epilogue pattern present")

    elif axis == "event_tokens":
        components = ["sync"]
        legal = bool(facts.get("has_dependency_chain"))
        # Upper structural bound: event tokens can overlap the sync wait behind compute.
        benefit = baseline.component("sync") if legal else 0.0
        model_evidence = "structural_bound"
        reason = ("event tokens overlap the sync wait behind compute (upper bound)"
                  if legal else "no dependency chain to overlap")

    else:  # pragma: no cover - guarded by REQUIRED_AXES
        raise KeyError(f"unknown axis '{axis}'")

    # An axis can never remove more time than the components it touches contain.
    cap_total = _affected_total(baseline, components)
    if not legal or not quantified:
        benefit = 0.0
    elif components:
        benefit = max(0.0, min(benefit, cap_total))
    evidence_type = _combined_evidence(model_evidence, baseline, components)

    return AxisResult(
        axis=axis,
        family=family,
        affected_components=components,
        benefit_ms=benefit,
        evidence_type=evidence_type,
        legality=1 if legal else 0,
        cost_tier=cost_tier,
        reason=reason,
        could_be_wrong_if=caveats,
        quantified=quantified,
    )


def evaluate_axes(facts: dict, baseline: BaselineCost,
                  coupling_per_replan: dict | None = None) -> list[AxisResult]:
    """Evaluate every catalog axis under the given representation facts and baseline."""
    return [evaluate_axis(a, facts, baseline, coupling_per_replan) for a in REQUIRED_AXES]
