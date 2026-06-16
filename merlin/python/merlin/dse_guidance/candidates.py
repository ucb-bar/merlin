"""Structural DSE-candidate discovery for VLAs.

This is the missing middle layer between "we have temporal metadata" and "here is a quantitative
ranking". It runs pressure detectors over a :class:`~merlin.dse_guidance.topology.VlaRuntimeTopology`
and emits **candidate certificates** — structural facts about which DSE axes the workload's shape
makes relevant, what would have to be proven/built to exploit them, and *what to measure before
ranking them*. It deliberately produces **no cycle numbers and no scores**: a candidate is valid
without calibration; the quantitative gap-closure comes later (and only with measurement).

    temporal metadata -> DSE candidate certificates -> measurement plan -> (calibrated) triage

Each detector keys off a single pressure signal (temporal reuse, rate mismatch, CPU coupling,
layout/packing, memory traffic, accumulator/epilogue, dynamic loop) so the evidence trail is
explicit and falsifiable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import topology as TOP
from merlin.dse_guidance.topology import VlaRuntimeTopology

# Static per-axis facts: what it would take to exploit the axis, and what to measure first.
CANDIDATE_CATALOG: dict[str, dict] = {
    "resident_action_head_weights": {
        "compiler_proof": "action-head weights immutable and live across the K-loop "
                          "(resident-pack candidate); backbone weights excluded",
        "hw_support": "resident weight store + pack-once/use-many interface",
        "measurements": ["action_head_weight_bytes", "weight_reload_bytes_per_step",
                         "pack_cost_per_step", "resident_capacity_required",
                         "measured memory bandwidth"],
        "could_be_wrong_if": ["weights exceed resident capacity",
                              "packing already hoisted out of the loop",
                              "DMA/weight traffic is not the bottleneck",
                              "host dispatch dominates total latency"],
        "benefit_unquantified": "head/backbone cost attribution or measured weight traffic "
                                "is unavailable",
    },
    "resident_prefix_kv": {
        "compiler_proof": "prefix/KV produced once and read-only across the K-loop",
        "hw_support": "resident state object with replan-scoped lifetime",
        "measurements": ["prefix_kv_bytes", "kv_reload_bytes_per_step", "resident_capacity_required"],
        "could_be_wrong_if": ["prefix/KV exceeds resident capacity",
                              "prefix/KV is cheap to recompute",
                              "KV traffic is not a limiting component"],
        "benefit_unquantified": "prefix/KV byte traffic is not measured/attributed",
    },
    "command_batching": {
        "compiler_proof": "the K-loop dependency graph is static and known at submit time",
        "hw_support": "persistent/batched command buffer submit",
        "measurements": ["dispatches_per_replan", "host_submit_ns", "command_encode_ns",
                         "batched_submit_ns"],
        "could_be_wrong_if": ["host dispatch is not on the critical path",
                              "batching raises per-submit latency",
                              "the runtime already coalesces submits"],
        "benefit_unquantified": "host dispatch/sync overhead is not measured",
    },
    "autonomous_K_loop": {
        "compiler_proof": "bounded K and a loop body expressible as a device-resident program",
        "hw_support": "device-side loop controller + dependency tracking",
        "measurements": ["K", "per_step_host_dispatch_ns", "per_step_sync_ns",
                         "loop_carried_state_bytes"],
        "could_be_wrong_if": ["K is data-dependent / unbounded",
                              "per-step host overhead is already negligible",
                              "on-device control costs area not justified by the saving"],
        "benefit_unquantified": "per-step host/sync overhead inside the loop is not measured",
    },
    "packed_layout_preservation": {
        "compiler_proof": "a packed/quantized weight layout is produced once and consumed by the "
                          "same op family across the loop without re-pack",
        "hw_support": "packed layout as a first-class, dispatch-crossing object",
        "measurements": ["pack_count_per_replan", "pack_bytes", "repacks_avoided"],
        "could_be_wrong_if": ["packing is not repeated",
                              "the op family changes layout between uses",
                              "dequant happens upstream regardless"],
        "benefit_unquantified": "repeated pack/unpack traffic is not measured",
    },
    "backbone_head_partition": {
        "compiler_proof": "a clean state boundary between the once-per-replan backbone and the "
                          "K-times head (the crossing tensors are enumerable)",
        "hw_support": "slow-path/fast-path engine split or time-shared scheduling",
        "measurements": ["backbone_cost", "head_cost_per_step", "state_crossing_bytes",
                         "deadline_slack"],
        "could_be_wrong_if": ["backbone and head costs are comparable (no clear slow/fast split)",
                              "state crossing is too large to move",
                              "a single engine already meets the deadline"],
        "benefit_unquantified": "backbone vs head cost split is not measured",
    },
    "decode_kv_cache_path": {
        "compiler_proof": "autoregressive decode with a growing KV cache and batch=1 GEMV shapes",
        "hw_support": "decode/GEMV-optimized datapath + resident KV cache object",
        "measurements": ["tokens_decoded", "kv_growth_bytes", "gemv_shape_distribution"],
        "could_be_wrong_if": ["the head is not autoregressive",
                              "throughput-GEMM utilisation is already adequate",
                              "KV fits trivially"],
        "benefit_unquantified": "decode-step memory/compute profile is not measured",
    },
    "accumulator_commit": {
        "compiler_proof": "matmul/conv + epilogue (bias/requant/activation) keeps the i32 "
                          "accumulator live to a fused commit",
        "hw_support": "accumulator object + in-hardware epilogue commit",
        "measurements": ["intermediate_i32_bytes", "epilogue_dispatch_count"],
        "could_be_wrong_if": ["the intermediate is already fused",
                              "the epilogue is memory-bound elsewhere",
                              "the accumulator does not fit"],
        "benefit_unquantified": "intermediate materialization traffic is not measured",
    },
    "async_chunk_overlap": {
        "compiler_proof": "the next replan is independent of the current chunk's execution",
        "hw_support": "double-buffered action chunks + async backbone/head scheduling",
        "measurements": ["replan_latency", "chunk_execution_time", "deadline_slack"],
        "could_be_wrong_if": ["the next replan depends on executed-chunk feedback",
                              "there is no deadline slack to exploit",
                              "double buffering exceeds memory"],
        "benefit_unquantified": "replan latency vs chunk execution time is not measured",
    },
}


@dataclass
class DseCandidate:
    id: str
    axis: str
    signal_type: str
    evidence: dict
    reason: str
    required_compiler_proof: str
    required_hw_support: str
    required_measurements: list[str]
    could_be_wrong_if: list[str]
    legality: str = "structural"
    benefit: str = "unquantified"
    reason_benefit_unquantified: str = ""


def _make(topo: VlaRuntimeTopology, axis: str, signal_type: str, evidence: dict,
          reason: str) -> DseCandidate:
    cat = CANDIDATE_CATALOG[axis]
    return DseCandidate(
        id=f"{topo.workload}.{axis}",
        axis=axis,
        signal_type=signal_type,
        evidence=evidence,
        reason=reason,
        required_compiler_proof=cat["compiler_proof"],
        required_hw_support=cat["hw_support"],
        required_measurements=list(cat["measurements"]),
        could_be_wrong_if=list(cat["could_be_wrong_if"]),
        reason_benefit_unquantified=cat["benefit_unquantified"],
    )


# --- pressure detectors: each returns (axis, signal_type, evidence, reason) tuples ----------

def _detect_temporal_reuse(topo, facts):
    out = []
    if not topo.has_repeated_head():
        return out
    inv = topo.loop_invariant_state()
    ev = {"K": topo.K, "loop_invariant_state": sorted(inv),
          "head_phases": [r.name for r in topo.head_phases()]}
    if any("weight" in s.lower() for s in inv):
        out.append(("resident_action_head_weights", "temporal_reuse", ev,
                    f"action-head weights are loop-invariant across K={topo.K} steps; a flat "
                    "capture sees one use and hides this axis"))
    out.append(("autonomous_K_loop", "temporal_reuse",
                {"K": topo.K, "loop_carried_state": sorted(topo.loop_carried_state())},
                f"a bounded K={topo.K} loop with loop-carried state can run device-side"))
    return out


def _detect_rate_mismatch(topo, facts):
    out = []
    if topo.backbone_phases() and topo.head_phases() and topo.control_rate_hz:
        ev = {"backbone_phases": [r.name for r in topo.backbone_phases()],
              "head_phases": [r.name for r in topo.head_phases()],
              "K": topo.K, "H": topo.H, "control_rate_hz": topo.control_rate_hz,
              "replan_deadline_ms": topo.replan_deadline_ms}
        out.append(("backbone_head_partition", "rate_mismatch", ev,
                    "backbone runs once per replan while the head runs K times — the rates differ, "
                    "so a slow/fast partition is a candidate"))
        out.append(("async_chunk_overlap", "rate_mismatch",
                    {"H": topo.H, "control_rate_hz": topo.control_rate_hz,
                     "replan_deadline_ms": topo.replan_deadline_ms},
                    "the robot executes H actions at the control rate, opening a window to overlap "
                    "the next replan with chunk execution"))
    return out


def _detect_cpu_coupling(topo, facts):
    out = []
    if topo.has_repeated_head():
        out.append(("command_batching", "cpu_coupling",
                    {"K": topo.K, "note": "K per-step submits are collapsible to one buffer"},
                    f"the K={topo.K} head steps issue repeated host submits a flat capture hides"))
    return out


def _detect_layout_packing(topo, facts):
    out = []
    inv = topo.loop_invariant_state()
    quantized = bool(facts and getattr(facts, "dtype", None)
                     and str(facts.dtype).lower() in ("i8", "int8", "i4", "int4", "fp8", "f8"))
    if topo.has_repeated_head() and (any("weight" in s.lower() for s in inv) or quantized):
        out.append(("packed_layout_preservation", "layout_packing",
                    {"K": topo.K, "quantized": quantized},
                    "the same (packed/quantized) weight layout is consumed every step; preserving "
                    "it across dispatches avoids re-packing"))
    return out


def _detect_accumulator_epilogue(topo, facts):
    out = []
    has_epi = bool(facts and getattr(facts, "has_epilogue", False))
    if has_epi:
        out.append(("accumulator_commit", "accumulator_epilogue", {"has_epilogue": True},
                    "matmul/conv + epilogue keeps a large i32 accumulator live; a fused commit "
                    "removes the intermediate materialization"))
    return out


def _detect_dynamic_loop(topo, facts):
    out = []
    if topo.workload_class == TOP.CLASS_AUTOREGRESSIVE:
        out.append(("decode_kv_cache_path", "dynamic_loop",
                    {"workload_class": topo.workload_class, "K": topo.K},
                    "autoregressive decode with a growing KV cache and batch=1 GEMV shapes wants "
                    "a decode/KV-cache-optimized path, not GEMM throughput"))
    return out


# prefix/KV residency keys off the recovered state-crossing boundary.
def _detect_prefix_kv(topo, facts):
    out = []
    crossings = topo.state_crossing_boundaries()
    kv = [c for c in crossings if "kv" in c["state"].lower() or "prefix" in c["state"].lower()
          or "feature" in c["state"].lower()]
    if kv:
        out.append(("resident_prefix_kv", "temporal_reuse",
                    {"crossings": kv, "reused_times": topo.K},
                    "prefix/KV/features are produced once and reused across the K-loop"))
    return out


_DETECTORS = (_detect_temporal_reuse, _detect_prefix_kv, _detect_rate_mismatch,
              _detect_cpu_coupling, _detect_layout_packing, _detect_accumulator_epilogue,
              _detect_dynamic_loop)


def detect(topo: VlaRuntimeTopology, capture_facts=None) -> list[DseCandidate]:
    """Run all pressure detectors; return structural candidates (deduped by axis, first wins)."""
    seen: set[str] = set()
    out: list[DseCandidate] = []
    for det in _DETECTORS:
        for hit in det(topo, capture_facts):
            axis, signal_type, evidence, reason = hit
            if axis in seen or axis not in CANDIDATE_CATALOG:
                continue
            seen.add(axis)
            out.append(_make(topo, axis, signal_type, evidence, reason))
    return out


def to_yaml_obj(candidates: list[DseCandidate]) -> dict:
    return {"dse_candidate_axes": [
        {"id": c.id, "axis": c.axis,
         "signal": {"type": c.signal_type, "evidence": c.evidence},
         "dse_implication": {"candidate_axis": c.axis, "reason": c.reason},
         "required_compiler_proof": c.required_compiler_proof,
         "required_hw_support": c.required_hw_support,
         "required_measurements": c.required_measurements,
         "could_be_wrong_if": c.could_be_wrong_if,
         "current_status": {"legality": c.legality, "benefit": c.benefit,
                            "reason_benefit_unquantified": c.reason_benefit_unquantified}}
        for c in candidates
    ]}


def markdown(topo: VlaRuntimeTopology, candidates: list[DseCandidate]) -> str:
    L = [f"# Structural DSE candidate axes — {topo.workload}\n"]
    L.append(f"- workload class: **{topo.workload_class}**  ·  K={topo.K}, H={topo.H}, "
             f"control_rate={topo.control_rate_hz} Hz\n")
    L.append("> These are **structural** candidates derived from the recovered workload contract. "
             "They are valid without calibration. Each lists what must be proven/built and **what "
             "to measure before any quantitative ranking** — no cycle numbers are claimed here.\n")
    if not candidates:
        L.append("_No DSE candidates: the workload exposes no multi-rate reuse, rate mismatch, or "
                 "epilogue pressure._\n")
        return "\n".join(L)
    for c in candidates:
        L.append(f"## {c.axis}")
        L.append(f"- **signal**: {c.signal_type} — {c.reason}")
        L.append(f"- **evidence**: `{c.evidence}`")
        L.append(f"- **needs (compiler)**: {c.required_compiler_proof}")
        L.append(f"- **needs (hw/runtime)**: {c.required_hw_support}")
        L.append(f"- **measure first**: {', '.join(c.required_measurements)}")
        L.append(f"- **status**: legality={c.legality}, benefit={c.benefit} "
                 f"({c.reason_benefit_unquantified})")
        L.append(f"- **could be wrong if**: {'; '.join(c.could_be_wrong_if)}")
        L.append("")
    return "\n".join(L)
