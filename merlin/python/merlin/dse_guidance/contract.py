"""Workload-contract analysis package — what Merlin hands a future DSE engine.

Merlin is not the DSE engine and does not pick a design. It is a compiler-based analysis that
recovers the HW/SW contract a flat capture erases and turns it into a package the DSE engine
consumes:

  abstraction_candidates  — which HW/SW abstractions the workload implies a need for, each with
                            the compiler proof and runtime/HW support required, the DSE knobs it
                            exposes, and (explicitly) what is NOT claimed;
  dse_readiness           — is this workload ready for a DSE tool to rank designs, and what is
                            still missing;
  measurement_plan        — what to measure next, split into proxy (host/runtime) vs target
                            (the proposed design) measurements.

This module *consolidates* the structural candidates (:mod:`candidates`), the numerical-contract
candidates (:mod:`numerical_contract`), and the requirement envelope (:mod:`design_envelope`) into
that package. It claims no speedup, cycle, or energy number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# DSE axis -> the concrete HW/SW abstraction object it implies + the knobs a DSE tool would sweep.
ABSTRACTION_MAP: dict[str, dict] = {
    "resident_action_head_weights": {
        "system_abstraction": "resident_weight_object",
        "dse_knobs": ["local_memory_capacity", "packed_weight_format", "dma_bandwidth",
                      "residency/replacement_policy", "weight_prefetch_overlap"]},
    "resident_packed_lowbit_weights": {
        "system_abstraction": "packed_lowbit_tensor + resident_weight_object + scale_object",
        "dse_knobs": ["resident_capacity_at_format", "packed_layout_support", "scale_object",
                      "low_bit_matmul_datapath"]},
    "resident_prefix_kv": {
        "system_abstraction": "prefix_kv_object",
        "dse_knobs": ["resident_kv_capacity", "kv_lifetime_policy", "kv_quantization"]},
    "autonomous_K_loop": {
        "system_abstraction": "bounded_loop_command + loop_carried_state_handle",
        "dse_knobs": ["device_loop_controller", "dependency_tracking", "resident_state_handles"]},
    "command_batching": {
        "system_abstraction": "command_buffer",
        "dse_knobs": ["submission_granularity", "persistent_command_graph"]},
    "command_buffer_per_replan": {
        "system_abstraction": "persistent_command_buffer",
        "dse_knobs": ["region_level_dispatch", "persistent_graph_submission"]},
    "packed_layout_preservation": {
        "system_abstraction": "packed_weight_cache / layout_persistent_buffer",
        "dse_knobs": ["packed_layout_object", "repack_avoidance"]},
    "native_lowbit_compute": {
        "system_abstraction": "low_bit_matmul_datapath + scale_object + accumulator_object",
        "dse_knobs": ["compute_precision", "accumulator_width", "requant_unit"]},
    "fused_dequant_matmul": {
        "system_abstraction": "dequant_fused_matmul",
        "dse_knobs": ["weight_load_prologue", "fused_dequant"]},
    "fused_requant_epilogue": {
        "system_abstraction": "accumulator_object + requant_epilogue",
        "dse_knobs": ["in_hw_epilogue_commit", "accumulator_visibility"]},
    "accumulator_commit": {
        "system_abstraction": "accumulator_object + requant_epilogue",
        "dse_knobs": ["accumulator_capacity", "epilogue_fusion"]},
    "quantized_KV_cache": {
        "system_abstraction": "resident_KV_cache (quantized)",
        "dse_knobs": ["kv_precision", "kv_capacity", "decode_gemv_datapath"]},
    "backbone_head_partition": {
        "system_abstraction": "phase_boundary / region_handle (slow/fast split)",
        "dse_knobs": ["engine_partition", "time_share_vs_split", "state_crossing_transport"]},
    "async_chunk_overlap": {
        "system_abstraction": "double_buffered_replan_state + async_queue",
        "dse_knobs": ["double_buffering", "async_scheduling", "deadline_slack_use"]},
}

_NOT_CLAIMED = ["speedup", "cycle_reduction", "energy_reduction", "accuracy"]


@dataclass
class AbstractionCandidate:
    axis: str
    system_abstraction: str
    why_this_exists: dict
    compiler_proof: str
    runtime_hw_support: str
    dse_knobs_exposed: list
    measurements_needed: list
    quantification_blocked_by: str
    legality: str = "structural"
    what_is_not_claimed: list = field(default_factory=lambda: list(_NOT_CLAIMED))


def _abstraction_for(axis: str) -> dict:
    return ABSTRACTION_MAP.get(axis, {"system_abstraction": axis, "dse_knobs": []})


def abstraction_candidates(structural_candidates, numerical) -> list[AbstractionCandidate]:
    """Merge structural + numerical candidates into unified HW/SW abstraction certificates."""
    out: list[AbstractionCandidate] = []
    seen: set[str] = set()
    # Structural candidates (candidates.DseCandidate).
    for c in structural_candidates:
        if c.axis in seen:
            continue
        seen.add(c.axis)
        m = _abstraction_for(c.axis)
        out.append(AbstractionCandidate(
            axis=c.axis, system_abstraction=m["system_abstraction"],
            why_this_exists={"signal": c.signal_type, "evidence": c.evidence,
                             "reason": c.reason, "attributed_facts": c.attributed_facts},
            compiler_proof=c.required_compiler_proof, runtime_hw_support=c.required_hw_support,
            dse_knobs_exposed=m["dse_knobs"], measurements_needed=list(c.required_measurements),
            quantification_blocked_by=c.quantification_blocked_by))
    # Numerical-contract candidates (numerical_contract.NumericalCandidate).
    if numerical is not None:
        for k in numerical.candidates:
            if k.axis in seen:
                continue
            seen.add(k.axis)
            m = _abstraction_for(k.axis)
            out.append(AbstractionCandidate(
                axis=k.axis, system_abstraction=m["system_abstraction"],
                why_this_exists={"signal": "numerical_contract", "evidence": k.evidence},
                compiler_proof="(numerical) " + ", ".join(k.required_accuracy_measurements[:1]),
                runtime_hw_support=k.required_hw_support, dse_knobs_exposed=m["dse_knobs"],
                measurements_needed=(list(k.required_accuracy_measurements)
                                     + list(k.required_performance_measurements)),
                quantification_blocked_by=k.quantification_blocked_by))
    return out


# ----------------------------------------------------------------- DSE readiness

@dataclass
class DseReadiness:
    workload: str
    fields: dict                       # name -> {available: bool, source: str}
    missing: list

    @property
    def ready(self) -> bool:
        return not self.missing


def dse_readiness(topo, attribution, numerical, cpu_coupling_available: bool) -> DseReadiness:
    head = attribution.role("repeated_head") if attribution else None
    has_role = head is not None and head.attribution_status == "attributed"
    has_dtype = numerical is not None and numerical.declared_quantization is not None
    fields = {
        "topology_recovered": {"available": True, "source": "recovered_from_prov_fqn"},
        "role_attribution": {"available": has_role,
                             "source": head.source if has_role else "unavailable"},
        "K_source": {"available": True, "source": "assumed_reference"},
        "deadline_source": {"available": topo.replan_deadline_ms is not None,
                            "source": "assumed_reference"},
        "dtype_contract": {"available": has_dtype, "source": "recovered_from_ir"},
        "state_lifetimes": {"available": bool(topo.loop_invariant_state()),
                            "source": "recovered_from_prov_fqn"},
        "dispatch_graph": {"available": has_role, "source": "recovered_from_ir"},
        "accuracy_constraints": {"available": False, "source": "unavailable"},
        "cpu_coupling": {"available": cpu_coupling_available,
                         "source": "proxy_measured" if cpu_coupling_available else "unavailable"},
    }
    missing = []
    if not fields["accuracy_constraints"]["available"]:
        missing.append("quantization accuracy gates (per candidate low-bit format)")
    if not fields["cpu_coupling"]["available"]:
        missing.append("real (target) command-submit / sync latency")
    if fields["K_source"]["source"] == "assumed_reference":
        missing.append("K / control-rate from the real deployment (currently reference values)")
    return DseReadiness(workload=topo.workload, fields=fields, missing=missing)


# ----------------------------------------------------------------- measurement plan

# How each measurement is obtainable: proxy (host/runtime, exists now) vs target (proposed design).
_PROXY_HINTS = ("dispatch", "host", "submit", "encode", "sync", "command")
_TARGET_HINTS = ("bandwidth", "cycles", "latency", "capacity", "memory", "kernel calibration",
                 "matmul cycles")
_ACCURACY_HINTS = ("accuracy", "argmax", "bit-exact", "stability")  # not "cos" (matches "cost")


def _classify_measurement(m: str) -> str:
    low = m.lower()
    if any(h in low for h in _ACCURACY_HINTS):
        return "accuracy_measurable_now"     # accuracy can be measured without final HW
    if any(h in low for h in _PROXY_HINTS):
        return "proxy_measured"              # host/runtime proxy obtainable now
    if any(h in low for h in _TARGET_HINTS):
        return "target_measured"             # needs the proposed design / cycle-exact target
    return "target_measured"


def measurement_plan(candidates: list[AbstractionCandidate]) -> dict:
    buckets: dict[str, list[str]] = {"accuracy_measurable_now": [], "proxy_measured": [],
                                     "target_measured": []}
    seen: set[str] = set()
    for c in candidates:
        for m in c.measurements_needed:
            if m in seen:
                continue
            seen.add(m)
            buckets[_classify_measurement(m)].append(m)
    return {
        "measurable_now": {
            "accuracy": sorted(buckets["accuracy_measurable_now"]),
            "runtime_proxy": sorted(buckets["proxy_measured"]),
        },
        "needs_target_design": sorted(buckets["target_measured"]),
        "note": "accuracy + runtime-proxy measurements need no future hardware; target_measured "
                "needs the proposed design (cycle-exact sim or implemented prototype).",
    }


# ----------------------------------------------------------------- emitters

def abstraction_yaml(candidates: list[AbstractionCandidate]) -> dict:
    return {"abstraction_candidates": [
        {"axis": c.axis, "system_abstraction_needed": c.system_abstraction,
         "why_this_exists": c.why_this_exists, "compiler_proof": c.compiler_proof,
         "runtime_hw_support": c.runtime_hw_support, "dse_knobs_exposed": c.dse_knobs_exposed,
         "measurements_needed": c.measurements_needed, "what_is_not_claimed": c.what_is_not_claimed,
         "status": {"legality": c.legality,
                    "quantification_blocked_by": c.quantification_blocked_by}}
        for c in candidates]}


def readiness_yaml(r: DseReadiness) -> dict:
    return {"dse_readiness": {"workload": r.workload, "ready_to_rank_designs": r.ready,
                              "fields": r.fields, "missing_before_ranking": r.missing}}


def workload_contract_report_md(topo, attribution, numerical, envelope,
                                cands: list[AbstractionCandidate], readiness: DseReadiness,
                                plan: dict) -> str:
    """The unified contract-analysis package for one workload (what a DSE engine consumes)."""
    head = attribution.role("repeated_head") if attribution else None
    L = [f"# Workload contract analysis — {topo.workload}\n"]
    L.append("> Merlin recovers the HW/SW contract a flat capture erases and hands a future DSE "
             "engine the abstractions the workload needs, the requirements any design must meet, "
             "and what is still missing. It does **not** pick a design or claim a speedup.\n")
    L.append(f"- class: **{topo.workload_class}**  ·  K={topo.K}, H={topo.H}, "
             f"control_rate={topo.control_rate_hz} Hz\n")

    L.append("## 1. Recovered structure\n")
    if head and head.attribution_status == "attributed":
        f = head.facts
        L.append(f"- repeated head ({head.source}): {f['matmul_count']} matmuls, "
                 f"{f['weight_bytes']/1e6:.0f} MB weights, {f['macs_per_invocation']/1e9:.1f} "
                 f"GMAC/step, reused x{topo.K}")
    bb = attribution.role("backbone_once") if attribution else None
    if bb:
        L.append(f"- backbone (once/replan): {bb.facts.get('matmul_count')} matmuls")
    L.append("")

    if numerical is not None:
        L.append("## 2. Numerical contract\n")
        L.append(f"- storage **{numerical.weight_storage_dtype}**, compute "
                 f"**{numerical.compute_dtype}**; lost: "
                 f"{', '.join(numerical.lost_structure) or 'none'} (severity {numerical.severity})\n")

    if envelope is not None:
        L.append("## 3. Requirements (hardware-independent)\n")
        for name in ("macs_per_replan", "resident_capacity_required",
                     "avoidable_weight_reload_bytes", "required_compute_rate",
                     "required_weight_bandwidth", "required_command_rate"):
            r = envelope.req(name)
            if r and r.value is not None:
                L.append(f"- {name} = {r.value:.3e} {r.unit} ({r.evidence})")
        L.append("- resident capacity by format: " + ", ".join(
            f"{k}={v/1e6:.0f}MB" for k, v in envelope.capacity_by_dtype_B.items()))
        L.append("")

    L.append("## 4. HW/SW abstraction candidates\n")
    L.append("| abstraction | DSE knobs | blocked by |")
    L.append("|-------------|-----------|------------|")
    for c in cands:
        L.append(f"| {c.system_abstraction} | {', '.join(c.dse_knobs_exposed) or '—'} | "
                 f"{c.quantification_blocked_by} |")
    L.append("\n_None claim a speedup/accuracy number — see `abstraction_candidates.yaml`._\n")

    L.append("## 5. Measurement plan\n")
    L.append(f"- measurable now (accuracy): {', '.join(plan['measurable_now']['accuracy']) or '—'}")
    L.append(f"- measurable now (runtime proxy): "
             f"{', '.join(plan['measurable_now']['runtime_proxy']) or '—'}")
    L.append(f"- needs target design: {', '.join(plan['needs_target_design']) or '—'}\n")

    L.append("## 6. DSE readiness\n")
    L.append(f"- ready to rank designs: **{readiness.ready}**")
    for m in readiness.missing:
        L.append(f"  - missing: {m}")
    L.append("")
    return "\n".join(L)
