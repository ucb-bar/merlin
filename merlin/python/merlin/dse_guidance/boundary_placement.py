"""HW/SW boundary-placement analysis — the boundary search space, not the choice.

Merlin does not pick the HW/SW boundary, run DSE, or claim speedup/cycles/area/energy, or a chosen/ranked design.
For each workload-implied abstraction it asks *where the responsibility could live* — compiler
transform, runtime/HAL object, command buffer / command ISA, accelerator ISA, device microcode /
controller, or fixed hardware datapath — and emits, per placement: what software vs. hardware would
manage, the compiler proof required, the runtime/ISA/HW support required, the metadata crossing the
boundary, the DSE knobs it creates, the overfit risk, and the missing evidence. The DSE tool later
chooses among these quantitatively; Merlin only makes the options explicit, grounded in the prior
phases (operator geometry, contract graph, memory/DMA, sharding, fusion, compiler proofs).

Honesty rules held throughout: placements are grounded in *which workloads actually show the
structure* (not asserted uniformly); abstractions whose enabling structure the flat dequantized /
attention-lowered capture erased (packed low-bit, scale, KV) are ``blocked`` / ``unavailable`` at
the levels that need it, with the missing evidence named; the boundary_pressure_score is defined as
**evidence breadth**, never performance, priority, or benefit.

The field and status vocabulary used throughout this module (``sources``, ``region_roles``,
``support``, ``cp_axis``, ``erased``/``kv``, ``metadata``, ``knobs``, ``risk``, ``levels``, and the
level/status values) is defined in ``docs/reference/dse_boundary_vocabulary.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---- boundary levels (allowed vocabulary) ----
L_COMPILER = "compiler_transform"
L_RUNTIME = "runtime_hal_object"
L_COMMAND = "command_buffer_or_command_isa"
L_ISA = "accelerator_isa"
L_MICROCODE = "device_microcode_or_controller"
L_DATAPATH = "fixed_hardware_datapath"
LEVELS = [L_COMPILER, L_RUNTIME, L_COMMAND, L_ISA, L_MICROCODE, L_DATAPATH]

# ---- placement status (allowed vocabulary) ----
S_STRONG = "strong_candidate"
S_POSSIBLE = "possible"
S_WEAK = "weak_candidate"
S_NA = "not_applicable"
S_BLOCKED = "blocked"
S_UNAVAIL = "unavailable"
STATUS = {S_STRONG, S_POSSIBLE, S_WEAK, S_NA, S_BLOCKED, S_UNAVAIL}

# ---- responsibility cell vocabulary ----
R_OWNS = "owns"
R_ASSISTS = "assists"
R_DECLARES = "declares"
R_CONSUMES = "consumes"
R_NA = "not_applicable"
R_UNKNOWN = "unknown"
RESP_CELLS = {R_OWNS, R_ASSISTS, R_DECLARES, R_CONSUMES, R_NA, R_UNKNOWN}

# level-generic role template: what software / hardware manage and what each level needs.
_LEVEL_ROLE = {
    L_COMPILER: dict(sw="compiler rewrites the workload; hardware sees ordinary ops",
                     hw="none (generic ops)", runtime="none", isa="none",
                     hw_support="generic matmul / elementwise datapath"),
    L_RUNTIME: dict(sw="runtime tracks the object's lifetime + layout",
                    hw="exploits the declared lifetime / layout",
                    runtime="object ABI + lifetime tracking + metadata", isa="none",
                    hw_support="addressable resident store"),
    L_COMMAND: dict(sw="submits higher-level commands instead of ops",
                    hw="command processor executes / loops / tracks deps",
                    runtime="command encoder + event/queue", isa="command opcodes + handles",
                    hw_support="device-side command processor"),
    L_ISA: dict(sw="compiler targets the semantic ISA op",
                hw="datapath executes the semantic instruction",
                runtime="dispatch only", isa="the semantic instruction + operands",
                hw_support="datapath implementing the op"),
    L_MICROCODE: dict(sw="submits a bounded region; device iterates",
                      hw="controller owns the loop / deps / prefetch / state",
                      runtime="bounded-region handoff", isa="region / loop descriptor",
                      hw_support="sequencer + state + dependency tracking"),
    L_DATAPATH: dict(sw="none (absorbed into the unit)", hw="hardwired unit",
                     runtime="none", isa="implicit", hw_support="dedicated fixed unit"),
}

# Erased-structure placement profile: only the compiler-dequant (f32, status-quo) path is present;
# every other level is blocked pending a low-bit capture + accuracy.
_ERASED_LEVELS = {L_COMPILER: S_POSSIBLE, L_RUNTIME: S_BLOCKED, L_COMMAND: S_BLOCKED,
                  L_ISA: S_BLOCKED, L_MICROCODE: S_BLOCKED, L_DATAPATH: S_BLOCKED}
# KV / attention structure is lowered away entirely — not even visible.
_KV_LEVELS = {L_COMPILER: S_NA, L_RUNTIME: S_UNAVAIL, L_COMMAND: S_UNAVAIL, L_ISA: S_UNAVAIL,
              L_MICROCODE: S_UNAVAIL, L_DATAPATH: S_UNAVAIL}


def _lv(ct, rh, cmd, isa, mc, fx) -> dict:
    return {L_COMPILER: ct, L_RUNTIME: rh, L_COMMAND: cmd, L_ISA: isa, L_MICROCODE: mc,
            L_DATAPATH: fx}


# ---- the boundary catalog: 27 abstractions × {sources, region_roles, support, cp_axis, erased,
#      metadata, knobs[(name,reason)], risk, levels{level:status}} ----
_K = "knobs"
_BOUNDARY_CATALOG: dict[str, dict] = {
    "resident_weight_object": dict(
        sources=["memory_hierarchy_envelope", "state_lifetime", "compiler_proof_matrix"],
        region_roles=["repeated_head", "backbone_once"], support="k_loop",
        cp_axis="resident_action_head_weights", erased=False,
        metadata="dtype, shape, layout, size_bytes, lifetime",
        knobs=[("resident_capacity", "weight bytes resident across the K-loop"),
               ("replacement_policy", "which resident objects evict"),
               ("lifetime_scope", "loop-invariant vs replan scope")],
        risk="a hardware cache would hide the semantic lifetime the compiler already knows",
        levels=_lv(S_POSSIBLE, S_STRONG, S_STRONG, S_WEAK, S_POSSIBLE, S_NA)),
    "resident_packed_weight_object": dict(
        sources=["numerical_contract", "memory_hierarchy_envelope"],
        region_roles=["repeated_head"], support="lowbit", cp_axis="resident_packed_lowbit_weights",
        erased=True, metadata="packed dtype, pack_format, scale_object_handle, lifetime",
        knobs=[("pack_format", "sub-byte packing layout"),
               ("resident_capacity", "packed resident bytes")],
        risk="packed residency overfits a weight layout; needs a low-bit capture",
        levels=None),
    "packed_lowbit_tensor": dict(
        sources=["numerical_contract", "dtype_capacity_table"], region_roles=["repeated_head"],
        support="lowbit", cp_axis="resident_packed_lowbit_weights", erased=True,
        metadata="storage dtype, pack_format, group_size, scale_object_handle",
        knobs=[("weight_format", "int4/fp8/... packed format"),
               ("group_size", "quant group granularity")],
        risk="format / layout overfit; capture is dequantized f32", levels=None),
    "scale_object": dict(
        sources=["numerical_contract"], region_roles=["repeated_head"], support="lowbit",
        cp_axis="resident_packed_lowbit_weights", erased=True,
        metadata="scale dtype, zero_point, granularity (per-tensor/-channel/-group)",
        knobs=[("scale_granularity", "per-tensor / channel / group"),
               ("zero_point", "asymmetric vs symmetric")],
        risk="scales folded away in the capture; needs the source quant metadata", levels=None),
    "prefix_kv_object": dict(
        sources=["workload_contract_graph", "state_lifetime"], region_roles=["prefix_builder"],
        support="decode", cp_axis="decode_kv_cache_path", erased=False, kv=True,
        metadata="kv dtype, heads, head_dim, kv_len (unavailable — attention lowered)",
        knobs=[("kv_capacity", "resident KV bytes")],
        risk="attention/KV lowered into matmul projections — structure not recovered", levels=None),
    "loop_carried_state_handle": dict(
        sources=["workload_contract_graph", "state_lifetime"], region_roles=["repeated_head"],
        support="k_loop", cp_axis="autonomous_K_loop", erased=False,
        metadata="state dtype, shape, update site, lifetime=across_K",
        knobs=[("state_capacity", "carried-state bytes"),
               ("update_in_loop", "in-loop vs host update")],
        risk="serializes the loop w.r.t. the carried state",
        levels=_lv(S_NA, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG, S_POSSIBLE)),
    "bounded_loop_command": dict(
        sources=["workload_contract_graph", "command_graph", "phase_rate_table"],
        region_roles=["repeated_head"], support="k_loop", cp_axis="autonomous_K_loop", erased=False,
        metadata="trip_count, body_region_handle, loop_carried/invariant handles, event_in/out",
        knobs=[("loop_bound_K", "device-side trip count"), ("unroll_factor", "loop unroll")],
        risk="needs bounded-K semantics + device-side state",
        levels=_lv(S_NA, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG, S_POSSIBLE)),
    "region_level_dispatch": dict(
        sources=["command_graph", "pipeline_envelope"], region_roles=["repeated_head",
        "backbone_once"], support="all", cp_axis="command_batching", erased=False,
        metadata="region_handle, dependency list",
        knobs=[("dispatch_granularity", "op vs region dispatch")],
        risk="requires a command ISA + device scheduler",
        levels=_lv(S_POSSIBLE, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG, S_NA)),
    "persistent_command_buffer": dict(
        sources=["command_graph"], region_roles=["repeated_head"], support="all",
        cp_axis="command_batching", erased=False,
        metadata="command stream, reusable across replans",
        knobs=[("buffer_reuse_scope", "per-replan vs persistent")],
        risk="static dependency graph required at submit time",
        levels=_lv(S_NA, S_POSSIBLE, S_STRONG, S_WEAK, S_STRONG, S_NA)),
    "event_token": dict(
        sources=["pipeline_envelope", "command_graph"], region_roles=["repeated_head",
        "backbone_once"], support="all", cp_axis="async_chunk_overlap", erased=False,
        metadata="event id, producer, consumer",
        knobs=[("event_depth", "in-flight events")],
        risk="requires a device synchronization primitive",
        levels=_lv(S_NA, S_STRONG, S_STRONG, S_WEAK, S_POSSIBLE, S_NA)),
    "async_queue": dict(
        sources=["pipeline_envelope", "dma_stream_table"], region_roles=["repeated_head",
        "backbone_once"], support="all", cp_axis="async_chunk_overlap", erased=False,
        metadata="queue id, depth",
        knobs=[("queue_depth", "async queue depth")],
        risk="requires async submission + completion signalling",
        levels=_lv(S_NA, S_STRONG, S_STRONG, S_WEAK, S_POSSIBLE, S_NA)),
    "producer_consumer_queue": dict(
        sources=["pipeline_envelope"], region_roles=["repeated_head", "backbone_once"],
        support="control_loop", cp_axis="async_chunk_overlap", erased=False,
        metadata="chunk producer, control consumer, depth",
        knobs=[("pc_queue_depth", "producer/consumer slack")],
        risk="only meaningful for a real producer/consumer rate split",
        levels=_lv(S_NA, S_STRONG, S_STRONG, S_WEAK, S_POSSIBLE, S_NA)),
    "double_buffered_action_chunk": dict(
        sources=["pipeline_envelope", "phase_rate_table"], region_roles=["repeated_head"],
        support="control_loop", cp_axis="async_chunk_overlap", erased=False,
        metadata="chunk buffer x2, action_horizon H",
        knobs=[("buffer_count", "double vs triple buffer")],
        risk="only applies to a VLA action chunk consumed at a control rate",
        levels=_lv(S_NA, S_STRONG, S_STRONG, S_NA, S_POSSIBLE, S_NA)),
    "matrix_engine": dict(
        sources=["operator_shape_table", "primitive_coverage_matrix", "resource_pressure_table"],
        region_roles=["repeated_head", "backbone_once"], support="dense", cp_axis=None,
        erased=False, metadata="tile shape, accumulator dtype",
        knobs=[("tile_M", "output tile rows"), ("tile_N", "output tile cols"),
               ("pe_array_dims", "PE grid")],
        risk="square matrix engine over-serves the skinny/GEMV decode shapes",
        levels=_lv(S_POSSIBLE, S_NA, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG)),
    "skinny_gemm_or_gemv_engine": dict(
        sources=["operator_shape_table", "primitive_coverage_matrix"],
        region_roles=["repeated_head"], support="gemv", cp_axis=None, erased=False,
        metadata="lane width, accumulator depth",
        knobs=[("lane_width", "GEMV lane width"), ("num_lanes", "vector lanes")],
        risk="GEMV unit under-serves the large dense GEMM",
        levels=_lv(S_POSSIBLE, S_NA, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG)),
    "epilogue_requant_unit": dict(
        sources=["fusion_epilogue", "resource_pressure_table"], region_roles=["repeated_head",
        "backbone_once"], support="epilogue", cp_axis="fused_requant_epilogue", erased=False,
        metadata="epilogue op set, requant scale, output dtype",
        knobs=[("fused_bias", "bias in epilogue"), ("requant_path", "requant in epilogue")],
        risk="requant numerics need accuracy measurement (the slot is proven, the math is not)",
        levels=_lv(S_POSSIBLE, S_NA, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG)),
    "dma_engine": dict(
        sources=["memory_hierarchy_envelope", "dma_stream_table"], region_roles=["repeated_head",
        "backbone_once"], support="all", cp_axis=None, erased=False,
        metadata="stream descriptors, directions",
        knobs=[("num_channels", "DMA channels"), ("prefetch_depth", "prefetch lookahead")],
        risk="hardware-managed cache would hide reuse the compiler can express",
        levels=_lv(S_WEAK, S_STRONG, S_STRONG, S_POSSIBLE, S_POSSIBLE, S_POSSIBLE)),
    "multi_stream_dma_descriptor": dict(
        sources=["dma_stream_table"], region_roles=["repeated_head"], support="all", cp_axis=None,
        erased=False, metadata="per-stream descriptor (weight/activation/output)",
        knobs=[("num_streams", "independent DMA streams")],
        risk="requires a descriptor format + multi-channel engine",
        levels=_lv(S_NA, S_STRONG, S_STRONG, S_POSSIBLE, S_POSSIBLE, S_NA)),
    "prefetch_descriptor": dict(
        sources=["dma_stream_table", "memory_hierarchy_envelope"], region_roles=["repeated_head"],
        support="all", cp_axis=None, erased=False, metadata="prefetch target, depth, trigger",
        knobs=[("prefetch_depth", "weight/activation prefetch lookahead")],
        risk="device-managed prefetch may diverge from the compiler's reuse plan",
        levels=_lv(S_WEAK, S_STRONG, S_STRONG, S_POSSIBLE, S_STRONG, S_POSSIBLE)),
    "partial_sum_object": dict(
        sources=["sharding_opportunities"], region_roles=["repeated_head", "backbone_once"],
        support="all", cp_axis=None, erased=False,
        metadata="partial dtype (i32), shard count, M×N tile",
        knobs=[("shard_count_K", "K-shards"), ("partial_dtype", "accumulator width")],
        risk="hardware-internal K-sharding hides the reduction from the compiler",
        levels=_lv(S_NA, S_WEAK, S_POSSIBLE, S_STRONG, S_STRONG, S_STRONG)),
    "accumulator_merge": dict(
        sources=["sharding_opportunities"], region_roles=["repeated_head"], support="all",
        cp_axis=None, erased=False, metadata="reduction radix, accumulator width",
        knobs=[("reduction_radix", "merge tree radix")],
        risk="reduction network cost unknown (not measured)",
        levels=_lv(S_NA, S_WEAK, S_POSSIBLE, S_STRONG, S_STRONG, S_STRONG)),
    "accumulator_commit": dict(
        sources=["fusion_epilogue"], region_roles=["repeated_head"], support="all",
        cp_axis="fused_requant_epilogue", erased=False,
        metadata="commit dtype, commit-in-epilogue",
        knobs=[("commit_dtype", "output dtype of the commit")],
        risk="a separate accumulator materialization may exist that the capture hid",
        levels=_lv(S_NA, S_NA, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG)),
    "fused_dequant_matmul": dict(
        sources=["numerical_contract", "fusion_epilogue"], region_roles=["repeated_head"],
        support="lowbit", cp_axis="resident_packed_lowbit_weights", erased=True,
        metadata="storage dtype, compute dtype, dequant-in-load",
        knobs=[("weight_format", "packed format"), ("dequant_in_load", "fuse dequant onto load")],
        risk="needs a low-bit capture; the f32 capture only shows compiler-dequant", levels=None),
    "fused_requant_epilogue": dict(
        sources=["fusion_epilogue"], region_roles=["repeated_head", "backbone_once"],
        support="epilogue", cp_axis="fused_requant_epilogue", erased=False,
        metadata="epilogue op set, requant scale, output dtype",
        knobs=[("requant_in_epilogue", "requant onto the matmul output"),
               ("epilogue_op_set", "bias/activation/requant subset")],
        risk="the epilogue slot is proven (bias); requant accuracy is unmeasured",
        levels=_lv(S_POSSIBLE, S_NA, S_POSSIBLE, S_POSSIBLE, S_POSSIBLE, S_POSSIBLE)),
    "native_lowbit_matmul": dict(
        sources=["numerical_contract", "accuracy_gated_dtype_candidates"],
        region_roles=["repeated_head"], support="lowbit",
        cp_axis="resident_packed_lowbit_weights", erased=True,
        metadata="lhs/rhs storage dtype, accumulator dtype, scale object, output dtype",
        knobs=[("compute_dtype", "int4/fp8 datapath"), ("accumulator_dtype", "i32 accumulate")],
        risk="accuracy + format overfit; needs a low-bit capture + accuracy gate", levels=None),
    "kv_cache_object": dict(
        sources=["workload_contract_graph"], region_roles=["prefix_builder"], support="decode",
        cp_axis="decode_kv_cache_path", erased=False, kv=True,
        metadata="kv dtype, heads, head_dim, growing length (unavailable — attention lowered)",
        knobs=[("kv_capacity", "resident KV bytes")],
        risk="attention/KV not recovered from the flat capture", levels=None),
    "decode_loop_controller": dict(
        sources=["workload_contract_graph", "command_graph", "phase_rate_table"],
        region_roles=["repeated_head"], support="decode", cp_axis="decode_kv_cache_path",
        erased=False, metadata="token trip count, kv handle, loop-carried token state",
        knobs=[("decode_bound", "max decode tokens"), ("kv_update_in_loop", "in-loop KV update")],
        risk="needs bounded decode semantics + (unavailable) KV structure",
        levels=_lv(S_NA, S_POSSIBLE, S_STRONG, S_POSSIBLE, S_STRONG, S_POSSIBLE)),
}

ABSTRACTIONS = tuple(_BOUNDARY_CATALOG)
REGION_ROLES = {"repeated_head", "backbone_once", "prefix_builder", "unknown"}


def catalog_rows() -> list[dict]:
    """Read-only view of the boundary catalog for downstream analysis (e.g. the strict abstraction
    necessity classifier). Exposes the discriminating fields without the level/knob detail; does NOT
    change how certificates or the committed boundary matrix are built."""
    return [
        {"abstraction": name, "support": spec["support"],
         "region_roles": list(spec["region_roles"]),
         "erased": bool(spec.get("erased", False)), "kv": bool(spec.get("kv", False)),
         "cp_axis": spec.get("cp_axis")}
        for name, spec in _BOUNDARY_CATALOG.items()]


def _levels_for(spec: dict) -> dict:
    if spec.get("kv"):
        return dict(_KV_LEVELS)
    if spec.get("erased") and spec.get("levels") is None:
        return dict(_ERASED_LEVELS)
    return spec["levels"]


# ============================================================ certificates + pressure score

@dataclass
class BoundaryCertificate:
    abstraction: str
    source_analyses: list
    supporting_workloads: list
    region_roles: list
    evidence_summary: str
    cp_matrix_axis: object                 # a compiler_proof_matrix axis or None
    required_compiler_proof: str
    compiler_proof_status: str             # proven_for_workload | assumed | unknown | unavailable
    dse_knobs: list                        # [{knob, reason, evidence}]
    boundary_levels: list                  # [{level, status, ...}]
    boundary_pressure_score: int
    pressure_components: dict
    erased: bool
    what_is_not_claimed: str = ("no speedup, cycles, area, or energy claim; a boundary "
                                "search-space candidate, not a chosen design")


def _supporting(spec, evidence_ctx) -> list:
    key = spec["support"]
    if key == "all" or key == "lowbit":     # lowbit opportunity exists for every f32 workload
        return sorted(evidence_ctx)
    return sorted(w for w, f in evidence_ctx.items() if f.get(key))


def _missing_for(status, spec) -> str:
    if status == S_BLOCKED:
        return ("low-bit capture (packed weights + scales) + per-format accuracy"
                if spec.get("erased") else "accuracy / numerical measurement")
    if status == S_UNAVAIL:
        return ("attention/KV structure (a loop-preserving capture)" if spec.get("kv")
                else "structure not recovered from the flat capture")
    if status in (S_STRONG, S_POSSIBLE, S_WEAK):
        return "per-unit throughput / latency / area (a design YAML) to choose this placement"
    return "n/a"


def build_certificate(name, spec, evidence_ctx, cp_proofs) -> BoundaryCertificate:
    levels = _levels_for(spec)
    wls = _supporting(spec, evidence_ctx)
    cp_axis = spec.get("cp_axis")
    if cp_axis and cp_axis in cp_proofs:
        proof_str, proof_status = cp_proofs[cp_axis]
    else:
        proof_str, proof_status = ("unavailable (no compiler_proof_matrix entry)", "unavailable")
    erased = bool(spec.get("erased") or spec.get("kv"))
    if spec.get("kv"):
        ev = f"{spec['risk']}; structure unavailable in the flat capture"
    elif spec.get("erased"):
        ev = f"{spec['risk']}; only the compiler-dequant (f32) path is present"
    else:
        ev = f"supported by {', '.join(spec['sources'])}; suggested in {', '.join(wls) or '—'}"

    blevels = []
    for lv in LEVELS:
        st = levels[lv]
        role = _LEVEL_ROLE[lv]
        crosses = lv != L_COMPILER and st not in (S_NA, S_UNAVAIL)
        per_proof = ("none (generic ops)" if lv == L_COMPILER
                     else proof_str if st not in (S_NA, S_UNAVAIL) else "n/a")
        risk = spec["risk"]
        if lv in (L_ISA, L_DATAPATH) and st in (S_STRONG, S_POSSIBLE):
            risk = risk + "; highest overfit risk at a fixed/native level"
        blevels.append({
            "level": lv, "status": st,
            "software_manages": role["sw"], "hardware_manages": role["hw"],
            "required_compiler_proof": per_proof,
            "required_runtime_support": role["runtime"],
            "required_isa_semantics": role["isa"],
            "required_hw_support": role["hw_support"],
            "metadata_crossing": (spec["metadata"] if crosses else "n/a"),
            "risk": risk, "missing_evidence": _missing_for(st, spec)})

    # boundary_pressure_score = EVIDENCE BREADTH (not performance / priority / benefit)
    comp = {
        "n_supporting_workloads": len(wls),
        "n_region_roles": len(spec["region_roles"]),
        "crosses_rate_boundary": int("backbone_once" in spec["region_roles"]
                                     or spec["support"] in ("control_loop", "decode")),
        "in_repeated_loop": int("repeated_head" in spec["region_roles"]),
        "compiler_provable": int(proof_status == "proven_for_workload"),
        "overfit_penalty": -int(levels.get(L_DATAPATH) == S_STRONG),
        "missing_evidence_penalty": -int(erased),
    }
    score = sum(comp.values())

    knobs = [{"knob": k, "reason": r,
              "evidence": ("recovered_from_ir" if not erased else "unavailable")}
             for k, r in spec["knobs"]]
    return BoundaryCertificate(
        abstraction=name, source_analyses=spec["sources"], supporting_workloads=wls,
        region_roles=spec["region_roles"], evidence_summary=ev, cp_matrix_axis=cp_axis,
        required_compiler_proof=proof_str, compiler_proof_status=proof_status, dse_knobs=knobs,
        boundary_levels=blevels, boundary_pressure_score=score, pressure_components=comp,
        erased=erased)


def build_certificates(evidence_ctx, cp_proofs) -> list[BoundaryCertificate]:
    return [build_certificate(n, s, evidence_ctx, cp_proofs)
            for n, s in _BOUNDARY_CATALOG.items()]


# ============================================================ responsibility split matrix

_RESP_FUNCTIONS = [
    "region_partitioning", "layout_selection", "dtype_selection", "weight_packing",
    "scale_metadata_management", "resident_object_lifetime", "K_loop_iteration",
    "loop_carried_state_update", "command_dependency_tracking", "event_synchronization",
    "DMA_prefetch", "buffer_allocation", "sharding_split", "partial_sum_merge",
    "epilogue_requant", "deadline_enforcement", "safety_action_commit",
]
_RESP_COLUMNS = ["compiler", "runtime_hal", "command_processor", "accelerator_isa",
                 "device_microcode", "datapath"]
# rows: (compiler, runtime_hal, command_processor, accelerator_isa, device_microcode, datapath)
_RESP_MATRIX = {
    "region_partitioning": (R_OWNS, R_ASSISTS, R_NA, R_NA, R_NA, R_NA),
    "layout_selection": (R_OWNS, R_ASSISTS, R_NA, R_CONSUMES, R_NA, R_CONSUMES),
    "dtype_selection": (R_OWNS, R_ASSISTS, R_NA, R_CONSUMES, R_NA, R_CONSUMES),
    "weight_packing": (R_OWNS, R_ASSISTS, R_NA, R_CONSUMES, R_NA, R_CONSUMES),
    "scale_metadata_management": (R_DECLARES, R_OWNS, R_CONSUMES, R_CONSUMES, R_NA, R_CONSUMES),
    "resident_object_lifetime": (R_DECLARES, R_OWNS, R_OWNS, R_NA, R_ASSISTS, R_NA),
    "K_loop_iteration": (R_ASSISTS, R_ASSISTS, R_OWNS, R_NA, R_OWNS, R_OWNS),
    "loop_carried_state_update": (R_DECLARES, R_ASSISTS, R_OWNS, R_NA, R_OWNS, R_OWNS),
    "command_dependency_tracking": (R_DECLARES, R_ASSISTS, R_OWNS, R_NA, R_OWNS, R_NA),
    "event_synchronization": (R_NA, R_OWNS, R_OWNS, R_NA, R_ASSISTS, R_NA),
    "DMA_prefetch": (R_ASSISTS, R_OWNS, R_OWNS, R_NA, R_ASSISTS, R_ASSISTS),
    "buffer_allocation": (R_ASSISTS, R_OWNS, R_OWNS, R_NA, R_ASSISTS, R_NA),
    "sharding_split": (R_OWNS, R_ASSISTS, R_ASSISTS, R_CONSUMES, R_ASSISTS, R_ASSISTS),
    "partial_sum_merge": (R_DECLARES, R_NA, R_ASSISTS, R_OWNS, R_OWNS, R_OWNS),
    "epilogue_requant": (R_ASSISTS, R_NA, R_ASSISTS, R_OWNS, R_OWNS, R_OWNS),
    "deadline_enforcement": (R_NA, R_OWNS, R_ASSISTS, R_NA, R_ASSISTS, R_NA),
    "safety_action_commit": (R_NA, R_OWNS, R_ASSISTS, R_NA, R_NA, R_NA),
}
# functions whose evidence the flat capture does not carry (runtime/IO) -> noted, not invented.
_RESP_NOTE = {
    "deadline_enforcement": "runtime concern; the deadline is derived from H/control_rate, "
                            "enforcement timing is unavailable in a model-forward capture",
    "safety_action_commit": "post-processing/runtime; not in the model-forward capture",
    "scale_metadata_management": "scales erased by the dequantized capture",
}


def responsibility_rows() -> list[dict]:
    out = []
    for fn in _RESP_FUNCTIONS:
        cells = dict(zip(_RESP_COLUMNS, _RESP_MATRIX[fn]))
        out.append({"function": fn, **cells, "note": _RESP_NOTE.get(fn, "")})
    return out


# ============================================================ emitters

def _lvl_status(cert: BoundaryCertificate, level: str) -> str:
    return next(b["status"] for b in cert.boundary_levels if b["level"] == level)


def hw_sw_boundary_matrix_csv(certs) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for c in certs:
        row = {"abstraction": c.abstraction}
        for lv in LEVELS:
            row[lv] = _lvl_status(c, lv)
        row["supporting_workloads"] = "; ".join(c.supporting_workloads) or "—"
        row["boundary_pressure_score"] = c.boundary_pressure_score
        rows.append(row)
    rows.sort(key=lambda r: -r["boundary_pressure_score"])
    return _csv(rows, ["abstraction"] + LEVELS + ["supporting_workloads",
                                                  "boundary_pressure_score"])


def boundary_candidate_contracts_yaml(certs) -> dict:
    return {"boundary_candidate_contracts": {
        "note": "HW/SW boundary-placement certificates — where each workload-implied abstraction "
                "could live (compiler / runtime-HAL / command-ISA / accelerator-ISA / microcode / "
                "datapath), what each side manages, the compiler proof and runtime/ISA/HW support "
                "required, and the missing evidence. Merlin emits the search space; the DSE tool "
                "chooses. No speedup/cycles/area/energy and no chosen design claimed. boundary_pressure_score "
                "is EVIDENCE BREADTH, not performance/priority.",
        "boundary_levels_vocabulary": LEVELS,
        "status_vocabulary": sorted(STATUS),
        "certificates": [
            {"abstraction": c.abstraction, "source_analyses": c.source_analyses,
             "supporting_workloads": c.supporting_workloads, "region_roles": c.region_roles,
             "evidence_summary": c.evidence_summary, "compiler_proof_matrix_axis": c.cp_matrix_axis,
             "required_compiler_proof": c.required_compiler_proof,
             "compiler_proof_status": c.compiler_proof_status,
             "boundary_pressure_score": c.boundary_pressure_score,
             "pressure_components": c.pressure_components,
             "dse_knobs": c.dse_knobs, "what_is_not_claimed": c.what_is_not_claimed,
             "boundary_levels": c.boundary_levels}
            for c in certs]}}


def responsibility_split_csv(rows) -> str:
    from merlin.dse_guidance.corpus import _csv
    return _csv(rows, ["function"] + _RESP_COLUMNS + ["note"])


def _cands_at(certs, level, statuses) -> list:
    return [c for c in certs if _lvl_status(c, level) in statuses]


def runtime_object_candidates_yaml(certs) -> dict:
    cs = _cands_at(certs, L_RUNTIME, {S_STRONG, S_POSSIBLE})
    return {"runtime_object_candidates": {
        "note": "abstractions whose runtime/HAL-object placement is a candidate (lifetime + layout "
                "managed by the runtime). Sketch only — no speedup claimed.",
        "candidates": [
            {"object": c.abstraction, "status": _lvl_status(c, L_RUNTIME),
             "metadata": next(b["metadata_crossing"] for b in c.boundary_levels
                              if b["level"] == L_RUNTIME),
             "supporting_workloads": c.supporting_workloads,
             "required_runtime_support": "object ABI + lifetime tracking + metadata"}
            for c in cs]}}


def command_isa_candidates_yaml(certs) -> dict:
    cs = _cands_at(certs, L_COMMAND, {S_STRONG, S_POSSIBLE})
    return {"command_isa_candidates": {
        "note": "abstractions whose command-buffer / command-ISA placement is a candidate. Sketch "
                "only — no speedup claimed.",
        "candidates": [
            {"command": c.abstraction, "status": _lvl_status(c, L_COMMAND),
             "supporting_workloads": c.supporting_workloads,
             "required_isa_semantics": "command opcodes + handles + event/queue"}
            for c in cs]}}


def isa_candidate_primitives_yaml(certs) -> dict:
    cs = _cands_at(certs, L_ISA, {S_STRONG, S_POSSIBLE})
    return {"isa_candidate_primitives": {
        "note": "abstractions whose accelerator-ISA placement is a candidate (a semantic "
                "instruction the datapath executes). Sketch only — no speedup claimed; "
                "blocked/erased numerical primitives are excluded until a low-bit capture exists.",
        "primitives": [
            {"primitive": c.abstraction, "status": _lvl_status(c, L_ISA),
             "supporting_workloads": c.supporting_workloads,
             "required_compiler_proof": c.required_compiler_proof,
             "metadata": c.boundary_levels[0]["metadata_crossing"] if False else None}
            for c in cs]}}


def boundary_dse_knobs_yaml(certs) -> dict:
    knobs = []
    for c in certs:
        strongest = max(c.boundary_levels,
                        key=lambda b: {"strong_candidate": 3, "possible": 2, "weak_candidate": 1}
                        .get(b["status"], 0))
        for k in c.dse_knobs:
            knobs.append({"knob": k["knob"], "abstraction": c.abstraction,
                          "boundary_level": strongest["level"], "reason": k["reason"],
                          "evidence": k["evidence"]})
    return {"boundary_dse_knobs": {
        "note": "DSE knobs created by the boundary-placement options, each with the abstraction it "
                "comes from, the strongest candidate boundary level, a reason, and evidence. "
                "Search-space dimensions only — no speedup/priority/benefit claimed.",
        "knobs": knobs}}


def interface_contract_sketches_md(certs) -> str:
    L = ["# Interface contract sketches (HAL / command / ISA)\n",
         "> Possible high-level interface sketches the workload evidence suggests. **These are "
         "sketches, not a final ISA/HAL design**, and make no speedup/area claim. The DSE tool "
         "would refine and choose.\n"]
    L.append("## Runtime / HAL object sketch — `resident_weight_object`\n")
    L.append("- fields: `dtype`, `shape`, `layout`, `lifetime`, `size_bytes`, "
             "`scale_object_handle` (if quantized)")
    L.append("- operations: `load`, `pin`, `reuse`, `evict`")
    L.append("- evidence: weights are loop-invariant across the K-loop "
             "(`resident_action_head_weights` = proven_for_workload)\n")
    L.append("## Command ISA sketch — `bounded_loop_command`\n")
    L.append("- fields: `trip_count`, `body_region_handle`, `loop_carried_state_handles`, "
             "`invariant_state_handles`, `event_in`, `event_out`")
    L.append("- evidence: the repeated head is a bounded K-loop with loop-invariant weights "
             "(`autonomous_K_loop` = assumed)\n")
    L.append("## Accelerator ISA primitive sketch — `matmul_packed_lowbit`\n")
    L.append("- fields: `lhs_dtype`, `rhs_storage_dtype`, `accumulator_dtype`, `scale_object`, "
             "`output_dtype`, `tile_shape`, `epilogue_mode`")
    L.append("- evidence: **blocked** — the capture is dequantized f32; this primitive needs a "
             "low-bit capture (packed layout + scales) + per-format accuracy before it is a "
             "candidate (`resident_packed_lowbit_weights` = unknown)\n")
    L.append("## Accelerator ISA primitive sketch — `gemv_dot_lanes`\n")
    L.append("- fields: `lane_width`, `num_lanes`, `accumulator_depth`, `reduction_tree_width`")
    L.append("- evidence: GEMV/skinny shapes dominate the decode workloads (P5 geometry); a square "
             "matrix engine covers them poorly (P5 regret)\n")
    L.append("These sketches correspond to the `runtime_object_candidates.yaml`, "
             "`command_isa_candidates.yaml`, and `isa_candidate_primitives.yaml` lists.\n")
    return "\n".join(L)


def boundary_report_md(certs, resp_rows) -> str:
    strong = []
    plausible_all = []      # compiler + HAL + command + (isa or microcode or datapath) all plausible
    for c in certs:
        sts = {b["level"]: b["status"] for b in c.boundary_levels}
        n_strong = sum(1 for s in sts.values() if s == S_STRONG)
        if n_strong >= 1 and not c.erased:
            strong.append(c)
        cand = {S_STRONG, S_POSSIBLE, S_WEAK}
        if (sts[L_COMPILER] in cand and sts[L_RUNTIME] in cand and sts[L_COMMAND] in cand
                and (sts[L_ISA] in cand or sts[L_MICROCODE] in cand or sts[L_DATAPATH] in cand)):
            plausible_all.append(c.abstraction)
    strong.sort(key=lambda c: -c.boundary_pressure_score)
    L = ["# HW/SW boundary-placement report\n",
         "> The boundary search space the workload evidence implies: for each abstraction, where it "
         "could sit and what each placement requires. **Merlin generates the options; the DSE tool "
         "chooses. No speedup / cycles / area / energy and no chosen design is claimed.** "
         "`boundary_pressure_score` is evidence breadth, not performance.\n"]
    L.append("## Strongly-suggested boundary placements (by evidence breadth)\n")
    L.append("| abstraction | top level(s) | supporting workloads | pressure (evidence) |")
    L.append("|---|---|---|---|")
    for c in strong[:10]:
        tops = ", ".join(b["level"] for b in c.boundary_levels if b["status"] == S_STRONG)
        L.append(f"| {c.abstraction} | {tops} | {', '.join(c.supporting_workloads)} | "
                 f"{c.boundary_pressure_score} |")
    L.append("")
    L.append("## Where all software/hardware placements are plausible (the genuine design axes)\n")
    L.append(", ".join(f"`{a}`" for a in plausible_all) or "—")
    L.append("")
    L.append("## Software-only management may explode command count\n")
    L.append("- `bounded_loop_command` / `region_level_dispatch`: a pure host loop submits a "
             "command per step (K×matmuls dispatches); a command buffer or device controller would "
             "remove the host re-dispatch. **requires command/ISA semantics.**")
    L.append("- `multi_stream_dma_descriptor`: software-issued per-tile DMA explodes; a descriptor "
             "engine batches it.")
    L.append("")
    L.append("## Hardware-only management may hide semantics the compiler knows\n")
    L.append("- `resident_weight_object`: a hardware cache rediscovers reuse the compiler already "
             "proved (loop-invariant weights) — a `resident_weight_object` keeps the lifetime "
             "explicit.")
    L.append("- `partial_sum_object` / `accumulator_merge`: hardware-internal K-sharding hides the "
             "reduction; a command/ISA-level shard keeps it visible to the compiler.")
    L.append("")
    L.append("## ISA / HAL objects the evidence suggests\n")
    L.append("- runtime objects: see `runtime_object_candidates.yaml`; command ops: "
             "`command_isa_candidates.yaml`; ISA primitives: `isa_candidate_primitives.yaml`; "
             "sketches: `interface_contract_sketches.md`.")
    L.append("")
    L.append("## Blocked / unavailable (honest)\n")
    blocked = [c.abstraction for c in certs if c.erased]
    L.append(f"- {', '.join('`'+a+'`' for a in blocked)} — packed low-bit / scale / KV structure "
             "is erased or lowered in the capture; these placements are `blocked`/`unavailable` "
             "until a low-bit (packed weights + scales) or loop-preserving capture exists.")
    L.append("")
    L.append("## Missing measurements before choosing a boundary\n")
    L.append("- per-unit throughput / latency / area / energy (a design YAML), per-format low-bit "
             "accuracy, per-phase timing, and host command/sync latency — named per certificate. "
             "Merlin does not choose; it bounds the search space.\n")
    return "\n".join(L)
