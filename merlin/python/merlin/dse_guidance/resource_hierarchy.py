"""Hierarchical resource analysis — which processing-unit shapes the workloads imply.

This maps the recovered operator geometry (P5) and the inter-op concurrency (P7-a) onto *candidate*
hardware-hierarchy options and processing-unit types, with explicit evidence for and against each.
It answers "when should a future DSE tool explore one bigger unit, multiple identical units, or
specialized units" — as a **structural** recommendation about what the search space should contain,
never a performance ranking. Nothing here is a speedup, cycle, or area claim.

Two honest boundaries hold throughout: classes with no visible operators (attention reduction, conv)
are reported with their evidence_against = ``unavailable`` rather than dropped, and every knob is a
search-space dimension, not a chosen value.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from merlin.dse_guidance import shape_taxonomy as ST
from merlin.dse_guidance.design_envelope import E_DERIVED, E_FQN, E_IR, E_NA

# The full P7-c hierarchy vocabulary (compute-shape engines + structural units).
HIER_UNITS = {"matrix_tile_engine", "skinny_gemm_or_gemv_engine", "reduction_tree", "systolic_array",
              "SIMD_vector_lanes", "multi_engine_cluster", "epilogue_unit", "DMA_engine",
              "loop_controller"}

# --- resource classes (what kind of work) ---
RC_DENSE = "dense_gemm"
RC_SKINNY = "skinny_gemm_or_gemv"
RC_ATTN = "attention_softmax_or_reduction"
RC_CONV = "conv_or_patch_embed"
RC_EPILOGUE = "epilogue_or_requant"
RC_DMA = "dma_or_memory"
RC_CONTROL = "control_or_dispatch"
RC_SPARSE = "sparse_or_skip"

# geometry shape_class -> primary compute resource class
_SHAPE_RC = {
    ST.SQUAREISH: RC_DENSE, ST.PROJECTION: RC_DENSE,
    ST.GEMV: RC_SKINNY, ST.TALL_SKINNY: RC_SKINNY, ST.WIDE_SKINNY: RC_SKINNY,
}

# shape_class -> candidate hierarchy unit options (P7-c vocabulary)
_SHAPE_HIER = {
    ST.SQUAREISH: ["matrix_tile_engine", "systolic_array"],
    ST.PROJECTION: ["matrix_tile_engine", "systolic_array"],
    ST.WIDE_SKINNY: ["matrix_tile_engine", "SIMD_vector_lanes"],
    ST.TALL_SKINNY: ["matrix_tile_engine", "SIMD_vector_lanes"],
    ST.GEMV: ["skinny_gemm_or_gemv_engine", "SIMD_vector_lanes"],
}


@dataclass
class ClusterHierarchy:
    shape_class: str
    op_count: int
    macs: int
    mac_fraction: float
    workloads: list
    hierarchy_options: list
    evidence_for: str
    evidence_against: str


def _op_rc(shape) -> str:
    return _SHAPE_RC.get(shape.shape_class, RC_SPARSE)


def cluster_hierarchy(all_shapes) -> list[ClusterHierarchy]:
    total = sum(s.macs for s in all_shapes) or 1
    groups: dict[str, list] = {}
    for s in all_shapes:
        groups.setdefault(s.shape_class, []).append(s)
    out = []
    for cls, ss in groups.items():
        macs = sum(s.macs for s in ss)
        wls = sorted({s.workload for s in ss})
        out.append(ClusterHierarchy(
            shape_class=cls, op_count=len(ss), macs=macs, mac_fraction=round(macs / total, 4),
            workloads=wls, hierarchy_options=_SHAPE_HIER.get(cls, ["multi_engine_cluster"]),
            evidence_for=f"{len(ss)} ops, {macs/total:.1%} of MACs, in {', '.join(wls)}",
            evidence_against=("very small MAC share" if macs / total < 0.02 else "")))
    out.sort(key=lambda c: -c.macs)
    return out


# --------------------------------------------------------------------------- resource pressure

@dataclass
class ResourcePressure:
    resource_class: str
    op_count: int
    macs: int
    mac_fraction: float
    workloads: list
    present: bool
    basis: str               # compute_macs | op_count | structural | unavailable


def resource_pressure(all_shapes, dags) -> list[ResourcePressure]:
    total = sum(s.macs for s in all_shapes) or 1
    by_rc: dict[str, list] = {}
    for s in all_shapes:
        by_rc.setdefault(_op_rc(s), []).append(s)
    rows: list[ResourcePressure] = []

    def add(rc, ss, basis):
        macs = sum(s.macs for s in ss)
        rows.append(ResourcePressure(
            resource_class=rc, op_count=len(ss), macs=macs,
            mac_fraction=round(macs / total, 4), workloads=sorted({s.workload for s in ss}),
            present=bool(ss), basis=basis))

    add(RC_DENSE, by_rc.get(RC_DENSE, []), "compute_macs")
    add(RC_SKINNY, by_rc.get(RC_SKINNY, []), "compute_macs")
    # epilogue: ops carrying a fused bias/activation (addmm) — counted by op, MACs ride on the GEMM
    epi = [s for s in all_shapes if s.epilogue]
    add(RC_EPILOGUE, epi, "op_count")
    # structural classes (not MAC mass): DMA for resident weights, control for the K-loop
    dma_wls = sorted({d.workload for d in dags})        # every workload reuses resident weights
    rows.append(ResourcePressure(RC_DMA, 0, 0, 0.0, dma_wls, True, "structural"))
    ctrl_wls = sorted({d.workload for d in dags if d.total_ops})
    rows.append(ResourcePressure(RC_CONTROL, 0, 0, 0.0, ctrl_wls, True, "structural"))
    # honestly-absent classes
    rows.append(ResourcePressure(RC_ATTN, 0, 0, 0.0, [], False, "unavailable"))
    rows.append(ResourcePressure(RC_CONV, 0, 0, 0.0, [], False, "unavailable"))
    rows.append(ResourcePressure(RC_SPARSE, 0, 0, 0.0, [], False, "unavailable"))
    return rows


# --------------------------------------------------------------------------- processing units

# candidate unit -> (resource classes it serves, dse knobs, compiler proof needed)
_UNIT_CATALOG = [
    ("matrix_engine", [RC_DENSE],
     ["tile_M", "tile_N", "tile_K", "pe_array_rows", "pe_array_cols"],
     "matmul tiling is shape-legal for the dense operators (no cross-tile dependency)"),
    ("skinny_gemm_or_gemv_engine", [RC_SKINNY],
     ["lane_width", "num_lanes", "accumulator_depth"],
     "the skinny/GEMV operators vectorize along the long output dimension"),
    ("attention_kv_engine", [RC_ATTN],
     ["kv_tile", "head_parallelism"],
     "requires attention structure (heads / KV) which is not recoverable from the flat capture"),
    ("conv_engine", [RC_CONV],
     ["spatial_tile", "channel_tile"],
     "requires conv structure which is absent from the captures"),
    ("epilogue_requant_unit", [RC_EPILOGUE],
     ["fused_bias", "fused_activation", "requant_path"],
     "the addmm epilogue (bias / activation) fuses onto the GEMM output"),
    ("dma_engine", [RC_DMA],
     ["prefetch_depth", "resident_capacity", "tiling_of_residency"],
     "loop-invariant weights are read-only across the K-loop and can be made resident"),
    ("loop_controller", [RC_CONTROL],
     ["loop_bound_K", "unroll_factor"],
     "the repeated head is a bounded K-loop with loop-invariant weights"),
    ("scalar_control_unit", [RC_CONTROL],
     ["dispatch_width"],
     "host dispatch / control flow drives the per-step command stream"),
]


@dataclass
class UnitCandidate:
    unit: str
    evidence_for: str
    evidence_against: str
    workloads: list
    region_roles: list
    dse_knobs: list
    compiler_proof_needed: str
    what_is_not_claimed: str


def processing_unit_candidates(all_shapes, pressure: list[ResourcePressure]) -> list[UnitCandidate]:
    rc_row = {p.resource_class: p for p in pressure}
    roles_by_rc: dict[str, set] = {}
    for s in all_shapes:
        roles_by_rc.setdefault(_op_rc(s), set()).add(s.region_role)
    out = []
    for unit, rcs, knobs, proof in _UNIT_CATALOG:
        served = [rc_row[rc] for rc in rcs if rc in rc_row]
        present = [p for p in served if p.present]
        wls = sorted({w for p in present for w in p.workloads})
        roles = sorted({r for rc in rcs for r in roles_by_rc.get(rc, set())})
        if present and any(p.basis == "compute_macs" for p in present):
            frac = sum(p.mac_fraction for p in present)
            for_ev = f"serves {', '.join(rcs)}: {frac:.0%} of MACs across {', '.join(wls)}"
            against = "" if frac >= 0.05 else "small MAC share — may not warrant a dedicated unit"
        elif present:
            n = sum(p.op_count for p in present)
            for_ev = (f"serves {', '.join(rcs)} ({served[0].basis}; "
                      f"{n or 'structural'} across {', '.join(wls) or 'all workloads'})")
            against = ""
        else:
            for_ev = "no supporting operators in the captures"
            against = "unavailable — the structure this unit needs is not in the flat capture"
        out.append(UnitCandidate(
            unit=unit, evidence_for=for_ev, evidence_against=against, workloads=wls,
            region_roles=roles, dse_knobs=knobs, compiler_proof_needed=proof,
            what_is_not_claimed="no speedup, throughput, cycle, or area is claimed; this is a "
                                "structural search-space candidate, not a chosen design"))
    return out


# --------------------------------------------------------------------------- structural hierarchy

@dataclass
class StructuralHint:
    hierarchy_option: str
    evidence_for: str
    supported_workloads: list
    evidence: str
    dse_knobs: list
    required_compiler_proof: str
    missing_measurements: list


def structural_hierarchy_hints(all_shapes, all_axes, dags) -> list[StructuralHint]:
    """Structural hierarchy units the rest of P7 implies but the shape-cluster map doesn't surface:
    reduction_tree (K-sharding), epilogue_unit (addmm), DMA_engine + loop_controller (the resident-
    weight K-loop), and multi_engine_cluster (inter-op parallelism). Each is grounded in recovered
    structure or marked unavailable — never invented."""
    hints: list[StructuralHint] = []

    kshard = [a for a in all_axes if a.axis == "K" and a.shardable[8]]
    kw = sorted({a.workload for a in kshard})
    hints.append(StructuralHint(
        "reduction_tree",
        f"{len(kshard)} K-shardable ops produce partial sums that must be merged across shards",
        kw, E_IR if kshard else E_NA, ["reduction_radix", "accumulator_width"],
        "K-sharding produces partial sums that a reduction tree merges", ["reduction latency"]))

    epi = [s for s in all_shapes if s.epilogue]
    ew = sorted({s.workload for s in epi})
    hints.append(StructuralHint(
        "epilogue_unit",
        f"{len(epi)} ops carry a fused bias/activation (addmm epilogue)",
        ew, E_IR if epi else E_NA, ["fused_bias", "fused_activation", "requant_path"],
        "the addmm epilogue fuses onto the GEMM output", ["epilogue throughput"]))

    headw = sorted({s.workload for s in all_shapes if s.region_role == "repeated_head"})
    hints.append(StructuralHint(
        "DMA_engine",
        "loop-invariant weights are read-only across the repeated head (reused every step)",
        headw, E_FQN if headw else E_NA, ["prefetch_depth", "resident_capacity"],
        "loop-invariant weights are read-only across the K-loop", ["DRAM bandwidth"]))
    hints.append(StructuralHint(
        "loop_controller",
        "the repeated head is a bounded K-loop driving the per-step command stream",
        headw, E_FQN if headw else E_NA, ["loop_bound_K", "unroll_factor"],
        "the repeated head runs K times per replan", []))

    par = [d.workload for d in dags if d.serialization != "mostly_sequential"]
    hints.append(StructuralHint(
        "multi_engine_cluster",
        (f"inter-op parallelism in {', '.join(par)} (independent ops become ready together)" if par
         else "no workload exposes > 1.5x inter-op parallelism — limited evidence for many engines"),
        par, E_DERIVED if par else E_NA, ["num_engines"],
        "independent operators in the DAG can occupy separate engines",
        ["scheduling / communication latency"]))
    return hints


# --------------------------------------------------------------------------- emitters

def cluster_to_hierarchy_csv(clusters: list[ClusterHierarchy]) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = [{"shape_class": c.shape_class, "op_count": c.op_count, "macs": c.macs,
             "mac_fraction": c.mac_fraction, "hierarchy_options": "; ".join(c.hierarchy_options),
             "workloads": "; ".join(c.workloads), "evidence_for": c.evidence_for,
             "evidence_against": c.evidence_against or "—"} for c in clusters]
    return _csv(rows, ["shape_class", "op_count", "macs", "mac_fraction", "hierarchy_options",
                       "workloads", "evidence_for", "evidence_against"])


def resource_pressure_csv(pressure: list[ResourcePressure]) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = [{"resource_class": p.resource_class, "present": p.present, "op_count": p.op_count,
             "macs": p.macs, "mac_fraction": p.mac_fraction, "basis": p.basis,
             "workloads": "; ".join(p.workloads) or "—"} for p in pressure]
    return _csv(rows, ["resource_class", "present", "op_count", "macs", "mac_fraction", "basis",
                       "workloads"])


def hierarchy_hints_yaml(clusters: list[ClusterHierarchy],
                         structural: list[StructuralHint] | None = None) -> dict:
    return {"parallel_hierarchy_hints": {
        "note": "operator clusters + structural units mapped to candidate hardware-hierarchy "
                "options. Structural hints for a future DSE search space — NOT a chosen design, "
                "NOT a speedup. compute-shape clusters cover the GEMM/GEMV engines; structural_units "
                "cover reduction_tree (K-sharding), epilogue_unit, DMA_engine, loop_controller, and "
                "multi_engine_cluster (inter-op parallelism).",
        "clusters": [
            {"shape_class": c.shape_class,
             "hierarchy_options": c.hierarchy_options,
             "evidence_for": c.evidence_for,
             "evidence_against": c.evidence_against or "none",
             "supported_workloads": c.workloads,
             "dse_knobs_exposed": ["unit_count", "unit_shape", "tile_dims"],
             "required_compiler_proof": "operator geometry is tile/lane-legal for the option "
                                        "(recovered_from_ir shapes)",
             "missing_measurements": ["per-unit throughput", "communication latency",
                                      "energy/area"],
             "evidence": E_IR}
            for c in clusters],
        "structural_units": [
            {"hierarchy_option": h.hierarchy_option,
             "evidence_for": h.evidence_for,
             "supported_workloads": h.supported_workloads,
             "dse_knobs_exposed": h.dse_knobs,
             "required_compiler_proof": h.required_compiler_proof,
             "missing_measurements": h.missing_measurements,
             "evidence": h.evidence}
            for h in (structural or [])]}}


def processing_unit_candidates_yaml(units: list[UnitCandidate]) -> dict:
    return {"processing_unit_candidates": {
        "note": "candidate processing-unit types implied by the resource pressure. Each carries "
                "evidence for/against; units whose structure is not in the capture are marked "
                "unavailable, not dropped. Knobs are search-space dimensions, not chosen values. "
                "No speedup/cycle/area claim.",
        "units": [
            {"unit": u.unit, "evidence_for": u.evidence_for,
             "evidence_against": u.evidence_against or "none",
             "workloads_supporting": u.workloads,
             "region_roles_supporting": u.region_roles,
             "dse_knobs_exposed": u.dse_knobs,
             "compiler_proof_needed": u.compiler_proof_needed,
             "what_is_not_claimed": u.what_is_not_claimed,
             "evidence": (E_IR if u.workloads else E_NA)}
            for u in units]}}


def processing_unit_report_md(units: list[UnitCandidate], pressure: list[ResourcePressure],
                              clusters: list[ClusterHierarchy], dags) -> str:
    dense = next((p for p in pressure if p.resource_class == RC_DENSE), None)
    skinny = next((p for p in pressure if p.resource_class == RC_SKINNY), None)
    avg_par = round(sum(d.available_parallelism for d in dags) / len(dags), 2) if dags else 0
    L = ["# Processing-unit & hierarchy guidance\n",
         "> Which processing-unit shapes the workloads imply, and whether the evidence favors one "
         "bigger unit, multiple identical units, or specialized units. **Structural search-space "
         "guidance — no speedup, cycle, or area claim.**\n"]
    L.append("## Resource pressure (where the work is)\n")
    L.append("| resource class | present | ops | MAC share | basis |")
    L.append("|---|---|---|---|---|")
    for p in pressure:
        L.append(f"| {p.resource_class} | {p.present} | {p.op_count} | "
                 f"{p.mac_fraction:.1%} | {p.basis} |")
    L.append("")
    L.append("## One bigger unit vs. many identical vs. specialized\n")
    L.append(f"- **Average inter-op parallelism is low ({avg_par}×)** (see "
             f"`dag_parallelism_report.md`): the dependency DAG is near-sequential, so **many "
             f"identical units would be hard to keep busy** by inter-op concurrency alone — the "
             f"parallelism to exploit is *intra-op* sharding.")
    if dense and skinny:
        L.append(f"- **Compute splits across two shapes:** dense GEMM is {dense.mac_fraction:.0%} of "
                 f"MACs ({', '.join(dense.workloads)}) while skinny/GEMV is {skinny.mac_fraction:.0%} "
                 f"({', '.join(skinny.workloads)}). This favors **specialized units** (a matrix "
                 f"engine *and* a GEMV/vector engine) over one universal unit.")
    L.append("- **Plus structural units:** an `epilogue_requant_unit` (the addmm bias/activation "
             "fuses onto the GEMM), a `dma_engine` (resident loop-invariant weights), and a "
             "`loop_controller` (the bounded K-loop) — each backed by recovered structure.")
    L.append("- **Honestly absent:** `attention_kv_engine` and `conv_engine` have no supporting "
             "operators in the captures (attention is lowered, no conv) — listed `unavailable`.")
    L.append("")
    L.append("## Candidate units\n")
    L.append("| unit | evidence_for | supporting workloads |")
    L.append("|---|---|---|")
    for u in units:
        L.append(f"| {u.unit} | {u.evidence_for} | {', '.join(u.workloads) or '—'} |")
    L.append("\n**Caveat (structural, not realized):** resource pressure and unit candidates are "
             "structural. They are **not a speedup**, throughput, cycle, or area claim; the missing "
             "measurements (per-unit throughput, communication latency, energy/area) are named per "
             "unit in `processing_unit_candidates.yaml`.\n")
    return "\n".join(L)
