"""Processing-unit multiplicity guidance — monolithic vs. replicated vs. heterogeneous.

This synthesizes the P7 resource pressure, inter-op parallelism, and sharding facts into the
explicit search-space question: does the workload evidence point toward **one bigger unit**,
**multiple identical units**, or **multiple specialized units**? It emits evidence for/against each
option (and what blocks the replicated case) — it does **not** select an architecture. The language
is "structurally suggests" / "future DSE should consider", never "best" or "optimal".

It reuses the P7 resource-pressure classes and parallelism results; nothing is re-measured here.
No speedup, cycle, or area claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import resource_hierarchy as RH
from merlin.dse_guidance.design_envelope import E_DERIVED, E_NA


@dataclass
class MultiplicityOption:
    option: str                       # one_bigger_unit | multiple_identical_units | multiple_specialized_units
    evidence_for: list
    evidence_against: list            # for the blocked case, the blockers
    candidate_units: list = field(default_factory=list)


def _pressure(pressure, rc):
    return next((p for p in pressure if p.resource_class == rc), None)


def guidance(all_shapes, dags, pressure, all_axes) -> dict:
    avg_par = round(sum(d.available_parallelism for d in dags) / len(dags), 3) if dags else 0.0
    n_seq = sum(1 for d in dags if d.serialization == "mostly_sequential")
    n_par = len(dags) - n_seq
    dense = _pressure(pressure, RH.RC_DENSE)
    skinny = _pressure(pressure, RH.RC_SKINNY)
    epi = _pressure(pressure, RH.RC_EPILOGUE)
    shape_classes = sorted({s.shape_class for s in all_shapes})
    n_classes = len([c for c in shape_classes])
    clean_mn = sum(1 for a in all_axes if a.axis in ("M", "N") and a.shardable[8]
                   and not a.has_tail[8])

    monolithic = MultiplicityOption(
        "one_bigger_unit",
        evidence_for=[
            f"low inter-op parallelism (avg {avg_par}×, {n_seq}/{len(dags)} workloads near-sequential)"
            " — a single large unit is not starved by inter-op concurrency",
            (f"dense GEMM concentrates {dense.mac_fraction:.0%} of MACs in one shape family"
             if dense and dense.mac_fraction > 0.5 else "some workloads are dominated by one op")],
        evidence_against=[
            f"{n_classes} distinct geometry classes coexist ({', '.join(shape_classes)})",
            "GEMV/decode shapes (skinny family) are a poor match for a square matrix unit",
            "phases run at different cadences (backbone vs head vs control) — pipelineable"])

    replicated = MultiplicityOption(
        "multiple_identical_units",
        evidence_for=[
            (f"{n_par} workload(s) expose some inter-op parallelism" if n_par
             else "large ops shard cleanly along M/N"),
            f"{clean_mn} (op,axis) M/N shards split with no tail — reduction-free replication"],
        evidence_against=[                                   # blocked_by
            "reduction/partial-sum cost for K-sharding is unknown (not measured)",
            "memory bandwidth is unknown — replicas may contend for weight reload",
            f"data dependencies serialize work (avg parallelism only {avg_par}×)"])

    specialized = MultiplicityOption(
        "multiple_specialized_units",
        evidence_for=[
            (f"distinct operator families coexist: dense GEMM {dense.mac_fraction:.0%} of MACs vs "
             f"skinny/GEMV {skinny.mac_fraction:.0%}" if dense and skinny
             else "decode/GEMV and large GEMM coexist"),
            (f"epilogue/requant appears on {epi.op_count} ops" if epi else "epilogue appears"),
            "DMA/memory can overlap compute (resident loop-invariant weights)",
            "backbone and head run at different rates (multi-rate contract)",
            "the control loop decouples from replan inference"],
        evidence_against=[],
        candidate_units=[
            {"unit": "matrix_engine", "for": "dense_gemm", "evidence": "recovered_from_ir"},
            {"unit": "vector_gemv_engine", "for": "skinny_gemm_or_gemv",
             "evidence": "recovered_from_ir"},
            {"unit": "epilogue_requant_unit", "for": "epilogue_or_requant",
             "evidence": "recovered_from_ir"},
            {"unit": "dma_engine", "for": "dma_or_memory", "evidence": "recovered_from_prov_fqn"},
            {"unit": "loop_controller", "for": "control_or_dispatch",
             "evidence": "recovered_from_prov_fqn"},
            {"unit": "scalar_control_unit", "for": "control_or_dispatch",
             "evidence": "recovered_from_prov_fqn"},
            {"unit": "kv_cache_unit", "for": "attention_softmax_or_reduction",
             "evidence": "unavailable"}])

    # the conclusion is a search-space implication, NOT an architecture selection
    return {
        "avg_inter_op_parallelism": avg_par,
        "options": [monolithic, replicated, specialized],
        "search_space_implication": (
            "the evidence structurally suggests a HETEROGENEOUS (specialized) resource search "
            "space: distinct operator families (dense GEMM + skinny/GEMV), a frequent epilogue, "
            "resident-weight DMA, and multi-rate phases all coexist, while low inter-op parallelism "
            "argues against many identical units kept busy by concurrency. A future DSE should "
            "explore specialized units; this is an evidence-based search-space implication, NOT an "
            "architecture selection."),
    }


# --------------------------------------------------------------------------- emitters

def _opt_obj(o: MultiplicityOption) -> dict:
    d = {"option": o.option, "evidence_for": o.evidence_for}
    if o.option == "multiple_identical_units":
        d["blocked_by"] = o.evidence_against
    else:
        d["evidence_against"] = o.evidence_against
    if o.candidate_units:
        d["candidate_units"] = o.candidate_units
    return d


def guidance_yaml(g: dict) -> dict:
    return {"processing_unit_guidance": {
        "note": "monolithic vs. replicated vs. heterogeneous processing-unit evidence. This emits "
                "evidence and search-space implications only — it does NOT select an architecture, "
                "and makes no speedup/cycle/area claim.",
        "avg_inter_op_parallelism": {"value": g["avg_inter_op_parallelism"], "evidence": E_DERIVED,
                                     "note": "work/span average across workloads — not a speedup"},
        "options": [_opt_obj(o) for o in g["options"]],
        "search_space_implication": g["search_space_implication"]}}


def heterogeneity_report_md(g: dict) -> str:
    by_opt = {o.option: o for o in g["options"]}
    L = ["# Heterogeneity report — one bigger / many identical / specialized\n",
         "> Evidence comparing three resource-multiplicity search spaces. **Evidence and "
         "implications only — no architecture is selected, no speedup claimed.**\n"]
    for opt in ("one_bigger_unit", "multiple_identical_units", "multiple_specialized_units"):
        o = by_opt[opt]
        L.append(f"## {opt}\n")
        L.append("**Evidence for:**")
        for e in o.evidence_for:
            L.append(f"- {e}")
        label = "Blocked by:" if opt == "multiple_identical_units" else "Evidence against:"
        if o.evidence_against:
            L.append(f"\n**{label}**")
            for e in o.evidence_against:
                L.append(f"- {e}")
        if o.candidate_units:
            L.append("\n**Candidate units:** "
                     + ", ".join(f"`{u['unit']}`" for u in o.candidate_units))
        L.append("")
    L.append("## Search-space implication\n")
    L.append(g["search_space_implication"])
    L.append("\n**Caveat (structural, not realized):** this is an evidence-based search-space "
             "implication. **No speedup**, throughput, cycle, or area is claimed, and no design is "
             "chosen; the missing measurements that block a quantitative decision (reduction cost, "
             "memory bandwidth, per-unit throughput, timing) are named in the option evidence.\n")
    return "\n".join(L)
