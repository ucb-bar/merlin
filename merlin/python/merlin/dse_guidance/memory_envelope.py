"""Memory-traffic / reuse envelope — the bytes each region moves, and what residency avoids.

This is the quantitative companion to :mod:`.state_lifetime`. From the recovered per-region facts
(weight + activation bytes, recovered_from_ir) and the loop count K it derives, per region: the
weight traffic if the region is NOT made resident (re-loaded every invocation), the *avoidable*
reload residency removes, the activation traffic over the replan, and the reuse factor. These are
the inputs a memory-system DSE consumes (local-memory capacity, DRAM bandwidth, resident objects).

It claims no speedup and no bandwidth requirement against any hardware — only the workload-side
byte envelope. Traffic that the flat capture does not expose (intermediate-materialization bytes,
layout-conversion bytes) is emitted as ``unavailable``, never invented.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.dse_guidance.design_envelope import (CAPACITY_FORMATS, ELEMENT_BYTES, E_ASSUMED,
                                                 E_DERIVED, E_IR, E_NA)


@dataclass
class RegionTraffic:
    region: str
    invocations: int                    # K for the repeated head, 1 for the backbone
    invocations_evidence: str           # E_ASSUMED (K) for head, E_IR for backbone-once
    weight_bytes: int                   # recovered_from_ir
    weight_traffic_if_nonresident: int  # weight_bytes * invocations (derived)
    avoidable_weight_reload: int        # weight_bytes * (invocations - 1) (derived; residency removes)
    activation_bytes_per_invocation: int
    activation_traffic_total: int       # activation_bytes * invocations (derived)
    reuse_factor: int                   # = invocations


def region_traffic(attribution) -> list[RegionTraffic]:
    """Per-region byte traffic envelope from recovered role facts (no invented traffic)."""
    out: list[RegionTraffic] = []
    if attribution is None:
        return out
    for r in attribution.regions:
        if r.attribution_status != "attributed":
            continue
        f = r.facts
        inv = int(f.get("invocations", 1))
        wb = int(f.get("weight_bytes", 0))
        ape = int(f.get("activation_bytes_per_invocation", 0))
        out.append(RegionTraffic(
            region=r.role, invocations=inv,
            invocations_evidence=(E_ASSUMED if inv != 1 else E_IR),  # head K assumed; backbone ×1
            weight_bytes=wb,
            weight_traffic_if_nonresident=wb * inv,
            avoidable_weight_reload=wb * max(inv - 1, 0),
            activation_bytes_per_invocation=ape,
            activation_traffic_total=int(f.get("activation_bytes_total", ape * inv)),
            reuse_factor=inv))
    return out


def to_yaml_obj(rows: list[RegionTraffic], workload: str) -> dict:
    return {"memory_envelope": {
        "workload": workload,
        "note": "byte traffic from recovered per-region facts; weight/activation bytes are "
                "recovered_from_ir, ×K scaling is derived from an assumed_reference K. "
                "intermediate-materialization and layout-conversion traffic are NOT exposed by a "
                "flat capture and are marked unavailable. No bandwidth/speedup claimed.",
        "regions": [
            {"region": r.region,
             "invocations": {"value": r.invocations, "evidence": r.invocations_evidence},
             "weight_bytes": {"value": r.weight_bytes, "evidence": E_IR},
             "weight_traffic_if_nonresident": {"value": r.weight_traffic_if_nonresident,
                                               "evidence": E_DERIVED},
             "avoidable_weight_reload": {"value": r.avoidable_weight_reload, "evidence": E_DERIVED},
             "activation_bytes_per_invocation": {"value": r.activation_bytes_per_invocation,
                                                 "evidence": E_IR},
             "activation_traffic_total": {"value": r.activation_traffic_total, "evidence": E_DERIVED},
             "reuse_factor": {"value": r.reuse_factor, "evidence": E_ASSUMED},
             "intermediate_materialization_bytes": {"value": None, "evidence": E_NA},
             "layout_conversion_bytes": {"value": None, "evidence": E_NA}}
            for r in rows],
    }}


def traffic_csv(packages) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for p in packages:
        for r in p.get("mem", []):
            rows.append({
                "workload": p["case"].workload, "region": r.region, "invocations": r.invocations,
                "weight_bytes": r.weight_bytes,
                "weight_traffic_if_nonresident": r.weight_traffic_if_nonresident,
                "avoidable_weight_reload": r.avoidable_weight_reload,
                "activation_bytes_per_invocation": r.activation_bytes_per_invocation,
                "activation_traffic_total": r.activation_traffic_total,
                "reuse_factor": r.reuse_factor,
            })
    return _csv(rows, ["workload", "region", "invocations", "weight_bytes",
                       "weight_traffic_if_nonresident", "avoidable_weight_reload",
                       "activation_bytes_per_invocation", "activation_traffic_total",
                       "reuse_factor"])


# ============================================================ P9 memory-hierarchy envelope
# A richer per-region byte breakdown (weight / activation-in / output / intermediate / scale / KV),
# reuse-lifetime + residency per object, and the dtype-scaled resident set. Builds on the P5
# operator shapes (per-op input/output bytes) + attribution invocations. Bytes the flat capture does
# not expose (intermediate-materialization, scale/zero-point, KV/prefix) are emitted ``unavailable``.

# P9-b lifetime + reload + abstraction vocabularies.
LT_WITHIN_OP = "within_op"
LT_WITHIN_REGION = "within_region"
LT_ACROSS_K = "across_K"
LT_ACROSS_REPLAN = "across_replan"
LT_ACROSS_INSTRUCTION = "across_instruction"
LT_UNKNOWN = "unknown"

RB_RELOAD_EACH = "reload_each_invocation"
RB_LOAD_ONCE_K = "load_once_reuse_K"
RB_RESIDENT = "resident_candidate"
RB_UNKNOWN = "unknown"


@dataclass
class RegionMemory:
    region: str
    invocations: int
    weight_bytes: int
    activation_input_bytes: int          # per invocation
    output_bytes: int                    # per invocation
    intermediate_bytes: object           # None -> unavailable (fused buffers not exposed)
    scale_bytes: object                  # None -> unavailable (dequantized capture)
    kv_bytes: object                     # None -> unavailable (attention lowered)
    read_bytes_resident: int             # weight(once) + activation_in * invocations
    write_bytes: int                     # output * invocations
    read_write_ratio: object
    reuse_factor: int
    avoidable_weight_reload: int
    resident_by_dtype: dict
    lifetime: str
    evidence: str


def region_memory(attribution, shapes) -> list[RegionMemory]:
    """Per-region memory envelope from the P5 operator shapes + attribution invocations."""
    by_role: dict[str, list] = {}
    for s in shapes:
        if s.region_role and s.region_role != "unknown":
            by_role.setdefault(s.region_role, []).append(s)
    out: list[RegionMemory] = []
    for role, ss in by_role.items():
        rr = attribution.role(role) if attribution else None
        inv = int(rr.facts.get("invocations", 1)) if (rr and rr.attribution_status == "attributed") else 1
        weight = sum(s.rhs_weight_bytes for s in ss)
        act_in = sum(s.lhs_bytes for s in ss)
        outp = sum(s.output_bytes for s in ss)
        elem = ELEMENT_BYTES.get(str(ss[0].dtype or "f32").strip().lower(), 4.0)
        n_params = weight / elem if elem else 0
        read_res = weight + act_in * inv
        write = outp * inv
        out.append(RegionMemory(
            region=role, invocations=inv, weight_bytes=weight, activation_input_bytes=act_in,
            output_bytes=outp, intermediate_bytes=None, scale_bytes=None, kv_bytes=None,
            read_bytes_resident=read_res, write_bytes=write,
            read_write_ratio=(round(read_res / write, 3) if write else None),
            reuse_factor=inv, avoidable_weight_reload=weight * max(inv - 1, 0),
            resident_by_dtype={f: int(n_params * ELEMENT_BYTES[f]) for f in CAPACITY_FORMATS},
            lifetime=(LT_ACROSS_K if role == "repeated_head" else LT_ACROSS_REPLAN),
            evidence=E_IR))
    out.sort(key=lambda r: -r.weight_bytes)
    return out


@dataclass
class ReuseRecord:
    object: str
    producer_region: str
    consumer_regions: list
    lifetime: str
    reload_behavior: str
    required_abstraction: str
    bytes: object
    evidence: str


def reuse_lifetime(region_mem: list[RegionMemory]) -> list[ReuseRecord]:
    """Per-object reuse/residency: the weight object (reused across the loop) and the activation
    object (recomputed each invocation) for each region. KV/scale objects are unavailable."""
    out: list[ReuseRecord] = []
    for rm in region_mem:
        across = rm.lifetime
        out.append(ReuseRecord(
            object=f"{rm.region}.weights", producer_region=rm.region, consumer_regions=[rm.region],
            lifetime=across,
            reload_behavior=(RB_LOAD_ONCE_K if rm.invocations > 1 else RB_RESIDENT),
            required_abstraction="resident_weight_object", bytes=rm.weight_bytes, evidence=E_IR))
        out.append(ReuseRecord(
            object=f"{rm.region}.activations", producer_region=rm.region,
            consumer_regions=[rm.region], lifetime=LT_WITHIN_REGION,
            reload_behavior=RB_RELOAD_EACH, required_abstraction="resident_activation_object",
            bytes=rm.activation_input_bytes, evidence=E_IR))
    return out


# --------------------------------------------------------------------------- P9 emitters

def memory_hierarchy_yaml(mem_by_workload: dict) -> dict:
    return {"memory_hierarchy_envelope": {
        "note": "per-region byte envelope. weight/activation/output bytes are recovered_from_ir; "
                "resident_by_dtype scales the weights by element width. intermediate-"
                "materialization, scale/zero-point, and KV/prefix bytes are NOT exposed by a flat "
                "dequantized capture and are marked unavailable, never invented. "
                "No bandwidth/speedup is claimed (that needs an explicit design YAML).",
        "workloads": [
            {"workload": wl, "regions": [
                {"region": rm.region, "invocations": rm.invocations,
                 "weight_bytes": {"value": rm.weight_bytes, "evidence": E_IR},
                 "activation_input_bytes": {"value": rm.activation_input_bytes, "evidence": E_IR},
                 "output_bytes": {"value": rm.output_bytes, "evidence": E_IR},
                 "intermediate_bytes": {"value": "unavailable", "evidence": E_NA},
                 "scale_bytes": {"value": "unavailable", "evidence": E_NA},
                 "kv_bytes": {"value": "unavailable", "evidence": E_NA},
                 "read_bytes_resident": {"value": rm.read_bytes_resident, "evidence": E_DERIVED},
                 "write_bytes": {"value": rm.write_bytes, "evidence": E_DERIVED},
                 "read_write_ratio": rm.read_write_ratio,
                 "avoidable_weight_reload": {"value": rm.avoidable_weight_reload,
                                             "evidence": E_DERIVED},
                 "resident_by_dtype": rm.resident_by_dtype, "lifetime": rm.lifetime}
                for rm in mems]}
            for wl, mems in mem_by_workload.items()]}}


def data_movement_csv(mem_by_workload: dict) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for wl, mems in mem_by_workload.items():
        for rm in mems:
            rows.append({
                "workload": wl, "region": rm.region, "invocations": rm.invocations,
                "weight_bytes": rm.weight_bytes, "activation_input_bytes": rm.activation_input_bytes,
                "output_bytes": rm.output_bytes, "intermediate_bytes": "unavailable",
                "scale_bytes": "unavailable", "kv_bytes": "unavailable",
                "read_bytes_resident": rm.read_bytes_resident, "write_bytes": rm.write_bytes,
                "read_write_ratio": rm.read_write_ratio if rm.read_write_ratio is not None
                else "unavailable",
                "avoidable_weight_reload": rm.avoidable_weight_reload,
                "resident_int8_B": rm.resident_by_dtype.get("int8"),
                "resident_bf16_B": rm.resident_by_dtype.get("bf16"), "lifetime": rm.lifetime})
    return _csv(rows, ["workload", "region", "invocations", "weight_bytes",
                       "activation_input_bytes", "output_bytes", "intermediate_bytes",
                       "scale_bytes", "kv_bytes", "read_bytes_resident", "write_bytes",
                       "read_write_ratio", "avoidable_weight_reload", "resident_int8_B",
                       "resident_bf16_B", "lifetime"])


def reuse_lifetime_csv(reuse_by_workload: dict) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for wl, recs in reuse_by_workload.items():
        for r in recs:
            rows.append({"workload": wl, "object": r.object, "producer_region": r.producer_region,
                         "consumer_regions": "; ".join(r.consumer_regions), "lifetime": r.lifetime,
                         "reload_behavior": r.reload_behavior,
                         "required_abstraction": r.required_abstraction, "bytes": r.bytes,
                         "evidence": r.evidence})
    return _csv(rows, ["workload", "object", "producer_region", "consumer_regions", "lifetime",
                       "reload_behavior", "required_abstraction", "bytes", "evidence"])


def memory_abstraction_candidates_yaml(reuse_by_workload: dict) -> dict:
    from collections import Counter
    c = Counter()
    for recs in reuse_by_workload.values():
        for r in recs:
            c[r.required_abstraction] += 1
    return {"memory_abstraction_candidates": {
        "note": "memory abstractions implied by the reuse/residency structure. Counts are how many "
                "(workload,object) pairs imply each. Structural — no bandwidth/speedup claim.",
        "candidates": [
            {"abstraction": a, "implied_by_n_objects": n,
             "evidence": E_IR if a in ("resident_weight_object", "resident_activation_object")
             else E_NA}
            for a, n in c.most_common()],
        "unavailable_abstractions": [
            {"abstraction": "prefix_kv_object", "reason": "attention/KV lowered into matmuls"},
            {"abstraction": "scale_sideband_object", "reason": "scales erased (dequantized capture)"},
            {"abstraction": "packed_weight_store", "reason": "weights stored f32 (no packed layout)"}]}}


def memory_envelope_report_md(mem_by_workload: dict) -> str:
    L = ["# Memory-hierarchy envelope report\n",
         "> Per-region byte envelope: weights, activations, outputs, and the dtype-scaled resident "
         "set. **No bandwidth/speedup is claimed** (no bandwidth feasibility) — that needs a design "
         "YAML. Bytes a flat dequantized capture cannot expose (intermediate, scale, KV) are "
         "`unavailable`.\n"]
    # top memory-pressure regions by weight bytes + dominant byte class
    allregs = [(wl, rm) for wl, mems in mem_by_workload.items() for rm in mems]
    allregs.sort(key=lambda x: -x[1].weight_bytes)
    L.append("## Top memory-pressure regions (by weight bytes)\n")
    L.append("| workload | region | weight B | act-in B/inv | output B/inv | reuse | "
             "avoidable reload B | dominant class |")
    L.append("|---|---|---|---|---|---|---|---|")
    for wl, rm in allregs[:8]:
        dom = max((("weight", rm.weight_bytes), ("activation", rm.activation_input_bytes * rm.invocations),
                   ("output", rm.output_bytes * rm.invocations)), key=lambda kv: kv[1])[0]
        L.append(f"| {wl} | {rm.region} | {rm.weight_bytes:,} | {rm.activation_input_bytes:,} | "
                 f"{rm.output_bytes:,} | {rm.reuse_factor}× | {rm.avoidable_weight_reload:,} | "
                 f"{dom}-dominated |")
    L.append("")
    # overall: weight vs activation vs output dominated
    tot_w = sum(rm.weight_bytes for _, rm in allregs)
    tot_a = sum(rm.activation_input_bytes * rm.invocations for _, rm in allregs)
    tot_o = sum(rm.output_bytes * rm.invocations for _, rm in allregs)
    dom = max((("weight", tot_w), ("activation", tot_a), ("output", tot_o)), key=lambda kv: kv[1])[0]
    L.append("## Findings\n")
    L.append(f"- **Memory pressure is {dom}-dominated** across the recaptured workloads "
             f"(weights {tot_w:,} B, activations {tot_a:,} B, outputs {tot_o:,} B).")
    top = allregs[0]
    L.append(f"- **Top avoidable-reload candidate:** `{top[0]}/{top[1].region}` — "
             f"{top[1].avoidable_weight_reload:,} B avoidable if weights are made resident "
             f"(= weight_bytes × (reuse − 1)).")
    L.append("- **Repeatedly implied abstractions:** `resident_weight_object` (weights reused "
             "across the loop) and `resident_activation_object` (activations recomputed per step) — "
             "see `memory_abstraction_candidates.yaml`.")
    L.append("\n## Missing for real bandwidth feasibility\n")
    L.append("- intermediate-materialization, scale/zero-point, and KV/prefix bytes (`unavailable` "
             "in a flat dequantized capture);")
    L.append("- a target memory hierarchy (capacities, bandwidths) — supplied via a design YAML, "
             "absent here. **No bandwidth or deadline feasibility is claimed.**\n")
    return "\n".join(L)
