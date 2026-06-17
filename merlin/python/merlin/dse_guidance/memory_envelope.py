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

from merlin.dse_guidance.design_envelope import E_ASSUMED, E_DERIVED, E_IR, E_NA


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
