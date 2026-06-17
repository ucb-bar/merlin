"""Candidate compute-primitive coverage — structural geometry only (NOT a performance model).

Given the real operator geometry (``operator_geometry.py``), this asks a single structural question
for each candidate primitive shape: *if a future accelerator's compute primitive were this shape,
how much of the real operator MAC mass would it cover with little padding waste, and how much would
it waste on tile tails?* Padding waste and tile utilisation are pure geometry — they say nothing
about latency, throughput, or energy. The output tells a DSE engine which primitive shapes its
search space must include to cover the workloads, and which primitives overfit one workload.

Candidate primitives
---------------------
* **Tile primitives** ``tile_TMxTN`` cover the output M×N with a TM×TN tile. First version keeps K
  **exact** (``padded_K = K``) and models only the M/N tile tails — the K reduction does not create
  output-tile waste, and modelling K padding needs a datapath assumption we deliberately avoid here.
    padded_M = ceil(M/TM)*TM ; padded_N = ceil(N/TN)*TN
    padded_macs = padded_M * padded_N * K ; true_macs = M*N*K
    padding_waste = padded_macs/true_macs - 1 ; tile_utilization = true_macs/padded_macs
    tail_fraction = (padded_M*padded_N - M*N) / (padded_M*padded_N)
* **GEMV-lane primitives** ``gemv_lane_L`` vectorise along ONE output dimension with a lane width L.
  Structural applicability rule (documented): a lane primitive applies only to shapes that are
  vector-like along an output dim — ``gemv_like`` and ``wide_skinny`` (vectorised along **N**) and
  ``tall_skinny`` (vectorised along **M**). For any other class the lane is marked
  ``applicable = False`` and contributes **zero** structural coverage — it is never falsely scored
  as good on a square GEMM it cannot serve. Lane padding is along the vector dim only:
    padded_vec = ceil(vec/L)*L ; padding_waste = padded_vec/vec - 1.

Every coverage number is ``derived_requirement`` from ``recovered_from_ir`` shapes. No speedup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from merlin.dse_guidance import shape_taxonomy as ST
from merlin.dse_guidance.design_envelope import E_DERIVED

# (name, TM, TN) tile primitives.
TILE_PRIMITIVES: list[tuple[str, int, int]] = [
    ("tile_8x8", 8, 8), ("tile_8x16", 8, 16), ("tile_16x16", 16, 16),
    ("tile_16x32", 16, 32), ("tile_32x32", 32, 32),
]
# (name, lane_width) GEMV-lane primitives.
GEMV_PRIMITIVES: list[tuple[str, int]] = [
    ("gemv_lane_64", 64), ("gemv_lane_128", 128), ("gemv_lane_256", 256),
]
# Geometry classes a GEMV lane structurally applies to, and which output dim it vectorises.
_LANE_VEC_DIM = {ST.GEMV: "N", ST.WIDE_SKINNY: "N", ST.TALL_SKINNY: "M"}

WASTE_BANDS = (0.05, 0.10, 0.25)   # "covered under X% padding waste"


def _ceil_to(x: int, t: int) -> int:
    return int(math.ceil(x / t) * t)


@dataclass
class OpPrimitiveCoverage:
    workload: str
    op_index: int
    shape_class: str
    region_role: str
    primitive: str
    primitive_kind: str          # "tile" | "gemv_lane"
    applicable: bool
    true_macs: int
    padded_M: int | None
    padded_N: int | None
    padded_K: int | None
    padded_macs: int | None
    padding_waste: float | None
    tile_utilization: float | None
    tail_fraction: float | None
    covered_under_5pct: bool
    covered_under_10pct: bool
    covered_under_25pct: bool


def _bands(waste: float | None) -> tuple[bool, bool, bool]:
    if waste is None:
        return (False, False, False)
    return tuple(waste <= b for b in WASTE_BANDS)  # type: ignore[return-value]


def tile_coverage(shape, name: str, TM: int, TN: int) -> OpPrimitiveCoverage:
    M, N, K = shape.M, shape.N, shape.K
    pM, pN, pK = _ceil_to(M, TM), _ceil_to(N, TN), K
    true_macs = M * N * K
    padded_macs = pM * pN * pK
    waste = padded_macs / true_macs - 1.0 if true_macs else None
    util = true_macs / padded_macs if padded_macs else None
    tail = (pM * pN - M * N) / (pM * pN) if pM * pN else None
    c5, c10, c25 = _bands(waste)
    return OpPrimitiveCoverage(
        workload=shape.workload, op_index=shape.op_index, shape_class=shape.shape_class,
        region_role=shape.region_role, primitive=name, primitive_kind="tile", applicable=True,
        true_macs=true_macs, padded_M=pM, padded_N=pN, padded_K=pK, padded_macs=padded_macs,
        padding_waste=round(waste, 6) if waste is not None else None,
        tile_utilization=round(util, 6) if util is not None else None,
        tail_fraction=round(tail, 6) if tail is not None else None,
        covered_under_5pct=c5, covered_under_10pct=c10, covered_under_25pct=c25)


def gemv_coverage(shape, name: str, L: int) -> OpPrimitiveCoverage:
    M, N, K = shape.M, shape.N, shape.K
    true_macs = M * N * K
    vec_dim = _LANE_VEC_DIM.get(shape.shape_class)
    if vec_dim is None:                    # not a vector-like shape: lane does not apply
        return OpPrimitiveCoverage(
            workload=shape.workload, op_index=shape.op_index, shape_class=shape.shape_class,
            region_role=shape.region_role, primitive=name, primitive_kind="gemv_lane",
            applicable=False, true_macs=true_macs, padded_M=None, padded_N=None, padded_K=None,
            padded_macs=None, padding_waste=None, tile_utilization=None, tail_fraction=None,
            covered_under_5pct=False, covered_under_10pct=False, covered_under_25pct=False)
    vec = N if vec_dim == "N" else M
    pvec = _ceil_to(vec, L)
    waste = pvec / vec - 1.0 if vec else None
    util = vec / pvec if pvec else None
    padded_macs = int(true_macs * (pvec / vec)) if vec else None
    c5, c10, c25 = _bands(waste)
    return OpPrimitiveCoverage(
        workload=shape.workload, op_index=shape.op_index, shape_class=shape.shape_class,
        region_role=shape.region_role, primitive=name, primitive_kind="gemv_lane", applicable=True,
        true_macs=true_macs,
        padded_M=(pvec if vec_dim == "M" else M), padded_N=(pvec if vec_dim == "N" else N),
        padded_K=K, padded_macs=padded_macs,
        padding_waste=round(waste, 6) if waste is not None else None,
        tile_utilization=round(util, 6) if util is not None else None,
        tail_fraction=round(waste / (1 + waste), 6) if waste is not None else None,
        covered_under_5pct=c5, covered_under_10pct=c10, covered_under_25pct=c25)


def coverage_for_op(shape) -> list[OpPrimitiveCoverage]:
    out = [tile_coverage(shape, n, tm, tn) for n, tm, tn in TILE_PRIMITIVES]
    out += [gemv_coverage(shape, n, L) for n, L in GEMV_PRIMITIVES]
    return out


def all_coverage(all_shapes) -> list[OpPrimitiveCoverage]:
    out: list[OpPrimitiveCoverage] = []
    for s in all_shapes:
        out.extend(coverage_for_op(s))
    return out


# --------------------------------------------------------------------------- aggregation

@dataclass
class PrimitiveWorkloadAgg:
    primitive: str
    primitive_kind: str
    workload: str
    op_count: int
    applicable_ops: int
    macs_total: int
    macs_covered_5: int
    macs_covered_10: int
    macs_covered_25: int
    mac_weighted_utilization: float | None
    coverage_under_10pct: float          # macs_covered_10 / macs_total


def aggregate_by_primitive_workload(cov: list[OpPrimitiveCoverage]) -> list[PrimitiveWorkloadAgg]:
    groups: dict[tuple, list] = {}
    for c in cov:
        groups.setdefault((c.primitive, c.primitive_kind, c.workload), []).append(c)
    out = []
    for (prim, kind, w), cs in groups.items():
        macs_total = sum(c.true_macs for c in cs)
        appl = [c for c in cs if c.applicable]
        m5 = sum(c.true_macs for c in cs if c.covered_under_5pct)
        m10 = sum(c.true_macs for c in cs if c.covered_under_10pct)
        m25 = sum(c.true_macs for c in cs if c.covered_under_25pct)
        padded = sum(c.padded_macs for c in appl if c.padded_macs)
        true_appl = sum(c.true_macs for c in appl)
        util = (true_appl / padded) if padded else None
        out.append(PrimitiveWorkloadAgg(
            primitive=prim, primitive_kind=kind, workload=w, op_count=len(cs),
            applicable_ops=len(appl), macs_total=macs_total,
            macs_covered_5=m5, macs_covered_10=m10, macs_covered_25=m25,
            mac_weighted_utilization=round(util, 6) if util is not None else None,
            coverage_under_10pct=round(m10 / macs_total, 6) if macs_total else 0.0))
    out.sort(key=lambda a: (a.primitive, a.workload))
    return out


@dataclass
class PrimitiveRegret:
    primitive: str
    primitive_kind: str
    op_count: int
    macs_total: int
    mac_weighted_utilization: float | None
    coverage_under_5pct: float
    coverage_under_10pct: float
    coverage_under_25pct: float
    worst_workload_coverage_10: float
    best_workload_coverage_10: float
    average_workload_coverage_10: float
    max_regret: float
    clusters_poorly_served: str
    workloads_where_overfit: str


def aggregate_regret(cov: list[OpPrimitiveCoverage],
                     per_wl: list[PrimitiveWorkloadAgg]) -> list[PrimitiveRegret]:
    by_prim: dict[str, list] = {}
    for c in cov:
        by_prim.setdefault(c.primitive, []).append(c)
    wl_by_prim: dict[str, list] = {}
    for a in per_wl:
        wl_by_prim.setdefault(a.primitive, []).append(a)

    out = []
    for prim, cs in by_prim.items():
        kind = cs[0].primitive_kind
        macs_total = sum(c.true_macs for c in cs) or 0
        appl = [c for c in cs if c.applicable]
        padded = sum(c.padded_macs for c in appl if c.padded_macs)
        true_appl = sum(c.true_macs for c in appl)
        util = (true_appl / padded) if padded else None
        cov5 = sum(c.true_macs for c in cs if c.covered_under_5pct) / macs_total if macs_total else 0
        cov10 = sum(c.true_macs for c in cs if c.covered_under_10pct) / macs_total if macs_total else 0
        cov25 = sum(c.true_macs for c in cs if c.covered_under_25pct) / macs_total if macs_total else 0
        wls = wl_by_prim.get(prim, [])
        per = [a.coverage_under_10pct for a in wls]
        worst = min(per) if per else 0.0
        best = max(per) if per else 0.0
        avg = sum(per) / len(per) if per else 0.0
        # clusters poorly served = geometry classes whose 10%-covered MAC share is < 0.5 for this primitive
        cls_macs: dict[str, list] = {}
        for c in cs:
            cls_macs.setdefault(c.shape_class, [0, 0])
            cls_macs[c.shape_class][0] += c.true_macs
            if c.covered_under_10pct:
                cls_macs[c.shape_class][1] += c.true_macs
        poor = sorted(cls for cls, (tot, ok) in cls_macs.items() if tot and ok / tot < 0.5)
        # overfit = workloads where this primitive covers >=0.9 while the worst workload is <0.5
        overfit = sorted(a.workload for a in wls
                         if a.coverage_under_10pct >= 0.9 and worst < 0.5) if best >= 0.9 else []
        out.append(PrimitiveRegret(
            primitive=prim, primitive_kind=kind, op_count=len(cs), macs_total=macs_total,
            mac_weighted_utilization=round(util, 6) if util is not None else None,
            coverage_under_5pct=round(cov5, 6), coverage_under_10pct=round(cov10, 6),
            coverage_under_25pct=round(cov25, 6),
            worst_workload_coverage_10=round(worst, 6), best_workload_coverage_10=round(best, 6),
            average_workload_coverage_10=round(avg, 6), max_regret=round(best - worst, 6),
            clusters_poorly_served="; ".join(poor), workloads_where_overfit="; ".join(overfit)))
    out.sort(key=lambda r: (-r.coverage_under_10pct, r.max_regret, r.primitive))
    return out


# --------------------------------------------------------------------------- emitters

_TILE_WASTE_COLS = ["workload", "op_index", "shape_class", "region_role", "primitive",
                    "primitive_kind", "applicable", "true_macs", "padded_M", "padded_N", "padded_K",
                    "padded_macs", "padding_waste", "tile_utilization", "tail_fraction",
                    "covered_under_5pct", "covered_under_10pct", "covered_under_25pct"]


def tile_waste_csv(cov: list[OpPrimitiveCoverage]) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = [{c: getattr(o, c) for c in _TILE_WASTE_COLS} for o in cov]
    return _csv(rows, _TILE_WASTE_COLS)


def primitive_coverage_matrix_csv(per_wl: list[PrimitiveWorkloadAgg]) -> str:
    from merlin.dse_guidance.case_study import _csv
    cols = ["primitive", "primitive_kind", "workload", "op_count", "applicable_ops", "macs_total",
            "macs_covered_5", "macs_covered_10", "macs_covered_25", "mac_weighted_utilization",
            "coverage_under_10pct"]
    rows = [{c: getattr(a, c) for c in cols} for a in per_wl]
    return _csv(rows, cols)


def primitive_regret_csv(regret: list[PrimitiveRegret]) -> str:
    from merlin.dse_guidance.case_study import _csv
    cols = ["primitive", "primitive_kind", "op_count", "macs_total", "mac_weighted_utilization",
            "coverage_under_5pct", "coverage_under_10pct", "coverage_under_25pct",
            "worst_workload_coverage_10", "best_workload_coverage_10", "average_workload_coverage_10",
            "max_regret", "clusters_poorly_served", "workloads_where_overfit"]
    rows = [{c: getattr(r, c) for c in cols} for r in regret]
    return _csv(rows, cols)


def coverage_report_md(per_wl: list[PrimitiveWorkloadAgg], regret: list[PrimitiveRegret]) -> str:
    L = ["# Candidate-primitive coverage report\n",
         "> For each candidate compute-primitive shape: how much of the real operator MAC mass it "
         "covers under a padding-waste band, and its MAC-weighted tile utilisation. **Structural "
         "geometry coverage only — no speedup, no cycle-count, no performance ranking.** Tile "
         "primitives keep K exact and model M/N tile tails; GEMV lanes apply only to vector-like "
         "shapes (see `primitive_coverage.py`).\n"]
    L.append("## Primitive coverage (MAC-weighted, all workloads)\n")
    L.append("| primitive | kind | MAC util | covered ≤5% | covered ≤10% | covered ≤25% |")
    L.append("|---|---|---|---|---|---|")
    for r in regret:
        util = f"{r.mac_weighted_utilization:.1%}" if r.mac_weighted_utilization is not None else "—"
        L.append(f"| {r.primitive} | {r.primitive_kind} | {util} | {r.coverage_under_5pct:.1%} | "
                 f"{r.coverage_under_10pct:.1%} | {r.coverage_under_25pct:.1%} |")
    L.append("")
    L.append("## Per-workload coverage under 10% padding waste\n")
    workloads = sorted({a.workload for a in per_wl})
    L.append("| primitive | " + " | ".join(workloads) + " |")
    L.append("|---|" + "---|" * len(workloads))
    by_prim: dict[str, dict] = {}
    for a in per_wl:
        by_prim.setdefault(a.primitive, {})[a.workload] = a.coverage_under_10pct
    for prim in [r.primitive for r in regret]:
        cells = [f"{by_prim.get(prim, {}).get(w, 0.0):.0%}" for w in workloads]
        L.append(f"| {prim} | " + " | ".join(cells) + " |")
    L.append("\nCovers X% of MACs ≈ a primitive of that shape would process X% of the real MAC mass "
             "with ≤ the stated padding waste. A low cell means that primitive **poorly covers** "
             "that workload's shapes and is workload-specific. See `primitive_coverage_matrix.csv` "
             "and `primitive_regret_table.csv`.\n")
    return "\n".join(L)


def cross_workload_report_md(regret: list[PrimitiveRegret],
                             per_wl: list[PrimitiveWorkloadAgg]) -> str:
    L = ["# Cross-workload primitive coverage & regret\n",
         "> Which candidate primitive shapes a future DSE search space should include because they "
         "cover the real operator geometry broadly, and which overfit a single workload. "
         "**Structural geometry only — no speedup.**\n"]
    L.append("## Coverage regret across workloads (10% waste band)\n")
    L.append("| primitive | avg cov | worst cov | best cov | max regret | poorly-served clusters |")
    L.append("|---|---|---|---|---|---|")
    for r in regret:
        L.append(f"| {r.primitive} | {r.average_workload_coverage_10:.0%} | "
                 f"{r.worst_workload_coverage_10:.0%} | {r.best_workload_coverage_10:.0%} | "
                 f"{r.max_regret:.0%} | {r.clusters_poorly_served or '—'} |")
    L.append("")
    if regret:
        widest = max(regret, key=lambda r: r.average_workload_coverage_10)
        worst_regret = max(regret, key=lambda r: r.max_regret)
        L.append("## Findings\n")
        L.append(f"- **Widest average structural coverage:** `{widest.primitive}` at "
                 f"{widest.average_workload_coverage_10:.0%} average per-workload coverage under "
                 f"10% waste — **suggests this primitive should be included in the future DSE "
                 f"search space.**")
        L.append(f"- **Worst cross-workload regret:** `{worst_regret.primitive}` "
                 f"(max_regret {worst_regret.max_regret:.0%}: best "
                 f"{worst_regret.best_workload_coverage_10:.0%} vs worst "
                 f"{worst_regret.worst_workload_coverage_10:.0%}) — **suggests this primitive is "
                 f"workload-specific**, not a general choice.")
        overfit = [r for r in regret if r.workloads_where_overfit]
        if overfit:
            o = overfit[0]
            L.append(f"- **Overfit primitives:** `{o.primitive}` covers "
                     f"`{o.workloads_where_overfit}` well but poorly covers the worst workload "
                     f"({o.worst_workload_coverage_10:.0%}).")
    L.append("\n**Caveat:** these are structural tile/lane coverage metrics — padding waste and "
             "utilisation are pure geometry. **No speedup**, latency, or performance is implied, "
             "and no hardware is assumed.\n")
    return "\n".join(L)
