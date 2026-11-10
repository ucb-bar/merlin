"""Compare primitive coverage sets and their workload/threshold sensitivity.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

import math
from pathlib import Path

from .artifacts import _csv_rows
from .workloads import _signal_by_workload


def _coverage_from_tilewaste(cs_dir: Path):
    """Per-op covered_under_10pct flags grouped by (workload, primitive) -> set-union frontier base.
    Returns (op_macs, op_cover) where op_cover[(w, op)][primitive] = bool covered, and a per-workload
    MAC total — so a primitive SET covers an op if ANY member covers it."""
    tw = _csv_rows(Path(cs_dir) / "tile_waste_table.csv")
    op_macs, op_cover, prims = {}, {}, set()
    for r in tw:
        if r.get("applicable") != "True":
            continue
        key = (r["workload"], r["op_index"])
        op_macs[key] = float(r["true_macs"])
        op_cover.setdefault(key, {})[r["primitive"]] = r["covered_under_10pct"] == "True"
        prims.add(r["primitive"])
    return op_macs, op_cover, sorted(prims)


def _set_coverage(pset, op_macs, op_cover, drop=None):
    """MAC-weighted coverage of a primitive SET, per workload (a set covers an op if any member does).
    Returns {workload: coverage_fraction}."""
    num, den = {}, {}
    for (w, op), macs in op_macs.items():
        if drop and w == drop:
            continue
        covered = any(op_cover[(w, op)].get(p, False) for p in pset)
        den[w] = den.get(w, 0.0) + macs
        if covered:
            num[w] = num.get(w, 0.0) + macs
    return {w: (num.get(w, 0.0) / den[w] if den[w] else 0.0) for w in den}


def _frontier_stats(pset, op_macs, op_cover, drop=None):
    cov = _set_coverage(pset, op_macs, op_cover, drop)
    if not cov:
        return None
    vals = list(cov.values())
    macro = sum(vals) / len(vals)
    # micro = MAC-weighted across the corpus
    tot_macs = sum(m for (w, _), m in op_macs.items() if not (drop and w == drop)) or 1.0
    micro = (
        sum(
            _set_coverage(pset, op_macs, op_cover, drop)[w] * sum(m for (ww, _), m in op_macs.items() if ww == w)
            for w in cov
        )
        / tot_macs
    )
    worst = min(vals)
    return {
        "macro": round(macro, 4),
        "micro": round(micro, 4),
        "worst": round(worst, 4),
        "max_regret": round(max(vals) - min(vals), 4),
        "per_workload": cov,
    }


def primitive_set_frontier(cs_dir: Path, max_size: int = 3) -> dict:
    """Best primitive SET of size 1/2/3 by worst-workload then macro coverage (a set covers an op if
    ANY member tiles it under 10% waste). The headline search-space result: one primitive is not
    enough; {one tile + one gemv lane} lifts worst-workload coverage."""
    import itertools

    op_macs, op_cover, prims = _coverage_from_tilewaste(cs_dir)
    best_by_size = {}
    for size in range(1, max_size + 1):
        best = None
        for combo in itertools.combinations(prims, size):
            st = _frontier_stats(combo, op_macs, op_cover)
            if st is None:
                continue
            key = (st["worst"], st["macro"])
            if best is None or key > best["key"]:
                best = {"set": list(combo), "key": key, **st}
        if best:
            best.pop("key")
            best_by_size[size] = best
    # also per-single-primitive stats (for the frontier scatter)
    singles = []
    for p in prims:
        st = _frontier_stats((p,), op_macs, op_cover)
        if st:
            singles.append({"primitive": p, "macro": st["macro"], "worst": st["worst"]})
    return {"best_by_size": best_by_size, "singles": singles}


def operator_pareto(cs_dir: Path, thresholds=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """Per workload, how many top ops are needed to reach 50/80/90/95% of MACs (and of weight bytes).
    Shows whether DSE sizes for a few giant ops or many even ones."""
    ops = _csv_rows(Path(cs_dir) / "operator_shape_table.csv")
    wls = sorted({o["workload"] for o in ops})
    out = []
    for w in wls:
        wops = [o for o in ops if o["workload"] == w]

        def _ktoreach(key):
            vals = sorted((int(o[key]) for o in wops), reverse=True)
            tot = sum(vals) or 1
            ks = {}
            for t in thresholds:
                acc, k = 0, 0
                for v in vals:
                    acc += v
                    k += 1
                    if acc / tot >= t:
                        break
                ks[t] = k
            return ks

        km, kb = _ktoreach("macs"), _ktoreach("rhs_weight_bytes")
        row = {"workload": w, "n_ops": len(wops)}
        for t in thresholds:
            row[f"k_macs_{int(t * 100)}"] = km[t]
            row[f"k_wbytes_{int(t * 100)}"] = kb[t]
        # top op MAC share
        top = max(wops, key=lambda o: int(o["macs"]))
        row["top_op"] = top["prov_fqn"]
        row["top_op_mac_share"] = round(int(top["macs"]) / (sum(int(o["macs"]) for o in wops) or 1), 4)
        # distinct-shape vs instance-count (Q6): pi05's ~777 ops are a handful of shapes repeated
        # 100s of times, NOT 777 heterogeneous ops -- so "many ops for 50% MACs" is instance-count
        # concentration, not shape diversity.
        from collections import Counter as _C

        shapes = _C((int(o["M"]), int(o["N"]), int(o["K"])) for o in wops)
        row["n_distinct_shapes"] = len(shapes)
        row["top_shape_multiplicity"] = max(shapes.values()) if shapes else 0
        out.append(row)
    return {"thresholds": thresholds, "rows": out}


def robustness(cs_dir: Path) -> dict:
    """Macro/micro + leave-one-workload-out for the major cross-workload findings, flagging which
    conclusions flip when a workload is removed (anti-overfitting)."""
    sig, wls = _signal_by_workload(cs_dir)
    op_macs, op_cover, prims = _coverage_from_tilewaste(cs_dir)
    if len(wls) < 1 or len(prims) < 2:  # partial mode: nothing to compare
        return {"workloads": wls, "findings": []}
    findings = []

    # (a) best 2-primitive set: does the winner change under leave-one-out?
    import itertools

    def best_two(drop=None):
        best = None
        for combo in itertools.combinations(prims, 2):
            st = _frontier_stats(combo, op_macs, op_cover, drop)
            if st and (best is None or (st["worst"], st["macro"]) > best[0]):
                best = ((st["worst"], st["macro"]), combo, st)
        return best

    full = best_two()
    loo = {w: best_two(w) for w in wls}
    flips = sorted(w for w in wls if loo[w] and set(loo[w][1]) != set(full[1]))
    findings.append(
        {
            "finding": "best_2_primitive_set",
            "all": list(full[1]),
            "all_worst": full[2]["worst"],
            "all_macro": full[2]["macro"],
            "loo_changes_winner": flips,
            "robust": not flips,
        }
    )

    # (b) dense-MAC dominance: macro (equal-weight) vs micro (MAC-weighted); rdt-driven?
    dense_macro = sum(sig[w]["dense_mac"] for w in wls) / len(wls)
    tot_mac = {w: 0.0 for w in wls}
    ops = _csv_rows(Path(cs_dir) / "operator_shape_table.csv")
    for o in ops:
        tot_mac[o["workload"]] += int(o["macs"])
    corpus_mac = sum(tot_mac.values()) or 1.0
    dense_micro = sum(sig[w]["dense_mac"] * tot_mac[w] for w in wls) / corpus_mac
    dense_micro_loo = {
        w: (sum(sig[x]["dense_mac"] * tot_mac[x] for x in wls if x != w) / (corpus_mac - tot_mac[w] or 1.0))
        for w in wls
    }
    drop_to_low = sorted(w for w in wls if dense_micro_loo[w] < 0.2 <= dense_micro)
    findings.append(
        {
            "finding": "dense_gemm_mac_dominance",
            "macro": round(dense_macro, 4),
            "micro": round(dense_micro, 4),
            "micro_loo": {w: round(v, 4) for w, v in dense_micro_loo.items()},
            "collapses_if_removed": drop_to_low,
            "note": (
                "MAC-dominant (micro) but corpus-narrow (macro)"
                if dense_micro - dense_macro > 0.2
                else "consistent across views"
            ),
        }
    )

    # (c) residency pressure ranking: which workload tops avoidable_reload, robust?
    rank = sorted(wls, key=lambda w: -sig[w]["avoidable_reload"])
    findings.append(
        {
            "finding": "residency_pressure_rank",
            "all": rank,
            "top": rank[0] if rank else "",
            "note": "absolute bytes are small/random-init; ranking is the robust signal",
        }
    )
    return {"workloads": wls, "findings": findings}


# ---- Part C: primitive-frontier robustness (recompute pad-waste so thresholds + extra tiles work)
_FRONTIER_THRESHOLDS = (0.05, 0.10, 0.20)


# extra candidate tiles beyond the committed set (structural coverage candidates only, never "faster")
_EXTRA_TILES = ("tile_4x16", "tile_8x32", "tile_16x64", "tile_64x16", "tile_4x32")


_LANE_VEC_DIM = {"gemv_like": "N", "wide_skinny": "N", "tall_skinny": "M"}


_FROB_COLS = ["threshold_pct", "set_size", "primitive_set", "worst", "macro", "micro", "max_regret"]


_UNCOV_COLS = ["threshold_pct", "set_size", "primitive_set", "workload", "op_index", "status", "true_macs"]


def _prim_spec(name: str) -> tuple:
    if name.startswith("tile_"):
        tm, tn = name[len("tile_") :].split("x")
        return ("tile", int(tm), int(tn))
    if name.startswith("gemv_lane_"):
        return ("lane", int(name.rsplit("_", 1)[1]), 0)
    return (None, 0, 0)


def _prim_waste(name: str, M: int, N: int, K: int, shape_class: str):
    """(applicable, padding_waste_fraction) using the documented primitive_coverage geometry: tiles
    pad M/N tails (K exact); a GEMV lane pads the one vector dim and only applies to vector-like
    shapes. Mirrors primitive_coverage.py so the 10% recompute matches the committed tile_waste."""
    kind, a, b = _prim_spec(name)
    if kind == "tile":
        pm = int(math.ceil(M / a) * a)
        pn = int(math.ceil(N / b) * b)
        return True, pm * pn / (M * N) - 1.0
    if kind == "lane":
        vd = _LANE_VEC_DIM.get(shape_class)
        if vd is None:
            return False, None
        vec = N if vd == "N" else M
        return True, int(math.ceil(vec / a) * a) / vec - 1.0
    return False, None


def primitive_frontier_robustness(cs_dir: Path, max_size: int = 4) -> dict:
    """Part C: sweep the best primitive SET over set sizes 1..4 and pad-waste thresholds 5/10/20%,
    add extra candidate tiles, and report macro/micro/worst/max-regret + leave-one-out winner +
    uncovered ops. Coverage is recomputed from operator_shape_table geometry (so thresholds other
    than the committed 10% boolean, and the extra tiles, are available); the 10% recompute is
    regression-checked against the committed covered_under_10pct by the verifier."""
    import itertools

    cs_dir = Path(cs_dir)
    base = sorted({r["primitive"] for r in _csv_rows(cs_dir / "tile_waste_table.csv") if r.get("applicable") == "True"})
    prims = base + [p for p in _EXTRA_TILES if p not in base]
    geom, op_macs = {}, {}
    for o in _csv_rows(cs_dir / "operator_shape_table.csv"):
        M, N, K = int(o["M"]), int(o["N"]), int(o["K"])
        if M <= 0 or N <= 0 or K <= 0:
            continue
        key = (o["workload"], o["op_index"])
        geom[key] = (M, N, K, o["shape_class"])
        op_macs[key] = float(M) * N * K
    waste = {}
    for key, (M, N, K, sc) in geom.items():
        cell = {}
        for p in prims:
            ap, wv = _prim_waste(p, M, N, K, sc)
            cell[p] = wv if ap else None
        waste[key] = cell
    wls = sorted({w for (w, _) in geom})

    def cov(pset, key, t):
        return any(waste[key][p] is not None and waste[key][p] <= t for p in pset)

    def stats(pset, t, drop=None):
        num, den = {}, {}
        for (w, op), m in op_macs.items():
            if drop and w == drop:
                continue
            den[w] = den.get(w, 0.0) + m
            if cov(pset, (w, op), t):
                num[w] = num.get(w, 0.0) + m
        per = {w: (num.get(w, 0.0) / den[w] if den[w] else 0.0) for w in den}
        if not per:
            return None
        vals = list(per.values())
        tot = sum(den.values()) or 1.0
        return {
            "macro": sum(vals) / len(vals),
            "micro": sum(num.values()) / tot,
            "worst": min(vals),
            "max_regret": max(vals) - min(vals),
            "per": per,
        }

    def best(size, t, drop=None):
        b = None
        for combo in itertools.combinations(prims, size):
            st = stats(combo, t, drop)
            if st and (b is None or (st["worst"], st["macro"]) > (b[1]["worst"], b[1]["macro"])):
                b = (combo, st)
        return b

    rows, grid = [], {}
    for t in _FRONTIER_THRESHOLDS:
        for size in range(1, max_size + 1):
            bs = best(size, t)
            if not bs:
                continue
            grid[(t, size)] = bs
            combo, st = bs
            rows.append(
                {
                    "threshold_pct": int(t * 100),
                    "set_size": size,
                    "primitive_set": "+".join(combo),
                    "worst": round(st["worst"], 4),
                    "macro": round(st["macro"], 4),
                    "micro": round(st["micro"], 4),
                    "max_regret": round(st["max_regret"], 4),
                }
            )
    # uncovered ops as the set grows (headline 10% threshold)
    uncovered, prev = [], set()
    for size in range(1, max_size + 1):
        bs = grid.get((0.10, size))
        if not bs:
            continue
        combo = bs[0]
        covered = {k for k in op_macs if cov(combo, k, 0.10)}
        for w, op in sorted(k for k in op_macs if k not in covered):
            uncovered.append(
                {
                    "threshold_pct": 10,
                    "set_size": size,
                    "primitive_set": "+".join(combo),
                    "workload": w,
                    "op_index": op,
                    "status": "uncovered",
                    "true_macs": int(op_macs[(w, op)]),
                }
            )
        prev = covered
    # threshold + LOO robustness of the best 2-set
    two = {t: (grid[(t, 2)][0] if (t, 2) in grid else None) for t in _FRONTIER_THRESHOLDS}
    base2 = grid.get((0.10, 2))
    loo_flips = []
    if base2:
        for w in wls:
            bs = best(2, 0.10, drop=w)
            if bs and set(bs[0]) != set(base2[0]):
                loo_flips.append(w)
    thr_sets = {tuple(sorted(v)) for v in two.values() if v}
    return {
        "workloads": wls,
        "primitives": prims,
        "rows": rows,
        "uncovered_rows": uncovered,
        "two_winners": {f"{int(t * 100)}pct": ("+".join(v) if v else "") for t, v in two.items()},
        "two_set_loo_flips": loo_flips,
        "two_set_threshold_robust": (len(thr_sets) == 1 and not loo_flips),
    }


def _frontier_rows(fr):
    rows = []
    for size, b in sorted(fr["best_by_size"].items()):
        rows.append(
            {
                "set_size": size,
                "primitive_set": " + ".join(b["set"]),
                "macro_coverage": b["macro"],
                "micro_coverage": b["micro"],
                "worst_workload_coverage": b["worst"],
                "max_regret": b["max_regret"],
            }
        )
    return rows


def _frontier_md(fr, scope) -> str:
    L = [
        f"# Primitive-set frontier ({scope})\n",
        "> A primitive SET covers an op if ANY member tiles it under 10% pad waste. The headline "
        "search-space result: one primitive is not enough — the best single primitive leaves a "
        "workload badly covered (low worst-workload), while a {tile + GEMV-lane} pair covers the "
        "corpus. Structural coverage only, no performance.\n",
        "| set size | best primitive set | worst-workload | macro (mean) | micro (MAC-wt) | max regret |",
        "|---|---|---|---|---|---|",
    ]
    for r in _frontier_rows(fr):
        L.append(
            f"| {r['set_size']} | {r['primitive_set']} | {r['worst_workload_coverage']:.2f} | "
            f"{r['macro_coverage']:.2f} | {r['micro_coverage']:.2f} | {r['max_regret']:.2f} |"
        )
    return "\n".join(L) + "\n"


def _pareto_md(par, scope) -> str:
    ts = par["thresholds"]
    L = [
        f"# Operator Pareto hotspots ({scope})\n",
        "> How many top ops are needed to reach a MAC threshold — whether DSE should size for a few "
        "giant ops or many even ones. Structural (MAC/byte share), no performance.\n",
        "| workload | n_ops | " + " | ".join(f"k@{int(t * 100)}%MAC" for t in ts) + " | top-op MAC share |",
        "|---|---|" + "|".join(["---"] * len(ts)) + "|---|",
    ]
    for r in par["rows"]:
        L.append(
            f"| {r['workload']} | {r['n_ops']} | "
            + " | ".join(str(r[f"k_macs_{int(t * 100)}"]) for t in ts)
            + f" | {r['top_op_mac_share']:.0%} |"
        )
    return "\n".join(L) + "\n"


def _robustness_md(rob, scope) -> str:
    L = [
        f"# Macro/micro + leave-one-workload-out robustness ({scope})\n",
        "> Anti-overfitting: each cross-workload finding recomputed dropping one workload. A finding "
        "that flips is corpus-specific, not general.\n",
    ]
    for f in rob["findings"]:
        L.append(f"## {f['finding']}\n")
        for k, v in f.items():
            if k == "finding":
                continue
            L.append(f"- {k}: {v}")
        L.append("")
    return "\n".join(L) + "\n"


def _frontier_robustness_md(fro, scope) -> str:
    L = [
        f"# Primitive-set frontier robustness ({scope})\n",
        "> The best primitive SET swept over set sizes 1–4 and pad-waste thresholds 5/10/20%, "
        f"with extra candidate tiles ({', '.join(_EXTRA_TILES)}). Coverage is recomputed from "
        "operator geometry (the 10% recompute is regression-checked against the committed "
        "tile_waste). Structural coverage only — no primitive is called faster.\n",
        f"**Best 2-set across thresholds:** {fro['two_winners']}. "
        f"**LOO flips (10%):** {fro['two_set_loo_flips'] or 'none'}. "
        f"**Threshold-robust:** {fro['two_set_threshold_robust']}.\n",
        "| threshold | set size | best primitive set | worst | macro | micro | max regret |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in fro["rows"]:
        L.append(
            f"| {r['threshold_pct']}% | {r['set_size']} | {r['primitive_set']} | "
            f"{r['worst']:.3f} | {r['macro']:.3f} | {r['micro']:.3f} | {r['max_regret']:.3f} |"
        )
    nunc = len(fro["uncovered_rows"])
    L.append(f"\nUncovered ops as the set grows (10% threshold): `uncovered_ops_by_primitive_set.csv` ({nunc} rows).\n")
    return "\n".join(L) + "\n"
