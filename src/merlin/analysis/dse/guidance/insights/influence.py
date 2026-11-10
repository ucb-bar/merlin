"""Measure corpus aggregation sensitivity and per-family evidence composition.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows
from .workloads import _signal_by_workload

# ---- Part D: macro/micro + leave-one-out influence (winner-stable vs magnitude-stable)
_INFLUENCE_METRICS = [
    ("dense_gemm_mac_fraction", "dense_mac"),
    ("gemv_skinny_mac_fraction", "gemv_mac"),
    ("true_gemv_mac_fraction", "true_gemv_mac"),
    ("skinny_gemm_mac_fraction", "skinny_gemm_mac"),
]


_INFLUENCE_COLS = [
    "metric",
    "macro",
    "micro",
    "macro_micro_gap",
    "worst_workload",
    "worst",
    "best_workload",
    "best",
    "most_influential",
    "loo_micro_at_influential",
    "max_loo_micro_delta",
    "winner_stable_magnitude_unstable",
]


_LOO_DELTA_COLS = ["metric", "workload_removed", "loo_micro", "delta_vs_full"]


def macro_micro_influence(cs_dir: Path) -> dict:
    """Part D: per cross-workload MAC-fraction metric, report macro (equal-weight) vs micro
    (MAC-weighted), worst/best workload, the most influential workload (largest leave-one-out micro
    swing), and crucially flag metrics whose WINNER is stable but whose MAGNITUDE is not (the
    dense-GEMM case: drop pi05 and micro jumps ~0.04 -> ~0.65 though skinny still wins)."""
    cs_dir = Path(cs_dir)
    sig, wls = _signal_by_workload(cs_dir)
    if len(wls) < 2:
        return {"workloads": wls, "rows": [], "loo_rows": []}
    tot_mac = {w: 0.0 for w in wls}
    for o in _csv_rows(cs_dir / "operator_shape_table.csv"):
        if o["workload"] in tot_mac:
            tot_mac[o["workload"]] += int(o["macs"])
    corpus = sum(tot_mac.values()) or 1.0
    rows, loo_rows = [], []
    for label, key in _INFLUENCE_METRICS:
        vals = {w: float(sig[w][key]) for w in wls}
        macro = sum(vals.values()) / len(wls)
        micro = sum(vals[w] * tot_mac[w] for w in wls) / corpus
        loo = {w: (sum(vals[x] * tot_mac[x] for x in wls if x != w) / ((corpus - tot_mac[w]) or 1.0)) for w in wls}
        infl = max(wls, key=lambda w: abs(loo[w] - micro))
        max_delta = abs(loo[infl] - micro)
        worst_w = min(wls, key=lambda w: vals[w])
        best_w = max(wls, key=lambda w: vals[w])
        rows.append(
            {
                "metric": label,
                "macro": round(macro, 4),
                "micro": round(micro, 4),
                "macro_micro_gap": round(abs(macro - micro), 4),
                "worst_workload": worst_w,
                "worst": round(vals[worst_w], 4),
                "best_workload": best_w,
                "best": round(vals[best_w], 4),
                "most_influential": infl,
                "loo_micro_at_influential": round(loo[infl], 4),
                "max_loo_micro_delta": round(max_delta, 4),
                "winner_stable_magnitude_unstable": ("yes" if max_delta > 0.2 else "no"),
            }
        )
        for w in wls:
            loo_rows.append(
                {
                    "metric": label,
                    "workload_removed": w,
                    "loo_micro": round(loo[w], 4),
                    "delta_vs_full": round(loo[w] - micro, 4),
                }
            )
    # residency pressure is a byte ranking, not a fraction — report the rank separately
    reload_rank = sorted(wls, key=lambda w: -float(sig[w]["avoidable_reload"]))
    return {"workloads": wls, "rows": rows, "loo_rows": loo_rows, "residency_pressure_rank": reload_rank}


def per_family_summary(cs_dir: Path) -> dict:
    """Stage C (§5E): decompose the corpus by family (iterative_denoise vs token_decode) so corpus
    averages don't hide family behavior — per-family op counts, attention share, and visible-linear
    fraction from work_coverage_table + workload_family_table."""
    cs_dir = Path(cs_dir)
    fam = {r["workload"]: r["family"] for r in _csv_rows(cs_dir / "workload_family_table.csv")}
    wc = {r["workload"]: r for r in _csv_rows(cs_dir / "work_coverage_table.csv")}
    agg = {}
    for w, f in fam.items():
        r = wc.get(w)
        if not r:
            continue
        a = agg.setdefault(
            f,
            {
                "family": f,
                "workloads": [],
                "lin_macs": 0,
                "attn_macs": 0,
                "n_attention_ops": 0,
                "n_softmax": 0,
                "n_norm": 0,
            },
        )
        a["workloads"].append(w)
        a["lin_macs"] += int(r["linear_gemm_macs"])
        a["attn_macs"] += int(r["attention_macs"])
        a["n_attention_ops"] += int(r["n_attention_ops"])
        a["n_softmax"] += int(r["n_softmax"])
        a["n_norm"] += int(r["n_normalization"])
    rows = []
    for f, a in sorted(agg.items()):
        tot = a["lin_macs"] + a["attn_macs"]
        rows.append(
            {
                "family": f,
                "n_workloads": len(a["workloads"]),
                "workloads": "; ".join(sorted(a["workloads"])),
                "linear_gemm_macs": a["lin_macs"],
                "attention_macs": a["attn_macs"],
                "visible_linear_fraction": round(a["lin_macs"] / tot, 4) if tot else 1.0,
                "n_attention_ops": a["n_attention_ops"],
                "n_softmax": a["n_softmax"],
                "n_norm": a["n_norm"],
            }
        )
    return {"rows": rows}


def _influence_md(inf, scope) -> str:
    L = [
        f"# Workload influence / magnitude stability ({scope})\n",
        "> Per cross-workload MAC-fraction metric: macro (equal-weight) vs micro (MAC-weighted), the "
        "most influential workload (largest leave-one-out micro swing), and a flag for metrics whose "
        "**winner is stable but whose magnitude is not** — a metric can keep its ranking yet move "
        "sharply when one workload is dropped.\n",
        "| metric | macro | micro | gap | most influential | max LOO micro Δ | winner-stable/magnitude-unstable |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in inf["rows"]:
        L.append(
            f"| {r['metric']} | {r['macro']:.3f} | {r['micro']:.3f} | {r['macro_micro_gap']:.3f}"
            f" | {r['most_influential']} | {r['max_loo_micro_delta']:.3f} | "
            f"**{r['winner_stable_magnitude_unstable']}** |"
        )
    L.append(
        f"\nResidency-pressure ranking (avoidable weight reload, bytes; ranking is the robust "
        f"signal, absolute bytes are random-init): {' > '.join(inf['residency_pressure_rank'])}."
    )
    L.append("Per-(metric, workload) leave-one-out micro: `leave_one_out_delta_table.csv`.\n")
    return "\n".join(L) + "\n"


_PER_FAMILY_COLS = [
    "family",
    "n_workloads",
    "workloads",
    "linear_gemm_macs",
    "attention_macs",
    "visible_linear_fraction",
    "n_attention_ops",
    "n_softmax",
    "n_norm",
]


def _per_family_md(pf, scope) -> str:
    L = [
        f"# Per-family decomposition ({scope})\n",
        "> Corpus split by family so averages don't hide family behavior. Linear-GEMM vs attention MAC "
        "mass and op mix per family (structural; recovered from IR).\n",
        "| family | workloads | linear MACs | attention MACs | visible_linear_frac | attn ops | softmax | norm |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in pf["rows"]:
        L.append(
            f"| {r['family']} | {r['n_workloads']} | {r['linear_gemm_macs']} | "
            f"{r['attention_macs']} | {r['visible_linear_fraction']:.3f} | {r['n_attention_ops']} | "
            f"{r['n_softmax']} | {r['n_norm']} |"
        )
    return "\n".join(L) + "\n"
