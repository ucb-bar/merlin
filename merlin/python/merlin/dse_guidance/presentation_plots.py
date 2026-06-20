"""Neutral PNG renderers for the insight-mining plot manifest.

Renders the plot-manifest entries that have a known renderer + present source columns into PNGs
under a run's ``generated_plots/``. Styling is neutral and every axis is a structural quantity
(counts / bytes / fractions / evidence tiers) — nothing here plots latency, throughput, or any
performance/speedup quantity. Output lands in the non-committed ``results/`` run folder, so PNG
determinism is not required (the byte-stable guarantee is on the committed case_study, not here).

If matplotlib is unavailable the renderer no-ops and returns ``[]`` (the manifest is still emitted).
"""
from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from pathlib import Path


def _have_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return True
    except Exception:
        return False


def _rows(p: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(p.read_text()))) if p.is_file() else []


# muted academic pastel palette (matches final_analysis.html / the user's reference plots)
PALETTE = ["#5e8db4", "#cf8a82", "#8fa674", "#d2a23f", "#9d7fae", "#7c9aa6", "#b08968"]
_BG = "#faf7f1"


def _pastel_cmap():
    """A soft cream->sage->deep sequential colormap (replaces viridis/YlGnBu)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "merlin_pastel", ["#f4efe6", "#d9d2bd", "#9caf88", "#5e8db4", "#3a5f7c"])


def _style():
    import matplotlib.pyplot as plt
    from cycler import cycler
    # conference-grade legibility: every text element >= 8 pt at the rendered dpi.
    plt.rcParams.update({
        "figure.figsize": (8.6, 5.0), "figure.dpi": 150, "savefig.dpi": 150,
        "font.size": 10, "font.family": "serif",
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#b9ad97",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b9ad97", "axes.labelcolor": "#2f2a23", "axes.titlecolor": "#2f2a23",
        "text.color": "#2f2a23", "xtick.color": "#5c5446", "ytick.color": "#5c5446",
        "axes.prop_cycle": cycler(color=PALETTE)})


# ---- honesty annotations: every figure carries its evidence tier + magnitude-scale source -----------
# tier: A = IR/measured exact, B = recovered + recompute-checked, C = config/assumed.
# scale: how to read any absolute magnitude on the axes (the reviewer-facing T1 separation).
_DEPLOY = "deployment-composition (real config; exact for layer-identical stacks)"
_CAPTURED = "captured-config (structural ratios/shapes; NOT deployment scale)"
_NOSCALE = "structural (counts/fractions; no absolute magnitude)"
_PLOT_META = {
    "evidence_type_by_workload": ("A/B/C", _NOSCALE, "Provenance mix of recovered facts per workload; bar = fact count by evidence tier."),
    "evidence_type_by_phase": ("A/B/C", _NOSCALE, "Provenance mix of recovered facts per analysis phase."),
    "shape_class_mac_share": ("A", _CAPTURED, "Per-workload MAC split by GEMM shape class (from IR shapes); composition to 1."),
    "primitive_coverage_heatmap": ("A", _NOSCALE, "Fraction of each workload's MACs a primitive covers at <=10% pad waste (top primitives)."),
    "primitive_regret_bar": ("A", _NOSCALE, "Per-primitive coverage and worst-case cross-workload regret (shape geometry only)."),
    "boundary_placement_heatmap": ("B", _NOSCALE, "HW/SW-boundary placement status per abstraction x level (top by boundary pressure)."),
    "resident_capacity_by_dtype": ("A", _CAPTURED, "Resident weight bytes per region by dtype. Scale is captured-config, not deployment."),
    "avoidable_reload_by_region": ("A", _CAPTURED, "Weight bytes re-loaded across the loop if non-resident (captured-config scale; log axis)."),
    "measurement_priority_bar": ("B", _NOSCALE, "How many abstraction candidates each missing measurement would unblock."),
    "critical_path_parallelism": ("A", _NOSCALE, "Inter-op available parallelism (work/span) from the IR dependency graph."),
    "decision_primitive_choice": ("A", _NOSCALE, "If DSE commits to one primitive: worst vs mean MAC coverage (<=10% waste)."),
    "decision_weight_residency": ("A", _CAPTURED, "Weight bytes moved vs loop count K (IR-recovered); reload-every-step vs resident. Captured-config bytes."),
    "decision_capacity_dtype": ("A", _CAPTURED, "# workloads weight-resident as on-chip budget grows, per dtype (captured-config bytes)."),
    "decision_sharding_cost": ("A", _CAPTURED, "Extra data-movement bytes from 2/4/8-way sharding along M/N/K (captured-config bytes)."),
    "primitive_set_frontier": ("A", _NOSCALE, "Primitive-set frontier: mean vs worst-workload coverage (upper-right = broadly useful)."),
    "operator_cumulative_mac": ("A", _NOSCALE, "Compute concentration: cumulative MAC share vs top-k operators (shape ratio)."),
    "boundary_necessity_matrix": ("B", _NOSCALE, "Abstraction necessity per workload (N/U/P/B/-); analysis over recovered facts."),
    "decision_sharding_per_top_op": ("A", _CAPTURED, "Sharding extra bytes for the top-MAC ops, normalized by output bytes."),
    "primitive_frontier_by_threshold": ("A", _NOSCALE, "Frontier robustness: worst coverage vs set size across pad-waste thresholds."),
    "macro_vs_micro_primitive_coverage": ("A", _NOSCALE, "Macro (mean) vs micro (MAC-weighted) vs worst coverage by primitive-set size."),
    "required_compute_envelope": ("A/C", _CAPTURED, "Required GMAC/s vs replan deadline (configured K). A REQUIREMENT, not measured perf."),
    "required_memory_movement_envelope": ("A/C", _CAPTURED, "Required weight B/s @100ms: residency removes a Kx factor. Requirement, not perf."),
    "required_command_rate_envelope": ("C", _CAPTURED, "Required dispatch/s vs deadline. PROXY (~12x undercount); measured only for small_llama."),
    "workload_influence_loo_delta": ("B", _NOSCALE, "Leave-one-out stability of corpus metrics (red = winner-stable, magnitude-unstable)."),
    "work_coverage_by_workload": ("A", _CAPTURED, "Recovered linear-GEMM vs attention MAC mass per workload (IR shapes; captured-config; log)."),
    "visible_linear_fraction": ("A", _NOSCALE, "Share of recovered MAC work that is linear-GEMM geometry (rest = attention)."),
    "deployment_magnitude": ("B", _DEPLOY, "Deployment params & MACs/replan by config-composition (embed + per-layer x real n_layers)."),
    "arithmetic_intensity_roofline": ("A/B", _DEPLOY, "HW-INDEPENDENT roofline: arithmetic intensity (MAC/byte) resident vs reload-every-step. No chip assumed."),
}


def _stamp(fig, plot_id):
    """Stamp the evidence-tier + scale-source badge and a one-line caption on a figure."""
    tier, scale, caption = _PLOT_META.get(
        plot_id, ("B", _NOSCALE, ""))
    fig.text(0.008, 0.985, f"evidence: Tier {tier}", fontsize=8.5, va="top", ha="left",
             family="monospace", color="#2f2a23",
             bbox=dict(boxstyle="round,pad=0.3", fc="#efe7d6", ec="#b9ad97", lw=0.6))
    fig.text(0.992, 0.985, f"scale: {scale}", fontsize=8.0, va="top", ha="right",
             style="italic", color="#5c5446")
    if caption:
        fig.text(0.5, 0.012, caption, fontsize=8.0, va="bottom", ha="center",
                 style="italic", color="#3a352c", wrap=True)


def _save(fig, out: Path, plot_id=None):
    # leave headroom for the top badges + a footer caption, then stamp.
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    if plot_id is not None:
        _stamp(fig, plot_id)
    fig.savefig(out)
    import matplotlib.pyplot as plt
    plt.close(fig)


# each renderer: (cs_dir, facts, ax) -> bool drew_something

def _stacked_count(ax, facts, key, series_key, title):
    groups = defaultdict(Counter)
    for f in facts:
        groups[f[key]][f[series_key]] += 1
    cats = sorted(groups)
    series = sorted({s for c in groups.values() for s in c})
    bottoms = [0] * len(cats)
    for s in series:
        vals = [groups[c].get(s, 0) for c in cats]
        ax.bar(cats, vals, bottom=bottoms, label=s)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title(title)
    ax.set_ylabel("fact count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, ncol=2)
    return True


def _r_evidence_by_workload(cs, facts, ax):
    return _stacked_count(ax, facts, "workload", "evidence_type", "Evidence type by workload")


def _r_evidence_by_phase(cs, facts, ax):
    return _stacked_count(ax, facts, "source_phase", "evidence_type", "Evidence type by phase")


def _r_shape_mac_share(cs, facts, ax):
    rows = _rows(cs / "shape_summary_by_workload.csv")
    if not rows:
        return False
    g = defaultdict(dict)
    for r in rows:
        g[r["workload"]][r["shape_class"]] = float(r["mac_fraction"])
    wls = sorted(g)
    classes = sorted({c for d in g.values() for c in d})
    bottoms = [0.0] * len(wls)
    for c in classes:
        vals = [g[w].get(c, 0.0) for w in wls]
        ax.bar(wls, vals, bottom=bottoms, label=c)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Shape-class MAC share by workload")
    ax.set_ylabel("MAC fraction")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    return True


def _r_primitive_coverage(cs, facts, ax):
    rows = _rows(cs / "primitive_coverage_matrix.csv")
    if not rows:
        return False
    wls = sorted({r["workload"] for r in rows})
    cov = defaultdict(dict)
    for r in rows:
        cov[r["primitive"]][r["workload"]] = float(r["coverage_under_10pct"])
    # top-N primitives by mean coverage (keeps the figure legible at print size)
    prims = sorted(cov, key=lambda p: -sum(cov[p].values()) / max(len(cov[p]), 1))[:16]
    mat = [[cov[p].get(w, 0.0) for w in wls] for p in prims]
    im = ax.imshow(mat, aspect="auto", cmap=_pastel_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(wls)), wls, rotation=20)
    ax.set_yticks(range(len(prims)), prims, fontsize=8)
    ax.set_title(f"Primitive x workload coverage (<=10% pad waste; top {len(prims)} primitives)")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return True


def _r_primitive_regret(cs, facts, ax):
    rows = _rows(cs / "primitive_regret_table.csv")
    if not rows:
        return False
    prims = [r["primitive"] for r in rows]
    x = range(len(prims))
    ax.bar([i - 0.2 for i in x], [float(r["coverage_under_10pct"]) for r in rows], 0.4,
           label="coverage <=10%")
    ax.bar([i + 0.2 for i in x], [float(r["max_regret"]) for r in rows], 0.4, label="max regret")
    ax.set_xticks(list(x), prims, rotation=35, fontsize=8)
    ax.set_title("Primitive coverage + cross-workload regret")
    ax.set_ylabel("MAC fraction")
    ax.legend(fontsize=8)
    return True


def _r_boundary_heatmap(cs, facts, ax):
    rows = _rows(cs / "hw_sw_boundary_matrix.csv")
    if not rows:
        return False
    levels = ["compiler_transform", "runtime_hal_object", "command_buffer_or_command_isa",
              "accelerator_isa", "device_microcode_or_controller", "fixed_hardware_datapath"]
    score = {"strong_candidate": 4, "possible": 3, "weak_candidate": 2, "blocked": 1,
             "not_applicable": 0, "unavailable": 0}
    rows = sorted(rows, key=lambda r: -int(r["boundary_pressure_score"]))[:14]
    abst = [r["abstraction"] for r in rows]
    mat = [[score.get(r[lv], 0) for lv in levels] for r in rows]
    im = ax.imshow(mat, aspect="auto", cmap=_pastel_cmap(), vmin=0, vmax=4)
    ax.set_xticks(range(len(levels)), [lv.replace("_", "\n") for lv in levels], fontsize=8)
    ax.set_yticks(range(len(abst)), abst, fontsize=8)
    ax.set_title("Boundary placement: abstraction x level (status)")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return True


def _r_resident_capacity(cs, facts, ax):
    rows = _rows(cs / "data_movement_table.csv")
    if not rows:
        return False
    labels = [f"{r['workload']}/{r['region']}" for r in rows]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [int(r["resident_int8_B"]) for r in rows], 0.4, label="int8")
    ax.bar([i + 0.2 for i in x], [int(r["resident_bf16_B"]) for r in rows], 0.4, label="bf16")
    ax.set_xticks(list(x), labels, rotation=30, fontsize=8)
    ax.set_title("Resident weight capacity by dtype (per region)")
    ax.set_ylabel("bytes")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return True


def _r_avoidable_reload(cs, facts, ax):
    rows = _rows(cs / "data_movement_table.csv")
    if not rows:
        return False
    labels = [f"{r['workload']}/{r['region']}" for r in rows]
    ax.bar(labels, [int(r["avoidable_weight_reload"]) for r in rows])
    ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=8)
    ax.set_title("Avoidable weight reload by region")
    ax.set_ylabel("bytes")
    ax.set_yscale("log")
    return True


def _r_measurement_priority(cs, facts, ax):
    rows = _rows(cs / "measurement_priority_table.csv")
    if not rows:
        return False
    # top-N only — the full table has ~30 rows and is unreadable at print size.
    rows = sorted(rows, key=lambda r: -int(r["n_candidates_unblocked"]))[:14]
    ax.barh([r["measurement"][:40] for r in rows][::-1],
            [int(r["n_candidates_unblocked"]) for r in rows][::-1])
    ax.set_title(f"Candidates unblocked per measurement (top {len(rows)})")
    ax.set_xlabel("candidates")
    ax.tick_params(axis="y", labelsize=8)
    return True


def _r_critical_path(cs, facts, ax):
    rows = _rows(cs / "critical_path_table.csv")
    if not rows:
        return False
    ax.bar([r["workload"] for r in rows], [float(r["available_parallelism"]) for r in rows])
    ax.set_title("Available inter-op parallelism (work/span)")
    ax.set_ylabel("work / span")
    ax.tick_params(axis="x", rotation=20)
    return True


# ---- decision-impact ("what-if") renderers: outcome as a function of a DSE knob choice ----

def _r_decision_primitive_choice(cs, facts, ax):
    """If DSE commits to ONE primitive: worst-case vs mean MAC coverage across workloads."""
    rows = _rows(cs / "primitive_coverage_matrix.csv")
    if not rows:
        return False
    by_prim = defaultdict(list)
    for r in rows:
        by_prim[r["primitive"]].append(float(r["coverage_under_10pct"]))
    prims = sorted(by_prim, key=lambda p: -min(by_prim[p]))      # safest single choice first
    worst = [min(by_prim[p]) for p in prims]
    mean = [sum(by_prim[p]) / len(by_prim[p]) for p in prims]
    x = range(len(prims))
    ax.bar([i - 0.2 for i in x], worst, 0.4, label="worst workload")
    ax.bar([i + 0.2 for i in x], mean, 0.4, label="mean workload")
    ax.set_xticks(list(x), prims, rotation=35, fontsize=8)
    ax.set_title("Decision: single primitive choice -> MAC coverage (<=10% waste)")
    ax.set_ylabel("MAC fraction covered")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    return True


def _r_decision_weight_residency(cs, facts, ax):
    """Weight bytes moved vs loop count: reload-every-step (linear) vs resident (flat)."""
    rows = [r for r in _rows(cs / "data_movement_table.csv")
            if r["region"] == "repeated_head" and int(r["weight_bytes"]) > 0]
    if not rows:
        return False
    kmax = max(max(int(r["invocations"]), 2) for r in rows)
    ks = list(range(1, kmax + 1))
    for r in rows:
        wb = int(r["weight_bytes"])
        line, = ax.plot(ks, [wb * k for k in ks], label=f"{r['workload']} reload")
        ax.plot(ks, [wb] * len(ks), "--", color=line.get_color(), alpha=0.6)
        kr = int(r["invocations"])
        ax.scatter([kr], [wb * kr], color=line.get_color(), zorder=5, s=18)
    ax.set_title("Decision: weight residency -> bytes moved vs loop count\n"
                 "(solid=reload every step, dashed=resident; dot=IR-recovered K (scf.for trip count))")
    ax.set_xlabel("head loop count K")
    ax.set_ylabel("weight bytes moved")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    return True


def _r_decision_capacity_dtype(cs, facts, ax):
    """How many workloads are fully weight-resident as the capacity budget grows, per dtype."""
    rows = _rows(cs / "dtype_capacity_table.csv")
    if not rows:
        return False
    cols = [("bf16_B", "bf16"), ("int8_B", "int8"), ("int4_B", "int4")]
    allv = [int(float(r[c])) for r in rows for c, _ in cols]
    lo, hi = min(allv), max(allv)
    import math
    budgets = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / 40)
               for i in range(41)] if hi > lo else [lo]
    for col, name in cols:
        sizes = [int(float(r[col])) for r in rows]
        ax.step(budgets, [sum(1 for s in sizes if s <= b) for b in budgets], where="post",
                label=name)
    ax.set_title("Decision: on-chip capacity + dtype -> repeated-head weights resident")
    ax.set_xlabel("on-chip capacity budget (bytes)")
    ax.set_ylabel(f"# workloads w/ repeated-head weights resident (of {len(rows)})")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    return True


def _r_decision_sharding_cost(cs, facts, ax):
    """Extra data-movement bytes added by sharding 2/4/8-ways along M / N / K."""
    rows = _rows(cs / "sharding_table.csv")
    if not rows:
        return False
    counts = [2, 4, 8]
    axes_ = ["M", "N", "K"]
    tot = {a: [0.0, 0.0, 0.0] for a in axes_}
    for r in rows:
        a = r["axis"]
        if a not in tot:
            continue
        per = float(r["per_extra_shard_bytes"])
        for j, s in enumerate(counts):
            if r.get(f"shardable_{s}") == "True":
                tot[a][j] += per * (s - 1)
    x = range(len(counts))
    for k, a in enumerate(axes_):
        ax.bar([i + (k - 1) * 0.25 for i in x], tot[a], 0.25, label=f"{a}-shard")
    ax.set_xticks(list(x), [f"{s}-way" for s in counts])
    ax.set_title("Decision: shard axis + count -> extra data-movement bytes")
    ax.set_ylabel("extra bytes (partial-sum / broadcast)")
    ax.set_yscale("symlog")
    ax.legend(fontsize=8)
    return True


# ---- P16 decision-frontier & robustness renderers ----

def _r_primitive_set_frontier(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    fr = IM.primitive_set_frontier(cs)
    singles = fr.get("singles", [])
    if not singles:
        return False
    ax.scatter([s["macro"] for s in singles], [s["worst"] for s in singles],
               c="#888", label="single primitive", zorder=3)
    for s in singles:
        ax.annotate(s["primitive"].replace("_", ""), (s["macro"], s["worst"]), fontsize=8,
                    xytext=(2, 2), textcoords="offset points")
    markers = {1: "o", 2: "*", 3: "P"}
    for size, b in fr.get("best_by_size", {}).items():
        ax.scatter([b["macro"]], [b["worst"]], marker=markers.get(size, "s"), s=160, zorder=5,
                   label=f"best {size}-set")
    ax.plot([0, 1], [0, 1], "--", color="#b9ad97", zorder=1)
    ax.set_xlabel("mean (macro) coverage")
    ax.set_ylabel("worst-workload coverage")
    ax.set_title("Primitive-set frontier (upper-right = broadly useful)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)
    return True


def _r_operator_cumulative_mac(cs, facts, ax):
    rows = _rows(cs / "operator_shape_table.csv")
    if not rows:
        return False
    for w in sorted({r["workload"] for r in rows}):
        vals = sorted((int(r["macs"]) for r in rows if r["workload"] == w), reverse=True)
        tot = sum(vals) or 1
        cum, acc = [], 0
        for v in vals:
            acc += v
            cum.append(acc / tot)
        ax.plot(range(1, len(cum) + 1), cum, marker=".", label=w)
    ax.axhline(0.9, ls="--", color="#b9ad97")
    ax.set_xlabel("top-k operators (by MACs)")
    ax.set_ylabel("cumulative MAC share")
    ax.set_title("How concentrated is compute? (steep = a few giant ops)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    return True


_NEC_RANK = {"necessary": 4, "useful": 3, "possible": 2, "blocked": 1, "not_applicable": 0}
_NEC_ABBR = {"necessary": "N", "useful": "U", "possible": "P", "blocked": "B", "not_applicable": "–"}


def _r_boundary_necessity_matrix(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    nec = IM.abstraction_necessity(cs)
    wls = nec["workloads"]
    rows = nec["rows"][:12]                       # already sorted necessary-first
    if not rows:
        return False
    mat = [[_NEC_RANK[r[w]] for w in wls] for r in rows]
    im = ax.imshow(mat, aspect="auto", cmap=_pastel_cmap(), vmin=0, vmax=4)
    ax.set_xticks(range(len(wls)), wls, rotation=20, fontsize=8)
    ax.set_yticks(range(len(rows)), [r["abstraction"] for r in rows], fontsize=8)
    for i, r in enumerate(rows):
        for j, w in enumerate(wls):
            ax.text(j, i, _NEC_ABBR[r[w]], ha="center", va="center", fontsize=8,
                    color="white" if mat[i][j] >= 3 else "black")
    ax.set_title("Abstraction necessity (N=necessary U=useful P=possible B=blocked –=N/A)")
    return True


def _r_decision_sharding_per_top_op(cs, facts, ax):
    ops = _rows(cs / "operator_shape_table.csv")
    shard = _rows(cs / "sharding_table.csv")
    if not ops or not shard:
        return False
    top = sorted(ops, key=lambda o: -int(o["macs"]))[:5]
    by_op = {}
    for r in shard:
        by_op.setdefault((r["workload"], r["op_index"]), {})[r["axis"]] = r
    axes_ = ["M", "N", "K"]
    labels, data = [], {a: [] for a in axes_}
    for o in top:
        key = (o["workload"], o["op_index"])
        outb = int(o["output_bytes"]) or 1
        labels.append(o["prov_fqn"].split(".")[-1][:14])
        for a in axes_:
            r = by_op.get(key, {}).get(a)
            extra = (float(r["per_extra_shard_bytes"]) * 7 / outb) if r else 0.0
            data[a].append(extra)
    x = range(len(labels))
    for k, a in enumerate(axes_):
        ax.bar([i + (k - 1) * 0.25 for i in x], data[a], 0.25, label=f"{a}-shard")
    ax.set_xticks(list(x), labels, rotation=30, fontsize=8)
    ax.set_ylabel("8-way extra bytes / output bytes")
    ax.set_title("Decision: shard top-MAC ops (extra bytes normalized by output)")
    ax.legend(fontsize=8)
    return True


# --------------------------------------------------------------------------- P17 decision plots

def _r_primitive_frontier_by_threshold(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    fro = IM.primitive_frontier_robustness(cs)
    if not fro["rows"]:
        return False
    by_thr = {}
    for r in fro["rows"]:
        by_thr.setdefault(r["threshold_pct"], []).append((r["set_size"], r["worst"]))
    for thr in sorted(by_thr):
        pts = sorted(by_thr[thr])
        ax.plot([s for s, _ in pts], [w for _, w in pts], marker="o", label=f"{thr}% pad waste")
    ax.set_xlabel("primitive set size")
    ax.set_ylabel("worst-workload coverage")
    ax.set_title("Frontier robustness: worst coverage vs set size, by threshold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    return True


def _r_macro_vs_micro_primitive_coverage(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    fro = IM.primitive_frontier_robustness(cs)
    rows = sorted((r for r in fro["rows"] if r["threshold_pct"] == 10), key=lambda r: r["set_size"])
    if not rows:
        return False
    sz = [r["set_size"] for r in rows]
    ax.plot(sz, [r["macro"] for r in rows], marker="o", label="macro (mean)")
    ax.plot(sz, [r["micro"] for r in rows], marker="s", label="micro (MAC-weighted)")
    ax.plot(sz, [r["worst"] for r in rows], marker="^", label="worst-workload")
    ax.set_xlabel("primitive set size (10% pad waste)")
    ax.set_ylabel("coverage")
    ax.set_title("Macro vs micro vs worst primitive coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    return True


def _r_required_compute_envelope(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    env = IM.timing_requirement_envelope(cs)
    rows = [r for r in env["rows"] if r["K_basis"] == "configured" and r["deadline_basis"] == "sweep"]
    if not rows:
        return False
    for w in env["workloads"]:
        pts = sorted((r["deadline_ms"], r["required_compute_MAC_per_s"]) for r in rows
                     if r["workload"] == w)
        if pts:
            ax.plot([d for d, _ in pts], [v / 1e9 for _, v in pts], marker=".", label=w)
    ax.set_xlabel("replan deadline (ms)")
    ax.set_ylabel("required GMAC/s (configured K)")
    ax.set_yscale("log")
    ax.set_title("Required compute envelope (a requirement, not measured performance)")
    ax.legend(fontsize=8, ncol=2)
    return True


def _r_required_memory_movement_envelope(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    env = IM.timing_requirement_envelope(cs)
    rows = [r for r in env["rows"] if r["K_basis"] == "configured" and r["deadline_basis"] == "sweep"
            and int(r["deadline_ms"]) == 100]
    if not rows:
        return False
    rows.sort(key=lambda r: -r["required_weight_B_per_s_nonresident"])
    w = [r["workload"] for r in rows]
    nr = [r["required_weight_B_per_s_nonresident"] for r in rows]
    rs = [r["required_weight_B_per_s_resident"] for r in rows]
    x = range(len(w))
    ax.bar([i - 0.2 for i in x], nr, 0.4, label="non-resident (reload x K)")
    ax.bar([i + 0.2 for i in x], rs, 0.4, label="resident (load once)")
    ax.set_yscale("log")
    ax.set_xticks(list(x), w, rotation=30, fontsize=8)
    ax.set_ylabel("required weight B/s @ 100 ms")
    ax.set_title("Required memory-movement envelope: residency removes a K x requirement")
    ax.legend(fontsize=8)
    return True


def _r_required_command_rate_envelope(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    env = IM.timing_requirement_envelope(cs)
    rows = [r for r in env["rows"] if r["K_basis"] == "configured" and r["deadline_basis"] == "sweep"]
    if not rows:
        return False
    for w in env["workloads"]:
        pts = sorted((r["deadline_ms"], r["required_command_rate_per_s"]) for r in rows
                     if r["workload"] == w)
        if pts:
            ax.plot([d for d, _ in pts], [v for _, v in pts], marker=".", label=w)
    ax.set_xlabel("replan deadline (ms)")
    ax.set_ylabel("required dispatch/s (PROXY, ~12x undercount)")
    ax.set_yscale("log")
    ax.set_title("Required command-rate envelope (proxy; measured only for small_llama)")
    ax.legend(fontsize=8, ncol=2)
    return True


def _r_work_coverage_by_workload(cs, facts, ax):
    rows = _rows(cs / "work_coverage_table.csv")
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["visible_linear_fraction"]))
    wl = [r["workload"] for r in rows]
    lin = [float(r["linear_gemm_macs"]) for r in rows]
    att = [float(r["attention_macs"]) for r in rows]
    x = range(len(wl))
    # grouped (NOT stacked) on a log axis — stacked bars on log mislead (heights don't add on log).
    ax.bar([i - 0.2 for i in x], lin, 0.4, label="linear GEMM MACs")
    ax.bar([i + 0.2 for i in x], att, 0.4, label="attention MACs (recovered)")
    ax.set_yscale("log")
    ax.set_xticks(list(x), wl, rotation=30, fontsize=8)
    ax.set_ylabel("recovered MACs (log10)")
    ax.set_title("Recovered work: linear-GEMM vs attention MAC mass (no config; from IR shapes)")
    ax.legend(fontsize=8)
    return True


def _r_visible_linear_fraction(cs, facts, ax):
    rows = _rows(cs / "work_coverage_table.csv")
    if not rows:
        return False
    rows.sort(key=lambda r: float(r["visible_linear_fraction"]))
    wl = [r["workload"] for r in rows]
    frac = [float(r["visible_linear_fraction"]) for r in rows]
    ax.barh(range(len(wl)), frac, color=PALETTE[0])
    ax.set_yticks(range(len(wl)), wl, fontsize=8)
    ax.set_xlabel("visible_linear_fraction = linear / (linear + attention)")
    ax.set_xlim(0, 1.02)
    ax.set_title("How much recovered MAC work is linear-GEMM geometry (rest = attention)")
    return True


def _r_workload_influence_loo_delta(cs, facts, ax):
    from merlin.dse_guidance import insight_mining as IM
    inf = IM.macro_micro_influence(cs)
    rows = inf["rows"]
    if not rows:
        return False
    labels = [r["metric"].replace("_mac_fraction", "").replace("_", " ") for r in rows]
    vals = [r["max_loo_micro_delta"] for r in rows]
    cols = [PALETTE[1] if r["winner_stable_magnitude_unstable"] == "yes" else PALETTE[0] for r in rows]
    ax.bar(range(len(rows)), vals, color=cols)
    ax.axhline(0.2, ls="--", color="#b9ad97", label="magnitude-unstable threshold")
    ax.set_xticks(range(len(rows)), labels, rotation=20, fontsize=8)
    ax.set_ylabel("max leave-one-out micro change")
    ax.set_title("Workload influence (red = winner-stable but magnitude-unstable)")
    ax.legend(fontsize=8)
    return True


def _r_deployment_magnitude(cs, facts, ax):
    """Deployment-scale params & MACs/replan by config-composition (the T1 magnitude fix)."""
    rows = _rows(cs / "real_config_magnitudes.csv")
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: -float(r["total_gemm_params"]))
    wl = [r["workload"] for r in rows]
    params = [float(r["total_gemm_params"]) for r in rows]
    macs = [float(r["gemm_macs_per_token"]) for r in rows]
    x = range(len(wl))
    ax.bar([i - 0.2 for i in x], params, 0.4, label="GEMM params")
    ax.bar([i + 0.2 for i in x], macs, 0.4, label="GEMM MACs / token")
    ax.set_yscale("log")
    ax.set_xticks(list(x), wl, rotation=25, fontsize=8)
    ax.set_ylabel("count (log10)")
    ax.set_title("Deployment magnitudes by config-composition (params, MACs/token)")
    ax.legend(fontsize=8)
    return True


def _r_arithmetic_intensity_roofline(cs, facts, ax):
    """HW-INDEPENDENT roofline: arithmetic intensity (MAC/byte), resident vs reload-every-step."""
    rows = _rows(cs / "arithmetic_intensity.csv")
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: -float(r["ai_resident_mac_per_byte"]))
    wl = [r["workload"] for r in rows]
    res = [float(r["ai_resident_mac_per_byte"]) for r in rows]
    non = [float(r["ai_nonresident_mac_per_byte"]) for r in rows]
    gain = [float(r["residency_gain"]) for r in rows]
    x = range(len(wl))
    ax.bar([i - 0.2 for i in x], non, 0.4, label="reload every step (floor = 1/dtype)")
    ax.bar([i + 0.2 for i in x], res, 0.4, label="weights resident")
    for i, g in zip(x, gain):
        ax.annotate(f"{g:.1f}x", (i + 0.2, res[i]), fontsize=8, ha="center", va="bottom",
                    color="#5c5446")
    ax.set_xticks(list(x), wl, rotation=25, fontsize=8)
    ax.set_ylabel("arithmetic intensity (MAC / byte)")
    ax.set_title("HW-independent roofline x-axis: AI resident vs reload (label = residency gain)")
    ax.legend(fontsize=8)
    return True


_RENDERERS = {
    "deployment_magnitude": _r_deployment_magnitude,
    "arithmetic_intensity_roofline": _r_arithmetic_intensity_roofline,
    "evidence_type_by_workload": _r_evidence_by_workload,
    "evidence_type_by_phase": _r_evidence_by_phase,
    "shape_class_mac_share": _r_shape_mac_share,
    "primitive_coverage_heatmap": _r_primitive_coverage,
    "primitive_regret_bar": _r_primitive_regret,
    "boundary_placement_heatmap": _r_boundary_heatmap,
    "resident_capacity_by_dtype": _r_resident_capacity,
    "avoidable_reload_by_region": _r_avoidable_reload,
    "measurement_priority_bar": _r_measurement_priority,
    "critical_path_parallelism": _r_critical_path,
    "decision_primitive_choice": _r_decision_primitive_choice,
    "decision_weight_residency": _r_decision_weight_residency,
    "decision_capacity_dtype": _r_decision_capacity_dtype,
    "decision_sharding_cost": _r_decision_sharding_cost,
    "primitive_set_frontier": _r_primitive_set_frontier,
    "operator_cumulative_mac": _r_operator_cumulative_mac,
    "boundary_necessity_matrix": _r_boundary_necessity_matrix,
    "decision_sharding_per_top_op": _r_decision_sharding_per_top_op,
    "primitive_frontier_by_threshold": _r_primitive_frontier_by_threshold,
    "macro_vs_micro_primitive_coverage": _r_macro_vs_micro_primitive_coverage,
    "required_compute_envelope": _r_required_compute_envelope,
    "required_memory_movement_envelope": _r_required_memory_movement_envelope,
    "required_command_rate_envelope": _r_required_command_rate_envelope,
    "workload_influence_loo_delta": _r_workload_influence_loo_delta,
    "work_coverage_by_workload": _r_work_coverage_by_workload,
    "visible_linear_fraction": _r_visible_linear_fraction,
}


def render_plots(plot_manifest, cs_dir, facts, out_dir) -> list[str]:
    """Render every manifest plot that has a renderer; return the list of rendered plot_ids."""
    if not _have_mpl():
        return []
    import matplotlib.pyplot as plt
    _style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cs_dir = Path(cs_dir)
    rendered = []
    manifest_ids = {p["plot_id"] for p in plot_manifest}
    # always-on headline figures (deployment-scale + HW-independent roofline) even if the manifest
    # predates them — they are the reviewer-facing magnitude/roofline figures (Phase B/C).
    extra = [{"plot_id": pid} for pid in ("deployment_magnitude", "arithmetic_intensity_roofline")
             if pid not in manifest_ids]
    for p in list(plot_manifest) + extra:
        r = _RENDERERS.get(p["plot_id"])
        if r is None or p.get("recommendation") == "omit":
            continue
        fig, ax = plt.subplots()
        try:
            ok = r(cs_dir, facts, ax)
        except Exception:
            ok = False
        if ok:
            _save(fig, out_dir / f"{p['plot_id']}.png", plot_id=p["plot_id"])
            rendered.append(p["plot_id"])
        else:
            plt.close(fig)
    return rendered
