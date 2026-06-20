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
    "primitive_regret_bar": ("A", _NOSCALE, "If you build ONE primitive: fraction of MACs covered (<=10% pad waste), mean across the corpus vs the worst single workload."),
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
    "arithmetic_intensity_roofline": ("A/B", _DEPLOY, "HW-INDEPENDENT roofline: AI (MAC/byte) on x; ridge is a parametric band, not a chip; arrows = residency moving each workload toward compute-bound."),
    "capture_fidelity": ("A/B", _NOSCALE, "Per feature x workload: structural contract recovered from IR (S/R/M) vs erased (x) vs intentionally not-claimed (—). The contribution, made visual."),
    "table_capture_summary": ("A", _NOSCALE, "Recovered loop contract per workload: K, repeated-region ops, loop-carried operands, KV cache — all Tier-A from scf.for."),
    "table_low_bit_tiers": ("A/B", _NOSCALE, "Low-bit tier per workload; int8 accuracy ratified by the measured gate; fp8/int4 never assumed."),
    "table_deployment_magnitudes": ("B", _DEPLOY, "Deployment params/MACs by config-composition; openVLA & tiny_llama are exact external anchors."),
    "table_arithmetic_intensity": ("A/B", _DEPLOY, "Arithmetic intensity resident vs reload + residency gain per workload (HW-independent)."),
    "realtime_requirement": ("A/B", _DEPLOY, "Weight-bandwidth a machine MUST provide to hit 30Hz real-time (resident vs reload); regime=design target, not a chip's performance."),
    "table_realtime_requirement": ("A/B", _DEPLOY, "Per VLA/VLM real-time regime: required compute + weight bandwidth (HW-independent floor; chunking/H & residency/K levers)."),
    "realtime_requirement_surface": ("A/B", _DEPLOY, "3D feasibility frontier: required compute (z) over target rate (x) x VLA workload (y). HW-independent floor, computed from recovered structure."),
    "sharding_scalability": ("A", _NOSCALE, "Transfer effect of parallelism: extra comm bytes per unit of useful output as PU count grows, by shard axis. Splitting rows (M) broadcasts weights (priciest); cols (N) cheapest; K needs a partial-sum reduction."),
    "sharding_comm_tradeoff": ("A", _CAPTURED, "Absolute extra communication (GB) added per shard count, by axis — the cost side of sharding (M broadcasts weights; K needs a partial-sum reduction). HW-independent bytes."),
    "lever_ablation": ("A/B", _DEPLOY, "Ablation: action-chunking (/H) then residency (/~K) each cut the weight bandwidth needed for 30Hz. A requirement reduction, not a speedup."),
    "capture_level_ablation": ("A", _NOSCALE, "Progressive capture-level ablation: flat (nothing named) -> high_level (attention/softmax/norm named) -> quant_qdq (low-bit dequant) across the corpus."),
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
    is_3d = any(getattr(a, "name", "") == "3d" for a in fig.axes)
    if is_3d:
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10)  # tight_layout breaks on 3d
    else:
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
    """If you build ONE primitive: the fraction of MACs it covers (<=10% pad waste), mean across the
    corpus vs the WORST single workload. Plain-language replacement for the 'regret' framing — the
    mean-vs-worst gap IS the cross-workload regret of committing to that one primitive."""
    from collections import defaultdict
    rows = _rows(cs / "primitive_coverage_matrix.csv")
    if not rows:
        return False
    cov = defaultdict(list)
    for r in rows:
        cov[r["primitive"]].append(float(r["coverage_under_10pct"]))
    prims = sorted(cov, key=lambda p: -sum(cov[p]) / len(cov[p]))
    mean = [sum(cov[p]) / len(cov[p]) for p in prims]
    worst = [min(cov[p]) for p in prims]
    x = range(len(prims))
    ax.bar([i - 0.2 for i in x], mean, 0.4, label="mean across workloads")
    ax.bar([i + 0.2 for i in x], worst, 0.4, label="worst single workload")
    ax.set_xticks(list(x), prims, rotation=35, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of MACs covered (<=10% pad waste)")
    ax.set_title("If you build ONE primitive: MAC coverage, mean vs worst workload")
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
    import matplotlib.lines as mlines
    # every workload gets a colour: solid = reload-every-step (grows x K), dashed = resident (flat),
    # the dot marks the model's ACTUAL K recovered from the scf.for trip count.
    rows = sorted(rows, key=lambda r: -int(r["weight_bytes"]))
    for i, r in enumerate(rows):
        wb, kr = int(r["weight_bytes"]), int(r["invocations"])
        col = PALETTE[i % len(PALETTE)]
        ax.plot(ks, [wb * k for k in ks], color=col, lw=1.6, zorder=3)
        ax.plot(ks, [wb] * len(ks), "--", color=col, alpha=0.55, zorder=2)
        ax.scatter([kr], [wb * kr], color=col, zorder=5, s=30, edgecolor="white", lw=0.6)
        if i < 4:                                          # label the heaviest few inline
            ax.annotate(f"{r['workload']} (K={kr})", (kr, wb * kr), fontsize=8,
                        xytext=(4, 4), textcoords="offset points", color=col)
    ax.set_title("Decision: weight residency -> bytes moved vs loop count K (dot = the model's actual K)")
    ax.set_xlabel("head loop count K")
    ax.set_ylabel("weight bytes moved (log10)")
    ax.set_yscale("log")
    ax.legend(handles=[mlines.Line2D([], [], color="#5c5446", label="reload every step (grows x K)"),
                       mlines.Line2D([], [], color="#5c5446", ls="--", label="resident (load once, flat)"),
                       mlines.Line2D([], [], color="#5c5446", marker="o", ls="", label="model's actual K (IR scf.for)")],
              fontsize=8, loc="upper left")
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
    # right-margin summary: # workloads for which the abstraction is necessary-or-useful (rank >= 3)
    strong = [sum(1 for w in wls if _NEC_RANK[r[w]] >= 3) for r in rows]
    iax = ax.inset_axes([1.03, 0.0, 0.16, 1.0])
    iax.barh(range(len(rows)), strong, color=PALETTE[2], height=0.7)
    iax.set_ylim(ax.get_ylim())
    iax.set_yticks([])
    iax.set_xlim(0, len(wls))
    iax.tick_params(labelsize=8)
    iax.set_xlabel(f"# need N/U\n(of {len(wls)})", fontsize=8)
    for i, v in enumerate(strong):
        iax.text(v + 0.1, i, str(v), va="center", fontsize=8, color="#5c5446")
    ax.set_title("Abstraction necessity (N=necessary U=useful P=possible B=blocked –=N/A) — "
                 "build top rows first")
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
    rows = [r for r in _rows(cs / "real_config_magnitudes.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: -float(r["total_gemm_params"]))
    wl = [r["workload"] for r in rows]
    params = [float(r["total_gemm_params"]) for r in rows]
    macs = [float(r["gemm_macs_per_token"] or 0) for r in rows]
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
    """HW-INDEPENDENT roofline DIAGRAM: AI on x (log), normalized attainable throughput on y; the ridge
    is a PARAMETRIC band over plausible machine balances (no chip). Each workload sits at its AI; an arrow
    shows residency moving it rightward (reload-every-step -> resident) toward compute-bound."""
    import math
    rows = [r for r in _rows(cs / "arithmetic_intensity.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: float(r["ai_resident_mac_per_byte"]))
    B0, Blo, Bhi = 2.0, 1.0, 4.0                       # illustrative ridge + plausible band (MAC/byte)
    xs = [0.3 * (10 ** (i * math.log10(8 / 0.3) / 200)) for i in range(201)]
    roof = lambda ai, B: min(1.0, ai / B)             # noqa: E731  normalized attainable (peak=1)
    ax.fill_between(xs, [roof(a, Bhi) for a in xs], [roof(a, Blo) for a in xs],
                    color=PALETTE[0], alpha=0.13, label=f"ridge band (machine balance {Blo:.0f}-{Bhi:.0f})")
    ax.plot(xs, [roof(a, B0) for a in xs], color="#5c5446", lw=1.6,
            label=f"roofline @ illustrative balance B={B0:.0f} MAC/byte")
    import matplotlib.lines as mlines
    nonres = float(rows[0]["ai_nonresident_mac_per_byte"])     # 0.5 at bf16, shared by all
    ax.scatter([nonres], [roof(nonres, B0)], color="#b9ad97", zorder=4, s=44, edgecolor="#5c5446")
    ax.annotate("all workloads if\nreloaded every step\n(AI=1/dtype)", (nonres, roof(nonres, B0)),
                fontsize=8, xytext=(6, -32), textcoords="offset points", color="#5c5446")
    wl_handles = []
    for i, r in enumerate(rows):
        res, g = float(r["ai_resident_mac_per_byte"]), float(r["residency_gain"])
        y, col = roof(res, B0), PALETTE[i % len(PALETTE)]
        ax.annotate("", xy=(res, y), xytext=(nonres, roof(nonres, B0)),
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.65, lw=1.2))
        ax.scatter([res], [y], color=col, zorder=5, s=48)
        wl_handles.append(mlines.Line2D([], [], color=col, marker="o", ls="",
                                        label=f"{r['workload']} ({g:.1f}x)"))
    ax.set_xscale("log")
    ax.set_xlim(0.3, 8)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("arithmetic intensity (MAC / byte) — a WORKLOAD property, no HW assumed")
    ax.set_ylabel("attainable throughput / peak (normalized; HW-independent)")
    ax.set_title("HW-independent roofline: residency moves each workload toward compute-bound")
    # two legends: the roofline/band, and the workloads (with residency gain) — keeps labels off the curve
    band_leg = ax.legend(fontsize=8, loc="upper left")
    ax.add_artist(band_leg)
    ax.legend(handles=wl_handles, fontsize=8, loc="lower right", ncol=2, title="workload (residency gain)",
              title_fontsize=8)
    return True


# ---- capture fidelity (the thesis figure) + slide tables ------------------------------------------
_FID_ORDER = ["strong", "recovered", "measured", "assumed", "erased", "not_claimed", "na"]
_FID_COLOR = {"strong": "#5e8db4", "recovered": "#8fa674", "measured": "#7c9aa6",
              "assumed": "#d2a23f", "erased": "#cf8a82", "not_claimed": "#c9bfa8", "na": "#efe7d6"}
_FID_GLYPH = {"strong": "S", "recovered": "R", "measured": "M", "assumed": "A",
              "erased": "x", "not_claimed": "—", "na": ""}


def _fid_state(s: str) -> str:
    s = str(s).lower()
    for k in ("strong", "recovered", "measured", "assumed", "erased", "not_claimed"):
        if s.startswith(k):
            return k
    return "na"


def _r_capture_fidelity(cs, facts, ax):
    """THE thesis figure: per feature x workload, what the loop-preserving capture recovers from IR
    (S=structural, R=recovered, M=measured-host) vs erased (x) vs intentionally not-claimed (-)."""
    from merlin.dse_guidance import insight_mining as IM
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    cf = IM.capture_fidelity(cs)
    wls, matrix = cf.get("workloads", []), cf.get("matrix", [])
    if not matrix:
        return False
    feats = [row["feature"] for row in matrix]
    states = [[_fid_state(row.get(w, "")) for w in wls] for row in matrix]
    cidx = {s: i for i, s in enumerate(_FID_ORDER)}
    cmap = ListedColormap([_FID_COLOR[s] for s in _FID_ORDER])
    grid = [[cidx[s] for s in row] for row in states]
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=len(_FID_ORDER) - 1)
    ax.set_xticks(range(len(wls)), wls, rotation=30, fontsize=8)
    ax.set_yticks(range(len(feats)), [f.replace("_", " ") for f in feats], fontsize=8)
    for i, row in enumerate(states):
        for j, s in enumerate(row):
            if _FID_GLYPH[s]:
                ax.text(j, i, _FID_GLYPH[s], ha="center", va="center", fontsize=8,
                        color="white" if s in ("strong", "erased", "measured") else "#2f2a23")
    # right-margin summary: # workloads recovered (S/R/M) per feature
    rec = [sum(1 for s in row if s in ("strong", "recovered", "measured")) for row in states]
    iax = ax.inset_axes([1.03, 0.0, 0.15, 1.0])
    iax.barh(range(len(feats)), rec, color=PALETTE[2], height=0.7)
    iax.set_ylim(ax.get_ylim())
    iax.set_yticks([])
    iax.set_xlim(0, len(wls))
    iax.tick_params(labelsize=8)
    iax.set_xlabel(f"# recov\n(of {len(wls)})", fontsize=8)
    for i, v in enumerate(rec):
        iax.text(v + 0.1, i, str(v), va="center", fontsize=8, color="#5c5446")
    handles = [mpatches.Patch(color=_FID_COLOR[s], label=lbl) for s, lbl in
               [("strong", "structural (S)"), ("recovered", "recovered-from-IR (R)"),
                ("measured", "measured-host (M)"), ("erased", "erased (x)"),
                ("not_claimed", "not-claimed (—)"), ("na", "n/a")]]
    ax.legend(handles=handles, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Capture fidelity: loop-preserving capture recovers the structural contract from IR")
    return True


def _h(n: float) -> str:
    n = float(n)
    for d, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= d:
            return f"{n / d:.2f}{suf}"
    return f"{n:.0f}"


def _render_table(ax, columns, rows, title):
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#b9ad97")
        if r == 0:
            cell.set_facecolor(PALETTE[0])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#fbf8f1" if r % 2 else "#efe7d6")
    ax.set_title(title, pad=14)
    return True


def _r_table_capture_summary(cs, facts, ax):
    rows = [r for r in _rows(cs / "loop_aware_contract.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: r["workload"])
    body = [[r["workload"], r["K_ir"], r["repeated_region_ops"], r["n_loop_carried"],
             (r["kv_cache_bytes_ir"] if r["kv_cache_bytes_ir"] not in ("", "n/a")
              else "n/a (prefix-KV)")] for r in rows]
    return _render_table(ax, ["workload", "K (IR)", "repeated\nregion ops", "loop-carried\noperands",
                              "KV cache (IR)"], body,
                         "Recovered loop contract (all Tier-A, from scf.for)")


def _r_table_low_bit_tiers(cs, facts, ax):
    rows = _rows(cs / "low_bit_visibility.csv")
    if not rows:
        return False
    order = {"native": 0, "qdq_int8": 1, "dequant_only": 2}
    rows = sorted(rows, key=lambda r: (order.get(r["tier"], 9), r["workload"]))
    body = [[r["workload"], r["tier"], r["storage"], r["scale"],
             r["accuracy_status"].split(" (")[0]] for r in rows]
    return _render_table(ax, ["workload", "tier", "storage", "scale", "int8 accuracy"], body,
                         "Low-bit visibility (int8 ratified by measured gate; fp8/int4 never assumed)")


def _r_table_deployment_magnitudes(cs, facts, ax):
    rows = [r for r in _rows(cs / "real_config_magnitudes.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: -float(r["total_gemm_params"]))
    body = [[r["workload"], r["total_layers"], _h(r["total_gemm_params"]),
             (_h(r["gemm_macs_per_token"]) if r["gemm_macs_per_token"] else "n/a"),
             ("anchor" if r["workload"] in ("openvla", "tiny_llama") else "composed")]
            for r in rows]
    return _render_table(ax, ["workload", "layers", "GEMM params", "MACs/token", "source"], body,
                         "Deployment magnitudes by config-composition (exact for layer-identical stacks)")


def _r_table_arithmetic_intensity(cs, facts, ax):
    rows = [r for r in _rows(cs / "arithmetic_intensity.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows = sorted(rows, key=lambda r: -float(r["residency_gain"]))
    body = [[r["workload"], r["K"], f"{float(r['ai_resident_mac_per_byte']):.2f}",
             f"{float(r['ai_nonresident_mac_per_byte']):.2f}",
             f"{float(r['residency_gain']):.1f}x"] for r in rows]
    return _render_table(ax, ["workload", "K", "AI resident\n(MAC/byte)", "AI reload\n(MAC/byte)",
                              "residency\ngain"], body,
                         "Arithmetic intensity (HW-independent; residency gain = (prefix+rep*K)/(prefix+rep))")


def _r_table_realtime_requirement(cs, facts, ax):
    rows = _rows(cs / "realtime_requirement.csv")
    if not rows:
        return False
    body = [[r["workload"], r["regime"], r["budget_ms"], r["required_GMAC_per_s"],
             r["required_weight_GBps_resident"], r["required_weight_GBps_reload"]] for r in rows]
    return _render_table(ax, ["workload", "real-time regime", "budget\n(ms)", "req\nGMAC/s",
                              "weight GB/s\nresident", "weight GB/s\nreload"], body,
                         "Real-time requirements (HW-independent floor; regime=design target, not a chip)")


def _r_realtime_requirement(cs, facts, ax):
    """Per VLA workload @ the 30Hz real-time baseline: required weight bandwidth resident vs reload
    (residency lever) with the required compute rate annotated. A REQUIREMENT, not a chip's performance."""
    rows = [r for r in _rows(cs / "realtime_requirement.csv")
            if r["regime"].startswith("VLA 30Hz")]
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["required_weight_GBps_reload"]))
    wl = [r["workload"] for r in rows]
    res = [float(r["required_weight_GBps_resident"]) for r in rows]
    rel = [float(r["required_weight_GBps_reload"]) for r in rows]
    gm = [float(r["required_GMAC_per_s"]) for r in rows]
    x = range(len(wl))
    ax.bar([i - 0.2 for i in x], rel, 0.4, label="reload every step")
    ax.bar([i + 0.2 for i in x], res, 0.4, label="weights resident")
    for i, g in zip(x, gm):
        ax.annotate(f"{g:.0f}\nGMAC/s", (i, max(res[i], rel[i])), fontsize=8, ha="center",
                    va="bottom", color="#5c5446")
    ax.set_yscale("log")
    ax.set_xticks(list(x), wl, rotation=20, fontsize=8)
    ax.set_ylabel("required weight bandwidth (GB/s, log10)")
    ax.set_title("Requirement to hit 30Hz real-time: weight bandwidth (residency removes a ~Kx factor)")
    ax.legend(fontsize=8)
    return True


# ---- ablation studies + honest 3D surfaces (P25) -------------------------------------------------
def _ax3d(ax):
    """Swap the 2D ax the render loop made for a 3D ax on the same figure (so _stamp still works)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)
    fig = ax.figure
    ax.remove()
    return fig.add_subplot(111, projection="3d")


def _r_realtime_requirement_surface(cs, facts, ax):
    """3D (honest): required compute (z, log) over target rate (x, 10-100Hz) x VLA workload (y). The
    real-time feasibility frontier — a HW-independent requirement, computed from recovered structure."""
    import math
    import numpy as np
    from merlin.dse_guidance import models as M
    vfam = ("flow_matching", "diffusion", "autoregressive_vla")
    vla = [(r["workload"], float(r["macs_per_replan"]), (M.MODEL_ARCH[r["workload"]].action_horizon or 1))
           for r in _rows(cs / "arithmetic_intensity.csv")
           if r["workload"] != "small_llama" and M.MODEL_ARCH.get(r["workload"])
           and M.MODEL_ARCH[r["workload"]].family in vfam]
    if not vla:
        return False
    vla.sort(key=lambda t: t[1] / t[2])                  # by per-action work
    ax3 = _ax3d(ax)
    rates = list(range(10, 101, 5))
    X, Y = np.meshgrid(rates, range(len(vla)))
    Z = np.array([[math.log10(macs * r / H / 1e9) for r in rates] for (_, macs, H) in vla])
    surf = ax3.plot_surface(X, Y, Z, cmap=_pastel_cmap(), edgecolor="#5c5446", lw=0.25, alpha=0.92)
    ax3.set_yticks(range(len(vla)))
    ax3.set_yticklabels([w for w, _, _ in vla], fontsize=8)
    ax3.set_xlabel("target rate (Hz)", fontsize=9, labelpad=6)
    ax3.set_zlabel("required GMAC/s (log10)", fontsize=9, labelpad=4)
    ax3.set_title("Required compute vs real-time rate (HW-independent floor)")
    ax3.figure.colorbar(surf, ax=ax3, fraction=0.025, pad=0.10, label="log10 GMAC/s")
    ax3.view_init(elev=22, azim=-62)
    return True


def _r_sharding_scalability(cs, facts, ax):
    """The transfer effect of parallelism: extra communication bytes added per UNIT of useful output,
    as PU count grows, by shard axis. The whole corpus is shardable on every axis, so the real cost is
    here: splitting rows (M) / cols (N) is ~free; splitting the contraction (K) needs a partial-sum
    reduction every step -> the ratio climbs. HW-independent (structural bytes), not a perf claim."""
    sh = _rows(cs / "sharding_table.csv")
    ops = {(r["workload"], r["op_index"]): r for r in _rows(cs / "operator_shape_table.csv")}
    if not sh or not ops:
        return False
    counts = [2, 4, 8]
    out_tot = sum(int(o.get("output_bytes", 0) or 0) for o in ops.values()) or 1
    ratio = {a: [] for a in ("M", "N", "K")}
    for a in ("M", "N", "K"):
        for n in counts:
            comm = sum(float(r["per_extra_shard_bytes"]) * (n - 1) for r in sh
                       if r["axis"] == a and r.get(f"shardable_{n}") == "True")
            ratio[a].append(comm / out_tot)
    labels = {"M": "M (split output rows; broadcast weights)",
              "N": "N (split output cols; partition weights)",
              "K": "K (split contraction; partial-sum reduction)"}
    for i, key in enumerate(("M", "N", "K")):
        ax.plot(counts, ratio[key], marker="o", lw=1.8, color=PALETTE[i], label=labels[key])
    ax.set_xticks(counts)
    ax.set_xlabel("number of processing units (shard count)")
    ax.set_ylabel("extra comm bytes / useful output bytes")
    ax.set_title("Transfer effect of parallelism: comm overhead per unit work, by shard axis")
    ax.legend(fontsize=8, title="shard axis (how the GEMM is split)", title_fontsize=8)
    return True


def _r_sharding_comm_tradeoff(cs, facts, ax):
    """The cost side of sharding: extra communication bytes (partial-sum reductions / broadcasts) added
    per shard count, by axis. Read with sharding_scalability: M/N split cheaply (no reduction), K-axis
    needs a partial-sum reduction -> higher comm. HW-independent (structural bytes), not a perf claim."""
    from collections import defaultdict
    sh = _rows(cs / "sharding_table.csv")
    if not sh:
        return False
    counts = [2, 4, 8]
    comm = {a: [0.0, 0.0, 0.0] for a in ("M", "N", "K")}
    for r in sh:
        a = r["axis"]
        if a not in comm:
            continue
        per = float(r["per_extra_shard_bytes"])
        for j, n in enumerate(counts):
            if r.get(f"shardable_{n}") == "True":
                comm[a][j] += per * (n - 1)
    for i, a in enumerate(("M", "N", "K")):
        ax.plot(counts, [c / 1e9 for c in comm[a]], marker="o", lw=1.8, color=PALETTE[i],
                label=f"{a}-axis")
    ax.set_xticks(counts)
    ax.set_yscale("log")
    ax.set_xlabel("number of processing units (shard count)")
    ax.set_ylabel("extra communication (GB, log10)")
    ax.set_title("Sharding cost: extra comm bytes vs PU count, by axis (K needs partial-sum reduction)")
    ax.legend(fontsize=8, title="shard axis", title_fontsize=8)
    return True


def _r_lever_ablation(cs, facts, ax):
    """Ablation: how action-chunking (/H) then residency (/~K) each cut the weight bandwidth a machine
    must provide to hit 30Hz, for EVERY VLA workload. A REQUIREMENT reduction, not a speedup. Each line
    descends across the two levers; the spread shows chunk size (H) dominates the starting point."""
    from merlin.dse_guidance import models as M
    vfam = ("flow_matching", "diffusion", "autoregressive_vla")
    rows = [r for r in _rows(cs / "arithmetic_intensity.csv")
            if r["workload"] != "small_llama" and M.MODEL_ARCH.get(r["workload"])
            and M.MODEL_ARCH[r["workload"]].family in vfam]
    if not rows:
        return False
    stages = ["reload,\nno chunk", "+ action\nchunk (/H)", "+ residency\n(/~K)"]
    rows.sort(key=lambda r: -float(r["weight_bytes_nonresident"]))
    for k, r in enumerate(rows):
        w = r["workload"]
        H = M.MODEL_ARCH[w].action_horizon or 1
        wb_non, wb_res = float(r["weight_bytes_nonresident"]), float(r["weight_bytes_resident"])
        vals = [wb_non / (1 / 30) / 1e9, wb_non / (H / 30) / 1e9, wb_res / (H / 30) / 1e9]
        col = PALETTE[k % len(PALETTE)]
        ax.plot(range(3), vals, marker="o", color=col, lw=1.8, label=f"{w} (H={H})")
        ax.annotate(f"{vals[-1]:,.0f}", (2, vals[-1]), fontsize=8, color=col,
                    xytext=(5, 0), textcoords="offset points", va="center")
    ax.set_yscale("log")
    ax.set_xticks(range(3), stages, fontsize=8)
    ax.set_xlim(-0.2, 2.6)
    ax.set_ylabel("required weight bandwidth @30Hz (GB/s, log10)")
    ax.set_title("Lever ablation: chunking + residency cut the 30Hz bandwidth requirement (all VLAs)")
    ax.legend(fontsize=8, ncol=2)
    return True


def _r_capture_level_ablation(cs, facts, ax):
    """Progressive capture-level ablation: what each capture LEVEL unlocks across the corpus —
    flat (nothing named) -> high_level (attention/softmax/norm named) -> quant_qdq (low-bit metadata)."""
    from collections import defaultdict
    from merlin.dse_guidance import insight_mining as IM
    rows = IM.capture_level_ablation(cs)["rows"]
    if not rows:
        return False
    feats = [("linalg_ext_softmax", "softmax/attention"), ("linalg_ext_layer_norm", "normalization"),
             ("quant_ext_dequantize", "low-bit dequant")]
    levels = ["flat", "high_level", "quant_qdq"]
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for key, _ in feats:
            agg[r["level"]][key] += int(r.get(key, 0) or 0)
    levels = [lv for lv in levels if lv in agg]
    if not levels:
        return False
    x = range(len(levels))
    width = 0.26
    for k, (key, lbl) in enumerate(feats):
        vals = [agg[lv][key] for lv in levels]
        ax.bar([i + (k - 1) * width for i in x], vals, width, label=lbl)
    ax.set_xticks(list(x), [lv.replace("_", "\n") for lv in levels], fontsize=8)
    ax.set_ylabel("named structures recovered (corpus total)")
    ax.set_title("Capture-level ablation: what each capture level unlocks (named ops across the corpus)")
    ax.legend(fontsize=8)
    return True


_RENDERERS = {
    "capture_fidelity": _r_capture_fidelity,
    "realtime_requirement": _r_realtime_requirement,
    "table_realtime_requirement": _r_table_realtime_requirement,
    "realtime_requirement_surface": _r_realtime_requirement_surface,
    "sharding_scalability": _r_sharding_scalability,
    "sharding_comm_tradeoff": _r_sharding_comm_tradeoff,
    "lever_ablation": _r_lever_ablation,
    "capture_level_ablation": _r_capture_level_ablation,
    "table_capture_summary": _r_table_capture_summary,
    "table_low_bit_tiers": _r_table_low_bit_tiers,
    "table_deployment_magnitudes": _r_table_deployment_magnitudes,
    "table_arithmetic_intensity": _r_table_arithmetic_intensity,
    "deployment_magnitude": _r_deployment_magnitude,
    "arithmetic_intensity_roofline": _r_arithmetic_intensity_roofline,
    "evidence_type_by_workload": _r_evidence_by_workload,
    "evidence_type_by_phase": _r_evidence_by_phase,
    "shape_class_mac_share": _r_shape_mac_share,
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
    extra = [{"plot_id": pid} for pid in (
        "capture_fidelity", "deployment_magnitude", "arithmetic_intensity_roofline",
        "realtime_requirement", "table_capture_summary", "table_low_bit_tiers",
        "table_deployment_magnitudes", "table_arithmetic_intensity",
        "table_realtime_requirement", "realtime_requirement_surface",
        "sharding_scalability", "sharding_comm_tradeoff", "lever_ablation",
        "capture_level_ablation") if pid not in manifest_ids]
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
