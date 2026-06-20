"""Final presentation-pass renderers (P26): clean, conference-ready restyle of the curated plot set.

Separate from ``presentation_plots.py`` (which stays the full study artifact set). This module reuses the
SAME data accessors (``_rows`` + the ``insight_mining`` accessors + ``models.MODEL_ARCH``) but draws in a
single clean visual identity and writes to ``case_study/final_presentation_pass/figures/`` plus a
``figure_manifest.csv``. Figures carry NO evidence-tier badge, NO scale chip, NO mid-axes text; a short
italic caveat subtitle appears only on high-risk plots. Tier/scale/caveat live in the manifest.

Discipline (unchanged): structural facts / requirement envelopes only — no speedup/cycles/area/energy/
throughput/optimality/measured-performance. Timing & roofline are requirement / sensitivity views.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from merlin.dse_guidance.presentation_plots import _rows  # reuse the CSV reader

# ---- palette + ink (user-specified) --------------------------------------------------------------
_BG = "#FDF7EF"           # cream background (fig + axes)
_INK = "#2E2D2C"          # ash-black: text, spines, shadow
_GOLD = "#AB9A89"         # emphasis (bold)
_BLUE = "#333351"         # emphasis
_PALETTE = ["#333351", "#0F3759", "#8B93A6", "#815E5E", "#7D886C", "#AB9A89"]
_HATCHES = ["", "////", "....", "xx", "\\\\", "||"]
_GRID = "#D8CFC0"


def _h(n) -> str:
    n = float(n)
    for d, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= d:
            return f"{n / d:.2f}{suf}"
    return f"{n:.0f}"


def _have(name):
    import matplotlib.font_manager as fm
    return name in {f.name for f in fm.fontManager.ttflist}


def _fonts():
    """(serif title family chain, sans body family chain) using what's installed."""
    serif = [f for f in ("DM Serif Display", "Noto Serif Display", "PT Serif", "Georgia", "Tinos") if _have(f)]
    sans = [f for f in ("Inter", "Inter Display", "Liberation Sans", "DejaVu Sans") if _have(f)]
    return (serif or ["serif"]), (sans or ["sans-serif"])


def _final_style():
    import matplotlib.pyplot as plt
    serif, sans = _fonts()
    plt.rcParams.update({
        "figure.figsize": (9.0, 5.2), "figure.dpi": 200, "savefig.dpi": 200,
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "font.family": sans, "font.size": 12,
        "axes.titlesize": 16, "axes.labelsize": 12.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
        "text.color": _INK, "axes.labelcolor": _INK, "axes.edgecolor": _INK,
        "xtick.color": _INK, "ytick.color": _INK, "axes.titlecolor": _INK,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": _GRID, "grid.alpha": 0.6,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0, "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": _GRID, "legend.facecolor": "#FBF6EC",
    })
    plt.rcParams["font.serif"] = serif


def _title(ax, title, subtitle=None):
    """Serif title; optional short italic caveat subtitle beneath it (high-risk plots only)."""
    serif, _ = _fonts()
    ax.set_title(title, fontfamily=serif, fontsize=16, color=_INK, pad=(20 if subtitle else 12),
                 loc="left", fontweight="bold")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", ha="left", va="bottom", fontsize=10.5,
                    fontstyle="italic", color=_GOLD)


_SHADOW = "#B7AD9B"   # solid taupe-grey extrusion block (matches the reference 3D-bar look)


def _shadow():
    import matplotlib.patheffects as pe
    return [pe.withSimplePatchShadow(offset=(2.0, -2.0), shadow_rgbFace=_INK, alpha=0.22)]


def _extrude(ax, bars, dx=4.5, dy=-4.5):
    """Give bars a 3D-block look: a SOLID offset rectangle behind each bar (corner meets corner),
    like the reference figures — not a soft/blurry drop shadow. Offset is in points (scale/log safe)."""
    import matplotlib.transforms as mt
    from matplotlib.patches import Rectangle
    off = mt.offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")
    z = (bars[0].get_zorder() if len(bars) else 3) - 0.2
    for b in bars:
        bb = b.get_bbox()
        ax.add_patch(Rectangle((bb.x0, bb.y0), bb.width, bb.height, transform=off, facecolor=_SHADOW,
                               edgecolor=_INK, linewidth=0.8, joinstyle="miter", zorder=z))


def _nice_limits(values, pad=0.10, force_zero=False):
    """Padded (lo, hi). For tight non-zero ranges we don't force 0 (bars not needlessly long)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if force_zero or lo <= 0 or lo / hi > 0.55:   # near-zero or wide range -> keep 0 baseline (honest)
        return (0, hi * (1 + pad))
    span = hi - lo
    return (lo - span * pad * 2, hi + span * pad)


def _bars(ax, x, heights, width, color, hatch="", label=None, shadow=True, zorder=3):
    bars = ax.bar(x, heights, width, color=color, edgecolor=_INK, linewidth=1.1, hatch=hatch,
                  label=label, zorder=zorder)
    if shadow:
        _extrude(ax, bars)
    return bars


def _value_labels(ax, xs, heights, fmt="{:.0f}", dy=3, color=None):
    for xv, h in zip(xs, heights):
        ax.annotate(fmt.format(h), (xv, h), xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=color or _INK)


def _callout(ax, xy, text, color, xytext=(14, 14)):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="offset points", fontsize=10.5, color=_INK,
                bbox=dict(boxstyle="round,pad=0.35", fc="#FBF6EC", ec=color, lw=1.4),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0))


def _booktable(ax, cols, rows, title, shade=None, subtitle=None):
    """Booktabs-style table: only top/header/bottom rules, no vertical lines, optional heat shading."""
    ax.axis("off")
    n = len(rows)
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", bbox=[0, 0, 1, 0.86])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("none")
        cell.get_text().set_color(_INK)
        if r == 0:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(_BLUE)
            cell.visible_edges = "B"
            cell.set_linewidth(1.4)
            cell.set_edgecolor(_INK)
        elif r == n:
            cell.visible_edges = "B"
            cell.set_linewidth(1.2)
            cell.set_edgecolor(_INK)
        if r == 0 and c == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(1.6)
            cell.set_edgecolor(_INK)
        if shade and r > 0:
            sv = shade(r - 1, c, rows[r - 1][c])
            if sv:
                cell.set_facecolor(sv)
    # top rule across header
    ax.axhline(0.86, color=_INK, lw=1.6, xmin=0.0, xmax=1.0)
    _title(ax, title, subtitle)
    return True


def _save_clean(fig, out: Path):
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor=_BG)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _legend(ax, **kw):
    leg = ax.legend(**kw)
    if leg:
        leg.get_frame().set_linewidth(0.8)
    return leg


# ==================================================================================================
# Curated clean renderers (reuse the existing data logic; redraw clean). Each returns True if drawn.
# ==================================================================================================

def _r_realtime_requirement(cs, ax):
    from merlin.dse_guidance import models as M
    rows = [r for r in _rows(cs / "realtime_requirement.csv") if r["regime"].startswith("VLA 30Hz")]
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["required_weight_GBps_reload"]))
    wl = [r["workload"] for r in rows]
    rel = [float(r["required_weight_GBps_reload"]) for r in rows]
    res = [float(r["required_weight_GBps_resident"]) for r in rows]
    x = list(range(len(wl)))
    w = 0.38
    _bars(ax, [i - w / 2 for i in x], rel, w, _PALETTE[3], _HATCHES[1], "reload every step")
    _bars(ax, [i + w / 2 for i in x], res, w, _PALETTE[4], _HATCHES[2], "weights resident")
    ax.set_yscale("log")
    ax.set_xticks(x, wl, fontsize=11)
    ax.set_ylabel("required weight bandwidth (GB/s, log)")
    ax.set_xlabel("VLA workload")
    _title(ax, "30 Hz requirement envelope: residency lowers required weight bandwidth",
           "requirement floor under the workload model — not a chip measurement")
    _legend(ax, loc="upper right")
    return True


def _r_lever_ablation(cs, ax):
    from merlin.dse_guidance import models as M
    vfam = ("flow_matching", "diffusion", "autoregressive_vla")
    rows = [r for r in _rows(cs / "arithmetic_intensity.csv")
            if r["workload"] != "small_llama" and M.MODEL_ARCH.get(r["workload"])
            and M.MODEL_ARCH[r["workload"]].family in vfam]
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["weight_bytes_nonresident"]))
    stages = ["reload,\nno chunk", "+ action\nchunk (/H)", "+ residency\n(/K)"]
    for k, r in enumerate(rows):
        w = r["workload"]
        H = M.MODEL_ARCH[w].action_horizon or 1
        wb_non, wb_res = float(r["weight_bytes_nonresident"]), float(r["weight_bytes_resident"])
        vals = [wb_non / (1 / 30) / 1e9, wb_non / (H / 30) / 1e9, wb_res / (H / 30) / 1e9]
        col = _PALETTE[k % len(_PALETTE)]
        ax.plot(range(3), vals, marker="o", color=col, lw=2.2, ms=7, label=f"{w} (H={H})")
        ax.annotate(f"{vals[-1]:,.0f}", (2, vals[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=col, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(3), stages)
    ax.set_xlim(-0.25, 2.7)
    ax.set_ylabel("required weight bandwidth @30 Hz (GB/s, log)")
    _title(ax, "System levers reduce the 30 Hz weight-bandwidth requirement",
           "requirement reduction (action horizon H, loop K from source/config) — not a speedup")
    _legend(ax, loc="upper right", ncol=2)
    return True


def _r_table_capture_summary(cs, ax):
    rows = [r for r in _rows(cs / "loop_aware_contract.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows.sort(key=lambda r: r["workload"])
    body = [[r["workload"], r["K_ir"], r["repeated_region_ops"], r["n_loop_carried"],
             (r["kv_cache_bytes_ir"] if r["kv_cache_bytes_ir"] not in ("", "n/a") else "n/a")]
            for r in rows]
    return _booktable(ax, ["workload", "K (IR)", "repeated ops", "loop-carried", "KV cache (B)"],
                      body, "Recovered loop / region / state contract (all from scf.for)")


def _r_operator_cumulative_mac(cs, ax):
    rows = _rows(cs / "operator_shape_table.csv")
    if not rows:
        return False
    wls = sorted({r["workload"] for r in rows})
    curves = {}
    for w in wls:
        vals = sorted((int(r["macs"]) for r in rows if r["workload"] == w), reverse=True)
        tot = sum(vals) or 1
        cum, acc = [], 0
        for v in vals:
            acc += v
            cum.append(acc / tot)
        curves[w] = cum
    # k90 = ops needed for 90% of MACs; large k90 => diffuse. Label the 3 most-diffuse individually;
    # bundle the rest (hot-op-dominated) into one pale band so their labels don't collide.
    def k90(cum):
        return next((i + 1 for i, c in enumerate(cum) if c >= 0.9), len(cum))
    diffuse = sorted(wls, key=lambda w: -k90(curves[w]))[:3]
    xmax = max(len(c) for c in curves.values())
    for w in wls:
        if w in diffuse:
            continue
        ax.plot(range(1, len(curves[w]) + 1), curves[w], lw=1.4, color=_PALETTE[2], alpha=0.45,
                zorder=2)
    hot = [w for w in wls if w not in diffuse]
    if hot:
        km = max(k90(curves[w]) for w in hot)
        ax.annotate(f"hot-op-dominated bundle\n({len(hot)} workloads: ≥90% in ≤{km} ops)",
                    (km, 0.93), xytext=(40, -36), textcoords="offset points", fontsize=10, color=_INK,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#FBF6EC", ec=_PALETTE[2], lw=1.2),
                    arrowprops=dict(arrowstyle="-", color=_PALETTE[2], lw=1.0))
    for i, w in enumerate(diffuse):
        cum = curves[w]
        col = _PALETTE[[3, 4, 1][i % 3]]
        ax.plot(range(1, len(cum) + 1), cum, lw=2.4, color=col, zorder=4)
        ax.annotate(f"{w} (diffuse)", (len(cum), cum[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=col, fontweight="bold")
    ax.axhline(0.9, ls="--", color=_GOLD, lw=1.4, zorder=1)
    ax.annotate("90% of MACs", (xmax, 0.9), xytext=(-8, 6), textcoords="offset points", color=_GOLD,
                fontsize=10, fontweight="bold", ha="right")
    ax.set_xlabel("top-k operators (ranked by MACs)")
    ax.set_ylabel("cumulative MAC share")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.5, xmax * 1.18)
    _title(ax, "Compute concentration: hot-op-dominated vs diffuse workloads")
    return True


def _r_visible_linear_fraction(cs, ax):
    rows = _rows(cs / "work_coverage_table.csv")
    if not rows:
        return False
    rows.sort(key=lambda r: float(r["visible_linear_fraction"]))
    wl = [r["workload"] for r in rows]
    frac = [float(r["visible_linear_fraction"]) for r in rows]
    y = list(range(len(wl)))
    bars = ax.barh(y, frac, color=_PALETTE[1], edgecolor=_INK, linewidth=1.1, hatch=_HATCHES[1])
    _extrude(ax, bars)
    for yi, f in zip(y, frac):
        ax.annotate(f"{f:.2f}", (f, yi), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=10, color=_INK)
    ax.set_yticks(y, wl)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("linear-GEMM share  =  linear / (linear + attention)")
    _title(ax, "Recovered MAC work: linear-GEMM vs attention geometry",
           "fraction of recovered linear+attention MACs — excludes erased / unmodeled work")
    return True


def _r_work_coverage_by_workload(cs, ax):
    rows = _rows(cs / "work_coverage_table.csv")
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["visible_linear_fraction"]))
    wl = [r["workload"] for r in rows]
    lin = [float(r["linear_gemm_macs"]) for r in rows]
    att = [float(r["attention_macs"]) for r in rows]
    x = list(range(len(wl)))
    w = 0.4
    _bars(ax, [i - w / 2 for i in x], lin, w, _PALETTE[0], _HATCHES[1], "linear GEMM")
    _bars(ax, [i + w / 2 for i in x], att, w, _PALETTE[4], _HATCHES[2], "attention (recovered)")
    ax.set_yscale("log")
    ax.set_xticks(x, wl, rotation=30)
    ax.set_ylabel("recovered MACs (log)")
    _title(ax, "Recovered work: linear-GEMM vs attention MAC mass",
           "from IR shapes (captured-config) — not deployment scale")
    _legend(ax, loc="upper right")
    return True


def _r_deployment_magnitude(cs, ax):
    rows = [r for r in _rows(cs / "real_config_magnitudes.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["total_gemm_params"]))
    wl = [r["workload"] for r in rows]
    params = [float(r["total_gemm_params"]) for r in rows]
    macs = [float(r["gemm_macs_per_token"] or 0) for r in rows]
    x = list(range(len(wl)))
    w = 0.4
    _bars(ax, [i - w / 2 for i in x], params, w, _PALETTE[1], _HATCHES[1], "GEMM params")
    _bars(ax, [i + w / 2 for i in x], macs, w, _PALETTE[5], _HATCHES[2], "GEMM MACs / token")
    ax.set_yscale("log")
    ax.set_xticks(x, wl, rotation=25)
    ax.set_ylabel("count (log)")
    _title(ax, "Deployment magnitudes by config-composition",
           "embed + per-layer × real n_layers (exact for layer-identical stacks)")
    _legend(ax, loc="upper right")
    return True


def _r_decision_weight_residency(cs, ax):
    rows = [r for r in _rows(cs / "data_movement_table.csv")
            if r["region"] == "repeated_head" and int(r["weight_bytes"]) > 0]
    if not rows:
        return False
    import matplotlib.lines as mlines
    kmax = max(max(int(r["invocations"]), 2) for r in rows)
    ks = list(range(1, kmax + 1))
    rows.sort(key=lambda r: -int(r["weight_bytes"]))
    for i, r in enumerate(rows):
        wb, kr = int(r["weight_bytes"]), int(r["invocations"])
        col = _PALETTE[i % len(_PALETTE)]
        ax.plot(ks, [wb * k for k in ks], color=col, lw=2.0, zorder=3)
        ax.plot(ks, [wb] * len(ks), "--", color=col, alpha=0.5, zorder=2)
        ax.scatter([kr], [wb * kr], color=col, zorder=5, s=42, edgecolor=_BG, lw=1.0)
        if i < 4:
            ax.annotate(f"{r['workload']} (K={kr})", (kr, wb * kr), xytext=(6, 5),
                        textcoords="offset points", fontsize=9.5, color=col)
    ax.set_yscale("log")
    ax.set_xlabel("head loop count K")
    ax.set_ylabel("weight bytes moved (log)")
    _title(ax, "Loop-visible residency: non-resident weight traffic grows with K",
           "bytes moved (not bandwidth); dot = model's K (IR scf.for); captured-config scale")
    _legend(ax, handles=[mlines.Line2D([], [], color=_INK, lw=2, label="reload every step (grows × K)"),
                         mlines.Line2D([], [], color=_INK, ls="--", label="resident (load once, flat)"),
                         mlines.Line2D([], [], color=_INK, marker="o", ls="", label="model's K (IR scf.for)")],
            loc="upper left")
    return True


def _r_decision_capacity_dtype(cs, ax):
    import math
    rows = _rows(cs / "dtype_capacity_table.csv")
    if not rows:
        return False
    cols = [("bf16_B", "bf16"), ("int8_B", "int8"), ("int4_B", "int4")]
    allv = [int(float(r[c])) for r in rows for c, _ in cols]
    lo, hi = min(allv), max(allv)
    budgets = [10 ** (math.log10(lo) + i * (math.log10(hi) - math.log10(lo)) / 60) for i in range(61)] \
        if hi > lo else [lo]
    for k, (col, name) in enumerate(cols):
        sizes = [int(float(r[col])) for r in rows]
        ax.step(budgets, [sum(1 for s in sizes if s <= b) for b in budgets], where="post",
                color=_PALETTE[k], lw=2.2, label=name)
    ax.set_xscale("log")
    ax.set_ylabel(f"# repeated-head regions whose weights fit (of {len(rows)})")
    ax.set_xlabel("on-chip capacity budget")
    # human-readable byte ticks
    import matplotlib.ticker as mt
    def _fmt(v, _):
        for d, s in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
            if v >= d:
                return f"{v / d:.0f}{s}"
        return f"{v:.0f}B"
    ax.xaxis.set_major_formatter(mt.FuncFormatter(_fmt))
    _title(ax, "Capacity × dtype: when repeated-head weights fit on-chip",
           "captured-config weight sizes; dtype sets the residency budget")
    _legend(ax, loc="lower right", title="weight dtype", title_fontsize=10)
    return True


_NEC_RANK = {"necessary": 4, "useful": 3, "possible": 2, "blocked": 1, "not_applicable": 0}
_NEC_ABBR = {"necessary": "N", "useful": "U", "possible": "P", "blocked": "B", "not_applicable": "–"}


def _nec_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("nec", ["#EFE6D6", "#C9C2B4", "#8B93A6", "#3D4A63", "#0F3759"])


def _r_boundary_necessity_matrix(cs, ax, full=False):
    from merlin.dse_guidance import insight_mining as IM
    nec = IM.abstraction_necessity(cs)
    wls = nec["workloads"]
    rows = nec["rows"] if full else nec["rows"][:10]
    if not rows:
        return False
    mat = [[_NEC_RANK[r[w]] for w in wls] for r in rows]
    ax.grid(False)                                          # no grid over heatmap cells
    ax.imshow(mat, aspect="auto", cmap=_nec_cmap(), vmin=0, vmax=4)
    ax.set_xticks(range(len(wls)), wls, rotation=30, fontsize=10)
    ax.set_yticks(range(len(rows)), [r["abstraction"].replace("_", " ") for r in rows], fontsize=10)
    for i, r in enumerate(rows):
        for j, w in enumerate(wls):
            ax.text(j, i, _NEC_ABBR[r[w]], ha="center", va="center", fontsize=9.5,
                    color=_BG if mat[i][j] >= 3 else _INK)
    strong = [sum(1 for w in wls if _NEC_RANK[r[w]] >= 3) for r in rows]
    iax = ax.inset_axes([1.04, 0.0, 0.15, 1.0])
    iax.grid(False)
    ib = iax.barh(range(len(rows)), strong, color=_PALETTE[4], height=0.66, edgecolor=_INK, lw=0.8)
    iax.set_ylim(ax.get_ylim())
    iax.set_yticks([])
    iax.set_xlim(0, len(wls))
    iax.set_xlabel(f"# need\n(of {len(wls)})", fontsize=9.5)
    iax.tick_params(labelsize=9)
    for i, v in enumerate(strong):
        iax.text(v + 0.15, i, str(v), va="center", fontsize=9, color=_INK)
    _title(ax, "Abstraction necessity across workloads — search the necessary axes first",
           "N necessary · U useful · P possible · B blocked (capture/evidence) · – n/a")
    return True


def _r_boundary_necessity_full_backup(cs, ax):
    return _r_boundary_necessity_matrix(cs, ax, full=True)


def _r_primitive_set_frontier(cs, ax):
    from merlin.dse_guidance import insight_mining as IM
    fr = IM.primitive_set_frontier(cs)
    singles = fr.get("singles", [])
    if not singles:
        return False
    ax.scatter([s["macro"] for s in singles], [s["worst"] for s in singles], c=_PALETTE[2],
               edgecolor=_INK, lw=0.8, s=70, zorder=3, label="single primitive")
    markers = {1: "o", 2: "*", 3: "P"}
    cols = {1: _PALETTE[3], 2: _PALETTE[5], 3: _PALETTE[1]}
    for size, b in fr.get("best_by_size", {}).items():
        ax.scatter([b["macro"]], [b["worst"]], marker=markers.get(size, "s"), s=320, zorder=5,
                   color=cols.get(size, _PALETTE[0]), edgecolor=_INK, lw=1.2,
                   label=f"best {size}-primitive set")
    best = fr.get("best_by_size", {})
    if 1 in best and 2 in best:
        _callout(ax, (best[1]["macro"], best[1]["worst"]),
                 f"1 primitive: worst {best[1]['worst']:.0%}", _PALETTE[3], xytext=(12, -28))
        _callout(ax, (best[2]["macro"], best[2]["worst"]),
                 f"2 primitives: worst {best[2]['worst']:.0%}", _PALETTE[5], xytext=(-150, 20))
    ax.plot([0, 1], [0, 1], "--", color=_GRID, zorder=1)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("mean coverage across workloads")
    ax.set_ylabel("worst-workload coverage")
    _title(ax, "Primitive-set frontier: one primitive is not robust across workloads",
           "structural pad-waste MAC coverage — does not rank hardware performance")
    _legend(ax, loc="lower right")
    return True


_FID_ORDER = ["strong", "recovered", "measured", "assumed", "erased", "not_claimed", "na"]
_FID_COLOR = {"strong": "#0F3759", "recovered": "#7D886C", "measured": "#8B93A6",
              "assumed": "#AB9A89", "erased": "#815E5E", "not_claimed": "#C9BFA8", "na": "#EFE6D6"}
_FID_GLYPH = {"strong": "S", "recovered": "R", "measured": "M", "assumed": "A",
              "erased": "✕", "not_claimed": "—", "na": ""}


def _fid_state(s):
    s = str(s).lower()
    for k in ("strong", "recovered", "measured", "assumed", "erased", "not_claimed"):
        if s.startswith(k):
            return k
    return "na"


def _r_capture_fidelity(cs, ax):
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
    ax.grid(False)                                          # no grid over heatmap cells
    ax.imshow([[cidx[s] for s in row] for row in states], aspect="auto", cmap=cmap,
              vmin=0, vmax=len(_FID_ORDER) - 1)
    ax.set_xticks(range(len(wls)), wls, rotation=30, fontsize=10)
    ax.set_yticks(range(len(feats)), [f.replace("_", " ") for f in feats], fontsize=10)
    for i, row in enumerate(states):
        for j, s in enumerate(row):
            if _FID_GLYPH[s]:
                ax.text(j, i, _FID_GLYPH[s], ha="center", va="center", fontsize=9.5,
                        color=_BG if s in ("strong", "erased", "measured") else _INK)
    handles = [mpatches.Patch(color=_FID_COLOR[s], label=lbl) for s, lbl in
               [("strong", "structural (S)"), ("recovered", "recovered-from-IR (R)"),
                ("measured", "measured-host (M)"), ("erased", "erased (✕)"),
                ("not_claimed", "not-claimed (—)"), ("na", "n/a")]]
    ax.legend(handles=handles, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              frameon=False)
    _title(ax, "Capture fidelity: what the loop-preserving capture recovers vs erases")
    return True


def _r_capture_level_ablation(cs, ax):
    from collections import defaultdict
    from merlin.dse_guidance import insight_mining as IM
    rows = IM.capture_level_ablation(cs)["rows"]
    if not rows:
        return False
    feats = [("linalg_ext_softmax", "softmax / attention"), ("linalg_ext_layer_norm", "normalization"),
             ("quant_ext_dequantize", "low-bit dequant")]
    order = ["flat", "high_level", "quant_qdq"]
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for key, _ in feats:
            agg[r["level"]][key] += int(r.get(key, 0) or 0)
    levels = [lv for lv in order if lv in agg]
    x = list(range(len(levels)))
    w = 0.26
    for k, (key, lbl) in enumerate(feats):
        vals = [agg[lv][key] for lv in levels]
        _bars(ax, [i + (k - 1) * w for i in x], vals, w, _PALETTE[k], _HATCHES[k + 1], lbl)
    ax.set_xticks(x, [lv.replace("_", "\n") for lv in levels])
    ax.set_ylabel("named structures recovered (corpus total)")
    _title(ax, "Capture-level ablation: what each capture level unlocks")
    _legend(ax, loc="upper left")
    return True


def _r_arithmetic_intensity_roofline(cs, ax):
    import math
    import matplotlib.lines as mlines
    rows = [r for r in _rows(cs / "arithmetic_intensity.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows.sort(key=lambda r: float(r["ai_resident_mac_per_byte"]))
    B0, Blo, Bhi = 2.0, 1.0, 4.0
    xs = [0.3 * (10 ** (i * math.log10(8 / 0.3) / 200)) for i in range(201)]
    roof = lambda ai, B: min(1.0, ai / B)  # noqa: E731
    ax.fill_between(xs, [roof(a, Bhi) for a in xs], [roof(a, Blo) for a in xs],
                    color=_PALETTE[2], alpha=0.18, label="hypothetical machine-balance band")
    ax.plot(xs, [roof(a, B0) for a in xs], color=_INK, lw=1.8,
            label=f"roofline @ hypothetical balance B={B0:.0f}")
    nonres = float(rows[0]["ai_nonresident_mac_per_byte"])
    ax.scatter([nonres], [roof(nonres, B0)], color=_PALETTE[5], zorder=4, s=60, edgecolor=_INK)
    ax.annotate("reload every step\n(AI = 1/dtype)", (nonres, roof(nonres, B0)), xytext=(8, -34),
                textcoords="offset points", fontsize=9.5, color=_INK)
    wl_handles = []
    for i, r in enumerate(rows):
        res, g = float(r["ai_resident_mac_per_byte"]), float(r["residency_gain"])
        y, col = roof(res, B0), _PALETTE[i % len(_PALETTE)]
        ax.annotate("", xy=(res, y), xytext=(nonres, roof(nonres, B0)),
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.7, lw=1.4))
        ax.scatter([res], [y], color=col, zorder=5, s=60, edgecolor=_INK, lw=0.6)
        wl_handles.append(mlines.Line2D([], [], color=col, marker="o", ls="", label=f"{r['workload']} ({g:.1f}×)"))
    ax.set_xscale("log")
    ax.set_xlim(0.3, 8)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("weight-stream arithmetic intensity (MAC / repeated-head weight byte)")
    ax.set_ylabel("normalized roofline bound (hypothetical balance)")
    _title(ax, "Residency shifts weight-stream arithmetic intensity across machine-balance regimes",
           "requirement / modeling view — not measured performance, not full-memory AI")
    band_leg = ax.legend(loc="upper left", fontsize=10)
    ax.add_artist(band_leg)
    ax.legend(handles=wl_handles, loc="lower right", ncol=2, fontsize=9.5,
              title="workload (residency gain)", title_fontsize=9.5)
    return True


def _r_sharding_scalability(cs, ax):
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
    lbl = {"M": "M (split rows; broadcast weights)", "N": "N (split cols; partition weights)",
           "K": "K (split contraction; partial-sum reduction)"}
    for i, key in enumerate(("M", "N", "K")):
        ax.plot(counts, ratio[key], marker="o", lw=2.2, ms=7, color=_PALETTE[i], label=lbl[key])
    ax.set_xticks(counts)
    ax.set_xlabel("number of processing units (shard count)")
    ax.set_ylabel("extra comm bytes / useful output bytes")
    _title(ax, "Transfer effect of parallelism: comm overhead per unit work, by shard axis",
           "structural communication bytes — not a performance result")
    _legend(ax, loc="upper left", title="how the GEMM is split", title_fontsize=9.5)
    return True


def _bt_low_bit(cs, ax):
    rows = _rows(cs / "low_bit_visibility.csv")
    if not rows:
        return False
    order = {"native": 0, "qdq_int8": 1, "dequant_only": 2}
    rows.sort(key=lambda r: (order.get(r["tier"], 9), r["workload"]))
    body = [[r["workload"], r["tier"], r["storage"], r["scale"], r["accuracy_status"].split(" (")[0]]
            for r in rows]
    return _booktable(ax, ["workload", "tier", "storage", "scale", "int8 accuracy"], body,
                      "Low-bit visibility per workload",
                      subtitle="int8 ratified by the measured gate; fp8/int4 never assumed")


def _bt_deploy(cs, ax):
    rows = [r for r in _rows(cs / "real_config_magnitudes.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["total_gemm_params"]))
    body = [[r["workload"], r["total_layers"], _h(r["total_gemm_params"]),
             (_h(r["gemm_macs_per_token"]) if r["gemm_macs_per_token"] else "n/a"),
             ("anchor" if r["workload"] in ("openvla", "tiny_llama") else "composed")] for r in rows]
    return _booktable(ax, ["workload", "layers", "GEMM params", "MACs/token", "source"], body,
                      "Deployment magnitudes by config-composition")


def _bt_arith(cs, ax):
    rows = [r for r in _rows(cs / "arithmetic_intensity.csv") if r["workload"] != "small_llama"]
    if not rows:
        return False
    rows.sort(key=lambda r: -float(r["residency_gain"]))
    body = [[r["workload"], r["K"], f"{float(r['ai_resident_mac_per_byte']):.2f}",
             f"{float(r['ai_nonresident_mac_per_byte']):.2f}", f"{float(r['residency_gain']):.1f}×"]
            for r in rows]
    return _booktable(ax, ["workload", "K", "AI resident", "AI reload", "residency gain"], body,
                      "Weight-stream arithmetic intensity",
                      subtitle="MAC per repeated-head weight byte — modeling view, not full-memory AI")


# plot_id -> (renderer, evidence_tier, scale, caveat, source_artifact, presentation_class)
_FINAL_META = {
    "table_capture_summary": (_r_table_capture_summary, "A", "structural", "", "loop_aware_contract.csv", "main"),
    "capture_fidelity": (_r_capture_fidelity, "A/B", "structural", "", "IM.capture_fidelity", "main"),
    "capture_level_ablation": (_r_capture_level_ablation, "A", "structural", "", "IM.capture_level_ablation", "main"),
    "primitive_set_frontier": (_r_primitive_set_frontier, "A", "structural", "structural pad-waste coverage; not hardware performance", "IM.primitive_set_frontier", "main"),
    "operator_cumulative_mac": (_r_operator_cumulative_mac, "A", "structural", "", "operator_shape_table.csv", "main"),
    "decision_weight_residency": (_r_decision_weight_residency, "A", "captured-config", "bytes moved, not bandwidth; captured-config scale", "data_movement_table.csv", "main"),
    "decision_capacity_dtype": (_r_decision_capacity_dtype, "A", "captured-config", "captured-config weight sizes", "dtype_capacity_table.csv", "main"),
    "realtime_requirement": (_r_realtime_requirement, "A/B", "deployment-composition", "requirement floor, not a chip measurement", "realtime_requirement.csv", "main"),
    "lever_ablation": (_r_lever_ablation, "A/B", "deployment-composition", "requirement reduction (H, K from source/config), not a speedup", "arithmetic_intensity.csv", "main"),
    "boundary_necessity_matrix": (_r_boundary_necessity_matrix, "B", "structural", "blocked = capture/evidence blocked", "IM.abstraction_necessity", "main"),
    "arithmetic_intensity_roofline": (_r_arithmetic_intensity_roofline, "A/B", "deployment-composition", "requirement/modeling view; not measured performance; not full-memory AI; hypothetical machine balance", "arithmetic_intensity.csv", "main"),
    "visible_linear_fraction": (_r_visible_linear_fraction, "A", "structural", "excludes erased/unmodeled work", "work_coverage_table.csv", "main"),
    "work_coverage_by_workload": (_r_work_coverage_by_workload, "A", "captured-config", "captured-config; not deployment scale", "work_coverage_table.csv", "backup"),
    "deployment_magnitude": (_r_deployment_magnitude, "B", "deployment-composition", "", "real_config_magnitudes.csv", "backup"),
    "sharding_scalability": (_r_sharding_scalability, "A", "structural", "structural comm bytes; not a performance result", "sharding_table.csv", "backup"),
    "boundary_necessity_full_backup": (_r_boundary_necessity_full_backup, "B", "structural", "blocked = capture/evidence blocked", "IM.abstraction_necessity", "backup"),
    "table_low_bit_tiers": (_bt_low_bit, "A/B", "structural", "fp8/int4 never assumed", "low_bit_visibility.csv", "backup"),
    "table_deployment_magnitudes": (_bt_deploy, "B", "deployment-composition", "", "real_config_magnitudes.csv", "backup"),
    "table_arithmetic_intensity": (_bt_arith, "A/B", "deployment-composition", "modeling view, not full-memory AI", "arithmetic_intensity.csv", "backup"),
}


_BND_LEVELS = [("compiler_transform", "compiler\ntransform"), ("runtime_hal_object", "runtime /\nHAL object"),
               ("command_buffer_or_command_isa", "command\nISA"), ("accelerator_isa", "accelerator\nISA"),
               ("device_microcode_or_controller", "microcode /\ncontroller"),
               ("fixed_hardware_datapath", "fixed\ndatapath")]
_BND_KEY = ["resident_weight_object", "skinny_gemm_or_gemv_engine", "partial_sum_object",
            "fused_requant_epilogue", "loop_carried_state_handle", "bounded_loop_command",
            "packed_lowbit_tensor", "scale_object", "dma_engine", "async_queue"]
_BND_MAP = {"strong_candidate": ("strong", "#0F3759"), "possible": ("possible", "#8B93A6"),
            "weak_candidate": ("weak", "#AB9A89"), "blocked": ("blocked", "#815E5E"),
            "unavailable": ("unavail", "#C9BFA8"), "not_applicable": ("—", "#EFE6D6")}


def emit_boundary_simplified(cs_dir, fp_dir):
    """Phase 3B: a clean CATEGORICAL boundary-placement matrix (key abstractions x levels) — png+csv+md.
    A boundary search-space view (where each abstraction COULD live), not a score."""
    import csv as _csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    cs_dir, fp_dir = Path(cs_dir), Path(fp_dir)
    raw = {r["abstraction"]: r for r in _rows(cs_dir / "hw_sw_boundary_matrix.csv")}
    absts = [a for a in _BND_KEY if a in raw]
    if not absts:
        return False
    cats = list(_BND_MAP)
    cidx = {c: i for i, c in enumerate(cats)}
    cmap = ListedColormap([_BND_MAP[c][1] for c in cats])
    _final_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    grid, csv_rows = [], []
    for a in absts:
        row, crow = [], {"abstraction": a}
        for col, _ in _BND_LEVELS:
            st = raw[a].get(col, "not_applicable")
            row.append(cidx.get(st, cidx["not_applicable"]))
            crow[col] = _BND_MAP.get(st, ("—",))[0]
        grid.append(row)
        csv_rows.append(crow)
    ax.grid(False)                                          # no grid over heatmap cells
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=len(cats) - 1)
    ax.set_xticks(range(len(_BND_LEVELS)), [lbl for _, lbl in _BND_LEVELS], fontsize=10)
    ax.set_yticks(range(len(absts)), [a.replace("_", " ") for a in absts], fontsize=10)
    for i, a in enumerate(absts):
        for j, (col, _) in enumerate(_BND_LEVELS):
            lab, color = _BND_MAP.get(raw[a].get(col, "not_applicable"), ("—", "#EFE6D6"))
            ax.text(j, i, lab, ha="center", va="center", fontsize=8.5,
                    color=_BG if color in ("#0F3759", "#815E5E") else _INK)
    _title(ax, "HW/SW boundary search space: where each abstraction could live",
           "candidate placements (strong/possible/weak/blocked) — a search space, not a score")
    handles = [mpatches.Patch(color=c, label=_BND_MAP[k][0]) for k, (_, c) in
               [(k, _BND_MAP[k]) for k in cats]]
    ax.legend(handles=handles, fontsize=9.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False)
    _save_clean(fig, fp_dir / "boundary_placement_simplified.png")
    with open(fp_dir / "boundary_placement_simplified.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["abstraction"] + [c for c, _ in _BND_LEVELS])
        w.writeheader()
        w.writerows(csv_rows)
    md = ["# Boundary-placement (simplified, categorical)\n",
          "Where each key abstraction *could* be implemented across the HW/SW stack — a **search space**, "
          "not a score. Cells: strong / possible / weak / blocked / unavail / — (n/a).\n",
          "| abstraction | " + " | ".join(lbl.replace("\n", " ") for _, lbl in _BND_LEVELS) + " |",
          "|" + "---|" * (len(_BND_LEVELS) + 1)]
    for crow in csv_rows:
        md.append("| " + crow["abstraction"] + " | "
                  + " | ".join(crow[c] for c, _ in _BND_LEVELS) + " |")
    (fp_dir / "boundary_placement_simplified.md").write_text("\n".join(md) + "\n")
    return True


def render_final(cs_dir, out_dir) -> list[str]:
    import csv as _csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _final_style()
    cs_dir, out_dir = Path(cs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, manifest = [], []
    for pid, (fn, tier, scale, caveat, src, klass) in _FINAL_META.items():
        fig, ax = plt.subplots()
        try:
            ok = fn(cs_dir, ax)
        except Exception as e:  # noqa: BLE001
            print(f"  [{pid}] FAILED: {type(e).__name__}: {e}")
            ok = False
        if ok:
            _save_clean(fig, out_dir / f"{pid}.png")
            done.append(pid)
            title = ax.get_title(loc="left") or ax.get_title()
            manifest.append({"plot_id": pid, "class": klass, "evidence_tier": tier, "scale": scale,
                             "title": title, "caveat": caveat, "source_artifact": src,
                             "file": f"figures/{pid}.png"})
        else:
            plt.close(fig)
    mpath = out_dir.parent / "figure_manifest.csv"
    with open(mpath, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["plot_id", "class", "evidence_tier", "scale", "title",
                                           "caveat", "source_artifact", "file"])
        w.writeheader()
        w.writerows(manifest)
    return done
