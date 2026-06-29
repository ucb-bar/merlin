#!/usr/bin/env python3
"""Presentation figures for the autocomp-on-Muon results, in the Merlin house style.

Reads the portable result bundle (no new runs, no LLM spend) and renders PNG+SVG.
Style comes from scripts/merlin_plotstyle.py — imported, never re-derived.
"""
import matplotlib
matplotlib.use("Agg")
import sys, os, json, csv, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so merlin_plotstyle imports
from merlin_plotstyle import *                              # noqa: F401,F403
use_merlin_style()

import numpy as np
import matplotlib.pyplot as plt

BUNDLE = Path("/scratch/agustin/projects/oscar-merlin/tmp/kernels/radiance_only_kernels")
IDX = BUNDLE / "index"
OUT = Path("/scratch/agustin/projects/oscar-merlin/artifacts/presentation/radiance_autocomp")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- problem identity
PROB = {
    0:  dict(short="matmul 64³",        src="anchor",   sig=True),
    1:  dict(short="conv patch-embed",  src="openvla",  sig=True),
    2:  dict(short="attn seq64",        src="rdt",      sig=False),
    3:  dict(short="attn seq96",        src="rdt",      sig=False),
    6:  dict(short="matmul K128",       src="smolvla",  sig=False),
    7:  dict(short="flash-attn",        src="pi05",     sig=False),
    10: dict(short="matmul K384",       src="smolvla",  sig=False),
}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=180, facecolor=BG)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name)


# ======================================================================= load data
def load_cost_by_problem():
    """(prob,run) -> sorted list of (iter, cum_usd, cum_tok)."""
    raw = collections.defaultdict(list)
    for r in csv.DictReader(open(IDX / "cost_by_problem.csv")):
        raw[(int(r["prob"]), r["run"])].append(
            (int(r["iteration"]), float(r["usd"]),
             int(r["input_tokens"]) + int(r["output_tokens"])))
    series = {}
    for key, rows in raw.items():
        rows.sort()
        it = [x[0] for x in rows]
        cusd = np.cumsum([x[1] for x in rows])
        ctok = np.cumsum([x[2] for x in rows])
        series[key] = (it, cusd, ctok)
    return series


def load_results():
    res = {}
    for r in csv.DictReader(open(BUNDLE / "results/metrics/results.csv")):
        res[int(r["prob"])] = r
    return res


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


SPEND = json.load(open(IDX / "muon-spend_total.json"))
SERIES = load_cost_by_problem()
RES = load_results()


def runtag(run):
    t = run.replace("built:muon_muon_", "").replace("_cyclotron", "")
    t = "_".join(t.split("_")[1:])          # drop leading prob id
    return t


def prob_of(run):
    return int(run.replace("built:muon_muon_", "").split("_")[0])


# ============================================================ fig1: spend per kernel
# distinct line identity (colour + dash + marker) so one legend disambiguates all runs
LINE_STYLE = {
    (0,  "iters5"):           dict(c=NAVY,  ls="-",  m="o", lab="P0 matmul 64³",        sig=True),
    (1,  "iters12_bw6harvest"): dict(c=BLUE, ls="-",  m="s", lab="P1 conv patch-embed (12 it)", sig=True),
    (1,  "iters5"):           dict(c=BLUE,  ls="--", m="s", lab="P1 conv patch-embed (4 it)",  sig=True),
    (10, "iters12_bw6harvest"): dict(c=MAUVE, ls="-", m="D", lab="P10 matmul K384",      sig=False),
    (7,  "iters5"):           dict(c=SAGE,  ls="-",  m="v", lab="P7 flash-attn",         sig=False),
    (6,  "iters12_bw6harvest"): dict(c=MAUVE, ls="--", m="D", lab="P6 matmul K128",      sig=False),
    (2,  "iters5"):           dict(c=SLATE, ls="-",  m="^", lab="P2 attn seq64",         sig=False),
    (3,  "iters5"):           dict(c=SLATE, ls="--", m="^", lab="P3 attn seq96",         sig=False),
}


def _style_key(prob, run):
    tag = "iters12_bw6harvest" if "iters12" in run else "iters5"
    return (prob, tag)


def fig1_spend_per_kernel():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2))
    keys = sorted(SERIES, key=lambda k: (k[0], k[1]))
    handles = None
    for ax, col, lab in ((axL, 1, "cumulative cost (USD)"), (axR, 2, "cumulative tokens (millions)")):
        style_ax(ax)
        for (prob, run) in keys:
            it, cusd, ctok = SERIES[(prob, run)]
            y = cusd if col == 1 else ctok / 1e6
            st = LINE_STYLE[_style_key(prob, run)]
            sig = st["sig"]
            ax.plot(it, y, st["ls"], color=st["c"], lw=2.9 if sig else 1.7,
                    alpha=1.0 if sig else 0.7, zorder=5 if sig else 3,
                    marker=st["m"], ms=5 if sig else 3.2, mec=INK, mew=0.6,
                    label=st["lab"])
        ax.set_xlabel("beam iteration")
        ax.set_ylabel(lab)
        ax.set_xlim(0.5, max(max(SERIES[k][0]) for k in keys) + 1.2)
        title(ax, "cost per iteration" if col == 1 else "tokens per iteration")
        if handles is None:
            handles, labels_ = ax.get_legend_handles_labels()
    # one shared legend in the empty upper-left of the left panel
    axL.legend(loc="upper left", fontsize=9, ncol=1, handlelength=2.4,
               borderpad=0.7, labelspacing=0.5)
    # gold callout on the priciest endpoint (P1 12-iter)
    it, cusd, _ = SERIES[(1, "built:muon_muon_1_beam_iters12_bw6harvest_cyclotron")]
    emph(axL, it[-1], cusd[-1], "  $21.98 — priciest", color=GOLD, fs=10, va="center")
    suptitle(fig, "Cumulative search spend per kernel  (plan: Gemini 3.5 Flash · code: Qwen3-Coder 480B)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig1_spend_per_kernel")


# ================================ fig8: per-kernel cost/iter + grand total (combined)
def fig8_spend_per_kernel_with_total():
    """New combined view (keeps fig1 + fig7 intact): cost per iteration per kernel on the
    left, the campaign grand-total cumulative curve on the right. No tokens panel."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2),
                                   gridspec_kw=dict(width_ratios=[1.32, 1]))
    keys = sorted(SERIES, key=lambda k: (k[0], k[1]))

    # --- left: per-kernel cumulative cost per iteration
    style_ax(axL)
    for (prob, run) in keys:
        it, cusd, _ = SERIES[(prob, run)]
        st = LINE_STYLE[_style_key(prob, run)]
        sig = st["sig"]
        axL.plot(it, cusd, st["ls"], color=st["c"], lw=2.9 if sig else 1.7,
                 alpha=1.0 if sig else 0.7, zorder=5 if sig else 3,
                 marker=st["m"], ms=5 if sig else 3.2, mec=INK, mew=0.6, label=st["lab"])
    axL.set_xlabel("beam iteration")
    axL.set_ylabel("cumulative cost (USD)")
    axL.set_xlim(0.5, max(max(SERIES[k][0]) for k in keys) + 1.2)
    axL.legend(loc="upper left", fontsize=9, ncol=1, handlelength=2.4,
               borderpad=0.7, labelspacing=0.5)
    it, cusd, _ = SERIES[(1, "built:muon_muon_1_beam_iters12_bw6harvest_cyclotron")]
    emph(axL, it[-1], cusd[-1], "  $21.98 — priciest", color=GOLD, fs=10, va="center")
    title(axL, "cost per iteration, per kernel")

    # --- right: grand-total cumulative cost over the whole campaign
    days = sorted(SPEND["by_day"].items())
    labels = [d.replace("2026-", "") for d, _ in days]
    cum = np.cumsum([v for _, v in days])
    x = np.arange(len(days))
    style_ax(axR)
    axR.fill_between(x, 0, cum, color=NAVY, alpha=0.12, zorder=2)
    axR.plot(x, cum, "-", color=NAVY, lw=3, marker="o", ms=6.5, mec=INK, mew=0.8, zorder=4)
    for xi, c in zip(x[:-1], cum[:-1]):
        axR.annotate(f"${c:.0f}", (xi, c), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=9, color=INK)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=9.5)
    axR.set_xlim(-0.3, len(days) - 1 + 0.7)
    axR.set_ylim(0, cum[-1] * 1.18)
    axR.set_xlabel("active campaign day (2026)")
    axR.set_ylabel("cumulative cost (USD)")
    emph(axR, x[-1], cum[-1], f"  ${SPEND['total_usd']:.2f}\n  {SPEND['calls']:,} calls",
         color=GOLD, fs=12, va="center", ha="left")
    title(axR, "campaign total")

    suptitle(fig, "Cumulative search spend — per kernel and campaign total")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig8_spend_per_kernel_with_total")


# ============================================================ fig2: spend by model
def fig2_spend_by_model():
    name = {
        "us.anthropic.claude-sonnet-4-6": "Claude\nSonnet 4.6",
        "gemini-3.5-flash":               "Gemini\n3.5 Flash",
        "gemini-3.1-pro-preview":         "Gemini\n3.1 Pro",
        "qwen.qwen3-coder-480b-a35b-v1:0": "Qwen3-Coder\n480B",
        "gemini-3-flash-preview":         "Gemini\n3 Flash",
    }
    role = {
        "us.anthropic.claude-sonnet-4-6": "early exploration",
        "gemini-3.5-flash":               "planner",
        "gemini-3.1-pro-preview":         "planner (early)",
        "qwen.qwen3-coder-480b-a35b-v1:0": "code generator",
        "gemini-3-flash-preview":         "planner (early)",
    }
    items = [(k, v) for k, v in SPEND["by_model"].items() if v >= 1.0]
    tiny = [(k, v) for k, v in SPEND["by_model"].items() if 0 < v < 1.0]
    items.sort(key=lambda x: -x[1])
    labels = [name.get(k, k) for k, _ in items]
    usd = [v for _, v in items]
    colors = [MAUVE if k == "us.anthropic.claude-sonnet-4-6" else
              (NAVY if k == "gemini-3.5-flash" else
               (BLUE if k == "qwen.qwen3-coder-480b-a35b-v1:0" else SLATE))
              for k, _ in items]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    style_ax(ax)
    x = np.arange(len(items))
    for xi, h, c in zip(x, usd, colors):
        vbars(ax, [xi], [h], c, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("cumulative cost (USD)")
    ax.set_ylim(0, max(usd) * 1.18)
    for xi, h, (k, _) in zip(x, usd, items):
        ax.annotate(f"${h:.0f}", (xi, h), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.annotate(role[k], (xi, 0), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=BG, fontstyle="italic")
    emph(ax, len(items) - 0.5, max(usd) * 1.08,
         f"total ${SPEND['total_usd']:.0f} · {SPEND['calls']:,} calls", color=GOLD, fs=12,
         ha="right", va="center")
    title(ax, "Where the budget went, by model")
    if tiny:
        note = " · ".join(f"{name.get(k, k).replace(chr(10), ' ')} ${v:.2f}" for k, v in tiny)
        fig.text(0.5, 0.015, f"negligible: {note}", ha="center", fontsize=8.3,
                 color=INK, fontstyle="italic")
    suptitle(fig, "Cumulative LLM spend across the full Muon search campaign")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "fig2_spend_by_model")


# ===================================================== fig3: baseline -> final cycles
def fig3_baseline_vs_final():
    probs = [0, 1]
    names = [f"{PROB[p]['short']}\n({PROB[p]['src']})" for p in probs]
    net = [(f(RES[p]["baseline_net"]), f(RES[p]["best_net"]), f(RES[p]["speedup_net"])) for p in probs]
    tot = [(f(RES[p]["baseline_total"]), f(RES[p]["best_total"]), f(RES[p]["speedup_total"])) for p in probs]

    fig, (axN, axT) = plt.subplots(1, 2, figsize=(14, 6.3))
    w = 0.34
    for ax, data, lab in ((axN, net, "net (kernel-only) cycles"), (axT, tot, "total (end-to-end) cycles")):
        style_ax(ax)
        x = np.arange(len(probs))
        bvals = [d[0] for d in data]
        fvals = [d[1] for d in data]
        vbars(ax, x - w/2, bvals, MAUVE, width=w)
        vbars(ax, x + w/2, fvals, NAVY, width=w)
        lo = min(bvals + fvals)
        hi = max(bvals + fvals)
        ax.set_ylim(lo * 0.0, hi * 1.16)   # cycles: 0 origin is meaningful (a count)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
        ax.set_ylabel(lab)
        for xi, d in zip(x, data):
            for off, val in ((-w/2, d[0]), (w/2, d[1])):
                ax.annotate(f"{val/1000:.0f}k", (xi + off, val), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=9, color=INK)
            emph(ax, xi, max(d[0], d[1]) * 1.085, f"{d[2]:.2f}×", color=GOLD, fs=14, ha="center")
        title(ax, lab.split(" cycles")[0].strip() + " speedup")
    axN.annotate("matmul best = the golden SMEM\ntechnique (RTL-confirmed)",
                 (0 + w/2, net[0][1]), xytext=(0.46, net[0][0]*0.86),
                 fontsize=9, color=BLUE, fontweight="bold", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3,
                                 connectionstyle="arc3,rad=-0.2"))
    from matplotlib.patches import Patch
    axT.legend(handles=[Patch(fc=MAUVE, ec=INK, label="baseline (Claude Code)"),
                        Patch(fc=NAVY, ec=INK, label="final best (search)")],
               loc="upper right", fontsize=9.5)
    suptitle(fig, "Baseline to final: the two kernels with significant wins")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3_baseline_vs_final")


# ===================================================== fig3b: net speedup, all measured
def fig3b_speedup_all():
    probs = [0, 1, 6, 10, 2]   # those with a net speedup, sorted by magnitude
    sp = [(p, f(RES[p]["speedup_net"])) for p in probs]
    sp = [x for x in sp if x[1]]
    sp.sort(key=lambda x: -x[1])
    labels = [f"P{p} {PROB[p]['short']}" for p, _ in sp]
    vals = [v for _, v in sp]
    colors = [NAVY if PROB[p]["sig"] else MAUVE for p, _ in sp]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    style_ax(ax)
    x = np.arange(len(sp))
    for xi, h, c in zip(x, vals, colors):
        vbars(ax, [xi], [h], c, width=0.6, base=1.0)   # value axis from 1.0× (no change)
    ax.set_ylim(1.0, max(vals) * 1.12)
    ax.text(len(sp) - 1, max(vals) * 1.06, "bars measured from 1.0× (no change)",
            ha="right", va="center", fontsize=9, color=INK, fontstyle="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("kernel-only (net) speedup  ×")
    for xi, (p, v) in zip(x, sp):
        col = GOLD if PROB[p]["sig"] else INK
        emph(ax, xi, v, f"{v:.2f}×", color=col, fs=11.5, ha="center", va="bottom") \
            if PROB[p]["sig"] else ax.annotate(f"{v:.3f}×", (xi, v), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=9.5, color=INK)
    ax.annotate("P3 attn96, P4 swiglu, P9 gelu landed exactly 1.000× (unchanged)",
                (0, max(vals) * 1.06), fontsize=9, color=INK, ha="left", fontstyle="italic")
    title(ax, "Where the search actually moved the needle")
    suptitle(fig, "Net speedup across every kernel we measured")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig3b_speedup_all")


# ===================================================== fig4: candidate yield funnel
def fig4_yield_funnel():
    rows = [json.loads(l) for l in open(IDX / "all_transforms.jsonl")]
    total = len(rows)
    compcorr = sum(1 for r in rows if r.get("correct"))
    oc = collections.Counter(r.get("outcome") for r in rows if r.get("outcome"))
    improved = oc.get("improved", 0)
    labeled = sum(oc.values())

    fig, (axF, axB) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    # funnel as horizontal bars (counts -> 0 origin meaningful)
    style_ax(axF, grid="x")
    stages = [("generated", total, SLATE),
              ("compiled & correct", compcorr, NAVY),
              ("improved on parent*", improved, BLUE)]
    y = np.arange(len(stages))[::-1]
    for yi, (lab, val, c) in zip(y, stages):
        hbars(axF, [yi], [val], c, height=0.58)
        axF.annotate(f"{val}", (val, yi), xytext=(7, 0), textcoords="offset points",
                     va="center", fontsize=11, fontweight="bold", color=INK)
    axF.set_yticks(y); axF.set_yticklabels([s[0] for s in stages], fontsize=10.5)
    axF.set_xlim(0, total * 1.14)
    axF.set_xlabel("candidates")
    emph(axF, total * 0.5, y[1], f"{100*compcorr/total:.0f}% compile+correct", color=GOLD, fs=12,
         ha="center", va="center")
    title(axF, "Candidate yield")

    # failure / outcome breakdown of the ledger-labeled subset
    style_ax(axB, grid="x")
    order = [("compile_error", MAUVE), ("regressed", SLATE),
             ("correct_no_gain", SAGE), ("improved", NAVY)]
    yb = np.arange(len(order))[::-1]
    for yi, (k, c) in zip(yb, order):
        hbars(axB, [yi], [oc.get(k, 0)], c, height=0.58)
        axB.annotate(f"{oc.get(k,0)}", (oc.get(k, 0), yi), xytext=(7, 0),
                     textcoords="offset points", va="center", fontsize=10.5,
                     fontweight="bold", color=INK)
    axB.set_yticks(yb); axB.set_yticklabels([k for k, _ in order], fontsize=10.5)
    axB.set_xlim(0, max(oc.values()) * 1.32)
    axB.set_xlabel(f"candidates (labeled subset, n={labeled})")
    title(axB, "Outcome breakdown*")
    axB.annotate("compile errors dominated by\nregister oversubscription\n(>256 physical registers)",
                 (oc["compile_error"], yb[0]), xytext=(max(oc.values()) * 0.95, 1.5),
                 fontsize=9, color=BLUE, fontweight="bold", ha="right", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2,
                                 connectionstyle="arc3,rad=0.25"))
    fig.text(0.5, 0.02, "* outcome labels exist only for the transform-ledger subset (probs 1,2,3,7); "
             "generated/correct counts span all 835 attempts.", ha="center", fontsize=8.3,
             color=INK, fontstyle="italic")
    suptitle(fig, "One in nine generated kernels is valid — and few of those improve")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, "fig4_yield_funnel")


# ===================================================== fig5: cost vs gain (ROI)
def fig5_cost_vs_gain():
    # per-problem total $ and tokens (sum runs), net speedup
    by_prob_usd = collections.defaultdict(float)
    by_prob_tok = collections.defaultdict(int)
    for (prob, run), (it, cusd, ctok) in SERIES.items():
        by_prob_usd[prob] += cusd[-1]
        by_prob_tok[prob] += ctok[-1]
    pts = []
    for p in PROB:
        sp = f(RES[p]["speedup_net"])
        if sp is None:
            continue
        pts.append((p, by_prob_usd[p], sp, by_prob_tok[p]))

    # per-point label offsets (pt) to avoid overlaps in the bottom cluster
    LBL = {0: (10, 12), 1: (-10, 14), 2: (8, 12), 3: (8, -18), 6: (10, 10),
           10: (0, 14)}
    HA = {0: "left", 1: "right", 2: "left", 3: "left", 6: "left", 10: "center"}

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    style_ax(ax, grid="both")
    ax.axhline(1.0, color=INK, lw=1.1, ls=":", zorder=2)
    for p, usd, sp, tok in pts:
        sig = PROB[p]["sig"]
        c = NAVY if sig else MAUVE
        size = 130 + tok / 1e5
        ax.scatter([usd], [sp], s=size, color=c, edgecolor=INK, linewidth=1.3,
                   zorder=5, alpha=0.92)
        ax.annotate(f"P{p} {PROB[p]['short']}", (usd, sp), xytext=LBL[p],
                    textcoords="offset points", fontsize=9.5, ha=HA[p],
                    fontweight="bold" if sig else "normal",
                    color=NAVY if sig else INK)
    ax.set_xlabel("search cost on this kernel (USD)")
    ax.set_ylabel("kernel-only (net) speedup  ×")
    ax.set_ylim(0.97, max(p[2] for p in pts) * 1.13)
    ax.set_xlim(0, max(p[1] for p in pts) * 1.16)
    emph(ax, 4.6, 2.28, "best ROI:\ncheap + 2.4×", color=GOLD, fs=10.5, ha="left", va="top")
    ax.text(ax.get_xlim()[1] * 0.98, 1.18, "bubble area ~ tokens", ha="right",
            fontsize=8.7, color=INK, fontstyle="italic")
    title(ax, "Return on search spend")
    suptitle(fig, "Cost vs. gain — most $ bought ≈1.0×; the matmul win was cheap")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig5_cost_vs_gain")


# ===================================================== fig6: utilization / headroom
def fig6_utilization():
    rows = [(p, f(RES[p]["util_net_pct"])) for p in PROB if f(RES[p]["util_net_pct"]) is not None]
    rows.sort(key=lambda x: x[1])
    labels = [f"P{p} {PROB[p]['short']}" for p, _ in rows]
    vals = [v for _, v in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    style_ax(ax, grid="x")
    y = np.arange(len(rows))
    for yi, (p, v) in zip(y, rows):
        c = NAVY if PROB[p]["sig"] else SLATE
        hbars(ax, [yi], [v], c, height=0.55)
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("FP utilization  (essential FLOPs ÷ fp32 SIMT peak 32 flop/cyc, kernel-only)  %")
    # headroom band
    ax.axvspan(max(vals), 100, color=GOLD, alpha=0.10, zorder=0)
    emph(ax, (max(vals) + 100) / 2, (len(rows) - 1) / 2, "headroom to peak", color=GOLD, fs=12,
         ha="center", va="center")
    title(ax, "Best kernel reaches ~38% of fp32 peak; residual is latency, not throughput")
    suptitle(fig, "FP utilization — conservative, kernel-only (the win is in bigger/fused problems)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig6_utilization")


# ===================================================== fig7: total cumulative cost
def fig7_cumulative_total():
    days = sorted(SPEND["by_day"].items())
    labels = [d.replace("2026-", "") for d, _ in days]    # MM-DD
    cum = np.cumsum([v for _, v in days])
    x = np.arange(len(days))

    fig, ax = plt.subplots(figsize=(10.5, 6))
    style_ax(ax)
    ax.fill_between(x, 0, cum, color=NAVY, alpha=0.12, zorder=2)
    ax.plot(x, cum, "-", color=NAVY, lw=3, marker="o", ms=7, mec=INK, mew=0.8, zorder=4)
    for xi, c in zip(x[:-1], cum[:-1]):    # last point labelled by the gold total callout
        ax.annotate(f"${c:.0f}", (xi, c), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.3, len(days) - 1 + 0.6)
    ax.set_ylim(0, cum[-1] * 1.16)
    ax.set_xlabel("active campaign day (2026)")
    ax.set_ylabel("cumulative cost (USD)")
    emph(ax, x[-1], cum[-1], f"  ${SPEND['total_usd']:.2f}\n  {SPEND['calls']:,} calls",
         color=GOLD, fs=13, va="center", ha="left")
    ax.annotate("late hybrid runs (Gemini-plan +\nQwen-code) drove the final climb",
                (x[-2] + 0.5, (cum[-2] + cum[-1]) / 2),
                xytext=(x[-3] - 0.2, cum[-1] * 0.82), fontsize=9, color=BLUE,
                fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3,
                                connectionstyle="arc3,rad=0.2"))
    title(ax, "Total spend accumulated to $153")
    suptitle(fig, "Cumulative cost of the entire Muon search campaign")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig7_cumulative_total")


if __name__ == "__main__":
    # report font availability
    from matplotlib import font_manager as fm
    have = set(f.name for f in fm.fontManager.ttflist)
    print("fonts: Inter=%s  DM Serif Display=%s" % ("Inter" in have, "DM Serif Display" in have))
    fig1_spend_per_kernel()
    fig2_spend_by_model()
    fig3_baseline_vs_final()
    fig3b_speedup_all()
    fig4_yield_funnel()
    fig5_cost_vs_gain()
    fig6_utilization()
    fig7_cumulative_total()
    fig8_spend_per_kernel_with_total()
    print("done ->", OUT)
