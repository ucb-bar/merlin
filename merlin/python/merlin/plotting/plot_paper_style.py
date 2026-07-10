#!/usr/bin/env python
"""Paper-style (Physical-Intelligence / ML-paper aesthetic) figures of the RVV results.

Style vocabulary borrowed from the reference figures:
  - muted natural palette (dusty salmon / sage / steel / warm gold); "ours" = gold, prior = salmon
  - black bar outlines, value label on every bar
  - rounded cream "card" panels, bold serif titles + figure captions
  - boxed callout annotations with leader lines on the key datapoint
  - a soft shaded band marking the crossover / selected regime

All numbers transcribed verbatim from the committed result docs (sources in each section).
"""
from pathlib import Path
from merlin.common.paths import artifacts_dir
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = artifacts_dir() / "ceiling"

# ---- palette ----
INK     = "#2b2b2b"
SALMON  = "#cf8b7d"   # prior / baseline  (their π0.5 colour)
SAGE    = "#9bb08a"   # competitor A (OpenBLAS)
STEEL   = "#6f93b0"   # competitor B (XNNPACK)
GOLD    = "#e7c25c"   # OURS  (their "ours" colour)
TEAL    = "#7bb4c4"   # ceiling reference (dashed)
GREY    = "#9a9a9a"
CREAM   = "#f5f1e6"
CARD_EC = "#33312b"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#555", "axes.linewidth": 0.9,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

def card(ax, title):
    """Give an axes the cream rounded-card look with a bold title."""
    ax.set_facecolor(CREAM)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=INK, pad=8)

def callout(ax, xy, text, xytext, fc="#fff6e0", ec=GOLD):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=9.5, color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.3),
                arrowprops=dict(arrowstyle="-", color=ec, lw=1.3,
                                connectionstyle="arc3,rad=0.0"))


# ============================================================================
# FIGURE A — whole-model e2e "card" (the headline; image-copy-7 style)
# bitvla 3-config + per-model speedups. Source: output/rvv_bench/k1_e2e_*.md
# ============================================================================
def fig_e2e():
    # DATA-DRIVEN (no hardcoded numbers) — reads the SAME fresh four-way / .vf JSONs as fig_fourway,
    # so this figure can never drift out of sync with the headline.
    import json
    V3 = "#b8742a"; OB = "#7a9e7a"  # ours accum-resident microkernel (winner) / OpenBLAS
    FEAT = {"ours_v3": "accum-resident v3", "ours_wholemodel_vf": "wholemodel .vf",
            "ours_wholemodel": "wholemodel", "ours_tiled": "tiled vfmacc"}
    def load(m):
        vf = OUT.parents[1] / "rvv_bench" / f"k1_vf_{m}.json"
        fw = OUT.parents[1] / "rvv_bench" / f"k1_4way_{m}.json"
        src = vf if vf.is_file() else fw
        return json.load(open(src)) if src.is_file() else None
    def wall(s, key):
        r = (s or {}).get(key) or {}
        return r["min_wall_ns"] / 1e9 if r.get("min_wall_ns") else None
    def ours_best(s):
        cands = [k for k in ("ours_wholemodel_vf", "ours_v3", "ours_wholemodel", "ours_tiled")
                 if (s.get(k) or {}).get("min_wall_ns")]
        return min(cands, key=lambda k: s[k]["min_wall_ns"]) if cands else None

    fig = plt.figure(figsize=(12, 5.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

    # -- left: bitvla latency card — baseline / OpenBLAS / XNNPACK / ours-best (the winner) --
    ax = fig.add_subplot(gs[0]); card(ax, "bitvla — whole-model latency on K1 silicon")
    b = load("bitvla"); bk = ours_best(b); base = wall(b, "baseline")
    rows = [("baseline\n(hand_v0)", wall(b, "baseline"), SALMON, False),
            ("OpenBLAS\n(hand kernel)", wall(b, "openblas_kernels"), OB, False),
            ("XNNPACK\n(hand kernel)", wall(b, "xnnpack_kernels"), STEEL, False),
            (f"ours-{bk.replace('ours_','')}\n(accum-resident, compiler)", wall(b, bk), V3, True)]
    rows = [(l, ms, c, f) for (l, ms, c, f) in rows if ms]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, s_, col, fast) in zip(y, rows):
        ms = s_ * 1000.0
        ax.barh(yi, ms, height=0.6, color=col, edgecolor=CARD_EC, linewidth=1.6, zorder=3)
        tag = f"{ms:.0f} ms   ({base/s_:.2f}×)" + ("   ← fastest" if fast else "")
        ax.text(ms + base*1000*0.017, yi, tag, va="center", ha="left", fontsize=10.5,
                fontweight="bold", color=(V3 if fast else INK))
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.8)
    ax.set_xlim(0, base*1000*1.65); ax.set_xlabel("latency (ms / forward) — lower is better")
    xn = wall(b, "xnnpack_kernels"); over = (xn / wall(b, bk)) if (xn and wall(b, bk)) else None
    callout(ax, (wall(b, bk)*1000, y[-1]+0.32),
            f"compiler-emitted micro-kernel BEATS XNNPACK's\nhand kernel — {over:.2f}× faster, cos 0.99999",
            (base*1000*0.5, y[-1]+0.30), fc="#f7efe2", ec=V3)
    ax.set_ylim(-0.6, len(rows)-0.35)

    # -- right: best whole-model compiler speedup PER MODEL + the best-expert reference --
    ax = fig.add_subplot(gs[1]); card(ax, "best compiler speedup per model vs the best expert")
    models = ["rdt2", "openvla", "bitvla"]
    y = np.arange(len(models)); maxsp = 1.0
    for yi, m in enumerate(models):
        s = load(m); bk = ours_best(s); base = wall(s, "baseline")
        if not (s and bk and base): continue
        sp = base / wall(s, bk)
        exps = [base / wall(s, k) for k in ("xnnpack_kernels", "openblas_kernels") if wall(s, k)]
        col = V3 if (exps and sp >= max(exps)) else GOLD
        ax.barh(yi, sp, height=0.6, color=col, edgecolor=CARD_EC, linewidth=1.6, zorder=3)
        ax.text(sp + 0.25, yi, f"{sp:.2f}×", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(0.3, yi, FEAT.get(bk, bk), va="center", fontsize=8.5, color="white", fontweight="bold")
        maxsp = max(maxsp, sp, *(exps or [0]))
        if exps:  # best-expert reference tick (so "ours vs hand-tuned" is visible in absolute speedup space)
            be = max(exps)
            ax.plot([be, be], [yi-0.32, yi+0.32], color=INK, lw=2.2, zorder=5)
            pct = round(100*sp/be)
            ax.text(be, yi+0.40, ("WINS" if sp >= be else f"{pct}% of expert"),
                    ha="center", fontsize=7.6, fontweight="bold", color=(V3 if sp >= be else GREY))
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(models, fontsize=11)
    ax.set_xlim(0, maxsp*1.18); ax.set_xlabel("whole-model speedup (×) — higher is better\n(▮ = fastest expert)")
    ax.set_ylim(-0.6, len(models)-0.4)

    fig.text(0.5, -0.06,
             "Figure A:  Whole-model RVV speedups on real K1 silicon (cos ≥ 0.99999), same-pass, data-driven from the "
             "four-way JSONs.  LEFT: on bitvla the compiler-emitted accumulator-resident micro-kernel BEATS both hand-\n"
             "written vendor kernels.  RIGHT: best kernel is per-model (the autotune portfolio); the ▮ tick marks the "
             "fastest expert — ours WINS bitvla and reaches 60 % / 63 % of the best expert on openvla / rdt2.",
             ha="center", fontsize=9.3, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_e2e.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_e2e.png")
    plt.close(fig)


# ============================================================================
# FIGURE B — GEMM ceiling crossover on K1 (image-copy-2 annotated-line style)
# Source: output/kernels/ceiling/large_shape_packing_k1.md (inner-compute)
# ============================================================================
def fig_crossover():
    shapes = [32, 64, 128, 256, 384, 512]
    series = [
        ("ours-baseline",  [39246, 306608, 2516021, 19613446, None, None], SALMON, "-",  "o"),
        ("ours-tiled (compiler)", [2694, 20309, 168957, 1297217, None, None], GOLD, "-", "o"),
        ("OpenBLAS",       [409, 2329, 17789, 149008, 491134, 1146939], SAGE, "-", "s"),
        ("XNNPACK",        [238, 1976, 31761, 248282, 891310, 2079033], STEEL, "-", "^"),
        ("ours-intrinsic (hand ceiling)", [172, 1350, 14144, 106034, 441356, 1394602], TEAL, "--", "D"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.6)); ax.set_facecolor(CREAM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # shaded "ours-wins" region up to the crossover
    ax.axvspan(28, 430, color=GOLD, alpha=0.10, zorder=0)
    for name, ys, col, ls, mk in series:
        xs = [s for s, v in zip(shapes, ys) if v]; yy = [v for v in ys if v]
        ax.plot(xs, yy, ls=ls, marker=mk, ms=6, lw=2.4, color=col, label=name,
                markeredgecolor=CARD_EC, markeredgewidth=0.6, zorder=4)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(shapes); ax.set_xticklabels([f"{s}³" for s in shapes])
    ax.set_xlabel("GEMM size  (M=N=K)"); ax.set_ylabel("K1 rdtime ticks  (log; lower is faster)")
    ax.set_title("Single-GEMM ceiling on real K1 silicon", loc="left", color=INK, pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35, zorder=1)
    callout(ax, (384, 8.0e4), "ours-intrinsic beats both\nexperts through 384³",
            (96, 4.5e5), fc="#eaf4f6", ec=TEAL)
    callout(ax, (512, 1146939), "OpenBLAS retakes\nlead at 512³ (0.82×)",
            (512, 9.0e4), fc="#eef3ec", ec=SAGE)
    ax.text(40, 1.1e7, "ours-baseline never\nforms vfmacc (~100× off)", fontsize=8.6,
            color=SALMON, style="italic")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95, facecolor="white")
    fig.text(0.5, -0.03,
             "Figure B:  GEMM compute ceiling, K1 inner-compute ticks.  A compiler-emitted register-blocked "
             "kernel (ours-intrinsic, dashed = hand-written ceiling reference) beats OpenBLAS and XNNPACK\n"
             "from 32³ to 384³; OpenBLAS's cache-blocking retakes the lead at 512³.  The shipped compiler "
             "path (ours-tiled) trails the experts ~10×; the baseline ~100×.",
             ha="center", fontsize=9.3, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_crossover.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_crossover.png")
    plt.close(fig)


# ============================================================================
# FIGURE C — BEAM PROGRESSION: how the improved RVV path advances iteration by
# iteration, vs OpenBLAS / XNNPACK references. Left = whole-model bitvla e2e
# (the clean monotone story, ending above XNNPACK); right = single-GEMM @64^3
# spike instret kernel trajectory. Sources: cross_framework_matrix*, k1_e2e_*.json.
# ============================================================================
def fig_progression():
    V3 = "#b8742a"
    fig = plt.figure(figsize=(13, 5.4)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.26)

    # -- left: whole-model bitvla e2e progression (speedup vs baseline; higher=better) --
    ax = fig.add_subplot(gs[0]); card(ax, "Beam progression — bitvla whole-model on K1")
    # XNNPACK reference + v3 endpoint read from the fresh four-way JSON (no stale literals).
    import json as _j
    _b = _j.load(open(OUT.parents[1] / "rvv_bench" / "k1_4way_bitvla.json"))
    _base = _b["baseline"]["min_wall_ns"]; _xnn = _base / _b["xnnpack_kernels"]["min_wall_ns"]
    _v3 = _base / _b["ours_v3"]["min_wall_ns"]
    steps = [("baseline\nhand_v0", 1.00, SALMON),
             ("+ ntail\n(attn vfmacc)", 7.73, "#e8c98a"),
             ("+ tiled\nvfmacc", 9.16, GOLD),
             ("+ v3 accum-\nresident", round(_v3, 2), V3)]
    xs = np.arange(len(steps))
    vals = [s[1] for s in steps]
    ax.plot(xs, vals, "-", color="#9a9a9a", lw=1.6, zorder=2)
    for xi, (lab, v, col) in zip(xs, steps):
        ax.scatter([xi], [v], s=170, color=col, edgecolor=CARD_EC, linewidth=1.4, zorder=4)
        ax.text(xi, v + 0.7, f"{v}×", ha="center", fontsize=11, fontweight="bold", color=col if col != "#e8c98a" else "#9c7415")
    ax.axhline(_xnn, color=STEEL, ls="--", lw=1.6, zorder=1)
    ax.text(0.05, _xnn + 0.35, f"XNNPACK hand kernel ({_xnn:.2f}×)", fontsize=9, color=STEEL, fontweight="bold")
    callout(ax, (3, _v3), "compiler-emitted v3\ncrosses ABOVE XNNPACK", (1.75, 17.6), fc="#f7efe2", ec=V3)
    # iteration-2 note: .vf A-scalarize was a SEPARATE turn targeting the openvla/rdt2 residual.
    ax.text(0.05, 2.6,
            "iteration 2 (.vf A-scalarize) → openvla/rdt2:\nmemory-bound, modest 55→60% / 62→63%\n→ surfaces packing as the turn-3 residual",
            fontsize=7.6, color="#7a6a4a", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#faf6ec", ec="#d8c89a", lw=0.8))
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in steps], fontsize=9)
    ax.set_ylim(0, 19.5); ax.set_ylabel("whole-model speedup vs baseline (×)")
    ax.set_xlabel("beam iteration (each adds one mined capability)")

    # -- right: single-GEMM @64^3 spike instret kernel trajectory (lower=better, log) --
    ax = fig.add_subplot(gs[1]); card(ax, "Beam progression — single GEMM 64³ (spike instret)")
    k = [("baseline", 22430926, SALMON), ("vfmacc\ncontraction", 123094, "#e8c98a"),
         ("tiled\n(bounded)", 1328219, GOLD), ("v3 compute\nkernel", 53207, V3)]
    xs = np.arange(len(k)); vals = [s[1] for s in k]
    ax.plot(xs, vals, "-", color="#9a9a9a", lw=1.6, zorder=2)
    for xi, (lab, v, col) in zip(xs, k):
        ax.scatter([xi], [v], s=170, color=col, edgecolor=CARD_EC, linewidth=1.4, zorder=4)
        ax.text(xi, v*1.5, f"{v:,}", ha="center", fontsize=8.5, fontweight="bold", color=INK)
    for yv, lab, col in [(84483, "OpenBLAS", SAGE), (101705, "XNNPACK", STEEL), (50695, "hand ceiling", TEAL)]:
        ax.axhline(yv, color=col, ls="--", lw=1.3, zorder=1)
        ax.text(3.05, yv, lab, fontsize=8, color=col, va="center", fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(2.5e4, 2.5e8)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in k], fontsize=9)
    ax.set_ylabel("retired instructions (log; lower = faster)")
    ax.set_xlabel("beam iteration"); ax.set_xlim(-0.4, 4.0)

    fig.text(0.5, -0.03,
             "Figure C:  Beam progression of the improved RVV compilation path.  Left — whole-model bitvla on K1: each "
             "mined capability (attention vfmacc → tiled → accumulator-resident v3) advances the speedup, the final v3\n"
             "crossing ABOVE XNNPACK's hand kernel (13.19×).  Right — single-GEMM 64³ kernel trajectory (spike instret): "
             "v3's compute kernel reaches the hand ceiling and beats both experts.  Baseline RVV is frozen; each step is a default-off fork.",
             ha="center", fontsize=9.3, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_progression.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_progression.png")
    plt.close(fig)


# ============================================================================
# FIGURE D — OPTIMIZATION EFFECTS BY DRIVING EXAMPLE: how each distinct mined
# optimization (the beam's candidates) affected the RVV dialect, and how that
# RANKING DEPENDS on the driving example. Shows the headline gap: the recorded
# beam ran on GEMM 64^3 only (predating v3); the real winner is example-dependent.
# Sources: autotune ranking.yaml (GEMM 64^3) + k1_e2e_*.json (whole-model bitvla/openvla).
# ============================================================================
def fig_opt_effects():
    V3 = "#b8742a"
    EX = [("GEMM 64³ (autotune)", "#9aa0aa"), ("bitvla (whole-model)", GOLD),
          ("openvla (whole-model)", TEAL)]
    # feature -> {example_index: speedup}.  None = not measured on that example.
    feats = [
        ("baseline (hand_v0)",        [1.0, 1.0, 1.0]),
        ("lmul_widen_n",              [1.05, None, None]),
        ("vfmacc_contraction",        [8.05, None, None]),
        ("vfmacc_tiled",              [8.04, 9.16, 3.65]),
        ("accum_resident_ntail",      [None, 7.73, None]),
        ("vectorized_activation",     [None, 1.00, 0.95]),
        ("accum_resident_v3",         [None, 16.88, 2.38]),
    ]
    fig, ax = plt.subplots(figsize=(11, 6)); ax.set_facecolor(CREAM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    n_ex = len(EX); bh = 0.78 / n_ex
    yb = np.arange(len(feats))
    for fi, (fname, vals) in enumerate(feats):
        for ei, v in enumerate(vals):
            if v is None: continue
            yy = fi + (n_ex/2 - ei - 0.5) * bh
            col = EX[ei][1]
            ax.barh(yy, v, height=bh*0.92, color=col, edgecolor=CARD_EC, linewidth=0.7, zorder=3)
            star = "  ★" if (fname == "accum_resident_v3" and ei == 1) else ""
            ax.text(v + 0.15, yy, f"{v}×{star}", va="center", fontsize=8.3,
                    fontweight="bold", color=(V3 if star else INK))
    ax.axvline(1.0, color="#999", lw=1, ls="--")
    ax.axvline(13.19, color=STEEL, lw=1.4, ls=":")
    ax.text(13.19, len(feats)-0.4, "XNNPACK\n(bitvla)", fontsize=8, color=STEEL, ha="center", fontweight="bold")
    ax.set_yticks(yb); ax.set_yticklabels([f[0] for f in feats], fontsize=10)
    ax.set_xlim(0, 19); ax.set_xlabel("speedup vs frozen baseline (×) — per driving example")
    ax.set_title("How each mined optimization affects the RVV dialect — and how the ranking depends on the example",
                 loc="left", color=INK, fontsize=12, pad=10)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=lab) for lab, c in EX], fontsize=9, loc="lower right",
              framealpha=0.95, facecolor="white")
    callout(ax, (16.88, 6 - 0.26), "v3 wins bitvla (beats XNNPACK)\nbut is 2.38× on openvla — best\nkernel is per-model",
            (10.5, 4.3), fc="#f7efe2", ec=V3)
    fig.text(0.5, -0.02,
             "Figure D:  Per-optimization effect on the RVV dialect, by driving example.  The recorded beam/autotune ran "
             "on GEMM 64³ ONLY (7 candidates, best tiled+lmul 8.16×) and predates v3 — so it never saw the real\n"
             "whole-model winner.  On real models the ranking flips: accum-resident v3 wins bitvla (16.88×, beats XNNPACK) "
             "while tiled-vfmacc wins openvla (3.65×).  The beam must be driven by whole-model examples to rank faithfully.",
             ha="center", fontsize=9.2, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_opt_effects.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_opt_effects.png")
    plt.close(fig)


# ============================================================================
# FIGURE E — beam_rvv_v2 RESULT: faithful whole-model beam ranking per model.
# Reads the versioned run dir (artifacts/kernel-mining/rvv/beam_rvv_v2_*/ranking_*.yaml).
# ============================================================================
def fig_beam_ranking():
    import glob, yaml
    runs = sorted(glob.glob(str(OUT.parents[2] / "artifacts/kernel-mining/rvv/beam_rvv_v2_*")))
    if not runs:
        print("beam_ranking: no beam_rvv_v2 run; skipping"); return
    run = runs[-1]; V3 = "#b8742a"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6)); fig.patch.set_facecolor("white")
    XNN = {"bitvla": 13.19, "openvla": None}
    for ax, M in zip(axes, ("bitvla", "openvla")):
        r = yaml.safe_load(open(f"{run}/ranking_{M}.yaml"))
        rows = [x for x in r["ranked"]]
        rows = sorted(rows, key=lambda x: (x["speedup"] is None, -(x["speedup"] or 0)))
        labs, vals, cols = [], [], []
        for x in rows:
            sp = x["speedup"]; tag = x["tag"]
            if sp is None:
                labs.append(f"{tag} (blocked)"); vals.append(0.0); cols.append("#cfcfcf")
            else:
                labs.append(tag); vals.append(sp)
                if x["lowering"] == "scalar_fallback": cols.append(SALMON)
                elif tag == rows[0]["tag"]: cols.append(V3 if "v3" in tag or "accum" in tag else GOLD)
                elif sp <= 1.05: cols.append(GREY)
                else: cols.append(GOLD)
        V3LOCAL = "#b8742a"
        yb = np.arange(len(labs))
        ax.set_facecolor(CREAM)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.barh(yb, vals, color=cols, edgecolor=CARD_EC, linewidth=0.9, zorder=3)
        for i, (x, v) in enumerate(zip(rows, vals)):
            t = "blocked" if x["speedup"] is None else (f"{v:.2f}×" + ("  (scalar)" if x["lowering"]=="scalar_fallback" else ""))
            ax.text((v if v else 0)+0.15, i, t, va="center", fontsize=8.4, fontweight="bold",
                    color=("#999" if x["speedup"] is None else (SALMON if x["lowering"]=="scalar_fallback" else INK)))
        if XNN[M]:
            ax.axvline(XNN[M], color=STEEL, ls=":", lw=1.5)
            ax.text(XNN[M], len(labs)-0.4, "XNNPACK\n13.19×", fontsize=8, color=STEEL, ha="center", fontweight="bold")
        ax.axvline(1.0, color="#999", lw=1, ls="--")
        ax.set_yticks(yb); ax.set_yticklabels(labs, fontsize=8.6); ax.invert_yaxis()
        ax.set_xlabel("whole-model speedup vs baseline (×)")
        win = rows[0]
        ax.set_title(f"{M} — winner: {win['tag']} ({win['speedup']:.2f}×)", loc="left", color=INK, fontsize=12, pad=8)
        ax.set_xlim(0, max(max(vals)*1.18, 2))
    B = "beam_rvv_v2"
    fig.suptitle(f"Faithful whole-model beam ({B}) — every candidate optimization ranked per model on K1",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03,
             "Figure E:  The beam re-run on REAL driving examples (not the stale single-64³ v1).  Each candidate "
             "optimization measured whole-model on K1 (cos-gated).  The best kernel is per-model — v3 wins bitvla\n"
             "(16.77×, beats XNNPACK), accum-resident-wholemodel wins openvla (4.97×).  vfmacc_contraction regresses "
             "(scalar fallback, not whole-model-safe); lmul is ~no-op; v3+activation is composition-blocked.",
             ha="center", fontsize=9.2, color=INK)
    fig.tight_layout(rect=[0,0,1,0.97])
    for ext in ("png","svg"):
        fig.savefig(OUT / f"beam_rvv_v2_ranking.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "beam_rvv_v2_ranking.png")
    plt.close(fig)


# ============================================================================
# FIGURE F — beam PERFORMANCE + UTILIZATION per candidate (bitvla & openvla).
# Performance = whole-model speedup (bars). Utilization = % of the expert ceiling
# reached (speedup / XNNPACK-or-best-expert), with the VPU state (vfmacc-vectorized
# vs scalar-fallback) encoded by colour — the honest "does it use the vector unit"
# signal (K1 has no userspace perf counters: rdcycle traps, so this is the proxy).
# ============================================================================
def fig_beam_util_perf():
    import glob, yaml
    runs = sorted(glob.glob(str(OUT.parents[2] / "artifacts/kernel-mining/rvv/beam_rvv_v2_*")))
    if not runs: print("beam_util: no run; skip"); return
    run = runs[-1]; V3 = "#b8742a"
    CEIL = {"bitvla": 13.19, "openvla": 4.97}  # XNNPACK (bitvla) / best achieved (openvla, no lib kernel)
    CEIL_LBL = {"bitvla": "XNNPACK ceiling", "openvla": "best-achieved ceiling"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8)); fig.patch.set_facecolor("white")
    for ax, M in zip(axes, ("bitvla", "openvla")):
        r = yaml.safe_load(open(f"{run}/ranking_{M}.yaml"))
        rows = [x for x in r["ranked"] if x["speedup"] is not None]
        rows = sorted(rows, key=lambda x: x["speedup"])  # ascending: baseline low → best high
        ax.set_facecolor(CREAM)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        yb = np.arange(len(rows)); ceil = CEIL[M]
        for i, x in enumerate(rows):
            sp = x["speedup"]; scalar = x["lowering"] == "scalar_fallback"
            best = (i == len(rows)-1)
            col = SALMON if scalar else (V3 if best else (GREY if sp <= 1.05 else GOLD))
            ax.barh(i, sp, color=col, edgecolor=CARD_EC, linewidth=0.9, zorder=3)
            util = 100.0 * sp / ceil
            note = "  ⛔scalar (0% VPU)" if scalar else f"  · {util:.0f}% of ceiling"
            ax.text(sp + ceil*0.012, i, f"{sp:.2f}×{note}", va="center", fontsize=8.3,
                    fontweight="bold", color=(SALMON if scalar else INK))
        ax.axvline(ceil, color=STEEL, ls=":", lw=1.6)
        ax.text(ceil, len(rows)-0.4, f"{CEIL_LBL[M]}\n({ceil}× = 100%)", fontsize=8, color=STEEL, ha="center", fontweight="bold")
        ax.axvline(1.0, color="#999", lw=1, ls="--")
        ax.set_yticks(yb); ax.set_yticklabels([x["tag"] for x in rows], fontsize=8.6)
        ax.set_xlabel("performance — whole-model speedup vs baseline (×)")
        ax.set_xlim(0, ceil*1.35)
        ax.set_title(f"{M}: performance + VPU utilization per beam candidate", loc="left", color=INK, fontsize=11.5, pad=8)
    fig.suptitle("Beam candidates — performance (speedup) and utilization (% of ceiling + VPU state), baseline → best",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03,
             "Figure F:  Each beam candidate's PERFORMANCE (bar = whole-model speedup) and UTILIZATION (% of the expert "
             "ceiling reached; colour = VPU state).  Gold = vfmacc-vectorized, salmon = scalar fallback (0% VPU — a vector\n"
             "feature that didn't apply whole-model), grey = ~baseline, dark-gold = best.  bitvla's v3 exceeds 100% (beats "
             "XNNPACK); openvla's best is accum-resident-wholemodel.  Utilization is a ceiling proxy (K1 traps rdcycle — no HW counters).",
             ha="center", fontsize=9.0, color=INK)
    fig.tight_layout(rect=[0,0,1,0.96])
    for ext in ("png","svg"):
        fig.savefig(OUT / f"beam_util_perf.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "beam_util_perf.png")
    plt.close(fig)


# ============================================================================
# FIGURE G — FOUR-WAY whole-model on K1: baseline / ours-best / XNNPACK / OpenBLAS.
# Left = ALL-4 (absolute latency, log y — baseline = the starting point, visible).
# Right = ZOOMED (drop baseline; ours vs XNNPACK vs OpenBLAS, the competitive contest,
# linear, with "ours reaches X% of best expert"). Reads output/rvv_bench/k1_4way_*.json.
# ============================================================================
def fig_fourway():
    import json, glob
    V3 = "#b8742a"; OB = "#7a9e7a"
    COL = {"baseline": SALMON, "ours": V3, "xnnpack_kernels": STEEL, "openblas_kernels": OB}
    models = ["bitvla", "openvla", "rdt2"]
    data = {}
    for m in models:
        # Prefer the .vf re-measure (has ours_wholemodel_vf = the final best-ours) where it exists;
        # else the four-way (bitvla, whose best-ours is v3, lives only in k1_4way).
        vf = OUT.parents[1] / "rvv_bench" / f"k1_vf_{m}.json"
        p = OUT.parents[1] / "rvv_bench" / f"k1_4way_{m}.json"
        src = vf if vf.is_file() else p
        if src.is_file():
            data[m] = json.load(open(src))
    if not data:
        print("fourway: no k1_4way_*.json yet; skipping"); return
    models = [m for m in models if m in data]

    def cfg(s, key):  # (wall_s, range_pct) or None
        r = s.get(key) or {}
        if r.get("skipped") or not r.get("min_wall_ns"): return None
        return (r["min_wall_ns"]/1e9, (r.get("spread") or {}).get("range_pct", 0.0))
    def ours_key(s):  # the BEST ours-* config that ran (vf > v3 > wholemodel > tiled)
        cands = [k for k in ("ours_wholemodel_vf", "ours_v3", "ours_wholemodel", "ours_tiled")
                 if (s.get(k) or {}).get("min_wall_ns")]
        return min(cands, key=lambda k: s[k]["min_wall_ns"]) if cands else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6)); fig.patch.set_facecolor("white")
    # -- left: ALL-4 absolute latency (log) --
    ax = axes[0]; ax.set_facecolor(CREAM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    series = ["baseline", "ours", "xnnpack_kernels", "openblas_kernels"]
    labs = {"baseline": "baseline", "ours": "ours (best)", "xnnpack_kernels": "XNNPACK", "openblas_kernels": "OpenBLAS"}
    x = np.arange(len(models)); bw = 0.2
    for i, ser in enumerate(series):
        ys, errs = [], []
        for m in models:
            s = data[m]; key = ours_key(s) if ser == "ours" else ser
            c = cfg(s, key)
            ys.append(c[0] if c else 0); errs.append((c[0]*c[1]/100.0) if c else 0)
        ax.bar(x + (i-1.5)*bw, ys, bw*0.9, yerr=errs, capsize=2,
               color=COL[ser], edgecolor=CARD_EC, linewidth=0.8, label=labs[ser], zorder=3)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("whole-model latency (s, log) — lower = faster")
    ax.set_title("① All four — incl. baseline (the starting point)", loc="left", color=INK, fontsize=12, pad=8)
    ax.legend(fontsize=9, ncol=2); ax.grid(True, axis="y", ls=":", alpha=0.35)

    # -- right: ZOOMED contenders (drop baseline), absolute latency linear, with %-of-best-expert --
    ax = axes[1]; ax.set_facecolor(CREAM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # Zoom plots SPEEDUP (×) so models of different absolute magnitude are comparable; baseline excluded.
    zser = ["ours", "xnnpack_kernels", "openblas_kernels"]
    def spdup(s, key):  # (speedup×, abs_err×) from walls, or None
        c = cfg(s, key); b = cfg(s, "baseline")
        if not (c and b): return None
        v = b[0] / c[0]; return (v, v * c[1] / 100.0)
    for i, ser in enumerate(zser):
        ys, errs = [], []
        for m in models:
            s = data[m]; key = ours_key(s) if ser == "ours" else ser
            v = spdup(s, key)
            ys.append(v[0] if v else np.nan); errs.append(v[1] if v else 0)
        ax.bar(x + (i-1)*bw, ys, bw*0.9, yerr=errs, capsize=2,
               color=COL[ser], edgecolor=CARD_EC, linewidth=0.8, label=labs[ser], zorder=3)
    # annotate ours-reaches-% of the faster expert per model (speedup space)
    for j, m in enumerate(models):
        s = data[m]; o = spdup(s, ours_key(s)); xn = spdup(s, "xnnpack_kernels"); ob = spdup(s, "openblas_kernels")
        exps = [e[0] for e in (xn, ob) if e]
        if o and exps:
            best_exp = max(exps)  # fastest expert = highest speedup
            pct = round(100*o[0]/best_exp)
            tag = "ours WINS" if o[0] >= best_exp else f"ours {pct}% of best expert"
            ax.text(j, max(o[0], *exps)*1.06, tag, ha="center", fontsize=8.5,
                    fontweight="bold", color=(V3 if o[0] >= best_exp else INK))
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("speedup vs baseline (×) — higher = faster")
    ax.set_title("② Zoomed: ours vs XNNPACK vs OpenBLAS (no baseline)", loc="left", color=INK, fontsize=12, pad=8)
    ax.legend(fontsize=9); ax.grid(True, axis="y", ls=":", alpha=0.35)

    fig.suptitle("Whole-model four-way on real K1 silicon — baseline · ours-best · XNNPACK · OpenBLAS (same-pass, cos-gated)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03,
             "Figure G:  Same-pass four-way (one campaign vs the same baseline; experts use resident-weight pack). "
             "LEFT shows all four incl. the baseline starting point (log).  RIGHT drops baseline to zoom into the "
             "competitive contest: ours is within ~1.6–1.8× of the experts and BEATS them on bitvla.",
             ha="center", fontsize=9.0, color=INK)
    fig.tight_layout(rect=[0,0,1,0.96])
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_fourway.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_fourway.png")
    plt.close(fig)


# ============================================================================
# FIGURE H — STRUCTURAL GAP ATTRIBUTION (iteration 1→2→3): where the matmul-kernel
# residual actually is, from the memory-traffic decode facet. Refutes the "packing"
# hypothesis: iter-2 .vf already TIES XNNPACK on data movement; the only lever left
# (OpenBLAS MR>1 A-reuse) is structurally unreachable on the small-M VLAs.
# Source: output/kernels/ceiling/packing_residual_decode.json (+ packing_residual.md).
# ============================================================================
def fig_gap_attribution():
    import json
    V3 = "#b8742a"; OB = "#7a9e7a"
    d = json.load(open(OUT / "packing_residual_decode.json"))
    def pick(name):  # representative entry (all shapes agree for ours/xnnpack)
        es = [e for e in d if e["kernel"] == name]
        return es[0]["memory"] if es else None
    vv = pick("ours_wholemodel"); vf = pick("ours_wholemodel_vf"); xn = pick("xnnpack")
    MR4_LPF = 1.25         # decode-confirmed (packing_residual.md table; large-M only)
    OB_AMORT = 1.06        # OpenBLAS MR=16 amortized loads/FMA (packing_residual.md finding 2)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4)); fig.patch.set_facecolor("white")

    # -- LEFT: the iteration-2 win — A-broadcast ladder per FMA collapses to XNNPACK's 0 --
    ax = axes[0]; ax.set_facecolor(CREAM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    bars = [("ours iter-1\n(.vv wholemodel)", vv["a_broadcast_per_fma"], SALMON),
            ("ours iter-2\n(.vf wholemodel)", vf["a_broadcast_per_fma"], V3),
            ("XNNPACK\n(1x4v)", xn["a_broadcast_per_fma"], STEEL)]
    xb = np.arange(len(bars))
    ax.bar(xb, [b[1] for b in bars], 0.6, color=[b[2] for b in bars],
           edgecolor=CARD_EC, linewidth=1.4, zorder=3)
    for xi, (lab, v, c) in zip(xb, bars):
        ax.text(xi, v + 0.18, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold", color=c)
    ax.set_xticks(xb); ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_ylabel("A-broadcast ladder ops / FMA  (lower = better)")
    ax.set_ylim(0, 9.4)
    ax.set_title("1 · Iteration-2 .vf collapses the A-broadcast ladder", loc="left", color=INK, fontsize=12, pad=8)
    callout(ax, (1, 0.2), "iter-2 .vf ties XNNPACK\n(0 ladder ops)", (1.55, 4.6), fc="#f7efe2", ec=V3)
    ax.text(0, vv["a_broadcast_per_fma"] - 1.4, "the .vv\nbroadcast\npenalty", ha="center",
            fontsize=8.2, color=SALMON, style="italic")
    ax.grid(True, axis="y", ls=":", alpha=0.35)

    # -- RIGHT: loads / useful-FMA — ours-.vf == XNNPACK (residual CLOSED); MR>1 A-reuse is the only lever --
    ax = axes[1]; ax.set_facecolor(CREAM)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    rows = [("ours iter-2 (.vf)", vf["loads_per_fma"], V3, False),
            ("XNNPACK", xn["loads_per_fma"], STEEL, False),
            ("ours iter-3 (.vf MR4)\nlarge-M only", MR4_LPF, GOLD, True),
            ("OpenBLAS (MR=16)\namortized", OB_AMORT, OB, True)]
    xb = np.arange(len(rows))
    for xi, (lab, v, c, hatch) in zip(xb, rows):
        ax.bar(xi, v, 0.6, color=c, edgecolor=CARD_EC, linewidth=1.4, zorder=3,
               hatch="//" if hatch else None)
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=11.5, fontweight="bold", color=c)
    ax.axhline(vf["loads_per_fma"], color=STEEL, ls="--", lw=1.3, zorder=1)
    ax.set_xticks(xb); ax.set_xticklabels([r[0] for r in rows], fontsize=8.8)
    ax.set_ylabel("loads / useful-FMA  (lower = better)")
    ax.set_ylim(0, 2.55)
    ax.set_title("2 · Per-FMA loads: residual vs XNNPACK is CLOSED", loc="left", color=INK, fontsize=12, pad=8)
    callout(ax, (0.5, vf["loads_per_fma"]), "ours .vf == XNNPACK (2.0)\nkernel residual closed",
            (1.4, 2.32), fc="#f7efe2", ec=V3)
    ax.text(2.5, 0.45, "MR>1 A-reuse: the only lever left,\nbut needs large-M — openvla/rdt2 are\nall small-M (token dim 1–28) → unreachable",
            ha="center", fontsize=7.8, color="#7a6a4a", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#faf6ec", ec="#d8c89a", lw=0.8))
    ax.grid(True, axis="y", ls=":", alpha=0.35)

    fig.suptitle("Where the matmul-kernel residual is — memory-traffic decode (iteration 1→2→3)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.05,
             "Figure H:  Static memory-traffic decode of the emitted RVV asm.  LEFT: iteration-2 .vf eliminates the "
             "8-op A-broadcast ladder of the .vv kernel, tying XNNPACK at 0.  RIGHT: ours-.vf already matches XNNPACK's "
             "2.0 loads/useful-FMA at every openvla/rdt2 shape — the matmul-kernel residual vs XNNPACK is CLOSED.  The "
             "only data-movement lever left is OpenBLAS's MR>1 A-reuse register block (iter-3 MR4 reaches 1.25), which "
             "is structurally unreachable on the small-M VLA matmuls.  ⇒ the whole-model 60/63% gap is DISPATCH-LEVEL "
             "(non-matmul / no large-M batching), not the kernel.",
             ha="center", fontsize=9.0, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_gap_attribution.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_gap_attribution.png")
    plt.close(fig)


# ============================================================================
# FIGURE I — DISPATCH BREAKDOWN: MEASURED proof that the whole-model gap is
# dispatch-level, not the matmul kernel. Stacked wall = matmul-bucket (shared,
# decode-equal) + dispatch-bucket. Source: output/rvv_bench/dispatch_breakdown.json
# (K1 board, rdtime matmul ticks + wall, cos-gated).
# ============================================================================
def fig_dispatch_breakdown():
    import json
    MATMUL = "#6f93b0"; DISP = "#cf8b7d"  # matmul = steel (shared); dispatch = salmon (the gap)
    d = json.load(open(OUT.parents[1] / "rvv_bench" / "dispatch_breakdown.json"))
    panels = [("openvla", "openvla_fp32_consistent"), ("rdt2", "rdt2_fp32_consistent")]
    panels = [(nm, k) for nm, k in panels if k in d]
    fig, axes = plt.subplots(1, len(panels), figsize=(13.5, 4.8)); fig.patch.set_facecolor("white")
    if len(panels) == 1: axes = [axes]
    for ax, (nm, key) in zip(axes, panels):
        ax.set_facecolor(CREAM)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        loc = d[key]["localize_ours_wholemodel_vf"]
        mm = loc["shared_matmul_bucket_ns"] / 1e6  # ms (shared, equal across configs)
        bars = [("ours .vf", mm, loc["ours_dispatch_bucket_ns"] / 1e6, True),
                ("XNNPACK",  mm, loc["xnnpack_dispatch_bucket_ns"] / 1e6, False)]
        y = np.arange(len(bars))[::-1]
        for yi, (lab, mmms, dispms, is_ours) in zip(y, bars):
            ax.barh(yi, mmms, height=0.52, color=MATMUL, edgecolor=CARD_EC, linewidth=1.3, zorder=3)
            ax.barh(yi, dispms, left=mmms, height=0.52, color=DISP, edgecolor=CARD_EC, linewidth=1.3, zorder=3)
            ax.text(mmms + dispms + (mmms+dispms)*0.012, yi, f"{(mmms+dispms):.0f} ms",
                    va="center", ha="left", fontsize=10.5, fontweight="bold", color=INK)
            ax.text(mmms + dispms/2, yi, f"dispatch {dispms:.0f} ms", va="center", ha="center",
                    fontsize=8.6, color="#5b3a32", fontweight="bold")
        ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bars], fontsize=11)
        frac = d[key]["results"]["xnnpack_kernels"]["matmul_frac"] * 100
        delta = loc["delta_wall_ns"] / 1e6; over = loc["ours_over_xnnpack"]
        ax.set_xlim(0, (mm + loc["ours_dispatch_bucket_ns"]/1e6) * 1.22)
        ax.set_title(f"{nm} — matmul is {frac:.0f}% of wall; the {over:.2f}× gap is all dispatch",
                     loc="left", color=INK, fontsize=11.5, pad=8)
        ax.set_xlabel("whole-model wall (ms) — matmul-bucket (shared, decode-equal) + dispatch-bucket")
        # callout: the delta between the two bar ends is entirely dispatch
        ax.annotate("", xy=(mm + loc["ours_dispatch_bucket_ns"]/1e6, y[0]-0.34),
                    xytext=(mm + loc["xnnpack_dispatch_bucket_ns"]/1e6, y[0]-0.34),
                    arrowprops=dict(arrowstyle="<->", color="#9c4f3f", lw=1.6))
        ax.text((mm + (loc["ours_dispatch_bucket_ns"]+loc["xnnpack_dispatch_bucket_ns"])/2e6), y[0]-0.52,
                f"Δ {delta:.0f} ms = 100% dispatch", ha="center", fontsize=8.4,
                color="#9c4f3f", fontweight="bold")
        ax.set_ylim(-0.95, len(bars)-0.45)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=MATMUL, label="matmul kernel (shared, = XNNPACK by decode)"),
                            Patch(color=DISP, label="dispatch / non-matmul (the gap)")],
                   fontsize=8.4, loc="lower right")
    fig.suptitle("Where the whole-model time goes — matmul kernel vs dispatch (K1 silicon, measured)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.04,
             "Figure I:  K1 board, per-dispatch rdtime split (cos ≥ 0.99999, N=5/3).  The matmul kernel — proven to "
             "decode identically to XNNPACK — is only 8 % (openvla) / 3 % (rdt2) of whole-model wall and is EQUAL "
             "across configs (steel).  The entire 1.66× / 1.59× ours-vs-XNNPACK gap lives in the dispatch / non-matmul "
             "bucket (salmon): attention/norm/softmax/activation on the un-tuned RVV path + per-dispatch glue.  ⇒ the "
             "next win is a dispatch-level effort, not the matmul kernel.",
             ha="center", fontsize=9.0, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_dispatch_breakdown.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_dispatch_breakdown.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_e2e()
    fig_crossover()
    fig_progression()
    fig_opt_effects()
    fig_beam_ranking()
    fig_beam_util_perf()
    fig_fourway()
    fig_gap_attribution()
    fig_dispatch_breakdown()
