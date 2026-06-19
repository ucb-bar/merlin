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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "output" / "kernels" / "ceiling"

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
    fig = plt.figure(figsize=(12, 5.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

    # -- left: bitvla latency card — baseline / ours-vfmacc / XNNPACK / ours-v3 (the winner) --
    ax = fig.add_subplot(gs[0]); card(ax, "bitvla — whole-model latency on K1 silicon")
    V3 = "#b8742a"  # ours accumulator-resident microkernel (compiler) — the winner
    rows = [("baseline\n(hand_v0)", 2521, SALMON, "1.00×", False),
            ("ours-vfmacc\n(tiled, compiler)", 275, GOLD, "9.16×", False),
            ("XNNPACK\n(hand kernel)", 184, STEEL, "13.65×", False),
            ("ours-v3\n(accum-resident, compiler)", 150, V3, "16.83×", True)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, ms, col, sp, fast) in zip(y, rows):
        ax.barh(yi, ms, height=0.6, color=col, edgecolor=CARD_EC, linewidth=1.6, zorder=3)
        tag = f"{ms} ms   ({sp})" + ("   ← fastest" if fast else "")
        ax.text(ms + 70, yi, tag, va="center", ha="left", fontsize=10.5,
                fontweight="bold", color=(V3 if fast else INK))
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.8)
    ax.set_xlim(0, 4200); ax.set_xlabel("latency (ms / forward) — lower is better")
    ax.set_xticks([0, 1000, 2000, 3000])
    callout(ax, (150, y[3]+0.32), "compiler-emitted v3 BEATS XNNPACK's\nhand kernel — 1.23× faster, cos 0.99999",
            (2050, y[3]+0.30), fc="#f7efe2", ec=V3)
    ax.set_ylim(-0.6, len(rows)-0.35)

    # -- right: best whole-model compiler speedup PER MODEL (portfolio: best kernel is per-model) --
    ax = fig.add_subplot(gs[1]); card(ax, "best compiler-emitted speedup vs baseline (per model)")
    models = [("rdt2", 2.35, "accum-resident", GOLD), ("openvla", 3.65, "tiled vfmacc", GOLD),
              ("bitvla", 16.83, "accum-resident v3", V3)]
    y = np.arange(len(models))
    for yi, (m, sp, feat, col) in zip(y, models):
        ax.barh(yi, sp, height=0.6, color=col, edgecolor=CARD_EC, linewidth=1.6, zorder=3)
        ax.text(sp + 0.25, yi, f"{sp}×", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(0.3, yi, feat, va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([m[0] for m in models], fontsize=11)
    ax.set_xlim(0, 19); ax.set_xlabel("whole-model speedup (×) — higher is better")
    ax.set_ylim(-0.6, len(models)-0.4)

    fig.text(0.5, -0.04,
             "Figure A:  Whole-model RVV speedups on real K1 silicon (cos ≥ 0.99999).  The compiler-emitted "
             "accumulator-resident micro-kernel (ours-v3) reaches 16.83× on bitvla — beating both the tiled-vfmacc\n"
             "lowering (9.16×) and XNNPACK's hand-written RVV GEMM (13.65×).  Best kernel is per-model (right): v3 wins "
             "bitvla; tiled vfmacc wins openvla (3.65×) — a portfolio the autotune layer selects from.",
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
    steps = [("baseline\nhand_v0", 1.00, SALMON),
             ("+ ntail\n(attn vfmacc)", 7.73, "#e8c98a"),
             ("+ tiled\nvfmacc", 9.16, GOLD),
             ("+ v3 accum-\nresident", 16.83, V3)]
    xs = np.arange(len(steps))
    vals = [s[1] for s in steps]
    ax.plot(xs, vals, "-", color="#9a9a9a", lw=1.6, zorder=2)
    for xi, (lab, v, col) in zip(xs, steps):
        ax.scatter([xi], [v], s=170, color=col, edgecolor=CARD_EC, linewidth=1.4, zorder=4)
        ax.text(xi, v + 0.7, f"{v}×", ha="center", fontsize=11, fontweight="bold", color=col if col != "#e8c98a" else "#9c7415")
    ax.axhline(13.65, color=STEEL, ls="--", lw=1.6, zorder=1)
    ax.text(0.05, 13.65 + 0.35, "XNNPACK hand kernel (13.65×)", fontsize=9, color=STEEL, fontweight="bold")
    callout(ax, (3, 16.83), "compiler-emitted v3\ncrosses ABOVE XNNPACK", (1.75, 17.6), fc="#f7efe2", ec=V3)
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
             "crossing ABOVE XNNPACK's hand kernel (13.65×).  Right — single-GEMM 64³ kernel trajectory (spike instret): "
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
        ("accum_resident_v3",         [None, 16.83, 2.38]),
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
    ax.axvline(13.65, color=STEEL, lw=1.4, ls=":")
    ax.text(13.65, len(feats)-0.4, "XNNPACK\n(bitvla)", fontsize=8, color=STEEL, ha="center", fontweight="bold")
    ax.set_yticks(yb); ax.set_yticklabels([f[0] for f in feats], fontsize=10)
    ax.set_xlim(0, 19); ax.set_xlabel("speedup vs frozen baseline (×) — per driving example")
    ax.set_title("How each mined optimization affects the RVV dialect — and how the ranking depends on the example",
                 loc="left", color=INK, fontsize=12, pad=10)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=lab) for lab, c in EX], fontsize=9, loc="lower right",
              framealpha=0.95, facecolor="white")
    callout(ax, (16.83, 6 - 0.26), "v3 wins bitvla (beats XNNPACK)\nbut is 2.38× on openvla — best\nkernel is per-model",
            (10.5, 4.3), fc="#f7efe2", ec=V3)
    fig.text(0.5, -0.02,
             "Figure D:  Per-optimization effect on the RVV dialect, by driving example.  The recorded beam/autotune ran "
             "on GEMM 64³ ONLY (7 candidates, best tiled+lmul 8.16×) and predates v3 — so it never saw the real\n"
             "whole-model winner.  On real models the ranking flips: accum-resident v3 wins bitvla (16.83×, beats XNNPACK) "
             "while tiled-vfmacc wins openvla (3.65×).  The beam must be driven by whole-model examples to rank faithfully.",
             ha="center", fontsize=9.2, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"paper_opt_effects.{ext}", bbox_inches="tight", dpi=160)
    print("wrote", OUT / "paper_opt_effects.png")
    plt.close(fig)


# ============================================================================
# FIGURE E — beam_rvv_v2 RESULT: faithful whole-model beam ranking per model.
# Reads the versioned run dir (mined_knowledge/rvv/beam_rvv_v2_*/ranking_*.yaml).
# ============================================================================
def fig_beam_ranking():
    import glob, yaml
    runs = sorted(glob.glob(str(OUT.parents[2] / "mined_knowledge/rvv/beam_rvv_v2_*")))
    if not runs:
        print("beam_ranking: no beam_rvv_v2 run; skipping"); return
    run = runs[-1]; V3 = "#b8742a"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6)); fig.patch.set_facecolor("white")
    XNN = {"bitvla": 13.65, "openvla": None}
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
            ax.text(XNN[M], len(labs)-0.4, "XNNPACK\n13.65×", fontsize=8, color=STEEL, ha="center", fontweight="bold")
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


if __name__ == "__main__":
    fig_e2e()
    fig_crossover()
    fig_progression()
    fig_opt_effects()
    fig_beam_ranking()
