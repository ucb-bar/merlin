#!/usr/bin/env python
"""Consolidated plot of EVERY measured baseline / OpenBLAS / XNNPACK / ours comparison.

Three metric families kept on separate axes because they are NOT comparable:
  (1) single-GEMM on real K1 silicon  — rdtime ticks, full 32^3..512^3 sweep (the headline crossover)
  (2) single-GEMM on spike            — retired instructions, IPC=1 proxy (ranks OUR approaches)
  (3) whole-model end-to-end on K1    — wall seconds vs the frozen baseline

Data is transcribed verbatim from:
  output/kernels/ceiling/cross_framework_matrix.md           (spike)
  output/kernels/ceiling/cross_framework_matrix_k1.jsonl     (K1 inner small shapes)
  output/kernels/ceiling/large_shape_packing_k1.md           (K1 sweep, inner + pack-included)
  output/rvv_bench/k1_e2e_{bitvla,smolvla,rdt2_mtail}.md      (whole-model e2e)

ours-intrinsic is a HAND-WRITTEN ceiling reference (a target, not a compiler result); every other
"ours-*" is emitted by the Merlin RVV pipeline (frozen baseline + a default-off feature).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[1] / "output" / "kernels" / "ceiling"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888", "savefig.facecolor": "white"})

C = {"OpenBLAS": "#e8893b", "XNNPACK": "#d4564a", "intrinsic": "#2596be",
     "tiled": "#3ea567", "vfmacc": "#7a52c7", "baseline": "#6b7280"}

shapes = [32, 64, 128, 256, 384, 512]
# (1) K1 silicon, inner-compute ticks  (large_shape_packing_k1.md)
k1 = {
    "OpenBLAS":  [409, 2329, 17789, 149008, 491134, 1146939],
    "XNNPACK":   [238, 1976, 31761, 248282, 891310, 2079033],
    "intrinsic": [172, 1350, 14144, 106034, 441356, 1394602],
    "tiled":     [2694, 20309, 168957, 1297217, None, None],
    "baseline":  [39246, 306608, 2516021, 19613446, None, None],
}
# (2) spike retired instructions  (cross_framework_matrix.md)
sp_shapes = [32, 64, 128]
sp = {
    "intrinsic": [6551, 50695, 399241],
    "OpenBLAS":  [11039, 84483, 664811],
    "XNNPACK":   [13289, 101705, 798857],
    "vfmacc":    [11883, 123094, None],
    "tiled":     [166251, 1328219, 10665305],
    "baseline":  [2804206, 22430926, 179441453],
}
# (3) whole-model e2e  (k1_e2e_*.md)  -- (model, baseline_s, opt_s, speedup, feature)
e2e = [("bitvla", 2.525, 0.274, 9.22, "fused_vfmacc_tiled"),
       ("openvla", 5.848, 1.619, 3.61, "fused_vfmacc_tiled"),
       ("rdt2", 73.71, 31.41, 2.35, "accumulator_resident_wholemodel")]

fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.95], wspace=0.28)

# ---- Panel 1: K1 silicon log-log sweep ----
ax = fig.add_subplot(gs[0])
order = ["baseline", "tiled", "OpenBLAS", "XNNPACK", "intrinsic"]
labels = {"baseline": "ours-baseline (compiler)", "tiled": "ours-tiled (compiler)",
          "OpenBLAS": "OpenBLAS (expert)", "XNNPACK": "XNNPACK (expert)",
          "intrinsic": "ours-intrinsic (hand ceiling)"}
for k in order:
    xs = [s for s, v in zip(shapes, k1[k]) if v is not None]
    ys = [v for v in k1[k] if v is not None]
    ax.plot(xs, ys, marker="o", ms=5, lw=2.2, color=C[k],
            ls="--" if k == "intrinsic" else "-", label=labels[k])
ax.axvline(512, color=C["OpenBLAS"], ls=":", lw=1, alpha=0.7)
ax.annotate("OpenBLAS\nretakes lead\n@512³", xy=(512, 1146939), xytext=(300, 1.7e6),
            color=C["OpenBLAS"], fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", color=C["OpenBLAS"], lw=1))
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(shapes); ax.set_xticklabels([f"{s}³" for s in shapes])
ax.set_xlabel("GEMM size (M=N=K)"); ax.set_ylabel("rdtime ticks  (lower = faster)")
ax.set_title("① Single GEMM · real K1 silicon\n(inner-compute; the crossover)", fontsize=11)
ax.grid(True, which="both", ls=":", alpha=0.35); ax.legend(fontsize=7.6, loc="upper left")

# ---- Panel 2: spike grouped log bars ----
ax = fig.add_subplot(gs[1])
border = ["intrinsic", "OpenBLAS", "XNNPACK", "vfmacc", "tiled", "baseline"]
import numpy as np
x = np.arange(len(sp_shapes)); bw = 0.13
for i, k in enumerate(border):
    vals = [v if v is not None else 0 for v in sp[k]]
    bars = ax.bar(x + (i - 2.5) * bw, vals, bw * 0.92, color=C[k], label=k)
    for j, v in enumerate(sp[k]):
        if v is None:
            ax.text(x[j] + (i - 2.5) * bw, 1.5e4, "not_run", rotation=90,
                    fontsize=6.5, ha="center", va="bottom", color="#888")
ax.set_yscale("log"); ax.set_ylim(1e4, 5e8)
ax.set_xticks(x); ax.set_xticklabels([f"{s}³" for s in sp_shapes])
ax.set_ylabel("retired instructions (IPC=1 proxy)")
ax.set_title("② Single GEMM · spike\n(proxy — ranks OUR approaches)", fontsize=11)
ax.grid(True, axis="y", ls=":", alpha=0.35); ax.legend(fontsize=7.6, ncol=2)

# ---- Panel 3: e2e speedup bars ----
# Measured XNNPACK-kernels whole-model speedup vs the SAME frozen baseline (third e2e column;
# output/rvv_bench/k1_e2e_xnnpack_bitvla.{json,md}). Only bitvla has the XNNPACK column today.
# (model -> speedup of XNNPACK-kernels over baseline). XNNPACK swaps ONLY the f32 GEMM kernel;
# graph/weights/runtime are identical to ours-optimized, so the bitvla gap (13.65 vs 9.18)
# isolates kernel-level headroom: XNNPACK's hand RVV GEMM is ~1.49x our vfmacc whole-model.
e2e_xnn = {"bitvla": 13.65}
ax = fig.add_subplot(gs[2])
names = [r[0] for r in e2e][::-1]
sp_v = [r[3] for r in e2e][::-1]
y = np.arange(len(names))
has_xnn = [e2e_xnn.get(n) is not None for n in names]
# Where an XNNPACK column exists, render the ours-optimized bar slimmer + above the XNNPACK bar.
h_full, h_pair = 0.55, 0.26
for i, n in enumerate(names):
    if has_xnn[i]:
        ax.barh(i + 0.16, sp_v[i], color="#e0a72e", height=h_pair, label=None)
        ax.barh(i - 0.16, e2e_xnn[n], color="#c1467a", height=h_pair, label=None)
    else:
        ax.barh(i, sp_v[i], color="#e0a72e", height=h_full)
for i, r in enumerate(e2e[::-1]):
    n = names[i]
    if has_xnn[i]:
        ax.text(r[3] + 0.15, i + 0.16, f"ours {r[3]}×", va="center", fontsize=8.5,
                fontweight="bold", color="#9c7415")
        ax.text(e2e_xnn[n] + 0.15, i - 0.16, f"XNNPACK {e2e_xnn[n]}×", va="center",
                fontsize=8.5, fontweight="bold", color="#a03062")
        ax.text(0.15, i + 0.16, f"{r[1]}s→{r[2]}s", va="center", fontsize=7, color="#222")
    else:
        ax.text(r[3] + 0.12, i, f"{r[3]}×", va="center", fontsize=11, fontweight="bold",
                color="#9c7415")
        ax.text(0.15, i, f"{r[1]}s→{r[2]}s", va="center", fontsize=8, color="#222")
ax.axvline(1.0, color="#999", lw=1)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0, 16); ax.set_xlabel("whole-model speedup vs frozen baseline")
ax.set_title("③ Whole-model e2e · K1\n(wall s; cos≥0.9999; ours vs XNNPACK kernel)", fontsize=11)
ax.grid(True, axis="x", ls=":", alpha=0.35)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#e0a72e", label="ours-tiled (vfmacc)"),
                   Patch(color="#c1467a", label="XNNPACK RVV GEMM")],
          fontsize=7.6, loc="lower right")

fig.suptitle("RVV GEMM — every measured comparison  (baseline · OpenBLAS · XNNPACK · ours)",
             fontsize=13, fontweight="bold", y=1.02)
for ext in ("png", "svg"):
    p = OUT / f"all_comparisons.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    print("wrote", p)


# ===========================================================================
# Op-coverage plot — the comparison BEYOND GEMM. Reads the measured cross_framework
# ops jsonl (gelu / sigmoid / int8-gemm / dwconv / conv2d / attention on real K1) so
# the coverage is visibly more than GEMM. One panel per op family; bars are K1 rdtime
# ticks (lower = faster); ours-vs-XNNPACK where the library has the op, ours-vs-ours
# for attention (no library attention primitive). not_run rows are drawn as a hatch.
def plot_op_coverage():
    import json
    src = OUT / "cross_framework_ops_k1.jsonl"
    if not src.is_file():
        print("op_coverage: no cross_framework_ops_k1.jsonl yet; skipping")
        return
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

    def sz(r):
        if r["op"] in ("gelu", "sigmoid"):
            n = r.get("size_n", 0)
            return f"{n//1024}K" if n >= 1024 else str(n)
        if r["op"] in ("int8_gemm",):
            return f"{r.get('M')}³"
        if r["op"] == "conv2d":
            return f"{r.get('M')}x{r.get('N')}x{r.get('K')}"
        if r["op"] == "attention_bmm":
            return f"{r.get('B')}x{r.get('M')}x{r.get('N')}x{r.get('K')}"
        if r["op"] == "dwconv":
            return f"{r.get('C')}c·3x3"
        return ""

    # color per source family
    col = {"xnnpack": C["XNNPACK"],
           "ours_scalar": C["baseline"], "ours_vectorized": C["vfmacc"],
           "ours_f32_baseline": C["baseline"], "ours_int8_w8a8": C["intrinsic"],
           "ours_conv_baseline": C["baseline"], "ours_conv_vfmacc": C["vfmacc"],
           "ours_bmm_baseline": C["baseline"], "ours_bmm_vfmacc": C["vfmacc"],
           "ours_depthwise": C["tiled"]}
    nice = {"xnnpack": "XNNPACK", "ours_scalar": "ours-scalar",
            "ours_vectorized": "ours-vectorized", "ours_f32_baseline": "ours-f32",
            "ours_int8_w8a8": "ours-int8-W8A8", "ours_conv_baseline": "ours-baseline",
            "ours_conv_vfmacc": "ours-vfmacc", "ours_bmm_baseline": "ours-baseline",
            "ours_bmm_vfmacc": "ours-vfmacc", "ours_depthwise": "ours-depthwise"}
    op_titles = {"gelu": "GELU (f32)", "sigmoid": "sigmoid (f32)",
                 "int8_gemm": "int8 GEMM (qd8/W8A8)", "dwconv": "depthwise conv (f32 3x3)",
                 "conv2d": "conv2d (f32, im2col→GEMM)",
                 "attention_bmm": "attention bmm (ours-vs-ours)"}
    ops = [o for o in ["gelu", "sigmoid", "int8_gemm", "dwconv", "conv2d", "attention_bmm"]
           if any(r["op"] == o for r in rows)]

    ncol = 3
    nrow = (len(ops) + ncol - 1) // ncol
    figh = plt.figure(figsize=(5.2 * ncol, 3.7 * nrow))
    for idx, op in enumerate(ops):
        axh = figh.add_subplot(nrow, ncol, idx + 1)
        orows = [r for r in rows if r["op"] == op]
        # x groups = distinct sizes (sorted); series = sources present
        sizes_seen = []
        for r in orows:
            s = sz(r)
            if s not in sizes_seen:
                sizes_seen.append(s)
        srcs = []
        for r in orows:
            if r["source"] not in srcs:
                srcs.append(r["source"])
        import numpy as _np
        xg = _np.arange(len(sizes_seen))
        bw = 0.8 / max(1, len(srcs))
        for si, s in enumerate(srcs):
            ys, xs, hatch_x = [], [], []
            for gi, gs in enumerate(sizes_seen):
                m = [r for r in orows if r["source"] == s and sz(r) == gs]
                v = (m[0].get("ticks") if m else None)
                xpos = gi + (si - (len(srcs) - 1) / 2) * bw
                if v:
                    ys.append(v); xs.append(xpos)
                else:
                    hatch_x.append(xpos)
            if xs:
                axh.bar(xs, ys, bw * 0.92, color=col.get(s, "#999"), label=nice.get(s, s))
            for hx in hatch_x:
                axh.bar([hx], [1], bw * 0.92, color="none", edgecolor="#bbb", hatch="//")
                axh.text(hx, 1.3, "not_run", rotation=90, fontsize=6, ha="center",
                         va="bottom", color="#999")
        axh.set_yscale("log")
        axh.set_xticks(xg); axh.set_xticklabels(sizes_seen, fontsize=8)
        axh.set_ylabel("rdtime ticks (log)", fontsize=8)
        axh.set_title(op_titles.get(op, op), fontsize=10)
        axh.grid(True, axis="y", ls=":", alpha=0.35)
        axh.legend(fontsize=6.8, ncol=2)
    figh.suptitle("Cross-framework op coverage on real K1 silicon — GEMM + GELU + sigmoid + "
                  "int8-GEMM + conv + depthwise + attention\n(inner-compute rdtime ticks, "
                  "bit-exact/cos-verified; attention is ours-vs-ours — no library primitive)",
                  fontsize=11.5, fontweight="bold", y=1.0)
    figh.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "op_coverage.png"
    figh.savefig(p, bbox_inches="tight", dpi=150)
    print("wrote", p)


plot_op_coverage()
