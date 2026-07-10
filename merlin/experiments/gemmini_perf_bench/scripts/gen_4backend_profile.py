#!/usr/bin/env python3
"""House-style FireSim L5 profile of the 4 AGENTIC backends + cached golden + IREE.

The 4 backends authored in the capsule_bench agentic A/B/C experiment (raw C++, scaffold C++,
Python-Merlin, Merlin+CIRCT) profiled for cycles + PE-array utilization on the shared kernels.
golden(C lib) + IREE numbers are reused from the earlier perf_full_0001 FireSim run (we already
have them). Styled with the single house module (scripts/merlin_plotstyle.py) — never re-derive.

IREE is drawn with a meaningful hatch (+ "*"): it is NOT directly comparable (per-dispatch rdcycle
dump, all-ones self-check rc=0 rather than exact-int golden) — the one sanctioned hatch case.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/path/to/oscar-merlin")
sys.path.insert(0, str(REPO / "scripts"))
from merlin_plotstyle import (use_merlin_style, style_ax, title, suptitle, emph, block_shadow,
                              BG, INK, GOLD, BLUE, NAVY, SLATE, MAUVE, SAGE, SERIF, SANS)

PB_RUNS = REPO / "merlin" / "experiments" / "gemmini_perf_bench" / "runs"
REPORTS = REPO / "merlin" / "experiments" / "gemmini_perf_bench" / "reports"
AGENTIC_RUN = "perf_4backends_0001"     # the new FireSim run (4 agentic arms)
CACHE_RUN   = "perf_full_0001"          # cached golden + iree_dialect cells

# series identity (consistent across the repo). IREE shares MAUVE hue but is hatched + starred
# because it is not directly comparable; CIRCT is the hero (NAVY, gold-emphasised).
# 5 solid palette hues + IREE as a meaningful hatched variant (not directly comparable → the one
# sanctioned hatch). GOLD stays reserved for the emphasis callout on the CIRCT winner.
SERIES = [
    ("golden",               "golden (C lib)",          GOLD,  ""),    # goldenish
    ("agentic_raw_cpp",      "raw C++",                  MAUVE, ""),
    ("agentic_scaffold_cpp", "scaffold C++ (Merlin)",   SLATE, ""),
    ("agentic_python",       "Python-Merlin",           NAVY,  ""),
    ("agentic_circt",        "Merlin+CIRCT",            SAGE,  ""),    # green
    ("iree_dialect",         "IREE dialect *",          GOLD,  "///"),  # same hue as golden + diagonal hatch
]


def _macs():
    import yaml
    doc = yaml.safe_load((REPO / "experiments/gemmini_perf_bench/kernels/kernel_corpus.yaml").read_text())
    return {k["id"]: k["macs"] for sec in doc if isinstance(doc[sec], list) for k in doc[sec]}


def _load():
    ag = json.loads((PB_RUNS / AGENTIC_RUN / "firesim_arm_results.json").read_text())
    ca = json.loads((PB_RUNS / CACHE_RUN / "firesim_arm_results.json").read_text())
    merged = {}
    for k in set(ag) | set(ca):
        merged[k] = {}
        merged[k].update({a: v for a, v in (ca.get(k) or {}).items() if a in ("golden", "iree_dialect")})
        merged[k].update({a: v for a, v in (ag.get(k) or {}).items() if a.startswith("agentic_")})
    return merged


def main():
    use_merlin_style()
    macs = _macs()
    fs = _load()
    arms = [s[0] for s in SERIES]
    # keep kernels where every series has a cycles cell
    rows = []
    for k in sorted(fs):
        c = {a: (fs[k].get(a) or {}).get("cycles") for a in arms}
        if all(c.get(a) for a in arms):
            rows.append((k, c))
    if not rows:
        print("no kernels with all series present yet"); return 1
    n = len(arms)
    KNAME = {
        "G01_multitile_sq_64x64x64":      "matmul 64³",
        "G06_acc_scale_i8_64x64x64":      "acc-scale 64³",
        "G07_relu_i8_64x64x64":           "matmul+ReLU 64³",
        "G08_large_sq_128x128x128":       "large matmul 128³",
        "K_attn_pv_64x64x64":             "attn P·V 64",
        "K_attn_qk_64x64x64":             "attn Q·K 64",
        "K_attn_qk_128x64x128":           "attn Q·K 128",
        "M00_smolvla_model_16x32x960_i8": "SmolVLA 16×32×960",
        "M01_smolvla_model_64x720x32_i8": "SmolVLA 64×720×32",
        "M02_smolvla_model_64x32x720_i8": "SmolVLA 64×32×720",
        "M03_openvla_vla_32x256x128_i8":  "OpenVLA 32×256×128",
        "M04_openvla_vla_32x128x256_i8":  "OpenVLA 32×128×256",
    }
    # Median appears as its OWN group at the end (like a 13th "kernel"), separated by a small gap.
    labels = [KNAME.get(k, k.split("_", 1)[0]) for k, _ in rows] + ["Median"]
    x = np.append(np.arange(len(rows)), len(rows) + 0.6)   # gap before the Median group
    w = 0.82 / n

    fig, (axc, axu) = plt.subplots(1, 2, figsize=(max(19, (len(rows) + 1) * 1.55), 7.8))
    for ax in (axc, axu):
        style_ax(ax)

    for i, (key, lab, col, hatch) in enumerate(SERIES):
        off = (i - (n - 1) / 2) * w
        cyc_r = np.array([c[key] for _, c in rows], float)
        util_r = np.array([100.0 * macs.get(k, 0) / (c[key] * 256) if c[key] else np.nan
                           for k, c in rows], float)
        # append the across-kernel MEDIAN as the final group
        cyc = np.append(cyc_r, np.median(cyc_r))
        util = np.append(util_r, np.median(util_r))
        # cycles (log) — block shadow per bar
        bc = axc.bar(x + off, cyc, w, color=col, edgecolor=INK, linewidth=1.0,
                     zorder=3, hatch=(hatch or None), label=lab)
        bu = axu.bar(x + off, util, w, color=col, edgecolor=INK, linewidth=1.0,
                     zorder=3, hatch=(hatch or None))
        for p in list(bc.patches) + list(bu.patches):
            block_shadow(ax=p.axes, x=p.get_x(), y=p.get_y(),
                         w=p.get_width(), h=p.get_height(), dx=2.2, dy=-2.2, z=2.2)

    # left: cycles, log scale, floor from data
    axc.set_yscale("log")
    allcyc = [c[a] for _, c in rows for a in arms]
    axc.set_ylim(0.7 * min(allcyc), 1.5 * max(allcyc))
    axc.set_ylabel("cycles (FireSim L5, log)", fontsize=18)
    title(axc, "Cycles — lower is better", fs=20)
    _handles, _labels = axc.get_legend_handles_labels()   # placed as a figure legend under the title

    # right: utilization, data-scaled
    allutil = [100.0 * macs.get(k, 0) / (c[a] * 256) for k, c in rows for a in arms if c[a]]
    axu.set_ylim(0, max(allutil) * 1.18)
    axu.set_ylabel("PE-array utilization (%)", fontsize=18)
    title(axu, "Utilization — higher is better", fs=20)

    # shared x-axis cosmetics: rotated single-line labels (no overlap) + a divider before "Median"
    div = (x[-2] + x[-1]) / 2.0
    for ax in (axc, axu):
        ax.set_xlim(-0.6, x[-1] + 0.6)
        ax.axvline(div, color=INK, ls=(0, (2, 3)), lw=1.0, alpha=0.35, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=16)
        ax.get_xticklabels()[-1].set_fontweight("bold")   # emphasise the Median group
        ax.tick_params(axis="y", labelsize=14)            # bigger numeric y-ticks for slides

    suptitle(fig, "FireSim L5 — the 4 agentic backends vs golden & IREE on shared kernels", y=1.00, fs=24)
    # legend just under the title, fully flattened to ONE row (ncol = all series)
    fig.legend(_handles, _labels, loc="upper center", ncol=len(_labels), fontsize=16,
               frameon=True, facecolor="white", edgecolor="#d9cfc0",
               bbox_to_anchor=(0.5, 0.955), columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    REPORTS.mkdir(exist_ok=True)
    png = REPORTS / "fig_4backend_profile.png"
    svg = REPORTS / "fig_4backend_profile.svg"
    fig.savefig(png, bbox_inches="tight", dpi=400, facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", facecolor=BG)
    print(f"wrote {png}\nwrote {svg}  (N={len(rows)} kernels)")
    # quick numeric digest
    for key, lab, *_ in SERIES:
        med = np.median([c[key] for _, c in rows])
        print(f"  {lab:26s} median cycles={med:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
