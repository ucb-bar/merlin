"""Figure suite for the RTL-derived checks + arcilator middle-tier work (styled via perf_style).

Figures (reports/fig_arc_*.png):
  landscape   — speed vs fidelity: where spike / static-checks / arc / verilator / FireSim sit (HERO)
  speed       — per-kernel wall time: arc middle-tier (est) vs verilator (boot-dominated), log
  bitexact    — corpus 20/20 bit-exact grid + per-capsule arc cycles
  latency     — arc cycles vs modelled memory latency (timing realism / ideal-mem knob)
  checks      — RTL-derived static checks vs the oracle on 383 real runs (confusion + recall lift)
  mutation    — pre-screen catches RTL failures in ms vs verilator seconds (iteration saving)
  hostcomm    — host<->accelerator telemetry (RoCC control + DMA bytes) the SoC shell hides
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import _pbcommon as PB
import perf_style as S

RF = PB.REPO / "merlin/targets/gemmini/contracts/rtl_facts"
ARC = json.loads((RF / "arc_results.json").read_text())
# MEASURED wall references (no hardcoded estimates): arc per-capsule wall_s from measure_arc_wall.py;
# verilator median per-kernel wall + firesim per-run from arc_results.rtl_wall_ref.
_aw = [c["wall_s"] for c in ARC["capsules"] if c.get("wall_s")]
ARC_WALL_MED = sorted(_aw)[len(_aw) // 2] if _aw else None
_ref = ARC.get("rtl_wall_ref", {})
VERILATOR_WALL = _ref.get("verilator_wall_s_median")     # measured, median over perf kernels
FIRESIM_WALL = _ref.get("firesim_per_run_s_typ")          # measured per-run machinery
import numpy as _np  # noqa


def _save(fig, name):
    out = PB.REPORTS / f"fig_arc_{name}.png"
    S.save_fig(fig, out)
    print(f"wrote {out}")


def fig_landscape():
    # speed (x, log; throughput = 1/measured-wall, higher=faster) vs fidelity (y). MEASURED where possible.
    arc_thru = 1.0 / ARC_WALL_MED if ARC_WALL_MED else 2.7e2
    veri_thru = 1.0 / VERILATOR_WALL if VERILATOR_WALL else 1.5e-3
    fsim_thru = 1.0 / FIRESIM_WALL if FIRESIM_WALL else 5e-3
    pts = [  # name, throughput(kernels/s), fidelity, color, note, (label dx,dy), (note dx,dy)
        ("spike (functional)", 1e3, 1.0, S.COLOR["baseline"], "fast, NOT faithful (cyc plateau ~120)", (0, 26), (0, -26)),
        ("static RTL-checks", 3e2, 2.0, S.COLOR["iree_dialect"], "~3 ms; structural only (no numerics)", (0, 26), (0, -26)),
        ("arc middle-tier (this work)", arc_thru, 3.2, S.COLOR["merlin_targetgen"], f"RTL numerics+cycles · NO boot · {ARC_WALL_MED*1e3:.0f} ms median", (0, 28), (0, -28)),
        ("verilator (L3)", veri_thru, 3.7, S.COLOR["golden"], f"RTL-faithful, boot-dominated (~{VERILATOR_WALL:.0f} s median)", (78, 14), (78, -2)),
        ("FireSim (L5)", fsim_thru, 4.4, S.COLOR["merlin_native"], f"RTL + FPGA (~{FIRESIM_WALL:.0f} s/run)", (78, 14), (78, -2)),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axhspan(2.9, 4.7, color=S.COLOR["merlin_targetgen"], alpha=0.06)
    ax.text(2.5e6, 3.8, "RTL-faithful\nband", ha="right", va="center", fontsize=8.5, color="#9a8f78", style="italic")
    for name, spd, fid, col, note, (lx, ly), (nx, ny) in pts:
        ax.scatter([spd], [fid], s=520, color=col, edgecolor=S.INK, lw=1.6, zorder=5)
        ax.annotate(name, (spd, fid), xytext=(lx, ly), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold", zorder=6)
        ax.annotate(note, (spd, fid), xytext=(nx, ny), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="#5A5A5A")
    # arc -> verilator speedup arrow (both RTL-faithful, arc far faster). ratio is MEASURED.
    ratio = (arc_thru / veri_thru) if (ARC_WALL_MED and VERILATOR_WALL) else 1e5
    ax.annotate("", xy=(arc_thru, 3.25), xytext=(veri_thru, 3.6),
                arrowprops=dict(arrowstyle="->", color=S.COLOR["merlin_targetgen"], lw=2, alpha=.8))
    ax.text(arc_thru ** 0.5 * veri_thru ** 0.5, 3.66, f"~{ratio:,.0f}× faster\nsame RTL fidelity",
            ha="center", fontsize=8.5, fontweight="bold", color=S.COLOR["merlin_targetgen"])
    ax.set_xscale("log")
    ax.set_xlabel("← slower            throughput = 1 / measured wall  (kernels/s, log)            faster →", fontsize=10)
    ax.set_yticks([1.0, 2.0, 3.2, 3.7, 4.4])
    ax.set_yticklabels(["functional\nnumerics", "structural\n(ISA-legal)", "RTL numerics\n+ cycles",
                        "RTL (full)", "RTL + FPGA"], fontsize=8)
    ax.set_ylabel("fidelity  →", fontsize=10)
    ax.set_ylim(0.3, 4.9); ax.set_xlim(min(veri_thru, fsim_thru) / 3, 1e4)
    ax.set_title("Where each Gemmini oracle sits — the arc middle-tier fills the spike↔verilator gap", pad=12)
    S.caption(fig, f"MEASURED: arc wall median {ARC_WALL_MED*1e3:.1f} ms (20 capsules); verilator wall "
              f"median {VERILATOR_WALL:.0f} s ({_ref.get('verilator_wall_s_n','?')} perf-kernel runs); "
              f"FireSim {FIRESIM_WALL:.0f} s/run. spike/static-checks placed qualitatively. Fidelity axis "
              f"is ordinal. Single run per point.")
    _save(fig, "landscape")


def fig_speed():
    # MEASURED arc wall per capsule (bars) + measured RTL-sim reference lines. arc & RTL are different
    # corpora, so RTL is shown as a measured reference band/line (median), not falsely paired per kernel.
    caps = [c for c in ARC["capsules"] if c.get("wall_s")]
    names = [c["capsule"].split("_")[0] for c in caps]
    arc_wall = [c["wall_s"] for c in caps]
    x = np.arange(len(caps))
    fig, ax = plt.subplots(figsize=(max(9, len(caps) * 0.5), 5.0))
    bars = ax.bar(x, arc_wall, 0.62, label="arc middle-tier (measured)", color=S.COLOR["merlin_targetgen"], edgecolor=S.INK, lw=0.8)
    if VERILATOR_WALL:
        ax.axhline(VERILATOR_WALL, color=S.COLOR["golden"], lw=2, ls="--", label=f"verilator median (measured, {VERILATOR_WALL:.0f} s)")
    if FIRESIM_WALL:
        ax.axhline(FIRESIM_WALL, color=S.COLOR["merlin_native"], lw=2, ls=":", label=f"FireSim per-run (measured, {FIRESIM_WALL:.0f} s)")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("wall time per run (s, log)")
    ax.set_title("Arc middle-tier wall time (measured) vs RTL-sim references — no SoC boot", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=8)
    ratio = VERILATOR_WALL / ARC_WALL_MED if (VERILATOR_WALL and ARC_WALL_MED) else None
    if ratio:
        S.badge(ax, len(caps) * 0.5, (ARC_WALL_MED * VERILATOR_WALL) ** 0.5,
                f"~{ratio:,.0f}× faster\nthan verilator", color=S.COLOR["merlin_targetgen"], fontsize=9)
    S.caption(fig, f"arc wall = MEASURED min-of-5 per capsule (isolated @Gemmini, ideal memory). verilator/"
              f"FireSim = MEASURED wall references (median / per-run); different corpora, so shown as "
              f"reference lines not per-kernel pairs. arc bit-exact on all 20 (see bit-exact figure).")
    _save(fig, "speed")


def fig_bitexact():
    caps = ARC["capsules"]; n = len(caps)
    fig, (axg, axc) = plt.subplots(1, 2, figsize=(12, max(4.5, n * 0.3)), gridspec_kw={"width_ratios": [1, 1.4]})
    # left: bit-exact grid
    axg.set_xlim(0, 1); axg.set_ylim(-1, n); axg.invert_yaxis(); axg.axis("off")
    for i, c in enumerate(caps):
        fc = S.HEAT_GOOD if c["bitexact"] else S.HEAT_BAD
        axg.add_patch(plt.Rectangle((0.04, i + 0.08), 0.5, 0.84, fc=fc, ec="white", lw=1.5))
        axg.text(0.29, i + 0.5, "✓" if c["bitexact"] else "✗", ha="center", va="center", color="white", fontweight="bold")
        axg.text(0.58, i + 0.5, c["capsule"][:24], ha="left", va="center", fontsize=7.5)
    ok = sum(c["bitexact"] for c in caps)
    axg.set_title(f"arc vs golden: {ok}/{n} BIT-EXACT", fontsize=11, fontweight="bold", pad=8)
    # right: cycles bar
    cc = [c for c in caps if c["cycles"]]
    axc.barh(range(len(cc)), [c["cycles"] for c in cc], color=S.COLOR["merlin_targetgen"], edgecolor=S.INK, lw=0.8)
    axc.set_yticks(range(len(cc))); axc.set_yticklabels([c["capsule"][:22] for c in cc], fontsize=7)
    axc.invert_yaxis(); axc.set_xlabel("arc cycles (accelerator, ideal memory)")
    axc.set_title("per-capsule arc cycle count", fontsize=10, pad=8)
    S.caption(fig, "20 bench capsules · arc output vs exact-int golden · isolated @Gemmini, ideal memory · single run. ✓ = bit-exact (0 mismatches).")
    _save(fig, "bitexact")


def fig_latency():
    sw = ARC["latency_sweep"]
    L = [s["latency"] for s in sw]; C = [s["cycles"] for s in sw]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(L, C, "-o", color=S.COLOR["merlin_targetgen"], markersize=9, markeredgecolor=S.INK, lw=2)
    for x, y in zip(L, C):
        ax.annotate(str(y), (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("modelled memory read latency (cycles)"); ax.set_ylabel("A2 matmul cycles")
    ax.set_title("Timing realism: arc cycles scale with memory latency (knob closes the ideal-mem gap)", pad=12)
    S.badge(ax, L[0], C[0], "latency 0 = ideal mem\n(238 cyc, bit-exact at every point)", color=S.COLOR["merlin_targetgen"], fontsize=8)
    S.caption(fig, "A2 single-tile matmul only (illustrative); x = simulator memory-read latency (an arc knob, not measured DRAM). Output bit-exact at every latency.")
    _save(fig, "latency")


def fig_checks():
    # confusion matrix + recall lift
    fig, (axm, axr) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    conf = [["TN 242", "FP 0"], ["FN 17", "TP 124"]]
    cols = [[S.HEAT_GOOD, S.HEAT_BAD], ["#E8E2D4", S.HEAT_GOOD]]
    axm.set_xlim(0, 2); axm.set_ylim(0, 2); axm.invert_yaxis(); axm.axis("off")
    for r in range(2):
        for c in range(2):
            axm.add_patch(plt.Rectangle((c + .03, r + .03, ), .94, .94, fc=cols[r][c], ec="white", lw=2))
            axm.text(c + .5, r + .5, conf[r][c], ha="center", va="center", fontsize=13, fontweight="bold", color="white" if cols[r][c] != "#E8E2D4" else S.INK)
    axm.text(1, -0.12, "check verdict →  (ok / reject)", ha="center", fontsize=8, color="#5A5A5A")
    axm.text(-0.12, 1, "oracle  (pass / fail)", rotation=90, va="center", ha="center", fontsize=8, color="#5A5A5A")
    axm.set_title("RTL-derived checks vs oracle\n383 real agent runs — 0 false positives", fontsize=10.5, fontweight="bold", pad=10)
    # recall lift bars
    axr.bar(["base checks", "+ fence_bracket\n+preload/config"], [91 / 141 * 100, 124 / 141 * 100],
            color=[S.COLOR["baseline"], S.COLOR["merlin_targetgen"]], edgecolor=S.INK, lw=1)
    for i, v in enumerate([91, 124]):
        axr.text(i, v / 141 * 100 + 1.5, f"{v}/141\n{v/141*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
    axr.set_ylim(0, 100); axr.set_ylabel("recall on real RTL-tier-class failures (%)")
    axr.set_title("recall lift from added general invariants\n(FP stays 0)", fontsize=10, pad=10)
    S.caption(fig, "383 REAL agent-run traces (biased sample, not random). 0/242 false positives. FN=17 = 9 numerical functional_mismatch (out of static-check scope by design) + 8 other (tool-crash / mode-specific).")
    _save(fig, "checks")


def fig_mutation():
    demo = json.loads((PB.REPORTS / "prescreen_mutation_demo.json").read_text())
    rows = [r for r in demo["rows"]]
    names = [r["mutant"].replace("_", "\n", 1)[:22] for r in rows]
    vs = [float(r["verilator_s"]) for r in rows]
    ms = [r["prescreen_ms"] / 1000.0 for r in rows]
    x = np.arange(len(rows)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, vs, w, label="verilator (s)", color=S.COLOR["golden"], edgecolor=S.INK, lw=0.8)
    ax.bar(x + w / 2, ms, w, label="pre-screen (s)", color=S.COLOR["merlin_targetgen"], edgecolor=S.INK, lw=0.8)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("time to verdict (s, log)")
    ax.set_title("Pre-screen catches RTL failures in ~ms vs verilator's ~minutes (0 false-positive on 'original')", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=9)
    for i, r in enumerate(rows):
        tag = "caught" if r["caught_before_rtl"] else ("pass" if r["verilator"] == "pass" else "MISS\n(numeric)")
        ax.text(i, max(vs) * 1.3, tag, ha="center", fontsize=7, color="#5A5A5A")
    S.caption(fig, "5 mutants on ONE capsule (A2), illustrative not statistical. verilator wall is boot-dominated + noisy (110-198 s). pre-screen ms is the static-check time. 1 numerical mutant missed by design.")
    _save(fig, "mutation")


def fig_hostcomm():
    hc = ARC.get("hostcomm", {})
    if not hc:
        return
    caps = list(hc)
    metrics = [("rocc cmds", "cmds", 1), ("mvin KB", "mvin_B", 1 / 1024), ("mvout KB", "mvout_B", 1 / 1024), ("busy %", "busy_pct", 1)]
    x = np.arange(len(metrics)); w = 0.8 / max(len(caps), 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for j, c in enumerate(caps):
        vals = [hc[c].get(k, 0) * s for _, k, s in metrics]
        b = ax.bar(x + (j - (len(caps) - 1) / 2) * w, vals, w, label=c.split("_")[0],
                   color=[S.COLOR["golden"], S.COLOR["merlin_targetgen"]][j % 2], edgecolor=S.INK, lw=0.8)
        S.bar_labels(ax, b, fmt="{:.0f}", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics], fontsize=8.5)
    ax.set_ylabel("value"); ax.legend(fontsize=9, title="capsule")
    ax.set_title("Host↔accelerator telemetry the arc tier exposes (RoCC control + DMA traffic + utilization)", pad=12)
    S.caption(fig, "Measured from the arc harness on 2 capsules (A2 single-tile, C0 MLP). Bytes/cmds are exact; busy% = accelerator-active cycles / total.")
    _save(fig, "hostcomm")


def main():
    S.use_style()
    for f in (fig_landscape, fig_speed, fig_bitexact, fig_latency, fig_checks, fig_mutation, fig_hostcomm):
        try:
            f()
        except Exception as e:
            print(f"  (skip {f.__name__}: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
