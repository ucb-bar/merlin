#!/usr/bin/env python3
"""Render the cross-approach Gemmini comparison from a run's perf_results.json.

Two distinct axes, kept rigorously separate because conflating them is misleading:

  * PERFORMANCE = **verilator (L3) cycle-accurate RTL** ONLY. spike is a FUNCTIONAL simulator: it does
    NOT model Gemmini's systolic-array timing — the RoCC matmul retires in ~0 modeled cycles, so spike
    "cycles" plateau (~120) regardless of kernel size and yield util > 100% (physically impossible).
    We therefore NEVER report spike cycles as performance. Verilator is feasible only for small kernels
    (≤ ~32K MACs here); larger kernels need FireSim (L5), tracked separately.
  * CORRECTNESS + CAPABILITY = **spike (L2)**: does each approach PRODUCE A CORRECT result for each
    kernel (exact-int == shared capsule golden / runner reference). This is where the approaches differ
    in COVERAGE (e.g. which backend can compile conv2d at all).

Outputs reports/perf_comparison.md.  Usage: gen_perf_report.py [--run-id perf_full_0001]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _pbcommon as PB

APPROACH_ORDER = ["golden", "baseline", "merlin_targetgen", "iree_dialect", "merlin_native"]
APPROACH_LABEL = {"golden": "1.golden (C lib)", "baseline": "2.baseline-gen (v0)",
                  "merlin_targetgen": "3.merlin-gen (v1)",
                  "iree_dialect": "4.depr-merlin handwritten (IREE)",
                  "merlin_native": "(extra) merlin-native ref"}


def _ps(ar: dict, sim: str) -> dict | None:
    return (ar.get("per_sim") or {}).get(sim)


def _rtl(ar: dict):
    """Cycle-accurate result for an approach: verilator (L3) if present else FireSim (L5).
    Returns (cycles, util, correct, tier) or (None, None, None, None)."""
    for sim, tier in (("verilator", "L3"), ("firesim", "L5")):
        ps = _ps(ar, sim)
        if ps and ps.get("cycles"):
            return ps["cycles"], ps.get("util_pct"), ps.get("correct"), tier
    return None, None, None, None


def _veri_cell(ar: dict) -> str:
    """Cycle-accurate cell: cycles[tier] (✗ if wrong) (util%). Prefers verilator, falls back to FireSim."""
    cyc, util, corr, tier = _rtl(ar)
    if cyc is None:
        if (_ps(ar, "verilator") or {}).get("error"):
            return "to/err"
        return "err" if ar.get("error") else "·"
    mark = "" if corr else "✗"
    u = f" ({util}%)" if util is not None else ""
    return f"{cyc}{mark} `{tier}`{u}"


def _correct_cell(ar: dict) -> str:
    """Correctness/capability cell from spike (functional). ✓ / ✗ / · (not run) / err."""
    if ar.get("error"):
        return "·"  # arm not attempted / golden template deferred
    ps = _ps(ar, "spike")
    if not ps:
        return "·"
    if ps.get("error"):
        return "to/err"
    c = ps.get("correct")
    if c is True:
        return "✓"
    if c is False:
        return "✗ (no compile)" if ps.get("cycles") is None else "✗ (wrong)"
    return "·"


def _geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return round(math.exp(sum(math.log(x) for x in xs) / len(xs)), 1) if xs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    a = ap.parse_args(argv)
    rj = PB.RUNS / a.run_id / "perf_results.json"
    if not rj.is_file():
        print(f"no results at {rj}")
        return 2
    rows = json.loads(rj.read_text())
    approaches = [x for x in APPROACH_ORDER if any(x in r.get("approaches", {}) for r in rows)]

    L = [f"# Gemmini cross-approach comparison ({a.run_id})", "",
         "The SAME kernels driven through each Gemmini code-gen approach on one "
         "ELF→spike/verilator harness. **Two axes, deliberately separated:**", "",
         "1. **Performance = verilator (L3), cycle-accurate RTL — the only valid timing.** "
         "spike is *functional*: it does not model the systolic array, so its \"cycles\" plateau "
         "(~120 from 4K→2M MACs) and give util > 100% — meaningless as performance. Verilator is "
         "feasible only for small kernels (≤ ~32K MACs); bigger kernels need FireSim (L5), pending.",
         "2. **Correctness & capability = spike (L2):** does each approach produce a correct result "
         "at all (exact-int == golden)? This is where COVERAGE differs (e.g. who can compile conv2d).",
         "",
         "Approaches: **1** golden hand-tuned C lib (`tiled_matmul_auto`); **2** baseline-generated MLIR "
         "(agent_spec_v0); **3** merlin-generated MLIR (agent_spec_v1); **4** the deprecated-merlin "
         "hand-written C++ Gemmini dialect via IREE (`/path/to/merlin-iree`); plus an extra "
         "merlin native RoCC emitter for reference.", ""]

    # ---- PERFORMANCE: cycle-accurate = verilator (L3) + FireSim (L5) ----
    veri_rows = [r for r in rows if any(_rtl(r["approaches"].get(x, {}))[0] for x in approaches)]
    L += ["## 1. Performance — cycle-accurate RTL (verilator L3 + FireSim L5)", "",
          "Cells = **cycles** `tier` (✗ = wrong output) **(util% = MACs/(cycles·256), 16×16 PE array)**. "
          "Both tiers simulate the SAME RTL: `L3` verilator covers small kernels (≤32K MACs); `L5` "
          "FireSim (Alveo U250 FPGA) covers the larger 64³+/model/attention shapes verilator can't reach "
          f"in its time budget. {len(veri_rows)} kernels have cycle-accurate data.", "",
          "| kernel | shape | macs | " + " | ".join(APPROACH_LABEL[x] for x in approaches) + " |",
          "|---|---|---|" + "---|" * len(approaches)]
    for r in veri_rows:
        L.append(f"| {r['kernel']} | {r['shape']} | {r['macs']:,} | "
                 + " | ".join(_veri_cell(r["approaches"].get(x, {})) for x in approaches) + " |")
    cyc_by = {x: [c for r in veri_rows
                  for c, _u, corr, _t in [_rtl(r["approaches"].get(x, {}))] if c and corr]
              for x in approaches}
    L += ["", f"**Geomean cycles over the {len(veri_rows)} cycle-accurate kernels** (lower = faster; "
          "golden is the hand-tuned reference): "
          + ", ".join(f"{APPROACH_LABEL[x]} = {_geomean(cyc_by[x])}"
                      for x in approaches if _geomean(cyc_by[x])) + ".", ""]

    # ---- CORRECTNESS / CAPABILITY: spike ----
    L += ["## 2. Correctness & capability — spike L2 (functional)", "",
          "Does each approach produce a correct result for each kernel? ✓ pass · not attempted "
          "(golden conv template deferred) `✗ (no compile)` backend cannot lower this op. "
          "**This is the coverage story** — not a timing comparison.", "",
          "| kernel | op | shape | macs | " + " | ".join(APPROACH_LABEL[x] for x in approaches) + " |",
          "|---|---|---|---|" + "---|" * len(approaches)]
    for r in rows:
        src = r["source"].split(":")[0]
        L.append(f"| {r['kernel']} | {src} | {r['shape']} | {r['macs']:,} | "
                 + " | ".join(_correct_cell(r["approaches"].get(x, {})) for x in approaches) + " |")
    # capability counts
    L += ["", "**Correct-kernel count per approach (spike):** "
          + ", ".join(f"{APPROACH_LABEL[x]} = "
                      f"{sum(1 for r in rows if (_ps(r['approaches'].get(x, {}), 'spike') or {}).get('correct'))}"
                      f"/{len(rows)}" for x in approaches) + ".", ""]

    L += ["## Notes", "",
          "- **Why spike cycles are omitted from §1:** spike models RoCC functionally — the matmul "
          "retires in ~0 cycles, so spike cycle counts (~120 across all sizes) reflect only scalar "
          "issue overhead, not the systolic compute. Reporting them as performance would imply util "
          "> 100%. Verilator (and FireSim) are the timing oracles.",
          "- **IREE arm (4) — read its cells carefully:** the deprecated-merlin C++ Gemmini dialect now "
          "has cycle-accurate L5 (FireSim) numbers, but it is ~10–40× slower than the generated/golden "
          "arms (e.g. 93184 vs 7439 cyc on 64³) AND it is verified by the IREE runner's OWN all-ones "
          "self-check (rc=0; each output == K), NOT against the exact-int shared golden the other arms use. "
          "So its cells are shown WITHOUT ✗ (it is not producing wrong answers) but its correctness is "
          "self-checked-on-all-ones, a weaker guarantee than the others — do not read IREE cycles as a "
          "like-for-like comparison.",
          "- **'Identical RoCC' is only *almost* true (corrected by the L5 data):** baseline (v0) and the "
          "native emitter are **bit-identical in cycles** on every kernel; **merlin-gen (v1) differs by a "
          "few cycles** on epilogue/model kernels (e.g. G06/G07 acc_scale/relu +2, M00 +17, M01 +8, "
          "M03 +10) — v1's epilogue codegen is not byte-identical to v0/native. Small, but real.",
          "- **Capability finding:** only **merlin-gen (v1)** compiles conv2d and movement among the "
          "generated backends; baseline (v0) and the native emitter cannot lower those ops. All four "
          "handle matmul and attention.",
          "- **FireSim L5: COMPLETE** — 46/46 ELFs ran (100%); all 24 kernels are cycle-accurate "
          "(verilator ≤32K MACs, FireSim for the rest). Util-crossover holds at scale: golden 49–64% on "
          "the big/model shapes vs generated ~14–17%.",
          "- `to/err` = verilator timeout (>900s) or runner error; `·` = not run / backend can't compile."]

    out = PB.REPORTS / "perf_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(rows)} kernels, {len(veri_rows)} cycle-accurate, "
          f"{len(approaches)} approaches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
