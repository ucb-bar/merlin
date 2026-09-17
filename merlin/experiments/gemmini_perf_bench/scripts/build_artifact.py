#!/usr/bin/env python3
"""Build a styled HTML poster of the Gemmini perf-bench results (image-copy-7 cream-card aesthetic).

Embeds the matplotlib figures as base64 data URIs (CSP-safe) into a parchment-themed page with the
consistent method->colour legend, headline numbers, and the two-axis story (performance vs capability).
Reads live numbers from perf_results.json so the callouts stay honest as data fills in.

Usage: build_artifact.py [--run-id perf_full_0001]  ->  writes reports/perf_poster.html
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import _pbcommon as PB
import perf_reporting as PR

CREAM = "#F6F1E7"; INK = "#2B2B2B"
COL = {"golden": "#6E93B0", "baseline": "#D98C84", "merlin_targetgen": "#E6B84C",
       "iree_dialect": "#9DB682", "merlin_native": "#C9A86B"}


def _b64(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.is_file() else ""


def _rtl(ar):
    for sim in ("verilator", "firesim"):
        ps = (ar.get("per_sim") or {}).get(sim) or {}
        if ps.get("cycles"):
            return ps["cycles"], ps.get("correct")
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    PR.refuse_legacy_cross_approach(run, "build_artifact.py")
    rows = json.loads((run / "perf_results.json").read_text())
    R = PB.REPORTS

    # headline numbers
    pair = []
    for r in rows:
        g, gc = _rtl(r["approaches"].get("golden", {}))
        m, mc = _rtl(r["approaches"].get("merlin_targetgen", {}))
        if g and m and gc and mc:
            pair.append(m / g)
    gm = math.exp(sum(map(math.log, pair)) / len(pair)) if pair else None
    n_rtl = sum(1 for r in rows if any(_rtl(r["approaches"].get(x, {}))[0]
                for x in ("golden", "baseline", "merlin_targetgen", "iree_dialect", "merlin_native")))
    cap = {x: sum(1 for r in rows if ((r["approaches"].get(x, {}).get("per_sim") or {})
                  .get("spike") or {}).get("correct"))
           for x in ("golden", "baseline", "merlin_targetgen", "iree_dialect", "merlin_native")}
    n_fs = sum(1 for r in rows for x in r["approaches"].values()
               if ((x.get("per_sim") or {}).get("firesim") or {}).get("cycles"))

    figs = {k: _b64(R / f"fig_{k}.png") for k in ("cycles", "capability", "spike_not_timing")}

    def card(title, sub, body):
        return f"""<section class="card">
          <h2>{title}</h2><p class="sub">{sub}</p>{body}</section>"""

    def img(key):
        return f'<img src="{figs[key]}" alt="{key}"/>' if figs[key] else \
               '<p class="todo">figure pending</p>'

    legend = "".join(
        f'<span class="chip"><i style="background:{COL[k]}"></i>{lab}</span>'
        for k, lab in [("golden", "golden (C lib)"), ("baseline", "baseline-gen v0"),
                       ("merlin_targetgen", "merlin-gen v1 · ours"),
                       ("iree_dialect", "IREE dialect (depr.)"), ("merlin_native", "merlin-native")])

    hl = (f"{1/gm:.1f}× faster" if gm and gm < 1 else (f"{gm:.1f}× cycles" if gm else "—"))

    html = f"""<title>Gemmini cross-approach performance</title>
<style>
  body{{margin:0;background:{CREAM};color:{INK};
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased;padding:32px 18px 60px}}
  .wrap{{max-width:980px;margin:0 auto}}
  h1{{font-size:30px;font-weight:800;margin:0 0 4px;letter-spacing:-.5px}}
  .lede{{color:#5A5A5A;font-size:15px;margin:0 0 22px;max-width:760px}}
  .legend{{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 26px}}
  .chip{{display:flex;align-items:center;gap:7px;font-size:13px;font-weight:600}}
  .chip i{{width:16px;height:16px;border:1.5px solid {INK};border-radius:4px;display:inline-block}}
  .kpis{{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 26px}}
  .kpi{{flex:1;min-width:150px;background:#fff;border:2px solid {INK};border-radius:16px;
    padding:14px 16px;box-shadow:3px 3px 0 rgba(43,43,43,.08)}}
  .kpi .v{{font-size:26px;font-weight:800}}
  .kpi .l{{font-size:12px;color:#5A5A5A;font-weight:600;margin-top:2px}}
  .card{{background:#fff;border:2px solid {INK};border-radius:18px;padding:20px 22px;margin:0 0 22px;
    box-shadow:4px 4px 0 rgba(43,43,43,.08)}}
  .card h2{{font-size:20px;font-weight:800;margin:0 0 2px}}
  .card .sub{{color:#5A5A5A;font-size:13.5px;margin:0 0 14px}}
  .card img{{width:100%;height:auto;border-radius:10px;display:block}}
  .todo{{color:#9a8f78;font-style:italic;padding:30px;text-align:center;border:2px dashed #cdc3ad;
    border-radius:10px}}
  .foot{{color:#7a7160;font-size:12.5px;margin-top:8px}}
  code{{background:#efe9da;padding:1px 5px;border-radius:4px;font-size:12.5px}}
</style>
<div class="wrap">
  <h1>Gemmini cross-approach performance</h1>
  <p class="lede">The same kernels driven through four Gemmini code-gen paths — hand-tuned C library,
  two generated MLIR backends, and the deprecated hand-written IREE dialect — measured on one
  ELF→simulator harness. Two axes, kept separate: <b>cycle-accurate performance</b> (verilator + FPGA)
  and <b>correctness/capability</b> (which backend compiles each op at all).</p>
  <div class="legend">{legend}</div>
  <div class="kpis">
    <div class="kpi"><div class="v">{hl}</div><div class="l">generated vs golden (geomean, cycle-accurate)</div></div>
    <div class="kpi"><div class="v">{cap['merlin_targetgen']}/{len(rows)}</div><div class="l">merlin-gen kernels correct (only backend w/ conv+movement)</div></div>
    <div class="kpi"><div class="v">{n_rtl}</div><div class="l">kernels with cycle-accurate data ({n_fs} via FireSim FPGA)</div></div>
  </div>
  {card("1 · Cycle-accurate performance",
        "Cycles per kernel — generated (gold) vs hand-tuned golden (steel). verilator for small "
        "kernels, FireSim FPGA for the larger model/attention shapes (same RTL).", img("cycles"))}
  {card("2 · Correctness &amp; capability",
        "Who actually compiles &amp; runs each kernel (spike). The conv2d/movement rows are the "
        "differentiator: only merlin-gen (v1) lowers them.", img("capability"))}
  {card("3 · Why spike is not a performance number",
        "spike is functional — its cycle count plateaus (~120) regardless of kernel size, giving "
        "impossible &gt;100% utilization. Only RTL (verilator/FireSim) is valid timing.",
        img("spike_not_timing"))}
  <p class="foot">Capability (spike, correct/total): golden {cap['golden']} · baseline {cap['baseline']}
   · merlin-gen {cap['merlin_targetgen']} · IREE {cap['iree_dialect']} · native {cap['merlin_native']}
   (of {len(rows)}). Generated backends (v0/v1/native) emit identical RoCC for matmul/attention →
   identical cycles; merlin-gen shown as representative.</p>
</div>"""
    out = R / "perf_poster.html"
    out.write_text(html)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
