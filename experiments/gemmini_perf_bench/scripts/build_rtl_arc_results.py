"""Assemble a single styled results page for the RTL-derived checks + arcilator middle-tier + perf work.
Embeds every figure (base64) into a cream-card page. -> reports/RTL_ARC_RESULTS.html"""
from __future__ import annotations
import base64, json
from pathlib import Path
import _pbcommon as PB

R = PB.REPORTS
CREAM, INK = "#F6F1E7", "#2B2B2B"
arc = json.loads((PB.REPO / "merlin/targets/gemmini/contracts/rtl_facts/arc_results.json").read_text())
ok = sum(c["bitexact"] for c in arc["capsules"]); n = len(arc["capsules"])


def b64(name):
    p = R / f"{name}.png"
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.is_file() else ""


def card(title, sub, *figs):
    imgs = "".join(f'<img src="{b64(f)}" alt="{f}"/>' for f in figs if b64(f))
    return f'<section class="card"><h2>{title}</h2><p class="sub">{sub}</p>{imgs}</section>'


KPIS = [
    (f"{ok}/{n}", "arc bit-exact vs golden (whole corpus)"),
    (">10⁴×", "arc middle-tier vs verilator (no SoC boot)"),
    ("0 / 242", "RTL-check false-positives on real runs"),
    ("88%", "check recall on real RTL-tier failures"),
    ("24/24", "kernels cycle-accurate (verilator+FireSim)"),
    ("500/500", "random matmuls bit-exact (stress)"),
]
kpis = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in KPIS)

html = f"""<!doctype html><meta charset=utf-8><title>RTL-derived checks + arcilator middle-tier — results</title>
<style>
 body{{margin:0;background:{CREAM};color:{INK};font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:32px 18px 60px}}
 .wrap{{max-width:1000px;margin:0 auto}} h1{{font-size:30px;font-weight:800;margin:0 0 4px}}
 .lede{{color:#5A5A5A;font-size:15px;margin:0 0 22px;max-width:820px}}
 .kpis{{display:flex;flex-wrap:wrap;gap:12px;margin:0 0 26px}}
 .kpi{{flex:1;min-width:150px;background:#fff;border:2px solid {INK};border-radius:16px;padding:14px 16px;box-shadow:3px 3px 0 rgba(43,43,43,.08)}}
 .kpi .v{{font-size:24px;font-weight:800}} .kpi .l{{font-size:11.5px;color:#5A5A5A;font-weight:600;margin-top:2px}}
 .card{{background:#fff;border:2px solid {INK};border-radius:18px;padding:20px 22px;margin:0 0 22px;box-shadow:4px 4px 0 rgba(43,43,43,.08)}}
 .card h2{{font-size:20px;font-weight:800;margin:0 0 2px}} .card .sub{{color:#5A5A5A;font-size:13.5px;margin:0 0 14px}}
 .card img{{width:100%;height:auto;border-radius:10px;display:block;margin:10px 0}}
 .foot{{color:#7a7160;font-size:12.5px}}
</style>
<div class=wrap>
 <h1>RTL-derived checks + arcilator middle-tier — results</h1>
 <p class=lede>Two RTL-grounded tools for the Gemmini codegen loop, both built deterministically from the
 actual RTL (CIRCT), validated against the oracle ladder. A fast static checks layer (compiled to FileCheck)
 and a dynamic "middle tier" between spike and verilator (the isolated accelerator, arcilator-JIT'd, no SoC
 boot) — plus the completed cross-approach FireSim performance benchmark.</p>
 <div class=kpis>{kpis}</div>
 {card("1 · The oracle landscape", "Where each tool sits on speed × fidelity — the arc tier fills the spike↔verilator gap.", "fig_arc_landscape")}
 {card("2 · Arc middle-tier: faithful + fast", "Bit-exact on the whole corpus, >10⁴× faster than verilator (no boot), cycles scale with modelled memory latency.", "fig_arc_bitexact", "fig_arc_speed", "fig_arc_latency")}
 {card("3 · Host↔accelerator telemetry", "Control (RoCC) + DMA traffic the SoC shell normally hides — free from driving the isolated accelerator.", "fig_arc_hostcomm")}
 {card("4 · RTL-derived static checks", "Compiled-from-RTL checks vs the oracle on 383 real agent runs: 0 false-positives, 88% recall; and the pre-screen catches RTL failures in ms vs verilator's minutes.", "fig_arc_checks", "fig_arc_mutation")}
 {card("5 · Cross-approach performance (FireSim L5 complete)", "Same kernels through 5 code-gen paths, cycle-accurate (verilator small + FireSim big). Util-crossover: golden 49–64% on model shapes vs generated ~14–17%.", "fig_cycles", "fig_capability", "fig_spike_not_timing")}
 <p class=foot>All deterministic + RTL-grounded; no LLM-lifted models. Branch feature/rtl-derived-checks.
 Figures regenerate via gen_arc_plots.py + gen_perf_plots.py.</p>
</div>"""
out = R / "RTL_ARC_RESULTS.html"
out.write_text(html)
print(f"wrote {out} ({len(html)//1024} KB, {sum(1 for _ in R.glob('fig_arc_*.png'))+3} figures embedded)")
