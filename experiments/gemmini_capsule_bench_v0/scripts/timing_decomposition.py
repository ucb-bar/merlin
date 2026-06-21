#!/usr/bin/env python3
"""DETAILED timing decomposition for abc4 — exact per-tool/per-tier wall, not coarse gaps.

Two budgets, kept separate (they were conflated before):
  (1) AGENT ACTIVE SESSION  — what the agent spent reaching a working dialect (excl. rate-limit sleep):
        think+generate (LLM)  = sum(result.duration_api_ms)
        + in-session tool exec, of which the sims are measured EXACTLY from capsule_result tier timing.
  (2) OPERATOR SIM COST      — the verilator/VCS barrier + audits (separate processes, NOT in agent time):
        exact per-tier sim_active_s / build_s harvested from EVERY capsule_result.json (per round+capsule)
        + the full-suite audit timing_rollup. This is the expensive part (verilator ~222 s/capsule).
  CIRCT screen wall is measured live (rtl_check_runner.prescreen wasn't timed in-run — instrumented here).
-> reports/abc4_analysis/timing_detailed.json
"""
from __future__ import annotations
import json, glob, sys, time
from pathlib import Path

EXP = Path("/scratch/agustin/projects/oscar-merlin/experiments/gemmini_capsule_bench_v0")
OUT = EXP / "reports" / "abc4_analysis"
sys.path.insert(0, "/scratch/agustin/projects/oscar-merlin/merlin/python")
from merlin.targetgen import rtl_check_runner as RCR
import yaml

ARMS = {"rb_abc4": ("raw_baseline", "baseline-C++"),
        "merlin_abc4": ("merlin_assisted", "merlin-xDSL"),
        "merlincirct_abc4": ("merlin_assisted", "merlin+CIRCT")}
TIER_TOOL = {"L2": "spike", "L3": "verilator/VCS", "L4": "verilator/VCS"}


def harvest(rid: str) -> dict:
    sd = ARMS[rid][0]
    d = EXP / "runs" / sd / rid
    st = yaml.safe_load((d / "qa_loop_state.yaml").read_text()) if (d / "qa_loop_state.yaml").is_file() else {}
    active = (st.get("cumulative") or {}).get("active_wall_s", 0.0)

    # think+generate from result events
    api_ms = total_ms = 0
    for tp in glob.glob(f"{d}/rounds/round_*.transcript.jsonl"):
        for ln in Path(tp).read_text(errors="ignore").splitlines():
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("type") == "result":
                api_ms += o.get("duration_api_ms", 0) or 0
                total_ms += o.get("duration_ms", 0) or 0

    # EXACT per-tier sim timing, summed over every per-round per-capsule grade (the agent-loop sims)
    sims = {"spike": {"runs": 0, "build_s": 0.0, "sim_s": 0.0},
            "verilator/VCS": {"runs": 0, "build_s": 0.0, "sim_s": 0.0}}
    circt_dirs = []
    for cr in (d / "_qa_work").glob("runs_*/runs/gemmini-capsule-bench/*/capsule_result.json"):
        try:
            r = json.loads(cr.read_text())
        except Exception:
            continue
        for tier, tv in (r.get("tiers") or {}).items():
            tm = tv.get("timing") or {}
            tool = TIER_TOOL.get(tier)
            if tool and tm:
                sims[tool]["runs"] += 1
                sims[tool]["build_s"] += tm.get("build_s") or 0.0
                sims[tool]["sim_s"] += tm.get("sim_active_s") or 0.0
        if (cr.parent / "generated" / "instruction_trace.json").is_file():
            circt_dirs.append(cr.parent)

    # measure CIRCT screen wall live (it wasn't timed in-run) over this arm's emitted traces
    circt_runs, circt_s = 0, 0.0
    for cd in circt_dirs[:60]:
        t0 = time.time()
        try:
            RCR.prescreen(cd); circt_runs += 1; circt_s += time.time() - t0
        except Exception:
            pass

    think_s = api_ms / 1000
    return {"arm": ARMS[rid][1], "active_wall_min": round(active / 60, 1),
            "agent_session": {
                "think_generate_s": round(think_s, 1),
                "think_pct_of_total_api": round(100 * think_s / max(total_ms / 1000, 1), 1),
                "in_session_spike_s": round(sims["spike"]["sim_s"] + sims["spike"]["build_s"], 2),
                "in_session_verilator_s": round(sims["verilator/VCS"]["sim_s"], 1)},
            "tool_wall_exact": {
                "spike": {"runs": sims["spike"]["runs"],
                          "total_s": round(sims["spike"]["sim_s"] + sims["spike"]["build_s"], 2),
                          "per_run_s": round((sims["spike"]["sim_s"] + sims["spike"]["build_s"]) / max(sims["spike"]["runs"], 1), 3)},
                "verilator/VCS": {"runs": sims["verilator/VCS"]["runs"],
                                  "total_s": round(sims["verilator/VCS"]["sim_s"], 1),
                                  "per_run_s": round(sims["verilator/VCS"]["sim_s"] / max(sims["verilator/VCS"]["runs"], 1), 1)},
                "CIRCT_screen": {"runs": circt_runs, "total_s": round(circt_s, 2),
                                 "per_run_s": round(circt_s / max(circt_runs, 1), 3)}}}


def operator_barrier() -> dict:
    """The expensive verilator the agents did NOT pay in-session — barrier + full audit (separate)."""
    fa = EXP / "reports" / "full_suite_audit.json"
    out = {}
    if fa.is_file():
        d = json.loads(fa.read_text())
        for rid, b in (d.get("backends") or {}).items():
            tr = b.get("timing_rollup") or {}
            n = d.get("n_capsules", 25)
            out[rid] = {"suite_wall_s": tr.get("suite_wall_s"), "sim_active_s": tr.get("sim_active_s"),
                        "per_capsule_verilator_s": round((tr.get("sim_active_s") or 0) / n, 1),
                        "workers": tr.get("max_workers")}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {ARMS[r][1]: harvest(r) for r in ARMS}
    res["_operator_verilator_barrier"] = operator_barrier()
    (OUT / "timing_detailed.json").write_text(json.dumps(res, indent=2))
    print("== AGENT ACTIVE SESSION (excl. sleep) — exact per-tool ==")
    print(f"  {'arm':14s} {'active(min)':>11} {'think+gen':>10} {'spike(in-sess)':>14} {'CIRCT-screen':>13}")
    for r in ARMS:
        t = res[ARMS[r][1]]; tw = t["tool_wall_exact"]
        print(f"  {t['arm']:12s} {t['active_wall_min']:>11} {t['agent_session']['think_generate_s']:>9.0f}s "
              f"spike={tw['spike']['total_s']:.1f}s/{tw['spike']['runs']}runs  "
              f"CIRCT={tw['CIRCT_screen']['total_s']:.1f}s/{tw['CIRCT_screen']['runs']} ({tw['CIRCT_screen']['per_run_s']*1000:.0f}ms/run)")
    print("\n== OPERATOR verilator barrier/audit (separate from agent; the EXPENSIVE part) ==")
    for rid, b in res["_operator_verilator_barrier"].items():
        print(f"  {rid}: suite_wall={b['suite_wall_s']:.0f}s  sim_active={b['sim_active_s']:.0f}s  "
              f"per-capsule verilator≈{b['per_capsule_verilator_s']:.0f}s ({b['workers']} workers)")
    print(f"\nwrote {OUT}/timing_detailed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
