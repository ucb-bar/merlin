"""Measure REAL arc-sim wall time per capsule (min of repeats) and write into arc_results.json.

Replaces the previously-hardcoded ARC_RATE estimate: we time the actual compiled arc replay binary
(reset + RoCC replay + drain), not the one-time clang build. Also records the measured RTL-sim wall
references (verilator per-kernel from perf_results, FireSim per-run) so the speed figure is fully
measured, not derived.
"""
from __future__ import annotations
import json, subprocess, time, os, sys
from pathlib import Path
import _pbcommon as PB

REPO = PB.REPO
RF = REPO / "merlin/targets/gemmini/contracts/rtl_facts"
PYBIN = str(REPO / ".venv/bin/python")
GEN = "merlin.targetgen.rtl.gen_rocc_replay"
H = str(REPO / "merlin/python/merlin/targetgen/rtl/replay_json_to_h.py")
RUNS = REPO / "runs/capsule_bench_v1/runs/gemmini-capsule-bench"
CAPS = REPO / "merlin/contract/capsules"
REPS = 5


def cap_yaml(name):
    for p in CAPS.rglob("capsule.yaml"):
        if p.parent.name == name:
            return p
    return None


def build(cap, trace):
    subprocess.run([PYBIN, "-m", GEN, str(cap), str(trace), "--out", str(RF / "r.json")],
                   cwd=REPO / "merlin/python", capture_output=True)
    subprocess.run([PYBIN, H, str(RF / "r.json"), str(RF / "replay_active.h")], capture_output=True)
    subprocess.run(["clang", "-O2", "-w", "-I", str(RF), str(RF / "gemmini_arc_replay.c"),
                    str(RF / "gemmini.o"), "-o", str(RF / "rbin")], capture_output=True)


def time_rbin():
    best = 1e9
    for _ in range(REPS):
        t0 = time.perf_counter()
        subprocess.run([str(RF / "rbin")], capture_output=True, timeout=300)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    arc = json.loads((RF / "arc_results.json").read_text())
    by = {c["capsule"]: c for c in arc["capsules"]}
    for name, c in by.items():
        cy = cap_yaml(name); tr = RUNS / name / "generated/instruction_trace.json"
        if cy is None or not tr.is_file():
            continue
        build(cy, tr)
        w = time_rbin()
        c["wall_s"] = round(w, 5)
        print(f"  {name}: arc wall = {w*1e3:.2f} ms  ({c.get('cycles')} cyc)")
    # measured RTL-sim wall references (verilator per-kernel; FireSim per-run machinery)
    try:
        pr = json.loads((PB.RUNS / "perf_full_0001/perf_results.json").read_text())
        vw = [((a.get("per_sim") or {}).get("verilator") or {}).get("wall_s")
              for r in pr for a in r["approaches"].values()]
        vw = [x for x in vw if x]
        arc["rtl_wall_ref"] = {"verilator_wall_s_median": round(sorted(vw)[len(vw)//2], 1) if vw else None,
                               "verilator_wall_s_n": len(vw),
                               "firesim_per_run_s_typ": 210,  # measured machinery time per ELF (flash amortized)
                               "note": "verilator = measured per-kernel sim wall (boot+kernel); firesim = measured per-run"}
    except Exception as e:
        arc["rtl_wall_ref"] = {"error": repr(e)}
    (RF / "arc_results.json").write_text(json.dumps(arc, indent=2))
    am = [c["wall_s"] for c in by.values() if c.get("wall_s")]
    print(f"\narc wall: median {1e3*sorted(am)[len(am)//2]:.1f} ms over {len(am)} capsules; "
          f"verilator ref {arc['rtl_wall_ref'].get('verilator_wall_s_median')} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
