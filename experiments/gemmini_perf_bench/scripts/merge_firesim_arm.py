#!/usr/bin/env python3
"""Merge firesim_arm_results.json into perf_results.json as the FireSim (L5) cycle-accurate tier.

Injects under each approach's per_sim["firesim"] = {cycles, util_pct, correct}. FireSim and verilator
both simulate the SAME RTL, so their cycles are directly comparable — verilator covers the small kernels,
FireSim the larger ones that exceed verilator's wall-clock budget.

Usage: merge_firesim_arm.py [--run-id perf_full_0001]
"""
from __future__ import annotations

import argparse
import json

import _pbcommon as PB


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    pr = run / "perf_results.json"
    fs = run / "firesim_arm_results.json"
    if not pr.is_file() or not fs.is_file():
        print(f"missing {pr if not pr.is_file() else fs}"); return 2
    rows = json.loads(pr.read_text())
    fsd = json.loads(fs.read_text())
    merged = 0
    for r in rows:
        cells = fsd.get(r["kernel"])
        if not cells:
            continue
        for arm, v in cells.items():
            if v.get("cycles") is None:
                continue
            ap_ = r["approaches"].setdefault(arm, {"approach": arm, "per_sim": {}})
            ap_.setdefault("per_sim", {})["firesim"] = {
                "cycles": v["cycles"], "correct": v.get("correct"),
                "util_pct": v.get("util_pct")}
            merged += 1
    pr.write_text(json.dumps(rows, indent=2))
    print(f"merged {merged} FireSim L5 cells into {pr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
