#!/usr/bin/env python3
"""Merge iree_arm_results.json into a run's perf_results.json as the `iree_dialect` approach.

The IREE arm (approach d) runs separately (run_iree_arm.py) because it uses the deprecated-merlin IREE
build tree + a per-shape rebuild, not the merlin contract harness. This stitches its per-kernel
spike cycles into the same perf_results.json the other approaches populate, under
approaches["iree_dialect"]["per_sim"]["spike"], so gen_perf_report.py renders it as the 4th column.
IREE is spike-only (no verilator), so its verilator cell stays `·`.

Usage: merge_iree_arm.py [--run-id perf_full_0001]
"""
from __future__ import annotations

import argparse
import json

import _pbcommon as PB


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    a = ap.parse_args(argv)
    run = PB.RUNS / a.run_id
    pr = run / "perf_results.json"
    ir = run / "iree_arm_results.json"
    if not pr.is_file():
        print(f"no perf_results.json at {pr}")
        return 2
    if not ir.is_file():
        print(f"no iree_arm_results.json at {ir}")
        return 2
    rows = json.loads(pr.read_text())
    iree = json.loads(ir.read_text())
    merged = 0
    for r in rows:
        rec = iree.get(r["kernel"])
        if not rec:
            continue
        if rec.get("error") and rec.get("cycles") is None:
            r.setdefault("approaches", {})["iree_dialect"] = {"approach": "iree_dialect",
                                                              "error": rec["error"], "per_sim": {}}
        else:
            r.setdefault("approaches", {})["iree_dialect"] = {
                "approach": "iree_dialect",
                "per_sim": {"spike": {"cycles": rec.get("cycles"),
                                      "correct": rec.get("correct"),
                                      "util_pct": rec.get("util_pct"),
                                      "wall_s": rec.get("wall_s")}}}
        merged += 1
    pr.write_text(json.dumps(rows, indent=2))
    print(f"merged iree_dialect into {merged}/{len(rows)} kernels of {pr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
