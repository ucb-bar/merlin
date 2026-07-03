#!/usr/bin/env python
"""Run the Muon FP32 perf-bench corpus through a backend and report GFLOP/s vs the FP peak.

The Muon analog of experiments/gemmini_perf_bench/scripts/run_perf_bench.py. For each kernel it runs
the full Muon tier ladder (L0 golden / L1 consistency / L2 cyclotron --timing perf) via the parallel
:mod:`merlin.targetgen.muon_capsule_runner`, then reports the achieved GFLOP/s as a percent of the
Muon SIMT FP peak (64 flop/cycle = 32 GFLOP/s @ 500 MHz) -- the conservative utilization headline.

Usage:
  run_muon_perf.py [--package artifacts/targets/muon/reference_v0] [--run-id ref_v0]
                   [--kernels-root experiments/muon_perf_bench_v0/kernels] [--timeout 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "merlin" / "python"))

from merlin.targetgen import muon_capsule_runner as MR
from merlin.targetgen.muon_oracles import flops_from_cb
from merlin.runtime.backends.muon import FP_PEAK_GFLOPS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muon FP32 perf bench")
    ap.add_argument("--package", default="artifacts/targets/muon/reference_v0")
    ap.add_argument("--kernels-root", default="experiments/muon_perf_bench_v0/kernels")
    ap.add_argument("--run-id", default="ref_v0")
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=None, help="output dir (default runs/muon/perf-bench/<run-id>)")
    a = ap.parse_args(argv)

    out_dir = Path(a.out) if a.out else (_REPO / "runs" / "muon" / "perf-bench" / a.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root = out_dir / "_capsule_runs"

    labels = set(a.labels.split(",")) if a.labels else None
    caps = MR.discover_capsules(str(_REPO / a.kernels_root), labels=labels, contract=a.contract)
    if not caps:
        print(f"no capsules under {a.kernels_root}", file=sys.stderr)
        return 2

    rows = []
    for cap in sorted(caps, key=lambda c: c["name"]):
        res = MR.run_capsule(cap, str(_REPO / a.package), runs_root=str(runs_root),
                             run_id=cap["name"], contract=a.contract, timeout=a.timeout)
        flops = _flops_from_capsule(cap)
        l2 = res["tiers"].get("L2", {})
        rows.append({
            "kernel": cap["name"], "status": res["status"], "flops": flops,
            "cycles": l2.get("cycles"), "gflops": l2.get("gflops"),
            "pct_fp_peak": l2.get("pct_fp_peak"),
            "l3_cert": res["tiers"].get("L3", {}).get("status"),
        })

    summary = {"target": "muon", "package": a.package, "run_id": a.run_id,
               "fp_peak_gflops": FP_PEAK_GFLOPS, "kernels": rows,
               "passed": sum(1 for r in rows if r["status"] == "pass"), "total": len(rows)}
    (out_dir / "perf_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "perf_table.md").write_text(_md_table(summary), encoding="utf-8")
    print(_md_table(summary))
    print(f"\nwrote {out_dir/'perf_results.json'} and perf_table.md")
    return 0 if summary["passed"] == summary["total"] else 1


def _flops_from_capsule(cap: dict) -> int | None:
    """2*M*K*N for a matmul/linear capsule, from its declared input/weight shapes."""
    op = cap.get("operation", {}).get("op")
    if op not in ("matmul", "linear"):
        return None
    by_role = {s.get("role"): s.get("shape") for s in cap.get("inputs", [])}
    lhs, w = by_role.get("input"), by_role.get("weight")
    if lhs and w and len(lhs) == 2 and len(w) == 2:
        m, k = lhs
        _, n = w
        return 2 * m * k * n
    return None


def _md_table(summary: dict) -> str:
    peak = summary["fp_peak_gflops"]
    lines = [f"# Muon FP32 perf bench -- {summary['package']} ({summary['run_id']})",
             "",
             f"Reported conservatively against the Muon SIMT FP peak "
             f"(64 flop/cycle = {peak:g} GFLOP/s @ 500 MHz). "
             f"L2 = cyclotron --timing; L3 = VCS-RTL difftest cert.",
             "",
             "| kernel | status | flops | cycles | GFLOP/s | % FP peak | L3 cert |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in summary["kernels"]:
        g = f"{r['gflops']:.3f}" if r["gflops"] is not None else "-"
        p = f"{r['pct_fp_peak']:.2f}%" if r["pct_fp_peak"] is not None else "-"
        lines.append(f"| {r['kernel']} | {r['status']} | {r['flops'] or '-'} | "
                     f"{r['cycles'] or '-'} | {g} | {p} | {r['l3_cert'] or '-'} |")
    lines.append("")
    lines.append(f"**{summary['passed']}/{summary['total']} pass.**")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
