#!/usr/bin/env python
"""Run the Muon FP32 perf-bench corpus through a backend and report GFLOP/s vs the FP peak.

Thin wrapper over the shared `merlin.benchharness.perf` driver: it supplies a Muon `BenchTargetSpec`
(the muon capsule runner + %FP-peak headline) and a matmul flop counter, then reuses the one perf
loop + report shared with every other target. L2 = cyclotron --timing; L3 = VCS-RTL difftest cert.

Usage:
  run_muon_perf.py [--package artifacts/targets/muon/reference_v0] [--run-id ref_v0]
                   [--kernels-root experiments/muon_perf_bench_v0/kernels] [--timeout 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "merlin" / "python"))

from merlin.benchharness.perf import run_perf, perf_table
from merlin.benchharness.spec import BenchTargetSpec
from merlin.runtime.backends.muon import FP_PEAK_GFLOPS
from merlin.targetgen import muon_capsule_runner as MR


def _flops(cap: dict) -> int | None:
    """2*M*K*N for a matmul/linear capsule, from its declared input/weight shapes."""
    if cap.get("operation", {}).get("op") not in ("matmul", "linear"):
        return None
    by_role = {s.get("role"): s.get("shape") for s in cap.get("inputs", [])}
    lhs, w = by_role.get("input"), by_role.get("weight")
    if lhs and w and len(lhs) == 2 and len(w) == 2:
        return 2 * lhs[0] * lhs[1] * w[1]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muon FP32 perf bench")
    ap.add_argument("--package", default="out/artifacts/targets/muon/reference_v0")
    ap.add_argument("--kernels-root", default="merlin/experiments/muon_perf_bench_v0/kernels")
    ap.add_argument("--run-id", default="ref_v0")
    ap.add_argument("--labels", default="public,dev")
    ap.add_argument("--contract", default="merlin/contract")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default=None, help="output dir (default runs/muon/perf-bench/<run-id>)")
    a = ap.parse_args(argv)

    # resolve a relative --contract against the repo root (robust to CWD)
    contract = a.contract if Path(a.contract).is_absolute() else str(_REPO / a.contract)
    spec = BenchTargetSpec(
        name="Muon", runner=MR, corpus_root=_REPO / a.kernels_root,
        labels=set(a.labels.split(",")) if a.labels else None, contract=contract, perf_tier="L2",
        perf_fields=lambda t: {"gflops": t.get("gflops"), "pct_fp_peak": t.get("pct_fp_peak")},
        peak_note=f"the Muon SIMT FP peak ({FP_PEAK_GFLOPS:g} GFLOP/s, 64 flop/cycle @ 500 MHz)")

    out_dir = Path(a.out) if a.out else (_REPO / "runs" / "muon" / "perf-bench" / a.run_id)
    try:
        summary = run_perf(spec, package=str(_REPO / a.package), run_id=a.run_id, out_dir=out_dir,
                           timeout=a.timeout, flops_fn=_flops, extra_tier="L3")
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2
    print(perf_table(summary))
    print(f"\nwrote {out_dir/'perf_results.json'} and perf_table.md")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
