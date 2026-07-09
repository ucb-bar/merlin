#!/usr/bin/env python
"""Agent-facing self-check for the Muon backend — REDACTED grade on cyclotron.

Thin wrapper over the shared, target-parametric `merlin.benchharness.selfcheck` driver: it supplies a
Muon `BenchTargetSpec` (the muon capsule runner + %FP-peak headline) and reuses the one redacted-grade
implementation shared with every other target. Runs the agent's submission over the public corpus on
**cyclotron** (~1 s/kernel) and prints pass/fail + failure plane + mismatch_count + cycles + %FP-peak —
never expected/golden values. Iterate until every capsule passes, then push %FP-peak up.

  python agent_selfcheck.py                      # grade submission/ over all public capsules
  python agent_selfcheck.py --capsule MG00_gemm_16x16x16
  python agent_selfcheck.py --submission submission --json   # machine-readable

This is a harness tool (not part of your graded package) — it may import Merlin; YOUR package may not.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# locate the in-repo merlin/python (harness side)
_HERE = Path(__file__).resolve().parent
for _c in (_HERE, *_HERE.parents):
    if (_c / "merlin" / "python").is_dir():
        sys.path.insert(0, str(_c / "merlin" / "python"))
        _REPO = _c
        break


def _spec(corpus_root: str):
    from merlin.benchharness.spec import BenchTargetSpec
    from merlin.targetgen import muon_capsule_runner as MR
    return BenchTargetSpec(
        name="Muon", runner=MR, corpus_root=Path(corpus_root), labels={"public", "dev"},
        contract=None,  # -> absolute default contract dir (robust to CWD)
        perf_tier="L2", perf_fields=lambda t: {"pct_fp_peak": t.get("pct_fp_peak")},
        peak_note="Muon SIMT FP peak (64 flop/cycle = 32 GFLOP/s @ 500 MHz)")


def grade(submission: str, capsules_root: str, runs_root: str, timeout: int,
          only: str | None = None) -> dict:
    """Return a REDACTED verdict (per-capsule pass/fail + plane + mismatch_count + cycles + %peak).

    Signature preserved for `run_muon_qa_loop.py` (imports this as SC and calls SC.grade)."""
    from merlin.benchharness.selfcheck import redacted_grade
    return redacted_grade(_spec(capsules_root), submission, runs_root, timeout, only=only)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muon backend self-check (redacted, cyclotron)")
    ap.add_argument("--submission", default="submission")
    ap.add_argument("--capsule", default=None, help="grade only this capsule")
    # prefer the capsules staged into the agent's workspace (./capsules); else the repo corpus
    _default_caps = "capsules" if Path("capsules").is_dir() else str(
        _REPO / "merlin/experiments/muon_perf_bench_v0/kernels")
    ap.add_argument("--capsules-root", default=_default_caps)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from merlin.benchharness.selfcheck import print_verdict
    runs = Path(tempfile.mkdtemp(prefix="muon_selfcheck_"))
    v = grade(a.submission, a.capsules_root, str(runs), a.timeout, only=a.capsule)
    if a.json:
        print(json.dumps(v, indent=2))
        return 0 if v["all_pass"] else 1
    print_verdict(_spec(a.capsules_root), v, perf_key="pct_fp_peak", perf_suffix="% peak")
    print("\nALL PASS — now push %FP-peak up." if v["all_pass"]
          else "\nNot all pass yet — fix the failing planes above.")
    return 0 if v["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
