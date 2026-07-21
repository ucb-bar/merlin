#!/usr/bin/env python3
"""Run the Gemmini conformance sweep from experiment.yaml (resumable).

  python merlin/experiments/gemmini_cert/run.py [--simulators spike,verilator] [--force]

Skips cells already marked correct in the ledger, so a long Verilator/FireSim sweep can be
killed and resumed. Writes FINDINGS-style results to stdout and the ledger.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from merlin.targetgen.eval.gemmini_dispatcher import run_sweep, summarize
from merlin.common.paths import repo_root

HERE = Path(__file__).resolve().parent


def _resolve(p: str) -> str:
    """Resolve a repo-relative config path (e.g. out/runs/gemmini/cert) against the repo root so the
    sweep lands in the canonical out/ tree regardless of the caller's cwd."""
    pp = Path(p)
    return str(pp if pp.is_absolute() else repo_root() / pp)


def main() -> int:
    cfg = yaml.safe_load((HERE / "experiment.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default=",".join(cfg["rungs"]))
    ap.add_argument("--simulators", default="spike")
    ap.add_argument("--runs-root", default=_resolve(cfg["runs_root"]))
    ap.add_argument("--ledger", default=_resolve(cfg["ledger"]))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    rows = run_sweep(args.rungs.split(","), args.simulators.split(","),
                     runs_root=args.runs_root, ledger_path=args.ledger,
                     force=args.force, timeout=args.timeout)
    print(summarize(rows))
    n_ok = sum(1 for r in rows if r["correct"])
    print(f"\n{n_ok}/{len(rows)} cells correct")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
