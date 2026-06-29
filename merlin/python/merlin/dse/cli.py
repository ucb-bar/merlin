"""``merlin-dse`` CLI: run the interface-DSE harness and phase-transition experiment.

Thin wrapper over ``merlin.dse``. Writes ``dse_result`` artifacts, ``exploitability_report``s,
and the headline ``phase_transition.csv`` (+ ``.png`` if matplotlib is present) under
``output/dse/<workload>/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from merlin.common import paths
from merlin.common.artifacts import yaml_artifact
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.synthesize import FEATURE_ACCUMULATOR, FEATURE_RESIDENT
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
from merlin.dse.experiment import phase_transition
from merlin.dse.harness import evaluate_feature
from merlin.dse.report import build_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-dse", description=__doc__)
    ap.add_argument("--workload", default="vla_action_chunk_decode")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--out", default=None, help="output dir (default output/dse/<workload>)")
    ap.add_argument("--no-experiment", action="store_true",
                    help="only emit dse_result for the single point, skip the sweep")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the scoreboard + hardware-vs-interface decision report")
    ap.add_argument("--search", action="store_true",
                    help="also run the MAP-Elites strategy portfolio + search reports")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else paths.repo_root() / "artifacts" / "dse" / args.workload

    # Single-point dse_result artifacts (both features), written under out/<feature>/.
    rpv = compute_rpv(build_region(H=args.H, reuse_count=args.H, K=256))
    for feature in (FEATURE_RESIDENT, FEATURE_ACCUMULATOR):
        res = evaluate_feature(args.workload, rpv, feature)
        yaml_artifact(f"{feature}/dse_result.yaml", res,
                      header=f"dse_result: {args.workload} / {feature}").write(out)

    if not args.no_experiment:
        res = phase_transition(out_dir=out, workload=args.workload)
        best = {}
        for r in res["rows"]:
            if r["dtype"] == "i8" and r["epilogue"] and r["best"]:
                best[r["H"]] = r["contract"]
        print(f"phase transition (i8, epilogue) best contract by H: {best}")

    if not args.no_report:
        rep = build_report(rpv, workload=args.workload, out_dir=out)
        cap = rep["capstone"]
        print(f"hardware-only best:   {cap['hardware_only_best']['strategy']} "
              f"(cycles={cap['hardware_only_best']['cycles']})")
        print(f"interface-aware best: {cap['interface_aware_best']['strategy']} "
              f"(cycles={cap['interface_aware_best']['cycles']}); "
              f"changes category={cap['best_interface_changes_category']}")

    if args.search:
        _run_search(rpv, args.workload, out)

    print(f"artifacts -> {out}")
    return 0


def _run_search(rpv: dict, workload: str, out: Path) -> None:
    from merlin.search import grid, map_elites
    from merlin.search.candidate import seed_candidates
    from merlin.search.evaluator import make_evaluator
    from merlin.search.reports import build_report as build_search_report

    ev = make_evaluator([(workload, rpv)])
    grid_rows = grid.grid_search_strategies(seed_candidates(), ev)
    me = map_elites.map_elites_search(seed_candidates(), ev, iterations=40, seed=0,
                                      workload_regime="decode_like")
    build_search_report(me["archive"], grid_rows=grid_rows, title=workload,
                        out_dir=out / "search")
    print(f"search portfolio: {me['occupied_cells']} behavior cells; "
          f"best={me['best'].artifact['id']}")


if __name__ == "__main__":
    sys.exit(main())
