"""``merlin-dse-guidance`` CLI.

Two modes:

  * single workload — build the flat/multi-rate reports, the flat-vs-multi-rate diff, the axis
    triage, and (when a measurement source is supplied) the CPU-coupling result and calibration
    anchor for one workload;
  * ``--study`` — run the exhaustive cross-workload study over every supported workload,
    synthesizing an analytical baseline + temporal view where no measured fixture exists.

Logic lives in :mod:`merlin.dse_guidance`; this is a thin wrapper. It reuses the
``design_pressure`` region loading and pressure-vector machinery (via
:mod:`merlin.dse_guidance.loader` / ``representation``) and never duplicates DSE cost logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from merlin.common import paths
from merlin.dse_guidance import aet_ingest, baseline_cost as BC, loader, synth, temporal as T
from merlin.dse_guidance.pipeline import run_guidance, write_artifacts
from merlin.dse_guidance.study import (discover_model_specs, discover_specs, run_model_study,
                                        run_study)


def _single(args) -> int:
    region = None
    if args.region_yaml or args.workload:
        try:
            region = loader.load_region(args.workload, args.region_yaml, H=args.H)
        except SystemExit:
            region = None  # may be a pure-temporal workload (e.g. smolvla_action_head)

    if args.temporal_metadata:
        temporal = T.load(args.temporal_metadata)
    elif region is not None:
        print("note: no --temporal-metadata; synthesizing from region reuse structure")
        temporal = T.parse(synth.synth_temporal(region, control_rate_hz=args.control_rate_hz))
    else:
        raise SystemExit("need --temporal-metadata (or a resolvable --workload/--region-yaml)")

    baseline = None
    if args.baseline_cost:
        baseline = BC.load(args.baseline_cost)
    elif region is not None and not args.structural_only:
        print("note: no --baseline-cost; synthesizing analytical baseline from region")
        baseline = BC.parse(synth.analytical_baseline_cost(region))
    else:
        print("note: no baseline cost — emitting structural outputs only (topology, capture "
              "fidelity, candidate axes). Quantitative triage needs --baseline-cost.")

    wl = baseline.workload if baseline is not None else temporal.workload
    coupling = aet_ingest.ingest(args.cpu_coupling, args.aet_run, workload=wl)

    result = run_guidance(temporal, baseline, region=region, coupling=coupling)
    out = Path(args.out) if args.out else (
        paths.repo_root() / "output" / "dse_guidance" / wl)
    write_artifacts(result, out)

    for wmsg in result.warnings:
        print(f"warning: {wmsg}")
    print(f"workload={wl}  class={result.topology.workload_class}  "
          f"capture-fidelity severity={result.fidelity.severity}")
    print(f"  structural DSE candidates: {[c.axis for c in result.candidate_axes]}")
    if result.triage_multirate is not None:
        print(f"  (quantitative, uncalibrated) flat top      = {_top_names(result.triage_flat)}")
        print(f"  (quantitative, uncalibrated) multirate top = {_top_names(result.triage_multirate)}")
    print(f"artifacts -> {out}")
    return 0


def _top_names(tr: dict, n: int = 3) -> list[str]:
    return [r["axis"] for r in tr["axes"]
            if r["priority_score"] and r["legality"]][:n]


def _study(args) -> int:
    if args.models:
        specs = discover_model_specs()
        default_sub = "study_models"
        kind = "real model captures"
    else:
        specs = discover_specs()
        default_sub = "study"
        kind = "semantic_memory regions"
    if not specs:
        raise SystemExit(f"no workloads discovered for the study ({kind})")
    out = Path(args.out) if args.out else (
        paths.repo_root() / "output" / "dse_guidance" / default_sub)
    summary = run_model_study(out) if args.models else run_study(specs, out)
    print(f"studied {summary['n_workloads']} workloads ({kind}): "
          f"{', '.join(summary['workloads'])}")
    print(f"artifacts -> {out}  (study_summary.csv, study_summary.md, <workload>/...)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-dse-guidance", description=__doc__)
    ap.add_argument("--workload", default=None,
                    help="synthetic name or semantic_memory benchmark name")
    ap.add_argument("--region-yaml", default=None, help="explicit workload_region YAML path")
    ap.add_argument("--temporal-metadata", default=None,
                    help="temporal_workload_metadata YAML (else synthesized from region)")
    ap.add_argument("--baseline-cost", default=None,
                    help="baseline_cost YAML (else analytical baseline synthesized from region)")
    ap.add_argument("--cpu-coupling", default=None, help="cpu_coupling measurements YAML")
    ap.add_argument("--aet-run", default=None,
                    help="aet run dir with metrics/summary_metrics.json (measured coupling)")
    ap.add_argument("--H", type=int, default=16, help="action horizon (synthetic region loader)")
    ap.add_argument("--control-rate-hz", type=float, default=30.0,
                    help="control rate for synthesized temporal metadata")
    ap.add_argument("--study", action="store_true",
                    help="run the exhaustive cross-workload study")
    ap.add_argument("--models", action="store_true",
                    help="with --study: study the real model zoo (captures under output/) "
                         "instead of the semantic_memory regions")
    ap.add_argument("--structural-only", action="store_true",
                    help="emit only the structural front-end (topology, capture fidelity, "
                         "candidate axes); skip the quantitative triage even if a region is given")
    ap.add_argument("--out", default=None, help="output dir")
    args = ap.parse_args(argv)

    if args.study:
        return _study(args)
    return _single(args)


if __name__ == "__main__":
    sys.exit(main())
