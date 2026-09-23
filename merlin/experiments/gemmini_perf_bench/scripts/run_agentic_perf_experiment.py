#!/usr/bin/env python3
"""Legacy example layout adapter for the installed checkpoint campaign CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import _pbcommon as PB
from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_cli
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT

from merlin.benchharness import runs_root
from merlin.targetgen.target_experiment import load_target_experiment


def _resolve_context(args, invocation: tuple[str, ...]) -> AD.ExecutionContext:
    target = load_target_experiment(args.descriptor)
    source = args.source_root or PB.REPO
    return AD.ExecutionContext(
        source_root=source,
        contract_root=args.contract_root or source / "merlin/contract",
        functional_runs_root=args.functional_runs_root or runs_root(target.target, "capsule-bench"),
        stage_root=args.stage_root or runs_root(target.target, "perf-bench") / "agent_stages",
        measurement_root=args.measurement_root or PB.RUNS,
        holdout_sources=HOLDOUT.HoldoutSourceContext(
            source_root=source,
            catalog_path=args.holdout_catalog or source / "experiments/catalog.yaml",
            core_package_root=args.core_package_root or source / "src/merlin",
            experiments_package_root=args.experiments_package_root
            or source / "packages/merlin-experiments/src/merlin_experiments",
            experiments_namespace_root=args.experiments_namespace_root
            or source / "packages/merlin-experiments/src/merlin",
        ),
        chia_wrapper=args.chia_wrapper or Path(__file__).resolve().with_name("chia_agentic_perf_experiment.py"),
        invocation=invocation,
        suite=args.suite or "gemmini-perf-bench",
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return checkpoint_cli.main(
        arguments,
        resolve_context=_resolve_context,
        invocation=(sys.executable, str(Path(__file__).resolve()), *arguments),
    )


if __name__ == "__main__":
    raise SystemExit(main())
