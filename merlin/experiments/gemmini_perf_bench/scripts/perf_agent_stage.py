#!/usr/bin/env python3
"""Compatibility entry point for native performance authoring layouts."""

from merlin_experiments.phase2.authoring_cli import main as authoring_main
from merlin_experiments.phase2.contracts import safe_component

from merlin.benchharness import runs_root
from merlin.common.paths import merlin_dir, repo_root


def _native_layout(args, target):
    run_id = safe_component(args.run_id, label="performance stage run id")
    return {
        "functional_runs_root": runs_root(target.target, "capsule-bench"),
        "stage_root": runs_root(target.target, "perf-bench") / "agent_stages" / run_id,
        "contract_root": merlin_dir() / "contract",
        "source_root": args.source_root,
    }


def main(argv=None):
    return authoring_main(argv, resolve_layout=_native_layout, suite="gemmini-perf-bench", source_root=repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
