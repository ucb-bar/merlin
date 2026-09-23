#!/usr/bin/env python3
"""Compatibility command translating native paired-measurement storage inputs."""

from merlin_experiments.phase2 import paired_cli

from merlin.benchharness import runs_root
from merlin.common.paths import merlin_dir, repo_root


def _layout(args, target):
    return {
        "functional_runs_root": args.functional_runs_root or runs_root(target.target, "capsule-bench"),
        "measurement_root": args.measurement_root or runs_root(target.target, "perf-bench"),
        "contract_root": args.contract_root or merlin_dir() / "contract",
    }


def main(argv=None):
    return paired_cli.main(argv, resolve_layout=_layout, source_root=repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
