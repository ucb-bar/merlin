"""Explicit-resource CLI for the installed checkpoint experiment controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_controller as CTRL
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT


def main(argv: list[str] | None = None, *, resolve_context=None, invocation: tuple[str, ...] | None = None) -> int:
    original_args = list(sys.argv[1:] if argv is None else argv)
    invocation = invocation or (sys.executable, "-m", "merlin_experiments.phase2.checkpoint_cli", *original_args)
    parser = argparse.ArgumentParser()
    resource_names = (
        "source_root",
        "contract_root",
        "functional_runs_root",
        "stage_root",
        "measurement_root",
        "holdout_catalog",
        "core_package_root",
        "experiments_package_root",
        "experiments_namespace_root",
        "chia_wrapper",
    )
    for name in resource_names:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=resolve_context is None)
    parser.add_argument("--suite", required=resolve_context is None)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument(
        "--perf-capsules", default="all", help="comma-separated performance capsules to measure, or 'all'"
    )
    parser.add_argument(
        "--perf-families", default="all", help="comma-separated performance families to measure, or 'all'"
    )
    parser.add_argument("--functional-submission-sha256", required=True)
    parser.add_argument(
        "--waive-functional-gate",
        action="append",
        default=[],
        metavar="PREDICATE",
        help="accept a NAMED completeness gap in the functional baseline instead of "
        "refusing (repeatable). Integrity predicates cannot be waived; asking is "
        "an error. Recorded in the manifest and marks the run not gate-clean.",
    )
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--perf-profile", type=Path, required=True)
    parser.add_argument("--gsim-certificate", type=Path, required=True)
    parser.add_argument("--gsim-certificate-sha256", required=True)
    parser.add_argument("--functional-gsim-certificate", type=Path)
    parser.add_argument("--functional-gsim-certificate-sha256")
    parser.add_argument(
        "--waive-functional-gsim-certificate",
        action="store_true",
        help="launch WITHOUT the public+hidden functional cross-validation "
        "certificate, accepting GSIM's functional verdicts on the same terms "
        "phase 1 accepted them. Recorded in the manifest and marks the run not "
        "gate-clean: the functional regrade becomes GSIM-only, uncorroborated "
        "by a second engine. Timing authority is unaffected -- it is pinned by "
        "the tuning certificate, which is never waivable.",
    )
    parser.add_argument("--heldout-qualification-timeout", type=int, default=600)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--wall-budget-seconds", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-timeout-seconds", type=int, required=True)
    parser.add_argument("--max-tool-calls", type=int, required=True)
    parser.add_argument("--tool-timeout-seconds", type=int, required=True)
    parser.add_argument("--smoke-replicates", type=int, default=1)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument(
        "--sim-workers",
        type=int,
        default=1,
        metavar="N",
        help="executions in flight per measurement cell (default 1 = serial)",
    )
    parser.add_argument("--generalization-count", type=int, default=4)
    parser.add_argument("--measurement-timeout", type=int, default=600)
    parser.add_argument("--gsim-max-cycles", type=int)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--hardware-counters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--telemetry-price-table", type=Path)
    parser.add_argument("--chia-python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(original_args)
    if resolve_context is None:
        context = AD.ExecutionContext(
            source_root=args.source_root,
            contract_root=args.contract_root,
            functional_runs_root=args.functional_runs_root,
            stage_root=args.stage_root,
            measurement_root=args.measurement_root,
            holdout_sources=HOLDOUT.HoldoutSourceContext(
                source_root=args.source_root,
                catalog_path=args.holdout_catalog,
                core_package_root=args.core_package_root,
                experiments_package_root=args.experiments_package_root,
                experiments_namespace_root=args.experiments_namespace_root,
            ),
            chia_wrapper=args.chia_wrapper,
            invocation=invocation,
            suite=args.suite,
        )
    else:
        context = resolve_context(args, invocation)
    _raw = {key: value for key, value in vars(args).items() if key not in (*resource_names, "suite", "dry_run")}
    _raw["context"] = context
    _raw["waive_functional_gate"] = tuple(_raw.pop("waive_functional_gate", ()) or ())
    config = AD.Config(**_raw)
    try:
        outcome = CTRL.run(config, dry_run=args.dry_run)
    except Exception as exc:
        print(f"NO-GO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcome if isinstance(outcome, dict) else {"manifest": str(outcome)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
