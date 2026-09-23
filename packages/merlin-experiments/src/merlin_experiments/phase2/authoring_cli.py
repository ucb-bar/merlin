"""Explicit-input command line for performance candidate authoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment
from merlin_experiments.phase2 import authoring
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2.contracts import StageGateError


def main(
    argv: list[str] | None = None,
    *,
    resolve_layout: Callable[[argparse.Namespace, TargetExperiment], dict[str, Path]] | None = None,
    suite: str | None = None,
    source_root: Path | None = None,
) -> int:
    # The sandbox gets this through sandbox_env, but the host lane imports candidate modules too --
    # the development-feedback evaluator runs capsule lowerings in-process, and that wrote a second
    # batch of caches into submission/mlir_oot/lowering/ nine minutes after the first. Set it here so
    # this process and every child it spawns inherit it, sandboxed or not.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument("--functional-submission-sha256", required=True)
    if resolve_layout is not None:
        parser.add_argument("--run-id", required=True)
    else:
        for name in ("functional-runs-root", "stage-root", "contract-root"):
            parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=source_root, required=source_root is None)
    if suite is None:
        parser.add_argument("--suite", required=True, help="explicit AET attribution suite")
    parser.add_argument("--model", required=True, help="explicit Codex model slug")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--wall-budget-seconds", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="optional assertion; must equal the frozen formal acceptance exact_count",
    )
    parser.add_argument(
        "--smoke-replicates",
        type=int,
        default=1,
        help="non-claim diagnostic count; must be smaller than the formal cohort",
    )
    parser.add_argument("--round-timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-tool-calls", type=int, default=100)
    parser.add_argument("--tool-timeout-seconds", type=int, default=900)
    parser.add_argument("--families", default="all")
    parser.add_argument("--capsules", default="all")
    parser.add_argument(
        "--waive-functional-gate",
        action="append",
        default=[],
        metavar="PREDICATE",
        help="accept a NAMED completeness gap in the functional baseline "
        "(repeatable). Forwarded by the coordinator so a trial applies exactly "
        "the waivers the campaign was launched with. Integrity predicates "
        "cannot be waived.",
    )
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--gsim-certificate", type=Path, required=True)
    parser.add_argument("--gsim-certificate-sha256", required=True)
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--telemetry-price-table", type=Path, required=True)
    parser.add_argument(
        "--descriptor",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)
    target_experiment = load_target_experiment(args.descriptor, source_root=args.source_root)
    layout = (
        resolve_layout(args, target_experiment)
        if resolve_layout is not None
        else {
            name: getattr(args, name) for name in ("functional_runs_root", "stage_root", "source_root", "contract_root")
        }
    )
    try:
        record = authoring.run_stage(
            **layout,
            suite=suite if suite is not None else args.suite,
            functional_run_id=args.functional_run_id,
            functional_submission_sha256=args.functional_submission_sha256,
            target_experiment=target_experiment,
            sandbox_inputs=PC.select_package_sandbox_inputs(target_experiment),
            model=args.model,
            effort=args.effort,
            wall_budget_seconds=args.wall_budget_seconds,
            rounds=args.rounds,
            round_timeout_seconds=args.round_timeout_seconds,
            replicates=args.replicates,
            smoke_replicates=args.smoke_replicates,
            max_tool_calls=args.max_tool_calls,
            tool_timeout_seconds=args.tool_timeout_seconds,
            families=args.families,
            capsules=args.capsules,
            codex_binary=args.codex_binary,
            waive_functional_gate=tuple(args.waive_functional_gate or ()),
            gsim_certificate=args.gsim_certificate,
            gsim_certificate_sha256=args.gsim_certificate_sha256,
            rtl_facts=args.rtl_facts,
            telemetry_price_table=args.telemetry_price_table,
        )
    except (StageGateError, PC.CampaignGateError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 2
    document = json.loads(record.read_text(encoding="utf-8"))
    if document["admission"]["consumable"] is not True:
        print(f"NO-GO: {document['admission']['refusal']}\nrecord: {record}", file=sys.stderr)
        return 2
    print(f"SEALED: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
