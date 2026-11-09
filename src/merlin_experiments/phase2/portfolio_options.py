"""Full-model portfolio CLI declarations and ordered input admission."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin.perf.execution_policy import (
    FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS,
    GLOBAL_AUTHORING_ROUND_MAX_SECONDS,
)
from merlin.perf.host_resources import HostResourcePolicy
from merlin_experiments.phase2 import broker_evidence as BE
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import fast_evaluation_installation as FAST

_DESCRIPTION = "Launch full-graph authoring with optional bounded isolated probes, never model simulation."
_GIB = 1024**3


@dataclass(frozen=True)
class PortfolioInvocation:
    """Admitted options and derived policies; no experiment execution is performed."""

    args: argparse.Namespace
    resource_policy: HostResourcePolicy
    fast_evaluation_configured: bool
    total_authoring: int


def build_parser(*, description: str | None = _DESCRIPTION) -> argparse.ArgumentParser:
    """Expose the launch interface without importing a native controller."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--deployment", type=Path, help="explicit installed portfolio deployment JSON")
    parser.add_argument(
        "--campaign-config",
        type=Path,
        required=True,
        help="existing campaign config or suite JSON carrying frozen run identity/waivers",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--optimization-baseline",
        type=Path,
        help="explicit immutable comparison compiler; never replaces frozen Phase-1 qualification",
    )
    parser.add_argument(
        "--optimization-baseline-sha256", help="required exact compiler-tree SHA-256 for optimization-baseline"
    )
    parser.add_argument(
        "--optimization-baseline-reason", default="host-selected immutable optimization comparison seed"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-emission-cache",
        type=Path,
        help="persistent host-owned exact baseline-emission cache; defaults beside output",
    )
    parser.add_argument(
        "--baseline-emission-cache-seed-run",
        type=Path,
        action="append",
        default=[],
        help="prior run whose exact successful baseline emissions seed the cache",
    )
    parser.add_argument(
        "--portfolio-analysis-workers",
        type=int,
        default=4,
        help="maximum host-admitted concurrent full-model analysis workers",
    )
    parser.add_argument(
        "--historical-reference",
        type=Path,
        help="explicit public historical reference bundle; not target timing calibration",
    )
    parser.add_argument(
        "--historical-reference-sha256", help="required SHA-256 of the exact host-owned historical bundle"
    )
    objective = parser.add_mutually_exclusive_group()
    objective.add_argument(
        "--objective-capsule", help="exact public full-model objective from the existing frozen functional inputs"
    )
    objective.add_argument(
        "--external-objective",
        type=Path,
        help="host JSON spec for one pinned already-normalized external complete-model source",
    )
    parser.add_argument("--external-objective-sha256", help="required exact SHA-256 of the external objective spec")
    parser.add_argument(
        "--portfolio-capsule",
        action="append",
        default=[],
        help="additional frozen full-model training objective; repeat for a portfolio",
    )
    parser.add_argument(
        "--portfolio-external-objective",
        type=Path,
        action="append",
        default=[],
        help="pinned external full-model training member; repeat for a portfolio",
    )
    parser.add_argument(
        "--portfolio-external-objective-sha256",
        action="append",
        default=[],
        help="exact SHA-256 paired by order with portfolio-external-objective",
    )
    parser.add_argument(
        "--fast-evaluation-calibration",
        type=Path,
        help=(
            "exact host analytical calibration JSON; enables accuracy-bounded evaluation only with all companion inputs"
        ),
    )
    parser.add_argument("--fast-evaluation-calibration-sha256", help="raw-file SHA-256 of fast-evaluation-calibration")
    parser.add_argument(
        "--fast-evaluation-quality-observer",
        type=Path,
        help="pinned Python host-reference adapter; target execution is forbidden",
    )
    parser.add_argument(
        "--fast-evaluation-quality-observer-sha256", help="raw-file SHA-256 of fast-evaluation-quality-observer"
    )
    parser.add_argument(
        "--fast-evaluation-quality-observer-symbol", help="explicit callable in the pinned quality-observer adapter"
    )
    parser.add_argument(
        "--fast-evaluation-classification-member-sha256",
        help="exact portfolio member receiving the classification top-1 budget",
    )
    parser.add_argument(
        "--fast-evaluation-held-out-corpus",
        action="append",
        nargs=3,
        default=[],
        metavar=("MEMBER_SHA256", "PATH", "CORPUS_SHA256"),
        help="bind one portfolio member to one exact held-out corpus file; repeat exactly four times",
    )
    parser.add_argument(
        "--fast-evaluation-maximum-model-seconds",
        type=float,
        default=60.0,
        help="serialized host analytical/quality ceiling per model (maximum 60s)",
    )
    parser.add_argument("--round-seconds", type=int, default=600)
    parser.add_argument(
        "--iteration-seconds",
        type=int,
        default=600,
        help=(
            "host-only full-graph compile/static-analysis ceiling; at most "
            f"{FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:g}s; does not extend authoring rounds or "
            "permit full-model simulation"
        ),
    )
    parser.add_argument("--max-tool-calls", type=int, default=40)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument(
        "--total-authoring-seconds",
        type=int,
        help="total reserved authoring budget across bounded rounds; defaults to one round",
    )
    parser.add_argument("--on-round-failure", choices=("stop", "resume-last-checkpoint"), default="stop")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="exact prior sealed candidate; starts an explicit newly frozen policy segment",
    )
    parser.add_argument(
        "--static-analysis-seed-checkpoint",
        type=Path,
        help="explicit prior global candidate whose static analysis may be imported",
    )
    parser.add_argument(
        "--static-analysis-seed-sha256", help="required exact SHA-256 of static-analysis-seed-checkpoint"
    )
    parser.add_argument(
        "--edit-contract",
        type=Path,
        help="host-approved edit_contract.json with sibling receipt.json initial source-file pins",
    )
    parser.add_argument(
        "--mechanism-catalog", type=Path, help="absolute immutable host compiler_mechanism_catalog_v1 JSON"
    )
    parser.add_argument("--mechanism-catalog-sha256", help="required exact raw-file SHA-256 for mechanism-catalog")
    parser.add_argument(
        "--mechanism-work-order", type=Path, help="absolute immutable host_prepared_mechanism_work_order_v1 JSON"
    )
    parser.add_argument(
        "--mechanism-work-order-sha256", help="required exact raw-file SHA-256 for mechanism-work-order"
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="validate and seal an existing checkpoint without a new paid authoring round",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="one compile/static analysis only; no Codex, telemetry, probes or sealing",
    )
    parser.add_argument(
        "--comparison-candidate", type=Path, help="preserved pre-edit compiler for changed-region validation"
    )
    parser.add_argument(
        "--compare-controlled-context",
        action="store_true",
        help="compare identical bounded queued work across the two validation revisions",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="compile and qualify changed host regions without any device simulation",
    )
    parser.add_argument(
        "--probe-interface", type=Path, help="host-selected separate short interface, never the full-model objective"
    )
    parser.add_argument(
        "--probe-runtime-receipt",
        type=Path,
        help="existing exact-ELF/engine warm diagnostic used for wall-time admission",
    )
    parser.add_argument(
        "--probe-profile",
        choices=("none", "occupancy"),
        default="none",
        help="optional minimal joint-busy counters on the isolated primitive only",
    )
    parser.add_argument("--source-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--functional-gate",
        type=Path,
        default=None,
        help="OPT-IN per-iteration execution gate: a JSON file "
        "(merlin_functional_gate_config_v1) naming the model payload dir, the "
        "toolchain (mlir-translate/clang/simulator paths, target triple, "
        "march/mabi, simulator ISA/extension), the MERLIN_RESULT expectations "
        "and a timeout. Each candidate's emitted program is built, run, and a "
        "`failed` verdict excludes it from selection and sealing.",
    )
    parser.add_argument(
        "--min-memory-available-gib",
        type=int,
        default=16,
        help="refuse/stop before host MemAvailable falls below this bound",
    )
    parser.add_argument(
        "--max-swap-used-gib", type=int, default=2, help="refuse/stop after host swap use exceeds this bound"
    )
    parser.add_argument(
        "--resource-sample-seconds", type=float, default=2.0, help="host memory/swap supervision interval"
    )
    parser.add_argument(
        "--resource-trip-samples",
        type=int,
        default=2,
        help="consecutive pressured samples required while a run is active",
    )
    return parser


def worker_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    """Serialize this interface for the frozen worker, not a second option roster.

    The parent admits options with ``parse_invocation``; the worker admits them again.
    Every declared value is forwarded, including defaults, with paths resolved before
    changing working/source roots. This builds arguments only; existing supervisors
    select the executable and own process lifetime. Unsupported new argparse actions
    refuse until their serialization contract is supplied.
    """
    actions = [action for action in build_parser()._actions if not isinstance(action, argparse._HelpAction)]
    if set(vars(args)) != {action.dest for action in actions}:
        raise ValueError("portfolio worker options differ from the declared interface")
    result: list[str] = []
    for action in actions:
        flag = next(option for option in action.option_strings if option.startswith("--"))
        value = True if action.dest == "source_worker" else getattr(args, action.dest)
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"portfolio worker {flag} must be boolean")
            if value:
                result.append(flag)
        elif isinstance(action, argparse._AppendAction):
            if not isinstance(value, list):
                raise ValueError(f"portfolio worker {flag} must be a list")
            for item in value:
                if action.dest == "fast_evaluation_held_out_corpus":
                    if not isinstance(item, (list, tuple)) or len(item) != 3:
                        raise ValueError("fast-evaluation corpus requires member, path and digest")
                    result.extend((flag, str(item[0]), str(Path(item[1]).resolve()), str(item[2])))
                elif action.nargs is None:
                    result.extend((flag, str(Path(item).resolve()) if action.type is Path else str(item)))
                else:
                    raise ValueError(f"unsupported portfolio worker append action: {flag}")
        elif isinstance(action, argparse._StoreAction) and action.nargs is None:
            if value is not None:
                result.extend((flag, str(Path(value).resolve()) if action.type is Path else str(value)))
        else:
            raise ValueError(f"unsupported portfolio worker argument action: {flag}")
    return tuple(result)


def parse_invocation(argv: list[str] | None = None, *, description: str | None = _DESCRIPTION) -> PortfolioInvocation:
    """Parse and admit explicitly selected inputs, retaining the launch refusal order."""
    parser = build_parser(description=description)
    args = parser.parse_args(argv)
    fast_evaluation_configured = FAST.validate_cli(args, parser)
    if args.baseline_emission_cache is None:
        args.baseline_emission_cache = args.output.resolve().parent / "_global_phase2_baseline_emission_cache_v1"
    else:
        args.baseline_emission_cache = args.baseline_emission_cache.resolve()
    if any(path.is_symlink() or not path.is_dir() for path in args.baseline_emission_cache_seed_run):
        parser.error("baseline-emission-cache-seed-run must name a real prior run directory")
    if (
        min(args.min_memory_available_gib, args.max_swap_used_gib) < 0
        or args.portfolio_analysis_workers < 1
        or args.resource_trip_samples < 1
        or not math.isfinite(args.resource_sample_seconds)
        or not 0.25 <= args.resource_sample_seconds <= 30.0
    ):
        parser.error("host resource limits/sample interval are invalid")
    resource_policy = HostResourcePolicy(
        minimum_memory_available_bytes=args.min_memory_available_gib * _GIB,
        maximum_swap_used_bytes=args.max_swap_used_gib * _GIB,
        consecutive_violations_to_stop=args.resource_trip_samples,
    )
    if bool(args.historical_reference) != bool(args.historical_reference_sha256):
        parser.error("historical-reference requires its exact historical-reference-sha256 pin")
    if bool(args.static_analysis_seed_checkpoint) != bool(args.static_analysis_seed_sha256):
        parser.error("static-analysis seed requires both checkpoint path and exact SHA-256")
    if args.static_analysis_seed_checkpoint:
        seed = args.static_analysis_seed_checkpoint
        if (
            not BE._is_sha256(args.static_analysis_seed_sha256)
            or seed.is_symlink()
            or not seed.is_file()
            or P2_CONTRACTS.sha256_file(seed) != args.static_analysis_seed_sha256
        ):
            parser.error("static-analysis seed checkpoint is linked, absent, or differs from its pin")
    if bool(args.mechanism_catalog) != bool(args.mechanism_catalog_sha256):
        parser.error("mechanism-catalog requires its exact mechanism-catalog-sha256 pin")
    if args.mechanism_catalog:
        catalog = args.mechanism_catalog
        if not args.edit_contract:
            parser.error("mechanism-catalog requires an explicit host edit-contract")
        if (
            not BE._is_sha256(args.mechanism_catalog_sha256)
            or not catalog.is_absolute()
            or catalog.resolve() != catalog
            or catalog.is_symlink()
            or not catalog.is_file()
            or catalog.stat().st_mode & 0o222
            or P2_CONTRACTS.sha256_file(catalog) != args.mechanism_catalog_sha256
        ):
            parser.error("mechanism-catalog must be an exact immutable absolute file")
    if bool(args.mechanism_work_order) != bool(args.mechanism_work_order_sha256):
        parser.error("mechanism-work-order requires its exact mechanism-work-order-sha256 pin")
    if args.mechanism_work_order:
        work_order = args.mechanism_work_order
        if not args.mechanism_catalog:
            parser.error("mechanism-work-order requires an explicit mechanism-catalog")
        if (
            not BE._is_sha256(args.mechanism_work_order_sha256)
            or not work_order.is_absolute()
            or work_order.resolve() != work_order
            or work_order.is_symlink()
            or not work_order.is_file()
            or work_order.stat().st_mode & 0o222
            or P2_CONTRACTS.sha256_file(work_order) != args.mechanism_work_order_sha256
        ):
            parser.error("mechanism-work-order must be an exact immutable absolute file")
    if args.validation_only and args.static_analysis_seed_checkpoint:
        parser.error("validation-only compares two revisions and cannot import an initial static seed")
    if args.historical_reference:
        from merlin_experiments.phase2.portfolio_checkpoint import load_historical_reference

        load_historical_reference(
            args.historical_reference, args.historical_reference_sha256, candidate_roots=(args.candidate,)
        )
    if args.analysis_only and (
        args.validation_only
        or args.resume_checkpoint
        or args.comparison_candidate
        or args.probe_interface
        or args.probe_runtime_receipt
        or args.probe_profile != "none"
        or args.compare_controlled_context
        or args.semantic_only
        or args.max_rounds != 1
        or args.total_authoring_seconds is not None
        or args.edit_contract
        or args.mechanism_catalog
        or args.mechanism_work_order
    ):
        parser.error("analysis-only excludes authoring/resume, qualification and profiling options")
    if bool(args.optimization_baseline) != bool(args.optimization_baseline_sha256):
        parser.error("optimization-baseline requires its exact optimization-baseline-sha256 pin")
    if args.optimization_baseline and (
        not BE._is_sha256(args.optimization_baseline_sha256)
        or hash_tree(args.optimization_baseline)["sha256"] != args.optimization_baseline_sha256
    ):
        parser.error("optimization-baseline digest does not match the explicit source")
    if bool(args.external_objective) != bool(args.external_objective_sha256):
        parser.error("external-objective requires its exact external-objective-sha256 pin")
    if len(args.portfolio_external_objective) != len(args.portfolio_external_objective_sha256):
        parser.error("each portfolio-external-objective requires its ordered exact SHA-256 pin")
    if len(set(args.portfolio_capsule)) != len(args.portfolio_capsule):
        parser.error("portfolio-capsule members must be distinct")
    total_authoring = args.total_authoring_seconds if args.total_authoring_seconds is not None else args.round_seconds
    if (
        min(args.max_rounds, total_authoring, args.round_seconds) <= 0
        or args.round_seconds > GLOBAL_AUTHORING_ROUND_MAX_SECONDS
    ):
        parser.error(
            f"authoring bounds must be positive and each round at most {GLOBAL_AUTHORING_ROUND_MAX_SECONDS:g} seconds"
        )
    if not 0 < args.iteration_seconds <= FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:
        parser.error(
            "iteration-seconds is the host-only full-graph static-analysis ceiling and must be "
            f"in (0, {FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:g}]"
        )
    if args.validation_only and (args.max_rounds != 1 or args.resume_checkpoint or args.total_authoring_seconds):
        parser.error("validation-only does not launch or resume an authoring sequence")
    if args.validation_only and args.mechanism_catalog:
        parser.error("mechanism-catalog is for an authored round, not validation-only comparison")
    if args.validation_only != bool(args.comparison_candidate):
        parser.error("validation-only requires exactly one comparison-candidate")
    if args.compare_controlled_context and not args.validation_only:
        parser.error("compare-controlled-context requires validation-only and comparison-candidate")
    if args.semantic_only and (
        not args.validation_only
        or args.compare_controlled_context
        or args.probe_interface
        or args.probe_runtime_receipt
    ):
        parser.error("semantic-only requires validation-only and excludes device profiling options")
    if bool(args.probe_interface) != bool(args.probe_runtime_receipt):
        parser.error("probe interface and runtime receipt must be supplied together")
    if args.probe_profile != "none" and not args.probe_interface:
        parser.error("an isolated probe profile requires its interface and matching runtime receipt")
    return PortfolioInvocation(args, resource_policy, fast_evaluation_configured, total_authoring)
