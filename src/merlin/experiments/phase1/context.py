"""Explicit phase-1 initialization, independent of native launcher location.

Imports are inert. Calling load_context is an invocation operation: it normalizes explicit
descriptor/root environment variables, sources missing tooling variables, and then reads the
descriptor. Call before importing environment-sensitive toolchain or evaluator implementations.
This process environment is shared; concurrent experiments belong in separate processes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.targetgen.contract.readback_policy import READBACK_TRANSPORTS


@dataclass(frozen=True)
class InvocationContext:
    repo: Path
    descriptor: Path
    experiment: Path
    target: str
    runs: Path
    reports: Path
    bundles: Path
    sourced_environment: tuple[str, ...]
    harness: Path | None = None
    #: New runs live at out/runs/<target>/phase1/<run-id>/; ``runs`` remains the legacy
    #: capsule-bench root, read only to resume a run that started there.
    phase_runs: Path | None = None
    readback_policy: str | None = None


def load_context(
    descriptor: str | Path,
    *,
    repo: str | Path,
    harness: str | Path | None = None,
    readback_policy: str | None = None,
    _legacy_target_reader: Callable[[Path], str] | None = None,
) -> InvocationContext:
    """Initialize one invocation, requiring an explicit descriptor and repository/work root.

    Explicit arguments select the invocation; process variables beat experiment.env defaults.
    Descriptor existence is checked before mutating the environment. Sidecar loading precedes
    descriptor parsing and output-root resolution, as in the historical launcher.

    The private callback is solely for the native compatibility adapter: it retains that edge's
    permissive target-name fallback and prior environment normalization (including unset defaults).
    Installed callers must not supply it. No fallback target or checkout discovery lives here.
    """
    if readback_policy is not None and readback_policy not in READBACK_TRANSPORTS:
        raise ValueError("unsupported invocation readback policy")
    descriptor = Path(descriptor).expanduser().resolve()
    root = Path(repo).expanduser().resolve()
    if _legacy_target_reader is None:
        if not descriptor.is_file():
            raise FileNotFoundError(f"phase-1 descriptor is not a readable file: {descriptor}")
        os.environ["MERLIN_TARGET_EXPERIMENT"] = str(descriptor)
        os.environ["MERLIN_REPO_ROOT"] = str(root)

    from merlin.targetgen.corpora import source_experiment_env

    sourced = source_experiment_env(descriptor=descriptor)
    if _legacy_target_reader is None:
        from merlin.targetgen.target_experiment import load_target_experiment, selected_experiment_contract

        experiment_descriptor = load_target_experiment(descriptor)
        selected_contract = selected_experiment_contract(experiment_descriptor, environment=os.environ)
        if selected_contract is not None:
            os.environ["MERLIN_TARGET_CONTRACT"] = str(selected_contract)
        target = experiment_descriptor.target
        experiment = experiment_descriptor.resource_path(".")
        bundles = experiment_descriptor.resource_path("input_bundles")
    else:
        target = _legacy_target_reader(descriptor)
        # The native edge intentionally accepts malformed legacy descriptors.
        from merlin.targetgen.target_experiment import descriptor_resources_root

        experiment = descriptor_resources_root(descriptor, root=root)
        bundles = experiment / "input_bundles"

    from merlin.benchharness import reports_root, runs_root
    from merlin.common.paths import phase_runs_root

    return InvocationContext(
        repo=root,
        descriptor=descriptor,
        experiment=experiment,
        target=target,
        runs=runs_root(target, "capsule-bench"),
        reports=reports_root("capsule-bench", target),
        bundles=bundles,
        sourced_environment=tuple(sourced),
        harness=Path(harness).resolve() if harness is not None else None,
        phase_runs=phase_runs_root(target, 1),
        readback_policy=readback_policy,
    )


def require_scaffolding(experiment: Path, target: str) -> None:
    """Require materialized bundles; generated-prompt modes need no authored task directory."""
    missing = [name for name in ("input_bundles",) if not (experiment / name).is_dir()]
    if missing:
        raise SystemExit(
            f"experiment dir {experiment} (target={target}) is missing run scaffolding: {', '.join(missing)}.\n"
            f"  • input_bundles/: prepare a reviewed corpus release; standalone generation now writes "
            f"to artifacts, not beside the descriptor. Legacy staging requires an explicit --dest.\n"
            f"  • task/: optional for generated prompts; legacy file-based modes require their explicit task file.\n"
            f"Only descriptor-driven steps (bundle generation, governance checks) work without them."
        )


def experiment_conditions(bundles: Path) -> list[str]:
    """Discover materialized native A/B condition families, retaining the legacy empty default."""
    conditions: set[str] = set()
    for directory in bundles.glob("*_hwbringup*"):
        index = directory.name.find("hwbringup")
        if index > 0 and directory.is_dir() and (directory / "input_bundle_manifest.yaml").is_file():
            conditions.add(directory.name[index:])
    return sorted(conditions) or ["hwbringup_v0"]


def add_context_arguments(parser) -> None:
    """Declare explicit installed invocation inputs without initializing an experiment."""
    parser.add_argument("--descriptor", type=Path, help="target experiment descriptor")
    parser.add_argument("--repo", type=Path, help="repository/work root for declared resources")
    parser.add_argument("--readback-policy", choices=READBACK_TRANSPORTS, default=None)


def resolve_context(args, parser, context=None) -> InvocationContext:
    """Resolve after argument parsing; only native launchers may supply lazy defaults."""
    if args.descriptor is not None or args.repo is not None:
        if args.descriptor is None or args.repo is None:
            parser.error("--descriptor and --repo must be supplied together")
        selected = getattr(args, "readback_policy", None)
        kwargs = {} if selected is None else {"readback_policy": selected}
        return load_context(args.descriptor, repo=args.repo, **kwargs)
    if context is not None:
        resolved = context() if callable(context) else context
        selected = getattr(args, "readback_policy", None)
        return replace(resolved, readback_policy=selected) if selected is not None else resolved
    parser.error("installed execution requires --descriptor and --repo")


def context_argv(context: InvocationContext) -> list[str]:
    """Trusted host-child arguments, never populated from candidate request fields."""
    args = ["--descriptor", str(context.descriptor), "--repo", str(context.repo)]
    if context.readback_policy is not None:
        args += ["--readback-policy", context.readback_policy]
    return args


def readback_kwargs(context: InvocationContext) -> dict:
    """Explicit trusted adapter selection; absent selection preserves legacy calls."""
    if context.readback_policy is None:
        return {}
    from merlin.targetgen.contract.readback_policy import ReadbackPolicy

    return {"readback_policy": ReadbackPolicy(context.readback_policy)}


def readback_record(context: InvocationContext) -> dict | None:
    policy = readback_kwargs(context).get("readback_policy")
    return None if policy is None else policy.record()


def verify_readback_record(context: InvocationContext, record: object) -> None:
    if record != readback_record(context):
        raise RuntimeError("frozen invocation readback policy changed or is absent")
