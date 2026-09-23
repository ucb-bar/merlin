"""Explicit phase-1 initialization, independent of native launcher location.

Imports are inert. Calling load_context is an invocation operation: it normalizes explicit
descriptor/root environment variables, sources missing tooling variables, and then reads the
descriptor. Call before importing environment-sensitive toolchain or evaluator implementations.
This process environment is shared; concurrent experiments belong in separate processes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


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


def load_context(
    descriptor: str | Path,
    *,
    repo: str | Path,
    harness: str | Path | None = None,
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
        from merlin.targetgen.target_experiment import load_target_experiment

        experiment_descriptor = load_target_experiment(descriptor)
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


def resolve_context(args, parser, context=None) -> InvocationContext:
    """Resolve after argument parsing; only native launchers may supply lazy defaults."""
    if args.descriptor is not None or args.repo is not None:
        if args.descriptor is None or args.repo is None:
            parser.error("--descriptor and --repo must be supplied together")
        return load_context(args.descriptor, repo=args.repo)
    if context is not None:
        return context() if callable(context) else context
    parser.error("installed execution requires --descriptor and --repo")


def context_argv(context: InvocationContext) -> list[str]:
    """Trusted host-child arguments, never populated from candidate request fields."""
    return ["--descriptor", str(context.descriptor), "--repo", str(context.repo)]
