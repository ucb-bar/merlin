"""Explicit backend initialization, independent of native launcher location.

Imports are inert. Calling load_context is an invocation operation: it normalizes explicit
descriptor/root environment variables, sources missing tooling variables, and then reads the
descriptor. Call before importing environment-sensitive toolchain or evaluator implementations.
This process environment is shared; concurrent experiments belong in separate processes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sysconfig
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from merlin.development.backend.options import RunOptions
from merlin.target.contract.harness.readback_policy import READBACK_TRANSPORTS

# ---- context -----------------------------------------------------------------------------------


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
    #: New runs live at out/runs/<target>/backend-generation/<run-id>/; ``runs`` remains the legacy
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

    from merlin.target.registry.descriptor_loading import source_experiment_env

    sourced = source_experiment_env(descriptor=descriptor)
    if _legacy_target_reader is None:
        from merlin.target.registry.descriptor_loading import (
            load_target_experiment,
            selected_experiment_contract,
        )

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
        from merlin.target.registry.experiment_descriptor import descriptor_resources_root

        experiment = descriptor_resources_root(descriptor, root=root)
        bundles = experiment / "input_bundles"

    from merlin.common.paths import reports_root, runs_root
    from merlin.common.paths import runs_root as generation_runs_root

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
        phase_runs=generation_runs_root(target, "backend-generation"),
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
    from merlin.target.contract.harness.readback_policy import ReadbackPolicy

    return {"readback_policy": ReadbackPolicy(context.readback_policy)}


def readback_record(context: InvocationContext) -> dict | None:
    policy = readback_kwargs(context).get("readback_policy")
    return None if policy is None else policy.record()


def verify_readback_record(context: InvocationContext, record: object) -> None:
    if record != readback_record(context):
        raise RuntimeError("frozen invocation readback policy changed or is absent")


# ---- runtime_environment -----------------------------------------------------------------------
# Invocation-scoped host setup, independent of native launcher location.
#
# Preparation does not mutate process state. Application is for serialized controller
# invocations only: existing graders inherit os.environ, so parallel experiments must
# use separate processes. This is not an integrity-policy or credential sandbox.

RETIRED_READONLY_ENV = "MERLIN_PINNED_SUBMISSION_READ_ONLY"

RETIRED_READONLY_MESSAGE = "pinned-submission read-only mode is unsupported; no certified-resume policy is active"


@dataclass(frozen=True)
class PreparedEnvironment:
    """Host-only environment and deliberately small, receipt-safe account metadata.

    Never serialize this object: environment values can contain credentials. Only
    account is suitable for the existing environment receipt.
    """

    environment: Mapping[str, str] = field(repr=False)
    account: Mapping[str, object]
    refusal: int | None = None
    message: str = ""


def framework_import_roots() -> tuple[str, ...]:
    """Active installation roots; no compatibility promise for another Python ABI."""
    from merlin.common.paths import python_import_roots

    paths = sysconfig.get_paths()
    roots = (*python_import_roots(), paths.get("purelib"), paths.get("platlib"))
    return tuple(dict.fromkeys(str(Path(root).resolve()) for root in roots if root))


def prepare_runtime_environment(
    options: RunOptions,
    base_env: Mapping[str, str],
    *,
    machine_defaults: Mapping[str, str],
    library_paths: tuple[Path, ...] = (),
) -> PreparedEnvironment:
    """Prepare the historical host setup with explicit machine configuration.

    The native edge resolves .env through its existing reader and supplies defaults;
    explicit base_env values win. Account probing is the only subprocess, runs only
    when requested, and receives the prepared environment explicitly. Errors never
    echo provider credentials or subprocess output.
    """
    environment = dict(base_env)
    account: dict[str, object] = {"config_dir": None, "email": None, "orgId": None}

    def set_value(key, value):
        environment[key] = value

    def default(key, value):
        if key not in environment:
            set_value(key, value)

    def remove(key):
        environment.pop(key, None)

    def result(refusal=None, message=""):
        return PreparedEnvironment(
            MappingProxyType(dict(environment)), MappingProxyType(dict(account)), refusal, message
        )

    # Refuse the historical request before authentication or provider setup. A
    # mount alone would not restore its removed hash-bound resume lifecycle.
    if environment.get(RETIRED_READONLY_ENV, "").strip() == "1":
        return result(7, RETIRED_READONLY_MESSAGE)

    from merlin.development.backend.authoring import profiles as author_profiles

    try:
        profile = author_profiles.validate_options(options)
    except ValueError as error:
        return result(7, str(error))
    if profile is None:
        remove(author_profiles.ENV)
    else:
        set_value(author_profiles.ENV, profile.name)
        if not profile.modern:
            for key in tuple(environment):
                if key.startswith("MERLIN_REFERENCE"):
                    remove(key)

    # Only the operator's typed invocation selects this evaluator treatment.
    # Ambient or author-provided environment values cannot opt a run into it.
    mode = options.compiler_execution
    if mode not in {"profile_strict_v1", "snapshot_external_v1"}:
        return result(7, "unknown compiler execution treatment")
    if mode == "snapshot_external_v1" and options.sandbox != "bwrap":
        return result(7, "snapshot_external_v1 requires the isolated bwrap compiler workspace")
    set_value("MERLIN_COMPILER_EXECUTION_MODE", mode)
    from merlin.development.backend.feedback.grading import worker_resources as sandbox_pool

    if getattr(options, "sandbox_reuse", sandbox_pool.PER_CALL) not in sandbox_pool.CHOICES:
        return result(7, "unknown sandbox reuse selection")
    set_value(sandbox_pool.ENV, getattr(options, "sandbox_reuse", sandbox_pool.PER_CALL))
    if options.compiler_access == "full_compiler":
        if mode != "snapshot_external_v1" or not all(
            (options.full_compiler_root, options.full_compiler_manifest, options.full_compiler_manifest_sha256)
        ):
            return result(7, "full compiler access requires an explicit isolated source selection")
        set_value(
            "MERLIN_FULL_COMPILER_SELECTION",
            json.dumps(
                {
                    "root": options.full_compiler_root,
                    "manifest_path": options.full_compiler_manifest,
                    "manifest_sha256": options.full_compiler_manifest_sha256,
                },
                sort_keys=True,
            ),
        )
    else:
        remove("MERLIN_FULL_COMPILER_SELECTION")

    budget = options.qa_timeout if options.model_budget_s is None else options.model_budget_s
    if budget:
        set_value("MERLIN_MODEL_BUDGET_S", str(budget))
    else:
        remove("MERLIN_MODEL_BUDGET_S")
    if library_paths:
        set_value(
            "LD_LIBRARY_PATH",
            os.pathsep.join(map(str, library_paths)) + os.pathsep + environment.get("LD_LIBRARY_PATH", ""),
        )
    if options.provider == "bedrock":
        from merlin.development.backend.providers.model_tiers import resolve

        set_value("CLAUDE_CODE_USE_BEDROCK", "1")
        set_value("AWS_REGION", options.aws_region)
        default("AWS_DEFAULT_REGION", options.aws_region)
        if options.aws_profile:
            set_value("AWS_PROFILE", options.aws_profile)
        if not environment.get("AWS_BEARER_TOKEN_BEDROCK"):
            bearer = machine_defaults.get("AWS_BEARER_TOKEN_BEDROCK")
            if bearer:
                set_value("AWS_BEARER_TOKEN_BEDROCK", bearer)
        if options.subagent_model:
            set_value("CLAUDE_CODE_SUBAGENT_MODEL", resolve(options.subagent_model))
        if options.background_model:
            background = resolve(options.background_model)
            set_value("ANTHROPIC_SMALL_FAST_MODEL", background)
            set_value("ANTHROPIC_DEFAULT_HAIKU_MODEL", background)
        for key, value in {
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "us.anthropic.claude-opus-4-6-v1",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "us.anthropic.claude-sonnet-4-6",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "ANTHROPIC_SMALL_FAST_MODEL": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "CLAUDE_CODE_SUBAGENT_MODEL": "us.anthropic.claude-sonnet-4-6",
        }.items():
            default(key, value)
    else:
        remove("CLAUDE_CODE_USE_BEDROCK")

    if options.experiment == "realistic" and options.arm == "merlin_assisted":
        roots = framework_import_roots()
        previous = environment.get("PYTHONPATH")
        set_value("PYTHONPATH", os.pathsep.join((*roots, *((previous,) if previous else ()))))
        if importlib.util.find_spec("xdsl") is None:
            return result(7, "xdsl is unavailable in the active controller interpreter")

    if options.account_config_dir:
        selected = options.account_config_dir
        if selected == "~" or selected.startswith("~/"):
            # Match expanduser's HOME-then-user-database precedence, but against
            # the prepared baseline rather than another invocation's ambient HOME.
            import pwd

            home = environment.get("HOME")
            if home is None:
                home = pwd.getpwuid(os.getuid()).pw_dir
            selected = home + selected[1:]
        else:
            # Explicit ~user expansion uses the user database, not HOME.
            selected = str(Path(selected).expanduser())
        config = str(Path(selected).absolute())
        set_value("CLAUDE_CONFIG_DIR", config)
        account["config_dir"] = config
        try:
            response = subprocess.run(
                ["claude", "auth", "status"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            status = json.loads(response.stdout)
            account.update({key: status.get(key) for key in ("email", "orgId", "loggedIn")})
            if not status.get("loggedIn"):
                return result(6, "selected account configuration is not logged in")
        except Exception:  # account refusal must not disclose command output or credentials
            return result(6, "could not read selected account authentication status")
    return result()


@contextmanager
def applied_environment(prepared: PreparedEnvironment):
    """Apply the complete prepared baseline; restore the exact pre-context environment.

    A supplied base may differ from ambient state. Auth and authoring must see
    identical environments, including absent keys. Continuation-created or changed
    keys are also restored. Concurrent use is unsupported.
    """
    before = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(prepared.environment)
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)
