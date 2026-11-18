"""Fail-closed boundary between the completed functional Arm-4 run and performance.

The performance campaign is allowed to consume exactly one content-addressed functional
submission.  It copies that submission into a run-private workspace, makes the copy read-only,
records the functional fork, and executes every untrusted package entrypoint in a credential-free,
networkless bwrap sandbox.  Pure helpers in this module make the refusal paths testable without
starting a simulator.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path

import yaml

from merlin.analysis.perf.search.fork import ForkPoint, candidate_states, check_invariants, fork_from
from merlin.common.digest import hash_tree
from merlin.development.backend.optimization.author_interface.prompt_inputs import (
    PerfCell,
    PerfPromptContractError,
)
from merlin.development.targetgen.oracles.schedule import CapsuleState, Verdict
from merlin.development.targetgen.packaging import loading as oot_runner
from merlin.development.targetgen.sandbox import grants as BW
from merlin.development.targetgen.sandbox import toolchain as TC
from merlin.development.targetgen.sandbox.answer_surfaces import (
    AnswerSurface,
    CapsuleMetadata,
    answer_surfaces,
    audit_hit_is_violation,
)
from merlin.target.registry.experiment_descriptor import TargetExperiment

# ---- campaign ----------------------------------------------------------------------------------


def inspect_functional_run(
    run_root: Path, run_id: str, expected_digest: str, *, waive: frozenset[str] | tuple[str, ...] | None = None
) -> FunctionalRun:
    """Validate one explicitly named, fully graded Arm-4 functional run.

    No directory search is performed: the caller supplies both the run ID and the whole-submission
    SHA-256, and all independently recorded digests plus the bytes on disk must agree with it.

    ``waive`` names COMPLETENESS predicates to accept as known gaps rather than refusals, so a
    campaign can be launched against a baseline that is incomplete in a stated way and still produce
    a real measurement. It is a set of predicate NAMES, never a boolean: a blanket force would waive
    whatever failed next, including something nobody looked at. Three rules keep it honest —

    * a predicate in :data:`UNWAIVABLE` cannot be waived at all, and asking is an error (those
      predicates decide whether the numbers mean anything, not how complete they are);
    * a waiver naming a predicate that did NOT fail is an error, because a stale waiver silently
      pre-authorises a future failure nobody has seen;
    * every accepted waiver is recorded on the returned :class:`FunctionalRun` as a
      :class:`Deviation` carrying what was actually observed, and travels with the result.

    Identity and structure (run id, path safety, digest agreement, absent evidence files) are never
    waivable and still raise immediately: without them there is no run to reason about.
    """
    from merlin.development.backend.optimization.campaign.functional_boundary import (
        _WAIVABLE_PREDICATES,
        UNWAIVABLE,
        CampaignGateError,
        Deviation,
        FunctionalRun,
        _digest_excluded_path,
        _graded_ratio,
        _has_symlink,
        _mapping_file,
        _validate_clean_run,
        _validate_scope_coverage,
        _validate_score,
        verify_private_input_snapshot,
    )

    requested = frozenset(waive or ())
    forbidden = sorted(requested & UNWAIVABLE)
    if forbidden:
        raise CampaignGateError(
            f"these gate predicates are UNWAIVABLE and cannot be overridden: {', '.join(forbidden)}. "
            f"They decide whether the measurement means anything (sandbox, answer mask, "
            f"answer-access audit, integrity, public/hidden identity separation), not how complete "
            f"the baseline is. A result produced with one of them waived would not be a weaker "
            f"result, it would be an unattributable one."
        )
    unknown = sorted(requested - _WAIVABLE_PREDICATES)
    if unknown:
        raise CampaignGateError(
            f"unknown gate predicate(s) in waive: {', '.join(unknown)}; "
            f"waivable predicates are: {', '.join(sorted(_WAIVABLE_PREDICATES))}"
        )
    if not run_id:
        raise CampaignGateError("an explicit functional run id is required")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise CampaignGateError("functional run id must be a simple directory name")
    if (
        len(expected_digest) != 64
        or expected_digest.lower() != expected_digest
        or any(c not in "0123456789abcdef" for c in expected_digest)
    ):
        raise CampaignGateError("functional submission digest must be an explicit lowercase SHA-256")

    arm_root = (Path(run_root) / "merlin_assisted").resolve()
    if not (arm_root / run_id).exists() and (Path(run_root) / run_id).is_dir():
        arm_root = Path(run_root).resolve()  # a phase run root: out/runs/<target>/phase1/<run-id>/
    run_dir = arm_root / run_id
    if run_dir.is_symlink() or not run_dir.is_dir() or arm_root not in run_dir.resolve().parents:
        raise CampaignGateError(f"explicit functional run does not resolve safely: {run_dir}")
    run_dir = run_dir.resolve()
    submission = run_dir / "submission"
    if not submission.is_dir():
        raise CampaignGateError(f"functional submission is absent: {submission}")
    linked = _has_symlink(submission)
    if linked is not None:
        raise CampaignGateError(f"functional submission contains a live symlink: {linked}")
    excluded = _digest_excluded_path(submission)
    if excluded is not None:
        raise CampaignGateError(f"functional submission contains a digest-excluded path: {excluded}")

    environment = _mapping_file(run_dir / "environment.yaml", yaml_file=True)
    if environment.get("validation_scope", "full") == "capsules":
        raise CampaignGateError("capsule-only publication requires explicit capsule_iteration admission")
    verify_private_input_snapshot(run_dir, environment)
    summary = _mapping_file(run_dir / "qa_loop_summary.yaml", yaml_file=True)
    freeze = _mapping_file(run_dir / "freeze.json")
    manifest = _mapping_file(run_dir / "run_manifest.yaml", yaml_file=True)
    public_score = _mapping_file(run_dir / "grading_public" / "score_capsule.json")
    hidden_score = _mapping_file(run_dir / "grading_hidden" / "score_capsule.json")

    if environment.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise CampaignGateError("functional evidence names a different run id")
    profile = environment.get("compiler_tool_profile")
    if profile is None:
        if not str(environment.get("bundle_id") or "").startswith("merlin_assisted_rtlchecks_"):
            raise CampaignGateError("functional run is not from the historical Arm-4 RTL-checks bundle")
    found: list[Deviation] = list(_validate_clean_run(environment, summary))
    if manifest.get("integrity_status") != "clean" or manifest.get("integrity_exempt") is not False:
        found.append(
            Deviation(
                "manifest_integrity_gate_failed",
                "functional run did not pass the manifest integrity gate "
                f"(integrity_status={manifest.get('integrity_status')!r}, "
                f"integrity_exempt={manifest.get('integrity_exempt')!r})",
            )
        )
    if manifest.get("gradeable") is not True:
        found.append(
            Deviation(
                "manifest_not_gradeable",
                f"functional run manifest is not gradeable (gradeable={manifest.get('gradeable')!r})",
            )
        )
    public = manifest.get("public_dev") or {}
    hidden = manifest.get("hidden") or {}
    if public.get("functional_pass") != 1 or hidden.get("functional_pass") != 1:
        found.append(
            Deviation("phase_functional_pass_missing", "functional run did not pass both public and hidden grades")
        )
    if public.get("highest_tier") != "L3":
        found.append(
            Deviation(
                "public_tier_below_cert",
                f"functional public run did not reach the required L3 tier "
                f"(highest_tier={public.get('highest_tier')!r})",
            )
        )
    n_public, _pub_ratio_found = _graded_ratio(public.get("passed"), label="public functional grade")
    n_hidden, _hid_ratio_found = _graded_ratio(hidden.get("passed"), label="hidden functional grade")
    found += _pub_ratio_found + _hid_ratio_found
    task_scope = environment.get("task_scope")
    if not isinstance(task_scope, dict):
        raise CampaignGateError("functional run lacks its sealed task scope")
    found += _validate_scope_coverage(task_scope, public_score, hidden_score, n_public=n_public, n_hidden=n_hidden)
    found += _validate_score(public_score, label="public functional grade", expected=n_public)
    found += _validate_score(hidden_score, label="hidden functional grade", expected=n_hidden)
    _pub_rows = public_score.get("per_capsule") if isinstance(public_score, dict) else None
    _hid_rows = hidden_score.get("per_capsule") if isinstance(hidden_score, dict) else None
    if isinstance(_pub_rows, list) and isinstance(_hid_rows, list):
        _shared = {str(r["capsule"]) for r in _pub_rows if isinstance(r, dict) and r.get("capsule")} & {
            str(r["capsule"]) for r in _hid_rows if isinstance(r, dict) and r.get("capsule")
        }
        if _shared:
            found.append(
                Deviation(
                    "capsule_identity_reused",
                    f"public and hidden grades reuse a capsule identity: {', '.join(sorted(_shared)[:4])}",
                )
            )
    if freeze.get("workspace_mutable_after_freeze") is not False or not freeze.get("frozen_at"):
        found.append(
            Deviation("freeze_not_immutable", "functional freeze did not record an immutable workspace and timestamp")
        )

    # Refuse everything not explicitly named, and name ALL of it in one message.
    blocking = [d for d in found if d.predicate not in requested]
    if blocking:
        _lines = "; ".join(f"[{d.predicate}] {d.detail}" for d in blocking)
        raise CampaignGateError(f"functional run refused by {len(blocking)} gate predicate(s): {_lines}")
    stale = sorted(requested - {d.predicate for d in found})
    if stale:
        raise CampaignGateError(
            f"waive names predicate(s) that did not fail: {', '.join(stale)}. A waiver for a "
            f"predicate nobody saw fail would silently pre-authorise a future failure; drop it."
        )

    observed = hash_tree(submission)["sha256"]
    recorded = {
        "requested": expected_digest,
        "run_manifest": manifest.get("submission_sha256"),
        "freeze": freeze.get("submission_sha256"),
        "freeze_recheck": freeze.get("submission_sha256_recheck"),
        "submission_tree": observed,
    }
    if any(value != expected_digest for value in recorded.values()):
        raise CampaignGateError(f"explicit functional submission digest does not match every record: {recorded}")
    return FunctionalRun(
        deviations=tuple(sorted(found, key=lambda d: d.predicate)),
        run_dir=run_dir,
        submission_dir=submission.resolve(),
        run_id=run_id,
        digest=expected_digest,
        public_capsules=n_public,
        hidden_capsules=n_hidden,
        public_score=public_score,
        hidden_score=hidden_score,
        frozen_at=str(freeze.get("frozen_at") or manifest.get("frozen_at") or ""),
    )


def materialize_perf_workspace(
    record: FunctionalRun, perf_root: Path, *, published_root: Path | None = None, target: str | None = None
) -> Path:
    """Copy the exact functional package into a new performance-only workspace."""
    from merlin.development.backend.optimization.campaign.functional_boundary import CampaignGateError, FunctionalRun

    perf_root = Path(perf_root).resolve()
    workspace = perf_root / "workspace"
    snapshot = workspace / "submission"
    if published_root is None:
        observed = materialize_readonly_tree(record.submission_dir, snapshot)
    else:
        if target is None:
            raise CampaignGateError("published compiler selection requires the admitted target")
        from merlin.development.backend.optimization.campaign.package_closure import inspect, materialize

        selected = inspect(record, published_root, target=target)
        materialize(record, selected, snapshot)
        observed = hash_tree(snapshot)["sha256"]
    if observed != record.digest:
        raise CampaignGateError(f"copied performance submission digest {observed} does not match {record.digest}")
    return snapshot


def materialize_readonly_tree(source: Path, snapshot: Path) -> str:
    """Copy a symlink-free, fully hashed tree once and make the destination read-only."""
    from merlin.development.backend.optimization.campaign.functional_boundary import (
        CampaignGateError,
        _digest_excluded_path,
        _has_symlink,
    )

    source = Path(source).resolve()
    snapshot = Path(snapshot).resolve()
    if not source.is_dir():
        raise CampaignGateError(f"immutable campaign source is not a directory: {source}")
    if snapshot.exists() or snapshot.is_symlink():
        raise CampaignGateError(f"immutable campaign snapshot already exists: {snapshot}")
    linked = _has_symlink(source)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign source contains a live symlink: {linked}")
    excluded = _digest_excluded_path(source)
    if excluded is not None:
        raise CampaignGateError(f"immutable campaign source contains a digest-excluded path: {excluded}")
    before = hash_tree(source)["sha256"]
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, snapshot, symlinks=False)
    linked = _has_symlink(snapshot)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign snapshot contains a symlink: {linked}")
    observed = hash_tree(snapshot)["sha256"]
    if observed != before:
        raise CampaignGateError(f"immutable campaign copy changed during materialization: {before} != {observed}")
    for path in sorted(snapshot.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    snapshot.chmod(0o555)
    return str(observed)


def functional_fork(record: FunctionalRun) -> ForkPoint:
    """Convert the public+hidden L2/L3 evidence into the immutable Phase-F fork record."""
    from merlin.development.backend.optimization.campaign.functional_boundary import CampaignGateError, FunctionalRun

    states: list[CapsuleState] = []
    tiers: set[str] = set()
    for score in (record.public_score, record.hidden_score):
        for row in score["per_capsule"]:
            passed = {
                str(tier): Verdict(str(status), record.digest)
                for tier, status in (row.get("tiers") or {}).items()
                if status == "pass"
            }
            tiers.update(passed)
            states.append(CapsuleState(str(row["capsule"]), record.digest, verdicts=passed))
    if not states or not tiers:
        raise CampaignGateError("cannot fork from an empty functional verdict set")
    return fork_from(states, tier_order=sorted(tiers), digest=record.digest, recorded_at=record.frozen_at or None)


def check_fork(fork: ForkPoint, snapshot: Path):
    """Check the copied compiler's current bytes against every Phase-F invariant."""
    digest = hash_tree(Path(snapshot))["sha256"]
    states = candidate_states(fork, digest=digest)
    return check_invariants(fork, states, provenance=fork.provenance)


def _remove_agent_home_mounts(argv: Sequence[str]) -> list[str]:
    """Remove the agent-launch credential bind from a package-only bwrap prefix.

    ``base_argv`` also serves paid agent launches and therefore binds ``~/.claude``.  A compiled
    submission never needs that state.  Strip both that bind and its nested projects mask, then clear
    the inherited environment so API/provider credentials cannot enter through environment variables.
    """
    from merlin.development.backend.optimization.campaign.functional_boundary import CampaignGateError

    home_claude = str(Path(os.path.expanduser("~/.claude")))
    out: list[str] = []

    def is_agent_home_path(value: str) -> bool:
        path = Path(value)
        home = Path(home_claude)
        return path == home or home in path.parents

    i = 0
    while i < len(argv):
        if argv[i] in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try") and i + 2 < len(argv):
            if is_agent_home_path(argv[i + 2]):
                i += 3
                continue
        if argv[i] == "--tmpfs" and i + 1 < len(argv) and is_agent_home_path(argv[i + 1]):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if any(is_agent_home_path(value) for value in out if value.startswith("/")):
        raise CampaignGateError("package sandbox still exposes the agent credential directory")
    return out


def package_sandbox_policy(
    te: TargetExperiment,
    workspace: Path,
    package: Path,
    *,
    inputs: PackageSandboxInputs | FrozenPackageSandboxInputs | None = None,
    scope: PackagePolicyScope | None = None,
) -> PackageSandboxPolicy:
    """Derive a credential-free, answer-closed bwrap policy for untrusted package entrypoints.

    ``scope`` shares one execution scope's parsed capsule metadata between calls; the resulting
    policy is identical to one derived without it.
    """
    from merlin.development.backend.optimization.campaign.functional_boundary import (
        CampaignGateError,
        FrozenPackageSandboxInputs,
        PackagePolicyScope,
        PackageSandboxInputs,
        PackageSandboxPolicy,
        select_package_sandbox_inputs,
    )

    workspace = Path(workspace).resolve()
    package = Path(package).resolve()
    if not workspace.is_dir() or not package.is_dir() or workspace == package:
        raise CampaignGateError("performance sandbox workspace and copied package must be distinct directories")
    from merlin.development.backend.feedback.execution.profile_runtime import _original_package_composition

    composition = _original_package_composition(te, inputs, scope)
    # Empty bundle is intentional: the package sees its input files and the derived toolchain, not the
    # experiment repository.  The policy-only escape avoids requiring an agent-input snapshot when no
    # bundle grants exist and does not expose any live input itself.
    if isinstance(inputs, FrozenPackageSandboxInputs):
        from merlin.development.backend.optimization.campaign.package_closure import restore

        inputs = restore(inputs.root, inputs.record)
    base_options = (
        {}
        if inputs is None
        else {"repo": inputs.repo if isinstance(inputs, FrozenPackageSandboxInputs) else inputs.paths.repo}
    )
    argv = BW.base_argv(workspace, {}, _policy_test_live_inputs=True, **base_options)
    if isinstance(inputs, FrozenPackageSandboxInputs):
        # No resolver service state belongs to this networkless frozen tool closure.
        for option in ("--ro-bind", "--ro-bind-try"):
            resolver = [option, TC.RESOLVE_DIR, TC.RESOLVE_DIR]
            for i in range(len(argv) - 3, -1, -1):
                if argv[i : i + 3] == resolver:
                    del argv[i : i + 3]
    argv = _remove_agent_home_mounts(argv)
    argv += ["--unshare-net", "--clearenv", "--setenv", "HOME", "/tmp", "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    if isinstance(inputs, FrozenPackageSandboxInputs):
        argv += list(inputs.argv)
        surfaces = inputs.surfaces
    elif inputs is None:
        # An exact original owner closes all mounts below in one final pass.
        # Resolve its tools here; discover the live answers only after its grants.
        discovered = (
            select_package_sandbox_inputs(te, scope=scope)
            if composition is None
            else PackageSandboxInputs(TC.ToolchainPaths.from_checkout(), TC._sim(te), TC.curated_harness_dir(te), ())
        )
        selected = {
            "paths": discovered.paths,
            "sim": discovered.sim,
            "harness": discovered.harness,
        }
        argv += TC.toolchain_binds(te, **selected)
        surfaces = discovered.surfaces
    else:
        selected = {"paths": inputs.paths, "sim": inputs.sim, "harness": inputs.harness}
        surfaces = inputs.surfaces
        memory_dir = next((str(surface.path) for surface in surfaces if surface.origin == "memory"), "")
        argv += TC.toolchain_binds(te, **selected, memory_dir=memory_dir)
    argv = _remove_agent_home_mounts(argv)
    probes = (
        inputs.probes
        if isinstance(inputs, FrozenPackageSandboxInputs)
        else tuple(TC.required_tool_probes(te, paths=selected["paths"], sim=selected["sim"]))
    )
    if not probes:
        raise CampaignGateError("package sandbox derives zero required tool probes")
    prefix = (
        inputs.env_prefix
        if isinstance(inputs, FrozenPackageSandboxInputs)
        else TC.sandbox_env(te, workspace, **selected)
    )
    if composition is None:
        argv = BW.apply_answer_masks(argv, surfaces)
        argv += ["--ro-bind", str(package), str(package)]
    else:
        argv, prefix, current = composition(
            [*argv, "--ro-bind", str(package), str(package)],
            prefix,
            target=te,
            scope=scope,
            selected_surfaces=surfaces,
        )
        surfaces = (*surfaces, *current)
        if inputs is None:
            discovered = replace(discovered, surfaces=current)
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise CampaignGateError(f"package sandbox exposes derived answer surfaces: {gaps}")
    policy = PackageSandboxPolicy(
        tuple(argv),
        gaps,
        probes,
        workspace,
        package,
        te,
        prefix,
        inputs if isinstance(inputs, FrozenPackageSandboxInputs) else None,
        discovered if inputs is None else inputs,
    )
    from merlin.development.backend.feedback.execution.profile_runtime import active_public_package_policy
    from merlin.development.backend.optimization.campaign.compiler_libraries import (
        apply as additional_library_policy,
    )

    return additional_library_policy(
        active_public_package_policy(policy, target=te, scope=scope) if composition is None else policy
    )


def _verify_frozen_policy(policy: PackageSandboxPolicy) -> None:
    from merlin.development.backend.optimization.campaign.functional_boundary import (
        CampaignGateError,
        PackageSandboxPolicy,
    )

    if policy.compiler_fork_sha256 is not None:
        from merlin.development.targetgen.authoring.library_admission import active_compiler_library, candidate_sources

        selection = active_compiler_library()
        if selection is None or candidate_sources(selection, policy.package)[1] != policy.compiler_fork_sha256:
            raise CampaignGateError("candidate compiler fork changed before execution")
    if policy.frozen_inputs is not None:
        from merlin.development.backend.optimization.campaign.package_closure import restore

        verified = restore(policy.frozen_inputs.root, policy.frozen_inputs.record)
        from merlin.development.backend.feedback.execution.profile_runtime import active_public_package_policy

        expected = active_public_package_policy(
            replace(policy, env_prefix=verified.env_prefix), target=policy.target_experiment
        )
        from merlin.development.backend.optimization.campaign.compiler_libraries import (
            apply as additional_library_policy,
        )

        expected = additional_library_policy(expected)
        if (
            verified != policy.frozen_inputs
            or policy.required_tools != verified.probes
            or policy.env_prefix != expected.env_prefix
        ):
            raise CampaignGateError("package execution policy differs from verified frozen inputs")


def run_tool_probes(policy: PackageSandboxPolicy, *, timeout: int = 60) -> list[dict]:
    """Run every descriptor-derived tool probe in the exact package sandbox."""
    from merlin.development.backend.optimization.campaign.functional_boundary import (
        CampaignGateError,
        PackageSandboxPolicy,
    )

    rows: list[dict] = []
    for probe in policy.required_tools:
        _verify_frozen_policy(policy)
        proc = subprocess.run(
            [*policy.argv, "bash", "-c", policy.env_prefix + probe.cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        row = {
            "label": probe.label,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-400:],
            "stderr": (proc.stderr or "")[-400:],
        }
        rows.append(row)
        if proc.returncode != 0:
            raise CampaignGateError(f"required sandbox tool probe {probe.label!r} failed with rc={proc.returncode}")
    if len(rows) != len(policy.required_tools):
        raise CampaignGateError("not every required package-sandbox tool probe ran")
    return rows


def _inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


_ACTIVE = ContextVar("performance_package_sandbox", default=None)

_PATCH_LOCK = threading.Lock()

_PATCH_DEPTH = 0

_ORIGINALS: dict = {}


def active_sandbox_policy() -> PackageSandboxPolicy | None:
    """The policy boxing the calling thread, or None when it is not inside one."""
    from merlin.development.backend.optimization.campaign.functional_boundary import PackageSandboxPolicy

    return _ACTIVE.get()


@contextlib.contextmanager
def boxed_entrypoints(policy: PackageSandboxPolicy) -> Iterator[None]:
    """Route oot_runner's untrusted package execution through ``policy`` for the CALLING THREAD.

    Safe to enter from several threads at once: each one sees its own policy, and the module-level
    patch is installed once and removed only when the last of them leaves.
    """
    from merlin.development.backend.optimization.campaign.functional_boundary import PackageSandboxPolicy

    global _PATCH_DEPTH

    token = _ACTIVE.set(policy)
    with _PATCH_LOCK:
        if _PATCH_DEPTH == 0:
            _ORIGINALS["run_entrypoint"] = oot_runner.run_entrypoint
            _ORIGINALS["build_package"] = oot_runner.build_package
            oot_runner.build_package = _refuse_build
            oot_runner.run_entrypoint = _run_boxed
        _PATCH_DEPTH += 1
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        with _PATCH_LOCK:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0:
                oot_runner.run_entrypoint = _ORIGINALS["run_entrypoint"]
                oot_runner.build_package = _ORIGINALS["build_package"]


def _refuse_build(pkg, *, timeout: int = 1800) -> None:
    from merlin.development.backend.optimization.campaign import functional_guard_execution
    from merlin.development.backend.optimization.campaign.functional_boundary import CampaignGateError

    active = getattr(oot_runner, "active_package_executor", None)
    if callable(active) and active() is not None and not functional_guard_execution.owns_executor(active()):
        return _ORIGINALS["build_package"](pkg, timeout=timeout)
    build = pkg.manifest.get("build") or {}
    if any(build.get(key) for key in ("configure", "command")):
        raise CampaignGateError("Arm-4 performance package declares an untrusted build step; no host build is allowed")


def _run_boxed(
    pkg,
    name: str,
    input_mlir: Path,
    output_json: Path | None = None,
    *,
    timeout: int = 600,
    write_bytecode: bool = False,
    artifact_profile=None,
    invocation_directory: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run one untrusted package entrypoint inside the CALLING THREAD's sandbox policy.

    Refusing when no policy is set is the whole point of reading it from the thread rather than from
    a closure: an invocation that arrives on a thread nobody boxed must not fall through to an
    unsandboxed execution, which is exactly what the previous save/restore could produce.
    """
    from merlin.development.backend.optimization.campaign import (
        functional_guard_execution,
        normal_resource_launch,
    )
    from merlin.development.backend.optimization.campaign.functional_boundary import CampaignGateError

    active = getattr(oot_runner, "active_package_executor", None)
    if callable(active) and active() is not None and not functional_guard_execution.owns_executor(active()):
        return _ORIGINALS["run_entrypoint"](
            pkg,
            name,
            input_mlir,
            output_json,
            timeout=timeout,
            write_bytecode=write_bytecode,
            artifact_profile=artifact_profile,
            invocation_directory=invocation_directory,
        )
    if write_bytecode or artifact_profile is not None:
        raise CampaignGateError("legacy package box has no reviewed artifact-profile or bytecode service")
    policy = active_sandbox_policy()
    if policy is None:
        raise CampaignGateError("untrusted package execution reached a thread that is not inside a sandbox policy")
    _verify_frozen_policy(policy)
    if pkg.directory.resolve() != policy.package:
        raise CampaignGateError("capsule runner attempted to execute a package outside the snapshot")
    if functional_guard_execution.active_guard() is not None and Path(input_mlir).is_symlink():
        raise CampaignGateError("functional guard compiler input is linked")
    if (
        functional_guard_execution.active_guard() is not None
        and output_json is not None
        and Path(output_json).is_symlink()
    ):
        raise CampaignGateError("functional guard compiler output destination is linked")
    input_mlir = Path(input_mlir).resolve()
    output_json = Path(output_json).resolve() if output_json is not None else None
    if not _inside(input_mlir, policy.workspace):
        raise CampaignGateError("untrusted package input is outside the performance workspace")
    if output_json is not None and not _inside(output_json, policy.workspace):
        raise CampaignGateError("untrusted package output is outside the performance workspace")
    if functional_guard_execution.active_guard() is not None:
        return functional_guard_execution.execute_isolated(
            policy, pkg, name, input_mlir, output_json, timeout=timeout, invocation_directory=invocation_directory
        )
    from merlin.development.backend.feedback.execution.profile_runtime import active_profile_service

    original_service = active_profile_service()
    if original_service is not None:
        # An original profile call sees exactly one input and output staging,
        # never sibling oracle or simulation records in the performance workspace.
        from merlin.development.backend.feedback.execution.profile_executor import _Executor

        if policy.selected_inputs is None:
            raise CampaignGateError("original profile call lost its explicit package runtime selection")
        with tempfile.TemporaryDirectory(prefix="profile-call-", dir=policy.workspace.parent) as raw:
            executor = _Executor(
                original_service,
                policy.target_experiment,
                pkg.directory,
                Path(raw),
                sandbox_inputs=policy.selected_inputs,
            )
            return executor.run_entrypoint(
                pkg, name, input_mlir, output_json, timeout=timeout, invocation_directory=invocation_directory
            )
    argv = oot_runner._resolve_argv(pkg, name, input_mlir, output_json)
    if oot_runner._needs_interpreter(pkg, argv):
        argv = [sys.executable, *argv]
    shell = policy.env_prefix + 'exec "$@"'
    command = [*policy.argv, "--chdir", str(policy.package), "bash", "-c", shell, "perf-package", *argv]
    boundary = normal_resource_launch.resources.active_resource_boundary()
    with normal_resource_launch.bounded_normal_scratch(
        policy.argv, owner=invocation_directory or policy.workspace, boundary=boundary
    ) as prefix:
        command = [*prefix, *command[len(policy.argv) :]]
        if invocation_directory is None:
            return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        from merlin.common.process import invocation_records as invocation_record

        return invocation_record.run(
            command,
            directory=invocation_directory,
            stage=name,
            inputs=(input_mlir,),
            outputs=(output_json,) if output_json else (),
            dependencies=(Path(__file__),),
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def completion_report(results: Sequence[Mapping], expected: Sequence[PerfCell]) -> dict:
    """Validate exact Phase-P cells without treating functional-simulator cycles as performance."""
    from merlin.development.backend.optimization.campaign.functional_boundary import CampaignGateError

    expected_rows = tuple(expected)
    for identity in expected_rows:
        if not isinstance(identity, PerfCell):
            raise CampaignGateError("performance expected identity must be a PerfCell")
        try:
            identity.validate()
        except PerfPromptContractError as exc:
            raise CampaignGateError(f"invalid expected performance identity: {exc}") from exc
    expected_cells = set(expected_rows)
    if not expected_cells:
        raise CampaignGateError("performance campaign has zero expected Arm-4 cells")
    if len(expected_cells) != len(expected_rows):
        raise CampaignGateError("performance expected cell identities contain duplicates")
    by_identity: dict[PerfCell, Mapping] = {}
    for row in results:
        if not isinstance(row, Mapping):
            raise CampaignGateError("performance result must be a mapping with an exact identity")
        identity = PerfCell(row.get("family"), row.get("capsule"), row.get("simulator"), row.get("replicate"))
        try:
            identity.validate()
        except PerfPromptContractError as exc:
            raise CampaignGateError(f"invalid reported performance identity: {exc}") from exc
        if identity in by_identity:
            raise CampaignGateError(f"performance results repeat cell identity {identity.label!r}")
        by_identity[identity] = row

    extras = sorted(set(by_identity) - expected_cells)
    if extras:
        raise CampaignGateError(
            f"performance results contain unexpected cell identities: {[cell.label for cell in extras]}"
        )

    reported = correct = cycles_measured = 0
    failed = 0
    for identity in sorted(expected_cells):
        cell = by_identity.get(identity)
        if cell is None:
            continue
        reported += 1
        is_correct = cell.get("correct") is True
        cycles = cell.get("cycles")
        has_cycles = (
            identity.simulator == "verilator"
            and isinstance(cycles, int)
            and not isinstance(cycles, bool)
            and cycles > 0
        )
        if identity.simulator == "spike":
            valid = is_correct and cycles is None
        else:
            provenance = cell.get("provenance") or {}
            valid = (
                is_correct
                and has_cycles
                and isinstance(provenance, Mapping)
                and provenance.get("tier") == "L3"
                and provenance.get("simulator") == "verilator"
                and provenance.get("derived_from_rtl") is True
                and provenance.get("cycle_accurate") is True
            )
        correct += int(is_correct)
        cycles_measured += int(has_cycles)
        failed += int(not valid)
    missing = len(expected_cells) - reported
    counts = {
        "expected": len(expected_cells),
        "reported": reported,
        "correct": correct,
        "cycles_measured": cycles_measured,
        "failed": failed,
        "missing": missing,
        "complete": not missing and not failed,
    }
    return counts


def completion_counts(results: Sequence[Mapping], expected: Sequence[PerfCell]) -> dict:
    """Require every exact Arm-4 Phase-P identity to carry valid simulator-specific evidence."""
    from merlin.development.backend.optimization.campaign.functional_boundary import CampaignGateError

    counts = completion_report(results, expected)
    if not counts["complete"]:
        raise CampaignGateError(
            f"Arm-4 performance reported {counts['reported']} of {counts['expected']} expected cells; "
            f"{counts['failed']} reported cell(s) failed simulator-specific completion evidence"
        )
    return counts
