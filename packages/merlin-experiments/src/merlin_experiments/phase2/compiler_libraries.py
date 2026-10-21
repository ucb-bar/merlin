"""Additional reviewed compiler leaves in the existing normal package sandbox.

This changes a Phase 2 compiler's explicit library environment, never the Phase
1 profile or independent evaluator. Every compared arm must select the same
release and pass the original functional gates in that environment.
"""

from __future__ import annotations

import hashlib
import shlex
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import wraps
from pathlib import Path, PurePosixPath

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.compiler_library import CompilerLibrarySelection, active_compiler_library, scoped_compiler_library
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    active_compiler_passes,
    scoped_compiler_passes,
)
from merlin.targetgen.compiler_profile_admission import active_compiler_imports
from merlin_experiments.phase1.feedback.compiler_pass_delegation import scoped_selected_parent_pass_delegation

from .contracts import StageGateError


def stage_selection(function):
    """Host-only stage API; never inherit a release from an unbound caller."""

    @wraps(function)
    def call(*args, **kwargs):
        selection = kwargs.get("compiler_library_selection")
        passes = kwargs.get("compiler_pass_selection")
        if selection is not None and type(selection) is not CompilerLibrarySelection:
            raise StageGateError("author stage requires an explicit reviewed compiler library owner")
        if selection is None and active_compiler_library() is not None:
            raise StageGateError("author stage cannot inherit an unbound additional compiler library")
        if selection is None and kwargs.get("diagnostic_library_stage"):
            raise StageGateError("diagnostic library stage requires its explicit reviewed release")
        if passes is not None and type(passes) is not CompilerPassSelection:
            raise StageGateError("author stage requires original live compiler pass selection")
        if passes is None and active_compiler_passes() is not None:
            raise StageGateError("author stage cannot inherit unbound compiler pass selection")
        with (scoped_compiler_library(selection), scoped_compiler_passes(passes),
              scoped_selected_parent_pass_delegation(passes)):
            return function(*args, **kwargs)

    return call


@contextmanager
def original_agent_sources(functional):
    """Close one public-mount metadata operation before any execution.

    Provenance is independently verified first. The original payload transaction
    coalesces only repeated import-admission getters while ``apply_agent`` builds
    its mount list; both payload boundaries remain fresh. No probe, compiler,
    agent or child wait occurs in this scope, and it retains no parent CPU leaf.
    """
    if active_compiler_library() is None or functional is None:
        yield
        return
    from merlin.common.bundle_snapshot import verified_snapshot_scope
    from merlin.targetgen.compiler_profile_admission import original_compiler_imports

    from . import functional_inputs

    if functional.host_provenance is None:
        raise StageGateError("additional author library requires the original frozen source provenance")
    verified = functional_inputs._verify_private_functional_provenance(functional.host_provenance)
    if verified["bundle"].get("compiler_tool_profile") is None:
        yield
        return
    with (
        verified_snapshot_scope(verified["workspace"], verified["bundle"], repo=verified["repo"]),
        original_compiler_imports(
            workspace=verified["workspace"],
            bundle_manifest=Path(functional.host_provenance["bundle_manifest"]),
            repo=verified["repo"],
        ),
    ):
        yield


def apply_agent(policy):
    """Exact leaf overlay shared by author, tool broker and policy recheck."""
    with original_agent_sources(policy.compiler_library_functional):
        return apply(policy)


def prompt_inputs(inputs):
    """Advertise exact selected guest leaves through the ordinary prompt facts."""
    selection = active_compiler_library()
    if selection is None:
        return inputs
    root = guest_root(selection)
    from .compiler_library_guidance import selected_inventory

    return replace(
        inputs,
        **(
            {"additional_compiler_library": selected_inventory()}
            if "additional_compiler_library" in inputs.__dataclass_fields__
            else {}
        ),
        allowed_paths=tuple(
            dict.fromkeys(
                (
                    *inputs.allowed_paths,
                    *(str(root / member.path) for member in selection.contract.members),
                )
            )
        ),
    )


def paired_functional_guard(function):
    """Requalify both arms in the newly selected library environment."""

    @wraps(function)
    def call(baseline, candidate, *args, **kwargs):
        selection = active_compiler_library()
        passes = active_compiler_passes()
        if selection is None and passes is None:
            return function(baseline, candidate, *args, **kwargs)
        record = None if selection is None else selection.record()
        baseline_kwargs = dict(kwargs)
        output = baseline_kwargs.get("grade_root")
        if output is not None:
            baseline_kwargs["grade_root"] = Path(output).with_name(Path(output).name + "_library_baseline")
        baseline_result = function(baseline, baseline, *args, **baseline_kwargs)
        result = function(baseline, candidate, *args, **kwargs)
        if selection is not None and selection.record() != record:
            raise StageGateError("additional compiler library changed during final functional qualification")
        result = {**result, "additional_compiler_library": record, "library_baseline_guard": baseline_result}
        if passes is not None:
            result["compiler_pass_selection"] = passes.record()
            try:
                result["compiler_pass_realizations"] = passes.check_functional_guard(
                    baseline, candidate, baseline_result, result, grade_root=output
                )
            except ValueError as error:
                result["status"] = "offending"
                result["offenders"] = [
                    *result.get("offenders", ()),
                    {"kind": "compiler_pass_final_realization_unavailable", "reason": str(error)},
                ]
        if baseline_result.get("status") != "clean":
            result["status"] = "offending"
            result["offenders"] = [*result.get("offenders", ()), {"kind": "additional_library_baseline_unqualified"}]
        return result

    return call


@contextmanager
def _policy_scopes(policy, passes):
    # Scope entry/exit reopens original owners, but the compiler children need
    # their own grants. Release this metadata leaf before invoking the body.
    stack = ExitStack()
    try:
        with shared_compute_pool(timeout_s=120):
            stack.enter_context(scoped_compiler_library(policy._compiler_library_selection))
            stack.enter_context(scoped_compiler_passes(passes))
            stack.enter_context(scoped_selected_parent_pass_delegation(passes))
        yield
    finally:
        unwound = False
        try:
            with shared_compute_pool(timeout_s=120):
                unwound = True
                stack.__exit__(*sys.exc_info())
        finally:
            if not unwound:
                # Failed admission must still revoke the thread's scopes.
                # Inject the refusal so no successful postcheck is claimed.
                stack.__exit__(*sys.exc_info())


def scoped_policy(function):
    """Carry a retained policy's fixed selection into broker and queue threads."""

    @wraps(function)
    def call(policy, *args, **kwargs):
        if policy._compiler_library_selection is None and active_compiler_library() is not None:
            raise StageGateError("ordinary policy cannot inherit an unbound additional compiler library")
        passes = getattr(policy, "_compiler_pass_selection", None)
        if passes is None and active_compiler_passes() is not None:
            raise StageGateError("ordinary policy cannot inherit unbound compiler pass selection")
        with _policy_scopes(policy, passes):
            return function(policy, *args, **kwargs)

    return call


def _original_sources(owner):
    """Reopen exact public Python owners; do not copy a directory grant."""
    from merlin.targetgen import tool_registry
    from merlin.targetgen.sandbox import bwrap as BW

    if owner is None:
        return {}
    owner.verify()
    _, grants = BW._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
    rows = list(owner.profile["common_sources"])
    for name in owner.profile["treatment_tools"]:
        if name in {"isa_dialect_generation", "lean_proof_authoring"}:
            rows.extend({"path": path} for path in tool_registry.spec(name).bundle_paths if path.endswith(".py"))
    selected = {}
    for row in rows:
        relative = row["path"].removeprefix("merlin/python/")
        matches = [
            source
            for declared, _destination, source in grants
            if declared == row["path"] or declared.endswith("/" + relative)
        ]
        if len(matches) != 1:
            raise StageGateError("additional library cannot resolve an original public Python owner")
        path = PurePosixPath(relative)
        source = matches[0]
        if (
            path.is_absolute()
            or str(path) != relative
            or ".." in path.parts
            or path.parts[0] != "merlin"
            or source.is_symlink()
            or not source.is_file()
            or row.get("sha256") is not None
            and hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise StageGateError("additional library lost original public source identity")
        selected[relative] = source
    return selected


def guest_root(selection):
    selection.record()
    return Path("/perf-compiler-library") / selection.contract.sha256 / "python"


def environment_prefix(prefix):
    """Derive the same selected import path for policy verification and launch."""
    selection = active_compiler_library()
    if selection is None:
        return prefix
    suffix = "export PYTHONPATH=" + shlex.quote(str(guest_root(selection))) + ":$PYTHONPATH; "
    return prefix if prefix.endswith(suffix) else prefix + suffix


def apply(policy):
    """Add exact readonly leaves to an already answer-masked normal policy.

    Namespace initializers and overlapping compiler algorithms come from the
    explicitly selected Phase 2 release. Other original public leaves remain
    available in that same Python tree. No private grader tree is mounted.
    """
    selection = active_compiler_library()
    if selection is None:
        return policy
    record = selection.record()
    package = getattr(policy, "package", None)
    if package is not None and (selection.root.is_relative_to(package) or Path(package).is_relative_to(selection.root)):
        raise StageGateError("additional compiler library cannot be candidate-owned")
    root = guest_root(selection)
    sources = _original_sources(active_compiler_imports())
    sources.update({member.path: selection.root / member.path for member in selection.contract.members})
    directories = {Path("/perf-compiler-library"), root.parent, root}
    for relative in sources:
        destination = root / relative
        directories.update(parent for parent in destination.parents if parent == root or parent.is_relative_to(root))
    mounts = []
    for directory in sorted(directories, key=lambda path: (len(path.parts), str(path))):
        mounts += ["--dir", str(directory)]
    for relative, source in sorted(sources.items()):
        mounts += ["--ro-bind", str(source), str(root / relative)]
    prefix = list(policy.argv)
    # Frozen policy verification can reapply its own exact selection. A prior
    # library tree is otherwise never accepted as an unverified mount overlay.
    positions = [i for i in range(len(prefix) - len(mounts) + 1) if prefix[i : i + len(mounts)] == mounts]
    namespace = "/perf-compiler-library"
    outside = prefix if not positions else prefix[: positions[0]] + prefix[positions[0] + len(mounts) :]
    if any(part == namespace or part.startswith(namespace + "/") for part in outside):
        raise StageGateError("additional compiler library mount selection changed: foreign or overriding namespace")
    if not positions:
        prefix += mounts
    elif len(positions) != 1:
        raise StageGateError("additional compiler library mount selection is ambiguous")
    if selection.record() != record:
        raise StageGateError("additional compiler library changed while deriving package mounts")
    return replace(policy, argv=tuple(prefix), env_prefix=environment_prefix(policy.env_prefix or ""))
