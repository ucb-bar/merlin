"""Additional reviewed compiler leaves in the existing normal package sandbox.

This changes a Phase 2 compiler's explicit library environment, never the Phase
1 profile or independent evaluator. Every compared arm must select the same
release and pass the original functional gates in that environment.
"""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import replace
from functools import wraps
from pathlib import Path, PurePosixPath

from merlin.targetgen.compiler_library import active_compiler_library, scoped_compiler_library
from merlin.targetgen.compiler_profile_admission import active_compiler_imports

from .contracts import StageGateError


def scoped_policy(function):
    """Carry a retained policy's fixed selection into broker and queue threads."""

    @wraps(function)
    def call(policy, *args, **kwargs):
        if policy._compiler_library_selection is None and active_compiler_library() is not None:
            raise StageGateError("ordinary policy cannot inherit an unbound additional compiler library")
        with scoped_compiler_library(policy._compiler_library_selection):
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
