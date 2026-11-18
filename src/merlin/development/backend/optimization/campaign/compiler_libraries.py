"""Additional reviewed compiler leaves in the existing normal package sandbox.

This changes a Phase 2 compiler's explicit library environment, never the Phase
1 profile or independent evaluator. Every compared arm must select the same
release and pass the original functional gates in that environment.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import shlex
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import wraps
from pathlib import Path, PurePosixPath

from merlin.common.process.cpu_pool import AUTHOR, cpu_priority, shared_compute_pool
from merlin.development.backend.feedback.execution.compiler_pass_delegation import (
    scoped_selected_parent_pass_delegation,
)
from merlin.development.backend.optimization.contracts import StageGateError, document_sha256, mapping_file, write_json
from merlin.development.targetgen.authoring.library_admission import (
    CompilerLibrarySelection,
    active_compiler_library,
    scoped_compiler_library,
)
from merlin.development.targetgen.authoring.pass_selection import (
    CompilerPassSelection,
    active_compiler_passes,
    scoped_compiler_passes,
)
from merlin.development.targetgen.authoring.profile_admission import active_compiler_imports
from merlin.evaluation.capsule.operations import setup_timing

# ---- compiler_libraries ------------------------------------------------------------------------


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
        with (
            scoped_compiler_library(selection),
            scoped_compiler_passes(passes),
            scoped_selected_parent_pass_delegation(passes),
        ):
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
    from merlin.common.storage.bundle_snapshots import verified_snapshot_scope
    from merlin.development.backend.optimization.author_interface import functional_inputs
    from merlin.development.targetgen.authoring.profile_admission import original_compiler_imports

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
    # Only these author-waited leaves outrank unrelated grading work; the body
    # keeps its caller's priority and process environment.
    stack = ExitStack()
    try:
        with cpu_priority(AUTHOR, export=False), shared_compute_pool(timeout_s=120):
            stack.enter_context(scoped_compiler_library(policy._compiler_library_selection))
            stack.enter_context(scoped_compiler_passes(passes))
            stack.enter_context(scoped_selected_parent_pass_delegation(passes))
        yield
    finally:
        unwound = False
        try:
            with cpu_priority(AUTHOR, export=False), shared_compute_pool(timeout_s=120):
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
    from merlin.common.storage.bundle_snapshots import guarded_projection
    from merlin.development.targetgen.authoring import tool_registry
    from merlin.development.targetgen.sandbox import grants as BW

    if owner is None:
        return {}
    owner.verify()
    # This also reopens live original host locations. Only its frozen-record
    # projection may reuse the issued guard; keep its live checks every call.
    _, grants = BW._snapshot_grants(owner.workspace, owner.bundle, owner.repo)
    rows = list(owner.profile["common_sources"])
    for name in owner.profile["treatment_tools"]:
        if name in {"isa_dialect_generation", "lean_proof_authoring"}:
            rows.extend({"path": path} for path in tool_registry.spec(name).bundle_paths if path.endswith(".py"))
    # Freeze only the inputs to this pure projection. The live owner is still
    # verified above on every call; candidate overlays and final masks below
    # never enter this request-owned cache.
    sources = tuple((row["path"], row.get("sha256")) for row in rows)
    return guarded_projection(
        owner.workspace,
        owner.bundle,
        owner.repo,
        ("phase2.original_public_sources.v1", sources),
        lambda: _original_source_projection(grants, sources),
    )


def _original_source_projection(grants, sources):
    selected = {}
    for declared_path, source_sha256 in sources:
        relative = declared_path.removeprefix("merlin/python/")
        matches = [
            source
            for declared, _destination, source in grants
            if declared == declared_path or declared.endswith("/" + relative)
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
            or source_sha256 is not None
            and hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256
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
    return _environment_prefix(prefix, guest_root(selection))


def _environment_prefix(prefix, root):
    """Pure formatting of a root already verified by the enclosing operation."""
    suffix = "export PYTHONPATH=" + shlex.quote(str(root)) + ":$PYTHONPATH; "
    return prefix if prefix.endswith(suffix) else prefix + suffix


@setup_timing.measured("package_policy.compiler_library")
def apply(policy):
    """Add exact readonly leaves to an already answer-masked normal policy.

    Namespace initializers and overlapping compiler algorithms come from the
    explicitly selected Phase 2 release. Other original public leaves remain
    available in that same Python tree. No private grader tree is mounted.
    """
    from merlin.development.backend.authoring.library.compiler_workspace import apply_policy
    from merlin.development.backend.authoring.library.compiler_workspace import selected as full_selection

    full = full_selection()
    if full is not None:
        return apply_policy(policy, full)
    selection = active_compiler_library()
    if selection is None:
        return policy
    record = selection.record()
    package = getattr(policy, "package", None)
    if package is not None and (selection.root.is_relative_to(package) or Path(package).is_relative_to(selection.root)):
        raise StageGateError("additional compiler library cannot be candidate-owned")
    # This bounded metadata operation already verified the full selection.
    # Derive formatting from that record; reopen all bytes at the exit below.
    root = Path("/perf-compiler-library") / record["contract_sha256"] / "python"
    sources = _original_sources(active_compiler_imports())
    sources.update({member.path: selection.root / member.path for member in selection.contract.members})
    from merlin.development.targetgen.authoring.library_admission import candidate_sources

    fork, fork_sha256 = candidate_sources(selection, package)
    previous_fork = getattr(policy, "compiler_fork_sha256", None)
    if previous_fork is not None and previous_fork != fork_sha256:
        raise StageGateError("candidate compiler fork changed after sandbox policy construction")
    sources.update(fork)
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
    return replace(
        policy,
        argv=tuple(prefix),
        env_prefix=_environment_prefix(policy.env_prefix or "", root),
        **({"compiler_fork_sha256": fork_sha256} if hasattr(policy, "compiler_fork_sha256") else {}),
    )


# ---- compiler_library_guidance -----------------------------------------------------------------
# Answer-free source inventory and projections of existing host edit authority.
#
# AST declarations describe reviewed import availability. They do not identify
# target capabilities, execute library code or authorize compiler edits.

MAX_DECLARATIONS = 256

MAX_PREVIEW = 320


def _declarations(body, prefix=""):
    rows = []
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not (prefix and node.name == "__init__"):
                continue
            row = {
                "symbol": prefix + node.name,
                "line": node.lineno,
                "kind": "class" if isinstance(node, ast.ClassDef) else "callable",
            }
            doc = ast.get_docstring(node)
            if doc:
                row["documentation_preview"] = doc[:MAX_PREVIEW]
                row["documentation_truncated"] = len(doc) > MAX_PREVIEW
            if isinstance(node, ast.ClassDef):
                rows.append(row)
                rows.extend(_declarations(node.body, prefix + node.name + "."))
            else:
                signature = ast.unparse(node.args)
                row["parameters_preview"] = signature[:MAX_PREVIEW]
                row["parameters_truncated"] = len(signature) > MAX_PREVIEW
                rows.append(row)
        elif not prefix and isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name != "*" and not name.startswith("_"):
                    rows.append({"symbol": name, "kind": "imported_name", "line": node.lineno})
    return rows


def _bounded_previews(declarations):
    """Share the preview budget without hiding later modules' entrypoints."""
    selected = [[] for _ in declarations]
    remaining = MAX_DECLARATIONS
    # Discover actual public definitions before their methods or imported names.
    # Round-robin within each tier keeps alphabetical module order from owning
    # the entire budget. Preserve original source order in each final preview.
    for tier in range(3):
        queues = []
        for rows in declarations:
            queues.append(
                [
                    row
                    for row in rows
                    if (2 if row["kind"] == "imported_name" else 1 if "." in row["symbol"] else 0) == tier
                ]
            )
        cursor = 0
        while remaining and any(cursor < len(rows) for rows in queues):
            for index, rows in enumerate(queues):
                if cursor < len(rows) and remaining:
                    selected[index].append(rows[cursor])
                    remaining -= 1
            cursor += 1
    return [sorted(rows, key=lambda row: row["line"]) for rows in selected]


def selected_inventory():
    """Read exact reviewed source bytes without importing their implementation."""
    selection = active_compiler_library()
    if selection is None:
        return None

    before = selection.record()
    # guest_root revalidates the entire selected library. Resolve once within
    # this invocation, keeping the original before/after custody checks below.
    root = guest_root(selection)
    members = {row.module: row for row in selection.contract.members}
    modules, declarations = [], []
    for module in sorted(selection.contract.public_modules):
        member = members[module]
        declarations.append(_declarations(ast.parse((selection.root / member.path).read_bytes()).body))
        modules.append(
            {
                "module": module,
                "source": str(root / member.path),
                "source_sha256": member.sha256,
            }
        )
    for module, available, retained in zip(modules, declarations, _bounded_previews(declarations), strict=True):
        module.update(
            declarations=retained,
            declaration_count=len(available),
            omitted_declarations=len(available) - len(retained),
        )
    if selection.record() != before:
        raise StageGateError("selected compiler library changed during public API inventory")
    return {
        "schema": "merlin.phase2.public_compiler_library_inventory.v2",
        "review_id": selection.contract.review_id,
        "contract_sha256": selection.contract.sha256,
        "public_modules": modules,
        "derivation": "AST declarations from exact reviewed source; no library imports executed",
        "declaration_scope": (
            "top-level definitions/imported names and public class methods; "
            "dynamic exports and value aliases unresolved"
        ),
        "preview_allocation": (
            "round-robin modules, public top-level definitions before methods before imported names; "
            "original source order within each retained module preview"
        ),
        "library_edit_authority": "read-only reviewed algorithms, identical for both compared arms",
        "candidate_edit_authority": "only the host-frozen compiler edit contract; inventory grants no edits",
        "target_adapter_applicability": "UNKNOWN from source availability; requires selected software/RTL checks",
        "independent_qualification": "required; source inventory supplies no numerical, runtime or timing verdict",
        "algorithm_origin": (
            "reviewed authored library code; selected software effects and RTL facts constrain legal policy"
        ),
    }


def prompt_text(inventory):
    if inventory is None:
        return ""
    return (
        "\n## Additional reviewed compiler library\n\n"
        "The JSON below lists this selected release's public modules, exact read-only source paths, "
        "declaration lines and bounded API/docstring previews. Imported names are descriptive; inspect "
        "the exact source for signatures and guarantees. This is import availability, not target support. "
        "Generic algorithms are reviewed authored code. Legal sizes, layouts, resource bounds and effects "
        "derive from the selected software/ISA specification and RTL facts; a qualified adapter must connect them "
        "to independently checked emitted instructions. Missing applicability or timing remains UNKNOWN.\n\n"
        "Tune or modify candidate-owned policies and transformation bodies, or add components only where "
        "the host-frozen edit contract allows them. Read `STAGE_CONTEXT.json` -> "
        "`frozen_compiler_edit_authority` for exact existing path/symbol, helper and v2 new-component authority "
        "when queued search is enabled. Its absence grants no new component. The automatic optimization "
        "inventory/template and candidate manifest are descriptive, not permission. Selected shared-library "
        "leaves remain read-only; changing that release needs a separate host review and same-release comparison.\n\n"
        "For each hypothesis, identify the owned entrypoint, expected emitted-program delta and affected "
        "public shapes; freeze the source revision, inspect emitted bytes/work and independent correctness, "
        "then inspect original paired measurements or explicitly qualified predictions. Use search-status "
        "for pending/refused/failed cells and the unchanged objective. Roofline gaps guide experiments; "
        "UNKNOWN or partial floors cannot establish convergence, and an IR annotation alone is no improvement.\n"
    )


def prompt_context(inputs):
    inventory = getattr(inputs, "additional_compiler_library", None)
    return {"additional_compiler_library": copy.deepcopy(inventory)} if inventory is not None else {}


def publish_edit_authority(authority, candidate):
    """Project the already-frozen authority, excluding private seed/host paths."""
    authority.check_integrity()
    if not authority.configured or authority.contract is None:
        raise StageGateError("public edit guidance requires the actual frozen compiler edit authority")
    context_path = Path(candidate).parent / "STAGE_CONTEXT.json"
    if context_path.is_symlink() or not context_path.is_file():
        raise StageGateError("queued author context is absent or linked")
    projection = {
        "schema": "public_frozen_compiler_edit_authority_v1",
        "initial_candidate_sha256": authority.binding["initial_candidate_sha256"],
        "contract_document_sha256": authority.binding["contract_document_sha256"],
        "contract": copy.deepcopy(authority.contract),
        "scope": "description of existing host authority; editing this projection grants nothing",
    }
    projection["sha256"] = document_sha256(projection)
    context = mapping_file(context_path)
    context["frozen_compiler_edit_authority"] = projection
    write_json(context_path, context)
    authority.check_integrity()
    return projection
