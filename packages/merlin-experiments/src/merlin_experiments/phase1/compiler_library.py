"""Prospective, explicit additional-library selection for initial authoring.

Reuse the reviewed leaf snapshot and sandbox overlay. Import availability is
neither a compiler-pass permission nor numerical or resource qualification.
Legacy runs cannot acquire this environment on resume.
"""

from __future__ import annotations

import argparse
import contextvars
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from merlin.targetgen.compiler_library import (
    CompilerLibrarySelection,
    active_compiler_library,
    scoped_compiler_library,
)
from merlin_experiments.phase2 import compiler_libraries as overlays
from merlin_experiments.phase2 import compiler_library_state as state

KEY = "additional_compiler_library"
_INITIAL = contextvars.ContextVar("initial_compiler_library", default=None)


def encoded(record):
    return None if record is None else json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def freeze_for_run(request, run_dir, *, prior, resuming):
    selection = request.compiler_library_selection
    from .joint_epochs import selected as selected_epoch
    epoch = selected_epoch(request.options)
    if epoch is not None and (selection is None or selection.contract.sha256 != epoch["new_contract_sha256"]):
        raise ValueError("joint epoch differs from the operator-reviewed library contract")
    if active_compiler_library() is not None:
        raise ValueError("initial authoring requires explicit selection, not an ambient library")
    if selection is not None and type(selection) is not CompilerLibrarySelection:
        raise ValueError("initial authoring requires an original reviewed library selection")
    if resuming:
        if prior is None and (run_dir / "compiler_library").exists():
            raise ValueError("resume refused: initial compiler library binding is missing")
        if (selection is None) != (prior is None):
            raise ValueError("resume refused: initial compiler library selection was added or removed")
        if prior is not None:
            state.verify_binding(prior, selection)
        return prior
    if selection is None:
        return None
    if request.options.sandbox != "bwrap":
        raise ValueError("initial additional compiler library requires the readonly bwrap transport")
    record = state.freeze_selection(selection, run_dir / "compiler_library/python")
    contract = run_dir / "compiler_library/contract.json"
    contract.write_bytes(_contract_bytes(selection))
    contract.chmod(0o444)
    return record


def verify(request, environment, expected):
    record = environment.get(KEY)
    if encoded(record) != expected:
        raise ValueError("initial compiler library binding changed after admission")
    selection = request.compiler_library_selection
    if record is None:
        if selection is not None:
            raise ValueError("initial compiler library lacks its frozen binding")
        return None
    snapshot = state.verify_binding(record, selection)
    _check_contract(snapshot, snapshot.root.parent / "contract.json")
    return snapshot


def append_guidance(workspace, run_dir, record, selection):
    if record is None:
        return
    from merlin_experiments.phase2.compiler_library_guidance import selected_inventory

    snapshot = state.verify_binding(record, selection)
    before = encoded(record)
    with scoped_compiler_library(snapshot):
        inventory = selected_inventory()
    serialized = json.dumps(inventory, sort_keys=True, indent=2)
    if len(serialized) > 24 << 10:
        (workspace / "COMPILER_LIBRARY_API.json").write_text(serialized + "\n")
        (run_dir / "COMPILER_LIBRARY_API.json").write_text(serialized + "\n")
        details = "Full baseline inventory: `COMPILER_LIBRARY_API.json`.\n"
    else:
        details = "```json\n" + serialized + "\n```\n"
    text = (
        "\n## Original selected additional compiler library\n\n"
        "The selected baseline leaves remain readonly; an explicitly enabled candidate fork "
        "is described below. Availability does not grant numerical or resource authority.\n\n"
        + details
    )
    from .joint_control import selected as joint_selected
    if joint_selected():
        from .joint_pass_guidance import prompt
        text += prompt()
    from merlin.targetgen.compiler_library_fork import enabled, stage
    if enabled(snapshot):
        stage(snapshot, workspace / "submission")
        text += (
            "\n## Editable generic compiler implementation\n\n"
            "Edit `submission/compiler_library/merlin/` alongside your empty target backend. "
            "Each request snapshots and hashes both together. Its exact selected module leaves "
            "replace the baseline implementation only inside the compiler sandbox. Put new helper "
            "modules in your backend package. All numerical, source-to-device, emitted-code and "
            "resource checks remain independently owned. Editing a checker copy does not change "
            "grading. Keep reusable host code and target-independent passes in this generic fork; "
            "do not generate CPU contraction fallbacks. Failed variants do not replace qualified "
            "source snapshots; retain your last working variant.\n"
        )
    task = workspace / "TASK.md"
    task.write_text(task.read_text() + text)
    (run_dir / "TASK.md").write_bytes(task.read_bytes())
    if encoded(record) != before:
        raise ValueError("initial compiler library changed during task staging")
    state.verify_binding(record, selection)


@contextmanager
def execution(prepared):
    snapshot = verify(prepared.request, prepared.environment, prepared.compiler_library_binding)
    if active_compiler_library() is not None:
        raise ValueError("initial execution cannot inherit an ambient compiler library")
    with scoped_compiler_library(snapshot):
        token = _INITIAL.set(None if snapshot is None else (snapshot, snapshot.root.parent / "contract.json"))
        try:
            yield
        finally:
            try:
                if active_compiler_library() is not snapshot:
                    raise ValueError("initial execution changed its original compiler library owner")
                verify(prepared.request, prepared.environment, prepared.compiler_library_binding)
            finally:
                _INITIAL.reset(token)


def _contract_bytes(selection):
    return encoded(selection.contract.record()) + b"\n"


def _check_contract(selection, path):
    selection.record()
    if (
        path.is_symlink()
        or path.resolve() != path
        or not path.is_file()
        or any(parent.is_symlink() for parent in path.parents)
        or path.stat().st_mode & 0o222
        or path.read_bytes() != _contract_bytes(selection)
    ):
        raise ValueError("initial compiler library child contract changed")


def child_arguments():
    """Original launcher's explicit CLI grant, never request or environment data."""
    selected = _INITIAL.get()
    if selected is None:
        return ()
    library, contract = selected
    if active_compiler_library() is not library:
        raise ValueError("initial compiler library child lost its original selected owner")
    _check_contract(library, contract)
    return (
        "--compiler-library-root",
        str(library.root),
        "--compiler-library-contract",
        str(contract),
        "--compiler-library-contract-sha256",
        library.contract.sha256,
    ) + _shared_pass_arguments(contract)


def _shared_pass_file(path, digest, *, expected=None):
    from merlin.common.digest import sha256_file
    from merlin.targetgen.shared_pass_contract import validate_shared_pass_requirement

    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or any(parent.is_symlink() for parent in path.parents)
        or path.stat().st_mode & 0o222
        or path.stat().st_size > 64 << 10
        or sha256_file(path) != digest
    ):
        raise ValueError("original shared-pass child requirement file changed or is not readonly")
    value = validate_shared_pass_requirement(json.loads(path.read_bytes()))
    if expected is not None and value != expected:
        raise ValueError("original shared-pass child requirement differs from the selected scope")
    return value


def _shared_pass_arguments(contract):
    from merlin.common.digest import sha256_file
    from merlin.targetgen.shared_pass_context import active_shared_pass_requirement

    expected = active_shared_pass_requirement()
    path = contract.parent.parent / "shared_pass_requirement.json"
    if expected is None:
        if path.exists() or path.is_symlink():
            raise ValueError("original shared-pass requirement was lost before child launch")
        return ()
    digest = sha256_file(path)
    _shared_pass_file(path, digest, expected=expected)
    return "--shared-pass-requirement", str(path), "--shared-pass-requirement-sha256", digest


def selected_child(function):
    """Load only an explicit original host CLI selection at a fresh child edge."""

    @wraps(function)
    def call(argv=None, *args, **kwargs):
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        state.add_selection_arguments(parser)
        parser.add_argument("--shared-pass-requirement", type=Path)
        parser.add_argument("--shared-pass-requirement-sha256")
        values, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
        selection = state.selection_from_arguments(values)
        path, digest = values.shared_pass_requirement, values.shared_pass_requirement_sha256
        if (path is None) != (digest is None):
            raise ValueError("shared-pass child requirement requires its exact path and SHA")
        if selection is None:
            if path is not None:
                raise ValueError("shared-pass child requirement lacks its original compiler library")
            if _INITIAL.get() is not None:
                raise ValueError("initial compiler child requires its explicit original library arguments")
            return function(argv, *args, **kwargs)
        if active_compiler_library() is not None or _INITIAL.get() is not None:
            raise ValueError("initial compiler child cannot replace an ambient selected owner")
        contract = Path(values.compiler_library_contract)
        _check_contract(selection, contract)
        state._readonly_snapshot(selection)
        required_path = contract.parent.parent / "shared_pass_requirement.json"
        requirement = None
        if path is None:
            if required_path.exists() or required_path.is_symlink():
                raise ValueError("original shared-pass child requirement arguments were omitted")
        else:
            if path != required_path:
                raise ValueError("shared-pass child requirement is foreign to the selected run")
            requirement = _shared_pass_file(path, digest)
        from merlin.targetgen.shared_pass_context import scoped_shared_pass_requirement

        with scoped_compiler_library(selection), scoped_shared_pass_requirement(requirement):
            token = _INITIAL.set((selection, contract))
            try:
                return function(remaining, *args, **kwargs)
            finally:
                try:
                    if path is not None:
                        _shared_pass_file(path, digest, expected=requirement)
                    child_arguments()
                finally:
                    _INITIAL.reset(token)

    return call


def background_target(function):
    """Carry the live original owner into one thread, retaining join failures."""
    if _INITIAL.get() is None:
        return function, None
    child_arguments()
    from merlin.targetgen.shared_pass_context import active_shared_pass_requirement, scoped_shared_pass_requirement

    requirement = active_shared_pass_requirement()
    selected = _INITIAL.get()
    context, failures = contextvars.Context(), []

    def body():
        library, _contract = selected
        try:
            with scoped_compiler_library(library), scoped_shared_pass_requirement(requirement):
                token = _INITIAL.set(selected)
                try:
                    try:
                        child_arguments()
                        function()
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        try:
                            child_arguments()
                        except BaseException as error:
                            failures.append(error)
                finally:
                    _INITIAL.reset(token)
        except BaseException as error:
            failures.append(error)

    def joined():
        child_arguments()
        if failures:
            raise RuntimeError("initial compiler library background owner failed") from failures[0]

    return lambda: context.run(body), joined


@dataclass(frozen=True)
class _Mounts:
    argv: tuple[str, ...]
    env_prefix: str


def sandbox_overlay(parts, *, workspace, bundle, context, private_run_dir):
    """Reuse the same exact-leaf overlay before the final answer-mask pass."""
    if active_compiler_library() is None:
        return parts, ""
    from merlin.targetgen.compiler_profile_admission import original_compiler_imports

    if bundle.get("compiler_tool_profile") is None:
        result = overlays.apply(_Mounts(tuple(parts), ""))
    else:
        if private_run_dir is None:
            raise ValueError("selected library overlay requires the original private run directory")
        with original_compiler_imports(
            workspace=workspace, bundle_manifest=private_run_dir / "input_bundle_manifest.yaml", repo=context.repo
        ):
            result = overlays.apply(_Mounts(tuple(parts), ""))
    return list(result.argv), result.env_prefix


def library_thread(function, **kwargs):
    """Keep the existing thread policy; transfer only the original library."""
    import threading

    target, check = background_target(function)
    thread = threading.Thread(target=target, **kwargs)
    if check is not None:
        thread._initial_library_join = check
    return thread


def join_background(thread):
    thread.join()
    check = getattr(thread, "_initial_library_join", None)
    if check is not None:
        check()
