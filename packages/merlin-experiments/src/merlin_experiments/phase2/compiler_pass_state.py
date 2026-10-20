"""Operator-selected live pass admission for the ordinary authoring CLI.

A reviewed record pins a factory, not a successful verdict. Loading always
reruns that original factory and requires the same live registry and release.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from merlin.targetgen.compiler_library import CompilerLibrarySelection
from merlin.targetgen.compiler_pass_selection import restore_original_selection

from .contracts import StageGateError


def add_selection_arguments(parser):
    parser.add_argument("--compiler-pass-selection", type=Path, help="operator-reviewed original pass selection JSON")
    parser.add_argument("--compiler-pass-selection-sha256", help="expected SHA-256 of the exact selection file bytes")
    parser.add_argument(
        "--compiler-pass-admission-root", type=Path, help="fresh operator directory for independent admission"
    )
    parser.add_argument(
        "--compiler-pass-preparation", type=Path, help="operator-selected policies for both copied compilers"
    )
    parser.add_argument(
        "--compiler-pass-preparation-sha256", help="expected SHA-256 of the exact preparation policy file"
    )


def preparation_from_arguments(args):
    """Choices never issue a pass; production requires explicit choices with it."""
    from .compiler_pass_preparation import read_preparation

    path, digest = args.compiler_pass_preparation, args.compiler_pass_preparation_sha256
    selected = any(
        value is not None
        for value in (
            args.compiler_pass_selection,
            args.compiler_pass_selection_sha256,
            args.compiler_pass_admission_root,
        )
    )
    if path is None and digest is None and not selected:
        return None
    if path is None or digest is None or not selected:
        raise StageGateError(
            "original pass selection and both compiler preparation policy inputs are required together"
        )
    try:
        return read_preparation(path, digest)
    except (OSError, ValueError) as error:
        raise StageGateError("original compiler preparation policy refused: " + str(error)) from error


def selection_from_arguments(args, *, library, stage_root, external_boundary=None):
    """Reopen original owners before authoring, without creating the stage root."""
    return load_selection(
        path=args.compiler_pass_selection,
        sha256=args.compiler_pass_selection_sha256,
        work_root=args.compiler_pass_admission_root,
        library=library,
        stage_root=stage_root,
        external_boundary=external_boundary,
    )


def load_selection(*, path, sha256, work_root, library, stage_root, external_boundary=None):
    if path is None and sha256 is None and work_root is None:
        return None
    if path is None or sha256 is None or work_root is None:
        raise StageGateError("compiler pass selection, exact file SHA-256 and admission root are required together")
    if type(library) is not CompilerLibrarySelection:
        raise StageGateError("compiler pass CLI admission requires the explicitly selected reviewed library")
    path, root, stage = Path(path), Path(work_root), Path(stage_root).absolute()
    try:
        if (
            not path.is_absolute()
            or path.resolve(strict=True) != path
            or not path.is_file()
            or path.stat().st_size > 2 << 20
            or stage.exists()
            or stage.is_symlink()
            or stage.resolve() != stage
            or any(parent.is_symlink() for parent in stage.parents)
            or root.exists()
            or root.is_symlink()
            or not root.is_absolute()
            or root.resolve() != root
            or any(parent.is_symlink() for parent in root.parents)
            or root.is_relative_to(stage)
            or stage.is_relative_to(root)
            or root.is_relative_to(library.root)
        ):
            raise ValueError(
                "selection needs bounded original bytes and a fresh admission root disjoint from the stage"
            )
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("compiler pass selection differs from its operator-pinned file SHA-256")
        record = json.loads(payload)
        selected_library = library.record()
        if not isinstance(record, dict) or record.get("binding", {}).get("library") != library.contract.sha256:
            raise ValueError("compiler pass selection binds a different reviewed library release")
        from merlin.common.paths import out_dir

        from .normal_resource_launch import selected_boundary

        temporary = os.environ.get("TMPDIR")
        if external_boundary is not None and not temporary:
            raise ValueError("bounded pass admission requires the original operator TMPDIR")
        roots = (root, out_dir(), *((Path(temporary),) if temporary else ()))
        with selected_boundary(external_boundary, writable_roots=roots):
            selected = restore_original_selection(record, work_root=root)
        if (
            library.record() != selected_library
            or path.read_bytes() != payload
            or selected.binding.library != library.contract.sha256
        ):
            raise ValueError("original pass or library selection changed during admission")
        return selected
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise StageGateError("original compiler pass admission refused: " + str(error)) from error
