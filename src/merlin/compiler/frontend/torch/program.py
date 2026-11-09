"""Freeze Python framework inputs before the out-of-process capture boundary.

Torch is imported only when the caller supplies a framework object. The archive
and example/reference data stay in the caller's capture workspace. Compilation
receives the same compiler-only staged inputs as a saved-export invocation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from merlin.compiler.artifact import digest
from merlin.compiler.frontend.torch.export import SavedExport


def freeze(program, *, example_inputs: tuple | None, workspace: Path):
    """Serialize one original program without numerical preparation or mode changes.

    Modules and example inputs are copied together, retaining shared references
    while preventing preparation from modifying the caller's module. Functions
    are traced through an ordinary module wrapper. Python side effects during
    tracing have the same restrictions as torch.export itself.
    """
    workspace = Path(workspace).absolute()
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError(f"capture requires a fresh workspace: {workspace}")
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Python model inputs require PyTorch in the calling interpreter; "
            "otherwise pass a SavedExport produced by your framework environment"
        ) from error

    if example_inputs is not None and not isinstance(example_inputs, tuple):
        raise TypeError("example_inputs must be a tuple of positional tensor inputs")
    if isinstance(program, torch.export.ExportedProgram):
        if example_inputs is not None:
            raise ValueError("an ExportedProgram already carries its example inputs")
        exported = program
        kind = "torch.export.ExportedProgram"
        modes = None
    else:
        if not isinstance(program, torch.nn.Module) and not callable(program):
            raise TypeError("compile requires source text, Path, SavedExport, or a PyTorch program")
        if example_inputs is None:
            raise ValueError("a module, FX graph or function requires example_inputs=(...)")
        if any(not isinstance(value, torch.Tensor) for value in example_inputs):
            raise ValueError("the capture adapter requires flat positional tensor inputs")
        kind = (
            "torch.fx.GraphModule"
            if isinstance(program, torch.fx.GraphModule)
            else "torch.nn.Module"
            if isinstance(program, torch.nn.Module)
            else "python.callable"
        )
        try:
            snapshot, inputs = copy.deepcopy((program, example_inputs))
        except Exception as error:
            raise ValueError(
                "cannot copy the program and its inputs without changing their state; "
                "export it explicitly and pass the resulting ExportedProgram or SavedExport"
            ) from error
        if not isinstance(snapshot, torch.nn.Module):
            function = snapshot

            class Function(torch.nn.Module):
                def forward(self, *args):
                    return function(*args)

            snapshot = Function()
        modes = {name: module.training for name, module in snapshot.named_modules()}
        # Explicitly select the documented exporter mode instead of inheriting a
        # changing framework default. No eval(), quantize_(), or input casts.
        exported = torch.export.export(snapshot, inputs, strict=False)

    workspace.mkdir(parents=True)
    original = workspace / "original"
    original.mkdir()
    archive = original / "program.pt2"
    temporary = original / "program.pt2.pending"
    try:
        with temporary.open("xb") as stream:
            torch.export.save(exported, stream)
        temporary.replace(archive)
        record = {
            "schema": "merlin.frozen-program.v1",
            "input_kind": kind,
            "torch_version": str(torch.__version__),
            "export_strict": False if modes is not None else None,
            "module_training": modes,
            "archive_sha256": digest(archive),
            "scope": "original_export; no_precision_change_or_compilation_claim",
        }
        (original / "program.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return SavedExport(archive, capture=workspace / "import"), record
