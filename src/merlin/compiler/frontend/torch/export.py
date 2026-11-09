"""Import an explicit saved export through the selected, pinned capture producer.

Reference inputs and outputs stay in the capture workspace. Only the existing
compiler-only staging contract crosses into compilation. Immutable parameters
and constants become exact-storage tensor constants; mutable buffers retain
their explicit state role and require the existing stateful application route.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.compiler.artifact import Artifact, digest
from merlin.compiler.request import Source


@dataclass(frozen=True)
class SavedExport:
    """A torch.export archive and a fresh workspace for capture evidence.

    Loading a framework archive invokes its framework deserializer. Select only
    archives you intend to load, just as for the existing capture command.
    """

    path: Path
    capture: Path

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path).resolve(strict=True))
        object.__setattr__(self, "capture", Path(self.capture).absolute())
        if not self.path.is_file():
            raise ValueError("saved export must be a file")


def prepare(export: SavedExport, *, max_initialization_bytes: int = 64 * 1024 * 1024, original_program=None):
    """Capture once and revalidate producer, source, trace, and argument custody."""
    from merlin.compiler.frontend.inputs import load_compile_inputs, stage_compile_inputs
    from merlin.compiler.frontend.torch import capture
    from merlin.compiler.frontend.torch.bindings import bind_lifted_arguments
    from merlin.integrations.model2mlir import capture_pin, capture_python, root

    workspace = export.capture
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError(f"capture requires a fresh workspace: {workspace}")
    checkout = root()
    pin = capture_pin(checkout)
    if pin.get("ok") is not True:
        raise ValueError(f"capture producer is not pinned: {pin.get('citation')}")
    interpreter = capture_python(checkout=checkout)
    original = digest(export.path)
    workspace.mkdir(parents=True)
    bundle = workspace / "reference"
    # The selected producer owns its framework environment. A caller's compiler
    # PYTHONPATH or user-site packages must not replace its pinned dependencies.
    command = [
        str(interpreter),
        "-I",
        str(Path(capture.__file__)),
        "--exported",
        str(export.path),
        "--out",
        str(bundle),
        "--m2m-dir",
        str(checkout),
    ]
    with (workspace / "capture.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"saved-export capture failed; see {workspace / 'capture.log'}")
    if digest(export.path) != original or capture_pin(checkout) != pin:
        raise ValueError("saved export or selected capture producer changed during import")
    receipt = json.loads((bundle / "capture_receipt.json").read_text())
    # The producer's source identity must name the same archive the caller selected.
    source = receipt.get("source", {})
    if source.get("sha256") != original:
        raise ValueError("capture receipt does not bind the selected saved export")
    stage = workspace / "compiler"
    stage_compile_inputs(bundle, stage)
    staged = load_compile_inputs(stage)
    bound = bind_lifted_arguments(stage, staged, max_initialization_bytes=max_initialization_bytes)
    if original_program is not None and original_program.get("archive_sha256") != original:
        raise ValueError("frozen Python program differs from the selected export archive")
    record = {
        "schema": "merlin.saved-export.v1",
        "archive_sha256": original,
        "capture_pin": pin,
        "capture_python": str(interpreter),
        "capture_workspace": str(workspace),
        "capture_receipt_sha256": digest(bundle / "capture_receipt.json"),
        "captured_mlir_sha256": staged.original_source_sha256,
        "staged_mlir_sha256": staged.program_sha256,
        "compiler_mlir_sha256": bound.record["bound_program_sha256"],
        "entry": staged.entry,
        "argument_binding": bound.record,
        "scope": "static_unquantized_tensor_inputs_and_immutable_initializers; importer_numerical_fidelity_unverified",
    }
    if original_program is not None:
        record["original_program"] = original_program
    (workspace / "frontend.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return Source(bound.program, origin=str(export.path)), stage, record, bound


def publish(compilation, stage: Path, record: dict, bound):
    """Retain compiler-only inputs as portable artifacts; never copy references."""
    from merlin.compiler.frontend.inputs import load_compile_inputs

    bound.revalidate(stage)
    staged = load_compile_inputs(stage)
    if (
        bound.record["bound_program_sha256"] != compilation.source_sha256
        or record["compiler_mlir_sha256"] != compilation.source_sha256
        or record["argument_binding"] != bound.record
    ):
        raise ValueError("compiled source differs from prepared export")
    destination = compilation.directory / "frontend"
    shutil.copytree(stage, destination)
    copied = load_compile_inputs(destination)
    if copied != staged:
        raise ValueError("compiler-only inputs changed while publishing")
    bound.revalidate(destination)
    program = compilation.directory / "frontend.mlir"
    program.write_text(bound.program, encoding="utf-8")
    receipt = compilation.directory / "frontend.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    products = tuple(
        Artifact.from_file(path, kind="frontend-input", role="frontend")
        for path in [*sorted(destination.iterdir()), program, receipt]
        if path.is_file()
    )
    result = replace(
        compilation, artifacts=compilation.artifacts + products, metadata={**compilation.metadata, "frontend": record}
    )
    result.write()
    return result
