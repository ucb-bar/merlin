"""Bind an existing compiler-only stage to public compilation and discovery.

These are source observations, not a numerical admission or a reference grant.
Only the closed staged roster is read; capture/evaluator directories are absent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _snapshot(directory):
    from merlin.compiler.artifact import digest
    from merlin.compiler.frontend.archives import _read_member
    from merlin.compiler.frontend.inputs import load_compile_inputs

    if directory.is_symlink() or any(parent.is_symlink() for parent in directory.parents) or not directory.is_dir():
        raise ValueError("compiler input directory must be real and present")
    manifest = _read_member(directory, "compile-inputs.json")
    staged = load_compile_inputs(directory)
    if staged.precision_policy_document is None:
        raise ValueError("staged program requires an explicit source-bound precision policy")
    # The loader has already checked the exact regular-file roster, every
    # signature/initializer and policy-to-captured-source correspondence.
    members = tuple((path.name, digest(path)) for path in sorted(directory.iterdir()))
    if _read_member(directory, "compile-inputs.json") != manifest:
        raise ValueError("compiler stage changed during selection")
    return staged, manifest, members


@dataclass(frozen=True)
class StagedProgram:
    """Select existing compiler-only inputs, never a model/evaluator bundle.

    Construction validates and freezes the actual stage. Numerical meaning must
    still be admitted by the original caller through the selected compiler.
    ``freeze_readonly_buffers`` explicitly chooses the freshly observed BUFFER
    bytes for every compiled invocation; later source-state updates are outside
    this snapshot contract. It grants no mutation, alias or accuracy permission.
    """

    directory: Path
    freeze_readonly_buffers: bool = False
    _selected_directory: Path = field(init=False, repr=False)
    _selected_freeze: bool = field(init=False, repr=False)
    _manifest: bytes = field(init=False, repr=False)
    _members: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.freeze_readonly_buffers) is not bool:
            raise ValueError("freeze_readonly_buffers requires an explicit Boolean")
        root = Path(self.directory).absolute()
        _, manifest, members = _snapshot(root)
        object.__setattr__(self, "directory", root)
        object.__setattr__(self, "_selected_directory", root)
        object.__setattr__(self, "_selected_freeze", self.freeze_readonly_buffers)
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_members", members)

    def revalidate(self):
        if self.freeze_readonly_buffers is not self._selected_freeze:
            raise ValueError("original buffer freeze consent changed after selection")
        if self.directory != self._selected_directory:
            raise ValueError("original compiler stage directory changed after selection")
        staged, manifest, members = _snapshot(self.directory)
        if manifest != self._manifest or members != self._members:
            raise ValueError("original compiler stage or precision policy changed after selection")
        return staged


@dataclass(frozen=True)
class StagedPreparation:
    """Actual stage/policy/bound-source identities; no semantic authorization."""

    stage: StagedProgram
    bound: object = field(repr=False)
    _record: str = field(init=False, repr=False)
    _policy: bytes = field(init=False, repr=False)
    _owners: tuple = field(init=False, repr=False)

    def __post_init__(self):
        from merlin.compiler.frontend.torch.bindings import BoundCapture

        if type(self.stage) is not StagedProgram or type(self.bound) is not BoundCapture:
            raise ValueError("preparation requires the original typed stage and immutable binding")
        staged = self.stage.revalidate()
        self.bound.revalidate(self.stage.directory, staged)
        record = {
            "schema": "merlin.staged-preparation.v1",
            "directory": str(self.stage.directory),
            "preparation_sha256": _sha(self.stage._manifest),
            "preparation_member": "compile-inputs.json",
            "numerical_policy_sha256": staged.precision_policy_sha256,
            "captured_source_sha256": staged.original_source_sha256,
            "staged_source_sha256": staged.program_sha256,
            "bound_source_sha256": self.bound.record["bound_program_sha256"],
            "members": dict(self.stage._members),
            "scope": "compiler staging/preparation identity only; numerical admission and calibration/reference "
            "qualification are independent",
        }
        if self.stage.freeze_readonly_buffers:
            record["buffer_freeze"] = self.bound.record["buffer_freeze"]
        if "quantization_legalization" in self.bound.record:
            record["quantization_legalization"] = self.bound.record["quantization_legalization"]
        object.__setattr__(self, "_record", json.dumps(record, sort_keys=True, allow_nan=False))
        object.__setattr__(self, "_policy", staged.precision_policy_document)
        object.__setattr__(self, "_owners", (self.stage, self.bound))

    def record(self):
        return json.loads(self._record)

    def policy(self):
        """Return selected policy data for original-caller admission, never a verdict."""
        return json.loads(self._policy)

    def revalidate_ir(self, source):
        """Bind normalized discovery IR to the actual immutable argument substitution."""
        from merlin.frontends.linalg_mlir import parse_mlir_text

        self.revalidate()
        source.revalidate()
        if not source.module.is_structurally_equivalent(parse_mlir_text(self.bound.program)):
            raise ValueError("discovery source differs from its original staged preparation")

    def revalidate(self, source=None):
        if self.stage is not self._owners[0] or self.bound is not self._owners[1]:
            raise ValueError("original staged preparation owners changed")
        staged = self.stage.revalidate()
        self.bound.revalidate(self.stage.directory, staged)
        record = self.record()
        if (
            self.stage.freeze_readonly_buffers is not ("buffer_freeze" in record)
            or str(self.stage.directory) != record["directory"]
            or staged.precision_policy_document != self._policy
            or staged.precision_policy_sha256 != record["numerical_policy_sha256"]
            or _sha(self.stage._manifest) != record["preparation_sha256"]
            or _sha(self.bound.program.encode()) != record["bound_source_sha256"]
        ):
            raise ValueError("original staged preparation identity changed")
        if source is not None and (source.kind != "mlir" or source.text != self.bound.program):
            raise ValueError("prepared request differs from its original bound source")


def prepare(program: StagedProgram, *, max_initialization_bytes):
    """Bind only existing admitted immutable parameters; perform no quantization."""
    from merlin.compiler.frontend.torch.bindings import bind_lifted_arguments
    from merlin.compiler.request import Source

    if type(program) is not StagedProgram:
        raise ValueError("staged preparation requires an explicit StagedProgram")
    staged = program.revalidate()
    bound = bind_lifted_arguments(
        program.directory,
        staged,
        max_initialization_bytes=max_initialization_bytes,
        freeze_readonly_buffers=program.freeze_readonly_buffers,
    )
    binding = StagedPreparation(program, bound)
    source = Source(bound.program, origin=str(program.directory / "program.mlir"))
    binding.revalidate(source)
    return source, binding
