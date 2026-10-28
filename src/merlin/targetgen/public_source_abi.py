"""Observe neutral public ABI metadata enrichment before ordinary lowering.

The selected capsule remains the source/numerical authority. A derived source
adds only checked function attributes. Original and normalized source, complete
public metadata and the actual helper call stay pinned through candidate use.
The existing emission observer consumes the exact normalized bytes unchanged;
no saved receipt can select source bytes or bypass its ordinary checks.

:func:`compiler_source` carries the original bytes, their hash and the exact
bytes the compiler receives together. A source-identity check (the mixed-program
plan's ``source_sha256``) then compares the compiler's claim with the bytes it
was given, while those bytes are re-derived here from the selected original and
never read back from a generated copy.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import uuid

import yaml
from merlin.common import invocation_record
from .contract.linalg_iface import bind_logical_io_metadata, is_linalg_on_tensors

_KEYS = ("label", "interface_mlir", "inputs", "component_program")


@dataclass(frozen=True)
class CompilerSource:
    """The selected original source and the exact bytes a compiler receives for it."""

    original_bytes: bytes
    compiler_bytes: bytes

    @property
    def original_sha256(self) -> str:
        return hashlib.sha256(self.original_bytes).hexdigest()

    @property
    def compiler_sha256(self) -> str:
        return hashlib.sha256(self.compiler_bytes).hexdigest()

    @property
    def enriched(self) -> bool:
        return self.compiler_bytes != self.original_bytes


@dataclass(frozen=True)
class _Selection:
    original: Path
    metadata: Path
    metadata_bytes: bytes
    declared: dict
    text: str
    inputs: list
    outputs: list


def _selection(capsule, source: Path, raw: bytes) -> _Selection | None:
    """The checked public selection, or None when the source passes through unchanged."""
    program = capsule.get("component_program")
    if capsule.get("label") != "public" or not isinstance(program, dict) or program.get("schema") != "merlin.component_program.v2":
        return None
    text = raw.decode("utf-8")
    if not is_linalg_on_tensors(text):
        return None
    original = source.resolve(strict=True)
    if source.is_symlink():
        raise ValueError("public ABI source cannot be a symlink")
    root = Path(capsule["__dir__"]).resolve(strict=True)
    metadata = root / "capsule.yaml"
    if metadata.is_symlink() or not original.is_relative_to(root):
        raise ValueError("public ABI metadata/source escapes its selected capsule")
    if metadata.stat().st_size > 2 << 20 or original.stat().st_size > 16 << 20:
        raise ValueError("public ABI enrichment exceeds its bounded input size")
    metadata_bytes = metadata.read_bytes()
    declared = yaml.safe_load(metadata_bytes)
    if any(capsule.get(key) != declared.get(key) for key in _KEYS):
        raise ValueError("public ABI metadata differs from the original loaded capsule")
    if (root / declared.get("interface_mlir", "capsule.interface.mlir")).resolve(strict=True) != original:
        raise ValueError("public ABI interface differs from the selected source")
    inputs, outputs = declared.get("inputs"), declared["component_program"].get("outputs")
    if type(inputs) is not list or type(outputs) is not list:
        raise ValueError("public ABI has no complete ordered inputs/outputs")
    return _Selection(original, metadata, metadata_bytes, declared, text, inputs, outputs)


def _unchanged(selection: _Selection, raw: bytes) -> None:
    if selection.original.read_bytes() != raw or selection.metadata.read_bytes() != selection.metadata_bytes:
        raise ValueError("public ABI source/metadata changed before observed transformation")


def compiler_source(capsule, source) -> CompilerSource:
    """Re-derive the exact bytes :func:`public_abi_source` gives a compiler for ``source``.

    The same selection checks apply and the derivation is the same pure,
    body-preserving enrichment of the selected original's bytes, so a caller can
    bind a compiler's source claim to the bytes it actually received while the
    binding still proves derivation from the selected public source.
    """
    source = Path(source)
    raw = source.read_bytes()
    selection = _selection(capsule, source, raw)
    if selection is None:
        return CompilerSource(raw, raw)
    rendered, _ = bind_logical_io_metadata(selection.text, selection.inputs, selection.outputs)
    _unchanged(selection, raw)
    return CompilerSource(raw, rendered.encode("utf-8"))


@contextlib.contextmanager
def public_abi_source(capsule, source, generated):
    source = Path(source)
    raw = source.read_bytes()
    selection = _selection(capsule, source, raw)
    if selection is None:
        yield source
        return
    work = Path(generated).resolve() / "public-source-abi" / uuid.uuid4().hex
    work.mkdir(parents=True, mode=0o700)
    normalized = work / "interface.mlir"
    receipt = work / "source_binding.json"
    with invocation_record.observe_call(
        work, stage="public_logical_io_enrichment", function=bind_logical_io_metadata,
        arguments={"scope": "public ordered ABI names, roles and exact tensor types only"},
        inputs=(selection.original, selection.metadata), outputs=(normalized, receipt),
        dependencies=(Path(__file__), Path(inspect.getsourcefile(bind_logical_io_metadata))),
    ) as call:
        _unchanged(selection, raw)
        rendered, report = bind_logical_io_metadata(selection.text, selection.inputs, selection.outputs)
        report.update(original_source=str(selection.original), public_metadata=str(selection.metadata),
                      public_metadata_sha256=hashlib.sha256(selection.metadata_bytes).hexdigest())
        normalized.write_text(rendered, encoding="utf-8")
        receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        call.returned()
    invocation_record.verify(call.path)
    # The normal lower_interface observer continues to check source==staged
    # input and complete target/object custody. This outer owner additionally
    # keeps the source-to-metadata transition live across that invocation.
    try:
        yield normalized
    finally:
        invocation_record.verify(call.path)
        if any(capsule.get(key) != selection.declared.get(key) for key in _KEYS):
            raise ValueError("public ABI metadata changed during compiler execution")
