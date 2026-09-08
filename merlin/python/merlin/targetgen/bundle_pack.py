"""Pack a captured model's weights into a target's constant blob, for ANY model.

WHY THIS EXISTS AS LIBRARY CODE. The only working constant-blob builder on this tree is a
hand-written packaging script under ``out/build/generated/``, and the payload actually shipped was
produced by something weaker still: a *rebinder* that copies the previous bundle's ``const_blob.bin``
byte-for-byte after asserting the candidate's read-only ABI prefix is byte-identical to ResNet-50's,
then text-splices the harness. That family cannot build a second model by construction. Its core --
the ABI walk, the manifest-to-argument mapping, the row pitch, the alignment -- is model-agnostic;
only its edges were ResNet-specific. This is that core, with the four defects that would have made it
fail on the other two models fixed and pinned:

**1. The read-only prefix is not always contiguous.** SmolVLA's ``kernel_abi.args`` contains
``arg0..arg808, arg810, arg811`` -- **``arg809`` is absent**, while ``params.global_program_plan``
lists all 812, because the exported flow-denoise graph genuinely never reads the incoming prefix
KV-cache (``%arg809`` occurs zero times in the interface MLIR). A packer that enumerates positionally
would shift every tensor after it by one and mis-address 2 of 811 weights while producing a blob of
exactly the right size. So arguments are mapped by their PARSED index, and a gap is reported rather
than closed up.

**2. ``bf16`` and ``i1`` have no numpy dtype.** 275 of SmolVLA's arguments are ``bf16``, which numpy
cannot represent; they are moved as raw 2-byte words, which is lossless because the safetensors
payload already holds the bf16 encoding. ``i1`` is one byte per element, matching the compiler's own
generic one-byte ``i1`` storage sizing.

**3. ``params.storage_encodings`` exists only for ResNet-50** (it comes from ``global_encoding``), so
sizing mutable tensors from it works for exactly one model. Absent, the physical size falls back to
the pitch formula, and which rule was used is recorded per tensor.

**4. The row pitch is a MESH DIMENSION and is therefore required, never defaulted.** The shipped
script writes ``align(shape[-1], 16)``; 16 is this target's systolic width. Library code may not bake
that in, and guessing it does not fail -- it silently mis-sizes every tensor while producing a
plausible blob. So :func:`pack` takes ``row_pitch_elements`` from the caller, who derives it from the
target's own facts (``capabilities.mesh`` in the capability manifest, itself CIRCT-derived), and
:func:`row_pitch_from_manifest` refuses when the manifest does not state it.

WHAT THIS DOES NOT DO. It does not render a harness and it does not choose a correctness gate: those
are per-model decisions (ResNet-50 checks 1000 logits exactly against an integer reference that only
exists for PT2E Q/DQ graphs; an LM needs a stated tolerance plus top-1 against
``golden_w8a8.independent.npy``, never ``golden.npy`` -- grading W8A8 against a weight-only-int8
reference is what once read as a codegen defect at cos 0.484).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

__all__ = ["ArgRef", "PackedTensor", "PackPlan", "parse_arg_index", "read_only_prefix",
           "element_bytes", "physical_nbytes", "plan", "row_pitch_from_manifest",
           "ELEMENT_BYTES", "DEFAULT_ALIGNMENT"]

#: Bytes per element, by the command buffer's own dtype spelling. ``bf16`` is moved as a raw 2-byte
#: word rather than converted: numpy has no bfloat16 and the safetensors payload already carries the
#: encoding, so a pass-through is lossless where a conversion would not be. ``i1`` is one byte,
#: matching the compiler's generic one-byte ``i1`` storage sizing.
ELEMENT_BYTES: Mapping[str, int] = {
    "i1": 1, "i8": 1, "u8": 1,
    "i16": 2, "u16": 2, "f16": 2, "bf16": 2,
    "i32": 4, "u32": 4, "f32": 4,
    "i64": 8, "u64": 8, "f64": 8,
}

#: Byte alignment between tensors in the blob. A property of the DMA, not of the mesh, and separate
#: from the row pitch for that reason.
DEFAULT_ALIGNMENT = 64


class BundlePackError(ValueError):
    """The bundle cannot be packed, and the message says which tensor and why."""


@dataclass(frozen=True)
class ArgRef:
    """One kernel-ABI argument: its declared name, its parsed index, and how it is accessed."""

    tensor: str
    index: int
    access: str

    def to_dict(self) -> dict[str, Any]:
        return {"tensor": self.tensor, "index": self.index, "access": self.access}


@dataclass(frozen=True)
class PackedTensor:
    """Where one tensor lands, how big it is physically, and which sizing rule decided that."""

    tensor: str
    index: int
    storage: str                 # "const" | "mutable"
    offset: int
    logical_bytes: int
    physical_bytes: int
    dtype: str
    shape: tuple[int, ...]
    sizing: str                  # "declared_storage_encoding" | "row_pitch"
    weight: str = ""             # the state-dict key this comes from, for a const tensor

    def to_dict(self) -> dict[str, Any]:
        return {"tensor": self.tensor, "index": self.index, "storage": self.storage,
                "offset": self.offset, "logical_bytes": self.logical_bytes,
                "physical_bytes": self.physical_bytes, "dtype": self.dtype,
                "shape": list(self.shape), "sizing": self.sizing, "weight": self.weight}


@dataclass
class PackPlan:
    """The full layout, plus every gap and assumption a reader would otherwise have to infer."""

    const: list[PackedTensor] = field(default_factory=list)
    mutable: list[PackedTensor] = field(default_factory=list)
    const_bytes: int = 0
    mutable_bytes: int = 0
    row_pitch_elements: int = 0
    alignment: int = DEFAULT_ALIGNMENT
    #: Argument indices declared by the program plan but ABSENT from the kernel ABI. Reported, never
    #: closed up: SmolVLA's arg809 is real and skipping it silently shifts every later tensor.
    absent_indices: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_bundle_pack_plan_v1",
                "const_bytes": self.const_bytes, "mutable_bytes": self.mutable_bytes,
                "row_pitch_elements": self.row_pitch_elements, "alignment": self.alignment,
                "absent_indices": list(self.absent_indices),
                "n_const": len(self.const), "n_mutable": len(self.mutable),
                "const": [t.to_dict() for t in self.const],
                "mutable": [t.to_dict() for t in self.mutable],
                "notes": list(self.notes)}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()


def parse_arg_index(name: Any, *, prefix: str = "arg") -> int | None:
    """The integer index in ``arg<N>``, or None. Parsed structurally, never by position.

    Returning None rather than raising lets a caller distinguish "this argument is not one of the
    positional graph inputs" from "this argument is malformed", which are different situations.
    """
    if not isinstance(name, str) or not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    if not tail.isdigit():
        return None
    return int(tail)


def read_only_prefix(kernel_abi: Mapping[str, Any]) -> tuple[tuple[ArgRef, ...],
                                                             tuple[ArgRef, ...]]:
    """``(read, write)`` argument refs, in declared order, with indices PARSED from the names.

    The ABI's own order is preserved rather than sorted: it is the order the harness passes pointers
    in, and re-ordering here would produce a layout the kernel does not agree with. A read argument
    appearing after a write one is refused, because the const/mutable split is what makes one blob
    read-only.
    """
    rows = kernel_abi.get("args")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise BundlePackError("the kernel ABI declares no argument list")
    read: list[ArgRef] = []
    write: list[ArgRef] = []
    seen_write = False
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BundlePackError(f"kernel ABI argument {position} is not a mapping")
        tensor = row.get("tensor")
        access = str(row.get("access") or "")
        if not isinstance(tensor, str) or not tensor:
            raise BundlePackError(f"kernel ABI argument {position} declares no tensor name")
        if access not in ("read", "write"):
            raise BundlePackError(
                f"kernel ABI argument {tensor!r} declares access {access!r}; only 'read' and "
                f"'write' decide which blob a tensor belongs in, and guessing would put a weight in "
                f"the mutable arena or an output in read-only memory")
        index = parse_arg_index(tensor)
        ref = ArgRef(tensor=tensor, index=-1 if index is None else index, access=access)
        if access == "read":
            if seen_write:
                raise BundlePackError(
                    f"read argument {tensor!r} appears after a write argument; the const blob is the "
                    f"read-only PREFIX of the ABI, so an interleaved order has no such prefix")
            read.append(ref)
        else:
            seen_write = True
            write.append(ref)
    if not read:
        raise BundlePackError("the kernel ABI declares no read-only arguments to pack")
    return tuple(read), tuple(write)


def element_bytes(dtype: Any) -> int:
    """Bytes per element for a declared dtype, or a refusal naming it.

    Fails closed on an unknown spelling. The shipped script's table omitted ``bf16``, which is 275 of
    SmolVLA's arguments -- and a KeyError there is the *good* outcome; a default of 4 would have
    doubled every one of them and produced a blob whose size looked deliberate.
    """
    key = str(dtype or "")
    if key not in ELEMENT_BYTES:
        raise BundlePackError(
            f"dtype {key!r} has no declared element size; add it to ELEMENT_BYTES with its width "
            f"rather than letting a default decide how many bytes a tensor occupies")
    return ELEMENT_BYTES[key]


def physical_nbytes(tensor: Mapping[str, Any], *, row_pitch_elements: int) -> int:
    """The bytes a tensor occupies once its last axis is padded to the row pitch."""
    raw = tensor.get("shape")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise BundlePackError(f"tensor {tensor.get('name', '?')!r} declares no shape")
    shape = [int(v) for v in raw]
    if any(v <= 0 for v in shape):
        raise BundlePackError(f"tensor shape {shape} has a non-positive extent")
    width = element_bytes(tensor.get("dtype"))
    if not shape:
        return width
    rows = math.prod(shape[:-1])
    pitch = _align(shape[-1], row_pitch_elements)
    return rows * pitch * width


def _align(value: int, amount: int) -> int:
    if amount <= 0:
        raise BundlePackError(f"alignment/pitch must be positive, got {amount}")
    return (value + amount - 1) // amount * amount


def row_pitch_from_manifest(manifest: Mapping[str, Any]) -> int:
    """The row pitch in elements, from the target's own CIRCT-derived mesh facts.

    REQUIRED FROM FACTS, never defaulted. The pitch is the systolic width; a wrong one does not fail,
    it mis-sizes every tensor in the blob while producing a file of a plausible size. So a manifest
    that does not state its mesh raises, and the caller has to go and derive it.
    """
    mesh = ((manifest or {}).get("capabilities") or {}).get("mesh") or {}
    cols = mesh.get("cols")
    if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
        raise BundlePackError(
            "the capability manifest states no capabilities.mesh.cols, so the row pitch cannot be "
            "derived; it is the systolic width and a guessed value mis-sizes every tensor in the "
            "blob without failing")
    return int(cols)


def plan(command_buffer: Mapping[str, Any], *, row_pitch_elements: int,
         weight_manifest: Mapping[str, Any] | None = None,
         alignment: int = DEFAULT_ALIGNMENT) -> PackPlan:
    """Lay out the const and mutable blobs for one emitted program.

    ``weight_manifest`` maps a stringified argument index to that argument's capture metadata (the
    ``weights.safetensors.manifest.json`` shape). It is checked against the read-only prefix by INDEX
    rather than by count, so a gap like SmolVLA's missing ``arg809`` is reported instead of shifting
    every later tensor by one.
    """
    tensors = command_buffer.get("tensors")
    if not isinstance(tensors, Mapping):
        raise BundlePackError("the command buffer declares no tensor table")
    abi = command_buffer.get("kernel_abi")
    if not isinstance(abi, Mapping):
        raise BundlePackError("the command buffer declares no kernel ABI, so it has no pointer order")
    read, write = read_only_prefix(abi)

    params = command_buffer.get("params") if isinstance(command_buffer.get("params"), Mapping) else {}
    encodings = params.get("storage_encodings")
    encodings = encodings if isinstance(encodings, Mapping) else {}

    out = PackPlan(row_pitch_elements=int(row_pitch_elements), alignment=int(alignment))

    declared = params.get("global_program_plan")
    bindings = (declared or {}).get("entry_bindings") if isinstance(declared, Mapping) else None
    if isinstance(bindings, Sequence) and not isinstance(bindings, (str, bytes)):
        planned = {i for i in (parse_arg_index(b if isinstance(b, str) else (b or {}).get("tensor"))
                               for b in bindings) if i is not None}
        present = {ref.index for ref in (*read, *write) if ref.index >= 0}
        absent = tuple(sorted(planned - present))
        out.absent_indices = absent
        if absent:
            out.notes.append(
                f"argument index/indices {list(absent)} are declared by the program plan and ABSENT "
                f"from the kernel ABI; they are NOT packed and every later tensor keeps its own "
                f"parsed index, because closing the gap would mis-address each of them")

    if weight_manifest is not None:
        keys = {int(k) for k in weight_manifest if str(k).isdigit()}
        read_indices = {ref.index for ref in read if ref.index >= 0}
        missing = sorted(read_indices - keys)
        extra = sorted(keys - read_indices)
        if missing:
            raise BundlePackError(
                f"the weight manifest has no entry for read argument index/indices {missing[:8]}; "
                f"packing them would need bytes nobody captured")
        if extra:
            out.notes.append(
                f"the weight manifest carries {len(extra)} entry/entries the ABI never reads "
                f"({extra[:8]}); they are not packed")

    for group, refs in (("const", read), ("mutable", write)):
        cursor = 0
        rows: list[PackedTensor] = []
        for ref in refs:
            tensor = tensors.get(ref.tensor)
            if not isinstance(tensor, Mapping):
                raise BundlePackError(
                    f"kernel ABI argument {ref.tensor!r} has no entry in the tensor table, so its "
                    f"size is unknown")
            shape = tuple(int(v) for v in (tensor.get("shape") or ()))
            dtype = str(tensor.get("dtype") or "")
            logical = math.prod(shape) * element_bytes(dtype) if shape else element_bytes(dtype)
            encoded = encodings.get(ref.tensor) if isinstance(encodings, Mapping) else None
            if isinstance(encoded, Mapping) and isinstance(encoded.get("storage_elements"), int) \
                    and encoded["storage_elements"] > 0:
                physical = int(encoded["storage_elements"]) * element_bytes(dtype)
                sizing = "declared_storage_encoding"
            else:
                physical = physical_nbytes(tensor, row_pitch_elements=row_pitch_elements)
                sizing = "row_pitch"
            entry = weight_manifest.get(str(ref.index)) if weight_manifest else None
            rows.append(PackedTensor(
                tensor=ref.tensor, index=ref.index, storage=group, offset=cursor,
                logical_bytes=logical, physical_bytes=physical, dtype=dtype, shape=shape,
                sizing=sizing,
                weight=str((entry or {}).get("weight") or "") if isinstance(entry, Mapping) else ""))
            cursor = _align(cursor + physical, alignment)
        if group == "const":
            out.const, out.const_bytes = rows, cursor
        else:
            out.mutable, out.mutable_bytes = rows, cursor

    by_rule: dict[str, int] = {}
    for row in (*out.const, *out.mutable):
        by_rule[row.sizing] = by_rule.get(row.sizing, 0) + 1
    out.notes.append(
        "sizing rule counts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_rule.items()))
        + ("; params.storage_encodings is absent for this model, so the pitch formula decided"
           if not encodings else ""))
    return out
