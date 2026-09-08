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

__all__ = ["ArgRef", "PackedTensor", "PackPlan", "SessionState", "session_states_from_contract",
           "ROLES", "parse_arg_index", "read_only_prefix",
           "element_bytes", "physical_nbytes", "plan", "row_pitch_from_manifest",
           "padded_tensor_bytes", "prepack_bytes", "write_const_blob", "WeightSource",
           "PREPACK_PERMUTATIONS",
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

#: Manifest fields that can name the source a packed tensor's bytes come from, most specific first.
#: A capture spells a parameter's source under ``weight`` and a graph INPUT's under ``name`` -- so
#: reading only the first key refuses the input, which is what happened: ResNet-50's ``arg216`` is
#: the image, its manifest entry is ``{"kind": "input", "name": "image"}``, and the packer asked the
#: source for ``arg216``. Both spellings are read, in order, rather than one being assumed.
WEIGHT_KEY_FIELDS = ("weight", "name")


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
    #: ``argument`` -- the ABI passes a pointer to this. ``seed`` -- it holds a carried state's
    #: INITIAL bytes in read-only memory and nothing points at it; the harness copies it into the
    #: mutable working copy before the session runs, and again after the warm invocation.
    role: str = "argument"

    def to_dict(self) -> dict[str, Any]:
        return {"tensor": self.tensor, "index": self.index, "storage": self.storage,
                "offset": self.offset, "logical_bytes": self.logical_bytes,
                "physical_bytes": self.physical_bytes, "dtype": self.dtype,
                "shape": list(self.shape), "sizing": self.sizing, "weight": self.weight,
                "role": self.role}


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
    #: Tensor names in the kernel ABI's own pointer order. RECORDED rather than reconstructed: with
    #: a recurrent session the const and mutable groups no longer partition the ABI in order, so
    #: ``const + mutable`` is not the call's argument order and using it would pass every pointer
    #: after the first carried state to the wrong parameter.
    abi_order: tuple[str, ...] = ()
    #: One row per declared carried state: which seed feeds which working copy from which output.
    carried: tuple[dict[str, Any], ...] = ()
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_bundle_pack_plan_v1",
                "const_bytes": self.const_bytes, "mutable_bytes": self.mutable_bytes,
                "row_pitch_elements": self.row_pitch_elements, "alignment": self.alignment,
                "absent_indices": list(self.absent_indices),
                "abi_order": list(self.abi_order), "carried": [dict(c) for c in self.carried],
                "n_const": len(self.const), "n_mutable": len(self.mutable),
                "const": [t.to_dict() for t in self.const],
                "mutable": [t.to_dict() for t in self.mutable],
                "notes": list(self.notes)}

    @property
    def arguments(self) -> tuple[PackedTensor, ...]:
        """Every row the ABI passes a pointer to, IN THE ABI's DECLARED ORDER.

        The only correct source for a call's argument list. ``const + mutable`` happens to equal it
        whenever every read argument precedes every write one -- true of a feed-forward model and
        FALSE the moment a recurrent session moves a carried input into the mutable blob.
        """
        by_name = {t.tensor: t for t in (*self.const, *self.mutable) if t.role == "argument"}
        missing = [name for name in self.abi_order if name not in by_name]
        if missing:
            raise BundlePackError(
                f"the ABI declares argument(s) {missing[:8]} that this plan lays out no pointer "
                f"for; a short pointer list shifts every later argument")
        return tuple(by_name[name] for name in self.abi_order)

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


#: What a packed row can be. A ``seed`` exists only because a carried state's initial bytes must
#: survive in read-only memory: the session overwrites the working copy every step, and the warm
#: invocation of a warm-then-measure profile overwrites it before the measured one -- so without a
#: re-seed the measured invocation is a DIFFERENT program from the warm one, which is the quiet way
#: a recurrent model's cycle count stops meaning anything.
ROLES: tuple[str, ...] = ("argument", "seed")


@dataclass(frozen=True)
class SessionState:
    """One value the session carries from an output back into an input, as the CAPTURE declares it.

    A recurrent program's carried state is BOTH read and written, so it cannot live in the read-only
    blob -- and the emitted command buffer cannot tell: within one invocation the state input really
    is read-only, and only the session contract knows the loop writes it back. So the placement is
    driven by the capture's declaration, and a state whose declaration does not match the ABI is
    refused rather than placed on a guess.
    """

    name: str
    input_arg: int
    output_index: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "input_arg": self.input_arg,
                "output_index": self.output_index}


def session_states_from_contract(contract: Mapping[str, Any]) -> tuple[SessionState, ...]:
    """The carried states a session contract declares, parsed structurally.

    Reads the contract's own ``states`` list. A contract declaring no states describes a
    feed-forward program and yields ``()``, which plans exactly as before.
    """
    rows = contract.get("states")
    if rows is None:
        return ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise BundlePackError("the session contract's 'states' is not a list of declarations")
    states: list[SessionState] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise BundlePackError(f"session contract states[{position}] is not a mapping")
        for required in ("input_arg", "output_index"):
            if not isinstance(row.get(required), int):
                raise BundlePackError(
                    f"session contract states[{position}] declares no integer {required!r}; a "
                    f"carried state whose endpoints are unknown cannot be placed, and placing its "
                    f"input in read-only memory is a write fault at step 1")
        states.append(SessionState(name=str(row.get("name") or f"state{position}"),
                                   input_arg=int(row["input_arg"]),
                                   output_index=int(row["output_index"])))
    seen_in: dict[int, str] = {}
    seen_out: dict[int, str] = {}
    for state in states:
        for key, table, label in ((state.input_arg, seen_in, "input_arg"),
                                  (state.output_index, seen_out, "output_index")):
            if key in table:
                raise BundlePackError(
                    f"session states {table[key]!r} and {state.name!r} both claim {label} {key}; "
                    f"two states sharing one endpoint would each overwrite the other's carry")
            table[key] = state.name
    return tuple(states)


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
         session_states: Sequence[SessionState] = (),
         alignment: int = DEFAULT_ALIGNMENT) -> PackPlan:
    """Lay out the const and mutable blobs for one emitted program.

    ``weight_manifest`` maps a stringified argument index to that argument's capture metadata (the
    ``weights.safetensors.manifest.json`` shape). It is checked against the read-only prefix by INDEX
    rather than by count, so a gap like SmolVLA's missing ``arg809`` is reported instead of shifting
    every later tensor by one.

    ``session_states`` are the carries the capture declares (:func:`session_states_from_contract`).
    Each named input keeps its bytes in the const blob as a ``seed`` and gains a ``mutable`` working
    copy that the ABI pointer targets, because the session writes the state back every step. Without
    it the plan puts a written tensor in read-only memory: SmolVLA's three carries all land in the
    const blob from the command buffer alone, since within ONE invocation they genuinely are reads.
    """
    tensors = command_buffer.get("tensors")
    if not isinstance(tensors, Mapping):
        raise BundlePackError("the command buffer declares no tensor table")
    abi = command_buffer.get("kernel_abi")
    if not isinstance(abi, Mapping):
        raise BundlePackError("the command buffer declares no kernel ABI, so it has no pointer order")
    read, write = read_only_prefix(abi)
    out_abi_order = tuple(ref.tensor for ref in (*read, *write))

    # Resolve each declared carry against the ABI before laying anything out, so a contract that
    # does not describe THIS program is refused rather than half-applied.
    carried_by_tensor: dict[str, dict[str, Any]] = {}
    if session_states:
        read_by_index = {ref.index: ref for ref in read if ref.index >= 0}
        for state in session_states:
            source = read_by_index.get(state.input_arg)
            if source is None:
                write_hit = next((r for r in write if r.index == state.input_arg), None)
                if write_hit is not None:
                    continue  # already a write argument; nothing to reclassify
                raise BundlePackError(
                    f"session state {state.name!r} declares input_arg {state.input_arg}, which is "
                    f"not a read argument of this kernel ABI; the contract does not describe this "
                    f"program and applying it would move some other tensor into mutable memory")
            if not 0 <= state.output_index < len(write):
                raise BundlePackError(
                    f"session state {state.name!r} declares output_index {state.output_index} but "
                    f"the ABI has {len(write)} write argument(s); an out-of-range carry would read "
                    f"the destination from whichever tensor happened to be there")
            destination = write[state.output_index]
            src_t = tensors.get(source.tensor)
            dst_t = tensors.get(destination.tensor)
            if not isinstance(src_t, Mapping) or not isinstance(dst_t, Mapping):
                raise BundlePackError(
                    f"session state {state.name!r} names tensors absent from the tensor table")
            src_dtype, dst_dtype = str(src_t.get("dtype") or ""), str(dst_t.get("dtype") or "")
            src_shape = tuple(int(v) for v in (src_t.get("shape") or ()))
            dst_shape = tuple(int(v) for v in (dst_t.get("shape") or ()))
            if src_dtype != dst_dtype or src_shape != dst_shape:
                raise BundlePackError(
                    f"session state {state.name!r} carries {destination.tensor} "
                    f"({dst_dtype} {list(dst_shape)}) back into {source.tensor} "
                    f"({src_dtype} {list(src_shape)}); a carry between different layouts copies "
                    f"the right byte count into the wrong elements, which no size check can see")
            carried_by_tensor[source.tensor] = {
                "state": state.name, "input_arg": state.input_arg,
                "output_index": state.output_index, "seed_tensor": source.tensor,
                "output_tensor": destination.tensor, "dtype": src_dtype,
                "shape": list(src_shape)}

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

    carried_working: list[ArgRef] = [ref for ref in read if ref.tensor in carried_by_tensor]
    for group, refs in (("const", read), ("mutable", (*write, *carried_working))):
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
            entry = (weight_manifest.get(str(ref.index)) or {}) if weight_manifest else {}
            # A carried state's const row is the SEED -- nothing points at it; its working copy
            # lives in the mutable blob and takes the ABI pointer.
            role = "seed" if (group == "const" and ref.tensor in carried_by_tensor) else "argument"
            rows.append(PackedTensor(
                tensor=ref.tensor, index=ref.index, storage=group, offset=cursor,
                logical_bytes=logical, physical_bytes=physical, dtype=dtype, shape=shape,
                sizing=sizing,
                weight=next((str(entry[f]) for f in WEIGHT_KEY_FIELDS
                             if isinstance(entry.get(f), str) and entry[f]), ""),
                role=role))
            cursor = _align(cursor + physical, alignment)
        if group == "const":
            out.const, out.const_bytes = rows, cursor
        else:
            out.mutable, out.mutable_bytes = rows, cursor

    out.abi_order = out_abi_order
    if carried_by_tensor:
        seeds = {t.tensor: t for t in out.const if t.role == "seed"}
        working = {t.tensor: t for t in out.mutable if t.tensor in carried_by_tensor}
        produced = {t.tensor: t for t in out.mutable if t.role == "argument"}
        rows_out = []
        for name, row in carried_by_tensor.items():
            rows_out.append({**row,
                             "seed_offset": seeds[name].offset,
                             "working_offset": working[name].offset,
                             "output_offset": produced[row["output_tensor"]].offset,
                             "bytes": working[name].physical_bytes})
        out.carried = tuple(sorted(rows_out, key=lambda r: r["input_arg"]))
        out.notes.append(
            f"{len(out.carried)} carried session state(s) moved OUT of the read-only blob: their "
            f"ABI pointers target mutable working copies and the const blob keeps a seed of each, "
            f"because the session writes the state back every step and the warm invocation of a "
            f"warm-then-measure profile would otherwise leave the measured one a different program")

    by_rule: dict[str, int] = {}
    for row in (*out.const, *out.mutable):
        by_rule[row.sizing] = by_rule.get(row.sizing, 0) + 1
    out.notes.append(
        "sizing rule counts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_rule.items()))
        + ("; params.storage_encodings is absent for this model, so the pitch formula decided"
           if not encodings else ""))
    return out


# ---------------------------------------------------------------------------------------------------
# writing the bytes
# ---------------------------------------------------------------------------------------------------

#: A source of one tensor's raw little-endian bytes, by the weight key the manifest names. Kept as a
#: protocol rather than a safetensors dependency so a caller can pack from a capture, a re-quantized
#: instance, or a test fixture without this module learning any container format.
WeightSource = "Callable[[str], bytes]"


#: How a declared ``(source_layout, packed_layout)`` transition reorders elements. ``None`` means the
#: transition is a pure RESHAPE -- the element sequence is unchanged, so the bytes are already in
#: packed order -- and a tuple is the axis permutation to apply to the SOURCE shape.
#:
#: Read from the recipe the command buffer declares, so nothing here is a fact about a target: a
#: backend that prepacks differently declares a different transition and gets refused until this
#: table describes it, which is the intended outcome. Silently treating an undescribed transition as
#: a reshape is what produced a blob whose 216 other tensors were byte-identical and whose dense
#: weight was transposed -- correct arithmetic on the wrong bytes, for one layer out of 54.
PREPACK_PERMUTATIONS: Mapping[tuple[str, str], tuple[int, ...] | None] = {
    ("OIHW", "CoK_dim_padded"): None,          # row-major flatten of I,H,W into K: same sequence
    ("NK", "KN_dim_padded"): (1, 0),           # transpose
}


def prepack_bytes(raw: bytes, recipe: Mapping[str, Any], *, dtype: str) -> bytes:
    """``raw`` reordered from the recipe's source layout into its packed layout.

    Operates on a BYTE view with an explicit item size, never on a decoded numeric array, so a dtype
    numpy cannot represent (``bf16``) is permuted losslessly: the elements move and their bytes are
    carried along untouched.

    An undeclared transition RAISES. The alternative -- assuming a reshape -- is exactly the failure
    this exists to prevent, and it is invisible whenever the element count happens to match, which
    for a transpose it always does.
    """
    import numpy as _np  # noqa: PLC0415

    source_layout = str(recipe.get("source_layout") or "")
    packed_layout = str(recipe.get("packed_layout") or "")
    key = (source_layout, packed_layout)
    if key not in PREPACK_PERMUTATIONS:
        raise BundlePackError(
            f"the buffer declares a weight prepack {source_layout!r} -> {packed_layout!r} that this "
            f"packer does not describe (it knows {sorted(PREPACK_PERMUTATIONS)}); refusing rather "
            f"than assuming a reshape, because an element reordering assumed away is invisible "
            f"whenever the element count matches -- which for a permutation it always does")
    permutation = PREPACK_PERMUTATIONS[key]
    if permutation is None:
        return raw
    source_shape = [int(v) for v in (recipe.get("source_shape") or ())]
    if len(source_shape) != len(permutation):
        raise BundlePackError(
            f"prepack {key} permutes {len(permutation)} axes but the recipe declares a "
            f"{len(source_shape)}-D source shape {source_shape}")
    width = element_bytes(dtype)
    expected = 1
    for extent in source_shape:
        expected *= extent
    if len(raw) != expected * width:
        raise BundlePackError(
            f"prepack {key} needs {expected * width} source byte(s) for shape {source_shape} "
            f"({dtype}) and got {len(raw)}")
    view = _np.frombuffer(raw, dtype=_np.uint8).reshape(*source_shape, width)
    moved = view.transpose(*permutation, len(source_shape))
    return _np.ascontiguousarray(moved).tobytes()


def padded_tensor_bytes(raw: bytes, tensor: PackedTensor, *, row_pitch_elements: int) -> bytes:
    """``raw`` re-laid-out to this tensor's physical footprint, or a refusal naming the mismatch.

    The last axis is padded to the row pitch and the trailing bytes are zero. Padding is written
    explicitly rather than left to whatever the blob was initialised with: a reader of the emitted
    file cannot tell an intentional zero from an uninitialised one, and a weight whose pad bytes
    carry the previous tensor's tail is a silent wrong answer on any engine that reads the full row.

    Operates on BYTES, never on a decoded numeric array. That is what lets ``bf16`` -- which numpy
    cannot represent -- travel losslessly: the capture already holds the encoding, so a move is
    exact where a conversion would not be.
    """
    width = element_bytes(tensor.dtype)
    logical = 1
    for extent in tensor.shape:
        logical *= int(extent)
    expected = logical * width
    if len(raw) != expected:
        raise BundlePackError(
            f"{tensor.tensor!r} ({tensor.dtype}, shape {list(tensor.shape)}) needs {expected} "
            f"byte(s) and the source supplied {len(raw)}; a short read would pack the next tensor's "
            f"bytes into this one's tail and a long one would silently truncate")
    if not tensor.shape:
        return raw.ljust(tensor.physical_bytes, b"\x00")
    cols = int(tensor.shape[-1])
    rows = logical // cols if cols else 0
    pitch = _align(cols, row_pitch_elements)
    if pitch == cols:
        out = raw
    else:
        row_src, row_dst = cols * width, pitch * width
        buf = bytearray(rows * row_dst)
        for r in range(rows):
            buf[r * row_dst:r * row_dst + row_src] = raw[r * row_src:(r + 1) * row_src]
        out = bytes(buf)
    if len(out) > tensor.physical_bytes:
        raise BundlePackError(
            f"{tensor.tensor!r} lays out to {len(out)} byte(s) but the plan reserved "
            f"{tensor.physical_bytes}; the layout and the writer disagree, which would shift every "
            f"later tensor in the blob")
    # The reserved footprint can exceed the pitched layout when the plan sized this tensor from a
    # declared storage encoding rather than from the pitch. Zero-filled to the reservation so the
    # NEXT tensor still lands on the offset the plan promised.
    return out.ljust(tensor.physical_bytes, b"\x00")


def write_const_blob(plan: PackPlan, source, out_path: Any, *,
                     weight_manifest: Mapping[str, Any] | None = None,
                     prepack_recipes: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Write the constant blob this plan describes, and return a receipt for what went into it.

    ``source`` is called with the weight KEY the manifest names for each argument (falling back to
    the argument's own tensor name when the manifest carries none) and returns that tensor's raw
    bytes. Every tensor in the plan's const region must be supplied: a hole would be indistinguishable
    from a zero weight, which trains and infers as a real number.

    Written by SEEKING to each tensor's planned offset rather than by appending, so the file the
    device reads and the layout the harness was rendered against cannot drift apart -- and the
    receipt records the digest of the bytes actually written, not of what was intended.
    """
    import hashlib as _hashlib  # noqa: PLC0415

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = weight_manifest or {}
    # Keyed by the tensor the recipe names, so a buffer that prepacks only some weights is handled
    # by looking each one up rather than by assuming a uniform treatment.
    recipes = {str(r.get("tensor")): r for r in prepack_recipes if isinstance(r, Mapping)}
    written: list[dict[str, Any]] = []
    digest = _hashlib.sha256()
    cursor = 0
    with target.open("wb") as stream:
        for tensor in plan.const:
            entry = manifest.get(str(tensor.index)) or {}
            key = next((str(entry[f]) for f in WEIGHT_KEY_FIELDS
                        if isinstance(entry.get(f), str) and entry[f]), tensor.tensor)
            try:
                raw = source(key)
            except KeyError as exc:
                raise BundlePackError(
                    f"the source has no bytes for {key!r} (argument {tensor.index}, tensor "
                    f"{tensor.tensor!r}); a missing weight cannot be packed as zeros because a zero "
                    f"weight is a real number the device would happily compute with") from exc
            if raw is None:
                raise BundlePackError(f"the source returned no bytes for {key!r}")
            staged = bytes(raw)
            recipe = recipes.get(tensor.tensor)
            if recipe is not None:
                staged = prepack_bytes(staged, recipe, dtype=tensor.dtype)
            body = padded_tensor_bytes(staged, tensor,
                                       row_pitch_elements=plan.row_pitch_elements)
            if tensor.offset < cursor:
                raise BundlePackError(
                    f"{tensor.tensor!r} is planned at offset {tensor.offset} but {cursor} bytes are "
                    f"already written; the plan's offsets are not monotonic")
            if tensor.offset > cursor:
                # Inter-tensor alignment padding, written explicitly for the same reason the row pad
                # is: an uninitialised gap is indistinguishable from a deliberate one.
                gap = b"\x00" * (tensor.offset - cursor)
                stream.write(gap)
                digest.update(gap)
                cursor = tensor.offset
            stream.write(body)
            digest.update(body)
            cursor += len(body)
            written.append({"tensor": tensor.tensor, "index": tensor.index, "weight": key,
                            "offset": tensor.offset, "bytes": len(body), "dtype": tensor.dtype,
                            "sizing": tensor.sizing,
                            "prepack": (f"{recipe.get('source_layout')}->"
                                        f"{recipe.get('packed_layout')}") if recipe else None})
        if cursor < plan.const_bytes:
            tail = b"\x00" * (plan.const_bytes - cursor)
            stream.write(tail)
            digest.update(tail)
            cursor += len(tail)
    if cursor != plan.const_bytes:
        raise BundlePackError(
            f"wrote {cursor} byte(s) for a plan that reserved {plan.const_bytes}; the harness was "
            f"rendered against the plan, so a size disagreement means the device reads the wrong "
            f"offsets")
    return {"schema": "merlin_const_blob_receipt_v1", "path": str(target),
            "bytes": cursor, "sha256": digest.hexdigest(),
            "n_tensors": len(written), "row_pitch_elements": plan.row_pitch_elements,
            "alignment": plan.alignment, "plan_digest": plan.digest(), "tensors": written}
