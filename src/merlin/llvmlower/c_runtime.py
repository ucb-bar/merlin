"""Generate the data-driven inputs the Merlin C runtime needs to drive a compiled model.

The C runtime (`merlin/runtime/c/merlin_model.c`) is generic: it builds MLIR memref
descriptors from an **argument table** and invokes the model's C interface. This module
emits, from the model's MLIR signature + safetensors manifest:

- ``model_gen.h``   — the arg table (kind, weight-blob offset, rank, dims, elem size) +
                      the output shape; consumed by the generic runtime.
- ``model_call.c``  — ``merlin_invoke(void **descs)`` unrolling the N-pointer
                      ``_mlir_ciface_forward`` call (C has no >1024-arg limit, but the
                      arity is model-specific, so it is generated once here).
- ``weights.bin``   — the raw safetensors payload (weights blob; mmap/embed as-is).
- ``model_io.h``    — embedded runtime inputs, output buffers, and optional state-carry map.

Nothing here is target-specific: the same artifacts feed the host and the spike builds.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .model_runner import parse_forward_signature
from .weight_payload import WeightPayload
from .weights_pack import load_safetensors_header

if TYPE_CHECKING:
    from .entry_weight_projection import EntryWeightProjection

DT_BYTES = {"f32": 4, "f64": 8, "bf16": 2, "f16": 2, "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1}
NP_OF = {
    "f32": np.float32,
    "f64": np.float64,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.bool_,
}
C_OF = {
    "f32": "float",
    "f64": "double",
    "i64": "long",
    "i32": "int",
    "i16": "int16_t",
    "i8": "signed char",
    "i1": "signed char",
    "bf16": "unsigned short",
    "f16": "unsigned short",
}

# Existing portable build-time append policy. This describes generated blob
# placement; it supplies no target DMA, tensor alignment or no-alias fact.
GENERATED_WEIGHT_ALIGNMENT = 64


def _align_weight_rows(rows, blob, appended, append):
    """Repair native element alignment without changing any weight alias relation.

    Captured byte offsets need not satisfy the native typed-load ABI. Relocate
    each affected overlapping component together; never split views or infer
    device DMA alignment. Already aligned components retain their exact bytes
    and offsets. Lazy span copies keep build memory bounded by the I/O chunk.
    """
    payload = WeightPayload()
    payload.extend(blob)
    payload.extend(appended)
    windows = []
    for index, (kind, start, _rank, shape, elem, _dtype) in enumerate(rows):
        if kind == "MERLIN_WEIGHT":
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError("runtime weight descriptor requires static nonnegative dimensions")
            size = prod(shape) * elem
            if start < 0 or start > len(payload) or start + size > len(payload):
                raise ValueError("runtime weight descriptor exceeds actual packed bytes")
            if size:
                windows.append((start, start + size, index))
    groups = []
    for start, end, index in sorted(windows):
        if groups and start < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append((start, index))
        else:
            groups.append([start, end, [(start, index)]])
    repairs = []
    for low, high, members in groups:
        if all(start % rows[index][4] == 0 for start, index in members):
            continue
        reference, reference_index = max(members, key=lambda item: rows[item[1]][4])
        alignment = rows[reference_index][4]
        shift = (-reference) % alignment
        if GENERATED_WEIGHT_ALIGNMENT % alignment or any((start + shift) % rows[index][4] for start, index in members):
            raise ValueError("overlapping weight views have incompatible native element alignment")
        prefix = (low - reference) % alignment
        copied = WeightPayload.zeros(prefix)
        copied.extend(payload[low:high])
        begin = append(copied) + prefix
        for start, index in members:
            kind, _offset, *rest = rows[index]
            rows[index] = (kind, begin + start - low, *rest)
        repairs.append(
            dict(
                descriptor_stage="before_entry_projection",
                source_start=low,
                size=high - low,
                destination_start=begin,
                arguments=[index for _, index in members],
            )
        )
    return repairs


def manifest_requires_weight_blob(manifest: dict) -> bool:
    """Whether any forward argument reads bytes from ``weights.safetensors``.

    Stub parameters receive a generated zero region and runtime inputs are embedded separately, so
    neither requires a source blob. Externalized buffers do: they name a ``weight`` entry just like
    an ordinary parameter.
    """
    for meta in manifest.values():
        if not isinstance(meta, dict):
            continue
        if meta.get("kind") == "param" and not meta.get("stub"):
            return True
        if meta.get("kind") == "buffer" and meta.get("weight"):
            return True
    return False


def _out_specs(mlir_path: str | Path) -> list[tuple[list[int], str]]:
    """Read the result ABI through the shared content-qualified signature cache."""
    from ..common.mlir_query import forward_signature

    _, results = forward_signature(mlir_path)
    if not results:
        raise ValueError(f"forward in {mlir_path} has no result")
    return results


def _bf16_bits(f32: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round-to-nearest-even) and return the raw uint16 bit patterns.

    numpy has no native bf16, so do it by hand: bf16 is the top 16 bits of the f32 encoding; RNE
    adds the round bias 0x7FFF + (lsb of the kept mantissa) before truncating. NaNs are preserved
    (their exponent is all-ones and mantissa nonzero, which survives the shift)."""
    u = f32.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    rounded = (u + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return rounded.astype(np.uint16)


def _embed_array(arr: np.ndarray, dt: str) -> str:
    # 16-bit floats have NO decimal C literal for an ``unsigned short`` storage array: writing
    # `unsigned short x = 0.125` truncates to 0. Emit the RAW 16-bit patterns instead — f16 via
    # numpy's native half, bf16 via RNE from f32 — so the embedded operands are bit-exact.
    if dt == "f16":
        bits = arr.astype(np.float16).view(np.uint16).ravel()
        return ",".join(str(int(v)) for v in bits)
    if dt == "bf16":
        bits = _bf16_bits(np.ascontiguousarray(arr)).ravel()
        return ",".join(str(int(v)) for v in bits)
    flat = arr.astype(NP_OF.get(dt, np.float32)).ravel()
    if dt in {"f32", "f64"}:
        # Decimal `nan`/`inf` are not C constants.  C compiler NaN builtins also
        # need the captured sign, quiet/signaling bit and payload explicitly: a
        # bare NAN macro would silently change the input bytes before execution.
        if dt == "f32":
            raw = flat.view(np.uint32)
            exponent, fraction, quiet, sign_mask, suffix = (0x7F800000, 0x007FFFFF, 0x00400000, 0x80000000, "f")
        else:
            raw = flat.view(np.uint64)
            exponent, fraction, quiet, sign_mask, suffix = (
                0x7FF0000000000000,
                0x000FFFFFFFFFFFFF,
                0x0008000000000000,
                0x8000000000000000,
                "",
            )

        def literal(value: np.floating, bits: np.unsignedinteger) -> str:
            encoded = int(bits)
            if encoded & exponent != exponent:
                return str(float(value))  # preserve the existing finite C spelling
            sign = "-" if encoded & sign_mask else ""
            payload = encoded & fraction
            if payload == 0:
                return f"{sign}__builtin_inf{suffix}()"
            kind = "nan" if payload & quiet else "nans"
            return f'{sign}__builtin_{kind}{suffix}("0x{payload & ~quiet:x}")'

        return ",".join(literal(value, bits) for value, bits in zip(flat, raw, strict=True))
    return ",".join(str(int(v) if "i" in dt else float(v)) for v in flat)


def _storage_nbytes(arr: np.ndarray, dt: str) -> int:
    """Size of the emitted C storage, independent of the capture container dtype."""
    return int(arr.size) * DT_BYTES[dt]


def _storage_bytes(arr: np.ndarray, dt: str) -> bytes:
    """Expected embedded storage words, without interpreting them as answers."""
    if dt == "bf16":
        value = _bf16_bits(np.ascontiguousarray(arr))
    elif dt == "f16":
        value = arr.astype(np.float16).view(np.uint16)
    else:
        value = arr.astype(NP_OF[dt])
    return np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"))).tobytes()


def generate(
    model_dir: str | Path,
    out_dir: str | Path,
    inputs_npz: str | Path,
    extra_npz: str | Path | None = None,
    *,
    ciface_name: str = "forward",
    invoke_name: str = "merlin_invoke",
    max_session_steps: int | None = None,
    prepared_dir: str | Path | None = None,
    entry_projection: EntryWeightProjection | None = None,
    entry_projection_source: str | Path | None = None,
    materialize_weights: bool = True,
    weights_source: str | Path | None = None,
    readonly_input_indices: tuple[int, ...] = (),
    max_readonly_input_bytes: int = 1 << 20,
) -> dict:
    """Emit the runtime-driving artifacts for a captured model into ``out_dir``.

    Non-weight args are embedded as C arrays: real inputs from ``inputs_npz`` (by order),
    and non-persistent buffers / lifted constants (rotary inv_freq, etc.) from
    ``extra_npz`` (matched by manifest name; ``buf::`` keys for buffers, bare keys for
    lifted constants). Weights stay in the blob, referenced by offset.

    ``prepared_dir`` is the authority for build-time ABI additions emitted while rewriting the
    captured module. It is deliberately separate from ``out_dir``, which only receives generated C
    artifacts. Callers that do no preparation may leave it unset.

    ``max_session_steps`` caps how many steps of a session corpus are embedded (streams and the
    correctness/quality references together, so step k still meets reference k). ``None`` embeds the
    whole corpus, which is what every existing caller gets.

    Hash-only generation derives the complete blob without writing it. Explicit
    ``weights_source`` may share immutable bytes only after exact verification.
    All writes replace atomically, so regenerating an output cannot overwrite
    an existing shared inode. Mutable invocation state is never shared here.
    """
    if type(materialize_weights) is not bool:
        raise TypeError("materialize_weights must be a bool")
    from merlin.runtime.input_observation import validate_selection

    validate_selection(readonly_input_indices, max_readonly_input_bytes)
    if weights_source is not None and not materialize_weights:
        raise ValueError("weights_source requires materialize_weights=True")
    weight_storage = os.environ.get("MERLIN_WEIGHT_PAYLOAD_STORAGE", "streamed")
    if weight_storage not in {"streamed", "materialized"}:
        raise ValueError("MERLIN_WEIGHT_PAYLOAD_STORAGE requires streamed or materialized")
    model_dir, out_dir = Path(model_dir).resolve(), Path(out_dir).resolve()
    prepared_dir = Path(prepared_dir).resolve() if prepared_dir is not None else out_dir
    if entry_projection is not None:
        from merlin.common import mlir_query

        from .entry_weight_projection import apply_entry_weight_projection, derive_entry_argument_table

        if entry_projection_source is None or ciface_name != entry_projection.contract.entry_symbol:
            raise ValueError("entry projection needs its original prepared source and matching C interface")
        if derive_entry_argument_table(model_dir, prepared_dir) != entry_projection.argument_table:
            raise ValueError("entry projection argument source identity disagrees")
        apply_entry_weight_projection(mlir_query.parse(Path(entry_projection_source)), entry_projection)
    elif entry_projection_source is not None:
        raise ValueError("entry projection source requires an explicit projection")
    # A build whose PREPARED module has panel-packed weight arguments must be handed the bundle whose
    # blob and manifest were packed to match. The argument table below is built from THIS bundle while
    # the compiled object follows the prepared module, so the stock bundle here would produce an
    # object indexing [N/NR][K][NR] weights against [K][N] bytes -- it links, it runs, and it computes
    # nonsense. The preparation step leaves its plan beside this call's work directory precisely so
    # the mistake is detectable here rather than in a wrong result hours later.
    from .weight_panel import guard_planned_pack

    guard_planned_pack(model_dir, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readonly_sources = []
    if readonly_input_indices:
        from merlin.runtime.input_observation import snapshot_sources

        source_paths = [model_dir / "model.mlir", inputs_npz, model_dir / "weights.safetensors.manifest.json", __file__]
        source_paths.extend(
            path
            for path in (
                model_dir / "weights.safetensors",
                model_dir / "session_contract.yaml",
                model_dir / "input_order.json",
                Path(extra_npz or model_dir / "extra.npz"),
            )
            if path.is_file()
        )
        readonly_sources = snapshot_sources(source_paths)
    sig = parse_forward_signature(model_dir / "model.mlir")
    if any(i >= len(sig) for i in readonly_input_indices):
        raise ValueError("readonly input index is outside the original capture arguments")
    man = json.loads((model_dir / "weights.safetensors.manifest.json").read_text())
    weights_path = model_dir / "weights.safetensors"
    if manifest_requires_weight_blob(man):
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"manifest declares stored parameters but the weight blob is absent: {weights_path}"
            )
        hdr, payload_off = load_safetensors_header(weights_path)
        blob = WeightPayload.file(weights_path, payload_off, weights_path.stat().st_size - payload_off)
    else:
        # A compiled ABI-only stage can be a pure identity/view with no parameters. Captures still
        # carry the argument manifest, but emitting a dummy safetensors container is neither part of
        # the model contract nor necessary for the runtime: qinner/stub bytes, if any, are appended
        # below exactly as for an ordinary blob.
        hdr, payload_off, blob = {}, 0, WeightPayload()
    # BLOB WEIGHTS (merlin.llvmlower.blob_weights): device-layout weights the preparation bound as
    # trailing arguments, and the original arguments only they replaced, whose bytes are dropped here.
    from . import blob_weights as _blob_weights

    blob_plan = _blob_weights.read(prepared_dir)
    if blob_plan.args and entry_projection is not None:
        raise ValueError("blob weights and an entry weight projection cannot share one argument table")
    if blob_plan.dead_args:
        hdr, blob = _blob_weights.compact(hdr, blob, man, blob_plan.dead_args, appended=blob_plan.nbytes)
    inputs = np.load(inputs_npz)
    extra_path = extra_npz or (model_dir / "extra.npz")
    extra = np.load(extra_path) if Path(extra_path).is_file() else {}
    lifted_names = sorted(k for k in getattr(extra, "files", []) if not k.startswith("buf::"))

    def buffer_array(name: str):
        # manifest buffer name b_a_b_c  <->  extra key buf::a.b.c
        for k in getattr(extra, "files", []):
            if k.startswith("buf::") and "b_" + k[len("buf::") :].replace(".", "_") == name:
                return np.ascontiguousarray(extra[k])
        raise KeyError(f"buffer {name!r} not in {extra_path}")

    # weights blob (payload, byte-identical to what the manifest offsets index). Anything the
    # bundle does NOT store as a packed weight but the compiled model still has to READ -- the
    # quantized-subclass inner tensors, the zero region a stubbed argument's dead descriptor points
    # at -- is APPENDED here and addressed by its offset past the payload, so it costs no C literals
    # and rides the same mmap/embed path the weights already use.
    appended = WeightPayload()

    def _append_blob(data: WeightPayload) -> int:
        """Place ``data`` after the weights payload (64-byte aligned) and return its blob offset."""
        pad = (-(len(blob) + len(appended))) % GENERATED_WEIGHT_ALIGNMENT
        appended.extend(WeightPayload.zeros(pad))
        begin = len(blob) + len(appended)
        appended.extend(data)
        return begin

    stub_zero_offsets: dict[int, int] = {}  # byte length -> offset of a shared zero region

    out_specs = _out_specs(model_dir / "model.mlir")
    out_shape, out_dt = out_specs[0]

    # manifest input name -> inputs.npz tuple order. Prefer the per-model
    # input_order.json the capture emits (authoritative — e.g. openvla's
    # input_ids/pixel_values, xr0's 11 inputs); fall back to the legacy hardcoded VLA
    # map only when the bundle predates it. Mirrors dispatch_runtime's resolution.
    order_path = model_dir / "input_order.json"
    if order_path.is_file():
        input_order = {k: int(v) for k, v in json.loads(order_path.read_text()).items()}
    else:
        input_order = {"img": 0, "img_mask": 1, "lang_tokens": 2, "lang_masks": 3, "state": 4, "noise": 5, "ids": 0}

    # arg table: weights -> (offset in blob); inputs/buffers/lifted -> embedded arrays.
    rows = []  # (kind, offset, rank, dims, elem_size, dtype)
    io_decls = []  # embedded C arrays
    embedded_arrays: dict[int, tuple[np.ndarray, str]] = {}
    static_io_bytes = 0  # bytes of STATIC storage the harness needs for the model's I/O (see below)
    embedded: set[int] = set()  # arg positions that GOT an embedded array (see the ptr table below)
    n_in = 0  # positional input counter (loaders with a single tuple)
    li = 0  # lifted-constant counter
    for i, (shape, dt) in enumerate(sig):
        meta = man[str(i)]
        elem = DT_BYTES[dt]
        if meta["kind"] == "param":
            if meta.get("stub"):
                # A quantized-subclass weight is STUBBED: the manifest names a placeholder the
                # packer never filled (the fused f32 weight is dead -- the contraction consumes the
                # int8 inner tensors through the quant-inner channel). Reading the placeholder's
                # offset with the ARGUMENT's shape walks off into whatever tensors follow it in the
                # blob; the numpy interpreter hands the same argument zeros. Point the descriptor at
                # a zero region instead, shared between equally sized stubs since they are read-only.
                nbytes = max(1, int(np.prod(shape)) * elem)
                if nbytes not in stub_zero_offsets:
                    stub_zero_offsets[nbytes] = _append_blob(WeightPayload.zeros(nbytes))
                rows.append(("MERLIN_WEIGHT", stub_zero_offsets[nbytes], len(shape), shape, elem, dt))
                continue
            begin, _ = hdr[meta["weight"]]["data_offsets"]
            rows.append(("MERLIN_WEIGHT", begin, len(shape), shape, elem, dt))
            continue
        # non-weight arg: resolve its data and embed it.
        name = meta.get("name", "")
        if meta["kind"] == "buffer" and meta.get("weight") in hdr:
            # An EXTERNALIZED buffer (BatchNorm running stats, for instance) is in the weights blob
            # like a parameter and is named under "weight", not "name" -- so it belongs in the blob
            # arg table, not embedded as a C array. Only a buffer that stayed a graph argument
            # comes from extra.npz, keyed by its FX arg name. Handling just the latter made every
            # externalized buffer a KeyError on an empty name.
            begin, _ = hdr[meta["weight"]]["data_offsets"]
            rows.append(("MERLIN_WEIGHT", begin, len(shape), shape, elem, dt))
            continue
        if meta["kind"] == "buffer":
            arr = buffer_array(name)
        elif name.startswith("c_lifted_tensor_"):
            arr = np.ascontiguousarray(extra[lifted_names[li]])
            li += 1
        elif name in input_order and f"in{input_order[name]}" in inputs.files:
            arr = np.ascontiguousarray(inputs[f"in{input_order[name]}"])
        else:
            arr = np.ascontiguousarray(inputs[f"in{n_in}"])
            n_in += 1
        io_decls.append(f"static {C_OF[dt]} merlin_in_{i}[] = {{{_embed_array(arr, dt)}}};")
        embedded_arrays[i] = (arr, dt)
        static_io_bytes += _storage_nbytes(arr, dt)
        embedded.add(i)
        rows.append(("MERLIN_INPUT", i, len(shape), shape, elem, dt))
    # QUANT-INNER ARGUMENTS. A torchao subclass's int8 `int_data`/`scale` are not `@forward`
    # arguments: the capture parks them in `extra.npz` under `qinner::` keys and leaves an
    # uninitialized `tensor.empty` in the graph. `merlin.llvmlower.qinner.lift` (run from the shared
    # preparation step) turns each into a TRAILING argument, so the same derivation appends matching
    # rows here and the real bytes to the blob. Without this the interpreter binds them and the
    # compiled binary reads whatever was in memory -- one bundle gating cos 1.0 on the host and
    # computing garbage on the board.
    from . import qinner as _qinner

    qinner_args = _qinner.plan_for_bundle(model_dir / "model.mlir")
    if qinner_args:
        for arg in qinner_args:
            size = int(np.prod(arg.shape)) * DT_BYTES[arg.dtype]
            begin = _append_blob(WeightPayload.array(size, lambda arg=arg: _qinner.resolve(extra, [arg])[0]))
            rows.append(("MERLIN_WEIGHT", begin, len(arg.shape), list(arg.shape), DT_BYTES[arg.dtype], arg.dtype))
    # The device group rewrite binds its blob weights after qinner's arguments and before quant-hoist's.
    blob_values = np.load(blob_plan.directory / _blob_weights.VALUES_FILE) if blob_plan.args else {}

    for arg in blob_plan.args:
        size = int(np.prod(arg["shape"])) * DT_BYTES[arg["dtype"]]
        begin = _append_blob(WeightPayload.array(size, lambda arg=arg: blob_plan.array(arg, blob_values)))
        rows.append(
            ("MERLIN_WEIGHT", begin, len(arg["shape"]), list(arg["shape"]), DT_BYTES[arg["dtype"]], arg["dtype"])
        )
    # BUILD-TIME WEIGHT-INVARIANT RESULTS. `quant_hoist.apply` appends these after qinner's
    # arguments in the prepared @forward signature. The plan and bytes live in `prepared_dir`
    # because they are products of preparation, not capture or C generation. Reading either without
    # the other is an ABI
    # error: leaving the new descriptors unbound makes a perfectly linked image compute from
    # arbitrary memory.
    from . import quant_hoist as _quant_hoist

    quant_hoist_args = _quant_hoist.read_plan(prepared_dir)
    quant_hoist_values = _quant_hoist.read_values(prepared_dir)
    if quant_hoist_args:
        present = set(getattr(quant_hoist_values, "files", quant_hoist_values.keys()))
        expected = {arg.key for arg in quant_hoist_args}
        if present != expected:
            raise ValueError(
                "quant-hoist plan/value disagreement: "
                f"missing={sorted(expected - present)}, extra={sorted(present - expected)}"
            )
        numpy_dtype = {
            "f64": np.float64,
            "f32": np.float32,
            "f16": np.float16,
            "i64": np.int64,
            "i32": np.int32,
            "i16": np.int16,
            "i8": np.int8,
            "u8": np.uint8,
        }

        def hoist_array(arg):
            if arg.dtype not in numpy_dtype:
                raise ValueError(f"quant-hoist argument {arg.key!r} has unsupported {arg.dtype}")
            arr = np.ascontiguousarray(quant_hoist_values[arg.key])
            if tuple(arr.shape) != tuple(arg.shape) or arr.dtype != np.dtype(numpy_dtype[arg.dtype]):
                raise ValueError(
                    f"quant-hoist argument {arg.key!r} bytes are {arr.shape}x{arr.dtype}, "
                    f"plan requires {arg.shape}x{arg.dtype}"
                )
            return arr

        for arg in quant_hoist_args:
            size = int(np.prod(arg.shape)) * DT_BYTES[arg.dtype]
            begin = _append_blob(WeightPayload.array(size, lambda arg=arg: hoist_array(arg)))
            rows.append(("MERLIN_WEIGHT", begin, len(arg.shape), list(arg.shape), DT_BYTES[arg.dtype], arg.dtype))
    n_sig_args = len(sig) + len(qinner_args) + len(blob_plan.args) + len(quant_hoist_args)
    # Output rows follow the input/weight rows in MLIR result order. Keeping every result is what
    # lets a captured decoder/LSTM expose its updated state instead of the runtime silently dropping
    # all but result zero.
    for output_index, (shape, dt) in enumerate(out_specs):
        rows.append(("MERLIN_OUTPUT", output_index, len(shape), shape, DT_BYTES[dt], dt))

    # Optional capture-owned state map. Numeric ABI indices are intentional: symbolic state names
    # are provenance for humans, while the captured signature is the executable contract. Each pair
    # says "after a step, copy output_index into input_arg before the next step".
    session_path = model_dir / "session_contract.yaml"
    state_pairs: list[tuple[int, int]] = []
    session: dict = {}
    if session_path.is_file():
        from ..common.yaml import load_yaml

        session = load_yaml(session_path)
        from ..common.schemas import validate_or_raise

        validate_or_raise(session, "session_contract")
        if not isinstance(session, dict) or int(session.get("version", 0)) != 1:
            raise ValueError(f"invalid session contract at {session_path}: expected version 1 mapping")
        for j, state in enumerate(session.get("states", ()) or ()):
            if not isinstance(state, dict):
                raise ValueError(f"{session_path}: states[{j}] must be a mapping")
            input_arg, output_index = int(state["input_arg"]), int(state["output_index"])
            if input_arg < 0 or input_arg >= len(sig):
                raise ValueError(f"{session_path}: states[{j}].input_arg is outside forward inputs")
            if output_index < 0 or output_index >= len(out_specs):
                raise ValueError(f"{session_path}: states[{j}].output_index is outside forward results")
            in_shape, in_dt = sig[input_arg]
            o_shape, o_dt = out_specs[output_index]
            if in_shape != o_shape or in_dt != o_dt:
                raise ValueError(
                    f"{session_path}: state {state.get('name', j)!r} ABI mismatch: "
                    f"input {input_arg} is {in_shape}x{in_dt}, output {output_index} is "
                    f"{o_shape}x{o_dt}"
                )
            state_pairs.append((input_arg, output_index))

    state_input_args = {pair[0] for pair in state_pairs}
    stream_specs: list[tuple[int, np.ndarray, str]] = []
    streams = session.get("streams", ()) or ()
    session_steps = int(session.get("steps", 0) or 0)
    if streams:
        stream_path = model_dir / str(session.get("inputs", "session_inputs.npz"))
        if not stream_path.is_file():
            raise FileNotFoundError(f"{session_path}: session input corpus is absent: {stream_path}")
        if readonly_input_indices:
            readonly_sources.extend(snapshot_sources([stream_path]))
        stream_data = np.load(stream_path)
        for j, stream in enumerate(streams):
            if not isinstance(stream, dict):
                raise ValueError(f"{session_path}: streams[{j}] must be a mapping")
            input_arg, key = int(stream["input_arg"]), str(stream["key"])
            if input_arg in state_input_args:
                raise ValueError(f"{session_path}: input arg {input_arg} cannot be both state and stream")
            if input_arg not in embedded_arrays:
                raise ValueError(f"{session_path}: stream input arg {input_arg} is not a runtime input")
            if key not in stream_data.files:
                raise ValueError(f"{session_path}: stream key {key!r} is absent from {stream_path}")
            arr = np.ascontiguousarray(stream_data[key])
            shape, dt = sig[input_arg]
            if list(arr.shape[1:]) != list(shape) or arr.shape[0] < 1:
                raise ValueError(f"{session_path}: stream {key!r} must have shape [steps, {shape}], got {arr.shape}")
            stream_specs.append((input_arg, arr, dt))
        step_counts = {int(arr.shape[0]) for _, arr, _ in stream_specs}
        if len(step_counts) != 1:
            raise ValueError(f"{session_path}: every input stream must have the same step count")
        session_steps = next(iter(step_counts))
    elif session:
        if session_steps < 1 or not state_pairs:
            raise ValueError(f"{session_path}: a stream-free session needs positive steps and carried state")
    else:
        session_steps = 1

    def _trajectory_reference(field: str) -> tuple[np.ndarray | None, int]:
        spec = session.get(field, {}) if session else {}
        if not spec:
            return None, 0
        if not isinstance(spec, dict) or spec.get("scope") != "trajectory":
            raise ValueError(f"{session_path}: {field}.scope must be trajectory")
        path = model_dir / str(spec.get("golden", ""))
        key = str(spec.get("key", "output0"))
        output_index = int(spec.get("output_index", 0))
        if not path.is_file():
            raise FileNotFoundError(f"{session_path}: {field} trajectory golden is absent: {path}")
        if output_index < 0 or output_index >= len(out_specs):
            raise ValueError(f"{session_path}: {field}.output_index is outside forward results")
        shape, dtype = out_specs[output_index]
        if dtype != "f32":
            raise ValueError(f"{session_path}: {field} trajectory currently requires an f32 output")
        with np.load(path) as data:
            if key not in data.files:
                raise ValueError(f"{session_path}: {field} trajectory key {key!r} is absent")
            values = np.ascontiguousarray(data[key], dtype=np.float32)
        if list(values.shape[1:]) != list(shape) or values.shape[0] < 1:
            raise ValueError(f"{session_path}: {field} trajectory must have shape [steps, {shape}], got {values.shape}")
        return values, output_index

    quality_values, quality_output_index = _trajectory_reference("quality")
    # Compiler correctness and model quality are different claims. New captures carry an eager
    # reference at the compiled precision under `correctness`, and an independently generated FP32
    # reference under `quality`. Legacy diagnostic captures had only `quality`; preserve their
    # execution while paper preflight requires the explicit split.
    correctness_values, correctness_output_index = _trajectory_reference("correctness")
    if correctness_values is None and quality_values is not None:
        correctness_values, correctness_output_index = quality_values, quality_output_index

    # SESSION-STEP BUDGET. Every stream step and every trajectory reference step is emitted as C
    # LITERALS, so the header grows with the corpus, not with the model: resnet50's 256-step,
    # 154 MB `session_inputs.npz` becomes a 770 MB `model_io.h` that costs ~7 GB of RSS to compile.
    # A run that only needs a few steps (a correctness check, a per-step latency measurement) can say
    # so. Truncating streams and references TOGETHER keeps step k comparing against reference k, and
    # `None` (the default) emits the whole corpus, so an unqualified build is byte-identical.
    if max_session_steps is not None:
        if max_session_steps < 1:
            raise ValueError("max_session_steps must be >= 1")
        if max_session_steps < session_steps:
            session_steps = int(max_session_steps)
            stream_specs = [(a, arr[:session_steps], dt) for a, arr, dt in stream_specs]
            if correctness_values is not None:
                correctness_values = correctness_values[:session_steps]
            if quality_values is not None:
                quality_values = quality_values[:session_steps]

    weight_alignment = _align_weight_rows(rows, blob, appended, _append_blob)

    # Project all ABI consumers together after their original source derivation.
    # Embedded array names remain keyed by original capture indices; descriptor
    # indices, state maps and stream pointer updates follow the same projection.
    runtime_indices = tuple(range(n_sig_args))
    descriptor_index = {index: index for index in runtime_indices}
    if entry_projection is not None:
        from .entry_weight_projection import compact_runtime_rows

        combined = WeightPayload()
        combined.extend(blob)
        combined.extend(appended)
        rows, packed = compact_runtime_rows(rows, combined, entry_projection)
        blob, appended = packed, WeightPayload()
        runtime_indices = entry_projection.retained_original_indices
        descriptor_index = {original: index for index, original in enumerate(runtime_indices)}
        state_pairs = [(descriptor_index[input_arg], output_index) for input_arg, output_index in state_pairs]
        n_sig_args = len(runtime_indices)

    readonly_observation = None
    if readonly_input_indices:
        from merlin.runtime.input_observation import write_observation

        combined = WeightPayload()
        combined.extend(blob)
        combined.extend(appended)
        arguments, observed_bytes = [], 0
        for original_index in readonly_input_indices:
            if original_index not in descriptor_index:
                raise ValueError("readonly input observation cannot observe a projected-away argument")
            meta = man[str(original_index)]
            if meta.get("access") not in (None, "read", "readonly") or meta.get("mutable") is True:
                raise ValueError("readonly input selection contradicts declared argument mutability")
            runtime_index = descriptor_index[original_index]
            kind, offset, rank, shape, elem, dtype = rows[runtime_index]
            size = prod(shape) * elem
            observed_bytes += size
            if size <= 0 or observed_bytes > max_readonly_input_bytes:
                raise ValueError("readonly input observation exceeds its complete byte budget")
            if kind == "MERLIN_WEIGHT":
                for other, other_runtime in descriptor_index.items():
                    declared = man.get(str(other), {})
                    if not (declared.get("access") in ("write", "readwrite") or declared.get("mutable") is True):
                        continue
                    other_kind, start, _rank, dims, width, _dtype = rows[other_runtime]
                    if other_kind == "MERLIN_WEIGHT" and start < offset + size and offset < start + prod(dims) * width:
                        raise ValueError("readonly input aliases a declared mutable weight argument")
                data = bytes(combined[offset : offset + size])
            elif kind == "MERLIN_INPUT":
                array, dt = embedded_arrays[original_index]
                array = next((arr[0] for arg, arr, _dt in stream_specs if arg == original_index), array)
                data = _storage_bytes(array, dt)
            else:
                raise ValueError("readonly input selection does not identify an original input or weight")
            if len(data) != size:
                raise ValueError("readonly input bytes disagree with their actual runtime descriptor")
            arguments.append(
                {
                    "original_index": original_index,
                    "runtime_index": runtime_index,
                    "kind": kind,
                    "weight_offset": offset if kind == "MERLIN_WEIGHT" else None,
                    "shape": list(shape),
                    "dtype": dtype,
                    "data": data,
                }
            )
        readonly_observation = write_observation(
            out_dir, arguments=arguments, sources=readonly_sources, max_bytes=max_readonly_input_bytes
        )
        static_io_bytes += observed_bytes + 8 * len(arguments)

    # model_gen.h
    h = [
        "/* Generated by merlin.llvmlower.c_runtime — do not edit. */",
        "#ifndef MERLIN_MODEL_GEN_H",
        "#define MERLIN_MODEL_GEN_H",
        '#include "merlin_model.h"',
        f"#define MERLIN_N_ARGS {len(rows)}",
        f"#define MERLIN_N_OUTPUTS {len(out_specs)}",
        f"#define MERLIN_N_STATE_PAIRS {len(state_pairs)}",
        f"#define MERLIN_SESSION_STEPS {session_steps}",
        f"#define MERLIN_HAS_SESSION_CORRECTNESS {1 if correctness_values is not None else 0}",
        f"#define MERLIN_HAS_SESSION_QUALITY {1 if quality_values is not None else 0}",
        f"#define MERLIN_OUT_ELEMS {int(np.prod(out_shape))}",
        f"#define MERLIN_OUT_LASTDIM {out_shape[-1] if out_shape else 1}",
        f"#define MERLIN_OUT_IS_F32 {int(out_dt == 'f32')}",
        f"#define MERLIN_OUT_IS_I64 {int(out_dt == 'i64')}",
        f"#define MERLIN_OUT_IS_I1 {int(out_dt == 'i1')}",
        "static const merlin_arg_t MERLIN_ARGS[MERLIN_N_ARGS] = {",
    ]
    for kind, off, rank, dims, elem, dt in rows:
        dimstr = ",".join(str(d) for d in dims) or "0"
        h.append(f"  {{{kind}, {off}L, {rank}, {{{dimstr}}}, {elem}}},")
    h += ["};"]
    h.append("#endif")

    io = [
        "/* Generated. Embedded runtime inputs. */",
        "#ifndef MERLIN_MODEL_IO_H",
        "#define MERLIN_MODEL_IO_H",
        "#include <math.h>",
        "#include <string.h>",
    ]
    io += io_decls
    for input_arg in sorted(state_input_args):
        arr, dt = embedded_arrays[input_arg]
        io.append(f"static const {C_OF[dt]} merlin_initial_{input_arg}[] = {{{_embed_array(arr, dt)}}};")
        static_io_bytes += _storage_nbytes(arr, dt)
    for input_arg, arr, dt in stream_specs:
        io.append(f"static const {C_OF[dt]} merlin_stream_{input_arg}[] = {{{_embed_array(arr, dt)}}};")
        static_io_bytes += _storage_nbytes(arr, dt)
    if correctness_values is not None:
        io.append("static const float merlin_correctness_golden[] = {" + _embed_array(correctness_values, "f32") + "};")
        static_io_bytes += int(correctness_values.nbytes)
    if quality_values is not None:
        io.append("static const float merlin_quality_golden[] = {" + _embed_array(quality_values, "f32") + "};")
        static_io_bytes += int(quality_values.nbytes)
    for i, (shape, dt) in enumerate(out_specs):
        nbytes = int(np.prod(shape)) * DT_BYTES[dt]
        io.append(f"static _Alignas(64) unsigned char merlin_out_{i}[{max(1, nbytes)}];")
    # Pointer table, indexed by arg position: the address of the embedded array, or NULL for an arg
    # the runtime reads from the weights blob. Keyed off ``embedded`` -- what the loop ABOVE
    # actually emitted -- and not off a re-derived "is it a param" test, which drifts the moment a
    # second kind of arg lives in the blob (an externalized buffer does) and then names a C array
    # that was never declared.
    io.append(
        "static void *MERLIN_INPUT_PTR[MERLIN_N_ARGS] = {"
        + ",".join(
            [*(f"(void*)merlin_in_{i}" if i in embedded else "0" for i in runtime_indices), *("0" for _ in out_specs)]
        )
        + "};"
    )
    io.append(
        "static void *MERLIN_OUTPUT_PTR[MERLIN_N_OUTPUTS] = {"
        + ",".join(f"(void*)merlin_out_{i}" for i in range(len(out_specs)))
        + "};"
    )
    pair_len = max(1, len(state_pairs))
    io.append(
        f"static const int MERLIN_STATE_INPUT_ARGS[{pair_len}] = {{"
        + (",".join(str(v[0]) for v in state_pairs) if state_pairs else "0")
        + "};"
    )
    io.append(
        f"static const int MERLIN_STATE_OUTPUT_INDICES[{pair_len}] = {{"
        + (",".join(str(v[1]) for v in state_pairs) if state_pairs else "0")
        + "};"
    )
    io.append("static void merlin_reset_session(void) {")
    for input_arg in sorted(state_input_args):
        arr, dt = embedded_arrays[input_arg]
        io.append(f"  memcpy(merlin_in_{input_arg}, merlin_initial_{input_arg}, {_storage_nbytes(arr, dt)}UL);")
    io.append("}")
    io.append("static void merlin_prepare_step(long step) {")
    io.append("  long s = step % MERLIN_SESSION_STEPS;")
    for input_arg, arr, dt in stream_specs:
        step_bytes = _storage_nbytes(arr[0], dt)
        io.append(
            f"  MERLIN_INPUT_PTR[{descriptor_index[input_arg]}] = "
            f"(void *)((const unsigned char *)merlin_stream_{input_arg} + s * {step_bytes}L);"
        )
    io.append("  (void)s;")
    io.append("}")
    io += [
        "typedef struct {",
        "  long steps; double min_cos; double max_rel; long top1;",
        "} merlin_trajectory_metrics_t;",
        "static merlin_trajectory_metrics_t merlin_correctness_metrics = {0, 1.0, 0.0, 0};",
        "static merlin_trajectory_metrics_t merlin_quality_metrics = {0, 1.0, 0.0, 0};",
        "static void merlin_compare_trajectory(const float *got, const float *ref, long n,",
        "                                      merlin_trajectory_metrics_t *metrics) {",
        "  double dot = 0.0, gn = 0.0, rn = 0.0, rmax = 0.0, errmax = 0.0;",
        "  long gi = 0, ri = 0;",
        "  for (long i = 0; i < n; i++) {",
        "    double g = got[i], r = ref[i], e = fabs(g - r);",
        "    dot += g * r; gn += g * g; rn += r * r;",
        "    if (fabs(r) > rmax) rmax = fabs(r); if (e > errmax) errmax = e;",
        "    if (got[i] > got[gi]) gi = i; if (ref[i] > ref[ri]) ri = i;",
        "  }",
        "  double denom = sqrt(gn) * sqrt(rn);",
        "  double cos = denom > 0.0 ? dot / denom : (gn == rn ? 1.0 : 0.0);",
        "  double rel = rmax > 0.0 ? errmax / rmax : errmax;",
        "  if (cos < metrics->min_cos) metrics->min_cos = cos;",
        "  if (rel > metrics->max_rel) metrics->max_rel = rel;",
        "  if (gi == ri) metrics->top1++; metrics->steps++;",
        "}",
        "static void merlin_validate_step(long step) {",
    ]
    if correctness_values is not None:
        correctness_steps = int(correctness_values.shape[0])
        correctness_elems = int(np.prod(correctness_values.shape[1:]))
        io += [
            f"  merlin_compare_trajectory((const float *)MERLIN_OUTPUT_PTR[{correctness_output_index}],",
            f"      merlin_correctness_golden + (step % {correctness_steps}L) * {correctness_elems}L,",
            f"      {correctness_elems}L, &merlin_correctness_metrics);",
        ]
    if quality_values is not None:
        quality_steps = int(quality_values.shape[0])
        quality_elems = int(np.prod(quality_values.shape[1:]))
        io += [
            f"  merlin_compare_trajectory((const float *)MERLIN_OUTPUT_PTR[{quality_output_index}],",
            f"      merlin_quality_golden + (step % {quality_steps}L) * {quality_elems}L,",
            f"      {quality_elems}L, &merlin_quality_metrics);",
        ]
    io += [
        "  (void)step;",
        "}",
        "static long merlin_correctness_steps(void) { return merlin_correctness_metrics.steps; }",
        "static long merlin_correctness_min_cos_ppm(void) {",
        "  return (long)(merlin_correctness_metrics.min_cos * 1000000.0);",
        "}",
        "static long merlin_correctness_max_rel_ppm(void) {",
        "  return (long)(merlin_correctness_metrics.max_rel * 1000000.0);",
        "}",
        "static long merlin_correctness_top1(void) { return merlin_correctness_metrics.top1; }",
        "static long merlin_quality_steps(void) { return merlin_quality_metrics.steps; }",
        "static long merlin_quality_min_cos_ppm(void) {",
        "  return (long)(merlin_quality_metrics.min_cos * 1000000.0);",
        "}",
        "static long merlin_quality_max_rel_ppm(void) {",
        "  return (long)(merlin_quality_metrics.max_rel * 1000000.0);",
        "}",
        "static long merlin_quality_top1(void) { return merlin_quality_metrics.top1; }",
    ]
    io.append("#endif")

    if not ciface_name.isidentifier() or not invoke_name.isidentifier():
        raise ValueError("ciface_name and invoke_name must be C/MLIR identifier-safe")

    # model_call.c — unrolled ciface invocation
    decl = ",".join(["void*"] * len(rows))
    call = ",".join(f"d[{i}]" for i in range(len(rows)))
    call_c = [
        "/* Generated. */",
        f"extern void _mlir_ciface_{ciface_name}({decl});",
        f"void {invoke_name}(void **d) {{ _mlir_ciface_{ciface_name}({call}); }}",
    ]

    weight_digest = hashlib.sha256()
    sources = (blob, appended) if weight_storage == "streamed" else (bytes(blob), bytes(appended))

    def write_weights(stream=None):
        for source in sources:
            chunks = source.chunks() if isinstance(source, WeightPayload) else (memoryview(source),)
            for chunk in chunks:
                weight_digest.update(chunk)
                if stream is not None:
                    stream.write(chunk)

    if materialize_weights:
        weights = out_dir / "weights.bin"
        # The temporary directory also makes link publication atomic when an
        # earlier generated destination already exists.
        with tempfile.TemporaryDirectory(dir=out_dir, prefix="weights-") as temporary:
            pending = Path(temporary) / "weights.bin"
            if weights_source is not None:
                write_weights()
                source = Path(weights_source).resolve(strict=True)
                with source.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if source.stat().st_size != len(blob) + len(appended) or digest != weight_digest.hexdigest():
                    raise ValueError("selected weight blob differs from this stage's derived bytes")
                os.link(source, pending)
            else:
                with pending.open("wb") as stream:
                    write_weights(stream)
            with pending.open("rb") as stream:
                actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if pending.stat().st_size != len(blob) + len(appended) or actual_digest != weight_digest.hexdigest():
                raise ValueError("realized weight blob changed or disagrees with derived bytes")
            pending.replace(weights)
    else:
        write_weights()
    (out_dir / "model_gen.h").write_text("\n".join(h) + "\n")
    (out_dir / "model_io.h").write_text("\n".join(io) + "\n")
    (out_dir / "model_call.c").write_text("\n".join(call_c) + "\n")
    # ``static_io_bytes`` is what the harness spends on the model's I/O in STATIC storage: every
    # embedded input array plus `static <T> OUT[MERLIN_OUT_ELEMS]` in model_main.c. A caller that has
    # to place things in a real board's address map needs it, because it is not a small constant --
    # a 256000-wide logits vector at sequence length 128 is 125 MiB of .bss on its own, and a
    # code-region reserve chosen without it puts the weights blob inside .bss, which surfaces only as
    # a linker "section .weights VMA overlaps section .bss" and reads as anything but a sizing error.
    output_bytes = sum(int(np.prod(shape)) * DT_BYTES[dt] for shape, dt in out_specs)
    info = {
        "n_args": len(rows),
        "n_outputs": len(out_specs),
        "outputs": out_specs,
        "n_state_pairs": len(state_pairs),
        "out_shape": out_shape,
        "out_dt": out_dt,
        "has_session_correctness": correctness_values is not None,
        "has_session_quality": quality_values is not None,
        "ciface_name": ciface_name,
        "invoke_name": invoke_name,
        "n_qinner": len(qinner_args),
        "n_quant_hoist": len(quant_hoist_args),
        "weights_bytes": len(blob) + len(appended),
        "weights_sha256": weight_digest.hexdigest(),
        "weight_alignment_repairs": weight_alignment,
        "static_io_bytes": static_io_bytes + output_bytes,
    }
    if readonly_observation is not None:
        info["readonly_input_observation"] = readonly_observation
    if entry_projection is not None:
        info["entry_projection"] = {
            "original_arguments": len(entry_projection.argument_table.entries),
            "retained_arguments": n_sig_args,
            "removed_original_indices": list(entry_projection.removed_original_indices),
            "original_to_projected": list(entry_projection.original_to_projected),
            "argument_table_sha256": entry_projection.argument_table.identity,
        }
    return info
