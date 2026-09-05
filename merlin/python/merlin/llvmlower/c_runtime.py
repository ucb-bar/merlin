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
- ``model_io.h``    — the embedded runtime inputs (ids etc.) as C arrays.

Nothing here is target-specific: the same artifacts feed the host and the spike builds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model_runner import parse_forward_signature
from .weights_pack import load_safetensors_header

DT_BYTES = {"f32": 4, "f64": 8, "bf16": 2, "f16": 2,
            "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1}
NP_OF = {"f32": np.float32, "f64": np.float64, "i64": np.int64, "i32": np.int32,
         "i8": np.int8, "i1": np.bool_}
C_OF = {"f32": "float", "f64": "double", "i64": "long", "i32": "int", "i8": "signed char",
        "i1": "signed char", "bf16": "unsigned short", "f16": "unsigned short"}


def _out_shape(mlir_path: str | Path) -> tuple[list[int], str]:
    from ..common.mlir_query import forward_signature

    _, results = forward_signature(mlir_path)
    shape, dtype = results[0]
    return shape, dtype


def _bf16_bits(f32: "np.ndarray") -> "np.ndarray":
    """Round float32 to bfloat16 (round-to-nearest-even) and return the raw uint16 bit patterns.

    numpy has no native bf16, so do it by hand: bf16 is the top 16 bits of the f32 encoding; RNE
    adds the round bias 0x7FFF + (lsb of the kept mantissa) before truncating. NaNs are preserved
    (their exponent is all-ones and mantissa nonzero, which survives the shift)."""
    u = f32.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    rounded = (u + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return rounded.astype(np.uint16)


def _embed_array(arr: "np.ndarray", dt: str) -> str:
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
    return ",".join(str(int(v) if "i" in dt else float(v)) for v in flat)


def generate(model_dir: str | Path, out_dir: str | Path,
             inputs_npz: str | Path, extra_npz: str | Path | None = None) -> dict:
    """Emit the runtime-driving artifacts for a captured model into ``out_dir``.

    Non-weight args are embedded as C arrays: real inputs from ``inputs_npz`` (by order),
    and non-persistent buffers / lifted constants (rotary inv_freq, etc.) from
    ``extra_npz`` (matched by manifest name; ``buf::`` keys for buffers, bare keys for
    lifted constants). Weights stay in the blob, referenced by offset.
    """
    model_dir, out_dir = Path(model_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = parse_forward_signature(model_dir / "model.mlir")
    man = json.loads((model_dir / "weights.safetensors.manifest.json").read_text())
    hdr, payload_off = load_safetensors_header(model_dir / "weights.safetensors")
    inputs = np.load(inputs_npz)
    extra_path = extra_npz or (model_dir / "extra.npz")
    extra = np.load(extra_path) if Path(extra_path).is_file() else {}
    lifted_names = sorted(k for k in getattr(extra, "files", []) if not k.startswith("buf::"))

    def buffer_array(name: str):
        # manifest buffer name b_a_b_c  <->  extra key buf::a.b.c
        for k in getattr(extra, "files", []):
            if k.startswith("buf::") and "b_" + k[len("buf::"):].replace(".", "_") == name:
                return np.ascontiguousarray(extra[k])
        raise KeyError(f"buffer {name!r} not in {extra_path}")

    # weights blob (payload, byte-identical to what the manifest offsets index). Anything the
    # bundle does NOT store as a packed weight but the compiled model still has to READ -- the
    # quantized-subclass inner tensors, the zero region a stubbed argument's dead descriptor points
    # at -- is APPENDED here and addressed by its offset past the payload, so it costs no C literals
    # and rides the same mmap/embed path the weights already use.
    blob = (model_dir / "weights.safetensors").read_bytes()[payload_off:]
    appended = bytearray()

    def _append_blob(data: bytes) -> int:
        """Place ``data`` after the weights payload (64-byte aligned) and return its blob offset."""
        pad = (-len(appended)) % 64
        appended.extend(b"\0" * pad)
        begin = len(blob) + len(appended)
        appended.extend(data)
        return begin

    stub_zero_offsets: dict[int, int] = {}   # byte length -> offset of a shared zero region

    out_shape, out_dt = _out_shape(model_dir / "model.mlir")

    # manifest input name -> inputs.npz tuple order. Prefer the per-model
    # input_order.json the capture emits (authoritative — e.g. openvla's
    # input_ids/pixel_values, xr0's 11 inputs); fall back to the legacy hardcoded VLA
    # map only when the bundle predates it. Mirrors dispatch_runtime's resolution.
    order_path = model_dir / "input_order.json"
    if order_path.is_file():
        input_order = {k: int(v) for k, v in json.loads(order_path.read_text()).items()}
    else:
        input_order = {"img": 0, "img_mask": 1, "lang_tokens": 2, "lang_masks": 3,
                       "state": 4, "noise": 5, "ids": 0}

    # arg table: weights -> (offset in blob); inputs/buffers/lifted -> embedded arrays.
    rows = []          # (kind, offset, rank, dims, elem_size, dtype)
    io_decls = []      # embedded C arrays
    static_io_bytes = 0  # bytes of STATIC storage the harness needs for the model's I/O (see below)
    embedded: set[int] = set()   # arg positions that GOT an embedded array (see the ptr table below)
    n_in = 0           # positional input counter (loaders with a single tuple)
    li = 0             # lifted-constant counter
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
                    stub_zero_offsets[nbytes] = _append_blob(bytes(nbytes))
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
            arr = np.ascontiguousarray(extra[lifted_names[li]]); li += 1
        elif name in input_order and f"in{input_order[name]}" in inputs.files:
            arr = np.ascontiguousarray(inputs[f"in{input_order[name]}"])
        else:
            arr = np.ascontiguousarray(inputs[f"in{n_in}"]); n_in += 1
        io_decls.append(f"static {C_OF[dt]} merlin_in_{i}[] = {{{_embed_array(arr, dt)}}};")
        static_io_bytes += int(arr.nbytes)
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
        for arg, arr in zip(qinner_args, _qinner.resolve(extra, qinner_args)):
            begin = _append_blob(np.ascontiguousarray(arr).tobytes())
            rows.append(("MERLIN_WEIGHT", begin, len(arg.shape), list(arg.shape),
                         DT_BYTES[arg.dtype], arg.dtype))
    n_sig_args = len(sig) + len(qinner_args)
    # output row (last)
    rows.append(("MERLIN_OUTPUT", 0, len(out_shape), out_shape,
                 DT_BYTES[out_dt], out_dt))

    # model_gen.h
    h = ["/* Generated by merlin.llvmlower.c_runtime — do not edit. */",
         "#ifndef MERLIN_MODEL_GEN_H", "#define MERLIN_MODEL_GEN_H",
         "#include \"merlin_model.h\"",
         f"#define MERLIN_N_ARGS {len(rows)}",
         f"#define MERLIN_OUT_ELEMS {int(np.prod(out_shape))}",
         f"#define MERLIN_OUT_LASTDIM {out_shape[-1] if out_shape else 1}",
         "static const merlin_arg_t MERLIN_ARGS[MERLIN_N_ARGS] = {"]
    for kind, off, rank, dims, elem, dt in rows:
        dimstr = ",".join(str(d) for d in dims) or "0"
        h.append(f"  {{{kind}, {off}L, {rank}, {{{dimstr}}}, {elem}}},")
    h += ["};"]
    h.append("#endif")

    io = ["/* Generated. Embedded runtime inputs. */",
          "#ifndef MERLIN_MODEL_IO_H", "#define MERLIN_MODEL_IO_H"]
    io += io_decls
    # Pointer table, indexed by arg position: the address of the embedded array, or NULL for an arg
    # the runtime reads from the weights blob. Keyed off ``embedded`` -- what the loop ABOVE
    # actually emitted -- and not off a re-derived "is it a param" test, which drifts the moment a
    # second kind of arg lives in the blob (an externalized buffer does) and then names a C array
    # that was never declared.
    io.append("static void *MERLIN_INPUT_PTR[MERLIN_N_ARGS] = {" + ",".join(
        f"(void*)merlin_in_{i}" if i in embedded else "0"
        for i in range(n_sig_args)) + ",0};")
    io.append("#endif")

    # model_call.c — unrolled ciface invocation
    decl = ",".join(["void*"] * len(rows))
    call = ",".join(f"d[{i}]" for i in range(len(rows)))
    call_c = ["/* Generated. */",
              f"extern void _mlir_ciface_forward({decl});",
              f"void merlin_invoke(void **d) {{ _mlir_ciface_forward({call}); }}"]

    (out_dir / "weights.bin").write_bytes(bytes(blob) + bytes(appended))
    (out_dir / "model_gen.h").write_text("\n".join(h) + "\n")
    (out_dir / "model_io.h").write_text("\n".join(io) + "\n")
    (out_dir / "model_call.c").write_text("\n".join(call_c) + "\n")
    # ``static_io_bytes`` is what the harness spends on the model's I/O in STATIC storage: every
    # embedded input array plus `static <T> OUT[MERLIN_OUT_ELEMS]` in model_main.c. A caller that has
    # to place things in a real board's address map needs it, because it is not a small constant --
    # a 256000-wide logits vector at sequence length 128 is 125 MiB of .bss on its own, and a
    # code-region reserve chosen without it puts the weights blob inside .bss, which surfaces only as
    # a linker "section .weights VMA overlaps section .bss" and reads as anything but a sizing error.
    return {"n_args": len(rows), "out_shape": out_shape, "out_dt": out_dt,
            "n_qinner": len(qinner_args),
            "weights_bytes": len(blob) + len(appended),
            "static_io_bytes": static_io_bytes + int(np.prod(out_shape)) * DT_BYTES[out_dt]}
