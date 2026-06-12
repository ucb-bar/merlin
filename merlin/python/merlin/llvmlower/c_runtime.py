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
    import re

    text = Path(mlir_path).read_text(encoding="utf-8")
    m = re.search(r"func\.func @forward\(.*?\)\s*->\s*\(?\s*tensor<([^>]+)>", text, re.S)
    parts = m.group(1).split("x")
    return [int(p) for p in parts[:-1]], parts[-1]


def _embed_array(arr: "np.ndarray", dt: str) -> str:
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

    # weights blob (payload, byte-identical to what the manifest offsets index)
    blob = (model_dir / "weights.safetensors").read_bytes()[payload_off:]
    (out_dir / "weights.bin").write_bytes(blob)

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
    n_in = 0           # positional input counter (loaders with a single tuple)
    li = 0             # lifted-constant counter
    for i, (shape, dt) in enumerate(sig):
        meta = man[str(i)]
        elem = DT_BYTES[dt]
        if meta["kind"] == "param":
            begin, _ = hdr[meta["weight"]]["data_offsets"]
            rows.append(("MERLIN_WEIGHT", begin, len(shape), shape, elem, dt))
            continue
        # non-weight arg: resolve its data and embed it.
        name = meta.get("name", "")
        if meta["kind"] == "buffer":
            arr = buffer_array(name)
        elif name.startswith("c_lifted_tensor_"):
            arr = np.ascontiguousarray(extra[lifted_names[li]]); li += 1
        elif name in input_order and f"in{input_order[name]}" in inputs.files:
            arr = np.ascontiguousarray(inputs[f"in{input_order[name]}"])
        else:
            arr = np.ascontiguousarray(inputs[f"in{n_in}"]); n_in += 1
        io_decls.append(f"static {C_OF[dt]} merlin_in_{i}[] = {{{_embed_array(arr, dt)}}};")
        rows.append(("MERLIN_INPUT", i, len(shape), shape, elem, dt))
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
    h += ["};",
          "/* pointers to embedded runtime inputs, indexed by arg position (NULL if not input) */"]
    # input pointer table
    ptrs = []
    for i, (shape, dt) in enumerate(sig):
        ptrs.append(f"merlin_in_{i}" if man[str(i)]["kind"] != "param" else "0")
    ptrs.append("0")  # output
    h.append("#endif")

    io = ["/* Generated. Embedded runtime inputs. */",
          "#ifndef MERLIN_MODEL_IO_H", "#define MERLIN_MODEL_IO_H"]
    io += io_decls
    io.append("static void *MERLIN_INPUT_PTR[MERLIN_N_ARGS] = {" + ",".join(
        f"(void*)merlin_in_{i}" if man[str(i)]["kind"] != "param" else "0"
        for i in range(len(sig))) + ",0};")
    io.append("#endif")

    # model_call.c — unrolled ciface invocation
    decl = ",".join(["void*"] * len(rows))
    call = ",".join(f"d[{i}]" for i in range(len(rows)))
    call_c = ["/* Generated. */",
              f"extern void _mlir_ciface_forward({decl});",
              f"void merlin_invoke(void **d) {{ _mlir_ciface_forward({call}); }}"]

    (out_dir / "model_gen.h").write_text("\n".join(h) + "\n")
    (out_dir / "model_io.h").write_text("\n".join(io) + "\n")
    (out_dir / "model_call.c").write_text("\n".join(call_c) + "\n")
    return {"n_args": len(rows), "out_shape": out_shape, "out_dt": out_dt,
            "weights_bytes": len(blob)}
