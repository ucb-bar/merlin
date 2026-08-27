"""Assemble a model SECTION into a self-contained bundle the K1 board build consumes.

Given a whole-model capture + a selected region set, :func:`build_section_bundle` slices the outlined
module to that section (:func:`..xdsl_dialects.lowering.section_mlir.emit_section_module`) and writes a
tiny "model" directory — ``model.mlir`` (the section ``@forward``) + a section-scoped
``weights.safetensors`` (+ manifest) + ``inputs.npz`` + ``input_order.json`` — that flows through the
EXISTING whole-model K1 build (``mining.k1.build_k1_binary`` → lower → ELF → ``run_on_k1``) unchanged.

Each section-``@forward`` argument is one of:
  * a MODEL WEIGHT (a driver block-arg whose model manifest entry is ``param``) — carried into the
    section's own safetensors (a subset of the model's weights, re-offset);
  * a MODEL INPUT (a driver block-arg tagged ``input``) — copied from the model's ``inputs.npz``;
  * a BOUNDARY ACTIVATION (an upstream kernel result) — the section's true input, fed from the region
    goldens (``region_goldens.npz``) when present, else a seeded tensor of the right shape (timing is
    valid either way; only the golden CHECK needs the real boundary tensor).
So compiling the whole model once, we build + run just the section we care about, on real hardware.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..xdsl_dialects.lowering.outline import outline_dispatches
from ..xdsl_dialects.lowering.section_mlir import emit_section_module

# safetensors dtype spellings for the element types the captures use.
_ST_DTYPE = {"f32": "F32", "f16": "F16", "bf16": "BF16", "i32": "I32", "i8": "I8", "i64": "I64"}
_NP_DTYPE = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "i8": np.int8, "i64": np.int64}


def _mlir_elem(t) -> str:
    """xDSL tensor element type -> the short dtype token used across the bundle."""
    s = str(t.element_type)
    return {"f32": "f32", "f16": "f16", "bf16": "bf16", "i32": "i32", "i8": "i8", "i64": "i64"}.get(
        s, "f32")


def _shape(t) -> list[int]:
    return [int(d) for d in t.get_shape()]


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal valid ``.safetensors`` (8-byte LE header length + JSON header + payload)."""
    header: dict[str, Any] = {}
    payload = bytearray()
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        dt = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("int32"): "I32",
              np.dtype("int8"): "I8", np.dtype("int64"): "I64"}.get(arr.dtype, "F32")
        begin = len(payload)
        payload += arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [begin, len(payload)]}
    hjson = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(hjson)) + hjson + bytes(payload))


def build_section_bundle(model_dir, region_ids, out_dir, *, seed: int = 0) -> dict:
    """Slice the model at ``model_dir`` to ``region_ids`` and write a K1-buildable section bundle to
    ``out_dir``. Returns a summary (section fqns, boundary arg roles, output shape). Deterministic:
    activation boundary tensors are seeded (or read from region_goldens.npz when present)."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from xdsl.ir import BlockArgument

    from ..xdsl_dialects._common import text as _text

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outlined = outline_dispatches(parse_mlir_file(model_dir / "model.mlir"))
    section, boundary, outputs = emit_section_module(outlined.module, set(region_ids))

    man = json.loads((model_dir / "weights.safetensors.manifest.json").read_text())
    model_inputs = np.load(model_dir / "inputs.npz")
    region_goldens = {}
    rg = model_dir / "region_goldens.npz"
    if rg.is_file():
        region_goldens = dict(np.load(rg))
    # weight name -> ndarray, from the model safetensors (via the header reader).
    from merlin.llvmlower.weights_pack import load_safetensors_header
    hdr, payload_off = load_safetensors_header(model_dir / "weights.safetensors")
    blob = (model_dir / "weights.safetensors").read_bytes()[payload_off:]

    def _weight_array(wname: str, shape, elem) -> np.ndarray:
        begin, end = hdr[wname]["data_offsets"]
        return np.frombuffer(blob[begin:end], dtype=_NP_DTYPE.get(elem, np.float32)).reshape(shape)

    rng = np.random.default_rng(seed)
    section_manifest: dict[str, Any] = {}
    section_weights: dict[str, np.ndarray] = {}
    input_arrays: list[np.ndarray] = []
    input_order: dict[str, int] = {}
    roles: list[str] = []
    n_in = 0
    for j, val in enumerate(boundary):
        shape, elem = _shape(val.type), _mlir_elem(val.type)
        if isinstance(val, BlockArgument):
            entry = man.get(str(val.index), {})
            if entry.get("kind") == "param":                       # a model weight
                wname = entry["weight"]
                section_weights[wname] = _weight_array(wname, shape, elem)
                section_manifest[str(j)] = {"kind": "param", "name": None, "weight": wname}
                roles.append(f"weight:{wname}")
                continue
            arr = np.ascontiguousarray(model_inputs[f"in{n_in}"])   # a model input
            roles.append("model_input")
        else:                                                       # boundary activation
            key = next((k for k in region_goldens if k.endswith("::out")
                        and np.prod(region_goldens[k].shape) == int(np.prod(shape))), None)
            arr = (region_goldens[key] if key is not None
                   else rng.standard_normal(shape).astype(_NP_DTYPE.get(elem, np.float32)))
            roles.append("boundary_activation" + ("" if key is None else f":{key}"))
        name = f"barg{j}"
        section_manifest[str(j)] = {"kind": "input", "name": name}
        input_order[name] = n_in
        input_arrays.append(arr)
        n_in += 1

    (out_dir / "model.mlir").write_text(_text(section))
    _write_safetensors(out_dir / "weights.safetensors", section_weights)
    (out_dir / "weights.safetensors.manifest.json").write_text(json.dumps(section_manifest, indent=2))
    np.savez(out_dir / "inputs.npz", **{f"in{i}": a for i, a in enumerate(input_arrays)})
    (out_dir / "input_order.json").write_text(json.dumps(input_order, indent=2))

    return {
        "out_dir": str(out_dir),
        "region_ids": sorted(set(region_ids)),
        "n_boundary_args": len(boundary),
        "boundary_roles": roles,
        "n_weights": len(section_weights),
        "n_inputs": len(input_arrays),
        "output_shape": _shape(outputs[0].type) if outputs else [],
    }
