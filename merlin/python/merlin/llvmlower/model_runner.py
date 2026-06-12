"""Run the whole lowered model on the host (ctypes) — the pre-spike oracle.

Assembles all ``@forward`` arguments in manifest order: 1106 weights as pointers into
the safetensors payload blob, 4 rotary buffers + 9 runtime inputs from the reference
capture, output appended last (buffer-results-to-out-params), then drives the 10-step
denoise loop by varying the timestep input — same loop the bare-metal main runs.
"""
from __future__ import annotations

import ctypes
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .abi import HostModel
from .weights_pack import load_safetensors_header

NP_DTYPES = {"F32": "float32", "BF16": "uint16", "I64": "int64", "I32": "int32",
             "I8": "int8", "BOOL": "bool"}


def parse_forward_signature(mlir_path: str | Path) -> list[tuple[list[int], str]]:
    """[(shape, dtype_str)] for every @forward argument, from the MLIR text."""
    text = Path(mlir_path).read_text(encoding="utf-8")
    m = re.search(r"func\.func @forward\((.*?)\) ->", text, re.S)
    args = re.findall(r"tensor<([^>]+)>", m.group(1))
    out = []
    for a in args:
        parts = a.split("x")
        dims = [int(p) for p in parts[:-1]]
        out.append((dims, parts[-1]))
    return out


@dataclass
class ModelIO:
    """Buffers and the argument list for one forward call."""

    arrays: dict[int, "object"]              # arg_index -> numpy array (kept alive)
    arg_list: list[tuple[int, list[int]]]    # (pointer, shape) per arg + output
    output: "object"                         # 1x50x32 f32


def build_io(mlir_path: str | Path, manifest_path: str | Path,
             safetensors_path: str | Path, reference_npz: str | Path,
             lifted_npz: str | Path) -> ModelIO:
    import numpy as np

    sig = parse_forward_signature(mlir_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    header, payload_off = load_safetensors_header(safetensors_path)
    blob = np.fromfile(safetensors_path, dtype=np.uint8)[payload_off:]
    ref = np.load(reference_npz)
    lifted = np.load(lifted_npz)

    arrays: dict[int, np.ndarray] = {}
    args: list[tuple[int, list[int]]] = []
    lifted_names = sorted(lifted.files)
    li = 0
    for idx in range(len(sig)):
        meta = manifest[str(idx)]
        shape, _ = sig[idx]
        if meta["kind"] in ("param", "buffer"):
            name = meta.get("weight") or meta["name"]
            if name in header:
                begin, _ = header[name]["data_offsets"]
                ptr = blob.ctypes.data + begin
                args.append((ptr, shape))
                continue
            key = "buf::" + name[2:].replace("_", ".")
            cands = [k for k in ref.files
                     if k.startswith("buf::") and
                     k.replace(".", "_").replace("buf::", "b_") == name]
            arr = np.ascontiguousarray(ref[cands[0]] if cands else ref[key])
        elif meta["name"].startswith("c_lifted_tensor_"):
            arr = np.ascontiguousarray(lifted[lifted_names[li]])
            li += 1
        else:
            arr = np.ascontiguousarray(ref[meta["name"]])
        arrays[idx] = arr
        args.append((arr.ctypes.data, shape))

    out = np.zeros((1, 50, 32), dtype=np.float32)
    arrays[-1] = out
    arrays[-2] = blob   # weight pointers index into this — must stay alive
    args.append((out.ctypes.data, [1, 50, 32]))
    return ModelIO(arrays=arrays, arg_list=args, output=out)


def run_denoise_loop(so_path: str | Path, io: ModelIO, manifest_path: str | Path,
                     steps: int = 10):
    """Euler flow-matching loop: x += dt * v(x, t), t = 1.0 -> 0.1. Returns final x."""
    import numpy as np

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    noise_idx = next(int(k) for k, v in manifest.items()
                     if v.get("name") == "noise")
    t_idx = next(int(k) for k, v in manifest.items()
                 if v.get("name") == "c_lifted_tensor_0")

    model = HostModel.load(str(so_path))
    x_t = io.arrays[noise_idx]
    t_buf = io.arrays[t_idx]
    dt = -1.0 / steps
    per_step = []
    for i in range(steps):
        t_buf[...] = 1.0 + i * dt
        model(io.arg_list)
        x_t[...] = x_t + dt * io.output
        per_step.append(x_t.copy())
    return np.stack(per_step)
