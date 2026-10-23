"""Bind captured forward arguments from explicit model inputs and weight files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dispatch_placement import DispatchRuntimeError


class _ModelInput:
    """A ``@forward`` argument bound from ``inputs.npz`` member ``key`` on every call."""

    __slots__ = ("element", "key", "shape")

    def __init__(self, key: str, shape: Any, element: str) -> None:
        self.key, self.shape, self.element = key, shape, element


def plan_forward_args(
    model_dir: Path,
    input_files: tuple[str, ...],
    *,
    _np_dtype,
    f32_to_bf16,
    f8e4m3fn_to_f32,
) -> list[np.ndarray | _ModelInput]:
    """Every ``@forward`` argument in signature order: the array of a weight, buffer or lifted constant,
    or the ``inputs.npz`` member a model input is bound from (``input_files`` are that file's members)."""
    from ..llvmlower.model_runner import parse_forward_signature
    from ..llvmlower.weights_pack import load_safetensors_header

    sig = parse_forward_signature(model_dir / "model.mlir")
    man = json.loads((model_dir / "weights.safetensors.manifest.json").read_text())
    hdr, payload_off = load_safetensors_header(model_dir / "weights.safetensors")
    blob = (model_dir / "weights.safetensors").read_bytes()[payload_off:]
    extra_path = model_dir / "extra.npz"
    extra = np.load(extra_path) if extra_path.is_file() else None
    extra_files = list(getattr(extra, "files", []))
    lifted_names = sorted(k for k in extra_files if not k.startswith("buf::"))
    # Per-model input map (manifest arg-name -> inputs.npz index), emitted by the consistent
    # capture; falls back to the legacy fixed map + positional order when absent.
    order_path = model_dir / "input_order.json"
    if order_path.is_file():
        input_order = {k: int(v) for k, v in json.loads(order_path.read_text()).items()}
    else:
        input_order = {"img": 0, "img_mask": 1, "lang_tokens": 2, "lang_masks": 3, "state": 4, "noise": 5, "ids": 0}

    def buffer_array(name: str) -> np.ndarray:
        for k in extra_files:
            if k.startswith("buf::") and "b_" + k[len("buf::") :].replace(".", "_") == name:
                return np.ascontiguousarray(extra[k])
        raise DispatchRuntimeError(f"buffer {name!r} not in {extra_path}")

    plan: list[np.ndarray | _ModelInput] = []
    n_in = 0
    li = 0
    for i, (shape, dt) in enumerate(sig):
        meta = man[str(i)]
        if meta["kind"] == "param":
            # Quantized-subclass weights are stubbed in the blob (the fused fp32 weight ARG is
            # dead -- the matmul consumes int8 int_data/scale via the qinner channel). Synthesize
            # a zero buffer of the arg's true shape/dtype for the dead descriptor.
            if meta.get("stub"):
                plan.append(np.zeros(shape, dtype=_np_dtype(dt)))
                continue
            begin, end = hdr[meta["weight"]]["data_offsets"]
            store_dt = hdr[meta["weight"]].get("dtype")  # safetensors storage dtype
            if store_dt == "F8_E4M3":
                # fp8 weight-only: stored 1-byte e4m3, decoded to the f32 arg the graph expects
                arr = f8e4m3fn_to_f32(np.frombuffer(blob[begin:end], np.uint8)).reshape(shape)
            else:
                arr = np.frombuffer(blob[begin:end], dtype=_np_dtype(dt)).reshape(shape)
            plan.append(np.ascontiguousarray(arr))
            continue
        name = meta.get("name", "") or ""
        if meta["kind"] == "buffer":
            # A registered buffer reaches us one of two ways, and only one of them was handled.
            # When the capture EXTERNALIZED it, the manifest names it under "weight" and the
            # bytes are in the safetensors blob like any parameter (BatchNorm running stats, for
            # instance). When it stayed a graph argument, the manifest carries the FX arg "name"
            # and the value is in extra.npz. Prefer the blob when it has the tensor; otherwise
            # fall back to extra.npz -- so neither capture shape is a missing-buffer error.
            wname = meta.get("weight")
            if wname and wname in hdr:
                begin, end = hdr[wname]["data_offsets"]
                arr = np.ascontiguousarray(np.frombuffer(blob[begin:end], dtype=_np_dtype(dt)).reshape(shape))
            else:
                arr = buffer_array(name)
        elif "lifted_tensor" in name:  # lifted get_attr constant (name varies by model)
            arr = np.ascontiguousarray(extra[lifted_names[li]])
            li += 1
        elif name in input_order and f"in{input_order[name]}" in input_files:
            plan.append(_ModelInput(f"in{input_order[name]}", shape, dt))
            continue
        else:
            plan.append(_ModelInput(f"in{n_in}", shape, dt))
            n_in += 1
            continue
        if dt == "bf16":
            plan.append(f32_to_bf16(arr).reshape(shape))
        else:
            plan.append(arr.astype(_np_dtype(dt)).reshape(shape))
    return plan
