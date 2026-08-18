#!/usr/bin/env python3
"""model2MLIR capture worker — runs INSIDE the m2m venv (the only interpreter with torch), never inside
the merlin venv. :mod:`merlin.targetgen.capsule_source` invokes it as a subprocess with the m2m venv's
python and ingests the artifacts it writes.

Given a request JSON on argv (a loader ``.py`` exposing ``get_model_and_inputs()``, a canonical dtype
token, and an output dir) it:
  1. builds the model + example inputs from the loader;
  2. casts / torchAO-quantizes per the dtype token (the SAME scheme table ``workloads/capture.py`` uses);
  3. lowers to linalg-on-tensors via ``m2m.convert`` (fx_importer backend), externalizing weights;
  4. asserts 0 opaque ops (a capsule whose program still has opaque ops is not a valid input program);
  5. runs the model EAGER on host CPU to produce the reference (golden) output;
  6. writes ``linalg.mlir``, ``weights.safetensors``, ``inputs.json``, ``golden.json``, ``meta.json``.

This file carries no target-name literal and no merlin import — it is a pure m2m/torch worker.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


# canonical dtype token -> (torchAO scheme | None, base-weight dtype cast | None). Mirrors
# workloads/capture.py SCHEME + DTYPE_CAST so a capsule's precision matches how models are captured.
_SCHEME = {
    "fp32": (None, None), "f32": (None, None),
    "bf16": (None, "bfloat16"), "fp16": (None, "float16"), "f16": (None, "float16"),
    "int8": ("int8_weight_only", None), "i8": ("int8_weight_only", None),
    "fp8": ("float8_weight_only_e4m3", None), "fp8_e4m3": ("float8_weight_only_e4m3", None),
}


def _load_loader(loader_py: Path):
    spec = importlib.util.spec_from_file_location("_capsule_loader", loader_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_model_and_inputs"):
        raise RuntimeError(f"{loader_py} must define get_model_and_inputs() -> (model, inputs)")
    return mod


def _quant_for(dtype: str):
    scheme, _cast = _SCHEME.get(dtype, (None, None))
    if scheme is None:
        return None
    from m2m.capture.torchao_pipeline import QuantizationConfig
    return QuantizationConfig(scheme=scheme)


def _to_native(t):
    """A torch tensor -> nested python lists of floats/ints (json-safe), via float64 for exactness."""
    import torch
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float64).cpu().tolist()
    if isinstance(t, (list, tuple)):
        return [_to_native(x) for x in t]
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="m2m capsule capture worker (runs in the m2m venv).")
    ap.add_argument("--loader", required=True, help="path to a .py exposing get_model_and_inputs()")
    ap.add_argument("--dtype", required=True, help="canonical dtype token (fp32/bf16/fp16/int8/fp8)")
    ap.add_argument("--out", required=True, help="output directory for the artifacts")
    ap.add_argument("--m2m-dir", default=os.environ.get("MERLIN_M2M_DIR"),
                    help="model2MLIR repo root (for sys.path if m2m isn't installed)")
    ap.add_argument("--func-name", default="forward")
    a = ap.parse_args(argv)

    if a.m2m_dir and a.m2m_dir not in sys.path:
        sys.path.insert(0, a.m2m_dir)

    import torch
    import m2m
    from m2m.coverage import opaque_report

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    loader = _load_loader(Path(a.loader))

    mdl, inputs = loader.get_model_and_inputs()
    _scheme, cast = _SCHEME.get(a.dtype, (None, None))
    if cast is not None:
        mdl = mdl.to(getattr(torch, cast))
        inputs = tuple(x.to(getattr(torch, cast)) if isinstance(x, torch.Tensor) and x.is_floating_point()
                       else x for x in inputs)
    mdl = mdl.eval()

    weights_path = str(out / "weights.safetensors")
    q = _quant_for(a.dtype)
    res = m2m.convert(mdl, inputs, backend="fx_importer", quantization=q,
                      level="linalg-on-tensors", func_name=a.func_name, weights_path=weights_path)
    opaque = opaque_report(res.mlir_text)
    n_opaque = sum(opaque.values())

    (out / "linalg.mlir").write_text(res.mlir_text, encoding="utf-8")

    # host torch-eager reference — THE golden. Run the (cast/quantized) model the compiler must reproduce.
    with torch.no_grad():
        y = mdl(*inputs)
    outputs = _to_native(y)
    input_prov = [_to_native(x) for x in inputs]

    (out / "inputs.json").write_text(json.dumps(input_prov), encoding="utf-8")
    (out / "golden.json").write_text(json.dumps(outputs), encoding="utf-8")
    meta = {
        "ok": bool(res.ok), "opaque": int(n_opaque), "opaque_detail": opaque,
        "path_taken": getattr(res, "path_taken", None), "dtype": a.dtype,
        "linalg_ops": res.mlir_text.count("linalg."), "func_name": a.func_name,
        "weights": weights_path,
    }
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # a machine-readable tail line the parent greps for, even if warnings precede it
    print("__M2M_CAPTURE__ " + json.dumps({"ok": meta["ok"], "opaque": meta["opaque"]}))
    return 0 if (res.ok and n_opaque == 0) else 3


if __name__ == "__main__":
    raise SystemExit(main())
