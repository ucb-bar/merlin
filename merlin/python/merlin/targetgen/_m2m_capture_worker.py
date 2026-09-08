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
  6. writes ``linalg.mlir``, ``weights.safetensors`` and its argument manifest,
     ``inputs.json``, ``golden.json``, ``meta.json``.

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

_CAPTURE_ABI_VERSION = 3


def _load_loader(loader_py: Path):
    spec = importlib.util.spec_from_file_location("_capsule_loader", loader_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_model_and_inputs"):
        raise RuntimeError(f"{loader_py} must define get_model_and_inputs() -> (model, inputs)")
    return mod


def _quant_for(dtype: str, override: str | None = None):
    """The torchAO config for ``dtype``, or ``override`` when the caller names a scheme explicitly.

    WHY AN OVERRIDE EXISTS. The default int8 scheme is WEIGHT-ONLY, which emits a float matmul over
    dequantized weights -- `linalg.matmul` tagged `aten.mm.default` with an f32 datapath. That is the
    right capture for a model ladder that quantizes weights only, and the WRONG program for a capsule
    meant to exercise an integer systolic datapath: no golden substitution can fix a program that
    contains no integer contraction. `int8_dyn_act_int8_weight` emits `aten._int_mm.default`
    accumulating in i32 -- the arithmetic the mesh actually runs.

    Selectable rather than switched, because the same map serves the whole-model recaptures, and
    whether the model ladder wants W8A8 is a separate decision that must not ride along silently.
    """
    scheme = override or _SCHEME.get(dtype, (None, None))[0]
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


def _mlir_dtype(torch_dtype) -> str:
    """Canonical MLIR element spelling for a captured torch tensor dtype.

    Do not infer this from JSON values: ``_to_native`` deliberately converts tensors through float64,
    so integral-looking values and genuinely integral tensors are indistinguishable after serialization.
    """
    import torch

    spelling = {
        torch.bool: "i1",
        torch.int8: "i8", torch.uint8: "ui8",
        torch.int16: "i16", torch.int32: "i32", torch.int64: "i64",
        torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32",
        torch.float64: "f64",
    }
    for name, mlir in (("float8_e4m3fn", "f8E4M3FN"),
                       ("float8_e5m2", "f8E5M2")):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            spelling[dtype] = mlir
    if torch_dtype not in spelling:
        raise RuntimeError(f"unsupported captured input dtype: {torch_dtype}")
    return spelling[torch_dtype]


def _scalars(mapping) -> dict:
    """The JSON-safe scalar entries of a loader-declared mapping (tensors and streams are dropped).

    A session spec carries the image/trajectory STREAM alongside its provenance; serializing that would
    put hundreds of megabytes of operand data into a meta file. Only scalars travel, so what is recorded
    is the loader's own statement about the data and never the data itself.
    """
    out = {}
    for key, value in (mapping or {}).items():
        if value is None or isinstance(value, (str, bool, int, float)):
            out[str(key)] = value
    return out


def _loader_provenance(mod, mdl, inputs) -> dict:
    """What the LOADER declares about the data this capture ran on -- never inferred here.

    A whole-model loader can be driven down more than one input path (a real, attributed dataset
    stream; a seeded synthetic one), and which path ran is a property of the loader's environment, not
    of the program. If nothing records it, a capsule that only ever proves COMPILER CORRECTNESS -- the
    compiled program reproduces the reference the same loader produced on the same inputs -- can be
    read as a statement about the model's accuracy on real data, which it is not.

    Two declaration shapes are honored because both already exist in the wild: a ``session_provenance``
    attribute on the returned module, and a ``get_session_spec`` function returning ``provenance`` +
    ``paper_ready``. FAIL CLOSED: a loader that declares neither yields status ``undeclared`` and a
    loader whose declaration RAISES yields ``error`` -- both distinct from "declared not synthetic", so
    an unknown can never be recorded as a claim.
    """
    prov: dict = {}
    ready = None
    status = "undeclared"
    error = None
    direct = getattr(mdl, "session_provenance", None)
    if isinstance(direct, dict):
        prov.update(_scalars(direct))
        status = "declared"
    if hasattr(mod, "get_session_spec"):
        try:
            spec = mod.get_session_spec(mdl, inputs)
        except Exception as exc:                       # noqa: BLE001 -- a raising declaration is a fact
            error = f"{type(exc).__name__}: {exc}"
            if status != "declared":
                status = "error"
        else:
            if isinstance(spec, dict):
                declared = spec.get("provenance")
                if isinstance(declared, dict):
                    prov.update(_scalars(declared))
                    status = "declared"
                if isinstance(spec.get("paper_ready"), bool):
                    ready = bool(spec["paper_ready"])
    return {"loader_provenance": prov, "loader_paper_ready": ready,
            "loader_provenance_status": status, "loader_provenance_error": error}


def _input_abi(inputs):
    """Flatten the loader's input pytree and preserve each tensor leaf's real shape and dtype."""
    import torch

    leaves, _spec = torch.utils._pytree.tree_flatten(inputs)
    bad = [type(x).__name__ for x in leaves if not isinstance(x, torch.Tensor)]
    if bad:
        raise RuntimeError(f"model loader inputs must have only tensor leaves; got {bad}")
    return leaves, [{"shape": list(x.shape), "dtype": _mlir_dtype(x.dtype)} for x in leaves]


def _output_abi(outputs):
    """Flatten model results and preserve the ABI of every tensor result.

    Whole-model captures historically recorded only the nested JSON values.  A list-shaped tensor and
    a tuple of tensors are indistinguishable in that representation, which made the parent silently
    retain only result zero.  The pytree flattening performed while torch still owns the values is the
    authoritative result cardinality and dtype record.
    """
    import torch

    leaves, _spec = torch.utils._pytree.tree_flatten(outputs)
    bad = [type(x).__name__ for x in leaves if not isinstance(x, torch.Tensor)]
    if bad:
        raise RuntimeError(f"model outputs must have only tensor leaves; got {bad}")
    return leaves, [{"shape": list(x.shape), "dtype": _mlir_dtype(x.dtype)} for x in leaves]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="m2m capsule capture worker (runs in the m2m venv).")
    ap.add_argument("--loader", required=True, help="path to a .py exposing get_model_and_inputs()")
    ap.add_argument("--dtype", required=True, help="canonical dtype token (fp32/bf16/fp16/int8/fp8)")
    ap.add_argument("--out", required=True, help="output directory for the artifacts")
    ap.add_argument("--m2m-dir", default=os.environ.get("MERLIN_M2M_DIR"),
                    help="model2MLIR repo root (for sys.path if m2m isn't installed)")
    ap.add_argument("--func-name", default="forward")
    ap.add_argument("--scheme", default="",
                    help="torchAO scheme name, overriding the dtype default (e.g. "
                         "int8_dyn_act_int8_weight for a true W8A8 integer contraction)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch RNG seed applied before the frozen loader constructs model/inputs")
    a = ap.parse_args(argv)

    if a.m2m_dir and a.m2m_dir not in sys.path:
        sys.path.insert(0, a.m2m_dir)

    import torch
    import m2m
    from m2m.coverage import opaque_report

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    loader = _load_loader(Path(a.loader))

    torch.manual_seed(int(a.seed))
    modules_before_loader_call = set(sys.modules)
    mdl, inputs = loader.get_model_and_inputs()
    loader_dependency_sources = []
    for module_name in sorted(set(sys.modules) - modules_before_loader_call):
        module = sys.modules.get(module_name)
        source_name = getattr(module, "__file__", None)
        if not source_name:
            continue
        source_path = Path(source_name)
        if source_path.suffix != ".py" or not source_path.is_file() or source_path == Path(a.loader):
            continue
        loader_dependency_sources.append({"module": module_name, "path": str(source_path.resolve())})
    # BEFORE any cast/quantization: what the loader says about the data it just built. Recorded for
    # every capture, so the capsule can never be silent about whether its inputs were real.
    provenance = _loader_provenance(loader, mdl, inputs)
    _scheme, cast = _SCHEME.get(a.dtype, (None, None))
    if cast is not None:
        mdl = mdl.to(getattr(torch, cast))
        inputs = tuple(x.to(getattr(torch, cast)) if isinstance(x, torch.Tensor) and x.is_floating_point()
                       else x for x in inputs)
    mdl = mdl.eval()

    weights_path = str(out / "weights.safetensors")
    q = _quant_for(a.dtype, a.scheme or None)
    quant_stats = None
    if q is not None:
        # Apply quantization explicitly so both conversion and the golden run consume
        # the SAME returned module.  Most TorchAO quantize_ schemes mutate in place,
        # which hid the bug here; PT2E correctly returns a new GraphModule, so running
        # the old ``mdl`` would compare compiled W8A8 against an fp32 reference.
        from m2m.capture.torchao_pipeline import apply_quantization

        calibration_inputs = None
        if q.scheme == "int8_static_act_int8_weight":
            # A loader may keep the benchmark input separate from calibration data.
            # This is important for honest benchmark capture: calibrating on the one
            # measured sample is accidental input specialization.  The hook is
            # model/target agnostic and returns the same positional-input tuples
            # accepted by m2m.convert/apply_quantization.  Older loaders retain the
            # session-stream fallback.
            calibration_hook = getattr(loader, "get_calibration_inputs", None)
            if callable(calibration_hook):
                calibration_inputs = calibration_hook(mdl, tuple(inputs))
            else:
                stream = getattr(mdl, "session_images", None)
                if isinstance(stream, torch.Tensor) and stream.shape[0] > 0:
                    calibration_inputs = ((stream[i],) for i in range(int(stream.shape[0])))
        mdl = apply_quantization(
            mdl,
            q,
            example_inputs=tuple(inputs),
            calibration_inputs=calibration_inputs,
        )
        quant_stats = getattr(mdl, "_m2m_quantization_stats", None)
    res = m2m.convert(mdl, inputs, backend="fx_importer", quantization=q,
                      quantization_preapplied=(q is not None),
                      level="linalg-on-tensors", func_name=a.func_name, weights_path=weights_path)
    opaque = opaque_report(res.mlir_text)
    n_opaque = sum(opaque.values())

    (out / "linalg.mlir").write_text(res.mlir_text, encoding="utf-8")

    # host torch-eager reference — THE golden. Run the (cast/quantized) model the compiler must reproduce.
    with torch.no_grad():
        y = mdl(*inputs)
    output_leaves, output_abi = _output_abi(y)
    output_values = [_to_native(x) for x in output_leaves]
    # Preserve the version-1 single-result JSON shape so existing operator capsules and caches remain
    # byte-compatible.  ``output_abi`` is what disambiguates one list-shaped tensor from many results.
    outputs = output_values[0] if len(output_values) == 1 else output_values
    input_leaves, input_abi = _input_abi(inputs)
    input_prov = [_to_native(x) for x in input_leaves]

    (out / "inputs.json").write_text(json.dumps(input_prov), encoding="utf-8")
    (out / "golden.json").write_text(json.dumps(outputs), encoding="utf-8")
    meta = {
        "ok": bool(res.ok), "opaque": int(n_opaque), "opaque_detail": opaque,
        # WHICH quantization actually produced this program. Without it a weight-only capture and a
        # W8A8 one are indistinguishable after the fact, and they are different arithmetic.
        "scheme": a.scheme or _SCHEME.get(a.dtype, (None, None))[0],
        "quantization_stats": quant_stats,
        "path_taken": getattr(res, "path_taken", None), "dtype": a.dtype,
        "linalg_ops": res.mlir_text.count("linalg."), "func_name": a.func_name,
        "weights": weights_path,
        # model2MLIR externalization emits the authoritative placeholder -> state/input map from the
        # exact exported/quantized graph it lowered.  Preserve that map now; recreating torch.export
        # later from the source loader can have a different argument list (notably tensor-subclass
        # quantization expands parameters into inner tensors).
        "weights_manifest": weights_path + ".manifest.json",
        "capture_abi_version": _CAPTURE_ABI_VERSION,
        "torch_seed": int(a.seed),
        "loader_dependency_sources": loader_dependency_sources,
        # The only authoritative dtype record that survives JSON serialization. The parent verifies this
        # independently against the captured @forward signature before declaring capsule inputs.
        "input_abi": input_abi,
        "output_abi": output_abi,
        # WHAT THE INPUTS WERE, as the loader itself declares them. The parent turns this into the
        # capsule's input-provenance record; without it a synthetic-input capture and a real-data one
        # are indistinguishable afterwards, and only one of them can back an accuracy statement.
        **provenance,
    }
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # a machine-readable tail line the parent greps for, even if warnings precede it
    print("__M2M_CAPTURE__ " + json.dumps({"ok": meta["ok"], "opaque": meta["opaque"]}))
    return 0 if (res.ok and n_opaque == 0) else 3


if __name__ == "__main__":
    raise SystemExit(main())
