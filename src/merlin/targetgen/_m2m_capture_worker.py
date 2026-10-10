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

This file carries no target-name dispatch. Scoped quantization recipes use the
selected worker package's shared Merlin admission screen inside the m2m/torch process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

# canonical dtype token -> (torchAO scheme | None, base-weight dtype cast | None). Mirrors
# workloads/capture.py SCHEME + DTYPE_CAST so a capsule's precision matches how models are captured.
_SCHEME = {
    "fp32": (None, None),
    "f32": (None, None),
    "bf16": (None, "bfloat16"),
    "fp16": (None, "float16"),
    "f16": (None, "float16"),
    "int8": ("int8_weight_only", None),
    "i8": ("int8_weight_only", None),
    "fp8": ("float8_weight_only_e4m3", None),
    "fp8_e4m3": ("float8_weight_only_e4m3", None),
}

_CAPTURE_ABI_VERSION = 6


def _capture_api_report(m2m) -> dict[str, list[str]]:
    """Report selected runtime API gaps; a signature check is not capture qualification."""
    from m2m.capture.bundle import write_bundle

    bundle_args = set(inspect.signature(write_bundle).parameters)
    convert_args = set(inspect.signature(m2m.convert).parameters)
    same_conversion_missing = [
        f"m2m/capture/bundle.py:write_bundle({name})"
        for name in sorted({"source_path", "capture_trace", "conversion_result"} - bundle_args)
    ]
    if importlib.util.find_spec("m2m.capture.provenance") is None:
        same_conversion_missing.append("m2m/capture/provenance.py")
    frontend_trace_missing = [
        f"m2m/api.py:convert({name})" for name in sorted({"capture_trace", "original_frontend_snapshot"} - convert_args)
    ]
    if importlib.util.find_spec("m2m.capture.trace") is None:
        frontend_trace_missing.append("m2m/capture/trace.py")
    static_integerization_missing = [
        member
        for module, member in (
            ("m2m.capture.pt2e_integerize", "m2m/capture/pt2e_integerize.py"),
            ("m2m.capture.pt2e_integer_reference", "m2m/capture/pt2e_integer_reference.py"),
        )
        if importlib.util.find_spec(module) is None
    ]
    return {
        "same_conversion_missing": same_conversion_missing,
        "frontend_trace_missing": frontend_trace_missing,
        "static_integerization_missing": static_integerization_missing,
    }


def _diagnostic_model_copy(out: Path, loader: Path, *, capture_api: dict[str, list[str]]) -> None:
    """Expose the exact converted MLIR for inventory, never as an admitted bundle.

    This deliberately omits the capture receipt and runtime input/golden ABI. An
    older Model2MLIR may have converted the model but cannot provide the modern
    same-conversion materialization contract.
    """

    def digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if (out / "capture_receipt.json").exists() or (out / "model.mlir").exists():
        raise ValueError("diagnostic copy cannot reuse a materialized capture directory")
    shutil.copyfile(out / "linalg.mlir", out / "model.mlir")
    names = (
        "model.mlir",
        "linalg.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.json",
        "golden.json",
        "frontend-trace.json",
        "pytorch-opset.json",
        "meta.json",
    )
    artifacts = {name: {"bytes": (out / name).stat().st_size, "sha256": digest(out / name)} for name in names}
    record = {
        "schema": "merlin.diagnostic_m2m_model.v1",
        "status": "diagnostic_raw_conversion",
        "phase0_admission": "not_granted",
        "source_closure_verified": False,
        "materialized_abi": False,
        "reason": "raw conversion has no same-conversion Model2MLIR bundle or producer capture receipt",
        "capture_api": capture_api,
        "loader": {"path": str(loader.absolute()), "sha256": digest(loader)},
        "artifacts": artifacts,
    }
    (out / "diagnostic-capture.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


def _seed_capture(seed: int, torch) -> dict:
    """Seed loader imports as well as construction; refuse nondeterministic kernels.

    This binds the numerical capture policy, not reproducibility across different
    framework builds or unrecorded external data sources. A loader must still
    declare its data/source ownership separately.
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "rngs": ["python", "numpy", "torch"],
        "deterministic_algorithms": "required",
        "scope": "selected framework build and inputs",
    }


def _load_loader(loader_py: Path):
    spec = importlib.util.spec_from_file_location("_capsule_loader", loader_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_model_and_inputs"):
        raise RuntimeError(f"{loader_py} must define get_model_and_inputs() -> (model, inputs)")
    return mod


def _loader_dependency_sources(modules_before: set[str], loader_py: Path) -> list[dict]:
    """Observe newly imported Python source after loader initialization and input creation.

    This is not a complete import/data closure or proof of the bytes Python executed.
    The hashes let later staging refuse source changes since this observation.
    """
    sources = []
    for name in sorted(set(sys.modules) - modules_before):
        source_name = getattr(sys.modules.get(name), "__file__", None)
        if not source_name:
            continue
        path = Path(source_name)
        if path.suffix != ".py" or not path.is_file() or path.resolve() == loader_py.resolve():
            continue
        sources.append(
            {
                "module": name,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sources


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
    """A torch tensor -> JSON-safe lists without rounding integer or boolean values."""
    import torch

    if isinstance(t, torch.Tensor):
        value = t.detach().cpu()
        return (value if not (value.is_floating_point() or value.is_complex()) else value.to(torch.float64)).tolist()
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
        torch.int8: "i8",
        torch.uint8: "ui8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
    }
    for name, mlir in (("float8_e4m3fn", "f8E4M3FN"), ("float8_e5m2", "f8E5M2")):
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
        except Exception as exc:  # noqa: BLE001 -- a raising declaration is a fact
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
    return {
        "loader_provenance": prov,
        "loader_paper_ready": ready,
        "loader_provenance_status": status,
        "loader_provenance_error": error,
    }


def _input_abi(inputs):
    """Flatten the loader's input pytree and preserve each tensor leaf's real shape and dtype."""
    import torch

    leaves, _spec = torch.utils._pytree.tree_flatten(inputs)
    bad = [type(x).__name__ for x in leaves if not isinstance(x, torch.Tensor)]
    if bad:
        raise RuntimeError(f"model loader inputs must have only tensor leaves; got {bad}")
    return leaves, [{"shape": list(x.shape), "dtype": _mlir_dtype(x.dtype)} for x in leaves]


def _freeze_calibration(samples, *, limit, normalize, torch):
    """Own bounded sample values before a generator can reuse its scratch storage."""
    from itertools import islice

    calibration = []
    for sample in islice(samples, limit):
        sample = normalize(sample)
        leaves, _ = _input_abi(sample)
        if any(leaf.is_complex() for leaf in leaves):
            raise RuntimeError("FP32 staging does not admit complex calibration inputs")
        calibration.append(torch.utils._pytree.tree_map(lambda leaf: leaf.detach().clone(), sample))
    return calibration


def _float_reference(mdl, inputs, torch) -> dict:
    """The untransformed model's outputs on ``inputs``, in the golden's JSON shape.

    Every RNG this worker seeds is saved and restored around the forward, so a loader whose forward
    draws random numbers produces the same golden afterwards as it would have without this run.
    """
    import numpy as np

    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        with torch.no_grad():
            y = mdl(*inputs)
        leaves, abi = _output_abi(y)
        leaves = [leaf.detach().clone() for leaf in leaves]
        values = [_to_native(x) for x in leaves]
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    return {"outputs": values[0] if len(values) == 1 else values, "output_abi": abi, "leaves": leaves}


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


def _integerized_agreement(before, after, *, atol: float, rtol: float) -> dict:
    """Compare two independently supplied executions on the capture input."""
    import torch

    left, left_abi = _output_abi(before)
    right, right_abi = _output_abi(after)
    rows = []
    compatible = left_abi == right_abi and bool(left)
    for original, rewritten in zip(left, right):
        lhs = original.detach().to(torch.float64)
        rhs = rewritten.detach().to(torch.float64)
        same_shape = lhs.shape == rhs.shape
        finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()) if same_shape else False
        if same_shape and finite and lhs.numel():
            delta = (lhs - rhs).abs()
            max_abs = float(delta.max())
            max_rel = float((delta / lhs.abs().clamp_min(1e-12)).max())
            within = bool(torch.all(delta <= atol + rtol * lhs.abs()))
        elif same_shape and finite:
            max_abs = max_rel = 0.0
            within = True
        else:
            max_abs = max_rel = 0.0
            within = False
        rows.append(
            {
                "max_abs": max_abs,
                "max_rel": max_rel,
                "within_tolerance": within,
                "finite": finite,
                "atol": atol,
                "rtol": rtol,
                "shape": list(original.shape),
            }
        )
    passed = compatible and len(rows) == len(left) and all(row["within_tolerance"] for row in rows)
    return {
        "status": "passed" if passed else "failed",
        "samples": 1,
        "atol": atol,
        "rtol": rtol,
        "max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "max_rel": max((row["max_rel"] for row in rows), default=0.0),
        "outputs": rows,
        "finite": compatible and all(row["finite"] for row in rows),
    }


def _fp32_stage_observation(
    original: dict, staged_leaves, original_input_abi: list[dict], staged_input_abi: list[dict]
) -> dict:
    """Measure a selected precision change; this is not an accuracy or quantization gate."""
    import torch

    before = original["leaves"]
    _, after_abi = _output_abi(tuple(staged_leaves))
    before_abi = original["output_abi"]
    if len(before) != len(staged_leaves) or not before:
        raise RuntimeError("FP32 staging changed output cardinality")
    rows = []
    for left, right, old, new in zip(before, staged_leaves, before_abi, after_abi, strict=True):
        if left.shape != right.shape:
            raise RuntimeError("FP32 staging changed output shape")
        if not left.is_floating_point():
            if right.dtype != left.dtype or not torch.equal(left, right):
                raise RuntimeError("FP32 staging changed an exact integer/bool result")
            rows.append(
                {
                    "shape": list(left.shape),
                    "original_dtype": old["dtype"],
                    "staged_dtype": new["dtype"],
                    "comparison": "exact_nonfloating",
                    "max_abs": 0.0,
                    "max_rel": 0.0,
                }
            )
            continue
        if right.dtype != torch.float32:
            raise RuntimeError("FP32 staging retained a non-FP32 floating output")
        lhs, rhs = left.detach().to(torch.float64), right.detach().to(torch.float64)
        if not bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()):
            raise RuntimeError("FP32 staging output comparison is nonfinite")
        delta = (lhs - rhs).abs()
        rows.append(
            {
                "shape": list(left.shape),
                "original_dtype": old["dtype"],
                "staged_dtype": new["dtype"],
                "comparison": "observed_floating",
                "max_abs": float(delta.max()) if delta.numel() else 0.0,
                "max_rel": float((delta / lhs.abs().clamp_min(1e-12)).max()) if delta.numel() else 0.0,
            }
        )
    return {
        "status": "observed",
        "scope": "original computation versus staged FP32; not accuracy equivalence",
        "original_input_abi": original_input_abi,
        "staged_input_abi": staged_input_abi,
        "original_output_abi": before_abi,
        "staged_output_abi": after_abi,
        "output_cardinality": len(rows),
        "output_metrics": rows,
    }


def _exported_integer_mm_count(module) -> int:
    """Count proven i8×i8→i32 contractions in the emitted, unnormalized MLIR."""
    from xdsl.dialects.builtin import IntegerType, TensorType

    def width(value) -> int | None:
        typ = getattr(value, "type", None)
        if not isinstance(typ, TensorType) or not isinstance(typ.element_type, IntegerType):
            return None
        return int(typ.element_type.width.data)

    if module is None:
        return 0
    count = 0
    for op in module.walk():
        if op.name != "linalg.generic" or len(op.operands) < 3 or len(op.results) != 1:
            continue
        prov = op.attributes.get("prov.op")
        if (
            str(getattr(prov, "data", "")) == "int_matmul"
            and [width(value) for value in op.operands[:2]] == [8, 8]
            and width(op.results[0]) == 32
        ):
            count += 1
    return count


def _framework_catalog(torch) -> dict:
    """Observe the registry in this exact capture interpreter, never another build."""
    owner = Path(__file__).with_name("_aten_opset_worker.py")
    try:
        spec = importlib.util.spec_from_file_location("_capture_opset", owner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.core_opset()
    except Exception as exc:  # noqa: BLE001 -- unavailable is not an empty registry
        return {
            "schema": "merlin.pytorch_opset.v1",
            "status": "not_available",
            "torch": str(torch.__version__),
            "n_all_aten": None,
            "reason": f"capture framework catalog unavailable: {type(exc).__name__}: {exc}",
        }


def _materialize_session(
    model, inputs, args, out: Path, *, loader, determinism: dict, dependencies: list, torch
) -> int:
    """Capture an explicit multi-program protocol; do not infer stages by model name."""
    if not args.materialize_bundle:
        raise ValueError("deferred multi-program capture requires --materialize-bundle")
    recipe = json.loads(Path(args.recipe).read_text()) if args.recipe else None
    quantized = args.dtype == "int8" and recipe is not None
    if (args.dtype not in {"fp32", "f32"} and not quantized) or args.scheme or args.already_quantized:
        raise ValueError("multi-program capture requires FP32 or an explicit static int8 recipe")
    if args.quantize_activation_contractions or args.integer_nonlinear:
        raise ValueError("multi-program capture requires a recipe, not additional numerical rewrites")
    if recipe is not None:
        from merlin.targetgen.quant_recipe import digest as recipe_digest

        if (
            not quantized
            or recipe.get("recipe_sha256") != recipe_digest(recipe)
            or (recipe.get("activation") or {}).get("mode") != "static"
        ):
            raise ValueError("multi-program int8 capture requires a valid static recipe")
    from m2m.capture.bundle import write_multi_program_bundle
    from m2m.capture.external_runtime import external_runtime_session
    from m2m.coverage import opaque_report

    session = external_runtime_session(model, tuple(inputs))
    if session.version != 2:
        raise ValueError("deferred capture must supply the explicit version-2 multi-program protocol")
    catalog = _framework_catalog(torch)
    catalog_bytes = (json.dumps(catalog, sort_keys=True, indent=2) + "\n").encode()
    metadata = {
        "capture_abi_version": _CAPTURE_ABI_VERSION,
        "dtype": args.dtype,
        "torch_seed": int(args.seed),
        "determinism": determinism,
        "loader_dependency_sources": dependencies,
        "loader_provenance": _scalars(dict(session.metadata.get("provenance") or {})),
        "loader_provenance_status": "declared",
        "loader_provenance_error": None,
        "loader_paper_ready": session.metadata.get("paper_ready"),
    }
    # Observe every real stage ABI before export; golden.npy stores only result zero.
    stage_abis, stage_references = {}, {}
    with torch.no_grad():
        for program in session.programs:
            _, input_abi = _input_abi(program.inputs)
            program.module.eval()
            reference = _float_reference(program.module, program.inputs, torch) if args.stage_fp32 else None
            if reference is None:
                _, output_abi = _output_abi(program.module(*program.inputs))
            else:
                stage_references[program.name] = reference
                output_abi = reference["output_abi"]
            stage_abis[program.name] = {"input_abi": input_abi, "output_abi": output_abi}
    programs = session.bundle_programs()
    selections = {program.name: "untransformed" for program in session.programs}
    # A recipe-quantized stage reaches the bundle writer already quantized, so a paper-ready stage
    # needs its quality trajectory from the untransformed program first. Every reference is taken
    # before any stage is quantized. FP32 staging freezes its own reference from the staged program.
    stage_sessions = {program.name: program.session for program in session.programs}
    if quantized and not args.stage_fp32:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _capture_session_reference import pre_quantization_session_reference

        stage_sessions = {
            program.name: pre_quantization_session_reference(program.module, program.inputs, program.session)
            for program in session.programs
        }
    stage_quants = {}
    if quantized or args.stage_fp32:
        from m2m.capture.bundle import _shared_tensor_inventory

        if quantized:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import _recipe_quantizer as RQ

        # PT2E stage rewrites must leave shared source weights unchanged.
        source_owner = torch.nn.ModuleList([program.module for program in session.programs])
        has_source_state = any(True for _ in source_owner.parameters()) or any(True for _ in source_owner.buffers())
        source_state = _shared_tensor_inventory(source_owner) if has_source_state else []
        for program, record in zip(session.programs, programs, strict=True):
            selected = False
            exported = None
            if quantized:
                exported = torch.export.export(program.module.eval(), program.inputs)
                selected = RQ._has_floating_recipe_work(exported.graph_module, recipe)
            selections[program.name] = "recipe" if selected else "no_recipe_work" if quantized else "fp32_staged"
            stage_args = argparse.Namespace(**vars(args))
            stage_args.out = str(out / "stages" / program.name)
            if quantized and not selected:
                stage_args.dtype, stage_args.recipe = "fp32", ""
            observed = []
            status = main(
                stage_args,
                prepared_program={
                    "loader": loader,
                    "module": program.module,
                    "inputs": program.inputs,
                    "dependencies": dependencies,
                    "session": stage_sessions[program.name],
                    "provenance": _scalars(dict(session.metadata.get("provenance") or {})),
                    "source_exported": exported if args.stage_fp32 else None,
                    "float_reference": stage_references.get(program.name),
                },
                completed=observed.append,
            )
            if status != 0:
                return status
            capture = observed[0]
            record.update(
                model=capture["module"],
                inputs=capture["inputs"],
                session=capture["session"],
                conversion_result=capture["conversion_result"],
                metadata=capture["metadata"],
            )
            if args.stage_fp32:
                stage_abis[program.name] = {key: capture["metadata"][key] for key in ("input_abi", "output_abi")}
            if capture["quant"] is not None:
                stage_quants[program.name] = capture["quant"]
        if source_state != (_shared_tensor_inventory(source_owner) if has_source_state else []):
            raise ValueError("multi-program source weights changed during precision capture")
        if quantized and not stage_quants:
            raise ValueError("int8 session has no recipe-realized program")
    # The writer owns exported ABI, carried state, goldens and cross-stage bindings.
    summary = write_multi_program_bundle(
        programs,
        dict(session.metadata),
        out,
        capture_trace=True,
        source_path=Path(args.loader),
        metadata=metadata,
        quantization_by_program=stage_quants if quantized else None,
        quantization_preapplied=quantized,
    )
    stages = []
    for program in session.programs:
        stage = out / "stages" / program.name
        catalog_path = stage / "pytorch-opset.json"
        catalog_path.write_bytes(catalog_bytes)
        trace = json.loads((stage / "frontend-trace.json").read_bytes())
        opaque = opaque_report((stage / "model.mlir").read_text())
        precision = trace.get("precision") or {}
        projected = precision.get("status") == "projected" or bool(precision.get("projections"))
        count = sum(opaque.values())
        meta_path = stage / "meta.json"
        meta = json.loads(meta_path.read_bytes())
        if any(meta.get(key) != value for key, value in stage_abis[program.name].items()):
            raise ValueError(f"stage {program.name!r} changed its declared input/output ABI")
        meta.update(stage_abis[program.name])
        meta.update(
            ok=count == 0 and not projected,
            opaque=count,
            opaque_detail=opaque,
            precision_realization=precision,
            framework_catalog={
                "path": catalog_path.name,
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "torch": catalog.get("torch"),
                "status": catalog.get("status", "unknown"),
            },
        )
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        # Metadata/catalog additions must be committed by the producer receipt,
        # not leave stale hashes from the writer's earlier materialization.
        from m2m.capture.provenance import write_capture_receipt

        receipt = write_capture_receipt(stage, source_path=Path(args.loader))
        stages.append(
            {
                "name": program.name,
                "precision_selection": selections[program.name],
                # Present only for transformed stages, so an untransformed session keeps its bytes.
                **(
                    {"numeric_deployment": _numeric_deployment(meta, staged=bool(args.stage_fp32))}
                    if selections[program.name] != "untransformed"
                    else {}
                ),
                "ok": meta["ok"],
                "opaque": count,
                "trace_status": trace.get("status", "unknown"),
                "receipt_sha256": hashlib.sha256((stage / "capture_receipt.json").read_bytes()).hexdigest(),
                "materialized_abi": receipt["materialized_abi"],
            }
        )
    report = {
        "schema": "merlin.model_session_capture.v1",
        "programs": stages,
        "determinism": determinism,
        "agentic": False,
        "recipe_sha256": recipe.get("recipe_sha256") if recipe else None,
        "stage_storage": (
            "separate; original source weight bytes verified unchanged"
            if quantized or args.stage_fp32
            else "untransformed"
        ),
        **({"source_state_unchanged": True} if args.stage_fp32 else {}),
        "session_contract_sha256": hashlib.sha256((out / "session_contract.yaml").read_bytes()).hexdigest(),
        "qualification": "capture only; no target lowering, execution or application accuracy claim",
    }
    (out / "session-receipt.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    ok = all(stage["ok"] for stage in stages)
    print(
        "__M2M_CAPTURE__ "
        + json.dumps(
            {
                "ok": ok,
                "opaque": sum(stage["opaque"] for stage in stages),
                "session": summary["session_kind"],
                "programs": len(stages),
            }
        )
    )
    return 0 if ok else 3


def _numeric_deployment(meta: dict, *, staged: bool) -> dict:
    """What a transformed session stage actually deploys, stated in its receipt, not implied.

    ``form`` names the precision path (FP32 staging, then the recipe's W8A8, or either alone). The
    contraction census separates what the recipe selected and integerized, what it selected but had to
    keep as dequantized float (with the source dtype that forced it), and what it never selected. The
    activation x activation matmuls in that last group are an explicit, counted scope note: a static
    weight recipe has no weight to quantize there, so they stay float by design.
    """
    receipt = meta.get("integerization_receipt") or {}
    staging = meta.get("fp32_staging")
    conversion = meta.get("precision_conversion") or {}
    audit = conversion.get("staged_precision_audit") or {}
    quantized = bool(receipt)
    form = (
        ("fp32_staged_then_recipe_w8a8" if staged else "recipe_w8a8")
        if quantized
        else ("fp32_staged" if staged else "untransformed")
    )
    record: dict = {"form": form}
    if staged:
        record["precision_staging"] = {
            "target_dtype": audit.get("target_dtype"),
            "audit_status": audit.get("status"),
            "checked_floating_values": audit.get("checked_floating_values"),
            "non_target_floating_values": audit.get("non_target_floating_values"),
            "graph_dtype_retargeting": conversion.get("graph_dtype_retargeting"),
            "retyped_dtype_decisions": len(conversion.get("dtype_decisions") or ()),
            "original_graph_sha256": conversion.get("original_graph_sha256"),
            "staged_graph_sha256": conversion.get("staged_graph_sha256"),
            "source_state_unchanged": bool(staging),
        }
    if quantized:
        by_kind = receipt.get("quantized_by_kind") or {}
        decisions = receipt.get("precision_decisions") or []
        census = {}
        for kind in ("linear", "conv2d", "matmul"):
            seen = int(receipt.get(f"{kind}_seen") or 0)
            selected = int((by_kind.get(kind) or {}).get("seen") or 0)
            integerized = int((by_kind.get(kind) or {}).get("integerized") or 0)
            census[kind] = {
                "seen": seen,
                "recipe_selected": selected,
                "integerized": integerized,
                "selected_kept_float": selected - integerized,
                "not_selected": seen - selected,
            }
        reasons: dict[str, int] = {}
        for row in decisions:
            if row.get("decision") != "integerized_i32":
                key = f"{row.get('decision')}:{row.get('kind')}:{row.get('source_dtype')}"
                reasons[key] = reasons.get(key, 0) + 1
        record.update(
            contractions=census,
            selected_kept_float_reasons=dict(sorted(reasons.items())),
            integer_contractions=int(receipt.get("exported_integer_mm_count") or 0),
        )
        unselected_matmuls = census["matmul"]["not_selected"]
        if unselected_matmuls:
            record["scope_notes"] = [
                {
                    "kind": "matmul",
                    "count": unselected_matmuls,
                    "reason": "activation x activation matmul (no weight operand); a static weight recipe "
                    "does not quantize it, so it stays float",
                }
            ]
    return record


def _write_leaf_constants(mdl, inputs, weights_path: str, dest: Path) -> str | None:
    """Write manifest-listed ``@forward`` buffers and lifted constants omitted from weights.

    Use m2m's bundle layout: ``buf::<dotted name>`` for buffers and the
    manifest name for constants. Missing leaves fail closed; another capture's
    constants cannot complete this capsule.
    """
    import numpy as np

    manifest_path = Path(weights_path + ".manifest.json")
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [meta or {} for meta in manifest.values()]
    # A buffer the externalizer STORED carries its `weight` key and is already in the weights file.
    buffers = {str(meta.get("name") or "") for meta in rows if meta.get("kind") == "buffer" and not meta.get("weight")}
    lifted = [str(meta.get("name") or "") for meta in rows if "lifted_tensor" in str(meta.get("name") or "")]
    if not buffers and not lifted:
        return None
    extra: dict = {}
    for name, tensor in mdl.named_buffers():
        if "b_" + name.replace(".", "_") in buffers:
            extra["buf::" + name] = tensor.detach().float().cpu().numpy()
    if lifted:
        from m2m.capture.bundle import _lifted_constants

        found: dict = {}
        _lifted_constants(mdl, tuple(inputs), found)
        extra.update({key: value for key, value in found.items() if key in lifted})
    recovered = {"b_" + key[len("buf::") :].replace(".", "_") for key in extra if key.startswith("buf::")}
    missing = sorted(buffers - recovered) + [name for name in lifted if name not in extra]
    if missing:
        # FAIL CLOSED: a partial file would be read as complete by a consumer that trusts it.
        raise RuntimeError(f"leaf arguments named by the manifest could not be recovered: {missing}")
    np.savez(dest, **extra)
    return str(dest)


def main(argv=None, *, prepared_program=None, completed=None) -> int:
    ap = argparse.ArgumentParser(description="m2m capsule capture worker (runs in the m2m venv).")
    ap.add_argument("--loader", required=True, help="path to a .py exposing get_model_and_inputs()")
    ap.add_argument("--dtype", required=True, help="canonical dtype token (fp32/bf16/fp16/int8/fp8)")
    ap.add_argument("--out", required=True, help="output directory for the artifacts")
    ap.add_argument(
        "--m2m-dir",
        default=os.environ.get("MERLIN_M2M_DIR"),
        help="model2MLIR repo root (for sys.path if m2m isn't installed)",
    )
    ap.add_argument("--func-name", default="forward")
    ap.add_argument(
        "--scheme",
        default="",
        help="torchAO scheme name, overriding the dtype default (e.g. "
        "int8_dyn_act_int8_weight for a true W8A8 integer contraction)",
    )
    ap.add_argument(
        "--recipe",
        default="",
        help="path to a quant_recipe_v1 JSON derived from the target; when given, the "
        "model is quantized under it and --scheme / the dtype default are not used",
    )
    ap.add_argument(
        "--already-quantized",
        action="store_true",
        help="capture the loader's own materialized numeric graph without applying a TorchAO recipe or scheme",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="Python/NumPy/Torch RNG seed applied before loader import and construction"
    )
    ap.add_argument("--agreement-atol", type=float, default=1e-3)
    ap.add_argument("--agreement-rtol", type=float, default=1e-3)
    ap.add_argument(
        "--stage-fp32",
        action="store_true",
        help="explicitly stage the captured floating frontend to FP32 with audited dtype operands",
    )
    ap.add_argument(
        "--materialize-bundle",
        action="store_true",
        help="also emit the full model2MLIR runtime bundle from this exact conversion and model instance",
    )
    ap.add_argument(
        "--quantize-activation-contractions",
        action="store_true",
        help="also quantize the contractions the scheme cannot reach (activation x activation "
        "matmuls, non-overlapping convolutions) in the scheme's own int8 dynamic form; see "
        "_activation_contractions.py",
    )
    ap.add_argument(
        "--integer-nonlinear",
        action="store_true",
        help="with --quantize-activation-contractions: also compute softmax, GELU and layer norm in "
        "integer arithmetic (I-BERT); a different numerical model, see _integer_nonlinear.py",
    )
    ap.add_argument(
        "--diagnostic-model-copy",
        action="store_true",
        help="copy raw converted MLIR for inventory with byte hashes, without a capture receipt or Phase 0 admission",
    )
    a = argv if isinstance(argv, argparse.Namespace) else ap.parse_args(argv)
    if a.materialize_bundle and a.diagnostic_model_copy:
        ap.error("--materialize-bundle and --diagnostic-model-copy are mutually exclusive")
    if not 0 <= a.seed < 2**32:
        ap.error("--seed must be an unsigned 32-bit integer")
    if not all(math.isfinite(value) and value >= 0 for value in (a.agreement_atol, a.agreement_rtol)):
        ap.error("agreement tolerances must be finite and nonnegative")
    if a.already_quantized and (a.recipe or a.scheme):
        ap.error("--already-quantized cannot be combined with --recipe or --scheme")
    if a.integer_nonlinear and not a.quantize_activation_contractions:
        # Defined only on top of the int8 contractions (the softmax's int8 numerators are the next
        # contraction's own operand), so it is refused without them rather than mixed into a float program.
        ap.error("--integer-nonlinear is defined on top of --quantize-activation-contractions only")
    if a.stage_fp32 and (
        (a.dtype not in {"fp32", "f32"} and not (a.dtype == "int8" and a.recipe))
        or a.scheme
        or a.already_quantized
        or a.quantize_activation_contractions
        or a.integer_nonlinear
    ):
        ap.error("--stage-fp32 requires FP32 or an explicit int8 recipe without additional rewrites")

    if a.m2m_dir and a.m2m_dir not in sys.path:
        sys.path.insert(0, a.m2m_dir)
    # Shared software admission belongs to the selected worker's own Merlin
    # package, in both source and wheel layouts, never another editable checkout.
    if a.recipe:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import m2m
    import torch
    from m2m.coverage import opaque_report

    capture_api = _capture_api_report(m2m)

    if a.stage_fp32:
        if capture_api["frontend_trace_missing"]:
            raise RuntimeError(
                "FP32 staging requires frontend trace APIs: " + ", ".join(capture_api["frontend_trace_missing"])
            )
        from m2m.capture.trace import materialize_frontend_precision

        if "retarget_float_dtype_arguments" not in inspect.signature(materialize_frontend_precision).parameters:
            raise RuntimeError("FP32 staging requires audited frontend dtype-argument retargeting")

    if a.materialize_bundle and capture_api["same_conversion_missing"]:
        raise RuntimeError(
            "selected Model2MLIR lacks same-conversion bundle/receipt APIs: "
            f"{capture_api['same_conversion_missing']}; "
            "use --diagnostic-model-copy only for unadmitted raw-model inventory"
        )
    if a.integer_nonlinear:
        # The integer nonlinears are captured as integer shifts and floor divisions. A bridge that
        # does not decompose them leaves each one an opaque call, so refuse here, naming them, rather
        # than capture a program that cannot link.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _integer_nonlinear as NL
        from m2m.ir.decompositions import DECOMPOSITION_TABLE

        missing = [name for name in NL.REQUIRED_DECOMPOSITIONS if name not in DECOMPOSITION_TABLE]
        if missing:
            raise SystemExit(
                f"--integer-nonlinear needs a Model2MLIR that decomposes {missing}; the selected one does not"
            )

    # Model2MLIR embeds this path in prov.weights_file. A relative --out would
    # otherwise leave a CWD-relative reference in the saved MLIR, which a
    # relocated/frozen corpus cannot safely resolve against its selected bytes.
    out = Path(a.out).absolute()
    if a.diagnostic_model_copy and out.exists() and any(out.iterdir()):
        raise ValueError("diagnostic model copy requires a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    determinism = _seed_capture(a.seed, torch)
    modules_before_loader = set(sys.modules)
    if prepared_program is None:
        loader = _load_loader(Path(a.loader))
        mdl, inputs = loader.get_model_and_inputs()
        loader_dependency_sources = _loader_dependency_sources(modules_before_loader, Path(a.loader))
    else:
        loader = prepared_program["loader"]
        mdl, inputs = prepared_program["module"], prepared_program["inputs"]
        loader_dependency_sources = prepared_program["dependencies"]
    if not isinstance(mdl, torch.nn.Module) and callable(getattr(mdl, "external_runtime_session", None)):
        return _materialize_session(
            mdl,
            inputs,
            a,
            out,
            loader=loader,
            determinism=determinism,
            dependencies=loader_dependency_sources,
            torch=torch,
        )
    # BEFORE any cast/quantization: what the loader says about the data it just built. Recorded for
    # every capture, so the capsule can never be silent about whether its inputs were real.
    provenance = (
        _loader_provenance(loader, mdl, inputs)
        if prepared_program is None
        else {
            "loader_provenance": prepared_program["provenance"],
            "loader_provenance_status": "declared",
            "loader_provenance_error": None,
            "loader_paper_ready": None,
        }
    )
    mdl.eval()
    source_layer_inventory = None
    if a.stage_fp32 and a.recipe:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _recipe_quantizer as RQ

        source_layer_inventory = RQ.layer_inventory(mdl)
    source_reference = prepared_program.get("float_reference") if prepared_program else None
    if a.stage_fp32 and source_reference is None:
        source_reference = _float_reference(mdl, inputs, torch)
    session = (
        prepared_program["session"]
        if prepared_program is not None
        else (
            loader.get_session_spec(mdl, tuple(inputs))
            if a.materialize_bundle and hasattr(loader, "get_session_spec")
            else None
        )
    )
    # Capture loader-owned calibration at original precision before export.
    calibration = None
    if a.stage_fp32 and a.recipe:
        hook = getattr(loader, "get_calibration_inputs", None)
        samples = None
        if callable(hook):
            samples = hook(mdl, tuple(inputs))
        else:
            stream = getattr(mdl, "session_images", None)
            if isinstance(stream, torch.Tensor) and stream.shape[0] > 0:
                samples = ((stream[i],) for i in range(int(stream.shape[0])))
        if samples is not None:
            calibration_limit = max(
                inspect.signature(RQ.apply_recipe).parameters["calibration_samples"].default,
                inspect.signature(RQ.agreement).parameters["limit"].default,
            )
            calibration = _freeze_calibration(samples, limit=calibration_limit, normalize=RQ._as_tuple, torch=torch)
    original_snapshot = {
        "status": "unavailable",
        "stage": "original",
        "reason": "selected model2MLIR does not expose frontend capture tracing",
    }
    trace_supported = "capture_trace" in inspect.signature(m2m.convert).parameters
    if trace_supported and not a.already_quantized:
        try:
            from m2m.capture.trace import capture_frontend_snapshot

            source_exported = prepared_program.get("source_exported") if a.stage_fp32 and prepared_program else None
            if a.stage_fp32 and source_exported is None:
                source_exported = torch.export.export(mdl, tuple(inputs))
            original_snapshot = capture_frontend_snapshot(
                source_exported if source_exported is not None else mdl, tuple(inputs), stage="original"
            )
        except Exception as exc:  # noqa: BLE001 -- unknown source counts are not fabricated
            original_snapshot = {
                "status": "unavailable",
                "stage": "original",
                "reason": f"original frontend capture failed: {type(exc).__name__}: {exc}",
            }
    elif a.already_quantized:
        original_snapshot["reason"] = "loader supplies an already-quantized graph; the original model was not captured"
    _scheme, cast = _SCHEME.get(a.dtype, (None, None))
    # THE FLOAT MODEL'S OWN OUTPUT, before any precision cast or quantization rewrites it. The golden
    # below is the transformed model -- what the compiler must reproduce -- and so can say nothing about
    # how far the transformation moved the network; this is the reference that can.
    transforms = (
        a.stage_fp32
        or cast is not None
        or bool(a.recipe)
        or bool(a.scheme)
        or a.quantize_activation_contractions
        or a.integer_nonlinear
        or (not a.already_quantized and _quant_for(a.dtype, None) is not None)
    )
    float_reference = source_reference if a.stage_fp32 else None
    if not a.stage_fp32 and transforms and not a.already_quantized:
        float_reference = _float_reference(mdl, inputs, torch)
    original_input_abi = _input_abi(inputs)[1] if a.stage_fp32 else None
    precision_conversion = None
    if a.stage_fp32:
        if original_snapshot.get("status") != "complete":
            raise RuntimeError("FP32 staging requires a complete original frontend snapshot")
        mdl, inputs, original_snapshot, precision_conversion = materialize_frontend_precision(
            source_exported,
            tuple(inputs),
            dtype=torch.float32,
            original_frontend_snapshot=original_snapshot,
            retarget_float_dtype_arguments=True,
        )
        audit = precision_conversion.get("staged_precision_audit") if isinstance(precision_conversion, dict) else None
        if (
            precision_conversion.get("original_graph_sha256") != original_snapshot["sha256"]
            or not isinstance(precision_conversion.get("staged_graph_sha256"), str)
            or len(precision_conversion["staged_graph_sha256"]) != 64
            or precision_conversion.get("graph_dtype_retargeting") != "schema_float_dtype_operands"
            or not isinstance(precision_conversion.get("dtype_decisions"), list)
            or not isinstance(audit, dict)
            or audit.get("status") != "complete"
            or audit.get("target_dtype") != "torch.float32"
            or audit.get("non_target_floating_values") != 0
            or type(audit.get("checked_floating_values")) is not int
            or audit["checked_floating_values"] < 1
        ):
            raise RuntimeError("FP32 staging lacks a complete exact-source precision audit")
    if cast is not None:
        try:
            from m2m.capture.trace import materialize_frontend_precision

            mdl, inputs, original_snapshot, precision_conversion = materialize_frontend_precision(
                mdl, tuple(inputs), dtype=getattr(torch, cast), original_frontend_snapshot=original_snapshot
            )
        except Exception as exc:  # noqa: BLE001 -- compatibility output retains unknown lineage
            precision_conversion = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            original_snapshot = {
                "stage": "original",
                "status": "unavailable",
                "reason": "precision conversion lacks exact original source ownership",
            }
            mdl = mdl.to(getattr(torch, cast))
            inputs = tuple(
                x.to(getattr(torch, cast)) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                for x in inputs
            )
    # PyTorch's public exported-model train/eval helper may return None.
    # The owned GraphModule remains the conversion source.
    mdl.eval()
    fp32_staging = None
    if a.stage_fp32:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _capture_session_reference import freeze_fp32_session_reference

        session = freeze_fp32_session_reference(mdl, inputs, session)
        staged_reference = _float_reference(mdl, inputs, torch)
        fp32_staging = _fp32_stage_observation(
            float_reference, staged_reference["leaves"], original_input_abi, _input_abi(inputs)[1]
        )
        fp32_staging.update(
            original_graph_sha256=original_snapshot["sha256"],
            staged_graph_sha256=precision_conversion["staged_graph_sha256"],
        )
        if calibration is not None:
            before_calibration = [_input_abi(sample)[1] for sample in calibration]
            calibration = [
                torch.utils._pytree.tree_map(
                    lambda leaf: leaf.to(torch.float32) if leaf.is_floating_point() else leaf, sample
                )
                for sample in calibration
            ]
            fp32_staging["calibration_source"] = "loader_declared_pre_transform"
            fp32_staging["calibration_sample_limit"] = calibration_limit
            fp32_staging["calibration_input_abis"] = [
                {"original": before, "staged": _input_abi(sample)[1]}
                for before, sample in zip(before_calibration, calibration, strict=True)
            ]
        if source_layer_inventory is not None:
            inventory_bytes = json.dumps(source_layer_inventory, sort_keys=True, separators=(",", ":")).encode()
            fp32_staging.update(
                source_layer_inventory=source_layer_inventory,
                source_layer_inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
            )

    weights_path = str(out / "weights.safetensors")
    recipe = json.loads(Path(a.recipe).read_text(encoding="utf-8")) if a.recipe else None
    if recipe is not None:
        from merlin.targetgen.quant_recipe import digest as recipe_digest

        if recipe.get("recipe_sha256") != recipe_digest(recipe):
            raise ValueError("selected quantization recipe digest does not match its content")
    q = None if (recipe is not None or a.already_quantized) else _quant_for(a.dtype, a.scheme or None)
    quant_stats = (
        {"applied": False, "why": "the loader declares its numeric graph already materialized"}
        if a.already_quantized
        else None
    )
    agreement = None
    if recipe is not None:
        # THE TARGET'S OWN QUANTIZATION. The recipe was derived from what the hardware's readout
        # holds; the quantizer that realises it is generic and lives beside this worker.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _recipe_quantizer as RQ
        from m2m.capture.torchao_pipeline import QuantizationConfig

        if not a.stage_fp32:
            hook = getattr(loader, "get_calibration_inputs", None)
            if callable(hook):
                calibration = list(hook(mdl, tuple(inputs)))
            else:
                stream = getattr(mdl, "session_images", None)
                if isinstance(stream, torch.Tensor) and stream.shape[0] > 0:
                    calibration = [(stream[i],) for i in range(int(stream.shape[0]))]
        reference = mdl
        static = recipe["activation"]["mode"] == "static"
        if not static:
            import copy

            reference = copy.deepcopy(mdl)  # quantize_ mutates in place
        mdl = RQ.apply_recipe(
            mdl,
            recipe,
            example_inputs=tuple(inputs),
            calibration_inputs=calibration,
            original_frontend_snapshot=original_snapshot if trace_supported else None,
            source_layer_inventory=source_layer_inventory,
        )
        quant_stats = getattr(mdl, "_recipe_quantization_stats", None)
        if fp32_staging is not None:
            fp32_staging["source_layer_plan_sha256"] = (
                quant_stats.get("plan_sha256") if isinstance(quant_stats, dict) else None
            )
        # The receipt: the recipe-quantized model against the floating-point one, on the
        # calibration stream and the capture input. Recorded, not judged here.
        agreement = RQ.agreement(reference, mdl, [tuple(inputs), *(calibration or [])])
        # The capture library is told which lowering family this graph is, and that the
        # quantization is already applied; the numbers are the recipe's.
        formats = {str(recipe[part]["dtype"]) for part in ("activation", "weight")}
        if formats == {"int8"}:
            provenance_scheme = "int8_static_act_int8_weight" if static else "int8_dyn_act_int8_weight"
        elif len(formats) == 1 and next(iter(formats)).startswith("fp8_"):
            fmt = next(iter(formats))
            provenance_scheme = f"{fmt}_{'static' if static else 'dynamic'}_act_weight"
        else:
            raise ValueError(f"no capture provenance scheme for mixed recipe formats {sorted(formats)}")
        q = QuantizationConfig(scheme=provenance_scheme)
    elif q is not None:
        # Apply quantization explicitly so both conversion and the golden run consume
        # the SAME returned module.  Most TorchAO quantize_ schemes mutate in place,
        # which hid the bug here; PT2E correctly returns a new GraphModule, so running
        # the old ``mdl`` would compare compiled W8A8 against an fp32 reference.
        from m2m.capture.torchao_pipeline import apply_quantization

        if q.scheme == "int8_static_act_int8_weight":
            # This TorchAO config needs a calibrated activation scale, which the
            # public model2MLIR apply_quantization(model, config) API does not
            # derive.  The recipe path above performs calibrated PT2E instead.
            raise RuntimeError("named static W8A8 requires a calibrated --recipe")
        quant_parameters = inspect.signature(apply_quantization).parameters
        quant_trace_options = {}
        if "original_frontend_snapshot" in quant_parameters:
            quant_trace_options["original_frontend_snapshot"] = original_snapshot
        if "example_inputs" in quant_parameters:
            # Whole-graph transforms need the same concrete inputs as conversion
            # to discover functional contractions and their tensor shapes.
            quant_trace_options["example_inputs"] = tuple(inputs)
        mdl = apply_quantization(mdl, q, **quant_trace_options)
        quant_stats = getattr(mdl, "_m2m_quantization_stats", None)
    integerization_receipt = None
    if q is not None and q.scheme == "int8_static_act_int8_weight" and not a.already_quantized:
        # Account for every selected PT2E contraction: genuine integer work or
        # an explicitly preserved non-FP32 floating Q/DQ operation. Preserving
        # precision is not accelerator/host admission. The independent reference
        # compares the complete output and counts only genuine integer work.
        from m2m.capture.pt2e_integerize import integerize_pt2e

        selected_engine = recipe.get("software_numerical_engine") if recipe is not None else None
        if selected_engine not in (None, "integer_reference"):
            raise ValueError(f"static int8 capture cannot realize numerical engine {selected_engine!r}")
        with torch.no_grad():
            portable_output = mdl(*inputs)
        independent = None
        independent_error = None
        independent_source = None
        if selected_engine == "integer_reference":
            try:
                from m2m.capture import pt2e_integer_reference as integer_reference

                source = Path(integer_reference.__file__).resolve()
                if not source.is_file():
                    raise RuntimeError("selected integer reference source is unavailable")
                independent_source = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                independent = integer_reference.run_pt2e_integer_reference(mdl, tuple(inputs))
                if not all(
                    hasattr(independent, field)
                    for field in ("contraction_count", "conv2d_count", "linear_count", "matmul_count")
                ):
                    raise RuntimeError("selected integer reference lacks complete contraction accounting")
            except Exception as exc:  # noqa: BLE001 -- record a failed selected reference, never substitute portable
                independent_error = f"{type(exc).__name__}: {exc}"[:2000]
                independent = None
        mdl, integerization_receipt = integerize_pt2e(mdl, tuple(inputs))
        from merlin.capture.integerization import contraction_partition

        try:
            integer_partition = contraction_partition(integerization_receipt)
        except ValueError as exc:
            integer_partition = None
            integerization_receipt["partition_error"] = str(exc)
        with torch.no_grad():
            integer_output = mdl(*inputs)
        portable_agreement = _integerized_agreement(
            portable_output, integer_output, atol=a.agreement_atol, rtol=a.agreement_rtol
        )
        portable_agreement["reference"] = "portable_pt2e"
        integerization_receipt["portable_agreement"] = portable_agreement
        if selected_engine == "integer_reference":
            if independent is None:
                golden_agreement = {
                    "status": "failed",
                    "reference": "pt2e_integer",
                    "reason": independent_error or "selected integer reference returned no result",
                }
            else:
                golden_agreement = _integerized_agreement(independent.output, integer_output, atol=0.0, rtol=0.0)
                executed = independent.contraction_count
                selected = quant_stats.get("annotated_contractions") if isinstance(quant_stats, dict) else None
                seen = integerization_receipt["quantized_contractions_seen"]
                integerized = integerization_receipt["quantized_contractions_integerized"]
                if integer_partition is None or executed != integerized or selected != seen:
                    golden_agreement["status"] = "failed"
                    golden_agreement["reason"] = (
                        "selected/observed census or independently executed integer partition differs"
                    )
                reference_leaves, reference_abi = _output_abi(independent.output)
                reference_bytes = (
                    json.dumps(
                        {
                            "schema": "merlin.capture.integer_reference.v1",
                            "output_abi": reference_abi,
                            "outputs": [_to_native(value) for value in reference_leaves],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                reference_path = out / "integer-reference.json"
                reference_path.write_bytes(reference_bytes)
                golden_agreement.update(
                    reference="pt2e_integer",
                    source=independent_source,
                    output={"path": reference_path.name, "sha256": hashlib.sha256(reference_bytes).hexdigest()},
                    executed_contractions={
                        "conv2d": independent.conv2d_count,
                        "linear": independent.linear_count,
                        "matmul": independent.matmul_count,
                        "total": executed,
                        "selected": selected,
                        "observed": seen,
                        "integerized": integerized,
                        "preserved": integer_partition["preserved"] if integer_partition else None,
                    },
                )
            integerization_receipt["golden_agreement"] = golden_agreement
        else:
            integerization_receipt["golden_agreement"] = portable_agreement
    act_census = None
    if a.quantize_activation_contractions:
        # THE CONTRACTIONS quantize_ CANNOT REACH. The scheme replaces Linear weights; a matmul of two
        # activations and a convolution keep their float arithmetic under a program declared int8.
        # Only the int8 dynamic form is defined for them, so any other quantization is refused rather
        # than silently mixed with it.
        scheme_name = getattr(q, "scheme", None)
        if recipe is not None or scheme_name != "int8_dyn_act_int8_weight":
            raise SystemExit(
                f"--quantize-activation-contractions realises the int8 dynamic scheme's own form only; "
                f"the capture is quantized under "
                f"{'a derived recipe' if recipe is not None else repr(scheme_name)}"
            )
        # A sibling of this worker, imported by bare name: the worker runs under the capture
        # interpreter, which cannot import the package.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _activation_contractions as AC

        act_census = AC.install(mdl)
    nl_census = None
    if a.integer_nonlinear:
        # THE NONLINEAR LAYERS BETWEEN THE CONTRACTIONS, in integer arithmetic. Defined only on top of the
        # int8 contractions (the softmax's int8 numerators are the next contraction's own operand), so it
        # is refused without them rather than mixed into a floating-point program.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import _integer_nonlinear as NL

        nl_census = NL.install(mdl)
    trace_options = {"capture_trace": True, "original_frontend_snapshot": original_snapshot} if trace_supported else {}
    res = m2m.convert(
        mdl,
        inputs,
        backend="fx_importer",
        quantization=q,
        quantization_preapplied=(q is not None),
        level="linalg-on-tensors",
        func_name=a.func_name,
        weights_path=weights_path,
        **trace_options,
    )
    opaque = opaque_report(res.mlir_text)
    n_opaque = sum(opaque.values())
    if integerization_receipt is not None:
        integerization_receipt["exported_integer_mm_count"] = _exported_integer_mm_count(res.module)
    integerization_ok = integerization_receipt is None or (
        integer_partition is not None
        and integerization_receipt.get("accumulator_bound_checked") is True
        and integerization_receipt["exported_integer_mm_count"] > 0
        and integerization_receipt["integer_mm_emitted"] <= integerization_receipt["exported_integer_mm_count"]
        and integerization_receipt["golden_agreement"]["status"] == "passed"
    )
    # A recipe requests a format; only the applied transform and its emitted
    # arithmetic establish the captured scheme. In particular, an already-
    # integer graph that skipped observers must not inherit the request's label.
    realized_scheme = (
        None if a.already_quantized else (a.scheme or _SCHEME.get(a.dtype, (None, None))[0] if recipe is None else None)
    )
    if recipe is not None and q is not None and quant_stats is not None:
        if (
            q.scheme == "int8_static_act_int8_weight"
            and quant_stats.get("api") == "pt2e"
            and integerization_receipt is not None
            and integerization_ok
        ):
            realized_scheme = q.scheme
        elif (
            q.scheme == "int8_dyn_act_int8_weight"
            and quant_stats.get("api") == "quantize_"
            and quant_stats.get("layers_quantized", 0) > 0
        ):
            realized_scheme = q.scheme
        elif (
            q.scheme.startswith("fp8_")
            and quant_stats.get("api") == "pt2e"
            and quant_stats.get("annotated_contractions", 0) > 0
        ):
            realized_scheme = q.scheme

    (out / "linalg.mlir").write_text(res.mlir_text, encoding="utf-8")
    frontend_trace = getattr(res, "capture_trace", None)
    if frontend_trace is None:
        frontend_trace = {
            "schema": "m2m.capture_trace.v1",
            "status": "unavailable",
            "reason": "selected conversion did not return a frontend trace",
            "original": original_snapshot,
        }
    trace_path = out / "frontend-trace.json"
    trace_bytes = (json.dumps(frontend_trace, sort_keys=True, indent=2) + "\n").encode()
    trace_path.write_bytes(trace_bytes)
    # Ask the existing catalog implementation in this very capture process.
    # A workload-specific environment must not inherit another venv's denominator.
    catalog = _framework_catalog(torch)
    catalog_path = out / "pytorch-opset.json"
    catalog_bytes = (json.dumps(catalog, sort_keys=True, indent=2) + "\n").encode()
    catalog_path.write_bytes(catalog_bytes)

    # host torch-eager reference — THE golden. Run the (cast/quantized) model the compiler must reproduce.
    with torch.no_grad():
        y = mdl(*inputs)
    # The census of the forward that produced the golden, taken before anything runs the model again.
    act_sites = act_census.to_dict() if act_census is not None else None
    nl_sites = nl_census.to_dict() if nl_census is not None else None
    output_leaves, output_abi = _output_abi(y)
    output_values = [_to_native(x) for x in output_leaves]
    # Preserve the version-1 single-result JSON shape so existing operator capsules and caches remain
    # byte-compatible.  ``output_abi`` is what disambiguates one list-shaped tensor from many results.
    outputs = output_values[0] if len(output_values) == 1 else output_values
    input_leaves, input_abi = _input_abi(inputs)
    input_prov = [_to_native(x) for x in input_leaves]

    (out / "inputs.json").write_text(json.dumps(input_prov), encoding="utf-8")
    (out / "golden.json").write_text(json.dumps(outputs), encoding="utf-8")
    if not a.already_quantized:
        # An untransformed capture's golden IS the float model's output.
        reference = outputs if float_reference is None else float_reference["outputs"]
        (out / "float_reference.json").write_text(json.dumps(reference), encoding="utf-8")
    extra_path = _write_leaf_constants(mdl, inputs, weights_path, out / "extra.npz")
    precision = frontend_trace.get("precision") or {}
    precision_exact = precision.get("status") != "projected" and not precision.get("projections")
    meta = {
        "ok": bool(res.ok) and precision_exact,
        "precision_realization": precision if precision else {"status": "unknown"},
        "precision_conversion": precision_conversion,
        **({"fp32_staging": fp32_staging} if fp32_staging is not None else {}),
        "opaque": int(n_opaque),
        "opaque_detail": opaque,
        # WHICH quantization actually produced this program. Without it a weight-only capture and a
        # W8A8 one are indistinguishable after the fact, and they are different arithmetic.
        "scheme": realized_scheme,
        **({"capture_quantization": "already_materialized"} if a.already_quantized else {}),
        # The recipe a capture ran under, by content digest, and how far its outputs sit from the
        # floating-point model's. Both absent on a scheme-named capture.
        "recipe_sha256": recipe.get("recipe_sha256") if recipe is not None else None,
        "recipe": recipe,
        "recipe_agreement": agreement,
        "quantization_stats": quant_stats,
        **({"integerization_receipt": integerization_receipt} if integerization_receipt is not None else {}),
        # The contractions quantized BESIDE the scheme's own (activation x activation, patch
        # convolutions), with every site the mode saw and its verdict, from the golden's forward.
        # Absent when the capture did not ask for them.
        **(
            {
                "activation_contractions": {
                    "form": {
                        "first_operand": "int8 symmetric, dynamic, one scale per row, [-127, 127]",
                        "second_operand": "int8 symmetric, dynamic, one scale per output column",
                        "accumulate": "int32 (torch._int_mm)",
                        "dequantize": "int32 -> float x row scale x column scale",
                        "min_reduction_exclusive": AC.MIN_REDUCTION,
                    },
                    **act_sites,
                }
            }
            if act_sites is not None
            else {}
        ),
        # The nonlinear layers computed in integer arithmetic, with the census of the golden's forward.
        **(
            {
                "integer_nonlinear": {
                    "form": {
                        "softmax": "fixed exponent grid ln2/2**8, integer I-BERT exp, int8 numerators "
                        "relative to the row's largest term, normalized by their integer sum",
                        "gelu": "per-row int16 input, relu(x) - |x| h(|x|) with h a degree-6 fixed-point "
                        "polynomial, one per-row dequantizing scale",
                        "layer_norm": "per-row int16 input, integer mean, variance and Newton square root, "
                        "one integer multiply and shift per element; affine weight and bias per channel",
                    },
                    **nl_sites,
                }
            }
            if nl_sites is not None
            else {}
        ),
        "capture_diagnostics": [str(item)[:2000] for item in (getattr(res, "diagnostics", None) or [])[:100]],
        "path_taken": getattr(res, "path_taken", None),
        "dtype": a.dtype,
        "linalg_ops": res.mlir_text.count("linalg."),
        "func_name": a.func_name,
        "weights": weights_path,
        # model2MLIR externalization emits the authoritative placeholder -> state/input map from the
        # exact exported/quantized graph it lowered.  Preserve that map now; recreating torch.export
        # later from the source loader can have a different argument list (notably tensor-subclass
        # quantization expands parameters into inner tensors).
        "weights_manifest": weights_path + ".manifest.json",
        # The @forward leaves the weights file does not hold (registered buffers, lifted constants),
        # or None when the capture has none (see _write_leaf_constants).
        "extra": extra_path,
        "capture_abi_version": _CAPTURE_ABI_VERSION,
        "frontend_trace": {
            "path": str(trace_path),
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "status": frontend_trace.get("status", "unknown"),
        },
        "framework_catalog": {
            "path": str(catalog_path),
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "torch": catalog.get("torch"),
            "status": catalog.get("status", "unknown"),
        },
        "torch_seed": int(a.seed),
        "determinism": determinism,
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
    if a.diagnostic_model_copy:
        _diagnostic_model_copy(out, Path(a.loader), capture_api=capture_api)
    if a.materialize_bundle:
        from m2m.capture.bundle import write_bundle

        # Reuse the actual typed conversion and its prepared ExportedProgram.
        # A second export can reorder lifted constants or sever source lineage.
        write_bundle(
            mdl,
            tuple(inputs),
            out,
            quant=q,
            quantization_preapplied=(q is not None),
            session=session,
            source_path=Path(a.loader),
            capture_trace=True,
            conversion_result=res,
        )
    if completed is not None:
        completed(
            dict(module=mdl, inputs=tuple(inputs), session=session, quant=q, conversion_result=res, metadata=meta)
        )
    # a machine-readable tail line the parent greps for, even if warnings precede it
    print("__M2M_CAPTURE__ " + json.dumps({"ok": meta["ok"], "opaque": meta["opaque"]}))
    return 0 if (res.ok and n_opaque == 0 and integerization_ok and precision_exact) else 3


if __name__ == "__main__":
    raise SystemExit(main())
