#!/usr/bin/env python3
"""Capture a model through model2MLIR inside its torch-enabled environment.

The request names a loader, dtype and output directory. The worker constructs and
quantizes the model, lowers it through m2m, rejects opaque operations, evaluates the
host reference and writes the MLIR, weights, inputs, golden output and metadata.
Scoped quantization recipes use the selected package's Merlin admission screen.
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
import sys
from pathlib import Path

from merlin.targetgen._m2m_capture_support import (
    _capture_api_report,
    _diagnostic_model_copy,
    _RecordedForwards,
    _seed_capture,
)

if __package__:
    from ._m2m_capture_artifacts import (
        _framework_catalog,
        finalize_session_artifacts,
        frontend_trace_bytes,
        write_conversion_status,
        write_frontend_trace,
    )
    from ._m2m_capture_artifacts import (
        _framework_cpu_dispatch as _framework_cpu_dispatch,
    )
    from ._m2m_capture_deployment import _mlir_dtype, _numeric_deployment
else:
    from merlin.targetgen._m2m_capture_artifacts import (
        _framework_catalog,
        finalize_session_artifacts,
        frontend_trace_bytes,
        write_conversion_status,
        write_frontend_trace,
    )
    from merlin.targetgen._m2m_capture_artifacts import (
        _framework_cpu_dispatch as _framework_cpu_dispatch,
    )
    from merlin.targetgen._m2m_capture_deployment import _mlir_dtype, _numeric_deployment

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


def _calibration_stream_limit(RQ, recipe) -> int:
    """Samples a calibration stream must supply: what the recipe observes, and the agreement receipt."""
    observed = RQ.calibration_limit(
        recipe, inspect.signature(RQ.apply_recipe).parameters["calibration_samples"].default
    )
    return max(observed, inspect.signature(RQ.agreement).parameters["limit"].default)


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
    from merlin.targetgen._dynamic_integer_reference import compare_outputs

    return compare_outputs(before, after, atol=atol, rtol=rtol, output_abi=_output_abi)


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


def _materialize_session(
    model, inputs, args, out: Path, *, loader, determinism: dict, dependencies: list, torch
) -> int:
    """Capture an explicit multi-program protocol; do not infer stages by model name."""
    if not args.materialize_bundle:
        raise ValueError("deferred multi-program capture requires --materialize-bundle")
    recipe = json.loads(Path(args.recipe).read_text()) if args.recipe else None
    quantized = args.dtype == "int8" and recipe is not None
    native = bool(getattr(args, "native_integer_source", False))
    if native and (args.dtype != "int8" or not args.already_quantized or recipe or args.stage_fp32):
        raise ValueError("native integer session requires unchanged signed source without precision rewrites")
    if (
        (args.dtype not in {"fp32", "f32"} and not quantized and not native)
        or args.scheme
        or (args.already_quantized and not native)
    ):
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
    if quantized or args.stage_fp32 or native:
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
            selections[program.name] = (
                "native_integer_v1"
                if native
                else "recipe"
                if selected
                else "no_recipe_work"
                if quantized
                else "fp32_staged"
            )
            stage_args = argparse.Namespace(**vars(args))
            stage_args.out = str(out / "stages" / program.name)
            # The joined writer below materializes this exact conversion once,
            # including its complete trajectory. Do not run that trajectory here too.
            stage_args.materialize_bundle = False
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
                capture_regions=True,
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
    if quantized or args.stage_fp32 or native:
        if source_state != (_shared_tensor_inventory(source_owner) if has_source_state else []):
            raise ValueError("multi-program source weights changed during bundle publication")
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
            frontend_trace=write_frontend_trace(stage / "frontend-trace.json", trace),
            framework_catalog={
                "path": catalog_path.name,
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "torch": catalog.get("torch"),
                "status": catalog.get("status", "unknown"),
            },
        )
        meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
        # Metadata/catalog/trace changes must be committed by the producer receipt,
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
            if quantized or args.stage_fp32 or native
            else "untransformed"
        ),
        **({"source_state_unchanged": True} if args.stage_fp32 or native else {}),
        "session_contract_sha256": hashlib.sha256((out / "session_contract.yaml").read_bytes()).hexdigest(),
        "qualification": "capture only; no target lowering, execution or application accuracy claim",
    }
    finalize_session_artifacts(out, report)
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
        "--qualify-qdq",
        action="store_true",
        help="require supported Q/DQ source correspondence from this exact conversion (no spec approval)",
    )
    ap.add_argument("--qdq-torch-wheel", help="pinned PyTorch source-oracle wheel used to verify installed bytes")
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
        "--native-integer-source",
        action="store_true",
        help="versioned unchanged native integer source; retain original frontend trace without quantization",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="Python/NumPy/Torch RNG seed applied before loader import and construction"
    )
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="intra-op CPU threads; reference bytes depend on it, so it is recorded and replayed (default 1)",
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
    qualify_qdq = getattr(a, "qualify_qdq", False)
    if qualify_qdq and (not a.m2m_dir or not getattr(a, "qdq_torch_wheel", None)):
        ap.error("--qualify-qdq requires --m2m-dir and --qdq-torch-wheel")
    native_integer_source = getattr(a, "native_integer_source", False)
    if native_integer_source and (
        not a.already_quantized
        or a.dtype != "int8"
        or a.recipe
        or a.scheme
        or a.stage_fp32
        or a.quantize_activation_contractions
        or a.integer_nonlinear
    ):
        ap.error("--native-integer-source requires unchanged int8 source without additional transformations")
    if a.materialize_bundle and a.diagnostic_model_copy:
        ap.error("--materialize-bundle and --diagnostic-model-copy are mutually exclusive")
    if not 0 <= a.seed < 2**32:
        ap.error("--seed must be an unsigned 32-bit integer")
    if not 1 <= a.cpu_threads <= 256:
        ap.error("--cpu-threads must be between 1 and 256")
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
    determinism = _seed_capture(a.seed, torch, a.cpu_threads)
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
    # Direct recipe captures also reach the bundle writer already quantized.
    # Freeze a real source trajectory first, including diagnostic sessions;
    # the compatibility quality filename must not be mistaken for FP32 evidence.
    if a.recipe and not a.stage_fp32 and session is not None:
        from merlin.targetgen._capture_session_reference import pre_quantization_session_reference

        session = pre_quantization_session_reference(mdl, tuple(inputs), session, include_diagnostic=True)
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
            calibration_limit = _calibration_stream_limit(RQ, json.loads(Path(a.recipe).read_text(encoding="utf-8")))
            calibration = _freeze_calibration(samples, limit=calibration_limit, normalize=RQ._as_tuple, torch=torch)
    original_snapshot = {
        "status": "unavailable",
        "stage": "original",
        "reason": "selected model2MLIR does not expose frontend capture tracing",
    }
    trace_supported = "capture_trace" in inspect.signature(m2m.convert).parameters
    # Explicit Q/DQ qualification observes the supplied Q/DQ graph itself. This
    # supplies no lineage to a floating model before external quantization.
    if trace_supported and (not a.already_quantized or native_integer_source or qualify_qdq):
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

        recorded = None
        if not a.stage_fp32:
            hook = getattr(loader, "get_calibration_inputs", None)
            if callable(hook):
                # A hook may decline (None): the stage then calibrates on its own example, as
                # without one. A stream is read only as far as the recipe and the receipt use it.
                samples = hook(mdl, tuple(inputs))
                if samples is not None:
                    from itertools import islice

                    recorded = _RecordedForwards(mdl, limit=inspect.signature(RQ.agreement).parameters["limit"].default)
                    with recorded:
                        calibration = list(islice(samples, _calibration_stream_limit(RQ, recipe)))
            else:
                stream = getattr(mdl, "session_images", None)
                if isinstance(stream, torch.Tensor) and stream.shape[0] > 0:
                    calibration = [(stream[i],) for i in range(int(stream.shape[0]))]
        reference = mdl
        static = recipe["activation"]["mode"] == "static"
        agreement_reference = recorded if static and recorded is not None and recorded.module is mdl else None
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
        agreement = RQ.agreement(
            reference if agreement_reference is None else agreement_reference,
            mdl,
            [tuple(inputs), *(calibration or [])],
        )
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
    # A recipe whose dynamic block emitted every integer contraction leaves PT2E nothing to integerize.
    pt2e_selected = not isinstance(quant_stats, dict) or quant_stats.get("annotated_contractions") != 0
    if q is not None and q.scheme == "int8_static_act_int8_weight" and not a.already_quantized and pt2e_selected:
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
    if (
        q is not None
        and q.scheme == "int8_static_act_int8_weight"
        and not a.already_quantized
        and not pt2e_selected
        and (recipe or {}).get("software_numerical_engine") == "integer_reference"
    ):
        from merlin.targetgen import _dynamic_integer_reference as dynamic_reference

        integerization_receipt = dynamic_reference.capture_reference(
            mdl, tuple(inputs), out, output_abi=_output_abi, to_native=_to_native, recipe=recipe
        )
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
    qdq_identity = None
    if qualify_qdq:
        from merlin.targetgen.qdq_capture import begin_worker

        qdq_identity = begin_worker(a, out)
        if qdq_identity is None:
            return 3
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
    # Preserve this boundary before later eager evaluation or bundle writing can
    # fail. A conversion refusal must remain visible even if a subsequent error
    # prevents final metadata from being written; this is not an execution grade.
    write_conversion_status(out / "conversion-status.json", res)
    opaque = opaque_report(res.mlir_text)
    n_opaque = sum(opaque.values())
    qdq_source = None
    if qualify_qdq:
        from merlin.targetgen.qdq_capture import check_worker

        qdq_source = check_worker(res, qdq_identity, out, opaque)
        if qdq_source["status"] != "matched":
            return 3
    if integerization_receipt is not None:
        integerization_receipt["exported_integer_mm_count"] = _exported_integer_mm_count(res.module)
    dynamic_only_reference = (integerization_receipt or {}).get("schema") == "merlin.dynamic_integer_reference.v1"
    integerization_ok = integerization_receipt is None or (
        (dynamic_only_reference or integer_partition is not None)
        and integerization_receipt.get("accumulator_bound_checked") is True
        and integerization_receipt["exported_integer_mm_count"] > 0
        and integerization_receipt["integer_mm_emitted"] <= integerization_receipt["exported_integer_mm_count"]
        and integerization_receipt["golden_agreement"]["status"] == "passed"
    )
    if dynamic_only_reference:
        dynamic_reference.validate_receipt(
            integerization_receipt,
            annotated=quant_stats.get("annotated_contractions"),
            dynamic_calls=(quant_stats.get("dynamic_contractions") or {}).get("device_calls"),
            exported=integerization_receipt["exported_integer_mm_count"],
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
    trace_bytes = frontend_trace_bytes(frontend_trace)
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
        **({"qdq_source": qdq_source} if qdq_source is not None else {}),
        "precision_realization": precision if precision else {"status": "unknown"},
        "precision_conversion": precision_conversion,
        **({"fp32_staging": fp32_staging} if fp32_staging is not None else {}),
        "opaque": int(n_opaque),
        "opaque_detail": opaque,
        # WHICH quantization actually produced this program. Without it a weight-only capture and a
        # W8A8 one are indistinguishable after the fact, and they are different arithmetic.
        "scheme": realized_scheme,
        **({"capture_quantization": "already_materialized"} if a.already_quantized else {}),
        **({"source_mode": "native_integer_v1"} if native_integer_source else {}),
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
    if dynamic_only_reference and a.materialize_bundle:
        dynamic_reference.bind_reference_source(out)
    if completed is not None:
        completed(
            dict(module=mdl, inputs=tuple(inputs), session=session, quant=q, conversion_result=res, metadata=meta)
        )
    # a machine-readable tail line the parent greps for, even if warnings precede it
    if not integerization_ok:
        print(
            "__M2M_INTEGERIZATION_FAILURE__ "
            + json.dumps(
                {key: integerization_receipt.get(key) for key in ("partition_error", "refusals", "golden_agreement")}
            )
        )
    print("__M2M_CAPTURE__ " + json.dumps({"ok": meta["ok"], "opaque": meta["opaque"]}))
    return 0 if (res.ok and n_opaque == 0 and integerization_ok and precision_exact) else 3


if __name__ == "__main__":
    # Quantization is followed by further FX cleanup/re-export in model2mlir.
    # Preserve vmap control effects AND live nested GraphModule state through
    # later export/cleanup too: those passes may share child modules with the
    # final eager golden model. No model arithmetic/framework installation edits.
    try:
        import _recipe_quantizer as _rq_effects
    except ImportError:
        from merlin.targetgen import _recipe_quantizer as _rq_effects
    with _rq_effects.preserve_fx_vmap_effects(), _rq_effects._pt2e_nested_graph_cleanup():
        raise SystemExit(main())
