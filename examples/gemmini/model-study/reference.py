#!/usr/bin/env python3
"""Interpret one captured forward independently of a device compiler.

This creates a fresh reference, not a recovered historical bundle or a complete
recurrent-session result. It also checks the ordinary device rewrite against the
reference interpreter. Expected arrays remain evaluator inputs only.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import time

import numpy as np


def canonical_outputs(values, specs, *, finite_required):
    """Match the declared output ABI without silently narrowing numeric values."""
    from merlin.model_check.full_output_audit import DTYPES

    if len(values) != len(specs):
        raise ValueError("reference output count differs from the declared ABI")
    result, representations = [], []
    for index, (value, (shape, dtype)) in enumerate(zip(values, specs, strict=True)):
        value = np.asarray(value)
        if list(value.shape) != list(shape) or dtype not in DTYPES:
            raise ValueError("reference output shape or type differs from the declared ABI")
        required = np.dtype(DTYPES[dtype])
        original_dtype = str(value.dtype)
        if dtype == "i1":
            if value.dtype not in (np.dtype("bool"), np.dtype("int8"), np.dtype("uint8")):
                raise ValueError("logical Boolean reference needs one-byte storage")
            if not np.all((value == 0) | (value == 1)):
                raise ValueError("logical Boolean reference contains noncanonical bytes")
            if not np.all(np.array(value, copy=True, order="C").view(np.uint8) <= 1):
                raise ValueError("logical Boolean reference contains noncanonical storage bytes")
            converted = value.astype(np.bool_, copy=False)
            if converted.tobytes() != value.tobytes():
                raise ValueError("logical Boolean normalization changed stored bytes")
            value = converted
        elif value.dtype != required:
            raise ValueError("reference output dtype differs from the declared ABI")
        if finite_required and value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError("reference violates the selected finite-output requirement")
        result.append(value.copy(order="C"))
        representations.append({"output": index, "declared_dtype": dtype,
                                "interpreter_dtype": original_dtype, "output_dtype": str(value.dtype),
                                "shape": list(shape), "storage_bytes_unchanged": True})
    return result, representations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "models", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("model", "program"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)

    import importlib.util
    spec = importlib.util.spec_from_file_location("model_study_reference_admission",
                                                Path(__file__).with_name("reference_admission.py"))
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    from merlin.common.digest import sha256_file
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.llvmlower.c_runtime import _out_specs
    from merlin.model_check.bundle import Bundle
    from merlin.model_check.catalog import ProgramCatalog, evaluate
    from merlin.model_check.models import load, selected_float_policy
    from merlin.model_check.references import reference_math_context
    from merlin.runtime import linalg_numpy as LN
    from merlin.runtime.dispatch_runtime import resolve_forward_args

    output = args.output.resolve()
    catalog_root = args.catalog.resolve(strict=True)
    catalog = ProgramCatalog.load(catalog_root)
    bundle = Bundle(args.bundle.resolve(strict=True))
    models_bytes = args.models.read_bytes()
    models_pin = {"path": str(args.models.resolve()), "sha256": hashlib.sha256(models_bytes).hexdigest()}
    if admission.pin(args.models) != models_pin:
        raise ValueError("reference models configuration changed while being read")
    if (bundle.root / "configuration/models.yaml").read_bytes() != models_bytes:
        raise ValueError("reference bundle and selected models configuration differ")
    models = load(args.models, source_bytes=models_bytes)
    model = models.model(args.model)
    model.program(args.program)
    policy = selected_float_policy(models)
    if catalog.target != models.target:
        raise ValueError("catalog target differs from selected models configuration")
    if output.exists() or any(output.is_relative_to(p.resolve())
                              for p in (catalog_root, catalog.stage, bundle.root)):
        raise ValueError("reference needs a fresh output outside its input directories")
    inputs = [Path(__file__), Path(admission.__file__), args.models, catalog_root / "catalog.json",
              catalog_root / "dev/device_signatures.json", catalog.ref_prepared,
              catalog.dev_prepared, Path(LN.__file__)]
    inputs = list(dict.fromkeys(p.resolve() for p in inputs))
    pins = [models_pin if str(p) == models_pin["path"] else
            {"path": str(p), "sha256": sha256_file(p)} for p in inputs]
    captured = strict_tree_sha256(catalog.stage)
    output.mkdir(parents=True)
    started = time.monotonic()
    with reference_math_context(model, bundle, output / "math"):
        observed = evaluate(catalog, resolve_forward_args(catalog.stage))
        contract = LN.numeric_contract()
    specs = _out_specs(catalog.stage / "model.mlir")
    expected, representation = canonical_outputs(observed.reference, specs,
                                                finite_required=policy["finite_required"])
    rewritten, _ = canonical_outputs(observed.rewritten, specs,
                                     finite_required=policy["finite_required"])
    exact = [a.tobytes() == b.tobytes() for a, b in zip(expected, rewritten, strict=True)]
    np.savez(output / "expected.npz", **{"out" + str(i): v for i, v in enumerate(expected)})
    np.savez(output / "rewritten.npz", **{"out" + str(i): v for i, v in enumerate(rewritten)})
    unchanged = (all(sha256_file(row["path"]) == row["sha256"] for row in pins)
                 and strict_tree_sha256(catalog.stage) == captured)
    document = {"schema": admission.SCHEMA,
                "status": "completed" if exact and all(exact) and unchanged else "failed",
                "selected_model": args.model, "selected_program": args.program,
                "numeric_contract": contract, "comparison_policy": policy,
                "outputs": [{"shape": list(v.shape), "dtype": str(v.dtype)} for v in expected],
                "representation": representation, "reference_vs_rewritten_exact": exact,
                "all_outputs_exact": bool(exact) and all(exact), "inputs_unchanged": unchanged,
                "inputs": pins, "capture": {"path": str(catalog.stage.resolve()), **captured},
                "models": models_pin, "producer": admission.pin(Path(__file__)),
                "catalog": admission.pin(catalog_root / "catalog.json"),
                "expected": {"path": str(output / "expected.npz"),
                             "sha256": sha256_file(output / "expected.npz")},
                "seconds": time.monotonic() - started,
                "scope": "one captured forward, independently interpreted; no candidate, hardware, model-quality or complete-session qualification"}
    if document["status"] == "completed":
        admission.admit(document, reference=output / "expected.npz",
                        reference_sha256=document["expected"]["sha256"], models=args.models,
                        model=args.model, program=args.program, stage=catalog.stage,
                        host_math_policy=contract.get("host_math", {}).get("policy", "native"),
                        comparison_policy=policy)
    (output / "result.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"status": document["status"], "result": str(output / "result.json")}), flush=True)
    return 0 if document["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
