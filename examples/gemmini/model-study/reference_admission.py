"""Admit a new captured-forward reference from positive, byte-bound evidence.

Historical producer records remain evidence in their original schemas. Missing
fields never acquire success or identity through compatibility defaults.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA = "merlin.model_study.captured_reference.v2"


def pin(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def unchanged(record):
    if type(record) is not dict or set(record) != {"path", "sha256"} or pin(record["path"]) != record:
        raise ValueError("reference selected input changed or lacks an exact file binding")


def admit(document, *, reference, reference_sha256, models, model, program, stage,
          host_math_policy, comparison_policy):
    """Require every positive axis and join it to this exact caller selection."""
    from merlin.compile.model_execution_inputs import strict_tree_sha256

    if type(document) is not dict or document.get("schema") != SCHEMA:
        raise ValueError("reference needs the explicit captured-reference v2 schema; legacy proof is not inferred")
    if (document.get("status") != "completed" or document.get("all_outputs_exact") is not True
            or document.get("inputs_unchanged") is not True):
        raise ValueError("reference did not positively complete all output and immutable-input checks")
    if document.get("selected_model") != model or document.get("selected_program") != program:
        raise ValueError("reference selected model or program differs")
    actual_reference = pin(reference)
    if actual_reference["sha256"] != reference_sha256 or document.get("expected") != actual_reference:
        raise ValueError("reference expected file differs from its admitted bytes")
    if document.get("models") != pin(models):
        raise ValueError("reference models configuration differs")
    selected_stage = Path(stage).resolve(strict=True)
    capture = {"path": str(selected_stage), **strict_tree_sha256(selected_stage)}
    if document.get("capture") != capture:
        raise ValueError("reference captured source or inputs differ")
    contract = document.get("numeric_contract")
    if type(contract) is not dict or not isinstance(contract.get("version"), str) or not contract["version"]:
        raise ValueError("reference has no positive interpreter numerical contract")
    observed = (contract.get("host_math") or {}).get("policy", "native")
    if observed != host_math_policy or document.get("comparison_policy") != comparison_policy:
        raise ValueError("reference numerical policies differ from selected build")
    inputs = document.get("inputs")
    if type(inputs) is not list or not inputs:
        raise ValueError("reference has no immutable producer/input roster")
    if len({row.get("path") for row in inputs if type(row) is dict}) != len(inputs):
        raise ValueError("reference input roster repeats or omits a path")
    for row in inputs:
        unchanged(row)
    required = [document.get("models"), document.get("producer"), document.get("catalog")]
    if any(type(row) is not dict or row not in inputs for row in required):
        raise ValueError("reference lacks its models, producer or catalog input binding")
    import numpy as np
    from merlin.llvmlower.c_runtime import NP_OF, _out_specs

    specs = _out_specs(selected_stage / "model.mlir")
    exact = document.get("reference_vs_rewritten_exact")
    if type(exact) is not list or len(exact) != len(specs) or not exact or any(v is not True for v in exact):
        raise ValueError("reference rewrite consistency does not cover every declared output")
    with np.load(reference, allow_pickle=False) as archive:
        if len(archive.files) != len(specs) or set(archive.files) != {f"out{i}" for i in range(len(specs))}:
            raise ValueError("reference NPZ output names or arity differ")
        outputs = []
        for i, (shape, dtype) in enumerate(specs):
            value = archive[f"out{i}"]
            wanted = np.dtype(np.bool_ if dtype == "i1" else np.float32 if dtype == "bf16" else NP_OF[dtype])
            if list(value.shape) != list(shape) or value.dtype != wanted:
                raise ValueError("reference output shape or dtype differs from captured ABI")
            if value.dtype.kind == "b" and np.any(value.view(np.uint8) > 1):
                raise ValueError("reference Boolean output contains noncanonical storage")
            if comparison_policy["finite_required"] and not np.isfinite(value).all():
                raise ValueError("reference violates selected finite-output requirement")
            outputs.append({"shape": list(value.shape), "dtype": str(value.dtype)})
    if document.get("outputs") != outputs:
        raise ValueError("reference declared output roster differs from its actual arrays")
    # Reopen every selected span after arrays, including the original capture.
    if pin(reference) != actual_reference or {"path": str(selected_stage), **strict_tree_sha256(selected_stage)} != capture:
        raise ValueError("reference or capture changed during admission")
    for row in inputs:
        unchanged(row)
    return {"schema": "merlin.model_study.reference_admission.v1", "status": "admitted",
            "selected_model": model, "selected_program": program, "capture": capture,
            "models": document["models"], "expected": actual_reference,
            "scope": "Current positive producer and selected input/output custody; no historical or model-quality claim"}
