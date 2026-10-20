"""Retain attention capture/reference evidence without inventing numeric gates."""

from __future__ import annotations

import shutil
from collections import Counter

import numpy as np

from merlin.common import mlir_query as query
from merlin.targetgen.capsule_source import _portable_weights_reference

from . import development_cohort as F


def retain(description, captured, expected, golden, directory):
    module = query.parse(captured.linalg_mlir)
    domains = Counter()
    for op in query.walk(module, "linalg.matmul", "linalg.generic"):
        if query.op_name(op) != "linalg.matmul" and query.attr_str(op, "prov.family") != "contraction":
            continue
        if len(op.operands) >= 2 and op.results:
            domains[tuple(query.type_shape_dtype(x.type)[1] for x in [*op.operands[:2], op.results[0]])] += 1
    count = description["contractions"]
    if domains != Counter({("i8", "i8", "i32"): count, ("f32", "f32", "f32"): count}):
        raise ValueError("complete attention capture lost a required integer QK or floating PV contraction")
    observed = [{"operand_result_types": list(key), "count": value} for key, value in sorted(domains.items())]
    directory.mkdir()
    metrics = []
    for abi, actual, wanted in zip(captured.meta["output_abi"], golden, expected, strict=True):
        if wanted.dtype != np.float32:
            np.testing.assert_array_equal(actual, wanted)
            metrics.append({"dtype": str(wanted.dtype), "comparison": "exact_int", "equal": True})
            continue
        actual = np.asarray(actual, dtype=np.float32)
        if not np.isfinite(actual).all() or not np.isfinite(wanted).all():
            raise ValueError("attention diagnostic produced nonfinite required outputs")
        difference = actual.astype(np.float64) - wanted.astype(np.float64)
        metrics.append(
            {
                "dtype": abi["dtype"],
                "comparison": "UNKNOWN; no selected floating policy",
                "bit_equal": np.array_equal(actual, wanted),
                "max_abs_error": float(np.max(np.abs(difference), initial=0)),
                "rel_l2": float(
                    np.linalg.norm(difference) / max(float(np.linalg.norm(wanted.astype(np.float64))), 1e-300)
                ),
            }
        )
    portable, relocation = _portable_weights_reference(captured.linalg_mlir, captured.weights_path, sidecar=True)
    (directory / "capture.original.interface.mlir").write_text(captured.linalg_mlir)
    (directory / "capture.interface.mlir").write_text(portable)
    (directory / "capture.pytorch.py").write_text(captured.pytorch_src)
    (directory / "capture-meta.json").write_bytes(F._bytes(captured.meta))
    for path in [captured.weights_path, captured.meta.get("weights_manifest")]:
        if path:
            shutil.copyfile(
                path,
                directory / ("capsule.weights.safetensors" + (".manifest.json" if str(path).endswith(".json") else "")),
            )
    np.savez(directory / "independent-reference.npz", **{f"output_{i}": x for i, x in enumerate(expected)})
    np.savez(directory / "captured-reference.npz", **{f"output_{i}": np.asarray(x) for i, x in enumerate(golden)})
    pending = {
        "schema": "merlin.phase0.complete_attention_pending_capsule.v1",
        "source_description": description,
        "observed_contraction_domains": observed,
        "weights_reference_relocation": relocation,
        "floating_diagnostics": metrics,
        "writer_admission": "REFUSED: selected complete attention numerical policy unavailable",
        "phase0_admitted": False,
        "accelerator_execution": False,
        "timing_qualified": False,
    }
    (directory / "pending-capsule.json").write_bytes(F._bytes(pending))
    return {
        "status": "complete_attention_capture_reference_diagnostic_contract_pending",
        "pending_capsule": str(directory / "pending-capsule.json"),
        "observed_contraction_domains": observed,
        "floating_diagnostics": metrics,
        "reason": pending["writer_admission"],
        "files": {str(p.relative_to(directory)): F._identity(p) for p in sorted(directory.rglob("*")) if p.is_file()},
    }
