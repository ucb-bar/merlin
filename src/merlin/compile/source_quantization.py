"""Typed contract for captures whose integer arithmetic is already in source MLIR.

An ``int8`` weight-only capture needs Merlin's W8A8 conversion. A torchAO W8A8
capture already contains quantize, signed integer matmul, and dequantize operations.
Its framework ``golden.npy`` is the W8A8 reference and the compiler must preserve
that graph, including any float contractions the source did not quantize.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from merlin.common import mlir_query as query
from merlin.targetgen.application_inventory import exact_int_mm_generic_operation, verify_capture_receipt

SOURCE_W8A8_SCHEME = "int8_dyn_act_int8_weight"


def inspect_source_w8a8(bundle: str | Path) -> dict:
    """Require a TorchAO-labelled graph with real signed i8×i8→i32 roots.

    Inspect parsed operations and reduction bodies, not textual occurrences of
    ``aten._int_mm`` (the provenance tag also appears on constants and splats).
    """
    source = Path(bundle) / "model.mlir"
    module = query.parse(source)
    scheme = query.attr_str(module, "prov.quantization")
    if scheme != SOURCE_W8A8_SCHEME:
        raise ValueError(
            f"--dtype int8_w8a8 requires module prov.quantization={SOURCE_W8A8_SCHEME!r}; {source} declares {scheme!r}"
        )
    candidates = [
        op for op in query.walk(module, "linalg.generic") if query.attr_str(op, "prov.aten") == "aten._int_mm.default"
    ]
    invalid = [op for op in candidates if not exact_int_mm_generic_operation(op)]
    if not candidates or invalid:
        raise ValueError(
            f"--dtype int8_w8a8 requires exact signed i8×i8→i32 _int_mm roots; "
            f"found {len(candidates)} candidates, {len(invalid)} malformed"
        )
    valid_macs = 0
    for op in candidates:
        (m, k), _ = query.type_shape_dtype(op.inputs[0].type)
        (wk, n), _ = query.type_shape_dtype(op.inputs[1].type)
        assert k == wk  # checked structurally by exact_int_mm_generic_operation
        valid_macs += m * n * k
    receipt = verify_capture_receipt(source)
    return {
        "source_scheme": scheme,
        # This is the integer ROOT's storage/accumulation type. It is deliberately
        # distinct from the torchAO recipe; quant_formats.registry.yaml's `int8`
        # entry describes a weight-only capture and is not this source scheme.
        "matrix_operand_format": "signed_int8",
        "matrix_accumulator_format": "int32",
        "integer_matmuls": len(candidates),
        "integer_valid_macs": valid_macs,
        "source_mlir_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "capture_receipt": receipt,
        "quantization_passes": "preserve_source",
        "reference_file": "golden.npy",
        "reference_kind": "torchao_w8a8_framework",
    }
