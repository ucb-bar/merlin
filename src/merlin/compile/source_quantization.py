"""Typed contract for captures whose integer arithmetic is already in source MLIR.

An ``int8`` weight-only capture needs Merlin's W8A8 conversion. A torchAO W8A8
capture already contains quantize, typed integer matmul, and dequantize operations.
Its framework ``golden.npy`` is the W8A8 reference and the compiler must preserve
that graph, including any float contractions the source did not quantize.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from merlin.common import mlir_query as query
from merlin.kernels.shapes import zero_initialised
from merlin.llvmlower.unsigned_byte_dot import byte_dot_kind
from merlin.targetgen import application_inventory

SOURCE_W8A8_SCHEME = "int8_dyn_act_int8_weight"


def inspect_source_w8a8(bundle: str | Path) -> dict:
    """Require a TorchAO-labelled graph with proven signed or unsigned byte→i32 roots.

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
    # The scheme name alone does not identify the activation arithmetic. Older
    # TorchAO captures leave these calls opaque, losing their quant_min/max and
    # eps arguments. Their legacy fallback transcribes a different reduced-range
    # profile from the explicit [-128, 127] graph in the current source captures.
    # An exact-source route must use its own captured operations, regardless of
    # which TorchAO version produced them.
    opaque_affine = [
        op.callee.root_reference.data
        for op in query.walk(module, "func.call")
        if op.callee.root_reference.data.startswith(
            ("torchao_choose_qparams_affine_default", "torchao_quantize_affine_default")
        )
    ]
    if opaque_affine:
        raise ValueError(
            "--dtype int8_w8a8 requires explicit captured activation quantization; "
            f"opaque TorchAO affine calls lose their numerical parameters: {sorted(set(opaque_affine))}"
        )
    candidates = [
        op for op in query.walk(module, "linalg.generic") if query.attr_str(op, "prov.aten") == "aten._int_mm.default"
    ]
    kinds = {op: byte_dot_kind(op) for op in candidates}
    invalid = [op for op in candidates if kinds[op] is None]
    nonzero_init = [op for op in candidates if not zero_initialised(op)]
    if not candidates or invalid or nonzero_init:
        raise ValueError(
            f"--dtype int8_w8a8 requires exact s8×s8 or u8×s8→i32 _int_mm roots with zero initializers; "
            f"found {len(candidates)} candidates, {len(invalid)} malformed, "
            f"{len(nonzero_init)} without a proven zero initializer"
        )
    for op in candidates:
        expected = ("uint8" if kinds[op] == "u8s8" else "int8", "int8")
        for index, label in enumerate(("lhs", "rhs")):
            declared = query.attr_str(op, "prov.int_mm_" + label + "_dtype")
            owner = op.inputs[index].owner
            produced = query.attr_str(owner, "prov.orig_dtype") if hasattr(owner, "attributes") else None
            witnesses = [value for value in (declared, produced) if value]
            if not witnesses or any(value != expected[index] for value in witnesses):
                raise ValueError(
                    "source integer operand signedness is unknown or contradicts the _int_mm body: "
                    f"{label} dtype evidence {witnesses}, arithmetic requires {expected[index]}; "
                    "recapture with the byte-typed frontend; old source bytes are not repaired"
                )
    valid_macs = 0
    for op in candidates:
        (m, k), _ = query.type_shape_dtype(op.inputs[0].type)
        (wk, n), _ = query.type_shape_dtype(op.inputs[1].type)
        assert k == wk  # checked structurally by exact_int_mm_generic_operation
        valid_macs += m * n * k
    receipt = application_inventory.verify_capture_receipt(source)
    return {
        "source_scheme": scheme,
        "activation_quantization": "explicit_source_graph",
        # Canonical matrix ABI after exact unsigned normalization. Original
        # signedness stays in source_matrix_operand_formats below. This differs
        # from the weight-only recipe registered under the plain `int8` token.
        "matrix_operand_format": "signed_int8",
        "matrix_accumulator_format": "int32",
        "source_matrix_operand_formats": {
            kind: list(kinds.values()).count(kind) for kind in sorted(set(kinds.values()))
        },
        "exact_unsigned_normalization_required": "u8s8" in kinds.values(),
        "integer_matmuls": len(candidates),
        "integer_valid_macs": valid_macs,
        "source_mlir_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "capture_receipt": receipt,
        "quantization_passes": "preserve_source",
        "reference_file": "golden.npy",
        "reference_kind": "torchao_w8a8_framework",
    }
