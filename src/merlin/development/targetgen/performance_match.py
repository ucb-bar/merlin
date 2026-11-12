"""Conservative source-to-performance-capsule form matching.

An equal M/K/N is useful for selecting a diagnostic, but it is not proof that
the capsule executes the captured operation.  In particular, a linalg.generic
contraction can encode a different body, layout, or quantization.  Only a
source recognizer which checks those details may return an exact arithmetic
form here.  Even that result says nothing about epilogue chains, compiler
placement, or target timing.
"""

from __future__ import annotations

from collections.abc import Mapping

from merlin.development.targetgen.application_inventory import exact_int_mm_geometry


def _dtype(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return {"int8": "i8", "int32": "i32", "float32": "f32"}.get(value, value)


def _capsule_matmul(descriptor: Mapping) -> tuple[tuple[int, int, int], str, str] | None:
    operation = descriptor.get("operation")
    if not isinstance(operation, Mapping) or operation.get("op") not in {"matmul", "fused_matmul_bias"}:
        return None
    attributes = operation.get("attributes")
    inputs = descriptor.get("inputs")
    if not isinstance(attributes, Mapping) or not isinstance(inputs, list):
        return None
    named_inputs = [item for item in inputs if isinstance(item, Mapping) and isinstance(item.get("name"), str)]
    named = {item["name"]: item for item in named_inputs}
    lhs, weight = named.get(attributes.get("lhs")), named.get(attributes.get("weight"))
    if not isinstance(lhs, Mapping) or not isinstance(weight, Mapping):
        return None
    a, w = lhs.get("shape"), weight.get("shape")
    if not (
        isinstance(a, list)
        and isinstance(w, list)
        and len(a) == len(w) == 2
        and all(type(axis) is int and axis > 0 for axis in (*a, *w))
        and a[1] == w[0]
    ):
        return None
    a_dtype, w_dtype = _dtype(lhs.get("dtype")), _dtype(weight.get("dtype"))
    if a_dtype is None or a_dtype != w_dtype:
        return None
    output_dtype = _dtype(attributes.get("output_dtype"))
    if output_dtype is None:
        return None
    return (a[0], a[1], w[1]), a_dtype, output_dtype


def match_operation_form(source: Mapping, descriptor: Mapping) -> dict[str, str]:
    """Classify relevance without promoting geometric resemblance to coverage.

    ``exact_arithmetic_form`` is presently defined only for a proven captured
    signed integer matmul and its standalone, epilogue-free capsule. A strict
    body/ABI match without weight-origin evidence is a separate structural
    candidate. New exact forms require their own semantic recognizer; equal
    geometry alone is never a coverage claim.
    """
    if source.get("independent_compute_demand") is not True:
        return {"status": "not_independent", "reason": "source row is not an independent compute demand"}
    if source.get("semantic_family") != "contraction":
        return {"status": "not_comparable", "reason": "capsule is not a proven form of this source family"}
    capsule = _capsule_matmul(descriptor)
    if capsule is None:
        return {"status": "not_comparable", "reason": "capsule has no validated rank-2 matmul ABI"}
    geometry = source.get("contraction_shape")
    if not isinstance(geometry, Mapping) or any(type(geometry.get(axis)) is not int for axis in ("M", "K", "N")):
        return {"status": "unknown_shape", "reason": "source has no exact static M/K/N"}
    dimensions, input_dtype, output_dtype = capsule
    if dimensions != tuple(geometry[axis] for axis in ("M", "K", "N")):
        return {"status": "off_shape", "reason": "capsule M/K/N differs from the captured operation"}
    source_dtype = _dtype(source.get("operand_format"))
    if source_dtype is None or source_dtype != input_dtype:
        return {"status": "dtype_mismatch", "reason": "capsule input dtype differs from the captured operation"}
    results = source.get("ordered_result_types")
    source_output = (
        _dtype(results[0].get("dtype"))
        if isinstance(results, list) and len(results) == 1 and isinstance(results[0], Mapping)
        else None
    )
    if source_output is None or source_output != output_dtype:
        return {"status": "dtype_mismatch", "reason": "capsule output dtype differs from the captured operation"}
    structural_int_mm = (
        source.get("shape_confidence") == "observed_iteration_space"
        and exact_int_mm_geometry(dict(source), require_quant_origin=False) == dimensions
        and descriptor["operation"]["op"] == "matmul"
        and not (descriptor["operation"].get("attributes") or {}).get("epilogue")
        and output_dtype == "i32"
    )
    if (
        structural_int_mm
        and source.get("source_integerization_byte_bound") is True
        and exact_int_mm_geometry(dict(source)) == dimensions
    ):
        return {
            "status": "exact_arithmetic_form",
            "reason": "strict i8 matmul ABI and body with byte-bound source integerization; no epilogue",
        }
    if structural_int_mm:
        return {
            "status": "structural_arithmetic_candidate",
            "reason": "strict i8 matmul ABI and body; source quantization proof is incomplete",
        }
    return {
        "status": "shape_dtype_candidate",
        "reason": "matching M/K/N and input dtype do not prove body, layout, quantization or epilogue equivalence",
    }
