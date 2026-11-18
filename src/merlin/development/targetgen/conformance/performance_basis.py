"""Static, capture-bound Phase 0 performance inputs, without timing claims.

Counts here describe selected normalized operations. They are not executions,
traffic measurements, achievable throughput, or full-network estimates.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from merlin.common.digest import is_sha256
from merlin.development.targetgen.applications.inventory import exact_int_mm_geometry

# ---- performance_basis -------------------------------------------------------------------------

SCHEMA = "merlin.phase0.performance_basis.v1"
_NON_DEMAND = frozenset({"structural", "component", "support_required"})
_ELEMENT_BYTES = {
    **{f"i{bits}": bits // 8 for bits in (8, 16, 32, 64)},
    **{f"ui{bits}": bits // 8 for bits in (8, 16, 32, 64)},
    **{f"f{bits}": bits // 8 for bits in (8, 16, 32, 64)},
    "bf16": 2,
}


def _tensor_bytes(types: object) -> tuple[int | None, str | None]:
    if not isinstance(types, list):
        return None, "ordered_tensor_types_absent"
    total = 0
    for tensor in types:
        if not isinstance(tensor, dict):
            return None, "tensor_type_not_a_mapping"
        shape, dtype = tensor.get("shape"), tensor.get("dtype")
        if not isinstance(shape, list) or any(type(extent) is not int or extent < 0 for extent in shape):
            return None, "tensor_shape_not_static"
        width = _ELEMENT_BYTES.get(dtype) if isinstance(dtype, str) else None
        if width is None:
            return None, "tensor_storage_width_unknown"
        elements = 1
        for extent in shape:
            elements *= extent
        total += elements * width
    return total, None


def _macs(row: dict) -> tuple[int | None, str | None]:
    if row.get("semantic_family") != "contraction":
        return None, "semantic_form_has_no_defined_mac_count"
    shape = row.get("contraction_shape")
    if row.get("shape_confidence") != "observed_iteration_space" or not isinstance(shape, dict):
        return None, "contraction_iteration_space_not_observed"
    dimensions = [shape.get(axis) for axis in ("M", "K", "N")]
    if any(type(extent) is not int or extent <= 0 for extent in dimensions):
        return None, "contraction_dimensions_not_static_positive_integers"
    operation = row.get("mlir_operation")
    if operation == "linalg.matmul":
        return dimensions[0] * dimensions[1] * dimensions[2], None
    if operation == "linalg.generic":
        from merlin.development.targetgen.applications.inventory import exact_int_mm_geometry

        # Static arithmetic mass follows the typed indexing/body form. The
        # separate source-quantization proof still requires quant origin.
        geometry = exact_int_mm_geometry(row, require_quant_origin=False)
        if geometry is not None and list(geometry) == dimensions:
            return dimensions[0] * dimensions[1] * dimensions[2], None
    return None, "contraction_body_not_a_recognized_single_mac_form"


def build_performance_basis(
    accounting: dict,
    performance_facts: dict,
    *,
    target: str,
    operation_accounting_sha256: str,
    performance_facts_artifact_sha256: str,
    selected_hardware_spec_sha256: str | None,
    raw_rtl_facts_sha256: str | None,
) -> dict:
    """Project exact selected rows and byte-bound inputs into diagnostic mass.

    ``*_artifact_sha256`` values identify the frozen JSON bytes, while the
    accounting document retains canonical hashes of its selected source views.
    """
    if accounting.get("schema") != "merlin.phase0.operation_accounting.v1":
        raise ValueError("performance basis requires Phase 0 operation accounting")
    if performance_facts.get("target") != target or not is_sha256(performance_facts.get("sha256")):
        raise ValueError("performance basis requires selected target performance facts")
    if not all(is_sha256(value) for value in (operation_accounting_sha256, performance_facts_artifact_sha256)):
        raise ValueError("performance basis requires artifact SHA256 identities")
    if selected_hardware_spec_sha256 is not None and not is_sha256(selected_hardware_spec_sha256):
        raise ValueError("performance basis hardware spec identity is malformed")
    if raw_rtl_facts_sha256 is not None and not is_sha256(raw_rtl_facts_sha256):
        raise ValueError("performance basis RTL facts identity is malformed")
    applications = {}
    overall = {
        "n_operations": 0,
        "independent_compute_demands": 0,
        "known_macs": 0,
        "unknown_macs_demands": 0,
        "known_static_tensor_bytes": 0,
        "unknown_static_tensor_bytes_demands": 0,
    }
    for label, app in sorted(accounting.get("applications", {}).items()):
        rows = []
        for entry in app["signatures"]:
            observed = entry["observed_signature"]
            count = entry["count"]
            independent = observed["disposition"] not in _NON_DEMAND
            placements = sorted(
                {
                    decision["placement"]
                    for decision in entry.get("software_admissions", [])
                    if decision.get("status") == "admitted" and isinstance(decision.get("placement"), str)
                }
            )
            result = {
                "application": label,
                "capture_sha256": app["capture_sha256"],
                "operation": entry["operation"],
                "mlir_operation": observed["mlir_operation"],
                "frontend_op": observed.get("frontend_op"),
                "provenance_op": observed.get("provenance_op"),
                "callee": observed.get("callee"),
                "count": count,
                "ordinals": copy.deepcopy(entry["ordinals"]),
                "semantic_family": observed.get("semantic_family"),
                "family_basis": observed.get("family_basis"),
                "contraction_shape": copy.deepcopy(observed.get("contraction_shape")),
                "shape_confidence": observed.get("shape_confidence"),
                "ordered_operand_types": copy.deepcopy(observed.get("ordered_operand_types")),
                "ordered_result_types": copy.deepcopy(observed.get("ordered_result_types")),
                "result_shapes": copy.deepcopy(observed.get("result_shapes")),
                "operand_dtypes": copy.deepcopy(observed.get("operand_dtypes")),
                "result_dtypes": copy.deepcopy(observed.get("result_dtypes")),
                "accumulator_dtypes": copy.deepcopy(observed.get("accumulator_dtypes")),
                "operand_format": observed.get("operand_format"),
                "indexing_maps": copy.deepcopy(observed.get("indexing_maps")),
                "iterator_types": copy.deepcopy(observed.get("iterator_types")),
                "body_operations": copy.deepcopy(observed.get("body_operations")),
                "layout_evidence": copy.deepcopy(observed.get("layout_evidence")),
                "quant_evidence": copy.deepcopy(observed.get("quant_evidence")),
                "weight_buffer_evidence": copy.deepcopy(observed.get("weight_buffer_evidence")),
                "source_weight_identity": copy.deepcopy(observed.get("source_weight_identity")),
                "weight_transform_gap": copy.deepcopy(observed.get("weight_transform_gap")),
                "disposition": observed["disposition"],
                "classification": entry["classification"],
                "placement": placements[0] if len(placements) == 1 else None,
                "placement_candidates": placements,
                "placement_reason": (
                    None
                    if len(placements) == 1
                    else "no_admitted_software_placement"
                    if not placements
                    else "multiple_admitted_software_placements"
                ),
                "support_partition": entry["support_partition"],
                "independent_compute_demand": independent,
            }
            if not independent:
                mac_count, mac_reason = None, "not_independent_compute_demand"
                operands, results, byte_reason = None, None, "not_independent_compute_demand"
            else:
                mac_count, mac_reason = _macs(observed)
                operands, operand_reason = _tensor_bytes(observed.get("ordered_operand_types"))
                results, result_reason = _tensor_bytes(observed.get("ordered_result_types"))
                byte_reason = operand_reason or result_reason
            byte_count = operands + results if byte_reason is None else None
            result["macs"] = {
                "per_occurrence": mac_count,
                "total": mac_count * count if mac_count is not None else None,
                "reason": mac_reason,
            }
            result["static_tensor_bytes"] = {
                "operands": operands,
                "results": results,
                "per_occurrence": byte_count,
                "total": byte_count * count if byte_count is not None else None,
                "reason": byte_reason,
            }
            overall["n_operations"] += count
            if independent:
                overall["independent_compute_demands"] += count
                if mac_count is not None:
                    overall["known_macs"] += mac_count * count
                elif observed.get("semantic_family") == "contraction":
                    overall["unknown_macs_demands"] += count
                if byte_count is not None:
                    overall["known_static_tensor_bytes"] += byte_count * count
                else:
                    overall["unknown_static_tensor_bytes_demands"] += count
            rows.append(result)
        if sum(row["count"] for row in rows) != app["n_mlir_operations"]:
            raise ValueError(f"performance basis operation count differs for {label}")
        applications[label] = {
            "capture_sha256": app["capture_sha256"],
            "capture_receipt": copy.deepcopy(app.get("capture_receipt")),
            "capture_quantization": copy.deepcopy(app.get("capture_quantization")),
            "capture_integerization": copy.deepcopy(app.get("capture_integerization")),
            "n_operations": app["n_mlir_operations"],
            "n_signatures": len(rows),
            "rows": rows,
        }
    expected = accounting.get("overall", {}).get("n_mlir_operations")
    if expected is not None and overall["n_operations"] != expected:
        raise ValueError("performance basis overall count differs from operation accounting")
    if accounting["status"] == "not_available":
        overall = {key: None for key in overall}
        overall["unknown_reason"] = "no_selected_detailed_application_inventory"
    return {
        "schema": SCHEMA,
        "target": target,
        "status": accounting["status"],
        "selected_inventory": copy.deepcopy(accounting.get("selected_inventory")),
        "sources": {
            "operation_accounting_sha256": operation_accounting_sha256,
            "performance_facts_artifact_sha256": performance_facts_artifact_sha256,
            "performance_facts_sha256": performance_facts["sha256"],
            "selected_inventory_sha256": accounting.get("inventory_sha256"),
            "selected_software_spec_sha256": accounting.get("selected_software_spec_sha256"),
            "selected_capability_contract_sha256": accounting.get("selected_capability_contract_sha256"),
            "selected_hardware_spec_sha256": selected_hardware_spec_sha256,
            "raw_rtl_facts_sha256": raw_rtl_facts_sha256,
        },
        "applications": applications,
        "overall": overall,
        "qualification": (
            "static selected-capture diagnostic only; tensor bytes count ABI tensor extents, "
            "not distinct storage or measured traffic; no timing, throughput or speedup claim"
        ),
    }


# ---- performance_match -------------------------------------------------------------------------
# Conservative source-to-performance-capsule form matching.
#
# An equal M/K/N is useful for selecting a diagnostic, but it is not proof that
# the capsule executes the captured operation.  In particular, a linalg.generic
# contraction can encode a different body, layout, or quantization.  Only a
# source recognizer which checks those details may return an exact arithmetic
# form here.  Even that result says nothing about epilogue chains, compiler
# placement, or target timing.


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
