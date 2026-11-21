"""Host-owned Phase 0 writer implementation."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from merlin.development.targetgen.corpus import binding as CS  # noqa: E402
from merlin.development.targetgen.corpus.materialization import write_description
from merlin.development.targetgen.grading import golden as CG  # noqa: E402
from merlin.development.targetgen.grading import golden_inputs as NF  # noqa: E402
from merlin.development.targetgen.grading import golden_store as GS  # noqa: E402
from merlin.development.testgen.generation.capsule_writer import _golden_cached
from merlin.development.testgen.generation.goldens import (
    _float_golden,
    _simt_golden,
    float_semantics,
    specir_oracle_source_identity,
)
from merlin.development.testgen.generation.microscaling_goldens import (
    _mx_attention_golden,
    _mx_gemv_batched_golden,
    _mx_golden,
)
from merlin.development.testgen.sealed_capture.generation import capture_source

# ---- writer ------------------------------------------------------------------------------------


def _integer_reference_bound(entry: dict, cap: dict) -> dict:
    """Screen concrete integer contraction stimuli against selected internal width."""
    from merlin.development.testgen.generation.capsule_writer import INTEGER_CONTRACTION_BOUND_OPS
    from merlin.target.capabilities.semantic_families import integer_partial_sum_bound

    semantics = entry.get("numerical_semantics") or {}
    operation = cap["operation"]
    op, attrs = operation["op"], operation.get("attributes") or {}
    if op == "quantized_producer_sum":
        producer = attrs.get("producer") or {}
        if producer.get("op") not in ("matmul", "conv2d"):
            raise ValueError("quantized pair lacks its original source contraction")
        return _integer_reference_bound(entry, {**cap, "operation": producer})
    if op == "spatial_producer_pool":
        producer = attrs.get("producer") or {}
        if producer.get("op") != "conv2d":
            raise ValueError("spatial pair lacks its original source convolution")
        return _integer_reference_bound(entry, {**cap, "operation": producer})
    if op == "component_program":
        from merlin.development.testgen.components.component_execution_budget import source_for_capsule
        from merlin.development.testgen.components.component_integer_bounds import derive

        return derive(source_for_capsule(cap), semantics)
    policy = (semantics.get("internal_arithmetic") or {}).get("full_operation_overflow_policy")
    if policy != "bounded_exact_requires_each_partial_sum":
        if op == "host_island_seam":
            raise ValueError("host-island integer contractions require a selected internal-width bound policy")
        return {"status": "unknown", "reason": "no selected full-operation internal-width bound policy"}
    if op not in INTEGER_CONTRACTION_BOUND_OPS:
        return {"status": "not_applicable", "reason": "this writer path has no modeled integer contraction"}
    if op == "scope_chain":
        families = attrs.get("scope_families")
        if (
            not isinstance(families, list)
            or len(families) < 3
            or families[:2] != ["movement", "contraction"]
            or any(family != "elementwise_map" for family in families[2:])
        ):
            raise ValueError("integer scope chain needs one selected contraction and only post-contraction maps")
    leaves = CG.materialize_capsule_leaves(cap)

    def bounded_matmul(lhs_name: str, rhs_name: str, *, initial_values=()) -> dict:
        if lhs_name not in leaves or rhs_name not in leaves:
            raise ValueError("integer contraction bound requires concrete lhs and weight operands")
        lhs, rhs = leaves[lhs_name], leaves[rhs_name]
        if len(lhs.shape) != 2 or len(rhs.shape) != 2 or lhs.shape[1] != rhs.shape[0]:
            raise ValueError("integer matmul bound requires matching rank-2 reduction extents")
        result = integer_partial_sum_bound(
            semantics,
            reduction_extent=lhs.shape[1],
            lhs_values=[int(v) for v in lhs.data],
            rhs_values=[int(v) for v in rhs.data],
            initial_values=initial_values,
        )
        if result["status"] == "may_overflow":
            from merlin.target.capabilities.semantic_families import integer_matrix_partial_sum_bound

            result = integer_matrix_partial_sum_bound(
                semantics,
                lhs_shape=lhs.shape,
                rhs_shape=rhs.shape,
                lhs_values=lhs.data,
                rhs_values=rhs.data,
                initial_values=initial_values,
            )
        if result["status"] != "proven_safe":
            raise ValueError(f"integer mathematical golden cannot qualify selected internal MAC width: {result}")
        return result

    if op == "resident_reuse":
        weight_name, matmuls = attrs.get("weight"), attrs.get("matmuls")
        if not isinstance(weight_name, str) or not isinstance(matmuls, list) or not matmuls:
            raise ValueError("resident reuse bound requires a weight and nonempty matmul roster")
        members = []
        for member in matmuls:
            if (
                not isinstance(member, dict)
                or not isinstance(member.get("lhs"), str)
                or not isinstance(member.get("out"), str)
            ):
                raise ValueError("resident reuse bound has a malformed matmul member")
            members.append(
                {
                    "lhs": member["lhs"],
                    "out": member["out"],
                    "partial_sum_bound": bounded_matmul(member["lhs"], weight_name),
                }
            )
        bounds = [member["partial_sum_bound"] for member in members]
        return {
            "status": "proven_safe",
            "scope": "each listed resident-weight contraction; not full-kernel execution",
            "bound": max(bound["bound"] for bound in bounds),
            "signed_positive_limit": bounds[0]["signed_positive_limit"],
            "mac_result_bits": bounds[0]["mac_result_bits"],
            "members": members,
        }

    if op == "host_island_seam":
        from merlin.runtime.engine.command_buffer import Tensor

        role, transform = attrs.get("comparison_role"), attrs.get("host_transform")
        if (
            role not in {"island", "no_island"}
            or {"island": "xor_low_bit", "no_island": "none"}[role] != transform
            or attrs.get("accelerator_contractions") != 2
            or attrs.get("shared_accelerator_epilogue") != "saturating_i32_to_i8"
            or (semantics.get("internal_arithmetic") or {}).get("signed_operand_bits") != 8
        ):
            raise ValueError("host-island bound requires the declared two-contraction i8 seam")
        lhs_name, first_weight_name, second_weight_name = (attrs.get("lhs"), attrs.get("weight0"), attrs.get("weight1"))
        if (
            not all(isinstance(name, str) for name in (lhs_name, first_weight_name, second_weight_name))
            or set(leaves) != {lhs_name, first_weight_name, second_weight_name}
            or len(leaves) != 3
        ):
            raise ValueError("host-island bound requires exactly its three concrete input tensors")
        lhs, first_weight, second_weight = (leaves[lhs_name], leaves[first_weight_name], leaves[second_weight_name])
        dimensions = tuple(attrs.get(key) for key in ("M", "K", "H", "N"))
        if (
            any(type(value) is not int or value < 1 for value in dimensions)
            or lhs.shape != dimensions[:2]
            or first_weight.shape != dimensions[1:3]
            or second_weight.shape != dimensions[2:]
            or any(tensor.dtype != "i8" for tensor in (lhs, first_weight, second_weight))
        ):
            raise ValueError("host-island bound operand ABI differs from its declared contractions")
        first = bounded_matmul(lhs_name, first_weight_name)
        # The first sum is proven inside the selected MAC width above. Its
        # saturating narrow and optional bitwise map therefore give the exact
        # signed i8 operand bytes consumed by the second contraction.
        middle = lhs.matmul(first_weight).to_i8()
        if transform == "xor_low_bit":
            mask = attrs.get("xor_mask")
            if type(mask) is not int or not 0 < mask < 128:
                raise ValueError("host-island bound has an invalid low-bit XOR mask")
            middle = Tensor(middle.shape, [value ^ mask for value in middle.data], "i8")
        second = integer_partial_sum_bound(
            semantics,
            reduction_extent=dimensions[2],
            lhs_values=middle.data,
            rhs_values=second_weight.data,
        )
        if second["status"] != "proven_safe":
            raise ValueError(f"integer mathematical golden cannot qualify selected internal MAC width: {second}")
        return {
            "status": "proven_safe",
            "scope": (
                "both host-island contractions on the concrete input and derived middle tensor; "
                "not full-kernel execution"
            ),
            "bound": max(first["bound"], second["bound"]),
            "signed_positive_limit": first["signed_positive_limit"],
            "mac_result_bits": first["mac_result_bits"],
            "members": [
                {"region": "contraction_0", "partial_sum_bound": first},
                {"region": "contraction_1", "partial_sum_bound": second},
            ],
        }

    if op == "sibling_feed":
        from merlin.development.targetgen.corpus.corpus_sibling_feed import merged_operand

        members = [
            {"region": region, "partial_sum_bound": bounded_matmul(attrs["lhs"], attrs[weight])}
            for region, weight in (("gate", "gate_weight"), ("up", "up_weight"))
        ]
        merged = merged_operand(cap, leaves, apply_epilogue=CG._apply_epilogue, narrow=CG._narrow_to_dtype)
        down = leaves[attrs["down_weight"]]
        if merged.shape[1] != down.shape[0]:
            raise ValueError("sibling-feed down projection reduction extent differs from the merged operand")
        result = integer_partial_sum_bound(
            semantics,
            reduction_extent=down.shape[0],
            lhs_values=[int(v) for v in merged.data],
            rhs_values=[int(v) for v in down.data],
        )
        if result["status"] == "may_overflow":
            from merlin.target.capabilities.semantic_families import integer_matrix_partial_sum_bound

            result = integer_matrix_partial_sum_bound(
                semantics,
                lhs_shape=merged.shape,
                rhs_shape=down.shape,
                lhs_values=merged.data,
                rhs_values=down.data,
            )
        if result["status"] != "proven_safe":
            raise ValueError(f"integer mathematical golden cannot qualify selected internal MAC width: {result}")
        members.append({"region": "down", "partial_sum_bound": result})
        bounds = [row["partial_sum_bound"] for row in members]
        return {
            "status": "proven_safe",
            "scope": "both sibling contractions and the down projection on the concrete inputs and the derived "
            "merged operand; not full-kernel execution",
            "bound": max(bound["bound"] for bound in bounds),
            "signed_positive_limit": bounds[0]["signed_positive_limit"],
            "mac_result_bits": bounds[0]["mac_result_bits"],
            "members": members,
        }

    def name(role, declared):
        return attrs.get(declared) or next((row["name"] for row in cap["inputs"] if row.get("role") == role), None)

    if op == "attention_qk":
        lhs_name, rhs_name = attrs.get("q"), attrs.get("k")
    else:
        lhs_name, rhs_name = name("input", "ifm" if op == "conv2d" else "lhs"), name("weight", "weight")
    if lhs_name not in leaves or rhs_name not in leaves:
        raise ValueError("integer contraction bound requires concrete lhs and weight operands")
    lhs, rhs = leaves[lhs_name], leaves[rhs_name]
    if op in {"matmul", "linear", "matmul_bias", "fused_matmul_bias"}:
        initial = [
            int(value) for key, tensor in leaves.items() if key not in {lhs_name, rhs_name} for value in tensor.data
        ]
        return bounded_matmul(lhs_name, rhs_name, initial_values=initial)
    if op == "attention_qk" and rhs.shape[-1] != lhs.shape[-1]:
        raise ValueError("attention score reduction extents differ between query and key")
    if op == "scope_chain" and rhs.shape[-1] != lhs.shape[-1]:
        raise ValueError("integer scope chain reduction extents differ between lhs and transposed weight")
    reduction_extent = rhs.shape[0] if op == "conv2d" else lhs.shape[-1]
    initial = [int(value) for key, tensor in leaves.items() if key not in {lhs_name, rhs_name} for value in tensor.data]
    result = integer_partial_sum_bound(
        semantics,
        reduction_extent=reduction_extent,
        lhs_values=[int(v) for v in lhs.data],
        rhs_values=[int(v) for v in rhs.data],
        initial_values=initial,
    )
    if result["status"] == "may_overflow" and op == "conv2d":
        from merlin.target.capabilities.semantic_families import integer_matrix_partial_sum_bound

        patches = CG.im2col(
            lhs,
            int(attrs["ci"]),
            int(attrs["kh"]),
            int(attrs["kw"]),
            stride=attrs["stride"],
            padding=attrs["padding"],
            dilation=attrs.get("dilation", [1, 1]),
            layout=attrs.get("layout", "nhwc"),
        )
        result = integer_matrix_partial_sum_bound(
            semantics,
            lhs_shape=patches.shape,
            rhs_shape=rhs.shape,
            lhs_values=patches.data,
            rhs_values=rhs.data,
            initial_values=initial,
        )
    if result["status"] != "proven_safe":
        raise ValueError(f"integer mathematical golden cannot qualify selected internal MAC width: {result}")
    return result


def _source_integer_reference_bound(entry: dict, cap: dict, directory: Path) -> dict:
    """Bound exact source i8 matmul inputs, including the zero initial sum.

    A source-backed model or quantized PyTorch graph can contain contractions
    whose internal operands are not the capsule's external inputs. Isolated
    PyTorch integer matmul has captured bytes; spec matmul has program operands
    that must be checked alongside its separately materialized capsule inputs.
    """
    semantics = entry.get("numerical_semantics") or {}
    internal = semantics.get("internal_arithmetic") or {}
    if internal.get("full_operation_overflow_policy") != "bounded_exact_requires_each_partial_sum":
        raise ValueError("source-backed integer contraction lacks a selected internal-width bound policy")
    if entry.get("source") == "spec" or entry.get("spec_ref"):
        return _spec_integer_reference_bound(entry, cap, directory)
    operation = cap.get("operation") or {}
    attrs = operation.get("attributes") or {}
    if (
        entry.get("kind") == "model"
        or entry.get("op") == "model"
        or not (entry.get("source") == "pytorch" or entry.get("pytorch_ref"))
        or entry.get("capture_op") != "int_matmul"
        or operation.get("op") != "matmul"
        or cap.get("kind") == "model"
        or (cap.get("numeric_policy") or {}) != {"compare": "exact_int", "dtype": "i32"}
        or semantics.get("operand_dtype") not in {"int8", "i8"}
        or semantics.get("accumulator_dtype") != "i32"
        or attrs.get("epilogue") not in ([], ())
        or len(cap.get("inputs") or []) != 2
        or not CG.is_exact_pytorch_integer_source(cap)
    ):
        raise ValueError("source-backed integer contraction lacks a verified isolated i8 matmul and exact inputs")
    scoped = {**cap, "__dir__": str(directory)}
    # This checks the saved host output against an independent recomputation
    # from the captured bytes. It also refuses missing/mismatched input bytes.
    CG.golden(scoped, directory)
    bound = _integer_reference_bound(entry, scoped)
    if bound["status"] != "proven_safe":
        raise ValueError("source-backed integer contraction has no proven internal partial-sum bound")
    return bound


def _spec_integer_reference_bound(entry: dict, cap: dict, directory: Path) -> dict:
    """Bound both the spec program and the separately materialized capsule operands."""
    from merlin.target.capabilities.semantic_families import integer_partial_sum_bound

    operation = cap.get("operation") or {}
    attrs = operation.get("attributes") or {}
    lhs_name, rhs_name, out_name = (attrs.get(key) for key in ("lhs", "weight", "out"))
    inputs = cap.get("inputs") or []
    by_name = {row.get("name"): row for row in inputs if isinstance(row, dict)}
    if (
        entry.get("kind") == "model"
        or entry.get("op") != "matmul"
        or operation.get("op") != "matmul"
        or cap.get("kind") == "model"
        or not isinstance(entry.get("spec_ref"), str)
        or cap.get("spec_ref") != entry["spec_ref"]
        or (cap.get("numeric_policy") or {}) != {"compare": "exact_int", "dtype": "i32"}
        or attrs.get("epilogue") not in ([], ())
        or not all(isinstance(name, str) for name in (lhs_name, rhs_name, out_name))
        or len(inputs) != 2
        or set(by_name) != {lhs_name, rhs_name}
        or (entry.get("numerical_semantics") or {}).get("operand_dtype") not in {"int8", "i8"}
        or (entry.get("numerical_semantics") or {}).get("accumulator_dtype") != "i32"
    ):
        raise ValueError("spec-backed integer contraction lacks an isolated exact i8 matmul")
    golden = GS.load_golden(directory) or {}
    if not isinstance(golden, dict):
        raise ValueError("spec-backed integer contraction lacks an independent golden document")
    provenance = golden.get("oracle_provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("spec-backed integer contraction lacks exact program operand provenance")
    saved_inputs = provenance.get("inputs") or {}
    if (
        golden.get("golden_source") != f"specir_program_{entry['spec_ref'].partition(':')[0]}"
        or provenance.get("spec_ref") != entry["spec_ref"]
        or not isinstance(saved_inputs, dict)
        or set(saved_inputs) != {lhs_name, rhs_name}
    ):
        raise ValueError("spec-backed integer contraction lacks exact program operand provenance")

    def matrix(name: str, role: str) -> list[list[int]]:
        spec, saved = by_name[name], saved_inputs[name]
        shape = spec.get("shape")
        rows = saved.get("decoded") if isinstance(saved, dict) else None
        if (
            spec.get("role") != role
            or spec.get("dtype") != "i8"
            or not isinstance(saved, dict)
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dim) is not int or dim < 1 for dim in shape)
            or saved.get("shape") != shape
            or not isinstance(rows, list)
            or len(rows) != shape[0]
            or any(
                not isinstance(row, list)
                or len(row) != shape[1]
                or any(type(value) is not int or not -128 <= value <= 127 for value in row)
                for row in rows
            )
        ):
            raise ValueError("spec-backed integer contraction has incomplete exact i8 operand values")
        return rows

    lhs, rhs = matrix(lhs_name, "input"), matrix(rhs_name, "weight")
    if len(lhs[0]) != len(rhs):
        raise ValueError("spec-backed integer matmul has mismatched reduction extents")
    recomputed = [
        [sum(lhs[m][k] * rhs[k][n] for k in range(len(rhs))) for n in range(len(rhs[0]))] for m in range(len(lhs))
    ]
    outputs = golden.get("outputs")
    observed = outputs.get(out_name) if isinstance(outputs, dict) else None
    if (
        not isinstance(observed, list)
        or any(not isinstance(row, list) or any(type(value) is not int for value in row) for row in observed)
        or observed != recomputed
    ):
        raise ValueError("spec-backed integer golden differs from exact operand recomputation")
    program_bound = integer_partial_sum_bound(
        entry["numerical_semantics"],
        reduction_extent=len(rhs),
        lhs_values=[value for row in lhs for value in row],
        rhs_values=[value for row in rhs for value in row],
    )
    if program_bound["status"] != "proven_safe":
        raise ValueError(f"spec-backed integer golden cannot qualify selected internal MAC width: {program_bound}")
    capsule_bound = _integer_reference_bound(entry, cap)
    if capsule_bound["status"] != "proven_safe":
        raise ValueError("spec-backed integer capsule inputs have no proven internal partial-sum bound")
    dominant = max((program_bound, capsule_bound), key=lambda bound: bound["bound"])
    return {
        **dominant,
        "scope": "spec-program and separately materialized integer-capsule matmul operands",
        "members": [
            {"operand_stream": "spec_program", "partial_sum_bound": program_bound},
            {"operand_stream": "capsule_materialized", "partial_sum_bound": capsule_bound},
        ],
    }
