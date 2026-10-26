"""Per-operation host support declarations beside an immutable compiler package.

A package precision lane is not an operation-support manifest. These declarations
are separately selected and digest-bound; no compiler payload is rewritten.
"""

from __future__ import annotations

import copy
import json
from merlin.frontends.linalg_integer_seams import SCHEMA as INTEGER_SEAM_SCHEMA, validate as validate_integer_seam, recognize as recognize_integer_seam
from collections.abc import Mapping
from merlin.frontends.linalg_quantization_seams import (
    SCHEMA as QUANTIZATION_SEAM_SCHEMA,
    validate as validate_quantization_seam,
    recognize as recognize_quantization_seam,
)

from merlin.common.digest import is_sha256
from merlin.frontends.connected_index_source_body import SCHEMA as CONNECTED_INDEX_SCHEMA, screen as screen_connected_index
from merlin.frontends.bounded_control_source_body import (
    SCHEMA as BOUNDED_CONTROL_SCHEMA, ROLES as BOUNDED_CONTROL_ROLES,
    validate as validate_bounded_control, screen as screen_bounded_control,
)
from merlin.frontends.linalg_embedding_gather import (
    SCHEMA as EMBEDDING_GATHER_SCHEMA,
)
from merlin.frontends.linalg_embedding_gather import (
    recognize as recognize_embedding_gather,
)
from merlin.frontends.linalg_embedding_gather import (
    validate_source_body as validate_embedding_gather_body,
)
from merlin.frontends.linalg_mixed_patterns import SCHEMA as MIXED_SCHEMA
from merlin.frontends.linalg_mixed_patterns import recognize as recognize_mixed
from merlin.frontends.linalg_mixed_patterns import validate_declaration as validate_mixed
from merlin.targetgen.host_linkage_contract import (
    required_composite_math_symbol,
    validate_linkage_contract,
    validate_source_linkage_contract,
)
from merlin.targetgen.software_spec import admit_operation, validate_quantization_parameters

SCHEMA = "merlin.host_capabilities.v1"


def validate_host_capabilities(
    document: dict, *, package_sha256: str | None = None, dtype_strategy: str | None = None
) -> dict:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"host capability spec must declare schema {SCHEMA}")
    if document.get("status") not in {"reviewed", "unreviewed"}:
        raise ValueError("host capability spec must declare review status")
    compiler = document.get("compiler")
    if not isinstance(compiler, dict) or (
        not is_sha256(compiler.get("package_sha256"))
        and not (document["status"] == "unreviewed" and compiler.get("package_sha256") is None)
    ):
        raise ValueError("host capability spec requires an exact compiler package SHA256")
    if not isinstance(compiler.get("dtype_strategy"), str) or not compiler["dtype_strategy"]:
        raise ValueError("host capability spec requires an explicit precision lane")
    if (
        package_sha256 is not None
        and compiler["package_sha256"] is not None
        and compiler["package_sha256"] != package_sha256
    ):
        raise ValueError("host capability spec is bound to a different compiler package")
    if dtype_strategy is not None and compiler["dtype_strategy"] != dtype_strategy:
        raise ValueError("host capability spec is bound to a different precision lane")
    operations = document.get("operations")
    if not isinstance(operations, list):
        raise ValueError("host capability spec operations must be a list")
    seen = set()
    for row in operations:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("host capability operations require unique IDs")
        seen.add(row["id"])
        if row.get("status", "reviewed") not in {"reviewed", "unreviewed"}:
            raise ValueError(f"host capability {row['id']} has an invalid review status")
        if row.get("placement") != "host" or not isinstance(row.get("signature"), dict) or not row["signature"]:
            raise ValueError("host capability operations require host placement and typed signature constraints")
        if "source_body" in row["signature"]:
            raise ValueError("source_body must be an explicit host declaration field, not a signature constraint")
        if "linkage_contract" in row:
            body = row.get("source_body")
            validate_source_linkage_contract(
                body.get("schema") if isinstance(body, dict) else None,
                body.get("operation") if isinstance(body, dict) else None,
                row["linkage_contract"],
            )
        if "source_body" in row:
            from merlin.frontends.bucketize_source import (
                STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA,
                validate_static_bucketize_source_body,
            )
            from merlin.frontends.linalg_boolean_patterns import (
                DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA,
                STATIC_BOOLEAN_SOURCE_BODY_SCHEMA,
                validate_dynamic_boolean_cast_source_body,
                validate_static_boolean_source_body,
            )
            from merlin.frontends.linalg_composite_math import (
                STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
                validate_static_composite_math_source_body,
            )
            from merlin.frontends.linalg_f32_maximum_patterns import (
                STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA,
                validate_static_f32_maximum_source_body,
            )
            from merlin.frontends.linalg_f32_reductions import (
                STATIC_F32_SUM_SOURCE_BODY_SCHEMA,
                validate_static_f32_sum_source_body,
            )
            from merlin.frontends.linalg_math_patterns import (
                STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
                validate_static_f32_math_source_body,
            )
            from merlin.frontends.linalg_patterns import (
                STATIC_PROJECTED_POINTWISE_BODY_SCHEMA,
                validate_static_pointwise_source_body,
                validate_static_projected_pointwise_source_body,
            )
            from merlin.frontends.linalg_reduction_source_body import (
                INTEGER_REDUCTION_TARGETS,
                STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA,
                validate_static_integer_reduction_source_body,
            )
            from merlin.frontends.prepared_index_source_body import (
                ROLES as INDEX_ROLES,
            )
            from merlin.frontends.prepared_index_source_body import (
                SCHEMA as PREPARED_INDEX_SOURCE_BODY_SCHEMA,
            )
            from merlin.frontends.prepared_index_source_body import (
                validate_declaration as validate_prepared_index_source_body,
            )

            allowed = {
                "id",
                "ops",
                "families",
                "family",
                "placement",
                "signature",
                "status",
                "numerical_contract",
                "evidence",
                "description",
                "source_body",
                "linkage_contract",
            }
            if set(row) - allowed or not (row.get("ops") or row.get("families") or row.get("family")):
                raise ValueError(f"host capability {row['id']} has unsupported fields or no selector")
            body = row["source_body"]
            if isinstance(body, dict) and body.get("schema") == BOUNDED_CONTROL_SCHEMA:
                validate_bounded_control(body)
                op, inputs, outputs = BOUNDED_CONTROL_ROLES[body["operation"]]
                if (row.get("ops") != [op] or "families" in row or "family" in row
                        or row["signature"] != {"ordered_operand_dtypes": inputs, "ordered_result_dtypes": outputs}
                        or row.get("numerical_contract", {}).get("status") != "unreviewed"):
                    raise ValueError("bounded control needs one exact scalar ABI and placement-only declaration")
            elif isinstance(body, dict) and body.get("schema") == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
                validate_static_boolean_source_body(body)
            elif isinstance(body, dict) and body.get("schema") == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA:
                validate_dynamic_boolean_cast_source_body(body)
            elif isinstance(body, dict) and body.get("schema") == STATIC_F32_MATH_SOURCE_BODY_SCHEMA:
                validate_static_f32_math_source_body(body)
            elif isinstance(body, dict) and body.get("schema") == STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA:
                validate_static_composite_math_source_body(body)
                if required_composite_math_symbol(body["schema"], body["operation"]) and "linkage_contract" not in row:
                    raise ValueError("composite math source requires an exact selected linkage contract")
            elif isinstance(body, dict) and body.get("schema") == EMBEDDING_GATHER_SCHEMA:
                validate_embedding_gather_body(body)
                if row.get("signature", {}).get("ordered_operand_dtypes") != ["i64", "bf16"] or row.get(
                    "signature", {}
                ).get("ordered_result_dtypes") != ["bf16"]:
                    raise ValueError("embedding gather declaration has a different exact typed signature")
            elif isinstance(body, dict) and body.get("schema") == STATIC_PROJECTED_POINTWISE_BODY_SCHEMA:
                validate_static_projected_pointwise_source_body(body)
            elif isinstance(body, dict) and body.get("schema") == STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA:
                validate_static_integer_reduction_source_body(body)
                matches = [
                    (frontend, mlir)
                    for frontend, (mlir, kind) in INTEGER_REDUCTION_TARGETS.items()
                    if kind == body["operation"]
                ]
                signature = row["signature"]
                operands = signature.get("ordered_operand_dtypes")
                results = signature.get("ordered_result_dtypes")
                expected_operands = (
                    ["i64", "i64", "i64"]
                    if body["operation"] == "i64_min_first_index"
                    else ["i64", "i64"]
                    if body["operation"] == "sum"
                    else None
                )
                if (
                    len(matches) != 1
                    or row.get("ops") != [matches[0][0]]
                    or row.get("families", ["reduction"]) != ["reduction"]
                    or row.get("family", "reduction") != "reduction"
                    or signature.get("family") != "reduction"
                    or (expected_operands is not None and operands != expected_operands)
                    or (expected_operands is None and operands not in (["i1", "i64"], ["i64", "i64"]))
                    or results != (["i64", "i64"] if body["operation"] == "i64_min_first_index" else ["i64"])
                ):
                    raise ValueError("integer-reduction source_body needs one exact frontend selector and typed ABI")
            elif isinstance(body, dict) and body.get("schema") in {PREPARED_INDEX_SOURCE_BODY_SCHEMA, CONNECTED_INDEX_SCHEMA}:
                validate_prepared_index_source_body(dict(body, schema=PREPARED_INDEX_SOURCE_BODY_SCHEMA))
                operands = row["signature"].get("ordered_operand_dtypes")
                results = row["signature"].get("ordered_result_dtypes")
                expected = INDEX_ROLES[body["operation"]]
                if (
                    row.get("ops") != ["aten.index.Tensor"]
                    or "families" in row
                    or "family" in row
                    or not isinstance(row["signature"].get("family"), str)
                    or not row["signature"]["family"]
                    or (expected[1] is not None and operands != list(expected[1]))
                    or (
                        expected[1] is None
                        and (
                            type(operands) is not list
                            or len(operands) < 2
                            or operands != ["i64"] * (len(operands) - 1) + ["i1"]
                        )
                    )
                    or results != list(expected[2])
                ):
                    raise ValueError("prepared-index source_body needs one exact frontend role and typed ABI")
            elif isinstance(body, dict) and body.get("schema") == STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA:
                validate_static_bucketize_source_body(body)
                if (
                    row.get("ops") != ["aten.bucketize.Tensor"]
                    or "families" in row
                    or "family" in row
                    or row["signature"]
                    != {
                        "family": "reduction",
                        "ordered_operand_dtypes": [
                            "bf16" if body["operation"] == "bf16_extended_f32_literal_boundary_count_i64" else "f32",
                            "f32",
                            "i64",
                        ],
                        "ordered_result_dtypes": ["i64"],
                    }
                    or not isinstance(row.get("numerical_contract"), dict)
                    or row["numerical_contract"].get("status") != "unreviewed"
                ):
                    raise ValueError("bucketize source_body needs one exact typed placement-only selector")
            elif isinstance(body, dict) and body.get("schema") == STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA:
                validate_static_f32_maximum_source_body(body)
                if (
                    row.get("ops") != ["aten.amax.default"]
                    or "families" in row
                    or "family" in row
                    or row["signature"]
                    != {
                        "family": "reduction",
                        "ordered_operand_dtypes": ["f32", "f32"],
                        "ordered_result_dtypes": ["f32"],
                    }
                    or not isinstance(row.get("numerical_contract"), dict)
                    or row["numerical_contract"].get("status") != "unreviewed"
                ):
                    raise ValueError("f32 maximum source_body needs one exact typed placement-only selector")
            elif isinstance(body, dict) and body.get("schema") == STATIC_F32_SUM_SOURCE_BODY_SCHEMA:
                validate_static_f32_sum_source_body(body)
                if (row.get("ops") != ["aten.sum.dim_IntList"] or "families" in row or "family" in row
                        or row["signature"] != {"family": "reduction",
                            "ordered_operand_dtypes": ["f32", "f32"], "ordered_result_dtypes": ["f32"]}
                        or row.get("numerical_contract", {}).get("status") != "unreviewed"):
                    raise ValueError("f32 sum source_body needs one exact typed placement-only selector")
            elif isinstance(body, dict) and body.get("schema") == MIXED_SCHEMA:
                validate_mixed(body)
                if required_composite_math_symbol(body["schema"], body["operation"]) and "linkage_contract" not in row:
                    raise ValueError("mixed math source requires its exact selected linkage contract")
                if row.get("numerical_contract", {}).get("status") != "unreviewed":
                    raise ValueError("mixed source-body routing declares no numerical qualification")
            elif isinstance(body, dict) and body.get("schema") == QUANTIZATION_SEAM_SCHEMA:
                validate_quantization_seam(body)
            elif isinstance(body, dict) and body.get("schema") == INTEGER_SEAM_SCHEMA:
                validate_integer_seam(body)
                if row.get("numerical_contract", {}).get("status") != "unreviewed":
                    raise ValueError("integer seam routing declares no numerical qualification")
            else:
                validate_static_pointwise_source_body(body)
        if "quantization_parameters" in row["signature"]:
            validate_quantization_parameters(
                row["signature"]["quantization_parameters"], source=f"host capability {row['id']!r}"
            )
        for selector in ("ops", "families"):
            if selector in row and (
                not isinstance(row[selector], list)
                or any(not isinstance(value, str) or not value for value in row[selector])
            ):
                raise ValueError(f"host capability {selector} must be explicit string lists")
    if not isinstance(document.get("evidence"), dict):
        raise ValueError("host capability spec must record evidence and unresolved obligations")
    return document


def _selected_index_bits(source_context: Mapping | None) -> int | None:
    """Read one caller-owned compiler observation, never inventing an index width."""
    from merlin.llvmlower.target_data_layout import selected_index_bits

    if not isinstance(source_context, Mapping) or set(source_context) != {"selected_index_observation"}:
        return None
    return selected_index_bits(source_context["selected_index_observation"])


def _screen_integer_reduction_source_body(
    declaration: dict, row: dict, signature: dict, source_operations: tuple | None, source_context: Mapping | None
) -> dict:
    """Bind every parsed source root to a closed reduction and selected-width premise."""
    from merlin.common import mlir_query as mq
    from merlin.frontends.linalg_extremum_patterns import recognize_static_i64_argmin
    from merlin.frontends.linalg_integer_reductions import recognize_static_integer_reduction
    from merlin.frontends.linalg_patterns import InvalidLinalgPattern
    from merlin.frontends.linalg_reduction_source_body import INTEGER_REDUCTION_TARGETS, serialized_reduction_pattern
    from merlin.targetgen.application_inventory import operation_structure

    width = _selected_index_bits(source_context)
    if width is None:
        return {"status": "unknown", "reason": "integer source_body needs a selected compiler index observation"}
    if source_operations is None or not source_operations:
        return {"status": "unknown", "reason": "source_body requires parsed source operations"}
    if (
        not isinstance(source_operations, tuple)
        or type(row.get("count")) is not int
        or row["count"] != len(source_operations)
        or len({id(op) for op in source_operations}) != len(source_operations)
    ):
        return {"status": "unsupported", "reason": "source_body occurrence roster differs from source row"}
    frontend = row.get("frontend_op")
    target = INTEGER_REDUCTION_TARGETS.get(frontend)
    if target is None or target != (row.get("mlir_operation"), declaration["source_body"]["operation"]):
        return {"status": "unsupported", "reason": "source_body frontend/root operation selector differs"}
    patterns = []
    for op in source_operations:
        if mq.op_name(op) != target[0] or mq.attr_str(op, "prov.aten") != frontend:
            return {"status": "unsupported", "reason": "source_body parsed operation identity differs"}
        try:
            pattern = (
                recognize_static_i64_argmin(op, index_bits=width)
                if target[1] == "i64_min_first_index"
                else recognize_static_integer_reduction(op, index_bits=width)
            )
        except InvalidLinalgPattern as exc:
            return {"status": "unsupported", "reason": f"source_body structural proof refused: {exc}"}
        if pattern.operation != target[1]:
            return {"status": "unsupported", "reason": "source_body reduction kind differs"}
        structure = operation_structure(op)
        inputs, outputs = structure["ordered_operand_types"], structure["ordered_result_types"]
        if (
            row.get("ordered_operand_types") != inputs
            or row.get("ordered_result_types") != outputs
            or signature.get("ordered_operand_dtypes") != [item["dtype"] for item in inputs]
            or signature.get("ordered_result_dtypes") != [item["dtype"] for item in outputs]
            or type(signature.get("rank")) is not int
            or signature["rank"] != len(pattern.output_shape)
            or not outputs
            or any(item["shape"] != list(pattern.output_shape) for item in outputs)
            or inputs[0]["shape"] != list(pattern.input_shape)
        ):
            return {"status": "unsupported", "reason": "source_body differs from observed typed tensor ABI"}
        patterns.append(serialized_reduction_pattern(pattern))
    return {
        "status": "admitted",
        "reason": "every parsed occurrence matches the selected-width closed integer source body",
        "proof": {
            "schema": declaration["source_body"]["schema"],
            "declaration": declaration["id"],
            "operation": target[1],
            "selected_index_observation": copy.deepcopy(dict(source_context["selected_index_observation"])),
            "patterns": patterns,
        },
    }


def _screen_f32_maximum_source_body(
    declaration: dict, row: dict, signature: dict, source_operations: tuple | None, source_context: Mapping | None
) -> dict:
    """Screen every exact prepared amax root against the selected-width source form."""
    from merlin.common import mlir_query as mq
    from merlin.frontends.linalg_f32_maximum_patterns import (
        recognize_static_f32_maximum,
        serialized_f32_maximum_pattern,
    )
    from merlin.frontends.linalg_patterns import InvalidLinalgPattern
    from merlin.targetgen.application_inventory import operation_structure

    width = _selected_index_bits(source_context)
    if width is None:
        return {"status": "unknown", "reason": "f32 maximum source_body needs a selected compiler index observation"}
    if source_operations is None or not source_operations:
        return {"status": "unknown", "reason": "source_body requires parsed source operations"}
    if (
        not isinstance(source_operations, tuple)
        or type(row.get("count")) is not int
        or row["count"] != len(source_operations)
        or row.get("mlir_operation") != "linalg.reduce"
        or row.get("frontend_op") != "aten.amax.default"
        or len({id(op) for op in source_operations}) != len(source_operations)
    ):
        return {"status": "unsupported", "reason": "f32 maximum source-body occurrence roster differs"}
    patterns = []
    for op in source_operations:
        if mq.op_name(op) != "linalg.reduce" or mq.attr_str(op, "prov.aten") != "aten.amax.default":
            return {"status": "unsupported", "reason": "f32 maximum parsed operation identity differs"}
        try:
            pattern = recognize_static_f32_maximum(op, index_bits=width)
        except InvalidLinalgPattern as exc:
            return {"status": "unsupported", "reason": f"f32 maximum structural proof refused: {exc}"}
        structure = operation_structure(op)
        inputs, outputs = structure["ordered_operand_types"], structure["ordered_result_types"]
        if (
            row.get("ordered_operand_types") != inputs
            or row.get("ordered_result_types") != outputs
            or signature.get("family") != "reduction"
            or signature.get("ordered_operand_dtypes") != ["f32", "f32"]
            or signature.get("ordered_result_dtypes") != ["f32"]
            or type(signature.get("rank")) is not int
            or signature["rank"] != len(pattern.output_shape)
            or inputs
            != [
                {"shape": list(pattern.input_shape), "dtype": "f32"},
                {"shape": list(pattern.output_shape), "dtype": "f32"},
            ]
            or outputs != [{"shape": list(pattern.output_shape), "dtype": "f32"}]
        ):
            return {"status": "unsupported", "reason": "f32 maximum source differs from observed typed tensor ABI"}
        patterns.append(serialized_f32_maximum_pattern(pattern))
    return {
        "status": "admitted",
        "reason": "every parsed occurrence matches the selected-width prepared f32 maximum body",
        "proof": {
            "schema": declaration["source_body"]["schema"],
            "declaration": declaration["id"],
            "operation": declaration["source_body"]["operation"],
            "selected_index_observation": copy.deepcopy(dict(source_context["selected_index_observation"])),
            "patterns": patterns,
        },
    }


def _screen_f32_sum_source_body(declaration, row, signature, source_operations, source_context):
    """Reprove every +zero-initialized addf reduction at the selected width."""
    from dataclasses import asdict

    from merlin.common import mlir_query as mq
    from merlin.frontends.linalg_f32_reductions import recognize_static_f32_sum
    from merlin.frontends.linalg_patterns import InvalidLinalgPattern
    from merlin.targetgen.application_inventory import operation_structure

    width = _selected_index_bits(source_context)
    if width is None:
        return {"status": "unknown", "reason": "f32 sum needs a selected compiler index observation"}
    if not source_operations:
        return {"status": "unknown", "reason": "f32 sum requires parsed source operations"}
    if (type(source_operations) is not tuple or type(row.get("count")) is not int
            or row["count"] != len(source_operations) or row.get("mlir_operation") != "linalg.reduce"
            or row.get("frontend_op") != "aten.sum.dim_IntList"
            or len({id(op) for op in source_operations}) != len(source_operations)):
        return {"status": "unsupported", "reason": "f32 sum source occurrence roster differs"}
    patterns = []
    for op in source_operations:
        if mq.attr_str(op, "prov.aten") != "aten.sum.dim_IntList":
            return {"status": "unsupported", "reason": "f32 sum source frontend identity differs"}
        try:
            pattern = recognize_static_f32_sum(op, index_bits=width)
        except InvalidLinalgPattern as error:
            return {"status": "unsupported", "reason": f"f32 sum structural proof refused: {error}"}
        structure = operation_structure(op)
        if (row.get("ordered_operand_types") != structure["ordered_operand_types"]
                or row.get("ordered_result_types") != structure["ordered_result_types"]
                or signature.get("family") != "reduction"
                or signature.get("ordered_operand_dtypes") != ["f32", "f32"]
                or signature.get("ordered_result_dtypes") != ["f32"]
                or signature.get("rank") != len(pattern.output_shape)):
            return {"status": "unsupported", "reason": "f32 sum source differs from observed typed ABI"}
        patterns.append(json.loads(json.dumps(asdict(pattern))))
    return {"status": "admitted", "reason": "every source root is a selected-width +zero addf reduction",
            "proof": {"schema": declaration["source_body"]["schema"], "declaration": declaration["id"],
                      "operation": declaration["source_body"]["operation"],
                      "patterns": patterns,
                      "selected_index_observation": copy.deepcopy(source_context["selected_index_observation"])}}


def _screen_source_body(
    declaration: dict, row: dict, signature: dict, source_operations: tuple | None, source_context: Mapping | None
) -> dict:
    """Match every supplied parsed occurrence; never treat an omitted source as evidence."""
    from dataclasses import asdict
    from merlin.frontends.linalg_f32_reductions import STATIC_F32_SUM_SOURCE_BODY_SCHEMA

    if declaration["source_body"]["schema"] == STATIC_F32_SUM_SOURCE_BODY_SCHEMA:
        return _screen_f32_sum_source_body(declaration, row, signature, source_operations, source_context)

    if declaration["source_body"]["schema"] == CONNECTED_INDEX_SCHEMA:
        return screen_connected_index(declaration, row, signature, source_operations, source_context)

    if declaration["source_body"]["schema"] == BOUNDED_CONTROL_SCHEMA:
        return screen_bounded_control(declaration, row, signature, source_operations, source_context)

    from merlin.common import mlir_query as mq
    from merlin.frontends.bucketize_source import (
        STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA,
        bucketize_pattern_fits_index,
        recognize_static_bucketize_body,
    )
    from merlin.frontends.linalg_boolean_patterns import (
        DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA,
        STATIC_BOOLEAN_SOURCE_BODY_SCHEMA,
        recognize_dynamic_boolean_cast_body,
        recognize_static_boolean_body,
    )
    from merlin.frontends.linalg_composite_math import (
        STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
        recognize_static_composite_math_body,
    )
    from merlin.frontends.linalg_f32_maximum_patterns import STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA
    from merlin.frontends.linalg_math_patterns import (
        STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
        recognize_static_f32_math_body,
    )
    from merlin.frontends.linalg_patterns import (
        STATIC_PROJECTED_POINTWISE_BODY_SCHEMA,
        InvalidLinalgPattern,
        recognize_static_pointwise,
        recognize_static_projected_pointwise,
    )
    from merlin.frontends.linalg_reduction_source_body import STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA
    from merlin.frontends.prepared_index_source_body import SCHEMA as PREPARED_INDEX_SOURCE_BODY_SCHEMA
    from merlin.frontends.prepared_index_source_body import screen_source_body as screen_prepared_index_source_body

    if declaration["source_body"]["schema"] == STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA:
        return _screen_integer_reduction_source_body(declaration, row, signature, source_operations, source_context)
    if declaration["source_body"]["schema"] == STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA:
        return _screen_f32_maximum_source_body(declaration, row, signature, source_operations, source_context)
    if declaration["source_body"]["schema"] == PREPARED_INDEX_SOURCE_BODY_SCHEMA:
        return screen_prepared_index_source_body(declaration, row, signature, source_operations, source_context)
    if declaration["source_body"]["schema"] == STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA:
        width = _selected_index_bits(source_context)
        if width is None:
            return {"status": "unknown", "reason": "bucketize source_body needs a selected compiler index observation"}
        if source_operations is None or not source_operations:
            return {"status": "unknown", "reason": "source_body requires parsed source operations"}
        if (
            not isinstance(source_operations, tuple)
            or type(row.get("count")) is not int
            or row["count"] != len(source_operations)
            or row.get("mlir_operation") != "linalg.generic"
            or row.get("frontend_op") != "aten.bucketize.Tensor"
            or len({id(op) for op in source_operations}) != len(source_operations)
            or any(mq.attr_str(op, "prov.aten") != "aten.bucketize.Tensor" for op in source_operations)
        ):
            return {"status": "unsupported", "reason": "bucketize source-body occurrence roster differs"}
        try:
            patterns = tuple(recognize_static_bucketize_body(op) for op in source_operations)
        except InvalidLinalgPattern as exc:
            return {"status": "unsupported", "reason": f"bucketize source-body structural proof refused: {exc}"}
        if any(not bucketize_pattern_fits_index(pattern, width) for pattern in patterns):
            return {"status": "unsupported", "reason": "bucketize source tensor exceeds selected signed index span"}
        if (
            any(signature.get("ordered_operand_dtypes") != list(pattern.ordered_types[:3]) for pattern in patterns)
            or any(pattern.operation != declaration["source_body"]["operation"] for pattern in patterns)
            or signature.get("ordered_result_dtypes") != ["i64"]
            or type(signature.get("rank")) is not int
            or any(signature["rank"] != len(pattern.shape) for pattern in patterns)
        ):
            return {"status": "unsupported", "reason": "bucketize source body differs from observed typed tensor ABI"}
        return {
            "status": "admitted",
            "reason": "every parsed occurrence matches the closed literal bucketize count",
            "proof": {
                "schema": STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA,
                "declaration": declaration["id"],
                "operation": declaration["source_body"]["operation"],
                "predicate": None,
                "patterns": [json.loads(json.dumps(asdict(pattern))) for pattern in patterns],
                "selected_index_observation": copy.deepcopy(dict(source_context["selected_index_observation"])),
            },
        }

    if source_operations is None or not source_operations:
        return {"status": "unknown", "reason": "source_body requires parsed source operations"}
    if (
        not isinstance(source_operations, tuple)
        or ("count" in row and (type(row["count"]) is not int or row["count"] != len(source_operations)))
        or row.get("mlir_operation") != "linalg.generic"
        or len({id(op) for op in source_operations}) != len(source_operations)
    ):
        return {"status": "unsupported", "reason": "source_body occurrence roster differs from source row"}
    try:
        body = declaration["source_body"]
        recognizer = {
            INTEGER_SEAM_SCHEMA: recognize_integer_seam,
            QUANTIZATION_SEAM_SCHEMA: recognize_quantization_seam,
            MIXED_SCHEMA: recognize_mixed,
            STATIC_BOOLEAN_SOURCE_BODY_SCHEMA: recognize_static_boolean_body,
            DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA: recognize_dynamic_boolean_cast_body,
            STATIC_F32_MATH_SOURCE_BODY_SCHEMA: recognize_static_f32_math_body,
            STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA: recognize_static_composite_math_body,
            STATIC_PROJECTED_POINTWISE_BODY_SCHEMA: recognize_static_projected_pointwise,
            EMBEDDING_GATHER_SCHEMA: recognize_embedding_gather,
        }.get(body["schema"], recognize_static_pointwise)
        patterns = tuple(recognizer(op) for op in source_operations)
    except InvalidLinalgPattern as exc:
        return {"status": "unsupported", "reason": f"source_body structural proof refused: {exc}"}
    if any(
        pattern.operation != body["operation"] or getattr(pattern, "predicate", None) != body.get("predicate")
        for pattern in patterns
    ):
        return {"status": "unsupported", "reason": "source_body operation or predicate differs"}
    operands, results, rank = (
        signature.get("ordered_operand_dtypes"),
        signature.get("ordered_result_dtypes"),
        signature.get("rank"),
    )
    if not isinstance(operands, list) or not isinstance(results, list) or type(rank) is not int:
        return {"status": "unknown", "reason": "source_body has no complete observed typed signature"}
    if any(
        operands != list(pattern.ordered_types[:-1])
        or results != [pattern.ordered_types[-1]]
        or rank != len(pattern.shape)
        for pattern in patterns
    ):
        return {"status": "unsupported", "reason": "source_body differs from observed typed signature"}
    return {
        "status": "admitted",
        "reason": "every parsed occurrence matches the declared static source body",
        "proof": {
            "schema": body["schema"],
            "declaration": declaration["id"],
            "operation": body["operation"],
            "predicate": body.get("predicate"),
            "patterns": [asdict(pattern) for pattern in patterns],
        },
    }


def admit_host_operation(
    selected: dict | None,
    row: dict,
    signature: dict,
    *,
    source_operations: tuple | None = None,
    source_context: Mapping | None = None,
) -> dict:
    """Screen every selected host profile independently from accelerator admission."""
    from merlin.targetgen.host_admission_preparation import PreparedHostCapabilities

    if selected is None:
        return {
            "status": "unknown",
            "review_status": "unknown",
            "reviewed": False,
            "reason": "no selected per-operation host capability spec",
            "profiles": [],
        }
    prepared = type(selected) is PreparedHostCapabilities
    if not isinstance(selected, dict) and not prepared:
        raise ValueError("selected host capabilities must be a profile mapping")
    profiles = []
    for name, selection in selected._items() if prepared else sorted(selected.items()):
        if not isinstance(selection, dict):
            raise ValueError("selected host capability profile must be a mapping")
        document = selection.get("capability_spec")
        if document is None:
            profiles.append(
                {
                    "profile": name,
                    "status": "unknown",
                    "reviewed": False,
                    "review_status": "unknown",
                    "reason": "host profile has no operation capability declaration",
                }
            )
            continue
        if not prepared:
            validate_host_capabilities(
                document, package_sha256=selection.get("package_sha256"), dtype_strategy=selection.get("dtype_strategy")
            )
        pinned = (
            is_sha256(selection.get("package_sha256"))
            and is_sha256(selection.get("capability_spec_sha256"))
            and document["compiler"].get("package_sha256") == selection["package_sha256"]
        )
        decisions = []
        for declaration in selected._declarations(name, row) if prepared else document["operations"]:
            # A named selector is narrower than a semantic family. A package
            # schedule matching linalg.matmul must not admit an unrelated
            # linalg.generic merely because both describe contractions. Family
            # selectors remain available when no exact ops were declared.
            identities = (row.get("frontend_op"), row["mlir_operation"])
            exact_ops = declaration.get("ops") or []
            if exact_ops and not any(identity in exact_ops for identity in identities):
                continue
            operation = next(
                (identity for identity in identities if identity in exact_ops),
                row["mlir_operation"],
            )
            decision = admit_operation({**document, "operations": [declaration]}, operation, signature, "host")
            if "declaration" in decision:
                if "source_body" in declaration:
                    screened = _screen_source_body(declaration, row, signature, source_operations, source_context)
                    if decision["status"] == "admitted" or source_operations is None:
                        decision = {**decision, "status": screened["status"], "reason": screened["reason"]}
                        if screened.get("proof") is not None:
                            decision["source_body_proof"] = screened["proof"]
                            if "linkage_contract" in declaration and decision["status"] == "admitted":
                                decision["linkage_requirement"] = validate_linkage_contract(
                                    declaration["linkage_contract"]
                                )
                decisions.append(decision)
        verdict = next(
            (decision for decision in decisions if decision["status"] == "admitted"),
            next((decision for decision in decisions if decision["status"] == "unknown"), None),
        )
        status = verdict["status"] if verdict else "unsupported"
        reason = verdict["reason"] if verdict else "no host operation declaration admits the observed signature"
        if (
            document["status"] != "reviewed" or (verdict and verdict["review_status"] != "reviewed")
        ) and status == "unsupported":
            status, reason = "unknown", "unreviewed host declarations cannot establish absence of operation support"
        if not pinned:
            status, reason = "unknown", "host package and capability spec byte identities are not selected together"
        profiles.append(
            {
                "profile": name,
                "status": status,
                "reason": reason,
                "review_status": verdict["review_status"] if verdict else document["status"],
                "reviewed": document["status"] == "reviewed"
                and pinned
                and (verdict is None or verdict["review_status"] == "reviewed"),
                "package_sha256": selection.get("package_sha256"),
                "capability_spec_sha256": selection.get("capability_spec_sha256"),
                "dtype_strategy": selection.get("dtype_strategy"),
                "decisions": decisions,
                **(
                    {"source_body_proof": verdict["source_body_proof"]}
                    if verdict and "source_body_proof" in verdict
                    else {}
                ),
                **(
                    {"linkage_requirement": verdict["linkage_requirement"]}
                    if verdict and "linkage_requirement" in verdict and status == "admitted"
                    else {}
                ),
            }
        )
    verdict = next(
        (profile for profile in profiles if profile["status"] == "admitted"),
        next((profile for profile in profiles if profile["status"] == "unknown"), None),
    )
    if verdict is None:
        verdict = {
            "status": "unsupported" if profiles else "unknown",
            "review_status": "unknown",
            "reviewed": False,
            "reason": "no selected host profile admits the observed operation signature",
        }
    return {
        **copy.deepcopy({key: verdict[key] for key in ("status", "review_status", "reviewed", "reason")}),
        "profiles": profiles,
        "qualification": "selected declaration screen; host lowering remains unverified",
        **(
            {
                "source_body_proof": {
                    **verdict["source_body_proof"],
                    "profile": verdict["profile"],
                    "capability_spec_sha256": verdict["capability_spec_sha256"],
                }
            }
            if "source_body_proof" in verdict
            else {}
        ),
        **(
            {
                "linkage_requirement": {
                    "profile": verdict["profile"],
                    "capability_spec_sha256": verdict["capability_spec_sha256"],
                    "declaration": verdict["source_body_proof"]["declaration"],
                    "contract": copy.deepcopy(verdict["linkage_requirement"]),
                }
            }
            if verdict.get("status") == "admitted"
            and verdict.get("reviewed") is True
            and "source_body_proof" in verdict
            and "linkage_requirement" in verdict
            else {}
        ),
    }
