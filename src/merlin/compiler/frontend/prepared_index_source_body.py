"""Closed, caller-premised source roles for two independent prepared index forms.

This screens parsed roots against a receipt/trace proof supplied by the trusted
caller. It does not produce that proof, associate prepared nodes, grant host
placement, or establish original/prepared or compiled numerical equivalence.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_json
from merlin.common.jsonio import strict_json_equal as _same_json
from merlin.compiler.frontend.linalg_integer_reductions import recognize_static_integer_reduction
from merlin.compiler.frontend.linalg_patterns import (
    InvalidLinalgPattern,
    _checked_static_linalg_shell,
    recognize_static_pointwise,
)
from merlin.compiler.lower.model.target_data_layout import selected_index_bits
from merlin.development.targetgen.application_inventory import operation_structure

SCHEMA = "merlin.prepared_index_source_body.v1"
PROOF_SCHEMA = "merlin.prepared_index_source.v1"
PROOF_STATUS = "source_index_forms_pending_build"
PROOF_SCOPE = (
    "prepared Boolean-mask count and independently literal-bounded indexed extract only; "
    "no cross-node association, host admission, compiled bounds, or numerical equivalence"
)
ROLES = {
    "boolean_extension": ("linalg.generic", ("i1", "i64"), ("i64",)),
    "mask_sum": ("linalg.reduce", ("i64", "i64"), ("i64",)),
    "literal_indexed_extract": ("linalg.generic", None, ("i1",)),
}
_TOP = {
    "schema",
    "status",
    "scope",
    "index_bits_premise",
    "raw_source_sha256",
    "normalized_source_sha256",
    "frontend_trace_sha256",
    "capture_receipt_sha256",
    "n_source_operations",
    "original_to_prepared_equivalence",
    "records",
}
_MASK = {
    "source_node_id",
    "compute_ordinals",
    "kind",
    "component_ordinals",
    "extension_ordinal",
    "reduction_ordinal",
    "scalar_ordinal",
    "count_interval",
    "compaction_and_index_data",
}
_EXTRACT = {
    "source_node_id",
    "compute_ordinals",
    "kind",
    "component_ordinals",
    "source_shape",
    "output_shape",
    "index_input_shapes",
    "input_maps",
    "index_sources",
}


def validate_declaration(body: object) -> dict[str, str]:
    """Validate a role selector, without admitting any host operation."""
    if (
        type(body) is not dict
        or set(body) != {"schema", "operation"}
        or body.get("schema") != SCHEMA
        or type(body.get("operation")) is not str
        or body.get("operation") not in ROLES
    ):
        raise ValueError("prepared index source_body needs one closed role")
    return dict(body)


def _dims(value: object, limit: int) -> bool:
    return type(value) is list and bool(value) and all(type(dim) is int and 0 < dim <= limit for dim in value)


def _ordinals(value: object, total: int) -> bool:
    return (
        type(value) is list
        and all(type(item) is int and 0 <= item < total for item in value)
        and value == sorted(set(value))
    )


def _selected_width(selected: object) -> int | None:
    width = selected_index_bits(selected)
    return width if width is not None and 2 <= width <= 128 else None


def _record_role(record: Mapping[str, Any], ordinal: int) -> str | None:
    if record["kind"] == "boolean_mask_count":
        if ordinal == record["extension_ordinal"]:
            return "boolean_extension"
        if ordinal == record["reduction_ordinal"]:
            return "mask_sum"
    elif record["kind"] == "literal_indexed_extract" and ordinal in record["compute_ordinals"]:
        return "literal_indexed_extract"
    return None


def _node_ids(op: Any) -> tuple[str, ...]:
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    value = op.attributes.get("prov.source_node_ids")
    if not isinstance(value, ArrayAttr) or any(not isinstance(item, StringAttr) for item in value.data):
        return ()
    return tuple(item.data for item in value.data)


def validate_source_record(value: object, *, selected: object) -> dict[str, Any]:
    """Check the complete JSON shape; source/trace truth is independently reproved privately."""
    width = _selected_width(selected)
    if (
        width is None
        or type(value) is not dict
        or set(value) != _TOP
        or value.get("schema") != PROOF_SCHEMA
        or value.get("status") != PROOF_STATUS
        or value.get("scope") != PROOF_SCOPE
        or type(value.get("index_bits_premise")) is not int
        or value["index_bits_premise"] != width
        or type(value.get("n_source_operations")) is not int
        or value["n_source_operations"] <= 0
        or value.get("original_to_prepared_equivalence") != "not_proved"
        or any(
            not is_sha256(value.get(key))
            for key in (
                "raw_source_sha256",
                "normalized_source_sha256",
                "frontend_trace_sha256",
                "capture_receipt_sha256",
            )
        )
        or type(value.get("records")) is not list
    ):
        raise ValueError("prepared index source record is incomplete or selects another source/index width")
    total = value["n_source_operations"]
    limit = (1 << (width - 1)) - 1
    seen: set[int] = set()
    nodes: set[str] = set()
    for record in value["records"]:
        if (
            type(record) is not dict
            or type(record.get("kind")) is not str
            or record.get("kind") not in {"boolean_mask_count", "literal_indexed_extract"}
            or set(record) != (_MASK if record["kind"] == "boolean_mask_count" else _EXTRACT)
            or type(record.get("source_node_id")) is not str
            or not record["source_node_id"]
            or record["source_node_id"] in nodes
            or not _ordinals(record.get("compute_ordinals"), total)
            or not record["compute_ordinals"]
            or not _ordinals(record.get("component_ordinals"), total)
            or not set(record["compute_ordinals"]) <= set(record["component_ordinals"])
            or seen.intersection(record["compute_ordinals"])
        ):
            raise ValueError("prepared index record has malformed or repeated source roles")
        nodes.add(record["source_node_id"])
        seen.update(record["compute_ordinals"])
        if record["kind"] == "boolean_mask_count":
            ext, red, scalar = (record[key] for key in ("extension_ordinal", "reduction_ordinal", "scalar_ordinal"))
            if (
                any(type(item) is not int or not 0 <= item < total for item in (ext, red, scalar))
                or record["compute_ordinals"] != sorted((ext, red))
                or scalar in (ext, red)
                or scalar not in record["component_ordinals"]
                or type(record.get("count_interval")) is not list
                or len(record["count_interval"]) != 2
                or type(record["count_interval"][0]) is not int
                or record["count_interval"][0] != 0
                or type(record["count_interval"][1]) is not int
                or not 0 < record["count_interval"][1] <= limit
                or record.get("compaction_and_index_data") != "separate_control_source_obligation"
            ):
                raise ValueError("prepared mask-count roles or extent are malformed")
        else:
            source_shape, output_shape = record.get("source_shape"), record.get("output_shape")
            indices, maps, sources = (record.get(key) for key in ("index_input_shapes", "input_maps", "index_sources"))
            if (
                len(record["compute_ordinals"]) != 1
                or not _dims(source_shape, limit)
                or not _dims(output_shape, limit)
                or type(indices) is not list
                or len(indices) != len(source_shape)
                or not all(_dims(shape, limit) for shape in indices)
                or type(maps) is not list
                or len(maps) != len(indices)
                or not all(type(item) is str and item for item in maps)
                or type(sources) is not list
                or len(sources) != len(indices)
            ):
                raise ValueError("prepared indexed extract shapes or maps are malformed")
            for axis, source in enumerate(sources):
                if (
                    type(source) is not dict
                    or set(source)
                    != {"axis", "range_ordinal", "range_literal_sha256", "addition_ordinals", "offset_literal_sha256"}
                    or type(source.get("axis")) is not int
                    or source["axis"] != axis
                    or type(source.get("range_ordinal")) is not int
                    or not 0 <= source["range_ordinal"] < total
                    or not is_sha256(source.get("range_literal_sha256"))
                    or not _ordinals(source.get("addition_ordinals"), total)
                    or source.get("offset_literal_sha256") is not None
                    and not is_sha256(source.get("offset_literal_sha256"))
                ):
                    raise ValueError("prepared indexed extract has malformed literal range source")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared index record is not strict JSON") from exc
    return value


def screen_source_body(
    declaration: dict, row: dict, signature: dict, source_operations: tuple | None, source_context: Mapping | None
) -> dict:
    """Screen an exact row using a caller-reproved record; remain conditional until private linked join."""
    if not isinstance(source_context, Mapping) or set(source_context) != {
        "selected_index_observation",
        "verified_index_source",
    }:
        return {"status": "unknown", "reason": "index source_body needs a caller-verified source/trace premise"}
    selected, record = source_context["selected_index_observation"], source_context["verified_index_source"]
    try:
        validate_source_record(record, selected=selected)
    except ValueError as exc:
        return {"status": "unsupported", "reason": str(exc)}
    ordinals = row.get("ordinals")
    if (
        type(ordinals) is not list
        or not ordinals
        or not _ordinals(ordinals, record["n_source_operations"])
        or type(row.get("count")) is not int
        or row["count"] != len(ordinals)
        or type(source_operations) is not tuple
        or len(source_operations) != len(ordinals)
        or len({id(op) for op in source_operations}) != len(source_operations)
        or row.get("frontend_op") != "aten.index.Tensor"
        or row.get("mlir_operation") != ROLES[declaration["source_body"]["operation"]][0]
    ):
        return {"status": "unsupported", "reason": "index row has no exact parsed source identity"}
    role = declaration["source_body"]["operation"]
    expected_dtypes = ROLES[role][1]
    roles = []
    for ordinal, op in zip(ordinals, source_operations, strict=True):
        matches = [item for item in record["records"] if ordinal in item["compute_ordinals"]]
        if len(matches) != 1 or _record_role(matches[0], ordinal) != role:
            return {"status": "unsupported", "reason": "index ordinal is not a proved prepared role"}
        if mq.op_name(op) != ROLES[role][0] or mq.attr_str(op, "prov.aten") != "aten.index.Tensor":
            return {"status": "unsupported", "reason": "index parsed op differs from role"}
        if _node_ids(op) != (matches[0]["source_node_id"],):
            return {"status": "unsupported", "reason": "index parsed root has another prepared node owner"}
        structure = operation_structure(op)
        inputs, outputs = structure["ordered_operand_types"], structure["ordered_result_types"]
        input_dtypes = tuple(item["dtype"] for item in inputs)
        result_dtypes = tuple(item["dtype"] for item in outputs)
        if (
            (expected_dtypes is not None and input_dtypes != expected_dtypes)
            or (role == "literal_indexed_extract" and input_dtypes != ("i64",) * (len(input_dtypes) - 1) + ("i1",))
            or result_dtypes != ROLES[role][2]
            or not _same_json(row.get("ordered_operand_types"), inputs)
            or not _same_json(row.get("ordered_result_types"), outputs)
            or signature.get("ordered_operand_dtypes") != list(input_dtypes)
            or signature.get("ordered_result_dtypes") != list(result_dtypes)
            or type(signature.get("rank")) is not int
            or signature["rank"] != len(outputs[0]["shape"])
        ):
            return {"status": "unsupported", "reason": "index source role has another typed tensor signature"}
        try:
            if role == "boolean_extension":
                pattern = recognize_static_pointwise(op)
                if pattern.operation != "arith.extui" or pattern.ordered_types != ("i1", "i64", "i64"):
                    raise InvalidLinalgPattern("mask extension is not exact unsigned Boolean cast")
            elif role == "mask_sum":
                pattern = recognize_static_integer_reduction(op, index_bits=record["index_bits_premise"])
                if pattern.operation != "sum" or pattern.input_type != "i1":
                    raise InvalidLinalgPattern("mask reduction does not sum a proved Boolean extension")
            else:
                from xdsl.dialects import tensor

                shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
                item = matches[0]
                extract = next((inner for inner in shell.body if type(inner) is tensor.ExtractOp), None)
                if extract is None:
                    raise InvalidLinalgPattern("index body has no tensor extraction")
                source_shape, source_dtype = mq.type_shape_dtype(extract.operands[0].type)
                if (
                    source_dtype != "i1"
                    or item["source_shape"] != list(source_shape)
                    or item["output_shape"] != list(shell.shape)
                    or item["index_input_shapes"] != [list(value.type.get_shape()) for value in op.inputs]
                    or item["input_maps"] != [str(mapping) for mapping in shell.input_maps]
                ):
                    raise InvalidLinalgPattern("indexed extract shape/map/source type differs")
        except (InvalidLinalgPattern, ValueError, AttributeError, TypeError) as exc:
            return {"status": "unsupported", "reason": f"index source role structural screen refused: {exc}"}
        roles.append({"ordinal": ordinal, "role": role})
    return {
        "status": "admitted",
        "reason": "every parsed ordinal has one caller-reproved prepared index role",
        "proof": {
            "schema": SCHEMA,
            "declaration": declaration["id"],
            "operation": role,
            "selected_index_observation": dict(selected),
            "source_record_sha256": sha256(canonical_json(record)).hexdigest(),
            "roles": roles,
        },
    }
