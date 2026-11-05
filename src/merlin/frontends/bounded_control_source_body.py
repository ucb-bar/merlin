"""Closed scalar control roles under a caller-reproved whole-source interval proof.

This source screen grants no linked-build, index-width or numerical qualification.
The trusted caller must reconstruct the complete source and reprove its bounds;
an operation name or a serialized ordinal list alone is never sufficient.
"""
from hashlib import sha256

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_json
from merlin.llvmlower.target_data_layout import selected_index_bits
from merlin.targetgen.application_inventory import operation_structure
from merlin.xdsl_dialects._common import text

SCHEMA = "merlin.bounded_control_source_body.v1"
PROOF_SCHEMA = "merlin.connected_bounded_control_source.v1"
ROLES = {
    "count_to_index": ("arith.index_cast", ["i64"], ["index"]),
    "bounded_cursor_add": ("arith.addi", ["index", "index"], ["index"]),
    "count_bound_compare": ("arith.cmpi", ["index", "index"], ["i1"]),
    "proved_count_assert": ("cf.assert", ["i1"], []),
}


def validate(body):
    if (type(body) is not dict or set(body) != {"schema", "operation"}
            or body.get("schema") != SCHEMA or body.get("operation") not in ROLES):
        raise ValueError("bounded control needs one closed source role")
    return dict(body)


def screen(declaration, row, signature, source_operations, source_context):
    """Require exact parsed module/ordinal/ABI plus independently proved bounds."""
    if (type(source_context) is not dict or set(source_context) != {
            "selected_index_observation", "verified_bounded_control_source"}):
        return {"status": "unknown", "reason": "bounded control requires caller-reproved source interval context"}
    selected = source_context["selected_index_observation"]
    record = source_context["verified_bounded_control_source"]
    width = selected_index_bits(selected)
    if (width is None or type(record) is not dict or record.get("schema") != PROOF_SCHEMA
            or record.get("status") != "source_proved_pending_linked_build"
            or record.get("index_bits") != width
            or any(not is_sha256(record.get(k)) for k in (
                "raw_source_sha256", "normalized_source_sha256", "parsed_module_sha256"))
            or type(record.get("roles")) is not list):
        return {"status": "unsupported", "reason": "bounded control record or selected width differs"}
    role = declaration["source_body"]["operation"]
    opname, inputs, outputs = ROLES[role]
    ordinals = row.get("ordinals")
    if (type(ordinals) is not list or not ordinals or any(type(x) is not int for x in ordinals)
            or ordinals != sorted(set(ordinals)) or row.get("count") != len(ordinals)
            or row.get("mlir_operation") != opname or type(source_operations) is not tuple
            or len(source_operations) != len(ordinals)
            or signature.get("ordered_operand_dtypes") != inputs
            or signature.get("ordered_result_dtypes") != outputs):
        return {"status": "unsupported", "reason": "bounded control row has another occurrence roster or ABI"}
    module = source_operations[0]
    while module.parent_op() is not None:
        module = module.parent_op()
    parsed = tuple(mq.walk(module))
    if (module.name != "builtin.module" or record.get("n_source_operations") != len(parsed)
            or sha256(text(module, generic=True).encode()).hexdigest() != record["parsed_module_sha256"]):
        return {"status": "unsupported", "reason": "bounded control parsed source differs from proved module"}
    selected_roles = []
    for ordinal, op in zip(ordinals, source_operations, strict=True):
        matches = [r for r in record["roles"] if isinstance(r, dict) and r.get("ordinal") == ordinal]
        if (not 0 <= ordinal < len(parsed) or parsed[ordinal] is not op
                or len(matches) != 1 or matches[0].get("role") != role):
            return {"status": "unsupported", "reason": "bounded control ordinal is not one proved role"}
        structure = operation_structure(op)
        if (op.name != opname or op.regions or op.successors
                or structure["ordered_operand_types"] != [{"shape": [], "dtype": x} for x in inputs]
                or structure["ordered_result_types"] != [{"shape": [], "dtype": x} for x in outputs]
                or row.get("ordered_operand_types") != structure["ordered_operand_types"]
                or row.get("ordered_result_types") != structure["ordered_result_types"]):
            return {"status": "unsupported", "reason": "bounded control parsed scalar signature differs"}
        selected_roles.append({"ordinal": ordinal, "role": role})
    return {"status": "admitted", "reason": "exact scalar roles have caller-reproved bounded source premises",
            "proof": {"schema": SCHEMA, "declaration": declaration["id"], "operation": role,
                      "source_record_sha256": sha256(canonical_json(record)).hexdigest(),
                      "selected_index_observation": dict(selected), "roles": selected_roles,
                      "scope": "source placement only; actual host lowering, effects and all outputs remain required"}}
