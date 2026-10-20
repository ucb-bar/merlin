"""Bounded public source and candidate declaration census, without cost prices."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from math import prod

from merlin.common import mlir_query as Q

MAX_TEXT = 8 << 20
MAX_OPERATIONS = 4096
NUMERIC_ATTRIBUTES = frozenset({
    "fastmath", "roundingmode", "rounding_mode", "overflow", "saturation",
    "accumulator_dtype", "output_dtype", "requant_shift", "acc_scale", "bound_lsb",
})


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def public_projection(value, *, depth=0):
    """Bound declarations and scrub paths; no artifact payload is projected."""
    if depth > 16:
        raise ValueError("candidate declaration depth exceeds limit")
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        if "/" in value or "\\" in value or len(value) > 512:
            return {"text_sha256": hashlib.sha256(value.encode()).hexdigest(), "text_withheld": True}
        return value
    if type(value) is list and len(value) <= 4096:
        return [public_projection(row, depth=depth + 1) for row in value]
    if type(value) is dict and len(value) <= 4096:
        return {key: public_projection(item, depth=depth + 1) for key, item in value.items()
                if type(key) is str and len(key) <= 128 and "/" not in key and "\\" not in key}
    raise ValueError("candidate declaration is unsupported or exceeds limit")


def _tensor(ty):
    from xdsl.dialects.builtin import MemRefType, TensorType

    if not isinstance(ty, (TensorType, MemRefType)):
        return {"dtype": str(ty), "shape": None, "shape_status": "not_shaped"}
    shape, dtype = Q.type_shape_dtype(ty)
    return {"shape": shape, "dtype": dtype,
            "shape_status": "static" if all(n >= 0 for n in shape) else "UNKNOWN"}


def _iteration_domain(op, kinds):
    """Original Linalg loop inversion plus exhaustive affine bounds, including keepdims reductions."""
    from merlin.perf.model_macs import _interval

    bounds = tuple(op.get_static_loop_ranges())
    if len(bounds) != len(kinds) or any(type(n) is not int or n <= 0 for n in bounds):
        raise ValueError("static source iteration bounds unavailable")
    if sum(n.bit_length() for n in bounds) > 256:
        raise ValueError("source demand arithmetic limit exceeded")
    maps = tuple(attr.data for attr in op.get_indexing_maps())
    if len(maps) != len(op.operands):
        raise ValueError("incomplete indexing map roster")
    for value, mapping in zip(op.operands, maps, strict=True):
        shape = tuple(value.type.get_shape())
        if mapping.num_symbols or mapping.num_dims != len(bounds) or len(mapping.results) != len(shape):
            raise ValueError("unsupported source indexing domain")
        for expr, extent in zip(mapping.results, shape, strict=True):
            lo, hi = _interval(expr, bounds)
            if extent <= 0 or lo < 0 or hi >= extent:
                raise ValueError("source indexing range not proved in bounds")
    return tuple(n for n, kind in zip(bounds, kinds) if kind == "parallel"), tuple(
        n for n, kind in zip(bounds, kinds) if kind == "reduction")


def source_observation(text):
    """Inventory complete parsed source, including scalar reduction/conversion bodies."""
    if type(text) is not str or len(text.encode()) > MAX_TEXT:
        raise ValueError("source text exceeds limit")
    # The original interface parser owns its closed grammar and refuses unknown ops.
    if "merlin_iface." in text:
        from merlin.targetgen.contract.interface_emit import parse_interface_mlir

        cb = parse_interface_mlir(text)
        from merlin.perf.work_volume import work_from_command_buffer

        work = work_from_command_buffer(cb)
        tensors = {name: {key: spec.get(key) for key in ("shape", "dtype", "role")}
                   for name, spec in cb.get("tensors", {}).items()}
        return {"status": "observed", "grammar": "public_interface", "tensors": tensors,
                "operations": [{"ordinal": i, "operation": row.get("opcode"),
                                "operands": row.get("operands", {}),
                                "numeric_attributes": {key: value for key, value in row.get("attributes", {}).items()
                                                       if key in NUMERIC_ATTRIBUTES}}
                               for i, row in enumerate(cb.get("commands", []))],
                "macs": work.exact_macs,
                "reduction_domain_points": None,
                "work_basis": "original interface, canonical command semantics; unsupported work remains UNKNOWN",
                "work_refusals": list(work.refusals),
                "semantic_equivalence": "UNKNOWN", "source_numeric_contract": "declared attributes only"}
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(text)
    operations = []
    identities = {}
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
    top = {op: i for i, op in enumerate(functions[0].body.block.ops) if op.name != "func.return"}
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        top = {}
    from merlin.perf.model_macs import _iterators, observe_model_macs

    mac_rows = [shape for _, shape in observe_model_macs(module)]
    reduction_points, reduction_missing = 0, False
    unknown_source_work = False
    for i, op in enumerate(module.walk()):
        if i >= MAX_OPERATIONS:
            raise ValueError("source operation limit exceeded")
        identities[op] = i
        unknown_source_work |= op.name == "builtin.unregistered" or op.name.startswith(("scf.", "cf."))
        props = {**op.attributes, **op.properties}
        iterators = props.get("iterator_types")
        domain = None
        kinds = _iterators(op)
        if kinds and "reduction" in kinds:
            try:
                parallel, reduction = _iteration_domain(op, kinds)
                if op.parent_op().name != "func.func":
                    raise ValueError("enclosing invocation multiplicity is not proved")
                domain = {"parallel": list(parallel), "reduction": list(reduction),
                          "domain_points": prod(parallel + reduction), "status": "derived"}
                reduction_points += domain["domain_points"]
            except (ValueError, AttributeError, NotImplementedError):
                domain = {"status": "UNKNOWN"}
                reduction_missing = True
        operations.append({
            "ordinal": i, "operation": Q.op_name(op),
            "source_op_index": top.get(op),
            "inputs": [_tensor(value.type) for value in op.operands],
            "results": [_tensor(value.type) for value in op.results],
            "dependencies": [identities.get(value.owner) for value in op.operands],
            "iterator_types": [str(value) for value in iterators] if iterators is not None else None,
            "numeric_attributes": {key: str(value) for key, value in props.items() if key in NUMERIC_ATTRIBUTES},
            "reduction_domain": domain,
            "scalar_body_operations": dict(sorted(Counter(Q.op_name(child)
                for region in op.regions for block in region.blocks for child in block.ops).items()))
                if kinds else None,
        })
    return {"status": "observed", "grammar": "parsed_mlir", "operations": operations,
            "operation_histogram": dict(sorted(Counter(row["operation"] for row in operations).items())),
            "macs": None if unknown_source_work or any(row.status != "derived" for row in mac_rows)
                    else sum(row.macs for row in mac_rows),
            "reduction_domain_points": None if reduction_missing or unknown_source_work else reduction_points,
            "semantic_equivalence": "UNKNOWN", "source_numeric_contract": "declared attributes only"}


def placement_observation(source, command_buffer):
    """Join explicit interface tensor types; execution/source-operation ownership stays UNKNOWN."""
    params = command_buffer.get("params") or {}
    placements = params.get("lane_placement") or []
    tensors = command_buffer.get("tensors") or {}
    joins = []
    for name, spec in (source.get("tensors") or {}).items():
        emitted = tensors.get(name)
        matches = isinstance(emitted, dict) and all(emitted.get(key) == spec.get(key) for key in ("shape", "dtype"))
        joins.append({"tensor": name, "source": spec,
                      "emitted": {key: emitted.get(key) for key in ("shape", "dtype")} if emitted else None,
                      "type_join": "matches" if matches else "UNKNOWN_OR_MISMATCH"})
    return {"status": "declared" if placements else "UNKNOWN", "declarations": placements,
            "tensor_type_joins": joins, "refusal": command_buffer.get("declined"),
            "source_operation_to_execution_lane": "UNKNOWN",
            "basis": "candidate placement and refusal declarations and exact named interface tensor types",
            "limitation": "matching names and types and placement declarations do not prove computation or execution"}
