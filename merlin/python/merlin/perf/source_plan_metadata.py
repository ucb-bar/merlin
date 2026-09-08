"""Artifact-bound source/storage and integer-epilogue metadata for whole-program plans.

The captured dispatch graph deliberately describes logical values, while a candidate command
buffer describes materialized physical tensors.  Keeping those records separate is useful, but it
also means a whole-model optimizer cannot reason about a boundary unless an independently checked
record joins them.  This module builds that join after the global-plan verifier has established both
halves.

Epilogue classification is similarly source-side.  Target instruction names are never consulted.
Only an exact, all-parallel integer scalar DAG, its tensor maps, and verified task ownership can
produce a stage.  Anything else remains an explicit refusal.  The record is discovery evidence; it
does not prove that a target can implement the stage or that replacing it is numerically equivalent.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SCHEMA = "source_plan_metadata_v1"
STORAGE_SCHEMA = "source_buffer_physical_storage_v1"
EPILOGUE_SCHEMA = "source_integer_epilogue_ownership_v1"

_STAGES = frozenset({"acc_scale", "bias", "activation", "requant", "narrow_store"})
_INTEGER_POINTWISE = frozenset({
    "arith.constant", "arith.addi", "arith.subi", "arith.muli", "arith.divsi",
    "arith.divui", "arith.remsi", "arith.remui", "arith.andi", "arith.ori",
    "arith.xori", "arith.shli", "arith.shrsi", "arith.shrui", "arith.maxsi",
    "arith.maxui", "arith.minsi", "arith.minui", "arith.extsi", "arith.extui",
    "arith.trunci", "arith.cmpi", "arith.select", "linalg.yield",
})
_CLAMP_NARROW = frozenset({
    "arith.constant", "arith.maxsi", "arith.maxui", "arith.minsi", "arith.minui",
    "arith.trunci", "linalg.yield",
})


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _props(op: Any, key: str) -> Any:
    return {**getattr(op, "attributes", {}), **getattr(op, "properties", {})}.get(key)


def _string_attr(op: Any, key: str) -> str | None:
    value = _props(op, key)
    value = getattr(value, "data", value)
    value = getattr(value, "data", value)
    return value if isinstance(value, str) and value else None


def _string_array_attr(op: Any, key: str) -> list[str] | None:
    value = _props(op, key)
    if value is None:
        return None
    try:
        result = []
        for item in value:
            item = getattr(item, "data", item)
            item = getattr(item, "data", item)
            if not isinstance(item, str) or not item:
                return None
            result.append(item)
        return result
    except TypeError:
        return None


def _tensor(value: Any) -> tuple[list[int], str] | None:
    from xdsl.dialects.builtin import TensorType

    ty = getattr(value, "type", None)
    if not isinstance(ty, TensorType):
        return None
    shape = [int(extent) for extent in ty.get_shape()]
    if any(extent <= 0 for extent in shape):
        return None
    return shape, str(ty.get_element_type())


def _integer_width(dtype: Any) -> int | None:
    if (not isinstance(dtype, str) or len(dtype) < 2 or dtype[0] != "i"
            or not dtype[1:].isdigit() or int(dtype[1:]) <= 0):
        return None
    return int(dtype[1:])


def _source_buffer_map(block: Any, operations: Sequence[Any], graph: Any,
                       problems: list[str]) -> tuple[dict[Any, str], dict[str, dict[str, Any]]]:
    """Recover the exact source SSA value represented by each logical graph buffer."""
    by_value: dict[Any, str] = {}
    origins: dict[str, dict[str, Any]] = {}
    argument_indices: set[int] = set()
    for name, buffer in graph.buffers.items():
        if buffer.kind != "arg":
            continue
        index = buffer.arg_index
        if type(index) is not int or not 0 <= index < len(block.args):
            problems.append(f"logical argument buffer {name!r} has no exact source argument")
            continue
        value = block.args[index]
        if value in by_value:
            problems.append("one source argument maps to multiple logical buffers")
            continue
        by_value[value] = name
        argument_indices.add(index)
        origins[name] = {"kind": "entry_argument", "argument_index": index}
    if argument_indices != set(range(len(block.args))):
        problems.append("logical argument buffers do not cover every source argument exactly once")
    if len(operations) != len(graph.nodes):
        problems.append("source operations and logical graph nodes have different lengths")
        return by_value, origins
    for index, (operation, node) in enumerate(zip(operations, graph.nodes, strict=True)):
        if len(operation.results) != len(node.outputs):
            problems.append(f"source operation {index} result count differs from logical graph")
            continue
        for result_index, (value, name) in enumerate(zip(
                operation.results, node.outputs, strict=True)):
            if value in by_value or name in origins:
                problems.append("source result/logical buffer mapping is not one-to-one")
                continue
            by_value[value] = name
            origins[name] = {
                "kind": "source_result", "source_operation_id": index,
                "result_index": result_index,
            }
    return by_value, origins


def _physical_storage(*, graph: Any, block: Any, operations: Sequence[Any],
                      value_tensors: Mapping[Any, str], tensors: Mapping[str, Any],
                      storage_encodings: Mapping[str, Any], problems: list[str]
                      ) -> tuple[dict[str, Any], dict[Any, str]]:
    by_value, origins = _source_buffer_map(block, operations, graph, problems)
    rows: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    represented: set[Any] = set()
    for value, tensor in value_tensors.items():
        source_buffer = by_value.get(value)
        if source_buffer is None:
            problems.append(f"materialized tensor {tensor!r} has no logical source buffer")
            continue
        represented.add(value)
        logical = graph.buffers[source_buffer]
        physical = tensors.get(tensor)
        checked = storage_encodings.get(tensor)
        source_type = _tensor(value)
        base = {
            "source_buffer": source_buffer,
            "source_origin": origins[source_buffer],
            "materialized_tensor": tensor,
            "logical": {"shape": list(logical.shape), "dtype": logical.dtype},
            "physical_tensor": {
                "shape": physical.get("shape") if isinstance(physical, Mapping) else None,
                "dtype": physical.get("dtype") if isinstance(physical, Mapping) else None,
                "role": physical.get("role") if isinstance(physical, Mapping) else None,
            },
        }
        if (source_type is None or source_type != (list(logical.shape), logical.dtype)
                or not isinstance(physical, Mapping)):
            problems.append(
                f"source/logical/physical identity for materialized tensor {tensor!r} is incomplete")
            continue
        contract = checked.get("contract") if isinstance(checked, Mapping) else None
        if not isinstance(contract, Mapping):
            unknown.append({**base, "reason": "no verified physical storage encoding"})
            continue
        layout = {key: contract.get(key) for key in (
            "axis_groups", "physical_shape", "strides_elements", "storage_elements",
            "offset_elements",
        )}
        encoding_sha256 = _digest(contract)
        layout_sha256 = _digest(layout)
        rows.append({
            **base,
            # Stable strings are representation identities suitable for exact form matching.
            # The checked contract remains in the enclosing verifier's storage_encodings table.
            "encoding": f"{contract.get('schema')}@sha256:{encoding_sha256}",
            "encoding_sha256": encoding_sha256,
            "encoding_contract_location": f"/storage_encodings/{tensor}/contract",
            "layout": f"static_strided_elements_v1@sha256:{layout_sha256}",
            "layout_sha256": layout_sha256,
            "layout_contract": layout,
            "proof_scope": checked.get("proof_scope"),
            "caller_materialization": checked.get("caller_materialization"),
            "emitted_consumer_addressing": checked.get("emitted_consumer_addressing"),
        })
    if set(value_tensors) != represented:
        problems.append("not every materialized source value has one logical buffer")
    rows.sort(key=lambda row: row["source_buffer"])
    unknown.sort(key=lambda row: row["source_buffer"])
    return {
        "schema": STORAGE_SCHEMA,
        "status": "complete" if not unknown else "partial",
        "materialized_source_values": len(value_tensors),
        "exact_physical_representations": len(rows),
        "rows": rows,
        "unknown": unknown,
        "proof_scope": (
            "exact source SSA/logical-buffer/materialized-tensor join plus verified bounded "
            "physical storage address contract"
        ),
        "not_proven": ["caller packing", "emitted consumer addressing", "runtime residency"],
    }, by_value


def _map_positions(amap: Any) -> tuple[int, ...] | None:
    from xdsl.ir.affine import AffineDimExpr

    if amap.num_symbols or not all(isinstance(expr, AffineDimExpr) for expr in amap.results):
        return None
    return tuple(expr.position for expr in amap.results)


def _broadcast_relation(value: Any, amap: Any, output_positions: tuple[int, ...],
                        output_shape: list[int]) -> str | None:
    info = _tensor(value)
    positions = _map_positions(amap)
    if info is None or positions is None:
        return None
    shape, _dtype = info
    if not shape and not positions:
        return "scalar"
    if len(shape) > len(output_shape) or len(positions) != len(shape):
        return None
    if shape == output_shape and positions == output_positions:
        return "exact"
    trailing = output_positions[-len(shape):]
    trailing_shape = output_shape[-len(shape):]
    if positions == trailing and all(left in (1, right)
                                      for left, right in zip(shape, trailing_shape, strict=True)):
        return "trailing_broadcast"
    return None


def _depends_on(value: Any, argument: Any, seen: set[Any] | None = None) -> bool:
    if value is argument:
        return True
    owner = getattr(value, "owner", None)
    if owner is None or not hasattr(owner, "operands"):
        return False
    seen = set() if seen is None else seen
    if value in seen:
        return False
    seen.add(value)
    return any(_depends_on(operand, argument, seen) for operand in owner.operands)


def _pointwise_contract(op: Any, chain_value: Any) -> tuple[dict[str, Any] | None, str]:
    """Return exact integer pointwise structure, or one fail-closed reason."""
    if op.name != "linalg.generic" or len(op.results) != 1:
        return None, "consumer is not a single-result linalg.generic"
    segment = _props(op, "operandSegmentSizes")
    maps_attr = _props(op, "indexing_maps")
    iterators = _props(op, "iterator_types")
    if segment is None or maps_attr is None or iterators is None:
        return None, "consumer lacks explicit operand segments, maps, or iterators"
    counts = list(segment.get_values())
    if len(counts) < 2 or counts[-1] != 1:
        return None, "consumer does not have one exact destination"
    n_inputs = counts[0]
    if len(op.operands) != n_inputs + 1 or list(op.operands[:n_inputs]).count(chain_value) != 1:
        return None, "consumer does not read the chain value exactly once"
    result = _tensor(op.results[0])
    chain = _tensor(chain_value)
    if result is None or chain is None or result[0] != chain[0]:
        return None, "consumer changes or lacks an exact static tensor domain"
    if _integer_width(result[1]) is None or _integer_width(chain[1]) is None:
        return None, "consumer enters non-integer arithmetic"
    iterator_values = [item.data.value for item in iterators]
    if iterator_values != ["parallel"] * len(result[0]):
        return None, "consumer is not one all-parallel output-stage domain"
    maps = [item.data for item in maps_attr]
    if len(maps) != len(op.operands):
        return None, "consumer maps do not cover every operand"
    output_positions = _map_positions(maps[-1])
    chain_index = list(op.operands[:n_inputs]).index(chain_value)
    if (output_positions is None or len(output_positions) != len(result[0])
            or sorted(output_positions) != list(range(len(result[0])))
            or _map_positions(maps[chain_index]) != output_positions):
        return None, "chain input and output maps are not the same full-rank permutation"
    body = op.regions[0].blocks[0] if len(op.regions) == 1 and len(op.regions[0].blocks) == 1 else None
    if body is None or len(body.args) != n_inputs + 1 or body.last_op is None \
            or body.last_op.name != "linalg.yield" or len(body.last_op.operands) != 1:
        return None, "consumer scalar region is not an exact one-yield body"
    if list(body.args[-1].uses):
        return None, "consumer destination initializer participates in scalar arithmetic"
    names = [nested.name for nested in body.ops]
    if any(name not in _INTEGER_POINTWISE for name in names):
        return None, "consumer scalar region contains non-integer or unsupported operations"
    yielded = body.last_op.operands[0]
    if not _depends_on(yielded, body.args[chain_index]):
        return None, "consumer result does not depend on the chain value"
    extra = []
    for operand_index in range(n_inputs):
        if operand_index == chain_index:
            continue
        relation = _broadcast_relation(
            op.operands[operand_index], maps[operand_index], output_positions, result[0])
        if relation is None:
            return None, "additional operand has no exact scalar/trailing/exact relation"
        extra.append({"operand_index": operand_index, "relation": relation,
                      "value": op.operands[operand_index], "body_argument": body.args[operand_index]})
    return {
        "n_inputs": n_inputs,
        "chain_index": chain_index,
        "chain": chain,
        "result": result,
        "body": body,
        "body_operations": names,
        "yielded": yielded,
        "extra": extra,
        "maps": [str(amap) for amap in maps],
    }, ""


def _direct_binary_yield(info: Mapping[str, Any], opcode: str) -> bool:
    owner = getattr(info["yielded"], "owner", None)
    if getattr(owner, "name", None) != opcode or len(owner.operands) != 2:
        return False
    chain_arg = info["body"].args[info["chain_index"]]
    extras = [item["body_argument"] for item in info["extra"]]
    return len(extras) == 1 and set(owner.operands) == {chain_arg, extras[0]}


def _explicit_stage(op: Any, info: Mapping[str, Any]) -> tuple[str, str, list[str], str] | None:
    stage = _string_attr(op, "prov.epilogue_stage")
    operation = _string_attr(op, "prov.epilogue_operation")
    roles = _string_array_attr(op, "prov.epilogue_operand_roles")
    present = any(_props(op, key) is not None for key in (
        "prov.epilogue_stage", "prov.epilogue_operation", "prov.epilogue_operand_roles"))
    if not present:
        return None
    if stage not in _STAGES or operation is None or roles is None or len(roles) != len(info["extra"]):
        return "", "", [], "explicit epilogue metadata is incomplete or inconsistent"
    if len(set(roles)) != len(roles):
        return "", "", [], "explicit epilogue operand roles are not unique"
    return stage, operation, roles, ""


def _classify_stage(op: Any, chain_value: Any, value_buffers: Mapping[Any, str]
                    ) -> tuple[dict[str, Any] | None, str]:
    info, reason = _pointwise_contract(op, chain_value)
    if info is None:
        return None, reason
    explicit = _explicit_stage(op, info)
    source = "exact_integer_scalar_dag"
    if explicit is not None:
        stage, operation, roles, reason = explicit
        if reason:
            return None, reason
        source = "explicit_source_attributes_plus_exact_integer_scalar_dag"
    else:
        extra_relations = [item["relation"] for item in info["extra"]]
        chain_width = _integer_width(info["chain"][1])
        output_width = _integer_width(info["result"][1])
        names = set(info["body_operations"])
        if (_direct_binary_yield(info, "arith.addi") and extra_relations
                and extra_relations[0] in {"scalar", "trailing_broadcast"}):
            stage, operation, roles = "bias", "integer_add", ["bias"]
        elif (_direct_binary_yield(info, "arith.muli") and extra_relations
              and extra_relations[0] in {"scalar", "trailing_broadcast"}):
            stage, operation, roles = "acc_scale", "integer_multiply", ["scale"]
        elif (not info["extra"] and output_width is not None and chain_width is not None
              and output_width < chain_width and "arith.trunci" in names
              and names <= _CLAMP_NARROW):
            stage = "narrow_store"
            operation = "saturating_integer_narrow" if names & {
                "arith.maxsi", "arith.maxui", "arith.minsi", "arith.minui"} else "integer_narrow"
            roles = []
        elif (not info["extra"] and output_width is not None and chain_width is not None
              and output_width < chain_width and "arith.trunci" in names):
            stage, operation, roles = "requant", "integer_requantize_and_narrow", []
        else:
            return None, "integer pointwise scalar DAG has no exact supported epilogue identity"
    extra_relations = [item["relation"] for item in info["extra"]]
    chain_width = _integer_width(info["chain"][1])
    output_width = _integer_width(info["result"][1])
    if stage == "bias" and (len(info["extra"]) != 1
                             or extra_relations[0] not in {"scalar", "trailing_broadcast"}):
        return None, "bias is not exactly one scalar or trailing-broadcast operand"
    if stage == "acc_scale" and not info["extra"]:
        return None, "acc_scale has no exact scale operand"
    if stage in {"activation", "narrow_store"} and info["extra"]:
        return None, f"{stage} unexpectedly consumes additional operands"
    if (stage == "narrow_store"
            and (chain_width is None or output_width is None or output_width >= chain_width)):
        return None, "narrow_store does not reduce an integer container width"
    inputs = []
    for operand_index, value in enumerate(op.operands[:info["n_inputs"]]):
        tensor = _tensor(value)
        relation = ("exact" if operand_index == info["chain_index"] else
                    next(item["relation"] for item in info["extra"]
                         if item["operand_index"] == operand_index))
        role = ("accumulator" if operand_index == info["chain_index"] else
                roles[[item["operand_index"] for item in info["extra"]].index(operand_index)])
        inputs.append({
            "source_buffer": value_buffers.get(value), "operand_index": operand_index,
            "role": role, "relation": relation,
            "shape": tensor[0] if tensor else None, "dtype": tensor[1] if tensor else None,
        })
    output = {
        "source_buffer": value_buffers.get(op.results[0]),
        "shape": info["result"][0], "dtype": info["result"][1],
    }
    semantic = {
        "stage": stage, "operation": operation, "inputs": inputs, "output": output,
        "indexing_maps": info["maps"], "scalar_operations": info["body_operations"],
    }
    return {
        **semantic,
        "semantic_sha256": _digest(semantic),
        "classification_source": source,
    }, ""


def _integer_epilogues(*, operations: Sequence[Any],
                       owners: Mapping[int, int], task_kinds: Mapping[int, str],
                       value_buffers: Mapping[Any, str],
                       is_contraction: Callable[[Any], bool]) -> dict[str, Any]:
    positions = {operation: index for index, operation in enumerate(operations)}
    roots: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not is_contraction(operation) or len(operation.results) != 1:
            continue
        accumulator = _tensor(operation.results[0])
        if accumulator is None or _integer_width(accumulator[1]) is None:
            continue
        task_index = owners.get(index)
        base = {
            "producer_source_operation_id": index,
            "producer_task_index": task_index,
            "producer_task_kind": task_kinds.get(task_index) if task_index is not None else None,
            "accumulator_source_buffer": value_buffers.get(operation.results[0]),
            "accumulator": {"shape": accumulator[0], "dtype": accumulator[1]},
            "source_operation_ids": [index],
            "stages": [],
            "reasons": [],
        }
        current = operation.results[0]
        completed = False
        for _step in range(len(operations)):
            uses = list(current.uses)
            if len(uses) != 1:
                base["reasons"].append(
                    "epilogue chain has fanout" if uses else
                    "epilogue chain terminates before a narrower integer output")
                break
            consumer = uses[0].operation
            consumer_index = positions.get(consumer)
            if consumer_index is None:
                base["reasons"].append("epilogue consumer is not a top-level source operation")
                break
            stage, reason = _classify_stage(consumer, current, value_buffers)
            if stage is None:
                base["reasons"].append(reason)
                break
            owner = owners.get(consumer_index)
            stage.update({
                "source_operation_id": consumer_index,
                "task_index": owner,
                "task_kind": task_kinds.get(owner) if owner is not None else None,
            })
            if owner is None:
                base["reasons"].append("epilogue stage has no verified task owner")
                break
            base["stages"].append(stage)
            base["source_operation_ids"].append(consumer_index)
            current = consumer.results[0]
            final = _tensor(current)
            if (final is not None and _integer_width(final[1]) is not None
                    and _integer_width(final[1]) < _integer_width(accumulator[1])):
                completed = True
                break
        else:
            base["reasons"].append("epilogue trace exceeds the bounded source operation count")
        base["classification"] = (
            "complete_integer_epilogue" if completed else
            "partial_integer_epilogue" if base["stages"] else "unclassified")
        base["reasons"] = sorted(set(base["reasons"]))
        roots.append(base)
    counts = {name: sum(root["classification"] == name for root in roots) for name in (
        "complete_integer_epilogue", "partial_integer_epilogue", "unclassified")}
    return {
        "schema": EPILOGUE_SCHEMA,
        "status": "verified",
        "contraction_roots": len(roots),
        "classification_counts": counts,
        "roots": roots,
        "proof_scope": (
            "exact integer scalar DAG, tensor indexing maps, source edges, and verified task "
            "ownership; target capability is deliberately excluded"
        ),
        "not_proven": [
            "target epilogue support", "replacement semantic equivalence",
            "emitted work deletion", "runtime correctness", "cycle improvement",
        ],
    }


def build_source_plan_metadata(*, block: Any, operations: Sequence[Any], graph: Any,
                               owners: Mapping[int, int], task_kinds: Mapping[int, str],
                               value_tensors: Mapping[Any, str], tensors: Mapping[str, Any],
                               storage_encodings: Mapping[str, Any],
                               is_contraction: Callable[[Any], bool]) -> dict[str, Any]:
    """Build the target-neutral metadata join after its component records verify."""
    problems: list[str] = []
    storage, value_buffers = _physical_storage(
        graph=graph, block=block, operations=operations, value_tensors=value_tensors,
        tensors=tensors, storage_encodings=storage_encodings, problems=problems)
    epilogues = _integer_epilogues(
        operations=operations, owners=owners, task_kinds=task_kinds,
        value_buffers=value_buffers, is_contraction=is_contraction)
    return {
        "schema": SCHEMA,
        "status": ("refused" if problems else
                   "verified" if storage["status"] == "complete" else "partial"),
        "physical_storage": storage,
        "integer_epilogue_ownership": epilogues,
        "problems": sorted(set(problems)),
        "proof_scope": (
            "artifact-bound source/storage identity and source-side integer epilogue ownership"
        ),
    }
