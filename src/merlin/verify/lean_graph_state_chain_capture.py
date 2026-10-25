"""Trusted pinned actual typed SSA correspondence for supported semantic transport.

Runs as an isolated subprocess against the read-only producer checkout. Physical
cell identities stay exact. This mode supports flat entry operations and refuses
allocation/resource identity changes requiring a separate operational bridge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_graph_state_capture import capture, external_runtime, file_ref


def transport(reference, source, target, contract_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    admission = capture(reference, source, target, destination / "actual", contract_path=contract_path)
    if not admission["projection_checked"]:
        raise ValueError("actual complete stage correspondence refused: " + admission["reason"]["detail"])
    from merlin.transforms.planning import blocks_in
    from merlin.verify.lean_state_effects import EffectOperation, EffectScheduleContract
    from merlin.verify.lean_state_projection import _operation_name, _parse, _projection_nodes, _value_key

    # Reconstruct only the already validated original immutable host contract.
    record = dict(admission["contract"])
    operations = []
    for row in record.pop("operations"):
        row = dict(row)
        for key in ("operand_types", "result_types", "reads", "writes", "attribute_names"):
            row[key] = tuple(row[key])
        operations.append(EffectOperation(**row))
    record["disjoint_resources"] = tuple(tuple(pair) for pair in record["disjoint_resources"])
    record["isa_records"] = tuple(record["isa_records"])
    original_contract = EffectScheduleContract(operations=tuple(operations), **record)
    original_contract.validate()
    before = _parse((destination / "actual/before.mlir").read_text(), original_contract)
    after = _parse((destination / "actual/after.mlir").read_text(), original_contract)
    projection = admission["projection"]
    block_index = projection["block_index"]
    source_block, target_block = blocks_in(before)[block_index], blocks_in(after)[block_index]
    source_ops, target_ops = list(source_block.ops), list(target_block.ops)
    if any(op.regions or op.successors for op in [*source_ops, *target_ops]):
        raise ValueError("semantic transport requires flat operations; nested/control bridge is unavailable")
    target_nodes, target_cells = _projection_nodes(after, target_block, original_contract)
    source_cells = projection["cells"]
    source_nodes = projection["nodes"]
    order = projection["order"]

    def values(module):
        operations = {op: index for index, op in enumerate(module.walk())}
        blocks = {block: index for index, block in enumerate(blocks_in(module))}
        rows = [value for op in module.walk() for value in op.results]
        rows += [value for block in blocks for value in block.args]
        return operations, blocks, {_value_key(value, operations, blocks): value for value in rows}

    source_indices, source_blocks, source_values = values(before)
    target_indices, target_blocks, target_values = values(after)
    correspondence = dict(zip(source_block.args, target_block.args, strict=True))
    for actual_index, original_index in enumerate(order):
        original, actual = source_ops[original_index], target_ops[actual_index]
        if len(original.results) != len(actual.results) or any(
            left.type != right.type for left, right in zip(original.results, actual.results, strict=True)
        ):
            raise ValueError("actual result slot/type correspondence changed")
        correspondence.update(zip(original.results, actual.results, strict=True))
    for actual_index, original_index in enumerate(order):
        original, actual = source_ops[original_index], target_ops[actual_index]
        if len(original.operands) != len(actual.operands) or any(
            correspondence.get(left) is not right or left.type != right.type
            for left, right in zip(original.operands, actual.operands, strict=True)
        ):
            raise ValueError("actual ordered typed operand correspondence changed")
    forward, logical_source, logical_target = [], [], [None] * len(target_cells)
    physical = []
    for index, key in enumerate(source_cells):
        if key.startswith("ssa:"):
            original_value = source_values[key]
            actual_value = correspondence.get(original_value)
            if actual_value is None:
                raise ValueError("SSA cell lies outside supported exact entry correspondence")
            target_key = _value_key(actual_value, target_indices, target_blocks)
            if original_value.type != actual_value.type:
                raise ValueError("typed state-cell mapping changed")
            cell = {
                "kind": "argument" if key.startswith("ssa:block:") else "temporary",
                "logical_identity": key,
                "type": str(original_value.type),
            }
        else:
            target_key = key
            cell = {"kind": "physical", "identity": key}
            physical.append(index)
        if target_key not in target_cells:
            raise ValueError("physical/allocation identity changed; this bridge is not available")
        mapped = target_cells.index(target_key)
        if cell["kind"] == "physical" and mapped != index:
            raise ValueError("physical cells cannot be renamed as arbitrary SSA temporaries")
        if logical_target[mapped] is not None:
            raise ValueError("actual state mapping is not injective")
        logical_source.append(cell)
        logical_target[mapped] = cell
        forward.append(mapped)
    if len(source_cells) != len(target_cells) or any(item is None for item in logical_target):
        raise ValueError("actual state mapping is not a complete bijection")
    backward = [forward.index(index) for index in range(len(forward))]

    def primitives(ops, nodes, cells, operation_indices, block_indices):
        result = []
        for op, node in zip(ops, nodes, strict=True):
            reads = node["reads"] if type(node) is dict else node.reads
            writes = node["writes"] if type(node) is dict else node.writes
            physical_reads = sorted(cells[index] for index in reads if not cells[index].startswith("ssa:"))
            physical_writes = sorted(cells[index] for index in writes if not cells[index].startswith("ssa:"))
            inputs = [_value_key(value, operation_indices, block_indices) for value in op.operands] + physical_reads
            outputs = [_value_key(value, operation_indices, block_indices) for value in op.results] + physical_writes
            core = {
                "operation": _operation_name(op),
                "operandTypes": [str(value.type) for value in op.operands],
                "resultTypes": [str(value.type) for value in op.results],
                "attributes": json.dumps(
                    {
                        "attributes": {key: str(value) for key, value in sorted(op.attributes.items())},
                        "properties": {key: str(value) for key, value in sorted(op.properties.items())},
                    },
                    sort_keys=True,
                ),
                "physicalInputs": physical_reads,
                "physicalOutputs": physical_writes,
            }
            result.append(
                {
                    "core": core,
                    "inputs": [cells.index(key) for key in inputs],
                    "outputs": [cells.index(key) for key in outputs],
                }
            )
        return result

    source_primitives = primitives(source_ops, source_nodes, source_cells, source_indices, source_blocks)
    target_primitives = primitives(target_ops, target_nodes, target_cells, target_indices, target_blocks)
    # This preflight improves refusals; the Lean query rechecks every mapping,
    # physical cell, primitive descriptor and ordered slot independently.
    for position, source_index in enumerate(order):
        primitive = source_primitives[source_index]
        expected = {
            "core": primitive["core"],
            "inputs": [forward[key] for key in primitive["inputs"]],
            "outputs": [forward[key] for key in primitive["outputs"]],
        }
        if expected != target_primitives[position]:
            raise ValueError("common primitive descriptor or ordered operand/result mapping differs")
    result = {
        "schema": "merlin.graph-state-chain.transport-projection.v1",
        "projection_checked": True,
        "actual_stage": file_ref(destination / "actual/capture.json"),
        "order": order,
        "renaming": {"forward": forward, "backward": backward, "physical": physical},
        "source_cell_names": source_cells,
        "target_cell_names": list(target_cells),
        "source_cells": logical_source,
        "target_cells": logical_target,
        "source_primitives": source_primitives,
        "target_primitives": target_primitives,
        "source_footprints": [{"reads": node["reads"], "writes": node["writes"]} for node in source_nodes],
        "target_footprints": [{"reads": list(node.reads), "writes": list(node.writes)} for node in target_nodes],
        "source_inputs": [file_ref(source), file_ref(target), file_ref(contract_path)],
        "owned_projection": file_ref(__file__),
        "external_runtime": external_runtime(),
        "trust": "pinned full actual typed IR/effect projection and primary ordered-slot correspondence; "
        "common primitive interpretation must faithfully implement admitted actual instructions",
        "public_acceptance_issued": False,
        "source_contract_approved": False,
    }
    (destination / "transport.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("reference", "source", "target", "contract", "destination"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    try:
        result = transport(args.reference, args.source, args.target, args.contract, args.destination)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        args.destination.mkdir(parents=True, exist_ok=True)
        result = {"projection_checked": False, "reason": {"type": type(error).__name__, "detail": str(error)}}
        (args.destination / "transport.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"projection_checked": result["projection_checked"]}))


if __name__ == "__main__":
    main()
