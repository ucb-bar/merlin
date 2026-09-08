"""Deterministic, fail-closed partition planning for a verified linalg graph."""
from __future__ import annotations

import math
from collections import Counter, OrderedDict, defaultdict

from xdsl.dialects.builtin import TensorType

from .frontend import _str_attr
from .full_graph import _partition_class, _tensor_dtype, _tensor_shape


_DTYPE_BYTES = {"i1": 1, "i8": 1, "f8E4M3FN": 1, "bf16": 2, "f16": 2,
                "i16": 2, "f32": 4, "i32": 4, "i64": 8, "index": 8}
_UNTAGGED_BRIDGES = {"linalg.transpose", "tensor.cast", "tensor.collapse_shape",
                     "tensor.expand_shape", "linalg.copy"}


def _shape(value) -> tuple[int, ...]:
    return _tensor_shape(value) or ()


def _bytes(value, dtype: str | None = None) -> int | None:
    if not isinstance(value.type, TensorType):
        return None
    shape = _shape(value)
    if any(extent < 0 for extent in shape):
        return None
    element = dtype or _tensor_dtype(value)
    return math.prod(shape) * _DTYPE_BYTES.get(element, 4)


def _regions(block) -> tuple[OrderedDict[str, dict], dict[str, str]]:
    rows: OrderedDict[str, dict] = OrderedDict()
    for op in block.ops:
        region_id = _str_attr(op, "prov.region_id")
        if not region_id:
            continue
        row = rows.setdefault(
            region_id,
            {"semantic": _str_attr(op, "prov.op"), "dtypes": set(), "shapes": set()},
        )
        for value in (*op.operands, *op.results):
            if (dtype := _tensor_dtype(value)) is not None:
                row["dtypes"].add(dtype)
            shape = _tensor_shape(value)
            if shape is not None:
                row["shapes"].add(shape)
    classes = {
        region_id: _partition_class(row["semantic"], row["dtypes"], row["shapes"])[0]
        for region_id, row in rows.items()
    }
    return rows, classes


def _physical_ops(block) -> list:
    physical = []
    for op in block.ops:
        semantic = _str_attr(op, "prov.op")
        if op.name == "linalg.matmul":
            inputs = list(op.inputs)
            if len(inputs) >= 2 and len(_shape(inputs[0])) == len(_shape(inputs[1])) == 2:
                lhs, rhs = _shape(inputs[0]), _shape(inputs[1])
                if lhs[1] == rhs[0]:
                    physical.append(op)
        elif op.name == "linalg.generic" and semantic == "batch_matmul":
            inputs, outputs = list(op.inputs), list(op.outputs)
            if len(inputs) != 2 or len(outputs) != 1:
                continue
            lhs, rhs, out = _shape(inputs[0]), _shape(inputs[1]), _shape(outputs[0])
            if (len(lhs) == len(rhs) == len(out) == 3 and lhs[0] == rhs[0] == out[0]
                    and lhs[1] == out[1] and lhs[2] == rhs[1] and rhs[2] == out[2]):
                physical.append(op)
    return physical


def _bias_consumer(op):
    if _str_attr(op, "prov.op") != "addmm" or not op.results:
        return None
    candidates = []
    for use in op.results[0].uses:
        consumer = use.operation
        if (consumer.name == "linalg.generic"
                and _str_attr(consumer, "prov.op") == "addmm"
                and _str_attr(consumer, "prov.fqn") == _str_attr(op, "prov.fqn")):
            candidates.append(consumer)
    return candidates[0] if len(candidates) == 1 else None


def _value_origin(value, block, op_index, region_to_partition, classes,
                  current_partition, bridges=(), seen=frozenset()) -> dict:
    key = id(value)
    if key in seen:
        return {"kind": "cycle", "bridges": list(bridges)}
    owner = getattr(value, "owner", None)
    if owner not in op_index:
        for index, argument in enumerate(block.args):
            if argument is value:
                return {"kind": "function_argument", "argument_index": index,
                        "bridges": list(bridges)}
        return {"kind": "external_value", "bridges": list(bridges)}
    region_id = _str_attr(owner, "prov.region_id")
    if region_id in region_to_partition and region_to_partition[region_id] != current_partition:
        return {"kind": "accelerator_partition",
                "partition_id": region_to_partition[region_id],
                "producer_region": region_id, "bridges": list(bridges)}
    if (region_id and region_to_partition.get(region_id) == current_partition
            and owner.operands):
        # Capture decompositions such as im2col/reshape may share the physical
        # contraction's provenance region.  They are not emitted by the mesh
        # kernel, so walk through them and record that host preprocessing is
        # required at this partition boundary.
        meaningful = []
        for operand in owner.operands:
            operand_owner = getattr(operand, "owner", None)
            if getattr(operand_owner, "name", "") in {"tensor.empty", "linalg.fill"}:
                continue
            meaningful.append(operand)
        if meaningful:
            return _value_origin(
                meaningful[0], block, op_index, region_to_partition, classes,
                current_partition,
                (*bridges, {"region_id": region_id, "op": owner.name,
                            "execution": "host_preprocess"}),
                seen | {key},
            )
    if region_id and classes.get(region_id) == "layout_bridge_candidate" and owner.operands:
        return _value_origin(owner.operands[0], block, op_index, region_to_partition,
                             classes, current_partition,
                             (*bridges, {"region_id": region_id, "op": owner.name}),
                             seen | {key})
    if region_id:
        return {"kind": "host_region", "region_id": region_id,
                "semantic": _str_attr(owner, "prov.op"), "bridges": list(bridges)}
    if owner.name in _UNTAGGED_BRIDGES and owner.operands:
        return _value_origin(owner.operands[0], block, op_index, region_to_partition,
                             classes, current_partition,
                             (*bridges, {"region_id": None, "op": owner.name}),
                             seen | {key})
    return {"kind": "host_unattributed_op", "op": owner.name,
            "op_index": op_index[owner], "bridges": list(bridges)}


def _frontier_consumers(value, op_index, region_to_partition, classes,
                        current_partition, seen=frozenset()) -> list[dict]:
    key = id(value)
    if key in seen:
        return []
    frontier = []
    for use in value.uses:
        consumer = use.operation
        if consumer not in op_index:
            continue
        region_id = _str_attr(consumer, "prov.region_id")
        target = region_to_partition.get(region_id)
        if target is not None and target != current_partition:
            frontier.append({"kind": "accelerator_partition", "partition_id": target,
                             "region_id": region_id, "op_index": op_index[consumer]})
            continue
        bridge = (
            classes.get(region_id) == "layout_bridge_candidate"
            or (not region_id and consumer.name in _UNTAGGED_BRIDGES)
        )
        if bridge and consumer.results:
            for result in consumer.results:
                frontier.extend(_frontier_consumers(
                    result, op_index, region_to_partition, classes,
                    current_partition, seen | {key}))
            continue
        frontier.append({"kind": "host_region" if region_id else "host_unattributed_op",
                         "region_id": region_id or None,
                         "semantic": _str_attr(consumer, "prov.op") or None,
                         "op": consumer.name, "op_index": op_index[consumer]})
    unique = {(row.get("kind"), row.get("partition_id"), row.get("region_id"),
               row.get("op_index")): row for row in frontier}
    return [unique[key] for key in sorted(unique, key=lambda x: tuple(str(v) for v in x))]


def plan_full_graph(workload) -> dict:
    funcs = [op for op in workload.module.walk() if op.name == "func.func"]
    if len(funcs) != 1:
        raise ValueError(f"expected one func.func, found {len(funcs)}")
    block = funcs[0].body.blocks[0]
    ops = list(block.ops)
    op_index = {op: index for index, op in enumerate(ops)}
    _, classes = _regions(block)
    physical = _physical_ops(block)

    partitions = []
    region_to_partition = {}
    for index, op in enumerate(physical):
        partition_id = f"atlas_p{index:04d}"
        bias = _bias_consumer(op)
        regions = [_str_attr(op, "prov.region_id")]
        if bias is not None:
            regions.append(_str_attr(bias, "prov.region_id"))
        for region_id in regions:
            if region_id in region_to_partition:
                raise ValueError(f"region {region_id} belongs to two physical partitions")
            region_to_partition[region_id] = partition_id
        inputs = list(op.inputs)
        if bias is not None:
            inputs.append(list(bias.inputs)[1])
        lhs, rhs = _shape(inputs[0]), _shape(inputs[1])
        if len(lhs) == 2:
            geometry = {"M": lhs[0], "K": lhs[1], "N": rhs[1]}
            kind = "matmul"
            kernel_id = f"matmul_{lhs[0]}_{lhs[1]}_{rhs[1]}"
        else:
            geometry = {"B": lhs[0], "M": lhs[1], "K": lhs[2], "N": rhs[2]}
            kind = "matmul_batched"
            kernel_id = f"matmul_batched_{lhs[0]}_{lhs[1]}_{lhs[2]}_{rhs[2]}"
        if bias is not None:
            kernel_id += "_bias"
        output = bias.results[0] if bias is not None else op.results[0]
        partitions.append({
            "partition_id": partition_id,
            "kind": kind,
            "source_semantic": _str_attr(op, "prov.op"),
            "fqn": _str_attr(op, "prov.fqn"),
            "capture_regions": regions,
            "capture_op_index": op_index[op],
            "geometry": geometry,
            "kernel_id": kernel_id,
            "bias_fused": bias is not None,
            "_inputs": inputs,
            "_output": output,
        })

    by_id = {row["partition_id"]: row for row in partitions}
    accelerator_edges = set()
    for row in partitions:
        abi_inputs = []
        names = ["A0", "W"] + (["B"] if row["bias_fused"] else [])
        device_dtypes = ["fp8_e4m3", "fp8_e4m3"] + (["bf16"] if row["bias_fused"] else [])
        for name, value, device_dtype in zip(names, row.pop("_inputs"), device_dtypes):
            origin = _value_origin(value, block, op_index, region_to_partition,
                                   classes, row["partition_id"])
            if origin["kind"] == "accelerator_partition":
                accelerator_edges.add((origin["partition_id"], row["partition_id"]))
            abi_inputs.append({
                "name": name,
                "capture_type": str(value.type),
                "capture_bytes": _bytes(value),
                "device_dtype": device_dtype,
                "device_bytes": _bytes(value, "f8E4M3FN" if device_dtype == "fp8_e4m3" else device_dtype),
                "origin": origin,
            })
        output = row.pop("_output")
        uses = [use for use in output.uses if use.operation in op_index]
        consumers = []
        for use in sorted(uses, key=lambda use: op_index[use.operation]):
            consumer = use.operation
            consumers.append({
                "op_index": op_index[consumer],
                "op": consumer.name,
                "region_id": _str_attr(consumer, "prov.region_id") or None,
                "semantic": _str_attr(consumer, "prov.op") or None,
            })
        frontier = _frontier_consumers(output, op_index, region_to_partition,
                                       classes, row["partition_id"])
        row["abi"] = {
            "inputs": abi_inputs,
            "outputs": [{
                "name": "Y0",
                "capture_type": str(output.type),
                "capture_bytes": _bytes(output),
                "device_dtype": "bf16",
                "device_bytes": _bytes(output, "bf16"),
                "consumers": consumers,
                "frontier_consumers": frontier,
            }],
        }
        row["lifetime"] = {
            "definition_op_index": op_index[getattr(output, "owner")],
            "last_frontier_use_op_index": max(
                [consumer["op_index"] for consumer in frontier],
                default=row["capture_op_index"],
            ),
        }

    adjacency = defaultdict(set)
    for source, target in accelerator_edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen = set()
    islands = []
    for row in partitions:
        root = row["partition_id"]
        if root in seen:
            continue
        stack, members = [root], []
        seen.add(root)
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        members.sort(key=lambda value: int(value.removeprefix("atlas_p")))
        member_set = set(members)
        incoming = sorted({a for a, b in accelerator_edges if b in member_set and a not in member_set})
        outgoing = sorted({b for a, b in accelerator_edges if a in member_set and b not in member_set})
        islands.append({"island_id": f"atlas_i{len(islands):04d}",
                        "partitions": members, "partition_count": len(members),
                        "incoming_accelerator_partitions": incoming,
                        "outgoing_accelerator_partitions": outgoing})

    for row in partitions:
        row["structurally_lowerable"] = True
        row["capture_semantics_executable"] = False
        row["lowering_basis"] = (
            "structural contraction matched to an isolated FP8-input/BF16-output emitter"
        )
        row["missing_capture_bridge"] = (
            "calibrated f32-to-FP8 input/weight quantization and BF16-to-f32 output conversion"
        )
    return {
        "schema": "atlas_whole_capture_partition_plan_v1",
        "claim": "compile/partition plan only; no whole-model image or execution claim",
        "partition_count": len(partitions),
        "structurally_lowerable_partition_count": len(partitions),
        "capture_semantics_executable_partition_count": 0,
        "accelerator_dependency_edges": [
            {"source": source, "target": target}
            for source, target in sorted(accelerator_edges)
        ],
        "maximal_accelerator_island_count": len(islands),
        "maximal_accelerator_islands": islands,
        "partitions": partitions,
        "kernel_variants": sorted({row["kernel_id"] for row in partitions}),
        "kernel_variant_count": len({row["kernel_id"] for row in partitions}),
        "source_semantics": dict(sorted(Counter(row["source_semantic"] for row in partitions).items())),
    }
