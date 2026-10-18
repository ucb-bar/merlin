"""Exhaustive, vectorized byte-origin proof of independent constant copy iterations.

Rectangular loop domains come from the actual CFG. Every selected lane executes
the original integer instructions and pointer checks. Conflicting reads/writes,
escaping loop results and opaque effects refuse batching, so executing
iterations together cannot change a data dependency or a write's final owner.
Scratch forwarding requires a disjoint read/write set and exact source epochs.
"""

from collections import Counter
from dataclasses import dataclass
from math import prod

import numpy as np
from xdsl.dialects.builtin import IntegerType

from merlin.perf.constant_control_cfg import MAX_OPERATIONS
from merlin.perf.host_cfg_activity import _INTEGER, _payload_bytes
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer

from .input_copy_integer import integer
from .input_copy_loops import control_dependencies, loop_axes, region_cost
from .input_copy_storage import Pointer, ReadReference

MAX_LANES = 1 << 18


@dataclass(frozen=True)
class _Address:
    root: object
    offsets: object


@dataclass(frozen=True)
class _Bytes:
    origins: tuple
    references: tuple = ()


def execute(
    region, values, incoming, memory, capacity, readonly, *, argument_alignment=None,
    argument_alignments=None, control_only=False,
):
    """Return exact dynamic block/copy counts and the unique loop exit."""
    root, loops, order = region
    if incoming != (root.start,):
        raise ValueError("batched copy initial edge differs from original constant bound")
    if len(loops) > 128 or sum(len(tuple(block.ops)) for block in order) > 10000:
        raise ValueError("batched copy static domain exceeds finite proof limits")
    _, sizes = loop_axes(loops)
    if len(sizes) > 8:
        raise ValueError("batched copy loop nesting exceeds finite proof limits")
    if region_cost(region) > MAX_OPERATIONS:
        raise ValueError("batched copy exceeds finite exhaustive operation budget")
    if loops[0] is not root:
        raise ValueError("batched copy needs its original outer iteration domain")
    inner = prod(sizes[1:])
    if inner > MAX_LANES:
        raise ValueError("batched copy inner domain exceeds finite lane budget")
    batch = max(1, MAX_LANES // inner)
    totals = Counter()
    counts = Counter()
    written = {}
    read = {}
    for start in range(0, sizes[0], batch):
        stop = min(start + batch, sizes[0])
        shape = (stop - start, *sizes[1:])
        result = _execute_chunk(
            root, loops, order, shape, start, values, memory, capacity, readonly, written, read,
            argument_alignment, argument_alignments,
            control_only,
        )
        counts.update(result[0])
        totals.update(result[1])
    return root.exit, counts, totals


def _execute_chunk(
    root, loops, order, shape, start, original, memory, capacity, readonly, written, read,
    argument_alignment, argument_alignments,
    control_only,
):
    depths, _ = loop_axes(loops)
    values = {key: _Address(value.root, value.offset) if isinstance(value, Pointer) else value
              for key, value in original.items()}
    edges = {(root.header, None): (True, ())}
    backedges = {(loop.latch, loop.header) for loop in loops}
    counts = Counter()
    totals = Counter()
    control_ops = control_dependencies((root, loops, order))[0] if control_only else None

    def domain(block):
        return tuple(size if any(block in loop.nodes and depths[loop.header] == axis for loop in loops) else 1
                     for axis, size in enumerate(shape))

    def owners(block):
        return tuple(next((loop.header for loop in loops if block in loop.nodes and depths[loop.header] == axis), None)
                     for axis in range(len(shape)))

    def selected(value, mask):
        return np.broadcast_to(value, np.broadcast_shapes(np.shape(value), np.shape(mask)))[
            np.broadcast_to(mask, np.broadcast_shapes(np.shape(value), np.shape(mask)))
        ]

    def bound(address, width, mask):
        if not isinstance(address, _Address):
            raise ValueError("batched copy has unknown pointer ancestry")
        offsets = selected(address.offsets, mask)
        if (np.any(offsets < 0) or np.any(offsets > capacity[address.root] - width)):
            raise ValueError("batched input copy is outside original bounded storage")
        return offsets

    def merge(parts, field=None):
        value = None
        for mask, candidate in parts:
            if field is not None:
                candidate = getattr(candidate, field)
            if value is None:
                value = np.where(mask, candidate, 0)
            else:
                value = np.where(mask, candidate, value)
        return value

    for block in order:
        incoming = [(mask, args) for (destination, _), (mask, args) in edges.items() if destination is block]
        if not incoming:
            continue
        block_shape = domain(block)
        masks = [np.broadcast_to(mask, block_shape) for mask, _ in incoming]
        mask = np.logical_or.reduce(masks)
        count = int(np.count_nonzero(mask))
        counts[block] += count
        if not count:
            continue
        totals["proof_operations"] += len(tuple(block.ops))
        header_loop = next((loop for loop in loops if loop.header is block), None)
        if header_loop is not None:
            axis = depths[header_loop.header]
            begin = start if axis == 0 else 0
            iv_shape = [1] * len(shape)
            iv_shape[axis] = shape[axis]
            valid = (np.arange(begin, begin + shape[axis]) < header_loop.trips).reshape(iv_shape)
            mask = mask & valid
            counts[block] = counts[block] - count + int(np.count_nonzero(mask))
            count = int(np.count_nonzero(mask))
            values[block.args[0]] = (
                header_loop.start + np.arange(begin, begin + shape[axis], dtype=np.int64) * header_loop.step
            ).reshape(iv_shape)
        else:
            if any(len(args) != len(block.args) for _, args in incoming):
                raise ValueError("batched copy phi arguments differ from original CFG")
            for index, argument in enumerate(block.args):
                parts = [(m, args[index]) for m, (_, args) in zip(masks, incoming, strict=True)]
                candidates = [v for _, v in parts]
                if control_only and any(value is None for value in candidates):
                    values[argument] = None
                elif (any(isinstance(value, _Bytes) for value in candidates)
                        and all(isinstance(value, _Bytes) or type(value) is int and value == 0
                                for value in candidates)):
                    width = _payload_bytes(argument.type)
                    if width is None or any(isinstance(v, _Bytes) and len(v.origins) != width for v in candidates):
                        raise ValueError("batched byte phi changes payload width")
                    references = tuple(
                        ReadReference(reference.root, reference.offsets, reference.epochs, m & reference.active)
                        for m, v in parts if isinstance(v, _Bytes) for reference in v.references
                    )
                    if len(references) > 256:
                        raise ValueError("batched scratch forwarding phi exceeds finite provenance bound")
                    values[argument] = _Bytes(tuple(merge([
                        (m, v.origins[i] if isinstance(v, _Bytes) else -1) for m, v in parts
                    ]) for i in range(width)), references)
                elif all(isinstance(value, _Address) for value in candidates):
                    if len({value.root for value in candidates}) != 1:
                        raise ValueError("batched pointer phi mixes storage roots")
                    values[argument] = _Address(candidates[0].root, merge(parts, "offsets"))
                elif all(isinstance(value, (int, np.ndarray, np.integer)) for value in candidates):
                    values[argument] = merge(parts)
                else:
                    raise ValueError("batched phi mixes byte and scalar provenance")
        for op in block.ops:
            args = [values.get(value) for value in op.operands] if control_only else [values[v] for v in op.operands]
            name = op.op_name.data if op.name == "builtin.unregistered" else op.name
            value = None
            pure_control = name in _INTEGER or name in {
                "llvm.mlir.constant", "llvm.icmp", "llvm.trunc", "llvm.sext", "llvm.zext", "llvm.select"
            }
            if control_only and name not in {"llvm.br", "llvm.cond_br"} and (
                op not in control_ops or not pure_control
            ):
                pass  # Memory and opaque results stay unknown; this path grants only CFG counts.
            elif name == "llvm.mlir.constant":
                value = constant_integer(op.results[0])
                if value is None and not control_only:
                    raise ValueError("batched copy contains noninteger constant")
            elif name in _INTEGER or name in {"llvm.icmp", "llvm.trunc", "llvm.sext", "llvm.zext", "llvm.select"}:
                value = None if control_only and any(v is None for v in args) else integer(op, args, mask)
                if name in _INTEGER:
                    totals["arithmetic"] += count
                elif name in {"llvm.trunc", "llvm.sext", "llvm.zext"}:
                    totals["conversions"] += count
            elif name == "llvm.getelementptr":
                indices = list(op.properties["rawConstantIndices"].get_values())
                if (len(indices) != 1 or op.properties["elem_type"] != IntegerType(8)
                        or not isinstance(args[0], _Address)):
                    raise ValueError("batched copy has an unproved pointer layout")
                offset = args[1] if indices[0] == -(1 << 31) else indices[0]
                if not isinstance(offset, (int, np.ndarray, np.integer)):
                    raise ValueError("batched pointer offset depends on tensor data")
                # Address formation itself may not overflow the selected register.
                pointer = args[0]
                a, b = selected(pointer.offsets, mask), selected(offset, mask)
                if int(a.min()) + int(b.min()) < -(1 << 63) or int(a.max()) + int(b.max()) >= 1 << 63:
                    raise ValueError("batched copy address arithmetic could wrap")
                value = _Address(pointer.root, pointer.offsets + offset)
            elif name == "llvm.load":
                width = _payload_bytes(op.results[0].type)
                if type(width) is not int or not 1 <= width <= 16:
                    raise ValueError("batched copy requires bounded fixed-width loads")
                address = args[0]
                bound(address, width, mask)
                if address.root in {0, 1}:
                    base = capacity[0] if address.root == 1 else 0
                    value = _Bytes(tuple(address.offsets + base + i for i in range(width)))
                elif address.root in readonly:
                    value = _Bytes((-1,) * width)
                elif address.root == 2:
                    raise ValueError("batched copy may not read the model output")
                else:
                    byte_origins, references = [], []
                    grid_shape = np.broadcast_shapes(np.shape(address.offsets), np.shape(mask))
                    grid_mask = np.broadcast_to(mask, grid_shape)
                    for i in range(width):
                        offsets = np.broadcast_to(address.offsets, grid_shape) + i
                        origins, reference = memory.read_batch(
                            address.root, offsets[grid_mask], written, read
                        )
                        grid = np.full(grid_shape, -1, dtype=np.int64)
                        epochs = np.zeros(grid_shape, dtype=np.int64)
                        grid[grid_mask], epochs[grid_mask] = origins, reference.epochs
                        byte_origins.append(grid)
                        references.append(ReadReference(address.root, offsets, epochs))
                    value = _Bytes(tuple(byte_origins), tuple(references))
                totals["payload_bytes"] += count * width
            elif name == "llvm.store":
                byte, address = args
                width = _payload_bytes(op.operands[0].type)
                if type(width) is not int or not 1 <= width <= 16:
                    raise ValueError("batched copy requires bounded fixed-width stores")
                offsets = bound(address, width, mask)
                if address.root in {0, 1} or address.root in readonly:
                    raise ValueError("batched copy writes model input, weight or read-only storage")
                if isinstance(byte, _Bytes):
                    if len(byte.origins) != width:
                        raise ValueError("batched input copy changes payload width")
                    origins = np.stack([selected(value, mask) for value in byte.origins], axis=1).ravel()
                elif isinstance(byte, (int, np.ndarray, np.integer)) and np.all(selected(byte, mask) == 0):
                    origins = np.full(len(offsets) * width, -1, dtype=np.int64)
                else:
                    raise ValueError("batched scratch store is not an unchanged input byte or zero")
                byte_offsets = (offsets[:, None] + np.arange(width)[None, :]).ravel()
                if address.root == 2:
                    memory.publish_output(byte_offsets, origins)
                else:
                    memory.store_batch(address.root, byte_offsets, origins, written, read)
                for reference in byte.references if isinstance(byte, _Bytes) else ():
                    source_mask = mask & reference.active
                    memory.forward(ReadReference(
                        reference.root, selected(reference.offsets, source_mask), selected(reference.epochs, source_mask)
                    ))
                totals["payload_bytes"] += count * width
            elif name == "llvm.intr.assume":
                tags, sizes = op.properties.get("op_bundle_tags"), op.properties.get("op_bundle_sizes")
                if (len(args) != 3 or op.operands[0].type != IntegerType(1)
                        or not isinstance(args[0], (int, np.ndarray, np.integer))
                        or np.any(selected(args[0], mask) == 0) or not isinstance(args[1], _Address)
                        or tags is None or sizes is None or [tag.data for tag in tags] != ["align"]
                        or list(sizes.get_values()) != [2] or type(args[2]) is not int):
                    raise ValueError("batched copy has an unproved assumption")
                pointer, alignment = args[1], args[2]
                declared = argument_alignment if pointer.root in {0, 1, 2} else getattr(
                    getattr(getattr(pointer.root, "alignment", None), "value", None), "data", None
                )
                if pointer.root in {0, 1, 2} and argument_alignments is not None:
                    declared = argument_alignments[pointer.root]
                if (type(declared) is not int or alignment <= 0 or alignment & (alignment - 1)
                        or declared % alignment or np.any(selected(pointer.offsets, mask) % alignment)):
                    raise ValueError("batched alignment assumption lacks original storage evidence")
            elif name in {"llvm.br", "llvm.cond_br"}:
                if name == "llvm.br":
                    outgoing = [(op.successors[0], mask, tuple(args))]
                else:
                    if not isinstance(args[0], (int, np.ndarray, np.integer)):
                        raise ValueError("batched branch depends on tensor data")
                    sizes = list(op.properties["operandSegmentSizes"].get_values())
                    condition = np.asarray(args[0]) != 0
                    outgoing = [(op.successors[0], mask & condition, tuple(args[1:1 + sizes[1]])),
                                (op.successors[1], mask & ~condition, tuple(args[1 + sizes[1]:]))]
                for successor, outgoing_mask, parameters in outgoing:
                    if (block, successor) in backedges:
                        continue
                    if successor not in root.nodes:
                        if successor is not root.exit or parameters:
                            raise ValueError("batched copy has an unproved exit")
                        continue
                    for axis, (before, after) in enumerate(zip(owners(block), owners(successor), strict=True)):
                        if before is not None and before is not after:
                            if parameters:
                                raise ValueError("batched copy has live values leaving a nested loop")
                            outgoing_mask = np.any(outgoing_mask, axis=axis, keepdims=True)
                    if (successor, block) in edges:
                        raise ValueError("batched copy repeats an ambiguous CFG edge")
                    edges[successor, block] = outgoing_mask, parameters
            else:
                raise ValueError(f"batched copy contains unsupported effect {name}")
            if op.results:
                if len(op.results) != 1 or value is None and not control_only:
                    raise ValueError("batched copy result has unknown provenance")
                values[op.results[0]] = value
    return counts, totals
