"""Structural natural-loop domains for conflict-free constant-control input copies."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.llvm import ICmpPredicateFlag, LLVMArrayType
from xdsl.ir import Operation

from merlin.analysis.programs.host_cfg_index import prepare_host_cfg
from merlin.compiler.dialects.lowering.analyses import constant_integer

# ---- input_copy_loops --------------------------------------------------------------------------


@dataclass(frozen=True)
class CopyLoop:
    header: object
    latch: object
    nodes: frozenset
    exit: object
    start: int
    step: int
    trips: int
    update: object


def loop_axes(loops):
    """Disjoint sibling loops reuse their nesting-depth axis, never a product."""
    depths = {loop.header: sum(loop.nodes < parent.nodes for parent in loops) for loop in loops}
    sizes = tuple(
        max(loop.trips for loop in loops if depths[loop.header] == depth) for depth in range(max(depths.values()) + 1)
    )
    return depths, sizes


def region_cost(region):
    """Bound lane work per block, without charging absent nesting dimensions."""
    from math import prod

    _, loops, order = region
    depths, sizes = loop_axes(loops)
    return sum(
        len(tuple(block.ops))
        * prod(
            size
            for axis, size in enumerate(sizes)
            if any(block in loop.nodes and depths[loop.header] == axis for loop in loops)
        )
        for block in order
    )


def control_dependencies(region):
    """Exact SSA cone for branches/induction updates; body values confer no counts."""
    from xdsl.ir import BlockArgument

    root, loops, _ = region
    induction = {loop.header.args[0] for loop in loops}
    pending = [loop.update for loop in loops]
    pending.extend(block.last_op.operands[0] for block in root.nodes if block.last_op.name == "llvm.cond_br")
    seen, operations, external = set(), set(), set()
    predecessors = {block: [] for block in root.nodes}
    for block in root.nodes:
        for successor in block.last_op.successors:
            if successor in predecessors:
                predecessors[successor].append(block)
    while pending:
        value = pending.pop()
        if value in seen or value in induction:
            continue
        seen.add(value)
        if isinstance(value, BlockArgument) and value.owner in root.nodes:
            for block in predecessors[value.owner]:
                branch = block.last_op
                if branch.name == "llvm.br":
                    arguments = branch.operands
                else:
                    sizes = list(branch.properties["operandSegmentSizes"].get_values())
                    arguments = (
                        branch.operands[1 : 1 + sizes[1]]
                        if branch.successors[0] is value.owner
                        else branch.operands[1 + sizes[1] :]
                    )
                pending.append(arguments[value.index])
        elif isinstance(value.owner, Operation) and value.owner.parent_block() in root.nodes:
            operations.add(value.owner)
            pending.extend(value.owner.operands)
        else:
            external.add(value)
    return frozenset(operations), tuple(sorted(external, key=id))


def copy_loops(function):
    """Derive increasing post-tested loops; never infer a bound from observed counts.

    One integer IV is the only loop-carried value. All exits are the proven latch
    test, and no intermediate value escapes the loop. Memory effects and tensor
    ancestry are checked by the separate batching executor before admission.
    """
    cfg = prepare_host_cfg(function)
    found = {}
    for latch in cfg.blocks:
        for header in cfg.successors[latch]:
            if not cfg.dominance.dominates(header, latch):
                continue
            nodes, pending = {header}, [latch]
            while pending:
                block = pending.pop()
                if block in nodes:
                    continue
                nodes.add(block)
                pending.extend(cfg.predecessors[block])
            try:
                loop = _post_tested(header, latch, frozenset(nodes), cfg)
            except ValueError as exc:
                logging.getLogger(__name__).debug("copy batching declined; retain scalar proof: %s", exc)
                continue
            if header in found:
                raise ValueError("copy loop has ambiguous backedges")
            found[header] = loop
    return found, cfg


def _post_tested(header, latch, nodes, cfg):
    if len(header.args) != 1 or not isinstance(header.args[0].type, IntegerType):
        raise ValueError("copy loop requires one integer induction argument")
    induction = header.args[0]
    branch = latch.last_op
    if branch.name != "llvm.cond_br" or len(branch.successors) != 2:
        raise ValueError("copy loop needs a conditional latch")
    side = list(branch.successors).index(header)
    exit_block = branch.successors[1 - side]
    if exit_block in nodes or exit_block.args:
        raise ValueError("copy loop has an unsupported live-out")
    for block in nodes:
        if any(successor not in nodes for successor in cfg.successors[block]) and block is not latch:
            raise ValueError("copy loop can exit before its latch")
        if block is not header and any(pred not in nodes for pred in cfg.predecessors[block]):
            raise ValueError("copy loop has an internal entry")
    outside = cfg.predecessors[header] - nodes
    if len(outside) != 1 or cfg.predecessors[header] & nodes != {latch}:
        raise ValueError("copy loop lacks one preheader and one latch")
    initial = next(iter(outside)).last_op
    if initial.name == "llvm.br" and initial.successors[0] is header:
        initial_values = list(initial.operands)
    elif initial.name == "llvm.cond_br" and list(initial.successors).count(header) == 1:
        sizes = list(initial.properties["operandSegmentSizes"].get_values())
        initial_values = list(
            initial.operands[1 : 1 + sizes[1]] if initial.successors[0] is header else initial.operands[1 + sizes[1] :]
        )
    else:
        initial_values = []
    if len(initial_values) != 1:
        raise ValueError("copy loop initial edge is ambiguous")
    comparison = branch.operands[0].owner
    if not isinstance(comparison, Operation) or comparison.name != "llvm.icmp":
        raise ValueError("copy loop lacks an integer exit comparison")
    predicate = ICmpPredicateFlag.from_int(comparison.properties["predicate"].value.data).value
    sizes = list(branch.properties["operandSegmentSizes"].get_values())
    carried = list(branch.operands[1 : 1 + sizes[1]] if side == 0 else branch.operands[1 + sizes[1] :])
    if len(carried) != 1:
        raise ValueError("copy loop backedge does not carry one induction update")
    update_value = carried[0]
    update = update_value.owner
    if (
        not isinstance(update, Operation)
        or update.name != "llvm.add"
        or len(update.operands) != 2
        or list(update.operands).count(induction) != 1
        or update.parent_block() not in nodes
        or update_value.type != induction.type
    ):
        raise ValueError("copy loop lacks an exact induction update")
    tested = comparison.operands[0]
    if tested is not update_value and tested is not induction:
        raise ValueError("copy loop does not test its current or updated induction")
    start = constant_integer(initial_values[0])
    bound = constant_integer(comparison.operands[1])
    step = constant_integer(next(value for value in update.operands if value is not induction))
    if (
        any(type(value) is not int for value in (start, bound, step))
        or step <= 0
        or bound < start
        or tested is update_value
        and start == bound
    ):
        raise ValueError("copy loop needs increasing constant bounds")
    if (predicate, side) not in {("eq", 1), ("ne", 0), ("slt", 0), ("ult", 0)}:
        raise ValueError("copy loop exit predicate is unsupported")
    if predicate in {"eq", "ne"} and (bound - start) % step:
        raise ValueError("copy loop cannot reach its equality bound")
    trips = (bound - start + step - 1) // step + (tested is induction)
    bits = induction.type.width.data
    maximum = (1 << bits) - 1 if predicate == "ult" else (1 << (bits - 1)) - 1
    if not 1 <= bits <= 64 or start < 0 or start + trips * step > maximum:
        raise ValueError("copy loop induction could wrap")
    for block in nodes:
        if any(use.operation.parent_block() not in nodes for arg in block.args for use in arg.uses):
            raise ValueError("copy loop argument escapes the proven domain")
        for op in block.ops:
            if any(use.operation.parent_block() not in nodes for result in op.results for use in result.uses):
                raise ValueError("copy loop intermediate escapes the proven domain")
    return CopyLoop(header, latch, nodes, exit_block, start, step, trips, update_value)


def copy_regions(function):
    loops, cfg = copy_loops(function)
    roots = {}
    for header, loop in loops.items():
        nested = [value for value in loops.values() if value.nodes <= loop.nodes]
        if any(
            op.name in {"llvm.inline_asm", "llvm.call", "llvm.alloca", "llvm.mlir.addressof"}
            for block in loop.nodes
            for op in block.ops
        ):
            continue
            # Crossed loops and unrecognized backedges cannot become a rectangular domain.
        if any(a.nodes & b.nodes and not (a.nodes <= b.nodes or b.nodes <= a.nodes) for a in nested for b in nested):
            continue
        backedges = {(value.latch, value.header) for value in nested}
        pending = set(loop.nodes)
        order = []
        while pending:
            ready = [
                b
                for b in cfg.blocks
                if b in pending and not any(p in pending and (p, b) not in backedges for p in cfg.predecessors[b])
            ]
            if not ready:
                break
            order.extend(ready)
            pending.difference_update(ready)
        if not pending:
            roots[header] = (loop, sorted(nested, key=lambda row: -len(row.nodes)), tuple(order))
    return roots

    # Preserve historical object lookup; implementation and completion state live here.


# ---- input_copy_storage ------------------------------------------------------------------------
# Bounded byte origins and forwarding epochs for transport proofs; no tensor values.


@dataclass(frozen=True)
class Pointer:
    root: object
    offset: int = 0


@dataclass(frozen=True)
class Copy:
    origins: tuple
    references: tuple = ()


@dataclass(frozen=True)
class ReadReference:
    root: object
    offsets: object
    epochs: object
    active: object = True


class CopyStorage:
    """Exact input-byte offsets, zero padding and initialization/consumption epochs.

    -2 means uninitialized and -1 means zero. Nonnegative entries identify the
    original input, weight or decoded accelerator result byte. No tensor payload
    is loaded by this instrument. Source epochs discharge only into a tracked
    later copy, an accelerator input transfer or unchanged output publication.
    """

    def __init__(self, capacity):
        if sum(size for root, size in capacity.items() if root in (0, 1)) >= 1 << 63:
            raise ValueError("input byte-origin storage exceeds finite exact index range")
        self.capacity = capacity
        self.origins = {}
        self.consumed = {}
        self.epochs = {}
        self.next_epoch = 0
        self.output_base = (
            sum(capacity[root] for root in (0, 1)) if all(root in capacity for root in (0, 1, 2)) else None
        )
        self.output_written = None
        self.pending_results = set()

    def _allocate(self, root):
        if root not in self.origins:
            if sum(self.capacity[key] for key in self.origins) + self.capacity[root] > 64 << 20:
                raise ValueError("input scratch exceeds the aggregate finite storage proof budget")
            self.origins[root] = np.full(self.capacity[root], -2, dtype=np.int64)
            self.consumed[root] = np.zeros(self.capacity[root], dtype=np.bool_)
            self.epochs[root] = np.zeros(self.capacity[root], dtype=np.int64)

    def __contains__(self, key):
        root, offset = key
        return root in self.origins and self.origins[root][offset] != -2

    def __getitem__(self, key):
        if key not in self:
            raise ValueError("input preparation reads uninitialized scratch")
        root, offset = key
        value = int(self.origins[root][offset])
        if value == -1:
            return "zero"
        if self.output_base is not None and value >= self.output_base:
            return 2, value - self.output_base
        base = self.capacity.get(0)
        return (1, value - base) if base is not None and value >= base else (0, value)

    def __setitem__(self, key, origin):
        root, offset = key
        if origin != "zero" and origin[0] == 1 and 0 not in self.capacity:
            raise ValueError("weight byte origins require the original input capacity")
        if origin != "zero" and origin[0] == 2 and self.output_base is None:
            raise ValueError("output byte origins require the original ABI capacities")
        self._allocate(root)
        if self.origins[root][offset] >= 0 and not self.consumed[root][offset]:
            raise ValueError("input scratch copy is overwritten before accelerator consumption")
        self.origins[root][offset] = (
            -1
            if origin == "zero"
            else origin[1] + (self.output_base if origin[0] == 2 else self.capacity[0] if origin[0] == 1 else 0)
        )
        self.consumed[root][offset] = False
        self.next_epoch += 1
        self.epochs[root][offset] = self.next_epoch

    def store_batch(self, root, offsets, origins, written, read=None):
        self._allocate(root)
        if np.any(offsets < 0) or np.any(offsets >= self.capacity[root]) or len(np.unique(offsets)) != len(offsets):
            raise ValueError("batched copy has out-of-bounds or overlapping writes")
        if root not in written:
            written[root] = np.zeros(self.capacity[root], dtype=np.bool_)
        if np.any(written[root][offsets]):
            raise ValueError("batched copy reorders conflicting scratch writes")
        if read is not None and root in read and np.any(read[root][offsets]):
            raise ValueError("batched copy reorders conflicting scratch reads and writes")
        if np.any((self.origins[root][offsets] >= 0) & ~self.consumed[root][offsets]):
            raise ValueError("input scratch copy is overwritten before accelerator consumption")
        written[root][offsets] = True
        self.origins[root][offsets] = origins
        self.consumed[root][offsets] = False
        self.next_epoch += 1
        self.epochs[root][offsets] = self.next_epoch

    def read_batch(self, root, offsets, written=None, read=None):
        if root in self.pending_results:
            raise ValueError("CPU reads an accelerator result before the command fence")
        if root not in self.origins or np.any(self.origins[root][offsets] == -2):
            raise ValueError("input preparation reads uninitialized scratch")
        if written is not None and root in written and np.any(written[root][offsets]):
            raise ValueError("batched copy reorders conflicting scratch reads and writes")
        if read is not None:
            if root not in read:
                read[root] = np.zeros(self.capacity[root], dtype=np.bool_)
            read[root][offsets] = True
        return self.origins[root][offsets].copy(), ReadReference(
            root, np.asarray(offsets).copy(), self.epochs[root][offsets].copy()
        )

    def forward(self, reference):
        """Discharge a source epoch only after an unchanged tracked destination exists.

        Every destination has its own consumption obligation. Forwarding edges
        point to later writes, so all finite chains must end at a decoded input
        transfer or at publication of accelerator result bytes to the output.
        A dead load, a stale epoch and an overwritten destination prove nothing.
        """
        if np.any(self.epochs[reference.root][reference.offsets] != reference.epochs):
            raise ValueError("scratch copy refers to a changed source write epoch")
        self.consumed[reference.root][reference.offsets] = True

    def publish_output(self, offsets, origins):
        if (
            self.output_base is None
            or np.any(origins < self.output_base)
            or np.any(offsets < 0)
            or np.any(offsets >= self.capacity[2])
            or len(np.unique(offsets)) != len(offsets)
        ):
            raise ValueError("CPU output store is not unchanged bounded accelerator result bytes")
        if self.output_written is None:
            self.output_written = np.zeros(self.capacity[2], dtype=np.bool_)
        if np.any(self.output_written[offsets]):
            raise ValueError("output publication repeats a destination byte")
        self.output_written[offsets] = True

    def result_rows(self, root, offset, rows, width, stride):
        """Record actual decoded result writes; no numerical result is invented."""
        if self.output_base is None or rows * width > 64 << 20:
            raise ValueError("accelerator result exceeds bounded storage proof")
        offsets = (offset + np.arange(rows)[:, None] * stride + np.arange(width)).ravel()
        if root == 2:
            self.publish_output(offsets, np.full(len(offsets), self.output_base, dtype=np.int64))
        else:
            self.store_batch(root, offsets, self.output_base + offsets, {})
            self.pending_results.add(root)

    def consume(self, root, offsets):
        if root not in self.origins or np.any(self.origins[root][offsets] == -2):
            raise ValueError("accelerator consumes uninitialized input scratch")
        if self.output_base is not None and np.any(self.origins[root][offsets] >= self.output_base):
            raise ValueError("accelerator input contains output-result bytes")
        self.consumed[root][offsets] = True

    def consume_rows(self, root, offset, rows, width, stride):
        """Consume the union of bounded row spans, without expanding repeated rows."""
        if root not in self.origins:
            raise ValueError("accelerator consumes uninitialized input scratch")
        if stride <= width:
            end = offset + (rows - 1) * stride + width
            self.consume(root, slice(offset, end))
            return
        for start in range(offset, offset + rows * stride, stride):
            self.consume(root, slice(start, start + width))

    def require_consumed(self):
        for root, origins in self.origins.items():
            if np.any((origins >= 0) & ~self.consumed[root]):
                raise ValueError("input copy scratch never feeds decoded accelerator input")


def scratch_limit(buffers):
    """Finite resource limit from the original logical convolution's patch extent."""
    weights, output = buffers[1]["shape"], buffers[2]["shape"]
    patches = 0
    if (
        len(weights) == len(output) == 2
        and weights[-1] == output[-1]
        and all(type(n) is int and n > 0 for n in [*weights, *output])
    ):
        patches = output[0] * weights[0]
    return min(64 << 20, max(1 << 20, patches))


def byte_gep_stride(typ):
    """Byte and nested byte-array indexing has no target-dependent padding."""
    stride = 1
    while isinstance(typ, LLVMArrayType):
        if not 0 < typ.size.data <= 1 << 24:
            raise ValueError("input preparation has unbounded aggregate pointer layout")
        stride *= typ.size.data
        typ = typ.type
    if typ != IntegerType(8) or stride > 1 << 24:
        raise ValueError("input preparation has unsupported pointer layout")
    return stride

    # Preserve historical object lookup; implementation and completion state live here.
