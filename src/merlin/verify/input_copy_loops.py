"""Structural natural-loop domains for conflict-free constant-control input copies."""

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.llvm import ICmpPredicateFlag
from xdsl.ir import Operation

from merlin.perf.host_cfg_index import prepare_host_cfg
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


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
    sizes = tuple(max(loop.trips for loop in loops if depths[loop.header] == depth)
                  for depth in range(max(depths.values()) + 1))
    return depths, sizes


def region_cost(region):
    """Bound lane work per block, without charging absent nesting dimensions."""
    from math import prod

    _, loops, order = region
    depths, sizes = loop_axes(loops)
    return sum(len(tuple(block.ops)) * prod(
        size for axis, size in enumerate(sizes)
        if any(block in loop.nodes and depths[loop.header] == axis for loop in loops)
    ) for block in order)


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
                    arguments = (branch.operands[1:1 + sizes[1]] if branch.successors[0] is value.owner
                                 else branch.operands[1 + sizes[1]:])
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
            except ValueError:
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
        initial_values = list(initial.operands[1:1 + sizes[1]] if initial.successors[0] is header
                              else initial.operands[1 + sizes[1]:])
    else:
        initial_values = []
    if len(initial_values) != 1:
        raise ValueError("copy loop initial edge is ambiguous")
    comparison = branch.operands[0].owner
    if not isinstance(comparison, Operation) or comparison.name != "llvm.icmp":
        raise ValueError("copy loop lacks an integer exit comparison")
    predicate = ICmpPredicateFlag.from_int(comparison.properties["predicate"].value.data).value
    sizes = list(branch.properties["operandSegmentSizes"].get_values())
    carried = list(branch.operands[1:1 + sizes[1]] if side == 0 else branch.operands[1 + sizes[1]:])
    if len(carried) != 1:
        raise ValueError("copy loop backedge does not carry one induction update")
    update_value = carried[0]
    update = update_value.owner
    if (not isinstance(update, Operation) or update.name != "llvm.add"
            or len(update.operands) != 2 or list(update.operands).count(induction) != 1
            or update.parent_block() not in nodes or update_value.type != induction.type):
        raise ValueError("copy loop lacks an exact induction update")
    tested = comparison.operands[0]
    if tested is not update_value and tested is not induction:
        raise ValueError("copy loop does not test its current or updated induction")
    start = constant_integer(initial_values[0])
    bound = constant_integer(comparison.operands[1])
    step = constant_integer(next(value for value in update.operands if value is not induction))
    if (any(type(value) is not int for value in (start, bound, step)) or step <= 0
            or bound < start or tested is update_value and start == bound):
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
        if any(op.name in {"llvm.inline_asm", "llvm.call", "llvm.alloca", "llvm.mlir.addressof"}
               for block in loop.nodes for op in block.ops):
            continue
        # Crossed loops and unrecognized backedges cannot become a rectangular domain.
        if any(a.nodes & b.nodes and not (a.nodes <= b.nodes or b.nodes <= a.nodes)
               for a in nested for b in nested):
            continue
        backedges = {(value.latch, value.header) for value in nested}
        pending = set(loop.nodes)
        order = []
        while pending:
            ready = [b for b in cfg.blocks if b in pending and not any(
                p in pending and (p, b) not in backedges for p in cfg.predecessors[b]
            )]
            if not ready:
                break
            order.extend(ready)
            pending.difference_update(ready)
        if not pending:
            roots[header] = (loop, sorted(nested, key=lambda row: -len(row.nodes)), tuple(order))
    return roots
