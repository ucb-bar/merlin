"""Bounded count-only domains for pretested loops with constant internal branches.

Every nested natural loop must be independently proved. Its body is acyclic
after removing exactly those backedges, and no value escapes. The masked integer
interpreter still evaluates every selected control lane; opaque results remain unknown.
These regions confer no memory, arithmetic, instruction or effect permission.
"""

from dataclasses import dataclass
from math import prod

from xdsl.dialects.builtin import IntegerType

from merlin.perf.host_cfg_activity import _constant_loop
from merlin.perf.host_cfg_index import prepare_host_cfg
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


@dataclass(frozen=True)
class _ControlLoop:
    header: object
    latch: object
    nodes: frozenset
    exit: object
    start: int
    step: int
    trips: int  # Header visits, including the final false condition.
    update: object
    count_only: bool = True


def control_regions(function):
    """Derive closed pretested rectangular domains without trusting annotations."""
    cfg = prepare_host_cfg(function)
    positions = {block: index for index, block in enumerate(cfg.blocks)}
    backedges = {
        (latch, header)
        for latch in cfg.blocks
        for header in cfg.successors[latch]
        if cfg.dominance.dominates(header, latch)
    }
    loops = {}
    for latch, header in backedges:
        if latch is header or len(header.args) != 1 or len(tuple(header.ops)) != 2:
            continue
        if not isinstance(header.args[0].type, IntegerType):
            continue
        nodes, pending = {header}, [latch]
        while pending:
            block = pending.pop()
            if block in nodes:
                continue
            nodes.add(block)
            pending.extend(cfg.predecessors[block])
        if any(block is not header and cfg.predecessors[block] - nodes for block in nodes):
            continue
        try:
            row, _ = _constant_loop(header, latch, nodes, cfg.predecessors, positions)
        except ValueError:
            continue
        if row.get("other_carried_values") or not 1 <= header.args[0].type.width.data <= 64:
            continue
        # The vector integer owner uses exact signed-host integers. Wider
        # unsigned domains remain available to the ordinary scalar interpreter.
        final = row["initial"] + row["trip_count"] * row["step"]
        if not 0 <= row["initial"] <= final < 1 << (header.args[0].type.width.data - 1):
            continue
        if any(
            use.operation.parent_block() not in nodes
            for block in nodes
            for value in (*block.args, *(r for op in block.ops for r in op.results))
            for use in value.uses
        ):
            continue
        loop = _ControlLoop(
            header,
            latch,
            frozenset(nodes),
            header.last_op.successors[1],
            row["initial"],
            row["step"],
            row["trip_count"] + 1,
            latch.last_op.operands[0],
        )
        if not loop.exit.args:
            loops[header] = loop

    from merlin.perf.constant_control_cfg import MAX_BLOCK_VISITS, MAX_COUNT_CACHE_KEYS, MAX_OPERATIONS
    from merlin.verify.input_copy_loops import control_dependencies, region_cost

    structural = {}
    for header, loop in loops.items():
        nested = sorted(
            (child for child in loops.values() if child.nodes <= loop.nodes), key=lambda child: -len(child.nodes)
        )
        edges = {(child.latch, child.header) for child in nested}
        if edges != {(a, b) for a, b in backedges if a in loop.nodes and b in loop.nodes}:
            continue  # An unsupported inner cycle cannot become a finite domain.
        if any(a.nodes & b.nodes and not (a.nodes <= b.nodes or b.nodes <= a.nodes) for a in nested for b in nested):
            continue
        pending, order = set(loop.nodes), []
        while pending:
            ready = [
                block
                for block in cfg.blocks
                if block in pending
                and not any(pred in pending and (pred, block) not in edges for pred in cfg.predecessors[block])
            ]
            if not ready:
                break
            order.extend(ready)
            pending.difference_update(ready)
        if pending:
            continue
        structural[header] = (loop, tuple(nested), tuple(order))
    leaves = {header: region for header, region in structural.items() if len(region[1]) == 1}
    if _leaf_trace_fits(cfg, loops, backedges, leaves, MAX_BLOCK_VISITS) is not False:
        return leaves
    found = dict(leaves)
    for header, region in structural.items():
        loop = region[0]
        if len(region[1]) == 1:
            continue  # Preserve the original leaf selection, including derived external keys.
        # Bound all distinct external-control keys, not only one invocation's
        # lanes. Otherwise a large outer region can consume the whole budget
        # before a cheaper inner region gets a chance. Unrecognized keys keep
        # the ordinary scalar fallback; this is selection, never count credit.
        external = control_dependencies(region)[1]
        varying = {value for value in external if constant_integer(value) is None}
        owners = {parent.header.args[0]: parent for parent in loops.values() if loop.nodes < parent.nodes}
        if varying - owners.keys():
            continue
        key_bound = prod(owners[value].trips for value in varying)
        if key_bound > MAX_COUNT_CACHE_KEYS or region_cost(region) * key_bound > MAX_OPERATIONS:
            continue
        found[header] = region
    return found


def _leaf_trace_fits(cfg, loops, backedges, leaves, limit):
    """Prefer the original leaf path when its conservative scalar visits fit.

    This is a CFG/domain bound, never a scalar retrace. Unknown cycles retain
    the old path. Count every acyclic branch arm, all final-false headers and
    one entry per summarized leaf; omitted leaf bodies cost no scalar visits.
    """
    if {(loop.latch, loop.header) for loop in loops.values()} != backedges:
        return None
    pending = set(cfg.blocks)
    while pending:
        ready = {
            block
            for block in pending
            if not any(pred in pending and (pred, block) not in backedges for pred in cfg.predecessors[block])
        }
        if not ready:
            return None  # Irreducible/unrecognized cycles have no finite forest bound.
        pending.difference_update(ready)
    if any(
        a.header is not b.header and a.nodes & b.nodes and not (a.nodes < b.nodes or b.nodes < a.nodes)
        for a in loops.values()
        for b in loops.values()
    ):
        return None
    total = 0
    for block in cfg.blocks:
        if any(block in region[0].nodes and block is not header for header, region in leaves.items()):
            continue
        total += prod(
            loop.trips if block is loop.header else loop.trips - 1
            for loop in loops.values()
            if block in loop.nodes and not (block is loop.header and block in leaves)
        )
        if total > limit:
            return False
    return True
