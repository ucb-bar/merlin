"""Bounded count-only domains for pretested loops with constant internal branches.

Only the innermost natural loop is summarized. Its body is acyclic after removing
the unique backedge, and no value escapes. The shared masked integer interpreter
still evaluates every selected control lane; opaque results remain unknown.
These regions confer no memory, arithmetic, instruction or effect permission.
"""

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerType

from merlin.perf.host_cfg_activity import _constant_loop
from merlin.perf.host_cfg_index import prepare_host_cfg


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
    found = {}
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
        # A nested/unrecognized backedge is not an acyclic iteration body.
        if {(a, b) for a, b in backedges if a in nodes and b in nodes} != {(latch, header)}:
            continue
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
        pending, order = set(nodes), []
        while pending:
            ready = [
                block
                for block in cfg.blocks
                if block in pending
                and not any(pred in pending and (pred, block) != (latch, header) for pred in cfg.predecessors[block])
            ]
            if not ready:
                break
            order.extend(ready)
            pending.difference_update(ready)
        if pending:
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
        if loop.exit.args:
            continue
        found[header] = (loop, (loop,), tuple(order))
    return found
