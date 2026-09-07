"""Structural task accounting for LLVM control-flow graphs, including source-native loops.

Control flow, dominance and task ordering are checked here. This does not prove arithmetic, loop
trip-count equivalence, termination, hardware ordering or performance. Those obligations remain
separate from a source/task ownership receipt.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


def analyze_task_cfg(function: Any, task_ids: Sequence[int]) -> dict[str, Any]:
    """Check owned CFG operations and mandatory task coverage on every returning path."""
    from xdsl.ir import Block
    from xdsl.irdl.dominance import DominanceInfo

    blocks = list(function.body.blocks)
    problems: set[str] = set()
    if not blocks:
        return {"status": "refused", "problems": ["kernel has no entry block"]}
    entry = blocks[0]
    block_set = set(blocks)
    successors = {}
    predecessors = {block: set() for block in blocks}
    for block in blocks:
        terminal = block.last_op
        if terminal is None or terminal.name not in {"llvm.br", "llvm.cond_br", "llvm.return"}:
            problems.add("kernel block has no supported control-flow terminator")
            successors[block] = ()
            continue
        successors[block] = tuple(terminal.successors)
        for successor in terminal.successors:
            if successor not in block_set:
                problems.add("kernel branch leaves its function")
            else:
                predecessors[successor].add(block)
    if problems:
        return {"status": "refused", "problems": sorted(problems)}
    reachable = set()
    pending = [entry]
    while pending:
        block = pending.pop()
        if block not in reachable:
            reachable.add(block)
            pending.extend(successors[block])
    if reachable != block_set:
        problems.add("owned kernel contains unreachable blocks")
    returns = [block for block in blocks if block.last_op.name == "llvm.return"]
    if not returns:
        problems.add("kernel has no returning path")
    dominators = DominanceInfo(function.body)
    positions = {op: index for block in blocks for index, op in enumerate(block.ops)}

    # Hoisted definitions do not execute a task's computation. Only constants, entry-address
    # plumbing and statically sized allocations qualify. In particular loads are never shared.
    constants = set()
    addresses = set(entry.args)
    shared = set()
    for operation in entry.ops:
        operands = tuple(operation.operands)
        if operation.name == "llvm.mlir.constant":
            constants.update(operation.results)
            shared.add(operation)
        elif operation.name == "llvm.ptrtoint" and operands and operands[0] in addresses:
            addresses.update(operation.results)
            shared.add(operation)
        elif operation.name == "llvm.add" and len(operands) == 2 and (
                (operands[0] in addresses and operands[1] in constants)
                or (operands[1] in addresses and operands[0] in constants)):
            addresses.update(operation.results)
            shared.add(operation)
        elif operation.name == "llvm.alloca" and operands and all(v in constants for v in operands):
            addresses.update(operation.results)
            shared.add(operation)

    counts: Counter[int] = Counter()
    shared_counts = Counter(op.name for op in shared)
    block_tasks = {}
    first_last = {}
    allowed_ids = {-2, -1, *task_ids}
    for block in blocks:
        order = []
        owners = set()
        for operation in block.ops:
            if operation.regions:
                problems.add("nested kernel regions require additional task ownership proof")
            for operand in operation.operands:
                definition = operand.owner
                definition_block = definition if isinstance(definition, Block) else definition.parent_block()
                if definition_block not in block_set:
                    problems.add("kernel operand has a definition outside the kernel CFG")
                elif definition_block is block and not isinstance(definition, Block):
                    if positions[definition] >= positions[operation]:
                        problems.add("kernel SSA definition does not precede its use")
                elif not dominators.dominates(definition_block, block):
                    problems.add("kernel SSA definition does not dominate its use")
            if operation in shared or operation.name == "llvm.return":
                continue
            attr = operation.attributes.get("merlin.global_task")
            owner = getattr(getattr(attr, "value", None), "data", None)
            if isinstance(owner, bool) or not isinstance(owner, int) or owner not in allowed_ids:
                problems.add(f"kernel operation lacks task ownership: {operation.name}")
                continue
            counts[owner] += 1
            if owner >= 0:
                owners.add(owner)
            order.append(len(task_ids) if owner == -2 else owner)
        if any(right < left for left, right in zip(order, order[1:])):
            problems.add("kernel block reverses scheduled task order")
        block_tasks[block] = owners
        first_last[block] = (order[0], order[-1]) if order else None

    # Propagate last executed task through empty control-flow blocks, not textual block order.
    last_out = {block: -1 for block in blocks}
    changed = True
    while changed:
        changed = False
        for block in blocks:
            incoming = max((last_out[p] for p in predecessors[block]), default=-1)
            span = first_last[block]
            if span and incoming > span[0]:
                problems.add("kernel control-flow edge reverses scheduled task order")
            outgoing = max(incoming, span[1] if span else -1)
            if outgoing != last_out[block]:
                last_out[block] = outgoing
                changed = True

    # A task's presence only in a bypassable branch must not establish mandatory source coverage.
    universe = set(task_ids)
    must_out = {block: set(universe) for block in blocks}
    changed = True
    while changed:
        changed = False
        for block in blocks:
            incoming = (set.intersection(*(must_out[p] for p in predecessors[block]))
                        if block is not entry and predecessors[block] else set())
            outgoing = incoming | block_tasks[block]
            if outgoing != must_out[block]:
                must_out[block] = outgoing
                changed = True
    if any(not universe.issubset(must_out[block]) for block in returns):
        problems.add("a returning CFG path bypasses an entire planned task")
    if not universe.issubset(counts):
        problems.add("planned task emits no owned computation or control flow")
    return {
        "schema": "compiler_task_cfg_evidence_v1", "status": "refused" if problems else "verified",
        "problems": sorted(problems), "blocks": len(blocks), "reachable_blocks": len(reachable),
        "return_blocks": len(returns), "emitted_operations_by_task": dict(sorted(counts.items())),
        "shared_prologue_operations": dict(sorted(shared_counts.items())),
        "proof_scope": "CFG reachability, SSA dominance, ordered and mandatory task ownership",
        "loop_trip_count_equivalence": "UNPROVEN", "numeric_equivalence": "UNPROVEN",
    }
