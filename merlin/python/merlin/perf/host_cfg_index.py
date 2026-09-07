"""Host-owned immutable CFG facts shared by independent emitted-program audits."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class PreparedHostCFG:
    """A function-identity-bound CFG index and dominance relation.

    This object never crosses the compiler or worker boundary.  The parsed function and its block
    objects are retained by identity so an index from another artifact cannot be substituted.
    """

    function: Any
    blocks: tuple[Any, ...]
    block_set: frozenset[Any]
    successors: Mapping[Any, tuple[Any, ...]]
    predecessors: Mapping[Any, frozenset[Any]]
    dominance: Any


class _IntervalDominanceInfo:
    """Exact dominance queries for one fully reachable CFG.

    Immediate dominators are computed in reverse-postorder, then represented as intervals in the
    dominator tree.  This avoids xDSL's quadratic sets-of-all-blocks implementation on generated
    whole-model CFGs while keeping the same improper-dominance query (`a` dominates itself).
    """

    def __init__(self, entry: Any, successors: Mapping[Any, tuple[Any, ...]],
                 predecessors: Mapping[Any, frozenset[Any]]):
        seen = {entry}
        postorder: list[Any] = []
        stack: list[tuple[Any, Any]] = [(entry, iter(successors[entry]))]
        while stack:
            block, edges = stack[-1]
            try:
                successor = next(edges)
            except StopIteration:
                postorder.append(block)
                stack.pop()
                continue
            if successor not in seen:
                seen.add(successor)
                stack.append((successor, iter(successors[successor])))
        reverse_postorder = tuple(reversed(postorder))
        position = {block: index for index, block in enumerate(reverse_postorder)}
        immediate = {entry: entry}

        def intersect(left: Any, right: Any) -> Any:
            while left is not right:
                while position[left] > position[right]:
                    left = immediate[left]
                while position[right] > position[left]:
                    right = immediate[right]
            return left

        changed = True
        while changed:
            changed = False
            for block in reverse_postorder[1:]:
                defined = [pred for pred in predecessors[block] if pred in immediate]
                if not defined:
                    continue
                new = defined[0]
                for pred in defined[1:]:
                    new = intersect(pred, new)
                if immediate.get(block) is not new:
                    immediate[block] = new
                    changed = True
        if len(immediate) != len(reverse_postorder):
            raise ValueError("reachable CFG has no complete immediate-dominator solution")

        children = {block: [] for block in reverse_postorder}
        for block, parent in immediate.items():
            if block is not entry:
                children[parent].append(block)
        clock = 0
        intervals: dict[Any, tuple[int, int]] = {}
        starts: dict[Any, int] = {}
        traversal: list[tuple[Any, bool]] = [(entry, False)]
        while traversal:
            block, exiting = traversal.pop()
            if exiting:
                intervals[block] = (starts[block], clock)
                clock += 1
                continue
            starts[block] = clock
            clock += 1
            traversal.append((block, True))
            traversal.extend((child, False) for child in reversed(children[block]))
        self._intervals = intervals

    def dominates(self, left: Any, right: Any) -> bool:
        left_start, left_end = self._intervals[left]
        right_start, right_end = self._intervals[right]
        return left_start <= right_start and right_end <= left_end

    def strictly_dominates(self, left: Any, right: Any) -> bool:
        return left is not right and self.dominates(left, right)


def _dominance(blocks: tuple[Any, ...], block_set: frozenset[Any],
               successors: Mapping[Any, tuple[Any, ...]],
               predecessors: Mapping[Any, frozenset[Any]]) -> Any:
    """Use the linear-space index for valid reachable CFGs; preserve xDSL on unreachable CFGs."""
    if not blocks:
        return None
    if any(successor not in block_set for block in blocks for successor in successors[block]):
        # Both consumers refuse an edge leaving the function before consulting dominance.
        return None
    reachable = set()
    pending = [blocks[0]]
    while pending:
        block = pending.pop()
        if block not in reachable:
            reachable.add(block)
            pending.extend(successors[block])
    if reachable != block_set:
        # Unreachable inputs are refused by both audits. Retain xDSL's relation so any additional
        # diagnostics remain byte-for-byte compatible with the legacy path.
        from xdsl.irdl.dominance import DominanceInfo
        return DominanceInfo(blocks[0].parent_region())
    return _IntervalDominanceInfo(blocks[0], successors, predecessors)


def prepare_host_cfg(function: Any) -> PreparedHostCFG:
    """Build common CFG/dominance facts once without making an audit verdict."""
    blocks = tuple(function.body.blocks)
    block_set = frozenset(blocks)
    predecessors: dict[Any, set[Any]] = {block: set() for block in blocks}
    successors: dict[Any, tuple[Any, ...]] = {}
    for block in blocks:
        terminal = block.last_op
        if terminal is None or terminal.name not in {"llvm.br", "llvm.cond_br", "llvm.return"}:
            successors[block] = ()
            continue
        edges = tuple(terminal.successors)
        successors[block] = edges
        for successor in edges:
            if successor in block_set:
                predecessors[successor].add(block)
    return PreparedHostCFG(
        function=function,
        blocks=blocks,
        block_set=block_set,
        successors=MappingProxyType(successors),
        predecessors=MappingProxyType({block: frozenset(rows)
                                       for block, rows in predecessors.items()}),
        dominance=_dominance(blocks, block_set, successors, {
            block: frozenset(rows) for block, rows in predecessors.items()}),
    )


def require_prepared_host_cfg(function: Any, prepared: PreparedHostCFG) -> PreparedHostCFG:
    """Refuse an index that is not bound to this exact parsed function and block sequence."""
    if prepared.function is not function:
        raise ValueError("prepared host CFG belongs to a different function object")
    live_blocks = tuple(function.body.blocks)
    if (len(live_blocks) != len(prepared.blocks)
            or any(live is not retained for live, retained in zip(
                live_blocks, prepared.blocks, strict=True))):
        raise ValueError("prepared host CFG block identity changed after construction")
    return prepared
