"""Observe completion fences over actual LLVM control-flow paths, not printed block order."""

from xdsl.dialects.builtin import UnitAttr

from .decode import _asm_template


def closing_fence_on_normal_returns(module) -> bool | None:
    """Every returning path of each command-bearing function must finish after a fence.

    This is a finite two-state CFG analysis: any inline assembly dirties the
    state and an effectful memory fence closes it. Unsupported calls, nested
    regions or terminators remain unknown. This observes fence placement only;
    it supplies no operand, numerical, dispatch or physical completion proof.
    """
    observed = False
    for function in module.body.block.ops:
        if function.name != "llvm.func" or not function.body.blocks:
            continue
        blocks = tuple(function.body.blocks)
        if not any(op.name == "llvm.inline_asm" for block in blocks for op in block.ops):
            continue
        observed = True
        if any(op.regions or op.name in ("llvm.call", "llvm.invoke") for block in blocks for op in block.ops):
            return None
        pending = [(blocks[0], False)]
        visited = set()
        returns = 0
        while pending:
            block, closed = pending.pop()
            if (block, closed) in visited:
                continue
            visited.add((block, closed))
            for op in block.ops:
                if op.name == "llvm.inline_asm":
                    closed = (
                        _asm_template(op) == "fence"
                        and not op.operands
                        and isinstance(op.has_side_effects, UnitAttr)
                        and "~{memory}" in op.constraints.data.split(",")
                    )
            end = block.last_op
            if end is None:
                return None
            if end.name == "llvm.return":
                returns += 1
                if not closed:
                    return False
            elif end.name in ("llvm.br", "llvm.cond_br"):
                if not end.successors or any(successor not in blocks for successor in end.successors):
                    return None
                pending.extend((successor, closed) for successor in end.successors)
            else:
                return None
        if not returns:
            return None
    return True if observed else None
