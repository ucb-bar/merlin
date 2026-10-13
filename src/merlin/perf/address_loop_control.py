"""Prove bounded induction updates used exclusively for addressing and loop control."""

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Operation

from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


def _address_uses(value):
    """Reject address expressions that escape as stored/returned scalar values."""
    pending, visited = [value], set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if not current.uses:
            return False
        for use in current.uses:
            op = use.operation
            if op.name == "llvm.ptrtoint" and len(op.results) == 1 and isinstance(op.results[0].type, IntegerType):
                pending.extend(op.results)
            elif (op.name == "llvm.inttoptr" and len(op.results) == 1
                  and isinstance(op.results[0].type, llvm.LLVMPointerType)):
                pending.extend(op.results)
            elif (op.name == "llvm.bitcast" and len(op.results) == 1
                  and isinstance(current.type, llvm.LLVMPointerType)
                  and isinstance(op.results[0].type, llvm.LLVMPointerType)):
                pending.extend(op.results)
            elif op.name == "llvm.getelementptr" and use.index == 0 and len(op.results) == 1:
                pending.extend(op.results)
            elif op.name in {"llvm.add", "llvm.sub"} and len(op.results) == 1:
                other = op.operands[1 - use.index] if len(op.operands) == 2 else None
                if other is None or constant_integer(other) is None or (op.name == "llvm.sub" and use.index != 0):
                    return False
                pending.extend(op.results)
            elif op.name == "llvm.load" and use.index == 0:
                continue
            elif op.name == "llvm.store" and use.index == 1:
                continue
            elif op.name == "llvm.inline_asm" and not op.results:
                # The independent selected target decoder owns opaque command effects.
                continue
            else:
                return False
    return True


def address_loop_updates(blocks, loops) -> frozenset[Operation]:
    """Use only already-proven constant-trip loops; preserve all payload arithmetic.

    A bounded induction value may index pointers and govern its exact backedge.
    A loaded offset, scalar value sink, opaque integer operand, or an unrecognized
    forwarding edge receives no exemption. Scalar memory traffic remains counted.
    """
    proved = set()
    for row, _nodes in loops:
        header, latch = blocks[row["header_block"]], blocks[row["latch_block"]]
        index = row["induction_argument_index"]
        induction = header.args[index]
        terminal = latch.last_op
        if isinstance(terminal, llvm.BrOp) and terminal.successor is header:
            incoming = terminal.arguments
        elif isinstance(terminal, llvm.CondBrOp) and terminal.then_block is header:
            incoming = terminal.then_arguments
        else:
            continue
        if index >= len(incoming):
            continue
        next_value = incoming[index]
        update = next_value.owner
        if not isinstance(update, Operation) or update.name != "llvm.add" or len(update.operands) != 2:
            continue
        if induction not in update.operands:
            continue
        other = next(operand for operand in update.operands if operand is not induction)
        if constant_integer(other) != row["step"]:
            continue
        tested_terminal = terminal if row.get("post_tested") else header.last_op
        if not isinstance(tested_terminal, llvm.CondBrOp):
            continue
        comparison = tested_terminal.cond.owner
        if not isinstance(comparison, llvm.ICmpOp):
            continue
        if any(use.operation is not tested_terminal or use.index != 0 for use in comparison.results[0].uses):
            continue
        address_seen, valid = False, True
        for value in (induction, next_value):
            for use in value.uses:
                op = use.operation
                if op is update and value is induction:
                    continue
                if op is comparison:
                    continue
                if op is terminal and value is next_value:
                    if isinstance(terminal, llvm.BrOp):
                        valid &= use.index == index
                    else:
                        valid &= use.index == 1 + index
                    continue
                if op.name == "llvm.getelementptr" and use.index > 0 and len(op.results) == 1:
                    if _address_uses(op.results[0]):
                        address_seen = True
                        continue
                valid = False
        if valid and address_seen:
            proved.add(update)
    return frozenset(proved)
