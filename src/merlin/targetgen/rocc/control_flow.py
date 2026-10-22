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


def constant_control_operand_witness(
    module, *, target, expected_trace, semantic_class, field, value,
    max_operations=200_000, max_blocks=20_000,
):
    """Prove a mode witness on a complete bounded constant-controlled path.

    This observes emitted LLVM operands, not device execution or tensor results.
    Loads stay unknown; branches depending on them, opaque calls, poison and
    exhausted bounds refuse the entire observation. Pointer operands never gain
    guessed addresses. The original supplied trace must re-decode identically.
    """
    from merlin.perf.constant_control_cfg import constant_control_counts
    from .decode import _operand, _parse_insn, _semantics, _Val, decode_module

    original = decode_module(module, target=target)
    if any(original.get(key) != expected_trace.get(key) for key in ("abi", "instructions")):
        raise ValueError("constant-control witness differs from original lowered-module trace")
    semantics = _semantics(target)
    observer = getattr(semantics, "operand_observer", None)
    if observer is None:
        return None
    isa = observer.isa_constants(target)
    functions = [op for op in module.body.block.ops if op.name == "llvm.func" and op.body.blocks
                 and any(child.name == "llvm.inline_asm" and _asm_template(child).startswith(".insn")
                         for child in op.walk())]
    # No interprocedural invocation or reachability assumption.
    if len(functions) != 1:
        return None
    function = functions[0]
    if any(op.regions or op.name in ("llvm.call", "llvm.invoke")
           for block in function.body.blocks for op in block.ops):
        return None
    found = False
    unresolved = False

    def visit(op, args):
        nonlocal found, unresolved
        if op.name != "llvm.inline_asm":
            return
        template = _asm_template(op)
        if template == "fence" and not op.operands and not op.results:
            return
        # One transport instruction with exactly its source registers. No hidden
        # branch/call, extra assembly instruction or unobserved output operand.
        if (op.results or len(args) != 2 or len(template.split(",")) != 6
                or any(char in template for char in (";", "\n", "\r"))
                or not isinstance(op.has_side_effects, UnitAttr)):
            raise ValueError("constant-control witness has opaque inline assembly")
        parsed = _parse_insn(template, isa["CUSTOM_OPCODE"])
        if parsed is None or not parsed[1] or [s.strip() for s in template.split(",")[-2:]] != ["$0", "$1"]:
            raise ValueError("constant-control witness has unsupported transport operands")
        operands = [_operand(_Val("const", value=arg) if type(arg) is int else _Val("unknown"))
                    for arg in args]
        label, decoded = observer.decode_instruction(parsed[0], *operands, isa)
        if label == "UNKNOWN":
            raise ValueError("constant-control witness has unresolved selected instruction")
        if label == semantic_class:
            observed = decoded.get(field)
            if observed is None:
                unresolved = True
            elif type(observed) is type(value) and observed == value:
                found = True

    try:
        constant_control_counts(function, max_operations=max_operations,
                                max_blocks=max_blocks, observe=visit,
                                observed_operations=frozenset(op for block in function.body.blocks
                                    for op in block.ops if op.name == "llvm.inline_asm"))
    except (ValueError, KeyError, TypeError, IndexError):
        return None  # Partial observations never establish the requested witness.
    return True if found else None if unresolved else False
