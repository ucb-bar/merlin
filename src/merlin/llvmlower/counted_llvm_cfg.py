"""Lift canonical, zero-based LLVM counted loops without executing iterations.

Instruction meanings stay with the original observer. This reader accounts for
every reachable block and rejects noncanonical control flow, loop-carried state,
unproved integer wrap, memory operations and unknown arithmetic. It establishes
neither instruction semantics nor external buffer ownership or completion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerType
from xdsl.traits import SymbolTable

from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Ptr, Scaled, Select, Sum, Var
from merlin.sched.ir.kernel import check_structure

from .static_llvm_cfg import StaticTraceError, _flags_absent


class CountedTraceUnavailable(ValueError):
    """The actual function is outside the selected compact reader's domain."""


@dataclass(frozen=True)
class CountedPointer:
    base: str
    offset: Expr = Const(0)


@dataclass(frozen=True)
class CountedAddress:
    """Pointer bits with symbolic backing, never an assumed integer base."""

    base: str
    offset: Expr


@dataclass(frozen=True)
class _Comparison:
    predicate: int
    left: Expr
    right: Expr


def canonical_counted_kernel(kernel):
    """Rename bound induction variables; preserve calls, operands and ordering."""
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("counted normalization requires valid current Kernel IR")
    serial = 0

    def expression(value, names):
        if type(value) is Var:
            return Var(names[value.name])
        if type(value) is Sum:
            return sum((expression(term, names) for term in value.terms), Const(0))
        if type(value) is Scaled:
            return expression(value.expr, names) * value.factor
        if type(value) is Select:
            return replace(
                value, var=names[value.var], then=expression(value.then, names), other=expression(value.other, names)
            )
        if type(value) is Ptr:
            return replace(value, offset=expression(value.offset, names))
        return value

    def body(statements, names):
        nonlocal serial
        result = []
        for statement in statements:
            if type(statement) is Loop:
                name = f"counted_{serial}"
                serial += 1
                result.append(replace(statement, var=name, body=body(statement.body, {**names, statement.var: name})))
            else:
                result.append(
                    replace(
                        statement,
                        args=tuple((key, expression(value, names)) for key, value in statement.args),
                        sets=tuple((key, expression(value, names)) for key, value in statement.sets),
                        assumes=tuple((key, expression(value, names)) for key, value in statement.assumes),
                    )
                )
        return tuple(result)

    return replace(kernel, body=body(kernel.body, {}))


def project_counted_function(
    function,
    arguments,
    *,
    observe,
    max_operations=4096,
    max_depth=32,
    pointer_bits=None,
    global_addresses=None,
):
    """Return actual Call/Loop statements under a supplied instruction observer.

    ``observe(op, inputs)`` returns a Call for a known result-free instruction,
    or None to let this reader interpret pure arithmetic/control. No dynamic
    loop instance is visited. Arguments bind exact pointer identities or affine
    integer values; they confer no extent, noalias, or instruction permissions.

    Loops must enter at zero, advance by a positive constant, use a strict upper
    bound and exit without results. Pretest loops normalize their induction to
    a zero-based tile index. Only 64-bit nonnegative, nonwrapping arithmetic is
    currently admitted. Unsupported forms must retain their original route.
    """
    if type(max_operations) is not int or max_operations < 1 or type(max_depth) is not int or max_depth < 1:
        raise ValueError("counted LLVM projection requires positive static budgets")
    if type(function) is not llvm.FuncOp or not function.body.blocks:
        raise CountedTraceUnavailable("counted LLVM projection requires a defined function")
    function.verify()
    blocks = tuple(function.body.blocks)
    if sum(len(tuple(block.ops)) for block in blocks) > max_operations or len(arguments) != len(blocks[0].args):
        raise CountedTraceUnavailable("counted LLVM static budget or argument arity differs")
    values = dict(zip(blocks[0].args, arguments, strict=True))
    globals_snapshot = dict(global_addresses or {})
    for argument, value in values.items():
        if isinstance(argument.type, llvm.LLVMPointerType):
            if type(value) is not CountedPointer or getattr(argument.type.addr_space, "data", 0):
                raise CountedTraceUnavailable("counted LLVM pointer identity or address space differs")
        elif (
            not isinstance(argument.type, IntegerType) or argument.type.width.data != 64 or not isinstance(value, Expr)
        ):
            raise CountedTraceUnavailable("counted LLVM requires explicit i64 scalar arguments")
    incoming = {block: [] for block in blocks}
    for block in blocks:
        for operation in block.ops:
            for successor in operation.successors:
                incoming[successor].append((block, operation))
    visited, domains = set(), {}

    def interval(value):
        if type(value) is Const:
            return value.value, value.value
        if type(value) is Var and value.name in domains:
            return 0, domains[value.name] - 1
        if type(value) is Sum:
            spans = [interval(term) for term in value.terms]
            return sum(a for a, _ in spans), sum(b for _, b in spans)
        if type(value) is Scaled:
            low, high = interval(value.expr)
            return min(low * value.factor, high * value.factor), max(low * value.factor, high * value.factor)
        if type(value) is Select:
            a, b = interval(value.then), interval(value.other)
            return min(a[0], b[0]), max(a[1], b[1])
        raise CountedTraceUnavailable("unknown counted LLVM scalar expression")

    def checked(value):
        low, high = interval(value)
        if low < 0 or high >= 1 << 63:
            raise CountedTraceUnavailable("counted LLVM scalar arithmetic may wrap or become negative")
        return value

    def read(env, value):
        if value not in env:
            raise CountedTraceUnavailable("unrepresented counted LLVM SSA value")
        return env[value]

    def walk(start, env, active=None, depth=0):
        if depth > max_depth:
            raise CountedTraceUnavailable("counted LLVM nesting budget exceeded")
        body, block = [], start
        while True:
            if block in visited:
                raise CountedTraceUnavailable("noncanonical or multiply entered counted LLVM block")
            visited.add(block)
            for operation in block.ops:
                inputs = tuple(read(env, value) for value in operation.operands)
                call = observe(operation, inputs)
                if call is not None:
                    unused_void = (
                        len(operation.results) == 1
                        and isinstance(operation.results[0].type, llvm.LLVMVoidType)
                        and not operation.results[0].uses
                    )
                    if (
                        type(call) is not Call
                        or operation.regions
                        or operation.successors
                        or (operation.results and not unused_void)
                        or operation.has_trait(llvm.IsTerminator)
                    ):
                        raise CountedTraceUnavailable(
                            "observed counted instruction must be result-free and non-control"
                        )
                    body.append(call)
                    continue
                try:
                    _flags_absent(operation)
                except StaticTraceError as exc:
                    raise CountedTraceUnavailable(str(exc)) from exc
                result = None
                if isinstance(operation, llvm.ConstantOp):
                    attr = operation.properties.get("value")
                    if not hasattr(attr, "value") or not hasattr(attr.value, "data"):
                        raise CountedTraceUnavailable("counted LLVM requires integer constants")
                    result = checked(Const(attr.value.data))
                elif isinstance(operation, (llvm.AddOp, llvm.SubOp, llvm.MulOp)):
                    left, right = inputs
                    if not isinstance(left, Expr) or not isinstance(right, Expr):
                        raise CountedTraceUnavailable("counted LLVM arithmetic requires scalar operands")
                    if isinstance(operation, llvm.AddOp):
                        result = checked(left + right)
                    elif isinstance(operation, llvm.SubOp):
                        result = checked(left + right * -1)
                    elif type(left) is Const or type(right) is Const:
                        result = checked(right * left.value if type(left) is Const else left * right.value)
                    else:
                        raise CountedTraceUnavailable("counted LLVM multiplication must be affine")
                elif isinstance(operation, llvm.ICmpOp):
                    if any(not isinstance(value, Expr) for value in inputs):
                        raise CountedTraceUnavailable("counted LLVM comparison requires scalars")
                    result = _Comparison(operation.predicate.value.data, *inputs)
                elif isinstance(operation, llvm.SelectOp):
                    condition, yes, no = inputs
                    if (
                        type(condition) is not _Comparison
                        or condition.predicate not in (0, 1)
                        or type(condition.left) is not Var
                        or condition.left.name not in domains
                        or type(condition.right) is not Const
                        or not isinstance(yes, Expr)
                        or not isinstance(no, Expr)
                    ):
                        raise CountedTraceUnavailable("counted LLVM select needs induction equality and scalar arms")
                    if condition.predicate == 1:
                        yes, no = no, yes
                    result = checked(Select(condition.left.name, condition.right.value, yes, no))
                elif isinstance(operation, llvm.AddressOfOp):
                    name = operation.global_name
                    if name.nested_references or name.root_reference.data not in globals_snapshot:
                        raise CountedTraceUnavailable("counted LLVM global address has no original binding")
                    declared = SymbolTable.lookup_symbol(function, name)
                    if (
                        not isinstance(declared, llvm.GlobalOp)
                        or declared.thread_local_ is not None
                        or declared.addr_space.value.data != 0
                    ):
                        raise CountedTraceUnavailable("counted LLVM global is missing or has unsupported storage")
                    result = globals_snapshot[name.root_reference.data]
                    if type(result) is not CountedPointer:
                        raise CountedTraceUnavailable("counted LLVM global needs an explicit pointer identity")
                elif isinstance(operation, llvm.GEPOp):
                    if (
                        pointer_bits != 64
                        or type(inputs[0]) is not CountedPointer
                        or operation.properties.get("inbounds") is not None
                    ):
                        raise CountedTraceUnavailable("counted LLVM GEP needs explicit width and non-inbounds mapping")
                    element = operation.properties["elem_type"]
                    indices = tuple(operation.properties["rawConstantIndices"].iter_values())
                    if not isinstance(element, IntegerType) or element.width.data % 8 or len(indices) != 1:
                        raise CountedTraceUnavailable("counted LLVM GEP needs one byte-sized integer element index")
                    index = inputs[1] if indices[0] == llvm.GEP_USE_SSA_VAL else Const(indices[0])
                    if not isinstance(index, Expr):
                        raise CountedTraceUnavailable("counted LLVM GEP requires an affine index")
                    result = CountedPointer(
                        inputs[0].base, checked(inputs[0].offset + index * (element.width.data // 8))
                    )
                elif isinstance(operation, llvm.PtrToIntOp):
                    if (
                        pointer_bits != 64
                        or type(inputs[0]) is not CountedPointer
                        or not isinstance(operation.results[0].type, IntegerType)
                        or operation.results[0].type.width.data != 64
                    ):
                        raise CountedTraceUnavailable("counted LLVM pointer conversion needs explicit equal widths")
                    result = CountedAddress(inputs[0].base, inputs[0].offset)
                elif isinstance(operation, llvm.IntToPtrOp):
                    if pointer_bits != 64 or type(inputs[0]) is not CountedAddress:
                        raise CountedTraceUnavailable(
                            "counted LLVM integer pointer conversion lacks a symbolic backing"
                        )
                    result = CountedPointer(inputs[0].base, inputs[0].offset)
                elif isinstance(operation, llvm.BrOp):
                    target = operation.successors[0]
                    if active is not None and len(active) > 4 and operation is active[3]:
                        if target is not active[0] or inputs != (active[1] + active[6],):
                            raise CountedTraceUnavailable("counted LLVM loop stride or backedge differs")
                        return body, active[5]
                    if not target.args:
                        if inputs or len(incoming[target]) != 1:
                            raise CountedTraceUnavailable("counted LLVM branch merge is unsupported")
                        block = target
                        break
                    edges = incoming[target]
                    backedges = [
                        (parent, branch)
                        for parent, branch in edges
                        if isinstance(branch, llvm.CondBrOp) and branch.then_block is target
                    ]
                    pretest = not backedges
                    if pretest:
                        backedges = [(parent, branch) for parent, branch in edges if branch is not operation]
                    if (
                        len(target.args) != 1
                        or len(inputs) != 1
                        or inputs[0] != Const(0)
                        or len(edges) != 2
                        or len(backedges) != 1
                    ):
                        raise CountedTraceUnavailable("counted LLVM loop needs one zero seed and one backedge")
                    latch = backedges[0][1]
                    header_branch = target.last_op
                    if pretest and (not isinstance(latch, llvm.BrOp) or not isinstance(header_branch, llvm.CondBrOp)):
                        raise CountedTraceUnavailable("counted LLVM pretest needs an unconditional latch")
                    condition = (header_branch if pretest else latch).cond.owner
                    if not isinstance(condition, llvm.ICmpOp):
                        raise CountedTraceUnavailable("counted LLVM loop condition is not an integer comparison")
                    bound = condition.rhs.owner
                    if not isinstance(bound, llvm.ConstantOp):
                        raise CountedTraceUnavailable("counted LLVM loop extent must be constant")
                    extent = bound.value.value.data
                    if type(extent) is not int or not 0 < extent < (1 << 63):
                        raise CountedTraceUnavailable("counted LLVM loop extent is not positive and representable")
                    step = 1
                    if pretest:
                        if (
                            len(latch.operands) != 1
                            or header_branch.then_arguments
                            or header_branch.else_arguments
                            or condition.lhs is not target.args[0]
                        ):
                            raise CountedTraceUnavailable("counted LLVM pretest carries unsupported state")
                        increment = latch.operands[0].owner
                        if not isinstance(increment, llvm.AddOp) or increment.lhs is not target.args[0]:
                            raise CountedTraceUnavailable("counted LLVM increment is not an induction plus constant")
                        literal = increment.rhs.owner
                        if not isinstance(literal, llvm.ConstantOp):
                            raise CountedTraceUnavailable("counted LLVM loop step must be constant")
                        step = literal.value.value.data
                        if type(step) is not int or step <= 0:
                            raise CountedTraceUnavailable("counted LLVM loop step must be positive")
                    count = (extent + step - 1) // step
                    if count * step >= 1 << 63:
                        raise CountedTraceUnavailable("counted LLVM final induction may wrap")
                    variable = Var(f"counted_{len(domains)}")
                    domains[variable.name] = count
                    induction = variable * step
                    selected_loop = (
                        (target, induction, extent, latch, header_branch, header_branch.else_block, step)
                        if pretest
                        else (target, variable, extent, latch)
                    )
                    nested, exit_block = walk(target, {**env, target.args[0]: induction}, selected_loop, depth + 1)
                    body.append(Loop(variable.name, count, tuple(nested)))
                    block = exit_block
                    if block.args:
                        raise CountedTraceUnavailable("counted LLVM loop results are unsupported")
                    break
                elif isinstance(operation, llvm.CondBrOp):
                    if active is not None and len(active) > 4 and operation is active[4]:
                        comparison = read(env, operation.cond)
                        if (
                            type(comparison) is not _Comparison
                            or comparison.predicate not in (2, 6)
                            or comparison.left != active[1]
                            or comparison.right != Const(active[2])
                            or operation.then_arguments
                            or operation.else_arguments
                        ):
                            raise CountedTraceUnavailable("counted LLVM pretest comparison differs")
                        block = operation.then_block
                        break
                    if active is None or operation is not active[3]:
                        raise CountedTraceUnavailable("conditional LLVM branch is not the selected loop latch")
                    header, variable, extent, _latch = active
                    comparison = read(env, operation.cond)
                    transferred = tuple(read(env, value) for value in operation.then_arguments)
                    expected = variable + 1
                    if (
                        type(comparison) is not _Comparison
                        or operation.then_block is not header
                        or comparison.predicate not in (2, 6)
                        or comparison.left != expected
                        or comparison.right != Const(extent)
                        or transferred != (expected,)
                        or operation.else_arguments
                    ):
                        raise CountedTraceUnavailable("counted LLVM loop stride, comparison or backedge differs")
                    return body, operation.else_block
                elif isinstance(operation, llvm.ReturnOp):
                    if active is not None or inputs:
                        raise CountedTraceUnavailable("counted LLVM returns inside a loop or returns a value")
                    return body, None
                else:
                    raise CountedTraceUnavailable(f"unsupported counted LLVM operation: {operation.name}")
                if result is not None:
                    if isinstance(result, Expr) and (
                        not isinstance(operation.results[0].type, IntegerType)
                        or operation.results[0].type.width.data != 64
                    ):
                        raise CountedTraceUnavailable("counted LLVM arithmetic requires explicit i64 representation")
                    env[operation.results[0]] = result
            else:
                raise CountedTraceUnavailable("counted LLVM block lacks an admitted terminator")

    result, end = walk(blocks[0], values)
    if end is not None or visited != set(blocks):
        raise CountedTraceUnavailable("counted LLVM projection did not account for the complete function")
    return tuple(result)
