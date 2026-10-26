"""Emit checked counted-loop index programs through a caller's LLVM builder.

The target callback lowers each actual instruction and owns its effects. This
bridge supplies host loops and integer address expressions only; completion
annotations must never disappear during emission.
"""

from __future__ import annotations

from merlin.sched.ir import Const, Loop, Scaled, Select, Sum, Var, check_structure


class FnBuilderIndexAdapter:
    """Adapt Merlin's ordinary LLVM FnBuilder without reconstructing a schedule.

    Only signed i64 host indices and ascending, positive-step loops are admitted.
    The target callback still implements actual instructions and their completion.
    This adapter does not infer target semantics or erase execution annotations.
    """

    def __init__(self, builder):
        if any(not callable(getattr(builder, name, None)) for name in
               ("const", "add", "add_i", "mul_i", "for_loop")):
            raise ValueError("LLVM index emission requires the FnBuilder interface")
        self.builder = builder

    def value(self, value):
        from xdsl.dialects.builtin import i64
        from xdsl.ir import SSAValue

        if type(value) is int:
            if not -(1 << 63) <= value < (1 << 63):
                raise ValueError("LLVM index value exceeds signed i64")
            return self.builder.const(value)
        if not isinstance(value, SSAValue) or value.type != i64:
            raise ValueError("LLVM index operand must have type i64")
        return value

    def op(self, opcode, lhs, rhs):
        if opcode not in ("add", "mul"):
            raise ValueError("unsupported LLVM index arithmetic")
        action = self.builder.add_i if opcode == "add" else self.builder.mul_i
        return action(self.value(lhs), self.value(rhs))

    def cmp(self, predicate, lhs, rhs):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import IntegerAttr, i64

        if predicate != "eq":
            raise ValueError("LLVM index selection only supports equality")
        return self.builder.add(llvm.ICmpOp(
            self.value(lhs), self.value(rhs),
            IntegerAttr(llvm.ICmpPredicateFlag.EQ.to_int(), i64),
        )).results[0]

    def select(self, condition, lhs, rhs):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import i1
        from xdsl.ir import SSAValue

        if not isinstance(condition, SSAValue) or condition.type != i1:
            raise ValueError("LLVM index selection requires an i1 condition")
        return self.builder.add(llvm.SelectOp(condition, self.value(lhs), self.value(rhs))).results[0]

    def loop(self, start, stop, step, callback):
        if (any(type(value) is not int for value in (start, stop, step))
                or not 0 <= start <= stop < (1 << 63) or not 0 < step < (1 << 63)
                or stop and step > (1 << 63) - stop):
            raise ValueError("LLVM index loop requires bounded ascending signed-i64 bounds and step")
        self.builder.for_loop(start, stop, step, callback, retain_loop=True)

    def loop_carried(self, start, stop, step, initial, callback):
        """Lower actual i64 completion handles through LLVM CFG block arguments.

        The target must encode its real handle as i64 and implement the wait;
        this method neither creates a completion nor substitutes a delay.
        Nested callbacks leave their real exit block as the loop latch.
        """
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import IntegerAttr, i64
        from xdsl.ir import Block

        from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll

        if (any(type(v) is not int for v in (start, stop, step))
                or not 0 <= start <= stop < 1 << 63 or not 0 < step < 1 << 63
                or stop + step >= 1 << 63 or type(initial) is not tuple or not initial):
            raise ValueError("LLVM carried loop requires explicit i64 handles and ascending static bounds")
        values = tuple(self.value(v) for v in initial)
        b = self.builder
        header = Block(arg_types=[i64] * (len(values) + 1))
        active = Block()
        after = Block(arg_types=[i64] * len(values))
        for block in (header, active, after):
            b.region.add_block(block)
        b.add(llvm.BrOp(header, b.const(start), *values))
        b.blk = header
        index, *carried = header.args
        condition = b.add(llvm.ICmpOp(index, b.const(stop),
            IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))).results[0]
        b.add(llvm.CondBrOp(condition, active, [], after, carried))
        b.blk = active
        result = callback(index, tuple(carried))
        if not isinstance(result, (tuple, list)) or len(result) != len(values):
            raise ValueError("LLVM carried loop callback changed its completion arity")
        result = tuple(self.value(v) for v in result)
        latch = b.add(llvm.BrOp(header, b.add_i(index, b.const(step)), *result))
        disable_loop_unroll(latch)
        b.blk = after
        return tuple(after.args)


class FnBuilderControlAdapter(FnBuilderIndexAdapter):
    """Ordinary typed LLVM control for checked spatial/convolution indices.

    Payload operations and target effects remain in the explicit callback.
    Signed comparisons are needed before clamping negative halo coordinates;
    unsigned comparisons retain the existing DMA in-bounds predicates.
    """

    def op(self, opcode, lhs, rhs, dtype="i64"):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import i1
        from xdsl.ir import SSAValue

        operations = {"add": llvm.AddOp, "sub": llvm.SubOp, "mul": llvm.MulOp,
                      "udiv": llvm.UDivOp, "urem": llvm.URemOp, "and": llvm.AndOp}
        if opcode not in operations or dtype not in ("i64", "i1") or dtype == "i1" and opcode != "and":
            raise ValueError("unsupported typed LLVM control arithmetic")
        if dtype == "i1":
            if any(not isinstance(x, SSAValue) or x.type != i1 for x in (lhs, rhs)):
                raise ValueError("LLVM predicate conjunction requires i1 operands")
        else:
            lhs, rhs = self.value(lhs), self.value(rhs)
        return self.builder.add(operations[opcode](lhs, rhs)).results[0]

    def cmp(self, predicate, lhs, rhs):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import IntegerAttr, i64

        flags = {"eq": llvm.ICmpPredicateFlag.EQ, "slt": llvm.ICmpPredicateFlag.SLT,
                 "ult": llvm.ICmpPredicateFlag.ULT, "ule": llvm.ICmpPredicateFlag.ULE}
        if predicate not in flags:
            raise ValueError("unsupported LLVM control predicate")
        return self.builder.add(llvm.ICmpOp(self.value(lhs), self.value(rhs),
            IntegerAttr(flags[predicate].to_int(), i64))).results[0]

    def branch(self, predicate, yes, no):
        from xdsl.dialects import llvm
        from xdsl.dialects.builtin import i1
        from xdsl.ir import Block, SSAValue

        if type(predicate) is bool:
            (yes if predicate else no)()
            return
        if not isinstance(predicate, SSAValue) or predicate.type != i1:
            raise ValueError("LLVM control branch requires an i1 predicate")
        taken, other, after = Block(), Block(), Block()
        for block in (taken, other, after):
            self.builder.region.add_block(block)
        self.builder.add(llvm.CondBrOp(predicate, taken, [], other, []))
        self.builder.blk = taken
        yes()
        self.builder.add(llvm.BrOp(after))
        self.builder.blk = other
        no()
        self.builder.add(llvm.BrOp(after))
        self.builder.blk = after


def emit_index_program(kernel, builder, lower_call):
    errors = check_structure(kernel)
    if errors:
        raise ValueError("invalid counted-loop program: " + "; ".join(errors))

    def validate(body):
        for statement in body:
            if isinstance(statement, Loop):
                if statement.iter_args or statement.yields or statement.results:
                    raise ValueError("raw index emission cannot erase carried completion; use the checked SCF bridge")
                validate(statement.body)
            elif any((statement.unit, statement.produces, statement.awaits, statement.stages,
                      statement.sets, statement.assumes)):
                raise ValueError("LLVM index bridge cannot erase execution or configuration annotations")
    validate(kernel.body)

    def value(expression, env):
        if isinstance(expression, Const):
            if not -(1 << 63) <= expression.value < (1 << 63):
                raise ValueError("index constant exceeds selected 64-bit builder")
            return expression.value
        if isinstance(expression, Var):
            return env[expression.name]
        if isinstance(expression, Sum):
            result = 0
            for term in expression.terms:
                result = builder.op("add", result, value(term, env))
            return result
        if isinstance(expression, Scaled):
            return builder.op("mul", value(expression.expr, env), expression.factor)
        if isinstance(expression, Select):
            condition = builder.cmp("eq", env[expression.var], expression.equals)
            return builder.select(condition, value(expression.then, env), value(expression.other, env))
        raise ValueError("LLVM index bridge needs explicit integer instruction operands")

    def emit(body, env):
        for statement in body:
            if isinstance(statement, Loop):
                builder.loop(0, statement.extent, 1,
                             lambda index: emit(statement.body, {**env, statement.var: index}))
            else:
                lower_call(statement.instr, {name: value(expr, env) for name, expr in statement.args})

    emit(kernel.body, {})
    return {"kernel_sha256": kernel.digest(), "index_bits": 64,
            "scope": "actual counted-loop/index emission; target effects and machine conformance remain separate"}
