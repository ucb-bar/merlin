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


def emit_index_program(kernel, builder, lower_call):
    errors = check_structure(kernel)
    if errors:
        raise ValueError("invalid counted-loop program: " + "; ".join(errors))

    def validate(body):
        for statement in body:
            if isinstance(statement, Loop):
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
