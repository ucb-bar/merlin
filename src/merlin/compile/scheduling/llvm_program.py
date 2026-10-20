"""Emit checked counted-loop index programs through a caller's LLVM builder.

The target callback lowers each actual instruction and owns its effects. This
bridge supplies host loops and integer address expressions only; completion
annotations must never disappear during emission.
"""

from __future__ import annotations

from merlin.sched.ir import Const, Loop, Scaled, Select, Sum, Var, check_structure


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
