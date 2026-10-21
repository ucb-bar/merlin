"""Compress finite affine primitive-call streams into checked ordinary host loops.

This changes static representation only. Every dynamic call, operand and its
order must match the original finite stream. No target effects, hardware loop
instructions or latency are inferred. Unsupported annotated calls remain literal.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, fields, replace

from merlin.sched.ir import Call, Const, Kernel, Loop, Ptr, Var, evaluate
from merlin.sched.ir.expr import Expr, Scaled, Select, Sum
from merlin.sched.ir.integer_domain import IntegerDomainError, check_integer_domain
from merlin.sched.ir.kernel import check_structure


@dataclass(frozen=True)
class CommandCompaction:
    kernel: Kernel
    original_calls: int
    static_calls: int
    loops: int
    comparisons: int


def _preflight(kernel, max_calls, index_bits):
    """Bound traversal before invoking the ordinary recursive IR checkers."""
    if type(kernel) is not Kernel or type(max_calls) is not int or max_calls <= 0:
        raise ValueError("command compaction needs a typed kernel and positive verification budget")
    if type(index_bits) is not int or index_bits not in (32, 64):
        raise ValueError("command compaction needs an explicit signed index width")
    low, high = -(1 << (index_bits - 1)), (1 << (index_bits - 1)) - 1
    pending, nodes, expression_nodes = [(item, 0) for item in kernel.body], 0, 0
    while pending:
        statement, depth = pending.pop()
        nodes += 1
        if depth > 64 or nodes > max_calls:
            raise ValueError("command compaction static/nesting verification budget exceeded")
        if type(statement) is Loop:
            if type(statement.extent) is not int or not 0 <= statement.extent <= high:
                raise ValueError("command loop extent exceeds the selected signed domain")
            pending.extend((item, depth + 1) for item in statement.body)
            continue
        if type(statement) is not Call or any(
            len(pairs) > 64 for pairs in (statement.args, statement.sets, statement.assumes)
        ):
            raise ValueError("command compaction requires bounded actual call operands/annotations")
        expressions = [
            (value.offset if type(value) is Ptr else value, 0)
            for pairs in (statement.args, statement.sets, statement.assumes)
            for _, value in pairs
        ]
        while expressions:
            value, level = expressions.pop()
            expression_nodes += 1
            if level > 64 or expression_nodes > max_calls * 64:
                raise ValueError("command compaction expression verification budget exceeded")
            literals = ()
            if type(value) is tuple:
                if len(value) > 64:
                    raise ValueError("command compaction annotation verification budget exceeded")
                expressions.extend((item, level + 1) for item in value)
            elif type(value) is Const:
                literals = (value.value,)
            elif type(value) is Scaled:
                literals = (value.factor,)
                expressions.append((value.expr, level + 1))
            elif type(value) is Select:
                literals = (value.equals,)
                expressions.extend(((value.then, level + 1), (value.other, level + 1)))
            elif type(value) is Sum:
                expressions.extend((item, level + 1) for item in value.terms)
            elif isinstance(value, Expr) and type(value) is not Var:
                raise ValueError("command compaction requires supported bounded expressions")
            if any(type(x) is not int or not low <= x <= high for x in literals):
                raise ValueError("command integer literal exceeds its selected signed domain (overflow)")


def _same_argument(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return struct.pack(">d", left) == struct.pack(">d", right)
    if type(left) is tuple:
        return len(left) == len(right) and all(_same_argument(a, b) for a, b in zip(left, right))
    if type(left) in (Const, Var, Sum, Scaled, Select, Ptr):
        return all(_same_argument(getattr(left, field.name), getattr(right, field.name)) for field in fields(left))
    return left == right


def _literal(call):
    return (
        type(call) is Call
        and not any((call.unit, call.produces, call.awaits, call.stages, call.sets, call.assumes))
        and all(type(expr) is Const for _, expr in call.args)
    )


def verify_command_compaction(original, actual, *, index_bits, max_calls=1_000_000):
    """Compare actual loop execution to the complete original primitive sequence.

    The verifier uses the existing index evaluator and signed intermediate
    checker, not a retained pattern, hash-only assertion or optimizer replay.
    Verification allocates no dynamic trace beyond the original finite input.
    """
    _preflight(original, max_calls, index_bits)
    _preflight(actual, max_calls, index_bits)
    if (
        type(original) is not Kernel
        or type(actual) is not Kernel
        or check_structure(original)
        or check_structure(actual)
    ):
        raise ValueError("command compaction requires valid original and actual kernel IR")
    if type(max_calls) is not int or max_calls <= 0 or len(original.body) > max_calls:
        raise ValueError("command compaction exceeds its finite verification budget")
    if any(type(call) is not Call for call in original.body):
        raise ValueError("command compaction original must be a finite primitive-call stream")
    if (original.name, original.args, original.attrs) != (actual.name, actual.args, actual.attrs):
        raise ValueError("command compaction changed invocation boundaries")
    check_integer_domain(original, index_bits)
    check_integer_domain(actual, index_bits)
    cursor = 0
    domains, static = {}, 0

    def count(body, depth=0):
        nonlocal static
        if depth > 64:
            raise ValueError("command compaction verifier nesting budget exceeded")
        total = 0
        for statement in body:
            static += 1
            if static > max_calls:
                raise ValueError("actual compacted static program exceeds its verification budget")
            if type(statement) is Loop:
                domains[id(statement)] = count(statement.body, depth + 1)
                total += statement.extent * domains[id(statement)]
            else:
                total += 1
        return total

    if count(actual.body) != len(original.body):
        raise ValueError("actual compacted command count differs from the original")

    def resolved_arg(expr, indices):
        if isinstance(expr, Expr):
            return Const(evaluate(expr, indices))
        if type(expr) is Ptr:
            return replace(expr, offset=Const(evaluate(expr.offset, indices)))
        return expr

    def check(body, indices, depth):
        nonlocal cursor
        if depth > 64:
            raise ValueError("command compaction verifier nesting budget exceeded")
        for statement in body:
            if type(statement) is Loop:
                if domains[id(statement)] == 0:
                    continue
                for i in range(statement.extent):
                    check(statement.body, {**indices, statement.var: i}, depth + 1)
            else:
                if cursor >= len(original.body):
                    raise ValueError("command compaction adds dynamic commands")
                resolved = replace(
                    statement, args=tuple((key, resolved_arg(expr, indices)) for key, expr in statement.args)
                )
                expected = original.body[cursor]
                expected_args = tuple((key, resolved_arg(expr, {})) for key, expr in expected.args)
                if (
                    replace(resolved, args=(), sets=(), assumes=()) != replace(expected, args=(), sets=(), assumes=())
                    or not _same_argument(resolved.sets, expected.sets)
                    or not _same_argument(resolved.assumes, expected.assumes)
                    or len(resolved.args) != len(expected_args)
                    or any(
                        ka != kb or not _same_argument(a, b) for (ka, a), (kb, b) in zip(resolved.args, expected_args)
                    )
                ):
                    raise ValueError("compacted actual operands/order differ from original commands")
                cursor += 1

    check(actual.body, {}, 0)
    if cursor != len(original.body):
        raise ValueError("command compaction drops dynamic commands")
    return {"dynamic_calls": cursor, "index_bits": index_bits, "semantics": "exact primitive sequence"}


def compact_command_stream(
    kernel,
    *,
    index_bits,
    max_period=64,
    max_comparisons=1_000_000,
    max_calls=1_000_000,
):
    """Infer repeated affine call blocks under explicit search and code budgets.

    The source is finite. Calls with dependencies/configuration annotations,
    pointers, floating operands or nonliteral operands remain unchanged. This
    pass neither removes setup nor changes the dynamic instruction count.
    A Phase 2 consumer may select this after checked scheduling, then use the
    ordinary counted-loop emitter and its original executable-size admission.
    """
    _preflight(kernel, max_calls, index_bits)
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("command compaction requires valid kernel IR")
    if any(type(n) is not int or n <= 0 for n in (max_period, max_comparisons, max_calls)):
        raise ValueError("command compaction requires positive explicit budgets")
    if len(kernel.body) > max_calls or any(type(call) is not Call for call in kernel.body):
        raise ValueError("command compaction expects a bounded finite primitive stream")
    check_integer_domain(kernel, index_bits)
    source, output, cursor, serial, comparisons = kernel.body, [], 0, 0, 0
    while cursor < len(source):
        best = None
        for period in range(1, min(max_period, (len(source) - cursor) // 2) + 1):
            comparisons += 2 * period
            if comparisons > max_comparisons:
                raise ValueError("command compaction search exceeds its comparison budget")
            first = source[cursor : cursor + period]
            second = source[cursor + period : cursor + 2 * period]
            if not all(_literal(call) for call in (*first, *second)):
                continue
            increments = []
            for left, right in zip(first, second, strict=True):
                comparisons += 1
                if comparisons > max_comparisons:
                    raise ValueError("command compaction search exceeds its comparison budget")
                if left.instr != right.instr or tuple(k for k, _ in left.args) != tuple(k for k, _ in right.args):
                    break
                increments.append(
                    tuple(b.value - a.value for (_, a), (_, b) in zip(left.args, right.args, strict=True))
                )
            if len(increments) != period:
                continue
            trips = 2
            while cursor + (trips + 1) * period <= len(source):
                matches = True
                for offset, (base, delta) in enumerate(zip(first, increments, strict=True)):
                    comparisons += 1
                    if comparisons > max_comparisons:
                        raise ValueError("command compaction search exceeds its comparison budget")
                    expected = replace(
                        base,
                        args=tuple(
                            (key, Const(expr.value + trips * step))
                            for (key, expr), step in zip(base.args, delta, strict=True)
                        ),
                    )
                    if expected != source[cursor + trips * period + offset]:
                        matches = False
                        break
                if not matches:
                    break
                trips += 1
            if period * trips <= period + 1:
                continue
            variable = f"compact_{serial}"
            reserved = {argument.name for argument in kernel.args}
            while variable in reserved:
                variable = "_" + variable
            body = tuple(
                replace(
                    call,
                    args=tuple(
                        (key, expr + Var(variable) * step) for (key, expr), step in zip(call.args, delta, strict=True)
                    ),
                )
                for call, delta in zip(first, increments, strict=True)
            )
            loop = Loop(variable, trips, body)
            try:
                check_integer_domain(replace(kernel, body=(loop,)), index_bits)
            except IntegerDomainError:
                continue
            saving = period * trips - period - 1
            if best is None or saving > best[0]:
                best = saving, trips * period, loop
        if best is None:
            output.append(source[cursor])
            cursor += 1
        else:
            output.append(best[2])
            cursor += best[1]
            serial += 1
    candidate = replace(kernel, body=tuple(output))
    verify_command_compaction(kernel, candidate, index_bits=index_bits, max_calls=max_calls)
    static_calls = sum(len(item.body) if type(item) is Loop else 1 for item in candidate.body)
    return CommandCompaction(candidate, len(source), static_calls, serial, comparisons)
