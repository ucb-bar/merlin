"""Bounded ordered unrolling; code growth follows an explicit compiler budget.

This mechanism selects no profitable factor. It preserves each original body
evaluation, including its effects and loop-carried state, in its original order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import Rewriter

from .certificates import ir_digest, ir_text, parse_ir
from .normalization import constant_integer


@dataclass(frozen=True)
class UnrollWitness:
    operation_index: int
    lower: int
    upper: int
    step: int
    factor: int
    index_bits: int
    iterations: int
    full_groups: int
    remainder: int
    source_sha256: str
    carried_types: tuple[str, ...]


def _depth(loop: scf.ForOp) -> int:
    depth = 0
    parent = loop.parent_op()
    while parent is not None:
        depth += isinstance(parent, scf.ForOp)
        parent = parent.parent_op()
    return depth


def _repeat_body(source: Block, target: Block, iv: SSAValue, state: Sequence[SSAValue]) -> tuple[SSAValue, ...]:
    mapper = dict(zip(source.args, (iv, *state), strict=True))
    operations = tuple(source.ops)
    terminal = operations[-1]
    if not isinstance(terminal, scf.YieldOp) or terminal.attributes or terminal.properties:
        raise ValueError("ordered unrolling requires the exact standard yield")
    for operation in operations[:-1]:
        target.add_op(operation.clone(mapper))
    return tuple(mapper.get(value, value) for value in terminal.operands)


def _unroll_one(loop: scf.ForOp, witness: UnrollWitness) -> None:
    source = loop.body.block
    if witness.iterations <= witness.factor:
        emitted = Block()
        state = tuple(loop.iter_args)
        for lane in range(witness.iterations):
            iv = arith.ConstantOp.from_int_and_width(witness.lower + lane * witness.step, builtin.IndexType())
            emitted.add_op(iv)
            state = _repeat_body(source, emitted, iv.result, state)
        replacements = list(emitted.ops)
        for operation in replacements:
            emitted.detach_op(operation)
        Rewriter.replace_op(loop, replacements, list(state))
        return
    bulk_upper = witness.lower + witness.full_groups * witness.factor * witness.step
    stride = arith.ConstantOp.from_int_and_width(witness.factor * witness.step, builtin.IndexType())
    bound = arith.ConstantOp.from_int_and_width(bulk_upper, builtin.IndexType())
    bulk_body = Block(arg_types=[argument.type for argument in source.args])
    state = tuple(bulk_body.args[1:])
    for lane in range(witness.factor):
        iv = bulk_body.args[0]
        if lane:
            offset = arith.ConstantOp.from_int_and_width(lane * witness.step, builtin.IndexType())
            addition = arith.AddiOp(iv, offset.result)
            bulk_body.add_ops([offset, addition])
            iv = addition.result
        state = _repeat_body(source, bulk_body, iv, state)
    bulk_body.add_op(scf.YieldOp(*state))
    bulk = scf.ForOp(loop.lb, bound.result, stride.result, loop.iter_args, bulk_body)
    replacements: list[Operation] = [stride, bound, bulk]
    result = bulk.results
    if witness.remainder:
        tail_body = Block(arg_types=[argument.type for argument in source.args])
        mapper = dict(zip(source.args, tail_body.args, strict=True))
        tail_body.add_ops(operation.clone(mapper) for operation in source.ops)
        tail = scf.ForOp(bound.result, loop.ub, loop.step, bulk.results, tail_body)
        replacements.append(tail)
        result = tail.results
    Rewriter.replace_op(loop, replacements, list(result))


def unroll_ordered_loops(
    module: builtin.ModuleOp, factors: Sequence[int], *, index_bits: int, max_added_operations: int = 4096
) -> tuple[UnrollWitness, ...]:
    """Fully inline small loops or partially unroll a bounded bulk plus exact tail.

    Factors index original nesting depth. Dynamic/negative bounds and unsupported
    metadata remain unchanged. Requested arithmetic overflow or code growth
    refuses the transformation. No tensor elements or trip-count-sized structures
    are allocated. Callers should transform a private clone for transactional use.
    """
    if (
        type(index_bits) is not int
        or not 2 <= index_bits <= 128
        or type(max_added_operations) is not int
        or max_added_operations < 1
        or any(type(factor) is not int or factor < 1 for factor in factors)
    ):
        raise ValueError("unrolling requires explicit positive factors, index width and growth budget")
    module.verify()
    original_count = sum(1 for _ in module.walk())
    candidates = [
        (index, operation, _depth(operation), ir_digest(operation))
        for index, operation in enumerate(module.walk())
        if isinstance(operation, scf.ForOp)
    ]
    witnesses = []
    for index, loop, depth, source_hash in reversed(candidates):
        if depth >= len(factors) or factors[depth] == 1 or loop.attributes or loop.properties:
            continue
        lower, upper, step = (constant_integer(value) for value in (loop.lb, loop.ub, loop.step))
        if lower is None or upper is None or step is None or lower < 0 or upper < lower or step < 1:
            continue
        if not all(isinstance(value.type, builtin.IndexType) for value in (loop.lb, loop.ub, loop.step)):
            continue
        factor = factors[depth]
        if upper + step * factor > (1 << (index_bits - 1)) - 1:
            raise ValueError("unrolled loop arithmetic exceeds the selected signed index width")
        count = (upper - lower + step - 1) // step
        groups, remainder = divmod(count, factor)
        # A conservative bound is checked before cloning any repeated body.
        body_size = sum(1 for _ in loop.walk())
        growth = min(count, factor) * (body_size + 3) + body_size + 8
        if sum(1 for _ in module.walk()) - original_count + growth > max_added_operations:
            raise ValueError("unrolling exceeds the selected operation growth budget")
        witness = UnrollWitness(
            index,
            lower,
            upper,
            step,
            factor,
            index_bits,
            count,
            groups,
            remainder,
            source_hash,
            tuple(str(value.type) for value in loop.iter_args),
        )
        _unroll_one(loop, witness)
        witnesses.append(witness)
    module.verify()
    return tuple(reversed(witnesses))


def validate_unrolled_change(change) -> None:
    """Bind actual emitted instructions, IV substitutions and state to the source.

    Trusted structural replay is explicit; arithmetic/fold obligations are checked
    by the independent host Lean route, not by this correspondence validator.
    """
    expected = {"factors", "index_bits", "max_added_operations"}
    if set(change.parameters) != expected:
        raise ValueError("unrolling parameters differ from the explicit source selection")
    change.validate_identity()
    source = parse_ir(change.before_ir)
    witnesses = unroll_ordered_loops(
        source,
        change.parameters["factors"],
        index_bits=change.parameters["index_bits"],
        max_added_operations=change.parameters["max_added_operations"],
    )
    if witnesses != change.certificate.witnesses or ir_text(source) != ir_text(parse_ir(change.after_ir)):
        raise ValueError("emitted unrolling differs from actual ordered source bodies and yields")
