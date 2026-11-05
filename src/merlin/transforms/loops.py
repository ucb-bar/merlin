"""Ordered loop segmentation with constant-size generated IR."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, SSAValue
from xdsl.rewriter import Rewriter

from .certificates import LoopWitness, ir_digest
from .normalization import constant_integer


@dataclass(frozen=True)
class Segment:
    lower: int
    upper: int
    step: int


def iter_segments(lower: int, upper: int, step: int, tile_iterations: int) -> Iterator[Segment]:
    """Lazy execution/reference utility; compiler emission never enumerates this."""
    if lower < 0 or upper < lower or step <= 0 or tile_iterations <= 0:
        raise ValueError("segmentation requires nonnegative bounds and positive steps/tiles")
    for start in range(lower, upper, step * tile_iterations):
        yield Segment(start, min(start + step * tile_iterations, upper), step)


def _loop_depth(op: scf.ForOp) -> int:
    depth = 0
    parent = op.parent_op()
    while parent is not None:
        depth += isinstance(parent, scf.ForOp)
        parent = parent.parent_op()
    return depth


def segment_loops(
    module: builtin.ModuleOp, tile_iterations: Sequence[int], *, index_bits: int
) -> tuple[LoopWitness, ...]:
    """Strip-mine loops by their original nesting depth, retaining exact order.

    Every iter_arg is forwarded through the inner and outer yields. Thus ordered
    reductions retain the original accumulator after each element, including
    wrapping arithmetic. Arbitrary body effects remain in their original order.
    Negative/dynamic bounds and non-index induction types are left unchanged.
    Requested static loops whose generated index arithmetic would overflow refuse.
    """
    if index_bits <= 1 or any(v <= 0 for v in tile_iterations):
        raise ValueError("positive tile iterations and an explicit index width are required")
    candidates = [
        (index, op, _loop_depth(op), ir_digest(op))
        for index, op in enumerate(module.walk())
        if isinstance(op, scf.ForOp)
    ]
    witnesses = []
    for index, loop, depth, original_digest in reversed(candidates):
        if depth >= len(tile_iterations):
            continue
        lower, upper, step = (constant_integer(v) for v in (loop.lb, loop.ub, loop.step))
        if lower is None or upper is None or step is None:
            continue
        if lower < 0 or upper < lower or step <= 0 or not isinstance(loop.lb.type, builtin.IndexType):
            continue
        tile = tile_iterations[depth]
        block_step = step * tile
        maximum = (1 << (index_bits - 1)) - 1
        # Both the bound add and final loop increments fit the selected index type.
        if upper + block_step > maximum or upper + step > maximum:
            raise ValueError("segmented loop index arithmetic exceeds the selected signed index width")
        if tile == 1:
            continue
        body = loop.body.block
        witness = LoopWitness(
            index, lower, upper, step, tile, index_bits, tuple(str(v.type) for v in loop.iter_args), original_digest
        )
        block_step_op = arith.ConstantOp.from_int_and_width(block_step, builtin.IndexType())
        outer_body = Block(arg_types=[builtin.IndexType(), *(v.type for v in loop.iter_args)])
        upper_add = arith.AddiOp(outer_body.args[0], block_step_op.result)
        upper_min = arith.MinUIOp(upper_add.result, loop.ub)
        inner_body = Block(arg_types=[v.type for v in body.args])
        mapper: dict[SSAValue, SSAValue] = dict(zip(body.args, inner_body.args, strict=True))
        inner_body.add_ops(op.clone(mapper) for op in body.ops)
        inner = scf.ForOp(outer_body.args[0], upper_min.result, loop.step, outer_body.args[1:], inner_body)
        inner.attributes.update(loop.attributes)
        outer_body.add_ops([upper_add, upper_min, inner, scf.YieldOp(*inner.results)])
        outer = scf.ForOp(loop.lb, loop.ub, block_step_op.result, loop.iter_args, outer_body)
        Rewriter.replace_op(loop, [block_step_op, outer], list(outer.results))
        witnesses.append(witness)
    return tuple(reversed(witnesses))


def ordered_loop_obligation(witness: LoopWitness) -> dict[str, int]:
    """The numerical projection consumed by a parameterized partition theorem."""
    return {
        "lower": witness.lower,
        "upper": witness.upper,
        "step": witness.step,
        "tile_iterations": witness.tile_iterations,
        "index_bits": witness.index_bits,
    }
