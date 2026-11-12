"""``layer-norm-chunked-sums``: sum a LayerNorm row in contiguous chunks, then sum the chunks.

NUMERICS-CHANGING and default off. The rewrite changes the order of floating-point additions, so the
result is not bit-identical to the source reduction; it is a different numerical model and is graded
against its own reference. Selecting it (``--pass layer-norm-chunked-sums``) is the caller's explicit
permission for two things the source does not grant:

* reassociation of the row sum's f32 additions;
* finite intermediates. Overflow, infinities, NaN payloads and exception order are not preserved
  as an unrestricted IEEE contract. A partial sum that overflows in one order may stay finite in
  another.

Only a ``linalg.reduce`` whose source provenance says it computes a LayerNorm (``prov.op =
"layer_norm"``) is eligible: the operation kind, never a model, shape or target name. It must also be
an f32 ``arith.addf`` over the innermost axis with a static extent divisible by the chunk size, and a
``+0.0`` splat initial value. Anything else is left exactly as it was. The rewrite expands the reduced
axis into ``[groups, chunk]``, sums each chunk into a fresh ``+0.0``-filled partial tensor, and points
the original reduction at the partial sums. Every loop extent is an operand dimension, so the result
re-parses and re-applies to nothing.

Half-precision inputs are never accumulated here. A source LayerNorm already computes its sums in
f32, as its opmath decomposition requires, so an eligible sum is an f32 sum.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

#: The attribute recording that a reduction was chunked, with the permissions it was chunked under.
MARKER = "merlin.normalization_reassociation"
#: The provenance op kind that makes a reduction eligible.
SOURCE_OP = "layer_norm"
DEFAULT_CHUNK = 32


def _positive_zero_splat(init) -> bool:
    """The reduction's initial value is a ``tensor.splat`` of the constant ``+0.0``."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr

    owner = init.owner
    if getattr(owner, "name", None) != "tensor.splat" or len(owner.operands) != 1:
        return False
    constant = owner.operands[0].owner
    if not isinstance(constant, arith.ConstantOp) or not isinstance(constant.value, FloatAttr):
        return False
    value = constant.value.value.data
    return value == 0.0 and math.copysign(1.0, value) > 0


def _eligible_shape(op, chunk_size: int) -> list[int] | None:
    """The reduced operand's static shape when ``op`` is an eligible f32 innermost-axis add."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import TensorType, f32
    from xdsl.dialects.linalg import ops as linalg

    if getattr(op.attributes.get("prov.op"), "data", None) != SOURCE_OP or MARKER in op.attributes:
        return None
    if len(op.operands) != 2 or len(op.results) != 1:
        return None
    value, init = op.operands
    input_type, output_type = value.type, init.type
    if not isinstance(input_type, TensorType) or not isinstance(output_type, TensorType):
        return None
    if input_type.element_type != f32 or output_type.element_type != f32:
        return None
    shape = list(input_type.get_shape())
    rank = len(shape)
    if not rank or any(size <= 0 for size in shape):
        return None
    if shape[-1] < 2 * chunk_size or shape[-1] % chunk_size:
        return None
    if list(output_type.get_shape()) != shape[:-1] or op.results[0].type != output_type:
        return None
    if tuple(op.dimensions.iter_values()) != (rank - 1,):
        return None
    body = list(op.regions[0].block.ops)
    if len(body) != 2 or not isinstance(body[0], arith.AddfOp) or not isinstance(body[1], linalg.YieldOp):
        return None
    if list(body[0].operands) != list(op.regions[0].block.args) or list(body[1].operands) != [body[0].result]:
        return None
    # The original init takes part only in the final reduction; requiring +0 keeps a nonzero initial
    # accumulator from being added once per chunk.
    if not _positive_zero_splat(init):
        return None
    return shape


def chunk_layer_norm_sums(
    module,
    *,
    allow_reassociation: bool = False,
    assume_finite_intermediates: bool = False,
    chunk_size: int = DEFAULT_CHUNK,
    report_out: dict | None = None,
    on_rewrite: Callable[[Any, tuple[Any, ...], Any], None] | None = None,
) -> int:
    """Chunk every eligible LayerNorm row sum; returns how many were rewritten.

    Both numerical permissions must be given, or nothing changes. ``on_rewrite(source, generated,
    result_owner)`` reports each rewrite's new operations to a caller that accounts for them (the
    source transform map); the reduction keeps its results, so it is its own result owner.
    """
    if type(chunk_size) is not int or chunk_size < 2:
        raise ValueError("chunk_size must be an integer of at least two")
    report = report_out if report_out is not None else {}
    report.update(rewritten=0, opted_in=bool(allow_reassociation and assume_finite_intermediates))
    if not report["opted_in"]:
        return 0

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, StringAttr, TensorType, f32, i64
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import InsertPoint, Rewriter

    from merlin.compiler.ir.provenance import carry_provenance

    contract = StringAttr(f"chunk={chunk_size}; explicit reassociation; finite intermediates assumed")
    for op in list(module.walk()):
        if not isinstance(op, linalg.ReduceOp):
            continue
        shape = _eligible_shape(op, chunk_size)
        if shape is None:
            continue
        value, init = op.operands
        rank = len(shape)
        groups = shape[-1] // chunk_size
        partial_type = TensorType(f32, [*shape[:-1], groups])
        expanded_shape = [*shape[:-1], groups, chunk_size]
        reassociation = ArrayAttr(
            [ArrayAttr([IntegerAttr(i, i64)]) for i in range(rank - 1)]
            + [ArrayAttr([IntegerAttr(rank - 1, i64), IntegerAttr(rank, i64)])]
        )
        expanded = tensor.ExpandShapeOp(value, (), reassociation, expanded_shape, TensorType(f32, expanded_shape))
        zero = arith.ConstantOp(FloatAttr(0.0, f32))
        empty = tensor.EmptyOp((), partial_type)
        fill = linalg.FillOp((zero.result,), (empty.tensor,), res=(partial_type,))
        body = Block(arg_types=[f32, f32])
        add = arith.AddfOp(body.args[0], body.args[1])
        body.add_ops([add, linalg.YieldOp(add.result)])
        partial = linalg.GenericOp(
            inputs=(expanded.results[0],),
            outputs=(fill.results[0],),
            body=Region(body),
            indexing_maps=ArrayAttr(
                [
                    AffineMapAttr(AffineMap.identity(rank + 1)),
                    AffineMapAttr(AffineMap(rank + 1, 0, tuple(AffineDimExpr(i) for i in range(rank)))),
                ]
            ),
            iterator_types=ArrayAttr(
                [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * rank
                + [linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION)]
            ),
            result_types=(partial_type,),
        )
        generated = (expanded, zero, empty, fill, partial)
        for new in generated:
            carry_provenance(new, op, "layer_norm_chunked_sum")
        partial.attributes[MARKER] = contract
        op.attributes[MARKER] = contract
        Rewriter.insert_op(list(generated), InsertPoint.before(op))
        op.operands = (partial.results[0], init)
        if on_rewrite is not None:
            on_rewrite(op, generated, op)
        report["rewritten"] += 1
    return report["rewritten"]
