"""Explicit native source forms without extending host admission schemas."""
from __future__ import annotations

from .linalg_patterns import InvalidLinalgPattern, StaticProjectedPointwisePattern, _checked_static_indexed_linalg_shell, _checked_static_linalg_shell, _recognize_static_scalar_pointwise

def recognize_static_f32_division(op) -> StaticProjectedPointwisePattern:
    """Check projected, unflagged f32 division without admitting it to host schemas."""
    shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    scalar = _recognize_static_scalar_pointwise(
        op, shell, additional_signatures={"arith.divf": (("f32", "f32"), "f32")},
    )
    if scalar.operation != "arith.divf":
        raise InvalidLinalgPattern("f32 division needs exactly one arith.divf and yield")
    return StaticProjectedPointwisePattern(
        scalar.operation,
        scalar.shape,
        scalar.ordered_types,
        tuple(tuple(value.type.get_shape()) for value in op.inputs),
        tuple(str(mapping) for mapping in shell.input_maps),
        scalar.predicate,
    )


def recognize_f32_relu_ordered_select(op) -> tuple[object, object]:
    """Recognize the closed f32 `x < +0 ? +0 : x` scalar region.

    This is a source arithmetic form, not a target ReLU admission. Its compare
    is ordered, so NaNs take the original input arm and negative zero survives.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.ir.affine import AffineMap
    from xdsl.traits import Pure

    shell = _checked_static_indexed_linalg_shell(op)
    rank = len(shell.shape)
    if (
        len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1
        or shell.ordered_types != ("f32", "f32", "f32")
        or shell.input_shapes != (shell.shape,)
        or shell.output_shapes != (shell.shape,)
        or shell.indexing_maps != (AffineMap.identity(rank),) * 2
        or shell.iterator_types != (IteratorType.PARALLEL,) * rank
    ):
        raise InvalidLinalgPattern("f32 zero select needs identity maps and one static f32 input/output")
    body = shell.body
    if tuple(item.name for item in body) != (
        "arith.constant", "arith.sitofp", "arith.cmpf", "arith.select", "linalg.yield",
    ):
        raise InvalidLinalgPattern("f32 zero select needs integer zero, cast, ordered compare and select")
    integer, cast, compare, select, yld = body
    for item, allowed in zip(body[:-1], ({"value"}, set(), {"predicate", "fastmath"}, set()), strict=True):
        if (
            not type(item).has_trait(Pure) or item.regions or item.successors
            or any(not name.startswith("prov.") for name in item.attributes)
            or set(item.properties) - allowed
        ):
            raise InvalidLinalgPattern("f32 zero select has an effect or unknown scalar property")
    zero = integer.properties.get("value")
    predicate = compare.properties.get("predicate")
    flags = compare.properties.get("fastmath")
    if (
        not isinstance(zero, IntegerAttr) or str(zero.type) != "i64" or zero.value.data != 0
        or not isinstance(predicate, IntegerAttr) or str(predicate.type) != "i64"
        or predicate.value.data != 4  # ordered less than
        or flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data)
        or any(len(item.results) != 1 for item in body[:-1])
        or str(integer.results[0].type) != "i64"
        or str(cast.results[0].type) != "f32" or str(compare.results[0].type) != "i1"
        or str(select.results[0].type) != "f32"
        or tuple(cast.operands) != (integer.results[0],)
        or tuple(compare.operands) != (shell.args[0], cast.results[0])
        or tuple(select.operands) != (compare.results[0], cast.results[0], shell.args[0])
        or type(yld) is not YieldOp or yld.regions or yld.successors or yld.properties
        or any(not name.startswith("prov.") for name in yld.attributes)
        or tuple(yld.operands) != (select.results[0],)
    ):
        raise InvalidLinalgPattern("f32 zero select scalar body differs from ordered less-than and select")
    return op.inputs[0], op.outputs[0]

