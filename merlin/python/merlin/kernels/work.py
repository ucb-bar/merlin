"""How much arithmetic a linalg op actually performs, recovered from its iteration space.

An op's cost proxy has to come from the LOOP NEST, not from the result shape. A matmul's ``K`` and a
convolution's reduction window do not appear in the output, so an output-element count understates a
contraction by exactly the factor that makes it expensive, and ranks a large elementwise op above a
small GEMM. The nest is recovered from the indexing maps:

    work = (product of iteration-space extents) * (scalar arith/math ops in the body)

For each iteration dim ``d`` the extent is the operand-shape entry at any map result that is the bare
``AffineDimExpr(d)``. Compound results (``d2*16 + d5`` in a conv) are skipped, because the same dims
occur bare in the filter map and supply the extent there — so a conv's full 7-deep nest is still
exact. A dim that ONLY ever occurs compounded cannot be pinned down, and
:func:`iteration_space` says so with its ``complete`` flag rather than silently understating.

``footprint_bytes`` is the complementary axis. A family can be negligible on arithmetic and dominant
on traffic, so a ranking on one alone is misleading and both are reported.

This lives in the library rather than in the census script that first grew it because two callers now
need the same weighting — the cross-model family census
(``build_tools/scripts/model_op_census.py``) and the per-contraction census (:mod:`.census`) — and two
copies of a cost proxy drift into two different rankings of the same model.
"""
from __future__ import annotations

from typing import Any

from ..common import mlir_query as mq

__all__ = ["body_arith_ops", "footprint_bytes", "iteration_space", "work_of"]

#: Named linalg ops carry no region, so their body arithmetic is implicit. Value = scalar arith ops
#: performed per iteration-space point.
NAMED_BODY_OPS: dict[str, int] = {
    "linalg.matmul": 2,          # mul + add
    "linalg.batch_matmul": 2,
    "linalg.matvec": 2,
    "linalg.fill": 0,            # a store, no arithmetic
    "linalg.copy": 0,
    "linalg.transpose": 0,       # pure movement
    "linalg.broadcast": 0,
    "linalg.reduce": 1,
}

#: Named contractions whose iteration space is NOT just the result shape: the reduction dim has to
#: come from an input. Value = (operand index, dim index within that operand).
NAMED_EXTRA_DIM: dict[str, tuple[int, int]] = {
    "linalg.matmul": (0, -1),        # K = lhs last dim
    "linalg.batch_matmul": (0, -1),
    "linalg.matvec": (0, -1),
}


def _shape_of(t) -> tuple[list[int], str]:
    try:
        return mq.type_shape_dtype(t)
    except Exception:  # noqa: BLE001 — non-tensor operand (scalar/index): no shape, no footprint
        return [], ""


def _generic_iteration_space(op) -> tuple[int, bool]:
    maps = op.properties.get("indexing_maps") or op.attributes.get("indexing_maps")
    if maps is None:
        return 0, False
    from xdsl.ir.affine import AffineDimExpr
    tensors = [*op.operands, *op.results]
    extents: dict[int, int] = {}
    ndims = 0
    for amap_attr, tensor in zip(list(maps.data), tensors):
        amap = amap_attr.data
        ndims = max(ndims, amap.num_dims)
        shape, _ = _shape_of(tensor.type)
        if len(shape) != len(amap.results):
            continue
        for res, extent in zip(amap.results, shape):
            if isinstance(res, AffineDimExpr):
                extents[res.position] = max(extents.get(res.position, 0), extent)
    if ndims == 0:
        return 0, False
    prod = 1
    for v in extents.values():
        prod *= max(v, 1)
    return prod, len(extents) == ndims


def _named_iteration_space(op, name: str) -> int:
    if name == "linalg.reduce" and op.operands:
        in_shape, _ = _shape_of(op.operands[0].type)
        iters = 1
        for d in in_shape:
            iters *= max(d, 1)
        return iters
    res_shape: list[int] = []
    if op.results:
        res_shape, _ = _shape_of(op.results[0].type)
    iters = 1
    for d in res_shape:
        iters *= max(d, 1)
    extra = NAMED_EXTRA_DIM.get(name)
    if extra is not None and op.operands:
        lhs_shape, _ = _shape_of(op.operands[extra[0]].type)
        if lhs_shape:
            iters *= max(lhs_shape[extra[1]], 1)
    return iters


def iteration_space(op) -> tuple[int, bool]:
    """``(product of iteration-space extents, complete?)`` for any linalg op.

    ``complete`` is False when at least one iteration dim never occurred as a bare dim expression in
    any indexing map, i.e. the returned product is a LOWER BOUND. Callers must propagate that rather
    than round it away — a partially recovered nest that reads as exact is how a heavy op gets ranked
    light.
    """
    name = mq.op_name(op)
    if name == "linalg.generic":
        return _generic_iteration_space(op)
    return _named_iteration_space(op, name), True


def body_arith_ops(op) -> int:
    """Scalar arithmetic ops in ``op``'s region — yields, constants and index reads are not work.

    A named contraction has no region, so its implicit body count comes from
    :data:`NAMED_BODY_OPS` (2 for a multiply-accumulate), defaulting to 1 for an unlisted named op so
    an unrecognized op is weighted as doing something rather than nothing.
    """
    name = mq.op_name(op)
    if name != "linalg.generic":
        return NAMED_BODY_OPS.get(name, 1)
    n = 0
    for region in op.regions:
        for inner in region.walk():
            inner_name = mq.op_name(inner)
            if inner_name.startswith(("arith.", "math.")) and not inner_name.endswith(".constant"):
                n += 1
    return n


def footprint_bytes(op) -> int:
    """Operand + result footprint in bytes. A dtype with no known width contributes 0, so a
    footprint is an underestimate rather than a guess at the element size."""
    return sum(mq.value_bytes(v.type) for v in [*op.operands, *op.results])


def work_of(op) -> tuple[int, bool]:
    """``(work, complete?)`` — iteration-space size times body arithmetic."""
    iters, complete = iteration_space(op)
    return iters * body_arith_ops(op), complete


def is_weighable(op: Any) -> bool:
    """Whether ``op`` is a linalg op this module weights (structural ops, not terminators)."""
    name = mq.op_name(op)
    return name.startswith("linalg.") and name not in ("linalg.yield", "linalg.index")
