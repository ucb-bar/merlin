"""Expose single-iteration reductions as parallel scalar tensor arithmetic.

The scalar body, including its accumulator input and rounding, is unchanged.
This is useful before placement: a degenerate contraction need not acquire a
matrix accelerator's operand quantization just because it was expressed as a
reduction. Placement and profitability remain explicit caller decisions.
"""

from __future__ import annotations

from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorType
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from .tensor_chains import pure_tensor_compute


def singleton_reduction_axes(op) -> tuple[int, ...]:
    """Prove independent output iterations and statically unit reductions.

    Restrict to tensor generics with projected dimension maps. Unknown/dynamic
    extents, symbols, index observers, captures, and effectful bodies refuse.
    Every parallel iterator must identify a distinct output dimension.
    """
    if (not isinstance(op, GenericOp) or not pure_tensor_compute(op)
            or len(op.outputs) != 1 or len(op.results) != 1
            or any(step.name == "linalg.index" for step in op.body.walk())):
        return ()
    iterators = tuple(it.data for it in op.iterator_types)
    reductions = tuple(i for i, it in enumerate(iterators) if it == IteratorType.REDUCTION)
    parallel = tuple(i for i, it in enumerate(iterators) if it == IteratorType.PARALLEL)
    if not reductions or len(parallel) + len(reductions) != len(iterators):
        return ()
    maps = tuple(attr.data for attr in op.indexing_maps)
    values = (*op.inputs, *op.outputs)
    if len(maps) != len(values) or any(not isinstance(v.type, TensorType) for v in values):
        return ()
    extents = {}
    for value, mapping in zip(values, maps, strict=True):
        shape = value.type.get_shape()
        if (mapping.num_symbols or mapping.num_dims != len(iterators)
                or len(mapping.results) != len(shape)):
            return ()
        for expr, extent in zip(mapping.results, shape, strict=True):
            if (not isinstance(expr, AffineDimExpr) or extent <= 0
                    or expr.position >= len(iterators)):
                return ()
            previous = extents.setdefault(expr.position, extent)
            if previous != extent:
                return ()
    if any(extents.get(i) != 1 for i in reductions):
        return ()
    output_axes = tuple(expr.position for expr in maps[-1].results)
    if len(output_axes) != len(parallel) or set(output_axes) != set(parallel):
        return ()
    if op.results[0].type != op.outputs[0].type:
        return ()
    return reductions


def parallelize_singleton_reductions(module, *, select=None):
    """Return a verified fresh module and transformed operations in that module.

    An optional selector chooses among proven candidates; it cannot override
    legality. No floating arithmetic, initializer, type, or SSA edge is removed.
    The original module is never mutated.
    """
    result = module.clone()
    transformed = []
    for op in result.walk():
        removed = singleton_reduction_axes(op)
        if not removed or (select is not None and not select(op)):
            continue
        kept = tuple(i for i in range(len(op.iterator_types)) if i not in removed)
        positions = {old: new for new, old in enumerate(kept)}
        replacements = [AffineConstantExpr(0) if i in removed else
                        AffineDimExpr(positions[i]) for i in range(len(op.iterator_types))]
        op.properties["indexing_maps"] = ArrayAttr([
            AffineMapAttr(attr.data.replace_dims_and_symbols(
                replacements, (), len(kept), 0)) for attr in op.indexing_maps])
        op.properties["iterator_types"] = ArrayAttr([tuple(op.iterator_types)[i] for i in kept])
        transformed.append((op, removed))
    result.verify()
    return result, tuple(transformed)
