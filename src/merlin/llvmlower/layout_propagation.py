"""Carry demanded tensor permutations through exact scalar operations.

Transpose cancellation is proved from permutations and affine maps, without
changing scalar arithmetic. Static slices and padding follow the same axis
permutation. Unknown operations retain an explicit transpose at their boundary.
The pass is opt-in; catalog binding must happen after this rewrite.
Optional reduction channel blocking is an explicit scheduling choice, disabled
by default. It changes only independent-channel interleaving, never reduction
order, and requires a divisible static trailing channel extent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod

from xdsl.dialects import tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    IntegerAttr,
    NoneAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineDimExpr, AffineExpr, AffineMap


@dataclass
class LayoutReport:
    requested: int = 0
    canceled: int = 0
    unit_views: int = 0
    scalar_regions: int = 0
    static_slices: int = 0
    ordered_reductions: int = 0
    blocked_reductions: int = 0
    boundary_transposes: int = 0

    def to_dict(self):
        return asdict(self)


def _static_type(value):
    ty = value.type
    if not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr) or any(x <= 0 for x in ty.get_shape()):
        return None
    return ty


def _permutation(op):
    """Read either a named transpose or the exact generic copy equivalent."""
    if op.name == "linalg.transpose":
        return tuple(op.permutation.get_values())
    if op.name != "linalg.generic" or len(op.operands) != 2 or len(op.results) != 1:
        return None
    ty = _static_type(op.results[0])
    if ty is None:
        return None
    rank = len(ty.get_shape())
    maps = op.properties["indexing_maps"].data
    if len(maps) != 2 or maps[1].data != AffineMap.identity(rank):
        return None
    source = maps[0].data
    if source.num_symbols or source.num_dims != rank or len(source.results) != rank:
        return None
    if not all(isinstance(x, AffineDimExpr) for x in source.results):
        return None
    inverse = tuple(x.position for x in source.results)
    if sorted(inverse) != list(range(rank)):
        return None
    if any(x.data.value != "parallel" for x in op.properties["iterator_types"].data):
        return None
    body = op.regions[0].block
    if len(body.args) != 2 or len(list(body.ops)) != 1:
        return None
    if body.last_op.attributes or body.last_op.properties:
        return None  # Carry annotated scalar regions rather than erasing their attribution.
    if body.last_op.name != "linalg.yield" or list(body.last_op.operands) != [body.args[0]]:
        return None
    return tuple(inverse.index(i) for i in range(rank))


def _input_layout(value, block):
    """Find a common existing permutation through exact pointwise producers.

    A full-rank unknown operand prevents moving a reduction; scalar/splat
    operands have no preferred physical layout. No producer is changed here.
    """
    ty = _static_type(value)
    op = value.owner
    if ty is None or not isinstance(op, Operation) or op.parent is not block:
        return None
    permutation = _permutation(op)
    if permutation is not None:
        return tuple(permutation.index(i) for i in range(len(permutation)))
    if op.name != "linalg.generic" or len(op.results) != 1:
        return None
    rank = len(ty.get_shape())
    maps = [m.data for m in op.properties["indexing_maps"].data]
    if (
        any(it.data.value != "parallel" for it in op.properties["iterator_types"].data)
        or len(op.properties["iterator_types"].data) != rank
        or len(op.outputs) != 1
        or maps[-1] != AffineMap.identity(rank)
        or not all(
            x.name.startswith(("arith.", "math.")) or x.name == "linalg.yield"
            for region in op.regions
            for x in region.walk()
        )
    ):
        return None
    candidates = []
    for operand, index_map in zip(op.inputs, maps):
        operand_ty = _static_type(operand)
        if operand_ty is None:
            return None
        if not operand_ty.get_shape() and index_map == AffineMap(rank, 0, ()):
            continue
        if operand_ty.get_shape() != ty.get_shape() or index_map != AffineMap.identity(rank):
            return None
        if isinstance(operand.owner, Operation) and operand.owner.name == "tensor.splat":
            continue
        candidate = _input_layout(operand, block)
        if candidate is None:
            return None
        candidates.append(candidate)
    return candidates[0] if candidates and all(x == candidates[0] for x in candidates) else None


def rewrite_module(module, *, reduction_channel_block: int = 0, pointwise_only: bool = False,
                   maximum_added_operations: int = 8192) -> LayoutReport:
    if type(reduction_channel_block) is not int or reduction_channel_block < 0:
        raise ValueError("reduction_channel_block must be a nonnegative integer")
    if type(pointwise_only) is not bool or type(maximum_added_operations) is not int or maximum_added_operations <= 0:
        raise ValueError("layout checking requires an explicit family and positive code budget")
    report = LayoutReport()
    cache = {}
    segments = {}
    added = 0
    if pointwise_only:
        from .layout_conformance import pointwise_kind

        for region in (region for op in module.walk() for region in op.regions):
            for block in region.blocks:
                segment = 0
                for op in block.ops:
                    segments[op] = segment
                    if pointwise_kind(op) is None:
                        segment += 1

    def eligible(op):
        return not pointwise_only or pointwise_kind(op) is not None

    def add(op, anchor):
        nonlocal added
        added += sum(1 for _ in op.walk())
        if pointwise_only and added > maximum_added_operations:
            raise ValueError("layout exceeds its explicit compilation-space cost contract")
        anchor.parent.insert_op_before(op, anchor)
        if pointwise_only:
            segments[op] = segments[anchor]
        return op.results[0]

    def demand(value, permutation, anchor):
        ty = _static_type(value)
        if ty is None or sorted(permutation) != list(range(len(ty.get_shape()))):
            raise ValueError("layout demand requires a positive static tensor and a permutation")
        rank = len(permutation)
        if permutation == tuple(range(rank)):
            return value
        key = (anchor.parent, value, permutation, segments.get(anchor))
        if key in cache:
            return cache[key]
        shape = ty.get_shape()
        out_ty = TensorType(ty.get_element_type(), [shape[i] for i in permutation])
        op = value.owner
        same_block = (isinstance(op, Operation) and op.parent is anchor.parent and eligible(op)
                      and (not pointwise_only or segments[op] == segments[anchor]))
        result = None
        if same_block and (old := _permutation(op)) is not None:
            report.canceled += 1
            result = demand(op.operands[0], tuple(old[i] for i in permutation), anchor)
        # Moving only unit axes preserves the flat element order and is a view.
        elif ([i for i in permutation if shape[i] != 1] == [i for i in range(rank) if shape[i] != 1]
              and (not pointwise_only or sum(d != 1 for d in shape) <= 1)):
            reassoc = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in range(rank)])])
            flat = add(
                tensor.CollapseShapeOp.create(
                    operands=[value],
                    result_types=[TensorType(ty.get_element_type(), [prod(shape)])],
                    properties={"reassociation": reassoc},
                ),
                anchor,
            )
            result = add(tensor.ExpandShapeOp(flat, [], reassoc, out_ty.get_shape(), out_ty), anchor)
            report.unit_views += 1
        elif same_block and op.name in ("tensor.empty", "tensor.splat"):
            if op.name == "tensor.empty" and op.operands:
                raise ValueError("static tensor.empty unexpectedly has dynamic operands")
            result = add(
                type(op).create(
                    operands=list(op.operands),
                    result_types=[out_ty],
                    properties=dict(op.properties),
                    attributes=dict(op.attributes),
                ),
                anchor,
            )
        elif not pointwise_only and same_block and op.name in ("tensor.extract_slice", "tensor.insert_slice"):
            expected = 1 if op.name == "tensor.extract_slice" else 2
            arrays = [op.properties.get(name) for name in ("static_offsets", "static_sizes", "static_strides")]
            if (
                len(op.operands) == expected
                and all(_static_type(x) is not None and len(x.type.get_shape()) == rank for x in op.operands)
                and all(x is not None and len(tuple(x.get_values())) == rank for x in arrays)
            ):
                data = [tuple(x.get_values()) for x in arrays]
                if all(x >= 0 for x in data[0]) and all(x > 0 for row in data[1:] for x in row):
                    inputs = [demand(x, permutation, anchor) for x in op.operands]
                    props = dict(op.properties)
                    for name, row in zip(("static_offsets", "static_sizes", "static_strides"), data):
                        props[name] = DenseArrayBase.from_list(i64, [row[i] for i in permutation])
                    result = add(
                        type(op).create(
                            operands=inputs, result_types=[out_ty], properties=props, attributes=dict(op.attributes)
                        ),
                        anchor,
                    )
                    report.static_slices += 1
        elif (same_block and op.name == "linalg.generic" and len(op.results) == 1
              and (not pointwise_only or op.results[0].has_one_use())):
            maps = [x.data for x in op.properties["indexing_maps"].data]
            iters = [x.data.value for x in op.properties["iterator_types"].data]
            # Keep the relative order of reduction axes and every scalar instruction.
            # linalg.index and effects would observe a changed iteration coordinate.
            scalar = all(
                x.name.startswith(("arith.", "math.")) or x.name == "linalg.yield"
                for region in op.regions
                for x in region.walk()
            )
            output_map = AffineMap(len(iters), 0, tuple(AffineExpr.dimension(i) for i in range(rank)))
            if (
                scalar
                and len(op.outputs) == 1
                and maps[-1] == output_map
                and iters[:rank] == ["parallel"] * rank
                and all(x == "reduction" for x in iters[rank:])
                and all(m.num_symbols == 0 and m.num_dims == len(iters) for m in maps)
            ):
                inverse = [permutation.index(i) for i in range(rank)]
                dims = [AffineExpr.dimension(inverse[i] if i < rank else i) for i in range(len(iters))]
                inputs, transformed_maps = [], []
                for operand, old_map in zip(op.operands, maps):
                    new_map = old_map.replace_dims_and_symbols(dims, [], len(iters), 0)
                    operand_ty = _static_type(operand)
                    if operand_ty is not None and len(operand_ty.get_shape()) == rank:
                        producer = operand.owner
                        shared = (pointwise_only and isinstance(producer, Operation)
                                  and producer.name == "linalg.generic" and not operand.has_one_use())
                        if shared:
                            # Read the existing shared value through its composed
                            # map. Cloning its arithmetic has no known cost benefit.
                            inputs.append(operand)
                        else:
                            inputs.append(demand(operand, permutation, anchor))
                            new_map = AffineMap(len(iters), 0, tuple(new_map.results[i] for i in permutation))
                    else:
                        inputs.append(operand)
                    transformed_maps.append(AffineMapAttr(new_map))
                props = dict(op.properties)
                props["indexing_maps"] = ArrayAttr(transformed_maps)
                result = add(
                    type(op).create(
                        operands=inputs,
                        result_types=[out_ty],
                        properties=props,
                        attributes=dict(op.attributes),
                        regions=[r.clone() for r in op.regions],
                    ),
                    anchor,
                )
                report.scalar_regions += 1
        if result is None:
            empty = add(tensor.EmptyOp([], out_ty), anchor)
            result = add(TransposeOp(value, empty, DenseArrayBase.from_list(i64, permutation), out_ty), anchor)
            report.boundary_transposes += 1
        cache[key] = result
        return result

    originals = [(op, _permutation(op) if eligible(op) else None) for op in module.walk()]
    for op, permutation in originals:
        if (not eligible(op) or permutation is None or not op.results[0].uses
                or _static_type(op.results[0]) is None):
            continue
        report.requested += 1
        result = demand(op.operands[0], permutation, op)
        if pointwise_only:
            # Cancellation may expose an alias rather than a fresh snapshot.
            # Keep the original materialization if a later opaque effect lies
            # between it and an observation of that alias.
            from .layout_conformance import unit_source

            backing = result
            while isinstance(backing.owner, Operation) and pointwise_kind(backing.owner) == "unit_view":
                backing = unit_source(backing.owner)
            snapshot = (isinstance(backing.owner, Operation)
                        and pointwise_kind(backing.owner) in {"generic", "transpose", "splat", "constant"})
            if not snapshot and any(use.operation.parent is not op.parent
                                    or segments.get(use.operation) != segments[op] for use in op.results[0].uses):
                continue
        op.results[0].replace_all_uses_with(result)
        for key, cached in list(cache.items()):
            if cached is op.results[0]:
                cache[key] = result
            if key[1] is op.results[0]:
                cache.setdefault((key[0], result, key[2], key[3]), cache.pop(key))
        op.parent.erase_op(op)
    if pointwise_only:
        # Remove only known, unused tensor computations. Scalar region order and
        # opaque operations remain intact; the independent checker sees the result.
        for operation in reversed(list(module.walk())):
            if (operation.parent is not None and eligible(operation) and operation.results
                    and all(isinstance(value.type, TensorType) and not value.uses for value in operation.results)):
                operation.parent.erase_op(operation)
        module.verify()
        return report
    # Consume a preferred input layout at a reduction boundary. Only move
    # parallel axes across reduction axes; preserve both subsequences exactly.
    # In particular floating-point reductions retain their original scalar order.
    cache.clear()  # Earlier demands may have been inserted after this reduction.
    for op in list(module.walk()):
        if op.name != "linalg.reduce" or len(op.operands) != 2 or len(op.results) != 1 or len(op.regions) != 1:
            continue
        source, init = op.operands
        ty = _static_type(source)
        if ty is None or _static_type(init) is None or _static_type(op.results[0]) is None:
            continue
        permutation = _input_layout(source, op.parent)
        if permutation is None:
            continue
        rank = len(ty.get_shape())
        reduced = tuple(op.dimensions.get_values())
        parallel = tuple(i for i in range(rank) if i not in reduced)
        if (
            reduced != tuple(sorted(set(reduced)))
            or not reduced
            or tuple(i for i in permutation if i in reduced) != reduced
            or tuple(i for i in permutation if i not in reduced) != parallel
        ):
            continue
        if not all(
            x.name.startswith(("arith.", "math.")) or x.name == "linalg.yield"
            for region in op.regions
            for x in region.walk()
        ):
            continue
        transformed = demand(source, permutation, op)
        # Explicit parallel-then-reduction loop order keeps one accumulator
        # hot while retaining the original reduction-axis lexicographic order.
        # A named reduce over interleaved physical axes otherwise streams every
        # accumulator through memory on each reduction iteration.
        loop_axes = (*parallel, *reduced)
        input_map = AffineMap(rank, 0, tuple(AffineExpr.dimension(loop_axes.index(i)) for i in permutation))
        output_map = AffineMap(rank, 0, tuple(AffineExpr.dimension(i) for i in range(len(parallel))))
        block = reduction_channel_block
        if block and parallel and permutation[-1] == parallel[-1] and ty.get_shape()[parallel[-1]] % block == 0:

            def split_last(value):
                old = value.type
                shape = old.get_shape()
                reassociation = ArrayAttr(
                    [ArrayAttr([IntegerAttr(i, i64)]) for i in range(len(shape) - 1)]
                    + [ArrayAttr([IntegerAttr(len(shape) - 1, i64), IntegerAttr(len(shape), i64)])]
                )
                expanded = TensorType(old.get_element_type(), [*shape[:-1], shape[-1] // block, block])
                result = add(tensor.ExpandShapeOp(value, [], reassociation, expanded.get_shape(), expanded), op)
                return result, reassociation

            split_source, _ = split_last(transformed)
            split_init, output_reassociation = split_last(init)
            input_dims = [AffineExpr.dimension(loop_axes.index(i)) for i in permutation[:-1]]
            input_dims += [AffineExpr.dimension(len(parallel) - 1), AffineExpr.dimension(rank)]
            output_dims = [AffineExpr.dimension(i) for i in range(len(parallel))] + [AffineExpr.dimension(rank)]
            replacement = GenericOp(
                [split_source],
                [split_init],
                op.regions[0].clone(),
                [
                    AffineMapAttr(AffineMap(rank + 1, 0, tuple(input_dims))),
                    AffineMapAttr(AffineMap(rank + 1, 0, tuple(output_dims))),
                ],
                [IteratorTypeAttr.parallel()] * len(parallel)
                + [IteratorTypeAttr.reduction()] * len(reduced)
                + [IteratorTypeAttr.parallel()],
                [split_init.type],
            )
            replacement.attributes.update(op.attributes)
            expanded_result = add(replacement, op)
            result = add(
                tensor.CollapseShapeOp.create(
                    operands=[expanded_result],
                    result_types=[op.results[0].type],
                    properties={"reassociation": output_reassociation},
                ),
                op,
            )
            report.blocked_reductions += 1
        else:
            replacement = GenericOp(
                [transformed],
                [init],
                op.regions[0].clone(),
                [AffineMapAttr(input_map), AffineMapAttr(output_map)],
                [IteratorTypeAttr.parallel()] * len(parallel) + [IteratorTypeAttr.reduction()] * len(reduced),
                [op.results[0].type],
            )
            replacement.attributes.update(op.attributes)
            result = add(replacement, op)
        op.results[0].replace_all_uses_with(result)
        op.parent.erase_op(op)
        report.ordered_reductions += 1
    module.verify()
    return report
