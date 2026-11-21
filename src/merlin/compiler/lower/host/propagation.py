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

from collections import Counter
from dataclasses import asdict, dataclass
from io import StringIO
from math import prod

from xdsl.dialects import arith, builtin, func, tensor
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
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp, YieldOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr, AffineMap

from merlin.compiler.ir.xdsl_io import DenseFloatPrinter
from merlin.compiler.lower.quant.readout import pure_scalar_body
from merlin.compiler.transforms.certificates import ir_text

# ---- layout_propagation ------------------------------------------------------------------------


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

    if per_tensor_quantization(op):
        return _input_layout(op.operands[0], block)
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
        # A projected channel vector has no conflicting physical permutation:
        # moving its unit axes is a view. Keep its exact affine broadcast map.
        # General multidimensional broadcasts still require their own layout.
        if (
            len(operand_ty.get_shape()) == rank
            and sum(d != 1 for d in operand_ty.get_shape()) <= 1
            and index_map.num_symbols == 0
            and index_map.num_dims == rank
            and len(index_map.results) == rank
            and all(
                (extent == 1 and expr == AffineExpr.constant(0))
                or (isinstance(expr, AffineDimExpr) and ty.get_shape()[expr.position] == extent)
                for extent, expr in zip(operand_ty.get_shape(), index_map.results)
            )
        ):
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


def rewrite_module(
    module,
    *,
    reduction_channel_block: int = 0,
    pointwise_only: bool = False,
    maximum_added_operations: int = 8192,
    selected_transposes=None,
) -> LayoutReport:

    if type(reduction_channel_block) is not int or reduction_channel_block < 0:
        raise ValueError("reduction_channel_block must be a nonnegative integer")
    if type(pointwise_only) is not bool or type(maximum_added_operations) is not int or maximum_added_operations <= 0:
        raise ValueError("layout checking requires an explicit family and positive code budget")
    report = LayoutReport()
    if selected_transposes is not None and (
        type(selected_transposes) is not tuple
        or any(type(op) is not TransposeOp or op not in set(module.walk()) for op in selected_transposes)
    ):
        raise ValueError("selected layouts require current typed transpose requests")
    cache = {}
    segments = {}
    added = 0
    if pointwise_only:
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
        same_block = (
            isinstance(op, Operation)
            and op.parent is anchor.parent
            and eligible(op)
            and (not pointwise_only or segments[op] == segments[anchor])
        )
        result = None
        if same_block and (old := _permutation(op)) is not None:
            report.canceled += 1
            result = demand(op.operands[0], tuple(old[i] for i in permutation), anchor)
        # Moving only unit axes preserves the flat element order and is a view.
        elif [i for i in permutation if shape[i] != 1] == [i for i in range(rank) if shape[i] != 1] and (
            not pointwise_only or sum(d != 1 for d in shape) <= 1
        ):
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
        elif same_block and per_tensor_quantization(op):
            result = add(
                type(op).create(
                    operands=[demand(op.operands[0], permutation, anchor), *op.operands[1:]],
                    result_types=[out_ty],
                    properties=dict(op.properties),
                    attributes=dict(op.attributes),
                ),
                anchor,
            )
            report.scalar_regions += 1
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
        elif (
            same_block
            and op.name == "linalg.generic"
            and len(op.results) == 1
            and (
                not pointwise_only
                or op.results[0].has_one_use()
                or selected_transposes is not None
                and sum(use.operation not in selected_transposes for use in op.results[0].uses) == 1
            )
        ):
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
                        shared = (
                            pointwise_only
                            and isinstance(producer, Operation)
                            and producer.name == "linalg.generic"
                            and not operand.has_one_use()
                        )
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

    originals = [
        (op, _permutation(op) if eligible(op) else None)
        for op in (module.walk() if selected_transposes is None else selected_transposes)
    ]
    for op, permutation in originals:
        if not eligible(op) or permutation is None or not op.results[0].uses or _static_type(op.results[0]) is None:
            continue
        report.requested += 1
        result = demand(op.operands[0], permutation, op)
        if pointwise_only:
            # Cancellation may expose an alias rather than a fresh snapshot.
            # Keep the original materialization if a later opaque effect lies
            # between it and an observation of that alias.

            backing = result
            while isinstance(backing.owner, Operation) and pointwise_kind(backing.owner) == "unit_view":
                backing = unit_source(backing.owner)
            snapshot = isinstance(backing.owner, Operation) and pointwise_kind(backing.owner) in {
                "generic",
                "quantized_pointwise",
                "transpose",
                "splat",
                "constant",
            }
            if not snapshot and any(
                use.operation.parent is not op.parent or segments.get(use.operation) != segments[op]
                for use in op.results[0].uses
            ):
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
            if (
                operation.parent is not None
                and eligible(operation)
                and operation.results
                and all(isinstance(value.type, TensorType) and not value.uses for value in operation.results)
            ):
                operation.parent.erase_op(operation)
        module.verify()
        return report
    if selected_transposes is not None:
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


def realize_input_layout(module, value, permutation, anchor):
    """Realize one explicit source layout without rewriting unrelated requests.

    The caller owns the new consumer and its semantic correspondence. A temporary
    tensor cast observes the rewritten value while the existing demand consumer
    preserves scalar bodies, coordinates and opaque effect boundaries.
    """
    ty = _static_type(value)
    if (
        ty is None
        or sorted(permutation) != list(range(len(ty.get_shape())))
        or anchor.parent is None
        or anchor not in set(module.walk())
    ):
        raise ValueError("input layout requires a current static source and consumer")
    result_type = TensorType(ty.element_type, [ty.get_shape()[i] for i in permutation])
    empty = tensor.EmptyOp([], result_type)
    request = TransposeOp(value, empty.results[0], DenseArrayBase.from_list(i64, permutation), result_type)
    marker = tensor.CastOp.create(operands=[request.results[0]], result_types=[result_type])
    for operation in (empty, request, marker):
        anchor.parent.insert_op_before(operation, anchor)
    report = rewrite_module(module, pointwise_only=True, selected_transposes=(request,))
    result = marker.operands[0]
    marker.parent.erase_op(marker)
    module.verify()
    return result, report


# ---- layout_conformance ------------------------------------------------------------------------
# Bounded coordinate correspondence across unchanged opaque IR boundaries.
#
# This checks a static pointwise/layout subset and opt-in ordered generic
# reductions, not arbitrary call, loop or device
# semantics. Unknown operations retain ordered typed identities and complete nested
# wiring. Tensor observations carry an effect epoch so materialized snapshots cannot
# silently become reads after an unknown effect. No tensor elements are enumerated.

_EXPRESSION_CACHE_LIMIT = 32768


def _attribute(attribute):
    stream = StringIO()
    DenseFloatPrinter(stream=stream).print_attribute(attribute)
    return stream.getvalue()


def _metadata(operation):
    # Default xDSL printing can quiet signaling NaNs in nested dense attributes.
    return tuple(
        tuple(sorted((key, _attribute(value)) for key, value in fields.items()))
        for fields in (operation.attributes, operation.properties)
    )


def _shape(value):
    typ = value.type
    if (
        type(typ) is builtin.TensorType
        and isinstance(typ.encoding, builtin.NoneAttr)
        and all(d > 0 for d in typ.get_shape())
    ):
        return typ.get_shape()
    return None


def _fields(op, properties):
    return (
        not op.successors
        and set(op.properties) == set(properties)
        and all(name.startswith("prov.") for name in op.attributes)
    )


def _reshape(op):
    if type(op) not in (tensor.CollapseShapeOp, tensor.ExpandShapeOp) or len(op.operands) != 1 or len(op.results) != 1:
        return False
    props = {"reassociation"} | ({"static_output_shape"} if type(op) is tensor.ExpandShapeOp else set())
    source, target = _shape(op.operands[0]), _shape(op.results[0])
    if (
        source is None
        or target is None
        or not _fields(op, props)
        or op.operands[0].type.get_element_type() != op.results[0].type.get_element_type()
    ):
        return False
    small, large = (target, source) if type(op) is tensor.CollapseShapeOp else (source, target)
    groups = tuple(tuple(int(axis.value.data) for axis in group) for group in op.reassociation)
    if (
        len(groups) != len(small)
        or tuple(i for group in groups for i in group) != tuple(range(len(large)))
        or any(not group or prod(large[i] for i in group) != size for group, size in zip(groups, small))
    ):
        return False
    return type(op) is tensor.CollapseShapeOp or tuple(op.static_output_shape.get_values()) == target


def unit_source(op):
    """Only insertion/removal/movement of unit axes, including flat view pairs."""
    if not _reshape(op):
        return None
    value = op.operands[0]
    if type(op) is tensor.ExpandShapeOp and type(value.owner) is tensor.CollapseShapeOp and _reshape(value.owner):
        collapse = value.owner
        source = collapse.operands[0]
        # Flattening several nonunit axes may materialize a strided input.
        # No physical-layout premise is supplied here. Admit only the adjacent
        # single-axis bias-view form; other reshapes remain opaque boundaries.
        if collapse.parent is not op.parent or op.prev_op is not collapse or sum(d != 1 for d in _shape(source)) > 1:
            return None
        value = source
    source, target = _shape(value), _shape(op.results[0])
    return value if tuple(d for d in source if d != 1) == tuple(d for d in target if d != 1) else None


def per_tensor_quantization(op):
    """The declared scalar-broadcast quantization boundary, with no axis state.

    Keep the operation opaque as a scalar function; moving coordinates never
    changes its parameters, arithmetic, rounding or saturation semantics.
    """
    name = getattr(getattr(op, "op_name", None), "data", None)
    fields = {"quant_min", "quant_max"}
    if name == "quant_ext.quantize_per_tensor":
        fields.add("output_dtype")
    elif name != "quant_ext.dequantize_per_tensor":
        return False
    return (
        isinstance(op, builtin.UnregisteredOp)
        and not op.regions
        and not op.successors
        and len(op.operands) == 3
        and len(op.results) == 1
        and set(op.properties) == fields
        and all(key == "op_name__" or key.startswith("prov.") for key in op.attributes)
        and _shape(op.operands[0]) is not None
        and _shape(op.operands[0]) == _shape(op.results[0])
        and _shape(op.operands[1]) == _shape(op.operands[2]) == ()
    )


def pointwise_kind(op):
    """Strict registered schemas accepted by the checked production family."""
    if per_tensor_quantization(op):
        return "quantized_pointwise"
    if type(op) is arith.ConstantOp and _fields(op, {"value"}) and not op.operands and len(op.results) == 1:
        return "constant"
    if type(op) is tensor.EmptyOp and _fields(op, ()) and not op.operands and len(op.results) == 1:
        return "empty" if _shape(op.results[0]) is not None else None
    if type(op) is tensor.SplatOp and _fields(op, ()) and len(op.operands) == len(op.results) == 1:
        return "splat" if _shape(op.results[0]) is not None else None
    if type(op) in (tensor.ExpandShapeOp, tensor.CollapseShapeOp):
        if unit_source(op) is not None:
            return "unit_view"
        # A flattening intermediate is transparent only to verified unit-view pairs.
        if (
            type(op) is tensor.CollapseShapeOp
            and _reshape(op)
            and op.results[0].uses
            and all(
                type(use.operation) is tensor.ExpandShapeOp and unit_source(use.operation) is op.operands[0]
                for use in op.results[0].uses
            )
        ):
            return "flat_pair"
        return None
    if type(op) is TransposeOp:
        if (
            not _fields(op, {"permutation"})
            or len(op.operands) != 2
            or len(op.results) != 1
            or _shape(op.operands[0]) is None
            or _shape(op.results[0]) is None
        ):
            return None
        body = op.body.block
        permutation = tuple(op.permutation.get_values())
        if (
            len(list(body.ops)) != 1
            or type(body.last_op) is not YieldOp
            or tuple(body.last_op.operands) != (body.args[0],)
            or _metadata(body.last_op) != ((), ())
            or sorted(permutation) != list(range(len(_shape(op.operands[0]))))
        ):
            return None
        return "transpose"
    if type(op) is not GenericOp or not _fields(op, {"indexing_maps", "iterator_types", "operandSegmentSizes"}):
        return None
    if (
        len(op.results) != 1
        or len(op.outputs) != 1
        or len(op.body.blocks) != 1
        or _shape(op.results[0]) is None
        or any(it.data.value != "parallel" for it in op.iterator_types)
    ):
        return None
    rank = len(_shape(op.results[0]))
    maps = tuple(item.data for item in op.indexing_maps)
    if (
        len(op.iterator_types) != rank
        or len(maps) != len(op.operands)
        or maps[-1] != AffineMap.identity(rank)
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (len(op.inputs), 1)
    ):
        return None
    for value, mapping in zip(op.operands, maps, strict=True):
        shape = _shape(value)
        if shape is None or mapping.num_symbols or mapping.num_dims != rank or len(mapping.results) != len(shape):
            return None
        for dimension, size in zip(mapping.results, shape, strict=True):
            if isinstance(dimension, AffineDimExpr) and 0 <= dimension.position < rank:
                if size != _shape(op.results[0])[dimension.position]:
                    return None
            elif not isinstance(dimension, AffineConstantExpr) or dimension.value != 0 or size != 1:
                return None
    if (
        not pure_scalar_body(op.body.block)
        or type(op.body.block.last_op) is not YieldOp
        or len(op.body.block.last_op.operands) != 1
        or not _fields(op.body.block.last_op, ())
    ):
        return None
    return "generic"


def ordered_reduction_bounds(op):
    """Exact static projected-index reduction domain, with unchanged loop order.

    This admits no reassociation or target layout capability. Output parallel
    axes stay canonical; each input may be read through its exact permutation.
    Uninitialized accumulator reads, affine gathers and index/effect observers
    remain outside this family.
    """
    if (
        type(op) is not GenericOp
        or not _fields(op, {"indexing_maps", "iterator_types", "operandSegmentSizes"})
        or len(op.results) != 1
        or len(op.outputs) != 1
        or len(op.body.blocks) != 1
        or _shape(op.results[0]) is None
        or len(op.body.block.args) != len(op.operands)
        or not pure_scalar_body(op.body.block, integer_extensions=True)
        or type(op.body.block.last_op) is not YieldOp
        or not _fields(op.body.block.last_op, ())
        or len(op.body.block.last_op.operands) != 1
    ):
        return None
    shape = _shape(op.results[0])
    rank, count = len(shape), len(op.iterator_types)
    kinds = tuple(item.data.value for item in op.iterator_types)
    maps = tuple(item.data for item in op.indexing_maps)
    if (
        not rank
        or count <= rank
        or kinds[:rank] != ("parallel",) * rank
        or kinds[rank:] != ("reduction",) * (count - rank)
        or len(maps) != len(op.operands)
        or tuple(op.properties["operandSegmentSizes"].get_values()) != (len(op.inputs), 1)
        or maps[-1] != AffineMap(count, 0, tuple(AffineExpr.dimension(i) for i in range(rank)))
        or _shape(op.outputs[0]) != shape
    ):
        return None
    if op.body.block.args[-1].uses and type(op.outputs[0].owner) is tensor.EmptyOp:
        return None
    bounds = [*shape, *([None] * (count - rank))]
    for operand, mapping in zip(op.operands, maps, strict=True):
        sizes = _shape(operand)
        if sizes is None or mapping.num_symbols or mapping.num_dims != count or len(mapping.results) != len(sizes):
            return None
        for expression, size in zip(mapping.results, sizes, strict=True):
            if isinstance(expression, AffineDimExpr) and 0 <= expression.position < count:
                axis = expression.position
                if bounds[axis] is not None and bounds[axis] != size:
                    return None
                bounds[axis] = size
            elif not isinstance(expression, AffineConstantExpr) or expression.value != 0 or size != 1:
                return None
    return tuple(bounds) if all(bound is not None for bound in bounds) else None


def connected_kind(op):
    """Existing pointwise family plus exactly checked ordered reductions."""
    return pointwise_kind(op) or ("reduction" if ordered_reduction_bounds(op) is not None else None)


def scalar_bodies(module):
    """Canonical complete scalar order, including captures and unused operations."""
    result = []
    for op in module.walk():
        if type(op) is not GenericOp:
            continue
        if len(op.body.blocks) != 1:
            continue
        block = op.body.block
        bindings = {arg: ("arg", i, _attribute(arg.type)) for i, arg in enumerate(block.args)}
        rows = []
        for i, scalar in enumerate(block.ops):
            operands = []
            for value in scalar.operands:
                if value in bindings:
                    operands.append(bindings[value])
                elif type(value.owner) is arith.ConstantOp:
                    operands.append(("capture_constant", _attribute(value.type), _metadata(value.owner)))
                else:
                    operands.append(("capture", _attribute(value.type)))
            rows.append(
                (scalar.name, _metadata(scalar), tuple(_attribute(v.type) for v in scalar.results), tuple(operands))
            )
            bindings.update({value: ("result", i, j) for j, value in enumerate(scalar.results)})
        result.append(tuple(rows))
    return tuple(result)


def layout_expression(module, *, include_work=False, connected=False):
    """Reconstruct observable coordinates and every opaque boundary's wiring."""
    kinds, epochs, identities, arguments, paths = {}, {}, {}, {}, {}
    classifier = connected_kind if connected else pointwise_kind
    classifications, metadata_cache, body_cache = {}, {}, {}

    # One expression observes an immutable module. Reaching the same producer
    # from another coordinate or effect epoch must recheck that observation,
    # but need not reprint its identical attributes and complete scalar body.
    # Keep these caches local so later mutations always get fresh identities.
    def classify(op):
        if op not in classifications:
            classifications[op] = classifier(op)
        return classifications[op]

    def metadata(op):
        if op not in metadata_cache:
            metadata_cache[op] = _metadata(op)
        return metadata_cache[op]

    def body(op):
        if op not in body_cache:
            body_cache[op] = scalar_bodies(op)[0]
        return body_cache[op]

    def register(block, path):
        if len(path) > 64:
            raise ValueError("layout coordinate checking exceeded its bounded region depth")
        paths[block] = path
        arguments.update({value: (path, i) for i, value in enumerate(block.args)})
        ordinal = 0
        for op in block.ops:
            kind = classify(op)
            if kind == "empty" and any(classify(use.operation) is None for use in op.results[0].uses):
                kind = None
            kinds[op], epochs[op] = kind, (path, ordinal)
            if kind is not None:
                continue
            identities[op] = (path, ordinal)
            ordinal += 1
            if op.successors:
                raise ValueError("layout coordinate checking refuses unstructured control flow")
            for i, region in enumerate(op.regions):
                if len(region.blocks) > 1:
                    raise ValueError("layout coordinate checking requires single-block opaque regions")
                for child in region.blocks:
                    register(child, (*path, identities[op][1], i))

    functions = [op for op in module.body.block.ops if type(op) is func.FuncOp]
    for function in functions:
        if len(function.body.blocks) > 1:
            raise ValueError("layout coordinate checking requires single-block functions")
        for block in function.body.blocks:
            register(block, (function.sym_name.data,))
    cache, visiting, observed_generics = {}, set(), set()

    def element(value, coordinates, epoch):
        shape = _shape(value)
        if shape is not None:
            coordinates = tuple(
                AffineExpr.constant(0) if d == 1 else c for d, c in zip(shape, coordinates, strict=True)
            )
        key = value, coordinates, epoch
        if key in cache:
            return cache[key]
        if len(cache) >= _EXPRESSION_CACHE_LIMIT or len(visiting) >= 256 or value in visiting:
            raise ValueError("layout coordinate checking exceeded its bounded expression budget")
        visiting.add(value)
        tensor_epoch = epoch if isinstance(value.type, builtin.TensorType) else None
        if value in arguments:
            result = ("argument", arguments[value], _attribute(value.type), coordinates, tensor_epoch)
        else:
            op = value.owner
            if not isinstance(op, Operation) or op not in kinds:
                raise ValueError("layout expression has an unknown value owner")
            kind = kinds[op]
            if kind == "constant":
                result = ("constant", _attribute(value.type), metadata(op), coordinates)
            elif kind == "transpose":
                permutation = tuple(op.permutation.get_values())
                result = element(
                    op.operands[0],
                    tuple(coordinates[permutation.index(i)] for i in range(len(coordinates))),
                    epochs[op],
                )
            elif kind == "unit_view":
                source = unit_source(op)
                nonunit = iter(c for c, d in zip(coordinates, shape, strict=True) if d != 1)
                source_coordinates = tuple(AffineExpr.constant(0) if d == 1 else next(nonunit) for d in _shape(source))
                result = element(source, source_coordinates, epoch)
            elif kind == "splat":
                result = element(op.operands[0], (), epochs[op])
            elif kind == "quantized_pointwise":
                result = (
                    "scalar_quantization",
                    metadata(op),
                    tuple(_attribute(v.type.get_element_type()) for v in (*op.operands, value)),
                    element(op.operands[0], coordinates, epochs[op]),
                    tuple(element(v, (), epochs[op]) for v in op.operands[1:]),
                )
            elif kind in {"generic", "reduction"}:
                observed_generics.add(op)
                rank = len(coordinates)
                block = op.body.block
                domain = ordered_reduction_bounds(op) if kind == "reduction" else None
                indices = (
                    (*coordinates, *(AffineExpr.dimension(i) for i in range(rank, len(domain))))
                    if domain
                    else coordinates
                )
                bindings = {
                    arg: (
                        operand,
                        tuple(mapping.data.replace_dims_and_symbols(list(indices), [], len(indices), 0).results),
                    )
                    for arg, operand, mapping in zip(block.args, op.operands, op.indexing_maps, strict=True)
                }
                scalar_cache = {}

                def scalar(item, depth=0):
                    if depth > 256:
                        raise ValueError("layout scalar expression exceeded its bounded nesting budget")
                    if item in scalar_cache:
                        return scalar_cache[item]
                    if item in bindings:
                        operand, input_coordinates = bindings[item]
                        answer = (
                            ("accumulator", _attribute(item.type))
                            if domain and item is block.args[-1]
                            else element(operand, input_coordinates, epochs[op])
                        )
                    elif isinstance(item.owner, Operation) and item.owner.parent is block:
                        owner = item.owner
                        answer = (
                            owner.name,
                            tuple(owner.results).index(item),
                            _attribute(item.type),
                            metadata(owner),
                            tuple(scalar(v, depth + 1) for v in owner.operands),
                        )
                    else:
                        answer = element(item, (), epochs[op])
                    scalar_cache[item] = answer
                    return answer

                yielded = scalar(block.last_op.operands[0])
                plain_copy = len(list(block.ops)) == 1 and metadata(block.last_op) == ((), ())
                if domain:
                    result = (
                        "ordered_reduction",
                        metadata(op)[0],
                        body(op),
                        domain,
                        element(op.outputs[0], coordinates, epochs[op]) if block.args[-1].uses else None,
                        yielded,
                    )
                else:
                    result = yielded if plain_copy else ("pointwise", metadata(op)[0], body(op), yielded)
            elif kind in {"empty", "flat_pair"}:
                raise ValueError("layout coordinate checker has no semantics for this tensor producer")
            elif type(op) is tensor.EmptyOp:
                raise ValueError("layout coordinate checker has no semantics for this tensor producer")
            else:
                result = (
                    "opaque_result",
                    identities[op],
                    tuple(op.results).index(value),
                    _attribute(value.type),
                    coordinates,
                    tensor_epoch,
                )
        visiting.remove(value)
        cache[key] = result
        return result

    def observe(value, epoch):
        if type(value.owner) is tensor.EmptyOp and value.owner in identities:
            return ("uninitialized_boundary", identities[value.owner], _attribute(value.type))
        shape = _shape(value)
        if isinstance(value.type, builtin.TensorType) and shape is None:
            # Unknown layouts are stable typed leaves, not interpreted coordinates.
            coordinates = ()
        else:
            coordinates = tuple(AffineExpr.dimension(i) for i in range(len(shape or ())))
        return element(value, coordinates, epoch)

    def boundaries(block):
        rows = []
        for op in block.ops:
            if kinds[op] is not None:
                continue
            rows.append(
                (
                    identities[op],
                    op.name,
                    metadata(op),
                    tuple(_attribute(v.type) for v in op.results),
                    tuple(observe(value, epochs[op]) for value in op.operands),
                    tuple(tuple(boundaries(child) for child in region.blocks) for region in op.regions),
                )
            )
            # Keep the complete checked row, including all operands and effect
            # epochs. Finished expression cones need no memoized entries at
            # the next opaque boundary; bound each observation independently
            # instead of limiting the number of boundaries in a whole model.
            cache.clear()
        return (tuple(_attribute(v.type) for v in block.args), tuple(rows))

    expression = (
        metadata(module),
        tuple(
            (function.sym_name.data, metadata(function), tuple(boundaries(block) for block in function.body.blocks))
            for function in functions
        ),
        tuple(ir_text(op) for op in module.body.block.ops if type(op) is not func.FuncOp),
    )
    if not include_work:
        return expression
    work = Counter()
    for op in observed_generics:
        for scalar in op.body.block.ops:
            if type(scalar) is YieldOp:
                continue
            # Typed arithmetic and lexical control scope, never provenance IDs
            # or cycle estimates. Count one complete static parallel domain.
            scalar_metadata = tuple(
                tuple((name, value) for name, value in fields if not name.startswith("prov."))
                for fields in metadata(scalar)
            )
            key = paths[op.parent], scalar.name, tuple(_attribute(v.type) for v in scalar.results), scalar_metadata
            domain = ordered_reduction_bounds(op) if connected and kinds[op] == "reduction" else _shape(op.results[0])
            work[key] += prod(domain)
    return expression, work
