"""Bounded coordinate correspondence across unchanged opaque IR boundaries.

This checks a static pointwise/layout subset, not arbitrary call, loop or device
semantics. Unknown operations retain ordered typed identities and complete nested
wiring. Tensor observations carry an effect epoch so materialized snapshots cannot
silently become reads after an unknown effect. No tensor elements are enumerated.
"""

from __future__ import annotations

from io import StringIO
from math import prod

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp, YieldOp
from xdsl.ir import Operation
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineExpr, AffineMap

from merlin.common.dense_float_ir import DenseFloatPrinter
from merlin.transforms.certificates import ir_text

from .quant_round_check import pure_scalar_body


def _attribute(attribute):
    stream = StringIO()
    DenseFloatPrinter(stream=stream).print_attribute(attribute)
    return stream.getvalue()


def _metadata(operation):
    # Default xDSL printing can quiet signaling NaNs in nested dense attributes.
    return tuple(tuple(sorted((key, _attribute(value)) for key, value in fields.items()))
                 for fields in (operation.attributes, operation.properties))


def _shape(value):
    typ = value.type
    if (type(typ) is builtin.TensorType and isinstance(typ.encoding, builtin.NoneAttr)
            and all(d > 0 for d in typ.get_shape())):
        return typ.get_shape()
    return None


def _fields(op, properties):
    return not op.successors and set(op.properties) == set(properties) and all(
        name.startswith("prov.") for name in op.attributes
    )


def _reshape(op):
    if type(op) not in (tensor.CollapseShapeOp, tensor.ExpandShapeOp) or len(op.operands) != 1 or len(op.results) != 1:
        return False
    props = {"reassociation"} | ({"static_output_shape"} if type(op) is tensor.ExpandShapeOp else set())
    source, target = _shape(op.operands[0]), _shape(op.results[0])
    if (source is None or target is None or not _fields(op, props)
            or op.operands[0].type.get_element_type() != op.results[0].type.get_element_type()):
        return False
    small, large = (target, source) if type(op) is tensor.CollapseShapeOp else (source, target)
    groups = tuple(tuple(int(axis.value.data) for axis in group) for group in op.reassociation)
    if (len(groups) != len(small) or tuple(i for group in groups for i in group) != tuple(range(len(large)))
            or any(not group or prod(large[i] for i in group) != size for group, size in zip(groups, small))):
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
        if (collapse.parent is not op.parent or op.prev_op is not collapse
                or sum(d != 1 for d in _shape(source)) > 1):
            return None
        value = source
    source, target = _shape(value), _shape(op.results[0])
    return value if tuple(d for d in source if d != 1) == tuple(d for d in target if d != 1) else None


def pointwise_kind(op):
    """Strict registered schemas accepted by the checked production family."""
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
        if (type(op) is tensor.CollapseShapeOp and _reshape(op) and op.results[0].uses
                and all(type(use.operation) is tensor.ExpandShapeOp and unit_source(use.operation) is op.operands[0]
                        for use in op.results[0].uses)):
            return "flat_pair"
        return None
    if type(op) is TransposeOp:
        if (not _fields(op, {"permutation"}) or len(op.operands) != 2 or len(op.results) != 1
                or _shape(op.operands[0]) is None or _shape(op.results[0]) is None):
            return None
        body = op.body.block
        permutation = tuple(op.permutation.get_values())
        if (len(list(body.ops)) != 1 or type(body.last_op) is not YieldOp
                or tuple(body.last_op.operands) != (body.args[0],) or _metadata(body.last_op) != ((), ())
                or sorted(permutation) != list(range(len(_shape(op.operands[0]))))):
            return None
        return "transpose"
    if type(op) is not GenericOp or not _fields(op, {"indexing_maps", "iterator_types", "operandSegmentSizes"}):
        return None
    if (len(op.results) != 1 or len(op.outputs) != 1 or len(op.body.blocks) != 1
            or _shape(op.results[0]) is None or any(it.data.value != "parallel" for it in op.iterator_types)):
        return None
    rank = len(_shape(op.results[0]))
    maps = tuple(item.data for item in op.indexing_maps)
    if (len(op.iterator_types) != rank or len(maps) != len(op.operands)
            or maps[-1] != AffineMap.identity(rank)
            or tuple(op.properties["operandSegmentSizes"].get_values()) != (len(op.inputs), 1)):
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
    if (not pure_scalar_body(op.body.block) or type(op.body.block.last_op) is not YieldOp
            or len(op.body.block.last_op.operands) != 1 or _metadata(op.body.block.last_op) != ((), ())):
        return None
    return "generic"


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
            rows.append((scalar.name, _metadata(scalar),
                         tuple(_attribute(v.type) for v in scalar.results), tuple(operands)))
            bindings.update({value: ("result", i, j) for j, value in enumerate(scalar.results)})
        result.append(tuple(rows))
    return tuple(result)


def layout_expression(module):
    """Reconstruct observable coordinates and every opaque boundary's wiring."""
    kinds, epochs, identities, arguments, paths = {}, {}, {}, {}, {}

    def register(block, path):
        if len(path) > 64:
            raise ValueError("layout coordinate checking exceeded its bounded region depth")
        paths[block] = path
        arguments.update({value: (path, i) for i, value in enumerate(block.args)})
        ordinal = 0
        for op in block.ops:
            kind = pointwise_kind(op)
            if kind == "empty" and any(pointwise_kind(use.operation) is None for use in op.results[0].uses):
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
    cache, visiting = {}, set()

    def element(value, coordinates, epoch):
        shape = _shape(value)
        if shape is not None:
            coordinates = tuple(
                AffineExpr.constant(0) if d == 1 else c for d, c in zip(shape, coordinates, strict=True))
        key = value, coordinates, epoch
        if key in cache:
            return cache[key]
        if len(cache) >= 32768 or len(visiting) >= 256 or value in visiting:
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
                result = ("constant", _attribute(value.type), _metadata(op), coordinates)
            elif kind == "transpose":
                permutation = tuple(op.permutation.get_values())
                result = element(op.operands[0], tuple(
                    coordinates[permutation.index(i)] for i in range(len(coordinates))), epochs[op])
            elif kind == "unit_view":
                source = unit_source(op)
                nonunit = iter(c for c, d in zip(coordinates, shape, strict=True) if d != 1)
                source_coordinates = tuple(AffineExpr.constant(0) if d == 1 else next(nonunit) for d in _shape(source))
                result = element(source, source_coordinates, epoch)
            elif kind == "splat":
                result = element(op.operands[0], (), epochs[op])
            elif kind == "generic":
                rank = len(coordinates)
                block = op.body.block
                bindings = {
                    arg: (operand, tuple(mapping.data.replace_dims_and_symbols(list(coordinates), [], rank, 0).results))
                    for arg, operand, mapping in zip(block.args, op.operands, op.indexing_maps, strict=True)
                }
                scalar_cache = {}

                def scalar(item, depth=0):
                    if depth > 256:
                        raise ValueError("layout scalar expression exceeded its bounded nesting budget")
                    if item in scalar_cache:
                        return scalar_cache[item]
                    if item in bindings:
                        operand, indices = bindings[item]
                        answer = element(operand, indices, epochs[op])
                    elif isinstance(item.owner, Operation) and item.owner.parent is block:
                        owner = item.owner
                        answer = (owner.name, tuple(owner.results).index(item), _attribute(item.type), _metadata(owner),
                                  tuple(scalar(v, depth + 1) for v in owner.operands))
                    else:
                        answer = element(item, (), epochs[op])
                    scalar_cache[item] = answer
                    return answer

                yielded = scalar(block.last_op.operands[0])
                result = yielded if len(list(block.ops)) == 1 else (
                    "pointwise", _metadata(op)[0], yielded)
            elif kind in {"empty", "flat_pair"}:
                raise ValueError("layout coordinate checker has no semantics for this tensor producer")
            elif type(op) is tensor.EmptyOp:
                raise ValueError("layout coordinate checker has no semantics for this tensor producer")
            else:
                result = ("opaque_result", identities[op], tuple(op.results).index(value),
                          _attribute(value.type), coordinates, tensor_epoch)
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
            rows.append((identities[op], op.name, _metadata(op), tuple(_attribute(v.type) for v in op.results),
                         tuple(observe(value, epochs[op]) for value in op.operands),
                         tuple(tuple(boundaries(child) for child in region.blocks) for region in op.regions)))
        return (tuple(_attribute(v.type) for v in block.args), tuple(rows))

    return (_metadata(module), tuple(
        (function.sym_name.data, _metadata(function), tuple(boundaries(block) for block in function.body.blocks))
        for function in functions
    ), tuple(ir_text(op) for op in module.body.block.ops if type(op) is not func.FuncOp))
