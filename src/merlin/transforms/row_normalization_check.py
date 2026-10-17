"""Check actual row-local source replay and restore the complete surrounding IR.

No selected row plan proves itself: original scalar regions, initializer data,
actual loop bounds, slices, recurrence and local producer/consumer SSA are
checked separately. Inverting checked islands must reconstruct the entire
original module, including every unselected operation and escaping observation.
This is a structural semantic check, not a Lean or machine-code receipt.
"""

from __future__ import annotations

from math import prod

from xdsl.dialects import arith, builtin, scf, tensor
from xdsl.dialects.linalg import ops as L
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter

from .row_normalization import local_type, row_input_specs


def _require(condition, reason):
    if not condition:
        raise ValueError("row normalization conformance: " + reason)


def _integers(value):
    return tuple(value.get_values())


def _slice(operation, source, coordinates, shape, *, destination=None):
    kind = tensor.ExtractSliceOp if destination is None else tensor.InsertSliceOp
    _require(type(operation) is kind and not operation.attributes, "missing or changed actual row slice")
    expected = (source, *coordinates) if destination is None else (source, destination, *coordinates)
    _require(tuple(operation.operands) == expected, "slice does not use the actual source, row or carried destination")
    offsets = tuple(0 if n == 1 else builtin.DYNAMIC_INDEX for n in shape[:-1]) + (0,)
    _require(_integers(operation.static_offsets) == offsets, "slice row/column offsets changed")
    _require(
        _integers(operation.static_sizes) == (1,) * (len(shape) - 1) + (shape[-1],),
        "slice lost the complete column domain",
    )
    _require(_integers(operation.static_strides) == (1,) * len(shape), "slice changes logical source coordinates")
    allowed = {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
    _require(not operation.properties.keys() - allowed, "unknown slice properties")


def _local_replay(record, loop, mapped, position, coordinates):
    group = record.source
    body = tuple(loop.body.block.ops)
    for original in group.operations:
        result = original.results[0]
        if isinstance(original, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            mapped[result] = mapped[original.operands[0]]
            _require(mapped[result].type == local_type(result, group), "elided view crosses row-local values")
            continue
        _require(position < len(body), "local source replay is incomplete")
        actual = body[position]
        position += 1
        _require(type(actual) is type(original), "local source operation kind or order changed")
        _require(actual.attributes == original.attributes, "source operation attributes changed")
        expected_operands = tuple(mapped.get(value, value) for value in original.operands)
        _require(
            tuple(actual.operands) == expected_operands, "local source SSA, initialization or continuation changed"
        )
        expected_type = result.type if isinstance(original, arith.ConstantOp) else local_type(result, group)
        _require(
            tuple(value.type for value in actual.results) == (expected_type,), "local result dtype/row extent changed"
        )
        if isinstance(original, L.GenericOp):
            expected_maps = tuple(
                AffineMap(1, 0, (AffineDimExpr(0),) if value.type.get_num_dims() else ()) for value in actual.operands
            )
            _require(
                tuple(x.data for x in actual.indexing_maps) == expected_maps,
                "local indexing changes source row elements",
            )
            _require(
                tuple(str(x) for x in actual.iterator_types) == (str(L.IteratorTypeAttr(L.IteratorType.PARALLEL)),),
                "local pointwise iteration is no longer parallel",
            )
            _require(actual.properties.keys() == original.properties.keys(), "generic lowering properties changed")
            _require(
                all(
                    actual.properties[key] == original.properties[key]
                    for key in original.properties
                    if key not in {"indexing_maps", "iterator_types"}
                ),
                "generic operand grouping changed",
            )
        elif isinstance(original, L.ReduceOp):
            _require(
                _integers(actual.dimensions) == (0,) and actual.properties.keys() == {"dimensions"},
                "reduction no longer covers the whole row in source order",
            )
        else:
            _require(actual.properties == original.properties, "source constant or initialization changed")
        _require(len(actual.regions) == len(original.regions), "source scalar regions were added/removed")
        for before, after in zip(original.regions, actual.regions, strict=True):
            _require(before.is_structurally_equivalent(after), "source scalar arithmetic or reduction order changed")
        mapped[result] = actual.results[0]
    _require(len(body) == position + 2, "extra or missing effects in row-local replay")
    inserted, yielded = body[position:]
    _slice(
        inserted,
        mapped[group.operations[-1].results[0]],
        coordinates,
        group.shape,
        destination=loop.body.block.args[1],
    )
    _require(
        type(yielded) is scf.YieldOp and tuple(yielded.operands) == (inserted.result,),
        "row loop does not carry the completed output",
    )
    _require(not yielded.attributes and not yielded.properties, "unknown row yield effects")


def _index_constant(operation, value):
    _require(
        type(operation) is arith.ConstantOp
        and not operation.operands
        and not operation.attributes
        and operation.result.type == builtin.IndexType()
        and operation.value.value.data == value
        and operation.properties.keys() == {"value"},
        "row bound/coordinate constant changed",
    )


def verify_row_normalization_record(record):
    """Reconstruct one actual complete-row wrapper and its original source DAG."""
    group = record.source
    ops = record.emitted
    _require(len(ops) == 5, "row wrapper has extra/missing operations")
    for op in ops:
        _require(not op.attributes, "unknown generated wrapper attributes")
    zero, end, one, empty, loop = ops
    for operation, value in ((zero, 0), (end, group.rows), (one, 1)):
        _index_constant(operation, value)
    output_type = group.source.type
    _require(
        type(empty) is tensor.EmptyOp
        and empty.tensor.type == output_type
        and not empty.operands
        and not empty.properties,
        "row output allocation has a changed domain",
    )
    _require(
        type(loop) is scf.ForOp
        and not loop.properties
        and tuple(loop.operands) == (zero.result, end.result, one.result, empty.tensor),
        "row loop omitted rows or changed its carried output",
    )
    _require(
        tuple(x.type for x in loop.results) == (output_type,) and len(loop.body.blocks) == 1,
        "row loop has unknown output/control flow",
    )
    _require(tuple(loop.body.block.arg_types) == (builtin.IndexType(), output_type), "row loop binding changed")
    body = tuple(loop.body.block.ops)
    position = 0
    coordinates = []
    row_axes = {}
    for axis, extent in enumerate(group.shape[:-1]):
        if extent == 1:
            continue
        _require(len(body) >= position + 4, "row coordinates are incomplete")
        stride, quotient, dimension, coordinate = body[position : position + 4]
        position += 4
        _index_constant(stride, prod(group.shape[axis + 1 : -1]))
        _index_constant(dimension, extent)
        _require(
            type(quotient) is arith.DivUIOp
            and not quotient.properties
            and not quotient.attributes
            and tuple(quotient.operands) == (loop.body.block.args[0], stride.result),
            "source row prefix quotient changed",
        )
        _require(
            type(coordinate) is arith.RemUIOp
            and not coordinate.properties
            and not coordinate.attributes
            and tuple(coordinate.operands) == (quotient.result, dimension.result),
            "source row prefix coordinate changed",
        )
        coordinates.append(coordinate.result)
        row_axes[axis] = coordinate.result
    mapped = {}
    for value, projection in row_input_specs(group):
        shape = value.type.get_shape()
        if not shape:
            mapped[value] = value
            continue
        _require(len(body) > position, "row source extraction is missing")
        extracted = body[position]
        position += 1
        offsets = tuple(row_axes[axis] for axis in projection if axis in row_axes)
        _require(
            type(extracted) is tensor.ExtractSliceOp and tuple(extracted.operands) == (value, *offsets),
            "projected row input source or coordinates changed",
        )
        _require(
            not extracted.attributes
            and not extracted.properties.keys()
            - {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"},
            "row input slice metadata changed",
        )
        _require(
            _integers(extracted.static_offsets)
            == tuple(builtin.DYNAMIC_INDEX if axis in row_axes else 0 for axis in projection),
            "row input broadcast offsets changed",
        )
        _require(
            _integers(extracted.static_sizes)
            == tuple(group.columns if axis == len(group.shape) - 1 else 1 for axis in projection),
            "row input slice lost the complete source column domain",
        )
        _require(_integers(extracted.static_strides) == (1,) * len(shape), "row input slice stride changed")
        local_shape = (group.columns,) if len(group.shape) - 1 in projection else ()
        _require(
            extracted.result.type == builtin.TensorType(value.type.get_element_type(), local_shape),
            "row extraction changes input type or complete domain",
        )
        mapped[value] = extracted.result
    _local_replay(record, loop, mapped, position, tuple(coordinates))
    _require(record.result is loop.results[0], "published result is not the completed row loop output")


def verify_row_normalizations(original, modified, records):
    """Check actual islands; invert them and compare the entire source module."""
    for record in records:
        verify_row_normalization_record(record)
    mapped = {}
    restored = modified.clone(value_mapper=mapped)
    for record in reversed(records):
        try:
            actual_ops = tuple(mapped[op.results[0]].owner for op in record.emitted)
            external = dict(mapped)
            source_ops = tuple(op.clone(value_mapper=external) for op in record.source.operations)
            Rewriter.replace_op(actual_ops[-1], source_ops, new_results=[source_ops[-1].results[0]])
            for op in reversed(actual_ops[:-1]):
                Rewriter.erase_op(op)
        except (KeyError, ValueError) as exc:
            raise ValueError("row normalization conformance: actual wrapper membership/escape changed") from exc
    _require(restored.is_structurally_equivalent(original), "complete outer source, interfaces or observations changed")
    modified.verify()
