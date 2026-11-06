"""Prove exact storage replay over inserted tensor axes.

Only direct scalar yields are movement: even floating arithmetic identities
can change signaling NaNs. The source and result here remain original SSA
values, and the destination initializer is never read by the scalar region.
"""
from __future__ import annotations
from math import prod

from xdsl.dialects.builtin import DenseArrayBase, NoneAttr, TensorType, i64
from xdsl.dialects.linalg.ops import BroadcastOp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr


_STORAGE_DTYPES = frozenset(('i8', 'i16', 'i32', 'i64',
                             'f16', 'bf16', 'f32', 'f64'))


def register_broadcast(context):
    """Use the installed standard named source carrier, without replacement."""
    existing = context.get_optional_op(BroadcastOp.name)
    if existing is None:
        context.load_op(BroadcastOp)
    elif existing is not BroadcastOp:
        raise ValueError('linalg.broadcast source carrier is not the standard typed op')


def _attr(op, name):
    # Conflicting attributes/properties must never silently select one meaning.
    if name in op.attributes and name in op.properties:
        return None
    return op.properties.get(name, op.attributes.get(name))


def _spec(value):
    ty = value.type
    if (not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr)
            or str(ty.element_type) not in _STORAGE_DTYPES):
        return None
    shape = tuple(ty.get_shape())
    if any(type(extent) is not int or extent <= 0 for extent in shape):
        return None
    return shape, ty.element_type


def _plain_attributes(op, allowed):
    return all(key in allowed or key.startswith('prov.')
               for table in (op.attributes, op.properties) for key in table)


def _direct_yield(op, src, init):
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].block
    body = list(block.ops)
    if (len(block.args) != 2 or len(body) != 1
            or block.args[0].type != src.type.element_type
            or block.args[1].type != init.type.element_type):
        return False
    term = body[0]
    return (term.name == 'linalg.yield' and len(term.operands) == 1
            and term.operands[0] is block.args[0]
            and not term.results and not term.regions and not term.successors
            and _plain_attributes(term, set()))


def _dim(expr, position):
    return isinstance(expr, AffineDimExpr) and expr.position == position


def recognize_broadcast(op):
    """Return original src/result and MOVEMENT attrs, or reject the source.

    Named broadcasts declare inserted axes explicitly. Generic broadcasts here
    retain the leading-axis proof; other affine storage maps are recognized by
    frontend_movement. Scalar arithmetic never acquires a storage-copy proof.
    """
    named = isinstance(op, BroadcastOp)
    if not named and op.name != 'linalg.generic':
        return None
    if len(op.results) != 1 or op.successors:
        return None
    if named:
        if not _plain_attributes(op, {'dimensions'}):
            return None
        src, init = op.input, op.init
        if len(op.operands) != 2:
            return None
    else:
        if not _plain_attributes(op, {'indexing_maps', 'iterator_types',
                                       'operandSegmentSizes', 'doc'}):
            return None
        inputs, outputs = list(op.inputs), list(op.outputs)
        if len(inputs) != 1 or len(outputs) != 1 or len(op.operands) != 2:
            return None
        src, init = inputs[0], outputs[0]
    result = op.results[0]
    ss, si, sd = _spec(src), _spec(init), _spec(result)
    if ss is None or si is None or sd is None or si != sd:
        return None
    source_shape, element_type = ss
    shape, result_element_type = sd
    leading = len(shape) - len(source_shape)
    if (leading <= 0
            or element_type != result_element_type
            or init.type != result.type or not _direct_yield(op, src, init)):
        return None
    if named:
        dimensions = _attr(op, 'dimensions')
        if (not isinstance(dimensions, DenseArrayBase)
                or dimensions.elt_type != i64
                or len(dimensions) != leading):
            return None
        inserted=tuple(dimensions.get_values())
        if (tuple(sorted(set(inserted))) != inserted
                or any(axis < 0 or axis >= len(shape) for axis in inserted)):
            return None
        retained=[axis for axis in range(len(shape)) if axis not in inserted]
        if tuple(shape[axis] for axis in retained) != source_shape:
            return None
        strides=[0]*len(shape)
        for source_axis,result_axis in enumerate(retained):
            strides[result_axis]=prod(source_shape[source_axis+1:])
        return {'src':src,'result':result,'attributes':{
            'form':'as_strided','strides':strides,'storage_offset':0}}
    else:
        if shape[leading:] != source_shape:
            return None
        maps = _attr(op, 'indexing_maps')
        iters = _attr(op, 'iterator_types')
        if maps is None or iters is None or len(maps) != 2:
            return None
        if [getattr(iterator, 'data', None) for iterator in iters] != ['parallel'] * len(shape):
            return None
        source_map, output_map = (item.data for item in maps)
        if any(m.num_dims != len(shape) or m.num_symbols != 0
               for m in (source_map, output_map)):
            return None
        if (len(source_map.results) != len(source_shape)
                or len(output_map.results) != len(shape)
                or not all(_dim(expr, axis) for axis, expr in enumerate(output_map.results))):
            return None
        for axis, expr in enumerate(source_map.results):
            # A constant index is equivalent only on an actually unit axis
            # present in both original source and trailing result shape.
            if not (_dim(expr, leading + axis)
                    or (isinstance(expr, AffineConstantExpr)
                        and expr.value == 0 and source_shape[axis] == 1)):
                return None
    return {'src': src, 'result': result, 'attributes': {'form': 'broadcast'}}
