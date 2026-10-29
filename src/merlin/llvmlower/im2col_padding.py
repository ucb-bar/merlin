"""Pack exact static windows directly from a logically zero-padded image.

This host tensor rewrite preserves the complete gather and its consumers. It
does not select a convolution implementation, device gather or latency policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
from xdsl.dialects.linalg.ops import FillOp, GenericOp, IndexOp, YieldOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.rewriter import InsertPoint, Rewriter

from .im2col_pack import _is_copy_body, _static_shape, _window_term


@dataclass(frozen=True)
class PaddingGather:
    gather: GenericOp
    insert: tensor.InsertSliceOp
    fill: FillOp
    empty: tensor.EmptyOp
    image: SSAValue
    zero: SSAValue
    geometry: tuple[int, ...]


def _plain(op, properties=()):
    return not op.attributes and set(op.properties) == set(properties) and not op.successors


def _single(value, operation):
    uses = tuple(value.uses)
    return len(uses) == 1 and uses[0].operation is operation


def _supported_scope(operation):
    parent = operation.parent_op()
    while parent is not None:
        if type(parent) is builtin.ModuleOp:
            attributes, properties = {"llvm.data_layout", "llvm.target_triple"}, set()
        elif type(parent) is func.FuncOp:
            attributes, properties = {"llvm.emit_c_interface"}, {"sym_name", "function_type", "sym_visibility"}
        elif type(parent) in (scf.ForOp, scf.IfOp):
            attributes, properties = set(), set()
        else:
            return False
        if any(key not in attributes and not key.startswith("prov.") for key in parent.attributes):
            return False
        if set(parent.properties) - properties:
            return False
        parent = parent.parent_op()
    return True


def match_padding_gather(operation, *, index_bits):
    """Read exact source semantics; unsupported shapes/effects remain unchanged."""
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("padding/gather requires an explicit bounded index width")
    if (
        type(operation) is not GenericOp
        or not _supported_scope(operation)
        or not _plain(operation, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
        or len(operation.inputs) != 1
        or len(operation.outputs) != 1
        or len(operation.results) != 1
        or len(operation.body.blocks) != 1
        or not _is_copy_body(operation)
        or not _plain(operation.body.block.last_op)
        or tuple(t.data for t in operation.get_iterator_types()) != (IteratorType.PARALLEL,) * 6
    ):
        return None
    insert, destination = operation.inputs[0].owner, operation.outputs[0].owner
    if type(insert) is not tensor.InsertSliceOp or type(destination) is not tensor.EmptyOp:
        return None
    fill = insert.dest.owner
    if type(fill) is not FillOp or len(fill.inputs) != 1 or len(fill.outputs) != 1:
        return None
    empty, zero = fill.outputs[0].owner, fill.inputs[0].owner
    if (
        type(empty) is not tensor.EmptyOp
        or not _plain(empty)
        or empty.operands
        or not _plain(fill, ("operandSegmentSizes",))
        or len(fill.body.blocks) != 1
        or len(tuple(fill.body.block.ops)) != 1
        or type(fill.body.block.last_op) is not YieldOp
        or tuple(fill.body.block.last_op.operands) != (fill.body.block.args[0],)
        or not _plain(fill.body.block.last_op)
        or type(zero) is not arith.ConstantOp
        or not _plain(zero, ("value",))
        or type(zero.value) is not builtin.IntegerAttr
        or zero.value.value.data != 0
        or not _plain(insert, ("static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"))
        or insert.offsets
        or insert.sizes
        or insert.strides
        or not _single(empty.results[0], fill)
        or not _single(fill.results[0], insert)
        or not _single(insert.results[0], operation)
        or not _plain(destination)
        or destination.operands
    ):
        return None
    image = insert.source
    values = (image, insert.dest, insert.results[0], operation.outputs[0], operation.results[0])
    if any(type(v.type) is not builtin.TensorType or type(v.type.encoding) is not builtin.NoneAttr for v in values):
        return None
    shape, padded, gathered = (
        _static_shape(image),
        _static_shape(insert.results[0]),
        _static_shape(operation.results[0]),
    )
    if shape is None or padded is None or gathered is None or len(shape) != 4 or len(padded) != 4 or len(gathered) != 6:
        return None
    if any(dimension <= 0 for dimensions in (shape, padded, gathered) for dimension in dimensions):
        return None
    elem = image.type.element_type
    if str(elem) not in ("i8", "i16", "i32") or any(v.type.element_type != elem for v in values):
        return None
    if zero.results[0].type != elem or operation.outputs[0].type != operation.results[0].type:
        return None
    offsets = tuple(insert.static_offsets.get_values())
    if (
        len(offsets) != 4
        or offsets[:2] != (0, 0)
        or tuple(insert.static_sizes.get_values()) != tuple(shape)
        or tuple(insert.static_strides.get_values()) != (1, 1, 1, 1)
        or any(o < 0 or o + size > extent for o, size, extent in zip(offsets, shape, padded, strict=True))
        or padded[:2] != shape[:2]
        or prod(padded) <= prod(shape)
    ):
        return None
    maps = tuple(a.data for a in operation.indexing_maps)
    if len(maps) != 2 or maps[1] != AffineMap.identity(6) or maps[0].num_dims != 6 or maps[0].num_symbols:
        return None
    from xdsl.ir.affine import AffineDimExpr

    coords = maps[0].results
    if len(coords) != 4 or coords[:2] != (AffineDimExpr(3), AffineDimExpr(0)):
        return None
    h, w = _window_term(coords[2], 4, 1), _window_term(coords[3], 5, 2)
    if h is None or w is None or min(*h, *w) <= 0:
        return None
    channels, kh, kw, n, oh, ow = gathered
    if (n, channels) != tuple(shape[:2]) or (oh - 1) * h[0] + (kh - 1) * h[1] >= padded[2]:
        return None
    if (ow - 1) * w[0] + (kw - 1) * w[1] >= padded[3]:
        return None
    limit = (1 << (index_bits - 1)) - 1
    if max(*(prod(s) * (elem.width.data // 8) for s in (shape, padded, gathered)), *padded, *h, *w) > limit:
        return None
    block = operation.parent_block()
    chain = {empty, fill, insert, destination, operation}
    if block is None or any(op.parent_block() is not block for op in chain):
        return None
    current = empty
    while current is not operation:
        if current is None or current not in chain | {zero}:
            return None
        current = current.next_op
    return PaddingGather(operation, insert, fill, empty, image, zero.results[0], (*h, *w, *offsets[2:], *shape[2:]))


def rewrite_padding_gathers(module, *, index_bits, max_added_operations=4096):
    """Construct guarded reads only within a fixed original operation roster."""
    if type(max_added_operations) is not int or max_added_operations < 0:
        raise ValueError("padding/gather construction budget must be nonnegative")
    module.verify()
    original = tuple(module.walk())
    result, added = [], 0
    for operation in original:
        match = match_padding_gather(operation, index_bits=index_bits)
        if match is None:
            continue
        # 6 indices, 8 constants, 6 arithmetic, 4 comparisons, 3 conjunctions,
        # if + 2 subtractions + extract + 2 scf yields + linalg yield + writer.
        count = 35
        if added + count > max_added_operations:
            continue
        sh, dh, sw, dw, top, left, height, width = match.geometry
        elem = match.image.type.element_type
        body = Block(arg_types=(elem,))
        indices = [IndexOp(i) for i in range(6)]
        constants = [
            arith.ConstantOp.from_int_and_width(v, builtin.IndexType())
            for v in (sh, dh, sw, dw, top, left, top + height, left + width)
        ]
        body.add_ops([*indices, *constants])

        def emit(cls, *args):
            op = cls(*args)
            body.add_op(op)
            return op.results[0]

        ix = [op.results[0] for op in indices]
        c = [op.results[0] for op in constants]
        h = emit(arith.AddiOp, emit(arith.MuliOp, ix[4], c[0]), emit(arith.MuliOp, ix[1], c[1]))
        w = emit(arith.AddiOp, emit(arith.MuliOp, ix[5], c[2]), emit(arith.MuliOp, ix[2], c[3]))
        checks = [
            emit(arith.CmpiOp, a, b, predicate)
            for a, b, predicate in ((h, c[4], "uge"), (h, c[6], "ult"), (w, c[5], "uge"), (w, c[7], "ult"))
        ]
        condition = emit(
            arith.AndIOp, emit(arith.AndIOp, checks[0], checks[1]), emit(arith.AndIOp, checks[2], checks[3])
        )
        true = Block()
        ih, iw = arith.SubiOp(h, c[4]), arith.SubiOp(w, c[5])
        read = tensor.ExtractOp(match.image, [ix[3], ix[0], ih.results[0], iw.results[0]], elem)
        true.add_ops([ih, iw, read, scf.YieldOp(read.results[0])])
        false = Block([scf.YieldOp(match.zero)])
        branch = scf.IfOp(condition, [elem], Region(true), Region(false))
        body.add_ops([branch, YieldOp(branch.results[0])])
        writer = GenericOp(
            inputs=[],
            outputs=list(operation.outputs),
            body=Region(body),
            indexing_maps=[builtin.AffineMapAttr(AffineMap.identity(6))],
            iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * 6,
            result_types=[operation.results[0].type],
        )
        if len(tuple(writer.walk())) != count:
            raise ValueError("padding/gather construction count differs from its preflight budget")
        Rewriter.insert_op(writer, InsertPoint.before(operation))
        operation.results[0].replace_all_uses_with(writer.results[0])
        for dead in (operation, match.insert, match.fill, match.empty):
            Rewriter.erase_op(dead)
        added += count
        result.append({"gather_shape": list(writer.results[0].type.get_shape()), "added_operations": count})
    return tuple(result)
