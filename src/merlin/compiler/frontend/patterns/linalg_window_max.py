"""Recognize the ordered BF16 window maximum that max pooling lowers to.

The source form is an NCHW ``linalg.generic`` with four parallel output loops
and two window reduction loops. Its body keeps the first equal maximum
(``x > acc`` is ordered, so -0 and +0 do not replace each other) and the last
NaN (``x`` unordered with itself replaces). The accumulator starts at a -inf
splat. Optional padding is an insert into a -inf splat; because -inf never
replaces the accumulator, padded reads are equivalent to skipping them.
Kernel, stride, dilation, and padding come from the indexing maps and the
window operand's shape, never from a model or operator name.

PyTorch's BF16 CPU kernel returns the canonical NaN 0x7FC0 for any window that
holds a NaN, while this body keeps the last NaN's payload. The admitted input
domain therefore excludes NaN; consumers refuse NaN inputs before writing.

Temporary: this recognizer matches the fixed body chain that model2MLIR's max
pooling decomposition emits. Its general replacement is a declared
implementation family for window reductions, chosen by Merlin cover selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from .linalg_patterns import InvalidLinalgPattern

NUMERIC_SEMANTICS = "ordered-window-max-first-equal-v1"
INPUT_DOMAIN = "non-nan-bf16"
NEG_INF_BF16 = 0xFF80


@dataclass(frozen=True)
class WindowMax:
    source: object  # unpadded BF16 input value
    prelude: tuple[object, ...]  # consumed producer operations
    window: tuple[int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    padding: tuple[int, int]


def _only_user(value, user) -> bool:
    return all(use.operation is user for use in value.uses)


def _neg_inf_splat(value, user):
    """Return (splat, constant) when value is an otherwise unused BF16 -inf splat."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import BFloat16Type, FloatAttr
    from xdsl.dialects.tensor import SplatOp

    splat = value.owner
    if type(splat) is not SplatOp or splat.properties or not _only_user(value, user):
        raise InvalidLinalgPattern("window maximum needs an otherwise unused -inf splat")
    if any(not key.startswith("prov.") for key in splat.attributes) or len(splat.operands) != 1:
        raise InvalidLinalgPattern("window maximum -inf splat has unknown attributes or operands")
    constant = splat.operands[0].owner
    literal = constant.properties.get("value") if type(constant) is ConstantOp else None
    if (
        not isinstance(literal, FloatAttr)
        or not isinstance(literal.type, BFloat16Type)
        or set(constant.properties) != {"value"}
        or any(not key.startswith("prov.") for key in constant.attributes)
        or int.from_bytes(literal.type.pack((literal.value.data,)), "little") != NEG_INF_BF16
    ):
        raise InvalidLinalgPattern("window maximum initializer must be a BF16 -inf literal")
    return splat, constant


def recognize_bf16_window_max(op) -> WindowMax:
    """Prove the window-maximum body, maps, initializer, and optional -inf padding."""
    from xdsl.dialects.arith import CmpfOp, OrIOp, SelectOp
    from xdsl.dialects.builtin import AffineMapAttr, BFloat16Type, DenseArrayBase, NoneAttr, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.dialects.tensor import EmptyOp, InsertSliceOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("window maximum needs input, window operand, and one result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("window maximum has an unknown Linalg property or successor")
    if any(not key.startswith("prov.") for key in op.attributes):
        raise InvalidLinalgPattern("window maximum has an unknown Linalg attribute")
    segments = op.properties["operandSegmentSizes"]
    if not isinstance(segments, DenseArrayBase) or tuple(segments.iter_values()) != (2, 1):
        raise InvalidLinalgPattern("window maximum has an inconsistent operand segment")
    values = (*op.inputs, *op.outputs, *op.results)
    if any(
        not isinstance(value.type, TensorType)
        or not isinstance(value.type.element_type, BFloat16Type)
        or not isinstance(value.type.encoding, NoneAttr)
        for value in values
    ):
        raise InvalidLinalgPattern("window maximum requires default-encoded BF16 tensors")
    padded, window_value = op.inputs
    shapes = [tuple(value.type.get_shape()) for value in values]
    if (
        len(shapes[0]) != 4
        or len(shapes[1]) != 2
        or shapes[2] != shapes[3]
        or len(shapes[2]) != 4
        or any(type(dim) is not int or dim <= 0 for shape in shapes for dim in shape)
    ):
        raise InvalidLinalgPattern("window maximum needs static positive NCHW, window, and result shapes")
    (n, c, height, width), (kh, kw), (on, oc, oh, ow) = shapes[0], shapes[1], shapes[2]
    if (n, c) != (on, oc):
        raise InvalidLinalgPattern("window maximum changes batch or channel extents")
    iterators = op.iterator_types.data
    if (
        any(not isinstance(item, IteratorTypeAttr) for item in iterators)
        or tuple(item.data for item in iterators) != (IteratorType.PARALLEL,) * 4 + (IteratorType.REDUCTION,) * 2
    ):
        raise InvalidLinalgPattern("window maximum needs four parallel and two reduction loops")
    maps = tuple(op.indexing_maps.data)
    if len(maps) != 3 or any(not isinstance(item, AffineMapAttr) for item in maps):
        raise InvalidLinalgPattern("window maximum needs three affine maps")
    source_map = maps[0].data
    d = [AffineDimExpr(i) for i in range(6)]
    stride = dilation = None
    if source_map.num_dims == 6 and source_map.num_symbols == 0 and len(source_map.results) == 4:
        from merlin.compiler.semantic.linalg.linalg_recognition import LinalgBridgeError, _linear_affine_expression

        try:
            rows = [_linear_affine_expression(expr, 6) for expr in source_map.results]
        except LinalgBridgeError as exc:
            raise InvalidLinalgPattern("window maximum source map is not integer affine") from exc
        if rows[0] == ((1, 0, 0, 0, 0, 0), 0) and rows[1] == ((0, 1, 0, 0, 0, 0), 0):
            (h_row, h_off), (w_row, w_off) = rows[2], rows[3]
            if (h_off, w_off) == (0, 0) and h_row[1:2] == w_row[1:2] == (0,) and h_row[0] == w_row[0] == 0:
                if h_row[3] == h_row[5] == w_row[2] == w_row[4] == 0:
                    stride, dilation = (h_row[2], w_row[3]), (h_row[4], w_row[5])
    if (
        stride is None
        or min(*stride, *dilation) < 1
        or maps[1].data != AffineMap(6, 0, (d[4], d[5]))
        or maps[2].data != AffineMap(6, 0, tuple(d[:4]))
    ):
        raise InvalidLinalgPattern("window maximum needs strided, dilated window maps over NCHW")
    if (
        ((height - dilation[0] * (kh - 1) - 1) // stride[0] + 1, (width - dilation[1] * (kw - 1) - 1) // stride[1] + 1)
        != (oh, ow)
        or height < dilation[0] * (kh - 1) + 1
        or width < dilation[1] * (kw - 1) + 1
    ):
        raise InvalidLinalgPattern("window maximum result extents differ from the floor window count")
    block = op.regions[0].block if len(op.regions) == 1 and len(op.regions[0].blocks) == 1 else None
    if block is None or len(block.args) != 3 or any(not isinstance(arg.type, BFloat16Type) for arg in block.args):
        raise InvalidLinalgPattern("window maximum needs one BF16 scalar body")
    body = tuple(block.ops)
    if tuple(type(inner) for inner in body) != (CmpfOp, CmpfOp, OrIOp, SelectOp, YieldOp):
        raise InvalidLinalgPattern("window maximum body must be cmpf uno, cmpf ogt, ori, select, yield")
    unordered, greater, either, select, yield_op = body
    x, _, acc = block.args
    if (
        any(
            inner.regions or inner.successors or any(not key.startswith("prov.") for key in inner.attributes)
            for inner in body
        )
        or any(set(cmp.properties) != {"predicate", "fastmath"} for cmp in (unordered, greater))
        or any(inner.properties for inner in (either, select, yield_op))
        or unordered.predicate.value.data != 14
        or tuple(unordered.operands) != (x, x)
        or greater.predicate.value.data != 2
        or tuple(greater.operands) != (x, acc)
        or any(cmp.fastmath.data for cmp in (unordered, greater))
        or set(either.operands) != {unordered.result, greater.result}
        or len(either.operands) != 2
        or tuple(select.operands) != (either.result, x, acc)
        or tuple(yield_op.operands) != (select.result,)
        or any(key for inner in (either, select, yield_op) for key in inner.attributes if not key.startswith("prov."))
        or any(
            len(tuple(value.uses)) != 1 for value in (unordered.result, greater.result, either.result, select.result)
        )
    ):
        raise InvalidLinalgPattern("window maximum body does not keep first equal maximum and last NaN")
    window = window_value.owner
    if type(window) is not EmptyOp or window.operands or not _only_user(window_value, op):
        raise InvalidLinalgPattern("window maximum window operand must be an otherwise unused tensor.empty")
    init, init_constant = _neg_inf_splat(op.outputs[0], op)
    prelude = [window, init, init_constant]
    source, padding = padded, (0, 0)
    insert = padded.owner
    if type(insert) is InsertSliceOp:
        if not _only_user(padded, op) or insert.offsets or insert.sizes or insert.strides:
            raise InvalidLinalgPattern("window maximum padding must be one static, otherwise unused insert")
        inserted = insert.source
        offsets = tuple(insert.static_offsets.iter_values())
        sizes = tuple(insert.static_sizes.iter_values())
        strides = tuple(insert.static_strides.iter_values())
        inner_shape = tuple(inserted.type.get_shape()) if isinstance(inserted.type, TensorType) else ()
        padding = offsets[2:4] if len(offsets) == 4 else (-1, -1)
        if (
            offsets[:2] != (0, 0)
            or min(padding) < 0
            or sizes != inner_shape
            or strides != (1, 1, 1, 1)
            or len(inner_shape) != 4
            or inner_shape[:2] != (n, c)
            or (inner_shape[2] + 2 * padding[0], inner_shape[3] + 2 * padding[1]) != (height, width)
            or not isinstance(inserted.type.element_type, BFloat16Type)
            or any(not key.startswith("prov.") for key in insert.attributes)
        ):
            raise InvalidLinalgPattern("window maximum padding must be symmetric -inf borders around NCHW input")
        pad_splat, pad_constant = _neg_inf_splat(insert.dest, insert)
        prelude += [insert, pad_splat]
        if pad_constant is not init_constant:
            prelude.append(pad_constant)
        source = inserted
    splats = {item for item in prelude if type(item) is type(init)}
    constants = {init_constant, *(item for item in prelude[3:] if type(item) is type(init_constant))}
    if any(use.operation not in splats for item in constants for use in item.results[0].uses):
        raise InvalidLinalgPattern("window maximum -inf literal has users outside its splats")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("window maximum Linalg operation does not verify") from exc
    return WindowMax(source, tuple(prelude), (kh, kw), stride, dilation, padding)
