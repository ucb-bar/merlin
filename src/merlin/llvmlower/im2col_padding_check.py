"""Independently check zero-padding elimination at the tensor gather boundary.

This reader proves logical coordinates and unchanged surrounding effects. It
does not establish physical storage, target support, or a performance benefit.
It neither imports the realizer nor accepts its selected plan as evidence.
"""

from __future__ import annotations

from io import StringIO
from math import prod

from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.linalg.ops import FillOp, GenericOp, IndexOp, YieldOp
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr, AffineMap

from merlin.common.dense_float_ir import DenseFloatPrinter


def _printed(attribute):
    stream = StringIO()
    DenseFloatPrinter(stream=stream).print_attribute(attribute)
    return stream.getvalue()


def _metadata(op):
    return tuple(
        tuple(sorted((k, _printed(v)) for k, v in fields.items())) for fields in (op.attributes, op.properties)
    )


def _details(op):
    return (
        op.name,
        _metadata(op),
        tuple(_printed(v.type) for v in (*op.operands, *op.results)),
        tuple(tuple(_printed(v.type) for v in block.args) for region in op.regions for block in region.blocks),
    )


def _plain(op, properties=()):
    if op.attributes or set(op.properties) != set(properties) or op.successors:
        raise ValueError("padding/gather has unknown attributes, properties or control flow")


def _shape(value):
    typ = value.type
    if type(typ) is not builtin.TensorType or type(typ.encoding) is not builtin.NoneAttr:
        raise ValueError("padding/gather requires unencoded tensor values")
    shape = tuple(typ.get_shape())
    if any(d <= 0 for d in shape):
        raise ValueError("padding/gather requires positive static dimensions")
    return shape


def _constant(n):
    return (n, 0, 0, 0, 0, 0, 0)


def _axis(n):
    if not 0 <= n < 6:
        raise ValueError("padding/gather coordinate uses an unknown dimension")
    return (0, *(int(i == n) for i in range(6)))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _mul(a, b):
    if any(a[1:]) and any(b[1:]):
        raise ValueError("padding/gather has a nonlinear coordinate")
    if any(a[1:]):
        a, b = b, a
    return tuple(a[0] * v for v in b)


def _affine(expr):
    if isinstance(expr, AffineConstantExpr):
        return _constant(expr.value)
    if isinstance(expr, AffineDimExpr):
        return _axis(expr.position)
    if isinstance(expr, AffineBinaryOpExpr):
        if expr.kind == AffineBinaryOpKind.Add:
            return _add(_affine(expr.lhs), _affine(expr.rhs))
        if expr.kind == AffineBinaryOpKind.Mul:
            return _mul(_affine(expr.lhs), _affine(expr.rhs))
    raise ValueError("padding/gather has an unsupported affine coordinate")


def _one_use(value, consumer):
    uses = tuple(value.uses)
    if len(uses) != 1 or uses[0].operation is not consumer:
        raise ValueError("padding storage has an additional observer or escaping use")


def _source(gather, index_bits):
    """Read the complete original zero-padding/copy island independently."""
    if type(gather) is not GenericOp:
        raise ValueError("padding source is not a typed gather")
    _plain(gather, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
    if len(gather.inputs) != 1 or len(gather.outputs) != 1 or len(gather.results) != 1:
        raise ValueError("padding gather has an unsupported interface")
    if len(gather.body.blocks) != 1:
        raise ValueError("padding gather has unknown scalar control flow")
    body = gather.body.block
    if len(body.args) != 2 or len(tuple(body.ops)) != 1 or type(body.last_op) is not YieldOp:
        raise ValueError("padding gather is not an exact scalar copy")
    _plain(body.last_op)
    if tuple(body.last_op.operands) != (body.args[0],) or body.args[1].uses:
        raise ValueError("padding gather reads its destination or changes its scalar value")
    insert, output = gather.inputs[0].owner, gather.outputs[0].owner
    if type(insert) is not tensor.InsertSliceOp or type(output) is not tensor.EmptyOp:
        raise ValueError("padding gather lacks a static insert and fresh output")
    fill = insert.dest.owner
    if type(fill) is not FillOp or len(fill.inputs) != 1 or len(fill.outputs) != 1 or len(fill.results) != 1:
        raise ValueError("padding gather lacks an exact fill")
    empty, zero = fill.outputs[0].owner, fill.inputs[0].owner
    if type(empty) is not tensor.EmptyOp or type(zero) is not arith.ConstantOp:
        raise ValueError("padding gather lacks a fresh zero-filled tensor")
    for op in (empty, output):
        _plain(op)
        if op.operands:
            raise ValueError("padding/gather has dynamic allocation sizes")
    _plain(fill, ("operandSegmentSizes",))
    _plain(zero, ("value",))
    if type(zero.value) is not builtin.IntegerAttr or zero.value.value.data != 0:
        raise ValueError("padding fill is not an exact typed integer zero")
    if len(fill.body.blocks) != 1 or len(fill.body.block.args) != 2:
        raise ValueError("padding fill has an unknown scalar body")
    fb = fill.body.block
    if len(tuple(fb.ops)) != 1 or type(fb.last_op) is not YieldOp or tuple(fb.last_op.operands) != (fb.args[0],):
        raise ValueError("padding fill changes the original scalar value")
    _plain(fb.last_op)
    _plain(insert, ("static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"))
    if insert.offsets or insert.sizes or insert.strides:
        raise ValueError("padding insert has dynamic coordinates")
    image = insert.source
    shape, padded, gathered = _shape(image), _shape(insert.result), _shape(gather.results[0])
    elem = image.type.element_type
    if (
        len(shape) != 4
        or len(padded) != 4
        or len(gathered) != 6
        or type(elem) is not builtin.IntegerType
        or elem.width.data not in (8, 16, 32)
        or elem.signedness.data == builtin.Signedness.UNSIGNED
        or any(v.type != insert.result.type for v in (empty.results[0], fill.results[0], insert.dest))
        or zero.results[0].type != elem
        or gather.outputs[0].type != gather.results[0].type
        or gather.results[0].type.element_type != elem
    ):
        raise ValueError("padding/gather changes shape or element representation")
    offsets = tuple(insert.static_offsets.get_values())
    if (
        len(offsets) != 4
        or offsets[:2] != (0, 0)
        or tuple(insert.static_sizes.get_values()) != shape
        or tuple(insert.static_strides.get_values()) != (1,) * 4
        or padded[:2] != shape[:2]
        or prod(padded) <= prod(shape)
        or any(o < 0 or o + n > d for o, n, d in zip(offsets, shape, padded, strict=True))
    ):
        raise ValueError("padding insert is not the complete in-bounds image")
    maps = tuple(a.data for a in gather.indexing_maps)
    if (
        len(maps) != 2
        or maps[1] != AffineMap.identity(6)
        or maps[0].num_dims != 6
        or maps[0].num_symbols
        or len(maps[0].results) != 4
        or tuple(it.data.value for it in gather.iterator_types) != ("parallel",) * 6
    ):
        raise ValueError("padding gather has an unsupported domain")
    n, c, h, w = tuple(_affine(e) for e in maps[0].results)
    sh, dh, sw, dw = h[5], h[2], w[6], w[3]
    if n != _axis(3) or c != _axis(0) or min(sh, dh, sw, dw) <= 0:
        raise ValueError("padding gather changed the image/window axes")
    if h != _add(_mul(_axis(4), _constant(sh)), _mul(_axis(1), _constant(dh))) or w != _add(
        _mul(_axis(5), _constant(sw)), _mul(_axis(2), _constant(dw))
    ):
        raise ValueError("padding gather has an unproved window coordinate")
    channels, kh, kw, batches, oh, ow = gathered
    hmax, wmax = (oh - 1) * sh + (kh - 1) * dh, (ow - 1) * sw + (kw - 1) * dw
    limit = (1 << (index_bits - 1)) - 1
    if (
        (batches, channels) != shape[:2]
        or hmax >= padded[2]
        or wmax >= padded[3]
        or max(
            hmax,
            wmax,
            *shape,
            *padded,
            *gathered,
            sh,
            dh,
            sw,
            dw,
            *(prod(s) * (elem.width.data // 8) for s in (shape, padded, gathered)),
        )
        > limit
    ):
        raise ValueError("padding/gather exceeds its original domain or current index width")
    chain = (empty, fill, insert, gather)
    for a, b in zip(chain, chain[1:], strict=False):
        _one_use(a.results[0], b)
    if any(op.parent_block() is not gather.parent_block() for op in (*chain, output)):
        raise ValueError("padding/gather crosses a block boundary")
    cursor = empty
    while cursor is not gather:
        if cursor is None or cursor not in (*chain, output, zero):
            raise ValueError("padding/gather crosses an operation/effect barrier")
        cursor = cursor.next_op
    return {
        "gather": gather,
        "removed": chain[:3],
        "image": image,
        "zero": zero.results[0],
        "shape": shape,
        "padded": padded,
        "gathered": gathered,
        "h": h,
        "w": w,
        "top": offsets[2],
        "left": offsets[3],
        "element_bytes": elem.width.data // 8,
    }


def _realization(source, writer, values, index_bits):
    if type(writer) is not GenericOp:
        raise ValueError("padding/gather replacement is not a typed writer")
    _plain(writer, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
    original = source["gather"]
    if (
        writer.inputs
        or tuple(writer.outputs) != (values[original.outputs[0]],)
        or tuple(v.type for v in writer.results) != tuple(v.type for v in original.results)
        or tuple(a.data for a in writer.indexing_maps) != (AffineMap.identity(6),)
        or tuple(it.data.value for it in writer.iterator_types) != ("parallel",) * 6
        or len(writer.body.blocks) != 1
        or len(writer.body.block.args) != 1
        or writer.body.block.args[0].uses
        or len(tuple(writer.walk())) > 35
    ):
        raise ValueError("padding/gather replacement changes its complete output domain")
    expressions, predicates = {}, {}
    branch = None
    ops = tuple(writer.body.block.ops)

    def remember(value, expression):
        low = expression[0] + sum(min(0, c * (n - 1)) for c, n in zip(expression[1:], source["gathered"]))
        high = expression[0] + sum(max(0, c * (n - 1)) for c, n in zip(expression[1:], source["gathered"]))
        if low < 0 or high > (1 << (index_bits - 1)) - 1:
            raise ValueError("padding/gather actual coordinate exceeds its current index width")
        expressions[value] = expression

    for op in ops[:-1]:
        if type(op) is IndexOp:
            _plain(op, ("dim",))
            remember(op.results[0], _axis(op.dim.value.data))
        elif type(op) is arith.ConstantOp:
            _plain(op, ("value",))
            if type(op.results[0].type) is not builtin.IndexType or type(op.value) is not builtin.IntegerAttr:
                raise ValueError("padding/gather coordinate constant is not an index")
            remember(op.results[0], _constant(op.value.value.data))
        elif type(op) in (arith.AddiOp, arith.MuliOp):
            _plain(op, ("overflowFlags",))
            if op.overflow_flags.data or any(v not in expressions for v in op.operands):
                raise ValueError("padding/gather arithmetic has unknown overflow or captures")
            a, b = (expressions[v] for v in op.operands)
            remember(op.results[0], (_add if type(op) is arith.AddiOp else _mul)(a, b))
        elif type(op) is arith.CmpiOp:
            _plain(op, ("predicate",))
            if any(v not in expressions for v in op.operands):
                raise ValueError("padding bounds use an unknown expression")
            predicates[op.results[0]] = frozenset(((op.predicate.value.data, *(expressions[v] for v in op.operands)),))
        elif type(op) is arith.AndIOp:
            _plain(op)
            if any(v not in predicates for v in op.operands):
                raise ValueError("padding bounds use an unproved predicate")
            predicates[op.results[0]] = predicates[op.operands[0]] | predicates[op.operands[1]]
        elif type(op) is scf.IfOp and branch is None and op is ops[-2]:
            _plain(op)
            branch = op
        else:
            raise ValueError("padding/gather has an eager read, subtraction or unknown scalar effect")
    top, left = source["top"], source["left"]
    height, width = source["shape"][2:]
    expected = frozenset(
        (
            (9, source["h"], _constant(top)),
            (6, source["h"], _constant(top + height)),
            (9, source["w"], _constant(left)),
            (6, source["w"], _constant(left + width)),
        )
    )
    if branch is None or predicates.get(branch.cond) != expected or len(branch.results) != 1:
        raise ValueError("padding/gather does not guard the exact original image rectangle")
    if len(branch.true_region.blocks) != 1 or len(branch.false_region.blocks) != 1:
        raise ValueError("padding/gather branch has unsupported control flow")
    true, false = branch.true_region.block, branch.false_region.block
    if true.args or false.args or len(tuple(true.ops)) != 4 or len(tuple(false.ops)) != 1:
        raise ValueError("padding/gather branch has extra operations or arguments")
    first, second, read, yield_true = tuple(true.ops)
    for op in (first, second):
        if type(op) is not arith.SubiOp:
            raise ValueError("padding/gather image subtraction is not branch-local")
        _plain(op, ("overflowFlags",))
        if op.overflow_flags.data or any(v not in expressions for v in op.operands):
            raise ValueError("padding/gather subtraction has unproved inputs")
        a, b = (expressions[v] for v in op.operands)
        expressions[op.results[0]] = _add(a, tuple(-v for v in b))
    if type(read) is not tensor.ExtractOp:
        raise ValueError("padding/gather branch does not extract the original image")
    _plain(read)
    desired = (_axis(3), _axis(0), _add(source["h"], _constant(-top)), _add(source["w"], _constant(-left)))
    if (
        read.operands[0] is not values[source["image"]]
        or tuple(expressions.get(v) for v in read.operands[1:]) != desired
    ):
        raise ValueError("padding/gather image or exact source coordinates changed")
    for terminator, cls, operand in (
        (yield_true, scf.YieldOp, read.results[0]),
        (false.last_op, scf.YieldOp, values[source["zero"]]),
        (ops[-1], YieldOp, branch.results[0]),
    ):
        _plain(terminator)
        if type(terminator) is not cls or tuple(terminator.operands) != (operand,):
            raise ValueError("padding/gather changed its original scalar zero/read result")
    members = set(writer.walk())
    for op in members - {writer}:
        if any(use.operation not in members for result in op.results for use in result.uses):
            raise ValueError("padding/gather scalar intermediates escape")


def verify_padding_gather_rewrite(before, after, *, index_bits):
    """Reconstruct all realized changes, or refuse any unaccounted IR change."""
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("padding/gather requires an explicit bounded index width")
    before.verify()
    after.verify()
    values, blocks, observed = {}, {}, []

    def bind(a, b):
        if _printed(a.type) != _printed(b.type):
            raise ValueError("padding/gather changed a complete value type")
        values[a] = b

    def header(a, b):
        return (
            a.name == b.name
            and _metadata(a) == _metadata(b)
            and tuple(values.get(v) for v in a.operands) == tuple(b.operands)
            and tuple(blocks.get(v) for v in a.successors) == tuple(b.successors)
            and tuple(_printed(v.type) for v in a.results) == tuple(_printed(v.type) for v in b.results)
            and len(a.regions) == len(b.regions)
        )

    def operation(a, b):
        if not header(a, b):
            raise ValueError("padding/gather changed unmodeled operations, metadata or dataflow")
        for x, y in zip(a.results, b.results, strict=True):
            bind(x, y)
        same = a.is_structurally_equivalent(b, dict(values) | blocks) and tuple(_details(o) for o in a.walk()) == tuple(
            _details(o) for o in b.walk()
        )
        if same:
            return
        if type(a) not in (builtin.ModuleOp, func.FuncOp, scf.ForOp, scf.IfOp):
            raise ValueError("padding/gather changed an unknown enclosing region")
        allowed = (
            {"llvm.emit_c_interface"}
            if type(a) is func.FuncOp
            else ({"llvm.data_layout", "llvm.target_triple"} if type(a) is builtin.ModuleOp else set())
        )
        if any(k not in allowed and not k.startswith("prov.") for k in a.attributes):
            raise ValueError("padding/gather has an unknown enclosing execution contract")
        properties = {"sym_name", "function_type", "sym_visibility"} if type(a) is func.FuncOp else set()
        if set(a.properties) - properties:
            raise ValueError("padding/gather has unknown enclosing properties")
        for ar, br in zip(a.regions, b.regions, strict=True):
            if len(ar.blocks) != len(br.blocks):
                raise ValueError("padding/gather changed control flow")
            blocks.update(zip(ar.blocks, br.blocks, strict=True))
            for ab, bb in zip(ar.blocks, br.blocks, strict=True):
                block(ab, bb)

    def block(a, b):
        if len(a.args) != len(b.args):
            raise ValueError("padding/gather changed block arguments")
        for x, y in zip(a.args, b.args, strict=True):
            bind(x, y)
        removals, candidates = {}, {}
        for op in a.ops:
            if type(op) is GenericOp:
                try:
                    source = _source(op, index_bits)
                except (ValueError, AttributeError, IndexError):
                    continue
                candidates[op] = source
                for removed in source["removed"]:
                    removals[removed] = op
        actual, i, pending = tuple(b.ops), 0, {}
        for old in a.ops:
            if old in pending:
                if i >= len(actual):
                    raise ValueError("padding/gather realization is missing")
                source = candidates[old]
                if pending[old] != set(source["removed"]):
                    raise ValueError("padding/gather retained part of its claimed eliminated padding")
                _realization(source, actual[i], values, index_bits)
                bind(old.results[0], actual[i].results[0])
                observed.append(
                    {
                        "image_shape": list(source["shape"]),
                        "padded_shape": list(source["padded"]),
                        "gather_shape": list(source["gathered"]),
                        "index_bits": index_bits,
                        "eliminated_padding_payload_bytes": prod(source["padded"]) * source["element_bytes"],
                        "gather_payload_bytes": prod(source["gathered"]) * source["element_bytes"],
                    }
                )
                del pending[old]
                i += 1
            elif i < len(actual) and header(old, actual[i]):
                operation(old, actual[i])
                i += 1
            elif old in removals:
                pending.setdefault(removals[old], set()).add(old)
            else:
                raise ValueError("padding/gather removed or reordered an unmodeled operation/effect")
        if pending or i != len(actual):
            raise ValueError("padding/gather has an incomplete or unaccounted realization")

    operation(before, after)
    return tuple(observed)
