"""Closed static floating tensor graphs, independent of target precision policy.

Recognize actual scalar bodies and affine maps. Provenance is retained by the
source digest but never selects semantics. No numerical equivalence to a target
or change in reduction order is established by this source-only reader.
"""

from hashlib import sha256
from math import copysign, isnan

from xdsl.dialects import arith, builtin, func, math, tensor
from xdsl.dialects.linalg.attrs import IteratorType
from xdsl.dialects.linalg.ops import CopyOp, FillOp, GenericOp, MatmulOp, ReduceOp, TransposeOp, YieldOp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.xdsl_dialects.fp8 import FP8_TYPE_NAMES

from .linalg_batch_matmul import BatchMatmulOp
from .linalg_bf16_add import check_bf16_chop_binary_body
from .linalg_text import parse_mlir_text


def require(condition, message):
    if not condition:
        raise ValueError("tensor graph: " + message)


def metadata(op, properties=()):
    require(
        all(k.startswith("prov.") for k in op.attributes) and set(op.properties) <= set(properties),
        "unknown semantic attributes or properties on " + op.name,
    )


def tensor_type(value):
    t = value.type
    require(
        type(t) is builtin.TensorType
        and type(t.encoding) is builtin.NoneAttr
        and str(t.element_type) in (*FP8_TYPE_NAMES, "f16", "bf16", "f32", "f64"),
        "ordinary floating tensor required",
    )
    return TensorType(tuple(t.get_shape()), str(t.element_type), "source-floating-semantics")


def scalar_constant(value):
    op = value.owner
    require(type(op) is arith.ConstantOp and type(op.value) is builtin.FloatAttr, "floating literal required")
    metadata(op, ("value",))
    literal = op.value.value.data
    require(not isnan(literal), "NaN literal payloads require an explicit bit representation")
    return literal


def scalar_block(op):
    require(len(op.regions) == 1 and len(op.regions[0].blocks) == 1, "one scalar block required")
    block = op.regions[0].block
    require(
        tuple(arg.type for arg in block.args)
        == tuple(v.type.element_type if isinstance(v.type, builtin.TensorType) else v.type for v in op.operands),
        "scalar argument types differ from tensor operands",
    )
    return block


def scalar_body(op, operations, operands):
    block = scalar_block(op)
    body = tuple(block.ops)
    require(tuple(type(inner) for inner in body) == (*operations, YieldOp), "scalar operation sequence differs")
    for inner, expected in zip(body, operands):
        metadata(inner, () if type(inner) is YieldOp else ("fastmath",))
        require(
            not getattr(inner.properties.get("fastmath"), "data", ())
            and tuple(inner.operands) == tuple(expected(block, body)),
            "scalar operands or flags differ",
        )
    return body


def floating_body(op, *, contraction=False):
    """Read explicit casts and arithmetic in SSA order, including their intermediate precision."""
    block = scalar_block(op)
    body = iter(block.ops)
    current = next(body, None)
    require(current is not None and len(block.args) > 1, "nonempty scalar program with inputs required")

    def take(cls, operands):
        nonlocal current
        require(type(current) is cls and tuple(current.operands) == tuple(operands), "scalar operands or opcode differ")
        metadata(current, () if cls is YieldOp else ("fastmath",))
        require(not getattr(current.properties.get("fastmath"), "data", ()), "scalar arithmetic flags differ")
        consumed = current
        current = next(body, None)
        return consumed

    values = []
    for arg in block.args[:-1]:
        if type(current) is arith.ExtFOp and tuple(current.operands) == (arg,):
            widened = take(arith.ExtFOp, (arg,)).results[0]
            require(widened.type.bitwidth > arg.type.bitwidth, "expected a widening conversion")
            arg = widened
        values.append(arg)
    compute = values[0].type
    require(all(v.type == compute for v in values), "scalar compute types differ")
    if contraction:
        product = take(arith.MulfOp, values).results[0]
        acc = block.args[-1]
        require(product.type == acc.type == compute, "contraction compute type differs")
        acc_first = (
            type(op) is GenericOp and type(current) is arith.AddfOp and tuple(current.operands) == (acc, product)
        )
        result = take(arith.AddfOp, (acc, product) if acc_first else (product, acc)).results[0]
        kind = contraction_kind(op)
        attrs = (("reduction_order", "source-unspecified"),)
        if any(arg.type != compute for arg in block.args[:-1]):
            kind = "widening_" + kind
            attrs += (("accumulator_first", acc_first), ("compute_dtype", str(compute)))
    elif len(values) == 2:
        result = take(arith.AddfOp, values).results[0]
        kind, attrs = "mixed_add", (("compute_dtype", str(compute)),)
    else:
        require(len(values) == 1, "unary conversion required")
        result = values[0]
        kind, attrs = "convert", (("compute_dtype", str(compute)),)
    require(result.type == compute, "scalar result type differs")
    if not contraction:
        narrowing = type(current) is arith.TruncFOp
        if narrowing:
            result = take(arith.TruncFOp, (result,)).results[0]
            require(result.type.bitwidth < compute.bitwidth, "expected a narrowing conversion")
        attrs += (("narrowing", "truncf" if narrowing else "none"),)
    require(result.type == block.args[-1].type, "scalar yield type differs")
    take(YieldOp, (result,))
    require(current is None, "unconsumed scalar operation")
    return kind, attrs


def contraction_kind(op):
    """Recognize standard rank-two/three contraction geometry, without provenance hints.

    Scalar arithmetic and initializer legality are checked by read_tensor_graph;
    matching maps alone never grants a numerical replacement.
    """
    if type(op) in (MatmulOp, BatchMatmulOp):
        return "batch_matmul" if type(op) is BatchMatmulOp else "matmul"
    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    if not isinstance(op.inputs[0].type, builtin.TensorType):
        return None
    rank = len(op.inputs[0].type.get_shape())
    if rank not in (2, 3):
        return None
    dims = tuple(AffineDimExpr(i) for i in range(rank + 1))
    batch = dims[:1] if rank == 3 else ()
    m, n, k = dims[-3:]
    maps = tuple(AffineMap(rank + 1, 0, v) for v in ((*batch, m, k), (*batch, k, n), (*batch, m, n)))
    if tuple(v.data for v in op.indexing_maps) != maps:
        return None
    if tuple(v.data for v in op.iterator_types) != (IteratorType.PARALLEL,) * rank + (IteratorType.REDUCTION,):
        return None
    return "batch_matmul" if rank == 3 else "matmul"


def read_tensor_graph(source, *, entry, target_identity):
    module = parse_mlir_text(source)
    module.verify()
    metadata(module)
    functions = tuple(module.body.block.ops)
    require(len(functions) == 1 and type(functions[0]) is func.FuncOp, "one function required")
    fn = functions[0]
    metadata(fn, ("sym_name", "function_type", "arg_attrs", "res_attrs", "sym_visibility"))
    require(fn.sym_name.data == entry and len(fn.body.blocks) == 1, "entry or control flow differs")
    require(
        all(not item.data for attrs in (fn.arg_attrs, fn.res_attrs) if attrs is not None for item in attrs),
        "semantic boundary attributes require explicit support",
    )
    block = fn.body.block
    nodes, values, used = [], {}, set()

    def emit(op, kind, inputs, attrs=()):
        require(len(op.results) == 1 and all(v in values for v in inputs), "unknown producer or result arity")
        node = SemanticNode(
            f"node{len(nodes)}",
            kind,
            tuple(values[v] for v in inputs),
            tensor_type(op.results[0]),
            tuple(sorted(attrs)),
        )
        nodes.append(node)
        values[op.results[0]] = node.id
        used.add(op)

    def empty(value):
        op = value.owner
        require(type(op) is tensor.EmptyOp and not op.operands, "static empty destination required")
        metadata(op)
        used.add(op)

    for i, arg in enumerate(block.args):
        node = SemanticNode(f"input{i}", "input", (), tensor_type(arg), effect="input")
        nodes.append(node)
        values[arg] = node.id
    operations = tuple(block.ops)
    require(operations and type(operations[-1]) is func.ReturnOp, "return required")
    for op in operations[:-1]:
        if type(op) in (arith.ConstantOp, tensor.EmptyOp, FillOp):
            continue
        if type(op) is tensor.SplatOp:
            metadata(op)
            require(
                len(op.operands) == 1 and op.operands[0].type == op.results[0].type.element_type,
                "static splat element type differs",
            )
            value = scalar_constant(op.operands[0])
            used.add(op.operands[0].owner)
            # Hexadecimal spelling preserves signed zero and infinities without
            # allowing non-finite JSON numbers into the semantic graph.
            emit(op, "splat", (), (("value_hex", value.hex()),))
        elif type(op) is CopyOp:
            metadata(op, ("operandSegmentSizes",))
            require(len(op.inputs) == len(op.outputs) == len(op.results) == 1, "copy arity differs")
            require(op.inputs[0].type == op.outputs[0].type == op.results[0].type, "copy requires matching types")
            scalar_body(op, (), (lambda b, ops: (b.args[0],),))
            empty(op.outputs[0])
            emit(op, "copy", tuple(op.inputs))
        elif type(op) is TransposeOp:
            metadata(op, ("permutation",))
            require(tuple(op.permutation.iter_values()) == (1, 0), "rank-two transpose required")
            src, dst = tensor_type(op.operands[0]), tensor_type(op.results[0])
            require(
                len(src.shape) == 2
                and dst.shape == src.shape[::-1]
                and src.dtype == dst.dtype
                and op.operands[1].type == op.results[0].type,
                "transpose result type differs",
            )
            scalar_body(op, (), (lambda b, ops: (b.args[0],),))
            empty(op.operands[1])
            emit(op, "transpose", (op.operands[0],))
        elif contraction_kind(op) is not None:
            metadata(op, ("operandSegmentSizes", "indexing_maps", "iterator_types"))
            require(len(op.inputs) == 2 and len(op.outputs) == len(op.results) == 1, "matmul arity differs")
            a, b, out = (tensor_type(v) for v in (*op.inputs, op.results[0]))
            batched = contraction_kind(op) == "batch_matmul"
            rank = 3 if batched else 2
            require(
                len(a.shape) == len(b.shape) == rank
                and a.shape[:-2] == b.shape[:-2]
                and a.shape[-1] == b.shape[-2]
                and out.shape == (*a.shape[:-2], a.shape[-2], b.shape[-1])
                and op.outputs[0].type == op.results[0].type,
                "matmul shape or element type differs",
            )
            dims = tuple(AffineDimExpr(i) for i in range(rank + 1))
            batch = dims[:1] if batched else ()
            m, n, k = dims[-3:]
            expected = ((*batch, m, k), (*batch, k, n), (*batch, m, n))
            require(
                tuple(m.data for m in op.indexing_maps) == tuple(AffineMap(rank + 1, 0, v) for v in expected),
                "matmul indexing differs",
            )
            kind, attrs = floating_body(op, contraction=True)
            fill = op.outputs[0].owner
            if type(fill) is tensor.SplatOp:
                metadata(fill)
                require(len(fill.operands) == 1, "one splat scalar required")
                initializer = fill.operands[0]
            else:
                require(
                    type(fill) is FillOp and len(fill.inputs) == len(fill.outputs) == 1, "zero-seeded matmul required"
                )
                metadata(fill, ("operandSegmentSizes",))
                scalar_body(fill, (), (lambda b, ops: (b.args[0],),))
                initializer = fill.inputs[0]
                require(fill.outputs[0].type == op.results[0].type, "matmul initializer type differs")
                empty(fill.outputs[0])
            value = scalar_constant(initializer)
            require(initializer.type == op.results[0].type.element_type, "matmul initializer type differs")
            require(value == 0 and copysign(1, value) > 0, "positive zero matmul seed required")
            used.update((fill, initializer.owner))
            emit(op, kind, tuple(op.inputs), attrs)
        elif type(op) is ReduceOp:
            metadata(op, ("dimensions",))
            axes = tuple(op.dimensions.iter_values())
            require(len(axes) == 1, "one reduction axis required")
            src, dst = tensor_type(op.input), tensor_type(op.results[0])
            require(
                0 <= axes[0] < len(src.shape)
                and dst.shape == src.shape[: axes[0]] + src.shape[axes[0] + 1 :]
                and src.dtype == dst.dtype
                and op.init.type == op.results[0].type,
                "reduction shape or element type differs",
            )
            body = tuple(op.region.block.ops)
            require(body and type(body[0]) in (arith.MaximumfOp, arith.AddfOp), "unsupported reduction")
            cls = type(body[0])
            kind = "reduce_max" if cls is arith.MaximumfOp else "reduce_sum"
            scalar_body(op, (cls,), (lambda b, ops: b.args, lambda b, ops: (ops[0].results[0],)))
            init = op.init.owner
            require(type(init) is tensor.SplatOp, "reduction needs a literal splat seed")
            value = scalar_constant(init.operands[0])
            require(
                value == float("-inf") if kind == "reduce_max" else value == 0 and copysign(1, value) > 0,
                "reduction seed differs",
            )
            # Initializer remains an explicit graph input to the reduction.
            emit(op, kind, (op.input, op.init), (("axis", axes[0]), ("reduction_order", "source-unspecified")))
        elif type(op) is tensor.ExpandShapeOp:
            metadata(op, ("reassociation", "static_output_shape"))
            src, dst = tensor_type(op.operands[0]), tensor_type(op.results[0])
            require(
                len(op.operands) == 1 and len(src.shape) == 1 and dst.shape == (src.shape[0], 1),
                "only rank-one to trailing singleton expansion supported",
            )
            require(
                src.dtype == dst.dtype
                and tuple(tuple(i.value.data for i in group) for group in op.reassociation) == ((0, 1),),
                "expansion reassociation differs",
            )
            require(
                tuple(op.static_output_shape.iter_values()) == dst.shape,
                "expansion output shape differs from result type",
            )
            emit(op, "expand_dims", (op.operands[0],), (("axis", 1),))
        elif type(op) is GenericOp:
            metadata(op, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
            require(
                len(op.outputs) == len(op.results) == 1 and all(a.data.value == "parallel" for a in op.iterator_types),
                "one parallel pointwise output required",
            )
            shape = tensor_type(op.results[0]).shape
            rank = len(shape)
            require(len(op.iterator_types) == rank, "loop rank differs")
            maps = tuple(m.data for m in op.indexing_maps)
            identity = AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank)))
            require(maps[-1] == identity and len(maps) == len(op.inputs) + 1, "result identity map required")
            require(op.outputs[0].type == op.results[0].type, "pointwise destination type differs")
            for value, mapping in zip(op.inputs, maps):
                ty = tensor_type(value)
                require(
                    len(ty.shape) == rank and all(a == b or a == 1 for a, b in zip(ty.shape, shape)),
                    "pointwise shape is not a static singleton broadcast",
                )
                expected = AffineMap(
                    rank,
                    0,
                    tuple(
                        AffineConstantExpr(0) if a == 1 and b != 1 else AffineDimExpr(i)
                        for i, (a, b) in enumerate(zip(ty.shape, shape))
                    ),
                )
                require(mapping == expected, "pointwise indexing differs")

            body = tuple(op.body.block.ops)
            kinds = {
                arith.AddfOp: "add",
                arith.SubfOp: "sub",
                arith.MulfOp: "mul",
                arith.DivfOp: "div",
                math.ExpOp: "exp",
            }
            attrs = ()
            if any(type(inner) is arith.BitcastOp for inner in body):
                require(len(op.inputs) == 2, "BF16 chop needs two operands")
                subtract = any(type(inner) is arith.SubfOp for inner in body)
                scalar_block(op)
                check_bf16_chop_binary_body(op, subtract=subtract)
                kind = "mixed_sub" if subtract else "mixed_add"
                attrs = (("compute_dtype", "f32"), ("narrowing", "bf16_high_bits"))
            elif any(type(inner) in (arith.ExtFOp, arith.TruncFOp) for inner in body) or len(body) == 1:
                kind, attrs = floating_body(op)
            elif body and type(body[0]) in kinds:
                cls = type(body[0])
                kind = kinds[cls]
                require(len(op.inputs) == (1 if cls is math.ExpOp else 2), "pointwise arity differs")
                require(
                    all(v.type.element_type == op.results[0].type.element_type for v in op.inputs),
                    "pointwise element type differs",
                )
                scalar_body(op, (cls,), (lambda b, ops: b.args[:-1], lambda b, ops: (ops[0].results[0],)))
            else:
                require(
                    len(op.inputs) == 1
                    and len(body) == 3
                    and type(body[0]) is arith.ConstantOp
                    and type(body[1]) is arith.MaximumfOp
                    and type(body[2]) is YieldOp,
                    "unsupported scalar body",
                )
                v = scalar_constant(body[0].results[0])
                require(
                    v == 0
                    and copysign(1, v) > 0
                    and body[1].results[0].type == op.results[0].type.element_type
                    and tuple(body[1].operands) == (op.body.block.args[0], body[0].results[0])
                    and tuple(body[2].operands) == tuple(body[1].results),
                    "ReLU scalar correspondence differs",
                )
                metadata(body[1], ("fastmath",))
                metadata(body[2])
                require(not body[1].fastmath.data, "ReLU flags differ")
                require(
                    tuple(a.type for a in op.body.block.args) == tuple(v.type.element_type for v in op.operands),
                    "ReLU scalar types differ",
                )
                kind = "relu"
            empty(op.outputs[0])
            emit(op, kind, tuple(op.inputs), attrs)
        else:
            raise ValueError("tensor graph: unsupported operation " + op.name)
    ret = operations[-1]
    metadata(ret)
    used.add(ret)
    require(all(v in values for v in ret.operands), "unknown returned value")
    require(used == set(operations), "unconsumed source operation")
    outputs = tuple(values[v] for v in ret.operands)
    return KernelRequest(
        tuple(nodes),
        outputs,
        tuple(f"output:{i}" for i in range(len(outputs))),
        tuple((f"input{i}", f"input:{i}") for i in range(len(block.args))),
        target_identity,
        source_identity=sha256(source.encode()).hexdigest(),
    )
