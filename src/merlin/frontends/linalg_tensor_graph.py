"""Closed static floating tensor graphs, independent of target precision policy.

Recognize actual scalar bodies and affine maps. Provenance is retained by the
source digest but never selects semantics. No numerical equivalence to a target
or change in reduction order is established by this source-only reader.
"""

from hashlib import sha256
from math import copysign, isnan

from xdsl.dialects import arith, builtin, func, math, tensor
from xdsl.dialects.linalg.ops import FillOp, GenericOp, MatmulOp, ReduceOp, TransposeOp, YieldOp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType

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
        and str(t.element_type) in ("f16", "bf16", "f32", "f64"),
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


def scalar_body(op, operations, operands):
    require(len(op.regions) == 1 and len(op.regions[0].blocks) == 1, "one scalar block required")
    block = op.regions[0].block
    require(
        tuple(arg.type for arg in block.args)
        == tuple(v.type.element_type if isinstance(v.type, builtin.TensorType) else v.type for v in op.operands),
        "scalar argument types differ from tensor operands",
    )
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
            f"node{len(nodes)}", kind, tuple(values[v] for v in inputs), tensor_type(op.results[0]), tuple(attrs)
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
        elif type(op) is MatmulOp:
            metadata(op, ("operandSegmentSizes", "indexing_maps"))
            require(len(op.inputs) == 2 and len(op.outputs) == len(op.results) == 1, "matmul arity differs")
            a, b, out = (tensor_type(v) for v in (*op.inputs, op.results[0]))
            require(
                len(a.shape) == len(b.shape) == 2
                and a.shape[1] == b.shape[0]
                and out.shape == (a.shape[0], b.shape[1])
                and a.dtype == b.dtype == out.dtype
                and op.outputs[0].type == op.results[0].type,
                "matmul shape or element type differs",
            )
            dims = tuple(AffineDimExpr(i) for i in range(3))
            expected = ((dims[0], dims[2]), (dims[2], dims[1]), (dims[0], dims[1]))
            require(
                tuple(m.data for m in op.indexing_maps) == tuple(AffineMap(3, 0, v) for v in expected),
                "matmul indexing differs",
            )
            scalar_body(
                op,
                (arith.MulfOp, arith.AddfOp),
                (
                    lambda b, ops: b.args[:2],
                    lambda b, ops: (ops[0].results[0], b.args[2]),
                    lambda b, ops: (ops[1].results[0],),
                ),
            )
            fill = op.outputs[0].owner
            require(type(fill) is FillOp and len(fill.inputs) == len(fill.outputs) == 1, "zero-seeded matmul required")
            metadata(fill, ("operandSegmentSizes",))
            scalar_body(fill, (), (lambda b, ops: (b.args[0],),))
            value = scalar_constant(fill.inputs[0])
            require(
                fill.inputs[0].type == op.results[0].type.element_type and fill.outputs[0].type == op.results[0].type,
                "matmul initializer type differs",
            )
            require(value == 0 and copysign(1, value) > 0, "positive zero matmul seed required")
            empty(fill.outputs[0])
            used.update((fill, fill.inputs[0].owner))
            emit(op, "matmul", tuple(op.inputs), (("reduction_order", "source-unspecified"),))
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
                require(ty.dtype == tensor_type(op.results[0]).dtype, "pointwise element type differs")
            body = tuple(op.body.block.ops)
            kinds = {
                arith.AddfOp: "add",
                arith.SubfOp: "sub",
                arith.MulfOp: "mul",
                arith.DivfOp: "div",
                math.ExpOp: "exp",
            }
            if body and type(body[0]) in kinds:
                cls = type(body[0])
                kind = kinds[cls]
                require(len(op.inputs) == (1 if cls is math.ExpOp else 2), "pointwise arity differs")
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
            emit(op, kind, tuple(op.inputs))
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
