"""Explicit synchronous floating-matmul callbacks inside ordinary model lowering.

Reuse the closed tensor reader to qualify source arithmetic. Everything outside
the selected contractions stays in the original MLIR for ordinary host lowering.
The caller selects a numerical replacement policy; this diagnostic seam does
not establish source equivalence, hardware support or experiment admission.
"""

from hashlib import sha256

from xdsl.dialects import arith, bufferization, builtin, func, memref, tensor
from xdsl.dialects.linalg.ops import FillOp
from xdsl.ir import Block, Region

from merlin.frontends.linalg_tensor_graph import contraction_kind, read_tensor_graph
from merlin.frontends.linalg_text import parse_mlir_text
from merlin.xdsl_dialects._common import text


def static_tensor_signature(source, *, entry):
    """Explicit ranked host ABI; integer IDs/masks never change contraction dtype."""
    module = parse_mlir_text(source)
    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not func.FuncOp or functions[0].sym_name.data != entry:
        raise ValueError("one matching source function required")
    fn = functions[0]
    result = {}
    for name, types in (("inputs", fn.function_type.inputs), ("outputs", fn.function_type.outputs)):
        if any(
            type(t) is not builtin.TensorType
            or type(t.encoding) is not builtin.NoneAttr
            or any(d <= 0 for d in t.get_shape())
            or str(t.element_type) not in ("f32", "f64", "i1", "i8", "i16", "i32", "i64")
            for t in types
        ):
            raise ValueError("host ABI requires supported positive static tensor types")
        result[name] = [list(t.get_shape()) for t in types]
        result[name + "_dtypes"] = [str(t.element_type) for t in types]
    return result


def contraction_source(op):
    """Extract and check actual operands, recurrence and positive-zero seed."""
    if contraction_kind(op) is None or len(op.inputs) != 2 or len(op.outputs) != 1:
        raise ValueError("one checked two-input matmul required")
    fill = op.outputs[0].owner
    if type(fill) is tensor.SplatOp:
        dependencies = (fill.operands[0].owner, fill)
    elif type(fill) is FillOp and len(fill.inputs) == len(fill.outputs) == 1:
        zero, empty = fill.inputs[0].owner, fill.outputs[0].owner
        if type(empty) is not tensor.EmptyOp:
            raise ValueError("matmul requires a static empty")
        dependencies = (zero, empty, fill)
    else:
        raise ValueError("matmul requires a literal fill or splat")
    if type(dependencies[0]) is not arith.ConstantOp:
        raise ValueError("matmul requires a constant initializer")
    block = Block(arg_types=[v.type for v in op.inputs])
    mapping = dict(zip(op.inputs, block.args, strict=True))
    for original in dependencies:
        block.add_op(original.clone(value_mapper=mapping))
    cloned = op.clone(value_mapper=mapping)
    # The same source SSA value can occupy both operand positions. Keep two
    # independent ABI parameters in the extracted source in that case too.
    cloned.operands = (*block.args, mapping[op.outputs[0]])
    block.add_op(cloned)
    block.add_op(func.ReturnOp(mapping[op.results[0]]))
    fn = func.FuncOp("contraction", ([v.type for v in op.inputs], [op.results[0].type]), Region(block))
    source = text(builtin.ModuleOp([fn]), generic=True)
    request = read_tensor_graph(source, entry="contraction", target_identity="source-only")
    return source, request


def outline_floating_matmuls(source, *, entry, numerical_policy):
    """Outline checked f32 named/canonical-generic contractions, retaining other ops.

    Callback ABI is three ranked memref descriptors (lhs, rhs, fresh output),
    synchronous, read-only inputs and complete output writes. The existing
    upstream ownership/deallocation pipeline owns the fresh output allocation.
    Host operations retain their original precision. Nested matmuls refuse;
    other contraction forms remain explicit host operations.
    """
    if type(numerical_policy) is not str or not numerical_policy.strip():
        raise ValueError("explicit numerical replacement policy required")
    module = parse_mlir_text(source)
    module.verify()
    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not func.FuncOp:
        raise ValueError("one source function required")
    fn = functions[0]
    if fn.sym_name.data != entry or len(fn.body.blocks) != 1:
        raise ValueError("entry or control flow differs")
    records = []
    for ordinal, op in enumerate(tuple(fn.walk())):
        if contraction_kind(op) is None:
            continue
        if op.parent_block() is not fn.body.block:
            raise ValueError("nested matmul requires explicit control-flow dispatch")
        kernel, request = contraction_source(op)
        if any(n.type.dtype != "f32" for n in request.nodes):
            raise ValueError("floating callback ABI requires f32 tensors")
        symbol = f"merlin_matmul_{len(records)}"
        if symbol == entry:
            raise ValueError("entry collides with callback symbol")
        types = [builtin.MemRefType(v.type.element_type, v.type.get_shape()) for v in (*op.inputs, *op.results)]
        buffers = [
            bufferization.ToBufferOp(operands=[v], result_types=[t], properties={"read_only": builtin.UnitAttr()})
            for v, t in zip(op.inputs, types[:2], strict=True)
        ]
        output = memref.AllocOp((), (), types[2])
        call = func.CallOp(symbol, [*buffers, output], [])
        result = bufferization.ToTensorOp(output, restrict=True, writable=True)
        fn.body.block.insert_ops_before([*buffers, output, call, result], op)
        op.results[0].replace_all_uses_with(result.tensor)
        fn.body.block.erase_op(op)
        declaration = func.FuncOp.external(symbol, types, [])
        declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
        module.body.block.add_op(declaration)
        records.append(
            {
                "symbol": symbol,
                "source_ordinal": ordinal,
                "source": kernel,
                "source_sha256": sha256(kernel.encode()).hexdigest(),
                "input_shapes": [list(n.type.shape) for n in request.nodes if n.op == "input"],
                "output_shape": list(request.nodes[-1].type.shape),
                "numerical_policy": numerical_policy,
            }
        )
    fn.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module.verify()
    return text(module, generic=True), records


def callback_bridge(records):
    """C-interface transport hook; failure aborts rather than publishing garbage.

    The selected runtime supplies descriptor validation, numerical conversion,
    execution and synchronous completion. There is no CPU arithmetic fallback.
    """
    lines = [
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "typedef int (*Dispatch)(int64_t,void*,void*,void*);",
        "static Dispatch dispatch;",
        "void merlin_set_dispatch(Dispatch f) { dispatch=f; }",
    ]
    for i, record in enumerate(records):
        if record["symbol"] != f"merlin_matmul_{i}":
            raise ValueError("callback symbol differs from outlined ordinal")
        lines.append(
            f"void _mlir_ciface_{record['symbol']}(void*a,void*b,void*c) {{"
            f"if(!dispatch || dispatch({i},a,b,c)) abort(); }}"
        )
    return "\n".join(lines) + "\n"
