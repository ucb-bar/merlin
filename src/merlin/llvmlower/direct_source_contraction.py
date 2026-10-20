"""Exact single signed-dot source binding without a compiler global plan.

This establishes typed source/positional pointer names only. Actual execution,
complete products, host writes and readout still require independent checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from xdsl.dialects import arith, builtin, func, tensor

from merlin.common import mlir_query as mq
from merlin.transforms.dense_ir import recognize_signed_integer_matmul


def bind_direct_source_contraction(source: Path, command_buffer: dict, proof: dict) -> dict:
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("direct source binding needs its exact ordinary source")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != proof.get("interface_sha256"):
        raise ValueError("direct source binding differs from its current typed proof")
    module = mq.parse(raw.decode())
    module.verify()
    functions = list(mq.walk(module, "func.func"))
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1 or list(module.body.block.ops) != functions:
        raise ValueError("direct source binding requires one closed source function")
    block = functions[0].regions[0].block
    ops = list(block.ops)
    if (
        len(block.args) != 2
        or len(ops) != 4
        or not isinstance(ops[0], arith.ConstantOp)
        or not isinstance(ops[1], tensor.SplatOp)
        or not isinstance(ops[3], func.ReturnOp)
    ):
        raise ValueError("direct source binding supports only zero constant, splat, one signed dot and return")
    constant, splat, dot, returned = ops
    if set(constant.properties) != {"value"} or splat.properties or returned.properties:
        raise ValueError("direct source binding has unknown source properties")
    clean = dot.clone()
    for nested in clean.walk():
        for name in tuple(nested.attributes):
            if name.startswith("prov."):
                del nested.attributes[name]
    recognize_signed_integer_matmul(clean, ("i8", "i8", "i32"))
    if (
        not isinstance(constant.value, builtin.IntegerAttr)
        or constant.value.value.data != 0
        or constant.value.type != builtin.i32
        or tuple(splat.operands) != (constant.results[0],)
        or tuple(dot.inputs) != tuple(block.args)
        or tuple(dot.outputs) != (splat.results[0],)
        or tuple(returned.operands) != (dot.results[0],)
    ):
        raise ValueError("direct source binding changes signed operands, zero initialization or complete return")
    for op in (constant, splat, returned):
        if any(not name.startswith("prov.") for name in op.attributes):
            raise ValueError("direct source binding has unknown source effects")
    record = next((row for row in proof.get("operations", []) if row.get("ordinal") == 2), {})
    shapes = [tuple(value.type.get_shape()) for value in (*block.args, dot.results[0])]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("direct source binding requires rank-two dense tensors")
    (m, k), (rhs_k, n), out = shapes
    if (
        min(m, k, n) <= 0
        or rhs_k != k
        or out != (m, n)
        or record.get("kind") != "multiply_accumulate"
        or record.get("initial_interval") != [0, 0]
        or record.get("reduction_extent") != k
        or record.get("result_bits") != 32
    ):
        raise ValueError("direct source proof does not establish the complete signed zero-initialized reduction")
    declaration = command_buffer.get("compiler_pointer_abi")
    tensors = command_buffer.get("tensors")
    if (
        type(declaration) is not dict
        or set(declaration) != {"version", "arguments"}
        or type(declaration["version"]) is not int
        or declaration["version"] != 1
        or type(declaration["arguments"]) is not list
        or len(declaration["arguments"]) != 3
        or any(type(name) is not str or not name for name in declaration["arguments"])
        or len(set(declaration["arguments"])) != 3
        or type(tensors) is not dict
        or set(tensors) != set(declaration["arguments"])
    ):
        raise ValueError("direct source binding has no complete original three-pointer ABI")
    for name, shape, dtype, role in zip(
        declaration["arguments"], shapes, ("i8", "i8", "i32"), ("input", "input", "output"), strict=True
    ):
        row = tensors[name]
        if (
            type(row) is not dict
            or row.get("shape") != list(shape)
            or row.get("dtype") != dtype
            or row.get("role") != role
            or set(row) - {"shape", "dtype", "role", "layout"}
            or row.get("layout", "dense_row_major") not in {"row_major", "dense_row_major"}
        ):
            raise ValueError("direct source logical pointer storage differs from typed source")
    lhs, rhs, output = declaration["arguments"]
    return {"ordinal": 2, "lhs": lhs, "rhs": rhs, "output": output, "m": m, "k": k, "n": n}
