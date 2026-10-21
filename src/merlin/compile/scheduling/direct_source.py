"""Read a closed typed signed-dot proposal and positional pointer declaration.

This public compiler helper checks current source against supplied analysis data.
That data is not an original proof receipt. It grants no numerical, execution,
source-owner, storage-lifetime or timing authority; independent host checkers
still establish every corresponding obligation on the actual emitted program.
Only standard tensor/arithmetic/function/linalg grammar is loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.parser import Parser

from merlin.transforms.dense_ir import recognize_signed_integer_matmul


def bind_direct_source_contraction(source: Path, command_buffer: dict, proof: dict) -> dict:
    if type(command_buffer) is not dict or type(proof) is not dict:
        raise ValueError("direct source proposal requires explicit data mappings")
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("direct source binding needs its exact ordinary source")
    if source.stat().st_size > 1 << 20:
        raise ValueError("direct source proposal exceeds its one-MiB lexical bound")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != proof.get("interface_sha256"):
        raise ValueError("direct source binding differs from its current typed proof")
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, linalg.Linalg, tensor.Tensor):
        context.load_dialect(dialect)
    module = Parser(context, raw.decode()).parse_module()
    module.verify()
    functions = [operation for operation in module.walk() if isinstance(operation, func.FuncOp)]
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
    records = proof.get("operations")
    if type(records) is not list or any(type(row) is not dict for row in records):
        raise ValueError("direct source proposal has no explicit scalar analysis records")
    matches = [row for row in records if row.get("ordinal") == 2]
    if len(matches) != 1:
        raise ValueError("direct source proposal has no unique complete reduction analysis")
    record = matches[0]
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
    if source.read_bytes() != raw:
        raise ValueError("direct source proposal changed during its typed read")
    return {"ordinal": 2, "lhs": lhs, "rhs": rhs, "output": output, "m": m, "k": k, "n": n}
