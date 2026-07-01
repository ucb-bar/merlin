"""Isolated Saturn-vectors target dialect (xDSL PROTOTYPE) — run hand_v0.

A NON-matmul (vector/SIMD) family: `vector.map` (elementwise combine + activation) and
`vector.reduce` (reduction). Self-contained and loaded dynamically (not in the core tree).
Includes a direct lowering to the target-independent command buffer (VECTOR_MAP / VREDUCE).

Note (honest, see SV findings): the merlin contract/schedule "decision" layers are
matmul-residency-specific, so the vector family lowers DIRECTLY from its interface dialect to
the command buffer — it needs no residency/packing decisions. That's an architectural
observation, not a gap: the certified interface is the command buffer.
"""
from __future__ import annotations

from typing import Any

from xdsl.ir import Block, Region, Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, prop_def, result_def
from xdsl.dialects.builtin import ArrayAttr, StringAttr, TensorType, ModuleOp, FunctionType, i32
from xdsl.dialects.func import FuncOp, ReturnOp

DIALECT_NAME = "saturn_vec"
OPS = ["map", "reduce"]


@irdl_op_definition
class VectorMapOp(IRDLOperation):
    """vector.map — elementwise combine(lhs,rhs) + optional activation, over equal-shape vectors."""
    name = "saturn_vec.map"
    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    combine = prop_def(StringAttr)         # "add" | "mul"
    activation = opt_prop_def(ArrayAttr)   # e.g. ["relu"]
    dst = result_def(TensorType)


@irdl_op_definition
class VectorReduceOp(IRDLOperation):
    """vector.reduce — reduce a vector to a length-1 tensor."""
    name = "saturn_vec.reduce"
    src = operand_def(TensorType)
    redop = prop_def(StringAttr)           # "sum"
    dst = result_def(TensorType)


SATURN_VEC_DIALECT = Dialect(DIALECT_NAME, [VectorMapOp, VectorReduceOp], [])


def _vec(n: int) -> TensorType:
    return TensorType(i32, [n])


def build_example(n: int = 64) -> ModuleOp:
    """A dot product expressed in the vector family: s = sum(x * w)."""
    blk = Block(arg_types=[_vec(n), _vec(n)])
    x, w = blk.args
    mul = VectorMapOp(operands=[x, w], result_types=[_vec(n)],
                      properties={"combine": StringAttr("mul"),
                                  "activation": ArrayAttr([])})
    red = VectorReduceOp(operands=[mul.dst], result_types=[_vec(1)],
                         properties={"redop": StringAttr("sum")})
    blk.add_ops([mul, red, ReturnOp(red.dst)])
    fn = FuncOp("forward", FunctionType.from_lists([_vec(n), _vec(n)], [_vec(1)]), Region([blk]))
    return ModuleOp([fn])


def lower_to_command_buffer(module: ModuleOp, *, input_names=("x", "w"),
                            output_name="s") -> dict[str, Any]:
    """Lower a saturn_vec xDSL module to the target-independent command buffer."""
    fn = next(op for op in module.walk() if op.name == "func.func")
    blk = fn.body.blocks[0]
    names: dict = {a: input_names[i] for i, a in enumerate(blk.args)}
    tensors: dict[str, Any] = {}
    for a in blk.args:
        tensors[names[a]] = {"shape": list(a.type.get_shape()), "dtype": "i32", "role": "input"}
    commands: list[dict] = []
    n_tmp = 0
    for op in blk.ops:
        if isinstance(op, VectorMapOp):
            dst = "t%d" % n_tmp
            n_tmp += 1
            names[op.dst] = dst
            commands.append({"opcode": "VECTOR_MAP",
                             "operands": {"lhs": names[op.lhs], "rhs": names[op.rhs], "dst": dst},
                             "attributes": {"combine": op.combine.data,
                                            "activation": [s.data for s in (op.activation or [])]}})
        elif isinstance(op, VectorReduceOp):
            names[op.dst] = output_name
            commands.append({"opcode": "VREDUCE",
                             "operands": {"src": names[op.src], "dst": output_name},
                             "attributes": {"op": op.redop.data}})
        elif op.name == "func.return":
            ret = op.operands[0]
            names.setdefault(ret, output_name)
            tensors[names[ret]] = {"shape": list(ret.type.get_shape()), "dtype": "i32",
                                   "role": "output"}
    return {"abi_version": "0.1", "target": DIALECT_NAME, "backend": "spike",
            "tensors": tensors, "commands": commands}


def get_dialect() -> Dialect:
    return SATURN_VEC_DIALECT
