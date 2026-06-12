"""Build the linalg-level input module for the MVP workload.

``repeated_rhs_matmul``: for i in 0..R-1: Y_i = A_i @ W, with one immutable W. Each
matmul is a real ``linalg.quantized_matmul`` (the standard MLIR idiom for
i8 x i8 -> i32 accumulation; zero points are 0). Outputs stay i32 accumulations —
the workload spec carries no epilogue (merlin/benchmarks/semantic_memory/
repeated_rhs_matmul.yaml: op_sequence [matmul]).
"""
from __future__ import annotations

from .._common import HAS_XDSL


def build_input_module(reuse: int = 4, m: int = 64, k: int = 128, n: int = 64):
    """func @repeated_rhs_matmul(A_0..A_{R-1}: mxk i8, W: kxn i8) -> R x (mxn i32)."""
    if not HAS_XDSL:
        return None
    from xdsl.ir import Block, Region
    from xdsl.dialects import arith
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import (FunctionType, IntegerAttr, ModuleOp, TensorType,
                                       i8, i32)
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops

    if reuse < 1:
        raise ValueError("reuse must be >= 1")
    At = TensorType(i8, [m, k])
    Wt = TensorType(i8, [k, n])
    Ot = TensorType(i32, [m, n])

    arg_types = [At] * reuse + [Wt]
    blk = Block(arg_types=arg_types)
    a_args, w = list(blk.args[:-1]), blk.args[-1]

    zp = arith.ConstantOp(IntegerAttr(0, 32))
    ops = [zp]
    outs = []
    for a in a_args:
        init = tensor_d.EmptyOp((), Ot)
        mm = linalg_ops.QuantizedMatmulOp(
            inputs=(a, w, zp.result, zp.result), outputs=(init.tensor,), res=(Ot,))
        ops += [init, mm]
        outs.append(mm.results[0])
    ops.append(ReturnOp(*outs))
    blk.add_ops(ops)
    fn = FuncOp("repeated_rhs_matmul",
                FunctionType.from_lists(arg_types, [Ot] * reuse), Region([blk]))
    return ModuleOp([fn])


def find_matmuls(module):
    """All linalg matmul-family ops in the module (quantized + plain)."""
    from xdsl.dialects.linalg import ops as linalg_ops

    found = []
    for op in module.walk():
        if isinstance(op, (linalg_ops.QuantizedMatmulOp, linalg_ops.MatmulOp)):
            found.append(op)
    return found


def matmul_lhs_rhs(mm):
    """(lhs, rhs) SSA values of a linalg matmul-family op (inputs[0], inputs[1])."""
    return mm.inputs[0], mm.inputs[1]
