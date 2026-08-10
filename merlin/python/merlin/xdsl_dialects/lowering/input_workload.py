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


def build_matmul_chain(dims=(8, 16, 12, 6), elem="f32"):
    """func @chain(A, W_1..W_{L}) -> mxn_L : a feed-forward stack of L matmuls where each
    layer's output is the next layer's LHS (the whole-model backbone). ``dims`` = the L+1 shared
    dimensions [m, k_1, k_2, ..., k_L]; ``elem`` selects f32 (plain matmul) or i8/i32 semantics.
    Returns (module, [weight arrays are the caller's to inject by name])."""
    if not HAS_XDSL:
        return None
    from xdsl.ir import Block, Region
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, f16, f32, f64
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops

    et = {"f16": f16, "f32": f32, "f64": f64}[elem]
    m = dims[0]
    ks = list(dims[1:])
    if len(ks) < 1:
        raise ValueError("a chain needs at least one matmul (dims = [m, k, n, ...])")
    a_t = TensorType(et, [m, ks[0]])
    w_ts = [TensorType(et, [ks[j], ks[j + 1]]) for j in range(len(ks) - 1)]
    blk = Block(arg_types=[a_t] + w_ts)
    cur = blk.args[0]
    ws = list(blk.args[1:])
    ops = []
    rows = m
    for j, w in enumerate(ws):
        out_t = TensorType(et, [rows, ks[j + 1]])
        init = tensor_d.EmptyOp((), out_t)
        mm = linalg_ops.MatmulOp(inputs=(cur, w), outputs=(init.tensor,), res=(out_t,))
        ops += [init, mm]
        cur = mm.results[0]
    ops.append(ReturnOp(cur))
    blk.add_ops(ops)
    fn = FuncOp("chain", FunctionType.from_lists([a_t] + w_ts, [cur.type]), Region([blk]))
    return ModuleOp([fn])


def build_vector_block(m: int = 8, k: int = 16, elem: str = "f32",
                       combine: str = "add", relu: bool = True):
    """func @vecblock(A: m×k, W1: k×k, W2: k×k) -> m×k : ``combine(relu(A@W1), A@W2)``.

    Exercises the non-matmul vector path alongside matmuls: two ``linalg.matmul`` layers, an
    optional relu (``linalg.max`` against a zero fill — the standard relu idiom, which lowers to an
    identity vector_map + relu activation), and an elementwise ``linalg.add``/``linalg.mul`` (a
    residual add or a gating multiply). ``combine`` ∈ {"add", "mul"}."""
    if not HAS_XDSL:
        return None
    from xdsl.ir import Block, Region
    from xdsl.dialects import arith
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import (FloatAttr, FunctionType, ModuleOp, TensorType,
                                       f16, f32, f64)
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops

    et = {"f16": f16, "f32": f32, "f64": f64}[elem]
    combine_op = {"add": linalg_ops.AddOp, "mul": linalg_ops.MulOp}[combine]
    a_t = TensorType(et, [m, k])
    w_t = TensorType(et, [k, k])
    o_t = TensorType(et, [m, k])
    blk = Block(arg_types=[a_t, w_t, w_t])
    A, W1, W2 = blk.args
    ops = []

    e1 = tensor_d.EmptyOp((), o_t)
    mm1 = linalg_ops.MatmulOp(inputs=(A, W1), outputs=(e1.tensor,), res=(o_t,))
    ops += [e1, mm1]
    cur = mm1.results[0]

    if relu:
        zc = arith.ConstantOp(FloatAttr(0.0, et))
        ze = tensor_d.EmptyOp((), o_t)
        zf = linalg_ops.FillOp(inputs=(zc.result,), outputs=(ze.tensor,), res=(o_t,))
        re = tensor_d.EmptyOp((), o_t)
        rl = linalg_ops.MaxOp(inputs=(cur, zf.results[0]), outputs=(re.tensor,), res=(o_t,))
        ops += [zc, ze, zf, re, rl]
        cur = rl.results[0]

    e2 = tensor_d.EmptyOp((), o_t)
    mm2 = linalg_ops.MatmulOp(inputs=(A, W2), outputs=(e2.tensor,), res=(o_t,))
    ops += [e2, mm2]

    ce = tensor_d.EmptyOp((), o_t)
    cmb = combine_op(inputs=(cur, mm2.results[0]), outputs=(ce.tensor,), res=(o_t,))
    ops += [ce, cmb]

    ops.append(ReturnOp(cmb.results[0]))
    blk.add_ops(ops)
    fn = FuncOp("vecblock", FunctionType.from_lists([a_t, w_t, w_t], [o_t]), Region([blk]))
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
