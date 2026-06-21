"""Build the ``merlin_iface`` module from the parsed program, and lower it to the
``gemmini`` target dialect.

``build_iface_module`` materializes the parsed :class:`~iface_ir.Program` as a real
xDSL module of ``merlin_iface`` ops wrapped in a ``func.func`` (so it round-trips and
``verify()``s). ``lower_to_gemmini`` is the conversion pass: it walks the verified
interface module and constructs the corresponding ``gemmini`` target module (also
verified). This is the genuine dialect + lowering-pass path the arm asks for.
"""
from __future__ import annotations

from xdsl.dialects.builtin import (ArrayAttr, FloatAttr, Float32Type, IntegerAttr,
                                    IntegerType, ModuleOp, StringAttr, TensorType,
                                    FunctionType, i32)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

from . import dialects as D
from . import iface_ir as IR


def dtype_to_type(dt: str):
    return IntegerType(int(dt[1:]))


def _tt(dtype: str, shape):
    return TensorType(dtype_to_type(dtype), list(shape))


def _arr(strs):
    return ArrayAttr([StringAttr(s) for s in strs])


def _iarr(ints):
    return ArrayAttr([IntegerAttr(int(v), i32) for v in ints])


def _f32(v):
    return FloatAttr(float(v), Float32Type())


def matmul_out_shape(prog: IR.Program, mm: IR.Matmul):
    """(M, N) for a matmul: M = lhs rows, N = resident weight cols."""
    lhs_t = prog.tensors[mm.lhs]
    w_name = prog.pack_src[mm.rhs]
    w_t = prog.tensors[w_name]
    return int(lhs_t.shape[0]), int(w_t.shape[1])


def conv_out_shape(prog: IR.Program, cv: IR.Conv2d):
    """(n_patches, out_channels) of the im2col matmul.

    n_patches = batch * out_h * out_w; out_channels from the weight cols.
    """
    ifm = prog.tensors[cv.ifm]
    n, h, w, _ci = ifm.shape  # NHWC
    kh, kw = int(cv.kernel[0]), int(cv.kernel[1])
    sh, sw = int(cv.stride[0]), int(cv.stride[1])
    pt, pl, pb, pr = (list(cv.padding) + [0, 0, 0, 0])[:4]
    oh = (h + pt + pb - kh) // sh + 1
    ow = (w + pl + pr - kw) // sw + 1
    n_patches = n * oh * ow
    w_t = prog.tensors[cv.rhs] if cv.rhs in prog.tensors else prog.tensors[prog.pack_src[cv.rhs]]
    co = int(w_t.shape[1])
    return int(n_patches), int(co)


def build_iface_module(prog: IR.Program) -> ModuleOp:
    blk = Block()
    vals: dict[str, object] = {}
    ops = []

    for op in prog.ops:
        if isinstance(op, IR.Pack):
            pass  # handled when we hit it below
    # Emit leaf tensors first (declaration order), then ops in program order.
    for name, t in prog.tensors.items():
        o = D.IfaceTensorOp(
            result_types=[_tt(t.dtype, t.shape)],
            properties={"tname": StringAttr(name), "role": StringAttr(t.role)})
        ops.append(o)
        vals[name] = o.res

    for op in prog.ops:
        if isinstance(op, IR.Pack):
            src = vals[op.src]
            o = D.IfaceResidentPackOp(
                operands=[src],
                result_types=[D.IfaceResidentType(src.type)],
                properties={"layout": StringAttr(op.layout)})
            ops.append(o)
            vals[op.dst] = o.res
        elif isinstance(op, IR.Matmul):
            m, n = matmul_out_shape(prog, op)
            o = D.IfaceMatmulOp(
                operands=[vals[op.lhs], vals[op.rhs]],
                result_types=[D.IfaceAccType(TensorType(i32, [m, n]))])
            ops.append(o)
            vals[op.dst] = o.acc
        elif isinstance(op, IR.Commit):
            acc_t = vals[op.src].type.element
            m, n = acc_t.get_shape()
            props = {"tname": StringAttr(op.dst), "epilogue": _arr(op.epilogue),
                     "output_dtype": StringAttr(op.output_dtype)}
            if op.acc_scale is not None:
                props["acc_scale"] = _f32(op.acc_scale)
            o = D.IfaceCommitOp(
                operands=[vals[op.src]],
                result_types=[_tt(op.output_dtype, [m, n])],
                properties=props)
            ops.append(o)
            vals[op.dst] = o.res
        elif isinstance(op, IR.Movement):
            src = vals[op.src]
            o = D.IfaceMovementOp(
                operands=[src], result_types=[src.type],
                properties={"tname": StringAttr(op.dst)})
            ops.append(o)
            vals[op.dst] = o.res
        elif isinstance(op, IR.Conv2d):
            np_, co = conv_out_shape(prog, op)
            props = {"tname": StringAttr(op.dst), "kernel": _iarr(op.kernel),
                     "stride": _iarr(op.stride), "padding": _iarr(op.padding),
                     "dilation": _iarr(op.dilation), "layout": StringAttr(op.layout),
                     "epilogue": _arr(op.epilogue),
                     "output_dtype": StringAttr(op.output_dtype)}
            if op.acc_scale is not None:
                props["acc_scale"] = _f32(op.acc_scale)
            o = D.IfaceConv2dOp(
                operands=[vals[op.ifm], vals[op.rhs]],
                result_types=[_tt(op.output_dtype, [np_, co])],
                properties=props)
            ops.append(o)
            vals[op.dst] = o.res
        elif isinstance(op, IR.Evict):
            o = D.IfaceEvictOp(operands=[vals[op.handle]])
            ops.append(o)

    blk.add_ops(ops + [ReturnOp()])
    fn = FuncOp("kernel", FunctionType.from_lists([], []), Region([blk]))
    return ModuleOp([fn])


def _func_block(module: ModuleOp) -> Block:
    fn = next(o for o in module.body.block.ops if isinstance(o, FuncOp))
    return fn.body.block


def lower_to_gemmini(iface_module: ModuleOp) -> ModuleOp:
    """Conversion pass: merlin_iface module -> gemmini target module."""
    src_blk = _func_block(iface_module)
    blk = Block()
    vals: dict[object, object] = {}  # old SSAValue -> new SSAValue
    ops = []

    for op in src_blk.ops:
        if isinstance(op, D.IfaceTensorOp):
            o = D.GemminiTensorOp(result_types=[op.res.type],
                                  properties={"tname": op.tname, "role": op.role})
            ops.append(o); vals[op.res] = o.res
        elif isinstance(op, D.IfaceResidentPackOp):
            src = vals[op.src]
            o = D.GemminiPackOp(operands=[src],
                                result_types=[D.GemminiResidentTensorType(src.type)],
                                properties={"layout": op.layout})
            ops.append(o); vals[op.res] = o.res
        elif isinstance(op, D.IfaceMatmulOp):
            o = D.GemminiMatmulOp(
                operands=[vals[op.lhs], vals[op.rhs]],
                result_types=[D.GemminiAccumulatorType(op.acc.type.element)])
            ops.append(o); vals[op.acc] = o.acc
        elif isinstance(op, D.IfaceCommitOp):
            props = {"tname": op.tname, "epilogue": op.epilogue,
                     "output_dtype": op.output_dtype}
            if op.acc_scale is not None:
                props["acc_scale"] = op.acc_scale
            o = D.GemminiCommitOp(operands=[vals[op.acc]],
                                  result_types=[op.res.type], properties=props)
            ops.append(o); vals[op.res] = o.res
        elif isinstance(op, D.IfaceMovementOp):
            src = vals[op.src]
            o = D.GemminiMovementOp(operands=[src], result_types=[op.res.type],
                                    properties={"tname": op.tname})
            ops.append(o); vals[op.res] = o.res
        elif isinstance(op, D.IfaceConv2dOp):
            props = {"tname": op.tname, "kernel": op.kernel, "stride": op.stride,
                     "padding": op.padding, "dilation": op.dilation,
                     "layout": op.layout, "epilogue": op.epilogue,
                     "output_dtype": op.output_dtype}
            if op.acc_scale is not None:
                props["acc_scale"] = op.acc_scale
            o = D.GemminiConv2dOp(operands=[vals[op.ifm], vals[op.rhs]],
                                  result_types=[op.res.type], properties=props)
            ops.append(o); vals[op.res] = o.res
        elif isinstance(op, D.IfaceEvictOp):
            o = D.GemminiReleaseOp(operands=[vals[op.handle]])
            ops.append(o)

    blk.add_ops(ops + [ReturnOp()])
    fn = FuncOp("gemmini_kernel", FunctionType.from_lists([], []), Region([blk]))
    return ModuleOp([fn])
