"""Lowering pass: ``merlin_iface`` module  ->  ``gemmini`` target-dialect module.

A dialect-conversion pass that walks the verified interface module in order and emits the
Gemmini target dialect. The two interesting rules:

* ``merlin_iface.conv2d`` -> a derived im2col leaf (``gemmini.tensor``) + ``gemmini.matmul``
  (carrying the im2col recipe) + ``gemmini.commit``. The im2col activation is materialized by
  the runner harness/reference/simulator from the recipe, so the device path is a plain matmul.
* everything else maps 1:1 to its ``gemmini.*`` counterpart.
"""
from __future__ import annotations

from xdsl.dialects.builtin import (DictionaryAttr, IntegerAttr, ModuleOp, StringAttr,
                                    TensorType, i32)
from xdsl.ir import Block, Region, SSAValue

from . import dialects as D


def _conv_out(h, w, kh, kw, stride, padding, dilation):
    sh, sw = stride
    pt, pl, pb, pr = padding
    dh, dw = dilation
    ho = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    return ho, wo


def lower_to_gemmini(iface: ModuleOp) -> ModuleOp:
    blk = Block()
    ops = []
    vmap: dict[SSAValue, SSAValue] = {}

    def add(op):
        ops.append(op)
        return op

    for op in iface.body.block.ops:
        if isinstance(op, D.IfaceTensorOp):
            g = add(D.GTensorOp(properties={"sym": op.sym, "role": op.role},
                                result_types=[op.res.type]))
            vmap[op.res] = g.res
        elif isinstance(op, D.IfaceResidentPackOp):
            g = add(D.GPackOp(operands=[vmap[op.src]],
                              properties={"sym": op.sym, "layout": op.layout},
                              result_types=[D.GResidentType()]))
            vmap[op.res] = g.res
        elif isinstance(op, D.IfaceMatmulOp):
            g = add(D.GMatmulOp(operands=[vmap[op.lhs], vmap[op.rhs]],
                                properties={"sym": op.sym},
                                result_types=[D.GAccType()]))
            vmap[op.res] = g.res
        elif isinstance(op, D.IfaceCommitOp):
            props = {"sym": op.sym, "epilogue": op.epilogue, "output_dtype": op.output_dtype}
            if op.acc_scale is not None:
                props["acc_scale"] = op.acc_scale
            g = add(D.GCommitOp(operands=[vmap[op.acc]], properties=props,
                                result_types=[op.res.type]))
            vmap[op.res] = g.res
        elif isinstance(op, D.IfaceMoveOp):
            g = add(D.GMoveOp(operands=[vmap[op.src]], properties={"sym": op.sym},
                              result_types=[op.res.type]))
            vmap[op.res] = g.res
        elif isinstance(op, D.IfaceEvictOp):
            add(D.GReleaseOp(operands=[vmap[op.handle]]))
        elif isinstance(op, D.IfaceConvOp):
            ifm_t = op.ifm.type
            n, h, w, c = (int(d) for d in ifm_t.get_shape())
            kh, kw, ci, co = (int(x.value.data) for x in op.kernel)
            stride = [int(x.value.data) for x in op.stride]
            padding = [int(x.value.data) for x in op.padding]
            dilation = [int(x.value.data) for x in op.dilation]
            ho, wo = _conv_out(h, w, kh, kw, stride, padding, dilation)
            m, k = n * ho * wo, kh * kw * ci
            im2col_name = op.sym.data + "_im2col"
            # derived im2col activation leaf (overridden from IFM via the recipe)
            im_t = add(D.GTensorOp(properties={
                "sym": StringAttr(im2col_name), "role": StringAttr("input")},
                result_types=[TensorType(op.ifm.type.get_element_type(), [m, k])]))
            recipe = DictionaryAttr({
                "source": op.ifm.owner.sym, "target": StringAttr(im2col_name),
                "kh": IntegerAttr(kh, 64), "kw": IntegerAttr(kw, 64), "ci": IntegerAttr(ci, 64),
                "stride": op.stride, "padding": op.padding, "dilation": op.dilation,
                "layout": op.layout})
            mm = add(D.GMatmulOp(operands=[im_t.res, vmap[op.rhs]],
                                 properties={"sym": StringAttr(op.sym.data + "_acc"),
                                             "im2col": recipe},
                                 result_types=[D.GAccType()]))
            props = {"sym": op.sym, "epilogue": op.epilogue, "output_dtype": op.output_dtype}
            if op.acc_scale is not None:
                props["acc_scale"] = op.acc_scale
            g = add(D.GCommitOp(operands=[mm.res], properties=props,
                                result_types=[op.res.type]))
            vmap[op.res] = g.res
        else:
            raise TypeError(f"cannot lower {op.name}")

    blk.add_ops(ops)
    out = ModuleOp(Region([blk]))
    out.verify()
    return out
