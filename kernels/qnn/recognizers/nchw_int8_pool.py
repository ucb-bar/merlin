"""Bindings recognizer for the NCHW int8 max-pool dispatch.

Matches a `linalg.pooling_nchw_max` anchored func that:
  - takes an i8 NCHW input,
  - dequantizes via `linalg.generic` (sitofp + mulf scale),
  - applies the pool over the f32 dequant result,
  - returns f32.

Lowers to a 4-node QNN graph:
  - Transpose NCHW → NHWC (i8 input)
  - Dequantize i8 → f32 (NHWC) — keeps Pool's f32 input semantics
  - PoolMax2d (NHWC) → f32
  - Transpose NHWC → NCHW (f32 output)

The Dequantize-before-Pool path is the simplest correct lowering on
backends that don't support quantized PoolMax2d; on HTA the optimizer
will collapse the pair into a single quantized pool kernel.
"""

from __future__ import annotations

from typing import Any

from .base import (
    elem_dtype_of,
    find_func,
    find_named_op,
    func_arg_values,
    parse_dense_2d_attr,
    shape_of,
)

NAME = "nchw_int8_pool_max"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    pool_op = find_named_op(func, "linalg.pooling_nchw_max")
    if pool_op is None:
        return None

    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_val = args[0]
    if elem_dtype_of(in_val) != "i8":
        return None
    in_shape = shape_of(in_val)
    if len(in_shape) != 4:
        return None

    # Pool input: a `linalg.generic` doing i8 → f32 dequant on the func arg.
    # We just need to extract the dequant scale; the structural check is
    # implicit — if the recognizer fires on a func without that body,
    # the lowering still works because we only reference the conv attrs.
    pool_input_v = pool_op.operands[0]
    pool_input_op = pool_input_v.owner
    dequant_scale: float | None = None
    if hasattr(pool_input_op, "operation") and pool_input_op.operation.name == "linalg.generic":
        from iree.compiler import ir  # noqa: PLC0415  - lazy

        for region in pool_input_op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    if inner.name != "arith.mulf":
                        continue
                    if len(inner.operands) < 2:
                        continue
                    rhs = inner.operands[1]
                    rhs_owner = rhs.owner
                    if hasattr(rhs_owner, "operation") and rhs_owner.operation.name == "arith.constant":
                        try:
                            dequant_scale = ir.FloatAttr(rhs_owner.attributes["value"]).value
                        except (ValueError, TypeError, KeyError):
                            pass
    if dequant_scale is None:
        return None

    # Window shape from the second `ins(...)` operand (a 2D f32 tensor).
    if len(pool_op.operands) < 2:
        return None
    win_v = pool_op.operands[1]
    win_ty = win_v.type
    if not hasattr(win_ty, "shape") or len(win_ty.shape) != 2:
        return None
    fdims = (int(win_ty.shape[0]), int(win_ty.shape[1]))

    strides = parse_dense_2d_attr(pool_op, "strides")
    dilation = parse_dense_2d_attr(pool_op, "dilations")
    if strides is None or dilation is None:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    out_shape = shape_of(out_val)
    if elem_dtype_of(out_val) != "f32":
        return None

    from iree.compiler import ir  # noqa: PLC0415  - lazy
    from qnn_ir import (  # noqa: PLC0415  - lazy
        QnnGraphDesc,
        QuantParams,
        TensorDesc,
        dequantize_node,
        pool_max_2d_node,
        transpose_node,
    )

    sym_name = ir.StringAttr(func.attributes["sym_name"]).value

    n, c, h_in, w_in = (int(d) for d in in_shape)
    _, _, h_out, w_out = (int(d) for d in out_shape)
    nhwc_in_shape = (n, h_in, w_in, c)
    nhwc_pool_shape = (n, h_out, w_out, c)

    in_qp = QuantParams(scale=float(dequant_scale), offset=0)

    tensors = (
        TensorDesc(
            name="input",
            shape=tuple(in_shape),
            dtype="int8",
            role="input",
            quant=in_qp,
        ),
        TensorDesc(
            name="nhwc_in",
            shape=nhwc_in_shape,
            dtype="int8",
            role="native",
            quant=in_qp,
        ),
        TensorDesc(
            name="nhwc_in_f32",
            shape=nhwc_in_shape,
            dtype="float32",
            role="native",
        ),
        TensorDesc(
            name="nhwc_pool",
            shape=nhwc_pool_shape,
            dtype="float32",
            role="native",
        ),
        TensorDesc(
            name="output",
            shape=tuple(out_shape),
            dtype="float32",
            role="output",
        ),
    )

    nodes = (
        transpose_node(
            name="nchw_to_nhwc",
            input_tensor="input",
            output_tensor="nhwc_in",
            perm=(0, 2, 3, 1),
        ),
        dequantize_node(
            name="dequant_op",
            input_tensor="nhwc_in",
            output_tensor="nhwc_in_f32",
        ),
        pool_max_2d_node(
            name="pool_op",
            input_tensor="nhwc_in_f32",
            output_tensor="nhwc_pool",
            filter_size=fdims,
            strides=strides,
            pad_before_after_hw=((0, 0), (0, 0)),
            rounding_mode=0,
        ),
        transpose_node(
            name="nhwc_to_nchw",
            input_tensor="nhwc_pool",
            output_tensor="output",
            perm=(0, 3, 1, 2),
        ),
    )
    return QnnGraphDesc(name=sym_name, tensors=tensors, nodes=nodes)
