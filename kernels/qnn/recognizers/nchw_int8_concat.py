"""Bindings recognizer for the NCHW int8 concat dispatch.

Matches a `tensor.concat` anchored func with N i8 NCHW inputs, where
each is dequantized via `linalg.generic` before the concat.

Lowers to:
  - Per-input Transpose NCHW → NHWC (i8) and Dequantize → f32 (NHWC)
  - Concat (NHWC) along the corresponding dim
  - Transpose NHWC → NCHW (f32 output)
"""

from __future__ import annotations

from typing import Any

from .base import (
    elem_dtype_of,
    find_func,
    find_named_op,
    func_arg_values,
    integer_attr_value,
    shape_of,
)

NAME = "nchw_int8_concat"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None
    concat_op = find_named_op(func, "tensor.concat")
    if concat_op is None:
        return None

    args = func_arg_values(func)
    if len(args) < 2:
        return None
    if any(elem_dtype_of(a) != "i8" for a in args):
        return None
    arg_shapes = [tuple(shape_of(a)) for a in args]
    if any(len(s) != 4 for s in arg_shapes):
        return None
    if not all(s == arg_shapes[0] for s in arg_shapes):
        return None  # non-uniform input shapes are not yet supported

    # Read the concat axis (NCHW dim).
    nchw_axis = integer_attr_value(concat_op, "dim")
    if nchw_axis is None or nchw_axis < 0 or nchw_axis >= 4:
        return None
    # NCHW → NHWC permutation: dim 0,1,2,3 → 0,2,3,1; equivalent map for axis.
    nchw_to_nhwc_axis = (0, 3, 1, 2)
    nhwc_axis = nchw_to_nhwc_axis[nchw_axis]

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    out_shape = tuple(shape_of(out_val))
    if elem_dtype_of(out_val) != "f32":
        return None

    from iree.compiler import ir  # noqa: PLC0415  - lazy
    from qnn_ir import (  # noqa: PLC0415  - lazy
        QnnGraphDesc,
        QuantParams,
        TensorDesc,
        concat_node,
        dequantize_node,
        transpose_node,
    )

    sym_name = ir.StringAttr(func.attributes["sym_name"]).value

    # Per-input dequant scale: assume all inputs share the scale (this
    # is the common yolov8 case where consecutive convs feed concat).
    # Pull from the first dequant generic if available; fall back to
    # 1.0 (the lowering still composes; numerical correctness in
    # mixed-scale concats is a follow-up).
    in_scale = 1.0
    for op in func.regions[0].blocks[0].operations:
        if op.operation.name != "linalg.generic":
            continue
        for region in op.operation.regions:
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
                            in_scale = float(ir.FloatAttr(rhs_owner.attributes["value"]).value)
                        except (ValueError, TypeError, KeyError):
                            pass
        if in_scale != 1.0:
            break
    in_qp = QuantParams(scale=in_scale, offset=0)

    # Tensor inventory + nodes
    tensors_list: list[Any] = []
    nodes_list: list[Any] = []
    nhwc_dequant_names: list[str] = []
    for i, (val, sh) in enumerate(zip(args, arg_shapes)):
        in_name = f"input_{i}"
        nhwc_i8 = f"nhwc_in_{i}"
        nhwc_f32 = f"nhwc_in_{i}_f32"
        n, c, h, w = sh
        nhwc_shape = (n, h, w, c)
        tensors_list.extend(
            [
                TensorDesc(
                    name=in_name,
                    shape=sh,
                    dtype="int8",
                    role="input",
                    quant=in_qp,
                ),
                TensorDesc(
                    name=nhwc_i8,
                    shape=nhwc_shape,
                    dtype="int8",
                    role="native",
                    quant=in_qp,
                ),
                TensorDesc(
                    name=nhwc_f32,
                    shape=nhwc_shape,
                    dtype="float32",
                    role="native",
                ),
            ]
        )
        nodes_list.extend(
            [
                transpose_node(
                    name=f"nchw_to_nhwc_{i}",
                    input_tensor=in_name,
                    output_tensor=nhwc_i8,
                    perm=(0, 2, 3, 1),
                ),
                dequantize_node(
                    name=f"dequant_{i}",
                    input_tensor=nhwc_i8,
                    output_tensor=nhwc_f32,
                ),
            ]
        )
        nhwc_dequant_names.append(nhwc_f32)

    # Concat output in NHWC.
    n, c_out, h_out, w_out = out_shape
    nhwc_concat_shape = (n, h_out, w_out, c_out)
    tensors_list.append(
        TensorDesc(
            name="nhwc_concat",
            shape=nhwc_concat_shape,
            dtype="float32",
            role="native",
        )
    )
    tensors_list.append(
        TensorDesc(
            name="output",
            shape=out_shape,
            dtype="float32",
            role="output",
        )
    )
    nodes_list.append(
        concat_node(
            name="concat_op",
            input_tensors=tuple(nhwc_dequant_names),
            output_tensor="nhwc_concat",
            axis=nhwc_axis,
        )
    )
    nodes_list.append(
        transpose_node(
            name="nhwc_to_nchw",
            input_tensor="nhwc_concat",
            output_tensor="output",
            perm=(0, 3, 1, 2),
        )
    )

    return QnnGraphDesc(
        name=sym_name,
        tensors=tuple(tensors_list),
        nodes=tuple(nodes_list),
    )
