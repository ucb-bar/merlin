"""Bindings recognizer: f32 NHWC×HWC depthwise Conv2D, no bias.

Mirrors `qnn_emit.parse_depthwise_conv_mlir`/`lower_depthwise_conv`.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    ParsedDepthwiseConv,
    lower_depthwise_conv,
)
from .base import (  # noqa: E402
    elem_dtype_of,
    find_func,
    find_named_op,
    find_tensor_constants,
    func_arg_values,
    func_name,
    parse_dense_2d_attr,
    shape_of,
    splat_constant_value,
)

NAME = "f32_depthwise_conv_nhwc_hwc"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    dw_op = find_named_op(func, "linalg.depthwise_conv_2d_nhwc_hwc")
    if dw_op is None:
        return None

    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_val = args[0]
    if elem_dtype_of(in_val) != "f32":
        return None
    in_shape = shape_of(in_val)
    if len(in_shape) != 4 or in_shape[0] != 1:
        return None

    weights = find_tensor_constants(func, rank=3, dtype="f32")
    if not weights:
        return None
    weight_const = weights[0]
    weight_shape = shape_of(weight_const.results[0])
    weight_value = splat_constant_value(weight_const)
    if weight_value is None:
        return None

    strides = parse_dense_2d_attr(dw_op, "strides")
    dilation = parse_dense_2d_attr(dw_op, "dilations")
    if strides is None or dilation is None:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    out_shape = shape_of(out_val)
    if elem_dtype_of(out_val) != "f32":
        return None

    parsed = ParsedDepthwiseConv(
        func_name=func_name(func),
        input_shape=tuple(in_shape),
        weight_shape=tuple(weight_shape),
        weight_constant_value=float(weight_value),
        output_shape=tuple(out_shape),
        strides=strides,
        dilation=dilation,
    )
    return lower_depthwise_conv(parsed)
