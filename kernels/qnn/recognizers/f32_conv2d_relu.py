"""Bindings recognizer: f32 NHWC×HWCF Conv2D + bias-add + ReLU.

Mirrors `qnn_emit.parse_conv2d_relu_mlir` + `qnn_emit.lower_conv2d_relu`.
Reuses the existing `ParsedConv2dRelu` + `lower_conv2d_relu` so the v2
output is byte-identical to v1 for the same input.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    ParsedConv2dRelu,
    lower_conv2d_relu,
)
from .base import (  # noqa: E402
    elem_dtype_of,
    find_func,
    find_named_op,
    find_tensor_constants,
    func_arg_values,
    func_name,
    has_op_in_func,
    parse_dense_2d_attr,
    shape_of,
    splat_constant_value,
    walk_inner_ops,
)

NAME = "f32_conv2d_nhwc_hwcf_bias_relu"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
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

    conv_op = find_named_op(func, "linalg.conv_2d_nhwc_hwcf")
    if conv_op is None:
        return None

    # Weight = first 4D fp32 tensor constant; bias = first 1D fp32 tensor constant.
    weights = find_tensor_constants(func, rank=4, dtype="f32")
    biases = find_tensor_constants(func, rank=1, dtype="f32")
    if not weights or not biases:
        return None
    weight_const = weights[0]
    bias_const = biases[0]

    weight_shape = shape_of(weight_const.results[0])
    bias_shape = shape_of(bias_const.results[0])
    if bias_shape[0] != weight_shape[3]:
        return None
    weight_value = splat_constant_value(weight_const)
    bias_value = splat_constant_value(bias_const)
    if weight_value is None or bias_value is None:
        return None

    # Defensive: bias-add and ReLU bodies must appear (legacy contract).
    if not has_op_in_func(func, "linalg.generic"):
        return None
    has_addf = False
    has_maximumf = False
    for op in walk_inner_ops(func):
        if op.operation.name != "linalg.generic":
            continue
        for region in op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    if inner.name == "arith.addf":
                        has_addf = True
                    elif inner.name == "arith.maximumf":
                        has_maximumf = True
    if not (has_addf and has_maximumf):
        return None

    strides = parse_dense_2d_attr(conv_op, "strides")
    dilation = parse_dense_2d_attr(conv_op, "dilations")
    if strides is None or dilation is None:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    if elem_dtype_of(out_val) != "f32":
        return None
    out_shape = shape_of(out_val)

    parsed = ParsedConv2dRelu(
        func_name=func_name(func),
        input_shape=tuple(in_shape),
        weight_shape=tuple(weight_shape),
        weight_constant_value=float(weight_value),
        bias_shape=tuple(bias_shape),
        bias_constant_value=float(bias_value),
        strides=strides,
        dilation=dilation,
        output_shape=tuple(out_shape),
    )
    return lower_conv2d_relu(parsed, compute_dtype=fp_dtype)
