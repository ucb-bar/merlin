"""Bindings recognizer: f32 elementwise binary (Add / Sub / Mul / Div)."""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    _BINARY_BODY_TO_QNN_OP,
    ParsedElementwiseBinary,
    lower_elementwise_binary,
)
from .base import (  # noqa: E402
    elem_dtype_of,
    find_func,
    find_named_op,
    func_arg_values,
    func_name,
    has_any_op_in_func,
    linalg_generic_body_op_names,
    shape_of,
    walk_inner_ops,
)

NAME = "f32_elementwise_binary"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    # Reject if a conv or pooling is present — those are different patterns.
    if has_any_op_in_func(
        func, ("linalg.conv_2d_nhwc_hwcf", "linalg.pooling_nhwc_max", "linalg.depthwise_conv_2d_nhwc_hwc")
    ):
        return None

    args = func_arg_values(func)
    if len(args) != 2:
        return None
    a_val, b_val = args
    if elem_dtype_of(a_val) != "f32" or elem_dtype_of(b_val) != "f32":
        return None
    a_shape = shape_of(a_val)
    b_shape = shape_of(b_val)
    if a_shape != b_shape:
        return None  # broadcasting not yet supported

    # Find the linalg.generic and inspect its body for exactly one of the
    # supported arith binary ops.
    found_qnn_op: str | None = None
    for op in walk_inner_ops(func):
        if op.operation.name != "linalg.generic":
            continue
        body_ops = linalg_generic_body_op_names(op)
        for arith_name, qnn_name in _BINARY_BODY_TO_QNN_OP.items():
            if arith_name in body_ops:
                if found_qnn_op is not None and found_qnn_op != qnn_name:
                    return None  # multiple distinct binary ops
                found_qnn_op = qnn_name
    if found_qnn_op is None:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    if elem_dtype_of(out_val) != "f32":
        return None
    out_shape = shape_of(out_val)
    if out_shape != a_shape:
        return None

    parsed = ParsedElementwiseBinary(
        func_name=func_name(func),
        op_type=found_qnn_op,
        input_shape=tuple(a_shape),
        output_shape=tuple(out_shape),
    )
    return lower_elementwise_binary(parsed)
