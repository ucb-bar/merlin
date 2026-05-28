"""Bindings recognizer: f32 NHWC max-pool (`linalg.pooling_nhwc_max`)."""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    ParsedMaxPool,
    lower_maxpool,
)
from .base import (  # noqa: E402
    elem_dtype_of,
    find_func,
    find_named_op,
    func_arg_values,
    func_name,
    parse_dense_2d_attr,
    shape_of,
)

NAME = "f32_maxpool_nhwc"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    pool_op = find_named_op(func, "linalg.pooling_nhwc_max")
    if pool_op is None:
        return None

    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_val = args[0]
    if elem_dtype_of(in_val) != "f32":
        return None
    in_shape = shape_of(in_val)
    if len(in_shape) != 4:
        return None

    # Filter operand is the second `ins` operand: a 2D fp32 tensor (window
    # shape; not actually read at runtime).
    if len(pool_op.operands) < 2:
        return None
    win_val = pool_op.operands[1]
    win_ty = win_val.type
    if not hasattr(win_ty, "shape") or not hasattr(win_ty, "element_type"):
        return None
    if str(win_ty.element_type) != "f32" or len(win_ty.shape) != 2:
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

    parsed = ParsedMaxPool(
        func_name=func_name(func),
        input_shape=tuple(in_shape),
        output_shape=tuple(out_shape),
        filter_size=fdims,
        strides=strides,
        dilation=dilation,
    )
    return lower_maxpool(parsed)
