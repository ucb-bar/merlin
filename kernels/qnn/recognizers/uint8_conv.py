"""Bindings recognizer: fixture-form uint8 Conv2D (`merlin.qnn.conv2d_uint8`).

Bridges any front-end that wraps a quantized Conv2D as a custom
`merlin.qnn.conv2d_uint8` op carrying per-tensor q-params on the `func.func`
attribute dict.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    ParsedUint8Conv,
    lower_uint8_conv,
)
from .base import (  # noqa: E402
    elem_dtype_of,
    find_func,
    find_named_op,
    find_tensor_constants,
    func_arg_values,
    func_name,
    parse_dense_2d_attr,
    parse_qparams_attr,
    shape_of,
    splat_constant_value,
)

NAME = "uint8_conv2d_fixture"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    custom_op = find_named_op(func, "merlin.qnn.conv2d_uint8")
    if custom_op is None:
        return None

    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_val = args[0]
    in_dtype = elem_dtype_of(in_val)
    if in_dtype not in ("ui8", "i8"):
        return None
    in_shape = shape_of(in_val)
    if len(in_shape) != 4 or in_shape[0] != 1:
        return None

    in_qp = parse_qparams_attr(func, "input_qparams")
    w_qp = parse_qparams_attr(func, "weight_qparams")
    b_qp = parse_qparams_attr(func, "bias_qparams")
    out_qp = parse_qparams_attr(func, "output_qparams")
    if not (in_qp and w_qp and b_qp and out_qp):
        return None

    # Weight: first 4D u8/i8 constant.
    weight_const = None
    for cand_dtype in (in_dtype, "ui8", "i8"):
        for cand in find_tensor_constants(func, rank=4, dtype=cand_dtype):
            weight_const = cand
            break
        if weight_const is not None:
            break
    if weight_const is None:
        return None
    weight_shape = shape_of(weight_const.results[0])
    weight_value = splat_constant_value(weight_const)
    if weight_value is None:
        return None

    # Bias: first 1D i32 constant.
    bias_consts = find_tensor_constants(func, rank=1, dtype="i32")
    if not bias_consts:
        return None
    bias_const = bias_consts[0]
    bias_shape = shape_of(bias_const.results[0])
    bias_value = splat_constant_value(bias_const)
    if bias_value is None:
        return None

    strides = parse_dense_2d_attr(custom_op, "strides")
    dilation = parse_dense_2d_attr(custom_op, "dilations")
    if strides is None or dilation is None:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    out_shape = shape_of(out_val)

    parsed = ParsedUint8Conv(
        func_name=func_name(func),
        input_shape=tuple(in_shape),
        weight_shape=tuple(weight_shape),
        weight_constant_value=int(weight_value),
        bias_shape=tuple(bias_shape),
        bias_constant_value=int(bias_value),
        strides=strides,
        dilation=dilation,
        output_shape=tuple(out_shape),
        input_qp=in_qp,
        weight_qp=w_qp,
        bias_qp=b_qp,
        output_qp=out_qp,
    )
    return lower_uint8_conv(parsed)
