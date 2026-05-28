"""Bindings recognizer: f32 elementwise unary (Relu / Relu6 / Sigmoid /
Tanh).

Body classification (matches v1):
  Relu     :  arith.maximumf %a, %zero  (no minimumf)
  Relu6    :  arith.maximumf + arith.minimumf
  Sigmoid  :  arith.negf + math.exp + arith.addf 1 + arith.divf
  Tanh     :  math.tanh, no math.exp
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    ParsedElementwiseUnary,
    lower_elementwise_unary,
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

NAME = "f32_elementwise_unary"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    if has_any_op_in_func(
        func, ("linalg.conv_2d_nhwc_hwcf", "linalg.pooling_nhwc_max", "linalg.depthwise_conv_2d_nhwc_hwc")
    ):
        return None

    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_val = args[0]
    if elem_dtype_of(in_val) != "f32":
        return None
    in_shape = shape_of(in_val)

    # Aggregate body op names across every linalg.generic (legacy
    # behaviour — the recognizer is module-wide).
    body_ops: set[str] = set()
    for op in walk_inner_ops(func):
        if op.operation.name == "linalg.generic":
            body_ops |= linalg_generic_body_op_names(op)

    if not body_ops:
        return None

    if "math.tanh" in body_ops and "math.exp" not in body_ops:
        op_type = "Tanh"
    elif "math.exp" in body_ops and "arith.divf" in body_ops and "arith.addf" in body_ops:
        op_type = "Sigmoid"
    elif "arith.maximumf" in body_ops and "arith.minimumf" in body_ops:
        op_type = "Relu6"
    elif "arith.maximumf" in body_ops and "arith.minimumf" not in body_ops:
        op_type = "Relu"
    else:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    if elem_dtype_of(out_val) != "f32":
        return None
    out_shape = shape_of(out_val)
    if out_shape != in_shape:
        return None

    parsed = ParsedElementwiseUnary(
        func_name=func_name(func),
        op_type=op_type,
        input_shape=tuple(in_shape),
        output_shape=tuple(out_shape),
    )
    return lower_elementwise_unary(parsed)
