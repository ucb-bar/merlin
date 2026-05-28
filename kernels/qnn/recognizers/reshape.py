"""Bindings recognizer: f32 reshape via `tensor.collapse_shape` /
`tensor.expand_shape`."""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    ParsedReshape,
    lower_reshape,
)
from .base import (  # noqa: E402
    elem_dtype_of,
    find_func,
    find_named_op,
    func_arg_values,
    func_name,
    shape_of,
)

NAME = "f32_reshape"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    if find_named_op(func, "tensor.collapse_shape") is None and find_named_op(func, "tensor.expand_shape") is None:
        return None

    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_val = args[0]
    if elem_dtype_of(in_val) != "f32":
        return None
    in_shape = shape_of(in_val)

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    if elem_dtype_of(out_val) != "f32":
        return None
    out_shape = shape_of(out_val)

    n_in = 1
    for d in in_shape:
        n_in *= d
    n_out = 1
    for d in out_shape:
        n_out *= d
    if n_in != n_out:
        return None

    parsed = ParsedReshape(
        func_name=func_name(func),
        input_shape=tuple(in_shape),
        output_shape=tuple(out_shape),
    )
    return lower_reshape(parsed)
