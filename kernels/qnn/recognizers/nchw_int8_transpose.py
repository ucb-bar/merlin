"""Bindings recognizer for the NCHW int8 transpose dispatch.

Matches `linalg.transpose` on i8 tensors and emits a QNN Transpose
node carrying the same `permutation` attribute.
"""

from __future__ import annotations

from typing import Any

from .base import (
    elem_dtype_of,
    find_func,
    find_named_op,
    func_arg_values,
    shape_of,
)

NAME = "nchw_int8_transpose"


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None
    anchor = find_named_op(func, "linalg.transpose")
    if anchor is None:
        return None

    args = func_arg_values(func)
    if len(args) != 1:
        return None
    in_val = args[0]
    if elem_dtype_of(in_val) != "i8":
        return None
    in_shape = tuple(shape_of(in_val))

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    if elem_dtype_of(out_val) != "i8":
        return None
    out_shape = tuple(shape_of(out_val))

    if "permutation" not in anchor.attributes:
        return None
    perm_attr = anchor.attributes["permutation"]
    try:
        perm = tuple(int(perm_attr[i]) for i in range(len(perm_attr)))
    except (TypeError, ValueError):
        return None
    if len(perm) != len(in_shape):
        return None

    from iree.compiler import ir  # noqa: PLC0415  - lazy
    from qnn_ir import (  # noqa: PLC0415  - lazy
        QnnGraphDesc,
        QuantParams,
        TensorDesc,
        transpose_node,
    )

    sym_name = ir.StringAttr(func.attributes["sym_name"]).value
    qp = QuantParams(scale=1.0, offset=0)

    tensors = (
        TensorDesc(
            name="input",
            shape=in_shape,
            dtype="int8",
            role="input",
            quant=qp,
        ),
        TensorDesc(
            name="output",
            shape=out_shape,
            dtype="int8",
            role="output",
            quant=qp,
        ),
    )
    nodes = (
        transpose_node(
            name="transpose_op",
            input_tensor="input",
            output_tensor="output",
            perm=perm,
        ),
    )
    return QnnGraphDesc(name=sym_name, tensors=tensors, nodes=nodes)
