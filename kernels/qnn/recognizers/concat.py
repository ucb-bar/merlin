"""Bindings recognizer: f32 channel-wise `tensor.concat`.

NOTE on v1↔v2 textual divergence: the legacy regex emitter scraped
source-level SSA names (`%a`, `%b`) out of the MLIR text and used them as
QNN tensor names. The bindings normalize block arguments to canonical
names (`%arg0`, `%arg1`); recovering source names would require either
regex on the MLIR text (forbidden) or a separate parser. We therefore
**use canonical names** here and the parity test for concat compares
*structure* (axis, shapes, dtype, op type) instead of bytes.

QNN tensor names are internal to the resulting `.qnn-ctx`; runtime
binding is by index, not name, so this is a textual difference only.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ..emit import (  # noqa: E402
    ParsedConcat,
    lower_concat,
)
from .base import (  # noqa: E402
    elem_dtype_of,
    find_func,
    find_named_op,
    func_arg_values,
    func_name,
    integer_attr_value,
    shape_of,
)

NAME = "f32_concat"


def try_recognize(
    module: Any,
    *,
    fp_dtype: str = "float32",
    **_: object,
) -> Any | None:
    func = find_func(module)
    if func is None:
        return None

    concat_op = find_named_op(func, "tensor.concat")
    if concat_op is None:
        return None

    args = func_arg_values(func)
    if len(args) < 2:
        return None
    if any(elem_dtype_of(a) != "f32" for a in args):
        return None

    arg_names = tuple(a.get_name().lstrip("%") for a in args)
    arg_shapes = [shape_of(a) for a in args]
    arg_id_to_index = {a.get_name(): i for i, a in enumerate(args)}

    # Match concat operands back to their func-arg index. Legacy contract:
    # every concat operand must be a func arg (no intermediate-op inputs).
    operand_indices: list[int] = []
    for opnd in concat_op.operands:
        ssa = opnd.get_name()
        if ssa not in arg_id_to_index:
            return None
        operand_indices.append(arg_id_to_index[ssa])

    ordered_input_names = tuple(arg_names[i] for i in operand_indices)
    ordered_input_shapes = tuple(arg_shapes[i] for i in operand_indices)

    axis = integer_attr_value(concat_op, "dim")
    if axis is None:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    out_shape = shape_of(out_val)
    if elem_dtype_of(out_val) != "f32":
        return None

    expected_axis = sum(s[axis] for s in ordered_input_shapes)
    if expected_axis != out_shape[axis]:
        return None
    for d in range(len(out_shape)):
        if d == axis:
            continue
        if any(s[d] != out_shape[d] for s in ordered_input_shapes):
            return None

    parsed = ParsedConcat(
        func_name=func_name(func),
        input_arg_names=ordered_input_names,
        input_shapes=ordered_input_shapes,
        output_shape=tuple(out_shape),
        axis=axis,
    )
    return lower_concat(parsed)
