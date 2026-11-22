"""Invocation-only validation of staged execution-result input restrictions."""

import numpy as np

from merlin.compiler.frontend.intake.execution_result_signature import validate_signature
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.lower.model.oracle import ScalarArg
from merlin.compiler.lower.options.owned_results import _NUMPY


def resolve(source, signature, arrays):
    signature = validate_signature(parse_mlir_text(source), signature)
    if len(arrays) != len(signature["inputs"]):
        raise ValueError("execution result arguments omit/change the complete source roster")
    values = {}
    constraints = signature["input_constraints"]
    for row, array in zip(signature["inputs"], arrays, strict=True):
        if (
            type(array) is not np.ndarray
            or array.dtype != np.dtype(_NUMPY[row["dtype"]])
            or not array.flags.c_contiguous
            or array.ndim != len(row["shape"])
            or any(size is not None and size != actual for size, actual in zip(row["shape"], array.shape, strict=True))
        ):
            raise ValueError("execution result invocation changes original storage/rank/static dimensions")
        if row["dtype"] == "i4" and np.any(array > 15):
            raise ValueError("execution result i4 inputs require unpacked canonical uint8 cells in 0..15")
        if row["dtype"] == "i1" and np.any(array.view(np.uint8) > 1):
            raise ValueError("execution Boolean inputs require canonical source storage")
    dimension_names = {size for row in constraints["inputs"] for size in row["shape"] if isinstance(size, str)}
    for name, coordinate in constraints["dimension_bindings"].items():
        values[name] = arrays[coordinate["argument"]].shape[coordinate["axis"]]
    for row, array in zip(constraints["inputs"], arrays, strict=True):
        if row.get("kind") == "scalar":
            if "bounds" in row and not row["bounds"][0] <= array.item() <= row["bounds"][1]:
                raise ValueError("execution result scalar is outside its original input bounds")
            if "scalar" in row:
                value = array.item()
                if (row["scalar"] in dimension_names and type(value) is not int
                        or row["scalar"] in values and values[row["scalar"]] != value):
                    raise ValueError("execution result scalar/axis dimensions disagree")
                values[row["scalar"]] = value
    for row, array in zip(constraints["inputs"], arrays, strict=True):
        for size, actual in zip(row["shape"], array.shape, strict=True):
            expected = values[size] if isinstance(size, str) else size
            if expected is not None and expected != actual:
                raise ValueError("execution result invocation violates original input dimensions")
    return tuple(
        ScalarArg.from_storage(array.tobytes(), row["dtype"]) if row["kind"] == "scalar" else array
        for row, array in zip(signature["inputs"], arrays, strict=True)
    )
