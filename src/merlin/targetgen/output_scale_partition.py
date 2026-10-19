"""Bind a logical column-scale obligation without assigning a physical schedule.

Independent contraction output columns may use different scalar readout gains.
The hardware vector width still describes the elements sharing one physical
gain. This record preserves the source gain vector and that width; target
packing, commands, addresses, legality and execution remain compiler obligations.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real

SCHEMA = "logical_output_column_partition_v1"


def interface_components(source):
    """Scalar-scale logical components of an explicit independent-column contraction.

    This proves only source arithmetic and exhaustive disjoint column coverage.
    Every component still needs the ordinary SW/HW declaration screen; packing,
    addresses, physical commands and execution remain candidate obligations.
    Unsupported source forms return None, never a permissive scalar default.
    """
    from merlin.runtime.commandbuffer import declared_acc_scale_by_column

    if not isinstance(source, dict) or source.get("operation") not in {"matmul", "conv2d"}:
        return None
    lhs, rhs, attrs = (source.get(key) for key in ("lhs", "rhs", "attributes"))
    if not isinstance(attrs, dict) or not all(isinstance(item, dict) for item in (lhs, rhs)):
        return None
    shapes = [item.get("shape") for item in (lhs, rhs)]
    if any(
        not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape) for shape in shapes
    ):
        return None
    if lhs.get("dtype") not in {"i8", "int8"} or rhs.get("dtype") != lhs.get("dtype"):
        return None
    a, w = shapes
    if len(w) != 2 or source.get("disjoint") is not True:
        return None
    if source["operation"] == "matmul":
        if len(a) != 2 or a[1] != w[0]:
            return None
        rows = a[0]
    else:
        kernel = attrs.get("kernel")
        if (
            len(a) != 4
            or not isinstance(kernel, list)
            or len(kernel) != 4
            or any(type(n) is not int or n < 1 for n in kernel)
            or kernel[2] != a[-1]
            or math.prod(kernel[:3]) != w[0]
            or kernel[3] != w[1]
        ):
            return None
        from merlin.runtime.commandbuffer import conv_out_dims

        geometry = [attrs.get(key) for key in ("stride", "padding", "dilation")]
        if any(
            not isinstance(value, list) or len(value) != arity or any(type(n) is not int or n < minimum for n in value)
            for value, arity, minimum in zip(geometry, (2, 4, 2), (1, 0, 1), strict=True)
        ):
            return None
        ho, wo = conv_out_dims(a[1], a[2], *kernel[:2], *geometry)
        if ho < 1 or wo < 1 or str(attrs.get("layout", "nhwc")).lower() != "nhwc":
            return None
        rows = a[0] * ho * wo
    stages = attrs.get("epilogue")
    if (
        not isinstance(stages, list)
        or "acc_scale" not in stages
        or any(stage not in {"bias", "bias_add", "acc_scale", "relu"} for stage in stages)
        or attrs.get("output_dtype") not in {"i8", "int8"}
    ):
        return None
    geometry = {"kernel", "stride", "padding", "dilation"}
    if set(attrs) - geometry - {"epilogue", "acc_scale_by_column", "output_dtype", "bias", "layout", "name"}:
        return None
    result = source.get("result")
    if result is not None and (result.get("shape") != [rows, w[1]] or result.get("dtype") != attrs["output_dtype"]):
        return None
    if any(stage in {"bias", "bias_add"} for stage in stages):
        bias = source.get("bias") or {}
        if bias.get("shape") != [w[1]] or bias.get("dtype") not in {"i32", "int32"}:
            return None
    try:
        gains = declared_acc_scale_by_column(attrs, w[1], op="logical output-column partition")
    except (ValueError, TypeError, OverflowError):
        return None
    # Unknown vector-valued attributes could couple columns. Geometry is shared
    # unchanged; all other arrays/mappings keep the partition unavailable.
    if any(
        isinstance(value, (list, dict))
        for key, value in attrs.items()
        if key not in geometry | {"epilogue", "acc_scale_by_column"}
    ):
        return None
    parts = []
    start = 0
    while start < len(gains):
        end = start + 1
        while end < len(gains) and gains[end] == gains[start]:
            end += 1
        parts.append(
            {"start": start, "stop": end, "gain": gains[start], "dimensions": {"M": rows, "K": w[0], "N": end - start}}
        )
        start = end
    return {
        "scope": "source_arithmetic_only",
        "columns": w[1],
        "parts": parts,
        "preserved": ["complete_reduction", "bias_lane", "stage_order", "rounding", "saturation"],
        "physical_implementation": "unverified",
    }


def obligation(gains, physical_columns):
    """A source-arithmetic obligation for gains varying within a physical block.

    Return None for an already block-uniform vector. Callers establish that
    these are independent source output columns; this function grants no such
    independence, hardware admission, implementation or timing authority.
    """
    if type(physical_columns) is not int or physical_columns < 1:
        raise ValueError("the physical scale-vector width must be a positive integer")
    if len(gains) == 0:
        raise ValueError("a column-scale vector must be nonempty")
    digest = hashlib.sha256()
    values = []
    for value in gains:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("column gains must be positive finite numeric values")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("column gains must be positive finite numeric values")
        # The caller retains the selected numeric format in the complete proof.
        # Bind its exact already-typed values without imposing another format.
        digest.update(value.hex().encode("ascii") + b"\n")
        values.append(value)
    if all(
        len(set(values[start : start + physical_columns])) == 1 for start in range(0, len(values), physical_columns)
    ):
        return None
    return {
        "schema": SCHEMA,
        "scope": "source_arithmetic_only",
        "columns": len(values),
        "physical_columns_per_scale": physical_columns,
        "gains_sha256": digest.hexdigest(),
        "requires": "independent_output_column_implementation",
    }
