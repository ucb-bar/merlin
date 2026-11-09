"""Validate declared runtime dimension bindings without evaluator data.

This checks the producer ABI and typed signature. It grants byte custody and
shape-contract consistency, not source bounds, ownership or numeric admission.
"""

from math import prod


def validate_dimensions(spec, buffers, scalars):
    names = set()
    for buffer in buffers:
        shape = buffer.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError("kernel tensor shape must have a declared rank")
        for size in shape:
            if isinstance(size, str) and size.isidentifier():
                names.add(size)
            elif type(size) is not int or size <= 0:
                raise ValueError("kernel dimensions require positive integers or named scalar bindings")
        if any(isinstance(n, str) for n in shape) and len(shape) != 1:
            raise ValueError("dynamic multidimensional kernel admission requires another profile")
    bounds = {s["name"]: s.get("bounds") for s in scalars}
    grid = spec.get("grid", [1])
    if not isinstance(grid, list) or not 1 <= len(grid) <= 3:
        raise ValueError("kernel logical grid requires one through three axes")
    maximum = []
    dynamic = bool(names)
    for extent in grid:
        if isinstance(extent, dict):
            if (set(extent) != {"dimension", "divisor"} or not isinstance(extent["dimension"], str)
                    or not extent["dimension"].isidentifier() or type(extent["divisor"]) is not int
                    or not 0 < extent["divisor"] < 2**63):
                raise ValueError("invalid runtime logical grid expression")
            dynamic = True
            name, divisor = extent["dimension"], extent["divisor"]
            names.add(name)
            interval = bounds.get(name)
            if interval is None or interval[0] < 0:
                raise ValueError("runtime grid requires a nonnegative bounded dimension")
            extent = (interval[1] + divisor - 1) // divisor
        if type(extent) is not int or not 0 < extent < 2**31:
            raise ValueError("kernel grid bound is outside its positive i32 domain")
        maximum.append(extent)
    if prod(maximum) >= 2**63:
        raise ValueError("kernel logical launch product exceeds its i64 domain")
    if any(bounds.get(name) is None or bounds[name][0] < 0 for name in names):
        raise ValueError("runtime dimensions require nonnegative bounded integer scalar bindings")
    if dynamic != (spec["schema"] == "t2m.shared-tensor-abi.v5"):
        raise ValueError("runtime dimensions and grids require explicit ABI v5")
    return dynamic
