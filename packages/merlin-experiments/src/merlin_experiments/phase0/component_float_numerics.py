"""Independent exact-rational reference for explicit finite floating DAGs.

No compiler, target model or emitted IR is evaluated. Format layouts are shared
data; arithmetic and rounding are derived here from the selected source policy.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from fractions import Fraction
from pathlib import Path

from merlin.runtime.fp8_formats import float_format_params, normal_range
from merlin.runtime.tensor import Tensor
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.component_program import analyze


def source_identity():
    modules = (
        __name__,
        "merlin.targetgen.component_program",
        "merlin.targetgen.component_float",
        "merlin.runtime.fp8_formats",
        "merlin.common.quant_formats",
        "merlin.runtime.tensor",
        "merlin.targetgen.capsule_inputs",
        "merlin.targetgen.input_palette",
    )
    from merlin.targetgen.component_float import FORMATS

    return {
        "modules_sha256": {
            name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
            for name in modules
        },
        "formats": {dtype: list(float_format_params(dtype)) for dtype in FORMATS},
    }


def rounded(value, dtype, rounding="rne", *, negative_zero=False):
    """Round an exact finite value once; reject overflow instead of inventing saturation."""
    if rounding not in {"rne", "rtz"}:
        raise ValueError("unsupported independent component rounding")
    if not isinstance(value, Fraction):
        if not math.isfinite(value):
            raise ValueError("floating component reference requires finite values")
        negative_zero = value == 0 and math.copysign(1, value) < 0
        value = Fraction(value)
    negative = value < 0 or value == 0 and negative_zero
    value = abs(value)
    if not value:
        return -0.0 if negative else 0.0
    _, mantissa, bias, _ = float_format_params(dtype)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = Fraction(2) ** exponent
    if value < power:
        exponent -= 1
    step = Fraction(2) ** max(exponent - mantissa, 1 - bias - mantissa)
    scaled = value / step
    integral, remainder = divmod(scaled.numerator, scaled.denominator)
    if rounding == "rne" and (
        2 * remainder > scaled.denominator or 2 * remainder == scaled.denominator and integral % 2
    ):
        integral += 1
    result = integral * step
    if result > Fraction(normal_range(dtype)[1]):
        raise ValueError("floating component intermediate overflows its declared format")
    return -float(result) if negative else float(result)


def add(lhs, rhs, dtype, rounding="rne"):
    negative_zero = lhs == rhs == 0 and math.copysign(1, lhs) < 0 and math.copysign(1, rhs) < 0
    return rounded(Fraction(lhs) + Fraction(rhs), dtype, rounding, negative_zero=negative_zero)


def multiply(lhs, rhs, dtype):
    return rounded(Fraction(lhs) * Fraction(rhs), dtype, negative_zero=math.copysign(1, lhs) != math.copysign(1, rhs))


def evaluate(capsule):
    typed = capsule["component_program"]
    program = analyze(
        capsule["operation"]["attributes"]["program"],
        operand_dtype=typed["selected_storage"]["operand"],
        accumulator_dtype=typed["selected_storage"]["accumulator"],
    )
    if program != typed or "floating_point" not in program or capsule["inputs"] != program["inputs"]:
        raise ValueError("floating reference requires the exact explicit typed source")
    env = materialize_capsule_leaves(capsule)
    for row in program["inputs"]:
        for value in env[row["name"]].data:
            if not math.isfinite(value) or rounded(value, row["dtype"]) != value:
                raise ValueError("floating leaf is not exactly representable in declared storage")
    for node in program["nodes"]:
        args = [env[name] for name in node["actual_inputs"]]
        op, dtype = node["op"], node["dtype"]
        if op == "matmul":
            lhs, rhs = args
            m, k = lhs.shape
            n = rhs.shape[1]
            values = []
            for i in range(m):
                for j in range(n):
                    acc = 0.0
                    for p in range(k):
                        acc = add(acc, multiply(lhs.data[i * k + p], rhs.data[p * n + j], dtype), dtype)
                    values.append(acc)
        elif op in {"add", "update"}:
            values = [
                rounded(add(a, b, node["compute_dtype"]), dtype, node["rounding"])
                for a, b in zip(args[0].data, args[1].data, strict=True)
            ]
        elif op == "convert":
            values = [rounded(value, dtype, node["rounding"]) for value in args[0].data]
        elif op == "relu":
            values = [value if value > 0 else 0.0 for value in args[0].data]
        elif op == "transpose":
            rows, cols = args[0].shape
            values = [args[0].data[row * cols + col] for col in range(cols) for row in range(rows)]
        else:
            values = list(args[0].data)
        env[node["name"]] = Tensor(tuple(node["shape"]), values, dtype)
    provenance = {
        row["name"]: {"shape": row["shape"], "dtype": row["dtype"], "decoded": list(env[row["name"]].data)}
        for row in program["inputs"]
    }
    return {row["name"]: env[row["actual_value"]].to_list() for row in program["outputs"]}, provenance


def preflight(entry, *, binding):
    """Bound source work before allocating any canonical tensors or goldens."""
    from .component_execution_budget import measure, source_for_entry

    limits = entry.get("reference_budget")
    if (
        not isinstance(limits, dict)
        or set(limits) != {"max_reference_work", "max_materialized_elements"}
        or any(type(value) is not int or value < 1 for value in limits.values())
    ):
        raise ValueError("floating component requires explicit positive reference work and element budgets")
    costs = measure(source_for_entry(entry, binding=binding))
    if any(costs[key.removeprefix("max_")] > limit for key, limit in limits.items()):
        raise ValueError("floating component exceeds selected reference budget")
    return costs
