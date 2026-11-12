"""Explicit deterministic stress inputs, independent of numerical answers or schedules.

Patterns select exactly representable scalar values along a declared tensor axis.
Format landmarks are derived from the shared format registry/codecs. They never
change an oracle's arithmetic, rounding order, tolerance or acceptance policy.
"""

from __future__ import annotations

import math

SCHEMA = "merlin.component_input_palette.v1"


def validate(declaration):
    if not isinstance(declaration, dict) or set(declaration) != {"schema", "inputs"} or declaration["schema"] != SCHEMA:
        raise ValueError("input palette requires the closed component_input_palette v1 schema")
    rows = declaration["inputs"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("input palette requires explicit input patterns")
    names = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "axis", "values", "offset"}:
            raise ValueError("input pattern requires name, axis, values and offset")
        name = row["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("input pattern names must be unique and explicit")
        names.add(name)
        if row["axis"] != "linear" and type(row["axis"]) is not int:
            raise ValueError("input pattern axis must be a dimension index or linear")
        if not isinstance(row["values"], list) or not row["values"]:
            raise ValueError("input pattern requires a nonempty scalar palette")
        if type(row["offset"]) is not int or not 0 <= row["offset"] < len(row["values"]):
            raise ValueError("input pattern offset must select a palette position")
        for value in row["values"]:
            if type(value) in (float, int) and math.isfinite(value):
                continue
            if isinstance(value, str) and value:
                continue
            raise ValueError("input palette values must be finite numbers or explicit format landmarks")
    return declaration


def scalar(value, dtype):
    from merlin.common.quant_formats import get
    from merlin.runtime import fp8_formats as formats

    descriptor = get(dtype)
    negative = isinstance(value, str) and value.startswith("-")
    token = value[1:] if negative else value
    if isinstance(value, str):
        if descriptor.is_float:
            exponent, mantissa, bias, scheme = formats.float_format_params(dtype)
            minimum, maximum = formats.normal_range(dtype)
            landmarks = {
                "zero": 0.0,
                "one": 1.0,
                "min_subnormal": 2.0 ** (1 - bias - mantissa),
                "min_normal": minimum,
                "max_finite": maximum,
                "half_ulp_one": 2.0 ** (-mantissa - 1),
                "cancellation_large": 2.0 ** (mantissa + 1),
                "infinity": float("inf"),
                "nan": float("nan"),
            }
            if token == "infinity" and scheme != "ieee":
                raise ValueError("selected float storage format cannot represent infinity")
            if token == "nan" and scheme == "mx_finite":
                raise ValueError("selected float storage format cannot represent NaN")
        else:
            bits = descriptor.element_bits
            landmarks = {"zero": 0, "one": 1, "max_signed": (1 << (bits - 1)) - 1, "min_signed": -(1 << (bits - 1))}
        if token not in landmarks:
            raise ValueError(f"input palette landmark {value!r} is unknown for {dtype!r}")
        value = landmarks[token]
        if negative:
            value = -value
    if descriptor.is_float:
        import numpy as np

        raw = formats.float_to_codes([value], dtype)
        decoded = float(formats._decode(np.asarray(raw), dtype)[0])
        if math.isnan(value):
            if not math.isnan(decoded):
                raise ValueError("input palette NaN cannot be represented in the selected storage format")
        elif decoded != value or (value == 0 and math.copysign(1, decoded) != math.copysign(1, value)):
            raise ValueError(f"input palette value {value!r} is not exactly representable in {dtype!r}")
        return value
    bits = descriptor.element_bits
    if type(value) not in (int, float) or int(value) != value or not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise ValueError(f"input palette value exceeds signed {dtype!r} storage")
    return int(value)


def pattern(declaration, *, name, dtype, index=None):
    validate(declaration)
    candidates = [row for row in declaration["inputs"] if row["name"] == name]
    if index is not None and name != f"arg{index}":
        candidates += [row for row in declaration["inputs"] if row["name"] == f"arg{index}"]
    if len(candidates) > 1:
        raise ValueError("input palette names one tensor through conflicting named and positional selectors")
    if not candidates:
        candidates = [row for row in declaration["inputs"] if row["name"] == "*"]
    if not candidates:
        return None
    row = candidates[0]
    return {**row, "values": [scalar(value, dtype) for value in row["values"]]}


def realize(declaration, *, name, shape, dtype, index=None):
    selected = pattern(declaration, name=name, dtype=dtype, index=index)
    if selected is None:
        return None
    if not shape or any(type(dim) is not int or dim < 1 for dim in shape):
        raise ValueError("input palette needs a positive static tensor shape")
    count = math.prod(shape)
    axis = selected["axis"]
    if axis == "linear":
        stride, extent = 1, count
    else:
        if not -len(shape) <= axis < len(shape):
            raise ValueError("input palette axis is outside its tensor rank")
        axis %= len(shape)
        stride, extent = math.prod(shape[axis + 1 :]), shape[axis]
    values, offset = selected["values"], selected["offset"]
    if extent < len(values):
        raise ValueError("input palette axis is too short to exercise every declared value")
    return [values[((position // stride) % extent + offset) % len(values)] for position in range(count)]


def findings(declaration, *, name, shape, dtype, index=None):
    """Observable operand landmarks; execution/numerical equivalence is unproved."""
    values = realize(declaration, name=name, shape=shape, dtype=dtype, index=index)
    if values is None:
        return []
    from merlin.common.quant_formats import get
    from merlin.runtime.fp8_formats import normal_range

    result = set()
    finite = [value for value in values if math.isfinite(value)]
    if any(not math.isfinite(value) for value in values):
        result.add("nonfinite_input")
    if any(value == 0 and math.copysign(1, value) < 0 for value in values):
        result.add("signed_zero_input")
    if finite and min(finite) < 0 < max(finite):
        result.add("signed_input")
    descriptor = get(dtype)
    if descriptor.is_float:
        minimum, maximum = normal_range(dtype)
        if any(0 < abs(value) < minimum for value in finite):
            result.add("subnormal_input")
        if any(abs(value) == maximum for value in finite):
            result.add("maximum_finite_input")
        alphabet = set(finite)
        if 1.0 in alphabet and scalar("half_ulp_one", dtype) in alphabet:
            result.add("rounding_tie_input")
        if any(
            -value in alphabet and any(0 < abs(small) < abs(value) for small in alphabet) for value in alphabet if value
        ):
            result.add("cancellation_input")
    return sorted(result)


def render_source_inputs(declaration, names, dtype):
    """Emit one closed input-only override for an ordinary builtin Torch loader."""
    palettes = [pattern(declaration, name=name, dtype=dtype, index=index) for index, name in enumerate(names)]
    used = {row["name"] for row in palettes if row is not None}
    if any(row["name"] not in used for row in declaration["inputs"]):
        raise ValueError("input palette names a tensor absent from the generated PyTorch program")
    # Source contains the selected concrete values and a closed input-only
    # materializer. Capture binds its bytes and the actual typed operands.
    literals = []
    for palette in palettes:
        if palette is None:
            literals.append("None")
            continue
        values = [
            repr(value) if math.isfinite(value) else "float(" + repr(str(value)) + ")" for value in palette["values"]
        ]
        fields_literal = [repr(key) + ": " + repr(value) for key, value in palette.items() if key != "values"]
        literals.append("{" + ", ".join(fields_literal) + ", 'values': [" + ", ".join(values) + "]}")
    return _PALETTE_INPUTS.replace("__PALETTES__", "[" + ", ".join(literals) + "]")


_PALETTE_INPUTS = """
_PALETTES = __PALETTES__
_original_get_model_and_inputs = get_model_and_inputs
def get_model_and_inputs():
    model, inputs = _original_get_model_and_inputs()
    selected = []
    for tensor, palette in zip(inputs, _PALETTES, strict=True):
        if palette is None:
            selected.append(tensor)
            continue
        shape = tuple(tensor.shape)
        axis = palette["axis"]
        if axis == "linear":
            stride, extent = 1, tensor.numel()
        else:
            if not -len(shape) <= axis < len(shape):
                raise ValueError("input palette axis is outside its tensor rank")
            axis %= len(shape)
            stride, extent = math.prod(shape[axis + 1:]), shape[axis]
        values, offset = palette["values"], palette["offset"]
        if extent < len(values):
            raise ValueError("input palette axis cannot exercise every declared value")
        flat = [values[((position // stride) % extent + offset) % len(values)]
                for position in range(tensor.numel())]
        selected.append(torch.tensor(flat, dtype=tensor.dtype, device=tensor.device).reshape(shape))
    return model, tuple(selected)
"""
