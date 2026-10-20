"""Closed typed mixed-precision scalar source bodies; no numerical or placement grant."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from struct import pack

from merlin.frontends.linalg_composite_math import _scalar, _yield
from merlin.frontends.linalg_patterns import InvalidLinalgPattern, _checked_static_linalg_shell

SCHEMA = "merlin.static_mixed_pointwise_body.v1"
FORMS = {
    "f64_f64_arith_addf": (("f64", "f64", "f64", "f64"), ("arith.addf", "linalg.yield")),
    "bf16_bf16_arith_addf": (("bf16", "bf16", "bf16", "bf16"), ("arith.addf", "linalg.yield")),
    "f64_arith_constant_arith_divf": (("f64", "f64", "f64"), ("arith.constant", "arith.divf", "linalg.yield")),
    "f64_arith_constant_math_powf": (("f64", "f64", "f64"), ("arith.constant", "math.powf", "linalg.yield")),
    "bf16_f32_arith_extf_arith_addf": (("bf16", "f32", "f32", "f32"), ("arith.extf", "arith.addf", "linalg.yield")),
    "f64_f32_arith_extf_arith_mulf": (("f64", "f32", "f64", "f64"), ("arith.extf", "arith.mulf", "linalg.yield")),
    "bf16_f32_arith_extf_arith_mulf": (("bf16", "f32", "f32", "f32"), ("arith.extf", "arith.mulf", "linalg.yield")),
    "bf16_arith_extf": (("bf16", "f32", "f32"), ("arith.extf", "linalg.yield")),
    "f64_f64_arith_mulf": (("f64", "f64", "f64", "f64"), ("arith.mulf", "linalg.yield")),
    "bf16_bf16_arith_mulf": (("bf16", "bf16", "bf16", "bf16"), ("arith.mulf", "linalg.yield")),
    "i1_f64_f64_arith_select": (("i1", "f64", "f64", "f64", "f64"), ("arith.select", "linalg.yield")),
    "i64_arith_sitofp": (("i64", "f64", "f64"), ("arith.sitofp", "linalg.yield")),
    "i32_arith_sitofp": (("i32", "f32", "f32"), ("arith.sitofp", "linalg.yield")),
    "f64_f64_arith_subf": (("f64", "f64", "f64", "f64"), ("arith.subf", "linalg.yield")),
    "f64_arith_truncf": (("f64", "f32", "f32"), ("arith.truncf", "linalg.yield")),
    "f32_arith_truncf": (("f32", "bf16", "bf16"), ("arith.truncf", "linalg.yield")),
    "f64_math_cos": (("f64", "f64", "f64"), ("math.cos", "linalg.yield")),
    "f64_math_sin": (("f64", "f64", "f64"), ("math.sin", "linalg.yield")),
}


def validate_declaration(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "operation"}
        or value.get("schema") != SCHEMA
        or value.get("operation") not in FORMS
    ):
        raise InvalidLinalgPattern("mixed source requires a closed typed declaration")
    return dict(value)


def ordered_types(value):
    return FORMS[validate_declaration(value)["operation"]][0]


@dataclass(frozen=True)
class MixedPattern:
    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_maps: tuple[str, ...]
    literal_bits: tuple[str, ...]


def recognize(op):
    from xdsl.dialects import arith, math
    from xdsl.dialects.builtin import FloatAttr

    shell = _checked_static_linalg_shell(op, singleton_projection_inputs=True)
    names = tuple(item.name for item in shell.body)
    matches = [key for key, (types, body) in FORMS.items() if types == shell.ordered_types and body == names]
    if len(matches) != 1:
        raise InvalidLinalgPattern("mixed source has no unique exact typed body")
    operation = matches[0]
    args = list(shell.args[:-1])
    scalar_ops = shell.body[:-1]
    literals = []
    result = None
    if names[0] == "arith.constant":
        const = scalar_ops[0]
        dtype = shell.ordered_types[-1]
        value = const.properties.get("value")
        result = _scalar(const, arith.ConstantOp, (), dtype, frozenset({"value"}))
        if (
            set(const.properties) != {"value"}
            or not isinstance(value, FloatAttr)
            or str(value.type) != dtype
            or not isfinite(value.value.data)
        ):
            raise InvalidLinalgPattern("mixed source constant is not a finite typed literal")
        literals.append(pack("<d", value.value.data).hex())
        nextop = scalar_ops[1]
        if nextop.name == "arith.divf":
            if value.value.data != 1.0:
                raise InvalidLinalgPattern("mixed reciprocal numerator differs from one")
            result = _scalar(nextop, arith.DivfOp, (result, args[0]), dtype, frozenset({"fastmath"}))
        elif nextop.name == "math.powf":
            result = _scalar(nextop, math.PowFOp, (result, args[0]), dtype, frozenset({"fastmath"}))
        else:
            raise InvalidLinalgPattern("mixed literal has no closed consumer")
    elif names[0] == "arith.extf" and len(scalar_ops) == 2:
        # The narrower input is the unique extension operand; operand order remains exact.
        narrower = [i for i, arg in enumerate(args) if str(arg.type) != shell.ordered_types[-1]]
        if len(narrower) != 1:
            raise InvalidLinalgPattern("mixed arithmetic has no unique narrow input")
        axis = narrower[0]
        args[axis] = _scalar(scalar_ops[0], arith.ExtFOp, (args[axis],), shell.ordered_types[-1])
        cls = {"arith.addf": arith.AddfOp, "arith.mulf": arith.MulfOp}.get(scalar_ops[1].name)
        if cls is None:
            raise InvalidLinalgPattern("mixed extension has an unknown arithmetic consumer")
        result = _scalar(scalar_ops[1], cls, tuple(args), shell.ordered_types[-1], frozenset({"fastmath"}))
    else:
        classes = {
            "arith.addf": arith.AddfOp,
            "arith.mulf": arith.MulfOp,
            "arith.subf": arith.SubfOp,
            "arith.select": arith.SelectOp,
            "arith.extf": arith.ExtFOp,
            "arith.truncf": arith.TruncFOp,
            "arith.sitofp": arith.SIToFPOp,
            "math.sin": math.SinOp,
            "math.cos": math.CosOp,
        }
        cls = classes.get(scalar_ops[0].name)
        if cls is None or len(scalar_ops) != 1:
            raise InvalidLinalgPattern("mixed source has an unknown scalar sequence")
        result = _scalar(scalar_ops[0], cls, tuple(args), shell.ordered_types[-1], frozenset({"fastmath"}))
    _yield(shell.body[-1], result)
    return MixedPattern(
        operation,
        shell.shape,
        shell.ordered_types,
        tuple(tuple(v.type.get_shape()) for v in op.inputs),
        tuple(str(m) for m in shell.input_maps),
        tuple(literals),
    )


def validate_pattern(value):
    from collections.abc import Mapping

    from xdsl.context import Context
    from xdsl.parser import Parser

    from merlin.frontends.linalg_patterns import validate_singleton_projection_map

    if not isinstance(value, Mapping) or set(value) != {
        "operation",
        "shape",
        "ordered_types",
        "input_shapes",
        "input_maps",
        "literal_bits",
    }:
        raise InvalidLinalgPattern("mixed serialized pattern is not closed")
    expected = ordered_types({"schema": SCHEMA, "operation": value["operation"]})
    shape, shapes, maps = value["shape"], value["input_shapes"], value["input_maps"]
    if (
        not isinstance(shape, (list, tuple))
        or not shape
        or any(type(d) is not int or d <= 0 for d in shape)
        or tuple(value["ordered_types"]) != expected
        or len(shapes) != len(expected) - 2
        or len(maps) != len(shapes)
    ):
        raise InvalidLinalgPattern("mixed serialized tensor roster differs")
    for ishape, text in zip(shapes, maps, strict=True):
        if (
            not isinstance(ishape, (list, tuple))
            or any(type(d) is not int or d <= 0 for d in ishape)
            or not isinstance(text, str)
        ):
            raise InvalidLinalgPattern("mixed serialized input shape differs")
        try:
            mapping = Parser(Context(), text).parse_affine_map()
        except Exception as exc:
            raise InvalidLinalgPattern("mixed serialized map cannot parse") from exc
        if str(mapping) != text:
            raise InvalidLinalgPattern("mixed serialized map is not canonical")
        validate_singleton_projection_map(mapping, tuple(shape), tuple(ishape))
    literals = value["literal_bits"]
    body = FORMS[value["operation"]][1]
    if not isinstance(literals, (list, tuple)) or len(literals) != int(body[0] == "arith.constant"):
        raise InvalidLinalgPattern("mixed literal roster differs")
    for bits in literals:
        from struct import unpack

        try:
            raw = bytes.fromhex(bits)
            number = unpack("<d", raw)[0]
        except Exception as exc:
            raise InvalidLinalgPattern("mixed literal is not canonical f64 bits") from exc
        if raw.hex() != bits or not isfinite(number) or (body[1] == "arith.divf" and number != 1.0):
            raise InvalidLinalgPattern("mixed literal violates its closed form")
    return dict(value)
