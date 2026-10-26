"""Closed, source-only scalar-region proofs for literal FP32 math forms.

These patterns describe parsed source operations. They neither admit host
placement nor prove selected lowering, libm suppliers, or numerical agreement
with the frontend that produced the capture.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from struct import pack, unpack

from merlin.frontends.linalg_patterns import (
    InvalidLinalgPattern,
    _checked_static_linalg_shell,
    _screen_static_linalg_source,
)

SCHEMA = "merlin.static_composite_math_source.v1"
STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA = "merlin.static_composite_math_body.v1"
SOURCE_ONLY_SCOPE = "source body only; host placement, selected libm supplier and numerical equivalence unproved"

_FORMS = {
    "clamp_upper_literal": (("f32", "f32", "f32"), 1, ()),
    "clamp_upper_literal_compare_select": (("f32", "f32", "f32"), 1, ()),
    "clamp_lower_literal": (("f32", "f32", "f32"), 1, ()),
    "clamp_literal_interval_f32": (("f32", "f32", "f32"), 2, ()),
    "fptosi_i8_of_rounded_literal_interval": (("f32", "i8", "i8"), 2, ()),
    "clamp_lower_f32_min_normal_compare_select": (("f32", "f32", "f32"), 1, ()),
    "clamp_symmetric_int8_grid_compare_select": (("f32", "f32", "f32"), 2, ()),
    "fptosi_i8_of_rounded_int8_grid_clamp": (("f32", "i8", "i8"), 2, ()),
    "reciprocal_f32": (("f32", "f32", "f32"), 1, ()),
    "reciprocal_signed_i64_to_f32": (("i64", "f32", "f32"), 1, ()),
    "literal_base_pow_f32": (("f32", "f32", "f32"), 1, ("math.powf",)),
    "tanh_gelu_f32": (("f32", "f32", "f32"), 4, ("math.tanh",)),
}
_POSITIVE_ONE = "0x3f800000"
# Exact bounds of the two quantization clamps: FLT_MIN (smallest positive normal f32), and the
# symmetric int8 grid [-127, 127] written as i64 literals converted by arith.sitofp (exact in f32).
_F32_MIN_NORMAL = "0x00800000"
_INT8_GRID_BOUNDS = (-127, 127)
_INT8_GRID_BITS = ("0xc2fe0000", "0x42fe0000")
_EXACT_LITERALS = {
    "clamp_lower_f32_min_normal_compare_select": (_F32_MIN_NORMAL,),
    "clamp_symmetric_int8_grid_compare_select": _INT8_GRID_BITS,
    "fptosi_i8_of_rounded_int8_grid_clamp": _INT8_GRID_BITS,
}
_CMPF_OLT, _CMPF_OGT = 4, 2
# Public mathematical coefficients, in the exact f32 source operation order.
_TANH_GELU_LITERALS = ("0x3f000000", _POSITIVE_ONE, "0x3d372713", "0x3f4c422a")


def validate_static_composite_math_source_body(declaration: object) -> dict[str, str]:
    """Validate an exact opt-in declaration; grant no host placement."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "operation"}
        or declaration.get("schema") != STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA
        or type(declaration.get("operation")) is not str
        or declaration["operation"] not in _FORMS
    ):
        raise InvalidLinalgPattern("source_body requires a closed static composite math v1 declaration")
    return dict(declaration)


def static_composite_math_ordered_types(declaration: object) -> tuple[str, ...]:
    """Return input/init/result types for one validated source-body form."""
    return _FORMS[validate_static_composite_math_source_body(declaration)["operation"]][0]


@dataclass(frozen=True)
class StaticCompositeMathPattern:
    schema: str
    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    literal_bits: tuple[str, ...]
    lowering_intrinsic_obligations: tuple[str, ...]


@dataclass(frozen=True)
class StaticCompositeMathSource:
    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticCompositeMathPattern], ...]


def _f32_bits(value: float) -> str:
    if not isinstance(value, float) or not isfinite(value):
        raise InvalidLinalgPattern("math source literal is not a finite f32 value")
    try:
        return f"0x{unpack('<I', pack('<f', value))[0]:08x}"
    except (OverflowError, ValueError) as exc:
        raise InvalidLinalgPattern("math source literal is outside finite f32") from exc


def _scalar(op, expected_class, operands: tuple, result_type: str, allowed_properties: frozenset[str] = frozenset()):
    from xdsl.dialects.arith import FastMathFlagsAttr
    from xdsl.traits import Pure

    if type(op) is not expected_class or not type(op).has_trait(Pure):
        raise InvalidLinalgPattern("math source scalar is not an exact registered pure operation")
    if (
        op.regions
        or op.successors
        or any(not name.startswith("prov.") for name in op.attributes)
        or not set(op.properties) <= allowed_properties
        or tuple(op.operands) != operands
        or len(op.results) != 1
        or str(op.results[0].type) != result_type
    ):
        raise InvalidLinalgPattern("math source scalar has unknown metadata, type or SSA operands")
    flags = op.properties.get("fastmath")
    if flags is not None and (not isinstance(flags, FastMathFlagsAttr) or flags.data):
        raise InvalidLinalgPattern("math source scalar has fast-math flags")
    return op.results[0]


def _literal(op) -> tuple[object, str]:
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import FloatAttr

    result = _scalar(op, ConstantOp, (), "f32", frozenset({"value"}))
    value = op.properties.get("value")
    if set(op.properties) != {"value"} or not isinstance(value, FloatAttr) or str(value.type) != "f32":
        raise InvalidLinalgPattern("math source constant is not an exact f32 literal")
    return result, _f32_bits(value.value.data)


def _yield(op, result) -> None:
    from xdsl.dialects.linalg.ops import YieldOp

    if (
        type(op) is not YieldOp
        or op.regions
        or op.successors
        or op.results
        or op.properties
        or any(not name.startswith("prov.") for name in op.attributes)
        or tuple(op.operands) != (result,)
    ):
        raise InvalidLinalgPattern("math source does not yield only its closed scalar result")


def _recognize_clamp(body, source):
    from xdsl.dialects.arith import MaximumfOp, MinimumfOp

    if len(body) != 3 or type(body[1]) not in (MinimumfOp, MaximumfOp):
        raise InvalidLinalgPattern("literal clamp is not one min/max and yield")
    bound, bits = _literal(body[0])
    result = _scalar(body[1], type(body[1]), (source, bound), "f32", frozenset({"fastmath"}))
    _yield(body[2], result)
    operation = "clamp_upper_literal" if type(body[1]) is MinimumfOp else "clamp_lower_literal"
    return operation, (bits,)


def _interval_values(bits, *, int8=False):
    lower, upper = (unpack("<f", pack("<I", int(value[2:], 16)))[0] for value in bits)
    if not (isfinite(lower) and isfinite(upper) and lower <= upper):
        raise InvalidLinalgPattern("literal clamp needs an ordered finite interval")
    if int8 and not (-128 <= lower <= upper <= 127 and lower.is_integer() and upper.is_integer()):
        raise InvalidLinalgPattern("rounded cast needs integral bounds within signed i8")
    return lower, upper


def _recognize_interval_clamp(body, source):
    from xdsl.dialects.arith import MaximumfOp, MinimumfOp

    if len(body) != 5:
        raise InvalidLinalgPattern("interval clamp needs two literals, maximum, minimum and yield")
    lower, low_bits = _literal(body[0])
    value = _scalar(body[1], MaximumfOp, (source, lower), "f32", frozenset({"fastmath"}))
    upper, high_bits = _literal(body[2])
    result = _scalar(body[3], MinimumfOp, (value, upper), "f32", frozenset({"fastmath"}))
    _yield(body[4], result)
    _interval_values((low_bits, high_bits))
    return "clamp_literal_interval_f32", (low_bits, high_bits)


def _predicate(op) -> object:
    predicate = op.properties.get("predicate")
    return getattr(getattr(predicate, "value", None), "data", None)


def _compare_select(compare, select, source, bound, predicate: int):
    """One ordered compare of source against bound, selecting bound when it holds."""
    from xdsl.dialects.arith import CmpfOp, SelectOp

    if _predicate(compare) != predicate:
        raise InvalidLinalgPattern("compare/select clamp uses a different ordered predicate")
    condition = _scalar(compare, CmpfOp, (source, bound), "i1", frozenset({"predicate", "fastmath"}))
    return _scalar(select, SelectOp, (condition, bound, source), "f32")


def _recognize_lower_min_normal_clamp(body, source):
    """x < FLT_MIN ? FLT_MIN : x, with exactly that literal (a positive-denominator guard)."""
    bound, bits = _literal(body[0])
    if bits != _F32_MIN_NORMAL:
        raise InvalidLinalgPattern("lower compare/select clamp is admitted only at exactly FLT_MIN")
    value = _compare_select(body[1], body[2], source, bound, _CMPF_OLT)
    _yield(body[3], value)
    return "clamp_lower_f32_min_normal_compare_select", (bits,)


def _int8_grid_bound(constant, cast, expected: int):
    from xdsl.dialects.arith import ConstantOp, SIToFPOp
    from xdsl.dialects.builtin import IntegerAttr

    literal = _scalar(constant, ConstantOp, (), "i64", frozenset({"value"}))
    value = constant.properties.get("value")
    if (
        set(constant.properties) != {"value"}
        or not isinstance(value, IntegerAttr)
        or str(value.type) != "i64"
        or type(value.value.data) is not int
        or value.value.data != expected
    ):
        raise InvalidLinalgPattern("int8-grid clamp bound is not the exact i64 literal")
    return _scalar(cast, SIToFPOp, (literal,), "f32")


def _recognize_symmetric_int8_grid_clamp(body, source):
    """max then min against sitofp(-127) and sitofp(127): the symmetric int8 quantization grid."""
    low, high = _INT8_GRID_BOUNDS
    lower = _int8_grid_bound(body[0], body[1], low)
    clamped = _compare_select(body[2], body[3], source, lower, _CMPF_OLT)
    upper = _int8_grid_bound(body[4], body[5], high)
    value = _compare_select(body[6], body[7], clamped, upper, _CMPF_OGT)
    _yield(body[8], value)
    return "clamp_symmetric_int8_grid_compare_select", _INT8_GRID_BITS


def _sole_result_of(value, shape: tuple[int, ...]):
    from xdsl.dialects.linalg.ops import GenericOp

    owner = value.owner
    if type(owner) is not GenericOp or len(owner.results) != 1 or value is not owner.results[0]:
        raise InvalidLinalgPattern("int8 cast producer is not the sole result of a linalg.generic")
    if tuple(value.type.get_shape()) != shape:
        raise InvalidLinalgPattern("int8 cast producer has a different static shape")
    return owner


def _checked_roundeven(op, shape: tuple[int, ...]) -> None:
    from xdsl.dialects.math import RoundEvenOp

    shell = _checked_static_linalg_shell(op)
    if len(op.inputs) != 1 or shell.ordered_types != ("f32", "f32", "f32") or shell.shape != shape:
        raise InvalidLinalgPattern("int8 cast producer chain does not start at an f32 roundeven")
    body = shell.body
    if len(body) != 2:
        raise InvalidLinalgPattern("int8 cast producer roundeven has extra or missing operations")
    result = _scalar(body[0], RoundEvenOp, (shell.args[0],), "f32", frozenset({"fastmath"}))
    _yield(body[1], result)
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL is the source grammar and type verifier
        raise InvalidLinalgPattern(f"int8 cast roundeven producer failed xDSL verification: {exc}") from exc


def _recognize_int8_grid_cast(op, body, source, shape: tuple[int, ...]):
    """fptosi f32 -> i8 whose operand is proved integral and within [-127, 127].

    The operand must be the sole result of the exact int8-grid clamp, whose operand is in
    turn the sole result of an f32 roundeven over the same static shape. For every non-NaN
    input that chain yields an integral f32 in [-127, 127], so the conversion is exact and
    in range. A NaN propagates through both and is outside this proof (non-finite native
    behaviour stays unqualified, as for the other host forms).
    """
    from xdsl.dialects.arith import FPToSIOp

    if len(body) != 2:
        raise InvalidLinalgPattern("int8 cast is not one fptosi and yield")
    result = _scalar(body[0], FPToSIOp, (source,), "i8")
    _yield(body[1], result)
    clamp = _sole_result_of(op.inputs[0], shape)
    pattern = recognize_static_composite_math_body(clamp)
    if pattern.operation == "clamp_literal_interval_f32":
        _interval_values(pattern.literal_bits, int8=True)
        _checked_roundeven(_sole_result_of(clamp.inputs[0], shape), shape)
        return "fptosi_i8_of_rounded_literal_interval", pattern.literal_bits
    if pattern.operation != "clamp_symmetric_int8_grid_compare_select":
        raise InvalidLinalgPattern("int8 cast operand is not the exact int8-grid clamp")
    _checked_roundeven(_sole_result_of(clamp.inputs[0], shape), shape)
    return "fptosi_i8_of_rounded_int8_grid_clamp", _INT8_GRID_BITS


def _recognize_compare_select_clamp(body, source):
    from xdsl.dialects.arith import CmpfOp, SelectOp

    if len(body) != 4:
        raise InvalidLinalgPattern("literal compare/select clamp has extra or missing operations")
    if _predicate(body[1]) == _CMPF_OLT:
        return _recognize_lower_min_normal_clamp(body, source)
    bound, bits = _literal(body[0])
    predicate = body[1].properties.get("predicate")
    if getattr(getattr(predicate, "value", None), "data", None) != 2:
        raise InvalidLinalgPattern("upper compare/select clamp must use ordered greater-than")
    condition = _scalar(body[1], CmpfOp, (source, bound), "i1", frozenset({"predicate", "fastmath"}))
    value = _scalar(body[2], SelectOp, (condition, bound, source), "f32")
    _yield(body[3], value)
    return "clamp_upper_literal_compare_select", (bits,)


def _recognize_reciprocal(body, source, *, signed_i64: bool):
    from xdsl.dialects.arith import DivfOp, SIToFPOp

    if signed_i64:
        if len(body) != 4:
            raise InvalidLinalgPattern("signed reciprocal is not cast, literal, division and yield")
        source = _scalar(body[0], SIToFPOp, (source,), "f32")
        literal, division, yld = body[1:]
    else:
        if len(body) != 3:
            raise InvalidLinalgPattern("f32 reciprocal is not literal, division and yield")
        literal, division, yld = body
    one, bits = _literal(literal)
    if bits != _POSITIVE_ONE:
        raise InvalidLinalgPattern("reciprocal numerator is not exact positive one")
    result = _scalar(division, DivfOp, (one, source), "f32", frozenset({"fastmath"}))
    _yield(yld, result)
    return ("reciprocal_signed_i64_to_f32" if signed_i64 else "reciprocal_f32"), (bits,)


def _recognize_pow(body, source):
    from xdsl.dialects.math import PowFOp

    if len(body) != 3:
        raise InvalidLinalgPattern("literal-base power is not literal, power and yield")
    base, bits = _literal(body[0])
    result = _scalar(body[1], PowFOp, (base, source), "f32", frozenset({"fastmath"}))
    _yield(body[2], result)
    return "literal_base_pow_f32", (bits,)


def _recognize_tanh_gelu(body, source):
    from xdsl.dialects.arith import AddfOp, MulfOp
    from xdsl.dialects.math import TanhOp

    if len(body) != 14:
        raise InvalidLinalgPattern("tanh GELU body is not the closed polynomial and yield")
    literals = tuple(_literal(op) for op in body[:4])
    bits = tuple(value for _, value in literals)
    if bits != _TANH_GELU_LITERALS:
        raise InvalidLinalgPattern("tanh GELU has different f32 coefficient bits")
    half, one, cubic, scale = (result for result, _ in literals)
    sequence = (
        (MulfOp, (source, source)),
        (MulfOp, (4, source)),
        (MulfOp, (cubic, 5)),
        (AddfOp, (source, 6)),
        (MulfOp, (scale, 7)),
        (TanhOp, (8,)),
        (AddfOp, (one, 9)),
        (MulfOp, (half, source)),
        (MulfOp, (11, 10)),
    )
    results: dict[int, object] = {}
    for index, (expected_class, operands) in enumerate(sequence, start=4):
        resolved = tuple(results[value] if type(value) is int else value for value in operands)
        results[index] = _scalar(body[index], expected_class, resolved, "f32", frozenset({"fastmath"}))
    _yield(body[13], results[12])
    return "tanh_gelu_f32", bits


def recognize_static_composite_math_body(op) -> StaticCompositeMathPattern:
    """Prove one exact typed source form, without admission or numeric claims."""
    shell = _checked_static_linalg_shell(op)
    if len(op.inputs) != 1:
        raise InvalidLinalgPattern("composite math source requires one input tensor")
    types = shell.ordered_types
    body = shell.body
    names = tuple(inner.name for inner in body)
    if names == ("arith.constant", "arith.maximumf", "arith.constant", "arith.minimumf", "linalg.yield"):
        if types != ("f32", "f32", "f32"):
            raise InvalidLinalgPattern("interval clamp requires f32 input/init/result")
        operation, literals = _recognize_interval_clamp(body, shell.args[0])
    elif names[:2] in (("arith.constant", "arith.minimumf"), ("arith.constant", "arith.maximumf")):
        if types != ("f32", "f32", "f32"):
            raise InvalidLinalgPattern("literal clamp requires f32 input/init/result")
        operation, literals = _recognize_clamp(body, shell.args[0])
    elif names[:3] == ("arith.constant", "arith.cmpf", "arith.select"):
        if types != ("f32", "f32", "f32"):
            raise InvalidLinalgPattern("literal compare/select clamp requires f32 input/init/result")
        operation, literals = _recognize_compare_select_clamp(body, shell.args[0])
    elif names == (
        "arith.constant",
        "arith.sitofp",
        "arith.cmpf",
        "arith.select",
        "arith.constant",
        "arith.sitofp",
        "arith.cmpf",
        "arith.select",
        "linalg.yield",
    ):
        if types != ("f32", "f32", "f32"):
            raise InvalidLinalgPattern("int8-grid compare/select clamp requires f32 input/init/result")
        operation, literals = _recognize_symmetric_int8_grid_clamp(body, shell.args[0])
    elif names == ("arith.fptosi", "linalg.yield"):
        if types != ("f32", "i8", "i8"):
            raise InvalidLinalgPattern("int8 cast requires f32 input and i8 init/result")
        operation, literals = _recognize_int8_grid_cast(op, body, shell.args[0], tuple(shell.shape))
    elif names[:2] == ("arith.constant", "arith.divf"):
        if types != ("f32", "f32", "f32"):
            raise InvalidLinalgPattern("f32 reciprocal requires f32 input/init/result")
        operation, literals = _recognize_reciprocal(body, shell.args[0], signed_i64=False)
    elif names[:2] == ("arith.sitofp", "arith.constant"):
        if types != ("i64", "f32", "f32"):
            raise InvalidLinalgPattern("signed reciprocal requires i64 input and f32 init/result")
        operation, literals = _recognize_reciprocal(body, shell.args[0], signed_i64=True)
    elif names[:2] == ("arith.constant", "math.powf"):
        if types != ("f32", "f32", "f32"):
            raise InvalidLinalgPattern("literal-base power requires f32 input/init/result")
        operation, literals = _recognize_pow(body, shell.args[0])
    elif names[:4] == ("arith.constant",) * 4:
        if types != ("f32", "f32", "f32"):
            raise InvalidLinalgPattern("tanh GELU requires f32 input/init/result")
        operation, literals = _recognize_tanh_gelu(body, shell.args[0])
    else:
        raise InvalidLinalgPattern("composite math source has no closed recognized body")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL is the source grammar and type verifier
        raise InvalidLinalgPattern(f"composite math source failed xDSL verification: {exc}") from exc
    return StaticCompositeMathPattern(SCHEMA, operation, shell.shape, types, literals, _FORMS[operation][2])


def validate_serialized_static_composite_math_pattern(record: object) -> StaticCompositeMathPattern:
    """Recheck the closed, literal-bit-preserving source description."""
    if not isinstance(record, Mapping) or set(record) != set(StaticCompositeMathPattern.__dataclass_fields__):
        raise InvalidLinalgPattern("serialized composite math pattern has missing or extra fields")
    operation = record.get("operation")
    if record.get("schema") != SCHEMA or type(operation) is not str or operation not in _FORMS:
        raise InvalidLinalgPattern("serialized composite math pattern has an unknown form")
    expected_types, n_literals, obligations = _FORMS[operation]
    shape, ordered_types = record.get("shape"), record.get("ordered_types")
    if (
        not isinstance(shape, (tuple, list))
        or not shape
        or any(type(dim) is not int or dim < 0 for dim in shape)
        or not isinstance(ordered_types, (tuple, list))
        or tuple(ordered_types) != expected_types
    ):
        raise InvalidLinalgPattern("serialized composite math shape or typed roster is invalid")
    literal_bits = record.get("literal_bits")
    if not isinstance(literal_bits, (tuple, list)) or len(literal_bits) != n_literals:
        raise InvalidLinalgPattern("serialized composite math literal roster is invalid")
    bits: list[str] = []
    for literal in literal_bits:
        if type(literal) is not str or len(literal) != 10 or not literal.startswith("0x"):
            raise InvalidLinalgPattern("serialized composite math literal is not canonical f32 bits")
        try:
            value = int(literal[2:], 16)
        except ValueError as exc:
            raise InvalidLinalgPattern("serialized composite math literal is not hexadecimal") from exc
        if literal != f"0x{value:08x}" or not isfinite(unpack("<f", pack("<I", value))[0]):
            raise InvalidLinalgPattern("serialized composite math literal is not a finite f32 bit pattern")
        bits.append(literal)
    if operation.startswith("reciprocal_") and tuple(bits) != (_POSITIVE_ONE,):
        raise InvalidLinalgPattern("serialized reciprocal does not use exact positive one")
    if operation in _EXACT_LITERALS and tuple(bits) != _EXACT_LITERALS[operation]:
        raise InvalidLinalgPattern("serialized quantization clamp bounds differ from their exact literals")
    if operation in {"clamp_literal_interval_f32", "fptosi_i8_of_rounded_literal_interval"}:
        _interval_values(bits, int8=operation == "fptosi_i8_of_rounded_literal_interval")
    if operation == "tanh_gelu_f32" and tuple(bits) != _TANH_GELU_LITERALS:
        raise InvalidLinalgPattern("serialized tanh GELU coefficient bits differ")
    listed = record.get("lowering_intrinsic_obligations")
    if not isinstance(listed, (tuple, list)) or tuple(listed) != obligations:
        raise InvalidLinalgPattern("serialized composite math intrinsic obligations differ")
    return StaticCompositeMathPattern(SCHEMA, operation, tuple(shape), expected_types, tuple(bits), obligations)


def screen_static_composite_math_source(path: Path, ordinals: tuple[int, ...]) -> StaticCompositeMathSource:
    """Bind requested operations to full raw/normalized source bytes and ordinals."""
    raw, normalized, evidence = _screen_static_linalg_source(path, ordinals, recognize_static_composite_math_body)
    return StaticCompositeMathSource(raw, normalized, evidence)
