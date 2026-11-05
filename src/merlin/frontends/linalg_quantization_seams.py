"""Closed source proofs for lowered scalar reciprocal and FP32-to-int8 quantization.

These proofs preserve the parsed scalar algorithm. They do not certify finite
inputs, frontend equivalence, the selected rounding implementation or native
execution. Those obligations remain with the numerical and linked-image checks.
"""
from dataclasses import dataclass

from merlin.frontends.linalg_composite_math import _literal, _scalar, _yield
from merlin.transforms.dense_ir import InvalidLinalgPattern, checked_static_linalg_shell

SCHEMA = "merlin.static_quantization_seam_body.v1"
TYPES = {
    "scalar_reciprocal_f32": ("f32", "f32", "f32"),
    "roundeven_clamp_f32_to_i8": ("f32", "f32", "i64", "i8", "i8"),
}


def validate(declaration):
    if (not isinstance(declaration, dict) or set(declaration) != {"schema", "operation"}
            or declaration.get("schema") != SCHEMA or declaration.get("operation") not in TYPES):
        raise InvalidLinalgPattern("quantization seam needs a closed typed declaration")
    return dict(declaration)


@dataclass(frozen=True)
class QuantizationPattern:
    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_maps: tuple[str, ...]
    literal_bits: tuple[str, ...]


def recognize(op):
    from xdsl.dialects import arith, math
    from xdsl.ir.affine import AffineMap

    shell = checked_static_linalg_shell(op, singleton_projection_inputs=True, allow_rank_zero=True)
    body, args = shell.body, shell.args
    shapes = tuple(tuple(v.type.get_shape()) for v in op.inputs)
    if shell.ordered_types == TYPES["scalar_reciprocal_f32"] and shapes == ((),) and shell.shape == ():
        if len(body) != 3:
            raise InvalidLinalgPattern("scalar reciprocal must be constant, division, yield")
        one, bits = _literal(body[0])
        if bits != "0x3f800000":
            raise InvalidLinalgPattern("scalar reciprocal numerator must be exactly one")
        result = _scalar(body[1], arith.DivfOp, (one, args[0]), "f32", frozenset({"fastmath"}))
        _yield(body[2], result)
        operation, literals = "scalar_reciprocal_f32", (bits,)
    elif shell.ordered_types == TYPES["roundeven_clamp_f32_to_i8"]:
        if (not shell.shape or shapes != (shell.shape, (), ()) or len(body) != 8
                or shell.input_maps[0] != AffineMap.identity(len(shell.shape))):
            raise InvalidLinalgPattern("quantization needs identity data and two rank-zero scalar inputs")
        mul, rounding, convert, add, lower, upper, cast, yld = body
        if len(lower.operands) != 2 or len(upper.operands) != 2:
            raise InvalidLinalgPattern("quantization clamp arity differs")
        lo, lo_bits = _literal(lower.operands[1].owner)
        hi, hi_bits = _literal(upper.operands[1].owner)
        if (lo_bits, hi_bits) != ("0xc3000000", "0x42fe0000"):
            raise InvalidLinalgPattern("quantization bounds must be exactly -128 and 127")
        flags = frozenset({"fastmath"})
        product = _scalar(mul, arith.MulfOp, (args[0], args[1]), "f32", flags)
        rounded = _scalar(rounding, math.RoundEvenOp, (product,), "f32", flags)
        zero = _scalar(convert, arith.SIToFPOp, (args[2],), "f32")
        shifted = _scalar(add, arith.AddfOp, (rounded, zero), "f32", flags)
        clamped_lo = _scalar(lower, arith.MaximumfOp, (shifted, lo), "f32", flags)
        clamped_hi = _scalar(upper, arith.MinimumfOp, (clamped_lo, hi), "f32", flags)
        result = _scalar(cast, arith.FPToSIOp, (clamped_hi,), "i8")
        _yield(yld, result)
        operation, literals = "roundeven_clamp_f32_to_i8", (lo_bits, hi_bits)
    else:
        raise InvalidLinalgPattern("source is not an exact lowered FP32 quantization seam")
    return QuantizationPattern(operation, shell.shape, shell.ordered_types, shapes,
                               tuple(str(m) for m in shell.input_maps), literals)
