"""Exact byte quantization of binary32 inputs by integer threshold search.

Each table entry is the first ordered IEEE code above an exact rational half
step (inclusive only when RNE chooses the upper code). Eight integer bisections
then implement RNE, zero point and saturation without a reciprocal or quotient.
"""

import math
import struct
from fractions import Fraction


def ordered_float(key):
    raw = key ^ 0x80000000 if key & 0x80000000 else ~key & 0xFFFFFFFF
    return struct.unpack("<f", raw.to_bytes(4, "little"))[0]


def thresholds(scale, zero_point, low, high):
    scale = Fraction(scale)
    if scale <= 0 or not ((-128 <= low < high <= 127) or (0 <= low < high <= 255)) or not low <= zero_point <= high:
        raise ValueError("exact quantization requires positive scale and signed or unsigned byte bounds")
    result = []
    for code in range(low, high):
        midpoint = Fraction(2 * (code - zero_point) + 1, 2) * scale
        inclusive = bool((code - zero_point) % 2)
        left, right = 0x007FFFFF, 0xFF800000  # ordered -inf and +inf
        while left < right:
            mid = (left + right) // 2
            value = ordered_float(mid)
            above = (
                value > 0
                if math.isinf(value)
                else (Fraction(value) >= midpoint if inclusive else Fraction(value) > midpoint)
            )
            if above:
                right = mid
            else:
                left = mid + 1
        result.append(left)
    return result + [0xFFFFFFFF]  # safe final lookup once bisection has converged


def lower(op):
    """Lower a literal-scale, signed-i8 exact quantizer; refuse contradictory metadata."""
    from xdsl.dialects import arith, cf, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        IndexType,
        IntegerAttr,
        TensorType,
        f32,
        i32,
    )
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    attrs = op.attributes
    nums = attrs["prov.quant.scale_numerator"].data
    dens = attrs["prov.quant.scale_denominator"].data
    if len(nums) != 1 or len(dens) != 1 or int(dens[0].data) <= 0:
        raise ValueError("exact quantization needs one positive rational denominator")
    scale = Fraction(int(nums[0].data), int(dens[0].data))
    if Fraction(_constant_of(op.operands[1])) != scale:
        raise ValueError("exact quantization scale operand differs from rational provenance")
    zp = _constant_of(op.operands[2])
    low, high = [op.properties[k].value.data for k in ("quant_min", "quant_max")]
    value = op.operands[0]
    if value.type.element_type != f32 or str(op.results[0].type.element_type) != "i8" or type(zp) is not int:
        raise ValueError("exact quantization currently requires binary32 to signed i8")
    for key, expected in (("zero_point", zp), ("quant_min", low), ("quant_max", high)):
        if attrs[f"prov.quant.{key}"].value.data != expected:
            raise ValueError("exact quantization parameters differ from provenance")
    storage = getattr(attrs.get("prov.quant.storage_dtype"), "data", None)
    limits = {"int8": (-128, 127), "uint8": (0, 255)}.get(storage)
    if limits is None or not limits[0] <= low < high <= limits[1]:
        raise ValueError("exact quantization range differs from its byte storage contract")
    table = thresholds(scale, zp, low, high)
    table_type = TensorType(i32, [len(table)])
    data = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(table_type, table))
    empty = tensor.EmptyOp((), op.results[0].type)
    body = Block(arg_types=[f32, op.results[0].type.element_type])

    def emit(operation):
        body.add_op(operation)
        return operation.results[0]

    def c(v):
        return emit(arith.ConstantOp(IntegerAttr(v, i32)))

    raw = emit(arith.BitcastOp(body.args[0], i32))
    magnitude = emit(arith.AndIOp(raw, c(0x7FFFFFFF)))
    finite_or_inf = emit(arith.CmpiOp(magnitude, c(0x7F800000), "ule"))
    nan_storage_zero = "merlin.quantizer_nan_storage_zero" in attrs
    if nan_storage_zero and not low <= 0 <= high:
        raise ValueError("storage-zero NaN requires zero inside the declared code range")
    if not nan_storage_zero:
        body.add_op(cf.AssertOp(finite_or_inf, "exact quantization refuses NaN"))
    negative = emit(arith.CmpiOp(raw, c(0), "slt"))
    key = emit(arith.XOrIOp(raw, emit(arith.SelectOp(negative, c(0xFFFFFFFF), c(0x80000000)))))
    lo, hi, one = c(0), c(high - low), c(1)
    for _ in range((high - low).bit_length()):
        mid = emit(arith.ShRUIOp(emit(arith.AddiOp(lo, hi)), one))
        index = emit(arith.IndexCastOp(mid, IndexType()))
        threshold = emit(tensor.ExtractOp(data.results[0], [index], i32))
        above = emit(arith.AndIOp(emit(arith.CmpiOp(key, threshold, "uge")), emit(arith.CmpiOp(lo, hi, "ult"))))
        lo = emit(arith.SelectOp(above, emit(arith.AddiOp(mid, one)), lo))
        hi = emit(arith.SelectOp(above, hi, mid))
    code = emit(arith.AddiOp(lo, c(low)))
    if nan_storage_zero:
        code = emit(arith.SelectOp(finite_or_inf, code, c(0)))
    narrowed = emit(arith.TruncIOp(code, body.args[1].type))
    body.add_op(linalg.YieldOp(narrowed))
    rank = len(value.type.get_shape())
    maps = [AffineMapAttr(AffineMap.identity(rank))] * 2
    generic = linalg.GenericOp(
        inputs=[value],
        outputs=[empty.tensor],
        body=Region(body),
        indexing_maps=ArrayAttr(maps),
        iterator_types=ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * rank),
        result_types=[op.results[0].type],
    )
    return (data, empty, generic), generic
