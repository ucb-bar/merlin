"""Round-half-to-even then saturate to an integer range, as cheap emitted code.

WHY THIS MODULE EXISTS. Quantizing an activation is `clamp(roundeven(x * inv_scale) + zp, qmin, qmax)`
-- the expression `merlin.llvmlower.passes_xdsl.lower_quant_ext` emits, and the one TorchAO PT2E
specifies. A backend whose target dialect lacks a float-to-int instruction has to *build* the
conversion, and the obvious construction is expensive: reconstruct the IEEE fields, shift by a clamped
amount, and separately blend two magic-constant cases for the rounding. MEASURED on a whole-model
W8A8 ResNet-50 CPU lane, that construction was ~61 emitted operations and ~49 of the 69 instructions
retired per output element, on a path that was ~100% of runtime.

This module states the cheap construction ONCE, as executable arithmetic plus the algebra that makes
it exact, so an emitter can implement it and a test can hold it to the reference instead of each
backend rediscovering (or mis-deriving) it. Nothing here is target-specific: no dtype set, no dialect,
no instruction. It is the identity, not a lowering.

THE IDENTITY. For a float `x` pre-clamped into `[-bound, bound]` with `bound <= 2**21`:

    y = x + 1.5 * 2**23          lands in [2**23, 2**24), where the f32 ulp is EXACTLY 1,
                                 so this add itself rounds to nearest-EVEN
    int(roundeven(x)) == (bits(y) & 0x7FFFFF) - 2**22

because in that binade `bits(y) & 0x7FFFFF == y - 2**23 == x + 2**22`. Two consequences follow, and
both are the point: the rounding needs no exponent blend (the `|x| >= 2**23` case a general
`round_near_even` must handle cannot occur), and the conversion needs no exponent reconstruction (the
mantissa field already IS the integer).

THE PRE-CLAMP MUST BE WIDER THAN THE FINAL RANGE. This is the easy thing to get wrong. Clamping to
`[qmin, qmax]` BEFORE rounding changes the answer: `127.6` must round to `128` and then saturate to
`127`, but pre-clamped to `127.0` it rounds to `127`. So the caller clamps the returned INTEGER, and
the float pre-clamp exists only to make the magic-constant identity applicable. Any `|x| > bound`
saturates either way, because `bound` exceeds the integer range and the clamp is monotone.

NOT AN APPROXIMATION. `roundeven` is IEEE round-to-nearest-even, which is what `math.roundeven`,
`numpy.rint`, and the Gemmini header's `ROUND_NEAR_EVEN` all compute; ties go to even in every case.
"""
from __future__ import annotations

import struct

__all__ = ["MAGIC", "MAX_BOUND", "MANTISSA_MASK", "MANTISSA_BIAS",
           "roundeven_to_int_bounded", "quantize_affine", "bound_is_usable"]

#: `1.5 * 2**23`. Adding it to a float in `[-2**21, 2**21]` lands the sum in `[2**23, 2**24)`.
MAGIC = 12582912.0
#: The largest pre-clamp bound for which the identity holds.
MAX_BOUND = 2097152.0                    # 2**21
MANTISSA_MASK = 0x7FFFFF
MANTISSA_BIAS = 1 << 22


def bound_is_usable(bound: float, qmin: int, qmax: int) -> bool:
    """Is ``bound`` both wide enough to preserve round-then-clamp and inside the magic binade?

    Wide enough means strictly outside the integer range, so a value that rounds ONTO a limit is not
    pre-clamped to it. Fail-closed: a caller that cannot satisfy both must emit the general
    construction rather than a cheaper wrong one.
    """
    return 0.0 < bound <= MAX_BOUND and abs(qmin) < bound and abs(qmax) < bound


def roundeven_to_int_bounded(x: float, *, bound: float = MAX_BOUND) -> int:
    """``int(roundeven(x))`` via the magic constant, for ``x`` pre-clamped to ``[-bound, bound]``.

    The Python reference for what an emitter should emit. Values outside the bound saturate to
    ``+/-int(bound)``, which the caller's integer clamp then maps to the same limit the unclamped
    value would have reached.
    """
    if not 0.0 < bound <= MAX_BOUND:
        raise ValueError(f"bound {bound} must be in (0, 2**21] for the magic-constant identity")
    clamped = _f32(min(max(x, -bound), bound))
    y = _f32(clamped + MAGIC)
    return (_bits(y) & MANTISSA_MASK) - MANTISSA_BIAS


def quantize_affine(x: float, inv_scale: float, zero_point: int,
                    qmin: int, qmax: int, *, bound: float = MAX_BOUND) -> int:
    """``clamp(roundeven(x * inv_scale) + zero_point, qmin, qmax)`` through the cheap construction.

    Reciprocal-FIRST, matching TorchAO's specified order (see ``lower_quant_ext``): ``x * (1/scale)``
    differs from ``x / scale`` at some rounding boundaries, so which one is written is load-bearing
    and this takes the reciprocal as its argument rather than the scale.
    """
    if not bound_is_usable(bound, qmin, qmax):
        raise ValueError(f"bound {bound} cannot preserve round-then-clamp for [{qmin}, {qmax}]")
    return min(max(roundeven_to_int_bounded(_f32(_f32(x) * _f32(inv_scale)), bound=bound)
                   + zero_point, qmin), qmax)


def _f32(value: float) -> float:
    """``value`` rounded to f32, so this reference rounds where the emitted f32 code rounds."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]
