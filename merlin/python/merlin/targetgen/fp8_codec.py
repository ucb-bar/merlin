"""OCP microscaling FP8 (E4M3) + E8M0 block-scale codecs — a target-agnostic Python port of the
hardware-semantics reference ``mx_fp_math.h`` (radiance-kernels ``lib/golden/mx_fp_math.h``, itself a
BSD-3 port of Spike/riscv-isa-sim's MX primitives).

Used to turn a golden's ``decoded`` fp8 palette values back into their exact 1-byte fp8 e4m3 codes (the
device preload a block-scaled matmul needs), and to decode an E8M0 block-scale code to its power-of-two
scale. The FP8 grid is standard OCP e4m3 (bias 7, one extra binade at the top, no infinities), so this
names no target — any mxfp8 datapath shares it.

Reference anchors (mx_fp_math.h): ``fp8_e4m3_to_code`` (lines 205-224), ``fp8_e4m3_decode`` (227-236),
``fpe8m0_decode`` (249-252), ``round_half_to_even`` (196-203).
"""
from __future__ import annotations

import math

_E4M3_BIAS = 7
_E4M3_EMIN = -6
_E4M3_EMAX = 8      # OCP e4m3 carries one binade past the nominal bias (max normal 448)


def _round_half_to_even(x: float) -> int:
    """Banker's rounding — matches the C reference's ``round_half_to_even`` (Python 3 ``int(round(x))``)."""
    fl = math.floor(x)
    frac = x - fl
    fli = int(fl)
    if frac < 0.5:
        return fli
    if frac > 0.5:
        return fli + 1
    return fli + 1 if (fli & 1) else fli


def fp8_e4m3_decode(code: int) -> float:
    """Decode a 1-byte E4M3 code (1 sign | 4 exp | 3 mant, bias 7) to a Python float."""
    code &= 0xFF
    s = (code >> 7) & 1
    e = (code >> 3) & 0xF
    m = code & 0x7
    if e == 0:
        val = (m / 8.0) * (2.0 ** (1 - _E4M3_BIAS))
    else:
        val = (1.0 + m / 8.0) * (2.0 ** (e - _E4M3_BIAS))
    return -val if s else val


def fp8_e4m3_encode(v: float) -> int:
    """Encode a float to its 1-byte E4M3 code (round-half-to-even), saturating off-grid magnitudes.
    Faithful inverse of :func:`fp8_e4m3_decode` across the WHOLE e4m3 grid — including the subnormal
    binade (``e==0``: value ``m * 2^-9``) — so re-encoding a golden's decoded palette is lossless. (The
    upstream ``fp8_e4m3_to_code`` flushes subnormals to zero; that is lossy for a real fp8 operand palette,
    which is why the device preload is reconstructed here rather than with the reference encoder.)"""
    if v == 0.0 or not math.isfinite(v):
        return 0
    s = 1 if math.copysign(1.0, v) < 0 else 0
    av = abs(v)
    E = int(math.floor(math.log2(av)))
    if E < _E4M3_EMIN:
        # subnormal grid: decode(e=0, m) = (m/8) * 2^(1-bias) = m * 2^-9, m in 1..7
        m = _round_half_to_even(av * (2.0 ** (_E4M3_BIAS + 3 - 1)))   # av / 2^-9
        if m <= 0:
            return 0
        if m >= 8:   # rounded up to the smallest normal (2^-6): e=1, m=0
            return (s << 7) | (((1) & 0xF) << 3)
        return (s << 7) | (m & 0x7)
    if E > _E4M3_EMAX:
        e_used, mant = _E4M3_EMAX, 6
    else:
        e_used = E
        base = 2.0 ** e_used
        delta = base / 8.0
        k = _round_half_to_even((av - base) / delta)
        if k >= 8:
            e_used += 1
            k = 0
            if e_used > _E4M3_EMAX:
                e_used, k = _E4M3_EMAX, 6
        else:
            hi = 6 if e_used == _E4M3_EMAX else 7
            k = min(max(k, 0), hi)
        mant = k
    return (s << 7) | (((e_used + _E4M3_BIAS) & 0xF) << 3) | (mant & 0x7)


def e8m0_decode(code: int) -> float:
    """Decode an E8M0 block-scale code to its power-of-two scale ``2^(code-127)`` (0xFF is NaN)."""
    code &= 0xFF
    if code == 0xFF:
        return math.nan
    return 2.0 ** (code - 127)


def fp8_e4m3_encode_row(values) -> list[int]:
    """Encode a flat sequence of floats to their E4M3 codes (row-major, for a tensor preload)."""
    return [fp8_e4m3_encode(float(v)) for v in values]


# ---------------------------------------------------------------------------
# Sub-byte OCP microscaling grids: FP6 (E3M2) and FP4 (E2M1)
# ---------------------------------------------------------------------------
# These two formats share the E4M3 layout convention (1 sign | E exp | M mant,
# bias ``2^(E-1)-1``, subnormal binade at ``e==0``) but reserve NO code for
# infinity/NaN -- every bit pattern is a finite value, so the top exponent is a
# normal binade. The reference decoders are cyclotron's ``fp_math::fp6_e3m2_decode``
# / ``fp4_e2m1_decode`` (radiance mxgemmini co-model), themselves ports of the MX
# RTL semantics. As with FP8 the encoder is a FAITHFUL inverse across the whole
# grid -- including the subnormal binade -- so re-encoding an operand palette that
# a golden already decoded is lossless (0 round-trip errors). Any mxfp6/mxfp4
# datapath shares these grids, so this names no target.


def _e8m0_decode(code: int, exp_bits: int) -> float:
    """Decode an UNSIGNED, MANTISSA-LESS OCP code: the whole field is a biased exponent.

    This is the MX block-SCALE type (E8M0), not an operand format. It has no sign bit, no mantissa and
    no zero -- the smallest code is ``2^(1-bias)``, not 0 -- and the all-ones code is reserved for NaN.
    That is a different grid from :func:`_ocp_fpx_decode`'s sign|exp|mant layout, which is why it is a
    separate function rather than a flag on that one: sharing the code path would mean threading "does
    this format have a sign bit, a mantissa, a subnormal binade, a NaN code" through every branch of an
    encoder whose whole structure assumes it does.
    """
    bias = (1 << (exp_bits - 1)) - 1
    code &= (1 << exp_bits) - 1
    if code == (1 << exp_bits) - 1:
        return float("nan")
    return 2.0 ** (code - bias)


def _e8m0_encode(v: float, exp_bits: int) -> int:
    """Encode a positive value to the nearest power-of-two E8M0 code, saturating off-grid.

    Raises for a negative value rather than encoding its magnitude. An unsigned format cannot represent
    a sign, and silently dropping one would turn a wrong-signed scale into plausible wrong numbers --
    the same failure this module's own header describes for e4m3-encoded-as-e5m2.
    """
    if v < 0.0:
        raise ValueError(f"E8M0 is unsigned and cannot represent {v!r}; a negative block scale is not "
                         f"a representable value, and encoding |v| would silently drop the sign")
    bias = (1 << (exp_bits - 1)) - 1
    top = (1 << exp_bits) - 2                  # all-ones is NaN, so the largest finite code is one less
    if not math.isfinite(v):
        return (1 << exp_bits) - 1             # NaN
    if v == 0.0:
        return 0                               # no zero on this grid; saturate to the smallest scale
    code = _round_half_to_even(math.log2(v)) + bias
    return int(min(max(code, 0), top))


def _ocp_fpx_decode(code: int, exp_bits: int, mant_bits: int) -> float:
    """Decode an OCP FPx code with ``exp_bits`` exponent and ``mant_bits`` mantissa
    bits (bias ``2^(exp_bits-1)-1``, subnormal binade at ``e==0``, no inf/NaN)."""
    bias = (1 << (exp_bits - 1)) - 1
    width = 1 + exp_bits + mant_bits
    code &= (1 << width) - 1
    s = (code >> (exp_bits + mant_bits)) & 1
    e = (code >> mant_bits) & ((1 << exp_bits) - 1)
    m = code & ((1 << mant_bits) - 1)
    scale = float(1 << mant_bits)
    if e == 0:
        val = (m / scale) * (2.0 ** (1 - bias))
    else:
        val = (1.0 + m / scale) * (2.0 ** (e - bias))
    return -val if s else val


def _ocp_fpx_encode(v: float, exp_bits: int, mant_bits: int) -> int:
    """Encode a float to its OCP FPx code (round-half-to-even), saturating off-grid
    magnitudes to the largest finite value. Faithful inverse of
    :func:`_ocp_fpx_decode` across the whole grid, subnormal binade included."""
    bias = (1 << (exp_bits - 1)) - 1
    emin = 1 - bias
    emax = ((1 << exp_bits) - 1) - bias  # top exponent is a normal binade (no inf/NaN)
    mant_max = (1 << mant_bits) - 1
    if v == 0.0 or not math.isfinite(v):
        return 0
    s = 1 if math.copysign(1.0, v) < 0 else 0
    sign = s << (exp_bits + mant_bits)
    av = abs(v)
    E = int(math.floor(math.log2(av)))
    if E < emin:
        # subnormal grid: decode(e=0, m) = (m / 2^mant_bits) * 2^(1-bias) = m * 2^(emin-mant_bits)
        m = _round_half_to_even(av * (2.0 ** (mant_bits - emin)))
        if m <= 0:
            return 0
        if m > mant_max:  # rounded up to the smallest normal (e=1, m=0)
            return sign | (1 << mant_bits)
        return sign | (m & mant_max)
    if E > emax:
        e_used, mant = emax, mant_max
    else:
        e_used = E
        base = 2.0 ** e_used
        delta = base / (1 << mant_bits)
        k = _round_half_to_even((av - base) / delta)
        if k > mant_max:
            e_used += 1
            k = 0
            if e_used > emax:
                e_used, k = emax, mant_max
        else:
            k = min(max(k, 0), mant_max)
        mant = k
    return sign | (((e_used + bias) & ((1 << exp_bits) - 1)) << mant_bits) | (mant & mant_max)


def ocp_encode(v: float, exp_bits: int, mant_bits: int, *, signed: bool = True) -> int:
    """Encode one value to an OCP fp code of ANY (exp_bits, mant_bits) width — the public form of the
    generic encoder the named e4m3/fp6/fp4 helpers below are thin wrappers over.

    Exposed because a caller that knows a format only through the quant-format registry has its
    ``exp_bits``/``mant_bits`` and no reason to know which named helper corresponds. Without this, such a
    caller either name-matches formats (which is how e5m2 gets encoded as e4m3) or gives up.

    ``signed`` comes from the registry entry and must be passed, not assumed. The sign bit used to be
    unconditional, which made the returned code ``exp_bits + mant_bits + 1`` wide — for the unsigned,
    mantissa-less block-scale type that is 9 bits, and packing it into a byte raises. A caller that
    knows the format only through the registry has ``signed`` for exactly this reason.
    """
    if not signed:
        if mant_bits:
            raise ValueError(f"no unsigned OCP grid with a mantissa is defined "
                             f"(exp_bits={exp_bits}, mant_bits={mant_bits}); the only unsigned OCP "
                             f"format is the mantissa-less block scale")
        return _e8m0_encode(v, exp_bits)
    return _ocp_fpx_encode(v, exp_bits, mant_bits)


def ocp_decode(code: int, exp_bits: int, mant_bits: int, *, signed: bool = True) -> float:
    """Decode an OCP fp code of ANY width — the inverse of :func:`ocp_encode`."""
    if not signed:
        if mant_bits:
            raise ValueError(f"no unsigned OCP grid with a mantissa is defined "
                             f"(exp_bits={exp_bits}, mant_bits={mant_bits})")
        return _e8m0_decode(code, exp_bits)
    return _ocp_fpx_decode(code, exp_bits, mant_bits)


def fp6_e3m2_decode(code: int) -> float:
    """Decode a 6-bit FP6 E3M2 code (1 sign | 3 exp | 2 mant, bias 3) to a Python float."""
    return _ocp_fpx_decode(code, exp_bits=3, mant_bits=2)


def fp6_e3m2_encode(v: float) -> int:
    """Encode a float to its 6-bit FP6 E3M2 code (round-half-to-even). Faithful inverse of
    :func:`fp6_e3m2_decode` across the whole grid (subnormal binade included)."""
    return _ocp_fpx_encode(v, exp_bits=3, mant_bits=2)


def fp4_e2m1_decode(code: int) -> float:
    """Decode a 4-bit FP4 E2M1 code (1 sign | 2 exp | 1 mant, bias 1) to a Python float."""
    return _ocp_fpx_decode(code, exp_bits=2, mant_bits=1)


def fp4_e2m1_encode(v: float) -> int:
    """Encode a float to its 4-bit FP4 E2M1 code (round-half-to-even). Faithful inverse of
    :func:`fp4_e2m1_decode` across the whole grid (subnormal binade included)."""
    return _ocp_fpx_encode(v, exp_bits=2, mant_bits=1)


def fp6_e3m2_encode_row(values) -> list[int]:
    """Encode a flat sequence of floats to their FP6 E3M2 codes (row-major, for a preload)."""
    return [fp6_e3m2_encode(float(v)) for v in values]


def fp4_e2m1_encode_row(values) -> list[int]:
    """Encode a flat sequence of floats to their FP4 E2M1 codes (row-major, for a preload)."""
    return [fp4_e2m1_encode(float(v)) for v in values]
