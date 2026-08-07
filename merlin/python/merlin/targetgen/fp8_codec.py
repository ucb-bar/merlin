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
