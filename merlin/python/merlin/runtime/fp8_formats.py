"""Format-aware fp8 decode + representable-value enumeration (OCP float8: e4m3fn and e5m2).

``fp8`` is not one format: **e4m3** (1 sign, 4 exp bias-7, 3 mantissa; no inf, NaN=S.1111.111, max 448)
and **e5m2** (1 sign, 5 exp bias-15, 2 mantissa; IEEE-like inf/NaN, max 57344) decode the SAME raw byte to
DIFFERENT floats. A grader that assumes one format silently mis-values operands of the other — so the
format must be carried, not assumed. This module is the single source of truth for the byte<->float
mapping (``runtime.dispatch_runtime`` and the corpus operand synthesis both use it), keyed on the format
token so a new format is one table row, never a per-caller branch.
"""
from __future__ import annotations

import numpy as np

# (exp_bits, mantissa_bits, exp_bias, has_inf) per canonical token. Byte width is always 1.
_FP8 = {
    "fp8_e4m3": (4, 3, 7, False),   # OCP float8_e4m3fn: finite-only, NaN = all-exp-1 & all-mantissa-1
    "fp8_e5m2": (5, 2, 15, True),   # OCP float8_e5m2: IEEE-like, inf = all-exp-1 & mantissa-0
}
# accepted aliases (MLIR/torch spellings) -> canonical token
_ALIASES = {"f8E4M3FN": "fp8_e4m3", "f8e4m3fn": "fp8_e4m3", "e4m3": "fp8_e4m3",
            "f8E5M2": "fp8_e5m2", "f8e5m2": "fp8_e5m2", "e5m2": "fp8_e5m2"}


def canonical_fp8(fmt: str) -> str:
    """Canonical fp8 token for a spelling, or raise for a non-fp8 / unknown format (fail closed)."""
    t = _ALIASES.get(fmt, fmt)
    if t not in _FP8:
        raise KeyError(f"unknown fp8 format {fmt!r} (known: {sorted(_FP8)})")
    return t


def fp8_to_f32(u8: np.ndarray, fmt: str) -> np.ndarray:
    """Decode fp8 bytes to float32 under the named format. Subnormals (exp==0), normals, and the
    format's inf/NaN encodings are all handled; the mapping is derived from (exp_bits, mantissa_bits,
    bias, has_inf), never hardcoded per format."""
    eb, mb, bias, has_inf = _FP8[canonical_fp8(fmt)]
    mmax = (1 << mb)                                   # mantissa denominator (e.g. 8 for e4m3)
    emax = (1 << eb) - 1                               # all-ones exponent
    u = np.ascontiguousarray(u8, np.uint8).astype(np.uint32)
    sign = np.where((u >> 7) & 1 == 1, np.float32(-1.0), np.float32(1.0))
    exp = (u >> mb) & emax
    man = (u & (mmax - 1)).astype(np.float32)
    sub = (man / mmax) * np.float32(2.0 ** (1 - bias))            # exp==0: subnormal
    nrm = (1.0 + man / mmax) * np.exp2(exp.astype(np.float32) - bias)
    val = (sign * np.where(exp == 0, sub, nrm)).astype(np.float32)
    top = (exp == emax)
    if has_inf:                                        # e5m2: exp all-1 -> inf (man 0) / NaN (man != 0)
        val = np.where(top & (man == 0), sign * np.float32(np.inf), val)
        val = np.where(top & (man != 0), np.float32(np.nan), val)
    else:                                              # e4m3fn: only all-1 exp & all-1 mantissa is NaN
        val = np.where(top & (man == (mmax - 1)), np.float32(np.nan), val)
    return val


def representable_values(fmt: str, *, finite_only: bool = True) -> list[float]:
    """The sorted, de-duplicated set of values the format can represent (decode all 256 byte patterns).
    Finite-only by default (drops inf/NaN), so callers get a value set that is EXACT in the format and
    safe to encode back — the basis for a format-correct operand palette (dynamic range incl. a subnormal
    and a near-max value), not a hand-picked list of small magnitudes."""
    vals = fp8_to_f32(np.arange(256, dtype=np.uint8), fmt)
    out = {float(v) for v in vals if not finite_only or np.isfinite(v)}
    return sorted(out)
