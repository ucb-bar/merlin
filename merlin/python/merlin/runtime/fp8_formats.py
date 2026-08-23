"""Format-aware float decode + representable-value enumeration.

Covers the OCP float8 pair (**e4m3fn**, **e5m2**) AND the microscaling / SIMT float formats a target may
declare from its own datapath facts: **fp6** (e3m2), **fp4** (e2m1), **fp16**, **bf16**, **f32**, plus the
**E8M0** block scale (8-bit exponent-only). ``fp8`` is not one format — e4m3 and e5m2 decode the SAME raw
byte to DIFFERENT floats — and the same is true across the wider family, so the format must be CARRIED, not
assumed. This module is the single source of truth for the code<->float mapping (``runtime.dispatch_runtime``
and the corpus operand synthesis both use it), keyed on the format token so a new format is one table row,
never a per-caller branch.

The mapping is DERIVED from each format's ``(exp_bits, mantissa_bits, bias, top-code scheme)`` — never a
baked value table — and every unknown spelling FAILS CLOSED (``KeyError``), exactly like the e4m3/e5m2 +
saturn ``float8`` fix. ``bias`` is itself the derived IEEE value ``(1 << (exp_bits - 1)) - 1``, kept in the
row only so the three MX widths read explicitly against their RTL definitions (FP6E3M2/FP4E2M1/e4m3).
"""
from __future__ import annotations

import numpy as np

# scheme = how the all-ones exponent code is interpreted (the only place formats of the same width differ):
#   "ieee"       exp==all-1 -> inf (mant 0) / NaN (mant != 0)        e5m2, fp16, bf16, f32
#   "e4m3fn"     only all-1 exp AND all-1 mant is NaN; NO inf        OCP float8 e4m3fn
#   "mx_finite"  no inf / no NaN; every code is a finite value       MX fp6 (e3m2) / fp4 (e2m1)
# Row = (exp_bits, mantissa_bits, bias, scheme). bias == (1 << (exp_bits - 1)) - 1 for every entry (derived).
_FORMATS = {
    "fp8_e4m3": (4, 3, 7, "e4m3fn"),    # OCP float8_e4m3fn: finite-only, NaN = all-exp-1 & all-mant-1, max 448
    "fp8_e5m2": (5, 2, 15, "ieee"),     # OCP float8_e5m2: IEEE-like, inf = all-exp-1 & mant-0, max 57344
    "fp6_e3m2": (3, 2, 3, "mx_finite"), # MX FP6 (FP6E3M2NearestFinder.scala): 1 sign | 3 exp bias-3 | 2 mant
    "fp4_e2m1": (2, 1, 1, "mx_finite"), # MX FP4: 1 sign | 2 exp bias-1 | 1 mant
    "fp16":     (5, 10, 15, "ieee"),    # IEEE binary16
    "bf16":     (8, 7, 127, "ieee"),    # bfloat16
    "f32":      (8, 23, 127, "ieee"),   # IEEE binary32
}
_FP8 = {"fp8_e4m3", "fp8_e5m2"}         # the fp8 subset (the fp8-only public API restricts to these)

# accepted aliases (MLIR / torch / manifest spellings) -> canonical token. The ``mxfp*`` manifest tokens
# map onto their MX float layout: an MX PE's mxfp8 IS an e4m3-layout operand (the saturating/NaN-free
# product/scale semantics live in the numeric reference, not in the operand code<->value map).
_ALIASES = {
    "f8E4M3FN": "fp8_e4m3", "f8e4m3fn": "fp8_e4m3", "e4m3": "fp8_e4m3", "mxfp8": "fp8_e4m3",
    "f8E5M2": "fp8_e5m2", "f8e5m2": "fp8_e5m2", "e5m2": "fp8_e5m2",
    "f6E3M2FN": "fp6_e3m2", "e3m2": "fp6_e3m2", "mxfp6": "fp6_e3m2",
    "f4E2M1FN": "fp4_e2m1", "e2m1": "fp4_e2m1", "mxfp4": "fp4_e2m1",
    "f16": "fp16", "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "f32", "fp32": "f32", "f8E8M0FNU": "e8m0",
}

# --- E8M0 block scale (8-bit exponent-only, no sign/mantissa), derived bias 127; code 0xFF is NaN. --------
E8M0 = {"bits": 8, "bias": (1 << (8 - 1)) - 1, "nan_code": (1 << 8) - 1}   # bias == 127


def canonical_float(fmt: str) -> str:
    """Canonical float token for a spelling, or raise for an unknown format (fail closed)."""
    t = _ALIASES.get(fmt, fmt)
    if t not in _FORMATS:
        raise KeyError(f"unknown float format {fmt!r} (known: {sorted(_FORMATS)})")
    return t


def canonical_fp8(fmt: str) -> str:
    """Canonical fp8 token for a spelling, or raise for a non-fp8 / unknown format (fail closed)."""
    t = _ALIASES.get(fmt, fmt)
    if t not in _FP8:
        raise KeyError(f"unknown fp8 format {fmt!r} (known: {sorted(_FP8)})")
    return t


def float_format_params(fmt: str) -> tuple[int, int, int, str]:
    """``(exp_bits, mantissa_bits, bias, scheme)`` for a canonical float token — the derived definition
    every consumer (decoder, numeric reference, encoder) shares, so nothing is fitted twice."""
    return _FORMATS[canonical_float(fmt)]


def normal_range(fmt: str) -> tuple[float, float]:
    """``(smallest positive NORMAL magnitude, largest finite magnitude)`` for ``fmt``, DERIVED.

    Both bounds come out of the format's own ``(exp_bits, mantissa_bits, bias, scheme)`` — never a
    tabulated 448 or 65504. The scheme decides how much of the top exponent code the format actually
    spends: ``ieee`` reserves all-ones for inf/NaN, ``e4m3fn`` spends all of it but the all-ones
    mantissa (which is why e4m3 reaches 448 and an IEEE reading of the same bits says 240), and
    ``mx_finite`` has no reservations at all.

    The pair is what a boundary needs to know whether an operand is REPRESENTABLE: below the first
    value the format is into subnormals (reduced mantissa, or zero outright on a datapath that
    flushes them), above the second it saturates.
    """
    eb, mb, bias, scheme = float_format_params(fmt)
    top = (1 << eb) - 1                                # all-ones exponent code
    if scheme == "ieee":                               # reserved for inf/NaN -> the one below it is max
        emax_code, mant_num = top - 1, (1 << (mb + 1)) - 1
    elif scheme == "e4m3fn":                           # only all-ones exp AND all-ones mantissa is NaN
        emax_code, mant_num = top, (1 << (mb + 1)) - 2
    else:                                              # "mx_finite": every code is a finite value
        emax_code, mant_num = top, (1 << (mb + 1)) - 1
    max_finite = (mant_num / float(1 << mb)) * (2.0 ** (emax_code - bias))
    return 2.0 ** (1 - bias), float(max_finite)


def _decode(codes: np.ndarray, fmt: str) -> np.ndarray:
    """Decode integer code patterns to float32 under ``fmt``. Subnormals (exp==0), normals, and the
    scheme's inf/NaN encodings are all handled; the mapping is derived from (exp_bits, mantissa_bits,
    bias, scheme), never hardcoded per format. Sign bit sits at ``exp_bits + mantissa_bits``."""
    eb, mb, bias, scheme = float_format_params(fmt)
    mmax = 1 << mb                                     # mantissa denominator
    emax = (1 << eb) - 1                               # all-ones exponent code
    u = np.ascontiguousarray(codes).astype(np.uint32)
    sign = np.where((u >> (eb + mb)) & 1 == 1, np.float32(-1.0), np.float32(1.0))
    exp = (u >> mb) & emax
    man = (u & (mmax - 1)).astype(np.float32)
    with np.errstate(over="ignore"):                   # top-exponent codes overflow to inf (filtered below)
        sub = (man / mmax) * np.float32(2.0 ** (1 - bias))        # exp==0: subnormal
        nrm = (1.0 + man / mmax) * np.exp2(exp.astype(np.float32) - bias)
    val = (sign * np.where(exp == 0, sub, nrm)).astype(np.float32)
    top = (exp == emax)
    if scheme == "ieee":                               # exp all-1 -> inf (man 0) / NaN (man != 0)
        val = np.where(top & (man == 0), sign * np.float32(np.inf), val)
        val = np.where(top & (man != 0), np.float32(np.nan), val)
    elif scheme == "e4m3fn":                           # only all-1 exp & all-1 mantissa is NaN
        val = np.where(top & (man == (mmax - 1)), np.float32(np.nan), val)
    # scheme == "mx_finite": every code is a finite value (MX fp6/fp4) -> nothing to special-case
    return val


def fp8_to_f32(u8: np.ndarray, fmt: str) -> np.ndarray:
    """Decode fp8 bytes to float32 under the named fp8 format (see :func:`_decode`)."""
    canonical_fp8(fmt)                                 # fail closed on a non-fp8 format
    return _decode(np.ascontiguousarray(u8, np.uint8), fmt)


def _grid_values(fmt: str) -> list[float]:
    """A bf16-safe exactly-representable value set for a WIDE format (f32) whose 2**bits code space is too
    large to enumerate. Values ``±(1 + m/8) * 2**e`` use a 3-bit mantissa (exact on the bf16 grid, hence
    exact in fp16/f32) across a modest exponent window — a genuine dynamic-range spread, derived from the
    format's own bias/exponent range, not a hand-picked magnitude list."""
    _, _, bias, _ = float_format_params(fmt)
    out: set[float] = set()
    for e in range(-min(bias, 12), min(bias, 13)):     # modest window inside the format's exponent range
        for m in range(8):
            v = (1.0 + m / 8.0) * (2.0 ** e)
            out.add(v)
            out.add(-v)
    return sorted(out)


def representable_values(fmt: str, *, finite_only: bool = True) -> list[float]:
    """The sorted, de-duplicated set of values the format can represent. For any format that fits in <=16
    bits every code pattern is decoded (EXACT); for a wide format (f32) a bf16-safe grid is derived instead
    (2**32 codes are not enumerable). Finite-only by default (drops inf/NaN), so callers get a value set that
    is exact in the format and safe to encode back — the basis for a format-correct operand palette."""
    t = canonical_float(fmt)
    eb, mb, _, _ = _FORMATS[t]
    total_bits = 1 + eb + mb
    if total_bits > 16:
        return _grid_values(t)
    vals = _decode(np.arange(1 << total_bits, dtype=np.uint32), t)
    out = {float(v) for v in vals if not finite_only or np.isfinite(v)}
    return sorted(out)


def e8m0_decode(code: int) -> float:
    """Decode an E8M0 block-scale code to its float scale ``2**(code - 127)`` (code 0xFF is NaN)."""
    c = int(code) & ((1 << E8M0["bits"]) - 1)
    if c == E8M0["nan_code"]:
        return float("nan")
    return float(2.0 ** (c - E8M0["bias"]))
