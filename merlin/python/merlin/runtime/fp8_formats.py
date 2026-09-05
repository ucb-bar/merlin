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


def storage_bits(fmt: str) -> int:
    """Width in bits of one STORED element of ``fmt``, derived from its own layout (1 sign + exp + mant)
    rather than read off the spelling. Scraping digits out of a token cannot do this: ``fp8_e4m3`` has
    three digit runs and concatenates to 843, which silently mis-sized every capacity computed from a
    float operand token. E8M0 is exponent-only (no sign, no mantissa) and carries its own width."""
    t = _ALIASES.get(fmt, fmt)
    if t == "e8m0":
        return int(E8M0["bits"])
    eb, mb, _bias, _scheme = float_format_params(t)
    return 1 + eb + mb


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


def codes_to_f32(codes, fmt: str) -> np.ndarray:
    """Decode integer code patterns of ANY registered float format to float32.

    The wider sibling of :func:`fp8_to_f32` (which fails closed on a non-fp8 format) and the exact
    inverse of :func:`float_to_codes`, so ``codes_to_f32(float_to_codes(v, f), f) == v`` for every
    value ``f`` can hold. It exists for the *readback* direction: a device console that prints a
    float result's STORED CONTAINER as an integer (which is what an integer-only ``OUT`` protocol
    does) hands back a code pattern, not a value, and decoding it needs the declared format.

    Each code is masked to the format's own storage width first, because a console that printed a
    signed container sign-extends: an ``int16_t`` holding ``0xBF80`` arrives as ``-16512``, and the
    mask is what turns it back into the pattern that was stored. A format at least 32 bits wide is
    reinterpreted directly (a float32 already IS its stored pattern), matching what
    :func:`float_to_codes` does in the same width range."""
    t = canonical_float(fmt)
    bits = storage_bits(t)
    mask = (1 << bits) - 1
    u = np.asarray(codes, dtype=np.int64) & np.int64(mask)
    if bits >= 32:
        return np.ascontiguousarray(u.astype(np.uint32)).view(np.float32).copy()
    return _decode(u.astype(np.uint32), t)


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


def float_to_codes(values, fmt: str) -> np.ndarray:
    """Encode float values to this format's integer CODE patterns — the inverse of :func:`_decode`.

    Derived by inverting the decoder itself rather than re-deriving a layout: for any format that fits in
    <=16 bits every code is decoded once and the nearest one is selected, so encode(decode(c)) == c by
    construction and the two directions can never drift apart. A wide format (f32) is the identity view
    (its 2**32 codes are not enumerable, and a float32 already IS the stored pattern).

    Ties land on the EVEN code (IEEE round-half-to-even). Values outside the finite range saturate to the
    largest finite magnitude of the matching sign rather than becoming inf — a preload operand that
    silently turned into inf would poison the whole comparison. NaN is refused (fail closed): the caller
    has a value the device cannot be handed.
    """
    t = canonical_float(fmt)
    a = np.asarray(values, dtype=np.float32).ravel()
    if np.isnan(a).any():
        raise ValueError(f"cannot encode NaN into {t} (fail closed)")
    bits = storage_bits(t)
    if bits >= 32:                                     # f32: the value already is the stored pattern
        return np.ascontiguousarray(a).view(np.uint32).copy()
    codes = np.arange(1 << bits, dtype=np.uint32)
    vals = _decode(codes, t)
    keep = np.isfinite(vals)
    codes, vals = codes[keep], vals[keep]
    order = np.argsort(vals, kind="stable")
    codes, vals = codes[order], vals[order].astype(np.float64)
    # Clamp to the grid FIRST. An out-of-range magnitude (including a float32 inf, which is what
    # `biggest * 2` becomes) otherwise makes both neighbour distances inf, and the nearest-of-two
    # comparison then silently selects the second-largest code instead of saturating.
    a64 = np.clip(a.astype(np.float64), vals[0], vals[-1])
    hi = np.clip(np.searchsorted(vals, a64), 1, len(vals) - 1)
    lo = hi - 1
    dlo, dhi = np.abs(a64 - vals[lo]), np.abs(vals[hi] - a64)
    pick = np.where(dhi < dlo, hi, lo)
    tie = dlo == dhi                                   # round-half-to-even on the CODE pattern
    if tie.any():
        even_hi = (codes[hi] & 1) == 0
        pick = np.where(tie, np.where(even_hi, hi, lo), pick)
    return codes[pick]


def encode_bytes(values, fmt: str) -> bytes:
    """Little-endian stored bytes for ``values`` in ``fmt`` — the device preload image. Refuses a
    sub-byte format, where an element has no standalone byte and packing is the caller's layout
    decision, not this module's."""
    t = canonical_float(fmt)
    bits = storage_bits(t)
    if bits % 8:
        raise ValueError(f"{t} stores {bits} bits per element — packing is the caller's layout choice")
    width = bits // 8
    return float_to_codes(values, t).astype(f"<u{width}").tobytes()
