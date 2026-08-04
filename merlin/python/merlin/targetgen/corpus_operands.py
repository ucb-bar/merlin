"""Rigorous, format-derived operand synthesis for the generated capsule corpus.

A capsule can only expose a compiler bug if its operands make that bug CHANGE THE OUTPUT. Row-identical or
near-symmetric operands hide whole bug classes: a wrong row stride, a swapped base offset, or a transposed
load all produce the SAME result on a matrix whose rows repeat or which equals its own transpose. The old
generator filled from 11 hand-picked magnitudes via a flat scalar hash, giving only ~6 distinct values and
~11 distinct rows of 32 — weak.

This module derives operands with three GUARANTEED properties, for any float format and any shape, with no
target literal and no regex:
  * every row is distinct and every column is distinct  -> row/col addressing & stride bugs change output;
  * the matrix is asymmetric (A != A^T)                 -> a transpose/layout bug is visible;
  * values span the format's representable range (a subnormal, mid, near-cap; both signs), exactly
    representable so the golden stays exact.

Construction: with a palette of P distinct representable values and P a PRIME >= max(rows, cols),
``v[r][c] = palette[(r + 2*c + salt) mod P]``. Prime P makes the row-stride 1 and col-stride 2 invertible
mod P, so rows differ (r1!=r2 => shifts differ), columns differ (2 coprime to P), and for any r!=c the
(1-2)*(r-c) offset is nonzero -> ``v[r][c] != v[c][r]`` (asymmetric). ``salt`` shifts the pattern per capsule
without breaking any property.
"""
from __future__ import annotations

from merlin.runtime.fp8_formats import canonical_float, representable_values

# Keep magnitudes modest so a long-K reduction cannot overflow bf16 accumulation, while still spanning the
# dynamic range (subnormal -> a few powers of two). Widened automatically if a format is too sparse here.
_MAG_CAP = 16.0


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _mix(r: int, c: int, s: int) -> int:
    """A deterministic 32-bit avalanche hash of ``(row, col, salt)`` — pseudo-random alphabet indices so a
    small-alphabet operand still gets distinct row/column tuples (no linear periodicity that would repeat a
    column every ``alphabet`` steps). Structural + integer-only (no regex, no float)."""
    x = ((r + 1) * 0x9E3779B1) ^ ((c + 1) * 0x85EBCA77) ^ ((s + 1) * 0xC2B2AE3D)
    x &= 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2545F491) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0x27D4EB2F) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _prime_at_least(k: int) -> int:
    n = max(2, k)
    while not _is_prime(n):
        n += 1
    return n


def derive_palette(fmt: str, n: int, *, mag_cap: float = _MAG_CAP) -> list[float]:
    """``n`` DISTINCT, exactly-representable values for ``fmt``, spread across a safe magnitude window and
    including both signs and the smallest positive (subnormal) value — derived from the format's own
    representable set, never a hand-picked list. Raises if the format cannot supply ``n`` distinct values."""
    canonical_float(fmt)                                  # fail closed on an unknown float format
    finite = representable_values(fmt)
    cap = mag_cap
    while True:
        pool = sorted({v for v in finite if 0.0 < abs(v) <= cap})   # nonzero, in-window
        if len(pool) >= n or cap >= max(abs(v) for v in finite):
            break
        cap *= 2.0                                        # sparse format -> widen the window
    if len(pool) < n:
        raise ValueError(f"format {fmt!r} has only {len(pool)} usable values, need {n}")
    # evenly-spaced picks across the sorted pool -> a genuine spread (subnormal..near-cap, both signs)
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def _usable_pool(fmt: str, max_alphabet: int | None, mag_cap: float = _MAG_CAP) -> list[float]:
    """Nonzero, exactly-representable values in a safe magnitude window (widened if the format is sparse),
    optionally capped to an evenly-spread ``max_alphabet`` values — the alphabet the operand is drawn from.
    A low-bit format (fp4 has 14 nonzero values; fp6 is LUT-limited to 16 per row) legitimately supplies a
    small alphabet; that is a datapath fact, not a degeneracy."""
    canonical_float(fmt)                                  # fail closed on an unknown float format
    finite = representable_values(fmt)
    cap = mag_cap
    while True:
        pool = sorted({v for v in finite if 0.0 < abs(v) <= cap})
        if len(pool) >= 2 or cap >= max(abs(v) for v in finite):
            break
        cap *= 2.0
    if max_alphabet is not None and len(pool) > max_alphabet:
        step = len(pool) / max_alphabet
        pool = [pool[int(i * step)] for i in range(max_alphabet)]
    return pool


def operand_values(shape: tuple[int, int], fmt: str, salt: int, *, max_alphabet: int | None = None,
                   mag_cap: float = _MAG_CAP) -> list[float]:
    """Flat row-major values for a 2-D operand of ``shape`` in ``fmt``, with distinct rows, distinct
    columns, and asymmetry guaranteed (see module docstring). ``salt`` (an int derived from the tensor
    name) shifts the pattern per capsule so operands vary but the guarantees hold.

    Two regimes: a format rich enough to supply a PRIME-sized palette (fp8/fp16/bf16/f32) uses the prime
    construction (byte-identical to before); a low-bit format whose alphabet is smaller than the shape (fp4,
    or LUT-capped fp6) uses a deterministic mixed construction that still yields distinct row/column TUPLES
    over a small alphabet, verified by :func:`rigor_findings` with a salt search (fail closed if none is
    rigorous). ``max_alphabet`` caps the alphabet (e.g. 16 for the fp6 LUT); ``mag_cap`` bounds operand
    magnitude (MX operands stay small so a block-scaled bf16 accumulate cannot overflow)."""
    rows, cols = shape
    p = _prime_at_least(max(rows, cols, 2))
    if max_alphabet is None:
        try:
            pal = derive_palette(fmt, p, mag_cap=mag_cap)  # prime construction (strongest guarantees)
            s = salt % p
            return [pal[(r + 2 * c + s) % p] for r in range(rows) for c in range(cols)]
        except ValueError:
            pass                                          # alphabet too small -> mixed construction below
    pool = _usable_pool(fmt, max_alphabet, mag_cap)
    a = len(pool)
    for attempt in range(256):                            # deterministic salt search over the small alphabet
        s = salt + attempt
        vals = [pool[_mix(r, c, s) % a] for r in range(rows) for c in range(cols)]
        if not rigor_findings(vals, shape):
            return vals
    raise ValueError(f"format {fmt!r} alphabet ({a} values) too small to build a rigorous "
                     f"{rows}x{cols} operand")


def rigor_findings(values: list[float], shape: tuple[int, int]) -> list[str]:
    """Advisory: report the ways an operand would HIDE a bug — duplicate rows, duplicate columns, or
    symmetry (A == A^T). Empty list == rigorous. Target-agnostic; used by the corpus-rigor gate to keep a
    regeneration from silently degrading operand quality."""
    rows, cols = shape
    grid = [tuple(values[r * cols:(r + 1) * cols]) for r in range(rows)]
    out: list[str] = []
    if len(set(grid)) != rows:
        out.append(f"duplicate rows: only {len(set(grid))} distinct of {rows} (row-addressing bugs invisible)")
    colset = {tuple(grid[r][c] for r in range(rows)) for c in range(cols)}
    if len(colset) != cols:
        out.append(f"duplicate columns: only {len(colset)} distinct of {cols} (col/stride bugs invisible)")
    if rows == cols and all(grid[r][c] == grid[c][r] for r in range(rows) for c in range(cols)):
        out.append("operand is symmetric (A == A^T): a transpose/layout bug produces identical output")
    if len({v for row in grid for v in row}) <= 1:
        out.append("operand is constant (degenerate): almost any bug is invisible")
    return out


# E8M0 exponent half-window: block scales sweep +/- this many powers of two around the bias (scale 1.0).
# Kept SMALL on purpose — one distinct exponent per lane over a wide tile would span 2**lanes of dynamic
# range and overflow the bf16 accumulate to inf (a golden any broken kernel matches). +/-3 = scale in
# [2**-3, 2**3], enough distinct exponents (a 7-value alphabet) that a mis-indexed scale changes the output.
_E8M0_SPAN = 3


def e8m0_scale_codes(shape: tuple[int, int], salt: int, *, center: int = 127) -> list[list[int]]:
    """A rigorous, numerically-safe E8M0 block-scale stream for a ``(groups, lanes)`` shape (A: lanes=M rows;
    B: lanes=N cols). An MX matmul applies one block scale per (K-group, lane); if a lane's scale is
    mis-indexed the output must CHANGE, so the per-lane exponents VARY (they sweep a small window centred on
    ``center`` == the E8M0 bias 127, i.e. scale 1.0), with no two ADJACENT lanes equal and each lane also
    varying across groups. The window is only ``+/-_E8M0_SPAN`` wide so ``2**(code-127)`` stays modest and a
    bf16 accumulate cannot overflow. ``salt`` shifts the pattern per capsule; codes are clamped to the valid
    ``[0, 254]`` E8M0 range (255 is the NaN code)."""
    groups, lanes = shape
    period = 2 * _E8M0_SPAN + 1                          # a `period`-value exponent alphabet
    codes = []
    for g in range(groups):
        row = []
        for li in range(lanes):
            off = ((li * 5 + g * 3 + salt) % period) - _E8M0_SPAN    # step 5 (coprime to 7) -> neighbours differ
            row.append(max(0, min(254, center + off)))
        codes.append(row)
    return codes


def scale_rigor_findings(codes: list[list[int]]) -> list[str]:
    """Advisory rigor for an E8M0 scale stream: a constant stream, or one where adjacent lanes never differ,
    hides scale-addressing bugs. Target-agnostic; used by the corpus-rigor gate so a regeneration cannot ship
    an all-equal (inert) scale stream. (An all-distinct check is deliberately NOT required: at wide tiles it
    would force a 2**lanes dynamic range — see :data:`_E8M0_SPAN`.)"""
    if not codes:
        return ["empty scale stream"]
    out: list[str] = []
    flat = [c for row in codes for c in row]
    if len(set(flat)) <= 1:
        out.append("scale stream is constant (degenerate): a mis-indexed block scale is invisible")
    if not any(any(row[c] != row[c + 1] for c in range(len(row) - 1)) for row in codes):
        out.append("no adjacent lanes differ in any group: lane-swap scale bugs invisible")
    return out
