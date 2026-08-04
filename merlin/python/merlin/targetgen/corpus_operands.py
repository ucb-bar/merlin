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

from merlin.runtime.fp8_formats import canonical_fp8, representable_values

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


def _prime_at_least(k: int) -> int:
    n = max(2, k)
    while not _is_prime(n):
        n += 1
    return n


def derive_palette(fmt: str, n: int) -> list[float]:
    """``n`` DISTINCT, exactly-representable values for ``fmt``, spread across a safe magnitude window and
    including both signs and the smallest positive (subnormal) value — derived from the format's own
    representable set, never a hand-picked list. Raises if the format cannot supply ``n`` distinct values."""
    canonical_fp8(fmt)                                    # fail closed on a non-fp8 format
    finite = representable_values(fmt)
    cap = _MAG_CAP
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


def operand_values(shape: tuple[int, int], fmt: str, salt: int) -> list[float]:
    """Flat row-major values for a 2-D operand of ``shape`` in ``fmt``, with distinct rows, distinct
    columns, and asymmetry guaranteed (see module docstring). ``salt`` (an int derived from the tensor
    name) shifts the pattern per capsule so operands vary but the guarantees hold."""
    rows, cols = shape
    p = _prime_at_least(max(rows, cols, 2))
    pal = derive_palette(fmt, p)
    s = salt % p
    return [pal[(r + 2 * c + s) % p] for r in range(rows) for c in range(cols)]


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
