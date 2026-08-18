"""Deterministic test-stimulus synthesis — the single source of truth for capsule leaf data.

Every capsule operand, every C corroboration program and every perf-bench harness must fill the
SAME bytes, or a "three-way bit-exact" agreement is comparing different problems. That shared fill
lives here, in one place, and the C emitters below are generated from the same parameters rather
than hand-restated at each call site.

**Why the indexing matters.** The earlier fill was ``a[k] = (seed*(k+1) + k*k) % span`` over the
FLAT row-major index. Modulo a span of 4 that expression has period 4 in ``k``, so any 2-D operand
whose row length is a multiple of 4 — i.e. every ML shape — got *every row identical* and only two
distinct values. Operands like that cannot expose a compiler bug: a wrong row stride, a swapped
base offset or a transposed load all produce the same result on a matrix whose rows repeat. A
mutation test on the gemmini C0 matmul found 4 of 6 deliberately wrong implementations PASSING.

The fill is therefore indexed by ``(row, col)`` through an avalanche hash instead of by flat
position. Determinism, the ``lo..hi`` range and the pure-integer construction are unchanged; what
changes is that rows and columns now differ from one another, so row/column/stride/transpose bugs
alter the output and the grader can see them.
"""
from __future__ import annotations

_U32 = 0xFFFFFFFF


def mix(r: int, c: int, s: int) -> int:
    """A deterministic 32-bit avalanche hash of ``(row, col, salt)``.

    Structural and integer-only (no regex, no float, no RNG), so the value is identical across
    runs, machines and Python versions — and is cheap to restate in C (see :func:`c_fill_lines`).
    """
    x = ((r + 1) * 0x9E3779B1) ^ ((c + 1) * 0x85EBCA77) ^ ((s + 1) * 0xC2B2AE3D)
    x &= _U32
    x ^= x >> 15
    x = (x * 0x2545F491) & _U32
    x ^= x >> 13
    x = (x * 0x27D4EB2F) & _U32
    x ^= x >> 16
    return x


def det_seed(name: str) -> int:
    """The per-tensor salt derived from the tensor's name. Never zero (a zero salt would make two
    differently-named tensors collide)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) or 1


def grid_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    """Collapse any rank to ``(rows, cols)``: the last dimension is the column axis and the product
    of the leading dimensions is the row axis. A rank-1 operand becomes a single row, so its
    elements still vary along the column axis."""
    dims = tuple(int(d) for d in shape)
    if not dims:
        return (1, 1)
    cols = dims[-1]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    return (max(rows, 1), max(cols, 1))


def fill(name: str, shape: tuple[int, ...], lo: int = 0, hi: int = 3) -> list[int]:
    """Flat row-major integer stimulus for ``shape``, deterministic in ``name``.

    Values lie in ``lo..hi`` inclusive. The result is indexed by ``(row, col)``, so distinct rows
    and distinct columns are the norm rather than the exception — see the module docstring.
    """
    rows, cols = grid_shape(shape)
    span = hi - lo + 1
    if span <= 0:
        raise ValueError(f"empty stimulus range: lo={lo} hi={hi}")
    s = det_seed(name)
    return [lo + (mix(r, c, s) % span) for r in range(rows) for c in range(cols)]


# --------------------------------------------------------------------------------------------
# C emission — the same fill, for the baremetal reference programs
# --------------------------------------------------------------------------------------------
# A C program that fills its own leaves must produce byte-identical data to `fill` above, so the
# expression is emitted from here rather than restated at each call site. Kept as plain C99 with
# uint32_t arithmetic so the wraparound matches the Python `& 0xFFFFFFFF` exactly.

C_MIX_FN = r"""
/* Deterministic (row,col,salt) avalanche hash -- byte-identical to merlin.common.stimulus.mix. */
static uint32_t merlin_mix(uint32_t r, uint32_t c, uint32_t s) {
  uint32_t x = ((r + 1u) * 0x9E3779B1u) ^ ((c + 1u) * 0x85EBCA77u) ^ ((s + 1u) * 0xC2B2AE3Du);
  x ^= x >> 15; x *= 0x2545F491u;
  x ^= x >> 13; x *= 0x27D4EB2Fu;
  x ^= x >> 16;
  return x;
}
"""


def c_fill_loop(dest: str, rows: str, cols: str, seed: str, *, cast: str = "elem_t",
                lo: int = 0, hi: int = 3, indent: str = "  ") -> str:
    """A C statement filling ``dest`` (a flat row-major buffer) with the same values :func:`fill`
    produces. ``rows``/``cols``/``seed`` are C expressions so the caller can pass macros."""
    span = hi - lo + 1
    off = f"{lo} + " if lo else ""
    return (f"{indent}for (uint32_t r=0;r<(uint32_t)({rows});r++) "
            f"for (uint32_t c=0;c<(uint32_t)({cols});c++) "
            f"{{ ({dest})[r*(uint32_t)({cols})+c] = ({cast})({off}merlin_mix(r,c,(uint32_t)({seed}))%{span}u); }}")


def c_fill_loop_2d(dest: str, rows: str, cols: str, seed: str, *, cast: str = "elem_t",
                   lo: int = 0, hi: int = 3, indent: str = "  ") -> str:
    """Like :func:`c_fill_loop` but for a C array declared as ``dest[rows][cols]``."""
    span = hi - lo + 1
    off = f"{lo} + " if lo else ""
    return (f"{indent}for (uint32_t r=0;r<(uint32_t)({rows});r++) "
            f"for (uint32_t c=0;c<(uint32_t)({cols});c++) "
            f"{{ ({dest})[r][c] = ({cast})({off}merlin_mix(r,c,(uint32_t)({seed}))%{span}u); }}")
