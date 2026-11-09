"""The bridge from a compiled model's calling convention to a matrix unit's GEMM kernel, for any unit.

:mod:`llvmlower.int8_contractions` turns a selected contraction into ``call @<prefix>_<n>`` on lowered
memref descriptors. This emits what those symbols resolve to, around a kernel the support provider
supplies. Nothing here knows an instruction: the kernel arrives as C source with one fixed signature,

    void <kernel>(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                  size_t m, size_t n, size_t k);

computing ``C[m][n] = bias[n] + sum_k at[k*m + i] * b[k*n + j]`` (both operands K-major, int32 wrapping
accumulation, ``bias`` may be NULL). Everything between that and the model is generic and lives here:

**Layout.** Both operands of an outer-product unit are K-major. The prepared IR hands over A as (M, K)
and B as (K, N), so B is already K-major and A is packed (transposed) into an aligned scratch buffer.
That pack is a real cost, O(M*K) against the kernel's O(M*N*K), and it is vectorised on RISC-V.

**Calling convention.** A lowered rank-2 memref is seven scalars and a returned one a struct of the same
shape. Extents are read from the DESCRIPTORS, never from the signature an entry was generated for.

**Nothing may produce a wrong answer.** Every condition the fast path needs (dense B and C with the
row strides the kernel assumes, aligned bases, a pack that fits) is checked at run time; a failure takes
a plain scalar contraction and is counted (``<prefix>_calls`` / ``<prefix>_fallbacks``), so a
systematically declined route is visible rather than silently slow.

**Ordering.** One scratch buffer serves every entry, so a pack is fenced against the previous call's
kernel reads (a decoupled vector unit lets scalar stores run ahead of outstanding vector loads), and a
batched entry fences between its slices for the same reason.

The provider owns the kernel and its scalar stand-in; this owns only what any such kernel needs around it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["DEFAULT_SCRATCH_BYTES", "ShimUnit", "emit_translation_unit", "is_batched", "scratch_bytes_for"]

#: Pack scratch when no signature is known. Unreachable from the rewrite, which emits one per call site.
DEFAULT_SCRATCH_BYTES = 1 << 16


class ShimUnit(str):
    """The emitted C source, carrying the scratch size (an image-fit fact) and the entry symbols."""

    scratch_bytes: int
    symbols: tuple[str, ...]

    def __new__(cls, text: str, *, scratch_bytes: int, symbols: tuple[str, ...]) -> ShimUnit:
        self = super().__new__(cls, text)
        self.scratch_bytes = int(scratch_bytes)
        self.symbols = tuple(symbols)
        return self


def is_batched(sig: Sequence[int]) -> bool:
    """``(B, M, N, K)`` rather than ``(M, N, K)``: the arity is the discriminator."""
    return len(tuple(sig)) == 4


def scratch_bytes_for(signatures: Mapping[str, Sequence[int]]) -> int:
    """The largest ``M * K`` any signature packs (a batched one packs one slice at a time)."""
    if not signatures:
        return DEFAULT_SCRATCH_BYTES
    return max(int(s[-3]) * int(s[-1]) for s in signatures.values())


def _c_identifier(name: str, what: str) -> str:
    if not name or not (name[0].isalpha() or name[0] == "_") or not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(f"{what} {name!r} is not a C identifier")
    return name


def emit_translation_unit(
    kernel_source: str,
    kernel_func: str,
    signatures: Mapping[str, Sequence[int]],
    *,
    alignment_bytes: int,
    prefix: str,
    scratch_bytes: int | None = None,
) -> ShimUnit:
    """The whole C translation unit: the provider's kernel plus one MLIR-ABI entry per signature.

    ``signatures`` is ``{symbol: (m, n, k)}`` or ``{symbol: (b, m, n, k)}`` from the rewrite's sidecar.
    ``alignment_bytes`` is the base alignment the unit's operand loads need (derived by the provider from
    the configuration's datapath width); it is required because a guessed one fails on a later beat of a
    load rather than at build time. ``prefix`` namespaces every symbol this unit defines besides the
    entries, so two units' shims can be linked into one image.
    """
    if int(alignment_bytes) < 1:
        raise ValueError(f"alignment_bytes={alignment_bytes} is not a byte alignment")
    _c_identifier(kernel_func, "kernel function")
    _c_identifier(prefix, "symbol prefix")
    for sym in signatures:
        _c_identifier(sym, "entry symbol")
    need = int(scratch_bytes if scratch_bytes is not None else scratch_bytes_for(signatures))
    entries = "\n".join(_entry(sym, sig, prefix) for sym, sig in sorted(signatures.items()))
    text = _UNIT.format(
        kernel=kernel_source,
        align=int(alignment_bytes),
        scratch=need,
        func=kernel_func,
        p=prefix,
        P=prefix.upper(),
        entries=entries,
        n_sigs=len(signatures),
    )
    return ShimUnit(text, scratch_bytes=need, symbols=tuple(sorted(signatures)))


def _entry(symbol: str, sig: Sequence[int], prefix: str) -> str:
    if is_batched(sig):
        b, m, n, k = (int(v) for v in sig)
        return f"""
/* {symbol}: generated for B={b} M={m} N={n} K={k}; extents are re-read from the descriptors. */
{prefix}_memref_3d_i32 {symbol}(
    int8_t *a_alloc, int8_t *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1, intptr_t a_s2,
    intptr_t a_st0, intptr_t a_st1, intptr_t a_st2,
    int8_t *b_alloc, int8_t *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1, intptr_t b_s2,
    intptr_t b_st0, intptr_t b_st1, intptr_t b_st2,
    int32_t *c_alloc, int32_t *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1, intptr_t c_s2,
    intptr_t c_st0, intptr_t c_st1, intptr_t c_st2)
{{
  {prefix}_memref_3d_i32 ret;
  ret.allocated = c_alloc; ret.aligned = c_aligned; ret.offset = c_off;
  ret.sizes[0] = c_s0; ret.sizes[1] = c_s1; ret.sizes[2] = c_s2;
  ret.strides[0] = c_st0; ret.strides[1] = c_st1; ret.strides[2] = c_st2;
  for (intptr_t bi = 0; bi < c_s0; ++bi) {{
    {prefix}_shim_gemm_i8(a_alloc, a_aligned, a_off + bi * a_st0, a_s1, a_s2, a_st1, a_st2,
                          b_alloc, b_aligned, b_off + bi * b_st0, b_s1, b_s2, b_st1, b_st2,
                          c_alloc, c_aligned, c_off + bi * c_st0, c_s1, c_s2, c_st1, c_st2);
    {prefix.upper()}_FENCE();   /* the next slice re-packs the scratch this slice's kernel read */
  }}
  return ret;
}}"""
    m, n, k = (int(v) for v in sig)
    return f"""
/* {symbol}: generated for M={m} N={n} K={k}; extents are re-read from the descriptors. */
{prefix}_memref_2d_i32 {symbol}(
    int8_t *a_alloc, int8_t *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1,
    intptr_t a_st0, intptr_t a_st1,
    int8_t *b_alloc, int8_t *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1,
    intptr_t b_st0, intptr_t b_st1,
    int32_t *c_alloc, int32_t *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1,
    intptr_t c_st0, intptr_t c_st1)
{{
  return {prefix}_shim_gemm_i8(a_alloc, a_aligned, a_off, a_s0, a_s1, a_st0, a_st1,
                               b_alloc, b_aligned, b_off, b_s0, b_s1, b_st0, b_st1,
                               c_alloc, c_aligned, c_off, c_s0, c_s1, c_st0, c_st1);
}}"""


_UNIT = """\
/* GENERATED by merlin.kernels.matrix_shim -- do not edit.
 *
 * The bridge from the compiled model's calling convention to a matrix unit's GEMM kernel. The kernel
 * below is the support provider's; everything after it is generic.
 *
 * {n_sigs} entry point(s); left-operand pack scratch = {scratch} bytes, aligned to {align}.
 */
{kernel}

#include <stdint.h>
#include <stddef.h>

typedef struct {{
  int32_t *allocated;
  int32_t *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
}} {p}_memref_2d_i32;

typedef struct {{
  int32_t *allocated;
  int32_t *aligned;
  intptr_t offset;
  intptr_t sizes[3];
  intptr_t strides[3];
}} {p}_memref_3d_i32;

#ifdef __riscv
# define {P}_FENCE() __asm__ volatile("fence rw, rw" ::: "memory")
#else
# define {P}_FENCE() __asm__ volatile("" ::: "memory")
#endif

/* Base alignment the unit's operand loads need, from the configuration's datapath width. */
#define {P}_ALIGN {align}

#ifndef {P}_LHS_SCRATCH_BYTES
#define {P}_LHS_SCRATCH_BYTES {scratch}
#endif
static int8_t {p}_lhs_pack[{P}_LHS_SCRATCH_BYTES] __attribute__((aligned({P}_ALIGN)));

/* Pack block edge: a locality knob only; every value computes the same bytes. */
#ifndef {P}_PACK_BLK
#define {P}_PACK_BLK 64
#endif

static unsigned long long g_{p}_calls = 0ULL;
static unsigned long long g_{p}_fallbacks = 0ULL;
unsigned long long {p}_calls(void) {{ return g_{p}_calls; }}
unsigned long long {p}_fallbacks(void) {{ return g_{p}_fallbacks; }}

/* Correct for any strides; the plainest loop that says what a contraction is. Wraps like the unit. */
static void {p}_scalar_gemm(int32_t *c, const int8_t *a, const int8_t *b, size_t m, size_t n, size_t k,
                            intptr_t a_st0, intptr_t a_st1, intptr_t b_st0, intptr_t b_st1,
                            intptr_t c_st0, intptr_t c_st1)
{{
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {{
      uint32_t sum = 0;
      for (size_t kk = 0; kk < k; ++kk)
        sum += (uint32_t)((int32_t)a[(intptr_t)i * a_st0 + (intptr_t)kk * a_st1]
                        * (int32_t)b[(intptr_t)kk * b_st0 + (intptr_t)j * b_st1]);
      c[(intptr_t)i * c_st0 + (intptr_t)j * c_st1] = (int32_t)sum;
    }}
}}

/* C = A @ B in int32, A (M,K) M-major, B (K,N) K-major. The output is WRITTEN: the rewrite routes only
 * contractions whose init is a zero fill (merlin.kernels.shapes.zero_initialised). */
{p}_memref_2d_i32 {p}_shim_gemm_i8(
    int8_t *a_alloc, int8_t *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1,
    intptr_t a_st0, intptr_t a_st1,
    int8_t *b_alloc, int8_t *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1,
    intptr_t b_st0, intptr_t b_st1,
    int32_t *c_alloc, int32_t *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1,
    intptr_t c_st0, intptr_t c_st1)
{{
  (void)a_alloc; (void)b_alloc; (void)b_s1;
  const int8_t *A = a_aligned + a_off;
  const int8_t *B = b_aligned + b_off;
  int32_t *C = c_aligned + c_off;
  const size_t M = (size_t)a_s0;
  const size_t K = (size_t)a_s1;
  const size_t N = (size_t)c_s1;

  {p}_memref_2d_i32 ret;
  ret.allocated = c_alloc; ret.aligned = c_aligned; ret.offset = c_off;
  ret.sizes[0] = c_s0; ret.sizes[1] = c_s1;
  ret.strides[0] = c_st0; ret.strides[1] = c_st1;

  if (M == 0 || N == 0) return ret;
  if (K == 0) {{
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        C[(intptr_t)i * c_st0 + (intptr_t)j * c_st1] = 0;
    return ret;
  }}
  g_{p}_calls += 1ULL;

  const int b_dense = (b_st1 == 1) && (b_st0 == (intptr_t)N) && (b_s0 == (intptr_t)K);
  const int c_dense = (c_st1 == 1) && (c_st0 == (intptr_t)N) && (c_s0 == (intptr_t)M);
  const int b_okalign = (((uintptr_t)B) % (uintptr_t){P}_ALIGN) == 0;
  const int c_okalign = (((uintptr_t)C) % (uintptr_t){P}_ALIGN) == 0;
  const int fits = (M <= (size_t){P}_LHS_SCRATCH_BYTES / K);
  if (!(b_dense && c_dense && b_okalign && c_okalign && fits)) {{
    g_{p}_fallbacks += 1ULL;
    {p}_scalar_gemm(C, A, B, M, N, K, a_st0, a_st1, b_st0, b_st1, c_st0, c_st1);
    return ret;
  }}

  /* Order this pack after the previous call's kernel reads of the same scratch. */
  {P}_FENCE();
#if defined(__riscv) && defined(__riscv_vector) && !defined({P}_SCALAR_PACK)
  /* Vectorised transpose: a strided gather down A's column, a contiguous store along the pack row. */
  for (size_t kb = 0; kb < K; kb += {P}_PACK_BLK) {{
    const size_t ke = (K - kb) < {P}_PACK_BLK ? K : kb + {P}_PACK_BLK;
    for (size_t i0 = 0; i0 < M; i0 += {P}_PACK_BLK) {{
      const size_t i1 = (M - i0) < {P}_PACK_BLK ? M : i0 + {P}_PACK_BLK;
      for (size_t kk = kb; kk < ke; ++kk) {{
        int8_t *dst = {p}_lhs_pack + kk * M;
        const int8_t *src = A + (intptr_t)kk * a_st1;
        size_t i = i0;
        while (i < i1) {{
          size_t vl;
          __asm__ volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"(i1 - i));
          __asm__ volatile("vlse8.v v24, (%0), %1" :: "r"(src + (intptr_t)i * a_st0), "r"((intptr_t)a_st0)
                           : "memory");
          __asm__ volatile("vse8.v v24, (%0)" :: "r"(dst + i) : "memory");
          i += vl;
        }}
      }}
    }}
  }}
#else
  for (size_t i0 = 0; i0 < M; i0 += {P}_PACK_BLK) {{
    const size_t i1 = (M - i0) < {P}_PACK_BLK ? M : i0 + {P}_PACK_BLK;
    for (size_t kb = 0; kb < K; kb += {P}_PACK_BLK) {{
      const size_t ke = (K - kb) < {P}_PACK_BLK ? K : kb + {P}_PACK_BLK;
      for (size_t i = i0; i < i1; ++i) {{
        const int8_t *arow = A + (intptr_t)i * a_st0;
        for (size_t kk = kb; kk < ke; ++kk)
          {p}_lhs_pack[kk * M + i] = arow[(intptr_t)kk * a_st1];
      }}
    }}
  }}
#endif

  {func}(C, {p}_lhs_pack, B, (const int32_t *)0, M, N, K);
  return ret;
}}
{entries}
"""
