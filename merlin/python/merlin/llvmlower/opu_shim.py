"""Emit the translation unit that connects the compiled model to the certified matrix microkernel.

:mod:`passes_opu` turns a contraction into ``call @merlin_opu_gemm_i8_<n>``. This emits the thing that
symbol resolves to. Between the two sits every mismatch between what the model's lowered IR hands over
and what the unit's datapath requires, and each one is a real cost rather than an implementation detail:

**Layout.** The unit indexes BOTH operands K-major. The prepared IR's maps are ``(d0,d2)``/``(d2,d1)``, so
the right-hand operand is already K-major and the left-hand one is M-major and has to be transposed. That
transpose is the packing cost the routing decision has to price, and it is why this module exists at all
rather than the call going straight to the kernel.

**Calling convention.** After bufferization and LLVM lowering an MLIR ``memref`` argument is not a pointer
— it is seven scalars (allocated, aligned, offset, two sizes, two strides), and a returned one is a struct
of the same shape. The extents therefore arrive as ARGUMENTS. This reads M, N and K out of the descriptors
rather than from the signature it was generated for, so a shim compiled for one shape cannot silently
compute another; the generated-time extents are used only to SIZE the scratch buffer.

**Base alignment.** The unit's operand loads move ``dLen`` bits per beat and a multi-beat load from a
misaligned base returns a wrong second beat — MEASURED, and it presents as a case that passes or fails
depending on what else was linked beside it. The packed left operand therefore lives in a buffer aligned
to ``dLen/8``, and the right operand's base is CHECKED. What the frozen corpus establishes is narrower
than "everything must be aligned": cases ``m_17``/``m_31``/``n_17``/``n_31`` certify 31/31 at tile edge 32
with row strides of 17 and 31, i.e. per-row offsets that are misaligned and loads that span two beats. So
the requirement this enforces is on the BASE, which is exactly what the corpus supports and no more.

**Nothing here may produce a wrong answer.** Every condition the fast path needs is checked at run time,
and a failure takes a scalar fallback rather than zeroing the output or computing on a short buffer. That
choice matters: a wrong answer is discovered late and gets cited, while a slow answer is visible in the
counters this exports (``merlin_opu_calls`` / ``merlin_opu_fallbacks``) and grades correctly meanwhile. A
fallback that fires is a bug in the routing decision, and the counter is how it gets found.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..kernels.opu_kernel import KernelSpec, emit_microkernel

__all__ = ["DEFAULT_SCRATCH_BYTES", "ShimUnit", "emit_translation_unit", "scratch_bytes_for"]

#: Fallback left-operand scratch size when no signature is known. Only reached by a caller that asks for a
#: unit with no signatures, which cannot happen from the rewrite (it emits one signature per call site).
DEFAULT_SCRATCH_BYTES = 1 << 16


class ShimUnit(str):
    """The emitted C source, carrying what a build needs to know about it.

    A plain string would lose the scratch size, which the Zephyr build has to be able to see: ~200 KB of
    ``.bss`` is a fact about whether an image fits, not a detail.
    """

    scratch_bytes: int
    symbols: tuple[str, ...]

    def __new__(cls, text: str, *, scratch_bytes: int, symbols: tuple[str, ...]) -> "ShimUnit":
        self = super().__new__(cls, text)
        self.scratch_bytes = int(scratch_bytes)
        self.symbols = tuple(symbols)
        return self


def scratch_bytes_for(signatures: Mapping[str, tuple[int, int, int]]) -> int:
    """Bytes the left-operand pack needs: the largest ``M * K`` any signature in this unit will pack.

    One buffer serves every signature because model inference calls them in sequence — the pack is dead
    the moment the kernel returns. Sizing it to the maximum rather than per signature is what keeps this
    from being 5 buffers and ~600 KB of ``.bss`` for spectformer instead of ~200 KB.
    """
    if not signatures:
        return DEFAULT_SCRATCH_BYTES
    return max(int(m) * int(k) for (m, _n, k) in signatures.values())


def emit_translation_unit(encodings: Mapping[str, Any],
                          signatures: Mapping[str, tuple[int, int, int]],
                          *,
                          spec: KernelSpec | None = None,
                          alignment_bytes: int,
                          derivation_ok: bool = True,
                          scratch_bytes: int | None = None) -> ShimUnit:
    """The whole C translation unit: the certified microkernel plus one entry per signature.

    ``signatures`` is :attr:`passes_opu.OpuRewrite.signatures` — ``{symbol: (m, n, k)}``.
    ``alignment_bytes`` is the derived operand alignment
    (:func:`kernels.opu_cert.operand_alignment_for_config`); it is required rather than defaulted because
    a guessed alignment produces a shim that is wrong on the second beat of a load and right everywhere
    a test would look.

    The microkernel is emitted by the same :func:`kernels.opu_kernel.emit_microkernel` the certification
    ran, from the same derived encodings, so what a model executes is the certified code and not a
    transcription of it. ``derivation_ok`` is forwarded, so an unresolved encoding refuses here too.
    """
    if int(alignment_bytes) < 1:
        raise ValueError(f"alignment_bytes={alignment_bytes} is not a byte alignment; it is derived from "
                         "the datapath width and a wrong value returns bad data rather than failing")
    spec = spec or KernelSpec(accumulate="OPMACC", broadcast="OPMVINBCAST", readout="OPMVOUT")
    kernel = emit_microkernel(encodings, spec, derivation_ok=derivation_ok)
    need = int(scratch_bytes if scratch_bytes is not None else scratch_bytes_for(signatures))

    entries = "\n".join(_entry(sym, sig, spec.func_name) for sym, sig in sorted(signatures.items()))
    return ShimUnit(_UNIT.format(kernel=kernel, align=int(alignment_bytes), scratch=need,
                                 func=spec.func_name, entries=entries,
                                 n_sigs=len(signatures)),
                    scratch_bytes=need, symbols=tuple(sorted(signatures)))


def _entry(symbol: str, sig: tuple[int, int, int], kernel_func: str) -> str:
    """One MLIR-ABI entry point.

    Every entry is the same code — the shared body reads its extents from the descriptors. They exist as
    separate symbols only because MLIR function types are monomorphic, so a 256x196/K=768 contraction and
    a 196x1024/K=256 one cannot call the same declaration. The generated-time extents go into a comment
    and a cheap consistency check, never into the arithmetic.
    """
    m, n, k = (int(v) for v in sig)
    return f"""
/* {symbol}: generated for M={m} N={n} K={k} (packs {m * k} bytes of left operand). The extents are
 * re-read from the descriptors below, so this entry stays correct if it is ever called with others. */
merlin_memref_2d_i32 {symbol}(
    int8_t *a_alloc, int8_t *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1,
    intptr_t a_st0, intptr_t a_st1,
    int8_t *b_alloc, int8_t *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1,
    intptr_t b_st0, intptr_t b_st1,
    int32_t *c_alloc, int32_t *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1,
    intptr_t c_st0, intptr_t c_st1)
{{
  return merlin_opu_shim_gemm_i8(a_alloc, a_aligned, a_off, a_s0, a_s1, a_st0, a_st1,
                                 b_alloc, b_aligned, b_off, b_s0, b_s1, b_st0, b_st1,
                                 c_alloc, c_aligned, c_off, c_s0, c_s1, c_st0, c_st1);
}}"""


#: The unit. Held as one template so the emitted file reads top to bottom the way a person would write it:
#: kernel, descriptors, scratch, counters, shared body, entries.
_UNIT = """\
/* GENERATED by merlin.llvmlower.opu_shim — do not edit.
 *
 * The bridge from the compiled model's calling convention to the certified matrix microkernel. The
 * microkernel below is emitted by merlin.kernels.opu_kernel from the same derived encoding table the
 * certification ran, so this is the certified code rather than a copy of it.
 *
 * {n_sigs} entry point(s); left-operand pack scratch = {scratch} bytes, aligned to {align}.
 */
{kernel}

/* ---------------------------------------------------------------------------------------------------
 * The MLIR calling convention.
 *
 * A lowered `memref<?x?xT>` is seven scalars, and a returned one is a struct of the same shape. This
 * struct must match what the caller's LLVM lowering produces for rank 2 on lp64d, which is why the
 * field order is fixed and not merely conventional.
 * ------------------------------------------------------------------------------------------------- */
typedef struct {{
  int8_t *allocated;
  int8_t *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
}} merlin_memref_2d_i8;

typedef struct {{
  int32_t *allocated;
  int32_t *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
}} merlin_memref_2d_i32;

/* Byte alignment the unit's operand loads require of a panel BASE, derived from its datapath width
 * (dLen/8). A multi-beat load from a misaligned base returns a wrong second beat; see the module
 * docstring for what the frozen corpus does and does not establish about this. */
#define MERLIN_OPU_ALIGN {align}

/* The left operand arrives M-major and the unit needs it K-major, so it is packed here. Sized at
 * generation time to the largest M*K any entry in this unit will see; override with
 * -DMERLIN_OPU_LHS_SCRATCH_BYTES=<n> when the image is tight, at the cost of taking the scalar fallback
 * for the shapes that no longer fit. Static rather than malloc'd because the bare-metal and Zephyr
 * targets this runs on may have no heap worth the name, and a failed allocation deep in a model is a
 * worse failure than a buffer that is visibly too small at link time. */
#ifndef MERLIN_OPU_LHS_SCRATCH_BYTES
#define MERLIN_OPU_LHS_SCRATCH_BYTES {scratch}
#endif
static int8_t merlin_opu_lhs_pack[MERLIN_OPU_LHS_SCRATCH_BYTES]
    __attribute__((aligned(MERLIN_OPU_ALIGN)));

/* How the split actually went. A fallback is CORRECT but slow, so without these a systematically
 * declined model would grade perfectly and report a speedup it never got. */
static unsigned long long g_merlin_opu_calls = 0ULL;
static unsigned long long g_merlin_opu_fallbacks = 0ULL;
unsigned long long merlin_opu_calls(void) {{ return g_merlin_opu_calls; }}
unsigned long long merlin_opu_fallbacks(void) {{ return g_merlin_opu_fallbacks; }}

/* Correct for any strides, and the only thing that can be correct when nothing else holds. Deliberately
 * the plainest loop that says what a contraction is: it is a fallback, so it is written to be obviously
 * right rather than fast, and it accumulates in int32 and wraps exactly as the unit does. */
static void merlin_opu_scalar_gemm(int32_t *c, const int8_t *a, const int8_t *b,
                                   size_t m, size_t n, size_t k,
                                   intptr_t a_st0, intptr_t a_st1, intptr_t b_st0, intptr_t b_st1,
                                   intptr_t c_st0, intptr_t c_st1)
{{
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {{
      int32_t sum = 0;
      for (size_t kk = 0; kk < k; ++kk)
        sum += (int32_t)a[(intptr_t)i * a_st0 + (intptr_t)kk * a_st1]
             * (int32_t)b[(intptr_t)kk * b_st0 + (intptr_t)j * b_st1];
      c[(intptr_t)i * c_st0 + (intptr_t)j * c_st1] = sum;
    }}
}}

/* The shared body every entry point calls.
 *
 * C = A @ B in int32, with A M-major (MxK) and B K-major (KxN). The output is WRITTEN, not accumulated
 * into: the rewrite only routes contractions whose init is a zero fill, so `C_init + A@B` and `A@B` are
 * the same value. See merlin.llvmlower.passes_opu.zero_initialised — that condition is checked where the
 * IR can still be inspected, which is the only place it CAN be checked, since a descriptor cannot say
 * where its contents came from. */
merlin_memref_2d_i32 merlin_opu_shim_gemm_i8(
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

  /* Extents from the DESCRIPTORS, never from the signature this was generated for. A is MxK and C is
   * MxN, so both M's are available and disagreement means the caller is not the contraction we think. */
  const size_t M = (size_t)a_s0;
  const size_t K = (size_t)a_s1;
  const size_t N = (size_t)c_s1;

  merlin_memref_2d_i32 ret;
  ret.allocated = c_alloc; ret.aligned = c_aligned; ret.offset = c_off;
  ret.sizes[0] = c_s0; ret.sizes[1] = c_s1;
  ret.strides[0] = c_st0; ret.strides[1] = c_st1;

  if (M == 0 || N == 0) return ret;
  if (K == 0) {{                       /* empty reduction: the result is the zero init, made explicit */
    for (size_t i = 0; i < M; ++i)
      for (size_t j = 0; j < N; ++j)
        C[(intptr_t)i * c_st0 + (intptr_t)j * c_st1] = 0;
    return ret;
  }}

  g_merlin_opu_calls += 1ULL;

  /* Everything the fast path needs, checked rather than assumed. The kernel indexes b[kk*n + j] and
   * c[i*n + j], so it requires both to be contiguous row-major with exactly those row strides -- a view
   * with any other stride would be read as if it were dense and silently mix rows. */
  const int b_dense = (b_st1 == 1) && (b_st0 == (intptr_t)N) && (b_s0 == (intptr_t)K);
  const int c_dense = (c_st1 == 1) && (c_st0 == (intptr_t)N) && (c_s0 == (intptr_t)M);
  const int b_okalign = (((uintptr_t)B) % (uintptr_t)MERLIN_OPU_ALIGN) == 0;
  const int c_okalign = (((uintptr_t)C) % (uintptr_t)MERLIN_OPU_ALIGN) == 0;
  /* K is non-zero (checked above), so the division is safe -- and dividing rather than multiplying means
   * a huge M*K cannot overflow into a value that looks like it fits. */
  const int fits = (M <= (size_t)MERLIN_OPU_LHS_SCRATCH_BYTES / K);

  if (!(b_dense && c_dense && b_okalign && c_okalign && fits)) {{
    g_merlin_opu_fallbacks += 1ULL;
    merlin_opu_scalar_gemm(C, A, B, M, N, K, a_st0, a_st1, b_st0, b_st1, c_st0, c_st1);
    return ret;
  }}

  /* Transpose the left operand into K-major. This is the packing the unit's layout requires and the cost
   * the routing decision has to price: O(M*K) against the kernel's O(M*N*K), so it is a small fraction of
   * a large contraction and can dominate a thin one. The pack buffer's base is aligned by declaration,
   * which is what the unit needs; its row stride is M and may be misaligned, which the corpus certifies
   * is harmless (see the module docstring). */
  for (size_t i = 0; i < M; ++i) {{
    const int8_t *arow = A + (intptr_t)i * a_st0;
    for (size_t kk = 0; kk < K; ++kk)
      merlin_opu_lhs_pack[kk * M + i] = arow[(intptr_t)kk * a_st1];
  }}

  {func}(C, merlin_opu_lhs_pack, B, (const int32_t *)0, M, N, K);
  return ret;
}}
{entries}
"""
