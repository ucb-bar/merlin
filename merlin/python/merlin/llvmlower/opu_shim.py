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

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..kernels.opu_kernel import KernelSpec, emit_microkernel

__all__ = ["CONTRACT_PATH", "DEFAULT_SCRATCH_BYTES", "OpuBuild", "ShimUnit", "UnitContract",
           "build_object", "derive_encodings", "emit_translation_unit", "load_contract",
           "scratch_bytes_for"]

#: Where a unit's derivation entry points are declared. Tracked and reviewed, so which sources the
#: compiler reads is a diff someone sees rather than a literal buried in a call.
CONTRACT_PATH = "merlin/contract/matrix_units.yaml"

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


def is_batched(sig: Sequence[int]) -> bool:
    """Whether a signature is ``(B, M, N, K)`` rather than ``(M, N, K)``.

    The arity IS the discriminator — a batched contraction has one more parallel dim, and that is the
    whole difference between the two entry shapes. Keeping it a length test rather than a flag means a
    caller cannot pass a batched signature and forget to say so.
    """
    return len(tuple(sig)) == 4


def scratch_bytes_for(signatures: Mapping[str, Sequence[int]]) -> int:
    """Bytes the left-operand pack needs: the largest ``M * K`` any signature in this unit will pack.

    One buffer serves every signature because model inference calls them in sequence — the pack is dead
    the moment the kernel returns. Sizing it to the maximum rather than per signature is what keeps this
    from being 5 buffers and ~600 KB of ``.bss`` for spectformer instead of ~200 KB.

    A BATCHED signature packs one batch slice at a time, so its demand is ``M * K`` too, not ``B * M * K``
    — the batch loop reuses the same buffer, which is only sound because the slices run in sequence.
    """
    if not signatures:
        return DEFAULT_SCRATCH_BYTES
    return max(int(s[-3]) * int(s[-1]) for s in signatures.values())


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


def _entry(symbol: str, sig: Sequence[int], kernel_func: str) -> str:
    """One MLIR-ABI entry point, rank-2 or rank-3 according to the signature's arity.

    Every entry is the same code — the shared body reads its extents from the descriptors. They exist as
    separate symbols only because MLIR function types are monomorphic, so a 256x196/K=768 contraction and
    a 196x1024/K=256 one cannot call the same declaration. The generated-time extents go into a comment
    and a cheap consistency check, never into the arithmetic.
    """
    if is_batched(sig):
        return _batched_entry(symbol, sig)
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


def _batched_entry(symbol: str, sig: Sequence[int]) -> str:
    """A rank-3 entry: one batch of independent contractions, run as a loop over the rank-2 body.

    The batch dimension is a LOOP, not a third tile axis. The unit contracts two dimensions and reduces a
    third; a batch is just many of those, and each slice's operands and result are disjoint, so the whole
    batched form is the existing certified path called `B` times with the leading offset advanced. That is
    why this reuses `merlin_opu_shim_gemm_i8` rather than growing a second kernel: every legality check
    (density, alignment, scratch capacity) and the scalar fallback come along unchanged, and the code the
    unit executes is still the certified microkernel.

    Offsets rather than pointers: the descriptor's `offset` is in ELEMENTS, so advancing it by
    `b * stride0` names the slice base without any cast, and the rank-2 body sees exactly the descriptor
    it would have seen had the slice been passed directly.

    The pack scratch is reused across slices, which is sound only because the slices run in sequence
    here -- see `scratch_bytes_for`, which sizes for one slice for the same reason.
    """
    b, m, n, k = (int(v) for v in sig)
    return f"""
/* {symbol}: generated for B={b} M={m} N={n} K={k} (packs {m * k} bytes per batch slice). The extents are
 * re-read from the descriptors below, so this entry stays correct if it is ever called with others. */
merlin_memref_3d_i32 {symbol}(
    int8_t *a_alloc, int8_t *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1, intptr_t a_s2,
    intptr_t a_st0, intptr_t a_st1, intptr_t a_st2,
    int8_t *b_alloc, int8_t *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1, intptr_t b_s2,
    intptr_t b_st0, intptr_t b_st1, intptr_t b_st2,
    int32_t *c_alloc, int32_t *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1, intptr_t c_s2,
    intptr_t c_st0, intptr_t c_st1, intptr_t c_st2)
{{
  merlin_memref_3d_i32 ret;
  ret.allocated = c_alloc; ret.aligned = c_aligned; ret.offset = c_off;
  ret.sizes[0] = c_s0; ret.sizes[1] = c_s1; ret.sizes[2] = c_s2;
  ret.strides[0] = c_st0; ret.strides[1] = c_st1; ret.strides[2] = c_st2;

  for (intptr_t bi = 0; bi < c_s0; ++bi) {{
    merlin_opu_shim_gemm_i8(a_alloc, a_aligned, a_off + bi * a_st0, a_s1, a_s2, a_st1, a_st2,
                            b_alloc, b_aligned, b_off + bi * b_st0, b_s1, b_s2, b_st1, b_st2,
                            c_alloc, c_aligned, c_off + bi * c_st0, c_s1, c_s2, c_st1, c_st2);
    /* WAR on the pack scratch, and it is specific to the batched form. Every slice packs into the SAME
     * buffer, so slice bi+1's scalar stores target bytes slice bi's kernel read -- and on a decoupled
     * vector machine the scalar core runs ahead of the vector load/store unit, so those stores can land
     * while the previous slice's operand loads are still outstanding. The rank-2 path cannot hit this:
     * it packs once and never writes the buffer again while a kernel is reading it.
     *
     * Neither of the two things that would have caught it can: the acceptance corpus hands the kernel
     * pre-transposed operands and never packs, and spike orders memory functionally.
     *
     * Guarded on the ISA rather than on the scalar stand-in: the host build of this same source is the
     * one the unit tests compile and run, and it is not RISC-V. */
#ifdef __riscv
    __asm__ volatile("fence rw, rw" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");   /* the compiler-barrier half, which is all a host needs */
#endif
  }}
  return ret;
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

/* The rank-3 form, for batched contractions. Same layout one rank up -- the lowered convention passes a
 * memref as (allocated, aligned, offset, sizes..., strides...), so the rank decides the arity and
 * nothing else changes. */
typedef struct {{
  int32_t *allocated;
  int32_t *aligned;
  intptr_t offset;
  intptr_t sizes[3];
  intptr_t strides[3];
}} merlin_memref_3d_i32;

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

/* Block edge for the pack loop below. A LOCALITY knob, not a hardware fact: every block edge computes
 * exactly the same bytes, so nothing about correctness depends on it and a wrong value costs time only.
 * The default is one cache line's worth of the M axis, which is the shape that makes each output line be
 * written once instead of once per surrounding row; override with -DMERLIN_OPU_PACK_BLK=<n>. */
#ifndef MERLIN_OPU_PACK_BLK
#define MERLIN_OPU_PACK_BLK 64
#endif

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
   * is harmless (see the module docstring).
   *
   * BLOCKED, because the ORDER of an O(M*K) loop decides whether it behaves like O(M*K) memory traffic or
   * like M passes over the whole buffer. Writing `pack[kk*M + i]` down a whole column of k puts every
   * consecutive store M bytes apart -- a different cache line each time -- and then the next row of A
   * revisits every one of those lines to fill its next byte. The buffer is M*K bytes and does not fit a
   * cache at these extents, so each line is fetched and evicted repeatedly and the pack costs far more
   * than the bytes it moves. Iterating a block at a time keeps both sides local: the read stays along A's
   * contiguous k, and the writes stay inside BLK output lines until they are finished with. */
  /* Order the pack against the PREVIOUS call's kernel reads.
   *
   * The batched loop below fences between slices for the WAR on this same scratch. The rank-2 path was
   * left unfenced on the reasoning that it "packs once and never writes the buffer again while a kernel
   * is reading it" -- true WITHIN one call, false ACROSS calls: every routed entry point shares this one
   * global buffer, so call N+1's scalar pack stores target bytes call N's kernel is still loading, and on
   * a decoupled vector machine the scalar core runs ahead of the vector load/store unit.
   *
   * MEASURED on the taped-out part: whole-model spectformer graded cos 0.98288649 / rel 1.847e-01 /
   * max_rel 14.9 with 57 routed calls sharing the scratch -- most values right, a few badly wrong, the
   * signature of a stale read rather than a geometry error. The identical image is bit-exact on FireSim
   * (cos 1.0 / rel 0.0), which is why neither the acceptance corpus (pre-transposed operands, never
   * packs) nor spike (orders memory functionally) nor the FPGA could catch it.
   *
   * Placed BEFORE the pack rather than after the call so it also covers the first call after any other
   * writer of this buffer, and so a fenceless tail cannot be reintroduced by an early return. */
#ifdef __riscv
  __asm__ volatile("fence rw, rw" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
#if defined(__riscv) && !defined(MERLIN_OPU_SCALAR_PACK)
  /* VECTORISED pack, and the loop order is swapped relative to the scalar form below.
   *
   * The scalar pack measured 24.79 cycles PER ELEMENT over 4,009,984 elements -- 87% of the whole routed
   * region, against a kernel that is only 13% of it. Two things cost that much, and the order fixes one
   * while the vector ops fix the other.
   *
   * ORDER: with `kk` innermost the WRITE walks `pack[kk*M + i]` at stride M -- a different line per store.
   * With `i` innermost the write is CONTIGUOUS along the pack row and the strided side becomes the READ of
   * A's column. On a machine whose per-line cost dominates, a sequential write stream is what you want,
   * and blocking keeps the strided reads inside a resident tile.
   *
   * VECTOR: `vlse8.v` gathers a whole column segment in one instruction (VLMAX = 64 at e8/m1 for a 512-bit
   * unit) and `vse8.v` lays it down contiguously, replacing 2*BLK scalar ops per segment with 3.
   *
   * vtype is set here and the kernel always issues its own `vsetvli` before using the unit, so leaving a
   * vtype behind cannot affect it. The scratch is not read by anything until the kernel call below. */
  for (size_t k0 = 0; k0 < K; k0 += MERLIN_OPU_PACK_BLK) {{
    const size_t k1 = (K - k0) < MERLIN_OPU_PACK_BLK ? K : k0 + MERLIN_OPU_PACK_BLK;
    for (size_t i0 = 0; i0 < M; i0 += MERLIN_OPU_PACK_BLK) {{
      const size_t i1 = (M - i0) < MERLIN_OPU_PACK_BLK ? M : i0 + MERLIN_OPU_PACK_BLK;
      for (size_t kk = k0; kk < k1; ++kk) {{
        int8_t *dst = merlin_opu_lhs_pack + kk * M;
        const int8_t *src = A + (intptr_t)kk * a_st1;
        size_t i = i0;
        while (i < i1) {{
          size_t vl;
          __asm__ volatile("vsetvli %0, %1, e8, m1, ta, ma" : "=r"(vl) : "r"(i1 - i));
          __asm__ volatile("vlse8.v v24, (%0), %1"
                           :: "r"(src + (intptr_t)i * a_st0), "r"((intptr_t)a_st0)
                           : "memory");
          __asm__ volatile("vse8.v v24, (%0)" :: "r"(dst + i) : "memory");
          i += vl;
        }}
      }}
    }}
  }}
#else
  for (size_t i0 = 0; i0 < M; i0 += MERLIN_OPU_PACK_BLK) {{
    const size_t i1 = (M - i0) < MERLIN_OPU_PACK_BLK ? M : i0 + MERLIN_OPU_PACK_BLK;
    for (size_t k0 = 0; k0 < K; k0 += MERLIN_OPU_PACK_BLK) {{
      const size_t k1 = (K - k0) < MERLIN_OPU_PACK_BLK ? K : k0 + MERLIN_OPU_PACK_BLK;
      for (size_t i = i0; i < i1; ++i) {{
        const int8_t *arow = A + (intptr_t)i * a_st0;
        for (size_t kk = k0; kk < k1; ++kk)
          merlin_opu_lhs_pack[kk * M + i] = arow[(intptr_t)kk * a_st1];
      }}
    }}
  }}
#endif

  {func}(C, merlin_opu_lhs_pack, B, (const int32_t *)0, M, N, K);
  return ret;
}}
{entries}
"""


# ==================================================================================================
# Building the object: derive the facts, emit, compile.
# ==================================================================================================


@dataclass(frozen=True)
class UnitContract:
    """One matrix unit's derivation entry points, as declared in :data:`CONTRACT_PATH`.

    Every field is an ADDRESS — a file to read or a declaration to look for — never a value. The values
    are read out of those files at build time and a value that cannot be read fails closed.
    """

    unit: str
    pin: str
    root_env: str
    path: str
    sources: dict[str, str]
    declarations: dict[str, Any]
    configs: dict[str, Any]
    kernel_roles: dict[str, str]

    def checkout(self) -> Path:
        """The directory the unit's sources live in, resolved the way the rest of the repo resolves it.

        ``paths.env`` rather than ``os.environ``: the checkout is declared in the gitignored ``.env``, and
        reading only the process environment made an earlier consumer of this silently skip on a host
        where the hardware was present.
        """
        from ..common.paths import env as _env
        root = _env(self.root_env)
        if not root:
            raise FileNotFoundError(
                f"${self.root_env} is unset, so the sources for {self.unit!r} cannot be located and its "
                "instruction encodings cannot be derived; refusing to emit a kernel from guessed words")
        return Path(root) / self.path if self.path else Path(root)

    def source(self, name: str) -> Path:
        rel = self.sources.get(name)
        if rel is None:
            raise KeyError(f"{self.unit!r} declares no {name!r} source in {CONTRACT_PATH}")
        return self.checkout() / rel

    def root(self) -> Path:
        """The integrating SoC's root, which is where its own configs live."""
        from ..common.paths import env as _env
        got = _env(self.root_env)
        if not got:
            raise FileNotFoundError(f"${self.root_env} is unset, so {self.unit!r} cannot be located")
        return Path(got)

    def config_scala(self) -> list[Path]:
        """Every file that may declare a named configuration of this unit.

        Both the unit's own generator and the integrating SoC's repo, because a heterogeneous config -- the
        unit on one tile beside something else -- is declared by the SoC, and those are the configs real
        bitstreams tend to be built from. Missing files are dropped rather than raising: a contract may name
        a config location that a given checkout does not have, and the failure that matters is "the named
        config was not found in any of them", which the geometry derivation already reports.
        """
        out = [self.checkout() / str(p) for p in self.configs.get("config_scala", ())]
        out += [self.root() / str(p) for p in self.configs.get("host_config_scala", ())]
        return [p for p in out if p.is_file()]

    def mixin_scala(self) -> list[Path]:
        return [p for p in (self.checkout() / str(m)
                            for m in self.configs.get("mixin_scala", ())) if p.is_file()]

    def geometry(self, config: str) -> tuple[int, int]:
        """``(tile_edge, operand_alignment_bytes)`` for a named configuration, derived from its own Scala."""
        from ..kernels import opu_cert
        cfgs, mixins = self.config_scala(), self.mixin_scala()
        return (opu_cert.tile_edge_for_config(config, config_scala=cfgs, mixin_scala=mixins),
                opu_cert.operand_alignment_for_config(config, config_scala=cfgs, mixin_scala=mixins))

    def spec(self, **overrides) -> KernelSpec:
        """The kernel spec for this unit, with the instruction roles taken from the contract."""
        roles = {k: self.kernel_roles[k] for k in ("accumulate", "broadcast", "readout")}
        return KernelSpec(**roles, **overrides)


def load_contract(unit: str, *, path: "str | Path | None" = None) -> UnitContract:
    """Read one unit's block out of the contract file."""
    import yaml

    from ..common.paths import repo_root
    p = Path(path) if path is not None else Path(repo_root()) / CONTRACT_PATH
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    units = payload.get("units") or {}
    if unit not in units:
        raise KeyError(f"{p} declares no matrix unit {unit!r}; declared: {sorted(units)}")
    block = units[unit]
    missing = [k for k in ("pin", "root_env", "sources", "declarations", "configs", "kernel_roles")
               if k not in block]
    if missing:
        raise ValueError(f"{p}: unit {unit!r} is missing {missing}")
    return UnitContract(unit=unit, pin=str(block["pin"]), root_env=str(block["root_env"]),
                        path=str(block.get("path", "")), sources=dict(block["sources"]),
                        declarations=dict(block["declarations"]), configs=dict(block["configs"]),
                        kernel_roles=dict(block["kernel_roles"]))


def derive_encodings(contract: UnitContract):
    """The unit's instruction encodings, derived from its RTL and cross-checked against its own header.

    Returns the cross-checked :class:`targetgen.rtl.opu_isa.IsaDerivation`. Its ``ok`` is forwarded into
    the emitter, which refuses to emit when the two sources disagree — the funct6 values are ChiselEnum
    ORDINALS, i.e. a function of declaration order in an RTL file, so a stale cross-check header and a
    moved slot are indistinguishable from one source alone.
    """
    from ..targetgen.rtl import opu_isa

    d = contract.declarations
    derived = opu_isa.derive(consts=contract.source("consts"),
                             instructions=contract.source("instructions"),
                             params=contract.source("params"),
                             funct6_enum=str(d["funct6_enum"]),
                             consts_container=str(d["consts_container"]),
                             insn_seq=str(d["insn_seq"]),
                             opcode_name=str(d["opcode_name"]),
                             form_funct3={str(k): str(v) for k, v in d["form_funct3"].items()})
    return opu_isa.crosscheck(derived, contract.source("crosscheck_header"),
                              pairs={str(k): str(v) for k, v in d["crosscheck_pairs"].items()})


@dataclass(frozen=True)
class OpuBuild:
    """What a build produced, and what it is a result ABOUT.

    ``provenance`` is not optional bookkeeping: the encodings in this object came from one revision of one
    checkout, and an object attributed to the wrong one is worse than no object because it links and runs.
    """

    object_path: Path
    source_path: Path
    signatures: dict[str, tuple[int, int, int]]
    alignment_bytes: int
    scratch_bytes: int
    tile_edge: int | None = None
    scalar_tile: bool = False
    #: Whether the tile loop was compiled with OpenMP, i.e. whether this object can reach more than one
    #: matrix unit. Recorded rather than inferred: on a chip with a unit per core, a serial object uses
    #: exactly one of them and still computes the right answer, so nothing else distinguishes the two.
    parallel_tiles: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"object": str(self.object_path), "source": str(self.source_path),
                "signatures": {k: list(v) for k, v in self.signatures.items()},
                "alignment_bytes": self.alignment_bytes, "scratch_bytes": self.scratch_bytes,
                "tile_edge": self.tile_edge, "scalar_tile": self.scalar_tile,
                "parallel_tiles": self.parallel_tiles,
                "provenance": self.provenance, "gaps": list(self.gaps)}


def build_object(signatures: Mapping[str, tuple[int, int, int]], work: "str | Path", *,
                 unit: str, config: str, cc: "str | Path", cflags: Sequence[str],
                 scalar_tile: bool = False,
                 parallel_tiles: bool = False,
                 scratch_bytes: int | None = None,
                 contract_path: "str | Path | None" = None) -> OpuBuild:
    """Derive the unit's facts, emit the translation unit, compile it, and record what it came from.

    ``unit`` names a block in :data:`CONTRACT_PATH` and is REQUIRED rather than defaulted: a default would
    make this module quietly about one particular extension, which is the overfit the repo's cardinal rule
    forbids. The caller knows which hardware it is building for; this does not.

    ``config`` names the elaborated hardware configuration (e.g. an OPU Shuttle config); the tile edge and
    the operand alignment are read out of ITS OWN declaration, so a build for a different configuration
    gets different geometry without any code change.

    ``scalar_tile`` compiles the tile body as the scalar stand-in instead of the unit's instructions. That
    is what lets a whole model be graded on a host or a plain simulator: it validates the routing, the
    pack, the descriptor ABI and the epilogue while proving nothing about the datapath, which is the
    certification's job. The build records which of the two it is, so a report cannot confuse them.

    ``parallel_tiles`` compiles the tile loop with OpenMP, which is the ONLY way an image reaches more
    than one matrix unit on a chip that has one per core: a routed contraction is a single opaque call by
    the time the parallel transform schedule runs, so the schedule cannot split it and the split has to
    live inside the kernel. Off by default, so a single-core image is compiled exactly as certified and
    acquires no runtime dependency. It is recorded on the build because a serial object on a multi-unit
    chip uses one unit and still computes the right answer -- there is no other symptom.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    contract = load_contract(unit, path=contract_path)

    # What revision is this a result about? Verified, never enforced by moving someone's checkout.
    from ..common import provenance as PROV
    gaps: list[str] = []
    # `reads` is exactly the contract's declared sources plus the config declarations, so an uncommitted
    # edit to one of THOSE is drift while a stray build log elsewhere in the tree is a note. Passing the
    # precise set rather than letting it default to the pin's requires_paths is what makes the answer about
    # this build instead of about the checkout in general.
    # Only the paths inside the PINNED checkout are checked against the pin: `host_config_scala` lives in
    # the integrating SoC's repo, which this pin does not describe and cannot speak for.
    reads = [*contract.sources.values(),
             *(str(c) for c in contract.configs.get("config_scala", ())),
             *(str(m) for m in contract.configs.get("mixin_scala", ()))]
    verification = PROV.verify(contract.pin, checkout=contract.checkout(), reads=reads)
    if not verification.ok:
        gaps.append(f"pin {contract.pin} drifted: "
                    + "; ".join([*verification.drift,
                                 *([f"missing {list(verification.missing_paths)}"]
                                   if verification.missing_paths else []),
                                 *([f"forbidden present {list(verification.forbidden_present)}"]
                                   if verification.forbidden_present else [])]))

    derived = derive_encodings(contract)
    if not derived.ok:
        raise ValueError(
            f"the encoding derivation for {unit!r} did not agree with its cross-check source "
            f"({[c for c in derived.crosschecks if not c.get('agrees')]}); refusing to build an object "
            "from an unresolved encoding, because a wrong field emits a neighbouring instruction rather "
            "than failing to assemble")

    tile_edge, alignment = contract.geometry(config)

    unit_src = emit_translation_unit(derived.encodings, signatures, spec=contract.spec(),
                                     alignment_bytes=alignment, derivation_ok=derived.ok,
                                     scratch_bytes=scratch_bytes)
    src = work / "merlin_opu_shim.c"
    src.write_text(str(unit_src), encoding="utf-8")

    cmd = [str(cc), *cflags, "-c", str(src), "-o", str(work / "merlin_opu_shim.o")]
    if parallel_tiles:
        # Defines _OPENMP, which is what the emitted pragma is guarded on.
        cmd[1:1] = ["-fopenmp"]
    if scalar_tile:
        # The host/simulator build needs the edge supplied, since there is no unit to ask for it.
        cmd[1:1] = ["-DOPU_SCALAR_TILE", f"-DOPU_TILE_EDGE={tile_edge}"]
    got = subprocess.run(cmd, capture_output=True, text=True)
    obj = work / "merlin_opu_shim.o"
    if got.returncode != 0 or not obj.is_file():
        raise RuntimeError(f"the emitted matrix-unit shim did not compile:\ncmd: {' '.join(cmd)}\n"
                           f"{got.stderr[-3000:]}")

    prov = PROV.record(pins={contract.pin: verification},
                       sources=[contract.source(k) for k in sorted(contract.sources)],
                       artifacts={"shim_object": obj, "shim_source": src})
    return OpuBuild(object_path=obj, source_path=src,
                    signatures={k: tuple(int(x) for x in v) for k, v in signatures.items()},
                    alignment_bytes=alignment, scratch_bytes=unit_src.scratch_bytes,
                    tile_edge=tile_edge, scalar_tile=scalar_tile, parallel_tiles=parallel_tiles,
                    provenance=prov, gaps=tuple(gaps))
