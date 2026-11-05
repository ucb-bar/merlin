/* Merlin's implementation of the MLIR C runtime symbols the lowered model calls.
 *
 * `memref.copy` lowers to a call to `memrefCopy` (normally provided by MLIR's
 * libmlir_c_runner_utils). Merlin owns the runtime ABI, so we provide it directly —
 * one freestanding implementation linked into both the host .so and the bare-metal
 * spike/Zephyr image. Signature + descriptor layout match upstream MLIR exactly.
 */
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* math.rsqrt lowers (via convert-math-to-libm) to rsqrtf/rsqrt, which are NOT in
 * standard libm/newlib. Provide them (reciprocal sqrt). */
float rsqrtf(float x) { return 1.0f / sqrtf(x); }
double rsqrt(double x) { return 1.0 / sqrt(x); }

/* The MLIR roundeven contract is independent of the caller's rounding mode.
 * Some native intrinsic lowerings also call these symbols. rint is unsuitable:
 * it follows the ambient mode. Work on the software ABI's IEEE binary words;
 * finite values need no FP arithmetic. NaNs use x+x to retain the platform's
 * normal quieting/canonicalization. This is a value contract, not an FP-flag or
 * errno guarantee. These checks establish C-format prerequisites, not RTL support. */
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && sizeof(double) == 8 &&
               FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
               DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "roundeven runtime requires IEEE binary32/binary64 C representations");

#define MERLIN_ROUNDEVEN(NAME, FT, UT, FRACTION, BIAS)                         \
FT NAME(FT x) {                                                              \
  UT bits;                                                                  \
  __builtin_memcpy(&bits, &x, sizeof(bits));                                  \
  const UT sign = bits & ((UT)1 << (sizeof(UT) * CHAR_BIT - 1));               \
  const UT magnitude = bits ^ sign;                                         \
  const unsigned exponent = (unsigned)(magnitude >> (FRACTION));              \
  if (exponent >= (BIAS) + (FRACTION))                                       \
    return exponent == 2 * (BIAS) + 1 ? x + x : x;                           \
  if (exponent < (BIAS) - 1) {                                              \
    bits = sign;                                                            \
  } else if (exponent == (BIAS) - 1) {                                      \
    const UT fraction_mask = ((UT)1 << (FRACTION)) - 1;                       \
    bits = sign | ((magnitude & fraction_mask) ? (UT)(BIAS) << (FRACTION) : 0); \
  } else {                                                                  \
    const unsigned shift = (BIAS) + (FRACTION) - exponent;                    \
    const UT mask = ((UT)1 << shift) - 1;                                    \
    const UT half = (UT)1 << (shift - 1);                                    \
    bits = sign | ((magnitude + half - 1 + ((magnitude >> shift) & 1)) & ~mask); \
  }                                                                         \
  FT result;                                                                \
  __builtin_memcpy(&result, &bits, sizeof(result));                           \
  return result;                                                            \
}
#ifndef MERLIN_EXTERNAL_ROUNDEVEN_F32
MERLIN_ROUNDEVEN(roundevenf, float, uint32_t, FLT_MANT_DIG - 1, FLT_MAX_EXP - 1)
#endif
MERLIN_ROUNDEVEN(roundeven, double, uint64_t, DBL_MANT_DIG - 1, DBL_MAX_EXP - 1)
#undef MERLIN_ROUNDEVEN

/* bf16 soft-conversion compiler-rt builtins. The Zephyr SDK's libgcc (riscv64,
 * gcc 12.2) does not export them, but models with bf16 activations / casts outside
 * the f32-accumulated matmul path (e.g. smolvla) emit float<->bf16 conversions.
 *
 * CRITICAL: the bf16 value must travel through the SAME register class the *caller*
 * (clang's `fmul bfloat` legalization) uses, or the call corrupts every bf16 element.
 * The bf16 ABI register class is target-dependent:
 *   - x86-64 SysV: `__bf16` is an SSE-class scalar passed/returned in xmm0. Declaring
 *     these helpers with `unsigned short` (an INTEGER-class type in rdi/ax) silently
 *     mismatches the convention — the bf16 arg/return goes through the wrong register,
 *     so `0.066f * 31.0f` came back as ~8.6e9 (host smolvla cos 0.083). Use `__bf16`.
 *   - riscv64 lp64d under clang: LLVM's legalized bf16 calls pass and return `__bf16`
 *     through fa0 even without a native bf16 ISA extension. Declaring the runtime helper as
 *     `unsigned short` returns in a0 instead. On K1 this turned every integer-to-bf16 conversion
 *     in smolVLA's dequantized weights into zero: the caller read the unchanged float in fa0 and
 *     stored its low 16 bits. SpacemiT clang-19 supports `__bf16` in freestanding mode, so use the
 *     real type for clang RISC-V too. Non-clang freestanding toolchains retain the integer fallback;
 *     their lowered model and runtime must be produced by an ABI-compatible compiler pair.
 * The conversion math is identical either way (done on a 16-bit pattern via memcpy);
 * only the declared parameter/return TYPE — hence the register class — differs.
 * __truncsfbf2 uses round-half-to-even (matches torch). */
#if defined(__x86_64__) || defined(__i386__) || \
    (defined(__riscv) && defined(__clang__))
typedef __bf16 merlin_bf16_t;
#else
typedef unsigned short merlin_bf16_t;
#endif
static inline unsigned short merlin_bf16_bits(merlin_bf16_t b) {
  unsigned short r;
  __builtin_memcpy(&r, &b, 2);
  return r;
}
static inline merlin_bf16_t merlin_bf16_from_bits(unsigned short r) {
  merlin_bf16_t b;
  __builtin_memcpy(&b, &r, 2);
  return b;
}
merlin_bf16_t __truncsfbf2(float f) {
  unsigned int x;
  __builtin_memcpy(&x, &f, 4);
  unsigned int exp = (x >> 23) & 0xFF, man = x & 0x7FFFFF;
  if (exp == 0xFF)
    return merlin_bf16_from_bits((unsigned short)((x >> 16) | (man ? 0x0040u : 0u))); /* inf / NaN */
  unsigned int bias = 0x7FFFu + ((x >> 16) & 1u);                                     /* round-half-even */
  return merlin_bf16_from_bits((unsigned short)((x + bias) >> 16));
}
float __extendbfsf2(merlin_bf16_t b) {
  unsigned int x = (unsigned int)merlin_bf16_bits(b) << 16;
  float f;
  __builtin_memcpy(&f, &x, 4);
  return f;
}

/* UnrankedMemRefType<char>: {rank, ptr-to-descriptor}.
 * Descriptor layout: void* allocated; void* aligned; int64_t offset;
 *                    int64_t sizes[rank]; int64_t strides[rank]. */
typedef struct {
  int64_t rank;
  void *descriptor;
} merlin_unranked_memref_t;

/* Each descriptor is indexed by ITS OWN rank, and a disagreement is refused rather than computed through.
 *
 * The strides of a ranked descriptor start at `2 + 1 + rank` words, so indexing the DESTINATION's strides
 * with the SOURCE's rank reads past them into whatever follows — in a real image, the adjacent
 * descriptor's `allocated`/`aligned` POINTER, which then gets multiplied by the element size and added to
 * the base. That is a defensive fix for a real hazard the previous code had, and it makes the precondition
 * upstream MLIR merely asserts into something this reports.
 *
 * WHAT IT IS NOT: it is NOT the explanation for the whole-model store fault on spike. That fault was
 * traced to this function — on deepjscc int8 the destination offset accumulates to 0xc0595c80, an arena
 * POINTER rather than a stride, and multiplying by a 4-byte element size gives exactly the 0x301657200 the
 * commit trace shows before mcause 7 — but the guard below does NOT fire on that run, so the two unranked
 * descriptors agree on rank and something else puts a pointer where a stride belongs. At the faulting
 * iteration the loaded strides are 1 and the loaded index is 0, i.e. the values being read are sane and
 * the pointer entered the accumulator at an earlier dimension; the loop reaches i = 11, so the rank is at
 * least 12 while this function's scratch index array is sized 16 and its comment assumes 6-D. Both facts
 * are recorded because they bound where to look next, not because they are the answer.
 *
 * The counters exist because refusing is still a wrong answer — the copy did not happen — and a run that
 * silently skipped copies would grade badly with no explanation. Refusing beats storing: a skipped copy is
 * a bad number, a wild store is a dead image or memory corrupted somewhere else entirely. */
unsigned long long g_merlin_memref_rank_mismatch = 0ULL;
long long g_merlin_memref_bad_src_rank = -1;
long long g_merlin_memref_bad_dst_rank = -1;
unsigned long long merlin_memref_rank_mismatches(void) { return g_merlin_memref_rank_mismatch; }

/* The per-dimension counter below is a fixed-size stack array, so a rank larger than it holds would be
 * written straight past its end -- a stack smash inside the one function this file exists to make safe,
 * and one whose damage would land on whatever the compiler placed next. The bound is checked rather than
 * commented about: the previous note said "memrefs in this model are <= 6-D" while the trace helper next
 * to it already capped its own loop at 16, and the runtime has since logged a rank of 11. Which model is
 * being run is not this function's business to assume. */
#define MERLIN_MEMREF_MAX_RANK 16
unsigned long long g_merlin_memref_rank_too_large = 0ULL;
long long g_merlin_memref_worst_rank = -1;
unsigned long long merlin_memref_ranks_too_large(void) { return g_merlin_memref_rank_too_large; }

/* OPT-IN DESCRIPTOR TRACE (-DMERLIN_MEMREF_TRACE), off by default so a normal build is byte-identical.
 *
 * This exists because the alternative is inferring descriptor contents from a spike commit trace, which is
 * how two wrong hypotheses got as far as being implemented. Printing what memrefCopy was actually handed --
 * rank, sizes, strides, bases -- answers directly which field holds a pointer.
 *
 * The console is reached through WEAK references: on a bare-metal image htif_puts/htif_putd resolve to the
 * harness, and in the host shared object they are absent and the calls are skipped. That keeps one file
 * building for both without a second copy or a build-system change. */
#ifdef MERLIN_MEMREF_TRACE
__attribute__((weak)) void htif_puts(const char *s);
__attribute__((weak)) void htif_putd(long v);

static void trace_desc(const char *tag, int64_t rank, void **desc) {
  if (!htif_puts || !htif_putd)
    return;
  int64_t *rest = (int64_t *)(desc + 2);
  htif_puts(tag);
  htif_puts(" rank ");
  htif_putd((long)rank);
  /* The descriptor's own ADDRESS. Without it there is no way to tell a wrong pointer being passed from a
   * right pointer whose contents are stale -- and the observed corruption (words 6..9 replaced by what
   * looks like an adjacent descriptor) is consistent with either. */
  htif_puts(" at ");
  htif_putd((long)(uintptr_t)desc);
  htif_puts(" aligned ");
  htif_putd((long)(uintptr_t)desc[1]);
  htif_puts(" off ");
  htif_putd((long)rest[0]);
  htif_puts(" sizes");
  for (int64_t i = 0; i < rank && i < 16; i++) {
    htif_puts(" ");
    htif_putd((long)rest[1 + i]);
  }
  htif_puts(" strides");
  for (int64_t i = 0; i < rank && i < 16; i++) {
    htif_puts(" ");
    htif_putd((long)rest[1 + rank + i]);
  }
  htif_puts("\n");
}
#endif

/* Bound a positive-stride view without overflowing the address calculation.
 * A failed bound leaves the original element-at-a-time copy path in place. */
static int copy_view_end(uintptr_t base, int64_t rank, const int64_t *sizes,
                         const int64_t *strides, int64_t elem_size,
                         uintptr_t *end) {
  int64_t last = 0, span, bytes;
  if (elem_size <= 0)
    return 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (sizes[i] <= 0 || strides[i] < 0 ||
        __builtin_mul_overflow(sizes[i] - 1, strides[i], &span) ||
        __builtin_add_overflow(last, span, &last))
      return 0;
  }
  if (__builtin_add_overflow(last, (int64_t)1, &last) ||
      __builtin_mul_overflow(last, elem_size, &bytes) ||
      (uint64_t)bytes > UINTPTR_MAX - base)
    return 0;
  *end = base + (uintptr_t)bytes;
  return 1;
}

void memrefCopy(int64_t elem_size, merlin_unranked_memref_t *src_u,
                merlin_unranked_memref_t *dst_u) {
  int64_t rank = src_u->rank;
  int64_t d_rank = dst_u->rank;
  if (rank != d_rank) {
    if (g_merlin_memref_rank_mismatch == 0ULL) {
      g_merlin_memref_bad_src_rank = rank;
      g_merlin_memref_bad_dst_rank = d_rank;
    }
    ++g_merlin_memref_rank_mismatch;
    return;
  }
  if (rank > MERLIN_MEMREF_MAX_RANK) {
    /* Same policy as the mismatch above: refuse and count. A skipped copy is a bad number; writing past
     * `idx` is corruption somewhere else entirely, which is far harder to trace back to here. */
    if (rank > g_merlin_memref_worst_rank)
      g_merlin_memref_worst_rank = rank;
    ++g_merlin_memref_rank_too_large;
    return;
  }
  void **sdesc = (void **)src_u->descriptor;
  void **ddesc = (void **)dst_u->descriptor;
#ifdef MERLIN_MEMREF_TRACE
  trace_desc("MEMREF src", rank, sdesc);
  trace_desc("MEMREF dst", d_rank, ddesc);
#endif
  char *s_aligned = (char *)sdesc[1];
  char *d_aligned = (char *)ddesc[1];
  int64_t *s_rest = (int64_t *)(sdesc + 2); /* offset, sizes[rank], strides[rank] */
  int64_t *d_rest = (int64_t *)(ddesc + 2);
  char *s_base = s_aligned + s_rest[0] * elem_size;
  char *d_base = d_aligned + d_rest[0] * elem_size;
  int64_t *sizes = s_rest + 1;
  int64_t *s_strides = s_rest + 1 + rank;
  int64_t *d_strides = d_rest + 1 + d_rank;   /* the DESTINATION's own rank */

  if (rank == 0) {
    memcpy(d_base, s_base, (size_t)elem_size);
    return;
  }

  int64_t total = 1;
  for (int64_t i = 0; i < rank; i++)
    total *= sizes[i];

  /* Copy a common contiguous suffix as one byte span. Preserve the old
   * lexicographic element order for overlapping or unbounded views: merging
   * their copies could otherwise change which source bytes are subsequently
   * read. Singleton axes impose no physical-stride constraint. */
  int64_t copy_rank = rank, chunk_elements = 1;
  uintptr_t s_end, d_end;
  if (total > 0 &&
      copy_view_end((uintptr_t)s_base, rank, sizes, s_strides, elem_size, &s_end) &&
      copy_view_end((uintptr_t)d_base, rank, sizes, d_strides, elem_size, &d_end) &&
      (s_end <= (uintptr_t)d_base || d_end <= (uintptr_t)s_base)) {
    for (int64_t i = rank - 1; i >= 0; --i) {
      if (sizes[i] != 1 &&
          (s_strides[i] != chunk_elements || d_strides[i] != chunk_elements))
        break;
      int64_t next;
      if (__builtin_mul_overflow(chunk_elements, sizes[i], &next) ||
          (uint64_t)next > SIZE_MAX / (uint64_t)elem_size)
        break;
      chunk_elements = next;
      copy_rank = i;
    }
  }
  total /= chunk_elements;

  int64_t idx[MERLIN_MEMREF_MAX_RANK] = {0};   /* bounded above, not assumed */
  for (int64_t lin = 0; lin < total; lin++) {
    int64_t s_off = 0, d_off = 0;
    for (int64_t i = 0; i < copy_rank; i++) {
      s_off += idx[i] * s_strides[i];
      d_off += idx[i] * d_strides[i];
    }
    memcpy(d_base + d_off * elem_size, s_base + s_off * elem_size,
           (size_t)elem_size * (size_t)chunk_elements);
    for (int64_t i = copy_rank - 1; i >= 0; i--) {
      if (++idx[i] < sizes[i])
        break;
      idx[i] = 0;
    }
  }
}
