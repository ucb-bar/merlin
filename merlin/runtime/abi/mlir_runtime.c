/* Merlin's implementation of the MLIR C runtime symbols the lowered model calls.
 *
 * `memref.copy` lowers to a call to `memrefCopy` (normally provided by MLIR's
 * libmlir_c_runner_utils). Merlin owns the runtime ABI, so we provide it directly —
 * one freestanding implementation linked into both the host .so and the bare-metal
 * spike/Zephyr image. Signature + descriptor layout match upstream MLIR exactly.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

/* math.rsqrt lowers (via convert-math-to-libm) to rsqrtf/rsqrt, which are NOT in
 * standard libm/newlib. Provide them (reciprocal sqrt). */
float rsqrtf(float x) { return 1.0f / sqrtf(x); }
double rsqrt(double x) { return 1.0 / sqrt(x); }

/* math.roundeven lowers to roundevenf/roundeven (round-half-to-even / banker's
 * rounding) — a C23 libm symbol the Zephyr SDK's picolibc/newlib doesn't export.
 * rintf with the default rounding mode (round-to-nearest-even) IS round-half-to-even,
 * so it's the exact implementation. (bitvla and other models emit math.roundeven.) */
float roundevenf(float x) { return rintf(x); }
double roundeven(double x) { return rint(x); }

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
 *   - riscv64 lp64d / bare-metal soft-bf16: no bf16 FP register; `__bf16` is ABI-passed
 *     as a 16-bit integer (a0) — `unsigned short` matches, and `__bf16` may be unavailable
 *     under -ffreestanding, so keep `unsigned short` there.
 * The conversion math is identical either way (done on a 16-bit pattern via memcpy);
 * only the declared parameter/return TYPE — hence the register class — differs.
 * __truncsfbf2 uses round-half-to-even (matches torch). */
#if defined(__x86_64__) || defined(__i386__)
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

  int64_t idx[16] = {0}; /* MLIR memrefs in this model are <= 6-D */
  for (int64_t lin = 0; lin < total; lin++) {
    int64_t s_off = 0, d_off = 0;
    for (int64_t i = 0; i < rank; i++) {
      s_off += idx[i] * s_strides[i];
      d_off += idx[i] * d_strides[i];
    }
    memcpy(d_base + d_off * elem_size, s_base + s_off * elem_size,
           (size_t)elem_size);
    for (int64_t i = rank - 1; i >= 0; i--) {
      if (++idx[i] < sizes[i])
        break;
      idx[i] = 0;
    }
  }
}
