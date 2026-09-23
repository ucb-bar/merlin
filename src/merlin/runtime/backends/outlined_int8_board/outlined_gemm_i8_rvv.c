/* Reusable K1 W8A8 GEMM.
 *
 * C[M,N] = A[M,K] @ B[K,N], signed i8 inputs and exact signed i32 accumulation.
 * The routing pass proves that the linalg init is zero, so this kernel overwrites C.
 * B's N dimension is contiguous after Merlin's AOT weight-layout conversion.  Four
 * output rows share each B vector load; e32/m4 gives 32 output columns at VLEN=256.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef MERLIN_OUTLINED_TRACE
#include <malloc.h>
#include <stdio.h>
#endif

#if defined(__riscv_vector) && !defined(MERLIN_OUTLINED_FORCE_SCALAR)
#include <riscv_vector.h>
#endif

typedef struct {
  int32_t *allocated;
  int32_t *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
} merlin_memref_2d_i32;

static void scalar_gemm(const int8_t *a, const int8_t *b, int32_t *c,
                        intptr_t M, intptr_t N, intptr_t K,
                        intptr_t as0, intptr_t as1,
                        intptr_t bs0, intptr_t bs1,
                        intptr_t cs0, intptr_t cs1) {
  for (intptr_t m = 0; m < M; ++m)
    for (intptr_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (intptr_t k = 0; k < K; ++k)
        acc += (int32_t)a[m * as0 + k * as1] * (int32_t)b[k * bs0 + n * bs1];
      c[m * cs0 + n * cs1] = acc;
    }
}

#if defined(__riscv_vector) && !defined(MERLIN_OUTLINED_FORCE_SCALAR)
/* One independent (four-row x VL-column) tile.  Keeping this noinline is intentional:
 * it is the one reusable microkernel LLVM optimizes, instead of 155 expanded loop nests. */
__attribute__((noinline))
static void rvv_tile4(const int8_t *a, const int8_t *b, int32_t *c,
                      intptr_t m0, intptr_t n0, size_t vl, intptr_t K,
                      intptr_t as0, intptr_t bs0, intptr_t cs0) {
  vint32m4_t c0 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t c1 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t c2 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t c3 = __riscv_vmv_v_x_i32m4(0, vl);
  for (intptr_t k = 0; k < K; ++k) {
    vint8m1_t w8 = __riscv_vle8_v_i8m1(b + k * bs0 + n0, vl);
    vint16m2_t w16 = __riscv_vsext_vf2_i16m2(w8, vl);
    c0 = __riscv_vwmacc_vx_i32m4(c0, (int16_t)a[(m0 + 0) * as0 + k], w16, vl);
    c1 = __riscv_vwmacc_vx_i32m4(c1, (int16_t)a[(m0 + 1) * as0 + k], w16, vl);
    c2 = __riscv_vwmacc_vx_i32m4(c2, (int16_t)a[(m0 + 2) * as0 + k], w16, vl);
    c3 = __riscv_vwmacc_vx_i32m4(c3, (int16_t)a[(m0 + 3) * as0 + k], w16, vl);
  }
  __riscv_vse32_v_i32m4(c + (m0 + 0) * cs0 + n0, c0, vl);
  __riscv_vse32_v_i32m4(c + (m0 + 1) * cs0 + n0, c1, vl);
  __riscv_vse32_v_i32m4(c + (m0 + 2) * cs0 + n0, c2, vl);
  __riscv_vse32_v_i32m4(c + (m0 + 3) * cs0 + n0, c3, vl);
}

static inline void rvv_drain(void) {
  /* Saturn's vector engine is decoupled from the scalar core.  Drain once per
   * worker after its complete shard, not once per 4xVL tile (TinyLlama has
   * roughly 26k tiles per inference). */
  __asm__ volatile("fence rw, rw" ::: "memory");
}

static void rvv_gemm(const int8_t *a, const int8_t *b, int32_t *c,
                     intptr_t M, intptr_t N, intptr_t K,
                     intptr_t as0, intptr_t bs0, intptr_t cs0) {
  const intptr_t nr = (intptr_t)__riscv_vsetvlmax_e32m4();
  const intptr_t mt = M / 4;
  const intptr_t nt = (N + nr - 1) / nr;
  const intptr_t tiles = mt * nt;
#ifdef MERLIN_OUTLINED_PARALLEL
#pragma omp parallel if(tiles >= 8)
  {
#pragma omp for schedule(static)
  for (intptr_t tile = 0; tile < tiles; ++tile) {
    intptr_t mb = tile / nt;
    intptr_t nb = tile - mb * nt;
    intptr_t n0 = nb * nr;
    size_t vl = __riscv_vsetvl_e32m4((size_t)(N - n0));
    rvv_tile4(a, b, c, mb * 4, n0, vl, K, as0, bs0, cs0);
  }
    rvv_drain();
  }
#else
  for (intptr_t tile = 0; tile < tiles; ++tile) {
    intptr_t mb = tile / nt;
    intptr_t nb = tile - mb * nt;
    intptr_t n0 = nb * nr;
    size_t vl = __riscv_vsetvl_e32m4((size_t)(N - n0));
    rvv_tile4(a, b, c, mb * 4, n0, vl, K, as0, bs0, cs0);
  }
  rvv_drain();
#endif
  /* The model's weight matmuls use M=8.  Keep a correct general tail so the
   * backend remains safe for other models without spending vector registers on it. */
  for (intptr_t m = mt * 4; m < M; ++m)
    for (intptr_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (intptr_t k = 0; k < K; ++k)
        acc += (int32_t)a[m * as0 + k] * (int32_t)b[k * bs0 + n];
      c[m * cs0 + n] = acc;
    }
}
#endif

merlin_memref_2d_i32 merlin_outlined_gemm_i8_body(
    int8_t *a_alloc, int8_t *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1,
    intptr_t a_st0, intptr_t a_st1,
    int8_t *b_alloc, int8_t *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1,
    intptr_t b_st0, intptr_t b_st1,
    int32_t *c_alloc, int32_t *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1,
    intptr_t c_st0, intptr_t c_st1) {
#ifdef MERLIN_OUTLINED_TRACE
  static unsigned long calls;
  const unsigned long call = __atomic_add_fetch(&calls, 1, __ATOMIC_RELAXED);
  fprintf(stderr,
          "outlined-enter %lu A=%ldx%ld[%ld,%ld] B=%ldx%ld[%ld,%ld] "
          "C=%ldx%ld[%ld,%ld] alloc=%zu start=%ld need=%zu\n",
          call, (long)a_s0, (long)a_s1, (long)a_st0, (long)a_st1,
          (long)b_s0, (long)b_s1, (long)b_st0, (long)b_st1,
          (long)c_s0, (long)c_s1, (long)c_st0, (long)c_st1,
          malloc_usable_size(c_alloc),
          (long)((char *)(c_aligned + c_off) - (char *)c_alloc),
          (size_t)(((c_s0 - 1) * c_st0 + (c_s1 - 1) * c_st1 + 1) *
                   (intptr_t)sizeof(int32_t)));
  fflush(stderr);
#endif
  (void)a_alloc; (void)b_alloc; (void)b_s0; (void)b_s1;
  const int8_t *A = a_aligned + a_off;
  const int8_t *B = b_aligned + b_off;
  int32_t *C = c_aligned + c_off;

#if defined(__riscv_vector) && !defined(MERLIN_OUTLINED_FORCE_SCALAR)
  const int dense =
      a_s0 == c_s0 && a_s1 == b_s0 && b_s1 == c_s1 &&
      a_st0 == a_s1 && a_st1 == 1 &&
      b_st0 == b_s1 && b_st1 == 1 &&
      c_st0 == c_s1 && c_st1 == 1;
  if (dense)
    rvv_gemm(A, B, C, a_s0, c_s1, a_s1, a_st0, b_st0, c_st0);
  else
#endif
    scalar_gemm(A, B, C, a_s0, c_s1, a_s1,
                a_st0, a_st1, b_st0, b_st1, c_st0, c_st1);

  merlin_memref_2d_i32 ret;
  /* The result BORROWS the destination.  MLIR's function-boundary ownership model
   * otherwise treats a tensor result as a fresh allocation and frees this field,
   * even when c_alloc is an alloca or is already freed through the input descriptor.
   * A null allocation pointer preserves the aligned data alias while making that
   * generated ownership cleanup a safe free(NULL). */
  ret.allocated = NULL; ret.aligned = c_aligned; ret.offset = c_off;
  ret.sizes[0] = c_s0; ret.sizes[1] = c_s1;
  ret.strides[0] = c_st0; ret.strides[1] = c_st1;
#ifdef MERLIN_OUTLINED_TRACE
  fprintf(stderr, "outlined-exit %lu\n", call);
  fflush(stderr);
#endif
  return ret;
}
