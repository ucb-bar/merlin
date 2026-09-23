// BOARD (RVV) OURS GEMM shim: routes the whole-model f32 linalg.matmul to OUR compiler-emitted
// register-blocked, accumulator-resident MR=4 RVV micro-kernel (the same kernel the ceiling driver
// kernels/ceiling_drivers/ours_intrinsic_gemm_driver.c measures standalone — "v3"). Its ONLY purpose
// is to let the OURS arm self-TIME its matmul bucket through the SAME -DMERLIN_DISPATCH_TIMING rdtime
// bracket the XNNPACK/OpenBLAS shims use, so the matmul-vs-dispatch split is MEASURED on both arms
// (closes the "ours matmul assumed == xnnpack" caveat). Same descriptor ABI as the other board shims.
//
// C[m,n] = sum_k A[m,k] * B[k,n]   (A row-major MxK, B row-major KxN [the weight], C row-major MxN).
// The micro-kernel keeps MR=4 accumulator vreg-groups resident across K, reads B row-major directly
// (no B-pack), and packs A into MR-row panels OUTSIDE the timed region (pack-excluded, matching the
// expert ceiling drivers' inner-compute scope — apples to apples). The e2e cosine gate verifies.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>

// Register-block rows (A scalars broadcast per K-step; OURS_MR accumulator vreg-groups held resident).
// Overridable at compile time (-DOURS_MR=<n>) so the matmul-bucket measurement can sweep the register
// blocking (MR=1 == the wholemodel_vf small-M floor; MR=7 matches XNNPACK 7x4v). At LMUL=4 each
// accumulator is 4 vregs, so MR*4 + 4(brow) must be <= 32 vregs => MR<=7 fits the register file exactly.
#ifndef OURS_MR
#define OURS_MR 4
#endif

// --- matmul-bucket timer: verbatim mechanism + SYMBOL NAMES from the XNNPACK shim, so the K1
//     harness `extern unsigned long long merlin_matmul_ticks(void)` resolves regardless of backend.
#ifdef MERLIN_DISPATCH_TIMING
unsigned long long g_merlin_matmul_ticks = 0ULL;
unsigned long long g_merlin_matmul_calls = 0ULL;
static inline unsigned long long merlin_rd_time(void) {
  unsigned long long t; __asm__ volatile("rdtime %0" : "=r"(t)); return t;
}
unsigned long long merlin_matmul_ticks(void) { return g_merlin_matmul_ticks; }
unsigned long long merlin_matmul_calls(void) { return g_merlin_matmul_calls; }
#endif

// 2-D memref descriptor (matches MLIR's MemRefDescriptor for rank 2 / lp64d).
typedef struct {
  float *allocated;
  float *aligned;
  intptr_t offset;
  intptr_t sizes[2];
  intptr_t strides[2];
} merlin_memref_2d_f32;

static size_t round_up_mr(size_t x) { return ((x + (OURS_MR - 1)) / OURS_MR) * OURS_MR; }

// One MR-row register-blocked panel: C[0..MR rows, n0..n0+vl) over the full K. The MR accumulators are
// held resident in vector registers across K; B read row-major (vle32) ONCE per K-step and reused across
// all MR vfmacc.vf (A scalars broadcast) -> loads/useful-FMA = (1 B-load + MR A-scalars)/MR -> 1+1/MR;
// MR independent accumulator chains hide the vfmacc latency. C stored once.
// RVV scalable vector types are sizeless (cannot be C-array elements), so the MR accumulators are
// named and preprocessor-guarded. Supports OURS_MR in 1..8 (8*4 vregs > 32 at LMUL=4 -> the backend
// spills for MR=8; MR<=7 is register-resident). Each #if block adds one resident accumulator chain.
#define OURS_ACC_MAX 8
#if OURS_MR > OURS_ACC_MAX
#error "OURS_MR exceeds the named-accumulator ceiling (8)"
#endif
// `bstride` = element stride between consecutive B-rows read in the K-loop (= N for row-major B,
// = vl for the contiguous N-tile-packed B path); `cstride` = element stride between C output rows (= N).
static inline void ours_panel(const float *ap, const float *b, float *c, int cstride, int bstride,
                              int nc, int K) {
  size_t vl = __riscv_vsetvl_e32m4((size_t)nc);
  vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#if OURS_MR > 1
  vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#endif
#if OURS_MR > 2
  vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#endif
#if OURS_MR > 3
  vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#endif
#if OURS_MR > 4
  vfloat32m4_t a4 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#endif
#if OURS_MR > 5
  vfloat32m4_t a5 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#endif
#if OURS_MR > 6
  vfloat32m4_t a6 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#endif
#if OURS_MR > 7
  vfloat32m4_t a7 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
#endif
  const float *bp = b;
  for (int k = 0; k < K; k++) {
    vfloat32m4_t brow = __riscv_vle32_v_f32m4(bp, vl);
    const float *a = ap + (size_t)k * OURS_MR;
    a0 = __riscv_vfmacc_vf_f32m4(a0, a[0], brow, vl);
#if OURS_MR > 1
    a1 = __riscv_vfmacc_vf_f32m4(a1, a[1], brow, vl);
#endif
#if OURS_MR > 2
    a2 = __riscv_vfmacc_vf_f32m4(a2, a[2], brow, vl);
#endif
#if OURS_MR > 3
    a3 = __riscv_vfmacc_vf_f32m4(a3, a[3], brow, vl);
#endif
#if OURS_MR > 4
    a4 = __riscv_vfmacc_vf_f32m4(a4, a[4], brow, vl);
#endif
#if OURS_MR > 5
    a5 = __riscv_vfmacc_vf_f32m4(a5, a[5], brow, vl);
#endif
#if OURS_MR > 6
    a6 = __riscv_vfmacc_vf_f32m4(a6, a[6], brow, vl);
#endif
#if OURS_MR > 7
    a7 = __riscv_vfmacc_vf_f32m4(a7, a[7], brow, vl);
#endif
    bp += bstride;  // next B row (N for row-major, vl for packed-contiguous)
  }
  __riscv_vse32_v_f32m4(c + (size_t)0 * cstride, a0, vl);
#if OURS_MR > 1
  __riscv_vse32_v_f32m4(c + (size_t)1 * cstride, a1, vl);
#endif
#if OURS_MR > 2
  __riscv_vse32_v_f32m4(c + (size_t)2 * cstride, a2, vl);
#endif
#if OURS_MR > 3
  __riscv_vse32_v_f32m4(c + (size_t)3 * cstride, a3, vl);
#endif
#if OURS_MR > 4
  __riscv_vse32_v_f32m4(c + (size_t)4 * cstride, a4, vl);
#endif
#if OURS_MR > 5
  __riscv_vse32_v_f32m4(c + (size_t)5 * cstride, a5, vl);
#endif
#if OURS_MR > 6
  __riscv_vse32_v_f32m4(c + (size_t)6 * cstride, a6, vl);
#endif
#if OURS_MR > 7
  __riscv_vse32_v_f32m4(c + (size_t)7 * cstride, a7, vl);
#endif
}

// The MLIR-ABI entry (unpacked rank-2 descriptors; struct-return destination-passing form).
merlin_memref_2d_f32 merlin_ours_gemm_f32(
    float *a_alloc, float *a_aligned, intptr_t a_off, intptr_t a_s0, intptr_t a_s1,
    intptr_t a_st0, intptr_t a_st1,
    float *b_alloc, float *b_aligned, intptr_t b_off, intptr_t b_s0, intptr_t b_s1,
    intptr_t b_st0, intptr_t b_st1,
    float *c_alloc, float *c_aligned, intptr_t c_off, intptr_t c_s0, intptr_t c_s1,
    intptr_t c_st0, intptr_t c_st1) {
  (void)a_alloc; (void)a_st0; (void)a_st1;
  (void)b_alloc; (void)b_s0; (void)b_st0; (void)b_st1;
  (void)c_alloc; (void)c_st0; (void)c_st1;

  const float *A = a_aligned + a_off;
  const float *B = b_aligned + b_off;
  float *C = c_aligned + c_off;
  const size_t M = (size_t)a_s0;   // A is M x K
  const size_t K = (size_t)a_s1;
  const size_t N = (size_t)c_s1;   // C is M x N

  merlin_memref_2d_f32 ret;
  ret.allocated = c_alloc; ret.aligned = c_aligned; ret.offset = c_off;
  ret.sizes[0] = c_s0; ret.sizes[1] = c_s1; ret.strides[0] = c_st0; ret.strides[1] = c_st1;

  if (M == 0 || N == 0 || K == 0) {
    if (M && N) memset(C, 0, M * N * sizeof(float));
    return ret;
  }

  const size_t Mpad = round_up_mr(M);
  // A-pack into MR-row panels (per call; A is the activation), zero-padded for m >= M. Done OUTSIDE
  // the timed bracket => pack-excluded inner-compute scope, matching the expert shims/ceiling drivers.
  float *Apack = (float *)malloc(Mpad * K * sizeof(float));
  if (!Apack) { memset(C, 0, M * N * sizeof(float)); return ret; }
  for (size_t mp = 0; mp < Mpad / OURS_MR; mp++)
    for (size_t k = 0; k < K; k++)
      for (size_t mr = 0; mr < OURS_MR; mr++) {
        size_t m = mp * OURS_MR + mr;
        Apack[(mp * K + k) * OURS_MR + mr] = (m < M) ? A[m * K + k] : 0.0f;
      }
  // C scratch padded to Mpad rows (the padded rows have A=0 -> 0 output, discarded). If M already
  // a multiple of MR we write straight into C.
  float *Cout = (Mpad == M) ? C : (float *)malloc(Mpad * N * sizeof(float));
  if (!Cout) { free(Apack); memset(C, 0, M * N * sizeof(float)); return ret; }

#ifdef OURS_PACK_B
  // Optional B-PACK (the XNNPACK/OpenBLAS packing lever): repack row-major B (K x N, stride N) into
  // per-N-tile CONTIGUOUS panels [tile][k*vl + j] so the K-loop reads B with stride vl (one cache
  // line fully used per load) instead of stride N (a new line every K-step). Packed OUTSIDE the timed
  // region (B is the resident weight) -> pack-excluded, matching the expert shims' resident pack.
  float *Bpack = (float *)malloc((size_t)K * N * sizeof(float));
  if (!Bpack) { if (Cout != C) free(Cout); free(Apack); memset(C, 0, M * N * sizeof(float)); return ret; }
  {
    size_t off = 0;
    for (size_t n0 = 0; n0 < N;) {
      size_t vl = __riscv_vsetvl_e32m4(N - n0);
      for (size_t k = 0; k < K; k++)
        for (size_t j = 0; j < vl; j++)
          Bpack[off + k * vl + j] = B[k * N + n0 + j];
      off += (size_t)K * vl;
      n0 += vl;
    }
  }
#endif

#ifdef MERLIN_DISPATCH_TIMING
  const unsigned long long _mm_t0 = merlin_rd_time();
#endif
  for (size_t mp = 0; mp < Mpad / OURS_MR; mp++) {
    const float *apanel = Apack + (size_t)mp * K * OURS_MR;
    float *cpanel = Cout + (size_t)mp * OURS_MR * N;
#ifdef OURS_PACK_B
    size_t off = 0;
    for (size_t n0 = 0; n0 < N;) {
      size_t vl = __riscv_vsetvl_e32m4(N - n0);
      // packed tile has row-stride vl (contiguous); C row-stride stays N.
      ours_panel(apanel, Bpack + off, cpanel + n0, (int)N, (int)vl, (int)vl, (int)K);
      off += (size_t)K * vl;
      n0 += vl;
    }
#else
    for (size_t n0 = 0; n0 < N;) {
      size_t nc = N - n0;
      size_t vl = __riscv_vsetvl_e32m4(nc);
      ours_panel(apanel, B + n0, cpanel + n0, (int)N, (int)N, (int)vl, (int)K);
      n0 += vl;
    }
#endif
  }
#ifdef MERLIN_DISPATCH_TIMING
  g_merlin_matmul_ticks += merlin_rd_time() - _mm_t0;
  g_merlin_matmul_calls += 1ULL;
#endif

#ifdef OURS_PACK_B
  free(Bpack);
#endif
  if (Cout != C) {
    for (size_t m = 0; m < M; m++) memcpy(C + m * N, Cout + m * N, N * sizeof(float));
    free(Cout);
  }
  free(Apack);
  return ret;
}
