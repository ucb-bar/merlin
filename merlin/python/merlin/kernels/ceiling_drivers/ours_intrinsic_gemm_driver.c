// CEILING driver: a COMPILER-EMITTED, register-blocked RVV GEMM micro-kernel
// (RVV intrinsics, NOT the upstream pack->outerproduct->vfmacc lowering),
// measured STANDALONE on spike on the SAME Saturn bare-metal harness + same
// inner_compute timing scope as the OpenBLAS / XNNPACK expert drivers.
//
// PURPOSE (scalable_gap task, sub-lever 2): the upstream tile->vectorize->
// bufferize lowering re-reads/re-writes the MR x NR accumulator THROUGH MEMORY
// every K-tile (a memref.copy in the K loop) instead of keeping it in vector
// registers across K — that operand-copy traffic, not the vfmacc chain, is the
// 15.7x gap. This driver demonstrates the alternative the experts use and that
// a dedicated RVV micro-kernel codegen pass would emit: a register-blocked,
// accumulator-resident, K-streaming inner kernel. The accumulator MR x (NR vreg
// group) lives in vector registers for the WHOLE K loop; only A scalars + a B
// row are loaded per K-step (vfmacc.vf), and C is stored once at the end.
//
// This is the compiler-emitted-intrinsic micro-kernel (riscv_vector.h
// intrinsics, scalable VLEN via vsetvl) — the SAME thing a tuned RVV codegen
// pass would lower the inner block to. It is shape-scalable (NR = vsetvlmax,
// MR=4 register block) and resident-weight (A/B packed OUTSIDE the timed region,
// exactly like the OpenBLAS column), so the timed cycles are pack-EXCLUDED and
// head-to-head with OpenBLAS's pack-excluded kernel-only number.

#include <stdint.h>
#include <riscv_vector.h>
#include "util.h"   // saturn: read_csr(mcycle), printf via HTIF

#ifndef GEMM_M
#define GEMM_M 64
#endif
#ifndef GEMM_N
#define GEMM_N 64
#endif
#ifndef GEMM_K
#define GEMM_K 64
#endif
#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

#define MR 4   // register block rows (A scalars broadcast); MR accumulator vreg-groups

static float A[M * K];      // logical A[m,k], row-major
static float B[K * N];      // logical B[k,n], row-major
static float Apack[M * K];  // packed: per MR-row panel, K groups of MR contiguous (col-major in panel)
static float C[M * N];      // row-major C[m,n]
static float Cref[M * N];

// One register-blocked micro-kernel call: C[mp*MR.., 0..N] = Apack(panel mp) * B.
// Accumulator (MR vreg-groups of vl lanes) is loop-carried in registers across K;
// B is read row-by-row (vle32), A scalars broadcast (vfmacc.vf). C stored once.
// nc = number of N columns in this strip (<= vl); vl = vsetvl for this strip.
static inline void microkernel_panel(const float* ap, const float* b,
                                      float* c, int ldc, int nc, int kk) {
  size_t vl = __riscv_vsetvl_e32m4((size_t)nc);
  // MR accumulators, register-resident across the whole K loop.
  vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  const float* bp = b;
  for (int k = 0; k < kk; k++) {
    vfloat32m4_t brow = __riscv_vle32_v_f32m4(bp, vl);   // B[k, 0..nc) contiguous
    const float* a = ap + (size_t)k * MR;                // MR contiguous A scalars for this k
    acc0 = __riscv_vfmacc_vf_f32m4(acc0, a[0], brow, vl);
    acc1 = __riscv_vfmacc_vf_f32m4(acc1, a[1], brow, vl);
    acc2 = __riscv_vfmacc_vf_f32m4(acc2, a[2], brow, vl);
    acc3 = __riscv_vfmacc_vf_f32m4(acc3, a[3], brow, vl);
    bp += N;                                             // next B row
  }
  __riscv_vse32_v_f32m4(c + 0 * ldc, acc0, vl);
  __riscv_vse32_v_f32m4(c + 1 * ldc, acc1, vl);
  __riscv_vse32_v_f32m4(c + 2 * ldc, acc2, vl);
  __riscv_vse32_v_f32m4(c + 3 * ldc, acc3, vl);
}

// The full GEMM, register-blocked over (M in MR-strips, N in vl-strips).
static void gemm_micro(const float* ap, const float* b, float* c) {
  for (int mp = 0; mp < M / MR; mp++) {
    const float* apanel = ap + (size_t)mp * K * MR;
    float* cpanel = c + (size_t)mp * MR * N;
    for (int n0 = 0; n0 < N;) {
      int nc = N - n0;
      size_t vl = __riscv_vsetvl_e32m4((size_t)nc);
      microkernel_panel(apanel, b + n0, cpanel + n0, N, (int)vl, K);
      n0 += (int)vl;
    }
  }
}

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A[m * K + k] = (float)(((m * 7 + k * 3) % 13) - 6) * 0.125f;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      B[k * N + n] = (float)(((k * 5 + n * 11) % 17) - 8) * 0.0625f;

  // scalar reference (row-major C), BEFORE timing
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
      Cref[m * N + n] = acc;
    }

  // PACK A into MR-row panels (col-major within panel). Default scope is
  // inner-compute: pack is OUTSIDE the timed region (resident-weight scenario,
  // exactly like OpenBLAS's hoisted ncopy). With -DPACK_INCLUDED the pack is
  // moved INSIDE the timed region (realistic end-use: pack + compute). B is used
  // row-major directly (already contiguous per K row).
  for (int i = 0; i < M * N; i++) C[i] = 0.0f;

#ifdef PACK_INCLUDED
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
#endif
  for (int mp = 0; mp < M / MR; mp++)
    for (int k = 0; k < K; k++)
      for (int mr = 0; mr < MR; mr++)
        Apack[(mp * K + k) * MR + mr] = A[(mp * MR + mr) * K + k];

  // TIMED region: only the register-blocked micro-kernel compute (pack excluded
  // unless -DPACK_INCLUDED).
#ifndef PACK_INCLUDED
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
#endif
  gemm_micro(Apack, B, C);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float d = C[i] - Cref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 2e-3f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < M * N; i++) checksum += C[i];

  printf("OURS intrinsic_microkernel  M=%d N=%d K=%d MR=%d\n", M, N, K, MR);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("C[0]=%d C[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(C[0] * 1000.0f), (int)(C[M * N - 1] * 1000.0f),
         (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
