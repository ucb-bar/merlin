// CEILING driver: OpenBLAS RVV sgemm inner kernel (sgemm_kernel_8x8_zvl128b.c),
// measured STANDALONE on spike. Fair-comparison mode = inner_compute: the A/B
// packing (ncopy/tcopy) is done OUTSIDE the timed region; only the kernel's
// compute call is wrapped in read_csr(mcycle) ... read_csr(mcycle).
//
// The OpenBLAS kernel computes, with A and B PRE-PACKED:
//   C[m,n] (col-major, ld=ldc) += alpha * sum_k A[m,k] * B[k,n]
// A packed by gemm_ncopy_8 layout: per 8-row M-panel, K groups of 8 contiguous
//   floats = A_pack[panel*K*8 + k*8 + mr] = A[panel*8+mr, k]   (col-major in panel)
// B packed by gemm_tcopy_8 layout: per 8-col N-panel, K groups of 8 contiguous
//   floats = B_pack[panel*K*8 + k*8 + nr] = B[k, panel*8+nr]   (row-major in panel)
//
// We provide the macro shims the kernel's `#include "common.h"` would supply
// (BLASLONG, FLOAT, CNAME), then #include the kernel body verbatim.

#include <stdint.h>
#include "util.h"   // saturn: read_csr(mcycle), printf via HTIF

// CNAME names the kernel entry; BLASLONG/FLOAT + riscv_vector.h come from the
// common.h shim (ceiling_drivers/common.h) that the kernel body #includes.
#define CNAME openblas_sgemm_kernel

// ---- the expert kernel, verbatim (pulls in common.h shim) -----------------
#include "sgemm_kernel_8x8_zvl128b.c"

// ---------------------------------------------------------------------------
#define M 64
#define N 64
#define K 64

static FLOAT A[M * K];        // logical A[m,k], row-major
static FLOAT B[K * N];        // logical B[k,n], row-major
static FLOAT Apack[M * K];    // ncopy-packed A
static FLOAT Bpack[K * N];    // tcopy-packed B
static FLOAT C[N * M];        // col-major: C[n*M + m]
static FLOAT Cref[N * M];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  // ---- init logical operands (deterministic, non-degenerate) -------------
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A[m * K + k] = (FLOAT)(((m * 7 + k * 3) % 13) - 6) * 0.125f;
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      B[k * N + n] = (FLOAT)(((k * 5 + n * 11) % 17) - 8) * 0.0625f;

  // ---- scalar reference (col-major C), computed BEFORE timing ------------
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++) {
      FLOAT acc = 0.0f;
      for (int k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
      Cref[n * M + m] = acc;
    }

  // ---- PACK A (ncopy_8) and B (tcopy_8) -- OUTSIDE the timed region -------
  const int MR = 8, NR = 8;
  for (int mp = 0; mp < M / MR; mp++)
    for (int k = 0; k < K; k++)
      for (int mr = 0; mr < MR; mr++)
        Apack[(mp * K + k) * MR + mr] = A[(mp * MR + mr) * K + k];
  for (int np = 0; np < N / NR; np++)
    for (int k = 0; k < K; k++)
      for (int nr = 0; nr < NR; nr++)
        Bpack[(np * K + k) * NR + nr] = B[k * N + (np * NR + nr)];

  // ---- zero C (the kernel does C += alpha*AB) ----------------------------
  for (int i = 0; i < N * M; i++) C[i] = 0.0f;

  // ---- TIMED region: only the kernel compute call ------------------------
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  openblas_sgemm_kernel(M, N, K, 1.0f, Apack, Bpack, C, M);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- verify against scalar reference -----------------------------------
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < N * M; i++) {
    float d = C[i] - Cref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 1e-3f) errors++;
  }
  // checksum so the optimizer can't elide the GEMM
  double checksum = 0.0;
  for (int i = 0; i < N * M; i++) checksum += C[i];

  printf("OPENBLAS sgemm_kernel_8x8_zvl128b  M=%d N=%d K=%d\n", M, N, K);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("C[0]=%d C[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(C[0] * 1000.0f), (int)(C[N * M - 1] * 1000.0f),
         (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
