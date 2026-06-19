// CEILING driver: XNNPACK RVV f32 GEMM microkernel
// (xnn_f32_gemm_ukernel_1x4v__rvv), measured STANDALONE on spike.
//
// Fair-comparison mode = inner_compute: the weight PRE-PACK (xnn_pack_f32_gemm_
// goi_w-equivalent) and bias setup are done OUTSIDE the timed region; only the
// microkernel compute calls (one per M row) are wrapped in read_csr(mcycle).
//
// The 1x4v kernel computes, per call, ONE activation row x all N columns:
//   c[n] = bias[n] + sum_k a[k] * W[n,k]      (mr=1)
// reading PRE-PACKED weights `w` linearly: per N-tile of NR=vsetvlmax_e32m4
//   lanes, [bias[NR]] then kc/4 weight panels of [NR] each (panel k = W[tile*NR
//   + n, k] for n in tile). We reproduce that packing here. To cover M=64 we
//   call the kernel 64 times (rows share the same packed W). A is row-major
//   with a_stride bytes/row; C is row-major with cm_stride bytes/row and
//   cn_stride bytes per N-tile.

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "util.h"   // saturn: read_csr(mcycle), printf via HTIF

// ---- the expert microkernel, verbatim (pulls in src/xnnpack/gemm.h shim) ---
#include "f32-gemm/gen/f32-gemm-1x4v-rvv.c"

// ---------------------------------------------------------------------------
#define M 64
#define N 64
#define K 64

static float A[M * K];          // activation A[m,k], row-major
static float W[N * K];          // weights W[n,k] ("goi": output-channel outer)
static float bias[N];
static float C[M * N];          // output C[m,n], row-major
static float Cref[M * N];
static float Wpack[N + N * K];  // packed: per N-tile bias[NR] + K*[NR] panels

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  const size_t NR = __riscv_vsetvlmax_e32m4();  // f32 lanes at LMUL=4

  // ---- init operands -----------------------------------------------------
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A[m * K + k] = (float)(((m * 7 + k * 3) % 13) - 6) * 0.125f;
  for (int n = 0; n < N; n++) {
    bias[n] = (float)((n % 5) - 2) * 0.5f;
    for (int k = 0; k < K; k++)
      W[n * K + k] = (float)(((k * 5 + n * 11) % 17) - 8) * 0.0625f;
  }

  // ---- scalar reference: C[m,n] = bias[n] + sum_k A[m,k]*W[n,k] ----------
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float acc = bias[n];
      for (int k = 0; k < K; k++) acc += A[m * K + k] * W[n * K + k];
      Cref[m * N + n] = acc;
    }

  // ---- PRE-PACK weights (goi -> streamed panels) -- OUTSIDE timing -------
  // For each N-tile of NR lanes: bias[NR], then for each k a panel of NR
  // weights W[tile*NR + lane, k]. (N % NR == 0 here: N=64, NR=16.)
  {
    size_t off = 0;
    for (size_t n0 = 0; n0 < (size_t)N; n0 += NR) {
      size_t tile = (n0 + NR <= (size_t)N) ? NR : ((size_t)N - n0);
      for (size_t lane = 0; lane < tile; lane++) Wpack[off + lane] = bias[n0 + lane];
      off += NR;  // kernel advances w by full nr even on a short tail tile
      for (int k = 0; k < K; k++) {
        for (size_t lane = 0; lane < tile; lane++)
          Wpack[off + lane] = W[(n0 + lane) * K + k];
        off += NR;
      }
    }
  }

  for (int i = 0; i < M * N; i++) C[i] = 0.0f;

  struct xnn_f32_default_params params;
  const size_t a_stride  = K * sizeof(float);
  const size_t cm_stride = N * sizeof(float);
  const size_t cn_stride = NR * sizeof(float);

  // ---- TIMED region: M kernel calls (mr=1 each), packing already done ----
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  for (int m = 0; m < M; m++) {
    xnn_f32_gemm_ukernel_1x4v__rvv(
        1, (size_t)N, (size_t)K * sizeof(float),
        &A[m * K], a_stride,
        Wpack,
        &C[m * N], cm_stride, cn_stride,
        &params);
  }
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- verify ------------------------------------------------------------
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float d = C[i] - Cref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 1e-3f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < M * N; i++) checksum += C[i];

  printf("XNNPACK xnn_f32_gemm_ukernel_1x4v__rvv  M=%d N=%d K=%d  NR=%d\n",
         M, N, K, (int)NR);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("C[0]=%d C[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(C[0] * 1000.0f), (int)(C[M * N - 1] * 1000.0f),
         (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
