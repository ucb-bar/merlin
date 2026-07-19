// CEILING driver: XNNPACK RVV fp16 GEMM microkernel
// (xnn_f16_gemm_minmax_ukernel_7x4v__rvvfp16arith), measured STANDALONE on the K1 with the SAME
// inner-compute / rdtime protocol as ours_gemm_f16_driver.c.
//
// NUMERICS ASYMMETRY (state this in any speed comparison): this kernel accumulates NATIVELY in
// fp16 (vfmacc_vf_f16m4 -- e16 operands AND e16 accumulator), rounding every partial sum to the
// 10-bit mantissa. Ours accumulates in f32 (vfwmacc.vf -- e16 operands, e32 accumulator) and
// rounds once. Ours is the more accurate contract; this XNNPACK kernel is exactly the kind our
// per-element fp16 gate REJECTS at 128^3. So this driver:
//   * VERIFY PASS means the kernel matches an fp16-ACCUMULATING reference (its OWN contract) -- the
//     timing is valid for what XNNPACK computes; and
//   * it ALSO prints COS / MAX_REL vs the f64-EXACT reference, which quantifies how far the fp16
//     accumulate drifts from truth (the caveat the speed ratio must carry).

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>
#include "util.h"

#include "f16-gemm/gen/f16-gemm-7x4v-minmax-rvvfp16arith.c"

#ifndef GEMM_M
#define GEMM_M 128
#endif
#ifndef GEMM_N
#define GEMM_N 128
#endif
#ifndef GEMM_K
#define GEMM_K 128
#endif
#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

typedef _Float16 f16;

static f16 A[M * K];               // activation A[m,k], row-major
static f16 W[N * K];               // weights W[n,k] ("goi")
static f16 bias[N];
static f16 C[M * N];               // output C[m,n], row-major
static double Cexact[M * N];       // f64-exact reference (accuracy caveat)
static f16 Cref16[M * N];          // fp16-ACCUMULATE reference (XNN's own contract)
static f16 Wpack[N + N * K];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  const size_t NR = __riscv_vsetvlmax_e16m4();   // f16 lanes at LMUL=4

  // ---- init operands: ~unit-normal magnitude with FINE granularity ----------------
  // Deliberately NOT coarse multiples of a power of two: with coarse operands every product and
  // partial sum stays f16-exact and the fp16-accumulate error vanishes (measured max-rel 0),
  // hiding the very asymmetry this arm exists to show. An LCG mapped to ~N(0,1) (sum of 3
  // uniforms) gives values whose products are NOT f16-representable, so the K=128 fp16 reduction
  // drifts -- the regime our f32-accumulate datapath is built for and ours' own workload uses.
  unsigned int rng = 0x1234567u;
  #define NEXTU ((rng = rng * 1103515245u + 12345u), ((double)((rng >> 9) & 0x7FFF) / 32768.0))
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      A[m * K + k] = (f16)((NEXTU + NEXTU + NEXTU - 1.5) * 2.0);
  for (int n = 0; n < N; n++) {
    bias[n] = (f16)0.0;
    for (int k = 0; k < K; k++)
      W[n * K + k] = (f16)((NEXTU + NEXTU + NEXTU - 1.5) * 2.0);
  }
  #undef NEXTU

  // ---- f64-EXACT reference (truth) and fp16-ACCUMULATE reference (XNN's contract) --
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      double accd = (double)bias[n];
      f16 acc16 = bias[n];
      for (int k = 0; k < K; k++) {
        accd += (double)A[m * K + k] * (double)W[n * K + k];
        // native fp16 accumulate with FUSED single-rounding semantics, matching the kernel's
        // vfmacc (round a*b+c once). Compute the fma in double, round once to f16.
        acc16 = (f16)((double)acc16 + (double)A[m * K + k] * (double)W[n * K + k]);
      }
      Cexact[m * N + n] = accd;
      Cref16[m * N + n] = acc16;
    }

  // ---- PRE-PACK weights (goi -> streamed panels), OUTSIDE timing -------------------
  {
    size_t off = 0;
    for (size_t n0 = 0; n0 < (size_t)N; n0 += NR) {
      size_t tile = (n0 + NR <= (size_t)N) ? NR : ((size_t)N - n0);
      for (size_t lane = 0; lane < tile; lane++) Wpack[off + lane] = bias[n0 + lane];
      off += NR;
      for (int k = 0; k < K; k++) {
        for (size_t lane = 0; lane < tile; lane++)
          Wpack[off + lane] = W[(n0 + lane) * K + k];
        off += NR;
      }
    }
  }

  for (int i = 0; i < M * N; i++) C[i] = (f16)0.0;

  struct xnn_f16_minmax_params params;
  params.scalar.min = (f16)(-1.0e4);
  params.scalar.max = (f16)( 1.0e4);
  const size_t a_stride  = K * sizeof(f16);
  const size_t cm_stride = N * sizeof(f16);
  const size_t cn_stride = NR * sizeof(f16);

  // ---- TIMED region: M/MR kernel calls (packing already done) ---------------------
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  const int MR = 7;
  for (int m = 0; m < M; m += MR) {
    size_t mr = (size_t)((M - m < MR) ? (M - m) : MR);
    xnn_f16_gemm_minmax_ukernel_7x4v__rvvfp16arith(
        mr, (size_t)N, (size_t)K * sizeof(f16),
        &A[m * K], a_stride, Wpack, &C[m * N], cm_stride, cn_stride, &params);
  }
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);
  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- VERIFY against XNN's OWN (fp16-accumulate) contract ------------------------
  int errors = 0;
  for (int i = 0; i < M * N; i++) {
    double d = (double)C[i] - (double)Cref16[i];
    if (d < 0) d = -d;
    // fp16 unit-in-last-place near the output magnitude; a few ulp of slack.
    double denom = fabs((double)Cref16[i]); if (denom < 1.0) denom = 1.0;
    if (d / denom > 5e-3) errors++;
  }

  // ---- ACCURACY vs f64-EXACT (the caveat) ----------------------------------------
  double dot = 0.0, no = 0.0, nr = 0.0, maxrel = 0.0;
  for (int i = 0; i < M * N; i++) {
    double o = (double)C[i], r = Cexact[i];
    dot += o * r; no += o * o; nr += r * r;
    double d = o - r; if (d < 0) d = -d;
    double denom = fabs(r); if (denom < 1e-3) denom = 1e-3;
    double rel = d / denom;
    if (rel > maxrel) maxrel = rel;
  }
  double cos = dot / (sqrt(no) * sqrt(nr) + 1e-12);

  double checksum = 0.0;
  for (int i = 0; i < M * N; i++) checksum += (double)C[i];

  printf("XNNPACK xnn_f16_gemm_7x4v__rvvfp16arith  M=%d N=%d K=%d  NR=%d\n", M, N, K, (int)NR);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("COS %d (x1e7)  MAX_REL %d (x1e7)  [vs f64-exact; fp16 ACCUMULATE]\n",
         (int)(cos * 1e7), (int)(maxrel * 1e7));
  printf("VERIFY %s errors=%d  [against fp16-accumulate reference, XNN's own contract]\n",
         errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
