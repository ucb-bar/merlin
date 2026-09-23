// CEILING driver: XNNPACK RVV int8 dynamic-quant GEMM microkernel
// (xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv): per-row dynamically-quantized
// int8 activations x per-channel int8 weights -> f32 output.
//   c[m,n] = bias[n] + filter_scale[n] * input_scale[m] *
//            sum_k (a_q[m,k] - 0) * (w_q[n,k])      [+ ksum*input_zero_point term]
//
// This is the int8 analog of xnnpack_gemm_driver.c. The weight PRE-PACK and the
// per-row activation quantization (input_scale/zero_point) are done OUTSIDE the
// timed region; only the kernel compute calls (one per M row, mr=1) are timed.
// NOTE on scope: per-row activation quantization (the dynamic qd8 step) is a real
// runtime cost that XNNPACK does in a separate convert pass — here it is hoisted
// OUT (same inner-compute footing as the f32 GEMM driver, which hoists its pack).
//
// Packed weight layout the kernel reads per N-tile of NR=vsetvlmax_e32m4 lanes:
//   [ ksum(i32)[NR] ] [ K * w_q(i8)[NR] ] [ filter_output_scale(f32)[NR] ] [ bias(f32)[NR] ]
// (ksum[n] = sum_k w_q[n,k], used to correct the activation zero-point.)

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "util.h"
#include "src/xnnpack/microparams.h"

// ---- the expert microkernel, verbatim --------------------------------------
#include "qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4v-minmax-rvv.c"

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

static float  Af[M * K];        // f32 activations (pre-quant)
static float  Wf[N * K];        // f32 weights (pre-quant)
static float  bias[N];
static int8_t Aq[M * K];        // per-row dynamically quantized activations
static int8_t Wq[N * K];        // per-channel quantized weights
static float  wscale[N];        // per-channel weight scale
static float  Cref[M * N];
static float  C[M * N];

// packed weight buffer: per NR-tile -> ksum[NR](i32) + K*[NR](i8) + scale[NR](f32) + bias[NR](f32)
static unsigned char Wpack[ (size_t)((N + 16) / 16) * 16 * (sizeof(int32_t) + (size_t)K + 2*sizeof(float)) + 256 ];
static struct xnn_qd8_quantization_params qp[M];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;
  const size_t NR = __riscv_vsetvlmax_e32m4();

  // ---- init f32 operands ---------------------------------------------------
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++)
      Af[m * K + k] = (float)(((m * 7 + k * 3) % 13) - 6) * 0.125f;
  for (int n = 0; n < N; n++) {
    bias[n] = (float)((n % 5) - 2) * 0.5f;
    for (int k = 0; k < K; k++)
      Wf[n * K + k] = (float)(((k * 5 + n * 11) % 17) - 8) * 0.0625f;
  }

  // ---- per-channel symmetric weight quant (qc8w) ---------------------------
  for (int n = 0; n < N; n++) {
    float amax = 1e-12f;
    for (int k = 0; k < K; k++) { float a = Wf[n*K+k]; if (a < 0) a = -a; if (a > amax) amax = a; }
    float s = amax / 127.0f;
    wscale[n] = s;
    for (int k = 0; k < K; k++) {
      int q = (int)lroundf(Wf[n*K+k] / s);
      if (q > 127) q = 127; if (q < -127) q = -127;
      Wq[n*K+k] = (int8_t)q;
    }
  }
  // ---- per-row dynamic activation quant (qd8), zero_point=0 (symmetric) -----
  for (int m = 0; m < M; m++) {
    float amax = 1e-12f;
    for (int k = 0; k < K; k++) { float a = Af[m*K+k]; if (a < 0) a = -a; if (a > amax) amax = a; }
    float s = amax / 127.0f;
    qp[m].zero_point = 0;
    qp[m].inv_scale = s;        // kernel multiplies by this (it is the scale, "inv_scale" = scale-to-f32)
    for (int k = 0; k < K; k++) {
      int q = (int)lroundf(Af[m*K+k] / s);
      if (q > 127) q = 127; if (q < -127) q = -127;
      Aq[m*K+k] = (int8_t)q;
    }
  }

  // ---- scalar reference from the QUANTIZED operands (what the kernel computes) ----
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      int32_t acc = 0;
      for (int k = 0; k < K; k++) acc += (int32_t)Aq[m*K+k] * (int32_t)Wq[n*K+k];
      Cref[m*N+n] = bias[n] + qp[m].inv_scale * wscale[n] * (float)acc;
    }

  // ---- PRE-PACK weights (OUTSIDE timing) -----------------------------------
  {
    unsigned char* p = Wpack;
    for (size_t n0 = 0; n0 < (size_t)N; n0 += NR) {
      size_t tile = (n0 + NR <= (size_t)N) ? NR : ((size_t)N - n0);
      // ksum[NR] (i32)
      int32_t* ks = (int32_t*)p;
      for (size_t lane = 0; lane < NR; lane++) {
        int32_t s = 0;
        if (lane < tile) for (int k = 0; k < K; k++) s += (int32_t)Wq[(n0+lane)*K + k];
        ks[lane] = s;
      }
      p += NR * sizeof(int32_t);
      // K * w_q[NR] (i8), panel per k
      for (int k = 0; k < K; k++) {
        for (size_t lane = 0; lane < NR; lane++)
          p[lane] = (unsigned char)(lane < tile ? Wq[(n0+lane)*K + k] : 0);
        p += NR;
      }
      // filter_output_scale[NR] (f32)
      float* fs = (float*)p;
      for (size_t lane = 0; lane < NR; lane++) fs[lane] = (lane < tile) ? wscale[n0+lane] : 0.0f;
      p += NR * sizeof(float);
      // bias[NR] (f32)
      float* bs = (float*)p;
      for (size_t lane = 0; lane < NR; lane++) bs[lane] = (lane < tile) ? bias[n0+lane] : 0.0f;
      p += NR * sizeof(float);
    }
  }

  for (int i = 0; i < M * N; i++) C[i] = 0.0f;

  struct xnn_f32_minmax_params params;
  params.scalar.min = -1e30f;
  params.scalar.max =  1e30f;
  const size_t a_stride  = K * sizeof(int8_t);
  const size_t cm_stride = N * sizeof(float);
  const size_t cn_stride = NR * sizeof(float);

  // ---- TIMED region: M kernel calls (mr=1), packing+quant already done -----
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  for (int m = 0; m < M; m++) {
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv(
        1, (size_t)N, (size_t)K * sizeof(int8_t),
        &Aq[m * K], a_stride,
        Wpack,
        &C[m * N], cm_stride, cn_stride,
        &params, &qp[m]);
  }
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- verify (vs the QUANTIZED reference; both use the same i8 operands) ---
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float d = C[i] - Cref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 1e-2f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < M * N; i++) checksum += C[i];

  printf("XNNPACK qd8_f32_qc8w_gemm_1x4v__rvv  M=%d N=%d K=%d  NR=%d\n", M, N, K, (int)NR);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("C[0]=%d C[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(C[0] * 1000.0f), (int)(C[M * N - 1] * 1000.0f), (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
