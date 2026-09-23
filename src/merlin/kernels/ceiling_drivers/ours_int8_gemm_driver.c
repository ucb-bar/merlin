// CEILING driver for OURS int8 W8A8 GEMM (the merlin vwmacc i8xi8->i32 + dynamic
// act-quant + requant datapath). Like ours_gemm_driver.c but the verify is the
// repo's int8 tier (W8A8 is an APPROXIMATION of the f32 product, so a strict abs
// band is wrong): we accept on cosine > 0.99 AND relative-L2 < 5e-2 vs the f32
// reference — the same fp32-tier gate (cos>0.99) the zephyr_model int8 verifier and
// the spike int8 sweep use. (XNNPACK's qd8 driver verifies vs its OWN quantized
// reference and so is bit-exact; the two are NOT the same correctness bar — the
// doc records both honestly.)
//
// inner-compute, descriptors hoisted, fill-only subtracted (identical to the f32
// ours_gemm_driver timing scope). Shape via -DGEMM_M= -DGEMM_N= -DGEMM_K=.

#include <stdint.h>
#include <stddef.h>
#include "util.h"
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

#ifndef GEMM_M
#error "shape must be injected via -DGEMM_M= -DGEMM_N= -DGEMM_K="
#endif
#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

static float OUT[MERLIN_OUT_ELEMS];
static float Cref[MERLIN_OUT_ELEMS];
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

static double dsqrt(double x){ double g=x>1?x:1; for(int i=0;i<60;i++) g=0.5*(g+x/g); return g; }

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;
  const float* A = (const float*)merlin_in_0;   // M x K
  const float* B = (const float*)merlin_in_1;   // K x N

  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
      Cref[m * N + n] = acc;
    }
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) OUT[i] = 0.0f;

  void* desc_ptrs[MERLIN_N_ARGS];
  for (int i = 0; i < MERLIN_N_ARGS; i++) {
    void* data = 0;
    switch (MERLIN_ARGS[i].kind) {
      case MERLIN_INPUT:  data = MERLIN_INPUT_PTR[i]; break;
      case MERLIN_OUTPUT: data = (void*)OUT; break;
      default: break;
    }
    merlin_descriptor_t* d = &DESCS[i];
    d->allocated = data; d->aligned = data; d->offset = 0;
    long stride = 1;
    for (int r = MERLIN_ARGS[i].rank - 1; r >= 0; r--) {
      d->sizes[r] = MERLIN_ARGS[i].dims[r];
      d->strides[r] = stride;
      stride *= MERLIN_ARGS[i].dims[r];
    }
    desc_ptrs[i] = d;
  }

  static volatile float fill_sink;
  unsigned long f0 = read_csr(mcycle);
  unsigned long fi0 = read_csr(minstret);
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) OUT[i] = 0.0f;
  unsigned long fi1 = read_csr(minstret);
  unsigned long f1 = read_csr(mcycle);
  fill_sink = OUT[0];
  unsigned long fill_cycles = f1 - f0;
  unsigned long fill_instrs = fi1 - fi0;

  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  merlin_invoke(desc_ptrs);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);
  unsigned long cycles_full = c1 - c0;
  unsigned long cycles = (cycles_full > fill_cycles) ? (cycles_full - fill_cycles) : 0;
  // Retired instructions on the SAME bracket as the timing. Without this the int8 arm is blind to
  // the one distinction that cracked the f32 gap -- emitting too many instructions vs stalling on
  // each -- and an int8 beam can only rank on wall time.
  unsigned long instrs_full = i1 - i0;
  unsigned long instrs = (instrs_full > fill_instrs) ? (instrs_full - fill_instrs) : 0;

  // ---- int8 verify: cosine + relative-L2 vs the f32 reference --------------
  double dot = 0, na = 0, nb = 0, diff2 = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
    double o = OUT[i], r = Cref[i];
    dot += o * r; na += o * o; nb += r * r; diff2 += (o - r) * (o - r);
    float d = OUT[i] - Cref[i]; if (d < 0) d = -d; if (d > maxabs) maxabs = d;
  }
  double cos = dot / (dsqrt(na) * dsqrt(nb) + 1e-12);
  double rel = dsqrt(diff2) / (dsqrt(nb) + 1e-12);
  int ok = (cos > 0.99) && (rel < 5e-2);

  double checksum = 0.0;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) checksum += OUT[i];

  printf("OURS merlin_rvv_int8_w8a8  M=%d N=%d K=%d\n", M, N, K);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("COS %d (x1e6)  REL %d (x1e6)  maxabs_err=%d (x1e6)\n",
         (int)(cos * 1e6), (int)(rel * 1e6), (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", ok ? "PASS" : "FAIL", ok ? 0 : 1);
  printf("CYCLES %lu\n", cycles);
  printf("CYCLES_FULL %lu\n", cycles_full);
  printf("FILL_CYCLES %lu\n", fill_cycles);
  printf("INSTRET %lu\n", instrs);
  printf("INSTRET_FULL %lu\n", instrs_full);
  return 0;
}
