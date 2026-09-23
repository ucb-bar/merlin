// CEILING driver for OURS on a single f32 BATCH-MATMUL (the attention shape):
//   (B,M,K) x (B,K,N) -> (B,M,N),  out[b,m,n] = sum_k A[b,m,k]*B[b,k,n].
//
// ATTENTION has NO library baseline (it is not an XNNPACK/OpenBLAS primitive), so this
// driver only ever measures OURS — the ours-vs-ours comparison (baseline bmm lowering vs
// the vfmacc feature). It mirrors ours_gemm_driver.c (inner-compute, descriptors hoisted,
// fill-only subtracted) but with a CORRECT batched scalar reference (the 2-D gemm driver's
// flat reference would be wrong for a block-diagonal bmm).
//
// Shapes injected via -DBMM_B= -DBMM_M= -DBMM_N= -DBMM_K=.

#include <stdint.h>
#include <stddef.h>
#include "util.h"
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

#ifndef BMM_B
#error "shape must be injected via -DBMM_B= -DBMM_M= -DBMM_N= -DBMM_K="
#endif
#define B BMM_B
#define M BMM_M
#define N BMM_N
#define K BMM_K

static float OUT[MERLIN_OUT_ELEMS];
static float Cref[MERLIN_OUT_ELEMS];
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  const float* A = (const float*)merlin_in_0;   // B x M x K
  const float* Bm = (const float*)merlin_in_1;  // B x K x N

  // ---- batched scalar reference (BEFORE timing) ----------------------------
  for (int b = 0; b < B; b++)
    for (int m = 0; m < M; m++)
      for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
          acc += A[(b * M + m) * K + k] * Bm[(b * K + k) * N + n];
        Cref[(b * M + m) * N + n] = acc;
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

  // ---- fill-only baseline (subtracted, as in ours_gemm_driver) -------------
  static volatile float fill_sink;
  unsigned long f0 = read_csr(mcycle);
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) OUT[i] = 0.0f;
  unsigned long f1 = read_csr(mcycle);
  fill_sink = OUT[0];
  unsigned long fill_cycles = f1 - f0;

  // ---- TIMED region: compiled bmm (fill + contraction) ---------------------
  unsigned long c0 = read_csr(mcycle);
  merlin_invoke(desc_ptrs);
  unsigned long c1 = read_csr(mcycle);
  unsigned long cycles_full = c1 - c0;
  unsigned long cycles = (cycles_full > fill_cycles) ? (cycles_full - fill_cycles) : 0;

  // ---- verify --------------------------------------------------------------
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
    float d = OUT[i] - Cref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 2e-3f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) checksum += OUT[i];

  printf("OURS merlin_rvv_bmm  B=%d M=%d N=%d K=%d\n", B, M, N, K);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("C[0]=%d C[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(OUT[0] * 1000.0f), (int)(OUT[MERLIN_OUT_ELEMS - 1] * 1000.0f),
         (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("CYCLES_FULL %lu\n", cycles_full);
  printf("FILL_CYCLES %lu\n", fill_cycles);
  return 0;
}
