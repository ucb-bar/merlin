// CEILING driver for OURS (Merlin RVV codegen forks), measured STANDALONE on
// spike on the SAME Saturn bare-metal harness as the OpenBLAS / XNNPACK expert
// drivers — so ours and the frameworks sit on identical footing.
//
// Our compiler emits ONE `model.o` per (fork, shape): the lowered fp32
// `linalg.fill` + `linalg.matmul` reachable through `_mlir_ciface_forward`,
// driven by the generic Merlin C runtime (`merlin_model.c` -> `merlin_run` ->
// `merlin_invoke` -> `_mlir_ciface_forward`). The per-shape arg table / ciface
// unroll come from the generated `model_gen.h` / `model_io.h` / `model_call.c`.
//
// Fair-comparison mode = inner_compute: the memref-descriptor build (cheap,
// O(n_args)=3 here) is done OUTSIDE the timed region; only `merlin_invoke`
// (i.e. the compiled fill+matmul) is wrapped in read_csr(mcycle). This mirrors
// the experts' "pack outside, time the kernel call".
//
// HONEST caveat (printed in the matrix): the experts time ONLY the GEMM
// microkernel compute; ours times `_mlir_ciface_forward`, which is the compiled
// linalg.fill (zero C) + linalg.matmul for this single-op workload — i.e. the
// GEMM plus a thin compiler-emitted wrapper, NOT a multi-op model. Same timer
// (mcycle on functional spike), same harness, inner-compute scope.

#include <stdint.h>
#include <stddef.h>
#include "util.h"            // saturn: read_csr(mcycle), printf via HTIF
#include "merlin_model.h"    // merlin_arg_t, merlin_descriptor_t, merlin_run/invoke
#include "model_gen.h"       // MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_OUT_ELEMS, ...
#include "model_io.h"        // merlin_in_*, MERLIN_INPUT_PTR  (A=in0, B=in1)

#ifndef GEMM_M
#error "shape must be injected via -DGEMM_M= -DGEMM_N= -DGEMM_K="
#endif
#define M GEMM_M
#define N GEMM_N
#define K GEMM_K

// The workload generator lays A as in0 (M*K, row-major) and B as in1 (K*N,
// row-major); output is M*N row-major. The embedded merlin_in_0 / merlin_in_1
// arrays (from model_io.h) hold exactly those values (the SAME inputs.npz the
// runner used). We recompute a scalar reference from them and verify.

static float OUT[MERLIN_OUT_ELEMS];
static float Cref[MERLIN_OUT_ELEMS];
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  const float* A = (const float*)merlin_in_0;   // M x K, row-major
  const float* B = (const float*)merlin_in_1;   // K x N, row-major

  // ---- scalar reference C[m,n] = sum_k A[m,k]*B[k,n], BEFORE timing --------
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      for (int k = 0; k < K; k++) acc += A[m * K + k] * B[k * N + n];
      Cref[m * N + n] = acc;
    }

  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) OUT[i] = 0.0f;

  // ---- build memref descriptors OUTSIDE timing (mirrors expert pre-pack) ---
  // We reproduce merlin_run's descriptor setup so the timed region is ONLY the
  // compiled compute (merlin_invoke), exactly like the experts time only the
  // kernel call. (merlin_run would build descriptors INSIDE the call; we hoist.)
  void* desc_ptrs[MERLIN_N_ARGS];
  for (int i = 0; i < MERLIN_N_ARGS; i++) {
    void* data = 0;
    switch (MERLIN_ARGS[i].kind) {
      case MERLIN_INPUT:  data = MERLIN_INPUT_PTR[i]; break;
      case MERLIN_OUTPUT: data = (void*)OUT; break;
      default: break;   // no weights in this workload
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

  // ---- TIMED region: only the compiled compute (fill + matmul) -------------
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  merlin_invoke(desc_ptrs);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- verify --------------------------------------------------------------
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
    float d = OUT[i] - Cref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    // fp32 GEMM at K up to 128: a relative-ish 2e-3 abs band on ~O(1) operands.
    if (d > 2e-3f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) checksum += OUT[i];

  printf("OURS merlin_rvv_fork  M=%d N=%d K=%d\n", M, N, K);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("C[0]=%d C[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(OUT[0] * 1000.0f), (int)(OUT[MERLIN_OUT_ELEMS - 1] * 1000.0f),
         (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
