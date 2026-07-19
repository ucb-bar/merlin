// CEILING driver for OURS (Merlin RVV codegen) on the BROADENED op families beyond
// gemm/activation: elementwise BINARY (mul/add), REDUCTION (rsum/rmax over the last
// dim), CLAMP/relu, and 2-D TRANSPOSE. One driver, one measurement protocol; the op
// (and thus the scalar reference) is selected at compile time:
//
//   -DOURS_REF_binary   -DBINOP_MUL | -DBINOP_ADD      (out[i] = a[i] OP b[i])
//   -DOURS_REF_reduce   -DREDOP_SUM | -DREDOP_MAX      (out[m] = reduce_n x[m,n])
//   -DOURS_REF_clamp    (+ -DCLAMP_LO=.. -DCLAMP_HI=..) (out[i] = clamp(x[i],lo,hi))
//   -DOURS_REF_transpose                                (out[c,r] = x[r,c])
//
// Fair-comparison mode = inner_compute: the memref-descriptor build + the scalar
// reference are OUTSIDE the timed region; only merlin_invoke (the compiled kernel
// pass) is bracketed by read_csr(mcycle)/read_csr(minstret). This mirrors the
// ours_activation / ours_gemm drivers so all OURS rows share one timing protocol.
//
// Shapes are read from the generated MERLIN_ARGS table (rank + dims), NOT hard-coded,
// so the same binary verifies whatever shape the workload bundle was lowered at.

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include "util.h"
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

static float OUT[MERLIN_OUT_ELEMS];
static float Yref[MERLIN_OUT_ELEMS];
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

static long arg_elems(int i) {
  long n = 1;
  for (int r = 0; r < MERLIN_ARGS[i].rank; r++) n *= MERLIN_ARGS[i].dims[r];
  return n;
}

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  // ---- scalar reference (BEFORE timing) ------------------------------------
#if defined(OURS_REF_binary)
  {
    const float* A = (const float*)merlin_in_0;
    const float* B = (const float*)merlin_in_1;
    for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
  #if defined(BINOP_MUL)
      Yref[i] = A[i] * B[i];
  #elif defined(BINOP_ADD)
      Yref[i] = A[i] + B[i];
  #else
    #error "binary: define -DBINOP_MUL or -DBINOP_ADD"
  #endif
    }
  }
#elif defined(OURS_REF_reduce)
  {
    const float* X = (const float*)merlin_in_0;
    const long M = MERLIN_ARGS[0].dims[0];
    const long Ncol = MERLIN_ARGS[0].dims[1];
    for (long m = 0; m < M; m++) {
  #if defined(REDOP_SUM)
      float acc = 0.0f;
      for (long n = 0; n < Ncol; n++) acc += X[m * Ncol + n];
  #elif defined(REDOP_MAX)
      float acc = X[m * Ncol + 0];
      for (long n = 1; n < Ncol; n++) { float v = X[m * Ncol + n]; if (v > acc) acc = v; }
  #else
    #error "reduce: define -DREDOP_SUM or -DREDOP_MAX"
  #endif
      Yref[m] = acc;
    }
  }
#elif defined(OURS_REF_clamp)
  {
    const float* X = (const float*)merlin_in_0;
    const float lo = CLAMP_LO, hi = CLAMP_HI;
    for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
      float v = X[i];
      v = v < lo ? lo : v;
      v = v > hi ? hi : v;
      Yref[i] = v;
    }
  }
#elif defined(OURS_REF_transpose)
  {
    const float* X = (const float*)merlin_in_0;
    const long R = MERLIN_ARGS[0].dims[0];
    const long C = MERLIN_ARGS[0].dims[1];
    for (long r = 0; r < R; r++)
      for (long c = 0; c < C; c++)
        Yref[c * R + r] = X[r * C + c];   // out is (C,R)
  }
#else
  #error "define one of -DOURS_REF_binary / _reduce / _clamp / _transpose"
#endif
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) OUT[i] = 0.0f;

  // ---- build memref descriptors OUTSIDE timing -----------------------------
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

  // ---- TIMED region: the compiled kernel pass ------------------------------
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  merlin_invoke(desc_ptrs);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- verify --------------------------------------------------------------
  // Reductions accumulate in a different order than the scalar reference (the RVV
  // tree-reduce vs sequential), so a sum over N=4096 unit-normals carries a larger
  // absolute drift than an elementwise op. Scale the band with the op.
#if defined(OURS_REF_reduce) && defined(REDOP_SUM)
  const float band = 5e-2f;
#else
  const float band = 2e-3f;
#endif
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
    float d = OUT[i] - Yref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > band) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) checksum += OUT[i];

  printf("OURS merlin_rvv_kernel N=%d\n", MERLIN_OUT_ELEMS);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("Y[0]=%d Y[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(OUT[0] * 1000.0f), (int)(OUT[MERLIN_OUT_ELEMS - 1] * 1000.0f),
         (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
