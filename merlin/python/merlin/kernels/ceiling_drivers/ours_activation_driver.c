// CEILING driver for OURS (Merlin RVV codegen) on a single f32 ELEMENTWISE
// ACTIVATION (gelu / sigmoid), the elementwise analog of ours_gemm_driver.c.
//
// Our compiler emits ONE model.o per (fork, N): the lowered linalg.generic whose
// body is the activation (math.erf-GELU or exp-sigmoid), reachable through
// _mlir_ciface_forward and driven by the generic Merlin C runtime. The single
// input is merlin_in_0 (N f32, the SAME inputs.npz the workload used).
//
// Fair-comparison mode = inner_compute: the memref-descriptor build is OUTSIDE the
// timed region; only merlin_invoke (the compiled activation pass) is timed. There is
// NO pre-pack for an elementwise op (unlike GEMM) — the timed region is the whole
// activation pass, which is the honest end-use cost.
//
// -DXNN_REF_gelu or -DXNN_REF_sigmoid selects the scalar reference.

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

static float ref_gelu(float x)    { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
static float ref_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  const float* X = (const float*)merlin_in_0;

  // ---- scalar reference (BEFORE timing) ------------------------------------
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) {
#if defined(XNN_REF_gelu)
    Yref[i] = ref_gelu(X[i]);
#elif defined(XNN_REF_sigmoid)
    Yref[i] = ref_sigmoid(X[i]);
#else
#error "define -DXNN_REF_gelu or -DXNN_REF_sigmoid"
#endif
    OUT[i] = 0.0f;
  }

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

  // ---- TIMED region: the compiled activation pass --------------------------
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
    float d = OUT[i] - Yref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 2e-3f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < MERLIN_OUT_ELEMS; i++) checksum += OUT[i];

  printf("OURS merlin_rvv_activation N=%d\n", MERLIN_OUT_ELEMS);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("Y[0]=%d Y[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(OUT[0] * 1000.0f), (int)(OUT[MERLIN_OUT_ELEMS - 1] * 1000.0f),
         (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
