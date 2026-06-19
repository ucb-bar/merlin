// CEILING driver: XNNPACK RVV f32 elementwise activation microkernel
// (vgelu / vsigmoid), measured STANDALONE on spike or K1.
//
// The kernel source + entry symbol + scalar-reference selector are injected at
// compile time:
//   -DXNN_KERNEL_SRC=\"f32-vgelu/gen/f32-vgelu-rvv-rational-12-10-div-u4v.c\"
//   -DXNN_KERNEL_FN=xnn_f32_vgelu_ukernel__rvv_rational_12_10_div_u4v
//   -DXNN_REF=gelu   (or -DXNN_REF=sigmoid)
//   -DVLEN_N=16384   (element count to sweep)
//
// Fair-comparison mode = inner_compute: input init + scalar reference are OUTSIDE
// the timed region; only the microkernel call (one pass over N elements) is wrapped
// in read_csr(mcycle). Activations are bandwidth-bound, so there is no pack/setup
// to hoist — the whole timed region is the activation pass (NOTE: this differs from
// GEMM where weights are pre-packed; here pack==N/A, the kernel reads input directly).
//
// The kernel's `batch` arg is in BYTES (it does `batch >>= XNN_LOG2_SIZEOF_FLOAT`),
// so we pass VLEN_N*sizeof(float).

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>
#include "util.h"                       // read_csr(mcycle)/printf (saturn or k1 shim)
#include "src/xnnpack/microparams.h"    // xnn_f32_default_params

// ---- the expert microkernel, verbatim --------------------------------------
#include XNN_KERNEL_SRC

// Under the saturn bare-metal build (-nostdlib) newlib libm's erff/expf error
// path references __errno; provide storage so the scalar reference links. On the
// glibc K1 build this is unused (define MERLIN_BAREMETAL only for spike).
#ifdef MERLIN_BAREMETAL
static int merlin_errno_storage;
int* __errno(void) { return &merlin_errno_storage; }
#endif

#ifndef VLEN_N
#define VLEN_N 16384
#endif
#define N VLEN_N

static float X[N];
static float Y[N];
static float Yref[N];

// scalar reference (matches the workload golden in workloads.py)
static float ref_gelu(float x)    { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
static float ref_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  // ---- init + scalar reference (OUTSIDE timing) ----------------------------
  for (int i = 0; i < N; i++) {
    // same distribution shape as the ours workload (~N(0,3)); deterministic.
    float x = (float)(((i * 2654435761u) >> 8) % 6000) * 0.001f - 3.0f;
    X[i] = x;
#if defined(XNN_REF_gelu)
    Yref[i] = ref_gelu(x);
#elif defined(XNN_REF_sigmoid)
    Yref[i] = ref_sigmoid(x);
#else
#error "define -DXNN_REF_gelu or -DXNN_REF_sigmoid"
#endif
    Y[i] = 0.0f;
  }

  struct xnn_f32_default_params params;

  // ---- TIMED region: one activation pass over N elements -------------------
  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  XNN_KERNEL_FN((size_t)N * sizeof(float), X, Y, &params);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // ---- verify --------------------------------------------------------------
  // XNNPACK gelu is a rational-12-10 approximation, sigmoid an rr2-p5 poly; both
  // differ from the libm reference by a small approximation error. Use a 2e-3 abs
  // band (same scale as the GEMM driver), which the approximations comfortably meet.
  int errors = 0;
  float maxabs = 0.0f;
  for (int i = 0; i < N; i++) {
    float d = Y[i] - Yref[i];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > 2e-3f) errors++;
  }
  double checksum = 0.0;
  for (int i = 0; i < N; i++) checksum += Y[i];

  printf("XNNPACK vunary N=%d\n", N);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("Y[0]=%d Y[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(Y[0] * 1000.0f), (int)(Y[N - 1] * 1000.0f), (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
