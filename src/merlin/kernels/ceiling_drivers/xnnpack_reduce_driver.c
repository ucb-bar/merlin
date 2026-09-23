// CEILING driver: XNNPACK RVV f32 REDUCTION microkernel (rsum / rmax), measured
// STANDALONE on K1. XNNPACK's rsum/rmax reduce a flat batch to ONE scalar; a model's
// softmax/norm reduction is over the LAST DIM of an (M,N) tensor -> M scalars. The
// honest use of the expert kernel for that is one call per row, so the timed region
// is M calls -- exactly how a real reduction-over-last-dim would consume it.
//
//   -DXNN_KERNEL_SRC=\"f32-rsum/gen/f32-rsum-rvv-u8v.c\"
//   -DXNN_KERNEL_FN=xnn_f32_rsum_ukernel__rvv_u8v
//   -DXNN_REDOP_SUM  (or -DXNN_REDOP_MAX)
//   -DRED_M=64 -DRED_N=4096
//
// rsum's kernel does *output += reduced * scale, so the output row must be pre-zeroed
// and scale=1. rmax overwrites output[0] but reads it as the running max seed, so it is
// seeded with -inf. Both params structs differ, so the op selects the right one.

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>
#include "util.h"
#include "src/xnnpack/microparams.h"

#include XNN_KERNEL_SRC

#ifndef RED_M
#define RED_M 64
#endif
#ifndef RED_N
#define RED_N 4096
#endif

static float X[RED_M * RED_N];
static float Y[RED_M];
static float Yref[RED_M];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  for (long i = 0; i < (long)RED_M * RED_N; i++)
    X[i] = (float)(((i * 2654435761u) >> 8) % 6000) * 0.001f - 3.0f;

  for (long m = 0; m < RED_M; m++) {
#if defined(XNN_REDOP_SUM)
    float acc = 0.0f;
    for (long n = 0; n < RED_N; n++) acc += X[m * RED_N + n];
#elif defined(XNN_REDOP_MAX)
    float acc = X[m * RED_N + 0];
    for (long n = 1; n < RED_N; n++) { float v = X[m * RED_N + n]; if (v > acc) acc = v; }
#else
#error "define -DXNN_REDOP_SUM or -DXNN_REDOP_MAX"
#endif
    Yref[m] = acc;
  }

#if defined(XNN_REDOP_SUM)
  struct xnn_f32_scale_params params;
  params.scalar.scale = 1.0f;
  for (long m = 0; m < RED_M; m++) Y[m] = 0.0f;      // rsum does *output += ...
#else
  struct xnn_f32_default_params params;
  for (long m = 0; m < RED_M; m++) Y[m] = -3.0e38f; // rmax seeds from output[0] (-inf; ffast-math)
#endif

  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  for (long m = 0; m < RED_M; m++)
    XNN_KERNEL_FN((size_t)RED_N * sizeof(float), X + m * RED_N, Y + m, &params);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

  // A vector tree-reduce of N=4096 unit-normals drifts from the sequential reference by
  // more than an elementwise op; scale the band for sum.
#if defined(XNN_REDOP_SUM)
  const float band = 5e-2f;
#else
  const float band = 2e-3f;
#endif
  int errors = 0;
  float maxabs = 0.0f;
  for (long m = 0; m < RED_M; m++) {
    float d = Y[m] - Yref[m];
    if (d < 0) d = -d;
    if (d > maxabs) maxabs = d;
    if (d > band) errors++;
  }
  double checksum = 0.0;
  for (long m = 0; m < RED_M; m++) checksum += Y[m];

  printf("XNNPACK reduce M=%d N=%d\n", RED_M, RED_N);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("Y[0]=%d Y[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(Y[0] * 1000.0f), (int)(Y[RED_M - 1] * 1000.0f), (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
