// CEILING driver: XNNPACK RVV f32 VCLAMP microkernel (relu / relu6), STANDALONE on K1.
// Same shape as the vunary driver but VCLAMP takes xnn_f32_minmax_params (not default),
// so it gets its own driver rather than complicating the transcendental one.
//
//   -DXNN_KERNEL_SRC=\"f32-vclamp/gen/f32-vclamp-rvv-u8v.c\"
//   -DXNN_KERNEL_FN=xnn_f32_vclamp_ukernel__rvv_u8v
//   -DCLAMP_LO=0.0f -DCLAMP_HI=6.0f -DVLEN_N=65536

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "util.h"
#include "src/xnnpack/microparams.h"

#include XNN_KERNEL_SRC

#ifndef VLEN_N
#define VLEN_N 65536
#endif
#ifndef CLAMP_LO
#define CLAMP_LO 0.0f
#endif
#ifndef CLAMP_HI
#define CLAMP_HI 6.0f
#endif
#define N VLEN_N

static float X[N];
static float Y[N];
static float Yref[N];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;
  const float lo = CLAMP_LO, hi = CLAMP_HI;

  for (int i = 0; i < N; i++) {
    float x = (float)(((i * 2654435761u) >> 8) % 6000) * 0.001f - 3.0f;
    X[i] = x;
    float v = x < lo ? lo : x;
    v = v > hi ? hi : v;
    Yref[i] = v;
    Y[i] = 0.0f;
  }

  struct xnn_f32_minmax_params params;
  params.scalar.min = lo;
  params.scalar.max = hi;

  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  XNN_KERNEL_FN((size_t)N * sizeof(float), X, Y, &params);
  unsigned long i1 = read_csr(minstret);
  unsigned long c1 = read_csr(mcycle);

  unsigned long cycles = c1 - c0;
  unsigned long instrs = i1 - i0;

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

  printf("XNNPACK vclamp N=%d\n", N);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("Y[0]=%d Y[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(Y[0] * 1000.0f), (int)(Y[N - 1] * 1000.0f), (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
