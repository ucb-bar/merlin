// CEILING driver: XNNPACK RVV f32 elementwise BINARY microkernel (vmul / vadd),
// measured STANDALONE on K1. The two-input analog of xnnpack_vunary_driver.c.
//
//   -DXNN_KERNEL_SRC=\"f32-vbinary/gen/f32-vmul-rvv-u8v.c\"
//   -DXNN_KERNEL_FN=xnn_f32_vmul_ukernel__rvv_u8v
//   -DXNN_BINOP_MUL  (or -DXNN_BINOP_ADD)   -- selects the scalar reference
//   -DVLEN_N=65536
//
// inner_compute: input init + scalar reference OUTSIDE the timed region; only the one
// pass over N elements (a[i] OP b[i]) is bracketed by read_csr. batch arg is in BYTES.

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "util.h"
#include "src/xnnpack/microparams.h"

#include XNN_KERNEL_SRC

#ifndef VLEN_N
#define VLEN_N 65536
#endif
#define N VLEN_N

static float A[N];
static float B[N];
static float Y[N];
static float Yref[N];

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  for (int i = 0; i < N; i++) {
    float a = (float)(((i * 2654435761u) >> 8) % 6000) * 0.001f - 3.0f;
    float b = (float)(((i * 40503u) >> 4) % 6000) * 0.001f - 3.0f + 2.0f;
    A[i] = a; B[i] = b;
#if defined(XNN_BINOP_MUL)
    Yref[i] = a * b;
#elif defined(XNN_BINOP_ADD)
    Yref[i] = a + b;
#else
#error "define -DXNN_BINOP_MUL or -DXNN_BINOP_ADD"
#endif
    Y[i] = 0.0f;
  }

  struct xnn_f32_default_params params;

  unsigned long c0 = read_csr(mcycle);
  unsigned long i0 = read_csr(minstret);
  XNN_KERNEL_FN((size_t)N * sizeof(float), A, B, Y, &params);
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

  printf("XNNPACK vbinary N=%d\n", N);
  printf("CHECKSUM %d (x1000)\n", (int)(checksum * 1000.0));
  printf("Y[0]=%d Y[last]=%d (x1000)  maxabs_err=%d (x1e6)\n",
         (int)(Y[0] * 1000.0f), (int)(Y[N - 1] * 1000.0f), (int)(maxabs * 1e6f));
  printf("VERIFY %s errors=%d\n", errors == 0 ? "PASS" : "FAIL", errors);
  printf("CYCLES %lu\n", cycles);
  printf("INSTRET %lu\n", instrs);
  return 0;
}
