// Small self-checking carrier for a compiler-emitted `gemmini_kernel` object.
//
// The tensor values match the public G0 single-tile representative.  The CPU
// computes its golden independently, after the compiler-emitted kernel has
// moved its result back to T_Y0.  The marker calls are ordinary control flow:
// Verilator continues to the stock tohost exit, while the GSIM harness can stop
// at their committed PCs without requiring a coherent host link.
#include <stdint.h>

#include "include/gemmini_testutils.h"

extern void gemmini_kernel(void*, void*, void*);

static elem_t T_W[DIM * DIM] row_align(1);
static elem_t T_A0[DIM * DIM] row_align(1);
static acc_t T_Y0[DIM * DIM] row_align_acc(1);

__attribute__((noinline, used)) int merlin_gsim_pass_marker(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
  return 0;
}

__attribute__((noinline, used)) int merlin_gsim_fail_marker(int errors) {
  __asm__ volatile("fence rw, rw" ::: "memory");
  return errors ? errors : 1;
}

int main(void) {
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM; ++j) {
      T_W[i * DIM + j] = (j & 2) ? 1 : 3;
      T_A0[i * DIM + j] = (j == 0 || j == 3) ? 1 : 3;
      T_Y0[i * DIM + j] = 0;
    }
  }

  gemmini_kernel(T_W, T_A0, T_Y0);
  gemmini_fence();

  int errors = 0;
  for (int i = 0; i < DIM; ++i) {
    for (int j = 0; j < DIM; ++j) {
      int32_t golden = 0;
      for (int k = 0; k < DIM; ++k)
        golden += (int32_t)T_A0[i * DIM + k] * (int32_t)T_W[k * DIM + j];
      errors += T_Y0[i * DIM + j] != golden;
    }
  }
  return errors ? merlin_gsim_fail_marker(errors) : merlin_gsim_pass_marker();
}
