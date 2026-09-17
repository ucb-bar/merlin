// Gemmini instruction cost-model calibration microbenchmarks (baremetal).
//
// Each KIND varies the COUNT of ONE instruction class while holding the rest fixed, so a
// linear regression over (KIND, COUNT) recovers a per-instruction cycle cost. rdcycle is
// read AROUND the Gemmini region only, so boot cost cancels and we measure the region.
// preload is always 1:1 with compute in real code, so it is folded into the compute cost
// (not a separate regressor). Run the SAME binary under Spike for the event counts.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#ifndef COUNT
#define COUNT 8
#endif
#ifndef KIND
#define KIND 0
#endif
#define K_MVIN 0
#define K_MVIN2 1
#define K_COMPUTE 2
#define K_MVOUT 3
#define K_CONFIG 4
#define K_FENCE 5
#define K_MATMUL 6

static elem_t A[DIM][DIM] row_align(1);
static elem_t B[DIM][DIM] row_align(1);
static elem_t C[DIM][DIM] row_align(1);

int main(void) {
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      A[i][j] = (elem_t)((i + j) % 4 - 1);
      B[i][j] = (elem_t)((i * 2 + j) % 5 - 2);
    }
  const uint32_t A_sp = 0, B_sp = DIM, C_sp = 2 * DIM;

  gemmini_flush(0);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_extended3_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(DIM * sizeof(elem_t));

  uint64_t t0 = read_cycles();
#if KIND == K_MVIN
  for (int i = 0; i < COUNT; i++) gemmini_mvin(A, A_sp);
  gemmini_fence();
#elif KIND == K_MVIN2
  for (int i = 0; i < COUNT; i++) gemmini_extended_mvin2(B, B_sp, DIM, DIM);
  gemmini_fence();
#elif KIND == K_COMPUTE
  gemmini_mvin(A, A_sp);
  gemmini_extended_mvin2(B, B_sp, DIM, DIM);
  for (int i = 0; i < COUNT; i++) {
    gemmini_preload(B_sp, C_sp);
    gemmini_compute_preloaded(A_sp, GARBAGE_ADDR);
  }
  gemmini_fence();
#elif KIND == K_MVOUT
  gemmini_mvin(A, A_sp);
  gemmini_extended_mvin2(B, B_sp, DIM, DIM);
  gemmini_preload(B_sp, C_sp);
  gemmini_compute_preloaded(A_sp, GARBAGE_ADDR);
  for (int i = 0; i < COUNT; i++) gemmini_mvout(C, C_sp);
  gemmini_fence();
#elif KIND == K_CONFIG
  for (int i = 0; i < COUNT; i++) gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
#elif KIND == K_FENCE
  for (int i = 0; i < COUNT; i++) { gemmini_mvin(A, A_sp); gemmini_fence(); }
#elif KIND == K_MATMUL
  gemmini_extended_mvin2(B, B_sp, DIM, DIM);
  for (int i = 0; i < COUNT; i++) {
    gemmini_mvin(A, A_sp);
    gemmini_preload(B_sp, C_sp);
    gemmini_compute_preloaded(A_sp, GARBAGE_ADDR);
    gemmini_mvout(C, C_sp);
  }
  gemmini_fence();
#endif
  uint64_t t1 = read_cycles();

  printf("KIND %d COUNT %d CYCLES %lu\n", KIND, COUNT, (unsigned long)(t1 - t0));
  return 0;
}
