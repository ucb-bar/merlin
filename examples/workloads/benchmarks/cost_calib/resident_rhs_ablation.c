// Stage-F L2 ablation: resident_packed_tensor (packed_rhs_policy).
//
// One DIM x DIM weight-stationary tile matmul, repeated REPS times with the same B.
//   VARIANT_BASELINE: B is re-staged (mvin2) on EVERY iteration  -> pack-in-loop.
//   VARIANT_HOISTED : B is staged once before the loop           -> the policy's action.
//   VARIANT_ORACLE  : B starts resident, never staged            -> perfect-residency bound.
// Identical compute/mvout per variant; the ONLY difference is RHS staging traffic.
// B is staged with mvin2, A with mvin, so RHS loads are separately countable in the
// spike commit log (funct: mvin=2, mvin2=1, preload=6, compute=4/5, mvout=3).
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#ifndef REPS
#define REPS 8
#endif

static elem_t A[DIM][DIM] row_align(1);
static elem_t B[DIM][DIM] row_align(1);
static elem_t C[DIM][DIM] row_align(1);
static full_t gold[DIM][DIM];

int main(void) {
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      A[i][j] = (elem_t)((i + 2 * j) % 5 - 2);
      B[i][j] = (elem_t)((3 * i + j) % 7 - 3);
    }
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      full_t acc = 0;
      for (size_t k = 0; k < DIM; k++)
        acc += (full_t)A[i][k] * (full_t)B[k][j];
      gold[i][j] = acc;
    }

  const uint32_t A_sp = 0, B_sp = DIM, C_sp = 2 * DIM;

  gemmini_flush(0);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_config_ld(DIM * sizeof(elem_t));                                  // loader 0: A
  gemmini_extended3_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);  // loader 1: B
  gemmini_config_st(DIM * sizeof(elem_t));

#if defined(COSTMODEL_TIME)
  uint64_t _t0 = read_cycles();
#endif
#if !defined(VARIANT_BASELINE)
  gemmini_extended_mvin2(B, B_sp, DIM, DIM);              // hoisted: stage B once (oracle: counted, then ignored)
#endif

  for (int r = 0; r < REPS; r++) {
#if defined(VARIANT_BASELINE)
    gemmini_extended_mvin2(B, B_sp, DIM, DIM);            // re-stage B every iteration
#endif
    gemmini_mvin(A, A_sp);
    gemmini_preload(B_sp, C_sp);
    gemmini_compute_preloaded(A_sp, GARBAGE_ADDR);
    gemmini_mvout(C, C_sp);
  }
  gemmini_fence();
#if defined(COSTMODEL_TIME)
  uint64_t _t1 = read_cycles();
  printf("REGION_CYCLES %lu\n", (unsigned long)(_t1 - _t0));
#endif

  int ok = 1;
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      if (C[i][j] != (elem_t)gold[i][j]) ok = 0;
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
