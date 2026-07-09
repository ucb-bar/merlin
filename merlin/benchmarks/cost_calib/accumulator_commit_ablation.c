// Stage-F L2 ablation: accumulator_commit (accumulator_commit_policy).
//
// REPS x (DIM x DIM matmul + bias + ReLU + requant to int8).
//   VARIANT_BASELINE: mvout the FULL i32 accumulator, CPU does bias+ReLU+i8 store
//                     -> accumulator materializes; output round-trips memory at 4 bytes/elem.
//   VARIANT_FUSED   : bias staged into the accumulator (mvin3), ReLU at store, single i8 mvout
//                     -> commit-after-epilogue, no i32 round-trip.
// Compute is identical; only the commit path changes. mvout bytes: baseline 4/elem, fused 1.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#ifndef REPS
#define REPS 8
#endif

static elem_t A[DIM][DIM] row_align(1);
static elem_t B[DIM][DIM] row_align(1);
static acc_t D[DIM][DIM] row_align_acc(1);    // bias
static acc_t ACC[DIM][DIM] row_align_acc(1);  // i32 mvout (baseline)
static elem_t C[DIM][DIM] row_align(1);

static elem_t sat8(acc_t v) { return v > 127 ? 127 : v < -128 ? -128 : (elem_t)v; }

int main(void) {
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      A[i][j] = (elem_t)((i + 2 * j) % 5 - 2);
      B[i][j] = (elem_t)((3 * i + j) % 7 - 3);
      D[i][j] = (acc_t)(j % 11 - 5);
    }

  const uint32_t A_sp = 0, B_sp = DIM;
  const uint32_t acc = 1u << 31, acc_accum = (1u << 31) | (1u << 30);
  const uint32_t acc_full = (1u << 31) | (1u << 29);

  gemmini_flush(0);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_config_ld(DIM * sizeof(elem_t));                                          // A
  gemmini_extended3_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1); // B
  gemmini_extended3_config_ld(DIM * sizeof(acc_t), MVIN_SCALE_IDENTITY, false, 2);  // bias

  gemmini_extended_mvin2(B, B_sp, DIM, DIM);
  gemmini_mvin(A, A_sp);

  for (int r = 0; r < REPS; r++) {
#if defined(VARIANT_BASELINE)
    gemmini_extended_config_st(DIM * sizeof(acc_t), NO_ACTIVATION, ACC_SCALE_IDENTITY);
    gemmini_preload(B_sp, acc);                       // overwrite accumulator
    gemmini_compute_preloaded(A_sp, GARBAGE_ADDR);
    gemmini_extended_mvout(ACC, acc_full, DIM, DIM);  // materialize full i32 accumulator
    gemmini_fence();
    for (size_t i = 0; i < DIM; i++)                  // CPU epilogue: bias + ReLU + requant
      for (size_t j = 0; j < DIM; j++) {
        acc_t v = ACC[i][j] + D[i][j];
        C[i][j] = sat8(v > 0 ? v : 0);
      }
#else
    gemmini_extended_config_st(DIM * sizeof(elem_t), RELU, ACC_SCALE_IDENTITY);
    gemmini_extended_mvin3(D, acc, DIM, DIM);         // bias into accumulator
    gemmini_preload(B_sp, acc_accum);                 // accumulate on top of bias
    gemmini_compute_preloaded(A_sp, GARBAGE_ADDR);
    gemmini_extended_mvout(C, acc, DIM, DIM);         // single ReLU'd i8 commit
#endif
  }
  gemmini_fence();

  int ok = 1;
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      acc_t g = D[i][j];
      for (size_t k = 0; k < DIM; k++)
        g += (acc_t)A[i][k] * (acc_t)B[k][j];
      if (C[i][j] != sat8(g > 0 ? g : 0)) ok = 0;
    }
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
