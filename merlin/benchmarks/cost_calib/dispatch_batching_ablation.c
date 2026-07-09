// Stage-F L2 ablation: command_buffer_batching (L7 runtime candidate).
//
// TILES independent DIM x DIM tile matmuls (mirrors the mined autocomp corpus shape: many
// small dispatches, median 39/kernel, 0.85 small-dispatch fraction).
//   VARIANT_BASELINE: per-tile config_ex/config_ld/config_st re-issued + fence after every
//                     tile (exactly the corpus pattern that motivated the candidate).
//   VARIANT_BATCHED : invariant config hoisted once; one fence per batch.
// The matmul work (mvin/preload/compute/mvout) is identical; the difference is pure
// command/sync overhead — config events + fences per useful command.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#ifndef TILES
#define TILES 16
#endif

static elem_t A[TILES][DIM][DIM] row_align(1);
static elem_t B[DIM][DIM] row_align(1);
static elem_t C[TILES][DIM][DIM] row_align(1);

static void config_all(void) {
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_extended3_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  gemmini_config_st(DIM * sizeof(elem_t));
}

int main(void) {
  for (size_t t = 0; t < TILES; t++)
    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++) {
        A[t][i][j] = (elem_t)((t + i + 2 * j) % 5 - 2);
        B[i][j] = (elem_t)((3 * i + j) % 7 - 3);
      }

  const uint32_t A_sp = 0, B_sp = DIM, C_sp = 2 * DIM;
  gemmini_flush(0);
#if defined(COSTMODEL_TIME)
  uint64_t _t0 = read_cycles();
#endif
#if !defined(VARIANT_BASELINE)
  config_all();                          // batched: configure once
  gemmini_extended_mvin2(B, B_sp, DIM, DIM);
#endif

  for (int t = 0; t < TILES; t++) {
#if defined(VARIANT_BASELINE)
    config_all();                        // re-config before every tiny dispatch
    gemmini_extended_mvin2(B, B_sp, DIM, DIM);
#endif
    gemmini_mvin(A[t], A_sp);
    gemmini_preload(B_sp, C_sp);
    gemmini_compute_preloaded(A_sp, GARBAGE_ADDR);
    gemmini_mvout(C[t], C_sp);
#if defined(VARIANT_BASELINE)
    gemmini_fence();                     // sync after every tile
#endif
  }
  gemmini_fence();                       // batched: one fence per batch
#if defined(COSTMODEL_TIME)
  uint64_t _t1 = read_cycles();
  printf("REGION_CYCLES %lu\n", (unsigned long)(_t1 - _t0));
#endif

  int ok = 1;
  for (size_t t = 0; t < TILES; t++)
    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++) {
        int32_t acc = 0;
        for (size_t k = 0; k < DIM; k++)
          acc += (int32_t)A[t][i][k] * (int32_t)B[k][j];
        if (C[t][i][j] != (elem_t)acc) ok = 0;
      }
  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
