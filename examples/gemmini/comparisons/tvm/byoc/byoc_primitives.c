/* One mathematical tensor intrinsic, with no tiling or hardware-loop policy.
 * TVM's Schedule owns all graph, tile, reduction and staging loops.
 */
#include <stdint.h>
#include "include/gemmini_params.h"
#ifndef NORM_STAT_IDS
#define NORM_STAT_IDS 2
#endif
#include "include/gemmini.h"

_Static_assert(DIM == 16 && sizeof(elem_t) == 1 && (elem_t)-1 < 0, "signed INT8 DIM16 required");
_Static_assert(sizeof(acc_t) == 4 && (acc_t)-1 < 0, "INT32 accumulator required");
_Static_assert(BANK_NUM * BANK_ROWS == 16384 && ACC_ROWS == 1024, "pinned row geometry required");
#ifndef ACC_READ_FULL_WIDTH
#error "Full accumulator readout required"
#endif
#ifdef ELEM_T_IS_FLOAT
#error "Integer Gemmini required"
#endif

/* C is loaded into the accumulator before each update. This gives the
 * intrinsic its declared C += A @ B semantics for arbitrary initial C and
 * allows legal schedule reordering, independently of retained device state.
 * It trades extra DMA/fences for a small, composable synchronous primitive.
 */
void tvm_gemmini_tile_update_i8_i32(const int8_t *a, const int8_t *b,
                                  int32_t *c, int64_t sa, int64_t sb, int64_t sc) {
  const uint32_t acc = (1u << 31) | (1u << 29);
  asm volatile("fence rw,rw" ::: "memory");
  gemmini_flush(0);
  gemmini_config_ex(OUTPUT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_config_st(sc * sizeof(acc_t));
  gemmini_extended3_config_ld(sa, MVIN_SCALE_IDENTITY, false, 0);
  gemmini_extended3_config_ld(sb, MVIN_SCALE_IDENTITY, false, 1);
  gemmini_extended3_config_ld(sc * sizeof(acc_t), MVIN_SCALE_IDENTITY, false, 2);
  gemmini_extended_mvin(a, 0, 16, 16);
  gemmini_extended_mvin2(b, 16368, 8, 16);
  gemmini_extended_mvin3(c, (1u << 31), 8, 16);
  gemmini_extended_preload(GARBAGE_ADDR, acc | (1u << 30), 16, 16, 8, 16);
  gemmini_extended_compute_preloaded(0, 16368, 16, 16, 8, 16);
  gemmini_extended_mvout(c, acc | (1u << 30), 8, 16);
  gemmini_fence();
  asm volatile("fence rw,rw" ::: "memory");
}
