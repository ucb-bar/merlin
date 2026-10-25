/* Bare-metal probe for Spike's accumulator-scale rounding: mvin acc value 664, mvout through a
 * 1/166 scale. 664 * f32(1/166) is just below 4 (3.99999...), so true round-to-nearest-even gives 4;
 * a model that rounds the "up" candidate in float gives 5. The exit code is the rounded result. */
#include <stdint.h>
#include "include/gemmini.h"

static acc_t in_[DIM][DIM] __attribute__((aligned(64)));
static elem_t out_[DIM][DIM] __attribute__((aligned(64)));

int main(void) {
  const uint32_t acc_addr = 1u << (ADDR_LEN - 1);
  in_[0][0] = 664;
  gemmini_config_ld(DIM * sizeof(acc_t));
  gemmini_config_ex(0, 0, 0);
  gemmini_extended_config_st(DIM * sizeof(elem_t), NO_ACTIVATION, 1.0f / 166.0f);
  gemmini_mvin(in_, acc_addr);
  gemmini_mvout(out_, acc_addr);
  gemmini_fence();
  return (int)out_[0][0];
}
