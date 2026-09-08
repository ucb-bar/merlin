#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

extern void gemmini_kernel(void *, void *, void *);

static int8_t input[3 * 4 * 4] row_align(1);
static int8_t weight[3 * 32] row_align(1);
static int32_t output[4 * 4 * 32] row_align_acc(1);

int main(void) {
  for (int c = 0; c < 3; ++c)
    for (int h = 0; h < 4; ++h)
      for (int w = 0; w < 4; ++w)
        input[(c * 4 + h) * 4 + w] = (int8_t)(((c * 11 + h * 5 + w * 3) % 15) - 7);
  for (int c = 0; c < 3; ++c)
    for (int o = 0; o < 32; ++o)
      weight[c * 32 + o] = (int8_t)(((c * 7 + o * 3) % 13) - 6);

  gemmini_kernel(input, weight, output);
  gemmini_fence();
  int bad = 0;
  for (int h = 0; h < 4; ++h) {
    for (int w = 0; w < 4; ++w) {
      for (int o = 0; o < 32; ++o) {
        int32_t expected = 0;
        for (int c = 0; c < 3; ++c)
          expected += (int32_t)input[(c * 4 + h) * 4 + w] * weight[c * 32 + o];
        int32_t got = output[(h * 4 + w) * 32 + o];
        if (got != expected && bad < 8)
          printf("MISMATCH h=%d w=%d o=%d got=%d expected=%d\n",
                 h, w, o, (int)got, (int)expected);
        bad += got != expected;
      }
    }
  }
  printf("RESULT descriptors=2 checked=512 bad=%d first_tile=%d second_tile=%d\n",
         bad, (int)output[0], (int)output[16]);
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}
