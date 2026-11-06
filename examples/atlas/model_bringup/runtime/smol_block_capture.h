#ifndef SMOL_BLOCK_CAPTURE_H
#define SMOL_BLOCK_CAPTURE_H
#include <stddef.h>
#include <stdint.h>

/* Optional fixed-capture qualification. The live block still computes every
 * activation; these references validate the entire returned DMA image. */
typedef struct {
  size_t output_mismatches, preserved_mismatches;
  uint64_t check_cycles;
} smol_capture_result;
extern const uint8_t *const smol_capture_expected[7];
extern smol_capture_result smol_capture_results[7];
#endif
