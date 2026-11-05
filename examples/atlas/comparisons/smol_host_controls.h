#ifndef SMOL_HOST_CONTROLS_H
#define SMOL_HOST_CONTROLS_H

#include <stddef.h>
#include <stdint.h>

/* Batch-one source-native Boolean pad/attention-boundary controls. Outputs
 * and caller scratch require disjoint, caller-sized storage. Invalid Boolean
 * inputs/extents are refused before any scratch or output writes.
 */
int smol_host_prefix_controls(const uint8_t *pad, const uint8_t *attention,
                             size_t rows, int64_t *groups,
                             uint8_t *mask, int64_t *positions);

/* Cached-prefix denoising: mask is [suffix_rows, prefix_rows+suffix_rows],
 * groups/positions each have suffix_rows elements. The prefix portion repeats
 * prefix_pad even for a padded suffix query, as in the pinned native source.
 * Positions use sum(prefix_pad)+cumsum(suffix_pad)-1. A zero-length prefix is
 * allowed with a null prefix_pad; suffix_rows must be positive. At most 256
 * suffix attention boundaries are admitted, keeping cumulative integer groups
 * exactly representable in the native suffix's BF16 attention-mask dtype.
 */
int smol_host_suffix_controls(const uint8_t *prefix_pad, size_t prefix_rows,
                             const uint8_t *suffix_pad,
                             const uint8_t *suffix_attention,
                             size_t suffix_rows, int64_t *groups,
                             uint8_t *mask, int64_t *positions);

#endif
