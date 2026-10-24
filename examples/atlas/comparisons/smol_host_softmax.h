#ifndef SMOL_HOST_SOFTMAX_H
#define SMOL_HOST_SOFTMAX_H

#include <stddef.h>
#include <stdint.h>

/* Fixed current-native CPU16 arithmetic policy for contiguous [rows, keys]
 * finite FP32 scores, including finite mask sentinels. Each row covers the
 * entire key axis. Caller supplies keys FP32 workspace elements and rows*keys
 * BF16 output elements; all buffers must be disjoint. Invalid extents,
 * nonfinite scores or non-nearest rounding are refused before output writes.
 * This software policy does not promise parity with other Torch/ISA backends.
 */
int smol_host_softmax_cpu16_bf16(const float *scores, size_t rows, size_t keys,
                               float *workspace, uint16_t *output);

#endif
