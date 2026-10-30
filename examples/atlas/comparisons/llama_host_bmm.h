#ifndef LLAMA_HOST_BMM_H
#define LLAMA_HOST_BMM_H

#include <stddef.h>
#include <stdint.h>

/* Explicit current-native BF16 emulation policy: each packed reduction pair
 * accumulates its upper member before its lower member, with FP32 FMA. It is
 * not a promise of parity with any other GEMM backend. Batch-major row-major
 * inputs/output; right is [batch,N,K] when transposed, else [batch,K,N].
 * Callers provide full extents and disjoint buffers. */
int llama_host_bmm_bf16_pair(const uint16_t *left, const uint16_t *right,
                            uint16_t *output, size_t batches, size_t rows,
                            size_t reduction, size_t columns, int right_transposed);

/* Distinct BF16 tensor boundaries: round score*scale, then add BF16 mask and
 * round again. Mask broadcasts [rows,columns] over batches. */
int llama_host_scale_mask_bf16(const uint16_t *scores, const uint16_t *mask,
                               float scale, uint16_t *output, size_t batches,
                               size_t rows, size_t columns);
int llama_host_merge_heads_bf16(const uint16_t *input, uint16_t *output,
                                size_t rows, size_t heads, size_t width);
/* Contiguous cache keys have absolute positions 0..keys-1. Runtime query
 * positions must be in that range; reject all invalid rows before writing. */
int llama_host_causal_mask_bf16(const int64_t *positions, uint16_t *output,
                                size_t rows, size_t keys);

#endif
