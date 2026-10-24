#ifndef SMOL_VISION_ATTENTION_H
#define SMOL_VISION_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

/* CPU16 BF16 flash arithmetic for the vision all-zero additive mask.
 * Q is [rows, heads, 64], K/V are [keys, heads, 64], and output matches Q.
 * One image per call; no causal mask or dropout. Keys must be a multiple of16,
 * at most1024; rows are at most1024 and heads at most12. Output is disjoint
 * from all inputs.
 * Returns zero on success. Failure can leave partial output after computation.
 */
int smol_vision_attention_cpu16_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    size_t rows, size_t keys, size_t heads, uint16_t *output);

#endif
