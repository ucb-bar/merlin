#ifndef TINY_BLOCK_H
#define TINY_BLOCK_H
#include <stddef.h>
#include <stdint.h>

/* Fixed checkpoint geometry: first block, prefill 128 then first decode 1.
 * Projection replies are the only external arithmetic boundary. The callback
 * must fill every row-major BF16 output word and return zero on completion. */
typedef int (*tiny_projection)(void *, unsigned, const uint16_t *, size_t,
                               size_t, size_t, uint16_t *);
size_t tiny_block_workspace_bytes(void);
int tiny_block_run(size_t rows, const uint16_t *entry,
                   const uint16_t *input_norm_weight, const uint16_t *post_norm_weight,
                   float epsilon, const uint16_t *cosine, const uint16_t *sine,
                   const uint16_t *past_keys, const uint16_t *past_values,
                   tiny_projection projection, void *context,
                   void *workspace, size_t workspace_bytes,
                   uint16_t *keys, uint16_t *values, uint16_t *output);
#endif
