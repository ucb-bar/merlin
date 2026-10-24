#ifndef LLAMA_HOST_LAYOUT_H
#define LLAMA_HOST_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

/* Batch-one layouts. All outputs require distinct storage from inputs.
 * These primitives do not generate rotary tables or implement attention. */
int llama_host_split_heads_bf16(const uint16_t *linear, uint16_t *output,
                                size_t rows, size_t heads, size_t width);
int llama_host_rotary_bf16(const uint16_t *input, const uint16_t *cosine,
                          const uint16_t *sine, uint16_t *output,
                          size_t heads, size_t rows, size_t width);
int llama_host_cache_append_bf16(const uint16_t *prefix, const uint16_t *tail,
                                uint16_t *output, size_t heads,
                                size_t past_rows, size_t rows, size_t width);
int llama_host_repeat_kv_bf16(const uint16_t *input, uint16_t *output,
                             size_t heads, size_t groups, size_t rows,
                             size_t width);

#endif
