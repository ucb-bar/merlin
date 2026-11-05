#ifndef LLAMA_HOST_ATTENTION_H
#define LLAMA_HOST_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t query_heads, kv_heads, dimension, rows, past_rows, table_rows;
    float scaling;
    const uint16_t *query, *key, *value, *past_keys, *past_values;
    const uint16_t *cosine_table, *sine_table;
    const int64_t *positions;
} llama_host_attention_inputs;

typedef struct {
    uint16_t *scratch;
    size_t scratch_words;
    float *softmax_scores;
    size_t softmax_scores_capacity;
    float *softmax_workspace;
    size_t softmax_capacity;
    uint16_t *keys, *values, *output;
    size_t cache_capacity, output_capacity;
} llama_host_attention_buffers;

/* Batch-one full-axis BF16 attention under the explicitly qualified host
 * arithmetic policies. Input Q/K/V are unrotated row-major linear replies;
 * prefix caches are previous outputs of this function. Tables contain only
 * immutable model/config parameters. All mutable buffers must be disjoint
 * from each other, descriptors and immutable inputs. Invalid positions,
 * extents, nonfinite words, rounding mode and aliases fail before writes.
 * Helper arithmetic failures return -2; discard all outputs on any failure.
 * Does not project Q/K/V/O, implement a complete graph, or launch a device.
 */
int llama_host_attention_bf16(const llama_host_attention_inputs *input,
                              const llama_host_attention_buffers *buffers);

#endif
