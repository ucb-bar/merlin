/* Test-only attention ABI. Keep field order/types aligned with the provider.
 * The explicit integration suite compiles against the real provider headers. */
#ifndef TEST_LLAMA_HOST_ATTENTION_H
#define TEST_LLAMA_HOST_ATTENTION_H
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
int llama_host_attention_bf16(const llama_host_attention_inputs *, const llama_host_attention_buffers *);
#endif
