/* NONNUMERICAL TEST DOUBLES. These satisfy only the host orchestration ABI.
 * RMSNorm, residual and SiLU are identity copies of their first operand.
 * Attention copies Q to output and transports raw K/V into compact caches.
 * There is no RoPE, normalization, softmax, reduction or BF16 arithmetic here.
 * Numerical qualification must use the real provider and independent oracles.
 */
#include "smol_host_ops.h"
#include "llama_host_attention.h"
#include <string.h>

unsigned tiny_test_rms_calls, tiny_test_residual_calls, tiny_test_silu_calls;

int smol_host_rmsnorm_bf16(const uint16_t *input, const uint16_t *weight,
                          float epsilon, uint16_t *output, size_t rows, size_t width) {
    (void)weight;
    (void)epsilon;
    ++tiny_test_rms_calls;
    memcpy(output, input, rows * width * sizeof(*output));
    return 0;
}

int smol_host_residual_bf16(const uint16_t *left, const uint16_t *right, uint16_t *output, size_t count) {
    (void)right;
    ++tiny_test_residual_calls;
    memcpy(output, left, count * sizeof(*output));
    return 0;
}

int smol_host_silu_mul_bf16(const uint16_t *gate, const uint16_t *up, uint16_t *output, size_t count) {
    (void)up;
    ++tiny_test_silu_calls;
    memcpy(output, gate, count * sizeof(*output));
    return 0;
}

int llama_host_attention_bf16(const llama_host_attention_inputs *a, const llama_host_attention_buffers *b) {
    /* Reject incorrect descriptors even though no attention math is executed. */
    if (a->query_heads != 32 || a->kv_heads != 4 || a->dimension != 64
        || a->table_rows != 160 || a->scaling != 0.125f
        || (a->rows != 128 && a->rows != 1)
        || a->past_rows + a->rows > 160
        || b->cache_capacity != 4 * 160 * 64 || b->output_capacity != 128 * 2048)
        return -1;
    for (size_t row = 0; row < a->rows; ++row)
        if (a->positions[row] != (int64_t)(a->past_rows + row)) return -1;
    size_t total = a->past_rows + a->rows;
    for (size_t head = 0; head < 4; ++head) {
        size_t start = head * total * 64;
        if (a->past_rows) {
            memcpy(b->keys + start, a->past_keys + head * a->past_rows * 64, a->past_rows * 64 * sizeof(*b->keys));
            memcpy(b->values + start, a->past_values + head * a->past_rows * 64,
                   a->past_rows * 64 * sizeof(*b->values));
        }
        for (size_t row = 0; row < a->rows; ++row) {
            size_t source = (row * 4 + head) * 64;
            size_t dest = start + (a->past_rows + row) * 64;
            memcpy(b->keys + dest, a->key + source, 64 * sizeof(*b->keys));
            memcpy(b->values + dest, a->value + source, 64 * sizeof(*b->values));
        }
    }
    memcpy(b->output, a->query, a->rows * 2048 * sizeof(*b->output));
    return 0;
}
