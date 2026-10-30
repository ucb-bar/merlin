#include "llama_host_attention.h"
#include "llama_host_layout.h"
#include "llama_rotary_table.h"
#include "llama_host_bmm.h"
#include "smol_host_softmax.h"
#include "smol_host_bits.h"

#include <fenv.h>
#include <math.h>

static int count3(size_t a, size_t b, size_t c, size_t *result) {
    if (!a || !b || !c || b > SIZE_MAX / a || c > SIZE_MAX / (a * b)) return -1;
    *result = a * b * c;
    return *result > SIZE_MAX / sizeof(uint16_t) ? -1 : 0;
}

static int accumulate(size_t *total, size_t value, size_t copies) {
    if (value > SIZE_MAX / copies || *total > SIZE_MAX - value * copies) return -1;
    *total += value * copies;
    return 0;
}

static int span(const void *pointer, size_t bytes) {
    return pointer && bytes && bytes - 1 <= UINTPTR_MAX - (uintptr_t)pointer;
}

static int overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

static int finite_words(const uint16_t *words, size_t count) {
    for (size_t i = 0; i < count; ++i)
        if ((words[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return 0;
    return 1;
}

int llama_host_attention_bf16(const llama_host_attention_inputs *a,
                              const llama_host_attention_buffers *b) {
    if (!span(a, sizeof(*a)) || !span(b, sizeof(*b))) return -1;
    if (!a->query_heads || !a->kv_heads || a->query_heads % a->kv_heads
        || !a->dimension || a->dimension % 2 || !a->rows || !a->table_rows
        || a->past_rows > SIZE_MAX - a->rows || !isfinite(a->scaling)
        || a->scaling <= 0 || fegetround() != FE_TONEAREST) return -1;
    size_t keys = a->past_rows + a->rows;
    size_t q, k, cache, repeated, scores, mask, cosine, table, prefix = 0;
    if (count3(a->query_heads, a->rows, a->dimension, &q)
        || count3(a->kv_heads, a->rows, a->dimension, &k)
        || count3(a->kv_heads, keys, a->dimension, &cache)
        || count3(a->query_heads, keys, a->dimension, &repeated)
        || count3(a->query_heads, a->rows, keys, &scores)
        || count3(1, a->rows, keys, &mask)
        || count3(1, a->rows, a->dimension, &cosine)
        || count3(1, a->table_rows, a->dimension, &table)
        || (a->past_rows && count3(a->kv_heads, a->past_rows, a->dimension, &prefix))) return -1;
    size_t needed = 0;
    if (accumulate(&needed, q, 3) || accumulate(&needed, k, 3)
        || accumulate(&needed, cosine, 2) || accumulate(&needed, repeated, 2)
        || accumulate(&needed, scores, 3) || accumulate(&needed, mask, 1)
        || needed > SIZE_MAX / sizeof(uint16_t) || keys > SIZE_MAX / sizeof(float)
        || scores > SIZE_MAX / sizeof(float)
        || a->rows > SIZE_MAX / sizeof(int64_t)
        || b->scratch_words < needed || b->softmax_capacity < keys
        || b->softmax_scores_capacity < scores
        || b->cache_capacity < cache || b->output_capacity < q) return -1;
    const void *immutable[] = {a, b, a->query, a->key, a->value,
        a->cosine_table, a->sine_table, a->positions, a->past_keys, a->past_values};
    size_t immutable_bytes[] = {sizeof(*a), sizeof(*b), q * 2, k * 2, k * 2,
        table * 2, table * 2, a->rows * sizeof(int64_t), prefix * 2, prefix * 2};
    const void *mutable[] = {b->scratch, b->softmax_scores, b->softmax_workspace, b->keys, b->values, b->output};
    size_t mutable_bytes[] = {needed * 2, scores * sizeof(float), keys * sizeof(float), cache * 2, cache * 2, q * 2};
    for (size_t i = 0; i < 10; ++i)
        if (immutable_bytes[i] && !span(immutable[i], immutable_bytes[i])) return -1;
    for (size_t i = 0; i < 6; ++i) {
        if (!span(mutable[i], mutable_bytes[i])) return -1;
        for (size_t j = 0; j < i; ++j)
            if (overlap(mutable[i], mutable_bytes[i], mutable[j], mutable_bytes[j])) return -1;
        for (size_t j = 0; j < 10; ++j)
            if (immutable_bytes[j] && overlap(mutable[i], mutable_bytes[i], immutable[j], immutable_bytes[j])) return -1;
    }
    for (size_t row = 0; row < a->rows; ++row)
        if (a->positions[row] < 0 || (uint64_t)a->positions[row] >= a->table_rows
            || (uint64_t)a->positions[row] >= keys) return -1;
    if (!finite_words(a->query, q) || !finite_words(a->key, k) || !finite_words(a->value, k)
        || !finite_words(a->cosine_table, table) || !finite_words(a->sine_table, table)
        || (prefix && (!finite_words(a->past_keys, prefix) || !finite_words(a->past_values, prefix)))) return -1;
    /* Every intermediate below is produced by the preceding C operation.
     * No native scores/probabilities/rotary/cache/output are inputs. */
    uint16_t *p = b->scratch;
    uint16_t *split_q = p;
    p += q;
    uint16_t *split_k = p;
    p += k;
    uint16_t *split_v = p;
    p += k;
    uint16_t *cos = p;
    p += cosine;
    uint16_t *sin = p;
    p += cosine;
    uint16_t *rot_q = p;
    p += q;
    uint16_t *rot_k = p;
    p += k;
    uint16_t *repeat_k = p;
    p += repeated;
    uint16_t *repeat_v = p;
    p += repeated;
    uint16_t *qk = p;
    p += scores;
    uint16_t *causal = p;
    p += mask;
    uint16_t *masked = p;
    p += scores;
    uint16_t *probabilities = p;
    p += scores;
    uint16_t *attended = p;
    if (llama_host_split_heads_bf16(a->query, split_q, a->rows, a->query_heads, a->dimension)
        || llama_host_split_heads_bf16(a->key, split_k, a->rows, a->kv_heads, a->dimension)
        || llama_host_split_heads_bf16(a->value, split_v, a->rows, a->kv_heads, a->dimension)
        || llama_host_rotary_lookup_bf16(a->cosine_table, a->sine_table, a->table_rows,
                                        a->dimension, a->positions, a->rows, cos, sin)
        || llama_host_rotary_bf16(split_q, cos, sin, rot_q, a->query_heads, a->rows, a->dimension)
        || llama_host_rotary_bf16(split_k, cos, sin, rot_k, a->kv_heads, a->rows, a->dimension)
        || llama_host_cache_append_bf16(a->past_keys, rot_k, b->keys, a->kv_heads, a->past_rows, a->rows, a->dimension)
        || llama_host_cache_append_bf16(a->past_values, split_v, b->values, a->kv_heads, a->past_rows, a->rows, a->dimension)
        || llama_host_repeat_kv_bf16(b->keys, repeat_k, a->kv_heads, a->query_heads / a->kv_heads, keys, a->dimension)
        || llama_host_repeat_kv_bf16(b->values, repeat_v, a->kv_heads, a->query_heads / a->kv_heads, keys, a->dimension)
        || llama_host_bmm_bf16_pair(rot_q, repeat_k, qk, a->query_heads, a->rows, a->dimension, keys, 1)
        || llama_host_causal_mask_bf16(a->positions, causal, a->rows, keys)
        || llama_host_scale_mask_bf16(qk, causal, a->scaling, masked, a->query_heads, a->rows, keys)) return -2;
    /* Exact BF16-to-FP32 expansion is a cast boundary, not a recomputation. */
    for (size_t i = 0; i < scores; ++i) b->softmax_scores[i] = bf16_to_float(masked[i]);
    if (smol_host_softmax_cpu16_bf16(b->softmax_scores, a->query_heads * a->rows, keys,
                                    b->softmax_workspace, probabilities)
        || llama_host_bmm_bf16_pair(probabilities, repeat_v, attended, a->query_heads,
                                     a->rows, keys, a->dimension, 0)
        || llama_host_merge_heads_bf16(attended, b->output, a->rows, a->query_heads, a->dimension)) return -2;
    return 0;
}
