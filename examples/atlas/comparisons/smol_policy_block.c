#include "smol_policy_block.h"
#include "smol_policy_ops.h"
#include "smol_host_attention.h"
#include "smol_host_bits.h"
#include "smol_host_ops.h"
#include "smol_host_rope.h"
#include <math.h>
#include <string.h>

static int project(smol_policy_block_state *s, smol_policy_project callback,
    void *context, unsigned role, unsigned input_dtype, const void *input,
    size_t rows, size_t k, unsigned output_dtype, void *output, size_t n) {
    int result = callback(context, role, input_dtype, input, rows, k, output_dtype, output, n);
    if (result) { s->error = (unsigned) result; return 1; }
    ++s->projections;
    return 0;
}

static int float_views(const float *input, uint16_t *output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(input[i])) return 1;
        uint16_t word = float_to_bf16(input[i]);
        /* Atlas BF16 projection results promoted to FP32 are exact views. */
        uint32_t original, promoted;
        float value = bf16_to_float(word);
        memcpy(&original, input + i, sizeof(original));
        memcpy(&promoted, &value, sizeof(promoted));
        if (original != promoted) return 1;
        output[i] = word;
    }
    return 0;
}

int smol_policy_block(const void *input, unsigned dtype, const int64_t *positions,
    const uint8_t *mask, uint16_t *cache_key, uint16_t *cache_value,
    const smol_policy_block_parameters *p, const smol_policy_block_frame *w,
    smol_policy_block_state *s, smol_policy_project callback, void *context) {
    if (!input || !positions || !mask || !cache_key || !cache_value || !p || !w
        || !s || !callback || dtype > SMOL_F32 || p->kind > SMOL_EXPERT_CROSS
        || s->started || s->projections || s->complete || s->error
        || !p->input_weight || !p->post_weight || !p->cosine || !p->sine
        || !p->table_rows || !isfinite(p->input_epsilon) || p->input_epsilon <= 0
        || !isfinite(p->post_epsilon) || p->post_epsilon <= 0) return 1;
    if (!w->normalized || !w->query || !w->key || !w->value || !w->rotary_query
        || !w->rotary_key || !w->joined_key || !w->joined_value || !w->attention
        || !w->projected || !w->residual || !w->postnorm || !w->gate || !w->up
        || !w->activated || !w->down || !w->output || !w->bf16_probabilities
        || !w->scores || !w->float_probabilities || !w->workspace
        || !w->cross_input || !w->cross_key || !w->cross_value) return 1;
    size_t rows = p->kind == SMOL_PREFIX ? 177 : 50;
    size_t width = p->kind == SMOL_PREFIX ? 960 : 720;
    size_t hidden = p->kind == SMOL_PREFIX ? 2560 : 2048;
    size_t keys = p->kind == SMOL_EXPERT_SELF ? 227 : 177;
    s->started = 1;
    int norm = dtype == SMOL_BF16
        ? smol_host_rmsnorm_bf16(input, p->input_weight, p->input_epsilon, w->normalized, rows, width)
        : smol_host_rmsnorm_f32_to_bf16(input, p->input_weight, p->input_epsilon, w->normalized, rows, width);
    if (norm || project(s, callback, context, SMOL_Q, SMOL_BF16, w->normalized, rows, width, SMOL_BF16, w->query, 960)
        || smol_host_rope_bf16(w->query, positions, p->cosine, p->sine, p->first_table_position,
            p->table_rows, rows, 15, 64, w->rotary_query)) goto failed;
    if (p->kind == SMOL_EXPERT_CROSS) {
        for (size_t i = 0; i < 177 * 320; ++i) w->cross_input[i] = bf16_to_float(cache_key[i]);
        if (project(s, callback, context, SMOL_K, SMOL_F32, w->cross_input, 177, 320, SMOL_F32, w->cross_key, 320)) goto failed;
        for (size_t i = 0; i < 177 * 320; ++i) w->cross_input[i] = bf16_to_float(cache_value[i]);
        if (project(s, callback, context, SMOL_V, SMOL_F32, w->cross_input, 177, 320, SMOL_F32, w->cross_value, 320)
            || float_views(w->cross_key, w->key, 177 * 320)
            || float_views(w->cross_value, w->value, 177 * 320)
            || smol_policy_cross_attention_bf16(w->rotary_query, w->key, w->value, mask,
                w->scores, w->float_probabilities, w->workspace, w->attention)) goto failed;
    } else {
        if (project(s, callback, context, SMOL_K, SMOL_BF16, w->normalized, rows, width, SMOL_BF16, w->key, 320)
            || project(s, callback, context, SMOL_V, SMOL_BF16, w->normalized, rows, width, SMOL_BF16, w->value, 320)
            || smol_host_rope_bf16(w->key, positions, p->cosine, p->sine, p->first_table_position,
                p->table_rows, rows, 5, 64, w->rotary_key)) goto failed;
        if (p->kind == SMOL_PREFIX) {
            memcpy(cache_key, w->rotary_key, 177 * 320 * sizeof(uint16_t));
            memcpy(cache_value, w->value, 177 * 320 * sizeof(uint16_t));
            memcpy(w->joined_key, cache_key, 177 * 320 * sizeof(uint16_t));
            memcpy(w->joined_value, cache_value, 177 * 320 * sizeof(uint16_t));
        } else {
            memcpy(w->joined_key, cache_key, 177 * 320 * sizeof(uint16_t));
            memcpy(w->joined_value, cache_value, 177 * 320 * sizeof(uint16_t));
            memcpy(w->joined_key + 177 * 320, w->rotary_key, 50 * 320 * sizeof(uint16_t));
            memcpy(w->joined_value + 177 * 320, w->value, 50 * 320 * sizeof(uint16_t));
        }
        if (smol_host_attention_cpu16_bf16(w->rotary_query, w->joined_key, w->joined_value, mask,
            rows, keys, 15, 5, 64, w->workspace, w->scores, w->bf16_probabilities, w->attention)) goto failed;
    }
    if (project(s, callback, context, SMOL_O, SMOL_BF16, w->attention, rows, 960, SMOL_BF16, w->projected, width)) goto failed;
    int residual = dtype == SMOL_BF16
        ? smol_host_residual_bf16(w->projected, input, w->residual, rows * width)
        : smol_host_residual_bf16_f32(w->projected, input, w->residual, rows * width);
    if (residual || smol_host_rmsnorm_bf16(w->residual, p->post_weight, p->post_epsilon, w->postnorm, rows, width)
        || project(s, callback, context, SMOL_GATE, SMOL_BF16, w->postnorm, rows, width, SMOL_BF16, w->gate, hidden)
        || project(s, callback, context, SMOL_UP, SMOL_BF16, w->postnorm, rows, width, SMOL_BF16, w->up, hidden)
        || smol_host_silu_mul_bf16(w->gate, w->up, w->activated, rows * hidden)
        || project(s, callback, context, SMOL_DOWN, SMOL_BF16, w->activated, rows, hidden, SMOL_BF16, w->down, width)
        || smol_host_residual_bf16(w->down, w->residual, w->output, rows * width)) goto failed;
    if (s->projections != 7) goto failed;
    s->complete = 1;
    return 0;
failed:
    if (!s->error) s->error = 1;
    return 2;
}
