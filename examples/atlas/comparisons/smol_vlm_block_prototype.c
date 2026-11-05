#include "smol_vlm_block_prototype.h"
#include "smol_host_attention.h"
#include "smol_host_bits.h"
#include "smol_host_controls.h"
#include "smol_host_ops.h"
#include "smol_host_rope.h"

#include <fenv.h>
#include <math.h>

static int span(const void *pointer, size_t bytes) {
    return pointer && bytes && bytes - 1 <= UINTPTR_MAX - (uintptr_t) pointer;
}

static int overlap(const void *left, size_t left_bytes,
                   const void *right, size_t right_bytes) {
    uintptr_t a = (uintptr_t) left, b = (uintptr_t) right;
    return a <= b ? b - a < left_bytes : a - b < right_bytes;
}

static int finite_bf16(const uint16_t *values, size_t count) {
    for (size_t i = 0; i < count; ++i) if (!isfinite(bf16_to_float(values[i]))) return 0;
    return 1;
}

static int fail(smol_block_state *state, int result) {
    state->failed = 1;
    state->result = result;
    return result;
}

static int call_linear(smol_block_state *state, smol_block_linear linear,
                        void *context, unsigned role, const uint16_t *input,
                        size_t input_width, uint16_t *output, size_t output_width) {
    if (!finite_bf16(input, SMOL_BLOCK_ROWS * input_width))
        return fail(state, SMOL_BLOCK_ARITHMETIC);
    state->last_role = role;
    state->callback_code = linear(context, role, input, SMOL_BLOCK_ROWS,
                               input_width, output, output_width);
    if (state->callback_code) return fail(state, SMOL_BLOCK_CALLBACK);
    if (!finite_bf16(output, SMOL_BLOCK_ROWS * output_width))
        return fail(state, SMOL_BLOCK_ARITHMETIC);
    ++state->completed_linears;
    return SMOL_BLOCK_OK;
}

int smol_vlm_block_prototype(const uint16_t *input, const uint8_t *pad,
                             const uint8_t *attention_boundaries,
                             const smol_block_parameters *p,
                             const smol_block_frame *w,
                             smol_block_state *s,
                             smol_block_linear linear, void *context) {
    const size_t rows = SMOL_BLOCK_ROWS, width = SMOL_BLOCK_WIDTH;
    const size_t hidden = SMOL_BLOCK_MLP, kv_width = SMOL_BLOCK_KV_HEADS * SMOL_BLOCK_HEAD_DIM;
    const size_t matrix_bytes = rows * width * sizeof(uint16_t);
    const size_t kv_bytes = rows * kv_width * sizeof(uint16_t);
    const size_t hidden_bytes = rows * hidden * sizeof(uint16_t);
    const size_t score_count = SMOL_BLOCK_HEADS * rows * rows;
    if (!span(s, sizeof(*s))) return SMOL_BLOCK_INVALID;
    if (s->started || s->completed_linears || s->last_role || s->failed
        || s->complete || s->cache_valid || s->callback_code || s->result)
        return SMOL_BLOCK_USED_STATE;
    if (!span(p, sizeof(*p)) || !span(w, sizeof(*w)) || !linear
        || !p->table_rows || p->table_rows > SIZE_MAX / sizeof(float) / (SMOL_BLOCK_HEAD_DIM / 2)
        || !isfinite(p->input_epsilon) || p->input_epsilon <= 0
        || !isfinite(p->post_epsilon) || p->post_epsilon <= 0
        || fegetround() != FE_TONEAREST) return SMOL_BLOCK_INVALID;
    size_t table_count = p->table_rows * (SMOL_BLOCK_HEAD_DIM / 2);
    const void *immutable[] = {input, pad, attention_boundaries, p, w,
        p->input_norm_weight, p->post_norm_weight, p->cosine, p->sine};
    const size_t immutable_bytes[] = {matrix_bytes, rows, rows, sizeof(*p), sizeof(*w),
        width * sizeof(uint16_t), width * sizeof(uint16_t), table_count * sizeof(float), table_count * sizeof(float)};
    void *mutable[] = {s, w->normalized, w->query, w->key, w->value, w->rotary_query,
        w->rotary_key, w->attention, w->projected, w->residual, w->postnorm, w->gate,
        w->up, w->mlp_input, w->down, w->output, w->scores, w->attention_workspace,
        w->probabilities, w->mask, w->groups, w->positions};
    const size_t mutable_bytes[] = {sizeof(*s), matrix_bytes, matrix_bytes, kv_bytes,
        kv_bytes, matrix_bytes, kv_bytes, matrix_bytes, matrix_bytes, matrix_bytes,
        matrix_bytes, hidden_bytes, hidden_bytes, hidden_bytes, matrix_bytes, matrix_bytes,
        score_count * sizeof(float), rows * sizeof(float), score_count * sizeof(uint16_t),
        rows * rows, rows * sizeof(int64_t), rows * sizeof(int64_t)};
    for (size_t i = 0; i < sizeof(immutable) / sizeof(*immutable); ++i)
        if (!span(immutable[i], immutable_bytes[i])) return SMOL_BLOCK_INVALID;
    for (size_t i = 0; i < sizeof(mutable) / sizeof(*mutable); ++i) {
        if (!span(mutable[i], mutable_bytes[i])) return SMOL_BLOCK_INVALID;
        for (size_t j = 0; j < i; ++j)
            if (overlap(mutable[i], mutable_bytes[i], mutable[j], mutable_bytes[j])) return SMOL_BLOCK_INVALID;
        for (size_t j = 0; j < sizeof(immutable) / sizeof(*immutable); ++j)
            if (overlap(mutable[i], mutable_bytes[i], immutable[j], immutable_bytes[j])) return SMOL_BLOCK_INVALID;
    }
    if (!finite_bf16(input, rows * width) || !finite_bf16(p->input_norm_weight, width)
        || !finite_bf16(p->post_norm_weight, width)) return SMOL_BLOCK_INVALID;
    for (size_t i = 0; i < table_count; ++i)
        if (!isfinite(p->cosine[i]) || !isfinite(p->sine[i])) return SMOL_BLOCK_INVALID;
    /* Validate position-table bounds before any callback acquires ownership. */
    int64_t position = - 1;
    for (size_t row = 0; row < rows; ++row) {
        if (pad[row] > 1 || attention_boundaries[row] > 1) return SMOL_BLOCK_INVALID;
        position += pad[row];
        if (position < p->first_table_position
            || (uint64_t) position - (uint64_t) p->first_table_position >= p->table_rows)
            return SMOL_BLOCK_INVALID;
    }
    s->started = 1;
    if (smol_host_prefix_controls(pad, attention_boundaries, rows, w->groups, w->mask, w->positions)
        || smol_host_rmsnorm_bf16(input, p->input_norm_weight, p->input_epsilon,
                                 w->normalized, rows, width)) return fail(s, SMOL_BLOCK_ARITHMETIC);
    if (call_linear(s, linear, context, SMOL_BLOCK_Q, w->normalized, width, w->query, width)
        || call_linear(s, linear, context, SMOL_BLOCK_K, w->normalized, width, w->key, kv_width)
        || call_linear(s, linear, context, SMOL_BLOCK_V, w->normalized, width, w->value, kv_width)) return s->result;
    if (smol_host_rope_bf16(w->query, w->positions, p->cosine, p->sine, p->first_table_position,
                            p->table_rows, rows, SMOL_BLOCK_HEADS, SMOL_BLOCK_HEAD_DIM, w->rotary_query)
        || smol_host_rope_bf16(w->key, w->positions, p->cosine, p->sine, p->first_table_position,
                               p->table_rows, rows, SMOL_BLOCK_KV_HEADS, SMOL_BLOCK_HEAD_DIM, w->rotary_key)
        || smol_host_attention_cpu16_bf16(w->rotary_query, w->rotary_key, w->value, w->mask,
                                          rows, rows, SMOL_BLOCK_HEADS, SMOL_BLOCK_KV_HEADS,
                                          SMOL_BLOCK_HEAD_DIM, w->attention_workspace, w->scores,
                                          w->probabilities, w->attention)) return fail(s, SMOL_BLOCK_ARITHMETIC);
    if (call_linear(s, linear, context, SMOL_BLOCK_O, w->attention, width, w->projected, width)) return s->result;
    if (smol_host_residual_bf16(w->projected, input, w->residual, rows * width)
        || smol_host_rmsnorm_bf16(w->residual, p->post_norm_weight, p->post_epsilon,
                                 w->postnorm, rows, width)) return fail(s, SMOL_BLOCK_ARITHMETIC);
    if (call_linear(s, linear, context, SMOL_BLOCK_GATE, w->postnorm, width, w->gate, hidden)
        || call_linear(s, linear, context, SMOL_BLOCK_UP, w->postnorm, width, w->up, hidden)) return s->result;
    if (smol_host_silu_mul_bf16(w->gate, w->up, w->mlp_input, rows * hidden)) return fail(s, SMOL_BLOCK_ARITHMETIC);
    if (call_linear(s, linear, context, SMOL_BLOCK_DOWN, w->mlp_input, hidden, w->down, width)) return s->result;
    if (smol_host_residual_bf16(w->down, w->residual, w->output, rows * width)
        || !finite_bf16(w->output, rows * width)) return fail(s, SMOL_BLOCK_ARITHMETIC);
    s->complete = 1;
    s->cache_valid = 1;
    s->result = SMOL_BLOCK_OK;
    return SMOL_BLOCK_OK;
}
