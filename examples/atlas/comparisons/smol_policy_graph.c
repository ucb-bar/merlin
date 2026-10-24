#include "smol_policy_graph.h"
#include "smol_policy_ops.h"
#include "smol_host_bits.h"
#include "smol_host_controls.h"
#include "smol_host_ops.h"
#include <math.h>
#include <string.h>

typedef struct {
    smol_policy_graph_project callback;
    void *user;
    smol_policy_state *state;
    unsigned site, layer;
} context;

static int project(void *opaque, unsigned role, unsigned dtype, const void *input,
    size_t rows, size_t k, unsigned output_dtype, void *output, size_t n) {
    context *c = opaque;
    int result = c->callback(c->user, c->site, c->layer, role, dtype, input,
                             rows, k, output_dtype, output, n);
    if (result) c->state->error = (unsigned) result;
    else ++c->state->projections;
    return result;
}

static int valid(const smol_policy_parameters *p, const smol_policy_frame *w,
    smol_policy_state *s, smol_policy_graph_project callback) {
    if (!p || !w || !s || !callback || s->error || !p->prefix_final_weight
        || !p->expert_final_weight || !p->language_embedding || !p->timestep || !p->time_embedding
        || !isfinite(p->language_scale) || p->language_scale <= 0
        || !isfinite(p->final_epsilon) || p->final_epsilon <= 0
        || !w->prefix_input || !w->state_projection || !w->prefix_final || !w->expert_final
        || !w->cache_key || !w->cache_value || !w->prefix_mask || !w->self_mask || !w->cross_mask
        || !w->groups || !w->positions || !w->cross_positions || !w->action_embedding
        || !w->action_time || !w->time_hidden || !w->time_activated || !w->suffix_input || !w->velocity) return 0;
    for (unsigned layer = 0; layer < 16; ++layer)
        if (p->prefix[layer].kind != SMOL_PREFIX || p->expert[layer].kind != 1 + layer % 2) return 0;
    return 1;
}

static int failed(smol_policy_state *s) {
    if (!s->error) s->error = 1;
    return 2;
}

static int same_float(float left, float right) {
    uint32_t a, b;
    memcpy(&a, &left, sizeof(a));
    memcpy(&b, &right, sizeof(b));
    return a == b;
}

int smol_policy_prefix(const uint16_t *image0, const uint16_t *image1,
    const float *state, const uint8_t *pad, const uint8_t *attention,
    const smol_policy_parameters *p, const smol_policy_frame *w, smol_policy_state *s,
    smol_policy_graph_project callback, void *user) {
    if (!valid(p, w, s, callback) || !image0 || !image1 || !state || !pad || !attention
        || s->prefix_ready || s->steps || s->projections) return 1;
    if (smol_host_prefix_controls(pad, attention, 177, w->groups, w->prefix_mask, w->positions)) return 1;
    context c = {callback, user, s, SMOL_STATE_PROJECT, 0};
    if (project(&c, 0, SMOL_F32, state, 1, 32, SMOL_F32, w->state_projection, 960)) return failed(s);
    for (size_t i = 0; i < 64 * 960; ++i) {
        w->prefix_input[i] = bf16_to_float(image0[i]);
        w->prefix_input[64 * 960 + i] = bf16_to_float(image1[i]);
    }
    for (size_t i = 0; i < 48 * 960; ++i)
        w->prefix_input[128 * 960 + i] = bf16_to_float(float_to_bf16(
            bf16_to_float(p->language_embedding[i]) * p->language_scale));
    for (size_t i = 0; i < 960; ++i) w->prefix_input[176 * 960 + i] = w->state_projection[i];
    const void *input = w->prefix_input;
    for (unsigned layer = 0; layer < 16; ++layer) {
        smol_policy_block_state block = {0};
        c.site = SMOL_PREFIX_LAYER;
        c.layer = layer;
        if (smol_policy_block(input, layer ? SMOL_BF16 : SMOL_F32, w->positions, w->prefix_mask,
            w->cache_key + layer * 177 * 320, w->cache_value + layer * 177 * 320,
            &p->prefix[layer], &w->block, &block, project, &c)) return failed(s);
        input = w->block.output;
    }
    if (smol_host_rmsnorm_bf16(input, p->prefix_final_weight, p->final_epsilon,
                             w->prefix_final, 177, 960) || s->projections != 113) return failed(s);
    s->prefix_ready = 1;
    return 0;
}

int smol_policy_step(const float *input_state, float timestep, float *next_state,
    const uint8_t *prefix_pad, const smol_policy_parameters *p, const smol_policy_frame *w,
    smol_policy_state *s, smol_policy_graph_project callback, void *user) {
    if (!valid(p, w, s, callback) || !input_state || !next_state || !prefix_pad
        || s->prefix_ready != 1 || s->steps >= 10 || s->projections != 113 + 116 * s->steps
        || !same_float(timestep, p->timestep[s->steps])) return 1;
    uint8_t suffix_pad[50], suffix_attention[50];
    for (size_t row = 0; row < 50; ++row) suffix_pad[row] = suffix_attention[row] = 1;
    if (smol_host_suffix_controls(prefix_pad, 177, suffix_pad, suffix_attention, 50,
                                  w->groups, w->self_mask, w->positions)) return 1;
    for (size_t row = 0; row < 50; ++row) {
        w->cross_positions[row] = w->positions[row] - w->positions[0];
        for (size_t key = 0; key < 177; ++key) w->cross_mask[row * 177 + key] = w->self_mask[row * 227 + key];
    }
    context c = {callback, user, s, SMOL_ACTION_IN, 0};
    if (project(&c, 0, SMOL_F32, input_state, 50, 32, SMOL_F32, w->action_embedding, 720)) return failed(s);
    const float *time = p->time_embedding + s->steps * 720;
    for (size_t row = 0; row < 50; ++row)
        for (size_t column = 0; column < 720; ++column) {
            w->action_time[row * 1440 + column] = w->action_embedding[row * 720 + column];
            w->action_time[row * 1440 + 720 + column] = time[column];
        }
    c.site = SMOL_TIME_IN;
    if (project(&c, 0, SMOL_F32, w->action_time, 50, 1440, SMOL_F32, w->time_hidden, 720)
        || smol_policy_silu_f32(w->time_hidden, w->time_activated, 50 * 720)) return failed(s);
    c.site = SMOL_TIME_OUT;
    if (project(&c, 0, SMOL_F32, w->time_activated, 50, 720, SMOL_F32, w->suffix_input, 720)) return failed(s);
    const void *input = w->suffix_input;
    for (unsigned layer = 0; layer < 16; ++layer) {
        smol_policy_block_state block = {0};
        c.site = SMOL_EXPERT_LAYER;
        c.layer = layer;
        if (smol_policy_block(input, layer ? SMOL_BF16 : SMOL_F32,
            layer % 2 ? w->cross_positions : w->positions, layer % 2 ? w->cross_mask : w->self_mask,
            w->cache_key + layer * 177 * 320, w->cache_value + layer * 177 * 320,
            &p->expert[layer], &w->block, &block, project, &c)) return failed(s);
        input = w->block.output;
    }
    if (smol_host_rmsnorm_bf16(input, p->expert_final_weight, p->final_epsilon,
                             w->expert_final, 50, 720)) return failed(s);
    for (size_t i = 0; i < 50 * 720; ++i) w->action_embedding[i] = bf16_to_float(w->expert_final[i]);
    c.site = SMOL_ACTION_OUT;
    c.layer = 0;
    if (project(&c, 0, SMOL_F32, w->action_embedding, 50, 720, SMOL_F32, w->velocity, 32)
        || smol_host_euler_f32(input_state, w->velocity, -0.1f, next_state, 50 * 32)) return failed(s);
    ++s->steps;
    return 0;
}
