#include "tiny_full_model.h"
#include "tiny_full_tiles.h"
#include "tiny_block.h"
#include "llama_host_attention.h"
#include "smol_host_ops.h"
#include "tiny_attention.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint16_t normalized[128 * 2048], query[128 * 2048];
    uint16_t key[128 * 256], value[128 * 256];
    uint16_t attended[128 * 2048], projected[128 * 2048], residual[128 * 2048];
    uint16_t post_norm[128 * 2048], gate[128 * 5632], up[128 * 5632];
    uint16_t activated[128 * 5632], down[128 * 2048];
    uint16_t attention_scratch[3014656];
    float scores[32 * 128 * 128], softmax[160];
    int64_t positions[128];
} tiny_full_block_workspace;

static size_t block_workspace_bytes(void) {
    return sizeof(tiny_full_block_workspace);
}

static float tiled_bf16_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t tiled_float_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000))
        return (uint16_t)((bits >> 16) | UINT32_C(0x40));
    return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16) & 1u)) >> 16);
}

static int hidden_rmsnorm(const uint16_t *input, const uint16_t *weight, float epsilon,
                          uint16_t *output, size_t rows, int tiled) {
    if (!tiled) return smol_host_rmsnorm_bf16(input, weight, epsilon, output, rows, 2048);
    /* Adapt smol_host_ops.c's native CPU8 sum at the fixed hidden width 2048.
     * Its 64 blocks accumulate in four chunks of 16, then combine groups and
     * lanes in this order. Read tiles directly; normalized output is row-major. */
    for (size_t row = 0; row < rows; ++row) {
        float partial[4][8] = {{0.0f}};
        for (size_t chunk = 0; chunk < 4; ++chunk) {
            float sums[4][8] = {{0.0f}};
            for (size_t item = 0; item < 16; ++item)
                for (size_t group = 0; group < 4; ++group)
                    for (size_t lane = 0; lane < 8; ++lane) {
                        size_t col = ((chunk * 16 + item) * 4 + group) * 8 + lane;
                        float value = tiled_bf16_float(input[tiny_full_bf16_index(row, col, 2048)]);
                        sums[group][lane] += value * value;
                    }
            for (size_t group = 0; group < 4; ++group)
                for (size_t lane = 0; lane < 8; ++lane) partial[group][lane] += sums[group][lane];
        }
        for (size_t group = 1; group < 4; ++group)
            for (size_t lane = 0; lane < 8; ++lane) partial[0][lane] += partial[group][lane];
        float squares = 0.0f;
        for (size_t lane = 0; lane < 8; ++lane) squares += partial[0][lane];
        float inverse = 1.0f / sqrtf(squares / 2048.0f + epsilon);
        for (size_t col = 0; col < 2048; ++col) {
            uint16_t normalized = tiled_float_bf16(tiled_bf16_float(input[tiny_full_bf16_index(row, col, 2048)]) * inverse);
            output[row * 2048 + col] = tiled_float_bf16(tiled_bf16_float(weight[col]) * tiled_bf16_float(normalized));
        }
    }
    return 0;
}

static int hidden_residual(const uint16_t *left, const uint16_t *right,
                           uint16_t *output, size_t rows, int tiled) {
    if (!tiled) return smol_host_residual_bf16(left, right, output, rows * 2048);
    /* Only decode uses a host residual in tile mode. Keep padded norm inputs zero. */
    memset(output, 0, 32 * 2048 * sizeof(*output));
    for (size_t col = 0; col < 2048; ++col) {
        size_t at = tiny_full_bf16_index(0, col, 2048);
        output[at] = tiled_float_bf16(tiled_bf16_float(left[at]) + tiled_bf16_float(right[at]));
    }
    return 0;
}

static int full_block_run(size_t rows, size_t past_rows, const uint16_t *entry,
                   const uint16_t *iw, const uint16_t *pw,
                   float epsilon, const uint16_t *cosine, const uint16_t *sine,
                   const uint16_t *past_keys, const uint16_t *past_values,
                   tiny_projection project, void *context,
                   tiny_full_attention attention, void *attention_context,
                   tiny_full_fused_projection fused_project, tiny_full_normalization normalize,
                   uint64_t capabilities, unsigned layer,
                   void *workspace, size_t workspace_bytes,
                   uint16_t *keys, uint16_t *values, uint16_t *output) {
    if ((rows != 128 && rows != 1) || !entry || !iw || !pw || !cosine || !sine
        || !project || !workspace || workspace_bytes < sizeof(tiny_full_block_workspace)
        || (uintptr_t) workspace % _Alignof(tiny_full_block_workspace)
        || !keys || !values || !output || (fused_project && !attention) || !isfinite(epsilon) || epsilon <= 0
        || (rows == 1 && (!past_rows || past_rows >= 160 || !past_keys || !past_values))
        || (rows == 128 && (past_rows || past_keys || past_values))) return -1;
    /* Fixed driver owns distinct allocation extents; no dynamic allocation or
     * hidden tensor state. A failed projection/host stage aborts this block. */
    tiny_full_block_workspace *w = workspace;
    int tiled = (capabilities & TINY_FULL_CAP_TILED_HIDDEN) != 0;
    int device_residual = rows == 128 && (capabilities & TINY_FULL_CAP_PREFILL_RESIDUAL);
    int device_rope = capabilities & (rows == 128 ? TINY_FULL_CAP_PREFILL_ROPE : TINY_FULL_CAP_DECODE_ROPE);
    for (size_t i = 0; i < rows; ++i) w->positions[i] = (int64_t)(past_rows + i);
    const uint16_t *qkv_input = w->normalized;
    if (capabilities & TINY_FULL_CAP_PRE_ATTENTION_NORM) {
        if (normalize(attention_context, layer, entry, rows)) return -2;
        qkv_input = entry;
    } else if (hidden_rmsnorm(entry, iw, epsilon, w->normalized, rows, tiled)) return -2;
    if (project(context, 0, qkv_input, rows, 2048, 2048, w->query)
        || project(context, 1, qkv_input, rows, 2048, 256, w->key)
        || project(context, 2, qkv_input, rows, 2048, 256, w->value)) return -3;
    if (attention) {
        if ((device_rope ? (rows == 128
                ? tiny_attention_prepare_rotated(rows, w->query, w->key, w->value, keys, values)
                : tiny_attention_append_rotated(past_rows, w->query, w->key, w->value,
                                                past_keys, past_values, keys, values))
             : tiny_attention_prepare(rows, past_rows, w->query, w->key, w->value, past_keys, past_values,
                                      cosine, sine, keys, values))
            || attention(attention_context, layer, w->query, keys, values, rows, past_rows + rows, w->attended)) return -4;
    } else {
        llama_host_attention_inputs a = {32, 4, 64, rows, past_rows, 160, 0.125f,
            w->query, w->key, w->value, past_keys, past_values, cosine, sine, w->positions};
        llama_host_attention_buffers b = {w->attention_scratch, 3014656, w->scores, 32 * 128 * 128,
            w->softmax, 160, keys, values, w->attended, 4 * 160 * 64, 128 * 2048};
        if (llama_host_attention_bf16(&a, &b)) return -4;
    }
    if (fused_project) {
        if (device_residual) {
            if (fused_project(attention_context, layer, 3, w->attended, rows, 2048, 2048, entry, w->residual)) return -5;
        } else if (fused_project(attention_context, layer, 3, w->attended, rows, 2048, 2048, NULL, w->projected)
                   || hidden_residual(entry, w->projected, w->residual, rows, tiled)) return -5;
        const uint16_t *post_input = w->post_norm;
        if (normalize) {
            if (normalize(attention_context, layer, w->residual, rows)) return -6;
            post_input = w->residual;
        } else if (smol_host_rmsnorm_bf16(w->residual, pw, epsilon, w->post_norm, rows, 2048)) return -6;
        if (fused_project(attention_context, layer, 4, post_input, rows, 2048, 5632, NULL, w->gate)
            || fused_project(attention_context, layer, 5, post_input, rows, 2048, 5632, w->gate, w->activated)) return -7;
        if (device_residual)
            return fused_project(attention_context, layer, 6, w->activated, rows, 5632, 2048, w->residual, output) ? -9 : 0;
        if (fused_project(attention_context, layer, 6, w->activated, rows, 5632, 2048, NULL, w->down)) return -9;
        return hidden_residual(w->residual, w->down, output, rows, tiled) ? -10 : 0;
    }
    if (project(context, 3, w->attended, rows, 2048, 2048, w->projected)) return -5;
    if (smol_host_residual_bf16(entry, w->projected, w->residual, rows * 2048)
        || smol_host_rmsnorm_bf16(w->residual, pw, epsilon, w->post_norm, rows, 2048)) return -6;
    if (project(context, 4, w->post_norm, rows, 2048, 5632, w->gate)
        || project(context, 5, w->post_norm, rows, 2048, 5632, w->up)) return -7;
    if (smol_host_silu_mul_bf16(w->gate, w->up, w->activated, rows * 5632)) return -8;
    if (project(context, 6, w->activated, rows, 5632, 2048, w->down)) return -9;
    return smol_host_residual_bf16(w->residual, w->down, output, rows * 2048) ? -10 : 0;
}

typedef struct {
    tiny_full_block_workspace block;
    uint16_t hidden[2][128 * 2048], normalized[128 * 2048];
} tiny_full_workspace;

typedef struct {
    tiny_full_projection callback;
    void *context;
    unsigned layer;
} projection_bridge;

static int project_layer(void *opaque, unsigned logical, const uint16_t *input,
                         size_t rows, size_t k, size_t width, uint16_t *output) {
    projection_bridge *bridge = opaque;
    return bridge->callback(bridge->context, bridge->layer, logical, input, rows, k, width, output);
}

size_t tiny_full_workspace_bytes(void) {
    return sizeof(tiny_full_workspace);
}

static int argmax_bf16(const uint16_t *logits, size_t row, size_t count, int tiled, uint32_t *token) {
    if (!logits || !token || count != 32000 || row >= 128) return -1;
    float maximum = 0;
    uint32_t best = 0;
    for (size_t i = 0; i < count; ++i) {
        uint16_t word = logits[tiny_full_logits_index(row, i, tiled)];
        if ((word & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return -1;
        uint32_t bits = (uint32_t)word << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        if (!i || value > maximum) {
            maximum = value;
            best = (uint32_t)i;
        }
    }
    *token = best;
    return 0;
}

int tiny_full_argmax_bf16(const uint16_t *logits, size_t count, uint32_t *token) {
    return argmax_bf16(logits, 0, count, 0, token);
}

int tiny_full_argmax_tiled_bf16(const uint16_t *logits, size_t row, size_t count, uint32_t *token) {
    return argmax_bf16(logits, row, count, 1, token);
}

static size_t logits_rows(const tiny_full_parameters *p, size_t rows) {
    return p && (p->capabilities & TINY_FULL_CAP_TILED_LOGITS) && rows == 1 ? 32 : rows;
}

_Static_assert(sizeof(tiny_full_parameters) == 424, "Tiny parameter ABI size");
_Static_assert(sizeof(tiny_full_phase) == 32, "Tiny continuation ABI size");
_Static_assert(sizeof(tiny_full_live_layout) == 64, "Tiny layout ABI size");

const tiny_full_live_layout tiny_full_compiled_live_layout = {1, sizeof(tiny_full_workspace),
    {offsetof(tiny_full_workspace, hidden[0]), offsetof(tiny_full_workspace, hidden[1])},
    sizeof(((tiny_full_workspace *)0)->hidden[0]),
    TINY_FULL_LAYERS * TINY_FULL_CACHE_WORDS * sizeof(uint16_t),
    sizeof(tiny_full_phase), sizeof(tiny_full_state)};

int tiny_full_get_live_layout(tiny_full_live_layout *layout) {
    if (!layout) return -1;
    *layout = tiny_full_compiled_live_layout;
    return 0;
}

static int valid_context(const tiny_full_parameters *p, const tiny_full_state *s,
                         void *workspace, size_t workspace_bytes) {
    if (!p || !s || !workspace || workspace_bytes < sizeof(tiny_full_workspace)
        || (uintptr_t)workspace % _Alignof(tiny_full_workspace)
        || s->bank > 1 || s->position > 159 || (s->position && s->position < 128)
        || !p->embedding || !p->final_norm || !p->cosine || !p->sine
        || (p->capabilities & ~TINY_FULL_CAPABILITIES)
        || ((p->capabilities & TINY_FULL_CAP_DECODE_ROPE) && !(p->capabilities & TINY_FULL_CAP_PREFILL_ROPE))
        || ((p->capabilities & TINY_FULL_CAP_TILED_HIDDEN)
            && (p->capabilities & (TINY_FULL_CAP_POST_O_NORM | TINY_FULL_CAP_PREFILL_RESIDUAL))
                != (TINY_FULL_CAP_POST_O_NORM | TINY_FULL_CAP_PREFILL_RESIDUAL))
        || ((p->capabilities & (TINY_FULL_CAP_PREFILL_RESIDUAL | TINY_FULL_CAP_PREFILL_ROPE | TINY_FULL_CAP_TILED_HIDDEN))
            && (!p->attention || !p->fused_project))
        || ((p->capabilities & TINY_FULL_CAP_POST_O_NORM) != 0) != (p->normalize != NULL)
        || ((p->capabilities & TINY_FULL_CAP_PRE_ATTENTION_NORM) && !(p->capabilities & TINY_FULL_CAP_POST_O_NORM))
        || (p->normalize && (!p->attention || !p->fused_project || p->epsilon != 1e-5f))
        || !isfinite(p->epsilon) || p->epsilon <= 0
        || !s->keys[0] || !s->keys[1] || !s->values[0] || !s->values[1]) return 0;
    for (size_t layer = 0; layer < TINY_FULL_LAYERS; ++layer)
        if (!p->input_norm[layer] || !p->post_norm[layer]) return 0;
    return 1;
}

static int valid_phase(const tiny_full_state *s, const tiny_full_phase *phase) {
    return phase && phase->active == 1 && !phase->reserved
        && phase->position == s->position && phase->bank == s->bank && phase->next_token == s->next_token
        && phase->rows == (s->position ? 1u : 128u) && phase->next_layer <= TINY_FULL_LAYERS;
}

int tiny_full_model_begin(const tiny_full_parameters *p, const tiny_full_state *s, tiny_full_phase *phase,
                         const uint32_t *tokens, size_t rows, void *workspace, size_t workspace_bytes) {
    if (!valid_context(p, s, workspace, workspace_bytes) || !phase || phase->active || !tokens
        || (s->position == 0 ? rows != 128 : rows != 1)) return -1;
    for (size_t i = 0; i < rows; ++i)
        if (tokens[i] >= 32000 || (rows == 1 && tokens[i] != s->next_token)) return -1;
    tiny_full_workspace *w = workspace;
    if (p->capabilities & TINY_FULL_CAP_TILED_HIDDEN) {
        if (rows == 1) memset(w->hidden[0], 0, 32 * 2048 * sizeof(uint16_t));
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < 2048; col += 16)
                memcpy(w->hidden[0] + tiny_full_bf16_index(row, col, 2048),
                       p->embedding + (size_t)tokens[row] * 2048 + col, 16 * sizeof(uint16_t));
    } else {
        for (size_t i = 0; i < rows; ++i)
            memcpy(w->hidden[0] + i * 2048, p->embedding + (size_t)tokens[i] * 2048, 2048 * 2);
    }
    *phase = (tiny_full_phase){s->position, s->bank, s->next_token, 0, (uint32_t)rows, 1, 0};
    return 0;
}

int tiny_full_model_layer(const tiny_full_parameters *p, const tiny_full_state *s, tiny_full_phase *phase,
                         unsigned layer, tiny_full_projection project, void *context,
                         void *workspace, size_t workspace_bytes) {
    if (!valid_context(p, s, workspace, workspace_bytes) || !valid_phase(s, phase) || !project
        || layer >= TINY_FULL_LAYERS || phase->next_layer != layer) return -1;
    tiny_full_workspace *w = workspace;
    projection_bridge bridge = {project, context, layer};
    size_t offset = (size_t)layer * TINY_FULL_CACHE_WORDS;
    int status = full_block_run(phase->rows, s->position, w->hidden[layer % 2],
        p->input_norm[layer], p->post_norm[layer], p->epsilon, p->cosine, p->sine,
        s->position ? s->keys[s->bank] + offset : NULL,
        s->position ? s->values[s->bank] + offset : NULL,
        project_layer, &bridge, p->attention, context, p->fused_project, p->normalize, p->capabilities, layer,
        &w->block, block_workspace_bytes(),
        s->keys[1 - s->bank] + offset, s->values[1 - s->bank] + offset, w->hidden[1 - layer % 2]);
    if (status) { phase->active = 2; return -2; }
    ++phase->next_layer;
    return 0;
}

int tiny_full_model_finish(const tiny_full_parameters *p, tiny_full_state *s, tiny_full_phase *phase,
                          tiny_full_projection project, void *context, void *workspace, size_t workspace_bytes,
                          uint16_t *logits, size_t logits_words) {
    if (!valid_context(p, s, workspace, workspace_bytes) || !valid_phase(s, phase) || !project || !logits
        || phase->next_layer != TINY_FULL_LAYERS || logits_words < logits_rows(p, phase->rows) * 32000) return -1;
    tiny_full_workspace *w = workspace;
    size_t rows = phase->rows;
    if (hidden_rmsnorm(w->hidden[0], p->final_norm, p->epsilon,
                      w->normalized, rows, (p->capabilities & TINY_FULL_CAP_TILED_HIDDEN) != 0)) {
        phase->active = 2; return -3;
    }
    if (project(context, 22, 7, w->normalized, rows, 2048, 32000, logits)) { phase->active = 2; return -4; }
    uint32_t next_token;
    if (argmax_bf16(logits, rows - 1, 32000, (p->capabilities & TINY_FULL_CAP_TILED_LOGITS) != 0, &next_token)) {
        phase->active = 2; return -5;
    }
    s->bank = 1 - s->bank;
    s->position += rows;
    s->next_token = next_token;
    phase->active = 0;
    return 0;
}

int tiny_full_model_step(const tiny_full_parameters *p, tiny_full_state *s,
                         const uint32_t *tokens, size_t rows, tiny_full_projection project, void *context,
                         void *workspace, size_t workspace_bytes, uint16_t *logits, size_t logits_words) {
    tiny_full_phase phase = {0};
    if (!project || !logits || logits_words < logits_rows(p, rows) * 32000
        || tiny_full_model_begin(p, s, &phase, tokens, rows, workspace, workspace_bytes)) return -1;
    for (unsigned layer = 0; layer < TINY_FULL_LAYERS; ++layer)
        if (tiny_full_model_layer(p, s, &phase, layer, project, context, workspace, workspace_bytes)) return -2;
    return tiny_full_model_finish(p, s, &phase, project, context, workspace, workspace_bytes, logits, logits_words);
}
