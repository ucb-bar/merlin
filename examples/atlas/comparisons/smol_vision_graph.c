#include "smol_vision_graph.h"
#include "smol_vision_ops.h"
#include "smol_vision_attention.h"
#include "smol_host_bits.h"
#include "smol_host_ops.h"
#include <math.h>

int smol_vision_patch16(const uint16_t *image, uint16_t *matrix) {
    if (!image || !matrix || image == matrix) return 1;
    for (size_t row = 0; row < 32; ++row)
        for (size_t column = 0; column < 32; ++column)
            for (size_t channel = 0; channel < 3; ++channel)
                for (size_t y = 0; y < 16; ++y)
                    for (size_t x = 0; x < 16; ++x)
                        matrix[((row * 32 + column) * 3 + channel) * 256 + y * 16 + x]
                            = image[(channel * 512 + row * 16 + y) * 512 + column * 16 + x];
    return 0;
}

int smol_vision_shuffle4(const uint16_t *input, uint16_t *output) {
    if (!input || !output || input == output) return 1;
    /* Source view32x8x3072, transpose8x32x3072, group four height rows,
     * transpose8x8x12288. Channels are grouped by vertical then horizontal. */
    for (size_t row = 0; row < 8; ++row)
        for (size_t column = 0; column < 8; ++column)
            for (size_t y = 0; y < 4; ++y)
                for (size_t x = 0; x < 4; ++x)
                    for (size_t channel = 0; channel < 768; ++channel)
                        output[((row * 8 + column) * 16 + y * 4 + x) * 768 + channel]
                            = input[((row * 4 + y) * 32 + column * 4 + x) * 768 + channel];
    return 0;
}

static int project(smol_vision_state *s, smol_vision_project callback, void *context,
                   unsigned layer, unsigned role, const uint16_t *input,
                   size_t rows, size_t k, uint16_t *output, size_t n) {
    int code = callback(context, layer, role, input, rows, k, output, n);
    if (code) { s->error = (unsigned) code; return 1; }
    ++s->projections;
    return 0;
}

int smol_vision_image_layer(const uint16_t *image, const smol_vision_parameters *p,
    const smol_vision_frame *w, smol_vision_state *s, smol_vision_project callback,
    void *context, unsigned layer) {
    if (!image || !p || !w || !s || !callback || layer >= 12 || s->started != (layer != 0)
        || s->projections != (layer ? 1 + 6 * layer : 0) || s->complete
        || s->error || !p->position || !p->post_weight || !p->post_bias || !p->gelu
        || !isfinite(p->epsilon) || p->epsilon <= 0 || !w->tokens || !w->normalized
        || !w->query || !w->key || !w->value || !w->attended || !w->projected
        || !w->large || !w->activated || !w->shuffle || !w->output) return 1;
    for (size_t layer = 0; layer < 12; ++layer)
        if (!p->layers[layer].norm1_weight || !p->layers[layer].norm1_bias
            || !p->layers[layer].norm2_weight || !p->layers[layer].norm2_bias) return 1;
    if (layer == 0) {
        s->started = 1;
        if (smol_vision_patch16(image, w->large)
            || project(s, callback, context, 0, SMOL_VISION_PATCH, w->large, 1024, 768, w->tokens, 768)
            || smol_host_residual_bf16(w->tokens, p->position, w->normalized, 1024 * 768)) goto failed;
        for (size_t i = 0; i < 1024 * 768; ++i) w->tokens[i] = w->normalized[i];
    }
    {
        const smol_vision_layer *q = &p->layers[layer];
        if (smol_vision_layernorm_bf16(w->tokens, q->norm1_weight, q->norm1_bias, p->epsilon, w->normalized, 1024)
            || project(s, callback, context, layer, SMOL_VISION_Q, w->normalized, 1024, 768, w->query, 768)
            || project(s, callback, context, layer, SMOL_VISION_K, w->normalized, 1024, 768, w->key, 768)
            || project(s, callback, context, layer, SMOL_VISION_V, w->normalized, 1024, 768, w->value, 768)
            || smol_vision_attention_cpu16_bf16(w->query, w->key, w->value, 1024, 1024, 12, w->attended)
            || project(s, callback, context, layer, SMOL_VISION_O, w->attended, 1024, 768, w->projected, 768)
            || smol_host_residual_bf16(w->tokens, w->projected, w->normalized, 1024 * 768)) goto failed;
        for (size_t i = 0; i < 1024 * 768; ++i) w->tokens[i] = w->normalized[i];
        if (smol_vision_layernorm_bf16(w->tokens, q->norm2_weight, q->norm2_bias, p->epsilon, w->normalized, 1024)
            || project(s, callback, context, layer, SMOL_VISION_FC1, w->normalized, 1024, 768, w->large, 3072)
            || smol_vision_gelu_bf16(w->large, p->gelu, w->activated, 1024 * 3072)
            || project(s, callback, context, layer, SMOL_VISION_FC2, w->activated, 1024, 3072, w->projected, 768)
            || smol_host_residual_bf16(w->tokens, w->projected, w->normalized, 1024 * 768)) goto failed;
        for (size_t i = 0; i < 1024 * 768; ++i) w->tokens[i] = w->normalized[i];
    }
    if (layer == 11) {
        if (smol_vision_layernorm_bf16(w->tokens, p->post_weight, p->post_bias, p->epsilon, w->normalized, 1024)
            || smol_vision_shuffle4(w->normalized, w->shuffle)
            || project(s, callback, context, 12, SMOL_VISION_CONNECTOR, w->shuffle, 64, 12288, w->output, 960)) goto failed;
        /* The source explicitly materializes sqrt960 in BF16 before image scaling. */
        for (size_t i = 0; i < 64 * 960; ++i)
            w->output[i] = float_to_bf16(bf16_to_float(w->output[i]) * 31.0f);
        if (s->projections != 74) goto failed;
        s->complete = 1;
    }
    return 0;
failed:
    if (!s->error) s->error = 1;
    return 2;
}

int smol_vision_image(const uint16_t *image, const smol_vision_parameters *p,
    const smol_vision_frame *w, smol_vision_state *s, smol_vision_project callback,
    void *context) {
    for (unsigned layer = 0; layer < 12; ++layer) {
        int code = smol_vision_image_layer(image, p, w, s, callback, context, layer);
        if (code) return code;
    }
    return 0;
}
