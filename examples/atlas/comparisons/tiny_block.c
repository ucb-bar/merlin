#include "tiny_block.h"
#include "llama_host_attention.h"
#include "smol_host_ops.h"
#include <math.h>
#include <stdint.h>

typedef struct {
    uint16_t normalized[128 * 2048], query[128 * 2048];
    uint16_t key[128 * 256], value[128 * 256];
    uint16_t attended[128 * 2048], projected[128 * 2048], residual[128 * 2048];
    uint16_t post_norm[128 * 2048], gate[128 * 5632], up[128 * 5632];
    uint16_t activated[128 * 5632], down[128 * 2048];
    uint16_t attention_scratch[3014656];
    float scores[32 * 128 * 128], softmax[129];
    int64_t positions[128];
} tiny_workspace;

size_t tiny_block_workspace_bytes(void) {
    return sizeof(tiny_workspace);
}

int tiny_block_run(size_t rows, const uint16_t *entry,
                   const uint16_t *iw, const uint16_t *pw,
                   float epsilon, const uint16_t *cosine, const uint16_t *sine,
                   const uint16_t *past_keys, const uint16_t *past_values,
                   tiny_projection project, void *context,
                   void *workspace, size_t workspace_bytes,
                   uint16_t *keys, uint16_t *values, uint16_t *output) {
    if ((rows != 128 && rows != 1) || !entry || !iw || !pw || !cosine || !sine
        || !project || !workspace || workspace_bytes < sizeof(tiny_workspace)
        || (uintptr_t) workspace % _Alignof(tiny_workspace)
        || !keys || !values || !output || !isfinite(epsilon) || epsilon <= 0
        || (rows == 1 && (!past_keys || !past_values))
        || (rows == 128 && (past_keys || past_values))) return -1;
    /* Fixed driver owns distinct allocation extents; no dynamic allocation or
     * hidden tensor state. A failed projection/host stage aborts this block. */
    tiny_workspace *w = workspace;
    for (size_t i = 0; i < rows; ++i) w->positions[i] = (int64_t)(rows == 128 ? i : 128);
    if (smol_host_rmsnorm_bf16(entry, iw, epsilon, w->normalized, rows, 2048)) return -2;
    if (project(context, 0, w->normalized, rows, 2048, 2048, w->query)
        || project(context, 1, w->normalized, rows, 2048, 256, w->key)
        || project(context, 2, w->normalized, rows, 2048, 256, w->value)) return -3;
    llama_host_attention_inputs a = {32, 4, 64, rows, rows == 1 ? 128 : 0, 160, 0.125f,
        w->query, w->key, w->value, past_keys, past_values, cosine, sine, w->positions};
    llama_host_attention_buffers b = {w->attention_scratch, 3014656, w->scores, 32 * 128 * 128,
        w->softmax, 129, keys, values, w->attended, 4 * 129 * 64, 128 * 2048};
    if (llama_host_attention_bf16(&a, &b)) return -4;
    if (project(context, 3, w->attended, rows, 2048, 2048, w->projected)) return -5;
    if (smol_host_residual_bf16(entry, w->projected, w->residual, rows * 2048)
        || smol_host_rmsnorm_bf16(w->residual, pw, epsilon, w->post_norm, rows, 2048)) return -6;
    if (project(context, 4, w->post_norm, rows, 2048, 5632, w->gate)
        || project(context, 5, w->post_norm, rows, 2048, 5632, w->up)) return -7;
    if (smol_host_silu_mul_bf16(w->gate, w->up, w->activated, rows * 5632)) return -8;
    if (project(context, 6, w->activated, rows, 5632, 2048, w->down)) return -9;
    return smol_host_residual_bf16(w->residual, w->down, output, rows * 2048) ? -10 : 0;
}
