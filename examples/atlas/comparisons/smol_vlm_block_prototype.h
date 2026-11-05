#ifndef SMOL_VLM_BLOCK_PROTOTYPE_H
#define SMOL_VLM_BLOCK_PROTOTYPE_H

#include <stddef.h>
#include <stdint.h>

enum { SMOL_BLOCK_ROWS=177, SMOL_BLOCK_WIDTH=960, SMOL_BLOCK_MLP=2560,
       SMOL_BLOCK_HEADS=15, SMOL_BLOCK_KV_HEADS=5, SMOL_BLOCK_HEAD_DIM=64 };
enum { SMOL_BLOCK_Q, SMOL_BLOCK_K, SMOL_BLOCK_V, SMOL_BLOCK_O,
       SMOL_BLOCK_GATE, SMOL_BLOCK_UP, SMOL_BLOCK_DOWN, SMOL_BLOCK_LINEARS };

/* Row-major BF16 native rows only; callback owns padding, packing, weights,
 * dispatch and output unpacking. All callback arguments remain owned until
 * success. A runtime failure can retain device ownership: the caller must
 * establish external quiescence before changing/reusing frame or context.
 * The callback implementation determines whether this is diagnostic output
 * injection or actual device execution; the composition makes no such claim.
 */
typedef int (*smol_block_linear)(void *context, unsigned role,
                                 const uint16_t *input, size_t rows,
                                 size_t input_width, uint16_t *output,
                                 size_t output_width);

typedef struct {
    const uint16_t *input_norm_weight, *post_norm_weight;
    float input_epsilon, post_epsilon;
    const float *cosine, *sine;
    int64_t first_table_position;
    size_t table_rows;
} smol_block_parameters;

/* Every buffer is caller-sized and disjoint. BF16 matrices use 177 native
 * rows: normalized/query/rotary_query/attention/projected/residual/postnorm/
 * down/output width 960; key/value/rotary_key width 320; gate/up/mlp_input
 * width 2560. Scores/probabilities are [15,177,177], mask [177,177], groups and
 * positions [177], attention_workspace [177]. Rotary_key/value remain resident
 * prefix cache storage after a successful invocation, until caller releases it.
 */
typedef struct {
    uint16_t *normalized, *query, *key, *value, *rotary_query, *rotary_key;
    uint16_t *attention, *projected, *residual, *postnorm;
    uint16_t *gate, *up, *mlp_input, *down, *output;
    float *scores, *attention_workspace;
    uint16_t *probabilities;
    uint8_t *mask;
    int64_t *groups, *positions;
} smol_block_frame;

typedef struct {
    unsigned started, completed_linears, last_role, failed, complete, cache_valid;
    int callback_code, result;
} smol_block_state;

enum { SMOL_BLOCK_OK, SMOL_BLOCK_INVALID, SMOL_BLOCK_USED_STATE,
       SMOL_BLOCK_ARITHMETIC, SMOL_BLOCK_CALLBACK };

/* Fixed selected VLM block BF16-input composition; no model loading, other
 * layers, time/state/action graph, RNG, raw runtime binding or host ELF.
 * State must initially be all-zero at known quiescence. It is single-use and
 * failure-latching; there is deliberately no automatic reset/retry operation.
 */
int smol_vlm_block_prototype(const uint16_t *input, const uint8_t *pad,
                             const uint8_t *attention_boundaries,
                             const smol_block_parameters *parameters,
                             const smol_block_frame *frame,
                             smol_block_state *state,
                             smol_block_linear linear, void *context);

#endif
