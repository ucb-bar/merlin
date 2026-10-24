#ifndef SMOL_POLICY_BLOCK_H
#define SMOL_POLICY_BLOCK_H
#include <stddef.h>
#include <stdint.h>

enum smol_policy_block_kind { SMOL_PREFIX, SMOL_EXPERT_SELF, SMOL_EXPERT_CROSS };
enum smol_policy_dtype { SMOL_BF16, SMOL_F32 };
enum smol_policy_role { SMOL_Q, SMOL_K, SMOL_V, SMOL_O, SMOL_GATE, SMOL_UP, SMOL_DOWN };
typedef int (*smol_policy_project)(void *, unsigned role, unsigned input_dtype,
    const void *, size_t rows, size_t k, unsigned output_dtype, void *, size_t n);

typedef struct {
    unsigned kind;
    const uint16_t *input_weight, *post_weight;
    float input_epsilon, post_epsilon;
    const float *cosine, *sine;
    int64_t first_table_position;
    size_t table_rows;
} smol_policy_block_parameters;

/* Caller-owned disjoint buffers. Ordinary BF16 matrices cover177*960 words,
 * gate/up/activated cover177*2560. Joined K/V cover227*320. Float score and
 * probability buffers cover15*177*227, softmax workspace covers227 floats.
 * Cross FP32 K/V buffers and input conversion cover177*320 floats. */
typedef struct {
    uint16_t *normalized, *query, *key, *value, *rotary_query, *rotary_key;
    uint16_t *joined_key, *joined_value, *attention, *projected, *residual;
    uint16_t *postnorm, *gate, *up, *activated, *down, *output;
    uint16_t *bf16_probabilities;
    float *scores, *float_probabilities, *workspace;
    float *cross_input, *cross_key, *cross_value;
} smol_policy_block_frame;

typedef struct { unsigned started, projections, complete, error; } smol_policy_block_state;

/* Prefix cache pointers are output for SMOL_PREFIX and immutable input for
 * either expert kind. Positions are source-native: prefix pad cumsum minus1,
 * expert self141..190, expert cross0..49. Mask is row-major rows*keys. */
int smol_policy_block(const void *input, unsigned input_dtype,
    const int64_t *positions, const uint8_t *mask,
    uint16_t *prefix_key, uint16_t *prefix_value,
    const smol_policy_block_parameters *, const smol_policy_block_frame *,
    smol_policy_block_state *, smol_policy_project, void *);
#endif
