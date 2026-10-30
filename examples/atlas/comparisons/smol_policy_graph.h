#ifndef SMOL_POLICY_GRAPH_H
#define SMOL_POLICY_GRAPH_H
#include "smol_policy_block.h"

enum smol_policy_site { SMOL_STATE_PROJECT, SMOL_ACTION_IN, SMOL_TIME_IN,
    SMOL_TIME_OUT, SMOL_ACTION_OUT, SMOL_PREFIX_LAYER, SMOL_EXPERT_LAYER };
typedef int (*smol_policy_graph_project)(void *, unsigned site, unsigned layer,
    unsigned role, unsigned input_dtype, const void *, size_t rows, size_t k,
    unsigned output_dtype, void *, size_t n);

typedef struct {
    smol_policy_block_parameters prefix[16], expert[16];
    const uint16_t *prefix_final_weight, *expert_final_weight;
    /* Raw learned BF16 embeddings for the fixed48 observation tokens. */
    const uint16_t *language_embedding;
    float language_scale, final_epsilon;
    /* Source function tables for the fixed ten-step timestep domain. These
     * contain only sinusoidal function values, never learned activations. */
    const float *timestep, *time_embedding;
} smol_policy_parameters;

typedef struct {
    smol_policy_block_frame block;
    float *prefix_input, *state_projection;
    uint16_t *prefix_final, *expert_final;
    uint16_t *cache_key, *cache_value;
    uint8_t *prefix_mask, *self_mask, *cross_mask;
    int64_t *groups, *positions, *cross_positions;
    float *action_embedding, *action_time, *time_hidden, *time_activated;
    float *suffix_input, *velocity;
} smol_policy_frame;

typedef struct { unsigned prefix_ready, steps, projections, error; } smol_policy_state;

/* Each image operand is the actual scaled64x960 output of smol_vision_image.
 * This function retains the source FP32 cat promotion from the state token. */
int smol_policy_prefix(const uint16_t *image0, const uint16_t *image1,
    const float *state, const uint8_t *pad, const uint8_t *attention,
    const smol_policy_parameters *, const smol_policy_frame *, smol_policy_state *,
    smol_policy_graph_project, void *);

/* Execute one real timestep MLP,16-layer expert, final norm/projection and Euler
 * update. Calls require the source timestep bits and monotonically ordered step.
 * The caller keeps all ten velocities and states if required for final checks. */
int smol_policy_step(const float *input_state, float timestep, float *next_state,
    const uint8_t *prefix_pad, const smol_policy_parameters *, const smol_policy_frame *,
    smol_policy_state *, smol_policy_graph_project, void *);
/* Complete semantic layer phases. Initialization is folded into layer0 and
 * final normalization/head/Euler into layer15. Counts reject skipped/repeated
 * layers; masks and positions regenerate from fixed pad/attention on each call.
 * Retain block.output, all prefix KV, current Euler input and published history. */
int smol_policy_prefix_layer(const uint16_t *, const uint16_t *, const float *,
    const uint8_t *, const uint8_t *, const smol_policy_parameters *, const smol_policy_frame *,
    smol_policy_state *, smol_policy_graph_project, void *, unsigned layer);
int smol_policy_step_layer(const float *, float, float *, const uint8_t *,
    const smol_policy_parameters *, const smol_policy_frame *, smol_policy_state *,
    smol_policy_graph_project, void *, unsigned layer);
#endif
