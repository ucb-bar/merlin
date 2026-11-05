#ifndef SMOL_VISION_GRAPH_H
#define SMOL_VISION_GRAPH_H
#include <stddef.h>
#include <stdint.h>

enum smol_vision_role {
    SMOL_VISION_PATCH, SMOL_VISION_Q, SMOL_VISION_K, SMOL_VISION_V,
    SMOL_VISION_O, SMOL_VISION_FC1, SMOL_VISION_FC2, SMOL_VISION_CONNECTOR
};

/* Projection inputs/outputs use row-major matrix layout, including flattened
 * CR/S patch operands. The callback owns projection, native bias and rounding.
 * It must preserve one activation scale across all panels of the invocation. */
typedef int (*smol_vision_project)(void *, unsigned layer, unsigned role,
    const uint16_t *, size_t rows, size_t k, uint16_t *, size_t n);

typedef struct {
    const uint16_t *norm1_weight, *norm1_bias, *norm2_weight, *norm2_bias;
} smol_vision_layer;

typedef struct {
    /* Position rows already indexed by the fixed full32x32 patch geometry. */
    const uint16_t *position;
    smol_vision_layer layers[12];
    const uint16_t *post_weight, *post_bias, *gelu;
    float epsilon;
} smol_vision_parameters;

typedef struct {
    /* Each small buffer contains1024*768 words; large contains1024*3072. */
    uint16_t *tokens, *normalized, *query, *key, *value, *attended, *projected;
    uint16_t *large, *activated;
    /* Shuffle contains64*12288 words; output contains64*960 words. */
    uint16_t *shuffle, *output;
} smol_vision_frame;

typedef struct { unsigned started, projections, complete, error; } smol_vision_state;

/* Fixed observation geometry: one BF16 CHW3x512x512 image. Attention is genuine
 * CPU flash arithmetic from smol_vision_attention, with full zero additive mask. */
int smol_vision_image(const uint16_t *, const smol_vision_parameters *,
    const smol_vision_frame *, smol_vision_state *, smol_vision_project, void *);

/* One complete layer, folding image initialization into layer0 and postnorm,
 * shuffle, connector and scaling into layer11. Projection counts enforce order.
 * Only tokens and published output remain live between successful phases. */
int smol_vision_image_layer(const uint16_t *, const smol_vision_parameters *,
    const smol_vision_frame *, smol_vision_state *, smol_vision_project, void *, unsigned layer);

/* Source pixel_shuffle(scale_factor4), kept public for focused layout controls. */
int smol_vision_shuffle4(const uint16_t *, uint16_t *);
int smol_vision_patch16(const uint16_t *, uint16_t *);
#endif
