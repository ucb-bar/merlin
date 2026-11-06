#ifndef TINY_FULL_MODEL_H
#define TINY_FULL_MODEL_H
#include <stddef.h>
#include <stdint.h>

#define TINY_FULL_LAYERS 22
#define TINY_FULL_CACHE_WORDS (4 * 160 * 64)
#define TINY_FULL_CAP_POST_O_NORM (UINT64_C(1) << 0)
/* Prefill o/down add their residual operand on Atlas. RoPE capabilities
 * select q/k projections that return rotated tiles for each phase. */
#define TINY_FULL_CAP_PREFILL_RESIDUAL (UINT64_C(1) << 1)
#define TINY_FULL_CAP_PREFILL_ROPE (UINT64_C(1) << 2)
/* The same normalize callback also runs the pre-attention norm, before q, for q/k/v FP8. */
#define TINY_FULL_CAP_PRE_ATTENTION_NORM (UINT64_C(1) << 3)
/* Decode RoPE requires PREFILL_ROPE and tables for positions 128 and 129. */
#define TINY_FULL_CAP_DECODE_ROPE (UINT64_C(1) << 4)
#define TINY_FULL_DEVICE_CAPABILITIES (TINY_FULL_CAP_POST_O_NORM | TINY_FULL_CAP_PREFILL_RESIDUAL | TINY_FULL_CAP_PREFILL_ROPE \
                                       | TINY_FULL_CAP_PRE_ATTENTION_NORM | TINY_FULL_CAP_DECODE_ROPE)
/* Host-only layout choice; device programs and generated capability metadata stay unchanged.
 * Requires POST_O_NORM and PREFILL_RESIDUAL. Hidden, o, and down BF16 buffers
 * retain complete native tiles, with zero padding for decode. */
#define TINY_FULL_CAP_TILED_HIDDEN (UINT64_C(1) << 32)
/* LM-head outputs retain complete native BF16 tiles, including decode padding.
 * The logits allocation must hold 128x32000 prefill or 32x32000 decode words. */
#define TINY_FULL_CAP_TILED_LOGITS (UINT64_C(1) << 33)
#define TINY_FULL_CAPABILITIES (TINY_FULL_DEVICE_CAPABILITIES | TINY_FULL_CAP_TILED_HIDDEN | TINY_FULL_CAP_TILED_LOGITS)

typedef int (*tiny_full_normalization)(void *, unsigned, const uint16_t *, size_t);

typedef int (*tiny_full_attention)(void *, unsigned, const uint16_t *, const uint16_t *,
                                   const uint16_t *, size_t, size_t, uint16_t *);

/* Fused roles 3..6 consume an optional residual or activated-gate operand.
 * Role 5 writes natural FP8 tiles; role 6 consumes those bytes directly. */
typedef int (*tiny_full_fused_projection)(void *, unsigned, unsigned, const uint16_t *,
                                         size_t, size_t, size_t, const uint16_t *, uint16_t *);

/* All allocation extents are caller-owned and mutually disjoint. Atlas
 * attention uses native BF16 projection tiles in fixed 160-row KV slots.
 * The legacy host attention path uses compact [4,position,64] KV entries.
 * Double buffering preserves the last complete state if a step fails. */
typedef struct {
    const uint16_t *embedding;
    const uint16_t *input_norm[TINY_FULL_LAYERS];
    const uint16_t *post_norm[TINY_FULL_LAYERS];
    const uint16_t *final_norm, *cosine, *sine;
    float epsilon;
    tiny_full_attention attention;
    tiny_full_fused_projection fused_project;
    tiny_full_normalization normalize;
    uint64_t capabilities;
} tiny_full_parameters;

typedef struct {
    uint16_t *keys[2], *values[2];
    size_t position;
    unsigned bank;
    uint32_t next_token;
} tiny_full_state;

/* Layers 0..21 use logical projections 0..6 in the original block order.
 * Layer22/logical7 is the complete LM head, width32000. Callback returns
 * every BF16 word produced by the actual preceding host stage. Atlas attention
 * uses tiled Q/K/V projection outputs and a tiled o_proj input. */
typedef int (*tiny_full_projection)(void *, unsigned, unsigned,
                                    const uint16_t *, size_t, size_t, size_t, uint16_t *);
/* Pointer-free continuation ABI. active is 0 before begin/after commit,
 * 1 during a step, and 2 after a failed arithmetic operation. */
typedef struct {
    uint64_t position;
    uint32_t bank, next_token, next_layer, rows, active, reserved;
} tiny_full_phase;

/* Offsets come from the compiled workspace type, never guessed by a runner. */
typedef struct {
    uint64_t version, workspace_bytes, hidden_offset[2], hidden_bytes;
    uint64_t cache_bank_bytes, phase_bytes, state_bytes;
} tiny_full_live_layout;
extern const tiny_full_live_layout tiny_full_compiled_live_layout;
int tiny_full_get_live_layout(tiny_full_live_layout *);
int tiny_full_model_begin(const tiny_full_parameters *, const tiny_full_state *,
                         tiny_full_phase *, const uint32_t *, size_t, void *, size_t);
int tiny_full_model_layer(const tiny_full_parameters *, const tiny_full_state *, tiny_full_phase *, unsigned,
                         tiny_full_projection, void *, void *, size_t);
int tiny_full_model_finish(const tiny_full_parameters *, tiny_full_state *, tiny_full_phase *,
                          tiny_full_projection, void *, void *, size_t, uint16_t *, size_t);
size_t tiny_full_workspace_bytes(void);
int tiny_full_argmax_bf16(const uint16_t *logits, size_t count, uint32_t *token);
/* Scan one logical row without reading other logical rows or physical padding. */
int tiny_full_argmax_tiled_bf16(const uint16_t *logits, size_t row, size_t count, uint32_t *token);
int tiny_full_model_step(const tiny_full_parameters *, tiny_full_state *,
                         const uint32_t *tokens, size_t rows,
                         tiny_full_projection, void *, void *workspace, size_t workspace_bytes,
                         uint16_t *logits, size_t logits_words);
#endif
