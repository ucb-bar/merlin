#ifndef SMOL_POLICY_RAW_H
#define SMOL_POLICY_RAW_H
#include "atlas_host.h"
#include "smol_policy_graph.h"
#include "smol_vision_graph.h"
#include "smol_attention.h"

typedef struct {
    const uint32_t *program;
    const uint8_t *weights;
    size_t program_words, rescale_word, physical_rows, physical_k, physical_n;
    size_t row_start, logical_rows, input_width, output_width, call;
    size_t output_offset, dram_bytes, weight_bytes, load_bytes, store_bytes;
    int activation_scale, weight_scale;
} smol_policy_panel;

typedef struct {
    size_t rows, input_width, output_width, first_panel, panel_count;
    const void *bias;
    unsigned site, layer, role, image, input_dtype, output_dtype, bias_dtype;
    int activation_scale;
} smol_policy_call;

/* BF16 operands and output, head dimension 64, ordinary (nonpacked) query tiles.
 * metadata is exactly descriptors, padded key counts, and deduplicated masks. */
typedef struct {
    const uint32_t *program;
    const uint8_t *metadata, *expected_mask;
    size_t program_words, metadata_bytes;
    size_t descriptor_offset, descriptor_bytes, key_count_offset, key_count_bytes, mask_offset, mask_bytes;
    size_t before_call, phase, index;
    unsigned kind;
    size_t rows, keys, query_heads, kv_heads;
    size_t q_offset, k_offset, v_offset, output_offset, output_bytes, dram_bytes, load_bytes, store_bytes;
    int q_exponent, k_exponent, v_exponent;
} smol_attention_panel;

typedef struct {
    atlas_host *host;
    const smol_policy_panel *panels;
    const smol_policy_call *calls;
    size_t panel_count, call_count, panel_cursor, call_cursor;
    uint8_t *dram, *activation_copy;
    uint16_t *input_copy;
    size_t dram_capacity, activation_capacity, input_capacity, max_polls;
    uint32_t *cycle_delta;
    atlas_host_buffer active_buffer;
    unsigned image, failed;
    int result;
    const smol_attention_panel *attention_panels;
    size_t attention_count, attention_cursor;
#ifdef ATLAS_FAST_MODE
    /* Retain both ownership descriptors even when a fault leaves hardware active. */
    atlas_host_buffer fast_buffers[2];
#endif
} smol_policy_raw_context;

enum { SMOL_RAW_OK, SMOL_RAW_INVALID, SMOL_RAW_SCALE, SMOL_RAW_RUNTIME,
       SMOL_RAW_GUARD, SMOL_RAW_INPUT, SMOL_RAW_OUTPUT };

int smol_policy_raw_graph(void *, unsigned site, unsigned layer, unsigned role,
    unsigned input_dtype, const void *, size_t rows, size_t k, unsigned output_dtype,
    void *, size_t n);
int smol_policy_raw_vision(void *, unsigned layer, unsigned role, const uint16_t *,
    size_t rows, size_t k, uint16_t *, size_t n);
int smol_policy_raw_attention(void *, unsigned kind, const uint16_t *, const uint16_t *,
    const uint16_t *, const uint8_t *, size_t rows, size_t keys,
    size_t query_heads, size_t kv_heads, uint16_t *);
#endif
