#ifndef TINY_ATTENTION_H
#define TINY_ATTENTION_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t step, layer, query_rows, keys, query_heads, kv_heads;
    int32_t q_exponent, k_exponent, v_exponent, p_exponent;
    const uint32_t *program;
    uint64_t program_words, q_offset, k_offset, v_offset, descriptor_offset, descriptor_bytes;
    uint64_t output_offset, output_bytes, total_bytes, load_bytes, store_bytes;
    const uint8_t *initial, *expected;
} tiny_attention_panel;
/* V2 metadata is parallel to the unchanged legacy panel table. Offsets in
 * written_spans are relative to output_offset, not arena addresses. */
typedef struct {
    uint64_t offset, bytes;
} tiny_attention_span;
enum { TINY_ATTENTION_V2 = 2, TINY_ATTENTION_Q_NORMAL_BF16 = 1,
       TINY_ATTENTION_Q_PACKED_BF16 = 2, TINY_ATTENTION_OUTPUT_FP8 = 2 };
typedef struct {
    uint64_t version, q_layout, output_dtype, q_bytes, kv_bytes;
    uint64_t key_count_offset, key_count_bytes, mask_offset, mask_bytes, output_extent_bytes;
    int32_t o_exponent;
    uint32_t packed_decode;
    const tiny_attention_span *written_spans;
    uint64_t written_span_count;
} tiny_attention_v2_panel;
typedef int (*tiny_attention_transport)(void *, const tiny_attention_panel *, uint8_t *, size_t);

/* Native projection BF16 tiles, with left/right 16-column halves per tile. */
size_t tiny_attention_index(size_t row, size_t column, size_t width);
int tiny_attention_pack(const uint16_t *, size_t rows, size_t width, int scale, uint8_t *, size_t capacity);
int tiny_attention_prepare(size_t rows, size_t past_rows, uint16_t *query,
    const uint16_t *key, const uint16_t *value, const uint16_t *past_keys, const uint16_t *past_values,
    const uint16_t *cosine, const uint16_t *sine, uint16_t *keys, uint16_t *values);
/* Prefill with q/k rotated on Atlas: fills empty caches from k/v tiles and
 * fails on any non-finite q/k word, exactly where the host rotation fails. */
int tiny_attention_prepare_rotated(size_t rows, const uint16_t *query, const uint16_t *key, const uint16_t *value,
                                   uint16_t *keys, uint16_t *values);
/* Append already rotated decode k and unmodified v to a prior native cache.
 * Validate the logical q/k row; physical padding is unused by the append. */
int tiny_attention_append_rotated(size_t past_rows, const uint16_t *query, const uint16_t *key,
    const uint16_t *value, const uint16_t *past_keys, const uint16_t *past_values,
    uint16_t *keys, uint16_t *values);
#endif
