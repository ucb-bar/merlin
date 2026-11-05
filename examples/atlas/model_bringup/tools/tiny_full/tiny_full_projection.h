#ifndef TINY_FULL_PROJECTION_H
#define TINY_FULL_PROJECTION_H
#include "tiny_projection.h"
#include "tiny_attention.h"
#include "tiny_full_model.h"

/* Companion metadata preserves the archived 120-byte tiny_panel ABI.
 * Output extent includes FP8 strided holes, while panel.store_bytes counts DMA.
 * Operand offsets are zero for legacy and gate panels; prefill device residual
 * o/down panels stage their residual and RoPE q/k panels the rotary tables there. */
typedef struct {
  size_t output_extent_bytes, operand_offset;
  int output_scale;
  unsigned role;
} tiny_fusion_panel;

/* A device-produced attention activation is usable by one matching o_proj. */
typedef struct {
  const uint8_t *data;
  size_t bytes, step, layer, producer_index;
  int exponent;
  unsigned valid, consumed;
} tiny_attention_fp8_identity;

typedef struct {
  size_t version, step, layer, rows, physical_rows, output_offset, total_bytes;
  size_t mask_offset, weight_offset, load_bytes, store_bytes;
  const uint32_t *program;
  size_t program_words;
  const uint8_t *mask, *weight, *expected;
  int exponent;
  unsigned flags;
} tiny_norm_panel;
typedef int (*tiny_norm_transport)(void *, const tiny_norm_panel *, uint8_t *, size_t);
/* With TINY_FULL_CAP_PRE_ATTENTION_NORM each layer and step has two norm panels:
 * the input_layernorm panel (this flag) for q/k/v, then the post_attention_layernorm panel. */
#define TINY_NORM_PRE_ATTENTION 1u
typedef struct {
  const uint16_t *input;
  size_t rows, step, layer, producer_index, consumer_call;
  int exponent;
  unsigned stage;
} tiny_norm_fp8_identity;

/* Read-only cos, sin, and -sin BF16 tiles for the four prefill row tiles,
 * followed, with DECODE_ROPE, by one row tile for each decode step. */
#define TINY_FULL_ROPE_TABLE_BYTES 24576
#define TINY_FULL_DECODE_ROPE_TABLE_BYTES 6144

typedef struct {
  tiny_projection_context panels;
  size_t calls, step_count;
  const tiny_attention_panel *attention_panels;
  size_t attention_count, attention_cursor;
  tiny_attention_transport attention_transport;
  void *attention_transport_context;
  /* Shared normalized tensors are packed once across q/k/v and gate/up. */
  uint8_t packed_a[128 * 5632];
  const uint16_t *packed_input;
  size_t packed_rows, packed_k, packed_call;
  int packed_scale;
  unsigned packed_role, packed_valid;
  const tiny_fusion_panel *fusions;
  const tiny_attention_v2_panel *attention_v2;
  tiny_attention_fp8_identity pending_attention;
  const tiny_norm_panel *norm_panels;
  size_t norm_count, norm_cursor, layer_count;
  tiny_norm_transport norm_transport;
  void *norm_transport_context;
  uint64_t capabilities;
  unsigned configured;
  tiny_norm_fp8_identity pending_norm;
  const uint8_t *rope_tables;
} tiny_full_projection_context;

int tiny_full_projection_configure(tiny_full_projection_context *, uint64_t capabilities);
void tiny_full_projection_invalidate(tiny_full_projection_context *);
int tiny_full_normalize(void *, unsigned layer, const uint16_t *, size_t rows);

int tiny_full_project(void *, unsigned layer, unsigned logical,
                      const uint16_t *, size_t rows, size_t k, size_t width, uint16_t *);
int tiny_full_fused_project(void *, unsigned layer, unsigned logical,
                            const uint16_t *, size_t rows, size_t k, size_t width,
                            const uint16_t *operand, uint16_t *);
int tiny_full_attend(void *, unsigned layer, const uint16_t *, const uint16_t *, const uint16_t *,
                     size_t rows, size_t keys, uint16_t *);
/* Overlay only device-written output spans into a saved whole-arena image. */
int tiny_full_attention_overlay_expected(const tiny_attention_panel *, const tiny_attention_v2_panel *,
                                         const uint8_t *expected, uint8_t *before, size_t bytes);
#endif
