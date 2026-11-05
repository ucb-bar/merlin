#include "atlas_host_layout.h"
#include "tiny_full_projection.h"
#include "tiny_full_tiles.h"
#ifndef TINY_FAST_HOST_LEVEL
#define TINY_FAST_HOST_LEVEL 2
#endif
#if ATLAS_FAST_MODE
#include "fast-assets.h"
#endif
/* Packing callbacks are provided by the embedding workload library. */
int host_matrix_scale_bf16(const uint16_t *, size_t, size_t, int *);
int host_matrix_pack_bf16_at_scale(const uint16_t *, size_t, size_t, size_t, size_t, size_t, size_t, int, uint8_t *, size_t);

#include <string.h>

static int prepare_packed(tiny_full_projection_context *full, int native_attention, unsigned logical,
                          const uint16_t *input, size_t rows, size_t k) {
  size_t m = rows == 128 ? 128 : 32;
  int shared = logical == 1 || logical == 2 || logical == 5;
  unsigned producer = logical == 5 ? 4 : 0;
  if (shared) {
    if (!full->packed_valid || full->packed_input != input || full->packed_rows != rows || full->packed_k != k
        || full->packed_role != producer || full->packed_call + logical - producer != full->calls) return -2;
  } else {
    full->packed_valid = 0;
    if (host_matrix_scale_bf16(input, native_attention && logical == 3 ? m : rows, k,
        &full->packed_scale)) return -2;
    if (native_attention && logical == 3) {
      if (tiny_attention_pack(input, rows, k, full->packed_scale, full->packed_a, m * k)) return -4;
    } else if (host_matrix_pack_bf16_at_scale(input, rows, k, m, k, 32, 32, full->packed_scale,
                                             full->packed_a, m * k)) return -4;
    full->packed_input = input;
    full->packed_rows = rows;
    full->packed_k = k;
    full->packed_call = full->calls;
    full->packed_role = logical;
    full->packed_valid = 1;
  }
  return 0;
}

_Static_assert(sizeof(tiny_panel) == 120, "Projection panel ABI size");

_Static_assert(sizeof(tiny_norm_panel) == 136, "Norm panel ABI size");

void tiny_full_projection_invalidate(tiny_full_projection_context *c) {
  if (!c) return;
  c->packed_valid = 0;
  c->pending_norm.stage = 0;
  c->pending_attention.valid = 0;
}

static size_t norm_sites(uint64_t capabilities) {
  return capabilities & TINY_FULL_CAP_PRE_ATTENTION_NORM ? 2 : 1;
}

static int valid_norm_panel(const tiny_full_projection_context *c, const tiny_norm_panel *p, size_t index, size_t sites) {
  size_t site = index / sites, rows = site < c->layer_count ? 128 : 1, m = rows == 128 ? 128 : 32;
  unsigned flags = sites == 2 && index % 2 == 0 ? TINY_NORM_PRE_ATTENTION : 0;
  return p && p->version == 1 && p->flags == flags && p->step == site / c->layer_count && p->layer == site % c->layer_count
      && p->rows == rows && p->physical_rows == m && p->output_offset == 2 * m * 2048
      && p->store_bytes == m * 2048 && p->mask_offset == 3 * m * 2048 + 256
      && p->weight_offset == p->mask_offset + 65536 && p->total_bytes == p->weight_offset + 131072 + 256
      && p->total_bytes <= c->panels.dram_capacity && p->load_bytes == 2 * m * 2048 + 196608
      && p->exponent >= -126 && p->exponent <= 119 && p->program && p->program_words
      && p->program_words <= ATLAS_IMEM_BYTES / 4 - 5 && p->mask && p->weight && p->expected;
}

int tiny_full_projection_configure(tiny_full_projection_context *c, uint64_t capabilities) {
  uint64_t device_capabilities = capabilities & TINY_FULL_DEVICE_CAPABILITIES;
  if (!c || (capabilities & ~TINY_FULL_CAPABILITIES) || c->calls || c->panels.cursor
      || c->attention_cursor || c->norm_cursor || (device_capabilities && !(capabilities & TINY_FULL_CAP_POST_O_NORM))
      || !(capabilities & TINY_FULL_CAP_PREFILL_ROPE) != !c->rope_tables) return -1;
  if ((capabilities & TINY_FULL_CAP_DECODE_ROPE)
      && (!(capabilities & TINY_FULL_CAP_PREFILL_ROPE) || c->step_count > 3)) return -1;
  if ((capabilities & TINY_FULL_CAP_TILED_HIDDEN)
      && (capabilities & (TINY_FULL_CAP_POST_O_NORM | TINY_FULL_CAP_PREFILL_RESIDUAL))
          != (TINY_FULL_CAP_POST_O_NORM | TINY_FULL_CAP_PREFILL_RESIDUAL)) return -1;
  if (!device_capabilities) {
    if (c->norm_panels || c->norm_count || c->norm_transport) return -1;
  } else {
    if (c->step_count < 1 || c->step_count > 3 || !c->layer_count || c->layer_count > 22
        || (c->layer_count < 22 && c->step_count != 1) || !c->norm_panels
        || c->norm_count != norm_sites(capabilities) * c->layer_count * c->step_count || !c->norm_transport
        || !c->attention_v2 || !c->attention_panels || !c->fusions || !c->panels.panels || !c->panels.panel_count
        || !c->panels.dram || !c->panels.transport) return -1;
    size_t sites = norm_sites(capabilities), norm_index = 0;
    size_t calls = c->layer_count == 22 ? 155 * c->step_count : 7 * c->layer_count;
    for (size_t call = 0, i = 0; i < c->panels.panel_count; ++call) {
      if (call >= calls) return -1;
      unsigned role = call % 155 == 154 ? 7 : (unsigned)(call % 155 % 7);
      size_t width = c->panels.panels[i].width, column = 0;
      /* Norm panels for this layer and step: input_layernorm (with the capability), then post-o. */
      size_t first = role == 7 ? 0 : (call / 155 * c->layer_count + call % 155 / 7) * sites;
      if (role != 7 && (first + sites > c->norm_count
          || (role == 0 && sites == 2 && !valid_norm_panel(c, c->norm_panels + first, first, sites))
          || (role == 4 && !valid_norm_panel(c, c->norm_panels + first + sites - 1, first + sites - 1, sites)))) return -1;
      const tiny_norm_panel *pre = c->norm_panels + first, *post = pre + sites - 1;
      do {
        const tiny_panel *p = c->panels.panels + i;
        if (p->logical != role || p->column != column || !p->n || p->n > width - column) return -1;
        if ((role == 4 || role == 5) && (p->a_scale != post->exponent || p->k != 2048 || p->width != 5632)) return -1;
        if (sites == 2 && role <= 2 && (p->a_scale != pre->exponent || p->k != 2048)) return -1;
        /* Device residual and RoPE operands exist exactly at their enabled sites. */
        const tiny_fusion_panel *f = c->fusions + i;
        size_t operand = 0;
        if (call < 155 && (role == 3 || role == 6) && (capabilities & TINY_FULL_CAP_PREFILL_RESIDUAL))
          operand = p->output_offset + f->output_extent_bytes + 256;
        if (role <= 1 && (capabilities & (call < 155 ? TINY_FULL_CAP_PREFILL_ROPE : TINY_FULL_CAP_DECODE_ROPE)))
          operand = p->output_offset + p->store_bytes + 256;
        if (role != 5 && f->operand_offset != operand) return -1;
        column += p->n;
        ++i;
        if (column < width && i == c->panels.panel_count) return -1;
      } while (column < width);
      if (role == 5) norm_index += sites;
      if (i == c->panels.panel_count && (call + 1 != calls || norm_index != c->norm_count)) return -1;
    }
  }
  c->capabilities = capabilities;
  c->configured = 1;
  tiny_full_projection_invalidate(c);
  return 0;
}

/* The call position selects the site: before q is the input_layernorm, before gate is post-o. */
int tiny_full_normalize(void *opaque, unsigned layer, const uint16_t *input, size_t rows) {
  tiny_full_projection_context *c = opaque;
  size_t sites = c ? norm_sites(c->capabilities) : 1;
  int pre = c && sites == 2 && c->calls % 155 == layer * 7;
  if (!c || !c->configured || !(c->capabilities & TINY_FULL_CAP_POST_O_NORM) || !input
      || layer >= 22 || (rows != 128 && rows != 1) || c->pending_norm.stage
      || !c->norm_panels || c->norm_cursor >= c->norm_count || (!pre && c->calls % 155 != layer * 7 + 4)
      || c->norm_cursor != (c->calls / 155 * c->layer_count + layer) * sites + (pre ? 0 : sites - 1)
      || !c->panels.dram || !c->norm_transport) {
    tiny_full_projection_invalidate(c);
    return -1;
  }
  const tiny_norm_panel *p = c->norm_panels + c->norm_cursor;
  if (!valid_norm_panel(c, p, c->norm_cursor, sites) || p->rows != rows) {
    tiny_full_projection_invalidate(c);
    return -2;
  }
  uint8_t *dram = c->panels.dram;
  if (c->capabilities & TINY_FULL_CAP_TILED_HIDDEN) {
    /* The embedding and decode residual producers clear every padded row. */
    memcpy(dram, input, p->output_offset);
  } else {
    memset(dram, 0, p->output_offset);
    for (size_t r = 0; r < rows; ++r)
      for (size_t col = 0; col < 2048; ++col) {
        size_t at = 2 * tiny_full_bf16_index(r, col, 2048);
        uint16_t value = input[r * 2048 + col];
        dram[at] = (uint8_t)value;
        dram[at + 1] = (uint8_t)(value >> 8);
      }
  }
  memset(dram + p->output_offset, 0xcd, p->store_bytes);
  memset(dram + p->output_offset + p->store_bytes, 0x5a, 256);
  memcpy(dram + p->mask_offset, p->mask, 65536);
  memcpy(dram + p->weight_offset, p->weight, 131072);
  memset(dram + p->weight_offset + 131072, 0xa5, 256);
  if (c->norm_transport(c->norm_transport_context, p, dram, p->total_bytes)) {
    tiny_full_projection_invalidate(c);
    return -3;
  }
  memcpy(c->packed_a, dram + p->output_offset, p->store_bytes);
  c->packed_valid = 0;
  c->pending_norm = (tiny_norm_fp8_identity){input, rows, p->step, layer, c->norm_cursor, c->calls, p->exponent, 1};
  ++c->norm_cursor;
  return 0;
}

static int fused_panels(tiny_full_projection_context *full, unsigned logical, const uint16_t *input,
                        size_t rows, size_t k, size_t width, const uint16_t *operand, uint16_t *output) {
  tiny_projection_context *c = &full->panels;
  int residual = (logical == 3 || logical == 6) && rows == 128 && (full->capabilities & TINY_FULL_CAP_PREFILL_RESIDUAL);
  if (!full->fusions || !full->attention_panels || !c->panels || !c->dram || !c->transport || !input || !output
      || logical < 3 || logical > 6 || (logical == 5 || residual ? operand == NULL : operand != NULL)
      || c->cursor >= c->panel_count) return -1;
  size_t start = c->cursor, column = 0, m = rows == 128 ? 128 : 32;
  int scale = c->panels[start].a_scale;
  int attention_fp8 = logical == 3 && full->attention_v2;
  int norm_fp8 = (logical == 4 || logical == 5) && (full->capabilities & TINY_FULL_CAP_POST_O_NORM);
  if (attention_fp8) {
    const tiny_attention_fp8_identity *a = &full->pending_attention;
    if (!a->valid || a->consumed || a->data != (const uint8_t *)input || a->bytes != m * k
        || a->step != full->calls / 155 || a->layer != full->calls % 155 / 7
        || !full->attention_cursor || a->producer_index != full->attention_cursor - 1
        || a->producer_index >= full->attention_count || a->exponent != scale
        || full->attention_v2[a->producer_index].o_exponent != scale) return -3;
    full->packed_valid = 0;
  } else if (norm_fp8) {
    const tiny_norm_fp8_identity *a = &full->pending_norm;
    if (!full->configured || a->stage != logical - 3 || a->input != input || a->rows != rows || k != 2048
        || a->step != full->calls / 155 || a->layer != full->calls % 155 / 7
        || a->consumer_call + logical - 4 != full->calls || a->exponent != scale
        || !full->norm_cursor || a->producer_index != full->norm_cursor - 1) return -3;
  } else if (logical != 6) {
    int prepared = prepare_packed(full, 1, logical, input, rows, k);
    if (prepared) return prepared;
    scale = full->packed_scale;
  }
  if (logical == 6) {
    size_t previous = start;
    while (previous && c->panels[previous - 1].logical == 5) {
      --previous;
      if (full->fusions[previous].role != 5 || full->fusions[previous].output_scale != scale) return -3;
    }
    if (previous == start) return -3;
  }
  /* Check all shards before staging or launching any of them. */
  size_t end = start;
  while (column < width) {
    if (end >= c->panel_count) return -3;
    const tiny_panel *p = c->panels + end;
    const tiny_fusion_panel *f = full->fusions + end;
    if (p->rows != rows || p->physical_rows != m || p->k != k || p->logical != logical || p->column != column
        || p->width != width || !p->n || p->n % 32 || p->n > width - column || p->a_scale != scale
        || !p->weights || !p->program || !p->program_words || p->program_words > ATLAS_IMEM_BYTES / 4 - 5
        || p->weight_bytes != k * p->n || p->output_offset != m * k + p->weight_bytes + 256
        || f->role != logical || !p->load_bytes || p->load_bytes % 32 || p->store_bytes % 32) return -3;
    size_t extent = logical == 5 ? (m / 32 - 1) * 32 * width + (column + p->n) * 32 : m * p->n * 2;
    size_t extra = logical == 5 || residual ? m * p->n * 2 : 0;
    size_t offset = extra ? p->output_offset + extent + 256 : 0;
    if (f->output_extent_bytes != extent || f->operand_offset != offset
        || p->store_bytes != m * p->n * (logical == 5 ? 1 : 2)
        || p->total_bytes != p->output_offset + extent + 256 + extra || p->total_bytes > c->dram_capacity
        || (logical == 5 && (f->output_scale != full->fusions[start].output_scale
                            || f->output_scale < -126 || f->output_scale > 127))
        || (logical != 5 && f->output_scale)) return -3;
    column += p->n;
    ++end;
  }
  for (; c->cursor < end; ++c->cursor) {
    const tiny_panel *p = c->panels + c->cursor;
    const tiny_fusion_panel *f = full->fusions + c->cursor;
#if ATLAS_FAST_MODE
    /* A and auxiliary operands are fully staged below; B is an immutable blob. */
    memset(c->dram + p->output_offset, 0xcd, f->output_extent_bytes);
#else
    memset(c->dram, 0xcd, p->total_bytes);
#endif
    memset(c->dram + p->output_offset - 256, 0xa5, 256);
    memset(c->dram + p->output_offset + f->output_extent_bytes, 0x5a, 256);
    memcpy(c->dram, logical == 6 || attention_fp8 ? (const uint8_t *)input : full->packed_a, m * k);
#if !ATLAS_FAST_MODE
    memcpy(c->dram + m * k, p->weights, p->weight_bytes);
#endif
    if (operand) {
      uint8_t *destination = c->dram + f->operand_offset;
      if (logical == 5 || (full->capabilities & TINY_FULL_CAP_TILED_HIDDEN)) {
        for (size_t tr = 0; tr < m / 32; ++tr)
          memcpy(destination + tr * p->n * 64,
                 (const uint8_t *)operand + (tr * width + p->column) * 64, p->n * 64);
      } else {
        /* Prefill rows fill every physical row, so the BF16 residual tiles are complete. */
        for (size_t r = 0; r < rows; ++r)
          for (size_t n = 0; n < p->n; ++n) {
            uint16_t value = operand[r * width + p->column + n];
            size_t at = tiny_full_bf16_index(r, n, p->n) * 2;
            destination[at] = (uint8_t)value;
            destination[at + 1] = (uint8_t)(value >> 8);
          }
      }
    }
    if (c->transport(c->transport_context, p, c->dram, p->total_bytes)) return -5;
    if (logical == 4 || logical == 5 || (full->capabilities & TINY_FULL_CAP_TILED_HIDDEN)) {
      size_t tile_bytes = logical == 5 ? 1024 : 2048;
#if ATLAS_FAST_MODE && TINY_FAST_HOST_LEVEL >= 2
      /* Adjacent column tiles are contiguous in both layouts within each tile row. */
      for (size_t tr = 0; tr < m / 32; ++tr) {
        size_t destination = (tr * (width / 32) + p->column / 32) * tile_bytes;
        size_t source = logical == 5 ? destination : tr * (p->n / 32) * tile_bytes;
        memcpy((uint8_t *)output + destination, c->dram + p->output_offset + source, p->n / 32 * tile_bytes);
      }
#else
      for (size_t tr = 0; tr < m / 32; ++tr)
        for (size_t tc = 0; tc < p->n / 32; ++tc) {
          size_t destination = (tr * (width / 32) + p->column / 32 + tc) * tile_bytes;
          size_t source = logical == 5 ? destination : (tr * (p->n / 32) + tc) * tile_bytes;
          memcpy((uint8_t *)output + destination, c->dram + p->output_offset + source, tile_bytes);
        }
#endif
    } else {
      for (size_t r = 0; r < rows; ++r)
        for (size_t n = 0; n < p->n; ++n) {
          size_t at = p->output_offset + tiny_full_bf16_index(r, n, p->n) * 2;
          output[r * width + p->column + n] = (uint16_t)c->dram[at] | (uint16_t)((uint16_t)c->dram[at + 1] << 8);
        }
    }
  }
  if (attention_fp8) full->pending_attention.consumed = 1;
  if (norm_fp8) full->pending_norm.stage = logical == 4 ? 2 : 0;
  return 0;
}

static int project_panels(tiny_full_projection_context *full, int native_attention, unsigned logical, const uint16_t *input, size_t rows, size_t k, size_t width,
                 uint16_t *output) {
  tiny_projection_context *c = &full->panels;
  if (!c || !c->panels || !c->dram || !c->transport || !input || !output || logical > 7 || (rows != 128 && rows != 1) ||
      !k || !width || width > 32000 || k > 5632 || c->cursor >= c->panel_count)
    return -1;
  size_t start = c->cursor, column = 0;
  int rope = logical <= 1 && (full->capabilities & (rows == 128 ? TINY_FULL_CAP_PREFILL_ROPE : TINY_FULL_CAP_DECODE_ROPE));
  size_t tables = rope ? (rows == 128 ? TINY_FULL_ROPE_TABLE_BYTES : TINY_FULL_DECODE_ROPE_TABLE_BYTES) : 0;
  size_t table_offset = rows == 128 ? 0 : TINY_FULL_ROPE_TABLE_BYTES
      + (full->calls / 155 - 1) * TINY_FULL_DECODE_ROPE_TABLE_BYTES;
  /* q/k/v consume the pre-attention norm FP8 in packed_a, bound to its H handle, in order. */
  int norm_fp8 = logical <= 2 && (full->capabilities & TINY_FULL_CAP_PRE_ATTENTION_NORM);
  int scale = full->pending_norm.exponent;
  if (norm_fp8) {
    const tiny_norm_fp8_identity *a = &full->pending_norm;
    if (!full->configured || a->stage != logical + 1 || a->input != input || a->rows != rows || k != 2048
        || a->step != full->calls / 155 || a->layer != full->calls % 155 / 7 || a->consumer_call + logical != full->calls
        || !full->norm_cursor || a->producer_index != full->norm_cursor - 1) return -3;
  } else {
    int prepared = prepare_packed(full, native_attention, logical, input, rows, k);
    if (prepared) return prepared;
    scale = full->packed_scale;
  }
  /* Validate the entire logical projection before writing its first panel.
   * Programs are precompiled for this session's global input scale. A new
   * scale fails closed; this wrapper does not synthesize instructions. */
  while (column < width) {
    if (c->cursor >= c->panel_count) {
      c->cursor = start;
      return -3;
    }
    const tiny_panel *p = c->panels + c->cursor++;
    size_t m = rows == 128 ? 128 : 32;
    if (p->rows != rows || p->physical_rows != m || p->k != k || p->logical != logical || p->column != column ||
        p->width != width || !p->n || p->n % 32 || p->n > width - column || k % 32 || p->a_scale != scale || !p->weights ||
        !p->program || !p->program_words || p->program_words > ATLAS_IMEM_BYTES / 4 - 5 || p->weight_bytes != k * p->n ||
        p->store_bytes != m * p->n * 2 || p->output_offset != m * k + p->weight_bytes + 256 ||
        p->total_bytes != p->output_offset + p->store_bytes + 256 + tables || p->total_bytes > c->dram_capacity ||
        !p->load_bytes || p->load_bytes % 32 || p->store_bytes % 32 || (rope && (!native_attention || !full->fusions
        || !full->rope_tables || full->fusions[c->cursor - 1].operand_offset != p->output_offset + p->store_bytes + 256))) {
      c->cursor = start;
      return -3;
    }
    column += p->n;
  }
  size_t end = c->cursor;
  c->cursor = start;
  for (; c->cursor < end; ++c->cursor) {
    const tiny_panel *p = c->panels + c->cursor;
#if ATLAS_FAST_MODE
    /* A and auxiliary operands are fully staged below; B is an immutable blob. */
    memset(c->dram + p->output_offset, 0xcd, p->store_bytes);
#else
    memset(c->dram, 0xcd, p->total_bytes);
#endif
    memset(c->dram + p->output_offset - 256, 0xa5, 256);
    memset(c->dram + p->output_offset + p->store_bytes, 0x5a, 256);
    memcpy(c->dram, full->packed_a, p->physical_rows * k);
#if !ATLAS_FAST_MODE
    memcpy(c->dram + p->physical_rows * k, p->weights, p->weight_bytes);
#endif
    if (rope) memcpy(c->dram + p->output_offset + p->store_bytes + 256, full->rope_tables + table_offset, tables);
    if (c->transport(c->transport_context, p, c->dram, p->total_bytes)) return -5;
    /* Producer physical order: M32,N32 tiles; each tile stores two
     * 32x16 halves. Tiled logits retain every physical decode row. */
    if ((native_attention && logical <= 2) || (logical == 7 && (full->capabilities & TINY_FULL_CAP_TILED_LOGITS))) {
#if ATLAS_FAST_MODE && TINY_FAST_HOST_LEVEL >= 2
      for (size_t tr = 0; tr < p->physical_rows / 32; ++tr) {
        size_t source = tr * (p->n / 32) * ATLAS_BF16_TILE_BYTES;
        size_t destination = (tr * (width / 32) + p->column / 32) * ATLAS_BF16_TILE_BYTES;
        memcpy((uint8_t *)output + destination, c->dram + p->output_offset + source,
               p->n / 32 * ATLAS_BF16_TILE_BYTES);
      }
#else
      for (size_t tr = 0; tr < p->physical_rows / 32; ++tr)
        for (size_t tc = 0; tc < p->n / 32; ++tc) {
          size_t source = (tr * (p->n / 32) + tc) * ATLAS_BF16_TILE_BYTES;
          size_t destination = (tr * (width / 32) + p->column / 32 + tc) * ATLAS_BF16_TILE_BYTES;
          memcpy((uint8_t *)output + destination, c->dram + p->output_offset + source, ATLAS_BF16_TILE_BYTES);
        }
#endif
      continue;
    }
    for (size_t r = 0; r < rows; ++r)
      for (size_t n = 0; n < p->n; ++n) {
        size_t tile = (r / 32) * (p->n / 32) + n / 32;
        size_t index =
            tile * (ATLAS_BF16_TILE_BYTES / 2) + (n % 32) / 16 * (ATLAS_BF16_HALF_BYTES / 2) + (r % 32) * 16 + n % 16;
        size_t at = p->output_offset + index * 2;
        output[r * width + p->column + n] = (uint16_t)c->dram[at] | (uint16_t)((uint16_t)c->dram[at + 1] << 8);
      }
  }
  if (norm_fp8) full->pending_norm.stage = logical < 2 ? logical + 2 : 0;
  return 0;
}

int tiny_full_project(void *opaque, unsigned layer, unsigned logical,
                       const uint16_t *input, size_t rows, size_t k, size_t width,
                       uint16_t *output) {
  tiny_full_projection_context *c = opaque;
  static const size_t widths[8] = {2048, 256, 256, 2048, 5632, 5632, 2048, 32000};
  if (!c || c->step_count < (c->attention_panels ? 1u : 2u) || c->step_count > 33
      || c->calls >= c->step_count * 155 || (c->attention_v2 && logical == 3)) { tiny_full_projection_invalidate(c); return -1; }
  size_t index = c->calls % 155;
  unsigned expected_layer = index == 154 ? 22 : (unsigned)(index / 7);
  unsigned expected_role = index == 154 ? 7 : (unsigned)(index % 7);
  if (layer != expected_layer || logical != expected_role || width != widths[expected_role]
      || k != (logical == 6 ? 5632 : 2048) || rows != (c->calls < 155 ? 128u : 1u)) { tiny_full_projection_invalidate(c); return -1; }
  int status = project_panels(c, c->attention_panels != NULL, logical, input, rows, k, width, output);
  if (!status) ++c->calls;
  else tiny_full_projection_invalidate(c);
  return status;
}

int tiny_full_fused_project(void *opaque, unsigned layer, unsigned logical, const uint16_t *input,
                            size_t rows, size_t k, size_t width, const uint16_t *operand, uint16_t *output) {
  tiny_full_projection_context *c = opaque;
  static const size_t widths[4] = {2048, 5632, 5632, 2048};
  if (!c || logical < 3 || logical > 6 || c->step_count < 1 || c->step_count > 33
      || c->calls >= c->step_count * 155 || c->calls % 155 != layer * 7 + logical || layer >= 22
      || width != widths[logical - 3] || k != (logical == 6 ? 5632u : 2048u)
      || rows != (c->calls < 155 ? 128u : 1u)) { tiny_full_projection_invalidate(c); return -1; }
  int status = fused_panels(c, logical, input, rows, k, width, operand, output);
  if (!status) ++c->calls;
  else tiny_full_projection_invalidate(c);
  return status;
}

_Static_assert(sizeof(tiny_attention_panel) == 176, "Attention panel ABI size");
_Static_assert(sizeof(tiny_attention_v2_panel) == 104, "Attention v2 companion ABI size");

int tiny_full_attention_overlay_expected(const tiny_attention_panel *p, const tiny_attention_v2_panel *v,
                                         const uint8_t *expected, uint8_t *before, size_t bytes) {
  if (!p || !v || !expected || !before || v->version != TINY_ATTENTION_V2 || !v->written_spans
      || !v->written_span_count || v->written_span_count > v->output_extent_bytes / 32
      || p->output_offset > bytes || v->output_extent_bytes > bytes - p->output_offset
      || p->output_bytes != v->output_extent_bytes) return -1;
  size_t end = 0, written = 0;
  for (size_t i = 0; i < v->written_span_count; ++i) {
    const tiny_attention_span *s = v->written_spans + i;
    if (!s->bytes || s->offset < end || s->offset > v->output_extent_bytes
        || s->bytes > v->output_extent_bytes - s->offset) return -1;
    end = s->offset + s->bytes;
    written += s->bytes;
  }
  if (written != p->store_bytes) return -1;
  for (size_t i = 0; i < v->written_span_count; ++i) {
    const tiny_attention_span *s = v->written_spans + i;
    memcpy(before + p->output_offset + s->offset, expected + s->offset, s->bytes);
  }
  return 0;
}

static int attend_v2(tiny_full_projection_context *c, const tiny_attention_panel *p,
                     const uint16_t *query, const uint16_t *keys, const uint16_t *values, uint16_t *output) {
  const tiny_attention_v2_panel *v = c->attention_v2 + c->attention_cursor;
  size_t rows = p->query_rows, key_rows = p->keys;
  size_t q_bytes = rows == 128 ? 524288 : 16384;
  size_t kv_bytes = (key_rows + 31) / 32 * 32 * 256 * 2;
  size_t descriptors = rows == 128 ? 128 : 64;
  size_t masks = rows == 128 ? 2048 : 4096;
  size_t extent = rows == 128 ? 262144 : 65536;
  if (v->version != TINY_ATTENTION_V2 || v->output_dtype != TINY_ATTENTION_OUTPUT_FP8
      || v->q_layout != (rows == 128 ? TINY_ATTENTION_Q_NORMAL_BF16 : TINY_ATTENTION_Q_PACKED_BF16)
      || v->packed_decode != (rows == 1) || p->step > 2 || rows != (p->step ? 1u : 128u)
      || key_rows != 128 + p->step || v->q_bytes != q_bytes || v->kv_bytes != kv_bytes
      || p->q_offset || p->k_offset != q_bytes || p->v_offset != q_bytes + kv_bytes
      || p->descriptor_offset != q_bytes + 2 * kv_bytes || p->descriptor_bytes != descriptors
      || v->key_count_offset != p->descriptor_offset + descriptors || v->key_count_bytes != 32
      || v->mask_offset != v->key_count_offset + 32 || v->mask_bytes != masks
      || p->output_offset != v->mask_offset + masks + 256 || v->output_extent_bytes != extent
      || p->output_bytes != extent || p->store_bytes != rows * 2048
      || p->total_bytes != p->output_offset + extent + 256 || p->total_bytes > c->panels.dram_capacity
      || p->load_bytes != q_bytes + 2 * kv_bytes + descriptors + 32 + masks
      || !p->initial || !p->expected || !p->program || !p->program_words
      || p->program_words > ATLAS_IMEM_BYTES / 4 - 5 || !v->written_spans || !v->written_span_count
      || v->written_span_count > extent / 32 || p->p_exponent != -8
      || p->q_exponent < -60 || p->q_exponent > 60 || p->k_exponent < -60 || p->k_exponent > 60
      || p->v_exponent < -60 || p->v_exponent > 60 || v->o_exponent < -60 || v->o_exponent > 60
      || (c->pending_attention.valid && !c->pending_attention.consumed)) return -2;
  size_t end = 0, written = 0;
  for (size_t i = 0; i < v->written_span_count; ++i) {
    const tiny_attention_span *s = v->written_spans + i;
    if (!s->bytes || s->offset % 32 || s->bytes % 32 || s->offset < end
        || s->offset > extent || s->bytes > extent - s->offset) return -2;
    if (rows == 1 && (v->written_span_count != 64 || s->offset != (i / 2) * 2048 + (i % 2) * 1024
                     || s->bytes != 32)) return -2;
    end = s->offset + s->bytes;
    written += s->bytes;
  }
  if (written != p->store_bytes) return -2;
#if ATLAS_FAST_MODE && TINY_FAST_HOST_LEVEL >= 1 && TINY_FAST_ATTENTION_INITIALS_ADMITTED
  /* build_fast admits these exact read-only templates and their linked panel offsets. */
  if (c->attention_panels != tiny_full_attention_panels || c->attention_count != TINY_FULL_ATTENTION_KERNELS) {
#endif
  for (size_t i = 0; i < extent; ++i) if (p->initial[p->output_offset + i]) return -2;
  for (size_t i = 0; i < 256; ++i)
    if (p->initial[p->output_offset - 256 + i] != 0xa5
        || p->initial[p->output_offset + extent + i] != 0x5a) return -2;
#if ATLAS_FAST_MODE && TINY_FAST_HOST_LEVEL >= 1 && TINY_FAST_ATTENTION_INITIALS_ADMITTED
  }
#endif
  for (size_t row = key_rows; row < (key_rows + 31) / 32 * 32; ++row)
    for (size_t column = 0; column < 256; ++column) {
      size_t at = tiny_full_bf16_index(row, column, 256);
      if (keys[at] || values[at]) return -2;
    }
  memcpy(c->panels.dram, p->initial, p->total_bytes);
  if (rows == 128) memcpy(c->panels.dram, query, q_bytes);
  else {
    memset(c->panels.dram, 0, q_bytes);
    for (size_t head = 0; head < 32; ++head)
      for (size_t d16 = 0; d16 < 4; ++d16)
        memcpy(c->panels.dram + (head / 8) * 4096 + d16 * 1024 + (head % 8) * 32,
               (const uint8_t *)query + head * 4096 + d16 * 1024, 32);
  }
  memcpy(c->panels.dram + p->k_offset, keys, kv_bytes);
  memcpy(c->panels.dram + p->v_offset, values, kv_bytes);
  if (c->attention_transport(c->attention_transport_context, p, c->panels.dram, p->total_bytes)) return -3;
  memcpy(output, c->panels.dram + p->output_offset, extent);
  c->packed_valid = 0;
  c->pending_attention = (tiny_attention_fp8_identity){(const uint8_t *)output, extent, p->step, p->layer,
                                                    c->attention_cursor, v->o_exponent, 1, 0};
  ++c->attention_cursor;
  return 0;
}

int tiny_full_attend(void *opaque, unsigned layer, const uint16_t *query, const uint16_t *keys,
                      const uint16_t *values, size_t rows, size_t key_rows, uint16_t *output) {
  tiny_full_projection_context *c = opaque;
  if (!c || !c->attention_panels || !c->attention_transport || !query || !keys || !values || !output
      || c->attention_cursor >= c->attention_count || layer >= 22 || (rows != 128 && rows != 1)) { tiny_full_projection_invalidate(c); return -1; }
  const tiny_attention_panel *p = c->attention_panels + c->attention_cursor;
  if (c->attention_v2) {
    if (!c->panels.dram || p->layer != layer || p->step != c->calls / 155
        || c->calls % 155 != layer * 7 + 3 || p->query_rows != rows || p->keys != key_rows
        || p->query_heads != 32 || p->kv_heads != 4) { tiny_full_projection_invalidate(c); return -2; }
    int status = attend_v2(c, p, query, keys, values, output);
    if (status) tiny_full_projection_invalidate(c);
    return status;
  }
  size_t q_bytes = (rows + 31) / 32 * 32 * 2048 * 2;
  size_t kv_bytes = (key_rows + 31) / 32 * 32 * 256 * 2;
  if (p->layer != layer || p->step != c->calls / 155 || c->calls % 155 != layer * 7 + 3
      || p->query_rows != rows || p->keys != key_rows || p->query_heads != 32 || p->kv_heads != 4
      || p->q_offset != 0 || p->k_offset != q_bytes || p->v_offset != q_bytes + kv_bytes
      || p->descriptor_offset != q_bytes + 2 * kv_bytes || !p->descriptor_bytes
      || p->descriptor_offset + p->descriptor_bytes > p->output_offset
      || p->output_bytes != q_bytes || p->store_bytes != q_bytes
      || p->output_offset + p->output_bytes + 256 != p->total_bytes
      || p->total_bytes > c->panels.dram_capacity || !p->initial || !p->expected || !p->program
      || !p->program_words || p->program_words > ATLAS_IMEM_BYTES / 4 - 5) { tiny_full_projection_invalidate(c); return -2; }
  memcpy(c->panels.dram, p->initial, p->total_bytes);
  memcpy(c->panels.dram + p->q_offset, query, q_bytes);
  memcpy(c->panels.dram + p->k_offset, keys, kv_bytes);
  memcpy(c->panels.dram + p->v_offset, values, kv_bytes);
  if (c->attention_transport(c->attention_transport_context, p, c->panels.dram, p->total_bytes)) { tiny_full_projection_invalidate(c); return -3; }
  memcpy(output, c->panels.dram + p->output_offset, p->output_bytes);
  ++c->attention_cursor;
  return 0;
}
