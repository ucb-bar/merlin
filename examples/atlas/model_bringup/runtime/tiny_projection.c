#include "atlas_host_layout.h"
#include "tiny_projection.h"
/* Packing callbacks are provided by the embedding workload library. */
int host_matrix_scale_bf16(const uint16_t *, size_t, size_t, int *);
int host_matrix_pack_bf16_at_scale(const uint16_t *, size_t, size_t, size_t, size_t, size_t, size_t, int, uint8_t *, size_t);

#include <string.h>

int tiny_project(void *opaque, unsigned logical, const uint16_t *input, size_t rows, size_t k, size_t width,
                 uint16_t *output) {
  tiny_projection_context *c = opaque;
  if (!c || !c->panels || !c->dram || !c->transport || !input || !output || logical > 6 || (rows != 128 && rows != 1) ||
      !k || !width || width > 5632 || k > 5632 || c->cursor >= c->panel_count)
    return -1;
  size_t start = c->cursor, column = 0;
  int scale;
  if (host_matrix_scale_bf16(input, rows, k, &scale)) return -2;
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
        p->total_bytes != p->output_offset + p->store_bytes + 256 || p->total_bytes > c->dram_capacity || !p->load_bytes ||
        p->load_bytes % 32 || p->store_bytes % 32) {
      c->cursor = start;
      return -3;
    }
    column += p->n;
  }
  size_t end = c->cursor;
  c->cursor = start;
  for (; c->cursor < end; ++c->cursor) {
    const tiny_panel *p = c->panels + c->cursor;
    memset(c->dram, 0xcd, p->total_bytes);
    memset(c->dram + p->output_offset - 256, 0xa5, 256);
    memset(c->dram + p->output_offset + p->store_bytes, 0x5a, 256);
    if (host_matrix_pack_bf16_at_scale(input, rows, k, p->physical_rows, k, 32, 32, scale, c->dram, p->physical_rows * k))
      return -4;
    memcpy(c->dram + p->physical_rows * k, p->weights, p->weight_bytes);
    if (c->transport(c->transport_context, p, c->dram, p->total_bytes)) return -5;
    /* Producer physical order: M32,N32 tiles; each tile stores two
     * 32x16 halves. Only logical rows survive decode padding. */
    for (size_t r = 0; r < rows; ++r)
      for (size_t n = 0; n < p->n; ++n) {
        size_t tile = (r / 32) * (p->n / 32) + n / 32;
        size_t index =
            tile * (ATLAS_BF16_TILE_BYTES / 2) + (n % 32) / 16 * (ATLAS_BF16_HALF_BYTES / 2) + (r % 32) * 16 + n % 16;
        size_t at = p->output_offset + index * 2;
        output[r * width + p->column + n] = (uint16_t)c->dram[at] | (uint16_t)((uint16_t)c->dram[at + 1] << 8);
      }
  }
  return 0;
}
