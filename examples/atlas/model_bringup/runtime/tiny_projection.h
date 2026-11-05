#ifndef TINY_PROJECTION_H
#define TINY_PROJECTION_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
  size_t rows, physical_rows, k, n, column, width;
  unsigned logical;
  int a_scale;
  const uint8_t *weights;
  size_t weight_bytes;
  const uint32_t *program;
  size_t program_words;
  size_t output_offset, total_bytes, load_bytes, store_bytes;
} tiny_panel;
typedef int (*tiny_transport)(void *, const tiny_panel *, uint8_t *, size_t);
typedef struct {
  const tiny_panel *panels;
  size_t panel_count, cursor;
  uint8_t *dram;
  size_t dram_capacity;
  tiny_transport transport;
  void *transport_context;
} tiny_projection_context;
int tiny_project(void *, unsigned, const uint16_t *, size_t, size_t, size_t, uint16_t *);
#endif
