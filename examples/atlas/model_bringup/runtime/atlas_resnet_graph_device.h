#ifndef ATLAS_RESNET_GRAPH_DEVICE_H
#define ATLAS_RESNET_GRAPH_DEVICE_H
#include "atlas_host.h"
enum { ATLAS_GRAPH_KERNELS = 54, ATLAS_GRAPH_PROGRAM_WORDS = 32763 };
typedef struct {
  const uint32_t *program;
  const uint8_t *weights;
  size_t program_words, patch_word, activation_bytes, weight_bytes;
  size_t output_bytes, output_offset, dram_bytes;
  int weight_scale;
} atlas_graph_kernel;
typedef enum {
  ATLAS_GRAPH_DEVICE_OK = 0,
  ATLAS_GRAPH_DEVICE_INVALID,
  ATLAS_GRAPH_DEVICE_SCALE,
  ATLAS_GRAPH_DEVICE_RUNTIME,
  ATLAS_GRAPH_DEVICE_GUARD,
  ATLAS_GRAPH_DEVICE_INPUT
} atlas_graph_device_result;
typedef struct {
  atlas_host *host;
  const atlas_graph_kernel *kernels;
  const uint32_t *rescale_words; /* 254 qualified words, exponent+126 */
  uint8_t *dram;
  size_t dram_capacity, max_polls;
  uint32_t *program_buffer;
  size_t program_capacity_words;
  /* Persistent throughout timeout/fault; external quiescence+reinit required
   * before any buffer/context reuse following a failed callback. */
  atlas_host_buffer active_buffer;
  unsigned completed, failed, failing_index;
  atlas_graph_device_result result;
  atlas_host_result runtime_result;
  int activation_scale[ATLAS_GRAPH_KERNELS], rescale_exponent[ATLAS_GRAPH_KERNELS];
  uint32_t sequence[ATLAS_GRAPH_KERNELS], cycles[ATLAS_GRAPH_KERNELS];
} atlas_graph_device_context;
/* Exact graph callback ABI. All live activation bytes come from C packing. */
int atlas_resnet_graph_device(void *, unsigned, const uint8_t *, size_t, int, uint8_t *, size_t);
#endif
