#ifndef SMOL_BLOCK_RAW_ADAPTER_H
#define SMOL_BLOCK_RAW_ADAPTER_H

#include "atlas_host.h"

/* Preserve the block builder's transitive parameter types when embedded in
 * Merlin; the standalone Atlas adapter has no dependency on that header. */
#if defined(__has_include)
#if __has_include(<smol_vlm_block_prototype.h>)
#include <smol_vlm_block_prototype.h>
#endif
#endif
enum { SMOL_RAW_ROWS = 177, SMOL_RAW_LINEARS = 7 };

/* Immutable admitted original programs and globally scaled B payloads.
 * Physical M=192, native M=177. The program's existing rescale literal is
 * fixed: a changed activation exponent refuses before launch. No instruction
 * rewriting or new compiler pass is performed by this adapter.
 */
typedef struct {
  const uint32_t *program;
  const uint8_t *weights;
  size_t program_words, rescale_word;
  size_t input_width, output_width;
  size_t activation_bytes, weight_bytes, output_bytes, output_offset, dram_bytes;
  int activation_scale, weight_scale;
} smol_raw_kernel;

enum { SMOL_RAW_OK, SMOL_RAW_INVALID, SMOL_RAW_SCALE, SMOL_RAW_RUNTIME, SMOL_RAW_GUARD, SMOL_RAW_INPUT, SMOL_RAW_OUTPUT };
typedef struct {
  atlas_host *host;
  const smol_raw_kernel *kernels; /* Seven entries in native callback order. */
  uint8_t *dram, *activation_copy;
  size_t dram_capacity, activation_capacity, max_polls;
  /* CPU dram storage must alias/map the source program's device DMA base
   * 0x100000000 through the selected platform or diagnostic backend. */
  atlas_host_buffer active_buffer;
  unsigned completed, failed, failing_index;
  int result;
  atlas_host_result runtime_result;
  int activation_scale[SMOL_RAW_LINEARS];
  uint32_t sequence[SMOL_RAW_LINEARS], cycle_delta[SMOL_RAW_LINEARS];
} smol_raw_context;

/* C composition callback. All buffers/control records remain alive and
 * unchanged across timeout/fault until external quiescence and reinitialization.
 * Initialization requires an all-zero progress/failure record and a known
 * quiescent, initialized atlas_host. No automatic retry/reset is provided.
 * This callback's source alone is not device/model/RTL/ISA execution evidence.
 */
int smol_block_raw_linear(void *context, unsigned role, const uint16_t *input, size_t rows, size_t input_width,
                          uint16_t *output, size_t output_width);

#endif
