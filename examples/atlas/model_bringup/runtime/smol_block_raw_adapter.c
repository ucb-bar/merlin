#include "smol_block_raw_adapter.h"
#if defined(ATLAS_SMOL_CAPTURE_CHECK) && ATLAS_SMOL_CAPTURE_CHECK
#include "atlas_host_check.h"
#include "smol_block_capture.h"
#endif
/* Packing callbacks are provided by the embedding workload library. */
int host_matrix_pack_bf16(const uint16_t *, size_t, size_t, size_t, size_t, size_t, size_t, uint8_t *, size_t, int *);

static int span(const void *pointer, size_t bytes) {
  return pointer && bytes && bytes - 1 <= UINTPTR_MAX - (uintptr_t)pointer;
}

static int overlap(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

static int reject(smol_raw_context *context, int result) {
  context->failed = 1;
  context->result = result;
  return result;
}

static uint16_t physical_output(const volatile uint8_t *dram, size_t offset, size_t row, size_t column, size_t columns) {
  size_t tile = (row / 32) * (columns / 32) + column / 32;
  size_t word = tile * (ATLAS_BF16_TILE_BYTES / 2) + ((column % 32) / 16) * (ATLAS_BF16_HALF_BYTES / 2) + (row % 32) * 16 +
                column % 16;
  return (uint16_t)((uint16_t)dram[offset + 2 * word] | ((uint16_t)dram[offset + 2 * word + 1] << 8));
}

int smol_block_raw_linear(void *opaque, unsigned role, const uint16_t *input, size_t rows, size_t input_width,
                          uint16_t *output, size_t output_width) {
  smol_raw_context *w = opaque;
  static const size_t inputs[SMOL_RAW_LINEARS] = {960, 960, 960, 960, 960, 960, 2560};
  static const size_t outputs[SMOL_RAW_LINEARS] = {960, 320, 320, 960, 2560, 2560, 960};
  if (!span(w, sizeof(*w))) return SMOL_RAW_INVALID;
  if (w->failed) return w->result;
  w->failing_index = role;
  if (sizeof(uintptr_t) < 8 || role >= SMOL_RAW_LINEARS || role != w->completed || rows != SMOL_RAW_ROWS ||
      input_width != inputs[role] || output_width != outputs[role] || !span(w->host, sizeof(*w->host)) || w->host->active ||
      !w->max_polls || !span(w->kernels, SMOL_RAW_LINEARS * sizeof(*w->kernels)) || !span(w->dram, w->dram_capacity) ||
      w->dram_capacity > 64 * 1024 * 1024 || !span(w->activation_copy, w->activation_capacity) ||
      w->activation_capacity > 64 * 1024 * 1024 || !span(input, rows * input_width * sizeof(uint16_t)) ||
      ((uintptr_t)input & 1) || !span(output, rows * output_width * sizeof(uint16_t)) || ((uintptr_t)output & 1))
    return reject(w, SMOL_RAW_INVALID);
  const smol_raw_kernel *k = &w->kernels[role];
  if (k->input_width != input_width || k->output_width != output_width || !k->program_words || k->program_words > 8192 ||
      k->rescale_word >= k->program_words - 1 || !span(k->program, k->program_words * sizeof(uint32_t)) ||
      ((uintptr_t)k->program & 3) || k->activation_bytes != 192 * input_width ||
      k->weight_bytes != input_width * output_width || k->output_bytes != 192 * output_width * sizeof(uint16_t) ||
      k->output_offset != k->activation_bytes + k->weight_bytes + 256 ||
      k->dram_bytes != k->output_offset + k->output_bytes + 256 || k->dram_bytes > w->dram_capacity ||
      k->activation_bytes > w->activation_capacity || !span(k->weights, k->weight_bytes) || k->activation_scale < -126 ||
      k->activation_scale > 127 || k->weight_scale < -126 || k->weight_scale > 127)
    return reject(w, SMOL_RAW_INVALID);
  int exponent = k->activation_scale + k->weight_scale;
  if (exponent < -126 || exponent > 127 ||
      k->program[k->rescale_word] != (((uint32_t)(exponent + 127) << 23) | UINT32_C(0x25f)))
    return reject(w, SMOL_RAW_INVALID);
  void *mutable[] = {w, w->host, w->dram, w->activation_copy, output};
  const size_t mutable_bytes[] = {sizeof(*w), sizeof(*w->host), w->dram_capacity, w->activation_capacity,
                                  rows * output_width * sizeof(uint16_t)};
  const void *immutable[] = {input, k->program, k->weights, w->kernels};
  const size_t immutable_bytes[] = {rows * input_width * sizeof(uint16_t), k->program_words * sizeof(uint32_t),
                                    k->weight_bytes, SMOL_RAW_LINEARS * sizeof(*k)};
  for (size_t i = 0; i < 5; ++i) {
    for (size_t j = 0; j < i; ++j)
      if (overlap(mutable[i], mutable_bytes[i], mutable[j], mutable_bytes[j])) return reject(w, SMOL_RAW_INVALID);
    for (size_t j = 0; j < 4; ++j)
      if (overlap(mutable[i], mutable_bytes[i], immutable[j], immutable_bytes[j])) return reject(w, SMOL_RAW_INVALID);
  }
  int activation_scale;
  if (host_matrix_pack_bf16(input, rows, input_width, 192, input_width, 32, 32, w->activation_copy, w->activation_capacity,
                            &activation_scale))
    return reject(w, SMOL_RAW_INPUT);
  if (activation_scale != k->activation_scale) return reject(w, SMOL_RAW_SCALE);
  w->activation_scale[role] = activation_scale;
  for (size_t i = 0; i < k->activation_bytes; ++i) w->dram[i] = w->activation_copy[i];
  for (size_t i = 0; i < k->weight_bytes; ++i) w->dram[k->activation_bytes + i] = k->weights[i];
  for (size_t i = k->activation_bytes + k->weight_bytes; i < k->output_offset; ++i) w->dram[i] = 0xa5;
  for (size_t i = k->output_offset; i < k->output_offset + k->output_bytes; ++i) w->dram[i] = 0xcd;
  for (size_t i = k->output_offset + k->output_bytes; i < k->dram_bytes; ++i) w->dram[i] = 0x5a;
  w->runtime_result = atlas_host_load_raw(w->host, k->program, k->program_words);
  if (w->runtime_result != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
  w->active_buffer = (atlas_host_buffer){(uintptr_t)ATLAS_DRAM_BASE, k->dram_bytes, ATLAS_HOST_DEVICE_READ_WRITE};
  uint32_t before = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base + ATLAS_CSR_CYCLES);
  w->runtime_result = atlas_host_start(w->host, &w->active_buffer, 1);
  if (w->runtime_result == ATLAS_HOST_OK) w->runtime_result = atlas_host_wait(w->host, w->max_polls);
  w->cycle_delta[role] = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base + ATLAS_CSR_CYCLES) - before;
  if (w->runtime_result != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
#if defined(ATLAS_SMOL_CAPTURE_CHECK) && ATLAS_SMOL_CAPTURE_CHECK
  uint64_t check_begin, check_end;
  __asm__ volatile("fence rw,rw\nrdcycle %0" : "=r"(check_begin) : : "memory");
  const uint8_t *expected = smol_capture_expected[role];
  smol_capture_result *check = &smol_capture_results[role];
  check->preserved_mismatches = atlas_host_check_exact(w->dram, expected, k->output_offset).mismatch_bytes;
  check->output_mismatches = atlas_host_check_exact(w->dram + k->output_offset, expected + k->output_offset,
                                                   k->output_bytes).mismatch_bytes;
  size_t tail = k->output_offset + k->output_bytes;
  check->preserved_mismatches += atlas_host_check_exact(w->dram + tail, expected + tail,
                                                       k->dram_bytes - tail).mismatch_bytes;
  __asm__ volatile("fence rw,rw\nrdcycle %0" : "=r"(check_end) : : "memory");
  check->check_cycles = check_end - check_begin;
  if (check->output_mismatches) return reject(w, SMOL_RAW_OUTPUT);
  if (check->preserved_mismatches) return reject(w, SMOL_RAW_INPUT);
#else
  for (size_t i = k->activation_bytes + k->weight_bytes; i < k->output_offset; ++i)
    if (((volatile uint8_t *)w->dram)[i] != 0xa5) return reject(w, SMOL_RAW_GUARD);
  for (size_t i = k->output_offset + k->output_bytes; i < k->dram_bytes; ++i)
    if (((volatile uint8_t *)w->dram)[i] != 0x5a) return reject(w, SMOL_RAW_GUARD);
  for (size_t i = 0; i < k->activation_bytes; ++i)
    if (((volatile uint8_t *)w->dram)[i] != w->activation_copy[i]) return reject(w, SMOL_RAW_INPUT);
  for (size_t i = 0; i < k->weight_bytes; ++i)
    if (((volatile uint8_t *)w->dram)[k->activation_bytes + i] != k->weights[i]) return reject(w, SMOL_RAW_INPUT);
#endif
  /* Validate complete physical output and positive-zero padded rows before
   * writing even the first native output word. */
  for (size_t row = 0; row < 192; ++row)
    for (size_t column = 0; column < output_width; ++column) {
      uint16_t value = physical_output(w->dram, k->output_offset, row, column, output_width);
      if ((row >= rows && value != 0) || (value & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return reject(w, SMOL_RAW_OUTPUT);
    }
  for (size_t row = 0; row < rows; ++row)
    for (size_t column = 0; column < output_width; ++column)
      output[row * output_width + column] = physical_output(w->dram, k->output_offset, row, column, output_width);
  w->sequence[role] = w->host->invocation_sequence;
  ++w->completed;
  w->result = SMOL_RAW_OK;
  return SMOL_RAW_OK;
}
