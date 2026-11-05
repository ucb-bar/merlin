#include "atlas_resnet_graph_device.h"
static int span(const void *p, size_t n) {
  return p && n && n - 1 <= UINTPTR_MAX - (uintptr_t)p;
}

static int overlap(const void *a, size_t na, const void *b, size_t nb) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x <= y ? y - x < na : x - y < nb;
}

static int reject(atlas_graph_device_context *w, atlas_graph_device_result result) {
  w->failed = 1;
  w->result = result;
  return (int)result;
}

int atlas_resnet_graph_device(void *opaque, unsigned index, const uint8_t *a, size_t a_bytes, int scale, uint8_t *out,
                              size_t out_bytes) {
  atlas_graph_device_context *w = opaque;
  if (!span(w, sizeof(*w))) return ATLAS_GRAPH_DEVICE_INVALID;
  if (w->failed) return (int)w->result;
  w->failing_index = index;
  if (index >= ATLAS_GRAPH_KERNELS || index != w->completed || !span(w->host, sizeof(*w->host)) || w->host->active ||
      !span(w->kernels, ATLAS_GRAPH_KERNELS * sizeof(*w->kernels)) || !span(w->rescale_words, 254 * sizeof(uint32_t)) ||
      ((uintptr_t)w->rescale_words & 3) || !w->max_polls || w->program_capacity_words != ATLAS_GRAPH_PROGRAM_WORDS ||
      !span(w->program_buffer, ATLAS_GRAPH_PROGRAM_WORDS * sizeof(uint32_t)) || ((uintptr_t)w->program_buffer & 3) ||
      !span(w->dram, w->dram_capacity) || w->dram_capacity > 64 * 1024 * 1024 || !span(a, a_bytes) || !span(out, out_bytes))
    return reject(w, ATLAS_GRAPH_DEVICE_INVALID);
  const atlas_graph_kernel *k = &w->kernels[index];
  if (!k->program_words || k->program_words > ATLAS_GRAPH_PROGRAM_WORDS || k->patch_word >= k->program_words - 1 ||
      !span(k->program, k->program_words * 4) || ((uintptr_t)k->program & 3) || !span(k->weights, k->weight_bytes) ||
      k->activation_bytes != a_bytes || k->output_bytes != out_bytes || a_bytes > 64 * 1024 * 1024 ||
      k->weight_bytes > 64 * 1024 * 1024 || a_bytes > SIZE_MAX - k->weight_bytes - 512 ||
      k->output_offset != a_bytes + k->weight_bytes + 256 || out_bytes > SIZE_MAX - k->output_offset - 256 ||
      k->dram_bytes != k->output_offset + out_bytes + 256 || k->dram_bytes > w->dram_capacity || scale < -126 ||
      scale > 127 || k->weight_scale < -126 || k->weight_scale > 127)
    return reject(w, ATLAS_GRAPH_DEVICE_INVALID);
  /* Mutable buffers and control records must never alias each other or any
   * immutable input. Validate before staging or patching even one byte. */
  const void *mutable[] = {w, w->host, w->dram, w->program_buffer, out};
  size_t sizes[] = {sizeof(*w), sizeof(*w->host), w->dram_capacity, ATLAS_GRAPH_PROGRAM_WORDS * 4, out_bytes};
  const void *immutable[] = {a, k->program, k->weights, w->kernels, w->rescale_words};
  size_t extents[] = {a_bytes, k->program_words * 4, k->weight_bytes, ATLAS_GRAPH_KERNELS * sizeof(*k), 254 * 4};
  for (unsigned i = 0; i < 5; ++i) {
    for (unsigned j = 0; j < i; ++j)
      if (overlap(mutable[i], sizes[i], mutable[j], sizes[j])) return reject(w, ATLAS_GRAPH_DEVICE_INVALID);
    for (unsigned j = 0; j < 5; ++j)
      if (overlap(mutable[i], sizes[i], immutable[j], extents[j])) return reject(w, ATLAS_GRAPH_DEVICE_INVALID);
  }
  int exponent = scale + k->weight_scale;
  if (exponent < -126 || exponent > 127) return reject(w, ATLAS_GRAPH_DEVICE_SCALE);
  /* The qualified table must retain the exact VLI_ALL m4 power-of-two words.
   * Checking the selected and baseline entries detects corrupted metadata. */
  uint32_t word = ((uint32_t)(exponent + 127) << 23) | UINT32_C(0x25f);
  if (w->rescale_words[exponent + 126] != word || w->rescale_words[126] != UINT32_C(0x3f80025f) ||
      k->program[k->patch_word] != UINT32_C(0x3f80025f))
    return reject(w, ATLAS_GRAPH_DEVICE_INVALID);
  for (size_t i = 0; i < k->program_words; ++i) w->program_buffer[i] = k->program[i];
  w->program_buffer[k->patch_word] = word;
  for (size_t i = 0; i < a_bytes; ++i) w->dram[i] = a[i];
  for (size_t i = 0; i < k->weight_bytes; ++i) w->dram[a_bytes + i] = k->weights[i];
  for (size_t i = a_bytes + k->weight_bytes; i < k->output_offset; ++i) w->dram[i] = 0xa5;
  for (size_t i = k->output_offset; i < k->output_offset + out_bytes; ++i) w->dram[i] = 0xcd;
  for (size_t i = k->output_offset + out_bytes; i < k->dram_bytes; ++i) w->dram[i] = 0x5a;
  w->runtime_result = atlas_host_load_raw(w->host, w->program_buffer, k->program_words);
  if (w->runtime_result != ATLAS_HOST_OK) return reject(w, ATLAS_GRAPH_DEVICE_RUNTIME);
  w->active_buffer = (atlas_host_buffer){ATLAS_DRAM_BASE, k->dram_bytes, ATLAS_HOST_DEVICE_READ_WRITE};
  uint32_t before = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base + ATLAS_CSR_CYCLES);
  w->runtime_result = atlas_host_start(w->host, &w->active_buffer, 1);
  if (w->runtime_result == ATLAS_HOST_OK) w->runtime_result = atlas_host_wait(w->host, w->max_polls);
  w->cycles[index] = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base + ATLAS_CSR_CYCLES) - before;
  if (w->runtime_result != ATLAS_HOST_OK) return reject(w, ATLAS_GRAPH_DEVICE_RUNTIME);
  w->activation_scale[index] = scale;
  w->rescale_exponent[index] = exponent;
  w->sequence[index] = w->host->invocation_sequence;
  for (size_t i = a_bytes + k->weight_bytes; i < k->output_offset; ++i)
    if (((volatile uint8_t *)w->dram)[i] != 0xa5) return reject(w, ATLAS_GRAPH_DEVICE_GUARD);
  for (size_t i = k->output_offset + out_bytes; i < k->dram_bytes; ++i)
    if (((volatile uint8_t *)w->dram)[i] != 0x5a) return reject(w, ATLAS_GRAPH_DEVICE_GUARD);
  for (size_t i = 0; i < a_bytes; ++i)
    if (((volatile uint8_t *)w->dram)[i] != a[i]) return reject(w, ATLAS_GRAPH_DEVICE_INPUT);
  for (size_t i = 0; i < k->weight_bytes; ++i)
    if (((volatile uint8_t *)w->dram)[a_bytes + i] != k->weights[i]) return reject(w, ATLAS_GRAPH_DEVICE_INPUT);
  for (size_t i = 0; i < out_bytes; ++i) out[i] = ((volatile uint8_t *)w->dram)[k->output_offset + i];
  ++w->completed;
  w->result = ATLAS_GRAPH_DEVICE_OK;
  return 0;
}
