/* Explicit pointer binding without invoking a model or arithmetic kernel. */
#include "merlin_descriptor_internal.h"
#include <string.h>

static int metadata_overlap(uintptr_t a, size_t an, uintptr_t b, size_t bn) {
  if (an > UINTPTR_MAX - a || bn > UINTPTR_MAX - b) return 1;
  return a < b + bn && b < a + an;
}

int merlin_prepare_strided_descriptor(void *base, size_t capacity_bytes,
                                      int element_bytes, int rank, int64_t offset,
                                      const int64_t *sizes, const int64_t *strides,
                                      merlin_descriptor_t *descriptor) {
  if (!base || !sizes || !strides || !descriptor || rank <= 0 || rank > MERLIN_MAX_RANK ||
      offset < 0 || (element_bytes != 1 && element_bytes != 2 && element_bytes != 4 && element_bytes != 8) ||
      (uintptr_t)base % (unsigned)element_bytes ||
      (uintptr_t)descriptor % _Alignof(merlin_descriptor_t)) return -1;
  uint64_t largest = (uint64_t)offset, elements = 1;
  merlin_arg_t argument = {MERLIN_INPUT, 0, rank, {0}, element_bytes};
  int64_t saved_strides[MERLIN_MAX_RANK];
  for (int d = 0; d < rank; ++d) {
    if (sizes[d] <= 0 || strides[d] <= 0 ||
        (uint64_t)sizes[d] > (uint64_t)INT64_MAX / elements ||
        (uint64_t)(sizes[d] - 1) > ((uint64_t)INT64_MAX - largest) / (uint64_t)strides[d]) return -1;
    elements *= (uint64_t)sizes[d];
    largest += (uint64_t)(sizes[d] - 1) * (uint64_t)strides[d];
    argument.dims[d] = (long)sizes[d];
    saved_strides[d] = strides[d];
  }
  if (largest + 1 > SIZE_MAX / (unsigned)element_bytes) return -1;
  size_t span = (size_t)(largest + 1) * (unsigned)element_bytes;
  if (span > capacity_bytes || capacity_bytes > UINTPTR_MAX - (uintptr_t)base ||
      metadata_overlap((uintptr_t)descriptor, sizeof(*descriptor), (uintptr_t)base, capacity_bytes) ||
      metadata_overlap((uintptr_t)descriptor, sizeof(*descriptor), (uintptr_t)sizes, rank * sizeof(*sizes)) ||
      metadata_overlap((uintptr_t)descriptor, sizeof(*descriptor), (uintptr_t)strides, rank * sizeof(*strides))) return -1;
  fill_descriptor(descriptor, base, &argument);
  descriptor->offset = offset;
  /* Reuse the same rank-exact layout as fill_descriptor. Copying word payloads
   * avoids indexing past the public oversized sizes[] member into strides[]. */
  unsigned char *packed = (unsigned char *)descriptor;
  for (int d = 0; d < rank; ++d)
    memcpy(packed + (3 + rank + d) * sizeof(int64_t), &saved_strides[d], sizeof(int64_t));
  return 0;
}

int merlin_prepare_descriptors(const merlin_arg_t *args, int n_args,
                               void *const *data_ptrs, const size_t *capacities,
                               merlin_descriptor_t *descs, void **descriptor_ptrs) {
  if (!args || !data_ptrs || !capacities || !descs || !descriptor_ptrs ||
      n_args <= 0 || n_args > MERLIN_MAX_DESCRIPTOR_ARGS) return -1;
  for (int i = 0; i < n_args; ++i) {
    if (!data_ptrs[i] || args[i].rank < 0 || args[i].rank > MERLIN_MAX_RANK ||
        args[i].elem_size <= 0) return -1;
    size_t elements = 1;
    for (int d = 0; d < args[i].rank; ++d) {
      if (args[i].dims[d] <= 0 || elements > (size_t)INT64_MAX / (size_t)args[i].dims[d]) return -1;
      elements *= (size_t)args[i].dims[d];
    }
    if (elements > SIZE_MAX / (size_t)args[i].elem_size ||
        elements * (size_t)args[i].elem_size > capacities[i]) return -1;
  }
  for (int i = 0; i < n_args; ++i) {
    fill_descriptor(&descs[i], data_ptrs[i], &args[i]);
    descriptor_ptrs[i] = &descs[i];
  }
  return 0;
}
