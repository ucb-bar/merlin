/* Explicit pointer binding without invoking a model or arithmetic kernel. */
#include "merlin_descriptor_internal.h"

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
