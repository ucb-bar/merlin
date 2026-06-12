/* Merlin C runtime: build memref descriptors from the arg table and invoke forward(). */
#include "merlin_model.h"

static void fill_descriptor(merlin_descriptor_t *d, void *data, const merlin_arg_t *a) {
  d->allocated = data;
  d->aligned = data;
  d->offset = 0;
  /* row-major contiguous strides */
  long stride = 1;
  for (int i = a->rank - 1; i >= 0; i--) {
    d->sizes[i] = a->dims[i];
    d->strides[i] = stride;
    stride *= a->dims[i];
  }
}

void merlin_run(const merlin_arg_t *args, int n_args, const void *weights_base,
                void *const *input_ptrs, void *out_buffer,
                merlin_descriptor_t *descs) {
  void *desc_ptrs[256];           /* descriptor pointers for the ciface call */
  for (int i = 0; i < n_args; i++) {
    void *data = 0;
    switch (args[i].kind) {
      case MERLIN_WEIGHT:
        data = (void *)((const char *)weights_base + args[i].offset);
        break;
      case MERLIN_INPUT:
        data = input_ptrs[i];
        break;
      case MERLIN_OUTPUT:
        data = out_buffer;
        break;
      default:
        break;
    }
    fill_descriptor(&descs[i], data, &args[i]);
    desc_ptrs[i] = &descs[i];
  }
  merlin_invoke(desc_ptrs);
}
