/* Merlin C runtime: build memref descriptors from the arg table and invoke forward(). */
#include "merlin_model.h"
#include <stdlib.h>

/* desc_ptrs fast-path size. For models with <= this many args the pointer table lives on
 * the stack (no malloc — keeps the common case and the bare-metal Zephyr path allocation-
 * free). Models with MORE args (e.g. the VLAs: smolvla 1120, pi05 831, groot 490) heap-
 * allocate the table instead: a fixed `desc_ptrs[256]` previously overflowed the stack
 * (cause 0xf store-fault on K1) for any model exceeding 256 args. */
#define MERLIN_DESC_PTRS_STACK 256

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
  /* descriptor pointers for the ciface call. Stack for the common case; heap when n_args
   * exceeds the stack table (a fixed 256-slot array overflowed the stack for the big VLAs). */
  void *stack_ptrs[MERLIN_DESC_PTRS_STACK];
  void **desc_ptrs = stack_ptrs;
  void **heap_ptrs = 0;
  if (n_args > MERLIN_DESC_PTRS_STACK) {
    heap_ptrs = (void **)malloc((size_t)n_args * sizeof(void *));
    desc_ptrs = heap_ptrs;          /* if malloc fails this is null; surfaces as a fault */
  }
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
  if (heap_ptrs) free(heap_ptrs);
}
