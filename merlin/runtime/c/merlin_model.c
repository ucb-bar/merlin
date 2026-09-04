/* Merlin C runtime: build memref descriptors from the arg table and invoke forward(). */
#include "merlin_model.h"
#include <stdlib.h>
#include <string.h>

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

void merlin_run_multi_with(const merlin_arg_t *args, int n_args, const void *weights_base,
                           void *const *input_ptrs, void *const *output_ptrs,
                           merlin_descriptor_t *descs, merlin_invoke_fn_t invoke) {
  /* descriptor pointers for the ciface call. Stack for the common case; heap when n_args
   * exceeds the stack table (a fixed 256-slot array overflowed the stack for the big VLAs). */
  void *stack_ptrs[MERLIN_DESC_PTRS_STACK];
  void **desc_ptrs = stack_ptrs;
  void **heap_ptrs = 0;
  if (n_args > MERLIN_DESC_PTRS_STACK) {
    heap_ptrs = (void **)malloc((size_t)n_args * sizeof(void *));
    desc_ptrs = heap_ptrs;          /* if malloc fails this is null; surfaces as a fault */
  }
  int output_index = 0;
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
        data = output_ptrs[output_index++];
        break;
      default:
        break;
    }
    fill_descriptor(&descs[i], data, &args[i]);
    desc_ptrs[i] = &descs[i];
  }
  invoke(desc_ptrs);
  if (heap_ptrs) free(heap_ptrs);
}

void merlin_run_multi(const merlin_arg_t *args, int n_args, const void *weights_base,
                      void *const *input_ptrs, void *const *output_ptrs,
                      merlin_descriptor_t *descs) {
  merlin_run_multi_with(args, n_args, weights_base, input_ptrs, output_ptrs, descs,
                        merlin_invoke);
}

void merlin_run(const merlin_arg_t *args, int n_args, const void *weights_base,
                void *const *input_ptrs, void *out_buffer,
                merlin_descriptor_t *descs) {
  void *outputs[1] = {out_buffer};
  merlin_run_multi(args, n_args, weights_base, input_ptrs, outputs, descs);
}

static size_t arg_bytes(const merlin_arg_t *arg) {
  size_t elems = 1;
  for (int i = 0; i < arg->rank; i++) elems *= (size_t)arg->dims[i];
  return elems * (size_t)arg->elem_size;
}

int merlin_commit_state(const merlin_arg_t *args, int n_args,
                        void *const *input_ptrs, void *const *output_ptrs,
                        int n_state_pairs, const int *input_args,
                        const int *output_indices) {
  for (int pair = 0; pair < n_state_pairs; pair++) {
    int input_arg = input_args[pair];
    int wanted_output = output_indices[pair];
    if (input_arg < 0 || input_arg >= n_args || args[input_arg].kind != MERLIN_INPUT ||
        wanted_output < 0 || input_ptrs[input_arg] == 0) return -1;
    int seen = 0, output_arg = -1;
    for (int i = 0; i < n_args; i++) {
      if (args[i].kind != MERLIN_OUTPUT) continue;
      if (seen++ == wanted_output) { output_arg = i; break; }
    }
    if (output_arg < 0 || output_ptrs[wanted_output] == 0) return -1;
    size_t input_bytes = arg_bytes(&args[input_arg]);
    size_t output_bytes = arg_bytes(&args[output_arg]);
    if (input_bytes != output_bytes) return -1;
    memcpy(input_ptrs[input_arg], output_ptrs[wanted_output], input_bytes);
  }
  return 0;
}
