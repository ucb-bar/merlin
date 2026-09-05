/* Merlin C runtime: drive a compiled model via its MLIR C interface.
 *
 * The runtime is generic and data-driven: a generated argument table
 * (merlin_arg_t[]) describes every forward() operand — its kind (weight blob
 * offset / runtime input / output), rank, dims and element size — and the runtime
 * builds the MLIR memref descriptor for each and invokes the model. Target-agnostic:
 * the same code drives host and bare-metal (spike/Zephyr) builds.
 */
#ifndef MERLIN_MODEL_H
#define MERLIN_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define MERLIN_WEIGHT 0
#define MERLIN_INPUT 1
#define MERLIN_OUTPUT 2

#define MERLIN_MAX_RANK 8

typedef struct {
  int kind;                       /* MERLIN_WEIGHT | INPUT | OUTPUT */
  long offset;                    /* byte offset into the weight blob (kind WEIGHT) */
  int rank;
  long dims[MERLIN_MAX_RANK];
  int elem_size;                  /* bytes per element */
} merlin_arg_t;

/* MLIR memref descriptor (ranked, up to MERLIN_MAX_RANK). Layout matches MLIR's
 * lowering of memref<...> exactly: {alloc, aligned, offset, sizes[], strides[]}. */
typedef struct {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[MERLIN_MAX_RANK];
  int64_t strides[MERLIN_MAX_RANK];
} merlin_descriptor_t;

/* Generated per model (model_call.c): unrolls the N-pointer ciface call. */
void merlin_invoke(void **descriptor_ptrs);
typedef void (*merlin_invoke_fn_t)(void **descriptor_ptrs);

/* Run forward() once.
 *   args        : the generated MERLIN_ARGS table (length n_args, OUTPUT last).
 *   n_args      : MERLIN_N_ARGS.
 *   weights_base: base pointer of the weight blob (weights.bin in memory).
 *   input_ptrs  : per-arg data pointers for INPUT args (NULL otherwise); MERLIN_INPUT_PTR.
 *   out_buffer  : caller-allocated output buffer (>= output elems * elem_size).
 *   descs       : caller-allocated scratch of n_args descriptors.
 */
void merlin_run(const merlin_arg_t *args, int n_args, const void *weights_base,
                void *const *input_ptrs, void *out_buffer,
                merlin_descriptor_t *descs);

/* Multi-result form used by stateful sessions. output_ptrs is indexed by MLIR result ordinal.
 * The single-result merlin_run above remains as a compatibility wrapper. */
void merlin_run_multi(const merlin_arg_t *args, int n_args, const void *weights_base,
                      void *const *input_ptrs, void *const *output_ptrs,
                      merlin_descriptor_t *descs);

/* Multi-program sessions link several compiled entry points into one process. This form selects
 * the generated invocation trampoline explicitly, avoiding a global `forward` dispatch or a
 * model-name branch in the runtime. */
void merlin_run_multi_with(const merlin_arg_t *args, int n_args, const void *weights_base,
                           void *const *input_ptrs, void *const *output_ptrs,
                           merlin_descriptor_t *descs, merlin_invoke_fn_t invoke);

/* Copy declared result buffers into their next-step input buffers. Returns 0 on success and -1
 * when a pair is out of range or its byte extents differ (state is never truncated silently). */
int merlin_commit_state(const merlin_arg_t *args, int n_args,
                        void *const *input_ptrs, void *const *output_ptrs,
                        int n_state_pairs, const int *input_args,
                        const int *output_indices);

#endif
