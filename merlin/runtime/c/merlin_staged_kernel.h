/* Explicit prepared-kernel callback ABI for an outlined staged generation.
 * All argument tables, pointers, descriptor arrays, state and scratch remain
 * live until synchronous compute/finish AND caller completion joins. The
 * controller calls prepare/finish; selected workers alone call compute.
 * Provider code owns device semantics, target facts and context requirements.
 */
#ifndef MERLIN_STAGED_KERNEL_H
#define MERLIN_STAGED_KERNEL_H
#include "merlin_model.h"
#include <stddef.h>
#include <stdint.h>

#define MERLIN_STAGED_KERNEL_ABI_VERSION 1u
typedef struct {
  unsigned abi_version;
  unsigned input_count, output_count, argument_count;
  const merlin_arg_t *arguments;
  void *const *data;
  const size_t *capacities;
  void *const *descriptors;
  uint64_t sequence;
  unsigned worker_count, selected_worker_mask;
  const unsigned *worker_harts;
} merlin_staged_kernel_invocation;
typedef int (*merlin_staged_kernel_prepare_fn)(void *, const merlin_staged_kernel_invocation *);
typedef int (*merlin_staged_kernel_compute_fn)(void *, const merlin_staged_kernel_invocation *, unsigned);
typedef int (*merlin_staged_kernel_finish_fn)(void *, const merlin_staged_kernel_invocation *);

/* Selected providers export these signatures under explicitly bound names:
 * int prepare(void *state, const merlin_staged_kernel_invocation *);
 * int compute(void *state, const merlin_staged_kernel_invocation *, unsigned slot);
 * int finish(void *state, const merlin_staged_kernel_invocation *);
 * Every callback is synchronous. A provider may not create an implicit team,
 * reenter the dispatcher, retain transient pointers or silently change math.
 */
#endif
