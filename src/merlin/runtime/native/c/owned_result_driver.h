/* Explicit execution-owned invocation; metadata is compiler-owned. */
#ifndef MERLIN_OWNED_RESULT_DRIVER_H
#define MERLIN_OWNED_RESULT_DRIVER_H
#include "owned_result_views.h"

enum { MERLIN_OWNED_INPUT, MERLIN_OWNED_OUTPUT };
enum {
  MERLIN_OWNED_DRIVER_INPUT = 100,
  MERLIN_OWNED_DRIVER_BUSY,
  MERLIN_OWNED_DRIVER_ALLOCATION,
  MERLIN_OWNED_DRIVER_OBSERVER
};
typedef struct {
  const char *dtype;
  size_t rank;
  const intptr_t *shape;
  const void *data;
  size_t bytes;
} MerlinOwnedInput;
typedef int (*MerlinOwnedObserver)(void *, int, size_t, const char *, size_t,
                                  const intptr_t *, const void *, size_t);
typedef struct {
  size_t input_count, output_count;
  const MerlinOwnedInput *inputs;
  const char *const *output_types;
  int (*invoke)(void **, MerlinOwnedResultView *, size_t);
  size_t max_bytes, max_allocations, max_output_bytes;
} MerlinOwnedPlan;

/* Observers may read their complete immutable payload only during the call.
 * All outputs are checked/copied and source owners released before callbacks.
 * A callback rejection stops publication and frees every private copy. */
int merlin_owned_driver_execute(const MerlinOwnedPlan *, void **,
                                MerlinOwnedObserver, void *);
const char *merlin_owned_driver_error(int);

/* Emitted by the source-typed driver generator. The input blob contains only
 * invocation bytes, never result references. Shapes are fixed by that driver;
 * scalar domain/dimension guards are executable on every call. */
int merlin_owned_driver_run(void *, size_t, MerlinOwnedObserver, void *);
#endif
