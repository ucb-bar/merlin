/* Generic synchronous control flow; target allocation context is selected by
 * the executor. No target placement, prediction or source arithmetic here. */
#include "owned_result_driver.h"
#include <stdlib.h>

extern int merlin_owned_results_begin(size_t, size_t);
extern void merlin_owned_results_end(void);

int merlin_owned_driver_execute(const MerlinOwnedPlan *plan, void **arguments,
                                MerlinOwnedObserver observe, void *context) {
  if (!plan || !observe || !plan->invoke || !plan->output_types ||
      !plan->output_count || plan->output_count > MERLIN_OWNED_RESULT_MAX_COUNT ||
      plan->input_count > MERLIN_OWNED_RESULT_MAX_COUNT ||
      (plan->input_count && (!plan->inputs || !arguments)) ||
      !plan->max_bytes || !plan->max_allocations || !plan->max_output_bytes)
    return MERLIN_OWNED_DRIVER_INPUT;
  MerlinOwnedResultView *views = calloc(plan->output_count, sizeof(*views));
  if (!views)
    return MERLIN_OWNED_DRIVER_ALLOCATION;
  void *copies[MERLIN_OWNED_RESULT_MAX_COUNT] = {0};
  int status = MERLIN_OWNED_DRIVER_BUSY, active = 0;
  if (!merlin_owned_results_begin(plan->max_bytes, plan->max_allocations))
    goto cleanup;
  active = 1;
  status = plan->invoke(arguments, views, plan->max_output_bytes);
  if (status)
    goto cleanup;
  for (size_t i = 0; i < plan->output_count; ++i) {
    copies[i] = malloc(views[i].bytes ? views[i].bytes : 1);
    if (!copies[i]) {
      status = MERLIN_OWNED_DRIVER_ALLOCATION;
      goto cleanup;
    }
    status = merlin_owned_result_copy(&views[i], copies[i], views[i].bytes);
    if (status)
      goto cleanup;
  }
  merlin_owned_results_end();
  active = 0;
  for (size_t i = 0; i < plan->input_count; ++i) {
    const MerlinOwnedInput *input = &plan->inputs[i];
    if (observe(context, MERLIN_OWNED_INPUT, i, input->dtype, input->rank,
                input->shape, input->data, input->bytes)) {
      status = MERLIN_OWNED_DRIVER_OBSERVER;
      goto cleanup;
    }
  }
  for (size_t i = 0; i < plan->output_count; ++i) {
    if (observe(context, MERLIN_OWNED_OUTPUT, i, plan->output_types[i],
                views[i].rank, views[i].sizes, copies[i], views[i].bytes)) {
      status = MERLIN_OWNED_DRIVER_OBSERVER;
      goto cleanup;
    }
  }
cleanup:
  if (active)
    merlin_owned_results_end();
  for (size_t i = 0; i < plan->output_count; ++i)
    free(copies[i]);
  free(views);
  return status;
}

const char *merlin_owned_driver_error(int status) {
  switch (status) {
  case MERLIN_OWNED_DRIVER_INPUT: return "invocation input contract failed";
  case MERLIN_OWNED_DRIVER_BUSY: return "another source invocation is active";
  case MERLIN_OWNED_DRIVER_ALLOCATION: return "driver copy allocation failed";
  case MERLIN_OWNED_DRIVER_OBSERVER: return "observation callback rejected payload";
  default: return merlin_owned_result_error(status);
  }
}
