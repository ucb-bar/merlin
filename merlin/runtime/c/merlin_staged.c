/* See merlin_staged.h for the caller-owned execution and completion contract. */
#include "merlin_staged.h"

enum { IDLE, ARM, COMPUTE, FINAL_ACK, ABORT };

static unsigned all_workers(const merlin_staged_team *t) {
  return (1u << t->config.worker_count) - 1u;
}

static void relax(merlin_staged_team *t) {
  if (t->hooks.relax) t->hooks.relax(t->context);
}

static int wait_bits(merlin_staged_team *t, atomic_uint *field) {
  const unsigned all = all_workers(t);
  for (uint64_t i = 0; i < t->config.max_wait_polls; ++i) {
    if (atomic_load_explicit(field, memory_order_acquire) == all) return 0;
    relax(t);
  }
  return MERLIN_STAGED_TIMEOUT;
}

static int wait_command(merlin_staged_team *t, unsigned expected) {
  for (uint64_t i = 0; i < t->config.max_wait_polls; ++i) {
    unsigned command = atomic_load_explicit(&t->command, memory_order_acquire);
    if (command == expected) return 0;
    if (command == ABORT) return MERLIN_STAGED_ABORTED;
    relax(t);
  }
  return MERLIN_STAGED_TIMEOUT;
}

static void sample_window(merlin_staged_team *t, unsigned slot, unsigned bit,
                          uint64_t *destination) {
  if (!t->config.enable_worker_windows) return;
  merlin_staged_worker_report *r = &t->report.workers[slot];
  uint64_t value = 0;
  int status = t->hooks.shared_window_ticks(t->context, &value);
  if (status) {
    if (!r->clock_reader_status) r->clock_reader_status = status;
    return;
  }
  *destination = value;
  r->clock_valid_mask |= bit;
}

static void finish_windows(merlin_staged_team *t, unsigned slot) {
  if (!t->config.enable_worker_windows) return;
  merlin_staged_worker_report *r = &t->report.workers[slot];
  if (r->clock_reader_status) r->clock_status = MERLIN_STAGED_CLOCK_READ_ERROR;
  else if (r->clock_valid_mask != MERLIN_STAGED_ALL_WINDOWS)
    r->clock_status = MERLIN_STAGED_CLOCK_INCOMPLETE;
  else if (r->arm_ready_tick > r->compute_begin_tick ||
           r->compute_begin_tick > r->compute_end_tick ||
           r->compute_end_tick > r->final_ack_tick)
    r->clock_status = MERLIN_STAGED_CLOCK_NONMONOTONIC;
  else r->clock_status = MERLIN_STAGED_CLOCK_VALID;
}

int merlin_staged_team_init(merlin_staged_team *t,
                            const merlin_staged_config *config,
                            const merlin_staged_hooks *hooks,
                            const merlin_staged_callbacks *callbacks,
                            void *context) {
  if (!t || !config || !hooks || !callbacks ||
      !hooks->current_hart || !hooks->worker_cycles ||
      !hooks->worker_instret || !callbacks->model_init || !callbacks->prepare ||
      !callbacks->compute || !callbacks->finish || !config->max_wait_polls || !config->sequence ||
      !config->worker_count || config->worker_count > MERLIN_STAGED_MAX_WORKERS ||
      config->enable_worker_windows > 1 ||
      (!config->enable_worker_windows && !hooks->global_ticks) ||
      (config->enable_worker_windows && !hooks->shared_window_ticks))
    return MERLIN_STAGED_INVALID;
  if (hooks->current_hart(context) != config->controller_hart)
    return MERLIN_STAGED_ROLE;
  for (unsigned i = 0; i < config->worker_count; ++i) {
    if (config->worker_harts[i] == config->controller_hart)
      return MERLIN_STAGED_INVALID;
    for (unsigned j = 0; j < i; ++j)
      if (config->worker_harts[i] == config->worker_harts[j])
        return MERLIN_STAGED_INVALID;
  }
  t->config = *config;
  t->hooks = *hooks;
  t->callbacks = *callbacks;
  t->context = context;
  atomic_init(&t->command, IDLE);
  atomic_init(&t->claimed, 0);
  atomic_init(&t->armed, 0);
  atomic_init(&t->done, 0);
  atomic_init(&t->exited, 0);
  t->report = (merlin_staged_report){0};
  t->report.sequence = config->sequence;
  t->report.global_clock_hart = config->worker_harts[0];
  t->report.worker_clock_windows_enabled = config->enable_worker_windows;
  for (unsigned i = 0; i < config->worker_count; ++i) {
    t->report.workers[i].hart = config->worker_harts[i];
    t->report.workers[i].clock_status = config->enable_worker_windows ?
      MERLIN_STAGED_CLOCK_INCOMPLETE : MERLIN_STAGED_CLOCK_DISABLED;
  }
  return 0;
}

static int abort_stage(merlin_staged_team *t, int status) {
  t->report.status = status;
  atomic_store_explicit(&t->command, ABORT, memory_order_release);
  t->report.quiescent = wait_bits(t, &t->exited) == 0;
  return t->report.quiescent ? status : MERLIN_STAGED_UNQUIESCED;
}

int merlin_staged_controller(merlin_staged_team *t) {
  if (!t) return MERLIN_STAGED_INVALID;
  if (t->hooks.current_hart(t->context) != t->config.controller_hart)
    return MERLIN_STAGED_ROLE;
  unsigned idle = IDLE;
  if (!atomic_compare_exchange_strong_explicit(&t->command, &idle, ARM,
                                               memory_order_release,
                                               memory_order_relaxed))
    return MERLIN_STAGED_USED;
  int status = wait_bits(t, &t->armed);
  if (status) return abort_stage(t, status);
  status = t->callbacks.model_init(t->context);
  if (!status) status = t->callbacks.prepare(t->context);
  if (status) return abort_stage(t, status);
  atomic_store_explicit(&t->command, COMPUTE, memory_order_release);
  status = wait_bits(t, &t->done);
  if (status) return abort_stage(t, status);
  for (unsigned i = 0; i < t->config.worker_count; ++i)
    if (t->report.workers[i].status)
      return abort_stage(t, t->report.workers[i].status);
  status = t->callbacks.finish(t->context);
  if (status) return abort_stage(t, status);
  atomic_store_explicit(&t->command, FINAL_ACK, memory_order_release);
  status = wait_bits(t, &t->exited);
  t->report.quiescent = status == 0;
  if (!status)
    for (unsigned i = 0; i < t->config.worker_count; ++i)
      if (t->report.workers[i].wait_status) {
        status = t->report.workers[i].wait_status;
        break;
      }
  if (!status && t->config.enable_worker_windows) {
    t->report.worker_clock_windows_valid = 1;
    for (unsigned i = 0; i < t->config.worker_count; ++i)
      if (t->report.workers[i].clock_status != MERLIN_STAGED_CLOCK_VALID)
        t->report.worker_clock_windows_valid = 0;
  }
  t->report.status = status;
  return !t->report.quiescent ? MERLIN_STAGED_UNQUIESCED : status;
}

int merlin_staged_worker(merlin_staged_team *t, unsigned slot) {
  if (!t || slot >= t->config.worker_count) return MERLIN_STAGED_INVALID;
  if (t->hooks.current_hart(t->context) != t->config.worker_harts[slot])
    return MERLIN_STAGED_ROLE;
  const unsigned bit = 1u << slot;
  if (atomic_fetch_or_explicit(&t->claimed, bit, memory_order_acq_rel) & bit)
    return MERLIN_STAGED_USED;
  int status = wait_command(t, ARM);
  if (status) goto exit;
  merlin_staged_worker_report *r = &t->report.workers[slot];
  sample_window(t, slot, MERLIN_STAGED_ARM_READY, &r->arm_ready_tick);
  if (!slot) {
    if (t->config.enable_worker_windows) {
      if (r->clock_valid_mask & MERLIN_STAGED_ARM_READY)
        t->report.global_begin = r->arm_ready_tick;
    } else t->report.global_begin = t->hooks.global_ticks(t->context);
  }
  atomic_fetch_or_explicit(&t->armed, bit, memory_order_release);
  status = wait_command(t, COMPUTE);
  if (status) goto exit;
  sample_window(t, slot, MERLIN_STAGED_COMPUTE_BEGIN, &r->compute_begin_tick);
  r->begin_cycle = t->hooks.worker_cycles(t->context);
  r->begin_instret = t->hooks.worker_instret(t->context);
  r->status = t->callbacks.compute(t->context, slot);
  r->end_instret = t->hooks.worker_instret(t->context);
  r->end_cycle = t->hooks.worker_cycles(t->context);
  sample_window(t, slot, MERLIN_STAGED_COMPUTE_END, &r->compute_end_tick);
  atomic_fetch_or_explicit(&t->done, bit, memory_order_release);
  status = wait_command(t, FINAL_ACK);
  if (!status) {
    sample_window(t, slot, MERLIN_STAGED_FINAL_ACK, &r->final_ack_tick);
    finish_windows(t, slot);
    if (!slot) {
      if (t->config.enable_worker_windows) {
        if (r->clock_valid_mask & MERLIN_STAGED_FINAL_ACK)
          t->report.global_end = r->final_ack_tick;
        t->report.global_interval_valid = r->clock_status == MERLIN_STAGED_CLOCK_VALID;
      } else {
        t->report.global_end = t->hooks.global_ticks(t->context);
        t->report.global_interval_valid = 1;
      }
    }
  }
exit:
  /* No callback is cancelled. Exiting after ABORT is a storage-lifetime ACK. */
  t->report.workers[slot].wait_status = status;
  finish_windows(t, slot);
  atomic_fetch_or_explicit(&t->exited, bit, memory_order_release);
  return status;
}
