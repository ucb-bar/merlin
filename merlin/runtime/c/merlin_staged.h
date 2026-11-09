/* Explicit single-stage, single-owner prepare/compute/finish dispatch.
 *
 * Target hooks and callbacks establish real hart identity, pinned execution,
 * coherent shared storage and synchronous device completion. C11 atomics only
 * publish those caller-owned objects; they do not establish device completion.
 * Initialize before workers enter, and retain all storage until quiescent=1.
 * No OpenMP, allocator, ISA instruction, timer ISR or background thread here.
 */
#ifndef MERLIN_STAGED_H
#define MERLIN_STAGED_H
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#define MERLIN_STAGED_MAX_WORKERS 8u
#define MERLIN_STAGED_ABI_VERSION 2u
enum {
  MERLIN_STAGED_CLOCK_DISABLED,
  MERLIN_STAGED_CLOCK_VALID,
  MERLIN_STAGED_CLOCK_READ_ERROR,
  MERLIN_STAGED_CLOCK_NONMONOTONIC,
  MERLIN_STAGED_CLOCK_INCOMPLETE
};
enum {
  MERLIN_STAGED_ARM_READY = 1u,
  MERLIN_STAGED_COMPUTE_BEGIN = 2u,
  MERLIN_STAGED_COMPUTE_END = 4u,
  MERLIN_STAGED_FINAL_ACK = 8u,
  MERLIN_STAGED_ALL_WINDOWS = 15u
};
enum {
  MERLIN_STAGED_OK = 0,
  MERLIN_STAGED_INVALID = -1,
  MERLIN_STAGED_ROLE = -2,
  MERLIN_STAGED_USED = -3,
  MERLIN_STAGED_TIMEOUT = -4,
  MERLIN_STAGED_UNQUIESCED = -5,
  MERLIN_STAGED_ABORTED = -6
};

typedef struct {
  unsigned controller_hart;
  unsigned worker_count;
  unsigned worker_harts[MERLIN_STAGED_MAX_WORKERS];
  uint64_t sequence;
  uint64_t max_wait_polls; /* finite liveness bound, not a time measurement */
  unsigned enable_worker_windows; /* default zero preserves legacy readers */
} merlin_staged_config;

typedef struct {
  unsigned (*current_hart)(void *);
  uint64_t (*global_ticks)(void *); /* called only by worker slot zero */
  uint64_t (*worker_cycles)(void *); /* called only by compute workers */
  uint64_t (*worker_instret)(void *);
  void (*relax)(void *); /* optional, must preserve worker context */
  /* Opt-in checked shared-clock reader, called only by the owning worker.
   * Return zero on success; a nonzero error preserves the raw reader status.
   * All workers must use the same declared monotonic clock domain. */
  int (*shared_window_ticks)(void *, uint64_t *);
} merlin_staged_hooks;

typedef struct {
  int (*model_init)(void *);
  int (*prepare)(void *);
  int (*compute)(void *, unsigned worker_slot);
  int (*finish)(void *);
} merlin_staged_callbacks;

typedef struct {
  unsigned hart;
  int status;
  int wait_status;
  uint64_t begin_cycle, end_cycle, begin_instret, end_instret;
  uint64_t arm_ready_tick, compute_begin_tick, compute_end_tick, final_ack_tick;
  unsigned clock_valid_mask, clock_status;
  int clock_reader_status;
} merlin_staged_worker_report;

typedef struct {
  int status;
  unsigned quiescent;
  unsigned global_interval_valid;
  unsigned worker_clock_windows_enabled, worker_clock_windows_valid;
  unsigned global_clock_hart;
  uint64_t sequence;
  uint64_t global_begin, global_end;
  merlin_staged_worker_report workers[MERLIN_STAGED_MAX_WORKERS];
} merlin_staged_report;

typedef struct {
  merlin_staged_config config;
  merlin_staged_hooks hooks;
  merlin_staged_callbacks callbacks;
  void *context;
  atomic_uint command, claimed, armed, done, exited;
  merlin_staged_report report;
} merlin_staged_team;

/* team_init and controller are controller-only; workers call worker exactly once
 * per initialized team, on their declared physical harts. The controller never
 * invokes a counter reader or a compute callback. No role has an implicit ID.
 * A failed callback's exact error is retained. Failure never grants safe reuse:
 * check report.quiescent even when controller returned an error.
 */
int merlin_staged_team_init(merlin_staged_team *, const merlin_staged_config *,
                            const merlin_staged_hooks *,
                            const merlin_staged_callbacks *, void *context);
int merlin_staged_controller(merlin_staged_team *);
int merlin_staged_worker(merlin_staged_team *, unsigned worker_slot);
#endif
