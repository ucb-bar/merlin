/* Per-op whole-model profiler shim (default-OFF; only linked when the K1 build is made with
 * op_profile=True, which also instruments the IR — see merlin/llvmlower/op_profile.py).
 *
 * The instrumented `@forward` calls merlin_prof_mark(id) immediately BEFORE each top-level op,
 * plus a sentinel mark before `func.return`. Each mark samples `rdtime` and credits the elapsed
 * ticks to the PREVIOUS mark's id — so ticks[i] is the cost of top-level op i, at one call per
 * op rather than an enter/exit pair.
 *
 * `rdtime` (not `rdcycle`) because the K1 kernel traps userspace rdcycle; it is the delegated
 * 24 MHz platform counter, the same timebase the K1 harness and the matmul-bucket timer use, so
 * the per-op ticks and the whole-model wall are directly comparable. Resolution is ~41.7 ns:
 * a single sub-42 ns op reads 0 or 1 tick, which is why the driver reports aggregates.
 *
 * Storage is a fixed-size static array (no malloc — a malloc here would show up in the very
 * dispatch overhead we are trying to measure). Ids beyond MERLIN_PROF_MAX_OPS are folded into
 * the overflow slot and reported, so a too-small table is loud rather than silent.
 */
#include <stdint.h>
#include <stdio.h>

#ifndef MERLIN_PROF_MAX_OPS
#define MERLIN_PROF_MAX_OPS 8192
#endif

static uint64_t merlin_prof_ticks[MERLIN_PROF_MAX_OPS];
static uint32_t merlin_prof_hits[MERLIN_PROF_MAX_OPS];
static uint64_t merlin_prof_overflow_ticks;
static uint32_t merlin_prof_overflow_hits;
static uint64_t merlin_prof_last_t;
static int32_t  merlin_prof_last_id = -1;
static uint64_t merlin_prof_marks;

static inline uint64_t merlin_prof_rdtime(void) {
  uint64_t t;
  __asm__ volatile("rdtime %0" : "=r"(t));
  return t;
}

/* Called from the instrumented IR. Keep this as short as possible: it runs once per top-level
 * op and its own cost is the profiler's perturbation budget. */
void merlin_prof_mark(int32_t id) {
  uint64_t now = merlin_prof_rdtime();
  int32_t prev = merlin_prof_last_id;
  if (prev >= 0) {
    uint64_t dt = now - merlin_prof_last_t;
    if (prev < MERLIN_PROF_MAX_OPS) {
      merlin_prof_ticks[prev] += dt;
      merlin_prof_hits[prev] += 1;
    } else {
      merlin_prof_overflow_ticks += dt;
      merlin_prof_overflow_hits += 1;
    }
  }
  merlin_prof_last_id = id;
  merlin_prof_last_t = merlin_prof_rdtime();   /* exclude this function's own bookkeeping */
  merlin_prof_marks += 1;
}

/* Emitted by the harness after merlin_run returns: one line per op that accumulated any ticks,
 * plus a summary the driver uses to check the profiler accounted for the whole run. */
void merlin_prof_dump(void) {
  uint64_t total = merlin_prof_overflow_ticks;
  for (int i = 0; i < MERLIN_PROF_MAX_OPS; i++) {
    if (merlin_prof_hits[i] == 0) continue;
    total += merlin_prof_ticks[i];
    printf("PROF %d %llu %u\n", i, (unsigned long long)merlin_prof_ticks[i],
           (unsigned)merlin_prof_hits[i]);
  }
  if (merlin_prof_overflow_hits)
    printf("PROF -1 %llu %u\n", (unsigned long long)merlin_prof_overflow_ticks,
           (unsigned)merlin_prof_overflow_hits);
  printf("METRIC prof_total_ticks %llu\n", (unsigned long long)total);
  printf("METRIC prof_marks %llu\n", (unsigned long long)merlin_prof_marks);
}
