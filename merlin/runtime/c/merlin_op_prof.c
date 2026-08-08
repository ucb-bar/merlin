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

/* On Zephyr there is no stdio on the console and no delegated `rdtime`: printk is the console and
 * mcycle is the counter the rest of the harness already reports. Selected by the build (the Zephyr
 * CMakeLists defines MERLIN_PROF_ZEPHYR) rather than sniffed, so the K1 path is untouched. */
#ifdef MERLIN_PROF_ZEPHYR
#include <zephyr/sys/printk.h>
#define MERLIN_PROF_PRINT printk
#else
#define MERLIN_PROF_PRINT printf
#endif

#ifndef MERLIN_PROF_MAX_OPS
#define MERLIN_PROF_MAX_OPS 8192
#endif

/* WHERE THE RUN CURRENTLY IS, readable from outside this file.
 *
 * The accumulate-and-dump-at-the-end design below answers "what was slow" and is useless for the
 * question that actually cost us two delivery rounds: "it stopped -- where?". A dump that runs after
 * merlin_run returns never runs if merlin_run does not return. This one volatile int is what the debug
 * harness's heartbeat reads, so an image that stops mid-inference still names the op it stopped in.
 * Declared weak in the generated harness so a build without the profiler still links. */
volatile int32_t merlin_prof_last_id = -1;

static uint64_t merlin_prof_ticks[MERLIN_PROF_MAX_OPS];
static uint32_t merlin_prof_hits[MERLIN_PROF_MAX_OPS];
static uint64_t merlin_prof_overflow_ticks;
static uint32_t merlin_prof_overflow_hits;
static uint64_t merlin_prof_last_t;
static uint64_t merlin_prof_marks;

static inline uint64_t merlin_prof_rdtime(void) {
  uint64_t t;
#ifdef MERLIN_PROF_ZEPHYR
  /* mcycle, not rdtime: this runs in M-mode on a bare SoC where the platform timer may be a slow
   * CLINT reference (50 kHz on one of these chips -- coarser than most individual ops), while mcycle
   * is the same counter METRIC cycles already reports, so per-op ticks and the whole-model wall stay
   * directly comparable. */
  __asm__ volatile("csrr %0, mcycle" : "=r"(t));
#else
  __asm__ volatile("rdtime %0" : "=r"(t));
#endif
  return t;
}

#if defined(MERLIN_PROF_ZEPHYR) && defined(MERLIN_PROF_HEARTBEAT_MS)
#include <zephyr/kernel.h>
#include <zephyr/arch/cpu.h>
#include <stdbool.h>

static int64_t merlin_prof_next_beat;

/* "Still running, and here is where" -- printed from the model's own thread, between two top-level
 * ops, at most every MERLIN_PROF_HEARTBEAT_MS.
 *
 * The first design was a k_timer, on the reasoning that a timer fires regardless of what the
 * scheduler is doing. It does -- and its printk CRASHED the run: this SoC's console is HTIF, whose
 * buffered/syscall-proxy path is not reentrant from interrupt context, and the corrupted `tohost`
 * word came back as `bad syscall #1243416269594910946`. So the beat happens where printing is already
 * known to be safe. A low-priority thread does not work either: the model runs on a pinned
 * COOPERATIVE worker that never yields, so a preemptible heartbeat is simply never scheduled
 * (measured: zero lines across a 100-second run).
 *
 * What this trades away: it cannot report from INSIDE a single op. That is the right trade -- the
 * last line still names the op it entered, which is the actual question ("where did it stop"), and it
 * costs nothing when the model is progressing normally.
 */
static void merlin_prof_beat(int32_t id) {
  static bool beat_once;
  int64_t now_ms = k_uptime_get();
  unsigned long ms;

  /* ALWAYS emit the first one, however fast the run is.
   *
   * Rate-limiting alone means a model that finishes inside one interval prints no ALIVE at all --
   * measured on the FPGA, where the whole inference is 1.4 seconds of simulated time against a 5-second
   * beat. Zero lines is then ambiguous in the worst way: it looks identical to a heartbeat that is
   * broken or was never linked in, which is exactly the doubt this line exists to remove. One
   * unconditional beat makes ABSENCE meaningful -- if you see none at all, the mechanism really is
   * dead, and that is worth knowing too.
   */
  if (beat_once && now_ms < merlin_prof_next_beat) {
    return;
  }
  beat_once = true;
  merlin_prof_next_beat = now_ms + (int64_t)MERLIN_PROF_HEARTBEAT_MS;
  __asm__ volatile("csrr %0, mstatus" : "=r"(ms));
  printk("ALIVE t=%lld op=%d hart=%d vs=%u\n", (long long)(now_ms / 1000), (int)id,
         arch_curr_cpu()->id, (unsigned)((ms >> 9) & 3));
}
#else
#define merlin_prof_beat(id) ((void)0)
#endif

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
  merlin_prof_last_id = id;                    /* published for the heartbeat, not just for us */
  merlin_prof_beat(id);
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
    MERLIN_PROF_PRINT("PROF %d %llu %u\n", i, (unsigned long long)merlin_prof_ticks[i],
           (unsigned)merlin_prof_hits[i]);
  }
  if (merlin_prof_overflow_hits)
    MERLIN_PROF_PRINT("PROF -1 %llu %u\n", (unsigned long long)merlin_prof_overflow_ticks,
           (unsigned)merlin_prof_overflow_hits);
  MERLIN_PROF_PRINT("METRIC prof_total_ticks %llu\n", (unsigned long long)total);
  MERLIN_PROF_PRINT("METRIC prof_marks %llu\n", (unsigned long long)merlin_prof_marks);
}
