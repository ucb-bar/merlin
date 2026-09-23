// K1 (SpacemiT, Bianbu glibc Linux) replacement for the saturn bare-metal `util.h`
// used by the ceiling drivers (openblas_sgemm_driver.c / xnnpack_gemm_driver.c /
// ours_gemm_driver.c). The drivers are COMPILED UNCHANGED — this header is placed
// FIRST on the include path so their `#include "util.h"` picks it up instead of the
// saturn one.
//
// On the K1 the userspace `cycle` CSR (mcycle/minstret) TRAPS as illegal (the Bianbu
// kernel does not delegate it). The `time` CSR IS delegated, so we map the drivers'
// `read_csr(mcycle)` to the platform `rdtime` counter (a fixed 24 MHz timebase, NOT
// core cycles) and `read_csr(minstret)` to 0 (instret is unavailable in userspace).
// So the K1 ceiling numbers are REAL-SILICON rdtime TICKS (inner-compute, same scope
// as spike's mcycle proxy) — reported as cycle_accurate=false; spike/FireSim remain the
// cycle-accurate authorities. printf comes from glibc (hosted Linux, not HTIF).
#ifndef MERLIN_K1_CEILING_UTIL_H
#define MERLIN_K1_CEILING_UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <linux/perf_event.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

// rdtime: the delegated platform timebase counter (24 MHz on this K1). Used as the
// drivers' "cycle" source so the inner-compute timing path is identical to spike's,
// just with a real-silicon tick instead of a functional-proxy mcycle.
static inline unsigned long merlin_k1_rdtime(void) {
  unsigned long t;
  __asm__ volatile("rdtime %0" : "=r"(t));
  return t;
}

// RETIRED-INSTRUCTION COUNT. The `minstret` CSR traps in userspace on this kernel, but the
// board's PMU is exposed via perf_event_open, so we self-monitor (pid=0) and read the
// hardware instruction counter instead of reporting 0.
//
// WHY THIS MATTERS: rdtime alone cannot tell the two ways a kernel is slow apart — emitting
// too many instructions (a codegen problem a schedule can fix) versus stalling on each one (a
// memory/dependency problem it cannot). Reading instret on the SAME bracket as rdtime yields
// instructions-per-tick, which separates them. Process-wide sampling cannot: these drivers
// spend most of their process in the scalar verification reference, which swamps the kernel.
//
// Fail-closed: if the counter cannot be opened, reads return 0 — the same honest "unavailable"
// the CSR mapping reported before, never a fabricated count.
// Opened EAGERLY in a constructor rather than lazily on first read: a lazy `static` inside a
// header-inlined function is per-translation-unit and proved fragile across the driver builds
// (some reported 0 while others counted correctly). One eager open, one flag, same answer
// everywhere.
static int merlin_k1_instret_fd_v = -1;

__attribute__((constructor)) static void merlin_k1_instret_init(void) {
  struct perf_event_attr a;
  memset(&a, 0, sizeof a);
  a.type = PERF_TYPE_HARDWARE;
  a.size = sizeof a;
  a.config = PERF_COUNT_HW_INSTRUCTIONS;
  a.disabled = 0;
  a.exclude_kernel = 1;
  a.exclude_hv = 1;
  merlin_k1_instret_fd_v = (int)syscall(__NR_perf_event_open, &a, 0 /*self*/, -1, -1, 0);
}

// The drivers print INSTRET unconditionally, and a 0 there is ambiguous between "no instructions"
// and "counter unavailable". Print the status once so a silent 0 can never be misread as a
// measurement (the same fail-closed rule the runner applies to timings).
__attribute__((destructor)) static void merlin_k1_instret_report(void) {
  printf("INSTRET_COUNTER %s\n", merlin_k1_instret_fd_v >= 0 ? "available" : "unavailable");
}

// The drivers call read_csr(mcycle) / read_csr(minstret) and are compiled UNCHANGED.
#define read_csr(reg) merlin_k1_read_csr_##reg()
static inline unsigned long merlin_k1_read_csr_mcycle(void)   { return merlin_k1_rdtime(); }
static inline unsigned long merlin_k1_read_csr_minstret(void) {
  int fd = merlin_k1_instret_fd_v;
  long long v = 0;
  if (fd < 0 || read(fd, &v, sizeof v) != (ssize_t)sizeof v) return 0UL;
  return (unsigned long)v;
}

#endif // MERLIN_K1_CEILING_UTIL_H
