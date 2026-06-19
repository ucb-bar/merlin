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

// rdtime: the delegated platform timebase counter (24 MHz on this K1). Used as the
// drivers' "cycle" source so the inner-compute timing path is identical to spike's,
// just with a real-silicon tick instead of a functional-proxy mcycle.
static inline unsigned long merlin_k1_rdtime(void) {
  unsigned long t;
  __asm__ volatile("rdtime %0" : "=r"(t));
  return t;
}

// The drivers call read_csr(mcycle) / read_csr(minstret). Map both here. minstret is
// not available in userspace on this kernel, so it reads 0 (the drivers' INSTRET line
// will be 0 on K1 — honest: we do not have a retired-instruction count on the board).
#define read_csr(reg) merlin_k1_read_csr_##reg()
static inline unsigned long merlin_k1_read_csr_mcycle(void)   { return merlin_k1_rdtime(); }
static inline unsigned long merlin_k1_read_csr_minstret(void) { return 0UL; }

#endif // MERLIN_K1_CEILING_UTIL_H
