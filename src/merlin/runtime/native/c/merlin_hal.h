/* Merlin HAL — the frozen, tiny hardware-abstraction seam the lean replay runtime
 * (merlin_program.c) and generated code call. Each platform implements it ONCE:
 *   - hal_linux.c            (pthread / glibc)        — K1 Linux, host
 *   - hal_baremetal_spike.c  (bump arena + HTIF)      — spike bare-metal
 *   - hal_zephyr.c           (k_thread / SMP)         — RTOS  (phase 3)
 *   - hal_simt.c             (grid launch)            — SIMT  (phase 3)
 *
 * Design goals (the whole point of this runtime):
 *   - HW-AGNOSTIC: generated code + the replay engine NEVER call libc/OS directly; they call
 *     this seam, so retargeting to a custom-ISA accelerator / SIMT core / research HW = implement
 *     this header once. No libomp, no hardcoded pthread, no platform #ifdef in the engine.
 *   - LEAN: one static arena (no per-op malloc); single + multi-core share ONE parallel_for path
 *     (single-core = the trivial serial impl); bare-metal/RTOS friendly (no heap, no stdio needed).
 *   - EXTENSIBLE: capability flags let a target advertise what it supports; SIMT gets the same
 *     seam via merlin_hal_launch (grid/block).
 */
#ifndef MERLIN_HAL_H
#define MERLIN_HAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- arena: ONE contiguous region the compiler statically planned (memory_plan.arena_bytes).
 * Intermediates bind to merlin_hal_arena_base() + offset; reset reclaims the whole arena between
 * inferences (no free()). The compiler guarantees arena_bytes <= the region the platform provides. */
void       merlin_hal_arena_init(size_t bytes);   /* provision the arena (>= program arena_bytes) */
uint8_t   *merlin_hal_arena_base(void);           /* base pointer; intermediates = base + offset   */
size_t     merlin_hal_arena_size(void);
void       merlin_hal_arena_reset(void);          /* reclaim the whole arena (between inferences)   */

/* ---- bulk memory (the engine uses these for view ops / I/O; libc-free targets supply their own) */
void      *merlin_hal_memcpy(void *dst, const void *src, size_t n);
void      *merlin_hal_memset(void *dst, int c, size_t n);

/* ---- timing + diagnostics (mcycle / clock_gettime / rdtime — platform's monotonic tick). */
uint64_t   merlin_hal_now_ticks(void);
void       merlin_hal_log(const char *msg);       /* one line; printf/printk/HTIF puts            */

/* ---- parallelism: ONE seam for single + multi-core. The engine calls parallel_for over a kernel's
 * outer iteration range; a worker runs body(i, ctx) for i in [lo, hi). n_workers<=1 OR a platform
 * with no threads runs it serially (correct, just not parallel). SIMT targets use merlin_hal_launch
 * (grid/block) instead — the same dispatch seam, different shape. */
typedef void (*merlin_hal_body_fn)(int64_t i, void *ctx);
void merlin_hal_parallel_for(int64_t lo, int64_t hi, merlin_hal_body_fn body, void *ctx,
                             int n_workers);

/* SIMT launch (phase 3): grid x block invocation of a per-lane kernel. Hosts without SIMT may
 * lower it to a parallel_for over the flattened grid (so the engine is uniform). */
typedef void (*merlin_hal_lane_fn)(int64_t gx, int64_t gy, int64_t gz,
                                   int64_t lx, int64_t ly, int64_t lz, void *ctx);
void merlin_hal_launch(int64_t gx, int64_t gy, int64_t gz,
                       int64_t bx, int64_t by, int64_t bz,
                       merlin_hal_lane_fn lane, void *ctx);

/* ---- capability flags: a target advertises what it supports; the replay/adapter fails CLOSED on
 * a program that needs a capability the target lacks (honest, never silent-wrong). */
enum {
  MERLIN_CAP_SCALAR   = 1u << 0,
  MERLIN_CAP_RVV      = 1u << 1,
  MERLIN_CAP_THREADS  = 1u << 2,   /* merlin_hal_parallel_for is genuinely concurrent             */
  MERLIN_CAP_SIMT     = 1u << 3,   /* merlin_hal_launch maps to real grid/block hardware          */
  MERLIN_CAP_CUSTOM   = 1u << 16,  /* base for target-specific capability bits (custom ISA)       */
};
uint32_t   merlin_hal_capabilities(void);
int        merlin_hal_has(uint32_t cap);   /* (capabilities() & cap) == cap */

#ifdef __cplusplus
}
#endif
#endif /* MERLIN_HAL_H */
