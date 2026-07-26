/* Minimal OpenMP runtime for Zephyr SMP — the multicore seam for merlin's RVV models.
 *
 * merlin's multicore lowering (llvmlower/pipeline.py, `parallel_harts=N`) wraps each
 * contraction in an `scf.forall`, which becomes `omp.parallel` + `omp.wsloop` and finally
 * a handful of `__kmpc_*` calls. On Linux/K1 those are satisfied by a real cross-built
 * libomp (build_tools/k1_openmp); Zephyr has no OpenMP runtime at all, so this file
 * provides exactly the symbol set that lowering emits, backed by a Zephyr thread pool.
 *
 * ONLY the symbols merlin's own IR references are implemented. That set is not guessed:
 * it is pinned by merlin/tests/rvv/test_rvv_parallel_pipeline.py, which compiles a matmul
 * through the real pipeline and asserts on the emitted declarations. This is NOT a general
 * OpenMP implementation — no dynamic/guided schedules, no nesting, no tasks, no reductions
 * beyond what the static worksharing loop needs.
 *
 * THREADING MODEL (this is a correctness constraint, not a tuning choice):
 *   one COOP-priority worker per hart, pinned 1:1 with k_thread_cpu_pin, created once and
 *   reused. Never oversubscribed, never preempted.
 *
 * Why it must be that way: the Saturn fork's arch/riscv/core/v.c has a documented vector
 * save/restore bug under preemption — samples/merlin_model_runner/prj.conf records yolov8n
 * failing bit-exactness with two DIFFERENT wrong hashes across two Kconfig variants, and
 * their workaround was to disable V entirely. Pinned, non-preemptible COOP workers never
 * hit the buggy context-switch path, which is the same argument that makes merlin's
 * existing single pinned worker safe, and the configuration under which the proven
 * zephyr-chipyard-sw samples (tiled_matmul_mt_pool, merlin_mt_rvv_dispatch) pass with V
 * enabled. merlin/tests/runtime/test_zephyr_multicore.py gates on 1-hart vs N-hart
 * BIT-EXACT equality, which is the detector if that assumption ever breaks.
 */
#ifndef MERLIN_LIBOMP_ZEPHYR_H
#define MERLIN_LIBOMP_ZEPHYR_H

#include <stdint.h>

/* Maximum harts this shim will fan out to. Raise together with CONFIG_MP_MAX_NUM_CPUS. */
#ifndef MERLIN_OMP_MAX_THREADS
#define MERLIN_OMP_MAX_THREADS 8
#endif

/* Per-worker stack. The lowered model's large intermediates are alloca'd in forward() on
 * the MASTER thread (buffer-results-to-out-params{hoist-static-allocs}); a worker only runs
 * the outlined loop body, so this is modest by design. */
#ifndef MERLIN_OMP_WORKER_STACK
#define MERLIN_OMP_WORKER_STACK (256 * 1024)
#endif

/* Start the pool with `n_threads` total (master + n_threads-1 pinned workers).
 * Idempotent; call once before the first parallel region so pool spin-up is not charged to
 * inference time. Returns the number of threads actually available (clamped to the number
 * of CPUs Zephyr reports and to MERLIN_OMP_MAX_THREADS), or a negative errno on failure. */
int merlin_omp_init(int n_threads);

/* Threads the pool will actually fan out to (0 before merlin_omp_init). */
int merlin_omp_num_threads(void);

#endif /* MERLIN_LIBOMP_ZEPHYR_H */
