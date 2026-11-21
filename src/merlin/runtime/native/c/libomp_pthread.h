/* Hosted OpenMP ABI provider used by Merlin's persistent-worker-pool policy.
 *
 * This is intentionally a small runtime for the exact __kmpc_* subset emitted by
 * Merlin's MLIR pipeline, not a general OpenMP implementation.  Initialization is
 * explicit or lazy; there are no ELF constructors/destructors.  Workers remain alive
 * until process exit, so repeated inference does not pay pthread create/join per region.
 */
#ifndef MERLIN_LIBOMP_PTHREAD_H
#define MERLIN_LIBOMP_PTHREAD_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MERLIN_OMP_PTHREAD_MAX_THREADS
#define MERLIN_OMP_PTHREAD_MAX_THREADS 64
#endif

/* Idempotently create a process-local team of n_threads including the caller/master.
 * A later request cannot resize an already active pool.  Returns the available team size. */
int merlin_omp_init(int n_threads);
/* Initialize from OMP_NUM_THREADS, or fallback_threads when it is unset. */
int merlin_omp_init_from_env(int fallback_threads);
int merlin_omp_num_threads(void);
/* CPU observed for the master (tid 0) or helper.  Returns -1 for an invalid tid. */
int merlin_omp_worker_cpu(int tid);

#ifdef __cplusplus
}
#endif
#endif
