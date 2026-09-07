#define _GNU_SOURCE
/* Persistent pthread implementation of the OpenMP ABI subset emitted by Merlin.
 * See libomp_pthread.h. */
#include "libomp_pthread.h"
#include "omp_static_schedule.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MERLIN_OMP_MAX_SHARED 16
#ifndef MERLIN_OMP_PTHREAD_WORKER_STACK
#define MERLIN_OMP_PTHREAD_WORKER_STACK (16u * 1024u * 1024u)
#endif

typedef struct {
  int32_t reserved_1, flags, reserved_2, reserved_3;
  const char *psource;
} ident_t;

typedef void (*micro0_t)(int32_t *, int32_t *);
typedef void (*micro1_t)(int32_t *, int32_t *, void *);
typedef void (*micro2_t)(int32_t *, int32_t *, void *, void *);
typedef void (*micro3_t)(int32_t *, int32_t *, void *, void *, void *);
typedef void (*micro4_t)(int32_t *, int32_t *, void *, void *, void *, void *);
typedef void (*micro5_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *);
typedef void (*micro6_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *);
typedef void (*micro7_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro8_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro9_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro10_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro11_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro12_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro13_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro14_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro15_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*micro16_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);

struct worker {
  pthread_t thread;
  sem_t start;
  sem_t done;
  int slot;
  int observed_cpu;
} __attribute__((aligned(64)));
static struct worker workers[MERLIN_OMP_PTHREAD_MAX_THREADS];
static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t dispatch_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t critical_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t ready_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready_cv = PTHREAD_COND_INITIALIZER;
static int initialized;
static int pool_size = 1;
static int master_cpu = -1;
static int started;
static void *task_fn;
static int task_argc, task_team;
static void *task_args[MERLIN_OMP_MAX_SHARED];

static _Thread_local int tls_tid;
static _Thread_local int tls_team = 1;
static _Thread_local int tls_depth;
static _Thread_local int tls_requested;

static void fail(const char *why) {
  fprintf(stderr, "FAIL merlin pthread OpenMP runtime: %s\n", why);
  abort();
}

static int observed_cpu(void) {
#ifdef __linux__
  return sched_getcpu();
#else
  return -1;
#endif
}

static int allowed_cpus(int *cpus, int capacity) {
#ifdef __linux__
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    fail("sched_getaffinity failed");
  int count = 0;
  for (int cpu = 0; cpu < CPU_SETSIZE && count < capacity; ++cpu)
    if (CPU_ISSET(cpu, &allowed)) cpus[count++] = cpu;
  if (count == 0) fail("process affinity mask contains no CPUs");
  return count;
#else
  long count = sysconf(_SC_NPROCESSORS_ONLN);
  if (count < 1) count = 1;
  if (count > capacity) count = capacity;
  for (int cpu = 0; cpu < count; ++cpu) cpus[cpu] = cpu;
  return (int)count;
#endif
}

static void call_micro(void *fn, int argc, void **args, int32_t tid) {
  int32_t bound = 0;
  switch (argc) {
    case 0: ((micro0_t)fn)(&tid, &bound); break;
    case 1: ((micro1_t)fn)(&tid, &bound, args[0]); break;
    case 2: ((micro2_t)fn)(&tid, &bound, args[0], args[1]); break;
    case 3: ((micro3_t)fn)(&tid, &bound, args[0], args[1], args[2]); break;
    case 4: ((micro4_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3]); break;
    case 5: ((micro5_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4]); break;
    case 6: ((micro6_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5]); break;
    case 7: ((micro7_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6]); break;
    case 8: ((micro8_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]); break;
    case 9: ((micro9_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]); break;
    case 10: ((micro10_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]); break;
    case 11: ((micro11_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]); break;
    case 12: ((micro12_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]); break;
    case 13: ((micro13_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]); break;
    case 14: ((micro14_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]); break;
    case 15: ((micro15_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]); break;
    case 16: ((micro16_t)fn)(&tid, &bound, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15]); break;
    default: fail("outlined region has more than 16 shared arguments");
  }
}

static void run_region(int tid, int team, void *fn, int argc, void **args) {
  int old_tid = tls_tid, old_team = tls_team, old_depth = tls_depth;
  tls_tid = tid; tls_team = team; tls_depth = old_depth + 1;
  call_micro(fn, argc, args, (int32_t)tid);
  tls_tid = old_tid; tls_team = old_team; tls_depth = old_depth;
}

static void *worker_main(void *opaque) {
  struct worker *worker = (struct worker *)opaque;
  pthread_mutex_lock(&ready_mu);
  worker->observed_cpu = observed_cpu();
  ++started;
  pthread_cond_signal(&ready_cv);
  pthread_mutex_unlock(&ready_mu);
  for (;;) {
    while (sem_wait(&worker->start) != 0)
      if (errno != EINTR) fail("worker semaphore wait failed");
    int team = task_team, argc = task_argc;
    void *fn = task_fn;
    void *args[MERLIN_OMP_MAX_SHARED];
    for (int i = 0; i < argc; ++i) args[i] = task_args[i];
    run_region(worker->slot, team, fn, argc, args);
    if (sem_post(&worker->done) != 0) fail("worker completion semaphore post failed");
  }
  return NULL;
}

int merlin_omp_init(int requested) {
  int cpus[MERLIN_OMP_PTHREAD_MAX_THREADS];
  int cpu_count = allowed_cpus(cpus, MERLIN_OMP_PTHREAD_MAX_THREADS);
  if (requested < 1) requested = 1;
  if (requested > MERLIN_OMP_PTHREAD_MAX_THREADS)
    requested = MERLIN_OMP_PTHREAD_MAX_THREADS;
  if (requested > cpu_count) requested = cpu_count;
  pthread_mutex_lock(&init_mu);
  if (!initialized) {
    pthread_attr_t attr;
    int attr_ok = pthread_attr_init(&attr) == 0;
    if (attr_ok && pthread_attr_setstacksize(&attr, MERLIN_OMP_PTHREAD_WORKER_STACK) != 0)
      attr_ok = 0;
    pool_size = 1;
    started = 0;
#ifdef __linux__
    cpu_set_t master_set;
    CPU_ZERO(&master_set);
    CPU_SET(cpus[0], &master_set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(master_set), &master_set) != 0)
      fail("could not pin OpenMP master thread");
#endif
    master_cpu = observed_cpu();
    if (attr_ok) {
      for (int slot = 1; slot < requested; ++slot) {
        workers[slot].slot = slot;
        workers[slot].observed_cpu = -1;
        if (sem_init(&workers[slot].start, 0, 0) != 0 ||
            sem_init(&workers[slot].done, 0, 0) != 0)
          fail("could not initialize worker semaphores");
#ifdef __linux__
        cpu_set_t worker_set;
        CPU_ZERO(&worker_set);
        CPU_SET(cpus[slot], &worker_set);
        if (pthread_attr_setaffinity_np(&attr, sizeof(worker_set), &worker_set) != 0)
          fail("could not set OpenMP worker affinity");
#endif
        if (pthread_create(&workers[slot].thread, &attr, worker_main, &workers[slot]) != 0)
          break;
        pool_size = slot + 1;
      }
      pthread_attr_destroy(&attr);
    }
    pthread_mutex_lock(&ready_mu);
    while (started != pool_size - 1) pthread_cond_wait(&ready_cv, &ready_mu);
    pthread_mutex_unlock(&ready_mu);
    initialized = 1;
  }
  int result = pool_size;
  pthread_mutex_unlock(&init_mu);
  return result;
}

int merlin_omp_init_from_env(int fallback_threads) {
  int requested = fallback_threads;
  const char *value = getenv("OMP_NUM_THREADS");
  if (value && *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < 1 || parsed > INT_MAX)
      fail("OMP_NUM_THREADS must be a positive integer");
    requested = (int)parsed;
  }
  return merlin_omp_init(requested);
}

int merlin_omp_num_threads(void) {
  pthread_mutex_lock(&init_mu);
  int result = initialized ? pool_size : 0;
  pthread_mutex_unlock(&init_mu);
  return result;
}

int merlin_omp_worker_cpu(int tid) {
  pthread_mutex_lock(&init_mu);
  int result = -1;
  if (initialized && tid >= 0 && tid < pool_size)
    result = tid == 0 ? master_cpu : workers[tid].observed_cpu;
  pthread_mutex_unlock(&init_mu);
  return result;
}

int32_t __kmpc_global_thread_num(ident_t *loc) { (void)loc; return tls_tid; }

void __kmpc_push_num_threads(ident_t *loc, int32_t gtid, int32_t n) {
  (void)loc; (void)gtid; tls_requested = n;
}

void __kmpc_fork_call(ident_t *loc, int32_t argc, void *microtask, ...) {
  (void)loc;
  if (argc < 0 || argc > MERLIN_OMP_MAX_SHARED) fail("invalid outlined-region arity");
  void *args[MERLIN_OMP_MAX_SHARED];
  va_list ap;
  va_start(ap, microtask);
  for (int i = 0; i < argc; ++i) args[i] = va_arg(ap, void *);
  va_end(ap);

  if (tls_depth > 0) {
    tls_requested = 0;
    run_region(0, 1, microtask, argc, args);
    return;
  }
  int requested = tls_requested;
  tls_requested = 0;
  if (requested < 1) {
    requested = merlin_omp_init_from_env(1);
  }
  int available = merlin_omp_init(requested);
  int team = requested < available ? requested : available;
  if (team < 1) team = 1;
  if (team == 1) { run_region(0, 1, microtask, argc, args); return; }

  /* One caller owns the published task until every worker completes.  This also makes
   * simultaneous application-thread entry deterministic instead of corrupting globals. */
  pthread_mutex_lock(&dispatch_mu);
  task_fn = microtask; task_argc = argc; task_team = team;
  for (int i = 0; i < argc; ++i) task_args[i] = args[i];
  /* Wake only helpers selected for this dispatch, avoiding a condition-variable broadcast to
   * workers outside a smaller requested team. */
  for (int slot = 1; slot < team; ++slot)
    if (sem_post(&workers[slot].start) != 0) fail("worker semaphore post failed");
  run_region(0, team, microtask, argc, args);
  for (int slot = 1; slot < team; ++slot)
    while (sem_wait(&workers[slot].done) != 0)
      if (errno != EINTR) fail("worker completion semaphore wait failed");
  pthread_mutex_unlock(&dispatch_mu);
}

void __kmpc_barrier(ident_t *loc, int32_t gtid) { (void)loc; (void)gtid; }
void __kmpc_critical(ident_t *loc, int32_t gtid, void *crit) {
  (void)loc; (void)gtid; (void)crit; pthread_mutex_lock(&critical_mu);
}
void __kmpc_end_critical(ident_t *loc, int32_t gtid, void *crit) {
  (void)loc; (void)gtid; (void)crit; pthread_mutex_unlock(&critical_mu);
}

static void static_init(int32_t tid, int32_t sched, int32_t *last, int64_t *lo,
                        int64_t *hi, int64_t *stride, int64_t incr) {
  if (sched != MERLIN_KMP_SCH_STATIC_CHUNKED && sched != MERLIN_KMP_SCH_STATIC)
    fail("only static OpenMP worksharing schedules are supported");
  (void)merlin_omp_static_split(tid, tls_team, lo, hi, stride, incr, last);
}

void __kmpc_for_static_init_8(ident_t *loc, int32_t tid, int32_t sched, int32_t *last,
  int64_t *lo, int64_t *hi, int64_t *stride, int64_t incr, int64_t chunk) {
  (void)loc; (void)chunk; static_init(tid, sched, last, lo, hi, stride, incr);
}
void __kmpc_for_static_init_8u(ident_t *loc, int32_t tid, int32_t sched, int32_t *last,
  uint64_t *lo, uint64_t *hi, int64_t *stride, int64_t incr, int64_t chunk) {
  int64_t l = (int64_t)*lo, h = (int64_t)*hi;
  (void)loc; (void)chunk; static_init(tid, sched, last, &l, &h, stride, incr);
  *lo = (uint64_t)l; *hi = (uint64_t)h;
}
void __kmpc_for_static_init_4(ident_t *loc, int32_t tid, int32_t sched, int32_t *last,
  int32_t *lo, int32_t *hi, int32_t *stride, int32_t incr, int32_t chunk) {
  int64_t l = *lo, h = *hi, s = 0;
  (void)loc; (void)chunk; static_init(tid, sched, last, &l, &h, &s, incr);
  *lo = (int32_t)l; *hi = (int32_t)h; *stride = (int32_t)s;
}
void __kmpc_for_static_init_4u(ident_t *loc, int32_t tid, int32_t sched, int32_t *last,
  uint32_t *lo, uint32_t *hi, int32_t *stride, int32_t incr, int32_t chunk) {
  int64_t l = *lo, h = *hi, s = 0;
  (void)loc; (void)chunk; static_init(tid, sched, last, &l, &h, &s, incr);
  *lo = (uint32_t)l; *hi = (uint32_t)h; *stride = (int32_t)s;
}
void __kmpc_for_static_fini(ident_t *loc, int32_t tid) { (void)loc; (void)tid; }

int omp_get_num_threads(void) { return tls_team; }
int omp_get_thread_num(void) { return tls_tid; }
int omp_get_max_threads(void) { int n = merlin_omp_num_threads(); return n > 0 ? n : 1; }
