/* Minimal OpenMP runtime for Zephyr SMP. See libomp_zephyr.h for the threading model and
 * why pinned COOP workers are a correctness requirement (the v.c V-state bug), not a tuning
 * choice.
 *
 * The implemented surface is exactly what merlin's multicore lowering emits:
 *   __kmpc_global_thread_num, __kmpc_fork_call, __kmpc_for_static_init_{4,4u,8,8u},
 *   __kmpc_for_static_fini, __kmpc_barrier, __kmpc_push_num_threads,
 *   __kmpc_critical / __kmpc_end_critical, omp_get_{num_threads,thread_num,max_threads}.
 * Signatures are taken from the actual emitted declarations, e.g.
 *   declare void @__kmpc_for_static_init_8u(ptr, i32, i32, ptr, ptr, ptr, ptr, i64, i64)
 *   declare void @__kmpc_fork_call(ptr, i32, ptr, ...)
 */
#include "libomp_zephyr.h"
#include "omp_static_schedule.h"

#include <stdarg.h>
#include <stddef.h>

#include <zephyr/kernel.h>
#include <zephyr/arch/cpu.h>
#include <zephyr/sys/printk.h>

/* ---- OpenMP ABI types ------------------------------------------------------------- */

/* %struct.ident_t = type { i32, i32, i32, i32, ptr } — source location; unused here. */
typedef struct {
	int32_t reserved_1;
	int32_t flags;
	int32_t reserved_2;
	int32_t reserved_3;
	const char *psource;
} ident_t;

/* The outlined region: microtask(&gtid, &bound_tid, shared...). */
typedef void (*kmpc_micro0_t)(int32_t *, int32_t *);
typedef void (*kmpc_micro1_t)(int32_t *, int32_t *, void *);
typedef void (*kmpc_micro2_t)(int32_t *, int32_t *, void *, void *);
typedef void (*kmpc_micro3_t)(int32_t *, int32_t *, void *, void *, void *);
typedef void (*kmpc_micro4_t)(int32_t *, int32_t *, void *, void *, void *, void *);

#define MERLIN_OMP_MAX_SHARED 4

/* ---- pool state --------------------------------------------------------------------- */

struct omp_worker {
	struct k_thread thread;
	struct k_sem start;
	struct k_sem done;
	int32_t gtid;
};

K_THREAD_STACK_ARRAY_DEFINE(merlin_omp_stacks, MERLIN_OMP_MAX_THREADS,
			    MERLIN_OMP_WORKER_STACK);

static struct omp_worker omp_workers[MERLIN_OMP_MAX_THREADS];
static struct k_mutex omp_critical;

static int omp_nthreads;              /* total incl. master; 0 until merlin_omp_init */
static int omp_started;               /* pool threads created */
static int omp_requested;             /* __kmpc_push_num_threads for the NEXT region */

/* The region currently being executed. Only the master writes these, and only while every
 * worker is parked on its `start` semaphore, so no lock is needed. */
static void *omp_task_fn;
static int omp_task_argc;
static void *omp_task_args[MERLIN_OMP_MAX_SHARED];
static int omp_task_nthreads;

/* hart id -> OpenMP thread id. This is CORRECTNESS-CRITICAL, not bookkeeping: the emitted
 * IR does NOT pass the outlined region's %tid argument to the worksharing loop, it calls
 * __kmpc_global_thread_num() and hands THAT to __kmpc_for_static_init_*:
 *
 *   %n = call i32 @__kmpc_global_thread_num(ptr @1)
 *   call void @__kmpc_for_static_init_8u(ptr @2, i32 %n, i32 34, ...)
 *
 * so if this mapping is wrong, two harts claim the same slice (and another claims none) and
 * the model silently computes the wrong answer. Deriving it from the running hart is exact
 * ONLY because workers are pinned 1:1 — a third reason pinning is required, alongside the
 * v.c V-state bug. Zephyr has no portable __thread here, and a lookup by k_current_get()
 * would be a search on the hot path; a hart-indexed table is O(1) and exact. */
static int8_t omp_gtid_of_hart[MERLIN_OMP_MAX_THREADS];
static int omp_master_cpu = -1;

static inline int omp_self_gtid(void)
{
	int cpu = (int)arch_curr_cpu()->id;

	if (cpu < 0 || cpu >= MERLIN_OMP_MAX_THREADS) {
		return 0;
	}
	return (int)omp_gtid_of_hart[cpu];
}

/* mstatus.VS/FS = Dirty. The proven zephyr-chipyard-sw samples set this in each V-using
 * worker entry (tiled_matmul_mt_pool's enable_vector_operations, merlin_hetero_runner's
 * set_vs_dirty): a freshly created thread can start with VS=Off, and the first vector
 * instruction would then trap. Harmless when the state is already dirty. */
#define MSTATUS_FS 0x00006000UL
#define MSTATUS_VS 0x00000600UL

static inline void omp_enable_vector(void)
{
	unsigned long mstatus;

	__asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
	mstatus |= MSTATUS_VS | MSTATUS_FS;
	__asm__ volatile("csrw mstatus, %0" ::"r"(mstatus));
}

static void omp_run_region(int32_t gtid)
{
	int32_t tid = gtid;
	int32_t bound = 0;
	void **a = omp_task_args;

	switch (omp_task_argc) {
	case 0:
		((kmpc_micro0_t)omp_task_fn)(&tid, &bound);
		break;
	case 1:
		((kmpc_micro1_t)omp_task_fn)(&tid, &bound, a[0]);
		break;
	case 2:
		((kmpc_micro2_t)omp_task_fn)(&tid, &bound, a[0], a[1]);
		break;
	case 3:
		((kmpc_micro3_t)omp_task_fn)(&tid, &bound, a[0], a[1], a[2]);
		break;
	case 4:
		((kmpc_micro4_t)omp_task_fn)(&tid, &bound, a[0], a[1], a[2], a[3]);
		break;
	default:
		/* Fail loudly: silently running the region with the wrong argument count would
		 * corrupt results rather than crash. */
		printk("FAIL merlin_omp: %d shared args > MERLIN_OMP_MAX_SHARED %d\n",
		       omp_task_argc, MERLIN_OMP_MAX_SHARED);
		k_panic();
	}
}

static void omp_worker_entry(void *p0, void *p1, void *p2)
{
	struct omp_worker *w = (struct omp_worker *)p0;
	int slot = (int)(intptr_t)p1;

	ARG_UNUSED(p2);
	omp_enable_vector();
	for (;;) {
		k_sem_take(&w->start, K_FOREVER);
		if (slot < omp_task_nthreads) {
			omp_run_region((int32_t)slot);
		}
		k_sem_give(&w->done);
	}
}

int merlin_omp_num_threads(void)
{
	return omp_nthreads;
}

int merlin_omp_init(int n_threads)
{
	int cpus = arch_num_cpus();

	if (omp_started) {
		return omp_nthreads;
	}
	if (n_threads < 1) {
		n_threads = 1;
	}
	if (n_threads > MERLIN_OMP_MAX_THREADS) {
		n_threads = MERLIN_OMP_MAX_THREADS;
	}
	if (n_threads > cpus) {
		/* More threads than harts would mean two V-using threads sharing a hart, i.e.
		 * exactly the preemption case the v.c bug lives in. Clamp instead. */
		printk("merlin_omp: clamping %d threads to %d CPUs\n", n_threads, cpus);
		n_threads = cpus;
	}

	omp_master_cpu = (int)arch_curr_cpu()->id;
	k_mutex_init(&omp_critical);
	omp_nthreads = n_threads;
	for (int i = 0; i < MERLIN_OMP_MAX_THREADS; i++) {
		omp_gtid_of_hart[i] = 0;
	}
	omp_gtid_of_hart[omp_master_cpu] = 0;   /* the master is OpenMP thread 0 */

	for (int i = 1; i < n_threads; i++) {
		struct omp_worker *w = &omp_workers[i];
		/* Pin worker i to a hart that is NOT the master's. Harts are numbered 0..cpus-1;
		 * walk them skipping the master's so the mapping stays 1:1 even when the master
		 * is not on hart 0 (the FireSim vector-tile case pins it to hart 1). */
		int cpu = (i <= omp_master_cpu) ? (i - 1) : i;

		k_sem_init(&w->start, 0, 1);
		k_sem_init(&w->done, 0, 1);
		w->gtid = cpu;
		omp_gtid_of_hart[cpu] = (int8_t)i;

		k_tid_t t = k_thread_create(&w->thread, merlin_omp_stacks[i],
					    MERLIN_OMP_WORKER_STACK, omp_worker_entry,
					    w, (void *)(intptr_t)i, NULL,
					    K_PRIO_COOP(0), 0, K_FOREVER);
		int rc = k_thread_cpu_pin(t, cpu);

		if (rc != 0) {
			printk("FAIL merlin_omp: k_thread_cpu_pin worker %d -> hart %d rc=%d\n",
			       i, cpu, rc);
			omp_nthreads = i;   /* run with what we successfully pinned */
			break;
		}
		k_thread_start(t);
	}
	omp_started = 1;
	printk("merlin_omp: %d threads (master hart %d) over %d CPUs\n",
	       omp_nthreads, omp_master_cpu, cpus);
	return omp_nthreads;
}

/* ---- the __kmpc_* surface ------------------------------------------------------------ */

int32_t __kmpc_global_thread_num(ident_t *loc)
{
	ARG_UNUSED(loc);
	return (int32_t)omp_self_gtid();
}

void __kmpc_push_num_threads(ident_t *loc, int32_t gtid, int32_t num_threads)
{
	ARG_UNUSED(loc);
	ARG_UNUSED(gtid);
	omp_requested = (int)num_threads;
}

void __kmpc_fork_call(ident_t *loc, int32_t argc, void *microtask, ...)
{
	va_list ap;
	int n;

	ARG_UNUSED(loc);
	if (!omp_started) {
		/* A parallel region before merlin_omp_init: run it serially rather than crash.
		 * Correct, just not fast — and visible in the log so it is never mistaken for a
		 * multicore run. */
		printk("merlin_omp: fork before init -> serial\n");
		merlin_omp_init(1);
	}

	if (argc > MERLIN_OMP_MAX_SHARED) {
		printk("FAIL merlin_omp: fork argc %d > %d\n", (int)argc, MERLIN_OMP_MAX_SHARED);
		k_panic();
	}
	va_start(ap, microtask);
	for (int i = 0; i < argc; i++) {
		omp_task_args[i] = va_arg(ap, void *);
	}
	va_end(ap);

	n = omp_requested > 0 ? omp_requested : omp_nthreads;
	omp_requested = 0;
	if (n > omp_nthreads) {
		n = omp_nthreads;
	}
	if (n < 1) {
		n = 1;
	}

	omp_task_fn = microtask;
	omp_task_argc = (int)argc;
	omp_task_nthreads = n;

	for (int i = 1; i < n; i++) {
		k_sem_give(&omp_workers[i].start);
	}
	omp_run_region(0);                       /* the master is thread 0 */
	for (int i = 1; i < n; i++) {
		k_sem_take(&omp_workers[i].done, K_FOREVER);
	}
	/* Returning from fork_call IS the implicit barrier at the end of the region. */
}

void __kmpc_barrier(ident_t *loc, int32_t gtid)
{
	ARG_UNUSED(loc);
	ARG_UNUSED(gtid);
	/* Every worksharing loop merlin emits is followed by the end-of-region join in
	 * __kmpc_fork_call, and the static schedule gives each thread a DISJOINT output slice
	 * (the reduction dim is never split), so there is no mid-region cross-thread
	 * dependency to synchronize. A no-op is correct for this emitted subset — it would
	 * NOT be for `omp barrier` inside a region, which merlin never emits. */
}

void __kmpc_critical(ident_t *loc, int32_t gtid, void *crit)
{
	ARG_UNUSED(loc);
	ARG_UNUSED(gtid);
	ARG_UNUSED(crit);
	k_mutex_lock(&omp_critical, K_FOREVER);
}

void __kmpc_end_critical(ident_t *loc, int32_t gtid, void *crit)
{
	ARG_UNUSED(loc);
	ARG_UNUSED(gtid);
	ARG_UNUSED(crit);
	k_mutex_unlock(&omp_critical);
}

/* Static worksharing loop. Computes this thread's contiguous chunk of [lower, upper].
 *
 * The `_4/_4u/_8/_8u` variants differ only in the induction-variable width/signedness of
 * the bounds; the block split is identical, so it is written once over int64_t and the
 * wrappers narrow. Only STATIC (schedtype 33/34) is emitted by merlin's lowering; a dynamic
 * schedule would arrive here as an unhandled schedtype and is rejected loudly rather than
 * silently mis-scheduled.
 */
static void omp_static_init(int32_t gtid, int32_t schedtype, int32_t *plastiter,
			    int64_t *plower, int64_t *pupper, int64_t *pstride,
			    int64_t incr, int64_t chunk)
{
	int nth = omp_task_nthreads > 0 ? omp_task_nthreads : 1;

	ARG_UNUSED(chunk);
	/* Only STATIC is emitted by merlin's lowering. A dynamic/guided schedule arriving here
	 * would need a work queue; mis-scheduling it as static would duplicate iterations, so
	 * reject it loudly instead. */
	if (schedtype != MERLIN_KMP_SCH_STATIC_CHUNKED && schedtype != MERLIN_KMP_SCH_STATIC) {
		printk("FAIL merlin_omp: unsupported schedule %d (static only)\n",
		       (int)schedtype);
		k_panic();
	}
	(void)merlin_omp_static_split(gtid, (int32_t)nth, plower, pupper, pstride, incr,
				      plastiter);
}

void __kmpc_for_static_init_8u(ident_t *loc, int32_t gtid, int32_t schedtype,
			       int32_t *plastiter, uint64_t *plower, uint64_t *pupper,
			       int64_t *pstride, int64_t incr, int64_t chunk)
{
	int64_t lo = (int64_t)*plower, up = (int64_t)*pupper;

	ARG_UNUSED(loc);
	omp_static_init(gtid, schedtype, plastiter, &lo, &up, pstride, incr, chunk);
	*plower = (uint64_t)lo;
	*pupper = (uint64_t)up;
}

void __kmpc_for_static_init_8(ident_t *loc, int32_t gtid, int32_t schedtype,
			      int32_t *plastiter, int64_t *plower, int64_t *pupper,
			      int64_t *pstride, int64_t incr, int64_t chunk)
{
	ARG_UNUSED(loc);
	omp_static_init(gtid, schedtype, plastiter, plower, pupper, pstride, incr, chunk);
}

void __kmpc_for_static_init_4u(ident_t *loc, int32_t gtid, int32_t schedtype,
			       int32_t *plastiter, uint32_t *plower, uint32_t *pupper,
			       int32_t *pstride, int32_t incr, int32_t chunk)
{
	int64_t lo = (int64_t)*plower, up = (int64_t)*pupper, st = 0;

	ARG_UNUSED(loc);
	omp_static_init(gtid, schedtype, plastiter, &lo, &up, &st, incr, chunk);
	*plower = (uint32_t)lo;
	*pupper = (uint32_t)up;
	*pstride = (int32_t)st;
}

void __kmpc_for_static_init_4(ident_t *loc, int32_t gtid, int32_t schedtype,
			      int32_t *plastiter, int32_t *plower, int32_t *pupper,
			      int32_t *pstride, int32_t incr, int32_t chunk)
{
	int64_t lo = (int64_t)*plower, up = (int64_t)*pupper, st = 0;

	ARG_UNUSED(loc);
	omp_static_init(gtid, schedtype, plastiter, &lo, &up, &st, incr, chunk);
	*plower = (int32_t)lo;
	*pupper = (int32_t)up;
	*pstride = (int32_t)st;
}

void __kmpc_for_static_fini(ident_t *loc, int32_t gtid)
{
	ARG_UNUSED(loc);
	ARG_UNUSED(gtid);
}

/* ---- the omp_* user API (a model may reference these via convert-math/libm paths) ---- */

int omp_get_num_threads(void)
{
	return omp_task_nthreads > 0 ? omp_task_nthreads : 1;
}

int omp_get_thread_num(void)
{
	return omp_self_gtid();
}

int omp_get_max_threads(void)
{
	return omp_nthreads > 0 ? omp_nthreads : 1;
}
