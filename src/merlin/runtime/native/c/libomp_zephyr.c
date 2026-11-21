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
#include <zephyr/sys/atomic.h>
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
typedef void (*kmpc_micro5_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro6_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro7_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro8_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro9_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro10_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro11_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro12_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro13_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro14_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro15_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
typedef void (*kmpc_micro16_t)(int32_t *, int32_t *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);

/* 16, not 4. The MLIR OpenMP lowering passes exactly ONE shared arg (a captured struct),
 * which is why 4 sufficed for every model region. The OPU tile loop is different: it is C
 * emitted by the compiler backend and compiled by clang, whose OpenMP codegen passes every
 * captured variable SEPARATELY -- the `collapse(2)` region passes 8. At 4 this panicked with
 * `fork argc 8 > 4` on the first routed contraction, so the 2-core OPU path had never once
 * executed. 16 leaves room for wider signatures (bias/requant variants capture more) while
 * keeping the loud failure for anything past it. */
#define MERLIN_OMP_MAX_SHARED 16

/* ---- pool state --------------------------------------------------------------------- */

/* Workers SPIN on a generation counter rather than blocking on a semaphore.
 *
 * This is forced by the Saturn fork's arch/riscv/core/v.c V save/restore bug, and it took a
 * measurement to see why: pinning COOP workers 1:1 to harts removes PREEMPTION, but
 * k_sem_take is a VOLUNTARY context switch — a blocked worker yields its hart to the idle
 * thread and switches back on wake, which runs v.c's save/restore exactly as preemption
 * would. With ~358 parallel regions per inference that is ~716 V context switches per hart,
 * and the measured result was a load access fault from garbage callee-saved registers on the
 * master. The SAME binary with the pool clamped to one thread (omp_threads=1) runs clean to
 * DONE, and the same lowering is bit-correct on the host — so it is the switching, not the
 * codegen.
 *
 * Spinning never leaves the thread, so v.c is never entered. It also happens to be the right
 * structure regardless: at this fork/join granularity a semaphore round-trip would dominate
 * the region itself, and the harts are dedicated (pinned 1:1, never oversubscribed) so there
 * is nothing else for them to run. Interrupts stay enabled; ISRs do not touch V under
 * CONFIG_RISCV_V_KERNEL_ONLY, and with no other ready thread on the hart they cause no switch.
 */
struct omp_worker {
	struct k_thread thread;
	atomic_t done;          /* generation this worker has finished */
	atomic_t entered;       /* generation this worker last STARTED (done lags it by one region) */
	atomic_t alive;         /* set by the worker the first time it actually executes */
	int32_t gtid;
};

/* How long merlin_omp_init waits for each worker to prove it is really executing, and how
 * long a region waits for a worker that was alive at init.
 *
 * WHY THIS EXISTS. Spinning means a worker that never runs does not stall the master, it makes
 * it burn forever. Measured on FireSim: a 2-hart image of a 22.4M-cycle model ran past 11.4
 * BILLION cycles -- 500x -- printing nothing after the pool banner, because the master spun on
 * a generation the worker never published. k_thread_cpu_pin() returning 0 and arch_num_cpus()
 * reporting 2 say the KERNEL believes the hart exists; neither proves it is executing code.
 * So liveness is now PROVEN by the worker itself before any region is dispatched, and a hart
 * that does not answer is dropped from the pool with a loud message instead of hanging the run.
 */
#ifndef MERLIN_OMP_LIVENESS_SPINS
#define MERLIN_OMP_LIVENESS_SPINS 200000000L
#endif

/* The region wait is bounded by WALL CLOCK, not a spin count, because the same image runs at
 * three wildly different speeds (spike ~10 MIPS, FireSim 25 MHz, K1 silicon) and any spin
 * budget that is generous on one is either instant-panic or effectively infinite on another.
 * k_uptime_get() is driven by the timer ISR, which keeps running while the master spins. */
#ifndef MERLIN_OMP_REGION_TIMEOUT_MS
#define MERLIN_OMP_REGION_TIMEOUT_MS 600000
#endif
/* Spins between "still waiting" heartbeats. A silent hang is unactionable; a hang that keeps
 * reporting which worker it is waiting for, and what that worker last published, is a bug
 * report. Cheap: one uptime read per 4M spins. */
#ifndef MERLIN_OMP_HEARTBEAT_SPINS
#define MERLIN_OMP_HEARTBEAT_SPINS (1L << 22)
#endif
/* Minimum wall time between stall reports for the same wait. */
#ifndef MERLIN_OMP_STALL_REPORT_MS
#define MERLIN_OMP_STALL_REPORT_MS 5000
#endif

/* Confine the pool to harts that actually have a vector unit, discovered at startup by asking each
 * hart (see omp_hart_has_vector). On for a VECTOR image; a scalar image sets this 0 because it may
 * legitimately use every hart -- that is the only way to reach a core without a vector unit, and on a
 * chip with more cores than vector units it is the difference between using the machine and using
 * part of it. Defaulted here so a caller that predates the flag still builds. */
#ifndef MERLIN_OMP_VECTOR_POOL
#define MERLIN_OMP_VECTOR_POOL 1
#endif

/* Harts the DISCOVERY pass may examine. Deliberately NOT MERLIN_OMP_MAX_THREADS: that is sized to the
 * threads this image asked for, and bounding the probe by it would only ever look at the first N harts
 * -- reintroducing the "the vector harts are 0..n-1" assumption the probe exists to remove. A 2-thread
 * image on a 3-core chip must still be able to discover that the vector units are on harts 0 and 2. */
#ifndef MERLIN_OMP_MAX_HARTS
#define MERLIN_OMP_MAX_HARTS 8
#endif

K_THREAD_STACK_ARRAY_DEFINE(merlin_omp_stacks, MERLIN_OMP_MAX_THREADS,
			    MERLIN_OMP_WORKER_STACK);

static struct omp_worker omp_workers[MERLIN_OMP_MAX_THREADS];
static struct k_mutex omp_critical;
static atomic_t omp_gen;              /* bumped once per parallel region by the master */

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
static int8_t omp_gtid_of_hart[MERLIN_OMP_MAX_HARTS];
static int omp_master_cpu = -1;

/* Per-hart TEAM state — the thread's id within the team currently executing on that hart, the
 * size of that team, and the parallel-region nesting depth.
 *
 * WHY A TEAM AND NOT JUST THE POOL SLOT. merlin's lowering DOES emit nested parallel regions:
 * of the 358 fork sites in a small_llama int8 module, four outlined regions
 * (`forward..omp_par.{68,111,239,278}`) themselves call __kmpc_fork_call. Treating the inner
 * fork like an outer one is fatal in two ways, both observed: the inner fork bumps the
 * generation counter, so the master returns from its region and waits for a generation the
 * workers raced past -- measured live, "waiting on worker 1 in region 68; it last published
 * 132" -- and the inner worksharing loop would be split across a team that is not executing it.
 *
 * So nested regions are SERIALIZED, exactly as libomp does with OMP_NESTED=false: the inner
 * region runs inline on the encountering thread as a team of one. Worksharing then needs the
 * CURRENT team (size 1, tid 0), which is why these are per-hart and saved/restored around the
 * nested call rather than read from the global omp_task_nthreads. */
static int8_t omp_tid_of_hart[MERLIN_OMP_MAX_HARTS];
static int8_t omp_team_of_hart[MERLIN_OMP_MAX_HARTS];
static int8_t omp_depth_of_hart[MERLIN_OMP_MAX_HARTS];

/* Bounded by HARTS, not by threads. Bounding a hart index by the thread count silently ALIASES: a
 * 2-thread image on a 3-core chip whose vector units sit on harts 0 and 2 mapped hart 2 onto slot 0,
 * so the worker and the master both claimed tid 0 -- half the tiles computed twice and half not at
 * all, a wrong answer rather than a fault. MERLIN_OMP_MAX_HARTS is exactly the bound the discovery
 * pass already uses for this reason (see its comment above). */
static inline int omp_self_cpu(void)
{
	int cpu = (int)arch_curr_cpu()->id;

	return (cpu < 0 || cpu >= MERLIN_OMP_MAX_HARTS) ? 0 : cpu;
}

static inline int omp_self_gtid(void)
{
	return (int)omp_tid_of_hart[omp_self_cpu()];
}

/* Size of the team this hart is currently part of; never 0, so it is safe as a divisor. */
static inline int omp_self_team(void)
{
	int n = (int)omp_team_of_hart[omp_self_cpu()];

	return n > 0 ? n : 1;
}

/* Enter/leave a region on this hart, returning the saved state so it can be restored. */
struct omp_team_save { int8_t tid, team, depth; };

static inline struct omp_team_save omp_team_enter(int cpu, int tid, int team)
{
	struct omp_team_save sv = { omp_tid_of_hart[cpu], omp_team_of_hart[cpu],
				    omp_depth_of_hart[cpu] };

	omp_tid_of_hart[cpu] = (int8_t)tid;
	omp_team_of_hart[cpu] = (int8_t)team;
	omp_depth_of_hart[cpu] = (int8_t)(sv.depth + 1);
	return sv;
}

static inline void omp_team_leave(int cpu, struct omp_team_save sv)
{
	omp_tid_of_hart[cpu] = sv.tid;
	omp_team_of_hart[cpu] = sv.team;
	omp_depth_of_hart[cpu] = sv.depth;
}

/* mstatus.VS/FS = Dirty. The proven zephyr-chipyard-sw samples set this in each V-using
 * worker entry (tiled_matmul_mt_pool's enable_vector_operations, merlin_hetero_runner's
 * set_vs_dirty): a freshly created thread can start with VS=Off, and the first vector
 * instruction would then trap. Harmless when the state is already dirty. */
#define MSTATUS_FS 0x00006000UL
#define MSTATUS_VS 0x00000600UL

/* Spin-loop relax. Deliberately NOT Zephyr's arch_spin_relax(): on RISC-V that is defined
 * only under CONFIG_FPU_SHARING (arch/riscv/core/ipi_clint.c), which this image sets to n
 * because FPU_SHARING mis-routes V-illegal-instruction traps — so referencing it would fail
 * to link. A compiler barrier is all that is needed; atomic_get() below is a real load each
 * iteration, and the RISC-V pause hint (Zihintpause) is not guaranteed on this SoC. */
static inline void omp_relax(void)
{
	__asm__ volatile("" ::: "memory");
}

static inline void omp_enable_vector(void)
{
	unsigned long mstatus;

	__asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
	mstatus |= MSTATUS_VS | MSTATUS_FS;
	__asm__ volatile("csrw mstatus, %0" ::"r"(mstatus));
}

/* Does the hart executing this call have a vector unit? Asked of the HARDWARE, not of a build
 * parameter or a device tree.
 *
 * Why it can be asked at all: `mstatus.VS` is WARL -- on a hart that implements V it is writable, and
 * on one that does not it is hardwired to zero. So enabling it and reading it back distinguishes the
 * two, and does so WITHOUT TRAPPING. That matters: the obvious probe (execute a vector instruction and
 * see if it faults) needs a recoverable trap handler per hart, and an unrecovered fault on a board we
 * cannot attach to is the very failure mode being diagnosed.
 *
 * `misa` bit 21 ('V') is the other half. Some cores tie misa to zero, so a 0 there is "not reported",
 * not "absent" -- it can confirm but must not veto. Requiring both would silently drop a working
 * vector hart on any core with a zeroed misa.
 *
 * This is the fact whose ABSENCE caused a shipped image to hang: a 3-core chip with vector units on
 * only 2 of its harts fanned a vector kernel out over all 3, and the worker on the scalar hart trapped
 * and never reached the barrier. The device tree could not have told us -- it lists identical cpu@N
 * nodes -- and a build-time list means someone has to know and tell us. The hart itself knows.
 */
static bool omp_hart_has_vector(void)
{
	unsigned long ms, misa = 0;

	omp_enable_vector();
	__asm__ volatile("csrr %0, mstatus" : "=r"(ms));
	if (((ms >> 9) & 3) == 0) {
		return false;           /* VS refused the write -> no vector state on this hart */
	}
	/* VS is live, so vlenb is legal to read here; a zero width would mean something stranger than
	 * "no vectors" and is not something to fan a kernel out over either. */
	__asm__ volatile("csrr %0, misa" : "=r"(misa));
	if (((misa >> 21) & 1) == 0 && misa != 0) {
		return false;           /* misa is readable AND says no V */
	}
	unsigned long vlenb = 0;
	__asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
	return vlenb != 0;
}

/* Invoke an outlined region. Takes fn/argc/args explicitly rather than reading the globals,
 * because a SERIALIZED NESTED region must run from a caller-local argument array: the globals
 * still describe the outer region that the other harts are executing right now, and
 * overwriting them would corrupt it. */
static void omp_call_micro(void *fn, int argc, void **a, int32_t gtid)
{
	int32_t tid = gtid;
	int32_t bound = 0;

	/* RE-ARM VECTOR STATE, on EVERY region entry and for EVERY team member including the master.
	 *
	 * Enabling mstatus.VS once per thread is not enough. VS lives in the thread's saved mstatus, and
	 * a Zephyr built WITHOUT CONFIG_RISCV_ISA_EXT_V never puts it there -- so a context switch
	 * restores VS = Off and the next vector instruction traps. The master is the one this bites: it
	 * enables VS in the generated harness, then merlin_omp_init creates and joins a probe thread per
	 * hart and creates the pool, and every one of those is a switch. It comes back with VS off and
	 * takes the trap on its own slice of the first parallel region -- which, on a build with
	 * FPU_SHARING=y, is mis-routed into the FP retry path and hangs with nothing printed. That is
	 * exactly the shape of the Kodiak report: single-hart passed, every multi-hart image failed
	 * silently, and the stall watchdog could not say so because the master runs the watchdog.
	 *
	 * The board config is fixed separately (Zephyr now manages V state there). This stays because it
	 * is two CSR accesses per region against a whole-image hang, and because it is correct on any
	 * tree, including one whose Kconfig cannot express V at all.
	 */
	omp_enable_vector();

	switch (argc) {
	case 0:
		((kmpc_micro0_t)fn)(&tid, &bound);
		break;
	case 1:
		((kmpc_micro1_t)fn)(&tid, &bound, a[0]);
		break;
	case 2:
		((kmpc_micro2_t)fn)(&tid, &bound, a[0], a[1]);
		break;
	case 3:
		((kmpc_micro3_t)fn)(&tid, &bound, a[0], a[1], a[2]);
		break;
	case 4:
		((kmpc_micro4_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3]);
		break;
	case 5:
		((kmpc_micro5_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4]);
		break;
	case 6:
		((kmpc_micro6_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case 7:
		((kmpc_micro7_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case 8:
		((kmpc_micro8_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case 9:
		((kmpc_micro9_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case 10:
		((kmpc_micro10_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case 11:
		((kmpc_micro11_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]);
		break;
	case 12:
		((kmpc_micro12_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]);
		break;
	case 13:
		((kmpc_micro13_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case 14:
		((kmpc_micro14_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13]);
		break;
	case 15:
		((kmpc_micro15_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14]);
		break;
	case 16:
		((kmpc_micro16_t)fn)(&tid, &bound, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
		break;
	default:
		/* Fail loudly: silently running the region with the wrong argument count would
		 * corrupt results rather than crash. */
		printk("FAIL merlin_omp: %d shared args > MERLIN_OMP_MAX_SHARED %d\n",
		       argc, MERLIN_OMP_MAX_SHARED);
		k_panic();
	}
}

/* Run the region the master published in the globals, as team member `gtid` of `team`. */
static void omp_run_region(int32_t gtid, int team)
{
	int cpu = omp_self_cpu();
	struct omp_team_save sv = omp_team_enter(cpu, (int)gtid, team);

	omp_call_micro(omp_task_fn, omp_task_argc, omp_task_args, gtid);
	omp_team_leave(cpu, sv);
}

static void omp_worker_entry(void *p0, void *p1, void *p2)
{
	struct omp_worker *w = (struct omp_worker *)p0;
	int slot = (int)(intptr_t)p1;
	atomic_val_t seen = 0;

	ARG_UNUSED(p2);
	omp_enable_vector();
	/* Prove to merlin_omp_init that this hart is genuinely executing, not merely that the
	 * kernel accepted the thread and the pin. */
	atomic_set(&w->alive, 1);
	for (;;) {
		atomic_val_t g;

		/* Spin (never block — see the struct comment). atomic_get is SEQ_CST, so the
		 * task globals the master published before bumping the generation are visible
		 * once the new generation is. */
		while ((g = atomic_get(&omp_gen)) == seen) {
			omp_relax();
		}
		seen = g;
		atomic_set(&w->entered, g);
		if (slot < omp_task_nthreads) {
			omp_run_region((int32_t)slot, omp_task_nthreads);
		}
		/* Publish AFTER the region: the master waits on this before reading results or
		 * starting the next region. */
		atomic_set(&w->done, g);
	}
}

int merlin_omp_num_threads(void)
{
	return omp_nthreads;
}

void merlin_omp_report_stacks(void)
{
	for (int i = 1; i < omp_nthreads; i++) {
		size_t unused = 0;

		if (k_thread_stack_space_get(&omp_workers[i].thread, &unused) == 0) {
			printk("STACK omp%d size=%u unused=%u used=%u\n", i,
			       (unsigned)MERLIN_OMP_WORKER_STACK, (unsigned)unused,
			       (unsigned)(MERLIN_OMP_WORKER_STACK - unused));
		}
	}
}

/* Which harts a vector kernel may run on, DISCOVERED by asking each hart itself.
 *
 * A probe has to execute ON the hart it is asking about, so each gets a short-lived thread pinned to
 * it. Cost is one thread create/join per hart at startup, once, against the alternative of a build
 * parameter someone has to know: on a heterogeneous SoC (more cores brought up than vector units
 * attached) a wrong guess is a DEADLOCK -- the worker on a scalar hart traps and never reaches the
 * barrier -- and nothing readable states the mapping.
 *
 * A hart that fails to run the probe at all is treated as unusable, which is the same conservative
 * answer the liveness check below gives: better a smaller pool than a hung image.
 */
static K_THREAD_STACK_ARRAY_DEFINE(merlin_probe_stacks, MERLIN_OMP_MAX_HARTS, 1024);
static struct k_thread merlin_probe_threads[MERLIN_OMP_MAX_HARTS];
static volatile int8_t omp_vec_of_hart[MERLIN_OMP_MAX_HARTS];     /* -1 unknown, 0 no, 1 yes */
static volatile uint32_t omp_vlenb_of_hart[MERLIN_OMP_MAX_HARTS];
/* mstatus.VS as the thread found it, BEFORE omp_enable_vector() ran. This is the fact that
 * distinguishes "Zephyr is managing vector state for us" (2/3, Initial or Clean) from "every thread
 * starts with it off and only our own CSR write saves us" (0) -- and a 0 here on a hart that
 * nonetheless reports a vlenb is the precise signature of the Kodiak multi-hart hang. */
static volatile uint8_t omp_vs_on_entry[MERLIN_OMP_MAX_HARTS];

static void omp_probe_entry(void *p0, void *p1, void *p2)
{
	int cpu = (int)(intptr_t)p0;
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);

	unsigned long ms0;

	__asm__ volatile("csrr %0, mstatus" : "=r"(ms0));
	omp_vs_on_entry[cpu] = (uint8_t)((ms0 >> 9) & 3);

	bool has_v = omp_hart_has_vector();
	unsigned long vlenb = 0;

	if (has_v) {
		__asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
	}
	omp_vlenb_of_hart[cpu] = (uint32_t)vlenb;
	/* Published LAST, and after a fence, so a reader that sees the capability also sees the width. */
	__asm__ volatile("fence rw, rw" ::: "memory");
	omp_vec_of_hart[cpu] = has_v ? 1 : 0;
}

static void omp_discover_vector_harts(int cpus)
{
	for (int i = 0; i < MERLIN_OMP_MAX_HARTS; i++) {
		omp_vec_of_hart[i] = -1;
		omp_vlenb_of_hart[i] = 0;
		omp_vs_on_entry[i] = 0xFF;              /* never probed */
	}
	for (int cpu = 0; cpu < cpus && cpu < MERLIN_OMP_MAX_HARTS; cpu++) {
		if (cpu == omp_master_cpu) {
			/* We are already running here; probe in place rather than schedule onto self. */
			omp_probe_entry((void *)(intptr_t)cpu, NULL, NULL);
			continue;
		}
		k_tid_t t = k_thread_create(&merlin_probe_threads[cpu], merlin_probe_stacks[cpu],
					    1024, omp_probe_entry, (void *)(intptr_t)cpu, NULL, NULL,
					    K_PRIO_COOP(0), 0, K_FOREVER);
		if (k_thread_cpu_pin(t, cpu) != 0) {
			omp_vec_of_hart[cpu] = 0;
			continue;
		}
		k_thread_start(t);
		/* Bounded join: a hart that never runs must not stall startup. */
		if (k_thread_join(t, K_MSEC(2000)) != 0) {
			k_thread_abort(t);
			omp_vec_of_hart[cpu] = 0;
		}
	}
	/* Report the discovered topology on the console. A log mailed back from a board we cannot
	 * reach then STATES which harts have vector units and how wide they are, instead of leaving
	 * it to be inferred -- which is how the wrong assumption survived in the first place. */
	printk("METRIC vector_harts");
	for (int cpu = 0; cpu < cpus && cpu < MERLIN_OMP_MAX_HARTS; cpu++) {
		if (omp_vec_of_hart[cpu] == 1) {
			printk(" %d", cpu);
		}
	}
	printk("\n");
	for (int cpu = 0; cpu < cpus && cpu < MERLIN_OMP_MAX_HARTS; cpu++) {
		printk("METRIC hart%d_vlen_bits %u\n", cpu, omp_vlenb_of_hart[cpu] * 8u);
	}
	/* Whether the RTOS handed each thread live vector state, or we had to switch it on ourselves.
	 * 255 = that hart was never probed. */
	for (int cpu = 0; cpu < cpus && cpu < MERLIN_OMP_MAX_HARTS; cpu++) {
		printk("METRIC hart%d_mstatus_vs %u\n", cpu, (unsigned)omp_vs_on_entry[cpu]);
	}
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

	/* Ask the hardware which harts can run vector code, then keep the pool inside that set.
	 * Runtime discovery rather than a build parameter: the same binary is then correct on a
	 * homogeneous chip, on one whose vector units are on a non-contiguous pair, and on one nobody
	 * has characterised for us -- and it cannot deadlock by fanning out onto a scalar hart.
	 *
	 * Only for a VECTOR image. A scalar one may use every hart, which is the point of having it. */
#if MERLIN_OMP_VECTOR_POOL
	omp_discover_vector_harts(cpus);
	if (omp_vec_of_hart[omp_master_cpu] != 1) {
		/* The master runs the model's serial regions and thread 0 of every parallel one, so a
		 * master without vectors is not a pool problem -- it is the wrong hart for this image. */
		printk("merlin_omp: WARNING master hart %d reports no vector unit\n", omp_master_cpu);
	}
	int usable = 0;
	for (int cpu = 0; cpu < cpus && cpu < MERLIN_OMP_MAX_HARTS; cpu++) {
		usable += (omp_vec_of_hart[cpu] == 1);
	}
	if (usable > 0 && n_threads > usable) {
		printk("merlin_omp: %d thread(s) requested but %d hart(s) have vector units; "
		       "running %d\n", n_threads, usable, usable);
		n_threads = usable;
	}
#endif
	omp_nthreads = n_threads;
	for (int i = 0; i < MERLIN_OMP_MAX_THREADS; i++) {
		omp_gtid_of_hart[i] = 0;
		/* Outside any region every hart is a team of one, thread 0, depth 0 — so a
		 * worksharing loop reached before the first fork runs whole, not sliced. */
		omp_tid_of_hart[i] = 0;
		omp_team_of_hart[i] = 1;
		omp_depth_of_hart[i] = 0;
	}
	omp_gtid_of_hart[omp_master_cpu] = 0;   /* the master is OpenMP thread 0 */

	for (int i = 1; i < n_threads; i++) {
		struct omp_worker *w = &omp_workers[i];
		/* Pin worker i to a hart that is NOT the master's. Harts are numbered 0..cpus-1;
		 * walk them skipping the master's so the mapping stays 1:1 even when the master
		 * is not on hart 0 (the FireSim vector-tile case pins it to hart 1). */
		int cpu = (i <= omp_master_cpu) ? (i - 1) : i;
#if MERLIN_OMP_VECTOR_POOL
		/* Walk the harts the PROBE found, skipping the master's. This supersedes both the
		 * 0..n-1 assumption and any build-time list: the set came from the hardware a moment
		 * ago, so a non-contiguous pair (units on harts 0 and 2, say) needs nobody to tell us. */
		{
			int k = 0;
			cpu = -1;
			for (int j = 0; j < cpus && j < MERLIN_OMP_MAX_HARTS; j++) {
				if (omp_vec_of_hart[j] != 1 || j == omp_master_cpu) {
					continue;
				}
				if (++k == i) {
					cpu = j;
					break;
				}
			}
			if (cpu < 0) {
				printk("merlin_omp: only %d vector hart(s) usable, running %d thread(s)\n",
				       k + 1, i);
				omp_nthreads = i;
				break;
			}
		}
#elif defined(MERLIN_OMP_HART_IDS)
		/* HETEROGENEOUS SoC: the harts that may run this pool are listed explicitly, because
		 * "hart 0..n-1" is an assumption and a wrong one is a DEADLOCK, not a slowdown -- a
		 * worker placed on a hart without a vector unit traps on its first vector instruction
		 * and never reaches the barrier its peers are waiting on. Nothing readable says which
		 * harts are vector-capable: the device tree lists identical cpu@N nodes. So the set is
		 * a build parameter (Board.vector_hart_ids), and this walks it skipping the master's. */
		{
			static const int hart_ids[] = MERLIN_OMP_HART_IDS;
			const int n_ids = (int)(sizeof(hart_ids) / sizeof(hart_ids[0]));
			int k = 0;
			cpu = -1;
			for (int j = 0; j < n_ids; j++) {
				if (hart_ids[j] == omp_master_cpu) {
					continue;       /* the master already owns this one */
				}
				if (++k == i) {
					cpu = hart_ids[j];
					break;
				}
			}
			if (cpu < 0) {
				/* More threads requested than usable harts. Shrink rather than pin a
				 * second worker onto a hart already running one, which would serialize
				 * silently and report a speed-up that never happened. */
				printk("merlin_omp: only %d usable hart(s), running %d thread(s)\n",
				       k + 1, i);
				omp_nthreads = i;
				break;
			}
		}
#endif

		atomic_set(&w->done, 0);
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

	/* Wait for each worker to PROVE it is executing, then keep only the contiguous prefix of
	 * live workers (thread ids must stay 0..n-1 for the static split to partition correctly).
	 * A hart that never answers is dropped and the pool shrinks -- the model still computes
	 * the right answer, just on fewer cores, and says so. */
	int live = 1;
	for (int i = 1; i < omp_nthreads; i++) {
		long spins = 0;
		while (!atomic_get(&omp_workers[i].alive) && spins < MERLIN_OMP_LIVENESS_SPINS) {
			omp_relax();
			spins++;
		}
		if (!atomic_get(&omp_workers[i].alive)) {
			printk("merlin_omp: WORKER %d (hart %d) NEVER RAN -- dropping it; the kernel "
			       "accepted the thread and the pin, but the hart is not executing\n",
			       i, omp_workers[i].gtid);
			break;
		}
		live = i + 1;
	}
	if (live != omp_nthreads) {
		printk("merlin_omp: pool reduced %d -> %d threads (results stay correct)\n",
		       omp_nthreads, live);
		omp_nthreads = live;
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

	/* NESTED REGION -> run it inline as a team of one (libomp's OMP_NESTED=false default).
	 * This must happen before the generation counter is touched: an inner fork that bumped
	 * the generation would release the workers into a region their master is not joining,
	 * and the master would then wait forever on a generation they had already raced past. */
	if (omp_depth_of_hart[omp_self_cpu()] > 0) {
		void *nargs[MERLIN_OMP_MAX_SHARED];
		int cpu = omp_self_cpu();
		struct omp_team_save sv;

		va_start(ap, microtask);
		for (int i = 0; i < argc; i++) {
			nargs[i] = va_arg(ap, void *);
		}
		va_end(ap);
		omp_requested = 0;
		sv = omp_team_enter(cpu, 0, 1);
		omp_call_micro(microtask, (int)argc, nargs, 0);
		omp_team_leave(cpu, sv);
		return;
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

	/* Release the workers by bumping the generation (SEQ_CST: everything above is visible
	 * to a worker that observes the new value), run the master's own slice, then spin until
	 * every worker has published this generation. */
	atomic_val_t g = atomic_add(&omp_gen, 1) + 1;

	omp_run_region(0, n);                    /* the master is thread 0 */
	for (int i = 1; i < n; i++) {
		long spins = 0;
		int64_t t0 = 0, next_report = MERLIN_OMP_STALL_REPORT_MS;

		while (atomic_get(&omp_workers[i].done) != g) {
			omp_relax();
			if (++spins < MERLIN_OMP_HEARTBEAT_SPINS) {
				continue;
			}
			spins = 0;
			if (t0 == 0) {
				t0 = k_uptime_get();
				continue;
			}

			int64_t dt = k_uptime_get() - t0;

			if (dt < next_report) {
				continue;
			}
			/* A worker that was alive at init has stopped answering for long enough that
			 * this is not just a big region. Say so, sparsely, then give up -- an unbounded
			 * silent wait is indistinguishable from slow progress from the outside, which is
			 * how a 22.4M-cycle model reached 11.4 BILLION cycles on FireSim before anyone
			 * noticed. Sparsely matters: printk goes out over HTIF/UART one character at a
			 * time, so a chatty stall report costs more than the region it is reporting on. */
			next_report = dt + MERLIN_OMP_STALL_REPORT_MS;
			printk("merlin_omp: waiting %lld ms on worker %d (hart %d) in region %ld; "
			       "it last published %ld, entered %ld, alive=%ld\n",
			       (long long)dt, i, omp_workers[i].gtid, (long)g,
			       (long)atomic_get(&omp_workers[i].done),
			       (long)atomic_get(&omp_workers[i].entered),
			       (long)atomic_get(&omp_workers[i].alive));
			if (dt > MERLIN_OMP_REGION_TIMEOUT_MS) {
				printk("FAIL merlin_omp: worker %d stalled in region %ld\n", i, (long)g);
				k_panic();
			}
		}
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
	/* The CURRENT team, not omp_task_nthreads: inside a serialized nested region the team is
	 * one thread and the loop must not be split, even though the outer region has n. */
	int nth = omp_self_team();

	ARG_UNUSED(chunk);
	/* Only STATIC is emitted by merlin's lowering. A dynamic/guided schedule arriving here
	 * would need a work queue; mis-scheduling it as static would duplicate iterations, so
	 * reject it loudly instead. */
	if (schedtype != MERLIN_KMP_SCH_STATIC_CHUNKED && schedtype != MERLIN_KMP_SCH_STATIC) {
		printk("FAIL merlin_omp: unsupported schedule %d (static only)\n",
		       (int)schedtype);
		k_panic();
	}
#ifdef MERLIN_OMP_DEBUG_SPLIT
	/* Opt-in (-DMERLIN_OMP_DEBUG_SPLIT=1, via MERLIN_OMP_DEBUG_SPLIT=1 in the build env).
	 * A wrong split does not look like a wrong answer, it looks like an out-of-range vector
	 * load on an unrelated hart, so being able to dump every partition without editing C is
	 * what turns that class of bug from a crash into a table. */
	{
		static int nseen;
		int64_t lo0 = *plower, up0 = *pupper;

		(void)merlin_omp_static_split(gtid, (int32_t)nth, plower, pupper, pstride, incr,
					      plastiter);
		printk("SPLIT %d gtid=%d nth=%d in=[%lld,%lld] out=[%lld,%lld] incr=%lld\n",
		       nseen++, (int)gtid, nth, (long long)lo0, (long long)up0,
		       (long long)*plower, (long long)*pupper, (long long)incr);
		return;
	}
#endif
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
	return omp_self_team();
}

int omp_get_thread_num(void)
{
	return omp_self_gtid();
}

int omp_get_max_threads(void)
{
	return omp_nthreads > 0 ? omp_nthreads : 1;
}
