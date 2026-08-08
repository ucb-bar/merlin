/* Make an exhausted heap say so, instead of faulting on a null pointer.
 *
 * The lowered model allocates every intermediate tensor with a plain `malloc` emitted by
 * `finalize-memref-to-llvm`, and generated code never checks the result. When the arena runs out,
 * the next store goes through address 0 and the run dies as `mcause=7 Store/AMO access fault,
 * mtval=0` -- a register dump that says nothing about memory, thousands of instructions away from
 * the allocation that actually failed. That is what whisper_tiny looked like on FireSim: a store
 * fault at op 337 whose real cause was a 54 MB `malloc` returning NULL two instructions earlier.
 *
 * `--wrap=malloc` is the least invasive way to get in front of it: the linker redirects the
 * generated code's calls here, `__real_malloc` is the libc one, and the failure path reports the
 * size that could not be satisfied and the last op the profiler entered. Linked into EVERY image,
 * not just debug builds -- an unchecked allocation failure is precisely the failure a person with
 * no debugger and one serial log cannot diagnose, and a compare-and-branch per allocation is
 * nothing beside the allocation itself.
 */
#include <stdlib.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/toolchain.h>

#ifndef MERLIN_BUILD_HASH
#define MERLIN_BUILD_HASH "unknown"
#endif

/* Defined for real by merlin_op_prof.c in an instrumented build. Weak here so a plain image links
 * and still reports something useful (-1 = "no per-op instrumentation in this binary"). */
__weak volatile int32_t merlin_prof_last_id = -1;

extern void *__real_malloc(size_t n);

void *__wrap_malloc(size_t n)
{
	void *p = __real_malloc(n);

	if (p == NULL && n != 0) {
		printk("FAIL alloc bytes=%zu op=%d hart=%d build_hash=%s\n",
		       n, (int)merlin_prof_last_id, arch_curr_cpu()->id, MERLIN_BUILD_HASH);
		/* Terminate in the protocol. Without these two lines the host-side runner waits out
		 * its whole timeout on a target that has already stopped -- on the FPGA that is hours
		 * of a single shared board spent on a run that ended in the first minute. */
		printk("=== MODELBLASTER_WALL_CYCLES === 0\n");
		printk("DONE\n");
		k_fatal_halt(K_ERR_KERNEL_PANIC);
	}
	return p;
}
