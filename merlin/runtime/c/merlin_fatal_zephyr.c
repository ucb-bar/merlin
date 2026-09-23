/* Make a fatal error survive into the console log, attributable, and gradeable.
 *
 * Only linked into DEBUG images (zephyr_model passes `debug=True`), because the point of it is a
 * board nobody here can attach to. Zephyr already prints a register dump on a fault:
 * CONFIG_EXCEPTION_DEBUG defaults `y` whenever PRINTK is on, and arch/riscv/core/fatal.c writes
 * mcause/mtval/mepc/mstatus and the GPRs through plain printk -- so the detail is already on the same
 * channel as OUT/METRIC/DONE. Three things it does NOT do, all of which cost us a round trip:
 *
 *   1. The dump is not tied to a binary. A log that ends in a fault has no `build_hash` near the
 *      failure, and the banner may be thousands of lines earlier or lost to a truncated capture.
 *   2. The run never terminates in the protocol's own terms, so the host-side parser reports only
 *      "this log has no OUT line" -- the same thing it says for a hang, an unfinished upload, and a
 *      board that never booted. Four different failures, one message.
 *   3. `k_sys_fatal_error_handler`'s default halts the CPU that faulted. On SMP the others keep
 *      running, so a fault on a worker can be followed by more output and read as noise.
 *
 * So: restate the identity, name the reason in one greppable line, and emit DONE so the log is a
 * complete record of a failed run rather than a truncated record of an ambiguous one.
 */
#include <zephyr/kernel.h>
#include <zephyr/fatal.h>
#include <zephyr/sys/printk.h>
#include <zephyr/arch/cpu.h>

#ifndef MERLIN_BUILD_HASH
#define MERLIN_BUILD_HASH "unknown"
#endif

static const char *merlin_fatal_reason(unsigned int reason)
{
	switch (reason) {
	case K_ERR_CPU_EXCEPTION:       return "cpu_exception";
	case K_ERR_SPURIOUS_IRQ:        return "spurious_irq";
	case K_ERR_STACK_CHK_FAIL:      return "stack_overflow";
	case K_ERR_KERNEL_OOPS:         return "kernel_oops";
	case K_ERR_KERNEL_PANIC:        return "kernel_panic";
	default:                        return "unknown";
	}
}

void k_sys_fatal_error_handler(unsigned int reason, const struct arch_esf *esf)
{
	unsigned long mcause = 0, mepc = 0, mtval = 0, mstatus = 0;

	ARG_UNUSED(esf);
	__asm__ volatile("csrr %0, mcause"  : "=r"(mcause));
	__asm__ volatile("csrr %0, mepc"    : "=r"(mepc));
	__asm__ volatile("csrr %0, mtval"   : "=r"(mtval));
	__asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));

	/* One line, everything needed to place the failure: which binary, which hart, which thread,
	 * what trapped, where, and -- because it is the bug that brought us here twice -- whether this
	 * hart had vector state enabled at the time. */
	printk("FAIL fatal reason=%u(%s) hart=%d thread=%s mcause=%lu mepc=0x%lx mtval=0x%lx "
	       "vs=%u fs=%u build_hash=%s\n",
	       reason, merlin_fatal_reason(reason), arch_curr_cpu()->id,
	       k_thread_name_get(k_current_get()) ? k_thread_name_get(k_current_get()) : "?",
	       mcause, mepc, mtval,
	       (unsigned)((mstatus >> 9) & 3), (unsigned)((mstatus >> 13) & 3),
	       MERLIN_BUILD_HASH);
	/* Terminate in the protocol so the host parser reports a FAILED run rather than an absent one.
	 * The WALL_CYCLES sentinel matters as much as DONE: it is the marker the FireSim runner blocks
	 * on, and without it a crashed run holds the FPGA for its entire timeout. A whisper build that
	 * faulted eight minutes in sat on the board for four hours before anyone could see why. */
	printk("=== MODELBLASTER_WALL_CYCLES === 0\n");
	printk("DONE\n");

	k_fatal_halt(reason);
}
