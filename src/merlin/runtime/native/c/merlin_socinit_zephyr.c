/* Raise this SoC's PLL before the RTOS brings up its console.
 *
 * Only linked when the board descriptor declares `chip_freq_hz` (see zephyr_model._cmakelists). The
 * numbers below are NOT written here: every base, offset and selector value arrives as a -D macro
 * derived from the target's own SDK headers by `merlin.runtime.sdk_facts`, so adding a second chip is
 * a descriptor entry, not an edit to this file. It refuses to compile if a macro is absent rather than
 * defaulting one -- a wrong PLL ratio or a wrong divisor is a console that emits garbage, which reads
 * as a corrupt program rather than as a misconfigured clock.
 *
 * Why it matters: without this the chip stays on its reset clock. On the SoC this was written for that
 * is 50 MHz while the vendor's own demos run at 500 MHz, so every inference we shipped was ten times
 * slower than the part can go -- on a model whose vendor-equivalent finishes in about fifteen seconds.
 *
 * ORDER IS THE WHOLE THING, and it is theirs, not ours (bmark-lib/simple_setup.c, `init_test`):
 *
 *   1. park every clock domain on the slow source BEFORE touching the PLL. Reprogramming a PLL that
 *      domains are actively sourcing from means running the core off a relocking PLL.
 *   2. program and enable the PLL.
 *   3. switch the domains onto it.
 *   4. re-derive the UART divisor. The divisor is a RATIO to a clock that step 3 just changed by 10x;
 *      leaving it is the difference between a working console and line noise.
 *
 * WHERE this runs is load-bearing, and the obvious choice is wrong. Running at PRE_KERNEL_1 priority 0
 * -- before everything -- means the SiFive UART driver initialises AFTERWARDS and writes a divisor
 * computed from CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC, i.e. for the clock the chip is no longer on. It
 * would silently undo step 4 and hand back the garbled console this exists to prevent. So it runs
 * immediately AFTER the serial driver (`CONFIG_SERIAL_INIT_PRIORITY + 1`, derived rather than a
 * literal), which also matches the vendor's order exactly: their step 1 is "bring the UART up on the
 * reset clock", which is precisely what the driver has just done for us. Zephyr's boot banner is
 * printed later still, from the kernel, so it comes out at the corrected baud.
 */
#include <zephyr/init.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/sys_io.h>
#include <stdint.h>

#if !defined(MERLIN_CHIP_FREQ_HZ) || !defined(MERLIN_PLL_BASE) || \
    !defined(MERLIN_CLKSEL_BASE) || !defined(MERLIN_SYS_CLK_HZ) || \
    !defined(MERLIN_UART_BASE) || !defined(MERLIN_UART_DIV_OFF) || !defined(MERLIN_UART_BAUD)
#error "merlin_socinit_zephyr.c needs the SDK-derived clock macros (runtime.sdk_facts.macros)"
#endif

static inline void wr32(uintptr_t base, unsigned off, uint32_t v)
{
	sys_write32(v, base + off);
}

/* Every clock-selector domain onto one source. The domains are a contiguous array of 32-bit
 * registers; `sdk_facts.macros` verifies that contiguity against the derived map before emitting
 * MERLIN_CLKSEL_N, so walking it here is checked rather than assumed. */
static void merlin_set_all_clocks(uint32_t sel)
{
	for (unsigned i = 0; i < (unsigned)MERLIN_CLKSEL_N; i++) {
		wr32((uintptr_t)MERLIN_CLKSEL_BASE, i * 4u, sel);
	}
}

static int merlin_soc_clock_init(void)
{
	const uintptr_t pll = (uintptr_t)MERLIN_PLL_BASE;
	/* Integer ratio only: the SDK's own demos pass fraction = 0, and a fractional ratio would need
	 * the 2^24 fixed-point term their formula defines. Refuse rather than round -- an off-by-one
	 * ratio is a chip running at the wrong speed with nothing to say so. */
	const uint32_t ratio = (uint32_t)(MERLIN_CHIP_FREQ_HZ / MERLIN_SYS_CLK_HZ);

	if ((uint64_t)ratio * (uint64_t)MERLIN_SYS_CLK_HZ != (uint64_t)MERLIN_CHIP_FREQ_HZ) {
		return 0;                       /* not an integer multiple: stay on the reset clock */
	}

	merlin_set_all_clocks((uint32_t)MERLIN_CLKSEL_SLOW);

	wr32(pll, MERLIN_PLL_PLLEN_OFF, 0);
	wr32(pll, MERLIN_PLL_MDIV_RATIO_OFF, 1);
	wr32(pll, MERLIN_PLL_RATIO_OFF, ratio);
	wr32(pll, MERLIN_PLL_FRACTION_OFF, 0);
	wr32(pll, MERLIN_PLL_ZDIV0_RATIO_OFF, 1);
	wr32(pll, MERLIN_PLL_ZDIV1_RATIO_OFF, 1);
	wr32(pll, MERLIN_PLL_LDO_ENABLE_OFF, 1);
	wr32(pll, MERLIN_PLL_PLLEN_OFF, 1);
	wr32(pll, MERLIN_PLL_POWERGOOD_VNN_OFF, 1);
	wr32(pll, MERLIN_PLL_PLLFWEN_B_OFF, 1);

	merlin_set_all_clocks((uint32_t)MERLIN_CLKSEL_PLL);

	/* Step 4. f_baud = f_sys / (DIV + 1), from the vendor's own driver. */
	wr32((uintptr_t)MERLIN_UART_BASE, MERLIN_UART_DIV_OFF,
	     (uint32_t)((MERLIN_CHIP_FREQ_HZ / MERLIN_UART_BAUD) - 1u));
	return 0;
}

/* SYS_INIT pastes the priority into a LINKER SECTION NAME, so it has to be a single numeric token --
 * `CONFIG_SERIAL_INIT_PRIORITY + 1` expands to "50 + 1" and the link fails with the memorably unhelpful
 * "Undefined initialization levels used." Hence a literal, with the relationship that actually matters
 * asserted at compile time instead of assumed: if the tree ever raises the serial priority past this,
 * the build stops rather than silently going back to letting the driver overwrite our divisor. */
#define MERLIN_SOCINIT_PRIO 55
BUILD_ASSERT(MERLIN_SOCINIT_PRIO > CONFIG_SERIAL_INIT_PRIORITY,
	     "the PLL hook must run AFTER the serial driver, or the driver's own divisor "
	     "(computed for the reset clock) overwrites the one for the post-PLL clock");

SYS_INIT(merlin_soc_clock_init, PRE_KERNEL_1, MERLIN_SOCINIT_PRIO);
