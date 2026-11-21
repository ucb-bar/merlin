/* Report the board's vector configuration, from the board itself.
 *
 * Every "what is this chip's VLEN" question in this project has so far been answered by inference:
 * the Kodiak board files put no `v` in `riscv,isa` at all and never state a width, and the only
 * numbers anywhere are per-sample `CONFIG_RISCV_VECTOR_MAX_LEN` values (256 in one sample, 512 in
 * three) which size Zephyr's per-thread SAVE AREA and therefore only bound the real VLEN from above.
 * Guessing wrong is not benign: fixed-width vector code built for 128 lands at double LMUL on a
 * 256-bit unit, which is the documented SpacemiT K1 trap (spills, no speedup).
 *
 * So: a few hundred bytes that print the answer. This ships alongside the model images and runs in
 * seconds, before anyone spends minutes uploading a multi-megabyte one.
 *
 * Printing order is deliberate — each line is useful even if the NEXT one faults, which turns a
 * hang into a located failure:
 *
 *   1. `PROBE hartid` proves the image booted and the console works at all.
 *   2. `PROBE misa` says whether the hardware even advertises V (bit 21), independent of any DT.
 *   3. `PROBE mstatus_vs` says whether vector state is ENABLED. This is the one that matters for
 *      merlin's images: `vlenb` and every vector instruction trap when VS==0, and on these boards VS
 *      is turned on by `reset.S` under CONFIG_FPU rather than by the V Kconfig.
 *   4. `PROBE vlenb` is the authoritative width in bytes (VLEN = vlenb * 8).
 *   5. `PROBE vlmax_e8` / `e32` derive it a second way, through `vsetvli` at LMUL=1, so a wrong
 *      `vlenb` cannot pass unnoticed.
 *
 * Reads are M-mode CSR reads and one `vsetvli`; nothing is written to memory-mapped state and no
 * vector data is touched, so this cannot disturb anything else on the chip.
 */
#include <stdint.h>

void console_init(void);
void htif_puts(const char *s);
void htif_putd(long v);
void htif_putc(char c);
void htif_exit(int code);

static void kv(const char *k, long v)
{
	htif_puts("PROBE ");
	htif_puts(k);
	htif_putc(' ');
	htif_putd(v);
	htif_putc('\n');
}

/* crt.S calls main() on EVERY hart, by design -- mstatus.VS and vlenb are per-hart facts and a
 * heterogeneous chip is exactly what this probe exists to reveal. But nothing serialised the shared
 * console, so on a multi-hart part the harts printed over each other and a real returned log came back
 * as interleaved characters. That is not cosmetic: it wasted the one round trip we get, and it hid an
 * under-declared VLEN behind a garbled `vlenb` line. It also raced `probe_memory`, where two harts
 * writing one address can make good DRAM report FAIL. So the harts take turns, lowest first.
 *
 * A TICKET rather than a lock, because the order matters as much as the exclusion -- a reader should
 * see hart 0 first -- and because a ticket needs no atomics: each hart only ever writes its own
 * successor's value. `probe_turn` is static and uninitialised, so it lives in .bss, which crt.S clears
 * before releasing any hart through `boot_ready`.
 *
 * Both waits are BOUNDED, and both fail toward printing rather than toward silence: a chip whose hart
 * ids are not contiguous (0 and 2, no 1) would otherwise leave the later hart waiting forever for a
 * turn that never comes, and a probe that prints nothing is indistinguishable from one that never
 * booted. The limits are order-of-magnitude, against the slowest case that matters: one hart's block is
 * a few hundred characters, i.e. tens of ms at 115200 baud, so ~1e6 cycles on a 50 MHz core.
 */
static volatile unsigned long probe_turn;

#define PROBE_TURN_SPINS  200000000UL   /* ~16 s at 50 MHz: "that hart is never coming" */
#define PROBE_QUIET_SPINS   5000000UL   /* ~0.4 s: long enough for one more hart's block */

static void probe_await_turn(unsigned long hartid)
{
	for (unsigned long spins = 0; probe_turn != hartid && spins < PROBE_TURN_SPINS; spins++) {
	}
}

/* Only the boot hart terminates the log, and only once the ticket has stopped advancing: htif_exit
 * signals the host and then spins forever, so a hart calling it mid-run would cut the other harts'
 * output off. Non-boot harts return instead, and crt.S parks them. */
static void probe_finish(unsigned long hartid)
{
	unsigned long last, quiet;

	__asm__ volatile("fence rw, rw" ::: "memory");   /* our lines are out before the next hart starts */
	if (probe_turn == hartid) {
		probe_turn = hartid + 1;
	}
	if (hartid != 0) {
		return;
	}
	last = probe_turn;
	for (quiet = 0; quiet < PROBE_QUIET_SPINS; quiet++) {
		if (probe_turn != last) {
			last = probe_turn;
			quiet = 0;
		}
	}
	htif_puts("DONE\n");
	htif_exit(0);
}

/* How fast is this core REALLY running?
 *
 * Not a curiosity. One of the two boards this ships to was left on its 50 MHz reset clock because our
 * Zephyr path never programmed the PLL, while the vendor's own demos run the part at 500 MHz -- so
 * every cycle count we reported was against a clock nobody had established. And now that there IS a
 * variant that programs the PLL, "did it take?" is a question a returned log has to be able to answer:
 * a PLL that silently did not engage looks like a model that is merely slow.
 *
 * mcycle counts core clocks; mtime counts a fixed reference the SDK states (MTIME_FREQ). Their ratio
 * over the same interval is the core frequency, measured rather than declared. Costs a fraction of a
 * second. Reported in kHz so it fits an integer print without losing anything that matters.
 */
static void probe_clock(void)
{
#ifdef MERLIN_MTIME_HZ
	uint64_t t0, t1, c0, c1;
	/* Roughly a tenth of a second of the reference clock -- long enough that the read overhead does
	 * not skew it, short enough that nobody notices. */
	const uint64_t ticks = (uint64_t)MERLIN_MTIME_HZ / 10u;

	__asm__ volatile("rdtime %0" : "=r"(t0));
	__asm__ volatile("rdcycle %0" : "=r"(c0));
	do {
		__asm__ volatile("rdtime %0" : "=r"(t1));
	} while (t1 - t0 < ticks);
	__asm__ volatile("rdcycle %0" : "=r"(c1));

	kv("mtime_hz_declared", (long)MERLIN_MTIME_HZ);
	if (t1 > t0) {
		kv("core_khz_measured",
		   (long)(((c1 - c0) * (uint64_t)MERLIN_MTIME_HZ) / ((t1 - t0) * 1000u)));
	}
#endif
}

/* Does the DRAM we linked for actually answer?
 *
 * We build gemmelos images for 1 GB because the chip's owner said 1 GB, while that chip's own linker
 * script declares 256 MB -- and the largest model needs a region well past 256 MB. If the smaller
 * number is the true one, the image writes into nothing and dies with no output, which is the exact
 * symptom we have twice mistaken for a hang. Settling that costs seconds here versus a
 * many-minute upload of the model that depends on the answer.
 *
 * Write-then-read a pattern at a few points across the region rather than walking it: a walk of
 * hundreds of megabytes is itself minutes on a slow core. `volatile` so the compiler cannot decide the
 * read is redundant, and a fence between them so the write is not still sitting in a buffer.
 */
static void probe_memory(void)
{
#if defined(MERLIN_REGION_BASE) && defined(MERLIN_REGION_BYTES)
	const uint64_t base = (uint64_t)MERLIN_REGION_BASE;
	const uint64_t span = (uint64_t)MERLIN_REGION_BYTES;

	kv("region_mb", (long)(span >> 20));
	/* Start at 1/8 rather than 0: the low part of the region holds this program. */
	for (int i = 1; i <= 8; i++) {
		uint64_t off = (span >> 3) * (uint64_t)i;
		volatile uint64_t *p;
		uint64_t want, got;

		if (off >= span) {
			off = span - 4096u;
		}
		p = (volatile uint64_t *)(uintptr_t)(base + off);
		want = 0x5A5A5A5A00000000ULL | (uint64_t)i;
		*p = want;
		__asm__ volatile("fence rw, rw" ::: "memory");
		got = *p;
		htif_puts("PROBE mem_mb ");
		htif_putd((long)(off >> 20));
		htif_puts(got == want ? " ok\n" : " FAIL\n");
	}
#endif
}

int main(void)
{
	uint64_t hartid = 0, misa = 0, mstatus = 0, vlenb = 0, vl8 = 0, vl32 = 0;

	/* Our turn first, then the console. On real silicon `console_init` programs the UART's clocks and
	 * baud divisor -- shared state -- and printing before it hangs the core, so it has to happen after
	 * the ticket and before the first `kv`. Re-running it on a later hart rewrites the same derived
	 * values, which is why every hart may safely call it rather than trusting hart 0 to have gone first. */
	__asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
	probe_await_turn((unsigned long)hartid);
	console_init();
	kv("hartid", (long)hartid);

	/* misa bit 21 ('V' is letter 22, i.e. bit index 21) tells us whether the hardware claims the
	 * vector extension at all -- the fact the device tree omits. Some cores tie misa to zero; a 0
	 * here means "unreadable/not reported", NOT "no vectors", which is why the next lines still run.
	 * The raw value prints as a NEGATIVE decimal because MXL sits in the top two bits, so the
	 * extension letters are also printed on their own -- a human reads this log, not a parser. */
	__asm__ volatile("csrr %0, misa" : "=r"(misa));
	kv("misa_ext_bits", (long)(misa & 0x3FFFFFFUL));
	kv("misa_v_bit", (long)((misa >> 21) & 1));

	/* mstatus.VS is bits [10:9]: 0 = Off (vector instructions trap), 1 = Initial, 2 = Clean,
	 * 3 = Dirty. Anything nonzero means vector state is live and the reads below are legal. */
	__asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
	kv("mstatus_vs", (long)((mstatus >> 9) & 3));

	if (((mstatus >> 9) & 3) == 0) {
		/* Reading vlenb with VS==Off traps, and a trap on someone else's bench looks like a hang.
		 * Stop here instead, having already reported the reason. */
		htif_puts("PROBE vector_state off - vlenb not read (would trap)\n");
		probe_finish((unsigned long)hartid);
		return 0;
	}

	__asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
	kv("vlenb", (long)vlenb);
	kv("vlen_bits", (long)(vlenb * 8));

	/* Independent derivation: VLMAX at LMUL=1 is VLEN/SEW, so e8 gives vlenb and e32 gives vlenb/4.
	 * If these disagree with vlenb above, the number is not to be trusted. */
	__asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vl8));
	kv("vlmax_e8", (long)vl8);
	__asm__ volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vl32));
	kv("vlmax_e32", (long)vl32);

	probe_clock();
	probe_memory();

	probe_finish((unsigned long)hartid);
	return 0;
}
