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

int main(void)
{
	uint64_t hartid = 0, misa = 0, mstatus = 0, vlenb = 0, vl8 = 0, vl32 = 0;

	/* Console first, or there is nothing to read the answers on. On real silicon this programs the
	 * UART's clocks and baud divisor; printing before it hangs the core. */
	console_init();

	__asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
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
		htif_puts("DONE\n");
		htif_exit(0);
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

	htif_puts("DONE\n");
	htif_exit(0);
	return 0;
}
