/* Runner-owned rv64 SoC carrier for the Muon RTL oracles.
 *
 * WHY THIS EXISTS. The stock carrier (radiance-kernels soc/main.c) is `for(;;) sink++;` — Rocket spins
 * while the Muon cores compute, and the kernel's own OUT/DONE bytes go to IO_COUT_ADDR
 * (SMEM_BASE + (1<<19) = 0x1FF080000 in SoC-physical), which the elaborated address map does not map at
 * ALL. Cyclotron mirrors that aperture to stdout; the RTL does not, so on Verilator/GSIM every byte is
 * stored into a hole. Measured: uart_chars=0 on every capsule, and every RTL grade degraded to
 * completion-only. (The comment blaming a `$finish`/flush race is wrong — GSIM flushes per character.)
 *
 * WHY THE CARRIER AND NOT THE KERNEL. Rocket is the only side that can reach the console: walking the
 * elaborated diplomatic graph, 9 of 105 master names reach `serial@10020000` — `Core 0 DCache` among
 * them — and ZERO of the 702 Muon master edges do. Pointing the kernel's putchar at the UART therefore
 * cannot work; the bytes have to leave through Rocket.
 *
 * MU_UART_BASE is supplied by the builder from the target's own elaborated address map (the device
 * tree's `stdout-path`), never hardcoded here — a literal would bake in one config's map.
 */
#include <stdint.h>

#ifndef MU_UART_BASE
#error "MU_UART_BASE must be defined by the builder from the target's derived console fact"
#endif

/* SiFive uart0 register file. txdata's bit 31 reads back as FULL, so a write is only accepted when it
 * is clear — polling it (rather than blind-storing) keeps the carrier safe on a sim whose UART
 * backpressures, since a dropped byte would look exactly like the bug this replaces.
 *
 * TXEN IS NOT OPTIONAL. A uart0 comes out of reset with the transmitter DISABLED: storing to txdata
 * before setting txctrl.txen is accepted by the bus and transmits nothing, so the harness's SimUART
 * never sees out_valid and the console stays empty — indistinguishable, from the outside, from the
 * unmapped-aperture bug. Measured exactly that: Rocket reached the carrier 73,682 times and
 * uart_chars stayed 0 until this enable was added.
 */
#define UART_REG(off)  (*(volatile uint32_t *)((uintptr_t)(MU_UART_BASE) + (off)))
#define UART_TXDATA    UART_REG(0x00)
#define UART_TXCTRL    UART_REG(0x08)
#define UART_DIV       UART_REG(0x18)
#define UART_TXFULL    (1u << 31)
#define UART_TXEN      (1u << 0)

static void uart_init(void)
{
    /* Enable TX and DO NOT TOUCH div. Rocket executes the BootROM before ever reaching this carrier
     * (traced: pc walks 0x10000.. long before 0x80000000), and the BootROM programs the divisor for the
     * clock it was built against. Writing our own rate over it produced exactly one 0xFF underrun frame
     * and then wedged the FULL poll below — a carrier that hangs the run is strictly worse than one that
     * prints nothing. Deriving a divisor here would need the clock fact; inheriting the one the boot
     * flow already set needs nothing and is right on silicon too.
     */
    UART_TXCTRL |= UART_TXEN;
}

static void up(char c)
{
    /* BOUNDED wait. An unconfigured or absent transmitter leaves FULL set forever, and an unbounded
     * spin there turns "the console does not work" into "the run hangs" — the same class of failure,
     * made much more expensive to diagnose. Give up on the byte instead; a short console is a visible
     * symptom, a wedged simulation is not. */
    for (int spin = 0; spin < (1 << 20); ++spin) {
        if (!(UART_TXDATA & UART_TXFULL)) {
            UART_TXDATA = (uint32_t)(unsigned char)c;
            return;
        }
    }
}

static void us(const char *s)
{
    while (*s) up(*s++);
}

int main(void)
{
    /* Stage 1 (this commit): prove the channel. If this string appears on the RTL console, Rocket can
     * reach the UART and `uart_chars` — the counter that exposed the original bug — is nonzero.
     * Result readback lands on top of this once the channel is established; doing it the other way round
     * would mean debugging a readback contract through a console that was never shown to work. */
    uart_init();
    us("MU_CARRIER_ALIVE\n");

    /* Hand the machine back exactly as the stock carrier did: the Muon cores are released and run to
     * completion independently, and the harness still decides the run is over on its own terms
     * (GPU-idle / finished-execution). Spinning here changes nothing about that contract. */
    for (;;) { __asm__ volatile("wfi"); }
    return 0;
}
