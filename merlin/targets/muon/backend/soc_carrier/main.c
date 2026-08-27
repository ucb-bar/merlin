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

/* SiFive uart0 register file: txdata at +0x00. Bit 31 of txdata reads back as FULL, so a write is only
 * accepted when it is clear. Polling it (rather than blind-storing) is what makes the carrier safe on a
 * sim whose UART backpressures — a dropped byte here would look exactly like the bug this replaces. */
#define UART_TXDATA (*(volatile uint32_t *)((uintptr_t)(MU_UART_BASE) + 0x00))
#define UART_TXFULL (1u << 31)

static void up(char c)
{
    while (UART_TXDATA & UART_TXFULL) { /* wait for room */ }
    UART_TXDATA = (uint32_t)(unsigned char)c;
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
    us("MU_CARRIER_ALIVE\n");

    /* Hand the machine back exactly as the stock carrier did: the Muon cores are released and run to
     * completion independently, and the harness still decides the run is over on its own terms
     * (GPU-idle / finished-execution). Spinning here changes nothing about that contract. */
    for (;;) { __asm__ volatile("wfi"); }
    return 0;
}
