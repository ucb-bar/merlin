# Running a whole model on a GSIM SoC

GSIM re-roots the circuit at `ChipTop`, which drops the `TestHarness` — and with it SimDRAM and
SimTSI. The documented flow works around that by running a DRAM-free RoCC stream, which is enough to
watch controller FSMs and nothing like enough to run a model: a model needs its weights somewhere.

Three pieces close that gap, and each replaces something the pruned harness used to provide.

| what was lost | replacement |
|---|---|
| SimDRAM | `axi_mem_harness.cpp` binds testchipip's `mm_magic_t` to the chip-boundary AXI port |
| SimTSI (loading) | write the ELF's `PT_LOAD` segments straight into the backing store, then boot at `0x8000_0000` |
| HTIF (results) | read the output buffer back out of DRAM by symbol — no console needed |

## Three findings worth keeping

**The DRAM path was believed dead and is not.** A probe that preloads two distinct values and loads
them retires both with exactly the preloaded bytes. What made it look dead was the cycle bound: the
first cold-miss load retires around cycle 1044, and the probe harness stopped at 400. At that bound
every AXI counter reads zero, which is indistinguishable from a request that never left the SoC.

**Size the backing store from the image, not from a guess.** A whole-model ELF puts its weights blob
high — measured, a `0x2_0000_0000` physical address — so a 1 GB window loads the text and silently
drops the weights. A model whose weights never arrived still runs, reading whatever the store held.

**`lui` sign-extends on RV64.** `lui t0, 0x80000` lands `t0` at `0xffff_ffff_8000_0000`, so the boot
jump misses the image entirely and the core wanders off. Build the constant by shift instead; it is
unambiguous at any XLEN. The commit trace shows this one plainly, which is the only reason it was
cheap to find.

## Use

    boot_dram.S            -> assemble, pack to 64-bit words, patch into ChipTop0.cpp's bootrom[]
    model_run_harness.cpp  -> link with ChipTop0.o, axi_mem_harness.o, mm.o, blackboxes.o
    ./model_run <elf> [cycles] [dump_addr dump_len]

`dump_addr` is the output buffer's address from the ELF's symbol table; the run prints its bytes as
`DUMP <addr> <len>` followed by hex, which is the whole result-extraction path.

## Where this currently stops, and why it is not the memory path

The image loads and runs. It stops inside `htif_puts`: `sd` to `tohost`, then a poll of `fromhost`
until a host acknowledges. There is no host here, so it spins — measured, 1.6M cycles in a
two-instruction loop.

Three ways out, and each is closed for a different reason. They are worth writing down together,
because each one individually looks like the thing to try next.

**Service HTIF from the harness.** Cannot work at this level. `tohost` (0x8004_4000) and `fromhost`
(0x8004_4008) are eight bytes apart, so they share one 64-byte cache line and the whole handshake
completes in L1 without reaching memory. Measured: after 2M cycles with `ar=3624 aw=554 w=4432`, and
`last_ar` pointing at that very line, DRAM there still reads all zeros.

**Use the UART instead.** The chip does expose `uart_0$$txd`, so the pin is observable, and this is
the right shape of answer. But building an image with `console='uart'` requires `sdk_dir` +
`sdk_chip`: the UART divisor and PLL facts are derived from a target SDK's own headers rather than
hardcoded, and no SDK describes this chipyard SoC.

**Read the results out of DRAM instead of printing them.** `gemmini_dram_read` makes this trivial for
any address, and the output buffer is a symbol (`OUT`). The catch is the same cache: the final writes
to it are the last thing the program does, so they are still dirty when the run stops, and nothing
here can force a writeback.

So the remaining gap is the HOST-INTERACTION boundary, not the memory boundary — which is the
opposite of what the earlier notes claimed. The tractable next step is a coherent path from the
harness into the memory system (what TSI provided before it was pruned), after which both the HTIF
handshake and the result readback become straightforward.
