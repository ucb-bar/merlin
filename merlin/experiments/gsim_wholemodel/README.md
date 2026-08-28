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
