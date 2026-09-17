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

## Fast numeric Gemmini smoke without TSI

The general whole-model path below still needs a coherent host boundary. A small self-checking
kernel does not: `gemmini_selfcheck.c` compares Gemmini's mvout buffer against an independently
computed CPU golden, then calls one of two noinline marker functions. `selfcheck_run_harness.cpp`
loads the ELF through the AXI backing store and stops when Rocket commits the pass or fail marker PC.
The result never has to leave the cache through a backdoor, and no console or TSI handshake is used.

Run it against an explicit emitted GSIM model and compiler-produced kernel object:

    python merlin/experiments/gsim_wholemodel/run_gemmini_selfcheck.py \
      --kernel-object "$KERNEL_OBJECT" \
      --gsim-model-dir "$GEMMINI_GSIM_MODEL_DIR" \
      --workdir out/artifacts/cache/gsim-gemmini-selfcheck/smoke \
      --cross-check-verilator

The completion contract is fail-closed: a pass requires the kernel entry, the pass marker, nonzero
Gemmini busy cycles, and both AXI read and write traffic. The report seals the ELF, compiler object,
emitted FIRRTL/model object, runner, and (when requested) Verilator binary. The exact same ELF is
used for the Verilator corroboration.

This is intentionally labeled `smoke_only`. GSIM is RTL-derived, but the current prebuilt emitted
model has no sealed source-revision stamp. It must not replace the L3 Verilator certification gate
until that revision and the GSIM cycle contract are pinned and reviewed. The runner does not alter
network policy and makes no network-isolation claim.

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

## The TSI route: right architecture, one signal short

The DRAM backdoor cannot answer HTIF (the handshake never leaves L1) and cannot read results back
(the final writes are still dirty). Both are host-side problems, and fesvr's `tsi_t` is the component
that already solves them: it loads THROUGH the SoC coherently and acts as the HTIF host. Only the
host-side driver was pruned — the DUT's `serial_tl_0` port is present and fully wired, and fesvr
ships the driver pre-built (`libfesvr.a`, `testchip_tsi.cc`, `testchip_htif.cc`).

`tsi_run_harness.cpp` implements it, mirroring testchipip's SimTSI tick order exactly rather than
re-deriving it (a serial handshake off by a cycle still simulates and silently moves the wrong phits).

Where it gets to, and the one thing left:

* with the STOCK bootrom the core reaches `0x10034` — the wfi-spin waiting for a TSI IPI, which is
  the correct state to be in. (A baked jump-to-DRAM ROM is WRONG for this path: it jumps before TSI
  has finished loading, and the core arrives at zeros.)
* the host side works. Measured over 500k cycles: `host_has_phit=499999`.
* the DUT never accepts one: `accepted_by_dut=0 dut_sent=0 dut_ready=0`. `serial_tl_0$$in$$ready` is
  low for the entire run.

So the remaining blocker is a single signal — the serial-TL receiver never asserting ready — and the
suspects are its clocking or reset. Driving `serial_tl_0$$clock_in` (toggled, and through reset as
well as after it) does not change it, so the phit interface's clock-domain handling under GSIM is
where to look next. The link counters in the harness are the instrument: they separate "the host is
not sending" from "the DUT is not listening", which is the distinction that took the longest to
establish and should not have to be re-established.

### Do not chase the clock — GSIM has no clock domains

The blackbox contract this model is built on states it plainly: *"GSIM abstracts Clock-typed ports
away (step() advances all registers), so the EICG clock-gate reduces to a no-op in this model and the
IO cells are plain passthroughs."*

So `serial_tl_0$$clock_in` and `clock_uncore` are inputs that GSIM does not model as clocks. Driving
them changes nothing, and the serial-TL block is ALREADY being clocked — every register advances on
each `step()`. Three experiments were spent learning this the slow way (toggle the uncore clock,
toggle the serial clock, clock it through reset); none of them could have worked, and none of them
tells you anything when it fails.

Which means the receiver holding `in_ready` low is not a clocking problem. The remaining suspects are
its reset, or a credit/init handshake the TL-over-serial adapter expects before it will accept a
phit. That is where to look, and the link counters distinguish it from a host-side fault in one run.
