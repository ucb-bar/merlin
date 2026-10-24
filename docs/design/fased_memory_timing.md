---
title: "Design note: timing an RTL simulator's memory the way a FireSim board does"
kind: design
status: current
owner: targetgen
last_verified: 2026-10-23
related: [simulator_selection, accumulator_load_order]
code_refs: [merlin/contract/external/sim_memory/mm_latency_pipe.h, merlin/contract/external/sim_memory/mm_latency_pipe.cc, merlin/contract/external/sim_memory/mm_perturb.h, merlin/contract/external/sim_memory/mm_perturb.cc, src/merlin/targetgen/mem_perturb.py, src/merlin/targetgen/tool_cli.py, src/merlin/common/provenance.py, merlin/contract/hardware_pins.yaml, merlin/tests/targetgen/test_mem_latency_pipe.py]
---

# Timing an RTL simulator's memory the way a FireSim board does

An RTL simulator and a FireSim board can elaborate the same SoC, run the same ELF, and still disagree on
every cycle count that reaches DRAM. The RTL is the same on both: cores, caches, the accelerator and the
L2. What differs is the model behind the SoC's AXI memory port. This note covers the opt-in memory model
that removes that one difference, `mm_latency_pipe_t`. It is a cycle-faithful copy of the DRAM timing
model a FireSim bitstream runs, and it is selectable in the same Verilator variant the memory-order tier
already builds ([simulator_selection](../guides/simulator_selection.md)).

## The difference it removes

| | chipyard RTL simulator (stock) | FireSim board (FASED default) |
|---|---|---|
| model behind the AXI port | testchipip `mm_magic_t` | FASED latency-bandwidth pipe |
| read: acceptance to first beat | 1 cycle | `read_latency` cycles (30 on the pinned Gemmini board) |
| write: data and address in, to B | 1 cycle | `write_latency` cycles (30) |
| outstanding requests | reads unbounded, one write burst | `max_reads` / `max_writes` (16 / 16) |
| W accepted before its AW | no | yes |
| response order | acceptance order | acceptance order |
| address, bank or row matter | no | no |

So an L2 miss costs about 1 cycle on the simulator and 30 on the board. Bandwidth-bound streaming
largely agrees, because both are capped by the same 64-bit port at one beat per cycle. Latency-exposed
work does not: compute waiting on a move-in, fences, short bursts, and scalar code missing in L1. A schedule search
ranked on the stock simulator underprices exactly the schedules that leave memory latency exposed. A
DRAMSim2 run (`+dramsim`) is no substitute: at the simulator's 500 MHz it charges about 11 to 26 cycles
per miss depending on bank and row, which is neither the board's 30 nor address-independent like the
board's model.

## What the board runs, and where that is recorded

The pinned Gemmini Rocket bitstream (`0af6f26e`, registered as `firesim_gemmini_rocket_30mhz` and
`firesim_gemmini_rocket_u250_30mhz`) was built at FireSim `b084672c` with platform config
`BaseXilinxAlveoU250Config`. The bitstream path names that config. That config is
`WithDefaultMemModel ++ ...`. `WithDefaultMemModel` sets the model to
`LatencyPipeConfig(BaseParams(maxReads = 16, maxWrites = 16))` with no LLC model. The pipe's read and
write latencies are runtime settings that default to 30.

These values are a property of the built image, its platform config and runtime configuration, not of
the RTL it was elaborated from. So they are declared on the bitstream's pin as data:

```yaml
memory_model:
  kind: fased_latency_bandwidth_pipe
  read_latency: 30
  write_latency: 30
  max_reads: 16
  max_writes: 16
  evidence: >-
    ...where each value was read...
```

`provenance.load_artifacts` checks the shape: a string `kind`, and integer or string values. A float,
list or nested mapping is refused. `mem_perturb.latency_pipe_knobs` checks the meaning: only
`fased_latency_bandwidth_pipe` is accepted, all four knobs are required, and each must be at least 1.
A bitstream with no declaration has no timing to reproduce, and asking for one is refused, never
defaulted. FASED's DDR3 models (FCFS, FR-FCFS) and its LLC model are different models, so a declaration
naming one is refused rather than approximated by the pipe.

One link in the chain is still asserted rather than read from bytes: a run's `runtime.conf` can
override the latencies. No run on this bitstream records a custom runtime configuration. Reading the
`runtime.conf` inside the bitstream tar on the lab host is the remaining byte-level check.

## The model

`merlin/contract/external/sim_memory/mm_latency_pipe.{h,cc}` reproduces FASED's
`SplitTransactionModel`, `LatencyPipe` and `AXI4Releaser`, read from firesim
`sim/midas/src/main/scala/midas/models/dram/` at `b084672c`:

| FASED | `mm_latency_pipe_t` |
|---|---|
| `ar.ready := ~pendingReads.full`; the counter counts AR fire to the last R beat | `ar_ready()`: `pending_reads < max_reads` |
| `aw.ready := ~pendingAWReq.full` (AW fire to B fire) | `aw_ready()`: `pending_aw < max_writes` |
| `w.ready := ~pendingWReq.full` (complete W bursts to B fire) | `w_ready()`: `pending_w < max_writes`. W is accepted before, with or after its AW |
| `newWReq`: the later of AW fire and last-W fire pairs a write | W beats paired with AWs in order; the write completes when its pair does |
| `releaseCycle := latency + tCycle - egressUnitDelay` (delay 1), then the one-entry `AXI4Releaser` queue | answered in cycle `t + latency` |
| `readPipe` / `writePipe` FIFOs, released back to back | `rq` / `bq` FIFOs, one response at a time, no bubble |

**Cycle accounting.** testchipip's SimDRAM calls `tick()` on the clock edge and registers the outputs
it reads afterwards for the next cycle. So every output is a function of the state left by the
previous tick, as each FASED output is a function of its registers. A request accepted in cycle `t`
enters FASED's pipe with release `t + L - 1`, leaves it that cycle, and is presented from the releaser
register one cycle later, at `t + L`. The C++ model releases it at `t + L` directly. At `L = 1` that is
the cycle after acceptance, which is `mm_magic_t`'s timing. Measured below, a latency-1 pipe reproduces
the stock simulator's cycle count exactly.

**Data** follows `mm_magic_t`: read data is sampled at AR acceptance, and a write lands when its W beat
is paired with its AW.

## Using it

1. Build the variant as for the memory-order tier. The same binary now carries three models: stock
   (default), perturbed (`+mem_perturb_seed`) and the pipe (`+mem_latency_pipe=1`). Its receipt lists
   every model source and a `models` table.
   `merlin-target-tools mem-perturb-variant --obj-dir DIR --makefile MK --out FRESH_DIR [--base-artifact PIN]`
2. Run an ELF under a pinned board's memory timing, or in order for the A/B:
   `merlin-target-tools mem-model-run --emulator FRESH_DIR/emulator --elf PROG --firesim-artifact firesim_gemmini_rocket_u250_30mhz`
   `merlin-target-tools mem-model-run --emulator FRESH_DIR/emulator --elf PROG --in-order`
   The knobs go through the environment, so the command line is the one the receipt records, with
   its preload. The JSON names the bitstream and its digest beside the result.
3. Believe a timed run only when `model_confirmed` is true, that is, when every memory channel's factory
   line in the console names the model the run asked for. An engine built before the pipe existed
   ignores its knobs and answers in order. `mem-model-run` exits 2 in that case.

From Python: `mem_perturb.pinned_latency_pipe(artifact)` gives the knobs and `mem_perturb.run_under(...)`
runs and records. By hand: `+mem_latency_pipe=1 +mem_latency_pipe_read_latency=30 ...`, or the
`MERLIN_MEM_LATENCY_PIPE[_<KNOB>]` environment variables. Asking for both the pipe and a perturbation
seed is a hard error, because they answer different questions.

## How it is verified

`merlin/tests/targetgen/test_mem_latency_pipe.py` builds the model against the real testchipip
`mm.h`/`mm.cc` of a chipyard build:

- **Against FASED's Chisel, cycle for cycle.** The reference is `FasedPipe`, a transcription of the
  Scala register for register: its three `SatUpDownCounter`s, its flow queues and its `pipe=true`
  releaser. It is written as the Chisel is written, not as the C++ is. A random, legal AXI4 master
  holds each request until it fires, paces W independently of AW, and stalls R and B. It drives the
  C++ model and records every cycle. The reference replays the master's half and must agree on the
  model's half (every ready, every valid, and every R id, last flag and data word) on every cycle. That
  is six configurations: FASED stock, stock saturated, latency 1, tight bounds, one outstanding read,
  and heavily backpressured. Each runs under three stimulus seeds, with write data checked at the end.
- **The traffic reaches the corners.** A separate test requires that the read bound and the write bound
  each backpressure, that R stalls, and that some W burst completes before its AW. Otherwise agreement
  on those paths would be vacuous.
- **Mutants fail.** Four plausible misreadings are built and each must disagree with the reference
  somewhere: answering a cycle early (the releaser register forgotten), acknowledging writes a cycle
  late, no read bound, and bounding W by beats instead of bursts.
- **Direct properties.** Answers come exactly `read_latency` after acceptance, and exactly
  `write_latency` after the later of AW and last W. Exactly `max_reads` reads are accepted when none
  drain. Off means the stock model. Bad knobs abort. The environment spelling is the same model.
- **The build.** A miniature Verilator-shaped build is compiled, then rebuilt as a variant into a
  relative output directory. Its emulator answers in 1, 9 and perturbed modes as asked.

Measured on real RTL, 2026-10-23. A variant was built from the pinned Gemmini Verilator build
(chipyard `009e85b0`, Gemmini `63f0b68a`, the bitstream's revisions). Its control relink is
byte-identical to the stock simulator. On a small TVM-compiled Gemmini program:

- **Default.** The run reported `in_order` and reproduced the stock simulator's cycle count exactly.
- **The board's 30/30/16/16.** The run reported `latency_pipe` and took 0.70% more cycles, with
  byte-identical program output.
- **Latency 1.** The pipe matched the stock count exactly, which confirms the cycle accounting above.

That program is mostly L2-resident, so the delta is small. Programs that stream from DRAM are where the
models part. The full record, with provenance, is the `verification/gemmini` product under the artifacts
root.

**Not yet verified:** agreement with a board measurement of the same ELF. No FireSim-measured
executable was available where this was built. The next step is to run an ELF already measured on
`alveo_u250_firesim_gemmini_rocket_30mhz` through `mem-model-run --firesim-artifact` and compare cycle
brackets.

## What it does not cover

- **DRAM.** This is a copy of a model, not of DRAM. It makes a simulator agree with a board that runs the
  same model, and it says nothing about a chip with a real memory controller.
- **The board's other differences.** FireSim's tweaks run every bus at 1 GHz where the simulator runs
  them at 500 MHz. Both are 1:1 ratios, so cycle counts are not skewed. The serial host link is 4 bits
  wide on the board and 32 in the simulator, so keep host calls outside timed regions. Target memory is
  16 GiB on the board and 256 MiB in the simulator.
- **Other bitstreams.** Only the `0af6f26e` bitstream declares a memory model. Any other needs its own
  declaration before its timing can be reproduced.
