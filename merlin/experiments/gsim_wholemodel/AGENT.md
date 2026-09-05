# AGENT.md — merlin/experiments/gsim_wholemodel

## Purpose

Run a whole-model image on a GSIM-simulated SoC. GSIM re-roots the circuit at `ChipTop`, dropping the
`TestHarness` and with it SimDRAM and SimTSI, so a model has nowhere to keep its weights, no way in,
and no way out. This supplies all three.

## What belongs here

- `boot_dram.S` — the one-word boot ROM that hands control to the DRAM-resident image.
- `model_run_harness.cpp` — loads the ELF's PT_LOAD segments into the AXI-backed store, runs, and
  reads results back out of DRAM by address.
- `axi_mem_harness.cpp` — the thin binding from ChipTop's AXI pins to testchipip's `mm_magic_t`.
- `gemmini_selfcheck.c` + `selfcheck_run_harness.cpp` — the small numeric route: the ELF compares its
  own Gemmini output and the harness observes its committed pass/fail marker PC, avoiding a cache
  backdoor for the result.
- `run_gemmini_selfcheck.py` — builds that ELF and the GSIM runner, seals hashes, and can corroborate
  the exact ELF once on Verilator. Its result is smoke-only until the emitted model revision is pinned.

## What does not belong here

- The AXI memory model itself (testchipip's `mm_magic_t`, bound by the AXI harness) — that is the
  simulator's side of the boundary, not the experiment's.
- Anything that assumes a console. There is no host to service HTIF here; results come from memory.
- Any claim that the current GSIM artifact replaces the L3 Verilator gate. It is RTL-derived and much
  faster, but remains a smoke lane until its source revision and cycle contract are sealed.

## Reading a result from this path

Absence of output is not absence of progress, and the cycle bound is the usual culprit: the first
cold-miss DRAM load retires around cycle 1044, so a run that stops in the hundreds shows every AXI
counter at zero and looks exactly like a dead memory path. Check `[axi] ar=` before concluding
anything, and check the committed PC before concluding the image is running.
