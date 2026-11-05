---
title: "Design note: accumulator loads are ordered by issue, not by completion"
kind: design
status: current
owner: verification
last_verified: 2026-10-26
related: [open_coded_nest_costs, capsule_phase_split, agentic_experiment_integrity]
code_refs: [examples/gemmini/verification/acc_race/acc_race_repro.c, examples/gemmini/verification/acc_race/vend_repro.c, examples/gemmini/verification/acc_race/classify.py, examples/gemmini/verification/acc_race/fence_vendor_resadd.py, merlin/tests/gemmini/test_acc_race_repro.py, src/merlin/targetgen/completion_order.py, src/merlin/targetgen/rtl/command_ordering.py, packages/merlin-experiments/src/merlin/targetgen/completion_order_gate.py]
---

# Accumulator loads are ordered by issue, not by completion

A program can be correct only because memory happens to answer in the order it was asked. On
Gemmini this shows up in one specific pattern: two loads into the same accumulator rows, the first
overwriting and the second accumulating. The reservation station orders the second load behind the
first only until the first has been *issued*, not until it has *completed*. When the accumulating
load's data comes back first, the overwriting load lands on top of it, and the readout is the first
operand alone.

The in-order simulators never show this. A capsule can pass the functional tier (L2) and the
cycle-accurate tier (L3) and still fail on the board. This note records the measurements that
established it (2026-09-24 and 2026-09-25) and the verification decisions that follow from them. The
reproducers are in [`examples/gemmini/verification/acc_race/`](../../examples/gemmini/verification/acc_race/README.md).

## How the failures are read

Every reproducer test classifies each wrong element by which operand survived:

- `lhs_only` means the overwriting operand survived alone. This is the race's signature.
- `rhs_only` means the accumulating operand survived alone.
- `other` means neither operand alone.

All failures below are `lhs_only`. No run produced an `rhs_only` or `other` element, so every error
is this one mechanism and not some other wrong answer.

## Evidence

### On the board

Board: the lean U250 FireSim bitstream (`gemmini_rocket_lean_u250_firesim_bitstream` in
`merlin/contract/hardware_pins.yaml`, digest `9d66b956…`). Program: the full reproducer, ELF
`deac1b50…`. Job 909 confirmed the staged binary was still intact after the run. Job 907 ran the same
ELF and printed identical lines. But its staged binary had been replaced before the post-run check,
so it is not counted.

| Test (256 × 512 tensor, 4 repetitions) | Repetitions failing | Wrong elements per repetition |
|---|---:|---|
| Overwrite, then accumulate on the second load unit, no fence (a compiler's residual add) | 4/4 | 32, 48, 96, 128 of 131,072 |
| The same with a fence between the two loads | 0/4 | 0 |
| Both loads on the same load unit | 4/4 | 320, 208, 304, 272 |
| Vendor `tiled_resadd_auto` on the 14,336 × 28 view of the tensor | 0/4 | 0 |
| Vendor call split so each accumulator loop is followed by a fence | 0/4 | 0 |
| Vendor call on the 784 × 512 view of the same bytes | 2/4 | 32, 48, all in the last 32 columns |
| Vendor call with a captured layer's scales and ReLU | 0/4 | 0 |

Spike ran the same ELF with zero wrong elements in every test.

**Whole model.** A ResNet-50 program with all 71 groups on the vendor library was rebuilt twice:

- **Control:** rebuilt against the stock harness. Its ELF (`16dd11d3…`) is byte-identical to the one
  recorded job 905 ran, which proves the blobs and flags were the same.
- **Fenced:** rebuilt against a harness whose residual add puts a fence between its two loads
  (`fence_vendor_resadd.py`).

Results:

- **Job 905 (stock):** the residual-add groups 19 and 27 exceed their declared bounds. Their maximum
  absolute errors are 53 and 46 LSB, with 69 and 101 elements over a bound of 1.
- **Jobs 913 and 915 (fenced ELF `aab155cc…`):** every group is within bound. The maximum error is
  1 LSB, or 2 against a bound of 4. The two runs' 71 per-group digests are identical. Fifty-two of
  them differ from the stock run, starting at group 19.

The program's end-to-end argmax and cosine lines read 0 on all three FireSim runs, fenced or not, so
they do not separate the two builds; the per-group bounds do. On Spike the fenced program's argmax
agrees with the reference.

### On an RTL simulator with a reordering memory model

The model is a seeded AXI memory model. It keeps AXI4 ordering per ID but completes responses with
different IDs out of order. It was built into a Verilator `GemminiRocketConfig`; engine digests are
`a974e325…` and `2ca5598e…` (the second adds latency keyed on address region).

- **Small reproducer, operands generated on the core:** the compiler-style pair failed in 5 of 10
  runs and the same-unit pair in 7 of 10, in-order runs included. The fenced pair failed in 0 of 10.
  The vendor variants failed in 0 of 10. Their operands are cache-resident, so no request reaches the
  memory model.
- **Vendor residual add on cold operands** (98 × 512, initialised data read once from DRAM): the
  stock build failed in 6 of 168 perturbed runs, with 32 to 96 wrong elements each. The fenced build
  failed in 0 of 168. Both passed in order. Every failure came from one profile: a heavy
  per-response latency tail (maximum 512 cycles, 10% of responses slow). Uniform latency,
  region-keyed latency (0 of 32) and a heavier tail (maximum 768, 20%: 0 of 32) did not trigger it.
- **An agent-authored phase-1 compiler** was run on two residual-add capsules it had already passed
  at L2 and L3, under short uniform latency:
  - the occupancy-partial capsule failed on 2 of 8 seeds (35 of 496 elements);
  - the transfer-split capsule failed on 5 of 8 seeds (12 to 41 of 1,280 elements);
  - with a fence between the loads, the transfer-split capsule failed on 0 of 8.
- **The same transfer-split program under the ladder's load-order check:**
  - loaded through the host interface: 0 of 8 seeds flagged;
  - preloaded into the backing store: 6 of 8 seeds flagged.

  Through the host interface, every operand sits in the last-level cache, so the model has nothing
  to reorder. On the fenced build, the static order trigger did not fire.

### Hardware revision

| What | Revision |
|---|---|
| Gemmini RTL | `gemmini_rtl` pin `8c3f9923…`, from a checkout whose HEAD was `63f0b68a…`, plus the reviewed `LoadController.scala` edit (bytes `780ee0a2…`). The edit removes one simulation-only assertion and changes no datapath. |
| Harness headers | `gemmini_isa_headers` pin `7c540b3a…`, from a checkout whose HEAD was `6b477a8b…`, plus the reviewed `gemmini.h` edit (bytes `007826db…`). |
| FireSim bitstream | `gemmini_rocket_lean_u250_firesim_bitstream`, `9d66b956…` |
| Simulators | Verilator `GemminiRocketConfig` `5b6c7021…`; reordering variants `a974e325…` and `2ca5598e…` |

## What follows from it

1. **An in-order simulator cannot certify this pattern.** Spike, Verilator with in-order DRAM and
   GSIM all answer in issue order. Their pass on a program that issues an overwrite and an accumulate
   into the same rows says nothing about the board. That includes the vendor library's own residual
   add, so "matches the vendor library" is not a correctness argument either.
2. **Same-unit issue does not order the loads.** Only a fence, or a dependency the hardware tracks,
   does. An emitter that overwrites and accumulates into the same rows must order them. That is a
   legality obligation, not a performance choice.
3. **A memory-order check must perturb memory the program actually reads.** That means:
   - preload the image into the backing store, not through the host interface, so operands are not
     cache-resident;
   - judge each perturbed run against a console that already passed the golden, because a reordered
     run legitimately takes a different number of cycles;
   - deal seeds across two profiles, because the two measured races need opposite settings. A
     back-to-back pair shows up under short uniform latency. A pair several loads apart, which is what
     the loop unroller issues, shows up only under a heavy per-response tail.
4. **A clean sweep is a sample, not a proof.** Eight seeds sample the schedule space. Report a clean
   sweep as "no failure in N seeds under these profiles", never as "race-free".
5. **FireSim is cycle-deterministic for a given ELF.** The same binary fails the same elements on
   every run. What changes the outcome is timing context, such as a different repetition or
   different code around the kernel. "Non-deterministic" here means timing-dependent, not random per
   run.
6. **The decision is static, at L2, over the executed stream.** The same hazard appeared in the
   scratchpad: a 16-row zero fill and the rows gathered into it, back to back in the load queue,
   wrong on FireSim in 8 of 12 runs while Spike and GSIM passed. `merlin.targetgen.completion_order`
   now decides every local memory at once. It reads the command stream the functional run executed
   (its commit log) and the target's derived ordering facts (`facts.command_ordering`, observed by
   driving the dispatch module the contract names: each funct's queue, same-queue release at issue
   or completion, tracked cross-queue overlaps, span formulas, and whether a queue's writes can land
   out of order). It reports a pair only when no ordering path exists. A flagged pair fails the
   functional tier with the command indices and rows. An undecidable stream is recorded as
   `unknown` with its reasons, never as clear. Its verdict replaces the accumulator-only rule above
   as the trigger of the perturbation sweep.
