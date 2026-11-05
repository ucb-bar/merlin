---
title: "Design note: what open-coding the mesh stream costs, and which lever pays where"
kind: design
status: current
owner: compiler
last_verified: 2026-10-07
related: [verified_compiler_roadmap, accumulator_load_order, performance_levers_per_archetype, macro_scheduling]
code_refs: [examples/gemmini/verification/probe_load_placement.py, merlin/tests/gemmini/test_load_placement_probe.py, merlin/contract/perf_reference_targets.yaml]
---

# What open-coding the mesh stream costs, and which lever pays where

A Gemmini program can tile a contraction in two ways:

- **Through the hardware loop sequencer** (`LOOP_WS`, `LOOP_CONV_WS`). The accelerator expands one
  descriptor into the whole mesh command stream itself.
- **By open-coding the stream** from the host: explicit `config`, `mvin`, `preload`, `compute` and
  `mvout` instructions. This is what a compiler's schedule produces once the sequencer is prohibited.

A cycle count from the first measures the hardware's tiling choice, not a compiler. This note records
what the second costs on the same device and what closes the gap. The measurements were taken on
2026-09-23.

## The whole-model price of leaving the sequencer

These are three entries in `merlin/contract/perf_reference_targets.yaml`. All of them have program
identity `compute_group_standalone_image`, the same 71-group ResNet-50 image as job 728.

| Job | Device | Kernel bodies | Whole-model cycles |
|---|---|---|---:|
| 728 | `firesim_gemmini_rocket_u250_30mhz` | 63 `LOOP_WS` + 31 `LOOP_CONV_WS` descriptors | 24,359,749 |
| 759 | `firesim_gemmini_rocket_u250_30mhz` | explicit nests in 70 of 71 groups | 43,900,637 |
| 760 | `firesim_gemmini_rocket_u250` | the job 759 ELF | 43,900,637 |
| 761 | `firesim_gemmini_rocket_u250` | the job 728 ELF | 24,359,749 |

- **Same device and scope:** open-coding costs 1.80× overall:
  - convolutions 2.11× (12,167,031 → 25,639,926 over 20 groups);
  - matmuls 1.66× (8,399,110 → 13,953,532 over 34);
  - residual adds 1.13× (3,782,243 → 4,288,294 over 16);
  - the four deep-stage convolutions 3.9× to 5.3×.
- **Group 1:** this group pools its readout, and the explicit nest does not spell the store
  controller's pooling descriptor. It keeps the vendor call in both arms (1,986,790 cycles), so the
  explicit program is not FSM-free.
- **Gate:** every row matches 71 of 71 per-group checksums against the numpy emulation of the same
  integer program, with argmax 21 and cosine 997,981 ppm. The rows are recorded as observed, not
  sealed, for the reason the ledger states.
- **Device control:** both ELFs gave cycle counts identical to the last digit on both registered
  U250 bitstreams. That is a measured equality for this program, not a general licence to compare
  across those two devices.
- **Reproducibility gap:** the emitter that built the explicit ELF (`f16f16c8…`) was never committed.
  The result is bound to the ELF bytes, not to a revision of this repository.

## Which placement lever closes it

The [placement probe](../../examples/gemmini/verification/probe_load_placement.py) builds one
bare-metal binary that carries the same contraction under several policies. Every arm runs the same
weight-stationary loop order over the same blocking and issues the same transfers to the same
addresses. Arms differ only in:

- **load configuration states:** one shared state, or one per operand;
- **move-in placement:** a burst before the contraction, or each tile just before its first use;
- **readout placement:** a block readout afterwards, or fused into the last contribution;
- **the stationary operand:** named once per run of moving rows, or on every compute.

The sequencer runs in the same binary as the element-for-element oracle, and every arm is bit-exact
against it. Counts compare only within one binary. A second binary with six arms in a different
order moved individual arms by up to 7.2%. That is enough to reverse a small effect but not a large
one.

Engine: GSIM `1a3de02a…` built from FIRRTL `089d053b…` (a 38-member GSIM-vs-Verilator equivalence
certificate). RTL: the `gemmini_rtl` pin `8c3f9923…`, from a checkout whose HEAD was `63f0b68a…`,
plus the reviewed `LoadController.scala` edit. Scope: one call of one contraction, CPU cycle counter
across the call including the closing fence.

| Shape (M×N×K), rows per tile | Sequencer | Baseline (1 state, burst, reuse) | Move-ins at point of use | Stationary reload instead of reuse |
|---|---:|---:|---|---|
| 64×512×1024, 16 | 142,070 | 229,372 | −22.3% (3 states) | +1.5% under the burst, +9.5% at point of use |
| 64×256×1024, 16 | 73,112 | 97,676 | −12.7% (3 states) | −0.6% under the burst, +10.6% at point of use |
| 49×512×1024, 7 | 140,456 | 269,548 | −0.6% (second binary) | +21.7% (removing it: −17.8%, −19.8% in two binaries) |
| 3136×64×64, 16 (ResNet-50 group 2) | 62,297 | 134,986 | −3.2% (3 states), +0.6% (1 state) | +32.6%, execute-controller busy +88% |

The best arm against the baseline was −22.5% on 64×512×1024 and −14.3% on 64×256×1024. That closes
59% and 57% of those shapes' gaps to the sequencer, and nothing on group 2's shape.

### What the counters show

The load and execute pair counters explain the move-in result. On 64×512×1024 the hoisted burst
leaves 92,031 cycles where only the load controller is busy and 1,023 where load and execute overlap.
Moving each transfer to its point of use turns that into 4,225 and 117,075.

The burst is several hundred transfers deep. The load reservation station holds 8 entries (read from
the RTL, not assumed). Once the transfers arrive a few at a time, compute can always be allocated
behind them. The instruction stream is the same in both arms; only its order changes.

### Results that did not pay

- **Load configuration states are not the constraint.** One state against three measures within
  0.3% when moves are at point of use, and between −0.4% and +0.5% everywhere. The burst depth is
  what limits; the state count is not.
- **Fused readout pays only alongside point-of-use moves.** It is −0.4% and −1.4% there, and a 3.7%
  loss on group 2's shape. It does lower peak accumulator pressure.

## Decisions

1. **The two levers are anti-correlated, so a scheduler chooses per geometry.**
   - A hoisted burst costs runtime only where the load side is the critical path.
   - A stationary reload costs runtime only where the mesh is.
   - No measured shape was limited by both, and neither change was ever a loss.

   Applied globally, either one is a no-op on half the shapes. They belong in the schedule search as
   per-group choices, not as a fixed recipe.
2. **The predicate is computable without a simulator.** A group is load-limited when its block's
   transfer count times rows per transfer exceeds the mesh work the block issues. A scheduler that
   already picks a blocking has both numbers.
3. **For convolutions, weight reuse comes first.** At a 7-row output tile (the geometry of the four
   worst deep-stage convolutions) the shift-in is amortised over 7 streamed rows instead of 16, and a
   reload per compute costs 22%. The same change measured on a 16-row matmul tile under a burst reads
   as nothing. That is how it went unnoticed.
4. **Do not spend effort on load-state hoisting.**
5. **Whole-model effect: an estimate, not a measurement.** Applying these per-shape ratios to job
   759's per-group cycles estimates 36.4M to 39.5M cycles, against 43.9M today. That is necessary
   and visibly not sufficient. The probe contains no convolution address stream, and the residual-add
   and mean groups were not in scope. Do not quote the estimate as a result.
