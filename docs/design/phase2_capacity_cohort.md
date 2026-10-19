---
title: "Design note: the Phase 2 capacity cohort, its scope, its roofline and its calibration check"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-17
related: [merlin_supplied_host_placement, spike_timing]
code_refs: [packages/merlin-experiments/src/merlin_experiments/phase2/metric_scope.py, packages/merlin-experiments/src/merlin_experiments/phase2/public_program_measurement.py, packages/merlin-experiments/src/merlin_experiments/phase0/form_perf.py, packages/merlin-experiments/src/merlin_experiments/phase0/capacity_families.py, packages/merlin-experiments/src/merlin_experiments/phase2/claims/capacity.py, packages/merlin-experiments/src/merlin_experiments/phase2/roofline_priority.py, packages/merlin-experiments/src/merlin_experiments/phase2/calibration_validation.py, src/merlin/perf/movement_peak.py, src/merlin/perf/staged_roofline.py, experiments/templates/phase0/performance.yaml]
---

# The Phase 2 capacity cohort

Phase 2 optimizes what decides accelerator performance:

- tiling, blocking and the dataflow choice;
- DMA scheduling and double-buffering;
- scratchpad and accumulator reuse;
- epilogue fusion and command-queue pressure.

The previous cohort could not exercise any of these. The form-perf and PK members top out around
M≤17, N≤32 and K=128. They fit every on-chip store many times over, so none of those decisions
changes their cycles. Phase 2 iterates on a calibrated Spike timing model with no RTL simulator in
the loop, so large shapes are affordable. This note covers four changes:

- the metric scope;
- the capacity-scaled families;
- the roofline used to prioritize and stop;
- the one-time calibration check on the RTL simulator.

## A. What a Phase 2 score contains

The scored scope comes from the frozen Phase 0 placement (`merlin.targetgen.frozen_placement`; see
`merlin_supplied_host_placement.md`). It never comes from candidate output.
`phase2.metric_scope` applies it as follows.

| Operations | Treatment |
| --- | --- |
| Placed on the accelerator | Scored: the candidate's work |
| Unplaced support (packing, layout, im2col, residency, readout bookkeeping, command issue) | Scored: host code that feeds the accelerator |
| Placed on the host | Excluded: Merlin-supplied host compute |

**Public-program route.** The candidate-whole-program route lets the candidate emit host code. It is
now selected only when the frozen placement keeps no compute on the host. A mixed capsule returns no
selection and takes the owned-host split, where Merlin compiles the host part. A capsule with no
frozen placement, or with nothing on the accelerator, is refused.

**Projection.** The projection re-binds the placement to the pinned program bytes and records
`metric_scope` and `scored`. Paired measurement records the scope on every model row.

**Mixed programs.** A cycle count from a mixed program still contains Merlin's host stage.
`scored_cycles` therefore returns UNKNOWN unless that stage was measured. `paired_delta` admits a
candidate-minus-baseline delta only when both arms link byte-identical Merlin host code, because the
host stage then cancels.

**Form-perf.** Shares are now computed over accelerator groups only. A host-placed group no longer
makes its application unpriced. A host-placed class is recorded as `merlin_supplied_host`, and it is
neither minted nor required. Scopes frozen before this change get the same treatment.

## B. Capacity-scaled families

The `PX` family in `experiments/templates/phase0/performance.yaml` declares policy only: which kinds
to build, at which multiples of which capacity. `phase0.capacity_families` reads every capacity from
the target's RTL facts:

- array geometry, element widths, store banks and capacities (`derived_bound.machine_from_facts`);
- the movement ports (`perf.movement_peak`);
- per-class dispatch entries plus command-queue depth;
- the dataflow modes (`timing_structure`).

Each member records:

- its family, variant and regime;
- the regime variable `x`;
- why it sits in that regime, with the numbers;
- its staged roofline.

A regime that lands fewer points on a target than the contract needs is withheld and reported. A
family the facts cannot host, for example one with no runtime dataflow choice, is skipped with the
reason.

**Example shapes.** The table uses the gemmini facts-v6 values:

- 16×16 mesh and 256 MAC/cycle;
- 256 KiB operand store in 4 banks and 64 KiB accumulator in 2 banks, with 2 loop contexts;
- 16 B/beat read and write ports, each with 16 in flight × 64 B;
- live command windows of 26 (execute), 16 (load) and 6 (store).

| Family | Exercises | Examples (M×K×N) |
| --- | --- | --- |
| knee | capacity knee, multi-level blocking, compute/DMA overlap | fits_double 64×464×48, 80×640×64; fits_single 96×816×112, 128×976×112, 128×1008×128; spills 128×1040×128, 176×1424×192, 256×2048×256 |
| deep_k | multi-panel reduction, reuse across many blocks | 80×{320, 656, 1072, 1472, 3280, 6560, 13104}×80 |
| short_m | DMA-bandwidth bound, short M and large N | M∈{1, 8}, K=256, N∈{208 … 4096} (movement-limited floor) |
| tails | non-aligned tails | 80×640×64 and 128×976×112, each with M+1, K+1, N−1 and all three |
| conv | strided, padded, pointwise and stem convolutions | 3×3 s1/s2 p1 at ci=64 on images of 32–128; 1×1 at ci=128; 7×7 s2 p3 at ci=3 on images of 147–591 |
| readout | epilogue, full-width accumulator readout | 176×16×192 to 368×16×352, full-width vs acc_scale vs relu+requant |
| chain | cross-op residency | 1, 2, 4 or 8 ops sharing a 512×64 weight (resident) or a 4096×64 weight (reloaded), M=64 |
| queue | command-queue back-pressure | one-panel streams of 4–380 output tiles against the 6-command store window |
| dataflow | dataflow choice | equal work as rows-heavy, depth-heavy or balanced, e.g. 4096×16×64, 64×1024×64 and 512×128×64 |

Per-channel and dynamic-scale epilogues need a direct builder that carries them. No such builder
exists yet, so they are not minted.

**Claims.** `phase2.claims.capacity` (`perf_capacity_claim.analyze_capacity_claim/v1`) fits each
`(family, variant, regime)` group separately. Within a group, cycles must be affine in `x` with a
positive slope, and r² and residuals must stay inside the frozen thresholds. A measurement below its
floor refutes the floor.

The analyzer reports slope ratios between adjacent regimes and asserts nothing about them. Profit
depends on the regime and can be negative: fewer commands can be slower once overlap is lost. The
claim is a set of per-regime scaling laws, never "fewer is better".

## C. The roofline

`merlin.perf.movement_peak` reads the movement peak from `timing_structure.memory_clients`:

- the peak is beat bytes × channel beats per cycle;
- direction is read structurally (a client with data on the response reads);
- the in-flight window becomes a binding latency, above which it limits the rate: 1024 B / 16 B per
  cycle = 64 cycles on facts-v6;
- the window caps the sustained rate only when a fitted round trip is supplied.

`derived_bound` and `capsule_roofline` accept this as a movement fact. Before this change, every
target's movement term was UNKNOWN.

`merlin.perf.staged_roofline` keeps the stages separate:

- **compute:** useful and padded MACs over the mesh peak;
- **movement:** unique, requested (best legal tiling) and physical (whole beats per row segment)
  bytes, with read and write priced on their own ports;
- **host:** the accelerator-affecting host stage, which is UNKNOWN until modelled.

Merlin-supplied host compute is recorded and excluded.

Compute and movement overlap (taking the max) only when the stores admit a second loop context.
Otherwise they are summed, and a serial host stage is always added. The attainable rate is quoted in
useful MACs per cycle.

These existing stage formulas have limits that the guidance gate does not repair.
Padding rounds M and K by array rows and N by array columns. Two resident contexts
permit the overlap composition without proving actual queue, DMA or bank scheduling;
unknown buffering currently selects the serial sum without establishing that execution
must serialize. Movement takes the maximum of read and write costs under an independent-port
premise. Custom LHS byte volumes use unrounded requested bytes; in the dense-LHS beat branch,
a custom RHS byte override is not charged to physical reads. Planned tiling and declared
residency also do not reconstruct actual emitted transfers, and accelerator-feeding host cost
is an externally supplied term. Reuse only within reviewed accounting premises; an omitted
or unproved term cannot become known merely because a numeric floor exists. Correcting these
formulas or replacing frozen cohort floors requires a separate reviewed change.

`phase2.roofline_priority` ranks members by worst-repeat attainment (floor / measured), furthest
below the roofline first. Partial numeric floors remain useful for exploratory ranking, but their
attainment cannot establish completion. It reports `converged` only when every repeat has a fully
derived floor without unresolved terms and every member reaches the declared
`roofline_target.attainment_min` (0.8) on its worst repeat. Missing terms still block completion if
the source document labels its remaining bound `derived`. Per-repeat status and original unresolved
reasons are retained in `floor_evidence`, `partial_floor`, `unknown_floor` and `floor_refuted`.
Below-target numeric gaps can still yield `continue`; once those gaps meet the target, any partial
or unknown floor yields `undecided`. A measurement below even a partial floor yields `floor_refuted`.
The V1 guidance adds explicit convergence eligibility and completion blockers without changing
floor arithmetic, cohort membership, objective weights or the frozen attainment target.

The existing `search_priorities` savings view and the feedback roofline cells are unchanged. They now
receive a movement term on the facts-only route as well.

## D. Calibration validation (design and selection; not run)

This step runs once, after Phase 1 and before Phase 2, on the RTL simulator (`$target_oracle:L3`).
`phase2.calibration_validation.select` takes cohort members by family and regime, as declared under
`requires_capacity_scope.calibration_validation`. It picks the cheapest member by staged floor unless
the policy says `costliest`. Packet-width points come from the ports.

Each point is a pair of schedules for the same shape. `decide` qualifies Spike ranking in a category
only when Spike orders at least 90% of the pairs the RTL simulator separates by more than 5% the same
way. Those are the native-timing gates of `docs/guides/spike_timing.md`.

Each category's verdict means the following:

- a missing observation leaves the category UNKNOWN;
- a qualified category with `reconcile_with_board` must still be reconciled against the board, since
  gSIM and stock FireSim are reported to disagree in those categories;
- a failing category is ranked by RTL measurement or reported as UNKNOWN.

Points selected on facts-v6 (19 shapes):

| Category | Points | Reconcile with board |
| --- | --- | --- |
| per_command_issue_cost | 32×16×32 (within window), 80×16×80 (back-pressure) | yes |
| dma_packet_width | 16×64×{16, 64, 65}: one beat, the largest request, one element past it | yes |
| dma_compute_overlap | 64×464×48, 128×1008×128, 128×1040×128 | yes |
| bank_conflicts | 64×464×48 | yes |
| configuration_and_fence_drains | chain of 1 and of 8 ops on a resident 512×64 weight | yes |
| epilogue_store_cost | 176×16×192, full-width vs relu+requant | no |
| looped_vs_unrolled | 96×816×112, 80×3280×80 | yes |
| short_tile_waves | 81×640×64, 81×641×63 | no |
| large_regime_ranking | GEMV 1×256×2048; 3×3 s2 conv at ci=64 on a 32² image; 7×7 s2 stem at ci=3 on a 147² image | no |

## Not changed

- Phase 0 was not run. The cohort is minted by the next generation, deferred into
  `phase2-deferred.json` like every other `_perf` family.
- Nothing was sealed, simulated or measured.
- Whether gSIM and FireSim disagree in each flagged category has not been measured.
- The accelerator-affecting host stage has no cycle model yet.
- The Merlin host stage of a split program is not yet measured separately. Until it is, only paired
  deltas over identical Merlin host code are scored.
