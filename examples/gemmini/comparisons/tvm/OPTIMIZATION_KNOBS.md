# Current optimization choices

[ARCHITECTURE.md](ARCHITECTURE.md) owns the implementation boundary;
[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md) owns execution budgets and readiness.

## Campaign re-audit (2026-10-27)

This anchor is retained for existing links. The current path uses stock
**ReplayTrace without a learned cost model**. The earlier evolutionary/XGB,
50-sample warmup and three-model plans are superseded. Their source-backed
analysis remains in [the previous audit](https://github.com/ucb-bar/merlin/blob/9530ee83d/examples/gemmini/comparisons/tvm/OPTIMIZATION_KNOBS.md).
Historical results retain their original compiler, oracle and hardware identities.

| Layer | Active choice | Fixed or excluded |
| --- | --- | --- |
| Relax | Start fusion=2/CSE off; stock fusion/CSE alternatives may share the same trial budget | Frozen inputs/scales/reference semantics; no automatic bias relocation, prepacking or scale/readout rewrite |
| TensorIR | Exact intrinsic rectangles, M/N/K reuse groups, legal OS/WS, traversal, hoisting/pipelining, staging and tails | Current compact search keeps epilogues at root; unroll options 0/16 |
| Default | Largest exact legal tile, bounded traffic-ranked reuse groups, actual allocation check | Heuristic baseline, not a learned or calibrated latency model |
| Search | Stock ReplayTrace and round-robin task scheduling, up to 25 total proposals per owner chunk/case | No XGB, feature training, learned ranking or guided phase |
| Measurement | Completed correct warm target cycles, converted to seconds at the TVM Runner boundary | Median wall watchdog; timeout bounds are not cycle measurements |
| Database | JSONDatabase and MetaScheduleApplyDatabase; exact structural reuse; retain measured default | No historical costs imported onto different hardware/source profiles |
| Physical memory | Deterministic aligned SRAM allocation, capacity/alias checks and required DMA synchronization | No arbitrary address search, general cross-function residency or arbitrary overlap optimizer |
| Runtime/export | Primitive command ABI, configuration shadowing, compatible WS reuse; static graph wrapper | Fixed custom integration, not stock TVM runtime or tuned graph passes |

The active tuner extracts contractions. Other supported families retain fixed
schedules or host execution; their cost is included in complete-call timings.
T01/T02 must include all heads, T04/T06 all outputs, and ResNet E2E the connected
parent. Isolated task minima do not establish those timings. See
[SPIKE_HANDOFF_EVIDENCE.md](SPIKE_HANDOFF_EVIDENCE.md) for current functional checks.

## Retained historical code

The TVM compatibility backend is isolated in `gemmini_legacy.py`; it is loaded
only for explicit old API/route requests. Its tests and old nine-template search
remain for reproduction, outside the active campaign. Shared `gemmini_schedule.py`
helpers and `gemmini_bias.py` are still used by current TensorIR lowering: the
latter implements explicit accumulator initialization, not automatic bias folding.
Deleting those modules as if they were unused would break the active backend.
