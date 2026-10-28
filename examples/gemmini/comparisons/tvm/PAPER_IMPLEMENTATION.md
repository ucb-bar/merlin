# TVM implementation: paper text and evidence boundaries

Current source: TVM `2c0857ab88b18d4a090f2dd6e401e06266af1877`; the
published Spike baseline is `bb4ad9f8d`, preceding the legacy-module isolation.

The text below describes the implementation and configured search, not a claim
that the FireSim experiment has finished. Citation keys are supplied in
[PAPER_REFERENCES.bib](PAPER_REFERENCES.bib); a LaTeX version is in
[PAPER_IMPLEMENTATION.tex](PAPER_IMPLEMENTATION.tex).

## Suggested methods text

We implement a Gemmini backend in a fork of Apache TVM v0.19.0, following TVM's
separation of graph and operator optimization [Chen et al., OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/chen).
The backend uses TensorIR and TVM's TensorIntrin interface: each intrinsic pairs
a mathematical computation description with a target implementation, and
schedule-driven tensorization replaces matching computation blocks
[Feng et al., ASPLOS 2023](https://arxiv.org/html/2207.04296v2).
The same pinned Gemmini software specification supplied to Merlin defines the
backend's capability and numerical requirements. Intrinsic recognition and
finite tests establish implementation coverage; they do not discharge every
specification proof obligation or qualify a deployed bitstream.

Graph preparation uses stock Relax legalization, constant folding, binding
canonicalization, operator/TIR fusion, reshape rewriting and dead-code
elimination. Common-subexpression elimination is an optional stock graph choice;
the initial configuration uses fusion level 2 with CSE disabled. Target schedule
rules compose ordinary TensorIR transformations for tiling, loop ordering,
blockization, cache placement and layout. Their traced choices expose intrinsic
tile dimensions, operand-reuse groups, dataflow, staging and legal
hoisting/pipelining alternatives. A capacity-checked, shape-derived schedule
provides the initial baseline. Necessary target-specific integration supplies
workload annotation, intrinsic implementations, final schedule placement, SRAM
allocation and DMA synchronization, together with a primitive command runtime
and static RV64 packaging. Explicit accumulator initialization may tensorize;
automatic bias relocation, constant-weight prepacking and scale/readout
normalization passes are excluded.

We use MetaSchedule's search-space and measurement interfaces
[Shao et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e894eafae43e68b4c8dfdacf742bcbf3-Abstract-Conference.html),
with stock ReplayTrace and no learned cost model. The configured allowance is
at most 25 schedule proposals per kernel case or ResNet owner chunk, shared
across its internal tasks and graph alternatives. Completed correct warm-cycle
measurements select schedules; cycles are converted to seconds at the TVM
Runner boundary. Measurements do not guide ReplayTrace sampling. Three baseline
timings calibrate a median-based wall watchdog; failed or timed-out attempts do
not supply completed cycle labels. Matching traces are applied through
JSONDatabase and MetaScheduleApplyDatabase, retaining the measured default for
misses or unsuccessful search. Final comparisons require rebuilding and timing
the complete kernel or connected model, including executed host work, packing,
DMA and completion; sums of isolated operator minima are not E2E results.

## Source support for implementation-specific statements

The papers establish TVM/TensorIR/MetaSchedule methodology, not the details of
this fork. The original TVM paper predates Relax. The MetaSchedule paper's
learning-driven evaluation is not our selected ReplayTrace policy.

| Claim | Implementation evidence |
| --- | --- |
| Stock graph pipeline and optional CSE | [prepare_gemmini_tensorir_workloads](https://github.com/grammy117/tvm/blob/bb4ad9f8d/python/tvm/relax/backend/contrib/gemmini_tensorir.py), used by extraction and compilation |
| TensorIntrin description/implementation contract | [TVM v0.19.0 TensorIntrin](https://github.com/apache/tvm/blob/v0.19.0/python/tvm/tir/function.py) and fork `gemmini_intrin.py`, `gemmini_epilogue.py`, `gemmini_bias.py` |
| Traced target schedules and default heuristic | [gemmini_search.py](https://github.com/grammy117/tvm/blob/bb4ad9f8d/python/tvm/relax/backend/contrib/gemmini_search.py), `gemmini_meta_schedule.py`, `gemmini_sketch.py` |
| Remaining custom placement and memory lowering | [ARCHITECTURE.md](ARCHITECTURE.md), fork `ScheduleGemminiTensorIR`, `_inline_cpu_producers`, `gemmini_memory.py` |
| ReplayTrace is sampling, not learned ranking | [stock ReplayTrace source](https://github.com/apache/tvm/blob/v0.19.0/src/meta_schedule/search_strategy/replay_trace.cc), fork `gemmini_meta_schedule.py`; [OPTIMIZATION_KNOBS.md](OPTIMIZATION_KNOBS.md) |
| Actual cycle measurement and unit conversion | [tune_firesim_chunks.py](tune_firesim_chunks.py), [firesim_runner_seconds.py](firesim_runner_seconds.py), [firesim_timeout.py](firesim_timeout.py) |
| Current baseline functional evidence | [SPIKE_HANDOFF_EVIDENCE.md](SPIKE_HANDOFF_EVIDENCE.md): 74 operator fixtures, 16 kernels, 18 chunks and connected ResNet parent; diagnostic Spike, not FireSim timing |
| Exact operation authority | [8ec99308 software-spec.yaml](https://github.com/ucb-bar/merlin/blob/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/target/software-spec.yaml); [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md) records implemented candidates and missing obligations |

## Experimental settings to report separately

- ResNet: four aggregate tuning slot-hours and 30 minutes maximum per final E2E
  run. The 16-kernel TVM sweep has a 24-slot-hour ceiling. Caps can leave tasks
  undertuned; publish actual trials, failures, timeouts, occupancy and coverage.
- Fresh extraction: 18 ResNet chunks, 21 distinct contractions and ten owner
  campaigns. The 16 kernel cases have 17 distinct contractions and 16 owners.
  Exact structural equality shares records. The initial plan reserves 20 of 25
  trials for the base graph and five for alternatives, not five learned trials.
- Three baseline jobs per task and repeat/confirmation runs are extra executions
  charged to the same time budget. “25 proposals” does not mean 25 valid unique
  measurements, 25 per head, an ML warmup, or convergence.
- The active tuner extracts contractions. Other families and CPU work are not
  independently optimized by this campaign. Fixed runtime configuration caching
  and legal WS tile reuse must be disclosed as part of the implementation.
- Full TinyLlama/SmolVLA are backlogged; their isolated kernels are not full-model
  results. The complete-call harness uses warm-repeat, not historical 2 MiB
  eviction. Compare systems only under matching explicitly reported profiles.

## Do not claim shared-hardware results before the receipts exist

Replace the preliminary sentence “The same FireSim bitstream, workloads, and
result tolerances were used as the Merlin run” with this protocol statement:

> The comparison protocol requires matching workload/input identities, numerical
> checking, target hardware and timing profiles across TVM and Merlin. We report
> the bitstream and configuration identities with the measurements.

Only after both actual runs are bound to those identities may the paper state
that the same bitstream and tolerances **were used**. The selected model protocol
uses the full Gemmini/Rocket U250; historical F2 kernel measurements are not an
interchangeable hardware baseline. ResNet's exact selected-protocol binding,
spec-pinned oracle provenance and remote qualification remain separate from
passing diagnostic Spike checks. Do not describe this fork as unmodified TVM,
all 51 fixtures as device ops, every DMA decision as tuned, or a TensorIntrin as
necessarily one hardware instruction.
