# General TVM optimization and DSE scope

The 70.64% host / 29.36% Gemmini numbers are **elapsed mcycle intervals by
function**, including memory and other stalls. They do not establish a
compute-bound CPU, a memory-bound system or Gemmini array utilization. The
exact ELF audit is `out/artifacts/audits/gemmini-resnet-elf-attribution/v1/latest/`.
It matches all 128 profiled calls: 74 contain host-only code and 54 reach the
Gemmini completion fence. Idle-device phases are an inference under that
platform fence contract, not measured utilization. A CPU stalled on memory
can leave Gemmini idle; both can be true.

Use ResNet's profile to select expensive computation to study. Compiler rules
must derive applicability from operations, shapes, layouts, reuse, arithmetic
contracts and target capabilities. Model names and captured node IDs may select
experiment inputs; they must never select production schedules or rewrites.
A convolution optimization can be general without speeding up a transformer;
validate the operation family beyond the motivating model before promoting it.

## Existing decisions and proposed additions

| Decision | Actual tradeoff | Availability for the next study | General scope |
| --- | --- | --- | --- |
| GEMM M/N/K reuse groups, traversal and OS/WS | DMA/command overhead and reuse versus accumulator/scratchpad pressure | Existing traced MetaSchedule decisions; most ResNet shapes remain untuned | Convolutions lowered to contractions, linear layers, attention matmuls |
| Cache hoisting and software pipeline | Retained data/overlap versus capacity, buffering and extra staging | Existing traced MetaSchedule decisions, subject to legality | Supported contraction shapes across models |
| Graph fusion boundary | Intermediate traffic and dispatch versus recomputation, live storage and a larger scheduling problem | Existing TVM FuseOps choice; now exposed as `--graph-fusion-level` in the FireSim tuning driver and saved as a Relax Trace | Quantize/layout/pointwise/epilogue chains in any graph |
| Host producer placement/materialization | Produce shared data once versus recompute near consumers to avoid storing intermediates | **Proposed integration** of stock AutoInline/RandomComputeLocation and producer scheduling; excluded from the current contraction-only FireSim search | Shared activation conversion, padding, layout transforms and pointwise chains |
| Host loop tiling/order and bounded unrolling | Contiguous accesses/cache locality versus indexing and instruction-cache pressure | **Proposed integration** of stock CPU schedule rules for the host tasks, with explicit resource/compile budgets | CPU loops accompanying any accelerator graph |
| Input-packing granularity | Full im2col/gather materialization versus bounded spatial panels or tile-local production | **Additional compiler work**, then expose legal producer placements/tile sizes to TVM; not an implemented flag | Convolution/gather-to-GEMM patterns, independently of the model |
| Epilogue materialization/readout placement | Intermediate INT32/FP32 traffic versus fusing a proven bias/ReLU/scale/requant path | Existing backend support for recognized integer patterns; the retained ResNet capture's FP32-bias chain does **not** currently qualify | Eligible integer contraction epilogues |

The host is the existing single-hart RV64GC target. Do not introduce vector or
intra-guest parallel-loop choices without matching hardware support. Multiple
independent FireSim jobs can still run concurrently across physical slots.

The first new knob to exercise is the **existing general fusion level**. The
prepared study audits levels 0 and 2 with packing enabled, groups equivalent
prepared IR before allocating paid jobs, and preserves replayable choices.
Fusion is fixed within each inner EvolutionarySearch/XGB campaign. This is not
a joint ML optimizer over graph structure and schedules; the graph controls
are explicitly separate from the ML-guided operator DSE.

Before investing in additional host tiling/unrolling, use the ELF and a diagnostic
profile with retired-instruction, available CPU memory-hierarchy and Gemmini
activity/wait counters to identify the limiting cost. LLVM already optimizes
these functions. Host producer scheduling and bounded input packing remain
hypotheses that can affect computation and memory traffic. Neither is enabled
merely by increasing contraction trials. Keep large fused host functions
bounded during design-space generation; the earlier all-task attempt already
exposed excessive preparation time there.

## Fixed transformations and numerical obligations

Do not spend DSE trials on removing dead code, folding immutable constants,
identity operations, an inapplicable CSE/bias switch, or multiple choices that
produce identical code. Constant packing stays enabled in the first study.
Keep legal automatic simplifications in normal passes. Retain choices only
where there is a measured tradeoff; do not hardcode the fastest tuple from a
model's experiment into production.

Calibration scales, grouping, rounding, saturation, bias domain and quality
thresholds are fixed by [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md).
They are not performance knobs. The retained ResNet source has all 54 epilogue
scale arrays constant within groups of 16, but its graph casts INT32 GEMM
results to FP32 and adds FP32 bias. The round-two source preparation moves all
54 additions into INT32 with explicit bounds below 2^24, proving equivalence
to the old FP32 ordering over admitted INT8 inputs. It preserves weights/scales
and freezes a new identity. Compiled correctness, placement and model accuracy
remain separate gates; no old qualification is inherited.

## Can ResNet improve through more tuning?

Yes: 20 of 21 distinct contractions remain without measured winners, including
the large spatial convolutions. This is much more scope than the classifier.
The profile also assigns 70.64% of operator cycles to host-only functions.
Their memory stalls and instruction costs need diagnosis before selecting
an optimization; that share is not proof of inefficient CPU compute.

For scale, if the measured 29.36% Gemmini-function portion halved while all
other work stayed fixed, model time would fall about **14.7%** (about **1.17×**
speedup). Reducing the 70.64% CPU-function portion by a quarter would save about
**17.7%** (about **1.21×**). These are Amdahl illustrations, not predicted gains.
The current 0.0495% classifier-only result does not determine either outcome.

## Next FireSim preparation and promotion

The current product is `out/artifacts/handoff/gemmini-firesim-dse-round2/v1/latest/`.
It contains 18 full-shape trained-weight regions, independent boundary fixtures,
host checks and 21 structural workload keys assigned to 10 tuning owners.
See [FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md) and its README for
actual readiness; source correctness is not numerical-policy admission or
FireSim qualification. `prepare_exported_chunks.py` accepts arbitrary explicit
source regions and preserves constants and every live edge; it chooses no
optimization or schedule. Package-call indices are not source-node indices.

Start with stem and one full bottleneck from each stage: 48 proposals total.
The continuation adds 20 proposals for the remaining distinct workload owners;
24 are reserved for TinyLlama once its current-policy exports are available.
Use one shared four-hour deadline, batches of four and a cumulative proposal
ledger. Reserve wall time for connected/full-model confirmation; stop tuning
early when the remaining deadline cannot accommodate confirmation.

Before paid DSE, close the numerical-admission gap and demonstrate the actual
[active-cycle cutoff](ADAPTER_CONTRACT.md) on the installed runner. Then keep
managers alive, batch builds/submissions and use every actually available
independent slot under shared physical locks. DDR3 remains in the bitstream.
Score complete measured active windows; preserve input/constant identities and
independent all-output checks. Full-model quality is a separate recorded gate.

Promoting a scheduling/lowering change requires independent shapes, tails,
shared producers/residual live-outs and refusal tests. Include at least another
convolution family plus M=1 and M>1 linear shapes for changes that also affect
contractions. Reuse existing TVM frontend/TIR/backend tests. Measure other models
before claiming that a ResNet gain transfers to them. No benchmark-name branch,
handwritten winning schedule, TVM core monkeypatch or unreported CPU fallback
is an acceptable substitute for general rules and stock search.
