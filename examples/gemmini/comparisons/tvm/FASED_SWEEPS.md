# FASED-memory operator and connected-workload tuning

This follow-up uses the opt-in latency-pipe Verilator variant, not the stock
magic-memory timing labels from `rtl-v1`. The simulator's verified control relink
matches the previously qualified base. The memory policy comes from
`firesim_gemmini_rocket_u250_30mhz` in the new pin registry: read/write latency
30/30 cycles and maximum outstanding reads/writes 16/16. Each accepted run must
confirm `latency_pipe` and all four settings in every reported channel.

The variant implementation is from Merlin `b44479ce5` / memory feature
`9dab33668`, with emulator SHA256
`18aeb413ccd867cbe46f1268dfc6d31401bc67c671034bffc317fe3555eb2d2f`.
Its base control SHA256 is
`a37f42b6d2f3413881faf154a94eaffaedf8d65b800e667200f523570d30d0d7`.
This binds the memory timing model; it does not establish agreement with a
FireSim run of the same ELF. The bitstream archive's `runtime.conf` still needs
byte-level verification, and the newer F2 screenshot is a separate deployment.

## Two levels of compiler choices

Per-operator tuning remains stock TVM MetaSchedule. It searches legal tile/reuse
factors, rectangles, OS/WS, traversal, hoisting, software pipelining, staging,
epilogue placement and unroll. Records are reusable only when mathematical IR
and capabilities match. Model names are experiment labels, never backend rules.
The new `--baseline-database` option replays and **remeasures** matching current
traces before search; historical costs never enter the new measurement database.
Fresh successful baseline records provide a non-regressing fallback after search.
Compilation also supports an explicit frozen fallback database through stock
`OrderedUnionDatabase`: a fresh primary match takes priority regardless of the
historical fallback cost; only primary misses consult fallback traces. Both
sources and per-function selection are recorded, and the explicit primary must
match at least one function. Fallback costs never enter the fresh measured library.
This matters for partial libraries: the initial SmolVLA fresh-only combination
improves concat placement but drops a matching historical projection trace to the
default. The retained-current combination is measured separately to preserve that
projection. Neither configuration's graph cost is inferred from operator labels.

Connected-workload tuning now has a bounded outer search. Graph preparation
choices live in stock Relax `Knob` / `Choice` / `Trace`: fusion levels 0/1/2/3,
constant weight packing, bias preload canonicalization and common-subexpression
elimination. All 32 combinations are exposed; the initial four-plan experiment
uses fusion 2/0 × packing on/off, plus a separate CSE candidate. The measured
subset is a resource budget, not a compiler rule for any model. Each alternative is
compiled and measured as one connected package, including CPU glue and
intermediate materialization. Whole-chunk cycles select the graph plan; summing
per-operator minima cannot select it. Semantic/prepared IR identities and exact
graph-choice traces are retained beside the final package.

Passes and knobs serve different purposes here. A pass implements a legal
transformation; search chooses whether that transformation is profitable for the
current graph. The CSE option invokes stock Relax `EliminateCommonSubexpr` before
legalization. It does not recognize a model or replace a particular quantizer.
The retained TinyLlama K/V test goes from six CPU calls to three, with two Gemmini
calls unchanged. `operators.o` and `constants.bin` are byte-identical, while the
host wrapper shares the producer. Warm cycles fall from 622,531 to 460,958 (25.95%)
under identical program, input, memory and timing identities. Both complete output
checks pass, as does the CSE wrong-oracle RTL control. This is a measured connected
region improvement, not full-model or SRAM-residency evidence.
Recompilation with the fresh validated operator library also passes at 460,958
warm cycles. This confirms library transfer and combined compilation; it adds no
further measured gain over the CSE candidate.

These choices do not implement a new cross-function SRAM ABI. Current SRAM reuse
is inside each scheduled function. The connected measurements expose the cost of
returned host intermediates and fusion boundaries; they must not be reported as
persistent on-chip intermediate, weight or KV caching. Such residency requires
compatible representations, alias/lifetime proofs, supported intervening
arithmetic and a completion contract before it becomes a legal compiler choice.

There is a more specific hardware/compiler boundary: the pinned local RTL's
`GemminiRocketConfig` selects a configuration enabling accumulator operand reads
and scratchpad output writes. Its `Configs.scala` digest matches the compiler
profile. Accumulator operands pass through the hardware scale/activation path
as INT8, not arbitrary full-width INT32/FP32 inputs. Current TVM contraction
intrinsics only accept INT8 A/B from scratchpad and return INT32 C in accumulator;
the normal store returns C to global memory. Those additional resident operand
forms are not exposed as searched variants. Thus the capacities are correctly
exposed, but the compiler ABI implements a smaller set of memory transitions than
the RTL. `graphs/residency-capability-audit.json` binds this finding to source
hashes. Extending the ABI and its numerical/lifetime proofs is required before
search can choose those transitions; a boolean knob alone cannot enable them.
For example, SmolVLA's `[50,1440]` FP32 concatenation occupies 288,000 bytes,
larger than either local memory. Retaining this transition needs compatible
row/block schedules across concatenation and rowmax/scale/quantization. The new
native concatenation record uses small accumulator DMA tiles, then materializes
the tensor in host memory; its isolated timing does not remove that handoff.

The framework is still bounded. Fusion level applies to the whole prepared
module, not independently to each edge. Inner search uses two-level perfect-tile
sampling with a 64-group sampling guard, divisor-based rectangles/tails, and
legal pipelining alternatives rather than arbitrary stage counts. CPU placement
is searchable for eligible functions; CPU-only extracted tasks are reported but
excluded. Four proposals do not establish convergence or a trained cross-model
cost predictor. Broader per-edge partitions, legality-derived sampling bounds,
CPU-only task tuning and resident-region representations are general compiler
extensions for subsequent work, not per-model exceptions to add here.

The CPU-only exclusion is substantial: the full SmolVLA action/time program has
four eligible tasks out of eleven unique tasks (three contractions and concat).
The compact transition has five excluded CPU tasks out of seven. Rowwise
abs/max reductions, maximum/divide scale computation and divide/round/clip/cast
quantization remain untuned. A general single-hart CPU MetaSchedule path for those
functions, together with graph fusion, is a higher-priority extension than
assuming GEMM tile search alone can close the complete-model gap.

A concrete general next candidate is concat followed by rowwise quantization:
compare materializing FP32 concat against traversing its input regions directly
for the shared row maximum/scale and writing the final INT8 codes. The latter
could remove the FP32 temporary, but must preserve the shared reduction, rounding
and all other live consumers. This should be an IR-pattern legality check with
searched placement, applicable to any matching graph. It is not implemented or
credited in the current timings.

The exact-IR audit also found why a tuned family name is insufficient. The
standalone ResNet reduction clips to `[-127,127]`; its connected first stage
fuses ReLU and clips to `[0,127]`. The mathematical workloads differ, so stock
structural matching correctly refuses the old record for every tested pair plan.
Both pair contractions then use defaults with 512 bytes of scratchpad and
1,024 bytes of accumulator each. This is a schedule coverage gap, not a measured
proof that it explains the screenshot's entire slowdown. The TinyLlama MLP has
similar misses; a separate inner campaign now targets its exact up-projection
task. Standalone KV/output winners cannot be assigned to these functions by name.

## Workload scope and resource limits

Three workloads run concurrently, with one runner and builder per operator
campaign and one separate serial graph sweep per workload. A separate TinyLlama
MLP operator worker and initial memory A/B controls also overlap, bounded to eight
simulators on the 20-logical-CPU, 15-GiB host. Thread counts are fixed to one;
observed available RAM remained about 10 GiB.

| Workload | Reusable operator study | Initial connected pilot |
| --- | --- | --- |
| TinyLlama | Decode KV, M1 K2048 N256 | Decode MLP: distinct normalized/residual inputs, gate/up 2048→5632, SiLU/product, down 5632→2048, residual |
| SmolVLA | Action output, M50 K720 N32 | Action projection 32→720, time concat, 1440→720, SiLU, 720→720 |
| ResNet50 | Layer4 reduction, full 7×7, K2048 N512 | That 1×1 convolution followed by 512-channel 3×3 convolution with full spatial extent and integer requant/ReLU |

Full reduction/channel dimensions, action tails and spatial padding are retained.
A separate M32 TinyLlama prefill KV operator is measured independently; a prefill
MLP fixture is prepared but not timed. Decode results never substitute for either.
Late ResNet
chunks do not establish early-layer performance. SmolVLA's action path does not
establish the one-camera prefix/vision performance.

The initial full connected pilots each consumed over 35 minutes without a
completed timing and were explicitly stopped for resource budget. Their negative
signal return codes, partial logs and `budget-stop.json` remain in `connected/`;
they are not cycle labels or numerical passes. Queued alternatives were canceled.
The full-size ResNet standalone operator also reached its 2,400-second wall cap;
its campaign was stopped before spending more trials on a graph-mismatched task.

The next connected sweeps use these smaller, documented regions:

| Workload | Bounded region | Relationship check |
| --- | --- | --- |
| TinyLlama decode | Two full M1 K2048 N256 K/V projections, separate row quantization expressions, FP32 dequantization and head-layout views | All four KV heads retained; before RoPE/cache update. Architectural surrogate, not an exact EL4 capture. CSE is an optional compiler choice that shares the repeated producer. |
| SmolVLA | Full M50 action projection 32→720, time concatenation, next quantization; codes `[50,1440]` and scales `[50,1]` are live outputs | Exact parent binding slice `[0,21)`, original constants and boundary inputs; all outputs match independent parent checkpoints exactly. |
| SmolVLA shorter transition | Full M50 input quantization, 32→720 projection and time concatenation; FP32 `[50,1440]` output | Exact parent prefix `[0,13)`, all 13 bindings structurally equal and output bit-exact to the parent's independent checkpoint. Ends before the large downstream quantizer; it cannot establish that consumer's performance. |
| ResNet50 | Center 2×2 output region of the pair, requiring a 4×4 first-convolution input/halo; all 2048→512→512 channels and both requant/ReLU chains | Original constants retained; all 24 intermediate values match the corresponding full-graph crops exactly. The second convolution is valid over the supplied halo. |

The ResNet region changes spatial row geometry. Its traces and timings cannot be
assigned to the full 7×7 graph or multiplied into a model latency estimate. These
regions keep real reductions, channel dimensions and producer/consumer work while
making a local sweep practical. `run-plan-amendment.json` records the change.

The inputs/weights are frozen synthetic shape/traffic surrogates, not the
selected EL4 numerical release. The independent reference evaluates those exact
graphs without executing compiler candidates, checks integer accumulation bounds
and explicit FP32 boundaries, and records every binding. In particular the older
surrogate uses Relax away-tie `round`; actual PyTorch `round` intake uses
ties-even `nearbyint`. Preserve those distinct recipe identities. These fixtures
do not fix or diagnose the historical complete TinyLlama mismatch.

Per-operator references use unscheduled mathematical TIR executed by host LLVM;
they are independent of the candidate schedule, not independent model-numerics
evidence. Connected fixtures use the separate NumPy semantic evaluator above.

Every new timing uses one unconditional warmup and one measured invocation of
the same buffers. This replaces the historical duration-dependent cold/warm
policy. Spike checks every output, input hashes and guards and must detect a
wrong reference; RTL checks every output using the same ELF. Startup, oracle
checking and HTIF are outside the measured interval. The campaign does not relax
connected `rtol=1e-5`, `atol=2e-6`, exact integer or finite-output requirements.
The existing per-operator oracle uses the stricter `atol=1e-6`.

Initial operator searches use four trials and a two-trial batch, plus retained
current/default/CPU controls. This is a bounded initial sweep, not convergence
evidence. Original connected pilots requested 60-million-cycle / 3,600-second wall
caps; compact sweeps use 24-million-cycle / 2,400-second bounds. Timeouts, numeric
failures and unsupported plans remain failures, never timing labels. Output
validation is outside the inference interval but inside the simulator budget;
seeing a `DSE_CYCLES` line without completed output checks is not an admitted cost.

The runs exposed two generic measurement issues. Prepared graph constants must
bind operator inputs: generating a random value for a lifted integer divisor
caused a host-reference divide by zero. `operator_inputs.py` now derives constant
and alias bindings from IR; a supplied boundary fixture also propagates through
unscheduled mathematical host TIR. The ResNet compact binding values were checked
against independent NumPy checkpoints. Separately, large floating output checks
can exceed the old additive validation allowance after inference finishes.
`--validation-overhead-cycles` exposes that allowance without adding it to the
timing objective. Exact failed candidates can be replayed in a fresh directory;
only a complete new pass supplies a database cost.

The final checker revision rejects nonfinite actual/reference values explicitly
and skips tolerance arithmetic for exactly equal finite FP32 words. This fixes an
overflow corner at finite extrema and speeds untimed validation of exact-copy
outputs. It changes no inference arithmetic or tolerance. Native-host tests cover
31 floating/integer edge cases. Before applying it, all active guests were built
and frozen: 110 ELF hashes stayed unchanged. The older study references contain
196,192 FP32 values with maximum absolute value 619.6254 and finite tolerance
bounds, so their successful labels cannot hit the repaired overflow corner.
`memory/source-v2-final` and `memory/source-v3-finite-checker` preserve both source
revisions. Later combined builds use the new checker and fresh execution receipts;
no checking-time reduction is reported as a model speedup.

## Confirmed measurements

All counts below are target cycles with confirmed 30/30/16/16 memory and one
warmup followed by one measured invocation. A single sample per candidate and a
small search budget do not establish convergence or statistical significance.
Repeated same-buffer chunk calls also do not reproduce the cache eviction caused
by an entire model's intervening weights and state. First-call costs are retained
alongside warm costs. Select the final stage/session objective and cache conditions
explicitly; the latency-pipe model prices misses that occur, but does not make a
small warm working set equivalent to full-model traffic.

| TinyLlama connected K/V graph choice | Warm cycles |
| --- | ---: |
| Fusion 2, packed weights, current matching operator trace | 622,531 |
| Fusion 2, unpacked weights | 1,069,944 |
| Fusion 0, packed weights | 1,244,731 |
| Fusion 0, unpacked weights | 1,242,833 |
| Fusion 2, packed weights, optional stock CSE | **460,958** |
| Same CSE choice, fresh validated operator library | **460,958** |

These are the same frozen two-projection program and boundary fixture. The CSE
gain is 25.95% relative to the current fused/packed control. The last row verifies
combined compilation with the fresh library; it is not an additional gain.

| Operator | Fresh current control | Selected warm cycles | Interpretation |
| --- | ---: | ---: | --- |
| TinyLlama decode K/V, M1 K2048 N256 | 134,254 | 134,254 | Current trace wins; three passing new proposals are slower. |
| TinyLlama prefill K/V, M32 K2048 N256 | 602,545 | 601,930 | 0.10% lower in one measurement; treat as a near tie. |
| SmolVLA action output, M50 K720 N32 | 67,811 | 63,322 | 6.62% lower warm cycles; first-call cost rises from 79,495 to 81,267, so this is a warm-objective choice. |
| SmolVLA full M50 time concatenation | No completed current/CPU control | 68,796 | Fresh successful replay after increasing the total validation budget; no speedup ratio is assigned. |
| ResNet compact-region spatial convolution | Default/CPU controls capped | 1,269,090 | Both spatial proposals pass (other: 1,509,726); no baseline speedup ratio. |

The exact TinyLlama MLP up-projection default and both sampled proposals reached
the 10-million-cycle cap, including a hoisted/pipelined candidate. The CPU control
reached its separate 500,000-cycle cap. No timing record or MLP winner was admitted;
`operators/tinyllama-decode-mlp-up/bounded-result.json` retains all four outcomes.

ResNet's two reduction proposals were one CPU and one accelerator schedule; both
capped, so no reduction winner is admitted. The shared 16-record library matches
the compact spatial convolution only. Its connected package retains default
reduction (512 B scratchpad / 1,024 B accumulator) and selects 49,664 B scratchpad /
8,192 B accumulator for the spatial function. The 8,192-byte intermediate remains
host-materialized. These allocations demonstrate usable memory exposure and
partial schedule coverage; they are not measured bandwidth or whole-region gains.
The retained LLVM also shows that an isolated operator and the connected package
can have the same sketch with different argument alias annotations. Their Spike
instruction counts differ in a way that operator addition cannot predict.
`graphs/resnet-isolated-connected-abi.json` records the actual declarations and
file hashes; it does not attribute the difference to alias metadata alone. Final
LLVM/ELF retention and connected remeasurement are therefore necessary even when
a schedule name matches.

Failed/capped controls remain in each campaign. The whole compact SmolVLA/ResNet
default controls exhausted their 2,400-second wall limits; remaining default-only
alternatives were canceled. Both larger SmolVLA fresh-library combinations also
exhausted 2,400 seconds without an admitted timing. The final unfinished connected
runs were closed at the user's wrap-up request, with explicit stop evidence.
These caps/stops are not numerical failures or inferred cycle counts. A printed
timing without completed validation remains an unadmitted diagnostic.
The shorter retained SmolVLA prefix printed 1,523,227 first-call and 1,469,345 warm
cycles before the wrap-up stop, but never completed its output check. Those
numbers remain in its raw log and are explicitly excluded from selected results.

The broader regression suite passed **207 tests** across Gemmini applications
and the sketch, TensorIR, search and baremetal backends. Documentation checks pass;
the structure check retains nine pre-existing module-size findings. Checker and
ordered-database targeted tests overlap the broader suite and are not added to
its count. `validation-summary.json` binds the retained logs.

## Reproduction and FireSim handoff

The study artifact root is:

```text
out/artifacts/perf-studies/gemmini-fased-workload-sweeps/v1/
  perf-studies_gemmini-fased-workload-sweeps_v1_20261024T073102Z_36899e8/
```

`local-env.sh` binds the actual TVM library, toolchain, source checkouts and base,
variant and pin-registry receipts. `run-plan.json` records budgets and scope.
`memory/` contains the exact memory binding and runner control. `fixtures/`
contains independent references; `operators/` contains stock databases and every
candidate; `connected/` contains graph traces, compiled packages and whole-chunk
measurements. All old magic-memory evidence remains separate.

The final FireSim parameter record (`firesim-parameters.json` in the local study product `perf-studies_gemmini-fased-workload-sweeps_v1_20261024T073102Z_36899e8`, not in Git; its validated operator records ship as TVM preset `fased-v1`)
contains complete stock traces and sampled decisions, graph choices, per-function
primary/fallback selection, first/warm cycles, actual renderer and ELF identities,
all failed/stopped campaigns, and the proposed complete-session protocol. It is a
preparation record, not a launch qualification. `source-final/` freezes source
deltas; the separate v2/v3 snapshots preserve measurement-era attribution.

Next priorities are actual selected-export boundary fixtures and stage profiling,
generic CPU-only task tuning, finer graph partition/fusion choices, and a proved
row/block residency ABI before exposing retention as a knob. Then remeasure full
stateful sessions and correlate the same ELF with the actual FireSim deployment.
This study neither establishes the screenshot's approximately 100× gap nor fixes
the historical full-model TinyLlama numerical mismatch.

Use the companion TVM `WORKLOAD_TUNING.md` for commands. A new FireSim compilation
must bind the selected graph choices and exact-matching operator records to its
actual exported programs. Inspect unmatched functions and CPU fallbacks again;
small-chunk coverage does not imply complete-model coverage. Do not blindly apply
a chunk's global fusion setting to every stage. Recheck the full stateful session,
original tolerance, all required outputs, final ELF memory/instructions, target
image/driver and complete timing boundary before admitting a FireSim result.
