---
title: Evaluate default compiler passes
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-20
related: [shared_pass_phase2, shared_source_lowering, portable_generated_target_admission, ../reference/shared_compiler_passes]
code_refs:
  - examples/gemmini/phase2/shared_pass_benchmarks.py
  - merlin/tests/gemmini/test_shared_pass_benchmark_inputs.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/llvmlower/device_preparation.py
  - src/merlin/llvmlower/device_build.py
  - packages/merlin-timing/include/merlin_timing/reserved_submission.hpp
---

# Evaluate default compiler passes

Measure the deterministic shared-pass policy before searching for a better
policy in Phase 2. A pass contributes only when the ordinary compiler consumes
its changed program and the emitted implementation satisfies its contracts.
An absent integration is recorded explicitly; selecting every option does not
establish a complete optimized compiler.

## Freeze the comparison and respect IR levels

Keep the original Phase 1 compiler immutable. Make baseline and candidate builds
with identical selected library bytes, hardware, software/numerical contracts,
runtime, toolchain, inputs, output and state checks and execution boundaries. Keep
original numerical failures visible. New library grants are a separately
recorded experiment and cannot be attributed to the original Phase 1 treatment.

Use the existing pipelines rather than a new universal pass runner:

| Level | Ordering and constraints |
| --- | --- |
| Typed source | Semantic/numerical admission and supported source rewrites before the relevant information disappears. Ordinary host lowering orders optional recomputation/graph work, rows, padding/gather, destinations, rounding and producer packing. Some selections are mutually incompatible and must retain their documented refusal. |
| Placement and dense scheduling | Discover eligible work, establish placement for actual selected shapes, then choose blocking, operand retention and reduction panels under current assigned storage. Source rewrites before device outlining require fresh placement. |
| Typed device schedule | Existing semantic admission, selected configuration cleanup, qualified fusion/prefetch candidates, allocation/staging and supported spills. Post-storage deletion follows its existing contract. |
| Completion and emission | Reconstruct lifetimes/completion after rewrites, emit through the selected adapter, then check actual decoded effects and final instruction/address/branch limits. |

`PassSpec` and `PipelineSession` enforce prerequisites, representation, freshness
and invalidation. Spills and allocation fallback require bounded repair and
fresh analyses. Instruction changes invalidate emission and final-size checks.
Some current stages have only one legal order; an ordering parameter does not
create new supported alternatives.

Record cumulative arms and relevant single-pass controls. For every pass and
workload retain one of: applied, no matching opportunity, unavailable contract,
legal fallback, or refused. Applied means consumed transformation with checked
realization, not a selected plan. A pure GEMM has no producer-packing, epilogue or
multi-output fusion opportunity merely because it is large.

## Connect transformations to actual device compilation

The original host may select `DeviceRouting.llvm_transform_chain`. Its exact
selection is frozen before source preparation and reaches the existing checked
LLVM transformation consumer before device object compilation. The default
`None` preserves the original route. Catalog objects and routes with no emitted
device LLVM cannot silently consume a selected chain.

This is a transport and custody interface. It does not interpret opaque LLVM,
create a target-IR consumer, supply numerical/completion laws or authorize a
device transformation. A scheduling rewrite still needs a qualified projection
of the actual operations/effects, an implementation, an independent realization
checker and applicable final-image checks. The public V26 compiler does not
declare the shared target-schedule consumer. Its runtime loops cannot simply
be handed to the finite straight-line `compile_schedule` family.

For bounded contractions in looped LLVM, the original host can instead select
the [staged device LLVM route](staged_device_llvm.md). It uses an exhaustive
provider-owned projector and emitter, applies shared row placement, tile order
and RHS reuse, then checks the actual LLVM consumed by object compilation.
Its finite checking bounds and completion envelope remain explicit; it does
not make other scheduling or fusion families applicable automatically.

## Prepare explicit public workload inputs

The public source generator contains six GEMMs in **M, K, N** order:
512x512x512; 12544x256x64; 12544x64x256; 3136x512x128;
3136x128x512; 784x1024x256. It emits complete ordered, zero-seeded signed-i8
contractions with raw i32 outputs, without tuned schedules or reference
compiler implementations.

Generate inputs and optionally preflight the selected public compiler:

```sh
python -B examples/gemmini/phase2/shared_pass_benchmarks.py \
  --output-dir "$TMPDIR/pass-benchmark-inputs" \
  --compiler "$PHASE1_COMPILER" \
  --compiler-sha256 "$PHASE1_COMPILER_TREE_SHA256" \
  --python "$SELECTED_COMPILER_PYTHON" \
  --mlir-install "$MERLIN_MLIR_INSTALL"
```

All four compiler/toolchain arguments must be supplied together. Omitting all
four only materializes source and geometry. Destinations must be new. The
compiler tree is checked against the independently selected identity before
and after compilation. The public manifest selects emission commands; emitted
LLVM-dialect MLIR is additionally checked by the selected `mlir-opt`.

The report is **compile-only**: no target execution, numerical admission,
device-pass application or speedup is inferred. Reported wall time covers each
row's emission, syntax verification and subsequent identity checks; it excludes
initial pinning and source generation. Compiler peak memory remains unknown.
Proposed bounded operand values require fresh capture, independent complete
references and numerical/linked-image admission before execution. They do not
replace the original policy or qualify arbitrary full-range i8 inputs.

These shapes are public development inputs, not held-out evaluation. The
Autocomp and Exo papers are comparison context:
[Autocomp](https://arxiv.org/abs/2505.18574),
[Exo 2](https://arxiv.org/abs/2411.07211).
Shape agreement alone does not establish matching input distributions, bias,
output conversion, layout, hardware, memory system or timing boundaries. Match
those before comparing speedups. Convolution labels are retained as unresolved
until batch/spatial/channel/kernel geometry, stride, padding, dilation, layout
and epilogue are explicit; the generator does not guess them.

## Keep reservation credits separate from service queues

The timing package already models service occupancy, waiting-to-start queues
and declared dependencies. The optional advisory consumer additionally composes
`ReservationWindow` and `Engine`:

```cpp
ReservationWindow credits(selected_entry_count);
auto result = submit_reserved(engine, credits, event,
                              CreditReleaseStage::ready);
```

The original adapter explicitly selects release at issue, start or readiness;
the example's `ready` is not a universal hardware law. Live dispatch entries
remain occupied until that declared stage, independently of service occupancy
and downstream queue slots. Missing completion, occupancy or credit release
remains unknown and cannot be reported as a zero-cost event.

This consumer is a conservative submitted-order advisory model, not a
ready-select out-of-order reorder-buffer simulator. The new header is outside
the frozen timing export inventories and supplies no default target consumer,
author grant or fitted price. Extracted structural capacities constrain it;
qualified event semantics and independently calibrated prices are separate
inputs. Its synthetic tests establish model behavior, not Gemmini timing.

## Measure only qualified products

Run functional execution and independent complete output and state checks first.
Inspect actual emitted instructions and effects for each applied pass. Retain
compilation time/peak memory, static code size, executed commands/requested bytes,
host/accelerator contributions and whole-model timing separately. Unknown
measurements stay unknown. Spike functional counts are not accelerator cycles.

Submit hardware measurements through the existing managed FireSim queue using
an explicit pinned hardware/driver selection and owned workload inputs. Verify
RTL/runtime/bitstream correspondence, timer and completion boundaries, warm and
measured windows, full correctness and queue lifecycle receipts. A free queue
slot does not establish those prerequisites. Never substitute the queue's
default hardware or invoke board phases directly.

Untuned defaults may reduce traffic without improving latency; overhead and
overlap can dominate. Full automatic asynchronous double buffering remains
separate implementation work where no qualified transformation exists.
Phase 2 tunes admitted policy choices after this default comparison, preserving
all contracts and freezing learned rules before held-out evaluation.
