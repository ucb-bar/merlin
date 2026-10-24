# FireSim cycle gap, TVM memory mapping, and the next tuning campaign

The [FASED sweep follow-up](FASED_SWEEPS.md) adds explicit latency-pipe timing and
compiler graph-choice search after this initial audit. Its new measurements and
implementation status supersede this audit's historical "not run/not implemented"
statements for those two features; cross-function SRAM residency remains absent.

Audit date: 2026-10-23. Compiler inspected: companion TVM `d4480d521`
(`gemmini/curated-handoff`); integration: Merlin `36899e8f4`. Changes from this
audit live in isolated `gemmini/workload-tuning` and
`comparisons/tvm-workload-tuning` worktrees. Original dirty/deployment trees and
historical measurement artifacts are preserved.

**The current backend exposes scratchpad and accumulator correctly to its
TensorIR scheduler. It is still predominantly an operator optimizer.** Existing
fusion, constant packing, SRAM allocation and software pipelining do not provide
cross-function or cross-session SRAM residency. Complete graph CPU work, staging,
call boundaries, recurrence and actual memory timing must be measured before
attributing the observed gap to a particular tile choice.

## Initial audit change inventory (2026-10-23)

The new work is experiment preparation, measurement visibility and verifier
repair. It does **not** add a hand-selected tile schedule or change the compiler
search space. That initial source audit confirmed no changes to TVM's
`python/tvm/`, `src/` or `include/` relative to `d4480d521`. The subsequent
`FASED_SWEEPS.md` work adds optional stock CSE and graph-option plumbing to the
Python backend; the native C++ compiler remains unchanged.

| Changed component | Actual effect |
| --- | --- |
| `workload_chunks.py` | Defines eleven experiment graphs and exact-program slicing. Model names select test inputs, not production compiler behavior. Connected surrogate formulas are test construction, not new fusion passes. |
| `workload_tuning.py` | Freezes a case, reports matching records/resources, compiles CPU/default/current/tuned variants and delegates later search to existing MetaSchedule. Separate case directories isolate workload/phase measurements. |
| Existing `rtl-v1` records | Copied unchanged and matched by structural IR and capability. No schedule values were manually edited or new winner chosen. A miss uses an explicitly reported unmeasured default. |
| `package_audit.py` / `compile_package.py` | Adds a hashed report of CPU/device calls, SRAM allocations, direct host edges and host memory. Does not change operator scheduling. |
| `qualify_package.py` / `verify_tensorir.py` | Detects actual compiled compute references, counts WS compute and admits explicit larger bounded runs. These repair qualification, not generated model arithmetic. |
| Merlin package-session builder | Adds optional buffered stage-cycle reporting around the existing session calls. It is diagnostic instrumentation, not fusion or memory-residency optimization. |
| Tests and documentation | Covers extraction boundaries, metadata preservation, resource reporting, qualifier controls and stage profiling; records audit evidence and limitations. |

Preparing connected chunks does not make the current tuner optimize their
connection. The wrapper still searches extracted functions independently; the
whole-chunk qualifier is a separate validation/measurement step. There is no
outer graph search or automatic whole-chunk winner promotion in this addition.

## Compiler choices versus fixed optimization passes

Search and passes are complementary: a pass implements a legal transformation;
a search policy can choose whether and where to apply it. A fixed profitability
heuristic can miss workload-dependent tradeoffs, but search has no guarantee of
global optimality either, especially with small budgets or incomplete features.
Keep semantic proofs and mandatory synchronization deterministic. Expose
profitability choices as measured alternatives rather than per-model recipes.

| Decision | Current compiler status | Intended policy |
| --- | --- | --- |
| M/N/K reuse factors, intrinsic rectangle/tails, OS/WS, traversal, hoisting, pipeline, producer/epilogue placement, unroll | Existing structural sketch alternatives and sampled TVM trace decisions; unchanged by this work | Continue per-workload MetaSchedule search. Structural alternatives are enumerated; sampled factors/locations/unroll use stock mutation. |
| CPU versus device for eligible functions | Existing union of CPU/device schedule spaces | Keep both eligible alternatives and measure; unsupported CPU-only tasks are currently excluded from tuning. |
| Relax fusion boundaries | Fixed `FuseOps` / `FuseTIR` preparation before task extraction | Add an outer compiler search over legal regions; changing only a TIR tile cannot undo an earlier graph partition. |
| Weight packing, bias/readout normalization | Mostly proved preparation/lowering; some API flags exist, but the tuning driver fixes their settings | Separate canonicalization needed for recognition from optional transformations with material storage/latency tradeoffs; expose the latter as graph-plan alternatives. An API boolean alone is not automatic search. |
| Host materialization versus on-chip intermediate retention | Host graph ABI today; no cross-function SRAM lifetime contract | Requires implementation and proof of a device-resident region/boundary ABI before it can become a legal knob. |
| Cross-call weight/KV residency | Not implemented | Requires session-aware lifetimes, representation, invalidation, recurrence and capacity contracts; cannot be enabled by a JSON switch. |
| SRAM bounds, alignment, alias/dependence checks, required waits, numeric equivalence | Mandatory compiler proofs/checks | Never search over bypassing these checks. A different legal schedule can eliminate a hazard; blindly omitting a fence cannot. |
| Load coalescing, first-K overwrite, config shadowing, static lifetime reuse | Existing deterministic lowering/runtime mechanisms | Retain valid defaults. Add alternatives only where there is a demonstrated profitability tradeoff, not merely to maximize knob count. |

The preferred extension is two levels of compiler search. First generate a small
set of legal graph plans from semantics, shapes, numerical policy and target
capabilities: fusion regions, producer/consumer layouts, packing/staging and
supported intermediate placement. Then schedule each plan's functions with the
existing TensorIR search. Rank finalists using **whole connected-chunk cycles**,
including CPU glue and transfers, with one fixed warmup/repetition policy. A sum
of individually fastest functions is only a screening estimate. Search keys must
bind the semantic program, transformed plan, phase/shapes, numerical contract,
capability/toolchain and measurement identity; model names remain labels.

For example, an INT32 accumulator output cannot simply remain in SRAM for a
following INT8 contraction if the intervening program requires CPU FP32 scaling,
SiLU or normalization. A legal retained edge needs compatible representation and
layout, all required arithmetic, no conflicting CPU consumer and a bounded
lifetime. Otherwise the graph candidate must retain the host boundary. Even
fusing two contractions into one function does not alone prove a spill-free edge.

A fixed semantic optimization pass is useful as an initial implementation and
baseline for this extension. It should expose legal alternatives to the outer
search when profitability varies. This is the path forward, **not functionality
already implemented or performance-qualified by the eleven prepared cases**.

The supplied screenshot is new external execution evidence, not a locally
reproduced FireSim receipt. It reports ResNet correctness and a TinyLlama
numerical failure on both Spike and FPGA. No raw UART, final ELF, package,
compiler identity or per-stage breakdown was supplied with it. The user later
clarified that an earlier well-optimized run was roughly one hundredth of these
cycles, explicitly an arbitrary reference. Treat that approximate 100× gap as a
motivation, not a controlled before/after result: its workload, hardware, numerical
policy and timing boundary are not bound. This audit cannot authenticate the FPGA
run or attribute a like-for-like slowdown from those numbers alone.

## First fix the units and comparison boundary

| Reported window | Cycles | At the stated 60 MHz target clock | At hypothetical 1 GHz |
| --- | ---: | ---: | ---: |
| ResNet image | 3,230,479,164 | 53.8413 **seconds** | 3.23048 seconds |
| TinyLlama historical 16-token prefill + one decode | 3,407,588,407 | 56.7931 seconds | 3.40759 seconds |

The screenshot's “54 ms at 60 MHz” is a factor-of-1,000 unit error. Actual
elapsed target time still requires the target clock, not the FPGA host/build
frequency. FireSim wall time is a different metric. Spike's reported counter
must not serve as target-cycle or throughput evidence.

The screenshot describes 16+1 TinyLlama; the current selected protocol is 32+32.
SmolVLA is prefix + ten denoise calls + action decode, and ResNet is one image.
Do not compare those different amounts of work, or the newer F2 RTL/memory model
to the pinned U250 run, as a controlled optimization experiment. Do not count
TinyLlama's loosened diagnostic run as a correctness-qualified EL4 result.

The reported 21 Gemmini / 47 CPU kernels also needs its counting definition:
unique compiled functions and dynamic calls differ. Offload fraction by count
is neither MAC coverage nor a latency breakdown. The new package audit records
both unique functions and actual graph calls.

## Scratchpad and accumulator audit

The relevant current compiler sources are in the companion TVM repository:
`python/tvm/relax/backend/contrib/gemmini_{target,intrin,sketch,memory,tensorir}.py`;
primitive lowering/runtime is `apps/gemmini/matmul.c`. This is the default
TensorIR route, not the legacy `block_tiling.py` / BYOC 1/2/4-tile helper.

| Property | What the source implements | Limit |
| --- | --- | --- |
| Hardware capacity | `GemminiCapabilities`: DIM=16, 16,384 scratchpad rows × 16 INT8 elements = **256 KiB**; 1,024 accumulator rows × 16 INT32 elements = **64 KiB**. Capability fingerprint and generated-header hash bind the preset. | Source-bound profile, not live hardware discovery. A matching header alone does not prove the same SoC/DRAM/image. |
| TVM visibility | `cache_read(..., local.gemmini_spad)` for A/B and `cache_write(..., local.gemmini_acc)` for C; tiled layout transforms and physical row alignment. | These are compiler-managed scopes, not a host-addressable allocation or persistent graph cache. |
| Allocation legality | `lower_memory` compacts allocations, injects software pipelining, assigns physical rows, reuses nonoverlapping sibling lifetimes and rejects overflow. Both pipeline slots count. | Per function; no bank-contention cost model or global cross-function allocator. |
| Synchronization | Host staging hazards receive DMA waits; cross-statement host hazards receive fences; function completion makes outputs visible. Remaining CPU accesses to SRAM are rejected. | Host-produced tiles can serialize loads; fences cannot just be deleted. |
| Accumulation | K chunks accumulate in INT32 SRAM; first zero-initialized K overwrites rather than adding a separate zero fill. Bias preload is retained. | Provisional 20-bit PE intermediate limits require safe K chunks; a larger K knob is not automatically legal. |
| Reuse / dataflow | Sampled M/N/K groups, traversal, operand hoisting, raw-command OS/WS, rectangular tails and double buffering. | Hardware-loop/FSM commands remain forbidden. The capability field named `dataflow=OS` is an ABI-profile label; the current primitive sketch space also supports WS. |
| Epilogue | Some exact scalar FP32 RNE scale/ReLU readouts fuse; ordinary supported CPU epilogues can be tiled with contraction. | Per-channel, differing rounding, nonlinear and normalization work often remains CPU. Quantization semantics cannot be changed to force fusion. |

This exposure is through the target-specific TensorIR rule, scopes and allocator,
not a general accelerator memory/timing model in the LLVM target. The target
remains RV64 LLVM with a Gemmini dispatch key. The current `MathematicalFeatures`
adapter feeds stock `PerStoreFeature` from traces before postprocessing into XGB;
it does not supply explicit bank-conflict, command-queue, DMA bandwidth or
cross-function residency features. Measured RTL labels must teach those costs
indirectly, and the nine preset traces are not a universally trained cost model.
Capacity legality is implemented; realistic whole-graph cost prediction is not.

Retained measured packages demonstrate that capacity is usable. In the old
ResNet 7×7 3×3 winner, placement records **14,640 scratchpad rows (234,240 B,
89.36%) and 784 accumulator rows (50,176 B, 76.56%)**. The 32-row TinyLlama KV
winner uses 2,304 / 128 rows (36,864 / 8,192 B). A low fraction on a one-row
kernel is not proof of a missing memory declaration or poor throughput; extra
residency trades against command count, staging and reuse. These are allocation
peaks, not measured array utilization or DRAM bandwidth.

Retained source: `out/artifacts/measurements/baremetal_verilator-gemminirocketconfig/`
in the original comparison tree, each model's `package-v5/placement.json`.
The portable nine-winner preset carries the original measurement/ELF identities.
The final FPGA binary must be checked separately to establish that these schedules
were actually applied there. In particular, a successful `--tuning-preset rtl-v1`
compile can have **zero** structural matches and use unmeasured defaults.

## Why chaining still matters

The actual pipeline is semantic Relax → legalization/fusion → independent TIR
functions → independent schedules → static host call graph. There is useful
within-function producer/epilogue fusion and packed constant folding. There is
no global search over adjacent contractions, no device-resident intermediate ABI
between returned functions, and no persistent weights/KV plan across model calls.
Each returned device function completes its work. Intermediate tensors and serial
operator workspace belong to the host graph wrapper.

The existing `model_workloads.py` emits repeated independent operators and
returns their outputs. Extraction deduplicates tasks and weights calls, but it
does not recreate the real producer-consumer chain. A weighted sum of its winner
cycles omits CPU-only tasks, graph routing, actual activations, recurrent state,
cache conditions and contention. It is an operator estimate, never measured E2E.
The historical timing policy also mixes a first-call interval for long kernels
with a second-call warm interval for short ones; those minima are not a uniform
warm-session baseline. `run_secs` in these databases stores **cycles**, despite its
stock TVM field name.

Some scale examples explain why whole-model residency is impossible:

- A TinyLlama 2048×5632 INT8 MLP weight is 11 MiB, larger than 256 KiB scratchpad.
  A 32×5632 INT32 intermediate is 704 KiB, larger than the accumulator. Tiling is
  mandatory; which tiles to retain is workload dependent.
- Canonical TinyLlama's FP32 KV tensor `[2,22,1,4,64,64]` is 2.75 MiB. It cannot
  remain entirely in either local memory. The validation session currently
  retains every call's outputs; 33 KV outputs alone occupy 90.75 MiB. Distinguish
  required live recurrence from output retention for validation when auditing
  the actual linked memory map.
- The old SmolVLA prefix pilot assumes 241 tokens from three cameras. Current EL4
  selects one camera. The new bootstrap cases cover the camera-independent
  50-action expert path; actual prefix/vision tuning must use the selected capture.

Potential costs to isolate are CPU packing/im2col, quantization and dequantization,
normalization/softmax/rotary/activation math, non-device contractions, host
intermediate copies, repeated weights, per-call fences, and staging placement.
They are hypotheses until the actual guest is profiled. Equal Spike/FPGA outputs
make a shared compiler/numerical difference plausible for TinyLlama; they do not
prove that exp/trig alone explains its error. Find the first divergent stage or
operation using the prescribed portable reference; preserve the tolerance.

## Infrastructure added now

Companion TVM `workload_chunks.py` / `workload_tuning.py` provide eleven bounded
cases, each with a separate immutable program, source/capability identities,
`rtl-v1` seed database, exact structural-match report, CPU exclusions, default and
current resource reports, and an isolated future campaign directory:

| Cases | Actual dimensions retained | Relationship to selected inference |
| --- | --- | --- |
| TinyLlama decode KV and Q/O; prefill KV | M=1 or 32; K=2048; N=256 or 2048 | 44 architectural sites/stage; 32 decode calls or one prefill |
| TinyLlama decode / prefill connected MLP | 2048→5632 gate/up, SiLU×up, 5632→2048, residual; M=1 or 32 | One FFN chunk, 22 layers/stage; normalized input and residual are distinct live-ins |
| SmolVLA action input / output | M=50; 32→720 and 720→32 | Once per denoise call, ten calls |
| SmolVLA connected action/time path | Action projection, concatenate time, 1440→720, SiLU, 720→720 | One camera-independent path per denoise call |
| ResNet layer4 reduction / spatial / connected pair | NHWC 7×7×2048→512 1×1, then 7×7×512 3×3, with integer requant/ReLU in pair | Two late bottleneck sites/image; complete spatial extent, including padding |

These synthetic shapes are **surrogates**, not selected numerical releases.
They intentionally preserve full reductions/channels, tails, intermediate buffers
and CPU glue instead of shrinking everything to 16×16. Architectural occurrence
counts are metadata, not measured end-to-end weights; actual quantization can
split a site. One site is executed in each operator case. Connected cases execute
actual data dependencies rather than independent repeated calls.

The `inventory` / `slice` commands are the preferred route for later real work:
cut a contiguous interval from the exact prepared `.pt2`/Relax graph, retain all
live-outs (including residual/cache branches), expose external dependencies and
keep constants and shapes unchanged. Opaque/global calls are refused. Use actual
captured boundary activations and independent outputs for numerical qualification.
No slicing of channels/reductions, inferred adjacency or model-name compiler
specialization is introduced.

`compile_package.py` now includes a hashed `memory-audit.json`: dynamic call
placement, unique function count, actual per-function SRAM rows converted with
the correct widths, host materialized edges and the existing host memory
preflight. Logical edge bytes are **not measured DMA/DRAM traffic**; direct named
edges are reported and alias/view expansion is explicitly excluded.

The newly compiled connected SmolVLA action/time surrogate has **3 Gemmini
calls, 10 CPU calls, 17 direct host tensor edges (1,119,400 logical bytes)** and
722,112 B of workspace. It has eleven unique functions and thirteen calls.
Only one of its four eligible tuning tasks matches the old preset; the two
TinyLlama connected MLPs have zero preset matches. All seven standalone historical
operator shapes match. This is concrete coverage/memory evidence, not a latency
attribution or proof that all host traffic reaches DRAM.

The existing package-session builder gains opt-in `--profile-stages`. It records
each warm/measured stage `model_run`, keeps aggregate timing, and prints stage
records only after successful validation. Routing/reset/checks are outside stage
windows; aggregate timing includes counter overhead. It adds no mid-interval UART
and does not change tolerance or state routing. Use uninstrumented runs for final
comparisons and instrumented runs for diagnosis. This is stage profiling, not yet
per-operator profiling inside a large stage.

The local package smoke exposed and fixed a verifier defect: database-replayed
functions can omit old M/N/K metadata, which made the qualifier expect auxiliary
DMA activity instead of array compute. Qualification now reads primitive
references from the compiled `operators.o` and counts raw WS compute as well as
OS compute. It preserves the numerical and negative controls. The runner CLI now
accepts explicit bounded chunk budgets up to 100 million cycles / 3,600 seconds,
with its small defaults unchanged; previously it rejected the runbook budgets.

## Current knobs and path forward

The current measured choices are preserved as stock traces in `rtl-v1`, not a
single universal tile setting. Per-case `current-knobs.json` retains matched
traces and their historical cycle labels; `resources-current.json` records actual
lowered allocation. Unknown structures get explicit default placement. The
seven non-smoke winner labels are KV decode 133,169; Q/O decode 1,285,544; KV
prefill 610,198; ResNet 1×1 5,159,256 and 3×3 15,926,916; SmolVLA action input
1,234,495 and output 67,956 cycles. These are historical synthetic operator
measurements, not predictions for the screenshot or the new connected chunks.

The retained trace choices are below. Reuse factors count intrinsic blocks,
not scalar elements; rectangle is M×N×K. Unroll and full trace decisions remain
in each case's `current-knobs.json`. These are historical winners, not newly
optimized settings for the connected chunks.

| Operator | Rectangle | Dataflow / traversal | Reuse M,N,K | Hoist / pipeline | Epilogue / staging |
| --- | --- | --- | --- | --- | --- |
| TinyLlama decode KV | 1×16×16 | OS / ij | 1,8,4 | off / off | tile / tile |
| TinyLlama decode Q/O | 1×16×16 | WS / ij | 1,4,4 | on / off | root / tile |
| TinyLlama prefill KV | 16×16×16 | OS / ji | 1,8,8 | off / on | tile / tile |
| ResNet layer4 1×1 | 7×16×16 | OS / ji | 1,16,2 | off / on | tile / root |
| ResNet layer4 3×3 | 7×16×16 | WS / ji | 7,16,48 | off / off | tile / tile |
| SmolVLA action input | 10×16×16 | OS / ij | 5,3,1 | on / on | tile / tile |
| SmolVLA action output | 10×16×16 | OS / ji | 1,1,9 | on / off | tile / tile |

1. Bind the actual FireSim ELF/package, compiler, selected programs/fixtures,
   image/driver, memory model and timing window. Record preset hits/defaults and
   CPU calls. Fix the units and reproduce correctness without changing tolerance.
2. Get stage attribution with the new session option, then add buffered per-call
   cycle attribution inside the expensive stage. Account for host packing,
   epilogues and state transfer before expanding the tile search space.
3. Replace surrogate cases with exact prepared graph slices and independent
   boundary fixtures. Cover TinyLlama prefill and decode separately, SmolVLA's
   actual one-camera prefix plus repeated expert, and ResNet early/mid/late shapes.
   Late-layer cases alone do not represent ResNet's large early feature maps.
4. Later tune each case's stock MetaSchedule database independently. Sweep the
   existing legal M/N/K reuse, rectangles, OS/WS, traversal, hoisting, pipeline,
   producer/epilogue placement and unroll decisions. Include CPU/default/current
   baselines and retain failures/caps. Do not reuse a winner solely by model name.
5. Qualify and time each **connected package** with one fixed warmup/repeat policy;
   only promote changes that improve that whole chunk. Operator improvements can
   lose after additional copies, larger workspace or extra staging waits.
6. Promote useful fusion/layout/residency strategies into general semantic and
   resource-checked passes, with a device-resident boundary ABI if justified.
   Session state allocation and cross-call residency need explicit lifetimes and
   completion/alias contracts; they are not solved by increasing a tile factor.
7. Rerun complete, correctness-qualified sessions on the same FireSim identity.
   Local Verilator magic-memory rankings are candidate guidance, not proof of
   performance under FireSim's memory model.

No tuning sweep or cloud job was part of the initial setup. The subsequent
local FASED sweeps and selected parameters are recorded in `FASED_SWEEPS.md`.
Follow the companion
TVM `WORKLOAD_TUNING.md` for preparation, compile, independent controls and later
search commands. Test and local build receipts accompany this audit's generated
artifact directory; no full-model performance improvement is claimed.

## Local validation and remaining execution limit

All eleven current-knob packages compiled, passed their integrity/instruction
audits, and contain memory reports. The backend/application suite passed 54 tests;
the final changed-code suite passed 31 (overlapping that suite), and the session
suite passed 36. The target-boundary check passed. Nine module-size findings and
eight regex-boundary findings exactly match the unchanged handoff base.

The independent NumPy reference for the SmolVLA action-output projection passed
positive and wrong-oracle controls on Spike. Its wrong-oracle RTL control passed
in approximately 280 wall seconds. The positive RTL run, including a warmup and
three measured invocations plus guards, hit its 900-second wall cap with no
timing output. It is retained as **failed or incomplete**, not a correctness pass
or cycle label. The fixture, exact guests, source bindings and failed receipts
are preserved. Positive RTL qualification and connected-chunk timings remain for
the next execution campaign; the runbook now shows an explicit larger wall cap,
which itself is not yet validated. No new performance winner was selected.

## PyTorch-export-to-ELF pipeline

This is the TVM comparison path. It imports the prepared PyTorch export directly;
it does not pass through the independently generated Merlin MLIR compiler. The
shared capture/recipe producer supplies the numerical program, and Merlin's
comparison integration supplies session orchestration and execution evidence.

| Stage | Current transformations/decisions | Audit or qualification boundary |
| --- | --- | --- |
| 1. Prepare and freeze model semantics | Select checkpoint, inputs, calibration, quantization/rounding and host-math contract. Expose KV/action/time state as inputs/outputs. These choices precede performance scheduling. | Freeze hashes and independent quantized references; report source-versus-quantized quality separately. Portable full-model numerical release remains incomplete for the selected three-model protocol. |
| 2. PyTorch export | Capture worker saves the exact successful prepared `ExportedProgram` as `program.pt2` using `torch.export.save`; stages are exported separately. | Bind the actual exported artifact, parameters, static shapes and state interface. Do not replace it with a similarly named synthetic graph. |
| 3. TVM intake | `program_io.import_program` loads the `.pt2`, registers serialized Q/DQ ops, and calls `from_exported_program`. Alternative Relax JSON intake is available for experiments. | Check import semantics, dtypes, shapes, constants and numerical policy against the independent program. The importer does not automatically implement every selected portable host-math policy. |
| 4. Mathematical preparation | Optional stock `EliminateCommonSubexpr`, then `LegalizeOps`, `AnnotateTIROpPattern`; with optimization enabled: `FoldConstant`, `CanonicalizeBindings`, `FuseOps`, `FuseTIR`, `RewriteDataflowReshape`, `DeadCodeElimination`. Then scale specialization, optional constant prepacking/bias canonicalization, readout normalization and workload annotation. | The same preparation feeds extraction and compilation. CSE, fusion level, packing and bias preload are recorded outer choices; semantic proofs and CPU fallbacks remain mandatory. These decisions determine the functions visible to inner search. |
| 5. Extract and search, when requested | Deduplicate TIR tasks with IR call weights; record unsupported exclusions. Legal CPU/device sketches, stock EvolutionarySearch, mathematical-store XGB features, builder/runner and JSONDatabase choose per-function schedules. | Bind mathematical references, actual ELF/hardware identities, valid cycle labels and caps. The FASED follow-up remeasures current records and new candidates with a fixed warm policy; the original setup did not run a campaign. IR call weights do not include external session recurrence automatically. |
| 6. Replay and schedule | `MetaScheduleApplyDatabase` replays matching records; `ScheduleGemminiTensorIR` preserves tuned choices, schedules supported misses with defaults, and records CPU fallback reasons. CPU producer inlining follows. | Report exact hits/misses and actual calls, not just unique function counts. Replaying a record is compilation, not fresh optimization or qualification. |
| 7. Tensorize and lower memory | Use proved primitive intrinsics. Lower init/blocks/match buffers, compact allocations, coalesce admissible loads, elide proved zero initialization, expand software pipelines, allocate physical SRAM rows and insert host-DMA waits/completion. | Check full-width INT32 accumulator accounting, INT8 scratchpad accounting, pipeline slots, overflow/range/rounding, lifetime/alias/row alignment and unsupported CPU SRAM accesses. No cross-function residency is created. |
| 8. Static host graph and LLVM | Custom `baremetal.py` plans graph tensors/views/constants and reused host workspace; TVM USMP plans large private allocations with `greedy_by_size`. Preserve bounded RNE quantization, lower operators to RV64GC/lp64d LLVM objects and emit a serial C `model_run` wrapper. | Check static buffer ABI, live/allocated host memory, explicit workspace and runtime dependency closure. This is a custom exporter, not stock TVM VM execution. The new `memory-audit.json` reports these boundaries. |
| 9. Relocatable package | Cross-compile wrapper/runtime, assemble constant blobs, include the qualified primitive adapter, build `libmodel.a` and partially link against `libm`/`libgcc`. | Verify adapter/header/capability/source/library bindings, unresolved symbols, executable instructions and prohibited hardware-loop commands; hash all package payloads. A package is not a boot ELF. |
| 10. Final guest/session ELF | For a chunk, `qualify_package.py` links startup, guards and independent fixtures. For complete inference, Merlin namespaces stage packages, links recurrence/routes, resets, validation, runtime and the explicit memory map. Optional stage counters are buffered. | Audit final ELF segments, entry/ABI, stack/weights/references/arena, symbols and instructions. Preserve every required output and carried state. Linked status does not imply execution success. |
| 11. Execute and admit a result | Spike functional controls, bounded Verilator candidate/chunk runs, then the complete session on its bound FireSim configuration. | Require positive and deliberately wrong-reference controls, all-output numeric agreement and exact executed identities. Use complete warm/measured intervals; keep timing overhead, CPU/transfers/state work and memory-model differences visible. |

The follow-up outer graph-plan search belongs between semantic intake and final
partition selection, feeding stages 4–8 repeatedly. It currently searches fusion
level, constant packing, bias preload and optional CSE with stock Relax traces;
normalization, legality and
memory proofs remain deterministic. Its objective comes from stage 11 for the
connected graph. Inner MetaSchedule search in stage 5 cannot select graph
transformations already committed in stage 4. See `FASED_SWEEPS.md` for the actual
executed budgets, results, coverage gaps and remaining residency work.
