# Generalization changes for TinyLlama, SmolVLA and ResNet-50

This log covers the shared-pass work on `feat/atlas-target` and the companion
Atlas work on `feat/generalized-tinyllama-layer`. Validation is scoped to these
three models. Other model families encountered during discovery are not part of
this validation task. No OOT compiler-generation loop is run.

## Floating component test generation

The [floating test design](../../docs/design/floating_component_tests.md) extends
the existing component DAG and writer with an explicit finite floating policy,
per-operation rounding, ReLU and conversions. Integer programs retain their
existing schema and evaluator. An independent rational reference checks each
rounding step, and source-derived work budgets precede input allocation.

The optional [source suite](phase0/floating-tests.yaml) adds eight public cases,
combining contraction, residual/fan-out, independent work, reuse, logical mutation
and observable rounding/movement boundaries. Other formats and shapes are
repository regressions, not extra OOT-agent cases. Generated capsules validate
against the normal schema and retain required oracle tiers. Their source-only
manifest does not establish hardware/runtime or physical scheduling qualification.

Validation covers upstream parsing, CPU execution of all eight programs under a
second FP16/FP32 policy, and numerical agreement with the selected Atlas model on
the finite E4M3/BF16 inputs. These checks do not execute a generated Atlas compiler
or establish DMA timing, full-chip RTL correctness or full-model readiness.

## Changes already required for the TinyLlama integration

| Change | Shared mechanism / target-specific part | Reason and validation |
|---|---|---|
| Explicit ordered floating contractions | Merlin checks input/accumulator formats, positive-zero initialization, scalar recurrence and K order; Atlas chooses its E4M3/BF16 policy | Basic matmul tests and exact device-tile references |
| Floating graph intake | Merlin reads typed elementwise operations, broadcast, transpose, reductions, reshape and matmul structurally | Attention + MLP fixtures; rejects unknown maps, fastmath and inconsistent types |
| Static shape and fill-type validation | Merlin checks expand-shape extents and fill-region scalar types beyond xDSL verification | Negative regression tests for both malformed forms |
| Batched matmul | Merlin registers the missing named operation and checks batch dimensions, maps and scalar body | TinyLlama QK/PV; unequal batch/contracting dimensions refuse |
| Explicit callback outlining | Merlin extracts checked f32 contractions, preserves other operations and all returns, and uses ranked memref arguments | All nine layer contractions plus returned K/V; duplicate operand and nested-control tests |
| Host ABI and ownership reuse | Merlin uses upstream bufferization/deallocation, its host compiler and existing `memrefCopy` runtime | Noncontiguous operands, fresh outputs and successful native loading |
| Checked scheduling/timing | Merlin consumes explicit overlap/time annotations and verifies emitted schedule evidence; Atlas supplies hardware timing | Source-order and optimized schedule tests; no claimed shared-pass cycle prediction |
| Compact counted K traversal | Atlas consumes the actual shared tile schedule with bounded DMA and exact BF16 accumulator readout/restore | K=5632 fits in 98/96 words; no split-K host summation |
| Explicit padding and boundary precision | Atlas test transport rounds finite f32 to E4M3, saturates, flushes subnormal/zero codes, pads M/N/K and trims verified output tiles | Full-width, reduced and uneven-width TinyLlama fixtures |
| Simulator instruction decoding | Atlas test harness decodes jump immediates for counted loops and verifies binary round-trips | Executed final counted-loop binaries |
| Verification at host/device boundaries | Test-only transport checks exact device arithmetic and compares each host region independently against Torch | FP8 midpoint sensitivity is reported separately, not hidden by a relaxed whole-output tolerance |
| Portable compiler-library export | Merlin exports an explicit source closure, including the batch parser | Isolated-import tests; whole-host helper is not yet part of authoring admission |
| FP16 and FP4/FP6 storage | Merlin provides explicit codecs/packed layouts, independently of instruction support | Numeric storage tests; no claim that Atlas executes FP4/FP6 |

The previous TinyLlama run used synthetic weights, hidden/intermediate widths
2048/5632 and sequence length 32. Source-order and optimized outputs were
bit-identical; all device tiles and host-partition checks passed. It remains a
host/device partitioned layer, not an all-device or pretrained-quality result.
A small f32 RMSNorm difference crosses an FP8 midpoint and gives approximately
54 maximum final-output drift against free-running Torch; this limitation is
retained in the report.

## Additional changes and validation in this pass

| Extension | Trigger and implementation | Checks |
|---|---|---|
| Canonical generic contractions | SmolVLA attention uses six `linalg.generic` batch contractions; the scaled Llama capture uses four. `contraction_kind` recognizes exact rank-two/three indexing maps and parallel/reduction iterators, then the existing reader checks scalar multiply/add/yield, types and shapes. | Rank-two/three, both add operand orders, wrong arithmetic, fastmath and alternate-map regressions. Alternate valid maps remain host work. |
| Literal splat initializers | These captures use `tensor.splat` positive zero. Extraction clones the actual constant and splat, then invokes the existing reader. The `linalg.fill` path retains its original checks. | Fill/splat cross-product tests; negative-zero and nonzero seeds refuse. No arbitrary accumulator state is admitted. |
| Typed static host boundaries | Integer embedding IDs, boolean masks and rank-zero batchnorm counters prevented the f32-only model wrapper from accepting these captures. Merlin now describes supported static tensor shapes/dtypes; Atlas delegates to it. | Integer/mask/scalar signature regression; dynamic tensors refuse. Runtime checks actual shape/dtype before native invocation. Device contractions still require f32. |
| Reusable model validation | Atlas factors its existing tile simulator transport out of the TinyLlama-specific checker. Fresh captures use the existing model2MLIR frontend and retained PyTorch recipes, with quantization disabled explicitly. | Original f32 host graph versus Torch; every device tile versus independently implemented ordered arithmetic; complete outputs versus the same host graph with software contraction callbacks. |

No new convolution pass was needed: existing model2MLIR decomposition emits
im2col plus matmul for ResNet-50. No production behavior is selected by model
name. The new compiler changes live in `frontends/linalg_tensor_graph.py` and
`llvmlower/floating_matmul_outline.py`; target execution remains in Atlas.
FP4/FP6 storage support is retained, but these Atlas runs use E4M3/BF16 only.

## Executed scope and findings

All fixtures use synthetic weights. Every listed contraction executes in the
pinned simulator, while other operators remain compiled host work. The scaled
Llama fixture covers embedding, two transformer layers and logits; it supplements
the previous full-width TinyLlama layer without claiming a complete 1.1B run.
SmolVLA covers the retained reduced prefix/expert denoising-step recipe, not the
full pretrained robot policy. ResNet-50 runs the complete architecture with
32×32 input; 224×224 compilation passes in both modes but has not been simulated.

| Fixture | Device contractions | Tiles per mode | Source-order cycles | Optimized cycles |
|---|---:|---:|---:|---:|
| Scaled Llama, width 128/intermediate 344, two layers, sequence 8 | 19 | 108 | 940,580 | 909,548 |
| Reduced SmolVLA prefix/expert step | 28 | 37 | 165,181 | 160,245 |
| Complete ResNet-50, 32×32 | 54 | 924 | 45,974,812 | 44,227,900 |

All 2,138 device tiles across these six runs match ordered arithmetic bit-for-bit.
Final outputs are bit-identical between scheduling modes. Original f32 compiled
host outputs match Torch within rtol/atol 2e-4 (maximum absolute differences:
9.54e-7, 2.98e-7 and 1.93e-5 respectively). Device-path outputs match the same
host image with independent software contraction arithmetic exactly. This last
check isolates device/tiling/transport correctness, not independent whole-model
approximation accuracy. Maximum absolute differences versus original f32 Torch
are approximately 0.181, 0.0963 and 8.083; no pretrained accuracy claim follows.

Cycles are npu_model device-launch totals only, excluding host work, conversion
and transport. They are not wall-clock performance or a generated-versus-handwritten
backend comparison. The exact model pin is
`47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`; selected RTL identity is
`0079c0541111197741a231c002e3843fa6f545b2`. Source/tool/artifact hashes and individual
results are retained under the companion worktree's `out/build/model-portfolio`.
See its `docs/generalized-model-validation.md` for reproduction commands.

## Remaining qualification

This proves the initial mixed host/device compilation path for the stated
fixtures, not unrestricted full-model support or OOT-authoring admission.
Production transport, dynamic shapes/control flow, complete pretrained models,
KV-cache decoding and an application-approved quantization policy remain work.
The isolated compiler-library export remains tested; the whole-host outlining
helper is not yet admitted into the compiler-authoring source closure. No OOT
compiler-generation or optimization-agent loop was started.

## Static scheduling and authoring-library extension

- Added a source-order fixed-block scheduler driven entirely by provider effects,
  resource capacities and natural issue advances. Serial and independent-overlap
  policies share existing timing verification.
- Added `schedule.drain` and `schedule.await_completion` with actual consumers.
  Fixed drains become target delays; event waits remain tokens in existing
  counted-loop IR and require target emission support.
- Added plan-to-emitted-call correspondence and delay/tail validation. Final
  binary decoding and hardware-specific timing verification remain OOT duties.
- Split pure contraction outlining from the installed host build driver and
  exported it alongside both scheduling APIs (46 members, 19 public modules).
- Added portability tests for pipelines, multiple issue, read/write hazards,
  resource reservations, long latencies, and a compiled event-driven loop.
- Added an Atlas fixed-only consumer using the existing model timing provider's
  conservative complete-machine occupancy. It schedules real matmul and VPU
  programs through Merlin, then verifies and executes the resulting words.
  This initial consumer rejects DMA and CFG; it does not replace the existing
  full-model scheduler or qualify whole-model OOT authoring by itself.

See the [design and API contract](../../docs/design/static_scheduling.md).
