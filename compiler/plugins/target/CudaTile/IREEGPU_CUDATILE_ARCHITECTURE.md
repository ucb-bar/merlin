# IREEGPU to CudaTile Architecture Investigation

Date: 2026-04-17

## Summary

The best architecture is not `linalg -> raw cuda_tile` and not `linalg -> amdgpu -> cuda_tile`.
The best architecture is:

```text
torch / linalg / tensor
  -> linalg ops configured with iree_gpu target + lowering attrs
  -> CudaTileKernelPlan
  -> cuda_tile dialect emission
  -> tilebc / tileiras / cubin
```

In other words: use `iree_gpu` as the GPU planning/configuration layer, not as a full replacement semantic op dialect. `iree_gpu` gives us target descriptions, lowering configs, MMA layout attrs, promotion metadata, and a few GPU-specific helper ops. It does not provide semantic ops for matmul, conv, reductions, softmax, elementwise, transpose, etc.; those remain in `linalg`, `tensor`, `vector`, and `scf` until later lowering.

The first concrete migration should replace the current string-tag contract (`cuda_tile.kernel_class`) with a structured plan extractor that reads:

- `linalg` semantics and indexing maps
- `tensor_ext.dispatch.tensor.load/store` slice metadata
- `IREE::GPU::TargetAttr`
- `IREE::GPU::LoweringConfigAttr`
- `IREE::GPU::MMAAttr` / `VirtualMMAAttr` / `DataTiledMMAAttr` where present

Then the cuda_tile backend should emit tileir from that plan.

## Why `iree_gpu` Is A Good Fit

`iree_gpu` explicitly describes itself as common GPU codegen functionality that may be hardware-specific but is independent of the final lowering target. See:

- `third_party/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td:18`
- `third_party/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td:26`
- `third_party/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td:28`

That is exactly the kind of layer we want above `cuda_tile`: hardware-aware enough to drive performance, but not tied to NVVM/ROCDL/SPIR-V emission.

Important reusable pieces:

- `LoweringConfigAttr`: flexible GPU lowering config carrying tiling levels, lowering strategy, and workgroup reordering. See `IREEGPUAttrs.td:24`.
- `MMAAttr`: target MMA intent and layouts, abstractly representing `C += A x B`. See `IREEGPUAttrs.td:249`.
- `DataTiledMMAAttr`: richer MMA tile/subgroup/interleaving layout for performance-sensitive kernels. See `IREEGPUAttrs.td:383`.
- `VirtualMMAAttr`: virtual/unrolled/interleaved MMA variants that delay committing to one native intrinsic layout. See `IREEGPUAttrs.td:517`.
- `TargetWgpAttr` / `TargetAttr`: normalized target feature/limit model across AMD CUs and NVIDIA SMs. See `IREEGPUAttrs.td:715` and `IREEGPUAttrs.td:810`.
- `coalesced_gather_dma`: an example of one high-level GPU memory op with multiple lowering paths (`amdgpu.gather_to_lds` or generic `vector.gather`). See `IREEGPUOps.td:230` and `IREEGPUOps.td:255`.

This gives us a tested design vocabulary for target-independent-but-hardware-aware GPU lowering.

## Why `iree_gpu` Is Not A Complete Semantic Layer

`iree_gpu` has only a small op surface:

- `iree_gpu.barrier_region`
- `iree_gpu.value_barrier`
- `iree_gpu.yield`
- `iree_gpu.buffer_resource_cast`
- `iree_gpu.coalesced_gather_dma`

See `third_party/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.td:26`.

It does not have first-class ops for:

- matmul
- conv1d/2d/3d
- reduction
- softmax
- elementwise
- transpose
- broadcast
- slice

Those semantics remain in `linalg`, `tensor`, `vector`, `scf`, and IREE dispatch/tensor-ext IR. So the phrase "`iree_gpu -> cuda_tile`" should mean "configured GPU codegen state -> cuda_tile", not "replace linalg with iree_gpu ops".

## Current CudaTile Architecture

Today the CudaTile target is mostly a target-side semantic recovery system:

- Preprocessing and translation passes attach string attrs like `cuda_tile.kernel_class`.
- `buildCudaTileKernel` walks the inner dispatch module, picks a primary op, recovers shapes/slices/fusion information, and emits tileir directly.
- This combines semantic classification, scheduling, fusion detection, and target emission in one place.

Key locations:

- Main backend entry: `compiler/plugins/target/CudaTile/CudaTileTarget.cpp:1812`
- String attr lookup: `compiler/plugins/target/CudaTile/CudaTileTarget.cpp:1819`
- Metadata fallback recovery: `compiler/plugins/target/CudaTile/CudaTileTarget.cpp:1923`
- Translation-time annotation pipeline: `compiler/plugins/target/CudaTile/CudaTileTarget.cpp:3288`
- Preprocessing pipeline: `compiler/plugins/target/CudaTile/CudaTileTarget.cpp:3691`

Current classification passes:

- Data movement pass tags copy/transpose/broadcast/slice/collapse/expand-like cases. See `compiler/src/merlin/Dialect/CudaTile/Transforms/ConvertDataMovementToCudaTile.cpp:7`.
- Elementwise pass tags `linalg.generic` with supported scalar body ops. See `ConvertElementwiseToCudaTile.cpp:73`.
- Reductions pass tags `linalg.reduce` and `linalg.generic` reductions with `cuda_tile.combiner`. See `ConvertReductionsToCudaTile.cpp:7`.
- Contractions pass tags named matmul and generic mul+add reductions as `"matmul"`. See `ConvertContractionsToCudaTile.cpp:7`.

This has been good for bring-up but weak as a long-term contract.

## Problems We Already Hit That Point To A Missing Planning Layer

These are not abstract concerns; they showed up in the current path.

1. Slice/broadcast semantics were implicit.
   - Extract/insert slice correctness depended on recovering `dispatch.tensor.load/store` offsets and strides in the backend.
   - Broadcast needed explicit reshape plus broadcast because `cuda_tile.broadcast` does not change rank. See `cuda_tile` broadcast docs at `third_party/cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Ops.td:549`.

2. Conv semantics were lost too early.
   - `conv2d_1x1_stride2` looked like a generic contraction after lowering.
   - The backend originally flattened it as contiguous matmul and ignored slice strides.
   - We had to add a special sliced-pointwise matmul kernel in `CudaTileTarget.cpp`.

3. Fusion is target-side guessing.
   - Softmax and fused reduction/elementwise are handled by scanning multiple generics and reconstructing patterns in the backend.
   - That should be a planned fused kernel form or at least an explicit lowering config.

4. Schedule is not an explicit contract.
   - `CudaTileOptions` provides tile sizes, but there is no structured schedule object equivalent to IREEGPU lowering config.
   - The backend chooses/assumes tile structure as it emits.

5. CudaTile op semantics are lower-level than linalg semantics.
   - `cuda_tile.mmaf` expects concrete 2D/3D tile shapes. See `Ops.td:1191`.
   - `cuda_tile.make_tensor_view` and `make_partition_view` need physical shapes/strides/tile partitioning. See `Ops.td:2593` and `Ops.td:4053`.
   - `cuda_tile.load_view_tko` and `store_view_tko` operate on tile views and view index spaces. See `Ops.td:2198` and `Ops.td:3837`.

These are exactly the kinds of decisions `iree_gpu` lowering config and target attrs are meant to make explicit.

## Equivalent Operation / Concept Mapping

| Concept | `linalg` / semantic IR | `iree_gpu` role | `cuda_tile` emission |
|---|---|---|---|
| Elementwise | `linalg.generic` with all parallel iterators and scalar body ops | Usually schedule/config only; no dedicated elementwise op | `addf`, `mulf`, `subf`, `divf`, `exp`, `sqrt`, `cmpf`, `select`, etc. |
| Broadcast | `linalg.broadcast` or generic indexing map | Layout/dimension-expansion config where useful | `reshape` then `broadcast`; rank changes must happen before broadcast |
| Transpose | `linalg.transpose` or generic indexing map | Layout/permutation/schedule config | `permute`, altered tensor-view strides, or tile load/store mapping |
| Slice / extract / insert | `tensor.extract_slice`, `tensor.insert_slice`, `dispatch.tensor.load/store` slices | Offset/size/stride planning; promotion decisions | `offset`, `make_tensor_view`, `load_ptr_tko`, `store_ptr_tko`, or view load/store |
| Reduction | `linalg.reduce`, `linalg.generic` with reduction iterators | Reduction tiling strategy, subgroup/workgroup choices | `cuda_tile.reduce`, scalar combiner body, loops for multi-tile reductions |
| Matmul | `linalg.matmul`, `linalg.batch_matmul`, generic contraction | `MMAAttr`, `VirtualMMAAttr`, `DataTiledMMAAttr`, lowering config | `cuda_tile.mmaf` / `mmai`, K loop, views, accumulator tile |
| Conv | named linalg conv or generic indexed contraction | Direct-conv or IGEMM config; map image/channel/filter loops to M/N/K | direct conv emitter or implicit-GEMM tile loop using `mmaf` |
| Softmax | decomposed reductions + elementwise or named softmax before decomposition | fused reduction/elementwise plan and tiling strategy | reduce max, subtract, exp, reduce sum, divide |
| Barrier / sync | usually absent from linalg; appears after parallelization | `barrier_region`, `value_barrier`, `gpu.barrier` | cuda_tile tokens/barriers/control sequencing where needed |
| Cooperative memory movement | tensor/memref slices after tiling | `coalesced_gather_dma`, promotion attrs, DMA config | tile views, `load_view_tko`, `store_view_tko`, pointer `offset`, tokens |
| Target properties | absent | `TargetAttr`, `TargetWgpAttr`, `TargetChipAttr` | tile sizes, grid shape, MMA choice, shared memory use, load width |

The key point: the semantic computation remains in `linalg`. `iree_gpu` provides schedule/target/MMA/memory intent. `cuda_tile` is the concrete tile-level executable IR.

## Should We Use `iree_gpu` Directly?

Yes, but in stages.

### Directly useful now

These are worth reusing if build dependencies are manageable:

- `IREE::GPU::TargetAttr`
- `IREE::GPU::LoweringConfigAttr`
- `IREE::GPU::MMAAttr`
- `IREE::GPU::VirtualMMAAttr`
- `IREE::GPU::DataTiledMMAAttr`
- `IREE::GPU::GPUPipelineOptionsAttr`
- `GPULoweringConfigUtils`
- selected target utilities for CUDA/HIP/Vulkan-style target descriptions

The CudaTile target currently does not depend on `IREEGPUDialect`. Its CMake deps are listed in `compiler/plugins/target/CudaTile/CMakeLists.txt:1`. Direct reuse would require adding the IREEGPU dialect/library dependency and includes such as:

```cpp
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/DerivedConfigUtils.h"
```

The local IREEGPU CMake target is defined at `third_party/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/CMakeLists.txt:8`.

### Not useful as a full replacement

Do not replace `linalg` semantic ops with `iree_gpu` ops, because the ops are not equivalent. `iree_gpu` does not have a matmul/conv/reduce/softmax dialect surface.

Do not route through `amdgpu` to get to `cuda_tile`; `amdgpu` is a late AMD bridge dialect, not a cross-vendor planning layer.

## Recommended Architecture

### Phase 1: Add a `CudaTileKernelPlan` extractor

Keep IR unchanged initially. Add a C++ planning layer inside or next to `CudaTileTarget.cpp`:

```cpp
struct CudaTileKernelPlan {
  enum class Kind {
    Copy,
    Transpose,
    Broadcast,
    Slice,
    Elementwise,
    Reduction,
    Matmul,
    Conv,
    Softmax,
    Unsupported,
  };

  Kind kind;
  IREE::GPU::TargetAttr target;
  IREE::GPU::LoweringConfigAttr loweringConfig;
  Attribute mmaKind; // MMAAttr, VirtualMMAAttr, DataTiledMMAAttr, or null.
  // Shapes, strides, indexing maps, reduction dims, fusion graph, bindings.
};
```

The important change is that `buildCudaTileKernel` should consume this plan, not rediscover semantics directly from raw IR every time.

Initial sources for the plan:

- existing linalg op/indexing map analysis
- `dispatch.tensor.load/store` offsets, sizes, strides
- current `CudaTileOptions`
- optional `IREE::GPU::LoweringConfigAttr`
- optional `IREE::GPU::TargetAttr`

This can be incremental and does not require a new MLIR dialect.

### Phase 2: Replace string tags with structured config

Current:

```mlir
cuda_tile.kernel_class = "matmul"
cuda_tile.op_name = "subf;exp"
```

Better:

```mlir
lowering_config = #iree_gpu.lowering_config<...>
cuda_tile.plan = #cuda_tile.plan<kind = matmul, ...> // optional later
```

The key is to stop using `kernel_class` as the only contract. It is too lossy for conv, sliced pointwise, fused reductions, and layout-sensitive paths.

### Phase 3: Target configured `linalg`, not raw `iree_gpu` ops

The stable compiler boundary should be:

```text
configured linalg dispatch + iree_gpu attrs -> CudaTileKernelPlan
```

Not:

```text
iree_gpu ops only -> cuda_tile
```

This matches how IREE uses `iree_gpu`: it configures and transforms GPU codegen, while semantic work remains in other dialects.

### Phase 4: Lower selected `iree_gpu` helper ops to cuda_tile

Once the plan path exists, add direct lowering for:

- `iree_gpu.value_barrier` / `barrier_region` where they survive to CudaTile
- `iree_gpu.coalesced_gather_dma` into cuda_tile view/pointer loads
- promotion/cache swizzle metadata into cuda_tile view/load hints if cuda_tile supports the equivalent

This is optional at first. The larger win is using IREEGPU attrs/configs.

## Direct Conv / IGEMM Notes

IREE has both direct convolution and IGEMM configuration hooks:

- `setDirectConvolutionLoweringConfig`
- `setIGEMMConvolutionLoweringConfig`
- `setMatmulLoweringConfig`
- `setReductionConfig`

See `third_party/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h:19`.

However, the local direct-conv config currently rejects non-unit strides:

- `third_party/iree/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp:1776`

Our current cuda_tile conv regression matrix has all tested `conv2d` stride/batch cases marked passing, while non-pointwise `conv1d` and non-pointwise `conv3d` remain expected failures:

- `compiler/plugins/target/CudaTile/test/conv_regression.py:276`
- `compiler/plugins/target/CudaTile/test/conv_regression.py:288`
- `compiler/plugins/target/CudaTile/test/conv_regression.py:295`

So we should borrow IREE's config structure and target modeling, but not blindly adopt its current direct-conv constraints.

## Risk Assessment

### Directly adopting IREEGPU attrs

Benefits:

- avoids inventing our own target and MMA abstraction
- better alignment with IREE compiler internals
- creates a path to eventually share scheduling/config utilities
- makes CudaTile less of a special monolith

Risks:

- CudaTile target must link more IREE codegen dialect libraries
- `iree_gpu` is internal and may churn
- some utilities assume LLVMGPU/SPIR-V pipelines
- current IREE conv/reduction heuristics may be behind our cuda_tile experiments

### Creating a Merlin-only GPU dialect

Benefits:

- full control over contract and stability
- can be exactly shaped around cuda_tile needs

Risks:

- duplicates IREEGPU concepts
- makes future integration with IREE GPU codegen harder
- likely reinvents target attrs, MMA attrs, pipeline options, and config utilities

### Recommendation

Start with direct reuse of IREEGPU attrs/configs where practical. Do not create a Merlin GPU dialect yet. Create a C++ `CudaTileKernelPlan` boundary first. If that boundary stabilizes and needs to be inspected/transformed by multiple passes, then promote only that small contract to MLIR.

## Proposed Milestones

1. Add IREEGPU dialect dependency to the CudaTile target.
2. Add `CudaTileKernelPlan` extraction from current dispatch IR.
3. Teach the extractor to read existing `cuda_tile.kernel_class` attrs only as a fallback.
4. Teach the extractor to read `IREE::GPU::LoweringConfigAttr` and `TargetAttr`.
5. Refactor `buildCudaTileKernel` into:
   - `extractCudaTileKernelPlan`
   - `emitCudaTileFromPlan`
   - per-kind emitters
6. Convert matmul first:
   - use `MMAAttr` / `VirtualMMAAttr` when available
   - fall back to existing tile sizes otherwise
7. Convert reduction second:
   - use lowering config to decide workgroup/subgroup/multi-kernel strategy
8. Convert conv third:
   - use a plan that preserves conv rank, batch, stride, dilation, padding, groups
   - keep current tested direct `conv2d` behavior
   - do not force materialized im2col as the only path
9. Convert fused softmax/fused reduction-elementwise:
   - represent fusion explicitly in the plan
   - stop relying on backend-side generic-op pattern guessing

## Concrete Next Step

The smallest useful code change is not a new dialect. It is a new file pair:

```text
compiler/plugins/target/CudaTile/CudaTileKernelPlan.h
compiler/plugins/target/CudaTile/CudaTileKernelPlan.cpp
```

Initial scope:

- Parse one dispatch inner module.
- Identify kind: elementwise, reduction, matmul, conv, data movement.
- Preserve binding indices, shapes, strides, offsets, indexing maps, reduce dims.
- Optionally read IREEGPU target/lowering config if present.
- Return a strongly typed `CudaTileKernelPlan`.

Then `CudaTileTarget.cpp` can be reduced toward dispatching on that plan instead of walking and guessing repeatedly.

That gives us the architectural boundary we need while keeping existing tests and current cuda_tile emission mostly intact.

## Expanded Rollout Plan

The migration should be an architectural refactor with a correctness ratchet, not a stop-the-world rewrite. The existing backend already runs real kernels, including non-trivial `conv2d` cases, so the first rule is: keep the executable path working while moving planning decisions out of the emitter.

The intended long-term shape is:

```text
Torch / StableHLO / Linalg import
  -> IREE flow dispatch formation
  -> CudaTile configuration pass
       - reads HAL executable target
       - builds or imports IREEGPU target/config attrs
       - attaches lowering configs to root linalg ops
  -> CudaTile plan extraction
       - reads configured linalg + tensor_ext dispatch loads/stores
       - classifies semantics and fusion
       - preserves bindings, shapes, strides, offsets, indexing maps
  -> CudaTile emission
       - consumes CudaTileKernelPlan only
       - emits cuda_tile tile/view/MMA/reduce/control ops
  -> tilebc
  -> tileiras
  -> cubin
  -> cuda_tile HAL runtime
```

The important policy is that the configured `linalg` IR remains the semantic contract. `iree_gpu` is used for target and lowering configuration. The CudaTile emitter should not require a new semantic dialect before it can generate code.

## Phase 0: Stabilize Current Behavior

Before refactoring, lock down the current behavior so architecture work cannot silently regress it.

Deliverables:

- Keep `compiler/plugins/target/CudaTile/test/conv_regression.py` and promote it from an ad hoc script to a documented manual/GPU regression harness.
- Add a smaller lit-style compile-only suite for plan extraction and annotation, because GPU execution is too slow and machine-dependent for every edit.
- Record current expected support exactly: elementwise/data movement/reductions/matmul/softmax plus `conv2d` pass cases, with non-pointwise `conv1d`/`conv3d` marked unsupported until planned.
- Add debug dumps gated behind an option, not local `llvm::errs()` edits, so plan extraction can be inspected consistently.

Success criteria:

- Existing cuda_tile smoke tests still compile.
- `conv_regression.py` continues to report no unexpected failures on an RTX A5000-like machine.
- No new architectural dependency on IREEGPU is required yet.

## Phase 1: Introduce `CudaTileKernelPlan`

This is the first real architecture phase. It should not change generated kernels initially.

Create:

```text
compiler/plugins/target/CudaTile/CudaTileKernelPlan.h
compiler/plugins/target/CudaTile/CudaTileKernelPlan.cpp
```

Initial plan model:

```cpp
enum class CudaTileKernelKind {
  Copy,
  ExtractSlice,
  InsertSlice,
  Transpose,
  Broadcast,
  Elementwise,
  Reduction,
  Matmul,
  Conv,
  FusedReductionElementwise,
  Unsupported,
};

struct CudaTileOperandPlan {
  int64_t binding = -1;
  SmallVector<int64_t> logicalShape;
  SmallVector<int64_t> physicalShape;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  bool isConstant = false;
};

struct CudaTileSchedulePlan {
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> threadTileSizes;
  SmallVector<int64_t> reductionTileSizes;
  SmallVector<int64_t, 3> gridDims;
  Attribute mmaKind;
  Attribute target;
  Attribute loweringConfig;
};

struct CudaTileKernelPlan {
  CudaTileKernelKind kind = CudaTileKernelKind::Unsupported;
  Operation *rootOp = nullptr;
  SmallVector<Operation *> fusedOps;
  SmallVector<CudaTileOperandPlan> operands;
  CudaTileSchedulePlan schedule;
  Type elementType;
  SmallVector<int64_t> resultShape;
};
```

Phase 1 extractor inputs:

- HAL binding subspans and `IREE::TensorExt::DispatchTensorLoadOp` / `DispatchTensorStoreOp`
- `linalg::LinalgOp` indexing maps, iterator types, DPS inputs/inits
- existing `cuda_tile.kernel_class` attrs as compatibility fallback only
- `CudaTileOptions` tile sizes
- optional lowering config attrs if already present

Phase 1 emitter rule:

- `buildCudaTileKernel` calls `extractCudaTileKernelPlan`.
- Existing kernel generator functions stay mostly unchanged.
- Dispatch happens on `plan.kind`.
- Each existing generator receives data from the plan instead of rediscovering it from arbitrary IR.

This is the key mechanical step. It lets us preserve Claude's working code paths while removing the architectural reason they keep growing into one large pattern-matching backend.

## Phase 2: Respect IREEGPU Configuration

Once plan extraction exists, add IREEGPU as an optional planning source.

Target integration:

- Add CMake deps for `iree::compiler::Codegen::Dialect::GPU::IR` and any small utility libraries needed to read attrs.
- Register `IREE::GPU::IREEGPUDialect` in `CudaTileTargetBackend::getDependentDialects`.
- Add `IREE::GPU::TargetAttr` to the cuda_tile executable target configuration using the same convention consumed by `getGPUTargetAttr`.
- Populate CUDA target info from `options.smArch` through `IREE::GPU::getCUDATargetDetails`.

Lowering config integration:

- First read existing `IREE::GPU::LoweringConfigAttr` if present.
- If absent, synthesize conservative configs from `CudaTileOptions`.
- Do not immediately call the full LLVMGPU kernel config pipeline inside CudaTile.
- Borrow utility functions where they are clearly independent, especially target attr lookup, lowering config accessors, MMA attr extraction, and reduction config structure.

Policy:

- Existing command-line tile options remain fallbacks.
- IREEGPU config wins when present.
- If an IREEGPU config is unusable for CudaTile, emit a clear diagnostic and fall back only when correctness is not at risk.

## Phase 3: E2E Bring-Up Order

The best order is not strictly "simple to complex"; it is "small surface area first, then performance-critical structure".

1. Data movement and elementwise

   This validates the plan boundary, binding mapping, shape/stride handling, rank-changing broadcast behavior, and pure tile load/store/storeback. These kernels do not need IREEGPU MMA config, but they should still read target/config attrs if attached so the path is exercised.

2. Reductions

   Reductions force the plan to represent reduction dims, combiners, full-reduction tile constraints, and fused post-reduction elementwise. Use IREEGPU reduction config structure as soon as possible, but keep the current simple full-reduce-in-one-tile path as the correctness fallback.

3. Matmul / MMA

   Matmul is where IREEGPU matters most. The plan should represent M/N/K, batch dims, transposed operands, constant weights, K-loop count, tile sizes, MMA kind, and optional epilogue fusion. Start with existing `mmaf` emission, then let IREEGPU `MMAAttr` / `VirtualMMAAttr` / `DataTiledMMAAttr` override the intrinsic/tile strategy when present.

4. Fused matmul epilogues

   Bias, activation, residual add, and simple elementwise epilogues should be explicitly represented as fused ops in the plan. Do not keep adding special post-op scans in the emitter.

5. Conv2d

   Preserve the current passing direct `conv2d` behavior, including 1x1, non-1x1, batch, odd sizes, and stride 2/3 cases. Refactor it behind a `ConvPlan` that preserves rank, batch, spatial shape, filter shape, strides, dilations, channel dims, and lowering mode. Do not require materialized im2col. Use implicit-GEMM/direct-conv emission as the baseline and let IREEGPU direct/IGEMM config choose strategy later.

6. Conv1d and conv3d

   Generalize the conv plan to spatial rank N instead of adding one-off emitters. For phase 1 support, `conv1d` can be represented as a degenerate rank-2 plan and `conv3d` as rank-3 direct convolution. Performance can lag correctness initially, but the representation must not bake in rank 2.

7. Softmax and normalization families

   Softmax, layer norm, RMS norm, and batch norm need explicit fused reduction-elementwise plans. The current dynamic generic-op walk is useful evidence, but it should not remain the contract.

8. Advanced memory movement

   Only after the above works, map selected `iree_gpu` helper ops such as `value_barrier`, `barrier_region`, and `coalesced_gather_dma` to cuda_tile barriers/views/loads where useful.

## Phase 4: Performance Path

Correct E2E support is not enough. The performance path needs explicit schedule ownership.

Near-term performance levers:

- Use `IREE::GPU::TargetAttr` to choose default subgroup size, shared memory assumptions, and legal MMA families.
- Use `IREE::GPU::LoweringConfigAttr::getWorkgroupTileSizes()` instead of only `CudaTileOptions::tileM/tileN/tileK`.
- Use `GPULoweringConfigUtils::getMmaKind` to recover MMA intent.
- Add tile-size validation against cuda_tile/tileiras requirements, especially power-of-two padding and supported MMA dimensions.
- Preserve an inspectable plan dump so contributors can see why a kernel got a particular schedule.

Medium-term performance levers:

- Add shared-memory/promotion fields to the plan rather than hard-coding direct global loads forever.
- Use IREEGPU promotion metadata when it maps cleanly to cuda_tile view/load constructs.
- Add plan variants for direct-load MMA, promoted MMA, and implicit-GEMM conv.
- Start measuring against IREE CUDA/LLVMGPU and native CUDA baselines, not only correctness against CUDA output.

## What Current Claude-Written Code Is Worth Saving

The current implementation is not architecturally clean, but a complete rewrite would throw away useful bring-up work. The right move is extract-and-encapsulate, not delete-and-rewrite.

Keep:

- `CudaTileOpEmitter`: this is valuable backend infrastructure. It hides cuda_tile builder quirks and should remain the emission utility.
- Existing generator functions for copy, transpose, extract slice, insert slice, broadcast, elementwise, reduction, matmul, direct conv2d, and strided pointwise matmul. They are working references and should be refactored to consume plans.
- `buildConvPlan` logic as the seed of a real `ConvPlan`, especially affine-map parsing for sliding-window access, stride/dilation recovery, and the pointwise/direct-conv distinction.
- The direct `generateConv2DKernel` path. It proves materialized im2col is not required for correctness and covers important stride/batch cases.
- `generateStridedPointwiseMatmul2DKernel`. This encodes a real bug fix: sliced pointwise conv/matmul cannot be treated as contiguous flattened matmul.
- The conv regression harness. It is much better than only hand-written all-ones tests because it uses random inputs and a NumPy oracle.
- The elementwise op mapping tables, with cleanup. The coverage is useful even though the tag contract is weak.

Rewrite or heavily refactor:

- `buildCudaTileKernel` control flow. It should become plan extraction plus plan-based dispatch, not a semantic recovery monolith.
- Multi-op fused generic scanning. This should move into plan extraction with an explicit fused DAG or ordered fused-op list.
- String-only attrs such as `cuda_tile.kernel_class` and `cuda_tile.op_name` as primary contracts. Keep them only as compatibility/fallback during migration.
- Binding discovery and shape/stride recovery. This belongs in one reusable extractor, not repeated inside matmul, fusion, data movement, and fallback paths.
- Constant weight handling in the matmul emitter. It is useful but currently too special-cased and incomplete for multi-K constant tiles.
- The preprocessing im2col assumption. Keep im2col as an optional strategy, not as the required conv path.
- `ConvertSCFToCudaTile.cpp` as currently written. It only tags SCF ops and does not perform a meaningful lowering.

Delete only after replacement:

- Copy fallback for unknown multi-op dispatches. It can silently produce wrong answers, so once plan extraction exists, unsupported dispatches should fail loudly unless they are proven pure copies.
- Backend-side "promote matmul as primary and fall through" control flow. The plan extractor should choose the root and fusion policy explicitly.

## First Implementation Slice After This Investigation

The first slice should be small and testable:

1. Add `CudaTileKernelPlan.{h,cpp}` with no IREEGPU dependency yet.
2. Move binding collection, static shape extraction, dispatch load/store slice extraction, and primary op selection into the extractor.
3. Make `buildCudaTileKernel` call the extractor and dispatch on `plan.kind`.
4. Wire only copy, extract slice, insert slice, broadcast, elementwise, reduction, and matmul through the plan.
5. Keep direct conv2d behind the existing path until the basic plan path is stable.
6. Add compile-only tests for plan-kind classification where possible.
7. Run the existing GPU regression harness after each moved kernel family.

The second slice should add optional IREEGPU reads:

1. Add the CMake and dialect dependency.
2. Add target attr construction for CUDA from `sm_XX`.
3. Teach `CudaTileKernelPlan` to store target/lowering config attrs opaquely.
4. Teach schedule selection to prefer IREEGPU workgroup tile sizes when present.
5. Verify existing tests still pass with and without explicit IREEGPU attrs.

The third slice should move conv:

1. Promote `ConvPlan` into the plan layer.
2. Generalize spatial rank in the representation.
3. Keep `generateConv2DKernel` as the rank-2 emitter.
4. Add rank-1 direct conv using the same plan semantics.
5. Add rank-3 correctness support only after rank-1/rank-2 are stable.

This path preserves working code, introduces IREEGPU in a controlled way, and gives contributors a stable place to add support without editing a thousand-line backend decision tree.

## Decision Log: Breadth First, Then Tiled MMA

The current priority is model breadth and a reusable lowering architecture, not exhausting every convolution variant and not immediately chasing peak matmul performance.

The near-term rule is:

```text
configured linalg/tensor dispatch
  -> CudaTileKernelPlan semantic family classification
  -> reusable tile/view/reduction/contraction emitter primitives
  -> existing specialized emitters only when they fit the plan
```

The semantic families we want the plan to expose first are:

- Data movement: copy, slice, insert slice, transpose, broadcast, reshape-like cases.
- Map: elementwise and broadcasted elementwise bodies.
- Reduction: sum, max, min, product, and simple generic reductions.
- Contraction: matmul, batch matmul, vector matmul, and generic contraction indexing maps.
- Windowed reduction: conv and pooling-like indexed sliding-window computations.
- Fused reduction-elementwise: softmax, layer norm, RMS norm, and similar decomposed model patterns.

Convolution remains important, but it should be treated as a `WindowedReduction` family, not as the center of the architecture. We should preserve direct `conv2d` support and avoid materialized im2col as the default path. Wider `conv1d`/`conv3d` support should come from a rank-aware windowed-reduction plan when it blocks model coverage, not from building a separate bespoke conv stack first.

Attention should be supported first as composition of existing families:

```text
QK matmul
  -> scale/map
  -> mask/map
  -> softmax/fused reduction-elementwise
  -> AV matmul
  -> epilogue/map
```

A fused FlashAttention-style kernel is a later performance specialization. It should reuse the same contraction/reduction/map planning concepts instead of introducing unrelated machinery.

The later performance track is simple tiled MMA first:

```text
ContractionPlan
  -> CTA tile of C
  -> K tile loop
  -> cuda_tile.mmaf/mmai where legal
  -> tail-safe loads/stores
  -> optional fused epilogue
```

This track should start simple and correct, then grow explicit schedule ownership:

- `CudaTileKernelPlan` records M/N/K, batch dims, indexing maps, operand layouts, transpose/stride facts, constant operands, and epilogue ops.
- `CudaTileSchedulePlan` records workgroup tile sizes, reduction tile sizes, MMA kind, target attrs, and lowering config attrs.
- Command-line tile options remain the fallback schedule.
- IREEGPU target/lowering/MMA attrs override the fallback when present.
- cuda_tile/tileiras can lower to latest NVIDIA instructions only if the emitted tile IR expresses the right operation and schedule intent; we should not assume it will automatically choose TMA, WGMMA, persistent scheduling, swizzles, or shared-memory promotion for us.

The concrete order after the current plan refactor is:

1. Finish semantic-family classification in `CudaTileKernelPlan`.
2. Move more shape, loop, binding, and fusion facts into the plan.
3. Route existing data movement, map, reduction, contraction, and direct `conv2d` emitters through plan fields without changing generated kernels.
4. Add compile-only plan tests and keep GPU regression as the correctness ratchet.
5. Once breadth is stable, implement the simple tiled-MMA contraction path as the first focused performance project.

## Implementation Track: Plan, MMA, IREEGPU, Async

The next implementation phases should stay layered:

1. Plan ownership: `CudaTileKernelPlan` is the single place that recognizes dispatch semantics, operand roles, binding IDs, tensor/view shapes, loop/reduction dims, fused op order, schedule attrs, and async/TMA hints. `CudaTileTarget.cpp` should become mostly an emitter dispatcher that consumes this plan.
2. Baseline performance path: improve the existing contraction emitter first. The first performance target is a simple CTA-tiled `mmaf`/`mmai` K loop with correct tails, transposed RHS support, batched/flattened M handling, and optional epilogue map fusion. This should not depend on IREEGPU attrs being present.
3. IREEGPU schedule bridge: treat IREEGPU as a planning/config layer, not as a replacement op dialect for cuda_tile lowering. Read generic `lowering_config` through IREE Codegen interfaces and, when it is specifically `#iree_gpu.lowering_config`, preserve `mma_kind`, tiling levels, promotion operands, subgroup/lane basis, and translation workgroup/subgroup sizes in `CudaTileSchedulePlan`.
4. IREEGPU-driven performance: once the baseline contraction path is stable, let IREEGPU schedule metadata override command-line defaults where the meaning is unambiguous. For example, workgroup tile sizes can map to CTA tile M/N/K, `mma_kind` can select tile IR MMA shape/type intent, and promotion attrs can drive shared-memory/cache-swizzle decisions.
5. Async/TMA track: model this as hints in `CudaTileAsyncPlan` first: `allow_tma`, async-copy use, pipeline depth, and latency/optimization hints. Only then lower to cuda_tile optimization hints such as `allow_tma` on `load_view_tko`/`store_view_tko`. Do not mix TMA enablement into generic correctness paths.

The important constraint is that each later phase must be optional. A plain linalg dispatch with no IREEGPU metadata must still lower correctly through cuda_tile. IREEGPU metadata should refine schedule and performance decisions, not become a new correctness dependency.

Current schedule consumption is intentionally conservative:

- The default `--iree-cuda-tile-tile-{m,n,k}` values remain the fallback for every dispatch.
- A contraction may consume `lowering_config.workgroup` as CTA M/N/K only when the output is rank-2, there is exactly one reduction loop, and the output indexing map directly identifies M and N iterator dimensions.
- Non-power-of-two explicit workgroup tile sizes are ignored for now because the current cuda_tile emitter pads tile shapes via power-of-two tile extents.
- `mma_kind` is recorded in the plan but not yet used to select a different cuda_tile MMA op. The current emitter still emits `mmaf`; selecting typed/target-specific MMA shapes needs a separate legality check against cuda_tile/tileiras support.

Current validation is split into compile-time plan inspection plus GPU-backed
correctness harnesses:

- `--iree-cuda-tile-dump-kernel-plan-to=<path-or->` prints the recognized `CudaTileKernelPlan`. `-` writes to stdout; a path appends to a text file. This is the intended way to inspect semantic classification, loop/reduction dimensions, operand bindings, contraction facts, and IREEGPU schedule extraction without local `llvm::errs()` edits.
- `compiler/plugins/target/CudaTile/test/plan_dump.mlir` is the first compile-time plan regression. It checks that a matmul carrying `#iree_gpu.lowering_config<{workgroup = [4, 4, 8]}>` is recognized as a contraction, records the IREEGPU lowering config, and maps workgroup tiles to contraction schedule tiles.
- `compiler/plugins/target/CudaTile/test/codegen_regression.py` is the breadth ratchet. It checks elementwise add, row reduction, plain matmul, and matmul carrying `#iree_gpu.lowering_config`. This is the fast "can we lower simple model building blocks end-to-end?" suite.
- `compiler/plugins/target/CudaTile/test/conv_regression.py` is the convolution boundary ratchet. The current expected boundary is 15 passing cases and 6 XFAIL cases. Passing cases cover 2D pointwise/direct convs with odd dimensions, stride 2/3, batch 2, 1D pointwise stride-1 batch, and 3D pointwise convs. Expected failures still cover non-pointwise 1D convs and non-pointwise 3D convs.

The next validation gap is broader plan coverage. Add plan-dump tests for elementwise maps, reductions, direct `conv2d`, pointwise conv-to-matmul, and unsupported windowed reductions so contributors get fast feedback before running the GPU harnesses.
