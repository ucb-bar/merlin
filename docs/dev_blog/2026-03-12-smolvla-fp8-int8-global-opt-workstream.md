# 2026-03-12: SmolVLA FP8/INT8 Global-Optimization Workstream

## 0. New-User Bootstrap (Verified 2026-03-13)

This section is the end-to-end onboarding path for users starting from a fresh
Merlin checkout.

### 0.1 Clone third-party repos used by SmolVLA export

From repository root:

```bash
mkdir -p third_party
cd third_party

git clone -b mlir-smolvla https://github.com/ucb-bar/Understanding-PI0.git
git clone https://github.com/huggingface/lerobot.git
```

### 0.2 Set up Understanding-PI0 Python environment

```bash
cd third_party/Understanding-PI0
uv python pin 3.12
uv sync --extra export_iree
cd ../..
```

### 0.3 Build Merlin tools with NPU plugin enabled

```bash
conda run -n merlin-dev uv run tools/build.py \
  --profile npu \
  --config release \
  --no-build-python-bindings \
  --no-enable-libbacktrace
```

Sanity check:

```bash
build/host-merlin-release/install/bin/iree-compile --iree-list-plugins
```

Expected plugin list includes `npu`.

### 0.4 Branch linkage for patched third-party trees

For this workstream, behavior depends on patched trees in:

- `third_party/Understanding-PI0`
- `third_party/iree-turbine`
- `third_party/iree_bar`
- `third_party/torch-mlir`

As long as your `third_party/*` checkouts point at the patched branch/commit,
the behavior documented here applies. Use this check:

```bash
for d in third_party/Understanding-PI0 third_party/iree-turbine third_party/iree_bar third_party/torch-mlir; do
  echo "== $d =="
  git -C "$d" rev-parse --abbrev-ref HEAD
  git -C "$d" rev-parse --short HEAD
done
```

When upstreaming, push each repository's branch independently and open separate
PRs per repository.

### 0.5 Export SmolVLA MLIRs through `models/smolVLA`

The new wrapper script interfaces with `third_party/Understanding-PI0` and
produces all three required MLIR variants:

- `models/smolVLA/smolVLA.mlir` (fp32 path)
- `models/smolVLA/smolVLA.q.int8.mlir` (forced int8 path)
- `models/smolVLA/smolVLA.q.fp8.mlir` (mixed fp8/int8 path)

If these files are not present, that means export has not been run yet in this
workspace (they are generated artifacts, not static committed inputs).

Run:

```bash
conda run -n merlin-dev uv run models/smolVLA/export_smolvla.py \
  --mode all \
  --device cuda
```

Quick command preview without execution:

```bash
conda run -n merlin-dev uv run models/smolVLA/export_smolvla.py \
  --mode all \
  --device cuda \
  --dry-run
```

Single-mode exports are also supported:

```bash
conda run -n merlin-dev uv run models/smolVLA/export_smolvla.py --mode fp32 --device cuda
conda run -n merlin-dev uv run models/smolVLA/export_smolvla.py --mode int8 --device cuda
conda run -n merlin-dev uv run models/smolVLA/export_smolvla.py --mode fp8 --device cuda
```

### 0.6 Direct Understanding-PI0 workflow (optional)

If you want the original direct flow inside `third_party/Understanding-PI0`,
use:

```bash
cd third_party/Understanding-PI0

uv run python scripts/quantizing_smolvla/inspect_fqns.py \
  --model-id lerobot/smolvla_base \
  --device cuda

uv run python scripts/quantizing_smolvla/quantize_mx.py \
  --model-id lerobot/smolvla_base \
  --device cuda \
  --out reports/smolvla_mx/smolvla_mx_quantized.pt \
  --report-json reports/smolvla_mx/quant_report.json

uv run python scripts/quantizing_smolvla/validate_one_step.py \
  --model-id lerobot/smolvla_base \
  --device cuda

uv run python scripts/export_iree.py \
  --model-id lerobot/smolvla_base \
  --device cuda \
  --print-readable \
  --out reports/smolvla_mx/smolvla_one_step.mlir
```

### 0.7 Minimum Reproducible Flow

Use this exact sequence from repo root:

```bash
# 1) Export mixed fp8/int8 SmolVLA MLIR
conda run -n merlin-dev uv run models/smolVLA/export_smolvla.py --mode fp8 --device cuda

# 2) Compile to global-opt with NPU target (short command)
conda run -n merlin-dev uv run tools/compile.py \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target npu_ucb \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

Core output directory from step 2:

```text
build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases/
```

Required files to inspect every run:

- `module.1.input.mlir`
- `module.4.global-optimization.mlir`

Then run NPU text ISA + simulator:

```bash
BIN=build/host-merlin-release/tools
DUMP=build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases

"$BIN/iree-opt" "$DUMP/module.4.global-optimization.mlir" \
  --iree-plugin=npu \
  --mlir-print-op-generic \
  > "$DUMP/module.4.global-optimization.generic.mlir"

"$BIN/npu-translate" \
  --allow-unregistered-dialect \
  --mlir-to-npu-text-isa \
  "$DUMP/module.4.global-optimization.generic.mlir" \
  > "$DUMP/smolvla.program.isa.txt"

conda run -n merlin-dev uv run python \
  compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py \
  "$DUMP/smolvla.program.isa.txt"

conda run -n merlin-dev uv run python \
  third_party/npu_model/compiler/scripts/run_simulator_smoke.py \
  "$DUMP/smolvla.program.isa.txt"
```

Where generated files live:

- exported MLIRs: `models/smolVLA/` (generated, gitignored)
- compiled artifacts/phases: `build/compiled_models/...` (build outputs, not committed)

## 1. Context and Goal

Goal for this workstream:

- compile the real SmolVLA graph (not a PoC) to `--compile-to=global-optimization`
- inspect both dumps every run:
  - `module.1.input.mlir`
  - `module.4.global-optimization.mlir`
- push computation toward low precision (`fp8`/`int8`) and avoid unnecessary QDQ-style expansion in runtime regions

Primary compile flow used for the latest rerun:

```bash
conda run -n merlin-dev uv run tools/compile.py \
  third_party/Understanding-PI0/reports/smolvla_mx/smolvla_one_step.mlir \
  --target spacemit_x60 \
  --quantized \
  --compile-to global-optimization \
  --dump-compilation-phases-to tmp/smolvla_global_opt_phases_quantfix12_softmax_fastpath_rerun \
  --iree-compile-arg=--iree-input-type=torch \
  --iree-compile-arg=--iree-opt-const-eval=false \
  --iree-compile-arg=--iree-global-opt-enable-demote-contraction-inputs-to-bf16=matmul
```

## 2. Why Vanilla IREE Was Not Enough For This Case

Vanilla IREE is semantically correct, but this exported graph already contains explicit decode/dequant math for MX payloads (`ui8 -> i32 -> bitcast f32 -> clamp/where -> bf16`).

That means global optimization can hoist and fuse parts, but it will not "re-infer" native FP8 attention kernels from an already expanded decode + mixed-precision graph.

<details>
<summary>Original exported graph snippet (decode path)</summary>

```mlir
%124 = torch.prims.convert_element_type %123, %int3_215
  : !torch.vtensor<[768,24],ui8>, !torch.int -> !torch.vtensor<[768,24],si32>
%125 = torch.operator "torch.aten.bitwise_left_shift.Tensor_Scalar"(%124, %int23)
  : (!torch.vtensor<[768,24],si32>, !torch.int) -> !torch.vtensor<[768,24],si32>
%126 = torch.aten.view.dtype %125, %int6_216
  : !torch.vtensor<[768,24],si32>, !torch.int -> !torch.vtensor<[768,24],f32>
%128 = torch.aten.clamp %127, %float1.175490e-38, %none_217
  : !torch.vtensor<[768,24],f32>, !torch.float, !torch.none -> !torch.vtensor<[768,24],f32>
%131 = torch.aten.where.self %129, %128, %130
  : !torch.vtensor<[768,24],i1>, !torch.vtensor<[768,24],f32>, !torch.vtensor<[768,24],f32>
    -> !torch.vtensor<[768,24],f32>
%132 = torch.prims.convert_element_type %131, %int15_228
  : !torch.vtensor<[768,24],f32>, !torch.int -> !torch.vtensor<[768,24],bf16>
```

</details>

## 3. Compiler Modifications

### 3.1 Preserve attention region compute type

File:

- `third_party/iree_bar/compiler/plugins/input/Torch/InputConversion/ConvertTMTensorToLinalgExt.cpp`

Change:

- removed forced `f32` region argument for `iree_linalg_ext.attention`
- region arg now uses frontend-derived `targetType`

```c++
// Preserve the frontend attention compute type instead of forcing f32.
block->addArgument(targetType, loc);
```

### 3.2 Keep attention decomposition in region score type

File:

- `third_party/iree_bar/compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

Change:

- removed hardcoded `f32` in attention decomposition
- uses `scoreType` from attention region argument

```c++
Type scoreType = getRegion().front().getArgument(0).getType();
Value s = computeQKAndElementwise(..., scoreType, ...);
```

### 3.3 Keep online-attention conversion in region score type

File:

- `third_party/iree_bar/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/TileAttention.cpp`

Change:

- removed hardcoded `f32` temp buffers/inits
- uses attention region score type end-to-end in this conversion

### 3.4 Torch softmax low-precision folding/decomposition

Files (both trees patched):

- `third_party/iree_bar/third_party/torch-mlir/lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp`
- `third_party/torch-mlir/lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp`

Changes:

- added `FoldSoftmaxIntConvertElementTypeOp`
- `AtenSoftmaxInt` fast-path: decompose directly in bf16/f16 when softmax is immediately cast down
- for bf16/f16 result types, force softmax accumulator type to bf16/f16

Result:

- `global linalg.softmax` moved from f32 to bf16 for the 23 softmax islands

### 3.5 Added quant IR analyzer

File:

- `tools/analyze_quant_ir.py`

Capabilities:

- compares `module.1.input.mlir` and `module.4.global-optimization.mlir`
- typed softmax counters (`linalg.softmax f32` vs `linalg.softmax bf16`)
- matmul signature counters and enforcement thresholds

## 4. Stage-by-Stage Results

Metrics from `tools/analyze_quant_ir.py`.

| Stage | input `attention region f32` | global `attention region f32` | global `linalg.softmax f32` | global `linalg.softmax bf16` | global `batch_matmul f32*bf16->f32` | global `batch_matmul bf16*bf16->f32` | global `truncf f32->bf16 outside initializer` |
|---|---:|---:|---:|---:|---:|---:|---:|
| quantfix3 | 36 | 36 | 23 | 0 | 23 | 0 | 213 |
| quantfix6_attn_online_dtype | 0 | 0 | 23 | 0 | 23 | 0 | 213 |
| quantfix8_demote_contract_bf16 | 0 | 0 | 23 | 0 | 0 | 23 | 259 |
| quantfix11_softmax_fastpath | 0 | 0 | 0 | 23 | 0 | 23 | 259 |
| quantfix12_softmax_fastpath_rerun | 0 | 0 | 0 | 23 | 0 | 23 | 259 |

Interpretation:

- Attention region type forcing bug is fixed.
- Softmax precision blocker is improved: softmax now bf16 in global-opt.
- Remaining primary blocker is the score path matmul accumulation pattern:
  `linalg.batch_matmul bf16*bf16 -> f32` (23 instances).

## 5. Expandable MLIR Evidence

<details>
<summary>Input (`quantfix12`) still has f32 score matmul islands</summary>

```mlir
%2462 = linalg.batch_matmul
  ins(%collapsed_1831, %collapsed_1832
      : tensor<15x291x64xf32>, tensor<15x64x291xf32>)
  outs(%2461 : tensor<15x291x291xf32>) -> tensor<15x291x291xf32>

%2467 = linalg.generic ... ins(%2465 : tensor<1x15x291x291xf32>)
  outs(%2466 : tensor<1x15x291x291xbf16>) {
^bb0(%in: f32, %out: bf16):
  %5333 = arith.truncf %in : f32 to bf16
  linalg.yield %5333 : bf16
}
```

</details>

<details>
<summary>Global-opt (`quantfix12`) now has bf16 softmax but score matmul still f32 output</summary>

```mlir
%1827 = linalg.batch_matmul
  ins(%collapsed_1078, %1826
      : tensor<15x291x64xbf16>, tensor<15x64x291xbf16>)
  outs(%1814 : tensor<15x291x291xf32>) -> tensor<15x291x291xf32>

%1832 = linalg.softmax dimension(2)
  ins(%1831 : tensor<15x291x291xbf16>)
  outs(%1830 : tensor<15x291x291xbf16>) -> tensor<15x291x291xbf16>
```

</details>

<details>
<summary>Input (`quantfix12`) attention op region is bf16 (fixed)</summary>

```mlir
%129 = iree_linalg_ext.attention ...
  ins(... : tensor<12x1024x64xbf16>, tensor<12x1024x64xbf16>,
            tensor<12x1024x64xbf16>, bf16, tensor<12x1024x1024xi1>)
  outs(%128 : tensor<12x1024x64xbf16>) {
^bb0(%arg13: bf16):
  iree_linalg_ext.yield %arg13 : bf16
} -> tensor<12x1024x64xbf16>
```

</details>

## 6. How IREE Handles FP8 vs Attention in This Tree

### 6.1 What exists already

- FP8 arithmetic coverage exists in e2e tests:
  - `third_party/iree_bar/tests/e2e/linalg/small_float_arith.mlir`
- ROCm FP8 matmul ukernel path exists:
  - `third_party/iree_bar/compiler/plugins/target/ROCM/builtins/mlir_ukernel/iree_uk_amdgpu_matmul_f8E4M3FN.mlir`
- Global-opt can keep many model matmuls in mixed low precision with explicit `f8E4M3FN -> bf16` scaling semantics.

### 6.2 What is still f32-centric

- Attention and softmax e2e references are mainly f32-typed:
  - `third_party/iree_bar/tests/e2e/linalg_ext_ops/attention.mlir`
  - `third_party/iree_bar/tests/e2e/linalg_ext_ops/attention_i1_mask.mlir`
  - `third_party/iree_bar/tests/e2e/linalg/softmax.mlir`
- LLVMGPU attention scheduling code currently models attention matmuls with f32 `c` type assumptions in schedule construction (`KernelConfig.cpp`).

Practical consequence for our pipeline:

- IREE does have FP8 matmul support.
- It does not automatically turn a decomposed score/softmax/value path into fully FP8 attention math when the frontend graph already carries mixed/f32 score semantics.

## 7. Current Status for NPU/Gemmini Matching

Implemented and validated now:

- NPU conversion pass now rewrites:
  - `linalg.batch_matmul` -> `npu_kernel.matmul`
  - `linalg.softmax` -> `npu_schedule.softmax_fragment`
  - `iree_linalg_ext.attention` -> `npu_kernel.ukernel_generic` (`npu_uk_gemma_attention_*`)
- NPU verifier now accepts batched matmul-like ranks and has explicit attention-family checks.
- Gemmini conversion now handles named `linalg.matmul` in addition to existing generic recovery.
- Gemmini plugin include/wiring blocker fixed (`GemminiOptions.cpp` include path).

## 8. Real-Model Coverage Check (Global-Opt Dump)

Validation was run on:

- `tmp/smolvla_global_opt_phases_quantfix12_softmax_fastpath_rerun/module.4.global-optimization.mlir`

### 8.1 NPU conversion coverage

Before `convert-linalg-to-npu-kernel`:

- `linalg.batch_matmul`: 46
- `linalg.softmax`: 23
- `iree_linalg_ext.attention`: 36

After `convert-linalg-to-npu-kernel`:

- `linalg.batch_matmul`: 0
- `linalg.softmax`: 0
- `iree_linalg_ext.attention`: 0
- `npu_kernel.matmul`: 113
- `npu_kernel.ukernel_generic`: 36
- `npu_schedule.softmax_fragment`: 23

After full NPU pipeline (`... -> convert-npu-schedule-to-isa`):

- `npu_kernel.*`: 0
- `npu_schedule.*`: 0
- `npu_isa.matmul_mxu0`: 185
- `npu_isa.vexp`: 59
- `npu_isa.vmul`: 118

<details>
<summary>NPU-lowered snippet (real global-opt file)</summary>

```mlir
%0 = npu_kernel.ukernel_generic "npu_uk_gemma_attention_bf16_bf16"(...)
...
%1 = npu_schedule.ukernel_launch "npu_uk_gemma_attention_bf16_bf16"(...)
...
%2 = npu_isa.matmul_mxu0 ...
```

</details>



## 9. Remaining Precision Blocker

We have removed the worst QDQ runtime behavior (bitcast/trunc now hoisted to initializers in this run), and softmax is bf16 in global-opt.

Remaining blocker before "near-zero fp32":

- 23 score-path `linalg.batch_matmul` islands in global-opt remain `bf16*bf16 -> f32`.
- In NPU-kernel-lowered form this appears as 24 `npu_kernel.matmul bf16*bf16->f32` (one additional non-batch contraction path).

## 10. Next Minimal Steps (Not Overdoing)

1. Keep current matcher path as baseline: it is now complete for both NPU and Gemmini on real global-opt forms.
2. If we decide to reduce fp32 further, add one targeted rewrite for score-matmul accumulators only (instead of broad precision changes).
3. Keep `tools/analyze_quant_ir.py` as a gate on `module.1.input.mlir` and `module.4.global-optimization.mlir`.

## 11. 2026-03-13 Continuation

### 11.1 Quantfix3 input/global-opt analysis (the two files we now always check)

Files:

- `tmp/smolvla_global_opt_phases_quantfix3/module.1.input.mlir`
- `tmp/smolvla_global_opt_phases_quantfix3/module.4.global-optimization.mlir`

Current counts in these dumps:

- input `linalg.batch_matmul ... -> tensor<...xf32>`: `23`
- global-opt `linalg.batch_matmul ... -> tensor<...xf32>`: `23`
- global-opt `linalg.softmax ... -> tensor<...xf32>`: `23`

<details>
<summary>Global-opt f32 island example (`quantfix3`)</summary>

```mlir
%1820 = linalg.fill ins(%cst_16 : f32) outs(%1819 : tensor<15x291x291xf32>) -> tensor<15x291x291xf32>
%1821 = linalg.batch_matmul ins(%collapsed_1078, %collapsed_1079
  : tensor<15x291x64xf32>, tensor<15x64x291xbf16>)
  outs(%1820 : tensor<15x291x291xf32>) -> tensor<15x291x291xf32>
```

</details>

### 11.2 NPU island handling (implemented)

File updated:

- `compiler/src/merlin/Dialect/NPU/Transforms/ConvertLinalgToNPUKernel.cpp`

What changed:

- generalized demotion rewrite from only `bf16 x bf16 -> f32` to float score paths including `f32 x bf16 -> f32`
- inserted explicit tensor elementwise casts:
  - cast float operands to `bf16` when needed (`truncf`)
  - run `npu_kernel.matmul` in `bf16`
  - widen back to `f32` with `extf` only where IR still expects `f32`

Result on real `quantfix3` global-opt after `convert-linalg-to-npu-kernel`:

- `npu_kernel.matmul ... -> tensor<...xf32>`: `0`
- `npu_kernel.matmul ... -> tensor<...xbf16>`: `47`
- `linalg.batch_matmul ... -> tensor<...xf32>`: `0`

<details>
<summary>NPU-demoted snippet (`quantfix3` global-opt lowered form)</summary>

```mlir
%1688 = npu_kernel.matmul %expanded, %__hoisted_tensor_960x32xbf16
  : tensor<1x32xbf16>, tensor<960x32xbf16> -> tensor<1x960xbf16>
%1689 = linalg.generic ... ins(%1688 : tensor<1x960xbf16>)
  outs(%1687 : tensor<1x960xf32>) {
^bb0(%in: bf16, %out: f32):
  %3561 = arith.extf %in : bf16 to f32
  linalg.yield %3561 : f32
}
```

</details>

Full NPU lowering still completes on the same file:

- `npu_kernel.*`: `0` after final pass
- `npu_schedule.*`: `0` after final pass
- `npu_isa.matmul_mxu0`: `185`
- `npu_isa.vexp`: `59`

### 11.3 NPU model execution validation (implemented)

Scripts validated:

- `compiler/src/merlin/Dialect/NPU/scripts/run_npu_model_smoke.sh`
- `compiler/src/merlin/Dialect/NPU/scripts/run_frontend_to_npu_model_e2e.sh`

Validated symbols/flows:

- `npu_uk_matmul_bf16_bf16_bf16`
- `npu_uk_gemma_attention_bf16_bf16`
- frontend hook e2e from `post-global-opt-hook.mlir`

Support updates made for this:

- `compiler/src/merlin/Dialect/NPU/scripts/emit_symbol_ukernel_isa.sh`
  - parses symbol-suffixed types for bf16/fp8/f32/i8/i32 families
- `compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py`
  - handles current `third_party/npu_model/npu_model/configs/isa_definition.py` path

### 11.4 Gemmini FP8 route (implemented)

Files updated:

- `compiler/plugins/target/Gemmini/GemminiOptions.h`
- `compiler/plugins/target/Gemmini/GemminiOptions.cpp`
- `compiler/plugins/target/Gemmini/PluginRegistration.cpp`
- `compiler/src/merlin/Dialect/Gemmini/Transforms/Passes.h`
- `compiler/src/merlin/Dialect/Gemmini/Transforms/ConvertToGemmini.cpp`
- `compiler/src/merlin/Dialect/Gemmini/IR/GemminiOps.cpp`
- `compiler/src/merlin/Dialect/Gemmini/Transforms/LowerGemminiToIREE.cpp`

New option:

- `--iree-gemmini-enable-fp8-matmul`

Behavior:

- recovers FP8 matmul forms (`f8E4M3FN/f8E4M3FN -> bf16|f32`) into `gemmini.matmul`
- keeps default behavior conservative (FP8 recovery off unless flag is enabled)
- lower-to-ISA and lower-back-to-IREE paths now handle FP8 matmul variants

Real-model (`quantfix3` global-opt) `util.func` coverage with current default path:

- `gemmini.matmul_tile`: `66`
- `gemmini.matmul`: `0` (all staged)
- remaining `linalg.matmul`: `1`

Tests added/updated:

- `compiler/src/merlin/Dialect/Gemmini/Transforms/tests/post-global-opt-hook.mlir`
  - includes FP8 function and checks with `--iree-gemmini-enable-fp8-matmul`
- `compiler/src/merlin/Dialect/Gemmini/Transforms/tests/matmul-lower-to-isa.mlir`
  - FP8 `gemmini.matmul -> gemmini.matmul_tile`
- `compiler/src/merlin/Dialect/Gemmini/Transforms/tests/lower-gemmini-to-iree.mlir`
  - FP8 tile lowering includes `arith.extf` path

## 12. Attention Flags Experiment (`2026-03-13`)

Tested IREE flags suggested for attention:

- `--iree-global-opt-enable-attention-v-transpose`
- `--iree-linalgext-attention-softmax-max=65504`

Compile commands were run both with and without those flags using:

- `tools/compile.py ... --compile-to global-optimization`
- phase dumps:
  - `tmp/smolvla_global_opt_phases_attnflags0` (without)
  - `tmp/smolvla_global_opt_phases_attnflags1` (with)

Result:

- `module.1.input.mlir` hashes are identical
- `module.4.global-optimization.mlir` hashes are identical
- key metrics are unchanged (`23` f32 score matmul islands in global-opt)

Conclusion for current tree + model path:

- these two flags are accepted by the compiler but are currently a no-op for this specific SmolVLA input/lowering path.

## 13. Reproducible Export -> Global-Opt -> NPU ISA -> `npu_model`

This is the concrete path from exported SmolVLA MLIR to NPU simulator input.

### 13.1 Compile exported SmolVLA to global-opt with phase dumps

```bash
# Baseline target bundle
conda run -n merlin-dev uv run tools/compile.py \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target spacemit_x60 \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

```bash
# NPU plugin path (from models/npu_ucb.yaml)
conda run -n merlin-dev uv run tools/compile.py \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target npu_ucb \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

```bash
# Gemmini plugin path (from models/gemmini_mx.yaml)
conda run -n merlin-dev uv run tools/compile.py \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target gemmini_mx \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

SmolVLA-specific compile knobs (`--iree-input-type=torch`,
`--iree-opt-const-eval=false`, demote-contraction flag) are now encoded in
`models/*.yaml` under `models: smolVLA`, so users do not pass those manually.

### 13.2 Always inspect the two key phase outputs

```bash
# Example for the npu_ucb fp8 build output directory:
ls -lh \
  build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases/module.1.input.mlir \
  build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases/module.4.global-optimization.mlir

conda run -n merlin-dev uv run python tools/analyze_quant_ir.py \
  build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases
```

The phase dumps are intentionally under `build/compiled_models/...` so they are
treated as generated build artifacts, not source-controlled inputs.

These are the two files we now treat as the ground truth for matching and
lowering:

- `module.1.input.mlir`
- `module.4.global-optimization.mlir`

### 13.3 Lower global-opt IR to NPU dialect/ISA

```bash
BIN=build/host-merlin-release/tools
DUMP=build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases

"${BIN}/iree-opt" "${DUMP}/module.4.global-optimization.mlir" \
  --iree-plugin=npu \
  --mlir-print-op-generic \
  --pass-pipeline='builtin.module(convert-linalg-to-npu-kernel,npu-verify-ukernel-symbols,convert-npu-kernel-to-schedule,npu-verify-ukernel-symbols,convert-npu-schedule-to-isa)' \
  > "${DUMP}/module.4.global-optimization.npu-isa.mlir"
```

Note: `--mlir-print-op-generic` is required here so non-NPU attrs/ops from
global-opt remain parseable by `npu-translate`.

### 13.4 Translate NPU ISA MLIR to text ISA

```bash
"${BIN}/npu-translate" \
  --allow-unregistered-dialect \
  --mlir-to-npu-text-isa \
  "${DUMP}/module.4.global-optimization.npu-isa.mlir" \
  > "${DUMP}/smolvla.program.isa.txt"
```

### 13.5 Validate ISA contract and run `npu_model` smoke

```bash
conda run -n merlin-dev uv run python \
  compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py \
  "${DUMP}/smolvla.program.isa.txt"

conda run -n merlin-dev uv run python \
  third_party/npu_model/compiler/scripts/run_simulator_smoke.py \
  "${DUMP}/smolvla.program.isa.txt"
```

The above contract-check + smoke path was validated using:

- real compile dump: `tmp/smolvla_global_opt_phases_verify_npu_ucb_real`
- ISA MLIR: `tmp/smolvla_global_opt_phases_quantfix12_softmax_fastpath_rerun/module.4.global-optimization.npu-isa.blog-generic.mlir`
- ISA text: `tmp/smolvla_global_opt_phases_quantfix12_softmax_fastpath_rerun/smolvla.blog-generic.isa.txt`


### 13.7 Quick matcher checks (exact commands)

```bash
rg -o "gemmini\\.[a-zA-Z_]+" -N \
  tmp/smolvla_global_opt_phases_verify_gemmini_mx_real2/module.4.global-optimization.mlir \
  | sort | uniq -c
```

```bash
rg -o "npu_isa\\.[a-zA-Z_]+" -N \
  tmp/smolvla_global_opt_phases_verify_npu_ucb_real/module.4.global-optimization.mlir \
  | sort | uniq -c
```
