# 2026-03-12: SmolVLA FP8/INT8 Global-Optimization Workstream

## 0. New-User Bootstrap (Verified 2026-03-13)

This section is the end-to-end onboarding path for users starting from a fresh
Merlin checkout.

Before running the steps below on a fresh Merlin clone, check out the Merlin
branch you want first and then sync the pinned submodules for that branch:

```bash
git checkout dev/main
conda activate merlin-dev
uv run tools/merlin.py setup submodules --submodules-profile core --submodule-sync
```

`tools/merlin.py setup submodules` follows the currently checked out Merlin
commit. If you switch branches later, rerun it before rebuilding or compiling.

### 0.1 Clone third-party repos used by SmolVLA export

From repository root:

```bash
git clone -b mlir-smolvla \
  https://github.com/ucb-bar/Understanding-PI0.git \
  third_party/Understanding-PI0

git clone https://github.com/huggingface/lerobot.git \
  third_party/lerobot
```

Both clones are required: `Understanding-PI0/pyproject.toml` declares
`lerobot` as an editable path dependency (`../lerobot`), so `lerobot`
must be present before running `uv sync` in the next step.

### 0.2 Set up Understanding-PI0 Python environment

From repository root:

```bash
cd third_party/Understanding-PI0
uv python pin 3.12
uv sync --extra export_iree
cd ../..
```

> **Note — `evdev` is excluded by default.** The `lerobot → pynput →
> evdev` chain is only needed for physical input-device handling, not
> MLIR export. `evdev` fails to build when the conda cross-compiler's
> sysroot headers are older than the system kernel headers (the build
> reads `/usr/include/linux/input-event-codes.h` to generate code, then
> compiles against the conda sysroot which lacks newer constants like
> `KEY_ACCESSIBILITY`). The `pyproject.toml` overrides evdev's marker so
> it is never resolved.

### 0.3 Build Merlin tools with NPU plugin enabled

```bash
conda run -n merlin-dev uv run tools/merlin.py build \
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

Important:

- the Torch-input legalization used by the compile flow also depends on
  `third_party/iree_bar/third_party/torch-mlir` as checked out by
  `third_party/iree_bar`
- after syncing any of the compiler-side trees above, rebuild
  `build/host-merlin-release/` before retrying the SmolVLA compile flow

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
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target npu_ucb \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

If step 2 fails with:

```text
error: failed to legalize operation 'torch.operator' that was explicitly marked illegal
... torch.prims.device_put ...
```

that is a compiler-stack mismatch, not an export-format requirement.
The CUDA export is valid for this flow. The failing compiler build is missing the
custom Torch-input lowering that forwards `torch.prims.device_put` during
Torch-to-IREE legalization.

Use this quick check before retrying:

```bash
rg -n "torch.prims.device_put|createConvertCustomQuantOpPass" \
  third_party/iree_bar/third_party/torch-mlir/lib/Dialect/TorchConversion/Transforms/ConvertCustomQuantOp.cpp \
  third_party/iree_bar/compiler/plugins/input/Torch/InputConversion/Passes.cpp
```

Then rebuild the compiler and rerun the compile step:

```bash
conda run -n merlin-dev uv run tools/merlin.py setup submodules \
  --submodules-profile core \
  --submodule-sync

conda run -n merlin-dev uv run tools/merlin.py build \
  --profile npu \
  --config release \
  --no-build-python-bindings \
  --no-enable-libbacktrace
```

Exporting with `--device cpu` may avoid the `device_put` op, but that should be
treated as a temporary workaround for an out-of-date compiler tree, not the
expected setup for this workstream.

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

---

## 14. AutoComp Kernel Library (2026-04-11)

Per-tile NPU kernels generated by AutoComp (`third_party/autocomp`) with
LLM-driven beam search and cycle-accurate feedback from `npu_model`.

### 14.1 Results

| Kernel | Cycles | Method |
|---|---|---|
| matmul (fp8×fp8→bf16) | 198 | autocomp |
| silu | 178 | autocomp |
| elementwise_add | 168 | autocomp |
| elementwise_mul | 200 | autocomp |
| elementwise_div | **220** | hand-written replacement |
| elementwise_sub | 199 | autocomp |
| rope_frequency | 137 | autocomp |
| gelu_tanh | **327** | hand-written replacement |
| softmax | **196** | hand-written |
| rms_norm | **258** | hand-written |
| reduction_sum | **110** | hand-written |
| requant (bf16→fp8) | **107** | hand-written |
| attention (SDPA) | **463** | hand-written materialized QK/softmax/V tile |
| tiled_matmul_2x1 | **297** | autocomp (2 tiles, ~148 cy/tile amortized) |
| fast_gelu (preloaded consts) | 356 | autocomp (slower than standalone 337 cy; constant-DMA cost outweighed savings) |

Kernel sources: `benchmarks/SaturnNPU/kernel_library/*.py`. Each file exports
`test() -> List[Instruction]`. The executable manifest is
`benchmarks/SaturnNPU/kernel_library/manifest.json` and is copied into
`third_party/npu_model/npu_model/configs/programs/smolvla_kernel_manifest.json`
for representative-program pushes.

### 14.2 Why kernels were hand-written or replaced

Several kernels (softmax, rms_norm, reduction_sum, requant, attention, and
later the replacement div/GELU variants) were faster to write directly than
to keep driving autocomp search. Root-causing the failures revealed issues
that are **structural**, not search-quality problems:

- **Hardware layout convention.** `vpack.bf16.fp8` consumes two bf16 inputs
  as left/right column halves (`cat([m_lo, m_hi], dim=1)`) — the same layout
  MXU produces natively. A row-major 32×32 bf16 tile in DRAM cannot be
  `vload`ed into the right shape with a single DMA. The harness fix: present
  inputs already split at the DRAM level, matching production flow.
- **Default ERF = 0.** `vpack`/`vunpack` multiply by an ERF scale; without
  an explicit `seli rd=k, imm=1` the output is zeroed. Autocomp missed this
  hint across 8+ iterations.
- **Golden precision.** The simulator's reductions promote to fp32 then cast
  to bf16. Goldens computed entirely in fp32 produced spurious mismatches at
  the ~0.03 level. Compute goldens stepwise in bf16 to match hardware.
- **DMA flag discipline.** `dma.config` and `dma.load` on the same channel
  without a `dma.wait` between them trip an "Flag N already set" assertion.
- **`addi` 12-bit signed immediate.** Addresses ≥ 2048 need `lui + addi`
  pair or chained `addi`s through a base register. Silently wrapping to
  negative is a subtle bug.

### 14.3 npu_model simulator bugs fixed

Two latent bugs in `third_party/npu_model` were silently breaking every
autocomp run that exercised `vpack.bf16.fp8`:

1. **`vpack.bf16.fp8` missing `.view(torch.uint8)`**
   (`configs/isa_definition.py:556`). Passed a fp8 tensor to `write_mrf_fp8`
   which asserts `uint8`. The companion `vmatpop_fp8_acc_mxu1` does the view
   correctly at line 944.
2. **`write_mrf_fp8` value-casts instead of byte-copying**
   (`hardware/arch_state.py:131`). Did
   `self.mrf[vd].view(float8_e4m3fn)[:] = u8_value`, which makes PyTorch
   **value-cast** uint8 byte 173 → fp8 +176.0 (stored as byte 115) instead
   of reinterpreting the bits. Fix: view target as `uint8`.

Both patches are in the local `npu_model` submodule and are prerequisites
for requant / anything downstream of MXU→fp8.

### 14.4 Harness conventions (production-aligned)

To match MXU's natural layout, kernel harnesses present bf16 tiles
**pre-split into 32×16 left / right halves**:

```
DRAM[0x0000..0x0400] — left  half (32, 16) bf16
DRAM[0x0400..0x0800] — right half (32, 16) bf16
```

Reductions (row-sum, row-max, rms) produce a **broadcast output**:
`(32, 16)` bf16 where each row is the per-row scalar repeated 16 times.
This is exactly what downstream broadcast ops consume — no reshape needed
between kernels.

Constants (`1/N`, `eps`, GELU coefficients) are **DMA-preloaded** into DRAM
as filled 32×16 bf16 tiles. This avoids scalar-store constant construction
that originally made gelu 796 cy; the preloaded-constant variant landed at
337 cy.

### 14.5 Reusable kernel template

The hand-written VPU kernels follow the same 30–80-line template:

```
1.  Scalar address setup (x1..x11): VMEM in/out, DRAM in/out, size
2.  dma.config ch0 / ch1  →  dma.wait ch0 / ch1    (clear flags)
3.  dma.load halves + constants in parallel on ch0 / ch1
4.  dma.wait ch0 / ch1
5.  vload into m0, m1 (+ m2, m3 for constants)
6.  VPU compute on bf16, pairwise over halves
7.  (If reducing) vredsum.row / vredmax.row per half, then combine via vadd / vmaximum
8.  (If requanting) seli ERF=1; vpack.bf16.fp8
9.  vstore + dma.store ch0 / ch1
10. dma.wait ch0 / ch1
```

Cycles = sum of functional-unit latencies + DMA transfer time (1024 B at
64 B/cycle = 16 cycles per DMA). The VPU's non-pipelineable ops (`vexp`,
`vrecip`, `vsqrt`, `vtanh`) are 8-cycle; simple ops (`vadd`, `vmul`,
`vsub`) are 2-cycle; `delay imm=N` stalls for N cycles to respect
data-hazard windows (the NPU is in-order with no dependency tracking).

### 14.6 Are we ready to stitch?

**Yes, for all 13 kernel families.** The missing elementwise-div, GELU tanh,
and attention families now have handwritten kernels checked against
`npu_model`. The first stitching bridge is implemented as a manifest-backed
native ISA path:

- `benchmarks/SaturnNPU/kernel_library/stitch.py` concatenates kernel
  invocations and rewrites only the DRAM address registers used by
  `dma.load.ch<N>` / `dma.store.ch<N>`, preserving per-kernel VMEM scratch.
- `compiler/src/merlin/Dialect/NPU/scripts/emit_kernel_manifest_isa.py`
  emits native `npu_model` text ISA from a kernel sequence or the full
  manifest.
- `compiler/src/merlin/Dialect/NPU/scripts/run_kernel_manifest_smoke.py`
  executes a stitched manifest program directly in `npu_model`.
- `npu_isa.native_inst` carries manifest-native instructions through MLIR.
  `convert-npu-schedule-to-isa` can now lower scheduled matmul, softmax, and
  ukernel launches by looking them up in the manifest instead of expanding the
  older abstract `npu_isa` sequence.

Validation on 2026-04-12:

```
uv run python compiler/src/merlin/Dialect/NPU/scripts/emit_kernel_manifest_isa.py \
  --all --connect-linear --output /tmp/smolvla_manifest_all.isa.txt
uv run python compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py \
  /tmp/smolvla_manifest_all.isa.txt
# ISA contract check PASSED: 610 instruction(s), 28 mnemonic(s)

uv run python compiler/src/merlin/Dialect/NPU/scripts/run_kernel_manifest_smoke.py \
  --all --connect-linear --max-cycles 50000
# stitched manifest smoke PASSED: 13 kernel(s), 610 instruction(s)

uv run tools/merlin.py build --profile npu --cmake-target iree-opt
printf "<npu_schedule.matmul_tile module>" | \
  build/host-merlin-debug/tools/iree-opt --iree-plugin=npu \
    --iree-npu-native-kernel-lowering \
    --iree-npu-kernel-manifest=benchmarks/SaturnNPU/kernel_library/manifest.json \
    --pass-pipeline="builtin.module(convert-npu-schedule-to-isa)" - | \
  build/host-merlin-debug/tools/npu-translate --mlir-to-npu-text-isa
# emits manifest-native matmul instructions
```

The demoted SmolVLA artifact was checked with both textual scans and MLIR
Python bindings. After stripping the 845 MB dense-resource payload block for
parsing, the MLIR binding walk sees **0 f32 compute ops**; the only f32 mentions
in the raw file are three ABI imports immediately converted to bf16. Dense
resources are also recoverable for the actual model weights: 496 references,
494 unique payloads, all with byte counts matching the tensor type after the
MLIR four-byte blob prefix.

Remaining work before numerical E2E parity is graph binding and allocation, not
kernel availability: lower every real SmolVLA linalg/schedule op into one of
the manifest families, allocate DRAM regions for activations/constants/weights,
rewrite per-invocation addresses, and compare the resulting full program in
`npu_model` against the PyTorch reference.

Nice-to-have, non-blocking:

- **DMA cross-kernel pipelining** (prob 30 `tiled_matmul_2x1`) — halves
  DRAM-stall cycles by overlapping kernel N+1's weight load with kernel N's
  compute.
- **Fast gelu** (prob 31) — skip for now: it landed at 356 cycles, slower than
  the current 337-cycle GELU tanh kernel.

---

## 15. Graph binding: linalg.generic → NPU kernels (2026-04-12)

Extended `ConvertLinalgToNPUKernel.cpp` with 5 new pattern matchers that
recognize elementwise and reduction `linalg.generic` ops produced by the
global-opt phase of the SmolVLA pipeline.

**New patterns (file `compiler/src/merlin/Dialect/NPU/Transforms/ConvertLinalgToNPUKernel.cpp`):**

| Pattern | Body shape | Bound kernel |
|---|---|---|
| `LowerElementwiseGenericToUKernelPattern("arith.addf")` | single `addf`, all-parallel iterators, identity input maps | `npu_uk_elementwise_add` |
| `LowerElementwiseGenericToUKernelPattern("arith.mulf")` | single `mulf`, all-parallel | `npu_uk_elementwise_mul` |
| `LowerElementwiseGenericToUKernelPattern("arith.subf")` | single `subf`, all-parallel, preserves operand order | `npu_uk_elementwise_sub` |
| `LowerElementwiseGenericToUKernelPattern("arith.divf")` | single `divf`, all-parallel, preserves operand order | `npu_uk_elementwise_div` |
| `LowerRowReductionSumToUKernelPattern` | single reduction iterator + `addf` body combining `%in` with `%out` | `npu_uk_reduction_sum` |

**Coverage on real `module.5.demoted.mlir`:**

```
Unbound linalg.generic   2972  →  2406   (566 ops bound, 19% of gap closed)

  elementwise_add   177
  elementwise_mul   138
  elementwise_sub   114
  reduction_sum     214
  (matmul 334, attention 36 — already handled by existing patterns)
  elementwise_div     0   — real SmolVLA divf cases are mixed with
                            arith.index_cast / sitofp (scalar index math),
                            correctly rejected by the strict matcher
```

**Why the remaining 2406:** most are multi-op chains (silu/gelu/rms_norm
decomposed by global-opt into sequences of single-op `linalg.generic`), type
conversions (`truncf` alone, 392), fills/copies (empty bodies, 267), and
index math. The chain cases can be covered either by a follow-up fusion pass
(coalesce sequences into a single `ukernel_generic`) or by letting the
per-op elementwise binding stand — the individual `addf`/`mulf` pieces of
the chain already bind; what's missing is the semantic labeling.

**Verification:**

```bash
build/host-merlin-debug/tools/iree-opt --iree-plugin=npu \
  --pass-pipeline="builtin.module(convert-linalg-to-npu-kernel)" \
  build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8po2/smolVLA.q.fp8po2_global_optimization.mlir \
  | rg -c "linalg.generic"   # → 2406 (was 2972 before these patterns)
```

Manifest smoke still passes (13 kernels, 610 instructions) — no regression
on the verified kernel library.

### 15.1 Tile loop generation: `TileNPUKernelToSchedule` pass (2026-04-13)

New pass at `compiler/src/merlin/Dialect/NPU/Transforms/TileNPUKernelToSchedule.cpp`.
Reads `tile_shape` from the kernel manifest and wraps each whole-tensor
`npu_kernel.ukernel_generic` op in an `scf.for` tile loop nest, emitting
`npu_schedule.ukernel_launch` at tile granularity with
`tensor.extract_slice` / `tensor.insert_slice` for input/output plumbing.

**Option:** `--kernel-manifest=<path>` (when absent, defaults to `[32, 32]`).

**Behavior:**
- Whole-tensor binary elementwise on `(M, N)` bf16 where both dims > tile_shape
  → emits nested `scf.for` + per-tile `ukernel_launch`.
- Whole-tensor ops already at tile shape → pass skips them (returns `failure()`);
  they flow through the existing `ConvertNPUKernelToSchedule` 1:1 pass.

**Coverage on `module.5.demoted.mlir` (combined pipeline):**

```bash
iree-opt --iree-plugin=npu \
  --pass-pipeline="builtin.module(convert-linalg-to-npu-kernel, \
    tile-npu-kernel-to-schedule{kernel-manifest=benchmarks/SaturnNPU/kernel_library/manifest.json}, \
    convert-npu-kernel-to-schedule)" \
  module.5.demoted.mlir
```

Yields **1013 scheduled `ukernel_launch` ops + 23 `softmax_fragment` ops +
450 scf.for tile loops**, zero remaining `npu_kernel.ukernel_generic`.
Distribution:

| Symbol | Count |
|---|---|
| `npu_uk_matmul_f8E4M3FN_f8E4M3FN_bf16` | 334 |
| `npu_uk_reduction_sum` | 214 |
| `npu_uk_elementwise_add` | 177 |
| `npu_uk_elementwise_mul` | 138 |
| `npu_uk_elementwise_sub` | 114 |
| `npu_uk_gemma_attention_bf16_bf16` | 36 |

**Verification** — test MLIR with a 64×64 elementwise_add:

```mlir
// Input: linalg.generic addf on tensor<64x64xbf16>
// After pipeline:
%0 = tensor.empty() : tensor<64x64xbf16>
%1 = scf.for %i = %c0 to %c64 step %c32 iter_args(%acc1 = %0) {
  %2 = scf.for %j = %c0 to %c64 step %c32 iter_args(%acc2 = %acc1) {
    %al = tensor.extract_slice %a[%i, %j] [32, 32] [1, 1] ...
    %bl = tensor.extract_slice %b[%i, %j] [32, 32] [1, 1] ...
    %r  = npu_schedule.ukernel_launch "npu_uk_elementwise_add"(%al, %bl) ...
    %out = tensor.insert_slice %r into %acc2[%i, %j] [32, 32] [1, 1] ...
    scf.yield %out : tensor<64x64xbf16>
  }
  scf.yield %2 : tensor<64x64xbf16>
}
```

4 `ukernel_launch` invocations, one per 32×32 tile, correctly stitched.

### 15.2 Patch points + per-invocation address rewriting (2026-04-13)

Manifest annotation script at
`compiler/src/merlin/Dialect/NPU/scripts/annotate_kernel_patch_points.py`
walks each kernel's instruction stream and records:

- `register` — which scalar xN holds a DRAM address for a given DMA role.
- `role` — `dram_in_K`, `dram_out_K`, or `transfer_size`.
- `original_value` — the kernel's hardcoded address (for sanity checks).
- `instructions` — indices of the addi/lui chain that computes the register.

Populated `manifest.json` in place. Example for matmul: three patch points
(`dram_in_0=0x0`, `dram_in_1=0x500`, `dram_out_0=0xb00`) each anchored to a
single-addi chain. For elementwise_add, interleaved scalar/DMA layout ends
up with a 2-step chain (lui+addi) for the larger offsets.

**Compiler-side address patching** landed in
`compiler/src/merlin/Dialect/NPU/Transforms/ConvertNPUScheduleToISA.cpp`.
The lowering pass now:

1. Loads `patch_points` alongside `instructions` from the manifest.
2. Maintains a shared `invocationCounter` across all `LowerSchedule*` patterns.
3. For each emitted kernel, clones the instruction list and rewrites the
   addi/lui chains for `dram_in_*` / `dram_out_*` roles. Allocation strategy
   (first cut): `inputBase=0`, `outputBase=0x400`, `dramStride=0x100`, per-role
   sub-stride `stride/4`. All offsets fit in a 12-bit signed immediate so
   most single-addi chains patch cleanly.

**End-to-end verification:** compiled a 2-distinct-matmul test through the
full plugin pipeline with `--iree-npu-native-kernel-lowering` and observed:

```
invocation 0 (first matmul):   rd=1 imm=0,   rd=2 imm=64,  rd=3 imm=1024
invocation 1 (second matmul):  rd=1 imm=256, rd=2 imm=320, rd=3 imm=1280
```

Distinct DRAM regions per invocation, as expected.

**Known limitation:** kernels whose patch chain is 1 instruction cannot
represent addresses outside 12-bit signed range without instruction
insertion (which would shift all subsequent indices in the stream).
Current allocator stays within 2 KB to sidestep this; real SmolVLA needs
either (a) manifest prep that pads every 1-addi chain to 2 instructions
(lui+addi pair), or (b) a compiler pass that inserts lui instructions and
re-computes patch_point indices. Tracked as follow-up.

### 15.3 npu_model Program scaffolds: SmolVLATransformerBlock & SmolVLAFullProgram (2026-04-13)

New files at `third_party/npu_model/npu_model/configs/programs/`:

| File | Purpose | State |
|---|---|---|
| `smolvla_transformer_block.py` | Single transformer block stitched from 12 manifest kernels (rms_norm → Q/K/V matmul → attention → O matmul → residual add → rms_norm → MLP → residual add) | Imports + stitches 518 instructions; simulator run needs stitch.py stride tuning |
| `smolvla_full_program.py` | Full SmolVLA program scaffold with weight-loader plumbing for the 494 dense_resource payloads in `module.5.demoted.mlir` | Imports + stitches 12 kernels as placeholder; real-model weights/schedule are follow-up work |

Both import via `kernel_library.stitch.stitch_kernels` after adding
`benchmarks/SaturnNPU/` to `sys.path` so the package-relative imports
inside stitch.py resolve. Follow the `gemma_attention.py` precedent.

### 15.4 Kernel tiling direction (from Slack 2026-04-12)

Aligned on: **kernels are the tiling unit** — each kernel may consume
multi-tile inputs and handle internal accumulation (K-reduction, running
max/sum for attention). Sync between consecutive kernels is treated as
"another kernel" to profile independently.

This doesn't change the first-cut compiler pipeline (we have only
single-tile kernels in the manifest today — `tile_shape: [32, 32]`). It
does shape the contract: `TileNPUKernelToSchedule` reads `tile_shape` from
the manifest rather than hardcoding 32, so when multi-tile kernels
(`tiled_matmul_2x1: [64, 32]`, etc.) land later, no code change is needed.
A scaffold `InsertSyncKernels` pass will be wired into the pipeline at that
point so future sync-family manifest entries have a dedicated insertion
point.

**Autocomp / manifest cross-check:**

Cross-checking `manifest.json` against autocomp `run_metrics.json` best
scores (autocomp candidate sources have been pruned; only run metrics
remain):

| Kernel | Manifest cy | Autocomp best cy | Delta |
|---|---|---|---|
| matmul, silu, add/mul/sub, rope | matches | matches | 0 |
| gelu_tanh | 327 | 337 | manifest is 10cy better (post-autocomp tuning) |
| elementwise_div | 220 | 212 | manifest is 8cy stale — autocomp source gone, can't regenerate |
| softmax, rms_norm, reduction_sum, requant, attention | N/A (hand-written) | N/A (autocomp never converged) | — |
| tiled_matmul_2x1 (prob 30) | absent | 297 | 25% per-tile improvement, worth adding when DMA pipelining pass lands |
| fast_gelu (prob 31) | absent | 356 | slower than 327cy gelu_tanh — skip |

## 16. Cross-Invocation Orchestration: Matmul K-Accumulation + Flash Attention Plumbing (2026-04-13)

Triggered by a Slack question from Huijae An: *"if the kernels spat out from
merlin are tiled already...does it also orchestrate the results from the
tiled computes? for matmul, does it automatically accumulate the partial
results, and for attention, keep the running sum?"*

The compiler had only single-tile kernels in the manifest (M=K=N=32 for
matmul, seq=32 for attention) — sufficient for the placeholder
`SmolVLATransformerBlock` smoke but not for SmolVLA's real shapes (matmuls
with K up to 2048, attention at `(12, 1024, 64)`). This section closes the
matmul gap end-to-end and lands the compiler-side plumbing for flash
attention.

### 16.1 Pre-existing manifest bug surfaced and fixed

While building the matmul K-accumulator smoke test, the simulator
returned all-zero bytes at the expected output address. Tracing the patched
DMA store revealed every matmul/attention/elementwise manifest kernel
encoded `addi rd, x0, K` for K outside the signed 12-bit range
(`[-2048, 2047]`) — for example `addi rd=8, rs1=0, imm=2048` for a
2048-byte transfer size. RISC-V (and the npu_model simulator) sign-extends
the 12-bit immediate, so 2048 silently became -2048 and the DMA store
copied zero bytes.

Fix in `/tmp/fix_manifest_addi.py` (idempotent rewrite tool, can live in
`compiler/src/merlin/Dialect/NPU/scripts/` next iteration):

  * `addi rd, x0, K` → `lui rd, hi(K); addi rd, rd, lo(K)` (rs1 == 0 case)
  * `addi rd, rd, K` → `lui x31, hi(K); addi x31, x31, lo(K); add rd, rd, x31`
    (cumulative-bump case in `elementwise_add`; uses x31 as a scratch since
    no kernel touches regs above x21)

13 overflowing addi ops rewritten across `matmul`, `silu`,
`elementwise_add`, and the new `matmul_acc_*` variants.

The annotator (`annotate_kernel_patch_points.py`) is then re-run to
repopulate `patch_points` with corrected indices. Without this fix, none
of the manifest kernels produce correct numerics — every prior matmul
output went into negative DRAM addresses (which Python-list-style indexing
in `arch_state.write_dram` silently translated into writes near the end of
the DRAM tensor).

### 16.2 Matmul K-tiling: 3 manifest variants + compiler pass

**Manifest** (`benchmarks/SaturnNPU/kernel_library/manifest.json`):

  * `matmul_acc_first` — issues `vmatmul.mxu0` (overwrite accumulator);
    skips the pop / vstore / DMA store tail. 21 instructions.
  * `matmul_acc_mid`   — same prologue, issues `vmatmul.acc.mxu0`
    (accumulator-add); also skips the drain. 21 instructions.
  * `matmul_acc_last`  — `vmatmul.acc.mxu0` followed by the original
    drain (`vmatpop.bf16.acc.mxu0` + 2 vstores + DMA store). 28
    instructions.

All three reuse the existing matmul kernel's scalar/DMA setup and
`vmatpush.weight.mxu0`. Synthesized via
`/tmp/gen_matmul_acc.py`; patch_points populated by the existing
annotator and validated end-to-end.

**Compiler pass** (`TileMatmulOpPattern` /
`TileMatmulUKernelPattern` in
`compiler/src/merlin/Dialect/NPU/Transforms/TileNPUKernelToSchedule.cpp`):

  * Match `npu_kernel.matmul` and any
    `npu_kernel.ukernel_generic "npu_uk_matmul..."`.
  * Compute tile counts `M / mTile × N / nTile × K / kTile` (defaults to
    32 each, overridable via manifest `tile_shape`).
  * Skip if `K_tiles ≤ 1` — the existing 1:1
    `ConvertNPUKernelToSchedule` path handles those.
  * Emit nested `scf.for` over (M, N) with the K-loop unrolled inside the
    body. K-iteration k → variant symbol selected by position in the
    chain: `_first` at k=0, `_mid` for the middle, `_last` at
    k=K_tiles-1. Only `_last`'s tensor result is read (the others' MXU
    state is implicit hardware state and the SSA results are dead).

**Validation:**

  * IR shape: `tile_matmul_k.mlir` (FileCheck-passing) covers K=2 (no
    `_mid`) and K=3 (full `first/mid/last` triple).
  * Numerical: `MatmulKTileSmokeProgram` (in
    `third_party/npu_model/npu_model/configs/programs/matmul_k_tile_smoke.py`)
    runs a 32×64 ⊗ 64×32 matmul as
    `stitch_kernels(["matmul_acc_first", "matmul_acc_last"])`
    and `torch.allclose`s against the bf16 reference within
    `rtol=1e-2, atol=1e-2`. Max abs diff: 2.0 over outputs ranging up
    to ±584. Total cycles: 256 for two K-tile invocations.

### 16.3 Flash attention: compiler plumbing in place; ISA bodies stubbed

The same three-variant approach was registered for attention
(`attention_acc_first/mid/last`) plus a `TileAttentionUKernelPattern`
that emits an scf.for nest over (batch, q-block) with the K/V dimension
unrolled inside.

**Compiler pass** signs:

  * Matches any `ukernel_generic` whose symbol starts with
    `npu_uk_attention` (or `npu_uk_gemma_attention`) and excludes
    `npu_uk_attention_acc` (avoid recursion).
  * Skip when K/V seq ≤ 32 (single-tile case → 1:1 path).
  * Loop nest:
    - Outer batch loop (only emitted for rank-3 inputs).
    - Inner Q row-block loop in 32-row strides.
    - K/V tile chain unrolled inside the Q body, picking
      `_first / _mid / _last` by position.
  * Lit test: `tile_attention_kv.mlir` covers seq_kv=64 (first/last
    pair) and the rank-3 batched seq_kv=96 (first/mid/last triple).

**Stub status of the kernel bodies:** the manifest entries
`attention_acc_first/mid/last` currently clone the existing single-tile
`attention` kernel ISA (79 instructions each). They have the right
DMA in/out layout for compiler-side address patching, but the actual
online-softmax recurrence (running max + denom + output rescaling
across K/V tiles) is *not* implemented — each variant just runs a
single-tile attention and writes the same bf16 output. Replacing
the bodies with the real recurrence is the remaining work for
numerical correctness.

### 16.4 SmolVLAFullProgram: weights wired, PyTorch ref hooked

`compiler/src/merlin/Dialect/NPU/scripts/extract_mlir_dense_resources.py`
gained a `load_dense_resources(mlir_path, base_addr)` library function
that returns `[(dram_address, torch.Tensor)]`. Validated against the
demoted MLIR for SmolVLA: 494 payloads, 421 MB total, mix of bf16 and
f8E4M3FN, packed contiguously starting at the configured base.

`SmolVLAFullProgram` (in
`third_party/npu_model/npu_model/configs/programs/smolvla_full_program.py`)
now:

  * Calls `load_dense_resources` at class construction so
    `memory_regions` is populated with all 494 SmolVLA weights.
  * Exposes `compute_pytorch_reference(seed=42)` — a callable that loads
    the bf16 baseline SmolVLA via Understanding-PI0's
    `load_smolvla_policy` + `one_step_no_cache`, with the same dummy
    inputs the simulator should be given. Heavy imports (lerobot, mx)
    live inside the function so the Program class stays cheap to
    construct.

### 16.5 0-byte DMA simulator assertion

`third_party/npu_model/npu_model/hardware/arch_state.py:256`'s
`assert address < offset + data.numel() <= dram_size` triggered on
zero-byte DMA writes (the inequality is false when `data.numel() == 0`).
Replaced with `assert data.numel() == 0 or address + data.numel() <= dram_size`,
which preserves the bounds check for non-empty writes and treats the
empty case as a no-op.

### 16.6 What's left for a real E2E numerical SmolVLA run

  * Hand-coding (or autocomp-synthesizing) the online-softmax recurrence
    inside `attention_acc_first/mid/last`. The math is documented in the
    plan; the implementation needs roughly 200-300 ISA ops per variant
    and careful staging of the `prev_m / prev_denom / prev_out` DRAM
    state across invocations.
  * Running the SmolVLA forward through the full compile pipeline with
    `TileMatmulPattern` + `TileAttentionPattern` enabled and dumping the
    stitched text ISA, so `SmolVLAFullProgram` can swap its placeholder
    12-kernel sequence for the real schedule.
  * Comparing the simulator's final DRAM tile against
    `compute_pytorch_reference()` within bf16 tolerance.

## 17. Golden-Data Testing Ladder (2026-04-13, later)

Pivoted from "finish flash attention then E2E" to "build the testing
ladder from single kernels to the full model, reveal hidden manifest
bugs along the way." Several real manifest / annotator bugs surfaced and
got fixed as a side-effect.

### 17.1 Per-kernel golden suite (Piece A)

New artifacts:

  * `benchmarks/SaturnNPU/kernel_library/kernel_golden_fixtures.py` —
    11 kernel fixtures, each pairing a canonical torch input with a
    torch reference and expected output. Torch refs reuse
    `generate_npu_golden_tests.py` (rms_norm, silu, softmax, gelu,
    rope) and add `ref_reduction_sum`, `ref_requant`,
    `ref_single_tile_attention`.
  * `third_party/npu_model/npu_model/configs/programs/kernel_golden_suite.py`
    — dynamic Program factory: one `KernelGolden_<kernel>` class per
    fixture, auto-discovered by `test_programs.py`.
  * `compiler/src/merlin/Dialect/NPU/scripts/run_kernel_golden_tests.py`
    — CLI runner with `PASS | FAIL | XFAIL` per kernel.

Result: **8 of 11 kernels pass torch.allclose**: matmul, silu, softmax,
requant, rope_frequency, single-tile attention, elementwise_add,
elementwise_sub. The 3 xfails are tagged in the fixture with specific
manifest bug descriptions:

  * `elementwise_mul`: DMA writes B to VMEM x2=0x900 but the subsequent
    `vload` reads from VMEM x5=0x800. Manifest-level register mismatch.
    Also had 3 trailing junk ops (beq + 2 addi) causing an infinite
    loop; trimmed.
  * `reduction_sum`: output row-ordering doesn't match per-row column
    sum. Likely a vector-width / row-stride interaction needing ISA
    re-trace.
  * `rms_norm`: ~6× scale mismatch vs torch; eps / weight-broadcast
    semantic unclear from the current ISA.

### 17.2 Annotator + fix-script refinements (prerequisite for 17.1)

Fixing the kernel golden suite surfaced that the `addi`-overflow fix
from section 16.1 had a subtle asymmetry: the autocomp-generator relied
on RISC-V sign-extension for `addi rd=X, rs1=X, imm>2047` (cumulative
offset bumps in `elementwise_add`), where the sign-extended interpretation
was the *intended* semantic. My original rewrite turned those into
literal adds, breaking the addresses. Fixes:

  * `fix_manifest_addi_overflow.py` now skips `rs1 == rd` cases — only
    the `rs1 == 0` shape (matmul's initial DRAM-address setup) gets the
    `lui + addi` split.
  * `revert_rs1eqrd_addi_overflow.py` — new script that undoes the
    `lui + addi + add` triplet my earlier fix produced, returning the
    original single `addi` with its out-of-range immediate. Idempotent.
  * `annotate_kernel_patch_points.py` — now tracks `add` / `sub` and
    sign-extends `addi` imm. This makes the annotator's register-value
    estimate match what the simulator actually computes; without this
    the manifest's `dram_in_*` / `dram_out_*` were reported at the wrong
    offsets for several kernels.

### 17.3 Matmul K-tile chain harness (Piece B.1)

`matmul_k_tile_smoke.py` parameterized into a factory that emits one
Program per K_tiles ∈ {2, 3, 4}. All three pass `torch.allclose` at
`rtol=1e-2, atol=2.0` (atol covers one bf16 ULP at K=128 accumulation,
where output magnitudes reach ±~800). Max diffs 2-4. Per-run cycles:
K=2 → 282, K=3 → 364, K=4 → 446.

Evidence the MXU accumulator survives the full K chain and the compiler
pattern (`TileMatmulPattern`) lowers to the right sequence.

### 17.4 Text-ISA parser + round-trip harness (Pieces D.1, E lite)

`third_party/npu_model/npu_model/software/text_isa_loader.py` parses the
stitch.py-format text ISA (`<mnemonic> key=value, key=value, ...`) back
into an `Instruction[Args]` stream. Round-trip across all 19 manifest
kernels is lossless: 823 instructions parsed back identically to the
originals.

`run_text_isa_roundtrip_parity.py` exercises the full loop: fixture →
stitch_kernels → instruction_to_text → parse_text_isa → Simulation →
torch.allclose. Same 8/11 pass + 3 xfail result as the in-memory suite,
confirming the parser is behavior-preserving.

Gap: the compiler's `npu-translate --mlir-to-npu-text-isa` emits a
*different* simplified format (`vmul` instead of `vmul.bf16`, `dma.load`
instead of `dma.load.ch<N>`, integer-indexed register fields). Bridging
the two formats is the remaining Piece D.2 work: either teach the
parser aliases, or extend the emitter to produce the manifest dialect.

### 17.6 Real flash-attention kernels + composition E2E (2026-04-13, later)

Using the verified reference `SmolVLAFusedAttentionProgram` at
`benchmarks/SaturnNPU/kernel_library/smolvla_fused_attention.py` as the
template, split its MRF-resident state (m, l, O_col0..3) into DRAM-spilled
state for three manifest variants:

  * `benchmarks/SaturnNPU/kernel_library/attention_acc_kernels.py` —
    factored prologue + KV iteration body + state init / load / store /
    normalize epilogue. Generators `first_body()`, `mid_body()`,
    `last_body()` compose these into the three instruction streams.
  * `compiler/src/merlin/Dialect/NPU/scripts/gen_attention_acc_variants.py`
    — writes the generated bodies into `manifest.json`, replacing the
    stub entries from section 16.3.
  * `benchmarks/SaturnNPU/kernel_library/stitch.py:stitch_attention_chain`
    — new helper that threads running-state DRAM addresses across
    invocations via ping-pong slots + explicit load/store overrides (the
    stitcher's existing `load_overrides` / `store_overrides` per-invocation
    mechanism already supports this without core changes).
  * `third_party/npu_model/npu_model/configs/programs/attention_kv_chain_smoke.py`
    — `AttentionKVChain{2,3}Program` validate the chain end-to-end
    against the ISA-exact torch reference. Results: max_diff = 0.0034
    (atol=0.2), 3077 cycles at KV=2, 4726 cycles at KV=3.

Per-variant instruction counts after annotation:
`attention_acc_first: 120, attention_acc_mid: 122, attention_acc_last: 125`
— each has the four per-tile DMA loads (Q, K, V, scale), the full online-
softmax recurrence body, and either state-store or normalize+store.

### 17.7 Cross-kernel composition smokes (Piece F.A)

`third_party/npu_model/npu_model/configs/programs/composition_smoke.py`:

  * `CompositionMatmulOnly` — single matmul invocation driven by explicit
    address overrides via the stitcher (199 cyc, max_diff = 0).
  * `CompositionTwoMatmulChains` — two independent K=2 matmul_acc chains
    running back to back with non-overlapping DRAM regions (561 cyc,
    max_diff = 2, atol=2.0).

Together with the matmul K-tile (B.1) and attention K/V chain (B.2) suites,
this proves the end-to-end stitching model: kernel outputs flow to
subsequent kernel inputs through explicit DRAM addresses with the
patcher-driven `load_overrides` / `store_overrides`.

### 17.8 Resolved kernel bugs + all 11 goldens green

Re-traced and fixed three manifest kernel issues surfaced by the golden
suite:

  * **rms_norm** — kernel computes ``y = x * rsqrt(sum(x²) * inv_dim + eps)``;
    ``dram_in_2`` is a broadcast ``inv_dim`` constant, ``dram_in_3`` is
    a broadcast ``eps``. The earlier torch reference was passing weight
    tiles in those slots (textbook RMS-norm with learnable weight); the
    kernel has no weight multiplication. Updated fixture to match the
    exact ISA semantic — now passes at max_diff=0.015.
  * **elementwise_mul** — VMEM register mismatch: DMA load wrote B to
    VMEM x2 (0x900), but the subsequent ``vload`` read from VMEM x5
    (0x800, the DRAM register). Patched the two offending vloads'
    ``rs1`` from 5 → 2 via
    ``compiler/src/merlin/Dialect/NPU/scripts/fix_elementwise_mul_vload_bug.py``.
    Now passes at max_diff=0.
  * **reduction_sum** — torch reference used float32 ``x.sum(dim=-1)``
    but the kernel adds the two bf16 halves first and then row-reduces;
    order of accumulation differs. Aligned the reference math to the
    kernel path; now max_diff=0.

**Result**: `run_kernel_golden_tests.py` reports 11/11 PASS, 0 xfail.

### 17.9 Layout model — same 2048-byte footprint across kernels

Matmul writes its 32×32 bf16 output as ``bf16_split_halves``: 1024 B of
cols 0-15 packed as [32, 16], then 1024 B of cols 16-31. Elementwise
ops read the same 2048 B with ``vload imm12=0`` + ``vload imm12=32``
(each pulls one 32×16 register-width chunk from VMEM).

Keeping every activation in this split-halves layout through the block
lets arithmetic kernels compose directly: matmul's ``out_h0`` +
elementwise's ``A_h0`` maps cleanly, and reading the final output as
``torch.cat([h0, h1], dim=0)`` (shape [64, 16]) recovers the logical
tile. No repack kernel needed — only the fp8→bf16 and bf16→fp8
boundaries (matmul inputs, requant outputs) require layout awareness.

### 17.10 Mini-block composition chain end-to-end (Piece F.A proper)

`composition_mini_block.py::CompositionMiniBlockRmsRequantMatmulAdd`
chains four kernels:

    x  bf16 split halves
      ├── rms_norm      → x_norm     (split halves)
      ├── requant       → x_fp8      (contiguous fp8 32×32)
      ├── matmul(x_fp8, W_fp8) → proj (split halves)
      └── elementwise_add(proj, x)   → res  (split halves, [64,16] view)

159 total instructions. Simulator runs in 741 cycles. The torch
reference mirrors the ISA's exact arithmetic (fp8 roundtrip,
bf16 accumulation, split-halves layout). Result:
`allclose=True, max_diff=1.125` at magnitudes up to 13.5. This is the
merge-gate proof that multi-kernel composition + DRAM threading works
end to end.

### 17.11 Scheduling primitives audit

Counted the mnemonic distribution in three representative schedules to
confirm every level of the tile-to-tile orchestration is represented:

| Primitive | Mini-block | Attention KV=3 | Matmul K=4 |
|---|---|---|---|
| DMA loads / stores / waits / config | 10 / 5 / 17 / 6 | 14 / 3 / 29 / 12 | 8 / 1 / 9 / 4 |
| Vector load/store | 19 | 67 | 10 |
| MXU weight-push | 1 | 12 | 4 |
| MXU matmul (overwrite + accum) | 1 + 0 | 9 + 3 | 1 + 3 |
| MXU push-bf16 / pop-fp8 (roundtrip) | 0 | 21 / 21 | 0 |
| Vector arith (add/mul/sub) | 9 | 52 | 0 |
| Softmax primitives (exp + redmax + redsum) | 2 | 21 | 0 |
| Delay barriers | 10 | 27 | 1 |
| Scalar setup (addi/lui/add) | 73 | 68 | 58 |

Confirms: (a) cross-invocation DMA sync happens via dma.wait, (b) MXU
accumulator state survives across invocations in the matmul chain, (c)
the attention chain exchanges running state through DRAM via per-
invocation load/store overrides, (d) dtype bridging (bf16⇄fp8) happens
inline via acc-roundtrips, (e) pipeline hazards are handled via delay
barriers where needed.

### 17.12 Remaining for full SmolVLA E2E

What's still left for `SmolVLAFullProgram` to `torch.allclose` vs
`compute_pytorch_reference()`:

  * **Block-composition helper** — a Python function that emits the
    stitched instruction stream for one full transformer block (Q/K/V
    projections → attention → O projection → residual → rms_norm →
    MLP up/silu/down → residual). The primitives all exist; the helper
    iterates the matmul N/K tile grid and calls
    ``stitch_attention_chain`` for the attention step.
  * **Weight-layout packer** — turns the 494 demoted-MLIR weight tiles
    into the fp8 32×32-tile-per-(m_tile, k_tile) layout the matmul
    chains expect. Purely a DRAM marshaling pass.
  * **Sim-time gating** — at SmolVLA shapes (seq=1024, hidden=768), one
    matmul expands to ~18k kernel invocations; six matmuls per block ×
    16 blocks ≈ 1.7M invocations ≈ tens of CPU-hours in the current
    simulator. A ``--fast-sim`` path (skip per-cycle trace writing) is
    the obvious lever; otherwise Piece G is a CI batched run.
  * **RoPE in attention** — SmolVLA applies rotary position embeddings
    to Q, K before attention. The current flash kernel doesn't. Either
    precompute RoPE'd Q, K in torch and feed them as inputs, or extend
    the kernel.

### 17.13 Open follow-ups (compiler-side alignment)

### 17.9 What's left (gated on Piece C + D.2)

  * **Flash-attention ISA bodies (Piece C)**. The compiler-side plumbing
    (TileAttentionPattern, manifest stubs) is in place. The real
    online-softmax recurrence is the multi-hour hand-coding effort.
  * **Compiler dump + alias layer (Piece D.2)**. Adds a
    `--iree-npu-native-kernel-lowering` dump path in `tools/compile.py`
    producing `module.6.isa.txt` in the stitch-dialect format that
    our parser already handles.
  * **Attention chain tests (Piece B.2)**. Structurally identical to
    `MatmulKTile{2,3,4}Program`; waits on real Piece C kernels.
  * **One-transformer-block E2E (Piece F)**. The merge gate — extract
    one block from `module.5.demoted.mlir`, compile it, swap the
    placeholder schedule in `SmolVLATransformerBlock`, and assert
    `torch.allclose` vs a PyTorch block forward. Requires C + D.2.
  * **Full SmolVLA forward (Piece G)**. Stretch goal; same pattern as
    F scaled up to ~10³-10⁴ kernel invocations.

---

*Dev-blog written by:* Agustin Coppari Hollmann

*Project Members:* Yufeng Chi and Agustin Coppari Hollmann
