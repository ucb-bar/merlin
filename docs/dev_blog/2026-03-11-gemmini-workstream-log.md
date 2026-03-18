# 2026-03-11: Gemmini Workstream Log

## Context and Goal

The Gemmini dialect path in Merlin is designed as a post-global-optimization
recovery flow:

- detect Gemmini-friendly semantics from normalized `linalg.generic`
- materialize `gemmini.*` ops
- optionally lower back to ordinary IREE/MLIR IR for downstream compatibility

Current status: **active development; no validation yet on simulated/programmed
or taped-out hardware in this repo flow**.

## Implementation Changes (Current In-Tree State)

Gemmini dialect IR currently models:

- `gemmini.matmul`
- `gemmini.matmul_tile`
- `gemmini.conv2d`
- `gemmini.requantize`
- `gemmini.clamp`

Gemmini passes currently implemented:

- `gemmini-convert-to-gemmini`
- `gemmini-lower-to-isa`
- `gemmini-canonicalize`
- `gemmini-lower-gemmini-to-iree`

Plugin wiring (`compiler/plugins/target/Gemmini`) runs these passes after global
optimization when `--iree-gemmini-enable` is set, for both:

- `func.func`
- `util.func`

Important plugin options:

- `--iree-gemmini-enable`
- `--iree-gemmini-lower-back-to-iree`
- `--iree-gemmini-enable-matmul`
- `--iree-gemmini-enable-fp8-matmul`
- `--iree-gemmini-enable-conv2d`
- `--iree-gemmini-enable-requantize`
- `--iree-gemmini-enable-clamp`
- `--iree-gemmini-dataflow={os|ws}`
- `--iree-gemmini-tile-m`, `--iree-gemmini-tile-n`, `--iree-gemmini-tile-k`

## What Worked

- Matmul recovery from canonical `linalg.generic` into `gemmini.matmul` for
  int8/int8/i32 patterns.
- Optional FP8 matmul recovery (`f8E4M3FN/f8E4M3FN -> bf16|f32`) behind
  `--iree-gemmini-enable-fp8-matmul`.
- Named `linalg.matmul` recovery support in addition to canonical generic forms.
- Conv2D recovery for CHW/FCHW-style int8/int8/i32 patterns with stride/dilation
  extraction from affine maps.
- Requantize and clamp recovery from expected scalar-op chains.
- `gemmini-lower-to-isa` currently stages `gemmini.matmul` into
  `gemmini.matmul_tile` with explicit tile metadata.
- `gemmini-lower-gemmini-to-iree` converts Gemmini ops back into linalg/arith
  forms to preserve compatibility with generic downstream pipelines.

## What Did Not Work / Current Limitations

- No direct hardware execution path is wired from Gemmini dialect in this tree.
- `gemmini-lower-to-isa` is currently a staged structural lowering step
  (`matmul -> matmul_tile`), not a final hardware packet/binary emission path.
- Recovery is intentionally strict and shape/type-specific:
  - mostly int8/int8/i32 matmul/conv patterns
  - requantize/clamp must match expected op sequences
- Non-matching patterns remain in baseline MLIR dialects (for example, fp8 add
  stays as `linalg.add`).

## Debugging Notes

Most useful loop while iterating on pattern matching:

1. run only `gemmini-convert-to-gemmini`
2. inspect whether recovery happened
3. run `gemmini-lower-to-isa` to check tile metadata propagation
4. run `gemmini-lower-gemmini-to-iree` to verify back-lowering correctness

Useful inspection knob for post-global integration:

- `--iree-gemmini-lower-back-to-iree=false`
  keeps `gemmini.*` visible in global-opt output for debugging.

## Test Coverage and Commands

Compiler lit tests exist under:

- `compiler/src/merlin/Dialect/Gemmini/Transforms/tests/`

Key files:

- `convert-to-gemmini.mlir`
- `matmul-lower-to-isa.mlir`
- `lower-gemmini-to-iree.mlir`
- `fp8-no-convert.mlir`
- `post-global-opt-hook.mlir`

Typical commands:

```bash
build/host-merlin-<config>/install/bin/iree-opt \
  compiler/src/merlin/Dialect/Gemmini/Transforms/tests/convert-to-gemmini.mlir \
  --iree-plugin=gemmini \
  --pass-pipeline='builtin.module(func.func(gemmini-convert-to-gemmini))'
```

```bash
build/host-merlin-<config>/install/bin/iree-compile \
  compiler/src/merlin/Dialect/Gemmini/Transforms/tests/post-global-opt-hook.mlir \
  --iree-input-type=none \
  --iree-hal-target-backends=llvm-cpu \
  --compile-to=global-optimization \
  --iree-plugin=gemmini \
  --iree-gemmini-enable \
  --iree-gemmini-lower-back-to-iree=false
```

## Reproduce Latest Stage (Checklist)

1. Build Gemmini-enabled compiler tools:
   - `conda run -n merlin-dev uv run tools/build.py --profile gemmini`
2. Confirm plugin load:
   - `build/host-merlin-debug/install/bin/iree-compile --iree-list-plugins`
3. Run transform tests under:
   - `compiler/src/merlin/Dialect/Gemmini/Transforms/tests/`
4. Run post-global hook test with:
   - `--iree-gemmini-enable`
   - `--iree-gemmini-lower-back-to-iree=false`
5. Inspect output for recovered/staged ops:
   - `gemmini.matmul`
   - `gemmini.matmul_tile`

Note: this confirms compiler pattern recovery/lowering behavior only; it is not
yet a hardware-validated execution path.

## Follow-Up Tasks

- Expand recovery beyond current strict canonical forms.
- Add stronger e2e tests for `conv2d`, `requantize`, and `clamp` post-global
  pipeline behavior.
- Define/implement a concrete downstream execution path from staged Gemmini IR
  to runtime-executable representation.
- Add simulator/hardware-oriented validation once backend/runtime path is ready.

## Extra: (TODO Clean-up)

### 8.2 Gemmini conversion coverage

Running on `util.func` scope (same scope used in post-global-opt hooks):

- `linalg.matmul` reduced from 67 to 1
- `gemmini.matmul` recovered: 66
- lowered form present: `gemmini.matmul_tile` (66)

<details>
<summary>Gemmini-lowered snippet (real global-opt file)</summary>

```mlir
%1717 = gemmini.matmul_tile %1715, %cst_322
  {dataflow = 0 : i32, lhsZeroPoint = 0 : i64, rhsZeroPoint = 0 : i64,
   tileK = 16 : i64, tileM = 16 : i64, tileN = 16 : i64}
  : tensor<50x720xi8>, tensor<720x720xi8> -> tensor<50x720xi32>
```

</details>

### 13.6 Gemmini FP8 note for reproducibility

Gemmini FP8 matching uses `--iree-gemmini-enable-fp8-matmul` and
`--iree-gemmini-lower-back-to-iree=false` in `models/gemmini_mx.yaml` so
Gemmini ops remain visible in the global-opt output for matcher development.

Use `build/host-merlin-release/tools/iree-compile --iree-list-plugins` to
check plugin availability (this build's `tools/` binary is the one used by
`tools/compile.py`).

Validated in this workspace (`2026-03-13`):

- `tmp/smolvla_global_opt_phases_verify_gemmini_mx_real2/module.4.global-optimization.mlir`
  contains `gemmini.matmul_tile` (`66` matches).
- `tmp/smolvla_global_opt_phases_verify_npu_ucb_real/module.4.global-optimization.mlir`
  contains NPU ISA ops (`npu_isa.matmul_mxu*`, `npu_isa.vexp`, `npu_isa.vmul`,
  DMA ops), confirming post-global matching happened.

### 13.8 Full VMFB status for `models/smolVLA/smolVLA.q.fp8.mlir` on Gemmini

Full compile was re-run in this workspace on `2026-03-13` with:

```bash
conda run -n merlin-dev uv run tools/compile.py \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target gemmini_mx \
  --quantized
```

Output directory:

- `build/compiled_models/smolVLA/gemmini_mx_RVV_smolVLA.q.fp8/`

Current result:

- global-opt / matcher coverage is still working as documented above
- full end-to-end VMFB generation still fails; the compile exits nonzero and
  does not produce a valid `smolVLA.q.fp8.vmfb`

First visible failure class in the current trace:

- unresolved executable materialization around the softmax-score path:
  - `tensor<291xi8, #iree_encoding.encoding<...>> -> tensor<291xi8>`
  - `tensor<291xi8> -> tensor<291xi1>`
- the failing dispatch still contains:
  - `linalg.batch_matmul ... : tensor<15x291x64xbf16> x tensor<15x64x291xbf16> -> tensor<15x291x291xf32>`
  - a following `linalg.generic` using mask tensors derived from `tensor<291xi8>`
- later in the same compile, executable translation also fails on a separate
  constant/type mismatch:
  - `dense_resource<torch_tensor_32_torch.bfloat16> : tensor<32xbf16>`
  - result type `tensor<32xf32>`

Interpretation:

- the new Gemmini-MX RISC-V intrinsics remove one LLVM backend gap, but they do
  not by themselves unblock end-to-end SmolVLA VMFB generation
- the current blocker is still downstream executable
  encoding/materialization/translation on this exported SmolVLA FP8 path
- this same broad failure class also reproduces on non-Gemmini full-compile
  targets in this tree, so it should not currently be treated as a
  Gemmini-specific regression

---

*Dev-blog written by:* Agustin Coppari Hollmann

*Project Members:* See [Gemmini-MX](https://github.com/ucb-bar/gemmini/tree/gemmini-mx) for the original Gemmini ISA authors
