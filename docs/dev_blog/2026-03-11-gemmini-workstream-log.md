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
- `--iree-gemmini-enable-conv2d`
- `--iree-gemmini-enable-requantize`
- `--iree-gemmini-enable-clamp`
- `--iree-gemmini-dataflow={os|ws}`
- `--iree-gemmini-tile-m`, `--iree-gemmini-tile-n`, `--iree-gemmini-tile-k`

## What Worked

- Matmul recovery from canonical `linalg.generic` into `gemmini.matmul` for
  int8/int8/i32 patterns.
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
   - `conda run -n merlin-dev python tools/build.py --profile gemmini`
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
