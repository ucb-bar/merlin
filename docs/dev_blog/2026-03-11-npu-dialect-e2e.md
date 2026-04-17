# 2026-03-11: NPU Dialect E2E Bring-Up

> **Repro pin:** merlin@[`e18fc562`](https://github.com/ucb-bar/merlin/commit/e18fc562c5c9a9601fc3e34a6d990a0427ddc255) · iree_bar@[`dd293bb513`](https://github.com/ucb-bar/iree_bar/commit/dd293bb513)
> **Status:** Active

## Context and Goal

Integrate the `third_party/npu_model/compiler` dialect stack into Merlin as an
IREE compiler plugin target that runs after global optimization and emits ISA
text consumable by `npu_model`.

## Implementation Changes

- Added three MLIR dialect layers in Merlin:
  - `npu_kernel`
  - `npu_schedule`
  - `npu_isa`
- Added lowering passes:
  - `convert-linalg-to-npu-kernel`
  - `convert-npu-kernel-to-schedule`
  - `convert-npu-schedule-to-isa`
- Added additional NPU passes:
  - `npu-verify-ukernel-symbols`
  - `npu-plan-memory`
- Added NPU plugin activation and policy options:
  - `--iree-npu-enable`
  - verifier strictness/fallback toggles
  - matmul weight-source toggle (`mxu0` vs `mxu1`)
  - deterministic memory planner controls
- Added translator:
  - `npu-translate --mlir-to-npu-text-isa`
- Added scripts:
  - simulator smoke runner
  - ISA contract checker
  - numerical parity harness
  - frontend-to-simulator e2e script

## What Worked

- NPU plugin loaded through `iree-compile --iree-list-plugins`.
- Post-global pipeline successfully lowered matmul-like linalg to `npu_isa`.
- ISA text emitted in `mnemonic key=value` style accepted by `npu_model`.
- Smoke tests for Gemma symbol families ran to completion in simulator.
- Basic identity-matmul parity check passed.

## What Did Not Work Initially

- New passes failed to compile:
  - missing `ModuleOp` include in pass files.
  - float8 API mismatch (`isFloat8E4M3FN`) with local MLIR version.
- New verifier test structure was inconsistent:
  - positive RUN attempted to parse both valid and intentionally-invalid split
    chunks.
- Contract checker was too strict by default:
  - `model_npu` ISA defs do not currently include DMA `flag` as required keys,
    while emitted text includes them.
- E2E translation from global-opt IR failed initially:
  - global-opt output had non-NPU attrs/ops requiring generic printing and
    tolerant translation parsing.

## Debugging Notes

- Build failures were isolated by running the project-standard build command and
  fixing compile errors one-by-one in pass files.
- ISA contract issues were validated by reading
  `third_party/npu_model/model_npu/configs/isa_definition.py` and comparing
  required keys against emitted text.
- E2E parser failures were resolved by:
  - compiling global-opt with `--mlir-print-op-generic`
  - translating with `--allow-unregistered-dialect`

## Test Coverage and Commands

Build:

```bash
conda run -n merlin-dev uv run tools/build.py --profile npu --config release --no-build-python-bindings --no-enable-libbacktrace
```

Plugin visibility:

```bash
build/host-merlin-release/install/bin/iree-compile --iree-list-plugins
```

NPU transform tests (manual RUN-equivalent execution):

```bash
build/host-merlin-release/install/bin/iree-opt ...
build/host-merlin-release/install/bin/iree-compile ...
```

Simulator smoke:

```bash
conda run -n merlin-dev ./compiler/src/merlin/Dialect/NPU/scripts/run_npu_model_smoke.sh \
  build/host-merlin-release/install/bin \
  npu_uk_gemma_mlp_f8E4M3FN_f8E4M3FN_f32
```

E2E global-opt to simulator:

```bash
conda run -n merlin-dev ./compiler/src/merlin/Dialect/NPU/scripts/run_frontend_to_npu_model_e2e.sh \
  build/host-merlin-release/install/bin
```

Numerical parity:

```bash
conda run -n merlin-dev ./compiler/src/merlin/Dialect/NPU/scripts/run_npu_model_smoke.sh \
  build/host-merlin-release/install/bin \
  npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32 \
  compiler/src/merlin/Dialect/NPU/scripts/outputs/npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32.txt \
  --parity
```

## Reproduce Latest Stage (Checklist)

1. Build tools with NPU profile.
2. Confirm plugin load:
   - `build/host-merlin-release/install/bin/iree-compile --iree-list-plugins`
3. Run transform tests in `compiler/src/merlin/Dialect/NPU/Transforms/tests/`.
4. Run smoke scripts for:
   - `npu_uk_gemma_mlp_*`
   - `npu_uk_gemma_attention_*`
5. Run parity for `npu_uk_matmul_*` with `--parity`.
6. Run frontend e2e script:
   - `run_frontend_to_npu_model_e2e.sh`
7. Verify generated outputs:
   - `compiler/src/merlin/Dialect/NPU/scripts/outputs/e2e/global_optimization.mlir`
   - `compiler/src/merlin/Dialect/NPU/scripts/outputs/e2e/program.isa.txt`

Note: this currently validates compiler/simulator integration shape and parser
compatibility, not taped-out hardware behavior.

## Follow-Up Tasks

- Add CI coverage for NPU transform checks + one simulator smoke test.
- Add shape-aware parity invocation in e2e script (derive M/K/N from IR).
- Add more ukernel symbol-family tests and negative policy tests.

---

*Dev-blog written by:* Agustin Coppari Hollmann

*Project Members:* Yufeng Chi, Huijae An and Agustin Coppari Hollmann
