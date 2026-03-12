# Merlin NPU Dialect Stack

This integrates the `third_party/npu_model/compiler` scaffold into Merlin as a
compiler plugin target.

Dialect layers:

- `npu_kernel`: kernel-level semantic ops
- `npu_schedule`: schedule/pipeline-level ops
- `npu_isa`: near-ISA ops translated to textual ISA

Lowering slice:

- `linalg.matmul` / matmul-like `linalg.generic`
- `-> npu_kernel`
- `-> npu_schedule`
- `-> npu_isa`
- `-> textual ISA` via `npu-translate --mlir-to-npu-text-isa`

Extra compiler stages:

- `npu-verify-ukernel-symbols`: validates symbol families + rank/type/shape contracts
- `npu-plan-memory`: deterministic DMA address/flag assignment on `npu_isa`

## Build

Use Merlin scripts:

```bash
conda run -n merlin-dev python tools/build.py --profile npu --config release
```

Or explicitly:

```bash
conda run -n merlin-dev python tools/build.py \
  --target host --config release \
  --plugin-compiler --no-plugin-runtime \
  --compiler-scope npu
```

## Lit tests

```bash
ctest --test-dir build/host-merlin-release -R Dialect/NPU/Transforms/tests -V
```

If `ctest` is unavailable in your environment, run test commands directly with:

```bash
build/host-merlin-release/install/bin/iree-opt ...
```

## Plugin options

Relevant options:

- `--iree-npu-enable`
- `--iree-npu-enable-ukernel-verify`
- `--iree-npu-strict-ukernel-verify`
- `--iree-npu-allow-unknown-ukernel-fallback`
- `--iree-npu-matmul-use-mxu1-weights`
- `--iree-npu-enable-memory-planner`
- `--iree-npu-dma-flag-modulo`
- `--iree-npu-load-base`
- `--iree-npu-weight-base`
- `--iree-npu-store-base`

## Manual simulator smoke

Generate ISA for a symbol ukernel and run `npu_model` smoke:

```bash
cd /scratch2/agustin/merlin
conda run -n merlin-dev ./compiler/src/merlin/Dialect/NPU/scripts/run_npu_model_smoke.sh \
  build/host-merlin-release/install/bin \
  npu_uk_gemma_mlp_f8E4M3FN_f8E4M3FN_f32
```

## ISA contract checker

Validate emitted text ISA against `model_npu/configs/isa_definition.py`:

```bash
uv run python compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py /path/to/program.isa.txt
```

Default behavior checks that all required keys from `isa_definition.py` are
present in emitted ISA lines. To require exact key matches:

```bash
uv run python compiler/src/merlin/Dialect/NPU/scripts/check_isa_contract.py \
  --strict-keys /path/to/program.isa.txt
```

## Numerical parity harness

For matmul-style ISA outputs:

```bash
uv run python compiler/src/merlin/Dialect/NPU/scripts/run_numerical_parity.py /path/to/program.isa.txt
```

## Frontend-to-simulator E2E

Single command to run:

1. `iree-compile --compile-to=global-optimization` with NPU plugin
2. `npu-translate --mlir-to-npu-text-isa`
3. ISA contract check
4. `npu_model` simulator smoke

```bash
conda run -n merlin-dev ./compiler/src/merlin/Dialect/NPU/scripts/run_frontend_to_npu_model_e2e.sh \
  build/host-merlin-release/install/bin
```

Notes:

- The script compiles with `--mlir-print-op-generic` so global-opt output with
  non-NPU dialect attrs remains parseable by `npu-translate`.
- `npu-translate` runs with `--allow-unregistered-dialect` to tolerate
  remaining non-NPU ops/attrs in global-opt IR.
- Add `--parity` as the 4th argument to include numerical parity checks.
